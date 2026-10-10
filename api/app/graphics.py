"""Everything on the page that is not recognised text.

1. The recognised glyph pixels are erased (inpainted) from the scan, so the text can
   be redrawn as real vector text without a ghost underneath.
2. The remaining picture is decomposed into
   * the page colour,
   * solid rectangles (header bands, shaded cells, boxes)  -> vector <rect>
   * horizontal / vertical rules (table borders, underlines) -> vector <rect>
   * anything else (logos, photos, signatures, stamps, text the OCR was not sure
     about) -> small image crops placed at their exact position.
   Low-confidence text is intentionally *not* erased, so it survives as an image
   and nothing on the page is ever lost.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .ocr import OcrPage, color_distance


@dataclass
class Rect:
    x: float
    y: float
    w: float
    h: float
    color: tuple[int, int, int]


@dataclass
class Crop:
    x: int
    y: int
    image: np.ndarray  # RGB


@dataclass
class Graphics:
    page_color: tuple[int, int, int]
    rects: list[Rect] = field(default_factory=list)
    crops: list[Crop] = field(default_factory=list)
    clean: np.ndarray | None = None   # inpainted page (suspect words kept as pixels)
    clean_all: np.ndarray | None = None  # every recognised word removed (for Word export)
    for_word: "Graphics | None" = None    # decomposition used by the Word writer


def text_mask(page: OcrPage, include_suspect: bool = False) -> np.ndarray:
    m = np.zeros((page.height, page.width), np.uint8)
    for w in page.words:
        if w.mask is None or (w.suspect and not include_suspect):
            continue
        x0, y0, x1, y1 = w.bbox
        m[y0:y1, x0:x1][w.mask[: y1 - y0, : x1 - x0]] = 255
    k = max(3, int(round(page.dpi / 100)) * 2 + 1)
    return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))


def erase_text(rgb: np.ndarray, page: OcrPage, include_suspect: bool = False) -> np.ndarray:
    m = text_mask(page, include_suspect)
    if not m.any():
        return rgb.copy()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    out = cv2.inpaint(bgr, m, 3, cv2.INPAINT_TELEA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def _dominant(pixels: np.ndarray) -> tuple[tuple[int, int, int], float]:
    """Most frequent colour (quantised to 8 levels / channel) and its share."""
    q = (pixels // 32).astype(np.int32)
    keys = q[:, 0] * 64 + q[:, 1] * 8 + q[:, 2]
    vals, counts = np.unique(keys, return_counts=True)
    k = vals[counts.argmax()]
    sel = pixels[keys == k]
    return tuple(int(c) for c in np.median(sel, axis=0)), counts.max() / len(keys)  # type: ignore[return-value]


def _lines(mask: np.ndarray, rgb: np.ndarray, min_len: int, max_thick: int) -> tuple[list[Rect], np.ndarray]:
    """Straight horizontal / vertical rules. Only the *thin* part of the directional
    opening is kept, so rules are separated from the filled boxes they touch."""
    rects: list[Rect] = []
    found = np.zeros_like(mask)
    for along, across in (((min_len, 1), (1, max_thick + 1)), ((1, min_len), (max_thick + 1, 1))):
        directional = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, along))
        thick = cv2.morphologyEx(directional, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, across))
        thin = cv2.subtract(directional, thick)
        n, lab, stats, cent = cv2.connectedComponentsWithStats(thin, 8)
        horizontal = along[0] > 1
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            length, span = (w, h) if horizontal else (h, w)
            t = area / max(1, length)            # mean thickness
            if t > max_thick or length < max(min_len, 10 * t) or span > 2 * t + 3:
                continue
            sel = lab[y:y + h, x:x + w] == i
            covered = sel.any(axis=0 if horizontal else 1).mean()
            if covered < 0.9:                    # dashed / broken: keep as image
                continue
            color = tuple(int(c) for c in np.median(rgb[y:y + h, x:x + w][sel], axis=0))
            t = max(1.0, round(t))
            if horizontal:
                rects.append(Rect(x, cent[i][1] - t / 2, w, t, color))  # type: ignore[arg-type]
            else:
                rects.append(Rect(cent[i][0] - t / 2, y, t, h, color))  # type: ignore[arg-type]
            found[y:y + h, x:x + w][sel] = 255
    return rects, found


def analyse(rgb: np.ndarray, page: OcrPage, vectorize: bool = True) -> Graphics:
    clean = erase_text(rgb, page)
    page_color, _ = _dominant(clean[::4, ::4].reshape(-1, 3))
    g = Graphics(page_color=page_color, clean=clean)
    suspects = any(w.suspect for w in page.words)
    g.clean_all = erase_text(rgb, page, True) if suspects else clean
    if not vectorize:
        g.for_word = g
        return g
    _vectorize(g, clean, page)
    if suspects:
        # Word export writes suspect words as text, so they must not survive as pixels
        g.for_word = Graphics(page_color=page_color, clean=g.clean_all, clean_all=g.clean_all)
        _vectorize(g.for_word, g.clean_all, page)
    else:
        g.for_word = g
    return g


def _vectorize(g: Graphics, clean: np.ndarray, page: OcrPage) -> None:
    H, W = clean.shape[:2]
    ink = (color_distance(clean, np.array(g.page_color, np.uint8)[None, None, :]) > 28).astype(np.uint8) * 255
    # drop scanner specks
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    min_len = max(20, int(page.dpi * 0.15))
    max_thick = max(6, int(page.dpi * 0.05))
    _decompose(clean, ink, (0, 0, W, H), g.page_color, g, min_len, max_thick, depth=0)
    _tidy_crops(g)


def _tidy_crops(g: Graphics, pad: int = 2) -> None:
    """Drop duplicate crops (fully inside another one) and anti-aliasing / noise
    fringes: slivers a few pixels thick hugging the edge of a vector box."""
    boxes = [(c.x, c.y, c.x + c.image.shape[1], c.y + c.image.shape[0]) for c in g.crops]
    keep = []
    for i, (c, b) in enumerate(zip(g.crops, boxes)):
        inside = any(j != i and o[0] <= b[0] and o[1] <= b[1] and o[2] >= b[2] and o[3] >= b[3]
                     and (o != b or j < i) for j, o in enumerate(boxes))
        if inside:
            continue
        x0, y0, x1, y1 = b[0] + pad, b[1] + pad, b[2] - pad, b[3] - pad
        if min(x1 - x0, y1 - y0) <= 4:
            near_edge = False
            for r in g.rects:
                rx0, ry0, rx1, ry1 = r.x, r.y, r.x + r.w, r.y + r.h
                overlaps = x0 <= rx1 + 4 and x1 >= rx0 - 4 and y0 <= ry1 + 4 and y1 >= ry0 - 4
                on_edge = (min(abs(y0 - ry0), abs(y1 - ry1), abs(y0 - ry1), abs(y1 - ry0)) <= 4
                           or min(abs(x0 - rx0), abs(x1 - rx1), abs(x0 - rx1), abs(x1 - rx0)) <= 4)
                if overlaps and on_edge:
                    near_edge = True
                    break
            if near_edge:
                continue
        keep.append(c)
    g.crops = keep


def _decompose(clean, ink, region, bg_color, g: Graphics, min_len: int, max_thick: int,
               depth: int) -> None:
    rx0, ry0, rx1, ry1 = region
    sub_ink = ink[ry0:ry1, rx0:rx1].copy()
    sub_rgb = clean[ry0:ry1, rx0:rx1]
    line_rects, line_px = _lines(sub_ink, sub_rgb, min_len, max_thick)
    for r in line_rects:
        g.rects.append(Rect(r.x + rx0, r.y + ry0, r.w, r.h, r.color))
    rest = cv2.subtract(sub_ink, line_px)
    # close tiny gaps so a logo / photo is one component
    rest_c = cv2.morphologyEx(rest, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(rest_c, 8)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 12 or (w < 4 and h < 4):
            continue
        comp = lab[y:y + h, x:x + w] == i
        px = sub_rgb[y:y + h, x:x + w][comp]
        color, _ = _dominant(px)
        # share of pixels close to that colour: robust to scanner / sensor noise
        share = float((np.abs(px.astype(np.int16) - np.array(color, np.int16)).max(axis=-1) <= 36).mean())
        fill = area / float(w * h)
        X, Y = x + rx0, y + ry0
        if fill > 0.92 and share > 0.6 and w >= 6 and h >= 6:
            # a solid box: the rect covers its fill colour only; whatever is drawn
            # in / around it (borders, logos...) is decomposed on top of it
            box = clean[Y:Y + h, X:X + w]
            same = color_distance(box, np.array(color, np.uint8)[None, None, :]) <= 28
            dom = comp & same
            ys, xs = np.nonzero(dom)
            g.rects.append(Rect(X + xs.min(), Y + ys.min(), xs.max() - xs.min() + 1,
                                ys.max() - ys.min() + 1, color))
            if depth < 3:
                inner = ((comp | ~same) & ~dom).astype(np.uint8) * 255
                inner = cv2.morphologyEx(inner, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
                if inner.any():
                    full = np.zeros_like(ink)
                    full[Y:Y + h, X:X + w] = inner
                    _decompose(clean, full, (X, Y, X + w, Y + h), color, g, min_len, max_thick,
                               depth + 1)
            continue
        # anything else stays pixel-exact
        pad = 2
        H, W = clean.shape[:2]
        cx0, cy0 = max(0, X - pad), max(0, Y - pad)
        cx1, cy1 = min(W, X + w + pad), min(H, Y + h + pad)
        g.crops.append(Crop(cx0, cy0, clean[cy0:cy1, cx0:cx1].copy()))
