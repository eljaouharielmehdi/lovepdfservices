"""Recover the visual style of every OCR'd word so it can be regenerated as real text
that overlays the original glyphs: text colour, font family + weight (chosen by
rendering candidates and comparing them with the scanned glyphs), font size,
baseline and horizontal scale."""
from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np

from . import fonts
from .ocr import Line, OcrPage, Word, color_distance
from .textutil import has_arabic, render

REF = 64          # reference em size used for measuring
PAD = 4           # ring (px) around a word used to sample its background


# ----------------------------------------------------------------- colour / mask
def _word_mask_and_color(rgb: np.ndarray, w: Word, others: list[tuple[int, int, int, int]]) -> None:
    """Segment the glyph pixels of a word, tighten its box to them and sample the
    text colour.  Glyph parts sticking out of the OCR box (Arabic dots, descenders,
    a cut digit) are included; table rules crossing the box are excluded."""
    H, W = rgb.shape[:2]
    x0, y0, x1, y1 = w.bbox
    pad = max(PAD, int(0.2 * (y1 - y0)))
    pad_x = max(pad, int(0.8 * (y1 - y0)))  # room to recover cut-off first / last glyphs
    X0, Y0, X1, Y1 = max(0, x0 - pad_x), max(0, y0 - pad), min(W, x1 + pad_x), min(H, y1 + pad)
    crop = rgb[Y0:Y1, X0:X1]
    ring = np.ones(crop.shape[:2], bool)
    ring[max(0, y0 - Y0 - 1):y1 - Y0 + 1, max(0, x0 - X0 - 1):x1 - X0 + 1] = False
    ring_px = crop[ring] if ring.any() else crop.reshape(-1, 3)
    bg = np.median(ring_px, axis=0).astype(np.uint8)
    dist = color_distance(crop, bg[None, None, :])
    inner = dist[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0]
    if inner.size == 0:
        w.mask = np.zeros((max(1, w.h), max(1, w.w)), bool)
        return
    thr, _ = cv2.threshold(inner.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(35.0, float(thr))
    raw = (dist > thr).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    keep = np.zeros(n, bool)
    bh = y1 - y0
    for i in range(1, n):
        cx, cy, cw, ch, area = stats[i]
        # must touch the OCR box
        if cx + cw <= x0 - X0 or cx >= x1 - X0 or cy + ch <= y0 - Y0 or cy >= y1 - Y0:
            continue
        # rules / cell borders: long and thin
        if (cw > 3 * bh and ch < 0.25 * bh) or (ch > 1.6 * bh and cw < 0.25 * bh):
            continue
        # components mostly outside the box belong to a neighbour
        ix = max(0, min(cx + cw, x1 - X0) - max(cx, x0 - X0))
        iy = max(0, min(cy + ch, y1 - Y0) - max(cy, y0 - Y0))
        if ix * iy < 0.3 * cw * ch:
            continue
        keep[i] = True
    if keep.any():
        _grow(keep, stats, bh, [(a - X0, b - Y0, c - X0, d - Y0) for a, b, c, d in others])
    full = keep[lab]
    if full.sum() < 3:
        w.mask = np.zeros((max(1, w.h), max(1, w.w)), bool)
        w.color = (0, 0, 0) if bg.mean() > 128 else (255, 255, 255)
        return
    ys, xs = np.nonzero(full)
    tx0, ty0, tx1, ty1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    w.bbox = (int(X0 + tx0), int(Y0 + ty0), int(X0 + tx1), int(Y0 + ty1))
    mask = full[ty0:ty1, tx0:tx1]
    w.mask = mask
    d = dist[ty0:ty1, tx0:tx1]
    core = mask & (d >= np.percentile(d[mask], 85))
    w.color = tuple(int(c) for c in np.median(crop[ty0:ty1, tx0:tx1][core], axis=0))  # type: ignore[assignment]


def _grow(keep: np.ndarray, stats: np.ndarray, bh: int, others) -> None:
    """Add ink components adjacent (same text line, small gap) to the word that no
    other word claims: fixes OCR boxes that cut off the first / last glyph."""
    def box_of(sel):
        idx = np.nonzero(sel)[0]
        return (stats[idx, 0].min(), stats[idx, 1].min(),
                (stats[idx, 0] + stats[idx, 2]).max(), (stats[idx, 1] + stats[idx, 3]).max())

    changed = True
    while changed:
        changed = False
        bx0, by0, bx1, by1 = box_of(keep)
        for i in range(1, len(keep)):
            if keep[i]:
                continue
            cx, cy, cw, ch, area = stats[i]
            if area < 4 or ch > 1.6 * bh or cw > 3 * bh:
                continue
            vo = max(0, min(cy + ch, by1) - max(cy, by0))
            if vo < 0.5 * ch:
                continue
            gap = max(bx0 - (cx + cw), cx - bx1, 0)
            if gap > 0.3 * bh:
                continue
            if any(cx < o[2] and cx + cw > o[0] and cy < o[3] and cy + ch > o[1] for o in others):
                continue
            keep[i] = True
            changed = True


# ------------------------------------------------------------------- font match
def _script(w: Word, line: Line) -> str:
    if has_arabic(w.text):
        return "arabic"
    if line.rtl and not any(c.isalpha() for c in w.text):
        return "arabic"   # digits / punctuation inside an Arabic line
    return "latin"


def _glyph_mask(text: str, face: fonts.FontFace, rtl: bool, size: tuple[int, int]) -> np.ndarray | None:
    r = render(text, face, REF, rtl)
    if r.ink is None:
        return None
    x0, y0, x1, y1 = r.ink
    g = cv2.resize(r.mask[y0:y1, x0:x1], size, interpolation=cv2.INTER_AREA)
    return g > 110


def stroke_width(mask: np.ndarray) -> float:
    """Mean stroke thickness ~ area / (perimeter / 2)."""
    m = mask.astype(np.uint8)
    area = float(m.sum())
    if area == 0:
        return 0.0
    contours, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    per = sum(cv2.arcLength(c, True) for c in contours)
    return 2.0 * area / max(1.0, per)


def _iou_parts(a: np.ndarray, b: np.ndarray) -> tuple[int, int]:
    return int((a & b).sum()), int((a | b).sum())


def _match_fonts(page: OcrPage) -> None:
    # scores[(line_idx, script)][(family, bold)] = [inter, union]
    scores: dict = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    per_word: dict[int, dict] = {}
    for li, ln in enumerate(page.lines):
        for w in ln.words:
            if w.mask is None or w.mask.sum() < 8:
                continue
            sc = _script(w, ln)
            ws = {}
            for fam in fonts.available_faces(sc):
                for bold in (False, True):
                    g = _glyph_mask(w.text, fonts.face(fam, bold), w.rtl or ln.rtl,
                                    (w.mask.shape[1], w.mask.shape[0]))
                    if g is None:
                        continue
                    i, u = _iou_parts(g, w.mask)
                    ws[(fam, bold)] = i / max(1, u)
                    s = scores[(li, sc)][(fam, bold)]
                    s[0] += i
                    s[1] += u
            per_word[id(w)] = ws

    # page level winner per script, overridable by clearly different lines
    page_tot: dict = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for (li, sc), d in scores.items():
        for k, (i, u) in d.items():
            page_tot[sc][k][0] += i
            page_tot[sc][k][1] += u

    def fam_score(d, fam):
        best = 0.0
        for bold in (False, True):
            i, u = d.get((fam, bold), (0, 0))
            best = max(best, i / max(1, u))
        return best

    page_fam = {}
    for sc, d in page_tot.items():
        fams = {k[0] for k in d}
        page_fam[sc] = max(fams, key=lambda f: fam_score(d, f))

    default = {"arabic": "NaskhArabic", "latin": "NotoSans"}
    weight_pref: dict[int, float] = {}
    for li, ln in enumerate(page.lines):
        for w in ln.words:
            sc = _script(w, ln)
            fam = page_fam.get(sc, default[sc])
            d = scores.get((li, sc))
            if d:
                line_best = max({k[0] for k in d}, key=lambda f: fam_score(d, f))
                if fam_score(d, line_best) > fam_score(d, fam) + 0.04:
                    fam = line_best
            w.face_key = fam
            w.bold = False
            if w.mask is None or w.mask.sum() < 8:
                continue
            size = (w.mask.shape[1], w.mask.shape[0])
            gr = _glyph_mask(w.text, fonts.face(fam, False), w.rtl or ln.rtl, size)
            gb = _glyph_mask(w.text, fonts.face(fam, True), w.rtl or ln.rtl, size)
            if gr is None or gb is None:
                continue
            so, sr, sb = stroke_width(w.mask), stroke_width(gr), stroke_width(gb)
            # signed preference for bold, normalised by the regular/bold gap
            weight_pref[id(w)] = (abs(so - sr) - abs(so - sb)) / max(1e-6, abs(sb - sr))
            w.bold = weight_pref[id(w)] > 0
        # a line is usually uniformly bold / regular: majority vote, weak votes follow
        prefs = [weight_pref[id(w)] for w in ln.words if id(w) in weight_pref]
        if len(prefs) >= 3:
            maj = sum(p > 0 for p in prefs) * 2 > len(prefs)
            for w in ln.words:
                if abs(weight_pref.get(id(w), 0.0)) < 0.35:
                    w.bold = maj


# ------------------------------------------------------------------ geometry
def _place(w: Word, r, s: float) -> None:
    """Font size, baseline, horizontal scale and centre for scale s (px per ref px)."""
    w.font_px = s * REF
    ix0, iy0, ix1, iy1 = r.ink
    w.baseline = w.bbox[3] - (iy1 - r.baseline_y) * s
    raw = w.w / max(1.0, (ix1 - ix0) * s)
    # the recognised text cannot plausibly produce this ink (misread / merged
    # glyphs): the scan pixels stay visible and the text layer goes invisible
    w.suspect = not 0.65 <= raw <= 1.5
    w.scale_x = float(np.clip(raw, 0.5, 2.0))
    off = ((ix0 + ix1) / 2 - (r.origin_x + r.advance / 2)) * s * w.scale_x
    w.center_x = (w.bbox[0] + w.bbox[2]) / 2 - off


def _width_fit(w: Word, r, s: float) -> float:
    """|log| of the horizontal stretch needed at scale s (0 = perfect fit)."""
    return abs(np.log(max(1e-3, w.w / max(1.0, (r.ink[2] - r.ink[0]) * s))))


def _fit_geometry(page: OcrPage) -> None:
    meas: dict[int, tuple] = {}
    for ln in page.lines:
        own = []
        for w in ln.words:
            r = render(w.text, fonts.face(w.face_key, w.bold), REF, w.rtl or ln.rtl)
            if r.ink is None:
                continue
            sc = w.h / max(1, r.ink[3] - r.ink[1])
            meas[id(w)] = (r, sc)
            own.append(sc)
        if not own:
            continue
        line_s = float(np.median(own))
        for w in ln.words:
            if id(w) not in meas:
                w.font_px = line_s * REF
                w.baseline = w.bbox[3]
                w.center_x = (w.bbox[0] + w.bbox[2]) / 2
                continue
            r, sc = meas[id(w)]
            s_use = sc if abs(sc - line_s) / line_s > 0.25 else line_s
            meas[id(w)] = (r, s_use)
            _place(w, r, s_use)

    # row consensus across the page (table rows, forms): words of one visual row
    # with the same style share a size and a baseline.  A word whose glyph box is
    # incomplete (dots outside the OCR box...) is snapped when that also fits its
    # width better.
    words = [w for w in page.words if id(w) in meas]
    rows: list[list[Word]] = []
    for w in sorted(words, key=lambda w: (w.bbox[1] + w.bbox[3]) / 2):
        cy = (w.bbox[1] + w.bbox[3]) / 2
        for row in rows:
            ry0 = np.median([x.bbox[1] for x in row])
            ry1 = np.median([x.bbox[3] for x in row])
            if ry0 <= cy <= ry1:
                row.append(w)
                break
        else:
            rows.append([w])
    for row in rows:
        for w in row:
            peers = [x for x in row if x is not w and x.bold == w.bold and x.face_key == w.face_key]
            if len(peers) < 2:
                continue
            s_row = float(np.median([meas[id(x)][1] for x in peers]))
            r, s_w = meas[id(w)]
            if abs(s_w - s_row) / s_row > 0.12 and _width_fit(w, r, s_row) < _width_fit(w, r, s_w):
                meas[id(w)] = (r, s_row)
                _place(w, r, s_row)
                base = float(np.median([x.baseline for x in peers]))
                if abs(w.baseline - base) < 0.2 * w.font_px:
                    w.baseline = base
        # snap words of one row on one baseline when they nearly agree
        groups: dict[int, list[Word]] = {}
        for w in row:
            groups.setdefault(int(round(w.font_px)), []).append(w)
        for same in groups.values():
            if len(same) > 1:
                med = float(np.median([w.baseline for w in same]))
                for w in same:
                    if abs(w.baseline - med) < 0.08 * w.font_px:
                        w.baseline = med


def style_page(rgb: np.ndarray, page: OcrPage) -> OcrPage:
    words = page.words
    boxes = [w.bbox for w in words]
    for i, w in enumerate(words):
        _word_mask_and_color(rgb, w, boxes[:i] + boxes[i + 1:])
    _match_fonts(page)
    _fit_geometry(page)
    return page
