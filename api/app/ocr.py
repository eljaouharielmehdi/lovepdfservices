"""Tesseract OCR with preprocessing tuned for regeneration:

* polarity / colour normalisation so white-on-blue headers, red text and text on
  shaded table cells are read like black-on-white text;
* a second sparse-text pass (psm 11) that recovers isolated words / single digits
  (table cells) that the page-layout pass drops;
* hOCR parsing into Words / Lines / Paragraphs with reading direction.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .textutil import has_arabic, is_rtl_text, normalize_ocr_text

DEFAULT_LANGS = os.environ.get("OCR_LANGS", "ara+eng")
MIN_CONF = float(os.environ.get("OCR_MIN_CONF", "45"))


@dataclass
class Word:
    text: str
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1 (pixels)
    conf: float
    rtl: bool = False
    # styling (filled by layout.style_page)
    color: tuple[int, int, int] = (0, 0, 0)
    face_key: str = ""
    bold: bool = False
    font_px: float = 0.0
    baseline: float = 0.0
    scale_x: float = 1.0
    center_x: float = 0.0
    suspect: bool = False   # text does not fit its ink: keep original pixels visible
    mask: np.ndarray | None = field(default=None, repr=False)

    @property
    def w(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def h(self) -> int:
        return self.bbox[3] - self.bbox[1]


@dataclass
class Line:
    words: list[Word]
    bbox: tuple[int, int, int, int]
    rtl: bool
    par: int
    x_size: float = 0.0

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


@dataclass
class OcrPage:
    width: int
    height: int
    lines: list[Line]
    dpi: float

    @property
    def words(self) -> list[Word]:
        return [w for ln in self.lines for w in ln.words]

    @property
    def text(self) -> str:
        out, last_par = [], None
        for ln in self.lines:
            if last_par is not None and ln.par != last_par:
                out.append("")
            out.append(ln.text)
            last_par = ln.par
        return "\n".join(out)


# --------------------------------------------------------------------------- prep
def local_background(rgb: np.ndarray, text_h: int = 40) -> np.ndarray:
    """Per-pixel background colour estimate (large median filter on a downscaled copy)."""
    h, w = rgb.shape[:2]
    f = 4
    small = cv2.resize(rgb, (max(1, w // f), max(1, h // f)), interpolation=cv2.INTER_AREA)
    k = max(15, (text_h * 3 // f) | 1)
    bg = cv2.medianBlur(small, min(k, 255))
    return cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)


def color_distance(rgb: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return np.abs(rgb.astype(np.int16) - ref.astype(np.int16)).max(axis=-1).astype(np.uint8)


def normalize_for_ocr(rgb: np.ndarray) -> np.ndarray:
    """Return a grayscale image with every glyph dark on white, whatever its colour
    or the colour of what is behind it."""
    bg = local_background(rgb)
    dist = color_distance(rgb, bg).astype(np.float32)
    norm = 255.0 - np.clip(dist * 2.2, 0, 255)
    return norm.astype(np.uint8)


def estimate_skew(rgb: np.ndarray, max_angle: float = 6.0) -> float:
    """Text skew in degrees (projection-profile method, coarse then fine)."""
    ink = normalize_for_ocr(rgb) < 128
    h, w = ink.shape
    f = min(1.0, 1200.0 / max(h, w))
    small = cv2.resize(ink.astype(np.uint8) * 255, (max(1, int(w * f)), max(1, int(h * f))),
                       interpolation=cv2.INTER_AREA)
    sh, sw = small.shape
    centre = (sw / 2, sh / 2)

    def score(a: float) -> float:
        m = cv2.getRotationMatrix2D(centre, a, 1.0)
        rot = cv2.warpAffine(small, m, (sw, sh), flags=cv2.INTER_NEAREST)
        rows = rot.sum(axis=1).astype(np.float64)
        return float(np.var(np.diff(rows)))

    best = max(np.arange(-max_angle, max_angle + 1e-6, 0.5), key=score)
    best = max(np.arange(best - 0.5, best + 0.5 + 1e-6, 0.05), key=score)
    return float(best)


def rotate_page(rgb: np.ndarray, angle: float) -> np.ndarray:
    """Rotate around the centre, filling uncovered corners with the paper colour."""
    if abs(angle) < 0.1:
        return rgb
    h, w = rgb.shape[:2]
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    fill = tuple(int(c) for c in np.median(border, axis=0))
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(rgb, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=fill)


# --------------------------------------------------------------------------- hocr
def _title_props(title: str) -> dict[str, list[str]]:
    props = {}
    for part in title.split(";"):
        part = part.strip()
        if not part:
            continue
        k, *v = part.split()
        props[k] = v
    return props


def _bbox(props) -> tuple[int, int, int, int]:
    return tuple(int(v) for v in props["bbox"][:4])  # type: ignore[return-value]


def run_tesseract(gray: np.ndarray, langs: str, psm: int, dpi: int) -> str:
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "in.png"
        cv2.imwrite(str(src), gray)
        cmd = ["tesseract", str(src), str(Path(td) / "out"), "-l", langs, "--oem", "1",
               "--psm", str(psm), "--dpi", str(dpi), "-c", "hocr_char_boxes=0", "hocr"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if res.returncode != 0:
            raise RuntimeError(f"tesseract failed: {res.stderr.strip()}")
        return (Path(td) / "out.hocr").read_text(encoding="utf-8")


_LINE_CLASSES = {"ocr_line", "ocr_header", "ocr_caption", "ocr_textfloat"}


def parse_hocr(hocr: str, par_offset: int = 0) -> list[Line]:
    root = ET.fromstring(hocr.encode("utf-8"))
    ns = "{http://www.w3.org/1999/xhtml}"
    lines: list[Line] = []
    par_id = par_offset
    for par in root.iter():
        if par.get("class") != "ocr_par":
            continue
        par_id += 1
        par_rtl = par.get("dir") == "rtl"
        for ln in par.iter():
            if ln.get("class") not in _LINE_CLASSES:
                continue
            lp = _title_props(ln.get("title", ""))
            words = []
            for w in ln.iter(f"{ns}span"):
                if w.get("class") != "ocrx_word":
                    continue
                text = normalize_ocr_text("".join(w.itertext()))
                if not text:
                    continue
                wp = _title_props(w.get("title", ""))
                conf = float(wp.get("x_wconf", ["0"])[0])
                words.append(Word(text, _bbox(wp), conf, rtl=is_rtl_text(text)))
            if not words:
                continue
            rtl = par_rtl or sum(w.rtl for w in words) * 2 >= len(words)
            xs = float(lp.get("x_size", ["0"])[0])
            lines.append(Line(words, _bbox(lp), rtl, par_id, xs))
    return lines


# --------------------------------------------------------------------------- api
def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if not inter:
        return 0.0
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])  # noqa: E731
    return inter / min(area(a), area(b))


def _plausible(w: Word, min_conf: float) -> bool:
    if w.conf < min_conf or w.w < 2 or w.h < 4:
        return False
    if not any(ch.isalnum() for ch in w.text):
        # isolated punctuation must be confident (avoid "erasing" specks / rules)
        return w.conf >= 80 and w.h < 120
    return True


def _too_big(w: Word, median_h: float) -> bool:
    """Logos / stamps / bullets recognised as a huge single letter."""
    alnum = sum(c.isalnum() for c in w.text)
    return median_h > 0 and alnum <= 2 and w.h > 2.5 * median_h


def _kind(text: str) -> str:
    if any(c.isalpha() for c in text):
        return "latin"
    return "digits"


def _latin_second_opinion(gray: np.ndarray, lines: list[Line], langs: str, dpi: int) -> None:
    """The Arabic model sometimes garbles Latin / digit strings (2025-0042 -> 2025-2).
    Re-read words without Arabic letters with the non-Arabic models only."""
    parts = [x for x in langs.split("+") if x]
    latin = "+".join(x for x in parts if x != "ara")
    if "ara" not in parts or not latin:
        return
    if all(has_arabic(w.text) for ln in lines for w in ln.words):
        return
    alt = [w for ln in parse_hocr(run_tesseract(gray, latin, 3, dpi)) for w in ln.words]
    arabic_boxes = [w.bbox for ln in lines for w in ln.words if has_arabic(w.text)]

    def inter(a, b):
        return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))

    def area(r):
        return max(1, (r[2] - r[0]) * (r[3] - r[1]))

    for ln in lines:
        for w in list(ln.words):
            if has_arabic(w.text) or w not in ln.words:
                continue
            cands = [a for a in alt if inter(a.bbox, w.bbox) > 0.5 * area(w.bbox)]
            if len(cands) != 1:
                continue
            a = cands[0]
            if a.text == w.text or has_arabic(a.text) or _kind(a.text) != _kind(w.text):
                continue
            if (a.bbox[2] - a.bbox[0]) < 0.8 * w.w:
                continue
            # characters a box of this size can hold (latin / digits ~0.55 em wide)
            expected = w.w / max(1.0, 0.55 * w.h)
            too_short = len(w.text) < 0.6 * expected and abs(len(a.text) - expected) < abs(len(w.text) - expected)
            if a.conf < (60.0 if too_short else min(85.0, w.conf - 5)):
                continue
            # never let the latin reading swallow part of an Arabic word
            if any(inter(a.bbox, ab) > 0.2 * area(ab) for ab in arabic_boxes):
                continue
            # other latin fragments covered by the better reading are merged into it
            for o in list(ln.words):
                if o is not w and not has_arabic(o.text) and inter(a.bbox, o.bbox) > 0.6 * area(o.bbox):
                    ln.words.remove(o)
            w.text, w.conf = a.text, a.conf
            w.bbox = (min(w.bbox[0], a.bbox[0]), min(w.bbox[1], a.bbox[1]),
                      max(w.bbox[2], a.bbox[2]), max(w.bbox[3], a.bbox[3]))


def _group_sparse(words: list[Word], par_offset: int) -> list[Line]:
    """Group single words from the sparse pass into lines by vertical overlap."""
    lines: list[Line] = []
    for w in sorted(words, key=lambda w: (w.bbox[1], w.bbox[0])):
        lines.append(Line([w], w.bbox, w.rtl, par_offset + len(lines) + 1, float(w.h)))
    return lines


def ocr_image(rgb: np.ndarray, dpi: float, langs: str = DEFAULT_LANGS,
              min_conf: float = MIN_CONF) -> OcrPage:
    h, w = rgb.shape[:2]
    gray = normalize_for_ocr(rgb)
    idpi = int(round(dpi))
    lines = parse_hocr(run_tesseract(gray, langs, 3, idpi))
    hs = [x.h for ln in lines for x in ln.words if x.conf >= min_conf]
    median_h = float(np.median(hs)) if hs else 0.0
    for ln in lines:
        ln.words = [x for x in ln.words if _plausible(x, min_conf) and not _too_big(x, median_h)]
    lines = [ln for ln in lines if ln.words]
    _latin_second_opinion(gray, lines, langs, idpi)

    # second pass: sparse text on what is left (single digits in table cells...)
    masked = gray.copy()
    for wd in (x for ln in lines for x in ln.words):
        x0, y0, x1, y1 = wd.bbox
        masked[max(0, y0 - 2):y1 + 2, max(0, x0 - 2):x1 + 2] = 255
    extra = []
    known = [x.bbox for ln in lines for x in ln.words]
    for ln in parse_hocr(run_tesseract(masked, langs, 11, idpi)):
        for wd in ln.words:
            if not _plausible(wd, max(min_conf, 60)) or _too_big(wd, median_h):
                continue
            if any(_iou(wd.bbox, k) > 0.2 for k in known):
                continue
            extra.append(wd)
            known.append(wd.bbox)
    par_max = max((ln.par for ln in lines), default=0)
    lines += _group_sparse(extra, par_max)
    # recompute line boxes from kept words, reading order: top-to-bottom
    for ln in lines:
        ln.bbox = (min(x.bbox[0] for x in ln.words), min(x.bbox[1] for x in ln.words),
                   max(x.bbox[2] for x in ln.words), max(x.bbox[3] for x in ln.words))
    lines.sort(key=lambda ln: (ln.par, ln.bbox[1]))
    return OcrPage(w, h, lines, dpi)
