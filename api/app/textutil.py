"""Script detection and glyph rendering helpers (Pillow + libraqm: real Arabic
shaping and bidi, the same HarfBuzz shaping Chromium uses)."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw

from . import fonts

_ARABIC_RE = re.compile("[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufefe]")
_BIDI_CTRL_RE = re.compile("[\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff]")
_RTL_RE = re.compile("[\u0590-\u08ff\ufb1d-\ufdff\ufe70-\ufefe]")


def has_arabic(text: str) -> bool:
    return bool(_ARABIC_RE.search(text))


def is_rtl_text(text: str) -> bool:
    return bool(_RTL_RE.search(text))


def normalize_ocr_text(text: str) -> str:
    """Clean Tesseract output: NFC, Arabic punctuation inside Arabic words, no stray
    tatweel-only tokens."""
    text = _BIDI_CTRL_RE.sub("", unicodedata.normalize("NFC", text)).strip()
    if has_arabic(text):
        # tesseract often outputs latin comma / semicolon / question mark
        text = text.replace(",", "،").replace(";", "؛").replace("?", "؟")
    return text


@dataclass
class Rendered:
    mask: np.ndarray      # uint8 coverage (0..255), shape (h, w)
    origin_x: float       # x of the start of the advance box inside mask
    baseline_y: float     # y of the baseline inside mask
    advance: float        # total advance width (px)
    ink: tuple[int, int, int, int] | None  # x0, y0, x1, y1 of ink inside mask


def _runs(text: str, primary: fonts.FontFace) -> list[tuple[str, fonts.FontFace]]:
    """Split text into runs rendered with primary face or its fallback (font fallback
    exactly like the css stack `primary, fallback`)."""
    cov = fonts.cmap(primary)
    fb = fonts.fallback_face(primary)
    runs: list[tuple[str, fonts.FontFace]] = []
    for ch in text:
        f = primary if (ord(ch) in cov or ch.isspace()) else fb
        if runs and runs[-1][1] == f:
            runs[-1] = (runs[-1][0] + ch, f)
        else:
            runs.append((ch, f))
    return runs


@lru_cache(maxsize=4096)
def render(text: str, face: fonts.FontFace, size: int, rtl: bool) -> Rendered:
    runs = _runs(text, face)
    if rtl:
        runs = runs[::-1]  # visual order for a single RTL word
    pad = size
    advs = []
    for t, f in runs:
        d = "rtl" if is_rtl_text(t) else "ltr"
        advs.append(fonts.pil_font(f, size).getlength(t, direction=d))
    advance = float(sum(advs))
    w = int(advance + 2 * pad) + 1
    h = int(size * 3)
    baseline = int(size * 1.8)
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    x = float(pad)
    for (t, f), a in zip(runs, advs):
        d = "rtl" if is_rtl_text(t) else "ltr"
        draw.text((x, baseline), t, fill=255, font=fonts.pil_font(f, size), anchor="ls",
                  direction=d)
        x += a
    arr = np.asarray(img)
    ys, xs = np.nonzero(arr > 64)
    ink = None if len(xs) == 0 else (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    return Rendered(arr, float(pad), float(baseline), advance, ink)
