"""Font registry shared by the OCR layout engine (measuring / matching glyphs with
Pillow + libraqm) and the HTML renderer (fonts are shipped to Gotenberg as assets,
so what is measured is exactly what Chromium draws)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from fontTools.ttLib import TTFont
from PIL import ImageFont

FONT_DIRS = [
    Path(os.environ.get("FONT_DIR", "/usr/share/fonts")),
]


@dataclass(frozen=True)
class FontFace:
    key: str          # css family name used in generated html, e.g. "F-NaskhArabic"
    family: str       # human family name, e.g. "Noto Naskh Arabic"
    script: str       # "arabic" | "latin"
    bold: bool
    filename: str     # e.g. "NotoNaskhArabic-Bold.ttf"

    @property
    def css_family(self) -> str:
        # one css family per (family, weight) keeps @font-face trivial
        return f"{self.key}-{'B' if self.bold else 'R'}"


# family key -> (human name, script, regular file, bold file)
_FAMILIES = {
    "NaskhArabic": ("Noto Naskh Arabic", "arabic", "NotoNaskhArabic-Regular.ttf", "NotoNaskhArabic-Bold.ttf"),
    "SansArabic": ("Noto Sans Arabic", "arabic", "NotoSansArabic-Regular.ttf", "NotoSansArabic-Bold.ttf"),
    "Amiri": ("Amiri", "arabic", "Amiri-Regular.ttf", "Amiri-Bold.ttf"),
    "KufiArabic": ("Noto Kufi Arabic", "arabic", "NotoKufiArabic-Regular.ttf", "NotoKufiArabic-Bold.ttf"),
    "NotoSans": ("Noto Sans", "latin", "NotoSans-Regular.ttf", "NotoSans-Bold.ttf"),
    "NotoSerif": ("Noto Serif", "latin", "NotoSerif-Regular.ttf", "NotoSerif-Bold.ttf"),
    "LiberationSans": ("Liberation Sans", "latin", "LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf"),
    "LiberationSerif": ("Liberation Serif", "latin", "LiberationSerif-Regular.ttf", "LiberationSerif-Bold.ttf"),
    "LiberationMono": ("Liberation Mono", "latin", "LiberationMono-Regular.ttf", "LiberationMono-Bold.ttf"),
}

# fallback used for characters missing from a face (punctuation in Arabic fonts...)
FALLBACK = {"arabic": "NotoSans", "latin": "NaskhArabic"}


@lru_cache(maxsize=None)
def find_file(filename: str) -> Path:
    for d in FONT_DIRS:
        for p in d.rglob(filename):
            return p
    raise FileNotFoundError(f"font {filename} not found in {FONT_DIRS}")


@lru_cache(maxsize=None)
def face(key: str, bold: bool) -> FontFace:
    name, script, reg, bld = _FAMILIES[key]
    return FontFace(key, name, script, bold, bld if bold else reg)


def available_faces(script: str | None = None) -> list[str]:
    keys = []
    for k, (_, s, reg, bld) in _FAMILIES.items():
        if script and s != script:
            continue
        try:
            find_file(reg)
            find_file(bld)
        except FileNotFoundError:
            continue
        keys.append(k)
    return keys


@lru_cache(maxsize=None)
def cmap(f: FontFace) -> frozenset[int]:
    return frozenset(TTFont(find_file(f.filename), lazy=True).getBestCmap().keys())


@lru_cache(maxsize=512)
def pil_font(f: FontFace, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(find_file(f.filename)), size,
                              layout_engine=ImageFont.Layout.RAQM)


def fallback_face(f: FontFace) -> FontFace:
    return face(FALLBACK[f.script], f.bold)


def css_stack(f: FontFace) -> str:
    return f"'{f.css_family}', '{fallback_face(f).css_family}'"
