"""Build the HTML (one SVG per page) that Gotenberg/Chromium prints into the
regenerated PDF. Coordinates are the scan's pixel grid (viewBox) so every word and
graphic lands exactly where it was on the original; the physical page size comes
from the scan DPI."""
from __future__ import annotations

import io
from html import escape

import numpy as np
from PIL import Image

from . import fonts
from .graphics import Graphics
from .ocr import OcrPage

BACKGROUND_MODES = ("vector", "original", "none")


def _hex(c) -> str:
    return "#%02x%02x%02x" % tuple(int(v) for v in c[:3])


def _png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG", optimize=True)
    return buf.getvalue()


def _jpeg(arr: np.ndarray, q: int = 90) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG", quality=q, subsampling=0)
    return buf.getvalue()


def _fmt(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".")


def build_page(page: OcrPage, gfx: Graphics, background: str = "vector",
               prefix: str = "p0") -> tuple[str, dict[str, bytes], set[fonts.FontFace]]:
    """Return (svg markup, assets, faces used) for one page."""
    W, H = page.width, page.height
    assets: dict[str, bytes] = {}
    used: set[fonts.FontFace] = set()
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
           f'viewBox="0 0 {W} {H}" width="{_fmt(W * 72 / page.dpi)}pt" '
           f'height="{_fmt(H * 72 / page.dpi)}pt" preserveAspectRatio="none">']

    if background == "original" and gfx.clean is not None:
        name = f"{prefix}_bg.jpg"
        assets[name] = _jpeg(gfx.clean, 92)
        out.append(f'<image href="{name}" x="0" y="0" width="{W}" height="{H}" preserveAspectRatio="none"/>')
    elif background == "vector":
        out.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="{_hex(gfx.page_color)}"/>')
        # big boxes first so smaller ones / rules / images stack on top
        for r in sorted(gfx.rects, key=lambda r: -(r.w * r.h)):
            out.append(f'<rect x="{_fmt(r.x)}" y="{_fmt(r.y)}" width="{_fmt(r.w)}" '
                       f'height="{_fmt(r.h)}" fill="{_hex(r.color)}"/>')
        for i, c in enumerate(gfx.crops):
            name = f"{prefix}_g{i}.png"
            assets[name] = _png(c.image)
            h, w = c.image.shape[:2]
            out.append(f'<image href="{name}" x="{c.x}" y="{c.y}" width="{w}" height="{h}" '
                       f'preserveAspectRatio="none"/>')
    else:
        out.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>')

    for ln in page.lines:
        rtl_line = ln.rtl
        for w in ln.words:
            if not w.face_key or w.font_px <= 0:
                continue
            f = fonts.face(w.face_key, w.bold)
            used.add(f)
            used.add(fonts.fallback_face(f))
            rtl = rtl_line or w.rtl
            hidden = 'fill-opacity="0" ' if w.suspect else ""
            out.append(
                f'<text transform="translate({_fmt(w.center_x)} {_fmt(w.baseline)}) '
                f'scale({w.scale_x:.4f} 1)" font-family="{escape(fonts.css_stack(f))}" '
                f'font-size="{_fmt(w.font_px)}" fill="{_hex(w.color)}" '
                f'{hidden}text-anchor="middle" '
                f'direction="{"rtl" if rtl else "ltr"}" xml:space="preserve">{escape(w.text)}</text>')
    out.append("</svg>")
    return "\n".join(out), assets, used


def font_css(faces: set[fonts.FontFace], assets: dict[str, bytes]) -> str:
    css = []
    for f in sorted(faces, key=lambda f: f.css_family):
        if f.filename not in assets:
            assets[f.filename] = fonts.find_file(f.filename).read_bytes()
        css.append(f"@font-face{{font-family:'{f.css_family}';src:url('{f.filename}');"
                   f"font-weight:normal;font-style:normal}}")
    return "\n".join(css)


def document(svg: str, width_pt: float, height_pt: float, faces: set[fonts.FontFace],
             assets: dict[str, bytes], lang: str = "ar") -> str:
    # every face is pre-loaded explicitly; the page is printed only once all fonts
    # are ready (waitForExpression on window.__ready)
    loads = ",".join(f"document.fonts.load(\"16px '{f.css_family}'\")" for f in sorted(faces, key=lambda f: f.css_family))
    return f"""<!doctype html>
<html lang="{lang}"><head><meta charset="utf-8"><title>document</title>
<style>
{font_css(faces, assets)}
@page {{ size: {_fmt(width_pt)}pt {_fmt(height_pt)}pt; margin: 0 }}
html, body {{ margin: 0; padding: 0 }}
svg {{ display: block; text-rendering: geometricPrecision }}
</style></head><body>
{svg}
<script>Promise.all([{loads}]).then(() => document.fonts.ready).then(() => {{ window.__ready = true }});</script>
</body></html>"""
