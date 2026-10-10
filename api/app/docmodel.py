"""Format-neutral page model (points, top-left origin) used to write Word files,
built either from a digital PDF page (PyMuPDF) or from an OCR'd scan."""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
import pymupdf
from PIL import Image

from . import fonts
from .pipeline import AnalysedPage
from .textutil import has_arabic, is_rtl_text, normalize_ocr_text


@dataclass
class DWord:
    text: str
    bbox: tuple[float, float, float, float]
    size: float                 # pt
    bold: bool = False
    italic: bool = False
    color: tuple[int, int, int] = (0, 0, 0)
    font: str = "Noto Sans"
    baseline: float | None = None   # pt


@dataclass
class DLine:
    words: list[DWord]
    rtl: bool
    group: int = 0              # paragraph / block id hint from the source

    @property
    def bbox(self):
        return (min(w.bbox[0] for w in self.words), min(w.bbox[1] for w in self.words),
                max(w.bbox[2] for w in self.words), max(w.bbox[3] for w in self.words))

    @property
    def size(self) -> float:
        return float(np.median([w.size for w in self.words]))

    @property
    def baseline(self) -> float:
        bs = [w.baseline for w in self.words if w.baseline is not None]
        if bs:
            return float(np.median(bs))
        return self.bbox[3] - 0.25 * self.size


@dataclass
class DRule:                    # thin horizontal / vertical line or filled box
    x: float
    y: float
    w: float
    h: float
    color: tuple[int, int, int]


@dataclass
class DPage:
    width: float
    height: float
    lines: list[DLine] = field(default_factory=list)
    rects: list[DRule] = field(default_factory=list)
    background: Image.Image | None = None   # non-text graphics, page sized
    bg_scale: float = 1.0                    # background px per pt


# ------------------------------------------------------------------ helpers
def logical_order(words: list[DWord], rtl: bool) -> list[DWord]:
    """Visual (x sorted) words of a line -> reading order. In RTL lines, runs of
    consecutive LTR words (e.g. "Thank you") keep their left-to-right order."""
    if not rtl:
        return sorted(words, key=lambda w: w.bbox[0])
    ws = sorted(words, key=lambda w: -w.bbox[2])
    out, run = [], []
    for w in ws:
        ltr = not is_rtl_text(w.text) and any(c.isalpha() for c in w.text)
        if ltr:
            run.append(w)
            continue
        out.extend(reversed(run))
        run = []
        out.append(w)
    out.extend(reversed(run))
    return out


def _clean_font_name(name: str) -> str:
    if "+" in name[:8]:
        name = name.split("+", 1)[1]
    base = name.split("-")[0].split(",")[0]
    # "NotoNaskhArabic" -> "Noto Naskh Arabic"
    out = ""
    for i, c in enumerate(base):
        if i and c.isupper() and (base[i - 1].islower()):
            out += " "
        out += c
    return out or "Noto Sans"


# ------------------------------------------------------------------ digital pdf
def from_pdf_page(page: pymupdf.Page, render_dpi: int = 150) -> DPage:
    W, H = page.rect.width, page.rect.height
    dp = DPage(W, H)
    flags = pymupdf.TEXT_PRESERVE_WHITESPACE | pymupdf.TEXT_MEDIABOX_CLIP | pymupdf.TEXT_PRESERVE_LIGATURES
    tp = page.get_textpage(flags=flags)
    d = page.get_text("dict", textpage=tp)
    words = page.get_text("words", textpage=tp)
    lines: dict[tuple[int, int], list[DWord]] = {}
    for x0, y0, x1, y1, text, b, l, _ in words:
        text = normalize_ocr_text(text)
        if not text:
            continue
        spans = []
        try:
            spans = d["blocks"][b]["lines"][l]["spans"]
        except (IndexError, KeyError):
            pass
        best, best_ov = None, 0.0
        for s in spans:
            sx0, sy0, sx1, sy1 = s["bbox"]
            ov = max(0, min(x1, sx1) - max(x0, sx0)) * max(0, min(y1, sy1) - max(y0, sy0))
            if ov > best_ov:
                best, best_ov = s, ov
        size, bold, italic, color, font = 11.0, False, False, (0, 0, 0), "Noto Sans"
        base = None
        if best:
            base = float(best["origin"][1])
            size = float(best["size"])
            fname = best["font"]
            bold = bool(best["flags"] & 16) or "bold" in fname.lower()
            italic = bool(best["flags"] & 2) or "italic" in fname.lower()
            c = best["color"]
            color = ((c >> 16) & 255, (c >> 8) & 255, c & 255)
            font = _clean_font_name(fname)
        lines.setdefault((b, l), []).append(DWord(text, (x0, y0, x1, y1), size, bold, italic, color, font, base))

    # merge pymupdf lines that sit on the same baseline in the same block
    for (b, _l), ws in sorted(lines.items(), key=lambda kv: (kv[0][0], min(w.bbox[1] for w in kv[1]))):
        rtl = sum(has_arabic(w.text) for w in ws) * 2 >= len(ws) and any(has_arabic(w.text) for w in ws)
        dp.lines.append(DLine(ws, rtl, b))
    dp.lines = _merge_same_row(dp.lines)
    for ln in dp.lines:
        ln.words = logical_order(ln.words, ln.rtl)

    # vector rules / boxes
    for dr in page.get_drawings():
        fill = dr.get("fill")
        stroke = dr.get("color")
        width = dr.get("width") or 1.0
        for item in dr["items"]:
            if item[0] == "re":
                r = item[1]
                col = fill or stroke
                if col is None:
                    continue
                if fill is None and stroke is not None:
                    # stroked rectangle -> 4 rules
                    for rx, ry, rw, rh in ((r.x0, r.y0, r.width, 0), (r.x0, r.y1, r.width, 0),
                                           (r.x0, r.y0, 0, r.height), (r.x1, r.y0, 0, r.height)):
                        dp.rects.append(_rule(rx, ry, rw, rh, width, stroke))
                else:
                    dp.rects.append(DRule(r.x0, r.y0, r.width, r.height, _rgb(col)))
            elif item[0] == "l" and stroke is not None:
                p1, p2 = item[1], item[2]
                if abs(p1.y - p2.y) < 0.5 or abs(p1.x - p2.x) < 0.5:
                    dp.rects.append(_rule(min(p1.x, p2.x), min(p1.y, p2.y), abs(p2.x - p1.x),
                                          abs(p2.y - p1.y), width, stroke))
    dp.background = _pdf_background(page, render_dpi)
    dp.bg_scale = render_dpi / 72
    return dp


def _rgb(c) -> tuple[int, int, int]:
    return tuple(int(round(v * 255)) for v in c[:3])  # type: ignore[return-value]


def _rule(x, y, w, h, width, color) -> DRule:
    if h == 0:
        return DRule(x, y - width / 2, w, width, _rgb(color))
    return DRule(x - width / 2, y, width, h, _rgb(color))


def _merge_same_row(lines: list[DLine]) -> list[DLine]:
    out: list[DLine] = []
    for ln in lines:
        x0, y0, x1, y1 = ln.bbox
        for o in out:
            ox0, oy0, ox1, oy1 = o.bbox
            if o.group == ln.group and o.rtl == ln.rtl:
                ov = min(y1, oy1) - max(y0, oy0)
                if ov > 0.6 * min(y1 - y0, oy1 - oy0) and abs(ln.size - o.size) < 0.25 * o.size:
                    o.words.extend(ln.words)
                    break
        else:
            out.append(ln)
    return out


def _pdf_background(page: pymupdf.Page, dpi: int) -> Image.Image:
    """The page rendered without its text (graphics and pictures only)."""
    doc = pymupdf.open()
    doc.insert_pdf(page.parent, from_page=page.number, to_page=page.number)
    p = doc[0]
    for b in p.get_text("dict")["blocks"]:
        if b.get("type") == 0:
            for ln in b["lines"]:
                for s in ln["spans"]:
                    p.add_redact_annot(pymupdf.Rect(s["bbox"]))
    p.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE,
                       graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
                       text=pymupdf.PDF_REDACT_TEXT_REMOVE)
    pix = p.get_pixmap(dpi=dpi, alpha=False)
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


# ------------------------------------------------------------------ ocr
def from_ocr(ap: AnalysedPage) -> DPage:
    k = 72.0 / ap.ocr.dpi
    dp = DPage(ap.ocr.width * k, ap.ocr.height * k)
    for ln in ap.ocr.lines:
        ws = []
        for w in ln.words:
            if not w.face_key:
                continue
            f = fonts.face(w.face_key, w.bold)
            x0, y0, x1, y1 = w.bbox
            ws.append(DWord(w.text, (x0 * k, y0 * k, x1 * k, y1 * k), round(w.font_px * k * 2) / 2,
                            w.bold, False, w.color, f.family, w.baseline * k))
        if ws:
            dp.lines.append(DLine(ws, ln.rtl, ln.par))
    dp.lines = _merge_same_row(dp.lines)
    for ln in dp.lines:
        ln.words = logical_order(ln.words, ln.rtl)
    g = ap.gfx
    for r in g.rects:
        dp.rects.append(DRule(r.x * k, r.y * k, r.w * k, r.h * k, r.color))
    bg = g.clean_all if g.clean_all is not None else g.clean
    if bg is not None:
        dp.background = Image.fromarray(bg)
        dp.bg_scale = 1 / k
    return dp
