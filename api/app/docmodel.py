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
from .textutil import has_arabic, is_rtl_text, normalize_ocr_text, render


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
    scale: float = 1.0              # horizontal character scale matching the original width
    ink_box: bool = False           # bbox is the glyph ink (OCR) rather than the advance box (PDF)


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
class DPicture:                 # logo / photo / signature / stamp / complex drawing
    x: float
    y: float
    w: float
    h: float
    png: bytes


@dataclass
class DPage:
    width: float
    height: float
    lines: list[DLine] = field(default_factory=list)
    rects: list[DRule] = field(default_factory=list)       # -> tables or native shapes
    pictures: list[DPicture] = field(default_factory=list)  # -> anchored pictures
    page_color: tuple[int, int, int] = (255, 255, 255)


# ------------------------------------------------------------------ helpers
def line_is_rtl(words: list[DWord]) -> bool:
    """Base direction from letters (not words): a line holding a real share of
    Arabic letters is right-to-left even if it also carries Latin words / numbers."""
    text = "".join(w.text for w in words)
    arabic = sum(1 for c in text if has_arabic(c) and c.isalpha())
    letters = sum(1 for c in text if c.isalpha())
    return arabic > 0 and arabic >= 0.3 * letters



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
def from_pdf_page(page: pymupdf.Page) -> DPage:
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
        rtl = line_is_rtl(ws)
        dp.lines.append(DLine(ws, rtl, b))
    dp.lines = _merge_same_row(dp.lines)
    for ln in dp.lines:
        ln.words = logical_order(ln.words, ln.rtl)
        for w in ln.words:
            settle_font(w)

    # vector rules / boxes
    for dr in page.get_drawings():
        fill = dr.get("fill")
        stroke = dr.get("color")
        width = dr.get("width") or 1.0
        for item in dr["items"]:
            if item[0] in ("re", "qu") and _is_simple(item):
                r = item[1] if item[0] == "re" else item[1].rect
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
    dp.pictures = _pdf_pictures(page)
    return dp


_FACE_BY_NAME = {fonts.face(k, False).family.lower(): k for k in fonts.available_faces()}


def settle_font(w: DWord) -> None:
    """A Latin / number word with characters the (Arabic) font lacks is set in the
    fallback family explicitly, so that what we measure is what Word draws."""
    key = _FACE_BY_NAME.get(w.font.lower())
    if key is None or has_arabic(w.text):
        return
    f = fonts.face(key, w.bold)
    cov = fonts.cmap(f)
    if any(ord(c) not in cov for c in w.text if not c.isspace()):
        w.font = fonts.fallback_face(f).family


def word_metrics(w: DWord, rtl: bool) -> tuple[float, float, float, float] | None:
    """(advance, left bearing, right bearing, ink width) in pt for the word set in
    the font Word will use, or None when that font is not one we can measure."""
    key = _FACE_BY_NAME.get(w.font.lower())
    if key is None or w.size <= 0:
        return None
    r = render(w.text, fonts.face(key, w.bold), 64, rtl)
    if r.ink is None or r.advance <= 0:
        return None
    f = w.size / 64
    x0, _, x1, _ = r.ink
    return (r.advance * f, (x0 - r.origin_x) * f, (r.origin_x + r.advance - x1) * f, (x1 - x0) * f)


def space_advance(w: DWord) -> float:
    key = _FACE_BY_NAME.get(w.font.lower())
    if key is None:
        return 0.25 * w.size
    return fonts.pil_font(fonts.face(key, w.bold), 64).getlength(" ") * w.size / 64


_CLOSE = set(".,:;!?\u060c\u061b\u061f)]}\u00bb%")
_OPEN = set("([{\u00ab")


def needs_space(prev: DWord | None, w: DWord) -> bool:
    """Word separation used when writing text: punctuation sticks to its word."""
    if w.text and all(c in _CLOSE for c in w.text):
        return False
    if prev is not None and prev.text and all(c in _OPEN for c in prev.text):
        return False
    return True


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


def _without_text(page: pymupdf.Page) -> pymupdf.Page:
    """Copy of the page with every text span removed (graphics and images kept)."""
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
    return p


def _is_simple(item) -> bool:
    """Drawing primitives that become native Word shapes (axis-aligned)."""
    if item[0] == "re":
        return True
    if item[0] == "l":
        p1, p2 = item[1], item[2]
        return abs(p1.y - p2.y) < 0.5 or abs(p1.x - p2.x) < 0.5
    if item[0] == "qu":
        q = item[1]
        return q.is_rectangular
    return False


def _clusters(rects: list[pymupdf.Rect], gap: float = 2.0) -> list[pymupdf.Rect]:
    out: list[pymupdf.Rect] = []
    for r in rects:
        r = pymupdf.Rect(r)
        merged = True
        while merged:
            merged = False
            for o in out:
                if pymupdf.Rect(o.x0 - gap, o.y0 - gap, o.x1 + gap, o.y1 + gap).intersects(r):
                    out.remove(o)
                    r |= o
                    merged = True
                    break
        out.append(r)
    return out


def _pdf_pictures(page: pymupdf.Page) -> list[DPicture]:
    """Embedded images and complex vector art (curves: logos, signatures, stamps),
    each cut out of a text-free rendering at its exact place and size."""
    area = page.rect
    regions: list[tuple[pymupdf.Rect, int]] = []
    for info in page.get_image_info():
        r = pymupdf.Rect(info["bbox"]) & area
        if r.is_empty or r.width < 1 or r.height < 1:
            continue
        native = info.get("width", 0) / max(1e-3, r.width / 72)
        regions.append((r, int(min(400, max(150, native)))))
    complex_boxes = []
    for dr in page.get_drawings():
        if dr.get("fill") is None and dr.get("color") is None:
            continue
        if not all(_is_simple(it) for it in dr["items"]):
            # the path box excludes half the stroke and the line caps
            grow = (dr.get("width") or 0) / 2 + 1 if dr.get("color") is not None else 0.5
            r = pymupdf.Rect(dr["rect"])
            r = pymupdf.Rect(r.x0 - grow, r.y0 - grow, r.x1 + grow, r.y1 + grow) & area
            if not r.is_empty:
                complex_boxes.append(r)
    regions += [(r, 300) for r in _clusters(complex_boxes)]
    if not regions:
        return []
    clean = _without_text(page)
    out = []
    for r, dpi in regions:
        pix = clean.get_pixmap(clip=r, dpi=dpi, alpha=False)
        out.append(DPicture(r.x0, r.y0, r.width, r.height, pix.tobytes("png")))
    return out


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
                            w.bold, False, w.color, f.family, w.baseline * k, ink_box=True))
        if ws:
            dp.lines.append(DLine(ws, ln.rtl, ln.par))
    dp.lines = _merge_same_row(dp.lines)
    for ln in dp.lines:
        ln.words = logical_order(ln.words, ln.rtl)
        for w in ln.words:
            settle_font(w)
    g = ap.gfx.for_word or ap.gfx
    dp.page_color = tuple(int(c) for c in g.page_color)  # type: ignore[assignment]
    for r in g.rects:
        dp.rects.append(DRule(r.x * k, r.y * k, r.w * k, r.h * k, r.color))
    for c in g.crops:
        h, w = c.image.shape[:2]
        buf = io.BytesIO()
        Image.fromarray(c.image).save(buf, "PNG", optimize=True)
        dp.pictures.append(DPicture(c.x * k, c.y * k, w * k, h * k, buf.getvalue()))
    return dp
