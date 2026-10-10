"""OCR regeneration pipeline: image / scanned PDF -> pages -> OCR -> style recovery
-> graphics decomposition -> HTML -> Gotenberg (Chromium) -> regenerated PDF."""
from __future__ import annotations

import asyncio
import io
import os
from dataclasses import dataclass

import numpy as np
import pymupdf
from PIL import Image, ImageOps

from . import gotenberg
from .graphics import Graphics, analyse
from .layout import style_page
from .ocr import DEFAULT_LANGS, OcrPage, estimate_skew, ocr_image, rotate_page
from .render_html import build_page, document

MAX_PAGES = int(os.environ.get("MAX_PAGES", "50"))
PDF_RASTER_DPI = int(os.environ.get("PDF_RASTER_DPI", "300"))
Image.MAX_IMAGE_PIXELS = int(os.environ.get("MAX_IMAGE_PIXELS", str(200_000_000)))


@dataclass
class PageImage:
    rgb: np.ndarray
    dpi: float


class InputError(ValueError):
    pass


A4_WIDTH_IN = 8.27


def _guess_dpi(img: Image.Image, width_px: int) -> float:
    dpi = img.info.get("dpi")
    if dpi and 50 <= float(dpi[0]) <= 2400:
        # exif / jfif density; ignore the meaningless 72/96 defaults of phones/screens
        d = float(dpi[0])
        if d not in (72.0, 96.0) or width_px / d < 14:
            return d
    # unknown: assume the picture is an A4-wide page
    return max(72.0, width_px / A4_WIDTH_IN)


def is_pdf(data: bytes) -> bool:
    return data[:1024].lstrip().startswith(b"%PDF")


def scan_dpi(page: pymupdf.Page) -> float | None:
    """Native resolution of a page that is one full-page scan (no resampling, so
    OCR sees the original pixels)."""
    infos = page.get_image_info()
    if len(infos) != 1:
        return None
    x0, y0, x1, y1 = infos[0]["bbox"]
    area = page.rect.width * page.rect.height
    if (x1 - x0) * (y1 - y0) < 0.85 * area or not infos[0].get("width"):
        return None
    d = infos[0]["width"] / ((x1 - x0) / 72)
    return float(min(600.0, max(150.0, d)))


def load_pages(data: bytes, dpi: float | None = None) -> list[PageImage]:
    """Decode an image (any Pillow format, multi-frame TIFF included) or a PDF into
    RGB page rasters with their resolution."""
    pages: list[PageImage] = []
    if is_pdf(data):
        doc = pymupdf.open(stream=data, filetype="pdf")
        if doc.page_count > MAX_PAGES:
            raise InputError(f"too many pages ({doc.page_count} > {MAX_PAGES})")
        for p in doc:
            rdpi = int(round(dpi or scan_dpi(p) or PDF_RASTER_DPI))
            pix = p.get_pixmap(dpi=rdpi, colorspace=pymupdf.csRGB, alpha=False)
            arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3).copy()
            pages.append(PageImage(arr, rdpi))
        return pages
    try:
        img = Image.open(io.BytesIO(data))
    except Exception as e:  # noqa: BLE001
        raise InputError(f"unsupported image: {e}") from e
    n = getattr(img, "n_frames", 1)
    if n > MAX_PAGES:
        raise InputError(f"too many frames ({n} > {MAX_PAGES})")
    for i in range(n):
        img.seek(i)
        fr = ImageOps.exif_transpose(img.copy())
        if fr.mode in ("RGBA", "LA", "P"):
            fr = fr.convert("RGBA")
            bg = Image.new("RGBA", fr.size, (255, 255, 255, 255))
            fr = Image.alpha_composite(bg, fr)
        arr = np.asarray(fr.convert("RGB")).copy()
        pages.append(PageImage(arr, dpi or _guess_dpi(img, arr.shape[1])))
    return pages


@dataclass
class AnalysedPage:
    image: PageImage
    ocr: OcrPage
    gfx: Graphics


def analyse_page(pi: PageImage, langs: str, vectorize: bool = True,
                 deskew: bool | str = False) -> AnalysedPage:
    if deskew:
        angle = estimate_skew(pi.rgb)
        # "auto" (Word export): straighten only clearly tilted pages
        if deskew is True or abs(angle) >= 0.3:
            pi = PageImage(rotate_page(pi.rgb, angle), pi.dpi)
    page = ocr_image(pi.rgb, pi.dpi, langs)
    style_page(pi.rgb, page)
    gfx = analyse(pi.rgb, page, vectorize=vectorize)
    return AnalysedPage(pi, page, gfx)


async def analyse_all(pages: list[PageImage], langs: str, vectorize: bool = True,
                      deskew: bool | str = False) -> list[AnalysedPage]:
    sem = asyncio.Semaphore(int(os.environ.get("OCR_WORKERS", "2")))

    async def one(p):
        async with sem:
            return await asyncio.to_thread(analyse_page, p, langs, vectorize, deskew)

    return await asyncio.gather(*(one(p) for p in pages))


async def regenerate_pdf(pages: list[AnalysedPage], background: str = "vector") -> bytes:
    async def one(i: int, ap: AnalysedPage) -> bytes:
        svg, assets, faces = build_page(ap.ocr, ap.gfx, background, prefix=f"p{i}")
        w_pt = ap.ocr.width * 72 / ap.ocr.dpi
        h_pt = ap.ocr.height * 72 / ap.ocr.dpi
        html = document(svg, w_pt, h_pt, faces, assets)
        return await gotenberg.html_to_pdf(html, assets, wait_expression="window.__ready === true")

    pdfs = await asyncio.gather(*(one(i, ap) for i, ap in enumerate(pages)))
    return merge_pdfs(list(pdfs))


def merge_pdfs(pdfs: list[bytes]) -> bytes:
    if len(pdfs) == 1:
        return pdfs[0]
    out = pymupdf.open()
    for b in pdfs:
        with pymupdf.open(stream=b, filetype="pdf") as d:
            out.insert_pdf(d)
    return out.tobytes(garbage=3, deflate=True)


async def ocr_to_pdf(data: bytes, langs: str = DEFAULT_LANGS, background: str = "vector",
                     dpi: float | None = None, deskew: bool = False) -> bytes:
    pages = load_pages(data, dpi)
    analysed = await analyse_all(pages, langs, vectorize=(background == "vector"), deskew=deskew)
    return await regenerate_pdf(analysed, background)
