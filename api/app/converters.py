"""Image <-> PDF, PDF -> image, Office <-> PDF converters."""
from __future__ import annotations

import io
import zipfile

import img2pdf
import numpy as np
import pymupdf
from PIL import Image

from . import gotenberg
from .docmodel import from_ocr, from_pdf_page
from .pipeline import InputError, MAX_PAGES, PageImage, analyse_all, is_pdf, scan_dpi
from .render_docx import build_docx

PAGE_SIZES = {  # points
    "A4": (595.276, 841.89),
    "A3": (841.89, 1190.551),
    "A5": (419.528, 595.276),
    "LETTER": (612.0, 792.0),
    "LEGAL": (612.0, 1008.0),
}

OFFICE_EXTENSIONS = {
    ".doc", ".docx", ".docm", ".dot", ".dotx", ".odt", ".ott", ".rtf", ".txt", ".wpd", ".wps",
    ".xls", ".xlsx", ".ods", ".csv", ".ppt", ".pptx", ".odp", ".html", ".htm", ".xml", ".epub",
}


# ------------------------------------------------------------------ image -> pdf
def _normalise_image(data: bytes) -> list[bytes]:
    """Return image streams img2pdf can embed losslessly. JPEG / JPEG2000 / PNG
    without alpha / TIFF pass through untouched (bit-exact); images with alpha
    are flattened on white; EXIF rotation is applied by img2pdf."""
    try:
        img = Image.open(io.BytesIO(data))
    except Exception as e:  # noqa: BLE001
        raise InputError(f"unsupported image: {e}") from e
    fmt = (img.format or "").upper()
    frames = getattr(img, "n_frames", 1)
    if frames > MAX_PAGES:
        raise InputError(f"too many frames ({frames} > {MAX_PAGES})")
    has_alpha = img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info)
    if fmt in ("JPEG", "JPEG2000") or (fmt in ("PNG", "TIFF", "GIF") and not has_alpha and frames == 1):
        return [data]
    out = []
    for i in range(frames):
        img.seek(i)
        fr = img.copy()
        if fr.mode in ("RGBA", "LA", "PA", "P"):
            fr = fr.convert("RGBA")
            bg = Image.new("RGBA", fr.size, (255, 255, 255, 255))
            fr = Image.alpha_composite(bg, fr).convert("RGB")
        elif fr.mode not in ("RGB", "L", "1", "CMYK"):
            fr = fr.convert("RGB")
        buf = io.BytesIO()
        fr.save(buf, "PNG", dpi=img.info.get("dpi", (96, 96)))
        out.append(buf.getvalue())
    return out


def images_to_pdf(images: list[bytes], page_size: str = "fit", orientation: str = "auto",
                  margin: float = 0.0) -> bytes:
    streams: list[bytes] = []
    for data in images:
        streams.extend(_normalise_image(data))
    if not streams:
        raise InputError("no image given")
    if len(streams) > MAX_PAGES:
        raise InputError(f"too many pages ({len(streams)} > {MAX_PAGES})")
    key = page_size.upper()
    kwargs: dict = {"rotation": img2pdf.Rotation.ifvalid}
    border = (margin, margin) if margin > 0 else None
    if key == "FIT":
        if border:
            kwargs["layout_fun"] = img2pdf.get_layout_fun(None, None, border=border)
    else:
        if key not in PAGE_SIZES:
            raise InputError(f"page_size must be fit or one of {', '.join(PAGE_SIZES)}")
        w, h = PAGE_SIZES[key]
        if orientation == "landscape":
            w, h = h, w
        kwargs["layout_fun"] = img2pdf.get_layout_fun(
            (w, h), None, border=border, fit=img2pdf.FitMode.into,
            auto_orient=(orientation == "auto"))
    try:
        return img2pdf.convert(streams, **kwargs)
    except Exception as e:  # noqa: BLE001
        raise InputError(f"cannot convert image: {e}") from e


# ------------------------------------------------------------------ pdf -> image
def parse_pages(spec: str | None, count: int) -> list[int]:
    if not spec:
        return list(range(count))
    pages: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo = int(a) if a else 1
            hi = int(b) if b else count
            pages.extend(range(lo - 1, min(hi, count)))
        else:
            pages.append(int(part) - 1)
    pages = [p for p in pages if 0 <= p < count]
    if not pages:
        raise InputError("no valid page selected")
    return pages


def pdf_to_images(pdf: bytes, fmt: str = "png", dpi: int = 150, pages: str | None = None,
                  quality: int = 92) -> tuple[bytes, str, str]:
    """Return (content, media type, extension). Several pages -> zip archive."""
    fmt = fmt.lower()
    if fmt == "jpeg":
        fmt = "jpg"
    if fmt not in ("png", "jpg", "webp", "tiff"):
        raise InputError("format must be png, jpg, webp or tiff")
    if not 36 <= dpi <= 1200:
        raise InputError("dpi must be between 36 and 1200")
    try:
        doc = pymupdf.open(stream=pdf, filetype="pdf")
    except Exception as e:  # noqa: BLE001
        raise InputError(f"invalid pdf: {e}") from e
    if doc.needs_pass:
        raise InputError("the pdf is password protected")
    sel = parse_pages(pages, doc.page_count)
    if len(sel) > MAX_PAGES:
        raise InputError(f"too many pages ({len(sel)} > {MAX_PAGES})")
    files = []
    for i in sel:
        pix = doc[i].get_pixmap(dpi=dpi, alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        buf = io.BytesIO()
        if fmt == "jpg":
            img.save(buf, "JPEG", quality=quality, dpi=(dpi, dpi), subsampling=0)
        elif fmt == "webp":
            img.save(buf, "WEBP", quality=quality)
        elif fmt == "tiff":
            img.save(buf, "TIFF", compression="tiff_deflate", dpi=(dpi, dpi))
        else:
            img.save(buf, "PNG", dpi=(dpi, dpi), optimize=False)
        files.append((f"page-{i + 1:03d}.{fmt}", buf.getvalue()))
    media = {"png": "image/png", "jpg": "image/jpeg", "webp": "image/webp", "tiff": "image/tiff"}[fmt]
    if len(files) == 1:
        return files[0][1], media, fmt
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files:
            z.writestr(name, data)
    return zbuf.getvalue(), "application/zip", "zip"


# ------------------------------------------------------------------ office <-> pdf
async def office_to_pdf(filename: str, data: bytes, pdfa: str | None = None,
                        landscape: bool | None = None, page_ranges: str | None = None) -> bytes:
    opts: dict = {}
    if pdfa:
        opts["pdfa"] = pdfa
    if landscape is not None:
        opts["landscape"] = landscape
    if page_ranges:
        opts["nativePageRanges"] = page_ranges
    return await gotenberg.office_to_pdf(filename, data, **opts)


def _page_is_scanned(page: pymupdf.Page) -> bool:
    text = page.get_text("text").strip()
    if len(text) >= 20:
        return False
    return bool(page.get_images()) or len(page.get_drawings()) > 0


async def pdf_to_docx(pdf: bytes, mode: str = "auto", langs: str = "ara+eng", dpi: int = 300) -> bytes:
    """mode: auto (OCR only pages without a text layer), text, ocr."""
    if not is_pdf(pdf):
        raise InputError("input is not a pdf")
    try:
        doc = pymupdf.open(stream=pdf, filetype="pdf")
    except Exception as e:  # noqa: BLE001
        raise InputError(f"invalid pdf: {e}") from e
    if doc.needs_pass:
        raise InputError("the pdf is password protected")
    if doc.page_count > MAX_PAGES:
        raise InputError(f"too many pages ({doc.page_count} > {MAX_PAGES})")
    plan = []
    for p in doc:
        use_ocr = mode == "ocr" or (mode == "auto" and _page_is_scanned(p))
        plan.append(use_ocr)
    ocr_imgs = []
    for p, use_ocr in zip(doc, plan):
        if use_ocr:
            rdpi = int(round(scan_dpi(p) or dpi))
            pix = p.get_pixmap(dpi=rdpi, colorspace=pymupdf.csRGB, alpha=False)
            arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3).copy()
            ocr_imgs.append(PageImage(arr, rdpi))
    analysed = iter(await analyse_all(ocr_imgs, langs) if ocr_imgs else [])
    pages = [from_ocr(next(analysed)) if use_ocr else from_pdf_page(p) for p, use_ocr in zip(doc, plan)]
    return build_docx(pages)
