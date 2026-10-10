"""iLovePDF-like conversion API (FastAPI).

  Office  -> PDF   POST /api/v1/word-to-pdf
  PDF     -> Word  POST /api/v1/pdf-to-word
  Images  -> PDF   POST /api/v1/image-to-pdf
  PDF     -> image POST /api/v1/pdf-to-image
  OCR  image/scan -> regenerated PDF   POST /api/v1/ocr/pdf
  OCR  image/scan -> editable Word     POST /api/v1/ocr/word
  OCR  image/scan -> text / layout     POST /api/v1/ocr/text , /api/v1/ocr/json
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from . import converters, gotenberg
from .docmodel import from_ocr
from .ocr import DEFAULT_LANGS
from .pipeline import InputError, analyse_all, load_pages, regenerate_pdf
from .render_docx import build_docx

__version__ = "1.0.0"
MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB", "100"))
LANG_RE_CHARS = set("abcdefghijklmnopqrstuvwxyz_+")

app = FastAPI(
    title="LovePDF services",
    version=__version__,
    description="Self-hosted iLovePDF-like API: Word/Image/OCR to PDF and back, with "
                "first-class Arabic (right-to-left, correctly shaped) support. Rendering "
                "is done by Gotenberg (Chromium + LibreOffice).",
)


@app.exception_handler(InputError)
async def _input_error(_: Request, exc: InputError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(gotenberg.GotenbergError)
async def _gotenberg_error(_: Request, exc: gotenberg.GotenbergError):
    return JSONResponse(status_code=502, content={"detail": str(exc)})


# ------------------------------------------------------------------ helpers
async def _read(f: UploadFile) -> bytes:
    data = await f.read()
    if not data:
        raise InputError(f"empty file: {f.filename}")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"file larger than {MAX_UPLOAD_MB:g} MB")
    return data


def _file(content: bytes, filename: str, media: str) -> Response:
    disp = f"attachment; filename=\"{filename.encode('ascii', 'replace').decode()}\"; " \
           f"filename*=UTF-8''{quote(filename)}"
    return Response(content, media_type=media, headers={"Content-Disposition": disp})


def _stem(name: str | None, default: str = "document") -> str:
    return Path(name or default).stem or default


def _langs(lang: str) -> str:
    lang = lang.strip().lower()
    if not lang or not set(lang) <= LANG_RE_CHARS:
        raise InputError("lang must look like 'ara+eng'")
    return lang


DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


# ------------------------------------------------------------------ meta
@app.get("/health")
async def health():
    try:
        g = await gotenberg.health()
        status = "up" if g.get("status") == "up" else "degraded"
    except Exception as e:  # noqa: BLE001
        g, status = {"error": str(e)}, "degraded"
    return {"status": status, "version": __version__, "gotenberg": g}


@app.get("/")
async def root():
    return {
        "service": "lovepdf-services",
        "version": __version__,
        "docs": "/docs",
        "endpoints": {
            "POST /api/v1/word-to-pdf": "Word / Office document -> PDF (LibreOffice in Gotenberg)",
            "POST /api/v1/pdf-to-word": "PDF -> editable .docx (RTL aware, OCR for scanned pages)",
            "POST /api/v1/image-to-pdf": "one or more images -> PDF (lossless)",
            "POST /api/v1/pdf-to-image": "PDF -> PNG/JPG/WEBP/TIFF (zip when several pages)",
            "POST /api/v1/ocr/pdf": "image or scanned PDF -> regenerated PDF (real text, same look)",
            "POST /api/v1/ocr/word": "image or scanned PDF -> editable .docx",
            "POST /api/v1/ocr/text": "image or scanned PDF -> plain text",
            "POST /api/v1/ocr/json": "image or scanned PDF -> words, boxes, fonts, colours",
        },
    }


# ------------------------------------------------------------------ office <-> pdf
@app.post("/api/v1/word-to-pdf", summary="Word / Office document to PDF")
async def word_to_pdf(
    file: Annotated[UploadFile, File(description=".docx .doc .odt .rtf .txt .xlsx .pptx ...")],
    pdfa: Annotated[Literal["PDF/A-1b", "PDF/A-2b", "PDF/A-3b"] | None, Form()] = None,
    landscape: Annotated[bool | None, Form()] = None,
    page_ranges: Annotated[str | None, Form(description="e.g. 1-3,5")] = None,
):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in converters.OFFICE_EXTENSIONS:
        raise InputError(f"unsupported document type '{ext}'")
    data = await _read(file)
    pdf = await converters.office_to_pdf(f"document{ext}", data, pdfa, landscape, page_ranges)
    return _file(pdf, f"{_stem(file.filename)}.pdf", "application/pdf")


@app.post("/api/v1/pdf-to-word", summary="PDF to editable Word (.docx)")
async def pdf_to_word(
    file: Annotated[UploadFile, File(description="PDF document")],
    mode: Annotated[Literal["auto", "text", "ocr"], Form(
        description="auto: OCR pages without text layer; text: never OCR; ocr: always OCR")] = "auto",
    lang: Annotated[str, Form(description="tesseract languages for OCR pages")] = DEFAULT_LANGS,
):
    data = await _read(file)
    docx = await converters.pdf_to_docx(data, mode, _langs(lang))
    return _file(docx, f"{_stem(file.filename)}.docx", DOCX)


# ------------------------------------------------------------------ images <-> pdf
@app.post("/api/v1/image-to-pdf", summary="Images to PDF (lossless, one page per image)")
async def image_to_pdf(
    files: Annotated[list[UploadFile], File(description="JPG / PNG / TIFF / WEBP / BMP / GIF ...")],
    page_size: Annotated[Literal["fit", "A4", "A3", "A5", "Letter", "Legal"], Form()] = "fit",
    orientation: Annotated[Literal["auto", "portrait", "landscape"], Form()] = "auto",
    margin: Annotated[float, Form(ge=0, le=144, description="margin in points")] = 0,
):
    images = [await _read(f) for f in files]
    pdf = converters.images_to_pdf(images, page_size, orientation, margin)
    return _file(pdf, f"{_stem(files[0].filename, 'images')}.pdf", "application/pdf")


@app.post("/api/v1/pdf-to-image", summary="PDF pages to images")
async def pdf_to_image(
    file: Annotated[UploadFile, File(description="PDF document")],
    format: Annotated[Literal["png", "jpg", "jpeg", "webp", "tiff"], Form()] = "png",
    dpi: Annotated[int, Form(ge=36, le=1200)] = 150,
    pages: Annotated[str | None, Form(description="e.g. 1-3,5 (default: all)")] = None,
    quality: Annotated[int, Form(ge=1, le=100)] = 92,
):
    data = await _read(file)
    content, media, ext = converters.pdf_to_images(data, format, dpi, pages, quality)
    return _file(content, f"{_stem(file.filename)}.{ext}", media)


# ------------------------------------------------------------------ OCR
async def _analyse(file: UploadFile, lang: str, dpi: float | None, vectorize: bool = True,
                   deskew: bool | str = False):
    data = await _read(file)
    pages = load_pages(data, dpi)
    return await analyse_all(pages, _langs(lang), vectorize=vectorize, deskew=deskew)


DESKEW_HELP = "straighten tilted scans / photos first (the output is then not tilted like the input)"


@app.post("/api/v1/ocr/pdf", summary="OCR an image / scanned PDF into a regenerated PDF")
async def ocr_pdf(
    file: Annotated[UploadFile, File(description="image (jpg, png, tiff...) or scanned PDF")],
    lang: Annotated[str, Form()] = DEFAULT_LANGS,
    background: Annotated[Literal["vector", "original", "none"], Form(
        description="vector: graphics rebuilt as vectors + image crops (default); original: the "
                    "text-free scan as background; none: text only on white")] = "vector",
    dpi: Annotated[float | None, Form(ge=50, le=1200, description="override scan resolution")] = None,
    deskew: Annotated[bool, Form(description=DESKEW_HELP)] = False,
):
    analysed = await _analyse(file, lang, dpi, vectorize=(background == "vector"), deskew=deskew)
    pdf = await regenerate_pdf(analysed, background)
    return _file(pdf, f"{_stem(file.filename)}.pdf", "application/pdf")


@app.post("/api/v1/ocr/word", summary="OCR an image / scanned PDF into an editable .docx")
async def ocr_word(
    file: Annotated[UploadFile, File()],
    lang: Annotated[str, Form()] = DEFAULT_LANGS,
    dpi: Annotated[float | None, Form(ge=50, le=1200)] = None,
    deskew: Annotated[Literal["auto", "true", "false"], Form(
        description="auto (default): straighten pages tilted by 0.3 degree or more - a Word "
                    "document cannot hold tilted text lines")] = "auto",
):
    mode = {"auto": "auto", "true": True, "false": False}[deskew]
    analysed = await _analyse(file, lang, dpi, deskew=mode)
    return _file(build_docx([from_ocr(a) for a in analysed]), f"{_stem(file.filename)}.docx", DOCX)


@app.post("/api/v1/ocr/text", summary="OCR to plain text")
async def ocr_text(
    file: Annotated[UploadFile, File()],
    lang: Annotated[str, Form()] = DEFAULT_LANGS,
    dpi: Annotated[float | None, Form(ge=50, le=1200)] = None,
    deskew: Annotated[bool, Form(description=DESKEW_HELP)] = False,
):
    analysed = await _analyse(file, lang, dpi, vectorize=False, deskew=deskew)
    text = "\n\f\n".join(a.ocr.text for a in analysed)
    return Response(text, media_type="text/plain; charset=utf-8")


@app.post("/api/v1/ocr/json", summary="OCR to structured layout (words, boxes, fonts, colours)")
async def ocr_json(
    file: Annotated[UploadFile, File()],
    lang: Annotated[str, Form()] = DEFAULT_LANGS,
    dpi: Annotated[float | None, Form(ge=50, le=1200)] = None,
    deskew: Annotated[bool, Form(description=DESKEW_HELP)] = False,
):
    analysed = await _analyse(file, lang, dpi, vectorize=False, deskew=deskew)
    out = []
    for a in analysed:
        o = a.ocr
        out.append({
            "width": o.width, "height": o.height, "dpi": o.dpi,
            "lines": [{
                "bbox": ln.bbox, "rtl": ln.rtl, "paragraph": ln.par, "text": ln.text,
                "words": [{
                    "text": w.text, "bbox": w.bbox, "confidence": w.conf,
                    "font": w.face_key, "bold": w.bold, "size_px": round(w.font_px, 2),
                    "color": "#%02x%02x%02x" % tuple(w.color), "baseline": round(w.baseline, 2),
                } for w in ln.words],
            } for ln in o.lines],
        })
    return {"pages": out}
