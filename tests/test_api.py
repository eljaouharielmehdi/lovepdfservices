"""End-to-end tests against the running docker-compose stack (BASE_URL).

Every conversion is checked for content (text in logical order, Arabic
included) and for looks (rendered pages compared pixel-wise with the source).
"""
import difflib
import io
import zipfile

import numpy as np
import pymupdf
import pytest
from docx import Document
from PIL import Image

from conftest import FIX, pdf_text, render, save_side_by_side, squash, visual_match

DOCX_MT = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def post(api, path, files, **data):
    r = api.post(path, files=files, data={k: str(v) for k, v in data.items()})
    assert r.status_code == 200, f"{path}: {r.status_code} {r.text[:500]}"
    return r


def ocr_readback(api, pdf: bytes, dpi: int = 200) -> str:
    """Rasterise a PDF and OCR it again: proves the *rendered* Arabic glyphs are
    correctly shaped and ordered (scrambled / disconnected Arabic does not OCR)."""
    img = Image.fromarray(render(pdf, dpi))
    buf = io.BytesIO()
    img.save(buf, "PNG", dpi=(dpi, dpi))
    return post(api, "/api/v1/ocr/text", {"file": ("page.png", buf.getvalue())}).text


def docx_text(data: bytes) -> str:
    d = Document(io.BytesIO(data))
    parts = [p.text for p in d.paragraphs]
    for t in d.tables:
        for row in t.rows:
            for c in row.cells:
                parts.append(c.text)
    return "\n".join(parts)


def docx_baseline_error(api, docx: bytes, reference_pdf: bytes) -> tuple[float, float]:
    """Render the .docx back to PDF and compare every line baseline with the source."""
    pdf = post(api, "/api/v1/word-to-pdf", {"file": ("x.docx", docx)}).content

    def lines(b):
        out = []
        for blk in pymupdf.open(stream=b, filetype="pdf")[0].get_text("dict")["blocks"]:
            for ln in blk.get("lines", []):
                if "".join(s["text"] for s in ln["spans"]).strip():
                    out.append((ln["spans"][0]["origin"][1], ln["bbox"][0]))
        return out

    ref, got = lines(reference_pdf), lines(pdf)
    errs = [min(abs(g[0] - r[0]) for g in got) for r in ref]
    return max(errs), float(np.mean(errs))


# ===================================================================== meta
def test_health(api):
    j = api.get("/health").json()
    assert j["status"] == "up"
    assert j["gotenberg"]["details"]["chromium"]["status"] == "up"
    assert j["gotenberg"]["details"]["libreoffice"]["status"] == "up"


def test_bad_input_is_rejected(api):
    r = api.post("/api/v1/ocr/pdf", files={"file": ("x.png", b"not an image")})
    assert r.status_code == 400
    r = api.post("/api/v1/pdf-to-image", files={"file": ("x.pdf", b"%PDF-broken")})
    assert r.status_code == 400
    r = api.post("/api/v1/word-to-pdf", files={"file": ("x.exe", b"MZ")})
    assert r.status_code == 400


# ===================================================================== word -> pdf
def test_word_to_pdf_arabic(api, sample_pdf, truth, out_dir):
    doc = pymupdf.open(stream=sample_pdf, filetype="pdf")
    assert doc.page_count == 1
    fonts = {f[3] for f in doc[0].get_fonts()}
    assert any("Naskh" in f for f in fonts), fonts          # Arabic font embedded
    text = squash(pdf_text(sample_pdf))
    for phrase in truth["sample_ar.docx"]:
        assert squash(phrase) in text, phrase                # logical order text layer
    # the rendered page must OCR back to the same Arabic: shaping + RTL are right
    back = squash(ocr_readback(api, sample_pdf))
    for phrase in ["تقرير الخدمات الرقمية", "تهدف هذه الوثيقة", "من اليمين إلى",
                   "داخل النص", "This paragraph is written in English"]:
        assert squash(phrase) in back, phrase
    Image.fromarray(render(sample_pdf, 100)).save(out_dir / "word_to_pdf.png")


# ===================================================================== pdf -> word
def test_pdf_to_word_arabic(api, sample_pdf, truth, out_dir):
    r = post(api, "/api/v1/pdf-to-word", {"file": ("sample_ar.pdf", sample_pdf)})
    assert r.headers["content-type"] == DOCX_MT
    (out_dir / "pdf_to_word.docx").write_bytes(r.content)
    d = Document(io.BytesIO(r.content))
    text = docx_text(r.content)
    # exact logical text, punctuation included
    assert "تقرير الخدمات الرقمية" in text
    assert "يحتوي هذا السطر على أرقام 2024 وكلمات إنجليزية مثل PDF و Word داخل النص العربي." in text
    assert "This paragraph is written in English to verify mixed-direction documents." in text
    # right-to-left paragraphs and runs
    xml = d.element.xml
    assert "<w:bidi/>" in xml and "<w:rtl/>" in xml
    # the table is a real Word table, right-to-left, 3 x 3
    assert len(d.tables) == 1
    t = d.tables[0]
    assert len(t.rows) == 3 and len(t.columns) == 3
    assert "<w:bidiVisual/>" in t._tbl.xml
    assert [c.text for c in t.rows[0].cells] == ["الاسم", "الكمية", "السعر"]
    assert [c.text for c in t.rows[2].cells] == ["دفتر", "2", "40"]
    # layout: every line baseline lands where it was in the PDF
    worst, mean = docx_baseline_error(api, r.content, sample_pdf)
    assert worst < 2.0 and mean < 1.0, (worst, mean)


# ===================================================================== image -> pdf
def test_image_to_pdf_lossless_jpeg(api):
    jpg = (FIX / "photo.jpg").read_bytes()
    pdf = post(api, "/api/v1/image-to-pdf", {"files": ("photo.jpg", jpg)}).content
    doc = pymupdf.open(stream=pdf, filetype="pdf")
    assert doc.page_count == 1
    xref = doc[0].get_images()[0][0]
    assert doc.xref_stream_raw(xref) == jpg               # JPEG embedded bit-exact


def test_image_to_pdf_png_pixels_and_size(api):
    png = (FIX / "invoice_ar.png").read_bytes()
    pdf = post(api, "/api/v1/image-to-pdf", {"files": ("invoice.png", png)}).content
    doc = pymupdf.open(stream=pdf, filetype="pdf")
    page = doc[0]
    src = Image.open(io.BytesIO(png))
    dpi = src.info["dpi"][0]
    assert abs(page.rect.width - src.width * 72 / dpi) < 1      # physical size kept
    assert abs(page.rect.height - src.height * 72 / dpi) < 1
    info = doc.extract_image(page.get_images()[0][0])
    emb = np.asarray(Image.open(io.BytesIO(info["image"])).convert("RGB"))
    assert np.array_equal(emb, np.asarray(src.convert("RGB")))  # pixel exact


def test_images_to_pdf_multi_a4(api):
    files = [("files", ("a.jpg", (FIX / "photo.jpg").read_bytes())),
             ("files", ("b.png", (FIX / "letter_ar.png").read_bytes()))]
    pdf = post(api, "/api/v1/image-to-pdf", files, page_size="A4", margin=20).content
    doc = pymupdf.open(stream=pdf, filetype="pdf")
    assert doc.page_count == 2
    # landscape photo -> landscape A4 (auto orientation), portrait letter -> portrait
    assert round(doc[0].rect.width) == 842 and round(doc[0].rect.height) == 595
    assert round(doc[1].rect.width) == 595 and round(doc[1].rect.height) == 842


# ===================================================================== pdf -> image
def test_pdf_to_image_single_and_zip(api, sample_pdf):
    r = post(api, "/api/v1/pdf-to-image", {"file": ("s.pdf", sample_pdf)}, format="png", dpi=144)
    assert r.headers["content-type"] == "image/png"
    img = Image.open(io.BytesIO(r.content))
    page = pymupdf.open(stream=sample_pdf, filetype="pdf")[0]
    assert abs(img.width - page.rect.width * 2) <= 1
    ref = render(sample_pdf, 144)
    assert np.array_equal(np.asarray(img.convert("RGB")), ref)

    two = pymupdf.open()
    two.insert_pdf(pymupdf.open(stream=sample_pdf, filetype="pdf"))
    two.insert_pdf(pymupdf.open(stream=sample_pdf, filetype="pdf"))
    r = post(api, "/api/v1/pdf-to-image", {"file": ("two.pdf", two.tobytes())}, format="jpg", dpi=72)
    assert r.headers["content-type"] == "application/zip"
    names = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
    assert names == ["page-001.jpg", "page-002.jpg"]


# ===================================================================== OCR -> regenerated PDF
@pytest.mark.parametrize("name", ["invoice_ar.png", "letter_ar.png"])
def test_ocr_regenerated_pdf_looks_like_original(api, truth, out_dir, name):
    src = (FIX / name).read_bytes()
    r = post(api, "/api/v1/ocr/pdf", {"file": (name, src)})
    pdf = r.content
    (out_dir / f"ocr_{name}.pdf").write_bytes(pdf)
    doc = pymupdf.open(stream=pdf, filetype="pdf")
    page = doc[0]
    orig = np.asarray(Image.open(io.BytesIO(src)).convert("RGB"))
    dpi = Image.open(io.BytesIO(src)).info["dpi"][0]

    # 1. same physical page size as the scan
    assert abs(page.rect.width - orig.shape[1] * 72 / dpi) < 1
    # 2. regenerated, not a picture: real embedded fonts, and no image covers the page
    assert any("Naskh" in f[3] for f in page.get_fonts())
    for info in page.get_image_info():
        x0, y0, x1, y1 = info["bbox"]
        assert (x1 - x0) * (y1 - y0) < 0.2 * page.rect.width * page.rect.height
    # 3. text layer: every expected phrase, in logical order
    text = squash(pdf_text(pdf))
    for phrase in truth[name]:
        assert squash(phrase) in text, phrase
    # 4. looks the same: ink overlap and colour error against the original scan
    regen = render(pdf, dpi)
    m = visual_match(orig, regen)
    save_side_by_side(orig, regen, out_dir / f"ocr_{name}_compare.png")
    assert m["precision"] > 0.93 and m["recall"] > 0.93, m
    assert m["mae"] < 6.0, m
    # 5. the regenerated glyphs OCR back to the same text as the original scan does
    #    (broken shaping / reversed Arabic would collapse this similarity)
    first = squash(post(api, "/api/v1/ocr/text", {"file": (name, src)}).text)
    back = squash(ocr_readback(api, pdf, round(dpi)))
    ratio = difflib.SequenceMatcher(None, first, back).ratio()
    assert ratio > 0.97, (ratio, first, back)


def test_ocr_pdf_from_scanned_pdf_and_background_modes(api, out_dir):
    # scanned pdf input (image inside a pdf)
    scan = post(api, "/api/v1/image-to-pdf", {"files": ("inv.png", (FIX / "invoice_ar.png").read_bytes())}).content
    for bg in ("vector", "original", "none"):
        pdf = post(api, "/api/v1/ocr/pdf", {"file": ("scan.pdf", scan)}, background=bg).content
        (out_dir / f"ocr_scan_{bg}.pdf").write_bytes(pdf)
        text = squash(pdf_text(pdf))
        assert squash("المجموع الكلي") in text and "19750" in text


# ===================================================================== OCR -> word / text / json
def test_ocr_to_word(api, truth, out_dir):
    src = (FIX / "invoice_ar.png").read_bytes()
    r = post(api, "/api/v1/ocr/word", {"file": ("invoice_ar.png", src)})
    (out_dir / "ocr_invoice.docx").write_bytes(r.content)
    d = Document(io.BytesIO(r.content))
    text = docx_text(r.content)
    for phrase in ["فاتورة ضريبية", "اسم العميل: شركة النور للتجارة", "العنوان: الرباط، المغرب",
                   "تاريخ الاستحقاق 15 مارس 2025", "المجموع الكلي: 19750 درهم",
                   "Thank you for your business. Invoice No. 2025-0042"]:
        assert phrase in text, phrase
    assert len(d.tables) == 1
    t = d.tables[0]
    assert len(t.rows) == 4 and len(t.columns) == 3
    assert [c.text for c in t.rows[0].cells] == ["المنتج", "الكمية", "السعر"]
    assert [c.text for c in t.rows[1].cells] == ["حاسوب محمول", "2", "8500"]
    assert "<w:bidiVisual/>" in t._tbl.xml
    # header cells keep their shading, the band / logo are on the page background
    assert 'w:fill="CF' in t._tbl.xml or 'w:fill="D0' in t._tbl.xml
    assert "behindDoc=\"1\"" in d.element.xml
    # rendered back, it looks like the scan
    pdf = post(api, "/api/v1/word-to-pdf", {"file": ("x.docx", r.content)}).content
    orig = np.asarray(Image.open(io.BytesIO(src)).convert("RGB"))
    regen = render(pdf, 200)
    save_side_by_side(orig, regen, out_dir / "ocr_word_compare.png")
    m = visual_match(orig, regen, tol=4)
    assert m["precision"] > 0.85 and m["recall"] > 0.85, m


def test_ocr_text_and_json(api, truth):
    src = (FIX / "letter_ar.png").read_bytes()
    text = post(api, "/api/v1/ocr/text", {"file": ("l.png", src)}).text
    assert "تقرير الخدمات الرقمية" in text
    assert "يحتوي هذا السطر على أرقام 2024 وكلمات إنجليزية مثل PDF و Word داخل النص" in text
    j = post(api, "/api/v1/ocr/json", {"file": ("l.png", src)}).json()
    lines = j["pages"][0]["lines"]
    title = lines[0]
    assert title["rtl"] and title["text"] == "تقرير الخدمات الرقمية"
    assert all(w["bold"] for w in title["words"])
    assert title["words"][0]["color"].startswith("#")


def test_pdf_to_word_scanned_pdf_uses_ocr(api):
    scan = post(api, "/api/v1/image-to-pdf", {"files": ("l.png", (FIX / "letter_ar.png").read_bytes())}).content
    r = post(api, "/api/v1/pdf-to-word", {"file": ("scan.pdf", scan)})
    text = docx_text(r.content)
    assert "تقرير الخدمات الرقمية" in text
    assert "This paragraph is written in English to verify mixed-direction documents." in text


# ===================================================================== hard input
def test_ocr_phone_capture(api, out_dir):
    """Tilted, noisy, JPEG q70, ~160 dpi, no DPI metadata."""
    src = (FIX / "phone_invoice.jpg").read_bytes()
    for deskew in (False, True):
        pdf = post(api, "/api/v1/ocr/pdf", {"file": ("phone.jpg", src)}, deskew=deskew).content
        (out_dir / f"ocr_phone_deskew_{deskew}.pdf").write_bytes(pdf)
        text = squash(pdf_text(pdf))
        for phrase in ["فاتورة ضريبية", "اسم العميل", "شركة النور للتجارة", "الرباط", "المغرب",
                       "حاسوب محمول", "طابعة ليزر", "المجموع الكلي", "19750", "8500", "2300",
                       "Thank you for your business", "2025-0042"]:
            assert squash(phrase) in text, (deskew, phrase)
        page = pymupdf.open(stream=pdf, filetype="pdf")[0]
        # the round logo must stay a picture, never become a giant letter
        sizes = [s["size"] for b in page.get_text("dict")["blocks"] for ln in b.get("lines", [])
                 for s in ln["spans"]]
        assert max(sizes) < 40, max(sizes)
        if not deskew:   # without deskew the output must look exactly like the photo
            orig = np.asarray(Image.open(io.BytesIO(src)).convert("RGB"))
            dpi = orig.shape[1] / (page.rect.width / 72)
            regen = render(pdf, dpi)
            save_side_by_side(orig, regen, out_dir / "ocr_phone_compare.png")
            m = visual_match(orig, regen, tol=3)
            assert m["precision"] > 0.9 and m["recall"] > 0.9, m
