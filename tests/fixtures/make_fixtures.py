"""Generate the Arabic / English test fixtures used by the test-suite.

Fixtures (written next to this file):
  sample_ar.docx      Word document: RTL Arabic title, paragraphs, table, mixed text
  invoice_ar.html     HTML source of a coloured Arabic invoice (rendered to an image)
  invoice_ar.png      "Scanned" invoice: colours, logo, table rules, Arabic + English
  letter_ar.png       "Scanned" letter rendered from sample_ar.docx at 200 DPI
  photo.jpg           Photographic-like JPEG (gradient + shapes) for image<->pdf
  ground_truth.json   Text that must be recovered by OCR

Usage:  python make_fixtures.py [--gotenberg http://localhost:3000]
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import pymupdf as fitz
import httpx
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor
from PIL import Image, ImageDraw, ImageFilter

HERE = Path(__file__).resolve().parent

AR_TITLE = "تقرير الخدمات الرقمية"
AR_P1 = (
    "تهدف هذه الوثيقة إلى اختبار تحويل المستندات العربية بشكل صحيح من اليمين إلى اليسار "
    "مع الحفاظ على تشكيل الحروف واتصالها وعدم تبعثر الكلمات."
)
AR_P2 = "يحتوي هذا السطر على أرقام 2024 وكلمات إنجليزية مثل PDF و Word داخل النص العربي."
EN_P = "This paragraph is written in English to verify mixed-direction documents."
TABLE = [["الاسم", "الكمية", "السعر"], ["قلم", "3", "15"], ["دفتر", "2", "40"]]


def _rtl_paragraph(p):
    ppr = p._p.get_or_add_pPr()
    bidi = OxmlElement("w:bidi")
    ppr.insert(0, bidi)
    for r in p.runs:
        rpr = r._r.get_or_add_rPr()
        rpr.append(OxmlElement("w:rtl"))
        fonts = rpr.find(qn("w:rFonts"))
        if fonts is None:
            fonts = OxmlElement("w:rFonts")
            rpr.insert(0, fonts)
        fonts.set(qn("w:cs"), "Noto Naskh Arabic")


def _run(p, text, size=14, bold=False, color=None):
    r = p.add_run(text)
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.name = "Noto Naskh Arabic"
    rpr = r._r.get_or_add_rPr()
    szcs = OxmlElement("w:szCs")
    szcs.set(qn("w:val"), str(int(size * 2)))
    rpr.append(szcs)
    if bold:
        rpr.append(OxmlElement("w:bCs"))
    if color:
        r.font.color.rgb = RGBColor.from_string(color)
    return r


def make_docx(path: Path) -> None:
    doc = Document()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _run(p, AR_TITLE, 24, bold=True, color="1F4E79")
    _rtl_paragraph(p)

    for text in (AR_P1, AR_P2):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        _run(p, text, 14)
        _rtl_paragraph(p)

    table = doc.add_table(rows=3, cols=3)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    tblpr = table._tbl.tblPr
    tblpr.append(OxmlElement("w:bidiVisual"))
    for i, row in enumerate(TABLE):
        for j, val in enumerate(row):
            cell = table.cell(i, j)
            cp = cell.paragraphs[0]
            cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
            _run(cp, val, 13, bold=(i == 0))
            _rtl_paragraph(cp)

    p = doc.add_paragraph()
    _run(p, EN_P, 12)
    doc.save(path)


INVOICE_HTML = """<!doctype html><html lang="ar" dir="rtl"><head><meta charset="utf-8">
<style>
@page { size: 210mm 297mm; margin: 0 }
body { margin:0; font-family:'Noto Naskh Arabic','Noto Sans',sans-serif; color:#222; }
.head { background:#0b5394; color:#fff; padding:28px 48px; display:flex;
        justify-content:space-between; align-items:center }
.head h1 { margin:0; font-size:40px }
.logo { width:90px; height:90px; border-radius:50%; background:#f1c232;
        border:6px solid #fff }
.body { padding:30px 48px; font-size:22px; line-height:1.7 }
.red { color:#c00000; font-weight:bold }
table { width:100%; border-collapse:collapse; margin-top:20px; font-size:22px }
th { background:#cfe2f3; }
th,td { border:2px solid #444; padding:10px 14px; text-align:center }
.en { direction:ltr; text-align:left; font-family:'Noto Sans',sans-serif; font-size:20px;
      color:#38761d }
.total { margin-top:24px; font-size:28px; font-weight:bold; color:#0b5394 }
</style></head><body>
<div class="head"><h1>فاتورة ضريبية</h1><div class="logo"></div></div>
<div class="body">
<p>اسم العميل: شركة النور للتجارة</p>
<p>العنوان: الرباط، المغرب</p>
<p class="red">تاريخ الاستحقاق 15 مارس 2025</p>
<table>
<tr><th>المنتج</th><th>الكمية</th><th>السعر</th></tr>
<tr><td>حاسوب محمول</td><td>2</td><td>8500</td></tr>
<tr><td>طابعة ليزر</td><td>1</td><td>2300</td></tr>
<tr><td>ورق طباعة</td><td>10</td><td>45</td></tr>
</table>
<p class="total">المجموع الكلي: 19750 درهم</p>
<p class="en">Thank you for your business. Invoice No. 2025-0042</p>
</div></body></html>"""

FORM_HTML = """<!doctype html><html lang="ar" dir="rtl"><head><meta charset="utf-8">
<style>
@page { size: 210mm 297mm; margin: 0 }
body { margin:0; padding:50px 60px; font-family:'Noto Naskh Arabic','Noto Sans',sans-serif;
       font-size:20px; color:#111 }
h2 { text-align:center; color:#7f6000; margin:0 0 30px }
.row { display:flex; justify-content:space-between; margin:14px 0 }
table { width:100%; margin-top:30px; border-collapse:collapse }
td { padding:8px 4px }
td.num { text-align:left; direction:ltr }
</style></head><body>
<h2>استمارة طلب خدمة</h2>
<div class="row"><span>الاسم: أحمد بن علي</span><span>التاريخ: 15/03/2025</span></div>
<div class="row"><span>المدينة: الدار البيضاء</span><span>الهاتف: 0612345678</span></div>
<div class="row"><span>رقم الملف: 2025-117</span><span>Reference: AB-77</span></div>
<table>
<tr><td><b>الخدمة</b></td><td><b>المدة</b></td><td class="num"><b>المبلغ</b></td></tr>
<tr><td>استشارة قانونية</td><td>ساعتان</td><td class="num">1200</td></tr>
<tr><td>ترجمة وثائق</td><td>ثلاثة أيام</td><td class="num">850</td></tr>
</table>
</body></html>"""

INVOICE_TRUTH = [
    "فاتورة ضريبية", "اسم العميل", "شركة النور للتجارة", "الرباط", "المغرب",
    "تاريخ الاستحقاق", "حاسوب محمول", "طابعة ليزر", "ورق طباعة", "المجموع الكلي",
    "19750", "8500", "2300", "Thank you for your business", "2025-0042",
]


SIGNATURE_SVG = (
    '<svg width="220" height="90" viewBox="0 0 220 90" xmlns="http://www.w3.org/2000/svg">'
    '<path d="M10 60 C 30 10, 50 10, 60 50 S 90 90, 110 40 S 150 5, 160 55 S 200 70, 210 30" '
    'fill="none" stroke="#1a3d7c" stroke-width="3" stroke-linecap="round"/></svg>')


def invoice_pdf_html(photo: Path) -> str:
    """The invoice plus an embedded photo and a drawn signature (digital PDF fixture)."""
    import base64

    b64 = base64.b64encode(photo.read_bytes()).decode()
    extra = (f'<div style="display:flex;justify-content:space-between;align-items:flex-end;'
             f'padding:10px 48px"><img src="data:image/jpeg;base64,{b64}" style="width:240px">'
             f'<div style="text-align:center;font-size:20px">{SIGNATURE_SVG}<br>التوقيع</div></div>')
    return INVOICE_HTML.replace("</div></body>", "</div>" + extra + "</body>")


def html_to_pdf(gotenberg: str, html: str) -> bytes:
    r = httpx.post(
        f"{gotenberg}/forms/chromium/convert/html",
        files={"index.html": ("index.html", html.encode(), "text/html")},
        data={"preferCssPageSize": "true", "printBackground": "true"},
        timeout=120,
    )
    r.raise_for_status()
    return r.content


def office_to_pdf(gotenberg: str, path: Path) -> bytes:
    r = httpx.post(
        f"{gotenberg}/forms/libreoffice/convert",
        files={"files": (path.name, path.read_bytes())},
        timeout=120,
    )
    r.raise_for_status()
    return r.content


def pdf_page_png(pdf: bytes, dpi: int, noise: bool = False) -> Image.Image:
    doc = fitz.open(stream=pdf, filetype="pdf")
    pix = doc[0].get_pixmap(dpi=dpi)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    if noise:  # mild scanner blur
        img = img.filter(ImageFilter.GaussianBlur(0.6))
    return img


def make_phone_capture(src: Path, path: Path) -> None:
    """A tilted, noisy, JPEG-compressed, lower resolution capture without DPI info."""
    import numpy as np

    im = Image.open(src).convert("RGB")
    im = im.resize((int(im.width * 0.8), int(im.height * 0.8)), Image.LANCZOS)
    im = im.rotate(1.2, resample=Image.BICUBIC, expand=False, fillcolor=(250, 250, 247))
    a = np.asarray(im).astype(np.int16)
    a = a + np.random.default_rng(1).normal(0, 6, (im.height, im.width, 1)).astype(np.int16)
    a = np.clip(a * 0.97 + 6, 0, 255).astype(np.uint8)
    Image.fromarray(a).save(path, "JPEG", quality=70)


def make_photo(path: Path) -> None:
    w, h = 1600, 1000
    img = Image.new("RGB", (w, h))
    px = img.load()
    for y in range(h):
        for x in range(w):
            px[x, y] = (x * 255 // w, y * 255 // h, 160)
    d = ImageDraw.Draw(img)
    d.ellipse((200, 200, 600, 600), fill=(250, 220, 40))
    d.rectangle((900, 300, 1400, 800), fill=(30, 120, 60))
    img.save(path, "JPEG", quality=92)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gotenberg", default="http://localhost:3000")
    a = ap.parse_args()

    make_docx(HERE / "sample_ar.docx")
    (HERE / "invoice_ar.html").write_text(INVOICE_HTML, encoding="utf-8")
    pdf_page_png(html_to_pdf(a.gotenberg, INVOICE_HTML), 200, noise=True).save(
        HERE / "invoice_ar.png", dpi=(200, 200))
    pdf_page_png(office_to_pdf(a.gotenberg, HERE / "sample_ar.docx"), 200, noise=True).save(
        HERE / "letter_ar.png", dpi=(200, 200))
    make_photo(HERE / "photo.jpg")
    make_phone_capture(HERE / "invoice_ar.png", HERE / "phone_invoice.jpg")
    (HERE / "form_ar.pdf").write_bytes(html_to_pdf(a.gotenberg, FORM_HTML))
    (HERE / "invoice_ar.pdf").write_bytes(html_to_pdf(a.gotenberg, invoice_pdf_html(HERE / "photo.jpg")))
    truth = {
        "invoice_ar.png": INVOICE_TRUTH,
        "letter_ar.png": [AR_TITLE, "تهدف هذه الوثيقة", "من اليمين إلى اليسار", "2024",
                          "PDF", "Word", "الكمية", "دفتر", "This paragraph is written in English"],
        "sample_ar.docx": [AR_TITLE, AR_P1, AR_P2, EN_P, "الكمية", "دفتر"],
    }
    (HERE / "ground_truth.json").write_text(json.dumps(truth, ensure_ascii=False, indent=1),
                                            encoding="utf-8")
    print("fixtures written to", HERE)


if __name__ == "__main__":
    main()
