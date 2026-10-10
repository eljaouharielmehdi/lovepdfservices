# lovepdfservices

A self-hosted, iLovePDF-like conversion API that runs with `docker compose`. Arabic is
fully supported: it is laid out right to left, the letters are correctly shaped and joined,
and mixed Arabic / English / numbers keep their logical order.

| Conversion | Endpoint | Engine |
|---|---|---|
| Word / Office → PDF | `POST /api/v1/word-to-pdf` | LibreOffice (Gotenberg) |
| PDF → Word (.docx) | `POST /api/v1/pdf-to-word` | own layout engine (PyMuPDF). Scanned pages go through OCR |
| Images → PDF | `POST /api/v1/image-to-pdf` | img2pdf (lossless, JPEG embedded bit-exact) |
| PDF → images | `POST /api/v1/pdf-to-image` | PyMuPDF (PNG / JPG / WEBP / TIFF, zip for several pages) |
| **OCR** image / scanned PDF → **regenerated PDF** | `POST /api/v1/ocr/pdf` | Tesseract 5 + style recovery + Chromium (Gotenberg) |
| OCR image / scanned PDF → editable Word | `POST /api/v1/ocr/word` | Tesseract 5 + layout engine |
| OCR → plain text / layout JSON | `POST /api/v1/ocr/text`, `/api/v1/ocr/json` | Tesseract 5 |

Interactive docs: <http://localhost:8000/docs>

## Run

```bash
docker compose up -d --build     # first build downloads the Tesseract "best" models
curl http://localhost:8000/health
```

Two containers are started:

* `gotenberg`: Gotenberg 8 (headless Chromium + LibreOffice), extended with the Amiri font and
  fontconfig aliases for Windows Arabic fonts (Traditional Arabic, Simplified Arabic, Sakkal
  Majalla, …). It is not exposed on the host.
* `api`: the FastAPI service on port **8000**. It runs Tesseract, recovers the layout and calls Gotenberg.

## Examples

```bash
# Word -> PDF
curl -F file=@report.docx http://localhost:8000/api/v1/word-to-pdf -o report.pdf

# PDF -> Word (auto: OCR only the pages that have no text layer)
curl -F file=@report.pdf -F mode=auto http://localhost:8000/api/v1/pdf-to-word -o report.docx

# images -> one PDF (fit = page is the image at its DPI; or A4/A3/A5/Letter/Legal)
curl -F files=@p1.jpg -F files=@p2.png -F page_size=A4 -F margin=20 \
     http://localhost:8000/api/v1/image-to-pdf -o images.pdf

# PDF -> images
curl -F file=@report.pdf -F format=png -F dpi=200 -F pages=1-3 \
     http://localhost:8000/api/v1/pdf-to-image -o pages.zip

# OCR a photo / scan into a regenerated PDF that looks like the original
curl -F file=@invoice.jpg -F lang=ara+eng http://localhost:8000/api/v1/ocr/pdf -o invoice.pdf

# OCR into an editable Word document
curl -F file=@invoice.jpg http://localhost:8000/api/v1/ocr/word -o invoice.docx
```

### Parameters

| Endpoint | Field | Values (default) |
|---|---|---|
| word-to-pdf | `file` | .docx .doc .odt .rtf .txt .xlsx .pptx … |
| | `pdfa` | `PDF/A-1b`, `PDF/A-2b`, `PDF/A-3b` |
| | `landscape`, `page_ranges` | `true`/`false`, `1-3,5` |
| pdf-to-word | `mode` | `auto` (OCR pages without text), `text`, `ocr` |
| | `lang` | Tesseract languages (`ara+eng`) |
| image-to-pdf | `files` (repeatable) | jpg png tiff webp bmp gif … |
| | `page_size` | `fit`, `A4`, `A3`, `A5`, `Letter`, `Legal` |
| | `orientation` | `auto`, `portrait`, `landscape` |
| | `margin` | points (0) |
| pdf-to-image | `format` | `png`, `jpg`, `webp`, `tiff` |
| | `dpi`, `pages`, `quality` | 150, all, 92 |
| ocr/* | `file` | image (any Pillow format, multi-page TIFF) or PDF |
| | `lang` | `ara+eng` (models installed: `ara eng fra`, more via build arg) |
| | `dpi` | override the scan resolution (default: from the file, else page = A4 wide) |
| ocr/pdf | `background` | `vector` (default), `original`, `none` |

## How the OCR → PDF regeneration works

The output is **not** the scan with a hidden text layer. The page is rebuilt from what was
recognised and then printed by Gotenberg's Chromium:

1. **Pre-processing.** The scan is normalised against a local background estimate. White text on
   coloured bands, red or green text and text in shaded table cells all become dark-on-white
   for Tesseract (LSTM, `tessdata_best` models).
2. **Two OCR passes.** A page-layout pass, then a sparse-text pass over whatever was not
   recognised, which picks up isolated words and single digits in table cells.
3. **Style recovery for every word:**
   * the glyph pixels are segmented, including dots and descenders outside the OCR box, and
     table rules are excluded;
   * **text color** is sampled from the stroke cores;
   * the **font family** is chosen by rendering candidates (Noto Naskh Arabic, Noto Sans
     Arabic, Amiri, Noto Kufi, Noto Sans/Serif, Liberation Sans/Serif/Mono) with HarfBuzz
     shaping and comparing them with the scanned glyphs;
   * **bold** is detected by comparing stroke thickness against the regular and bold renders;
   * **size, baseline and width** are fitted so the regenerated ink box matches the original
     ink box.
4. **Graphics.** The recognised text is erased (inpainted) from the scan, then the rest is
   decomposed:
   * the page color, filled boxes and table rules become **vector rectangles**;
   * logos, photos, stamps and signatures stay as **pixel-exact image crops**;
   * text that OCR was not confident about is kept as an image too, so nothing is lost.
5. **Rendering.** Each page becomes one SVG in the scan's pixel coordinates (so every word sits
   where it was), at the scan's physical size. The fonts are sent to Gotenberg as assets, so
   what was measured is exactly what Chromium draws. Arabic words are drawn with
   `direction="rtl"` and fitted with a `scale()` transform. SVG `textLength` is avoided on
   purpose because Chromium then lays out the letters one by one and breaks Arabic joining.

The result has selectable, searchable text whose copy-paste gives logical-order Arabic, and its
fonts are embedded.

`background=original` keeps the text-free scan as the page background instead of vectorising it.
`background=none` gives text only on white.

## How PDF / OCR → Word keeps the look

* Paragraphs are rebuilt from lines (by gap and pitch). Alignment and indentation are measured
  against the text column, and each paragraph gets an **exact line pitch** placed so the first
  baseline lands where it was. Word and LibreOffice both put the baseline at 0.8 × the line
  height.
* Arabic paragraphs get `w:bidi` and Arabic runs `w:rtl`, `w:cs` fonts and `szCs`. Mixed lines are
  written in logical order. Note that in a bidi paragraph Word reads `jc="right"` as the *end*
  (left) side, and the writer accounts for that.
* Ruled tables become real Word tables: column widths, row heights, merged cells, borders,
  cell shading, and right-to-left column order (`w:bidiVisual`).
* All non-text graphics (bands, logos, photos) are placed on a page-sized picture anchored behind
  the text, so the page looks like the original while all text stays editable.

## Tests

The end-to-end suite runs against the compose stack and checks content and looks for every
conversion:

* text in logical order;
* the **rendered pages are OCR'd again**, so scrambled or disconnected Arabic would fail;
* pixel / ink comparison with the source;
* Word line baselines within 2 pt of the source PDF;
* lossless image embedding.

```bash
docker compose up -d --build
pip install -r tests/requirements.txt
python tests/fixtures/make_fixtures.py --gotenberg http://localhost:3000   # optional: regenerate fixtures
BASE_URL=http://localhost:8000 pytest -v tests
```

Side-by-side comparison images and all outputs are written to `out/`.

## Configuration (api environment)

| Variable | Default | |
|---|---|---|
| `GOTENBERG_URL` | `http://gotenberg:3000` | |
| `OCR_LANGS` | `ara+eng` | default OCR languages |
| `OCR_WORKERS` | `2` | pages OCR'd in parallel per request |
| `OCR_MIN_CONF` | `45` | words below this confidence stay as image pixels |
| `MAX_UPLOAD_MB` / `MAX_PAGES` | `100` / `50` | |
| `PDF_RASTER_DPI` | `300` | resolution used to OCR PDF pages |

Extra OCR languages: `docker compose build --build-arg OCR_LANGS="ara eng fra deu" api`, then send `lang=ara+deu`.

## Licenses

PyMuPDF is AGPL-3.0. Tesseract, Gotenberg, Pillow, OpenCV, img2pdf and python-docx are
permissively licensed. The fonts are SIL OFL.

## Samples

The `samples/` folder holds real outputs produced by the test suite. Each comparison image shows
the original on the left and the regenerated output on the right:

* `ocr_invoice_original_vs_regenerated.png`, `ocr_invoice_regenerated.pdf`: OCR → PDF of a coloured Arabic invoice
* `ocr_letter_original_vs_regenerated.png`: OCR → PDF of an Arabic / English letter with a table
* `ocr_phone_photo_original_vs_regenerated.png`: a tilted, noisy, JPEG-compressed phone capture
* `ocr_to_word_original_vs_docx.png`, `ocr_invoice.docx`: OCR → editable Word
* `pdf_to_word_arabic.docx`, `word_to_pdf_arabic.pdf`: Word ⇄ PDF

## Known limits

* OCR accuracy is Tesseract's. A word whose recognised text cannot produce its ink (a misread)
  keeps its original pixels visible, with the text layer invisible on top, so the page still
  looks exact. Words below `OCR_MIN_CONF` stay as pixels too.
* Fonts are matched against the bundled free families. A document set in another typeface is
  regenerated with the closest one, at the same size, weight, color and position.
* Handwriting is kept as image crops, not recognised.
