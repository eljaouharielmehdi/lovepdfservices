import json
import os
import re
import unicodedata
from pathlib import Path

import httpx
import numpy as np
import pymupdf
import pytest
from PIL import Image

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000").rstrip("/")
FIX = Path(__file__).parent / "fixtures"
OUT = Path(os.environ.get("TEST_OUTPUT", Path(__file__).parent.parent / "out"))


@pytest.fixture(scope="session")
def api():
    with httpx.Client(base_url=BASE_URL, timeout=600) as c:
        r = c.get("/health")
        assert r.status_code == 200, r.text
        yield c


@pytest.fixture(scope="session")
def truth():
    return json.loads((FIX / "ground_truth.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def out_dir():
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT


@pytest.fixture(scope="session")
def sample_pdf(api, out_dir):
    """sample_ar.docx converted by the service itself (digital Arabic PDF)."""
    r = api.post("/api/v1/word-to-pdf", files={"file": ("sample_ar.docx", (FIX / "sample_ar.docx").read_bytes())})
    assert r.status_code == 200, r.text
    (out_dir / "sample_ar.pdf").write_bytes(r.content)
    return r.content


# ---------------------------------------------------------------- helpers
_PUNCT = re.compile(r"[\s‎‏‪-‮⁦-⁩.,:;!?،؛؟()\-]+")


def squash(text: str) -> str:
    """Letters only, in logical order (spacing / punctuation placement ignored)."""
    return _PUNCT.sub("", unicodedata.normalize("NFC", text))


def pdf_text(pdf: bytes) -> str:
    doc = pymupdf.open(stream=pdf, filetype="pdf")
    return "\n".join(p.get_text("text") for p in doc)


def render(pdf: bytes, dpi: float, page: int = 0) -> np.ndarray:
    doc = pymupdf.open(stream=pdf, filetype="pdf")
    pix = doc[page].get_pixmap(dpi=int(round(dpi)), alpha=False)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)


def ink(rgb: np.ndarray, bg=None, thr: int = 70) -> np.ndarray:
    if bg is None:
        bg = np.array([255, 255, 255])
    return np.abs(rgb.astype(np.int16) - np.asarray(bg, np.int16)).max(axis=-1) > thr


def visual_match(a: np.ndarray, b: np.ndarray, tol: int = 2) -> dict:
    """Ink precision / recall with a small spatial tolerance + mean colour error."""
    import cv2

    h, w = a.shape[:2]
    b = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)
    k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    ia, ib = ink(a).astype(np.uint8), ink(b).astype(np.uint8)
    da, db = cv2.dilate(ia, k), cv2.dilate(ib, k)
    precision = float((ib & da).sum() / max(1, ib.sum()))
    recall = float((ia & db).sum() / max(1, ia.sum()))
    blur_a = cv2.GaussianBlur(a, (0, 0), 1.5).astype(np.int16)
    blur_b = cv2.GaussianBlur(b, (0, 0), 1.5).astype(np.int16)
    mae = float(np.abs(blur_a - blur_b).mean())
    return {"precision": precision, "recall": recall, "mae": mae}


def save_side_by_side(a: np.ndarray, b: np.ndarray, path: Path) -> None:
    h = a.shape[0]
    bb = np.asarray(Image.fromarray(b).resize((a.shape[1], h)))
    sep = np.zeros((h, 12, 3), np.uint8)
    sep[..., 0] = 255
    Image.fromarray(np.hstack([a, sep, bb])).save(path)
