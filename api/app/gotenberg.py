"""Minimal async client for Gotenberg 8 (Chromium HTML->PDF, LibreOffice office->PDF,
PDF engines merge)."""
from __future__ import annotations

import os

import httpx

GOTENBERG_URL = os.environ.get("GOTENBERG_URL", "http://gotenberg:3000").rstrip("/")
TIMEOUT = float(os.environ.get("GOTENBERG_TIMEOUT", "300"))


class GotenbergError(RuntimeError):
    pass


async def _post(path: str, files: list, data: dict) -> bytes:
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        r = await client.post(f"{GOTENBERG_URL}{path}", files=files, data=data)
    if r.status_code != 200:
        raise GotenbergError(f"gotenberg {path} -> {r.status_code}: {r.text[:500]}")
    return r.content


async def html_to_pdf(index_html: str, assets: dict[str, bytes] | None = None,
                      wait_expression: str | None = None, **opts) -> bytes:
    files = [("files", ("index.html", index_html.encode("utf-8"), "text/html"))]
    for name, data in (assets or {}).items():
        files.append(("files", (name, data, "application/octet-stream")))
    data = {
        "preferCssPageSize": "true",
        "printBackground": "true",
        "marginTop": "0", "marginBottom": "0", "marginLeft": "0", "marginRight": "0",
    }
    if wait_expression:
        data["waitForExpression"] = wait_expression
    data.update({k: str(v).lower() if isinstance(v, bool) else str(v) for k, v in opts.items()})
    return await _post("/forms/chromium/convert/html", files, data)


async def office_to_pdf(filename: str, content: bytes, **opts) -> bytes:
    files = [("files", (filename, content, "application/octet-stream"))]
    data = {k: str(v).lower() if isinstance(v, bool) else str(v) for k, v in opts.items()}
    return await _post("/forms/libreoffice/convert", files, data)


async def health() -> dict:
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(f"{GOTENBERG_URL}/health")
    return r.json()
