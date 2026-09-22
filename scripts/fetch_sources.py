"""Download every confirmed source in data/sources.yaml into data/raw/.

    uv run python -m scripts.fetch_sources           # skip files that already exist
    uv run python -m scripts.fetch_sources --force   # re-download

Only sources with url_confirmed: true are fetched. A `url` may be a direct PDF or a
publication landing page; for a landing page the first linked PDF on the publisher's
domains is used (and printed, so it can be recorded). Every file is checked to be a PDF;
the size and page count are printed. data/raw/ is gitignored (see licence notes).
"""

import argparse
import html as html_lib
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from pypdf import PdfReader

from backend.app.rag.ingest import DEFAULT_RAW_DIR, DEFAULT_SOURCES, load_sources

HEADERS = {"User-Agent": "Mozilla/5.0 (iba-triage source fetcher; research use)"}
PDF_HOSTS = ("who.int", "ncdc.gov.ng")


def is_pdf(data: bytes) -> bool:
    return data[:1024].lstrip().startswith(b"%PDF-")


def pdf_links(page: str, base: str) -> list[str]:
    """Candidate PDF links on a landing page, publisher domains only.

    WHO pages mark the main file with onclick="sendGaEvent('Download', '<url>')"; those come
    first, then other PDF-looking links (quoted or unquoted href) in page order.
    """
    page = html_lib.unescape(page)
    download_buttons = re.findall(r"sendGaEvent\('Download',\s*'([^']+)'\)", page)
    hrefs = re.findall(r"""href=(?:["']([^"']+)["']|([^\s>]+))""", page, re.IGNORECASE)
    candidates = download_buttons + [quoted or bare for quoted, bare in hrefs]
    links = []
    for href in candidates:
        url = urljoin(base, href)
        host = (urlparse(url).hostname or "").lower()
        lower = url.lower()
        looks_pdf = ".pdf" in lower or "/bitstreams/" in lower or "download=true" in lower
        if looks_pdf and any(host == h or host.endswith("." + h) for h in PDF_HOSTS):
            links.append(url)
    return list(dict.fromkeys(links))


def get(client: httpx.Client, url: str, attempts: int = 4) -> httpx.Response:
    """GET with retries on 5xx / network errors (NCDC's CDN returns intermittent 522s)."""
    for attempt in range(attempts):
        try:
            resp = client.get(url)
            if resp.status_code < 500:
                return resp
        except httpx.TransportError:
            if attempt == attempts - 1:
                raise
        if attempt < attempts - 1:
            time.sleep(5 * 2**attempt)
    return resp


def fetch_pdf(client: httpx.Client, url: str) -> tuple[bytes, str]:
    resp = get(client, url)
    resp.raise_for_status()
    if is_pdf(resp.content):
        return resp.content, url
    for link in pdf_links(resp.text, str(resp.url)):
        try:
            sub = get(client, link)
        except httpx.HTTPError:
            continue
        if sub.status_code == 200 and is_pdf(sub.content):
            return sub.content, link
    raise ValueError("no PDF at the URL or linked from the page")


def describe(path: Path) -> str:
    pages = len(PdfReader(path).pages)
    return f"{path.stat().st_size / 1_048_576:6.2f} MB, {pages:4d} pages"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download confirmed guideline PDFs.")
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args()

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    with httpx.Client(headers=HEADERS, follow_redirects=True, timeout=60) as client:
        for source in load_sources(args.sources):
            path = args.raw_dir / source.file
            if not source.url_confirmed:
                print(f"SKIP  {source.doc_id:<26} url not confirmed (add {path} manually)")
                continue
            if path.exists() and not args.force:
                print(f"HAVE  {source.doc_id:<26} {describe(path)}  {path}")
                continue
            try:
                data, used = fetch_pdf(client, source.url or "")
            except (httpx.HTTPError, ValueError) as exc:
                failures += 1
                print(f"FAIL  {source.doc_id:<26} {type(exc).__name__}: {exc}")
                continue
            path.write_bytes(data)
            print(f"GOT   {source.doc_id:<26} {describe(path)}  {path}")
            if used != source.url:
                print(f"      resolved PDF link: {used}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
