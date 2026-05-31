"""HTML / PDF fetcher with SSRF protection."""

from __future__ import annotations

import asyncio
import ipaddress
import re
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx
import trafilatura
from loguru import logger


@dataclass
class Chunk:
    """A unit of source text addressable by URL + offset within the page."""

    url: str
    title: str
    text: str
    source_idx: int = 0  # nth chunk in the source page


_PRIVATE_HOST_PATTERNS = (
    re.compile(r"^localhost$", re.I),
    re.compile(r"^127\."),
    re.compile(r"^10\."),
    re.compile(r"^192\.168\."),
    re.compile(r"^169\.254\."),
)


def _is_safe_url(url: str) -> bool:
    """Reject obviously-internal targets (basic SSRF guard)."""
    try:
        p = urlparse(url)
    except Exception:  # noqa: BLE001
        return False
    if p.scheme not in ("http", "https"):
        return False
    host = (p.hostname or "").lower()
    if not host:
        return False
    for pat in _PRIVATE_HOST_PATTERNS:
        if pat.match(host):
            return False
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_private or ip.is_loopback or ip.is_link_local:
            return False
    except ValueError:
        # not a literal IP; hostname is fine
        pass
    return True


def _split_text(text: str, max_chars: int = 1500, overlap: int = 100) -> list[str]:
    """Sliding-window chunking on character boundaries."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


class Fetcher:
    """Async HTML / PDF fetcher.

    For HTML pages we use ``trafilatura`` to extract the main text. PDFs
    are handled via PyMuPDF (lazy-imported to keep startup cheap).
    """

    def __init__(self, *, timeout_s: float = 25.0, max_chars: int = 1500) -> None:
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_s),
            follow_redirects=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (compatible; dr-agent/0.1; "
                    "+https://github.com/1552964482/deepresearch)"
                )
            },
        )
        self._max_chars = max_chars

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "Fetcher":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def fetch(self, url: str, *, title: str = "") -> list[Chunk]:
        if not _is_safe_url(url):
            logger.debug("skipping unsafe URL: {}", url)
            return []
        try:
            resp = await self._client.get(url)
        except Exception as e:  # noqa: BLE001
            logger.debug("fetch error {}: {}", url, e)
            return []
        if resp.status_code >= 400:
            return []
        ctype = resp.headers.get("content-type", "").lower()
        if "pdf" in ctype or url.lower().endswith(".pdf"):
            text = await asyncio.to_thread(self._extract_pdf, resp.content)
        else:
            text = await asyncio.to_thread(self._extract_html, resp.text, url)
        if not text:
            return []
        return [
            Chunk(url=url, title=title, text=t, source_idx=i)
            for i, t in enumerate(_split_text(text, max_chars=self._max_chars))
        ]

    @staticmethod
    def _extract_html(html: str, url: str) -> str:
        try:
            extracted = trafilatura.extract(
                html,
                url=url,
                include_comments=False,
                include_tables=True,
                favor_recall=True,
            )
            return extracted or ""
        except Exception as e:  # noqa: BLE001
            logger.debug("trafilatura failed for {}: {}", url, e)
            return ""

    @staticmethod
    def _extract_pdf(content: bytes) -> str:
        try:
            import fitz  # PyMuPDF
        except ImportError:
            return ""
        try:
            with fitz.open(stream=content, filetype="pdf") as doc:
                return "\n\n".join(page.get_text() for page in doc)
        except Exception as e:  # noqa: BLE001
            logger.debug("pymupdf failed: {}", e)
            return ""
