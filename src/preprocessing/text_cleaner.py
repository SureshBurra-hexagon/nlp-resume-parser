from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Union

EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
PHONE_PATTERN = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}")
WHITESPACE_PATTERN = re.compile(r"\s+")


def extract_text_from_pdf(source: Union[str, Path, bytes]) -> str:
    """Extract plain text from a PDF file path, Path object, or raw bytes.

    Args:
        source: A file-system path (str or Path) or the raw PDF bytes.

    Returns:
        Concatenated text from all pages, separated by newlines.

    Raises:
        ImportError: If ``pypdf`` is not installed.
        FileNotFoundError: If *source* is a path that does not exist.
        ValueError: If *source* is not a valid PDF or the type is unsupported.
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pypdf is required: pip install pypdf") from exc

    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")
        reader = PdfReader(str(path))
    elif isinstance(source, (bytes, bytearray)):
        reader = PdfReader(io.BytesIO(source))
    else:
        raise ValueError(f"Unsupported source type: {type(source)!r}")

    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n".join(pages)


def extract_text_from_txt(source: Union[str, Path, bytes]) -> str:
    """Extract plain text from a .txt file path, Path object, or raw bytes.

    Args:
        source: A file-system path (str or Path) or the raw file bytes.

    Returns:
        The decoded text content of the file.

    Raises:
        FileNotFoundError: If *source* is a path that does not exist.
        ValueError: If *source* type is unsupported.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Text file not found: {path}")
        return path.read_text(encoding="utf-8", errors="replace")
    elif isinstance(source, (bytes, bytearray)):
        return source.decode("utf-8", errors="replace")
    else:
        raise ValueError(f"Unsupported source type: {type(source)!r}")


def clean_text(text: str) -> str:
    """Normalize resume text for downstream NLP tasks."""
    if not text:
        return ""
    cleaned = text.replace("\u00a0", " ").replace("\t", " ").replace("\r", " ").replace("\n", " ")
    cleaned = WHITESPACE_PATTERN.sub(" ", cleaned).strip().lower()
    return cleaned


def extract_contact_entities(text: str) -> dict:
    """Extract contact-level entities from text."""
    if not text:
        return {"emails": [], "phones": []}
    return {
        "emails": sorted(set(EMAIL_PATTERN.findall(text))),
        "phones": sorted(set(PHONE_PATTERN.findall(text))),
    }
