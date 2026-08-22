from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Union

EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
PHONE_PATTERN = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}")
WHITESPACE_PATTERN = re.compile(r"\s+")

# Unified patterns dict for use by downstream consumers (e.g. app.py).
# Groups commonly-needed compiled regexes by category.
PATTERNS: dict[str, re.Pattern[str]] = {
    "email": EMAIL_PATTERN,
    "phone": PHONE_PATTERN,
    "years_experience": re.compile(r"\b(\d{1,2})\+?\s+years?\b", re.IGNORECASE),
    "url": re.compile(r"https?://[^\s]+|www\.[^\s]+", re.IGNORECASE),
    "linkedin": re.compile(r"linkedin\.com/in/[\w-]+", re.IGNORECASE),
    "github": re.compile(r"github\.com/[\w-]+", re.IGNORECASE),
    "section_header": re.compile(
        r"^(summary|profile|objective|about|experience|work experience|employment|"
        r"professional experience|education|academic background|academics|skills|"
        r"technical skills|core skills|competencies|projects|project experience|"
        r"certifications|licenses|credentials)\s*:?$",
        re.IGNORECASE | re.MULTILINE,
    ),
}


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


def identify_sections(text: str) -> dict[str, str]:
    """Identify and extract named sections from resume text.

    Delegates to :func:`src.ner.resume_entities.extract_sections` and returns
    a mapping of canonical section names (e.g. ``"experience"``, ``"skills"``)
    to their concatenated text content.
    """
    from src.ner.resume_entities import extract_sections  # local import avoids circular deps
    return extract_sections(text)


def extract_contact_entities(text: str) -> dict:
    """Extract contact-level entities from text."""
    if not text:
        return {"emails": [], "phones": []}
    return {
        "emails": sorted(set(EMAIL_PATTERN.findall(text))),
        "phones": sorted(set(PHONE_PATTERN.findall(text))),
    }


def extract_contact_info(text: str) -> dict:
    """Extract a comprehensive set of contact details from resume text.

    Returns a dict with the following keys:
        - emails (list[str]): email addresses found
        - phones (list[str]): phone numbers found
        - linkedin (list[str]): LinkedIn profile URLs found
        - github (list[str]): GitHub profile URLs found
        - urls (list[str]): other web URLs found (LinkedIn/GitHub excluded)
        - name (str | None): best-guess full name (first non-empty line)
    """
    if not text:
        return {"emails": [], "phones": [], "linkedin": [], "github": [], "urls": [], "name": None}

    emails = sorted(set(PATTERNS["email"].findall(text)))
    phones = sorted(set(PATTERNS["phone"].findall(text)))
    linkedin = sorted(set(PATTERNS["linkedin"].findall(text)))
    github = sorted(set(PATTERNS["github"].findall(text)))

    all_urls = sorted(set(PATTERNS["url"].findall(text)))
    excluded = set(linkedin) | set(github)
    urls = [u for u in all_urls if not any(ex in u for ex in excluded)]

    # Heuristic: the name is typically the first non-empty line of a resume.
    name: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not PATTERNS["email"].search(stripped) and not PATTERNS["phone"].search(stripped):
            name = stripped
            break

    return {
        "emails": emails,
        "phones": phones,
        "linkedin": linkedin,
        "github": github,
        "urls": urls,
        "name": name,
    }
