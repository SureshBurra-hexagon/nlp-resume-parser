from __future__ import annotations

from html import unescape
from pathlib import Path
import re
import zipfile


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _extract_txt_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract_html_text(path: Path) -> str:
    html = path.read_text(encoding="utf-8")
    html = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
    html = re.sub(r"(?is)<[^>]+>", " ", html)
    return unescape(_normalize_whitespace(html))


def _extract_docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        document_xml = archive.read("word/document.xml").decode("utf-8", errors="ignore")
    text = re.sub(r"(?i)</w:p>", "\n", document_xml)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    return _normalize_whitespace(unescape(text))


def _extract_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF parsing requires `pypdf`. Install dependencies from requirements.txt.") from exc

    reader = PdfReader(str(path))
    chunks = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        chunks.append(page_text)
    return _normalize_whitespace("\n".join(chunks))


def extract_text_from_file(file_path: str) -> str:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Resume file not found: {path}")
    if path.is_dir():
        raise IsADirectoryError(f"Expected a file but received a directory: {path}")

    suffix = path.suffix.lower()
    if suffix == ".txt":
        text = _extract_txt_text(path)
    elif suffix in {".html", ".htm"}:
        text = _extract_html_text(path)
    elif suffix == ".docx":
        text = _extract_docx_text(path)
    elif suffix == ".pdf":
        text = _extract_pdf_text(path)
    else:
        raise ValueError(f"Unsupported resume file format `{suffix}`. Use .txt, .html, .docx, or .pdf")

    if not text.strip():
        raise ValueError(f"No extractable text found in file: {path}")
    return text
