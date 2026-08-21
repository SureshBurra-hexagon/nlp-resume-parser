from __future__ import annotations

from pathlib import Path
import types
import sys
import zipfile

import pytest

from src.ingestion.file_text_extractor import extract_text_from_file
from src.utils.resume_parser import parse_resume_file


def _write_minimal_docx(path: Path, body_text: str) -> None:
    content_types = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>
"""
    rels = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
"""
    document = f"""<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>{body_text}</w:t></w:r></w:p>
  </w:body>
</w:document>
"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", rels)
        archive.writestr("word/document.xml", document)


def test_extract_text_from_txt_file(tmp_path):
    resume = tmp_path / "resume.txt"
    resume.write_text("Email: jane@example.com Skills: Python SQL", encoding="utf-8")
    extracted = extract_text_from_file(str(resume))
    assert "jane@example.com" in extracted
    assert "Python" in extracted


def test_extract_text_from_html_file(tmp_path):
    resume = tmp_path / "resume.html"
    resume.write_text(
        "<html><body><h1>Resume</h1><p>Skills: React TypeScript</p><script>ignore-me</script></body></html>",
        encoding="utf-8",
    )
    extracted = extract_text_from_file(str(resume))
    assert "Resume" in extracted
    assert "React TypeScript" in extracted
    assert "ignore-me" not in extracted


def test_extract_text_from_docx_file(tmp_path):
    resume = tmp_path / "resume.docx"
    _write_minimal_docx(resume, "Skills: AWS Docker Kubernetes")
    extracted = extract_text_from_file(str(resume))
    assert "AWS Docker Kubernetes" in extracted


def test_extract_text_from_pdf_file_with_stubbed_reader(tmp_path, monkeypatch):
    resume = tmp_path / "resume.pdf"
    resume.write_bytes(b"%PDF-1.4\n")

    class _Page:
        def extract_text(self) -> str:
            return "Skills: NLP FastAPI"

    class _Reader:
        def __init__(self, _path: str) -> None:
            self.pages = [_Page()]

    monkeypatch.setitem(sys.modules, "pypdf", types.SimpleNamespace(PdfReader=_Reader))
    extracted = extract_text_from_file(str(resume))
    assert "NLP FastAPI" in extracted


def test_parse_resume_file_extracts_structured_fields(tmp_path):
    resume = tmp_path / "resume.txt"
    resume.write_text("Email jane@example.com Skills: Python SQL NLP", encoding="utf-8")
    parsed = parse_resume_file(str(resume))
    assert "jane@example.com" in parsed["emails"]
    assert {"python", "sql", "nlp"}.issubset(set(parsed["skills"]))


def test_extract_text_from_file_rejects_unsupported_extension(tmp_path):
    resume = tmp_path / "resume.md"
    resume.write_text("# Resume", encoding="utf-8")
    with pytest.raises(ValueError):
        extract_text_from_file(str(resume))
