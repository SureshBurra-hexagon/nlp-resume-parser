"""
Tests for the preprocessing module.
"""

import pytest

from src.preprocessing import (
    clean_text,
    extract_contact_info,
    identify_sections,
    preprocess_text,
    tokenize,
    remove_stopwords,
    sentence_tokenize,
)


SAMPLE_RESUME = """
John Doe
john.doe@example.com | +1-555-123-4567
linkedin.com/in/johndoe | github.com/johndoe

Education
B.Tech in Computer Science
University of Technology, 2018 – 2022
GPA: 3.8/4.0

Experience
Software Engineer, Acme Corp, 2022 – 2024
- Developed REST APIs using Python and Flask
- Managed PostgreSQL databases

Skills
Python, Machine Learning, NLP, SQL, Docker, React

Projects
Resume Parser NLP Project
- Built a resume parsing tool using spaCy and scikit-learn

Certifications
AWS Certified Developer – 2023
"""


class TestCleanText:
    def test_removes_excessive_newlines(self):
        text = "line1\n\n\n\nline2"
        cleaned = clean_text(text)
        assert "\n\n\n" not in cleaned

    def test_normalises_unicode_dashes(self):
        text = "Jan 2020\u2013Dec 2022"
        assert "–" not in clean_text(text)

    def test_returns_string(self):
        assert isinstance(clean_text("hello world"), str)

    def test_empty_string(self):
        assert clean_text("") == ""


class TestTokenize:
    def test_basic_tokenization(self):
        tokens = tokenize("Hello world")
        assert "Hello" in tokens or "hello" in tokens.copy()

    def test_returns_list(self):
        assert isinstance(tokenize("test"), list)

    def test_non_empty(self):
        tokens = tokenize("Natural Language Processing")
        assert len(tokens) > 0


class TestRemoveStopwords:
    def test_removes_common_words(self):
        tokens = ["i", "am", "a", "software", "engineer"]
        filtered = remove_stopwords(tokens)
        assert "i" not in filtered
        assert "software" in filtered

    def test_case_insensitive(self):
        tokens = ["The", "Python", "Language"]
        filtered = remove_stopwords(tokens)
        assert "The" not in filtered
        assert "Python" in filtered


class TestSentenceTokenize:
    def test_splits_sentences(self):
        text = "I work at Acme Corp. I build software."
        sentences = sentence_tokenize(text)
        assert len(sentences) >= 2

    def test_returns_list(self):
        assert isinstance(sentence_tokenize("Hello."), list)


class TestPreprocessText:
    def test_returns_list(self):
        result = preprocess_text("Hello World")
        assert isinstance(result, list)

    def test_lowercase(self):
        result = preprocess_text("PYTHON", lowercase=True)
        for token in result:
            assert token == token.lower()

    def test_no_punctuation(self):
        result = preprocess_text("Hello, world!", remove_punct=True)
        assert "," not in result
        assert "!" not in result

    def test_empty_string(self):
        result = preprocess_text("")
        assert result == []


class TestExtractContactInfo:
    def test_email(self):
        info = extract_contact_info("Contact: john.doe@example.com")
        assert info["email"] == "john.doe@example.com"

    def test_phone(self):
        info = extract_contact_info("Phone: +1-555-123-4567")
        assert info["phone"] is not None

    def test_linkedin(self):
        info = extract_contact_info("linkedin.com/in/johndoe")
        assert info["linkedin"] is not None

    def test_github(self):
        info = extract_contact_info("github.com/johndoe")
        assert info["github"] is not None

    def test_empty_returns_none_values(self):
        info = extract_contact_info("no contact info here")
        assert info["email"] is None
        assert info["phone"] is None

    def test_full_resume(self):
        info = extract_contact_info(SAMPLE_RESUME)
        assert info["email"] == "john.doe@example.com"


class TestIdentifySections:
    def test_identifies_education(self):
        sections = identify_sections(SAMPLE_RESUME)
        assert "education" in sections

    def test_identifies_experience(self):
        sections = identify_sections(SAMPLE_RESUME)
        assert "experience" in sections

    def test_identifies_skills(self):
        sections = identify_sections(SAMPLE_RESUME)
        assert "skills" in sections

    def test_returns_dict(self):
        assert isinstance(identify_sections(SAMPLE_RESUME), dict)

    def test_education_content(self):
        sections = identify_sections(SAMPLE_RESUME)
        assert "Computer Science" in sections.get("education", "")

    def test_skills_content(self):
        sections = identify_sections(SAMPLE_RESUME)
        assert "Python" in sections.get("skills", "")
