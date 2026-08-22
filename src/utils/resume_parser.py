from __future__ import annotations

import re

from src.ingestion.file_text_extractor import extract_text_from_file
from src.ner.resume_entities import (
    estimate_experience_years,
    extract_certifications,
    extract_education_entities,
    extract_sections,
)
from src.preprocessing.text_cleaner import clean_text, extract_contact_entities

SKILL_KEYWORDS = {
    "python",
    "java",
    "sql",
    "aws",
    "docker",
    "kubernetes",
    "pytorch",
    "tensorflow",
    "nlp",
    "fastapi",
    "streamlit",
    "react",
    "javascript",
    "typescript",
    "tableau",
    "ci/cd",
}


def extract_skills(text: str) -> list[str]:
    normalized = clean_text(text)
    tokens = set(filter(None, re.split(r"[^a-z0-9+/.\-]+", normalized)))

    def contains(skill: str) -> bool:
        return skill in tokens or (not skill.isalnum() and skill in normalized)

    return sorted(skill for skill in SKILL_KEYWORDS if contains(skill))


def parse_resume(text: str) -> dict:
    normalized = clean_text(text)
    contacts = extract_contact_entities(text)
    skills = extract_skills(text)
    sections = extract_sections(text)
    education = extract_education_entities(text)
    certifications = extract_certifications(text)
    experience_years = estimate_experience_years(text)
    return {
        "normalized_text": normalized,
        "emails": contacts["emails"],
        "phones": contacts["phones"],
        "skills": skills,
        "sections": sections,
        "education": education,
        "certifications": certifications,
        "experience_years": experience_years,
    }


def parse_resume_file(file_path: str) -> dict:
    return parse_resume(extract_text_from_file(file_path))
