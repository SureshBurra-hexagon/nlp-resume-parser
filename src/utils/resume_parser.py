from __future__ import annotations

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
    tokens = set(normalized.replace(",", " ").split())
    return sorted(skill for skill in SKILL_KEYWORDS if skill in tokens)


def parse_resume(text: str) -> dict:
    normalized = clean_text(text)
    contacts = extract_contact_entities(text)
    skills = extract_skills(text)
    return {
        "normalized_text": normalized,
        "emails": contacts["emails"],
        "phones": contacts["phones"],
        "skills": skills,
    }
