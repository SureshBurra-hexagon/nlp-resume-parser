"""High-level ResumeParser used by the Streamlit app (app.py).

Wires together the existing preprocessing, NER, and utils modules into a
single `.parse(text)` entry-point that returns a structured dict.
"""
from __future__ import annotations

import re

from src.ner.resume_entities import (
    CERTIFICATION_PATTERNS,
    DEGREE_PATTERNS,
    estimate_experience_years,
    extract_certifications,
    extract_education_entities,
    extract_sections,
)
from src.preprocessing.text_cleaner import (
    PATTERNS,
    clean_text,
    extract_contact_info,
)

# ---------------------------------------------------------------------------
# Skill keyword lists
# ---------------------------------------------------------------------------

_TECHNICAL_SKILLS = {
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "scala", "kotlin", "swift", "ruby", "php", "r", "matlab",
    "sql", "nosql", "postgresql", "mysql", "mongodb", "redis", "sqlite",
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ansible",
    "git", "linux", "bash", "powershell",
    "pytorch", "tensorflow", "keras", "scikit-learn", "numpy", "pandas",
    "nlp", "ml", "deep learning", "machine learning", "computer vision",
    "fastapi", "flask", "django", "spring", "react", "angular", "vue",
    "streamlit", "tableau", "power bi", "spark", "hadoop", "kafka",
    "ci/cd", "devops", "mlops", "rest", "graphql", "grpc",
}

_SOFT_SKILLS = {
    "communication", "leadership", "teamwork", "problem solving",
    "critical thinking", "time management", "adaptability", "creativity",
    "collaboration", "attention to detail", "project management",
    "analytical", "self-motivated", "organized", "multitasking",
}


def _extract_technical_skills(text: str) -> list[str]:
    normalized = clean_text(text)
    tokens = set(filter(None, re.split(r"[^a-z0-9+/#.\-]+", normalized)))
    found = []
    for skill in _TECHNICAL_SKILLS:
        if " " in skill:
            if skill in normalized:
                found.append(skill)
        elif skill in tokens:
            found.append(skill)
    return sorted(found)


def _extract_soft_skills(text: str) -> list[str]:
    normalized = clean_text(text)
    return sorted(skill for skill in _SOFT_SKILLS if skill in normalized)


def _extract_education_structured(text: str) -> list[dict]:
    """Return a list of structured education dicts for the app's display_education()."""
    # Extract canonical degree labels from existing helper
    degree_labels = extract_education_entities(text)
    # Try to pull year ranges near each degree mention
    year_pattern = re.compile(r"\b(?:19|20)\d{2}\b")
    years = [int(y) for y in year_pattern.findall(text)]

    entries = []
    for degree in degree_labels:
        entry: dict = {
            "degree": degree,
            "institution": None,
            "start_year": years[0] if len(years) >= 2 else None,
            "end_year": years[1] if len(years) >= 2 else (years[0] if years else None),
            "gpa": None,
        }
        # Attempt to find a GPA value
        gpa_match = re.search(r"gpa[:\s]*([0-4]\.\d{1,2})", text, re.IGNORECASE)
        if gpa_match:
            entry["gpa"] = gpa_match.group(1)
        # Attempt to find institution name following "university|college|institute|school"
        inst_match = re.search(
            r"([\w\s]+(?:university|college|institute|school|academy)[\w\s]*)",
            text,
            re.IGNORECASE,
        )
        if inst_match:
            entry["institution"] = inst_match.group(1).strip()[:80]
        entries.append(entry)
    return entries


def _extract_named_entities(text: str) -> dict[str, list[str]]:
    """Best-effort NER without spaCy — returns an empty dict if spaCy unavailable."""
    try:
        import spacy  # type: ignore
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text[:5000])  # cap for performance
        entities: dict[str, list[str]] = {}
        for ent in doc.ents:
            entities.setdefault(ent.label_, []).append(ent.text)
        return entities
    except Exception:
        return {}


class ResumeParser:
    """Parse a resume text string into structured fields.

    Parameters
    ----------
    use_ner:
        When *True*, attempt to run spaCy NER for entity extraction.
        Falls back gracefully if spaCy or its model is not installed.
    """

    def __init__(self, use_ner: bool = True) -> None:
        self.use_ner = use_ner

    def parse(self, text: str) -> dict:
        """Parse *text* and return a structured result dict.

        Returns
        -------
        dict with keys:
            raw_text, contact, skills, education, entities, sections,
            certifications, experience_years
        """
        if not text or not text.strip():
            return {
                "raw_text": text,
                "contact": {},
                "skills": {"technical": [], "soft": []},
                "education": [],
                "entities": {},
                "sections": {},
                "certifications": [],
                "experience_years": None,
            }

        contact_raw = extract_contact_info(text)
        # Flatten to single values for display_contact() which uses .get("email")
        contact = {
            "email": contact_raw["emails"][0] if contact_raw["emails"] else None,
            "phone": contact_raw["phones"][0] if contact_raw["phones"] else None,
            "linkedin": contact_raw["linkedin"][0] if contact_raw["linkedin"] else None,
            "github": contact_raw["github"][0] if contact_raw["github"] else None,
            "url": contact_raw["urls"][0] if contact_raw["urls"] else None,
            "name": contact_raw["name"],
        }

        skills = {
            "technical": _extract_technical_skills(text),
            "soft": _extract_soft_skills(text),
        }

        education = _extract_education_structured(text)
        entities = _extract_named_entities(text) if self.use_ner else {}
        sections = extract_sections(text)
        certifications = extract_certifications(text)
        experience_years = estimate_experience_years(text)

        return {
            "raw_text": text,
            "contact": contact,
            "skills": skills,
            "education": education,
            "entities": entities,
            "sections": sections,
            "certifications": certifications,
            "experience_years": experience_years,
        }
