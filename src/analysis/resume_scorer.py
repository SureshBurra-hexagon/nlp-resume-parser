from __future__ import annotations

from src.analysis.experience_classifier import classify_experience_level
from src.ner.resume_entities import (
    estimate_experience_years,
    extract_certifications,
    extract_education_entities,
    extract_sections,
)
from src.utils.resume_parser import extract_skills

_EDUCATION_WEIGHTS = {"bachelor": 10, "master": 15, "mba": 15, "phd": 20}
_MAX_SKILL_SCORE = 30
_MAX_EXPERIENCE_SCORE = 20
_SECTION_BONUS_PER_SECTION = 4
_REQUIRED_SECTIONS = {"summary", "experience", "education", "skills"}
_MAX_SECTION_SCORE = 20
_CERT_BONUS_PER_CERT = 5
_MAX_CERT_SCORE = 15

_SALARY_RANGES: dict[str, tuple[int, int]] = {
    "entry": (40_000, 65_000),
    "junior": (55_000, 85_000),
    "mid": (80_000, 120_000),
    "senior": (110_000, 160_000),
    "staff": (150_000, 200_000),
    "principal": (180_000, 240_000),
}


def _education_score(text: str) -> float:
    entities = extract_education_entities(text)
    return min(float(sum(_EDUCATION_WEIGHTS.get(e, 0) for e in entities)), 20.0)


def _skill_score(text: str) -> float:
    skills = extract_skills(text)
    return min(len(skills) * 3.0, float(_MAX_SKILL_SCORE))


def _experience_score(years: int | None) -> float:
    if years is None:
        return 0.0
    return min(years * 2.0, float(_MAX_EXPERIENCE_SCORE))


def _section_score(text: str) -> float:
    found = set(extract_sections(text).keys())
    present = found & _REQUIRED_SECTIONS
    return min(len(present) * _SECTION_BONUS_PER_SECTION, float(_MAX_SECTION_SCORE))


def _cert_score(text: str) -> float:
    certs = extract_certifications(text)
    return min(len(certs) * _CERT_BONUS_PER_CERT, float(_MAX_CERT_SCORE))


def compute_quality_score(resume_text: str) -> dict:
    """Score a resume across multiple quality dimensions (0–100).

    Returns a dict with:
      - ``total_score``: float in [0, 100]
      - ``education_score``: float
      - ``skill_score``: float
      - ``experience_score``: float
      - ``section_score``: float
      - ``certification_score``: float
    """
    years = estimate_experience_years(resume_text)
    edu = _education_score(resume_text)
    skill = _skill_score(resume_text)
    exp = _experience_score(years)
    sec = _section_score(resume_text)
    cert = _cert_score(resume_text)
    total = min(edu + skill + exp + sec + cert, 100.0)
    return {
        "total_score": round(total, 2),
        "education_score": edu,
        "skill_score": skill,
        "experience_score": exp,
        "section_score": sec,
        "certification_score": cert,
    }


def estimate_salary(resume_text: str) -> dict:
    """Estimate a salary band based on experience level.

    Returns a dict with:
      - ``level``: experience level label
      - ``salary_min``: lower bound (USD)
      - ``salary_max``: upper bound (USD)
      - ``currency``: always "USD"
    """
    years = estimate_experience_years(resume_text)
    level = classify_experience_level(years)
    lo, hi = _SALARY_RANGES.get(level, (40_000, 65_000))
    return {"level": level, "salary_min": lo, "salary_max": hi, "currency": "USD"}
