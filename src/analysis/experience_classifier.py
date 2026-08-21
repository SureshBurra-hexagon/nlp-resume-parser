from __future__ import annotations

from src.ner.resume_entities import estimate_experience_years

_LEVEL_THRESHOLDS = [
    (0, "entry"),
    (2, "junior"),
    (5, "mid"),
    (8, "senior"),
    (12, "staff"),
    (16, "principal"),
]

_CAREER_PATHS: dict[str, list[str]] = {
    "entry": ["junior", "mid", "senior"],
    "junior": ["mid", "senior"],
    "mid": ["senior", "staff"],
    "senior": ["staff", "principal"],
    "staff": ["principal"],
    "principal": [],
}

_DOMAIN_TITLES: dict[str, list[str]] = {
    "data science": ["data scientist", "ml engineer", "research scientist", "ai engineer"],
    "software engineering": ["software engineer", "backend engineer", "full stack engineer", "developer"],
    "frontend": ["frontend engineer", "ui engineer", "react developer", "frontend developer"],
    "devops": ["devops engineer", "sre", "platform engineer", "infrastructure engineer"],
    "analytics": ["data analyst", "business analyst", "analytics engineer", "bi developer"],
    "management": ["engineering manager", "tech lead", "staff engineer", "principal engineer"],
}


def classify_experience_level(years: int | None) -> str:
    """Map years of experience to a career level label."""
    if years is None:
        return "entry"
    level = "entry"
    for threshold, label in _LEVEL_THRESHOLDS:
        if years >= threshold:
            level = label
    return level


def career_path_projection(current_level: str) -> list[str]:
    """Return typical next career levels from the current level."""
    return _CAREER_PATHS.get(current_level, [])


def classify_domain(resume_text: str) -> str:
    """Classify the professional domain of a resume using title keywords."""
    normalized = resume_text.lower()
    best_domain = "general"
    best_count = 0
    for domain, titles in _DOMAIN_TITLES.items():
        count = sum(1 for title in titles if title in normalized)
        if count > best_count:
            best_count = count
            best_domain = domain
    return best_domain


def analyze_career(resume_text: str) -> dict:
    """Return experience level, domain, and career path projection for a resume.

    Returns a dict with:
      - ``experience_years``: extracted years (or None)
      - ``level``: experience level label
      - ``domain``: inferred professional domain
      - ``next_levels``: list of projected next career levels
    """
    years = estimate_experience_years(resume_text)
    level = classify_experience_level(years)
    domain = classify_domain(resume_text)
    return {
        "experience_years": years,
        "level": level,
        "domain": domain,
        "next_levels": career_path_projection(level),
    }
