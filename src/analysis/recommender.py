from __future__ import annotations

from src.analysis.ats_scorer import compute_ats_score
from src.analysis.resume_scorer import compute_quality_score
from src.ner.resume_entities import extract_certifications, extract_sections
from src.utils.resume_parser import extract_skills

_SKILL_SUGGESTIONS: dict[str, list[str]] = {
    "data_science": ["python", "pytorch", "tensorflow", "sql", "nlp"],
    "frontend": ["react", "typescript", "javascript"],
    "devops": ["docker", "kubernetes", "aws", "ci/cd"],
    "analytics": ["sql", "tableau", "python"],
    "general": ["python", "sql", "aws"],
}

_SECTION_TIPS: dict[str, str] = {
    "summary": "Add a concise professional summary to introduce yourself.",
    "experience": "Include a dedicated work experience section with role details and achievements.",
    "education": "Add an education section listing your degrees and institutions.",
    "skills": "Add a skills section to make your technical competencies easy to scan.",
    "certifications": "List relevant certifications to strengthen your profile.",
    "projects": "Highlight key projects to demonstrate hands-on experience.",
}


def _missing_sections(resume_text: str) -> list[str]:
    present = set(extract_sections(resume_text).keys())
    return sorted(section for section in _SECTION_TIPS if section not in present)


def _missing_skills(resume_text: str, domain: str) -> list[str]:
    present = set(extract_skills(resume_text))
    suggested = _SKILL_SUGGESTIONS.get(domain, _SKILL_SUGGESTIONS["general"])
    return [skill for skill in suggested if skill not in present]


def generate_recommendations(
    resume_text: str,
    *,
    domain: str = "general",
    job_description: str = "",
) -> dict:
    """Generate actionable improvement recommendations for a resume.

    Returns a dict with:
      - ``quality_score``: overall quality score (0–100)
      - ``section_recommendations``: tips for missing sections
      - ``skill_recommendations``: skills to add for the target domain
      - ``ats_score``: ATS match score if ``job_description`` is provided (or None)
      - ``ats_missing_keywords``: JD keywords absent from the resume (or [])
      - ``certification_count``: number of recognised certifications found
      - ``overall_tips``: list of high-level improvement suggestions
    """
    quality = compute_quality_score(resume_text)
    section_recs = {
        section: _SECTION_TIPS[section] for section in _missing_sections(resume_text)
    }
    skill_recs = _missing_skills(resume_text, domain)
    certs = extract_certifications(resume_text)

    ats_score: float | None = None
    ats_missing: list[str] = []
    if job_description.strip():
        ats_result = compute_ats_score(resume_text, job_description)
        ats_score = ats_result["score"]
        ats_missing = ats_result["missing_keywords"]

    tips: list[str] = []
    if quality["total_score"] < 50:
        tips.append("Your resume score is below 50 — focus on adding missing sections and skills.")
    if not certs:
        tips.append("Consider adding professional certifications to strengthen your credentials.")
    if skill_recs:
        tips.append(f"Add domain-relevant skills: {', '.join(skill_recs[:3])}.")
    if ats_score is not None and ats_score < 0.5:
        tips.append("Your resume matches fewer than half the job description keywords — tailor it to the role.")

    return {
        "quality_score": quality["total_score"],
        "section_recommendations": section_recs,
        "skill_recommendations": skill_recs,
        "ats_score": ats_score,
        "ats_missing_keywords": ats_missing,
        "certification_count": len(certs),
        "overall_tips": tips,
    }
