from src.ner.resume_entities import (
    estimate_experience_years,
    extract_certifications,
    extract_education_entities,
    extract_sections,
)


def test_extract_sections_basic():
    text = (
        "Summary: Experienced software engineer\n"
        "Experience: Led backend projects\n"
        "Skills: Python, AWS\n"
        "Education: BS Computer Science"
    )
    sections = extract_sections(text)
    assert "summary" in sections
    assert "experience" in sections
    assert "skills" in sections
    assert "education" in sections


def test_extract_sections_empty_text():
    assert extract_sections("") == {}


def test_extract_sections_no_headers():
    text = "Python developer with 5 years of experience in machine learning"
    sections = extract_sections(text)
    assert "summary" in sections


def test_extract_sections_alias_matching():
    text = "Work Experience:\nBuilt microservices\nAcademics:\nBS Computer Science"
    sections = extract_sections(text)
    assert "experience" in sections
    assert "education" in sections


def test_extract_education_bachelor():
    assert "bachelor" in extract_education_entities("Bachelor of Science in Computer Science")


def test_extract_education_master():
    assert "master" in extract_education_entities("Master of Science in Data Science")


def test_extract_education_mba():
    assert "mba" in extract_education_entities("holds an MBA from a top business school")


def test_extract_education_phd():
    assert "phd" in extract_education_entities("PhD in Artificial Intelligence")


def test_extract_education_multiple():
    text = "BS in CS and MBA and PhD research"
    result = extract_education_entities(text)
    assert "bachelor" in result
    assert "mba" in result
    assert "phd" in result


def test_extract_education_empty():
    assert extract_education_entities("") == []


def test_extract_certifications_aws():
    assert "aws certified" in extract_certifications("AWS Certified Solutions Architect")


def test_extract_certifications_pmp():
    assert "pmp" in extract_certifications("holds PMP certification")


def test_extract_certifications_scrum():
    assert "scrum master" in extract_certifications("Certified Scrum Master (CSM)")


def test_extract_certifications_gcp():
    assert "gcp certified" in extract_certifications("Google Cloud Certified Professional")


def test_extract_certifications_empty():
    assert extract_certifications("") == []


def test_estimate_experience_years_single():
    assert estimate_experience_years("5 years of experience") == 5


def test_estimate_experience_years_returns_max():
    assert estimate_experience_years("3 years in Python and 7 years in Java") == 7


def test_estimate_experience_years_none():
    assert estimate_experience_years("software engineer") is None


def test_estimate_experience_years_with_plus():
    assert estimate_experience_years("10+ years of professional experience") == 10
