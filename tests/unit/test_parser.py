from src.utils.resume_parser import parse_resume


def test_parse_resume_extracts_core_fields():
    text = "Email: jane@example.com Skills: Python SQL NLP CI/CD"
    parsed = parse_resume(text)
    assert "jane@example.com" in parsed["emails"]
    assert set(parsed["skills"]) >= {"python", "sql", "nlp"}
    assert "ci/cd" in parsed["skills"]


def test_parse_resume_extracts_sections_and_resume_entities():
    text = """
    Summary: Data Scientist with 6 years experience
    Education: Bachelor of Science in Computer Science
    Certifications: AWS Certified Solutions Architect, PMP
    Skills: Python SQL NLP
    """
    parsed = parse_resume(text)

    assert parsed["sections"]["summary"] == "data scientist with 6 years experience"
    assert parsed["sections"]["education"] == "bachelor of science in computer science"
    assert set(parsed["education"]) >= {"bachelor"}
    assert set(parsed["certifications"]) >= {"aws certified", "pmp"}
    assert parsed["experience_years"] == 6
