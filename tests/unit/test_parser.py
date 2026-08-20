from src.utils.resume_parser import parse_resume


def test_parse_resume_extracts_core_fields():
    text = "Email: jane@example.com Skills: Python SQL NLP CI/CD"
    parsed = parse_resume(text)
    assert "jane@example.com" in parsed["emails"]
    assert set(parsed["skills"]) >= {"python", "sql", "nlp"}
    assert "ci/cd" in parsed["skills"]
