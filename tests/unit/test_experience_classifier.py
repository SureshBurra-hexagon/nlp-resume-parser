from src.analysis.experience_classifier import (
    analyze_career,
    career_path_projection,
    classify_domain,
    classify_experience_level,
)


def test_classify_experience_level_entry():
    assert classify_experience_level(0) == "entry"
    assert classify_experience_level(None) == "entry"


def test_classify_experience_level_junior():
    assert classify_experience_level(2) == "junior"
    assert classify_experience_level(3) == "junior"


def test_classify_experience_level_mid():
    assert classify_experience_level(5) == "mid"
    assert classify_experience_level(7) == "mid"


def test_classify_experience_level_senior():
    assert classify_experience_level(8) == "senior"
    assert classify_experience_level(10) == "senior"


def test_classify_experience_level_staff():
    assert classify_experience_level(12) == "staff"


def test_classify_experience_level_principal():
    assert classify_experience_level(16) == "principal"
    assert classify_experience_level(20) == "principal"


def test_career_path_projection_entry():
    path = career_path_projection("entry")
    assert "junior" in path
    assert "mid" in path


def test_career_path_projection_principal_is_empty():
    assert career_path_projection("principal") == []


def test_career_path_projection_unknown_level():
    assert career_path_projection("unknown") == []


def test_classify_domain_data_science():
    text = "Data scientist with expertise in machine learning and NLP. ML engineer background."
    assert classify_domain(text) == "data science"


def test_classify_domain_devops():
    text = "DevOps engineer. SRE and platform engineer experience with infrastructure automation."
    assert classify_domain(text) == "devops"


def test_classify_domain_general_fallback():
    text = "Experienced professional seeking new opportunities."
    assert classify_domain(text) == "general"


def test_analyze_career_returns_all_keys():
    result = analyze_career("Senior software engineer with 9 years of experience.")
    assert "experience_years" in result
    assert "level" in result
    assert "domain" in result
    assert "next_levels" in result


def test_analyze_career_experience_years():
    result = analyze_career("Data scientist with 6 years experience in NLP and machine learning.")
    assert result["experience_years"] == 6
    assert result["level"] == "mid"


def test_analyze_career_no_years():
    result = analyze_career("Software engineer with strong Python skills.")
    assert result["experience_years"] is None
    assert result["level"] == "entry"
