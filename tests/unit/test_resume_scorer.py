from src.analysis.resume_scorer import compute_quality_score, estimate_salary


RICH_RESUME = (
    "Summary: Senior Data Scientist with 8 years of experience.\n"
    "Experience: Led ML teams at multiple companies.\n"
    "Education: Master of Science in Computer Science. Bachelor of Science in Mathematics.\n"
    "Skills: Python, SQL, NLP, PyTorch, AWS, Docker, Kubernetes, React, TensorFlow, FastAPI\n"
    "Certifications: AWS Certified Solutions Architect. PMP certification holder."
)

SPARSE_RESUME = "Python developer."


def test_compute_quality_score_returns_all_keys():
    result = compute_quality_score(RICH_RESUME)
    assert "total_score" in result
    assert "education_score" in result
    assert "skill_score" in result
    assert "experience_score" in result
    assert "section_score" in result
    assert "certification_score" in result


def test_compute_quality_score_rich_resume_high():
    result = compute_quality_score(RICH_RESUME)
    assert result["total_score"] > 50


def test_compute_quality_score_sparse_resume_low():
    result = compute_quality_score(SPARSE_RESUME)
    assert result["total_score"] < 50


def test_compute_quality_score_bounded_0_to_100():
    result = compute_quality_score(RICH_RESUME)
    assert 0 <= result["total_score"] <= 100


def test_compute_quality_score_education_contributes():
    result = compute_quality_score(RICH_RESUME)
    assert result["education_score"] > 0


def test_compute_quality_score_skills_contribute():
    result = compute_quality_score(RICH_RESUME)
    assert result["skill_score"] > 0


def test_compute_quality_score_certifications_contribute():
    result = compute_quality_score(RICH_RESUME)
    assert result["certification_score"] > 0


def test_compute_quality_score_empty_resume():
    result = compute_quality_score("")
    assert result["total_score"] == 0.0


def test_estimate_salary_returns_all_keys():
    result = estimate_salary(RICH_RESUME)
    assert "level" in result
    assert "salary_min" in result
    assert "salary_max" in result
    assert "currency" in result


def test_estimate_salary_currency_usd():
    result = estimate_salary(RICH_RESUME)
    assert result["currency"] == "USD"


def test_estimate_salary_min_less_than_max():
    result = estimate_salary(RICH_RESUME)
    assert result["salary_min"] < result["salary_max"]


def test_estimate_salary_senior_level():
    result = estimate_salary("Senior engineer with 9 years of experience.")
    assert result["level"] == "senior"
    assert result["salary_min"] >= 100_000


def test_estimate_salary_entry_level():
    result = estimate_salary("Recent graduate looking for first role.")
    assert result["level"] == "entry"
    assert result["salary_max"] <= 70_000
