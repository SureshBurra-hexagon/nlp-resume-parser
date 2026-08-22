from src.analysis.recommender import generate_recommendations


MINIMAL_RESUME = "Python developer."

GOOD_RESUME = (
    "Summary: Data Scientist with 6 years of experience in NLP and machine learning.\n"
    "Experience: Built production ML pipelines at multiple startups.\n"
    "Education: Master of Science in Computer Science.\n"
    "Skills: Python, SQL, NLP, PyTorch, AWS, Docker, TensorFlow, FastAPI\n"
    "Certifications: AWS Certified Solutions Architect."
)

JD = (
    "Looking for a Python engineer with NLP, machine learning, and FastAPI skills. "
    "Docker and AWS experience required. SQL is a plus."
)


def test_generate_recommendations_returns_all_keys():
    result = generate_recommendations(MINIMAL_RESUME)
    assert "quality_score" in result
    assert "section_recommendations" in result
    assert "skill_recommendations" in result
    assert "ats_score" in result
    assert "ats_missing_keywords" in result
    assert "certification_count" in result
    assert "overall_tips" in result


def test_generate_recommendations_no_jd_ats_score_is_none():
    result = generate_recommendations(MINIMAL_RESUME)
    assert result["ats_score"] is None
    assert result["ats_missing_keywords"] == []


def test_generate_recommendations_with_jd_ats_score_provided():
    result = generate_recommendations(GOOD_RESUME, job_description=JD)
    assert result["ats_score"] is not None
    assert isinstance(result["ats_score"], float)


def test_generate_recommendations_minimal_resume_has_tips():
    result = generate_recommendations(MINIMAL_RESUME)
    assert len(result["overall_tips"]) > 0


def test_generate_recommendations_good_resume_fewer_section_recs():
    good = generate_recommendations(GOOD_RESUME)
    minimal = generate_recommendations(MINIMAL_RESUME)
    assert len(good["section_recommendations"]) <= len(minimal["section_recommendations"])


def test_generate_recommendations_skill_recommendations_list():
    result = generate_recommendations(MINIMAL_RESUME, domain="devops")
    assert isinstance(result["skill_recommendations"], list)


def test_generate_recommendations_domain_filters_skills():
    devops_result = generate_recommendations(MINIMAL_RESUME, domain="devops")
    ds_result = generate_recommendations(MINIMAL_RESUME, domain="data_science")
    assert devops_result["skill_recommendations"] != ds_result["skill_recommendations"]


def test_generate_recommendations_no_certs_tip_present():
    result = generate_recommendations(MINIMAL_RESUME)
    assert any("certification" in tip.lower() for tip in result["overall_tips"])


def test_generate_recommendations_quality_score_float():
    result = generate_recommendations(GOOD_RESUME)
    assert isinstance(result["quality_score"], float)
    assert 0 <= result["quality_score"] <= 100


def test_generate_recommendations_good_resume_cert_count():
    result = generate_recommendations(GOOD_RESUME)
    assert result["certification_count"] >= 1
