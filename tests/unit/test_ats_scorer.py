from src.analysis.ats_scorer import compute_ats_score, extract_jd_keywords


JD = (
    "We are looking for a Python engineer with experience in machine learning, "
    "NLP, FastAPI, Docker, and cloud platforms such as AWS. "
    "Strong SQL skills required. Experience with PyTorch or TensorFlow is a plus."
)

MATCHING_RESUME = (
    "Python developer with 5 years experience in machine learning, NLP, and FastAPI. "
    "Proficient in Docker, AWS, SQL, and PyTorch."
)

UNRELATED_RESUME = "Graphic designer with expertise in Photoshop, Illustrator, and brand identity."


def test_extract_jd_keywords_returns_list():
    keywords = extract_jd_keywords(JD)
    assert isinstance(keywords, list)
    assert len(keywords) > 0


def test_extract_jd_keywords_top_n_respected():
    keywords = extract_jd_keywords(JD, top_n=5)
    assert len(keywords) <= 5


def test_compute_ats_score_high_for_matching_resume():
    result = compute_ats_score(MATCHING_RESUME, JD)
    assert result["score"] > 0.3
    assert "python" in result["matched_keywords"]
    assert isinstance(result["matched_keywords"], list)
    assert isinstance(result["missing_keywords"], list)
    assert result["total_keywords"] > 0


def test_compute_ats_score_low_for_unrelated_resume():
    result = compute_ats_score(UNRELATED_RESUME, JD)
    assert result["score"] < 0.5


def test_compute_ats_score_empty_jd():
    result = compute_ats_score(MATCHING_RESUME, "")
    assert result["score"] == 0.0
    assert result["matched_keywords"] == []
    assert result["total_keywords"] == 0


def test_compute_ats_score_empty_resume():
    result = compute_ats_score("", JD)
    assert result["score"] == 0.0


def test_compute_ats_score_matched_plus_missing_equals_total():
    result = compute_ats_score(MATCHING_RESUME, JD)
    assert len(result["matched_keywords"]) + len(result["missing_keywords"]) == result["total_keywords"]
