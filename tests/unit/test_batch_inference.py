from src.optimization.inference import ResumeBatchProcessor, cached_parse_resume


SAMPLE_TEXT = (
    "Summary: Python engineer\n"
    "Skills: Python, AWS\n"
    "Education: Bachelor of Science in Computer Science\n"
    "Certifications: AWS Certified Solutions Architect\n"
    "5 years of experience in cloud infrastructure"
)


def test_cached_parse_resume_returns_expected_keys():
    result = cached_parse_resume(SAMPLE_TEXT)
    assert "normalized_text" in result
    assert "emails" in result
    assert "phones" in result
    assert "skills" in result
    assert "sections" in result
    assert "education" in result
    assert "certifications" in result
    assert "experience_years" in result


def test_cached_parse_resume_extracts_skills():
    result = cached_parse_resume(SAMPLE_TEXT)
    assert "python" in result["skills"]
    assert "aws" in result["skills"]


def test_cached_parse_resume_extracts_education():
    result = cached_parse_resume(SAMPLE_TEXT)
    assert "bachelor" in result["education"]


def test_cached_parse_resume_extracts_certifications():
    result = cached_parse_resume(SAMPLE_TEXT)
    assert "aws certified" in result["certifications"]


def test_cached_parse_resume_extracts_experience_years():
    result = cached_parse_resume(SAMPLE_TEXT)
    assert result["experience_years"] == 5


def test_cached_parse_resume_returns_plain_types():
    result = cached_parse_resume(SAMPLE_TEXT)
    assert isinstance(result["emails"], list)
    assert isinstance(result["phones"], list)
    assert isinstance(result["skills"], list)
    assert isinstance(result["sections"], dict)
    assert isinstance(result["education"], list)
    assert isinstance(result["certifications"], list)


def test_batch_processor_parse_single():
    processor = ResumeBatchProcessor()
    result = processor.parse(SAMPLE_TEXT)
    assert "normalized_text" in result
    assert "skills" in result


def test_batch_processor_parse_batch():
    processor = ResumeBatchProcessor()
    texts = [SAMPLE_TEXT, "React frontend developer with 3 years experience"]
    results = processor.parse_batch(texts)
    assert len(results) == 2
    assert all("normalized_text" in r for r in results)


def test_batch_processor_normalized_batch():
    processor = ResumeBatchProcessor()
    texts = [SAMPLE_TEXT, "React frontend developer with 3 years experience"]
    normalized = processor.normalized_batch(texts)
    assert len(normalized) == 2
    assert all(isinstance(s, str) for s in normalized)


def test_cached_parse_resume_empty_text():
    result = cached_parse_resume("")
    assert result["skills"] == []
    assert result["emails"] == []
