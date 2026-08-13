"""
Tests for the model module.
"""

import pytest

from src.model import (
    ResumeParser,
    SkillsExtractor,
    extract_education,
    TECHNICAL_SKILLS,
    SOFT_SKILLS,
)


SAMPLE_RESUME = """
Jane Smith
jane.smith@email.com | +1-555-987-6543
linkedin.com/in/janesmith

Summary
Data scientist with 4 years of experience in machine learning and NLP.

Education
M.S. in Data Science
Stanford University, 2016 – 2018
GPA: 3.9/4.0

B.S. in Statistics
UCLA, 2012 – 2016

Experience
Data Scientist, TechCorp, 2018 – Present
- Built NLP pipelines using Python and spaCy
- Trained transformer-based models for text classification
- Deployed models on AWS using Docker and Kubernetes

Junior Data Analyst, StartupX, 2016 – 2018
- Performed data analysis with pandas and numpy
- Created dashboards in Tableau

Skills
Python, R, Machine Learning, Deep Learning, NLP, TensorFlow, PyTorch,
scikit-learn, pandas, numpy, SQL, Docker, Kubernetes, AWS, Git

Certifications
AWS Certified Machine Learning Specialty – 2021

Projects
Sentiment Analysis System
- Built a BERT-based sentiment classifier achieving 92% accuracy
"""


class TestSkillsExtractor:
    def setup_method(self):
        self.extractor = SkillsExtractor()

    def test_extracts_technical_skills(self):
        skills = self.extractor.extract("Proficient in Python, Docker, and SQL")
        assert "python" in skills["technical"]
        assert "docker" in skills["technical"]
        assert "sql" in skills["technical"]

    def test_extracts_soft_skills(self):
        skills = self.extractor.extract("Strong leadership and communication skills")
        assert "leadership" in skills["soft"] or "communication" in skills["soft"]

    def test_returns_dict_with_keys(self):
        skills = self.extractor.extract("Python developer")
        assert "technical" in skills
        assert "soft" in skills

    def test_no_skills(self):
        skills = self.extractor.extract("This text has no relevant skills.")
        assert isinstance(skills["technical"], list)
        assert isinstance(skills["soft"], list)

    def test_deduplication(self):
        skills = self.extractor.extract("Python Python Python developer")
        assert skills["technical"].count("python") == 1

    def test_skills_database_non_empty(self):
        assert len(TECHNICAL_SKILLS) > 0
        assert len(SOFT_SKILLS) > 0


class TestExtractEducation:
    def test_detects_degree(self):
        text = "M.S. in Computer Science\nMIT, 2018-2020"
        results = extract_education(text)
        assert len(results) >= 1

    def test_extracts_year_range(self):
        text = "B.Tech in CS\nUniversity, 2014 – 2018"
        results = extract_education(text)
        assert results[0]["start_year"] == "2014"
        assert results[0]["end_year"] == "2018"

    def test_extracts_gpa(self):
        text = "B.S. Mathematics\nUCLA, 2012-2016\nGPA: 3.7/4.0"
        results = extract_education(text)
        assert any(r["gpa"] is not None for r in results)

    def test_empty_text(self):
        results = extract_education("")
        assert results == []


class TestResumeParser:
    def setup_method(self):
        self.parser = ResumeParser(use_ner=False)

    def test_parse_returns_dict(self):
        result = self.parser.parse(SAMPLE_RESUME)
        assert isinstance(result, dict)

    def test_parse_has_required_keys(self):
        result = self.parser.parse(SAMPLE_RESUME)
        for key in ["raw_text", "sections", "contact", "skills", "education"]:
            assert key in result, f"Missing key: {key}"

    def test_contact_extraction(self):
        result = self.parser.parse(SAMPLE_RESUME)
        assert result["contact"]["email"] == "jane.smith@email.com"

    def test_skills_extraction(self):
        result = self.parser.parse(SAMPLE_RESUME)
        assert "python" in result["skills"]["technical"]

    def test_sections_identified(self):
        result = self.parser.parse(SAMPLE_RESUME)
        # At least one expected section should be present
        present = set(result["sections"].keys())
        assert present & {"education", "experience", "skills"} != set()

    def test_education_extraction(self):
        result = self.parser.parse(SAMPLE_RESUME)
        assert isinstance(result["education"], list)
