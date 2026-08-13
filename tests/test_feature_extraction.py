"""
Tests for the feature extraction module.
"""

import pytest
import numpy as np

from src.feature_extraction import (
    HandcraftedFeatureExtractor,
    TFIDFExtractor,
)


SAMPLE_CORPUS = [
    "Python developer with machine learning experience",
    "Data scientist skilled in NLP and deep learning",
    "Software engineer with React and Node.js skills",
]


class TestTFIDFExtractor:
    def test_fit_transform_shape(self):
        extractor = TFIDFExtractor(max_features=50)
        matrix = extractor.fit_transform(SAMPLE_CORPUS)
        assert matrix.shape[0] == len(SAMPLE_CORPUS)
        assert matrix.shape[1] <= 50

    def test_transform_requires_fit(self):
        extractor = TFIDFExtractor()
        with pytest.raises(RuntimeError):
            extractor.transform(SAMPLE_CORPUS)

    def test_fit_then_transform(self):
        extractor = TFIDFExtractor(max_features=50)
        extractor.fit(SAMPLE_CORPUS)
        matrix = extractor.transform(SAMPLE_CORPUS)
        assert matrix.shape[0] == len(SAMPLE_CORPUS)

    def test_vocabulary_not_empty(self):
        extractor = TFIDFExtractor(max_features=50)
        extractor.fit(SAMPLE_CORPUS)
        assert len(extractor.vocabulary) > 0


class TestHandcraftedFeatureExtractor:
    def setup_method(self):
        self.extractor = HandcraftedFeatureExtractor()
        self.text = (
            "John Doe\n"
            "john@example.com | +1-555-123-4567\n"
            "linkedin.com/in/johndoe\n\n"
            "Education\n"
            "B.Tech CS, MIT, 2018–2022 GPA: 3.9\n\n"
            "Skills\n"
            "Python, ML, NLP\n\n"
            "Experience\n"
            "• Developed APIs using Flask\n"
            "• Led a team of 5 engineers\n"
        )

    def test_returns_dict(self):
        features = self.extractor.extract(self.text)
        assert isinstance(features, dict)

    def test_has_email(self):
        features = self.extractor.extract(self.text)
        assert features["has_email"] is True

    def test_has_phone(self):
        features = self.extractor.extract(self.text)
        assert features["has_phone"] is True

    def test_has_linkedin(self):
        features = self.extractor.extract(self.text)
        assert features["has_linkedin"] is True

    def test_has_gpa(self):
        features = self.extractor.extract(self.text)
        assert features["has_gpa"] is True

    def test_bullet_count(self):
        features = self.extractor.extract(self.text)
        assert features["num_bullet_points"] >= 2

    def test_section_flags(self):
        features = self.extractor.extract(self.text)
        assert features["has_education_section"] is True
        assert features["has_skills_section"] is True
        assert features["has_experience_section"] is True

    def test_no_features_on_empty(self):
        features = self.extractor.extract("")
        assert features["has_email"] is False
        assert features["has_phone"] is False

    def test_to_matrix(self):
        matrix = self.extractor.to_matrix(SAMPLE_CORPUS)
        assert matrix.shape[0] == len(SAMPLE_CORPUS)
        assert matrix.shape[1] == len(self.extractor.feature_names)
        assert matrix.dtype == float

    def test_feature_names_consistent(self):
        names = self.extractor.feature_names
        features = self.extractor.extract(self.text)
        assert set(names) == set(features.keys())
