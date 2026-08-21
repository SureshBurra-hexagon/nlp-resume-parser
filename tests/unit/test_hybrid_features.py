import numpy as np

from src.embedding.hybrid_features import HybridFeatureExtractor


SAMPLE_TEXTS = [
    "python nlp machine learning fastapi",
    "react javascript frontend ui components",
    "sql tableau business analytics reporting",
    "aws docker kubernetes devops infrastructure",
]


def test_fit_transform_returns_matrix():
    extractor = HybridFeatureExtractor()
    matrix = extractor.fit_transform(SAMPLE_TEXTS)
    assert matrix.shape[0] == len(SAMPLE_TEXTS)
    assert matrix.shape[1] > 0


def test_transform_same_shape_as_fit_transform():
    extractor = HybridFeatureExtractor()
    extractor.fit(SAMPLE_TEXTS)
    transformed = extractor.transform(SAMPLE_TEXTS)
    fit_transformed = HybridFeatureExtractor().fit_transform(SAMPLE_TEXTS)
    assert transformed.shape == fit_transformed.shape


def test_custom_feature_dimensions():
    extractor = HybridFeatureExtractor(word_max_features=500, char_max_features=300)
    matrix = extractor.fit_transform(SAMPLE_TEXTS)
    assert matrix.shape[1] <= 800


def test_sparse_output_is_numeric():
    extractor = HybridFeatureExtractor()
    matrix = extractor.fit_transform(SAMPLE_TEXTS)
    dense = matrix.toarray()
    assert np.isfinite(dense).all()


def test_save_and_load(tmp_path):
    extractor = HybridFeatureExtractor()
    extractor.fit(SAMPLE_TEXTS)
    path = str(tmp_path / "hybrid.joblib")
    extractor.save(path)
    loaded = HybridFeatureExtractor.load(path)
    original = extractor.transform(SAMPLE_TEXTS).toarray()
    restored = loaded.transform(SAMPLE_TEXTS).toarray()
    np.testing.assert_array_equal(original, restored)
