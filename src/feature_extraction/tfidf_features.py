from __future__ import annotations

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfFeatureExtractor:
    """TF-IDF feature extractor for resume classification."""

    def __init__(self, max_features: int = 3000, ngram_range: tuple[int, int] = (1, 2)) -> None:
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)

    def fit(self, texts: list[str]) -> "TfidfFeatureExtractor":
        self.vectorizer.fit(texts)
        return self

    def transform(self, texts: list[str]):
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: list[str]):
        return self.vectorizer.fit_transform(texts)

    def save(self, path: str) -> None:
        joblib.dump(self.vectorizer, path)

    @classmethod
    def load(cls, path: str) -> "TfidfFeatureExtractor":
        instance = cls()
        instance.vectorizer = joblib.load(path)
        return instance
