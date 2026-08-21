from __future__ import annotations

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


class HybridFeatureExtractor:
    """Combine word and character TF-IDF features for richer resume representations."""

    def __init__(
        self,
        word_max_features: int = 3000,
        char_max_features: int = 1500,
        word_ngram_range: tuple[int, int] = (1, 2),
        char_ngram_range: tuple[int, int] = (3, 5),
    ) -> None:
        self.vectorizer = FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        max_features=word_max_features,
                        ngram_range=word_ngram_range,
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        max_features=char_max_features,
                        ngram_range=char_ngram_range,
                    ),
                ),
            ]
        )

    def fit(self, texts: list[str]) -> "HybridFeatureExtractor":
        self.vectorizer.fit(texts)
        return self

    def transform(self, texts: list[str]):
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: list[str]):
        return self.vectorizer.fit_transform(texts)

    def save(self, path: str) -> None:
        joblib.dump(self.vectorizer, path)

    @classmethod
    def load(cls, path: str) -> "HybridFeatureExtractor":
        instance = cls()
        instance.vectorizer = joblib.load(path)
        return instance
