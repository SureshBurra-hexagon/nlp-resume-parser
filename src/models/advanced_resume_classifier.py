from __future__ import annotations

import joblib
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline


def build_advanced_pipeline(random_state: int = 42) -> Pipeline:
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    max_features=3000,
                    ngram_range=(1, 2),
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    max_features=1500,
                    ngram_range=(3, 5),
                ),
            ),
        ]
    )
    classifier = VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(random_state=random_state, max_iter=2000)),
            ("sgd", SGDClassifier(random_state=random_state, loss="log_loss", max_iter=2000)),
            ("nb", ComplementNB(alpha=0.5)),
        ],
        voting="soft",
    )
    return Pipeline([("features", features), ("classifier", classifier)])


class AdvancedResumeClassifier:
    """Advanced classifier using hybrid features and a soft-voting ensemble."""

    def __init__(self, random_state: int = 42, pipeline: Pipeline | None = None) -> None:
        self.model = pipeline or build_advanced_pipeline(random_state=random_state)

    def fit(self, texts: list[str], labels: list[str]) -> "AdvancedResumeClassifier":
        self.model.fit(texts, labels)
        return self

    def predict(self, texts: list[str]):
        return self.model.predict(texts)

    def predict_proba(self, texts: list[str]):
        return self.model.predict_proba(texts)

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str) -> "AdvancedResumeClassifier":
        return cls(pipeline=joblib.load(path))
