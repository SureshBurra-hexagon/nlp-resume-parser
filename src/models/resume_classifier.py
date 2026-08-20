from __future__ import annotations

import joblib
from sklearn.linear_model import LogisticRegression


class ResumeClassifier:
    """Baseline resume classifier using Logistic Regression."""

    def __init__(self, random_state: int = 42, max_iter: int = 1000) -> None:
        self.model = LogisticRegression(random_state=random_state, max_iter=max_iter)

    def fit(self, features, labels) -> None:
        self.model.fit(features, labels)

    def predict(self, features):
        return self.model.predict(features)

    def predict_proba(self, features):
        return self.model.predict_proba(features)

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str) -> "ResumeClassifier":
        instance = cls()
        instance.model = joblib.load(path)
        return instance
