from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.embedding.transformer_features import HuggingFaceTextEmbedder, TextEmbedder


class TransformerResumeClassifier:
    """Transformer embedding + logistic regression classifier."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        random_state: int = 42,
        embedder: TextEmbedder | None = None,
        classifier: LogisticRegression | None = None,
    ) -> None:
        self.model_name = model_name
        self.random_state = random_state
        self.embedder = embedder or HuggingFaceTextEmbedder(model_name=model_name)
        self.classifier = classifier or LogisticRegression(max_iter=3000, random_state=random_state)

    def fit(self, texts: list[str], labels: list[str]) -> "TransformerResumeClassifier":
        embeddings = self.embedder.embed(texts)
        if embeddings.size == 0:
            raise ValueError("Cannot train transformer classifier with an empty dataset.")
        self.classifier.fit(embeddings, labels)
        return self

    def predict(self, texts: list[str]) -> np.ndarray:
        embeddings = self.embedder.embed(texts)
        return self.classifier.predict(embeddings)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        embeddings = self.embedder.embed(texts)
        return self.classifier.predict_proba(embeddings)

    def save(self, path: str) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": self.model_name,
            "random_state": self.random_state,
            "classifier": self.classifier,
        }
        joblib.dump(payload, output_path)

    @classmethod
    def load(
        cls,
        path: str,
        embedder: TextEmbedder | None = None,
    ) -> "TransformerResumeClassifier":
        payload: Any = joblib.load(path)
        if isinstance(payload, dict):
            if "classifier" not in payload:
                raise ValueError("Unsupported transformer classifier artifact format: missing `classifier` payload.")
            model_name = str(payload.get("model_name", "distilbert-base-uncased"))
            random_state = int(payload.get("random_state", 42))
            classifier = payload["classifier"]
            return cls(
                model_name=model_name,
                random_state=random_state,
                embedder=embedder or HuggingFaceTextEmbedder(model_name=model_name),
                classifier=classifier,
            )

        if isinstance(payload, LogisticRegression):
            return cls(embedder=embedder, classifier=payload)

        raise ValueError("Unsupported transformer classifier artifact format.")
