import numpy as np

from src.models.transformer_resume_classifier import TransformerResumeClassifier


class DummyEmbedder:
    def embed(self, texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            tokens = text.lower().split()
            rows.append(
                [
                    float(tokens.count("python")),
                    float(tokens.count("react")),
                    float(tokens.count("sql")),
                    float(len(tokens)),
                ]
            )
        return np.array(rows, dtype=np.float32)


def test_transformer_classifier_fit_predict_and_proba():
    texts = [
        "python fastapi nlp pipelines",
        "python machine learning sql",
        "react frontend ui components",
        "react javascript design systems",
    ]
    labels = ["data_science", "data_science", "frontend", "frontend"]

    model = TransformerResumeClassifier(embedder=DummyEmbedder())
    model.fit(texts, labels)

    prediction = model.predict(["python sql ml"])
    probabilities = model.predict_proba(["python sql ml"])

    assert prediction[0] == "data_science"
    assert probabilities.shape == (1, 2)
    assert abs(float(probabilities[0].sum()) - 1.0) < 1e-6


def test_transformer_classifier_save_and_load(tmp_path):
    texts = [
        "python fastapi nlp pipelines",
        "react frontend ui components",
    ]
    labels = ["data_science", "frontend"]

    model = TransformerResumeClassifier(embedder=DummyEmbedder())
    model.fit(texts, labels)

    artifact = tmp_path / "transformer_classifier.joblib"
    model.save(str(artifact))

    loaded = TransformerResumeClassifier.load(str(artifact), embedder=DummyEmbedder())
    prediction = loaded.predict(["react typescript ui"])

    assert prediction[0] == "frontend"
