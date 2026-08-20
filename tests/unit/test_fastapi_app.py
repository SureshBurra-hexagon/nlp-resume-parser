from pathlib import Path

from fastapi.testclient import TestClient

from fastapi_app import main as api
from src.feature_extraction.tfidf_features import TfidfFeatureExtractor
from src.models.resume_classifier import ResumeClassifier
from src.preprocessing.text_cleaner import clean_text


def _write_artifacts(model_dir: Path) -> None:
    texts = [
        clean_text("python fastapi machine learning"),
        clean_text("react javascript frontend"),
    ]
    labels = ["data_science", "frontend"]

    extractor = TfidfFeatureExtractor(max_features=50)
    features = extractor.fit_transform(texts)

    classifier = ResumeClassifier(max_iter=2000)
    classifier.fit(features, labels)

    model_dir.mkdir(parents=True, exist_ok=True)
    extractor.save(str(model_dir / "tfidf.joblib"))
    classifier.save(str(model_dir / "classifier.joblib"))


def test_health_reports_missing_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "MODEL_DIR", tmp_path / "missing-models")
    api.load_artifacts.cache_clear()

    client = TestClient(api.app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "artifacts_loaded": False}


def test_parse_endpoint_returns_core_fields():
    client = TestClient(api.app)

    response = client.post(
        "/parse",
        json={"text": "Email jane@example.com Skills Python SQL NLP CI/CD"},
    )

    assert response.status_code == 200
    body = response.json()
    assert "jane@example.com" in body["emails"]
    assert set(body["skills"]) >= {"python", "sql", "nlp"}
    assert body["normalized_text"] == "email jane@example.com skills python sql nlp ci/cd"


def test_predict_endpoint_returns_prediction(tmp_path, monkeypatch):
    model_dir = tmp_path / "baseline"
    _write_artifacts(model_dir)
    monkeypatch.setattr(api, "MODEL_DIR", model_dir)
    api.load_artifacts.cache_clear()

    client = TestClient(api.app)
    response = client.post("/predict", json={"text": "Python FastAPI NLP pipelines"})

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_profile_category"] == "data_science"
    assert "python" in body["parsed_resume"]["skills"]


def test_predict_endpoint_requires_trained_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "MODEL_DIR", tmp_path / "baseline")
    api.load_artifacts.cache_clear()

    client = TestClient(api.app)
    response = client.post("/predict", json={"text": "Python FastAPI NLP pipelines"})

    assert response.status_code == 503
    assert "Run `python scripts/train.py` first" in response.json()["detail"]
