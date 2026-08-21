from pathlib import Path

from fastapi.testclient import TestClient

from fastapi_app import main as api
from src.models.advanced_resume_classifier import AdvancedResumeClassifier
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


def _write_advanced_artifacts(model_dir: Path) -> None:
    texts = [
        clean_text("python fastapi machine learning"),
        clean_text("python nlp data pipelines"),
        clean_text("react javascript frontend"),
        clean_text("react component design"),
    ]
    labels = ["data_science", "data_science", "frontend", "frontend"]

    classifier = AdvancedResumeClassifier()
    classifier.fit(texts, labels)

    model_dir.mkdir(parents=True, exist_ok=True)
    classifier.save(str(model_dir / "advanced_pipeline.joblib"))


def test_health_reports_missing_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "BASELINE_MODEL_DIR", tmp_path / "missing-baseline")
    monkeypatch.setattr(api, "ADVANCED_MODEL_DIR", tmp_path / "missing-advanced")
    api.load_artifacts.cache_clear()
    api.load_advanced_artifacts.cache_clear()

    client = TestClient(api.app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "baseline_artifacts_loaded": False,
        "advanced_artifacts_loaded": False,
    }


def test_parse_endpoint_returns_core_fields():
    client = TestClient(api.app)

    response = client.post(
        "/parse",
        json={
            "text": "Email jane@example.com\nSummary: Data Scientist with 6 years experience\nEducation: Bachelor of Science\nCertifications: AWS Certified Solutions Architect\nSkills: Python SQL NLP CI/CD"
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert "jane@example.com" in body["emails"]
    assert set(body["skills"]) >= {"python", "sql", "nlp"}
    assert body["normalized_text"] == "email jane@example.com summary: data scientist with 6 years experience education: bachelor of science certifications: aws certified solutions architect skills: python sql nlp ci/cd"
    assert body["experience_years"] == 6
    assert body["certifications"] == ["aws certified"]


def test_predict_endpoint_returns_prediction(tmp_path, monkeypatch):
    model_dir = tmp_path / "baseline"
    _write_artifacts(model_dir)
    monkeypatch.setattr(api, "BASELINE_MODEL_DIR", model_dir)
    api.load_artifacts.cache_clear()

    client = TestClient(api.app)
    response = client.post("/predict", json={"text": "Python FastAPI NLP pipelines"})

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_profile_category"] == "data_science"
    assert "python" in body["parsed_resume"]["skills"]
    assert body["confidence"] is not None


def test_predict_endpoint_requires_trained_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "BASELINE_MODEL_DIR", tmp_path / "baseline")
    api.load_artifacts.cache_clear()

    client = TestClient(api.app)
    response = client.post("/predict", json={"text": "Python FastAPI NLP pipelines"})

    assert response.status_code == 503
    assert "Run `python scripts/train.py` first" in response.json()["detail"]


def test_advanced_predict_endpoints_return_predictions(tmp_path, monkeypatch):
    model_dir = tmp_path / "advanced"
    _write_advanced_artifacts(model_dir)
    monkeypatch.setattr(api, "ADVANCED_MODEL_DIR", model_dir)
    api.load_advanced_artifacts.cache_clear()

    client = TestClient(api.app)
    response = client.post("/predict/advanced", json={"text": "Python FastAPI NLP pipelines"})

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_profile_category"] == "data_science"
    assert body["confidence"] is not None

    batch_response = client.post(
        "/predict/advanced/batch",
        json={"texts": ["Python FastAPI NLP pipelines", "React component libraries"]},
    )

    assert batch_response.status_code == 200
    predictions = batch_response.json()["predictions"]
    assert len(predictions) == 2
    assert predictions[0]["predicted_profile_category"] == "data_science"


def test_evaluate_endpoint_returns_metrics():
    client = TestClient(api.app)
    response = client.post(
        "/evaluate",
        json={
            "labels": ["data_science", "frontend", "frontend"],
            "predictions": ["data_science", "frontend", "data_science"],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["sample_count"] == 3
    assert set(body["metrics"]) >= {"accuracy", "precision_macro", "recall_macro", "f1_macro"}


def test_analyze_endpoint_returns_summary():
    client = TestClient(api.app)
    response = client.post(
        "/analyze",
        json={
            "texts": [
                "Python FastAPI NLP with 5 years experience",
                "React JavaScript TypeScript frontend engineer with 3 years experience",
            ]
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["sample_count"] == 2
    assert body["average_token_count"] > 0
    assert body["average_skill_count"] > 0
    assert "python" in body["top_skills"] or "react" in body["top_skills"]


def test_parse_endpoint_requires_auth_when_configured(monkeypatch):
    monkeypatch.setenv("API_BEARER_TOKEN", "secret-token")
    client = TestClient(api.app)

    unauthorized = client.post("/parse", json={"text": "Python FastAPI NLP"})
    assert unauthorized.status_code == 401
    assert "Unauthorized" in unauthorized.json()["detail"]

    monkeypatch.delenv("API_BEARER_TOKEN", raising=False)
