from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.advanced_resume_classifier import AdvancedResumeClassifier
from src.feature_extraction.tfidf_features import TfidfFeatureExtractor
from src.models.resume_classifier import ResumeClassifier
from src.optimization.inference import ResumeBatchProcessor

BASELINE_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "baseline"
ADVANCED_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "advanced"
BATCH_PROCESSOR = ResumeBatchProcessor()

app = FastAPI(
    title="NLP Resume Parser API",
    description="FastAPI demo for parsing resumes and serving baseline profile predictions.",
    version="0.2.0",
)


class ResumeTextRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw resume text to parse or classify.")


class ParsedResumeResponse(BaseModel):
    normalized_text: str
    emails: list[str]
    phones: list[str]
    skills: list[str]
    sections: dict[str, str]
    education: list[str]
    certifications: list[str]
    experience_years: int | None


class PredictionResponse(BaseModel):
    parsed_resume: ParsedResumeResponse
    predicted_profile_category: str
    confidence: float | None = None


class BatchResumeTextRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="A batch of raw resume texts.")


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    baseline_artifacts_loaded: bool
    advanced_artifacts_loaded: bool


def baseline_artifacts_available(model_dir: Path | None = None) -> bool:
    artifact_dir = model_dir or BASELINE_MODEL_DIR
    return (artifact_dir / "tfidf.joblib").exists() and (artifact_dir / "classifier.joblib").exists()


def advanced_artifacts_available(model_dir: Path | None = None) -> bool:
    artifact_dir = model_dir or ADVANCED_MODEL_DIR
    return (artifact_dir / "advanced_pipeline.joblib").exists()


@lru_cache(maxsize=None)
def load_artifacts(model_dir: str) -> tuple[TfidfFeatureExtractor, ResumeClassifier]:
    artifact_dir = Path(model_dir)
    if not baseline_artifacts_available(artifact_dir):
        raise FileNotFoundError("Model artifacts are unavailable. Run `python scripts/train.py` first.")
    extractor = TfidfFeatureExtractor.load(str(artifact_dir / "tfidf.joblib"))
    classifier = ResumeClassifier.load(str(artifact_dir / "classifier.joblib"))
    return extractor, classifier


@lru_cache(maxsize=None)
def load_advanced_artifacts(model_dir: str) -> AdvancedResumeClassifier:
    artifact_dir = Path(model_dir)
    if not advanced_artifacts_available(artifact_dir):
        raise FileNotFoundError("Advanced model artifacts are unavailable. Run `python scripts/train_advanced.py` first.")
    return AdvancedResumeClassifier.load(str(artifact_dir / "advanced_pipeline.joblib"))


def _prediction_confidence(probabilities) -> float | None:
    if probabilities is None:
        return None
    return float(max(probabilities))


def _parsed_resume_response(text: str) -> ParsedResumeResponse:
    return ParsedResumeResponse(**BATCH_PROCESSOR.parse(text))


@app.get("/", tags=["meta"])
def read_root() -> dict[str, str]:
    return {"message": "NLP Resume Parser API", "docs_url": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        baseline_artifacts_loaded=baseline_artifacts_available(BASELINE_MODEL_DIR),
        advanced_artifacts_loaded=advanced_artifacts_available(ADVANCED_MODEL_DIR),
    )


@app.post("/parse", response_model=ParsedResumeResponse, tags=["resume"])
def parse_resume_endpoint(payload: ResumeTextRequest) -> ParsedResumeResponse:
    return _parsed_resume_response(payload.text)


@app.post("/predict", response_model=PredictionResponse, tags=["resume"])
def predict_resume_category(payload: ResumeTextRequest) -> PredictionResponse:
    try:
        extractor, classifier = load_artifacts(str(BASELINE_MODEL_DIR))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    parsed_resume = _parsed_resume_response(payload.text)
    features = extractor.transform([parsed_resume.normalized_text])
    probabilities = classifier.predict_proba(features)[0]
    prediction = classifier.predict(features)[0]

    return PredictionResponse(
        parsed_resume=parsed_resume,
        predicted_profile_category=str(prediction),
        confidence=_prediction_confidence(probabilities),
    )


@app.post("/predict/advanced", response_model=PredictionResponse, tags=["resume"])
def predict_advanced_resume_category(payload: ResumeTextRequest) -> PredictionResponse:
    try:
        classifier = load_advanced_artifacts(str(ADVANCED_MODEL_DIR))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    parsed_resume = _parsed_resume_response(payload.text)
    probabilities = classifier.predict_proba([parsed_resume.normalized_text])[0]
    prediction = classifier.predict([parsed_resume.normalized_text])[0]

    return PredictionResponse(
        parsed_resume=parsed_resume,
        predicted_profile_category=str(prediction),
        confidence=_prediction_confidence(probabilities),
    )


@app.post("/predict/advanced/batch", response_model=BatchPredictionResponse, tags=["resume"])
def predict_advanced_resume_batch(payload: BatchResumeTextRequest) -> BatchPredictionResponse:
    try:
        classifier = load_advanced_artifacts(str(ADVANCED_MODEL_DIR))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    parsed_resumes = [ParsedResumeResponse(**parsed) for parsed in BATCH_PROCESSOR.parse_batch(payload.texts)]
    normalized_texts = [parsed.normalized_text for parsed in parsed_resumes]
    predictions = classifier.predict(normalized_texts)
    probabilities = classifier.predict_proba(normalized_texts)

    return BatchPredictionResponse(
        predictions=[
            PredictionResponse(
                parsed_resume=parsed_resume,
                predicted_profile_category=str(prediction),
                confidence=_prediction_confidence(probability_row),
            )
            for parsed_resume, prediction, probability_row in zip(parsed_resumes, predictions, probabilities, strict=True)
        ]
    )
