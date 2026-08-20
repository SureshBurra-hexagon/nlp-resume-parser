from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.feature_extraction.tfidf_features import TfidfFeatureExtractor
from src.models.resume_classifier import ResumeClassifier
from src.utils.resume_parser import parse_resume

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "baseline"

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


class PredictionResponse(BaseModel):
    parsed_resume: ParsedResumeResponse
    predicted_profile_category: str


class HealthResponse(BaseModel):
    status: str
    artifacts_loaded: bool


def artifacts_available(model_dir: Path | None = None) -> bool:
    artifact_dir = model_dir or MODEL_DIR
    return (artifact_dir / "tfidf.joblib").exists() and (artifact_dir / "classifier.joblib").exists()


@lru_cache(maxsize=None)
def load_artifacts(model_dir: str) -> tuple[TfidfFeatureExtractor, ResumeClassifier]:
    artifact_dir = Path(model_dir)
    if not artifacts_available(artifact_dir):
        raise FileNotFoundError("Model artifacts are unavailable. Run `python scripts/train.py` first.")
    extractor = TfidfFeatureExtractor.load(str(artifact_dir / "tfidf.joblib"))
    classifier = ResumeClassifier.load(str(artifact_dir / "classifier.joblib"))
    return extractor, classifier


@app.get("/", tags=["meta"])
def read_root() -> dict[str, str]:
    return {"message": "NLP Resume Parser API", "docs_url": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health_check() -> HealthResponse:
    return HealthResponse(status="ok", artifacts_loaded=artifacts_available())


@app.post("/parse", response_model=ParsedResumeResponse, tags=["resume"])
def parse_resume_endpoint(payload: ResumeTextRequest) -> ParsedResumeResponse:
    return ParsedResumeResponse(**parse_resume(payload.text))


@app.post("/predict", response_model=PredictionResponse, tags=["resume"])
def predict_resume_category(payload: ResumeTextRequest) -> PredictionResponse:
    try:
        extractor, classifier = load_artifacts(str(MODEL_DIR))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    parsed_resume = ParsedResumeResponse(**parse_resume(payload.text))
    prediction = classifier.predict(extractor.transform([parsed_resume.normalized_text]))[0]

    return PredictionResponse(
        parsed_resume=parsed_resume,
        predicted_profile_category=str(prediction),
    )
