from __future__ import annotations

from collections import Counter, defaultdict, deque
from functools import lru_cache
from pathlib import Path
import logging
import os
import sys
import time

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import classification_metrics
from src.models.advanced_resume_classifier import AdvancedResumeClassifier
from src.feature_extraction.tfidf_features import TfidfFeatureExtractor
from src.models.resume_classifier import ResumeClassifier
from src.optimization.inference import ResumeBatchProcessor

BASELINE_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "baseline"
ADVANCED_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "advanced"
BATCH_PROCESSOR = ResumeBatchProcessor()
LOG = logging.getLogger("resume_api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="NLP Resume Parser API",
    description=(
        "Phase 4 FastAPI backend for parsing, analysis, baseline/advanced prediction, "
        "and evaluation with authentication, rate limiting, and structured error responses."
    ),
    version="0.4.0",
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


class ErrorResponse(BaseModel):
    error: str
    message: str
    detail: str | None = None


class EvaluationRequest(BaseModel):
    labels: list[str] = Field(..., min_length=1)
    predictions: list[str] = Field(..., min_length=1)


class EvaluationResponse(BaseModel):
    sample_count: int
    metrics: dict[str, float]


class AnalyzeResponse(BaseModel):
    sample_count: int
    average_token_count: float
    average_skill_count: float
    average_experience_years: float
    top_skills: list[str]


_AUTH_SCHEME = HTTPBearer(auto_error=False)
_RATE_LIMIT_BUCKETS: dict[str, deque[float]] = defaultdict(deque)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_csv(name: str, default: str) -> list[str]:
    return [item.strip() for item in os.getenv(name, default).split(",") if item.strip()]


def _configured_token() -> str:
    return os.getenv("API_BEARER_TOKEN", "").strip()


def _rate_limit_window_seconds() -> int:
    return _env_int("API_RATE_LIMIT_WINDOW_SECONDS", 60)


def _rate_limit_requests() -> int:
    return _env_int("API_RATE_LIMIT_REQUESTS", 120)


def _auth_key(credentials: HTTPAuthorizationCredentials | None, request: Request) -> str:
    if credentials is not None and credentials.credentials:
        return f"token:{credentials.credentials}"
    return f"ip:{request.client.host if request.client else 'unknown'}"


def clear_rate_limit_state() -> None:
    _RATE_LIMIT_BUCKETS.clear()


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


async def verify_access_token(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_AUTH_SCHEME),
) -> None:
    expected_token = _configured_token()
    if not expected_token:
        return
    provided_token = credentials.credentials.strip() if credentials else ""
    if provided_token != expected_token:
        raise HTTPException(status_code=401, detail="Unauthorized: invalid or missing bearer token.")


@app.middleware("http")
async def add_request_logging(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    LOG.info("%s %s status=%s duration_ms=%.2f", request.method, request.url.path, response.status_code, elapsed_ms)
    return response


@app.middleware("http")
async def add_rate_limiting(request: Request, call_next):
    credentials = await _AUTH_SCHEME(request)
    now = time.time()
    window = _rate_limit_window_seconds()
    limit = _rate_limit_requests()
    bucket = _RATE_LIMIT_BUCKETS[_auth_key(credentials, request)]

    while bucket and now - bucket[0] > window:
        bucket.popleft()

    if len(bucket) >= limit:
        return JSONResponse(
            status_code=429,
            content=ErrorResponse(
                error="rate_limit_exceeded",
                message="Rate limit exceeded.",
                detail=f"Allowed {limit} requests per {window} seconds.",
            ).model_dump(),
        )

    bucket.append(now)
    return await call_next(request)


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error="http_error", message="Request failed.", detail=str(exc.detail)).model_dump(),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(error="validation_error", message="Request validation failed.", detail=str(exc)).model_dump(),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception):
    LOG.exception("Unhandled exception", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(error="internal_server_error", message="Unexpected internal server error.").model_dump(),
    )


from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=_env_csv("API_CORS_ALLOW_ORIGINS", "*"),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.post(
    "/parse",
    response_model=ParsedResumeResponse,
    responses={401: {"model": ErrorResponse}, 429: {"model": ErrorResponse}},
    tags=["resume"],
)
def parse_resume_endpoint(payload: ResumeTextRequest, _: None = Depends(verify_access_token)) -> ParsedResumeResponse:
    return _parsed_resume_response(payload.text)


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={401: {"model": ErrorResponse}, 429: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    tags=["resume"],
)
def predict_resume_category(payload: ResumeTextRequest, _: None = Depends(verify_access_token)) -> PredictionResponse:
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


@app.post(
    "/predict/advanced",
    response_model=PredictionResponse,
    responses={401: {"model": ErrorResponse}, 429: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    tags=["resume"],
)
def predict_advanced_resume_category(payload: ResumeTextRequest, _: None = Depends(verify_access_token)) -> PredictionResponse:
    try:
        classifier = load_advanced_artifacts(str(ADVANCED_MODEL_DIR))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    parsed_resume = _parsed_resume_response(payload.text)
    probabilities = classifier.predict_proba([parsed_resume.normalized_text])[0]
    prediction = classifier.model.classes_[probabilities.argmax()]

    return PredictionResponse(
        parsed_resume=parsed_resume,
        predicted_profile_category=str(prediction),
        confidence=_prediction_confidence(probabilities),
    )


@app.post(
    "/predict/advanced/batch",
    response_model=BatchPredictionResponse,
    responses={401: {"model": ErrorResponse}, 429: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    tags=["resume"],
)
def predict_advanced_resume_batch(
    payload: BatchResumeTextRequest,
    _: None = Depends(verify_access_token),
) -> BatchPredictionResponse:
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


@app.post(
    "/evaluate",
    response_model=EvaluationResponse,
    responses={401: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 429: {"model": ErrorResponse}},
    tags=["analysis"],
)
def evaluate_predictions(payload: EvaluationRequest, _: None = Depends(verify_access_token)) -> EvaluationResponse:
    if len(payload.labels) != len(payload.predictions):
        raise HTTPException(status_code=422, detail="`labels` and `predictions` must be the same length.")

    metrics = classification_metrics(payload.labels, payload.predictions)
    return EvaluationResponse(sample_count=len(payload.labels), metrics=metrics)


@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    responses={401: {"model": ErrorResponse}, 429: {"model": ErrorResponse}},
    tags=["analysis"],
)
def analyze_resumes(payload: BatchResumeTextRequest, _: None = Depends(verify_access_token)) -> AnalyzeResponse:
    parsed_batch = BATCH_PROCESSOR.parse_batch(payload.texts)
    token_counts = [len(parsed["normalized_text"].split()) for parsed in parsed_batch]
    skill_counts = [len(parsed["skills"]) for parsed in parsed_batch]
    experience_values = [parsed["experience_years"] for parsed in parsed_batch if parsed["experience_years"] is not None]

    skill_counter = Counter(skill for parsed in parsed_batch for skill in parsed["skills"])

    sample_count = len(parsed_batch)
    return AnalyzeResponse(
        sample_count=sample_count,
        average_token_count=float(sum(token_counts) / sample_count),
        average_skill_count=float(sum(skill_counts) / sample_count),
        average_experience_years=float(sum(experience_values) / len(experience_values)) if experience_values else 0.0,
        top_skills=[skill for skill, _ in skill_counter.most_common(10)],
    )
