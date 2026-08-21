from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import classification_metrics
from src.feature_extraction.tfidf_features import TfidfFeatureExtractor
from src.models.advanced_resume_classifier import AdvancedResumeClassifier
from src.models.resume_classifier import ResumeClassifier
from src.preprocessing.text_cleaner import clean_text


def benchmark_baseline(texts: list[str], labels: list[str], model_dir: Path) -> dict:
    extractor = TfidfFeatureExtractor.load(str(model_dir / "tfidf.joblib"))
    classifier = ResumeClassifier.load(str(model_dir / "classifier.joblib"))
    start = time.perf_counter()
    predictions = classifier.predict(extractor.transform(texts))
    elapsed = time.perf_counter() - start
    return {
        "metrics": classification_metrics(labels, predictions),
        "latency_seconds": elapsed,
    }


def benchmark_advanced(texts: list[str], labels: list[str], model_dir: Path) -> dict:
    classifier = AdvancedResumeClassifier.load(str(model_dir / "advanced_pipeline.joblib"))
    start = time.perf_counter()
    predictions = classifier.predict(texts)
    elapsed = time.perf_counter() - start
    return {
        "metrics": classification_metrics(labels, predictions),
        "latency_seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the phase 4 advanced resume classifier")
    parser.add_argument("--data", default="data/sample_resumes/test.csv")
    parser.add_argument("--baseline-model-dir", default="models/baseline")
    parser.add_argument("--advanced-model-dir", default="models/advanced")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    cleaned = df["resume_text"].fillna("").map(clean_text).tolist()
    labels = df["label"].tolist()

    baseline_results = benchmark_baseline(cleaned, labels, Path(args.baseline_model_dir))
    advanced_results = benchmark_advanced(cleaned, labels, Path(args.advanced_model_dir))

    print("Baseline results:")
    print(json.dumps(baseline_results, indent=2))
    print("Advanced results:")
    print(json.dumps(advanced_results, indent=2))


if __name__ == "__main__":
    main()
