from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import classification_metrics
from src.models.transformer_resume_classifier import TransformerResumeClassifier
from src.preprocessing.text_cleaner import clean_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the phase 1 transformer resume classifier")
    parser.add_argument("--data", default="data/sample_resumes/test.csv")
    parser.add_argument("--model-dir", default="models/transformer")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    texts = df["resume_text"].fillna("").map(clean_text).tolist()
    labels = df["label"].tolist()

    model_dir = Path(args.model_dir)
    classifier = TransformerResumeClassifier.load(str(model_dir / "transformer_classifier.joblib"))

    start = time.perf_counter()
    predictions = classifier.predict(texts)
    elapsed = time.perf_counter() - start

    results = {
        "sample_count": len(labels),
        "model_name": classifier.model_name,
        "metrics": classification_metrics(labels, predictions),
        "latency_seconds": elapsed,
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
