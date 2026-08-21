from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.transformer_resume_classifier import TransformerResumeClassifier
from src.preprocessing.text_cleaner import clean_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the phase 1 transformer resume classifier")
    parser.add_argument("--data", default="data/sample_resumes/train.csv")
    parser.add_argument("--model-dir", default="models/transformer")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    texts = df["resume_text"].fillna("").map(clean_text).tolist()
    labels = df["label"].tolist()

    classifier = TransformerResumeClassifier(model_name=args.model_name)
    classifier.fit(texts, labels)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    classifier.save(str(model_dir / "transformer_classifier.joblib"))

    metadata = {
        "training_examples": len(texts),
        "model_name": args.model_name,
        "labels": sorted(set(labels)),
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Transformer training complete. Artifacts saved to {model_dir}")


if __name__ == "__main__":
    main()
