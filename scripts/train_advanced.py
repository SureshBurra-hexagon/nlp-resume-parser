from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.augmentation.resume_augmenter import augment_training_data
from src.optimization.hyperparameter_search import optimize_advanced_model
from src.preprocessing.text_cleaner import clean_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the phase 3 advanced resume classifier")
    parser.add_argument("--data", default="data/sample_resumes/train.csv")
    parser.add_argument("--model-dir", default="models/advanced")
    parser.add_argument("--disable-augmentation", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    cleaned_texts = df["resume_text"].fillna("").map(clean_text).tolist()
    labels = df["label"].tolist()

    if args.disable_augmentation:
        training_texts, training_labels = cleaned_texts, labels
    else:
        training_texts, training_labels = augment_training_data(cleaned_texts, labels)

    optimization_result = optimize_advanced_model(training_texts, training_labels)
    classifier = optimization_result["model"]

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    classifier.save(str(model_dir / "advanced_pipeline.joblib"))

    metadata = {
        "training_examples": len(training_texts),
        "original_examples": len(cleaned_texts),
        "best_params": optimization_result["best_params"],
        "best_score": optimization_result["best_score"],
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Advanced training complete. Artifacts saved to {model_dir}")
    if optimization_result["best_score"] is not None:
        print(f"Best cross-validation f1_macro: {optimization_result['best_score']:.4f}")


if __name__ == "__main__":
    main()
