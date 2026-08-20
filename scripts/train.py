from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.feature_extraction.tfidf_features import TfidfFeatureExtractor
from src.models.resume_classifier import ResumeClassifier
from src.preprocessing.text_cleaner import clean_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline resume classifier")
    parser.add_argument("--data", default="data/sample_resumes/train.csv")
    parser.add_argument("--model-dir", default="models/baseline")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df["cleaned_text"] = df["resume_text"].fillna("").map(clean_text)

    extractor = TfidfFeatureExtractor(max_features=2000)
    features = extractor.fit_transform(df["cleaned_text"].tolist())

    classifier = ResumeClassifier()
    classifier.fit(features, df["label"].tolist())

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    extractor.save(str(model_dir / "tfidf.joblib"))
    classifier.save(str(model_dir / "classifier.joblib"))

    print(f"Training complete. Artifacts saved to {model_dir}")


if __name__ == "__main__":
    main()
