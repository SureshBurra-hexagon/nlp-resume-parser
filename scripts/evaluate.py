from __future__ import annotations

import argparse

import pandas as pd

from src.evaluation.metrics import classification_metrics
from src.feature_extraction.tfidf_features import TfidfFeatureExtractor
from src.models.resume_classifier import ResumeClassifier
from src.preprocessing.text_cleaner import clean_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline resume classifier")
    parser.add_argument("--data", default="data/sample_resumes/test.csv")
    parser.add_argument("--model-dir", default="models/baseline")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    cleaned = df["resume_text"].fillna("").map(clean_text).tolist()

    extractor = TfidfFeatureExtractor.load(f"{args.model_dir}/tfidf.joblib")
    classifier = ResumeClassifier.load(f"{args.model_dir}/classifier.joblib")

    predictions = classifier.predict(extractor.transform(cleaned))
    metrics = classification_metrics(df["label"].tolist(), predictions)

    print("Evaluation metrics:")
    for key, value in metrics.items():
        print(f"- {key}: {value:.4f}")


if __name__ == "__main__":
    main()
