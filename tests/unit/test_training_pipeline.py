import pandas as pd

from src.evaluation.metrics import classification_metrics
from src.feature_extraction.tfidf_features import TfidfFeatureExtractor
from src.models.resume_classifier import ResumeClassifier
from src.preprocessing.text_cleaner import clean_text


def test_training_pipeline_produces_predictions_and_metrics():
    df = pd.DataFrame(
        {
            "resume_text": [
                "python nlp fastapi machine learning",
                "react javascript frontend ui",
                "sql tableau business analytics",
                "aws docker kubernetes devops",
            ],
            "label": ["data", "frontend", "analytics", "devops"],
        }
    )

    cleaned = df["resume_text"].map(clean_text).tolist()
    y = df["label"].tolist()

    extractor = TfidfFeatureExtractor(max_features=100)
    X = extractor.fit_transform(cleaned)

    model = ResumeClassifier(max_iter=2000)
    model.fit(X, y)
    pred = model.predict(X)

    metrics = classification_metrics(y, pred)
    assert set(metrics.keys()) == {"accuracy", "precision_macro", "recall_macro", "f1_macro"}
    assert metrics["accuracy"] >= 0.5
