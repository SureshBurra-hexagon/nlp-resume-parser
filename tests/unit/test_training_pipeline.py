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
    X_train_text, X_test_text = cleaned[:3], cleaned[3:]
    y_train, y_test = df["label"].tolist()[:3], df["label"].tolist()[3:]

    extractor = TfidfFeatureExtractor(max_features=100)
    X_train = extractor.fit_transform(X_train_text)
    X_test = extractor.transform(X_test_text)

    model = ResumeClassifier(max_iter=2000)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = classification_metrics(y_test, pred)
    assert set(metrics.keys()) == {"accuracy", "precision_macro", "recall_macro", "f1_macro"}
    assert len(pred) == len(y_test)
    for value in metrics.values():
        assert 0.0 <= value <= 1.0
