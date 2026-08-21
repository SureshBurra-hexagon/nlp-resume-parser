import pandas as pd

from src.models.advanced_resume_classifier import AdvancedResumeClassifier
from src.optimization.hyperparameter_search import optimize_advanced_model
from src.preprocessing.text_cleaner import clean_text


def test_advanced_classifier_predicts_on_cleaned_text():
    texts = [
        clean_text("python nlp fastapi machine learning"),
        clean_text("python data pipelines machine learning"),
        clean_text("react javascript frontend ui"),
        clean_text("react design systems frontend"),
        clean_text("sql tableau business analytics"),
        clean_text("sql reporting business insights"),
        clean_text("aws docker kubernetes devops"),
        clean_text("aws ci/cd infrastructure automation"),
    ]
    labels = [
        "data_science",
        "data_science",
        "frontend",
        "frontend",
        "analytics",
        "analytics",
        "devops",
        "devops",
    ]

    model = AdvancedResumeClassifier()
    model.fit(texts, labels)
    prediction = model.predict([clean_text("python fastapi nlp pipelines")])

    assert prediction[0] == "data_science"


def test_hyperparameter_search_returns_fitted_model():
    df = pd.DataFrame(
        {
            "resume_text": [
                "python nlp fastapi machine learning",
                "python data pipelines mlops",
                "react javascript frontend ui",
                "typescript component libraries frontend",
                "sql tableau business analytics",
                "sql reporting stakeholder dashboards",
                "aws docker kubernetes devops",
                "aws infrastructure automation ci/cd",
            ],
            "label": [
                "data_science",
                "data_science",
                "frontend",
                "frontend",
                "analytics",
                "analytics",
                "devops",
                "devops",
            ],
        }
    )
    cleaned = df["resume_text"].map(clean_text).tolist()

    result = optimize_advanced_model(cleaned, df["label"].tolist())
    prediction = result["model"].predict([clean_text("aws kubernetes platform automation")])

    assert result["best_params"]
    assert result["best_score"] is not None
    assert prediction[0] == "devops"
