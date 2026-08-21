from __future__ import annotations

from collections import Counter

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.models.advanced_resume_classifier import AdvancedResumeClassifier, build_advanced_pipeline


def optimize_advanced_model(
    texts: list[str],
    labels: list[str],
    random_state: int = 42,
) -> dict:
    class_counts = Counter(labels)
    min_class_count = min(class_counts.values()) if class_counts else 0

    if min_class_count < 2:
        model = AdvancedResumeClassifier(random_state=random_state)
        model.fit(texts, labels)
        return {
            "model": model,
            "best_params": {},
            "best_score": None,
        }

    pipeline = build_advanced_pipeline(random_state=random_state)
    cv = StratifiedKFold(n_splits=min(2, min_class_count), shuffle=True, random_state=random_state)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid={
            "features__word__max_features": [1500, 3000],
            "features__char__max_features": [1000, 1500],
            "classifier__lr__C": [0.75, 1.25],
        },
        scoring="f1_macro",
        cv=cv,
        n_jobs=None,
    )
    search.fit(texts, labels)

    return {
        "model": AdvancedResumeClassifier(pipeline=search.best_estimator_),
        "best_params": search.best_params_,
        "best_score": float(search.best_score_),
    }
