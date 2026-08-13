"""
Evaluation metrics for resume parsing models.

Implements:
- Accuracy, Precision, Recall, F1-Score (micro / macro / weighted)
- Confusion matrix
- ROC-AUC (binary or one-vs-rest)
- Pretty-print evaluation report
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np

try:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.preprocessing import label_binarize
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False


def compute_metrics(
    y_true: List[Union[int, str]],
    y_pred: List[Union[int, str]],
    average: str = "weighted",
) -> Dict[str, float]:
    """Compute standard classification metrics.

    Args:
        y_true: Ground-truth labels.
        y_pred: Model predictions.
        average: Averaging strategy for multi-class metrics
            (``"micro"``, ``"macro"``, or ``"weighted"``).

    Returns:
        Dictionary with keys ``accuracy``, ``precision``, ``recall``,
        ``f1``.
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for evaluation metrics.")

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, average=average, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, average=average, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, y_pred, average=average, zero_division=0)
        ),
    }


def classification_report_str(
    y_true: List[Union[int, str]],
    y_pred: List[Union[int, str]],
    labels: Optional[List[str]] = None,
) -> str:
    """Return a formatted classification report string.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        labels: Optional list of label names for the report.

    Returns:
        Multi-line string report.
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required.")
    return classification_report(y_true, y_pred, target_names=labels, zero_division=0)


def compute_confusion_matrix(
    y_true: List[Union[int, str]],
    y_pred: List[Union[int, str]],
) -> np.ndarray:
    """Compute the confusion matrix.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.

    Returns:
        2-D numpy array of shape ``(n_classes, n_classes)``.
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required.")
    return confusion_matrix(y_true, y_pred)


def plot_confusion_matrix(
    y_true: List[Union[int, str]],
    y_pred: List[Union[int, str]],
    labels: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
) -> None:
    """Plot a heatmap of the confusion matrix.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        labels: Class names for axis tick labels.
        title: Plot title.
        save_path: Optional file path to save the figure.
    """
    if not PLOT_AVAILABLE:
        raise ImportError("matplotlib and seaborn are required for plotting.")

    cm = compute_confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels or "auto",
        yticklabels=labels or "auto",
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
    plt.close(fig)


def compute_roc_auc(
    y_true: List[Union[int, str]],
    y_score: np.ndarray,
    multi_class: str = "ovr",
    labels: Optional[List[str]] = None,
) -> float:
    """Compute ROC-AUC score.

    Args:
        y_true: Ground-truth labels.
        y_score: Predicted probabilities or decision scores.
            Shape ``(n_samples,)`` for binary or ``(n_samples, n_classes)``
            for multi-class.
        multi_class: ``"ovr"`` (one-vs-rest) or ``"ovo"`` (one-vs-one).
            Only used when ``y_score`` is 2-D.
        labels: Class labels for binarization (required for multi-class).

    Returns:
        ROC-AUC score as a float.
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required.")
    if y_score.ndim == 1:
        return float(roc_auc_score(y_true, y_score))
    return float(
        roc_auc_score(y_true, y_score, multi_class=multi_class, average="weighted")
    )


def print_evaluation_summary(
    y_true: List[Union[int, str]],
    y_pred: List[Union[int, str]],
    labels: Optional[List[str]] = None,
    average: str = "weighted",
) -> None:
    """Print a full evaluation summary to stdout.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        labels: Optional class label names.
        average: Averaging strategy.
    """
    metrics = compute_metrics(y_true, y_pred, average=average)
    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1-Score  : {metrics['f1']:.4f}")
    print("-" * 60)
    print("CLASSIFICATION REPORT")
    print("-" * 60)
    print(classification_report_str(y_true, y_pred, labels=labels))
    print("=" * 60)
