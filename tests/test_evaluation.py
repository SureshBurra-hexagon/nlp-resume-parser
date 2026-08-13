"""
Tests for the evaluation module.
"""

import pytest
import numpy as np

from src.evaluation import (
    compute_metrics,
    compute_confusion_matrix,
    classification_report_str,
    print_evaluation_summary,
)


Y_TRUE = ["education", "experience", "skills", "education", "experience", "other"]
Y_PRED = ["education", "experience", "skills", "experience", "experience", "other"]


class TestComputeMetrics:
    def test_perfect_prediction(self):
        metrics = compute_metrics(Y_TRUE, Y_TRUE)
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["f1"] == pytest.approx(1.0)

    def test_imperfect_prediction(self):
        metrics = compute_metrics(Y_TRUE, Y_PRED)
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0
        assert 0.0 <= metrics["f1"] <= 1.0

    def test_returns_all_keys(self):
        metrics = compute_metrics(Y_TRUE, Y_PRED)
        for key in ("accuracy", "precision", "recall", "f1"):
            assert key in metrics

    def test_binary(self):
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1]
        metrics = compute_metrics(y_true, y_pred, average="binary")
        assert 0.0 <= metrics["f1"] <= 1.0


class TestComputeConfusionMatrix:
    def test_shape(self):
        cm = compute_confusion_matrix(Y_TRUE, Y_PRED)
        n_classes = len(set(Y_TRUE + Y_PRED))
        assert cm.shape == (n_classes, n_classes)

    def test_perfect_diagonal(self):
        cm = compute_confusion_matrix(Y_TRUE, Y_TRUE)
        assert (cm == np.diag(np.diag(cm))).all()


class TestClassificationReportStr:
    def test_returns_string(self):
        report = classification_report_str(Y_TRUE, Y_PRED)
        assert isinstance(report, str)

    def test_contains_precision(self):
        report = classification_report_str(Y_TRUE, Y_PRED)
        assert "precision" in report.lower()


class TestPrintEvaluationSummary:
    def test_runs_without_error(self, capsys):
        print_evaluation_summary(Y_TRUE, Y_PRED)
        captured = capsys.readouterr()
        assert "EVALUATION SUMMARY" in captured.out
