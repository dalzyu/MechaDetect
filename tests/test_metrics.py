import pytest
import torch

from aigc_detector.metrics import binary_auprc, binary_auroc, confusion_matrix, macro_f1
from scripts.evaluate_performance import summarize


def test_binary_ranking_metrics_are_perfect_for_perfect_ordering() -> None:
    target = torch.tensor([0, 0, 1, 1])
    score = torch.tensor([0.1, 0.2, 0.8, 0.9])
    assert binary_auroc(target, score) == pytest.approx(1.0)
    assert binary_auprc(target, score) == pytest.approx(1.0)


def test_binary_ranking_metrics_handle_ties() -> None:
    target = torch.tensor([0, 0, 1, 1])
    score = torch.ones(4)
    assert binary_auroc(target, score) == pytest.approx(0.5)
    assert binary_auprc(target, score) == pytest.approx(0.5)


def test_confusion_matrix_and_macro_f1() -> None:
    target = torch.tensor([0, 1, 2, 2])
    prediction = torch.tensor([0, 1, 1, 2])
    matrix = confusion_matrix(target, prediction)
    assert matrix.tolist() == [[1, 0, 0], [0, 1, 0], [0, 1, 1]]
    assert macro_f1(matrix) == pytest.approx((1.0 + 2 / 3 + 2 / 3) / 3)


def test_track5_summary_counts_tampered_and_fully_generated_as_positive() -> None:
    target = torch.tensor([0, 1, 2])
    probabilities = torch.tensor(
        [
            [0.90, 0.10],
            [0.10, 0.90],
            [0.05, 0.95],
        ]
    )
    result = summarize(target, probabilities)
    assert result["binary_accuracy"] == 1.0
    assert result["binary_confusion_matrix"] == [[1, 0], [0, 2]]
    assert result["ai_positive_auroc"] == pytest.approx(1.0)
