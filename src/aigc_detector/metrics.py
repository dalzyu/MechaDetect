from __future__ import annotations

import torch
from torch import Tensor


def confusion_matrix(target: Tensor, prediction: Tensor, classes: int = 3) -> Tensor:
    matrix = torch.zeros((classes, classes), dtype=torch.long)
    for actual, predicted in zip(
        target.reshape(-1).tolist(), prediction.reshape(-1).tolist(), strict=True
    ):
        matrix[int(actual), int(predicted)] += 1
    return matrix


def macro_f1(matrix: Tensor) -> float:
    scores = []
    for index in range(matrix.shape[0]):
        true_positive = matrix[index, index].float()
        false_positive = matrix[:, index].sum().float() - true_positive
        false_negative = matrix[index, :].sum().float() - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append((2 * true_positive / denominator).item() if denominator else 0.0)
    return sum(scores) / len(scores)


def balanced_accuracy(matrix: Tensor) -> float:
    recalls = []
    for index in range(matrix.shape[0]):
        total = matrix[index, :].sum().item()
        recalls.append(matrix[index, index].item() / total if total else 0.0)
    return sum(recalls) / len(recalls)


def binary_auroc(target: Tensor, score: Tensor) -> float:
    target = target.detach().float().reshape(-1)
    score = score.detach().float().reshape(-1)
    positives = target == 1
    negatives = target == 0
    positive_count = int(positives.sum())
    negative_count = int(negatives.sum())
    if not positive_count or not negative_count:
        return float("nan")
    sorted_score, order = torch.sort(score)
    sorted_target = target[order]
    ranks = torch.empty_like(sorted_score)
    _, counts = torch.unique_consecutive(sorted_score, return_counts=True)
    offset = 0
    for count_tensor in counts:
        count = int(count_tensor)
        mean_rank = (offset + 1 + offset + count) / 2.0
        ranks[offset : offset + count] = mean_rank
        offset += count
    positive_rank_sum = ranks[sorted_target == 1].sum()
    mann_whitney = positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    return (mann_whitney / (positive_count * negative_count)).item()


def binary_auprc(target: Tensor, score: Tensor) -> float:
    target = target.detach().float().reshape(-1)
    score = score.detach().float().reshape(-1)
    positive_count = int((target == 1).sum())
    if not positive_count:
        return float("nan")
    sorted_score, order = torch.sort(score, descending=True)
    sorted_target = target[order]
    true_positive = torch.cumsum(sorted_target, dim=0)
    ranks = torch.arange(1, len(sorted_target) + 1, dtype=torch.float32)
    precision = true_positive / ranks
    recall = true_positive / positive_count
    distinct_last = torch.ones_like(sorted_target, dtype=torch.bool)
    distinct_last[:-1] = sorted_score[:-1] != sorted_score[1:]
    grouped_precision = precision[distinct_last]
    grouped_recall = recall[distinct_last]
    previous_recall = torch.cat((torch.zeros(1), grouped_recall[:-1]))
    return ((grouped_recall - previous_recall) * grouped_precision).sum().item()


def multiclass_macro_auroc(target: Tensor, probabilities: Tensor, classes: int = 3) -> float:
    values = []
    for index in range(classes):
        one_vs_rest = (target == index).float()
        values.append(binary_auroc(one_vs_rest, probabilities[:, index]))
    finite = [value for value in values if value == value]
    return sum(finite) / len(finite) if finite else float("nan")
