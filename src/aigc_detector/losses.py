from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .constants import Provenance
from .model import ProvenanceOutput


@dataclass(frozen=True)
class LossWeights:
    provenance_original: float = 1.0
    provenance_transformed: float = 1.0
    prediction_consistency: float = 0.5
    feature_consistency: float = 0.0
    mask_focal: float = 1.0
    mask_dice: float = 1.0
    ema_consistency: float = 0.0


def hierarchical_classification_loss(output: ProvenanceOutput, provenance: Tensor) -> Tensor:
    aigc_target = (provenance == int(Provenance.FULLY_AIGC)).float()
    aigc_loss = F.binary_cross_entropy_with_logits(output.aigc_logit, aigc_target)
    non_aigc = provenance != int(Provenance.FULLY_AIGC)
    if non_aigc.any():
        tamper_target = (provenance[non_aigc] == int(Provenance.TAMPERED)).float()
        tamper_loss = F.binary_cross_entropy_with_logits(
            output.tamper_logit[non_aigc], tamper_target
        )
    else:
        tamper_loss = output.tamper_logit.sum() * 0.0
    return aigc_loss + tamper_loss


def focal_bce_with_logits(logits: Tensor, target: Tensor, gamma: float = 2.0) -> Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    probability = logits.sigmoid()
    pt = torch.where(target >= 0.5, probability, 1.0 - probability)
    return ((1.0 - pt).pow(gamma) * bce).mean()


def soft_dice_loss(logits: Tensor, target: Tensor, epsilon: float = 1e-6) -> Tensor:
    probability = logits.sigmoid()
    numerator = 2.0 * (probability * target).sum() + epsilon
    denominator = probability.sum() + target.sum() + epsilon
    return 1.0 - numerator / denominator


def mask_supervision_loss(
    token_logits: list[Tensor], token_targets: list[Tensor | None]
) -> tuple[Tensor, Tensor]:
    focal: list[Tensor] = []
    dice: list[Tensor] = []
    for logits, target in zip(token_logits, token_targets, strict=True):
        if target is None:
            continue
        target = target.to(device=logits.device, dtype=logits.dtype)
        if target.shape != logits.shape:
            raise ValueError(f"Mask target {tuple(target.shape)} != logits {tuple(logits.shape)}")
        focal.append(focal_bce_with_logits(logits, target))
        dice.append(soft_dice_loss(logits, target))
    zero = token_logits[0].sum() * 0.0 if token_logits else torch.tensor(0.0)
    return (torch.stack(focal).mean(), torch.stack(dice).mean()) if focal else (zero, zero)


def confidence_gated_kl(
    student_probabilities: Tensor,
    teacher_probabilities: Tensor,
    threshold: float = 0.8,
) -> Tensor:
    confidence = teacher_probabilities.max(dim=-1).values
    eligible = confidence >= threshold
    if not eligible.any():
        return student_probabilities.sum() * 0.0
    return F.kl_div(
        student_probabilities[eligible].clamp_min(1e-7).log(),
        teacher_probabilities[eligible],
        reduction="batchmean",
    )


def provenance_robustness_loss(
    original: ProvenanceOutput,
    transformed: ProvenanceOutput,
    *,
    provenance: Tensor,
    weights: LossWeights | None = None,
    token_mask_targets: list[Tensor | None] | None = None,
    provenance_class_weights: Tensor | None = None,
    teacher_probabilities: Tensor | None = None,
    teacher_confidence_threshold: float = 0.8,
) -> tuple[Tensor, dict[str, Tensor]]:
    del provenance_class_weights
    weights = weights or LossWeights()
    provenance_original = hierarchical_classification_loss(original, provenance)
    provenance_transformed = hierarchical_classification_loss(transformed, provenance)
    prediction_consistency = F.mse_loss(original.probabilities, transformed.probabilities)
    feature_consistency = (
        1.0
        - F.cosine_similarity(original.provenance_features, transformed.provenance_features, dim=-1)
    ).mean()
    mask_focal, mask_dice = mask_supervision_loss(
        original.token_tamper_logits,
        token_mask_targets or [None] * len(original.token_tamper_logits),
    )
    ema_consistency = (
        confidence_gated_kl(
            transformed.probabilities,
            teacher_probabilities,
            teacher_confidence_threshold,
        )
        if teacher_probabilities is not None
        else transformed.probabilities.sum() * 0.0
    )
    components = {
        "provenance_original": provenance_original,
        "provenance_transformed": provenance_transformed,
        "prediction_consistency": prediction_consistency,
        "feature_consistency": feature_consistency,
        "mask_focal": mask_focal,
        "mask_dice": mask_dice,
        "ema_consistency": ema_consistency,
    }
    total = (
        weights.provenance_original * provenance_original
        + weights.provenance_transformed * provenance_transformed
        + weights.prediction_consistency * prediction_consistency
        + weights.feature_consistency * feature_consistency
        + weights.mask_focal * mask_focal
        + weights.mask_dice * mask_dice
        + weights.ema_consistency * ema_consistency
    )
    return total, components
