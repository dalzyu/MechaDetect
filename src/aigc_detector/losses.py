from __future__ import annotations

"""Loss functions and multi-objective optimization for robust provenance detection.

Mathematical formulations:
1. Binary AI-positive Classification Loss:
   - Authentic images are negative.
   - Fully generated and AI-edited images are positive.
   - `L_cls = BCE(ai_positive_logit, provenance != AUTHENTIC)`.

2. Localized Edit Mask Supervision:
   - For images with ground-truth binary edit masks, masks are converted to
     fractional patch occupancy and supervised using focal BCE and soft Dice.

3. Transformation Consistency:
   - Prediction consistency uses the two-class [authentic, AI-positive]
     probabilities.
   - Feature consistency uses cosine distance between normalized embeddings.

4. Teacher Distillation:
   - Confidence-gated KL divergence matches the EMA teacher's binary output.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .constants import Provenance
from .model import ProvenanceOutput


@dataclass(frozen=True)
class LossWeights:
    """Relative hyperparameter weights balancing the multi-objective loss components."""

    provenance_original: float = 1.0       # Binary AI-positive BCE on clean images
    provenance_transformed: float = 1.0    # Binary AI-positive BCE on transformed images
    prediction_consistency: float = 0.5    # MSE between clean and transformed probabilities
    feature_consistency: float = 0.0       # Cosine distance between clean and transformed embeddings
    mask_focal: float = 1.0                # Focal BCE loss on localized edit masks
    mask_dice: float = 1.0                 # Soft Dice loss on localized edit masks
    ema_consistency: float = 0.0           # Confidence-gated KL divergence from EMA teacher


def ai_classification_loss(output: ProvenanceOutput, target: Tensor) -> Tensor:
    """Classify authentic/human-edited images as negative (0) and AI as positive (1)."""
    if target.dtype.is_floating_point:
        ai_target = target
    else:
        ai_target = (target != int(Provenance.AUTHENTIC)).float()
    return F.binary_cross_entropy_with_logits(output.ai_positive_logit, ai_target)

def focal_bce_with_logits(logits: Tensor, target: Tensor, gamma: float = 2.0) -> Tensor:
    """Focal Binary Cross-Entropy with logits for handling extreme class imbalance in patch masks.

    Down-weights easy well-classified background patches and focuses gradients on ambiguous boundaries.
    """
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    probability = logits.sigmoid()
    pt = torch.where(target >= 0.5, probability, 1.0 - probability)
    return ((1.0 - pt).pow(gamma) * bce).mean()


def soft_dice_loss(logits: Tensor, target: Tensor, epsilon: float = 1e-6) -> Tensor:
    """Differentiable soft Dice loss for localized patch mask overlap optimization."""
    probability = logits.sigmoid()
    numerator = 2.0 * (probability * target).sum() + epsilon
    denominator = probability.sum() + target.sum() + epsilon
    return 1.0 - numerator / denominator


def mask_supervision_loss(
    token_logits: list[Tensor], token_targets: list[Tensor | None]
) -> tuple[Tensor, Tensor]:
    """Compute combined Focal BCE and soft Dice losses across localized token tamper masks.

    Args:
        token_logits: List of length B containing raw patch tamper logits [N_tokens].
        token_targets: List of length B containing target fractional occupancy [N_tokens],
                       or None if the image has no ground-truth tamper mask.

    Returns:
        Tuple of (mean_focal_loss, mean_dice_loss). Returns zeros if no masks are present.
    """
    focal_losses: list[Tensor] = []
    dice_losses: list[Tensor] = []

    for logits, target in zip(token_logits, token_targets, strict=True):
        if target is None:
            continue
        target = target.to(device=logits.device, dtype=logits.dtype)
        if target.shape != logits.shape:
            raise ValueError(f"Mask target shape {tuple(target.shape)} != logits {tuple(logits.shape)}")
        focal_losses.append(focal_bce_with_logits(logits, target))
        dice_losses.append(soft_dice_loss(logits, target))

    zero = token_logits[0].sum() * 0.0 if token_logits else torch.tensor(0.0)
    return (
        (torch.stack(focal_losses).mean(), torch.stack(dice_losses).mean())
        if focal_losses
        else (zero, zero)
    )


def confidence_gated_kl(
    student_probabilities: Tensor,
    teacher_probabilities: Tensor,
    threshold: float = 0.8,
    student_logit: Tensor | None = None,
) -> Tensor:
    """Compute KL divergence from EMA teacher to student, gated on high teacher confidence."""
    confidence = teacher_probabilities.max(dim=-1).values
    eligible = confidence >= threshold
    if not eligible.any():
        return student_probabilities.sum() * 0.0

    if student_logit is not None:
        eligible_logits = student_logit[eligible]
        student_log_probs = torch.stack(
            (F.logsigmoid(-eligible_logits), F.logsigmoid(eligible_logits)), dim=-1
        )
    else:
        student_log_probs = student_probabilities[eligible].clamp_min(1e-7).log()

    return F.kl_div(
        student_log_probs,
        teacher_probabilities[eligible],
        reduction="batchmean",
    )

def provenance_robustness_loss(
    original: ProvenanceOutput,
    transformed: ProvenanceOutput,
    *,
    provenance: Tensor | None = None,
    ai_target: Tensor | None = None,
    weights: LossWeights | None = None,
    token_mask_targets: list[Tensor | None] | None = None,
    teacher_probabilities: Tensor | None = None,
    teacher_confidence_threshold: float = 0.8,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute composite multi-objective robustness loss across clean and transformed views.

    Args:
        original: Predictions on clean / original images.
        transformed: Predictions on transformed / perturbed images.
        provenance: Integer ground-truth class labels [B].
        weights: Hyperparameter weights for loss terms (default: standard balanced weights).
        token_mask_targets: Optional patch-level tamper occupancy targets.
        teacher_probabilities: Optional soft probability targets from EMA teacher model.
        teacher_confidence_threshold: Minimum teacher confidence required for KL distillation.

    Returns:
        Tuple of (scalar_total_loss, dict_of_unweighted_individual_loss_components).
    """
    weights = weights or LossWeights()

    # 1. Binary AI-positive classification on clean and transformed views.
    target = ai_target if ai_target is not None else provenance
    if target is None:
        raise ValueError("Either ai_target or provenance must be provided")
    prov_orig = ai_classification_loss(original, target)
    prov_trans = ai_classification_loss(transformed, target)
    # 2. Representation & prediction consistency under transformations
    pred_consistency = F.mse_loss(original.probabilities, transformed.probabilities)
    feat_consistency = (
        1.0
        - F.cosine_similarity(original.provenance_features, transformed.provenance_features, dim=-1)
    ).mean()

    # 3. Patch-level localized tampering supervision
    mask_focal, mask_dice = mask_supervision_loss(
        original.token_tamper_logits,
        token_mask_targets or [None] * len(original.token_tamper_logits),
    )

    # 4. Confidence-gated EMA teacher distillation
    ema_loss = (
        confidence_gated_kl(
            transformed.probabilities,
            teacher_probabilities,
            teacher_confidence_threshold,
            student_logit=transformed.ai_positive_logit,
        )
        if teacher_probabilities is not None
        else transformed.probabilities.sum() * 0.0
    )

    components = {
        "provenance_original": prov_orig,
        "provenance_transformed": prov_trans,
        "prediction_consistency": pred_consistency,
        "feature_consistency": feat_consistency,
        "mask_focal": mask_focal,
        "mask_dice": mask_dice,
        "ema_consistency": ema_loss,
    }

    total = (
        weights.provenance_original * prov_orig
        + weights.provenance_transformed * prov_trans
        + weights.prediction_consistency * pred_consistency
        + weights.feature_consistency * feat_consistency
        + weights.mask_focal * mask_focal
        + weights.mask_dice * mask_dice
        + weights.ema_consistency * ema_loss
    )

    return total, components
