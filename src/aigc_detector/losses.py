from __future__ import annotations

"""Loss functions and multi-objective optimization for robust provenance detection.

Mathematical formulations:
1. Hierarchical Classification Loss:
   - Global AIGC Loss: Binary cross-entropy on whether the image is fully synthesized:
         L_aigc = BCE(aigc_logit, y_aigc)
   - Conditional Tamper Loss: Computed ONLY over authentic/tampered images:
         L_tamper = BCE(tamper_logit[non_aigc], y_tamper[non_aigc])
   - Total Classification: L_cls = L_aigc + L_tamper
   - Why exclude fully-AIGC images from tamper loss?
     Generative models frequently produce local inconsistencies or artifacts (e.g. garbled text).
     Forcing the tamper branch to score these as "untampered" creates gradient conflict with the
     global AIGC head. Decoupling them allows each head to specialize cleanly.

2. Localized Tamper Mask Supervision:
   - For images with ground-truth binary tampering masks (e.g. SID-Set spliced regions),
     masks are converted to fractional patch occupancy and supervised using:
     - Focal BCE Loss: Focuses gradient updates on hard-to-classify boundary patches (gamma = 2.0).
     - Soft Dice Loss: Maximizes area overlap between predicted suspicious patches and ground truth.
         L_mask = L_focal + L_dice

3. Representation & Prediction Consistency:
   - When an image undergoes content-preserving transformations T(x) (JPEG, blur, noise, crop):
     - Prediction Consistency: MSE between probabilities P(x) and P(T(x)).
     - Feature Consistency: Cosine distance between normalized feature vectors.
   - Enforces that compression or resizing cannot easily flip provenance classifications.

4. Teacher Distillation (EMA Consistency):
   - Confidence-Gated KL Divergence between student probabilities on transformed images
     and an exponential moving average (EMA) teacher model on clean images.
   - Distillation only applies when the teacher model is highly confident (confidence >= threshold, default 0.80).
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

    provenance_original: float = 1.0       # Hierarchical cross-entropy on clean images
    provenance_transformed: float = 1.0    # Hierarchical cross-entropy on transformed images
    prediction_consistency: float = 0.5    # MSE between clean and transformed output probabilities
    feature_consistency: float = 0.0       # Cosine distance between clean and transformed embeddings
    mask_focal: float = 1.0                # Focal BCE loss on localized tamper patch masks
    mask_dice: float = 1.0                 # Soft Dice loss on localized tamper patch masks
    ema_consistency: float = 0.0           # Confidence-gated KL divergence from EMA teacher


def hierarchical_classification_loss(output: ProvenanceOutput, provenance: Tensor) -> Tensor:
    """Compute hierarchical classification loss over a batch of images.

    Decouples global generative detection from localized manipulation detection.
    Fully-AIGC samples are cleanly excluded from the tamper classification loss.

    Args:
        output: Model output container containing `aigc_logit` and `tamper_logit`.
        provenance: Ground-truth integer labels [B] where values match `Provenance` enum:
                    0 = AUTHENTIC, 1 = TAMPERED, 2 = FULLY_AIGC.

    Returns:
        Scalar loss tensor combining global AIGC loss and conditional tamper loss.
    """
    # 1. Global AI generation decision: Is the image fully synthetic?
    aigc_target = (provenance == int(Provenance.FULLY_AIGC)).float()
    aigc_loss = F.binary_cross_entropy_with_logits(output.aigc_logit, aigc_target)

    # 2. Conditional tamper decision: Conditioned on NOT being fully synthetic, was it manipulated?
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
) -> Tensor:
    """Compute KL divergence from EMA teacher to student, gated on high teacher confidence.

    Only transfers supervision when the teacher's maximum class probability exceeds `threshold`.
    Prevents noisy teacher pseudo-labels from corrupting student representation learning.
    """
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

    # 1. Classification losses on clean and transformed views
    prov_orig = hierarchical_classification_loss(original, provenance)
    prov_trans = hierarchical_classification_loss(transformed, provenance)

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
