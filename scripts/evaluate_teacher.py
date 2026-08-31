#!/usr/bin/env python3
"""External evaluation and promotion protocol for MechaDetect teacher models.

Evaluates teacher checkpoints outside DDP on frozen validation IDs across the
complete single-transform severity grid. Emits clean/mean/worst,
family/severity/generator/domain/worst-domain metrics, calibrates a global
validation threshold maximizing balanced accuracy subject to both recalls >= 0.82,
checks the teacher promotion gate, and emits the shared promotion report and
metadata sidecar.

Never evaluates or selects on test, test_unseen, or organizer demonstration data.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
if str(_PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from aigc_detector.config import load_config
from aigc_detector.constants import Provenance, Transformation
from aigc_detector.dataset import load_manifest_frame, parse_provenance
from aigc_detector.manifests import (
    assert_forbidden_demonstration_data_absent,
    file_sha256,
    manifest_digest,
)
from aigc_detector.metrics import balanced_accuracy, binary_auroc, confusion_matrix
from aigc_detector.model import ai_generated_probability, binary_probabilities
from aigc_detector.preprocessing import RenderPolicy, render_for_model
from aigc_detector.runtime import load_local_environment
from aigc_detector.transforms import TransformSpec, apply_transform

load_local_environment(_PROJECT_ROOT)

# Teacher architecture defaults
TEACHER_CANONICAL_FAMILY = "dinov3_vith16"
TEACHER_FULL_PARAMETER_COUNT = 872606207
TEACHER_INPUT_SIZE = [3, 224, 224]
TEACHER_PREPROCESSING_VERSION = "square_jpeg95"

# Gate thresholds
TEACHER_GATE_MIN_AUROC = 0.96
TEACHER_GATE_MIN_RECALL = 0.82


# =========================================================================== #
# Robustness Conditions Specification
# =========================================================================== #


@dataclass(frozen=True)
class RobustnessCondition:
    name: str
    family: str  # "clean", "jpeg", "blur", "resize", "noise", "color", "crop"
    severity: float | None
    spec: TransformSpec | None
    description: str


SINGLE_TRANSFORM_GRID: tuple[RobustnessCondition, ...] = (
    RobustnessCondition("clean", "clean", None, None, "Unaltered input"),
    # JPEG Compression
    RobustnessCondition(
        "jpeg90", "jpeg", 90.0, TransformSpec(Transformation.JPEG, 90.0), "JPEG quality 90"
    ),
    RobustnessCondition(
        "jpeg70", "jpeg", 70.0, TransformSpec(Transformation.JPEG, 70.0), "JPEG quality 70"
    ),
    RobustnessCondition(
        "jpeg50", "jpeg", 50.0, TransformSpec(Transformation.JPEG, 50.0), "JPEG quality 50"
    ),
    RobustnessCondition(
        "jpeg30", "jpeg", 30.0, TransformSpec(Transformation.JPEG, 30.0), "JPEG quality 30"
    ),
    # Gaussian Blur
    RobustnessCondition(
        "blur0.5", "blur", 0.5, TransformSpec(Transformation.BLUR, 0.5), "Gaussian blur radius 0.5"
    ),
    RobustnessCondition(
        "blur1.0", "blur", 1.0, TransformSpec(Transformation.BLUR, 1.0), "Gaussian blur radius 1.0"
    ),
    RobustnessCondition(
        "blur2.0", "blur", 2.0, TransformSpec(Transformation.BLUR, 2.0), "Gaussian blur radius 2.0"
    ),
    # Resize
    RobustnessCondition(
        "resize_half",
        "resize",
        0.50,
        TransformSpec(Transformation.RESIZE, 0.50),
        "Resize 0.50x + bicubic upscale",
    ),
    RobustnessCondition(
        "resize_quarter",
        "resize",
        0.25,
        TransformSpec(Transformation.RESIZE, 0.25),
        "Resize 0.25x + bicubic upscale",
    ),
    # Gaussian Noise
    RobustnessCondition(
        "noise0.02",
        "noise",
        0.02,
        TransformSpec(Transformation.NOISE, 0.02),
        "Gaussian noise std 0.02",
    ),
    RobustnessCondition(
        "noise0.05",
        "noise",
        0.05,
        TransformSpec(Transformation.NOISE, 0.05),
        "Gaussian noise std 0.05",
    ),
    RobustnessCondition(
        "noise0.10",
        "noise",
        0.10,
        TransformSpec(Transformation.NOISE, 0.10),
        "Gaussian noise std 0.10",
    ),
    # Color Jitter
    RobustnessCondition(
        "color_jitter20",
        "color",
        0.20,
        TransformSpec(Transformation.COLOR, 0.20),
        "Color jitter +/-20%",
    ),
    # Center Crop
    RobustnessCondition(
        "crop80",
        "crop",
        0.80,
        TransformSpec(Transformation.CROP, 0.80),
        "Center crop 80% + resize back",
    ),
)

CONDITIONS_BY_NAME: dict[str, RobustnessCondition] = {
    cond.name: cond for cond in SINGLE_TRANSFORM_GRID
}

ALLOWED_FAMILIES: tuple[str, ...] = ("jpeg", "blur", "resize", "noise", "color", "crop")


# =========================================================================== #
# Validation Manifest Safety & Isolation Guards
# =========================================================================== #


def validate_manifest_safety(frame: pd.DataFrame) -> pd.DataFrame:
    """Ensure manifest contains valid validation rows and strictly zero test/demo leakage.

    Raises ValueError if:
      - manifest contains only test, test_unseen, or train data.
      - manifest does not contain 'validation' split when split column is present.
      - manifest contains forbidden organizer demonstration data.
    """
    if "split" in frame.columns:
        splits = set(frame["split"].dropna().astype(str).str.lower().unique())
        test_splits = {"test", "test_unseen", "test_seen"}

        # Manifest is exclusively test data
        if splits and splits.issubset(test_splits):
            raise ValueError(
                f"Manifest split(s) {splits} are test/test_unseen. "
                "Teacher promotion must NEVER select, evaluate, or calibrate on test data!"
            )

        if splits and ("validation" not in splits):
            raise ValueError(
                f"Manifest split(s) {splits} do not contain 'validation'. "
                "Teacher promotion requires frozen validation IDs only."
            )

        # Filter strictly to validation
        val_frame = frame[frame["split"].astype(str).str.lower() == "validation"].copy()
    else:
        val_frame = frame.copy()

    if len(val_frame) == 0:
        raise ValueError("Validation manifest is empty; no validation records found.")

    # Guard against forbidden demonstration data (COCO val2017 / DALL-E Advanced)
    assert_forbidden_demonstration_data_absent(val_frame)

    for index, row in val_frame.iterrows():
        text = " ".join(
            str(row.get(column, ""))
            for column in ("dataset", "generator", "image_path", "original_path", "Category")
        ).lower()
        compact = re.sub(r"[^a-z0-9]+", "", text)
        if ("coco" in compact and "val2017" in compact) or (
            "dalle" in compact and "advanced" in compact
        ):
            raise ValueError(
                f"Forbidden organizer demonstration data detected in row {index}: {text[:100]}. "
                "Never evaluate or select promotion checkpoints on organizer demo data."
            )

    return val_frame


# =========================================================================== #
# Core Metric Calculations & Threshold Calibration
# =========================================================================== #


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def compute_binary_metrics_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Compute binary classification metrics at a given decision threshold."""
    y_true_t = torch.from_numpy(y_true).long()
    y_pred_t = (torch.from_numpy(y_score) >= threshold).long()

    matrix = confusion_matrix(y_true_t, y_pred_t, classes=2)
    tn = int(matrix[0, 0].item())
    fp = int(matrix[0, 1].item())
    fn = int(matrix[1, 0].item())
    tp = int(matrix[1, 1].item())

    n_pos = tp + fn
    n_neg = tn + fp
    total = len(y_true)

    tpr = tp / n_pos if n_pos > 0 else 0.0
    tnr = tn / n_neg if n_neg > 0 else 0.0
    bal_acc = balanced_accuracy(matrix)
    acc = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * tpr / (precision + tpr) if (precision + tpr) > 0 else 0.0

    return {
        "threshold": float(round(threshold, 6)),
        "accuracy": _finite(acc),
        "balanced_accuracy": _finite(bal_acc),
        "binary_recall": _finite(tpr),
        "ai_positive_recall": _finite(tpr),
        "authentic_recall": _finite(tnr),
        "binary_specificity": _finite(tnr),
        "binary_precision": _finite(precision),
        "binary_f1": _finite(f1),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n": total,
        "n_pos": n_pos,
        "n_neg": n_neg,
    }


def compute_condition_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute complete condition metrics including AUROC and threshold metrics."""
    metrics = compute_binary_metrics_at_threshold(y_true, y_score, threshold)

    target_t = torch.from_numpy(y_true).float()
    score_t = torch.from_numpy(y_score).float()
    auroc = binary_auroc(target_t, score_t)
    metrics["ai_positive_auroc"] = _finite(auroc)
    metrics["auroc"] = _finite(auroc)
    return metrics


def calibrate_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    min_recall: float = TEACHER_GATE_MIN_RECALL,
) -> tuple[float, dict[str, Any], bool]:
    """Calibrate one global validation threshold maximizing balanced accuracy.

    Subject to:
      AI-positive recall >= min_recall (0.82)
      Authentic recall >= min_recall (0.82)

    Tie-breaking policy (fully deterministic):
      1. Maximum balanced accuracy
      2. Minimum |TPR - TNR| (balanced performance across both classes)
      3. Minimum |threshold - 0.5| (closest to nominal decision boundary)
      4. Smallest threshold value

    Returns:
      (chosen_threshold, metrics_at_chosen_threshold, satisfies_constraint)
    """
    y_true_bool = y_true.astype(bool)
    n_pos = int(np.sum(y_true_bool))
    n_neg = int(np.sum(~y_true_bool))

    if n_pos == 0 or n_neg == 0:
        # Cannot calibrate with single-class data
        fallback_t = 0.5
        fallback_metrics = compute_binary_metrics_at_threshold(y_true, y_score, fallback_t)
        return fallback_t, fallback_metrics, False

    # Candidate thresholds: fine search grid plus unique observed scores
    fine_grid = np.linspace(0.01, 0.99, 981)
    unique_scores = np.unique(y_score)
    candidates = np.unique(np.concatenate([fine_grid, unique_scores]))
    candidates.sort()

    eligible_candidates: list[tuple[float, float, float, float]] = []
    all_candidates: list[tuple[float, float, float, float]] = []

    for t in candidates:
        pred = y_score >= t
        tp = int(np.sum(y_true_bool & pred))
        tn = int(np.sum(~y_true_bool & ~pred))
        tpr = tp / n_pos
        tnr = tn / n_neg
        bal_acc = (tpr + tnr) / 2.0
        diff = abs(tpr - tnr)
        dist_05 = abs(t - 0.5)

        all_candidates.append((bal_acc, -diff, -dist_05, t, tpr, tnr))
        if tpr >= min_recall and tnr >= min_recall:
            eligible_candidates.append((bal_acc, -diff, -dist_05, t, tpr, tnr))

    if eligible_candidates:
        # Sort descending by primary criteria
        eligible_candidates.sort(
            key=lambda item: (item[0], item[1], item[2], -item[3]), reverse=True
        )
        best = eligible_candidates[0]
        chosen_threshold = float(best[3])
        satisfies_constraint = True
    else:
        # Fallback: maximize balanced accuracy, then closest to achieving constraint
        all_candidates.sort(
            key=lambda item: (item[0], min(item[4], item[5]), item[1], item[2], -item[3]),
            reverse=True,
        )
        best = all_candidates[0]
        chosen_threshold = float(best[3])
        satisfies_constraint = False

    chosen_metrics = compute_binary_metrics_at_threshold(y_true, y_score, chosen_threshold)
    return chosen_threshold, chosen_metrics, satisfies_constraint


def check_teacher_gate(
    clean_auroc: float | None,
    calibrated_tpr: float | None,
    calibrated_tnr: float | None,
    satisfies_recall_constraint: bool,
    min_auroc: float = TEACHER_GATE_MIN_AUROC,
    min_recall: float = TEACHER_GATE_MIN_RECALL,
) -> tuple[bool, list[str]]:
    """Check teacher promotion gate criteria.

    Criteria:
      1. Clean AUROC > min_auroc (0.96)
      2. AI-positive recall at calibrated threshold >= min_recall (0.82)
      3. Authentic recall at calibrated threshold >= min_recall (0.82)
      4. Recalls constraint is satisfied simultaneously at the calibrated threshold

    Returns:
      (passed, failed_reasons)
    """
    failed_reasons: list[str] = []

    if clean_auroc is None:
        failed_reasons.append("Clean AUROC is missing or undefined")
    elif clean_auroc <= min_auroc:
        failed_reasons.append(
            f"Clean AUROC {clean_auroc:.4f} <= {min_auroc} (required > {min_auroc})"
        )

    if not satisfies_recall_constraint:
        best_tpr = f"{calibrated_tpr:.4f}" if calibrated_tpr is not None else "None"
        best_tnr = f"{calibrated_tnr:.4f}" if calibrated_tnr is not None else "None"
        failed_reasons.append(
            f"Recall constraint violation: no threshold achieves both AI-positive and authentic recall >= {min_recall} "
            f"(calibrated: TPR={best_tpr}, TNR={best_tnr})"
        )

    if calibrated_tpr is None:
        failed_reasons.append("AI-positive recall at calibrated threshold is missing")
    elif calibrated_tpr < min_recall:
        failed_reasons.append(
            f"AI-positive recall at calibrated threshold {calibrated_tpr:.4f} < {min_recall} (required >= {min_recall})"
        )

    if calibrated_tnr is None:
        failed_reasons.append("Authentic recall at calibrated threshold is missing")
    elif calibrated_tnr < min_recall:
        failed_reasons.append(
            f"Authentic recall at calibrated threshold {calibrated_tnr:.4f} < {min_recall} (required >= {min_recall})"
        )

    passed = len(failed_reasons) == 0
    return passed, failed_reasons


# =========================================================================== #
# Aggregated Evaluation Metrics Computation
# =========================================================================== #


def aggregate_evaluation_metrics(
    condition_scores: dict[str, np.ndarray],
    y_true: np.ndarray,
    metadata_rows: list[dict[str, Any]],
    calibrated_threshold: float,
    satisfies_recall_constraint: bool,
) -> dict[str, Any]:
    """Compute the full hierarchy of evaluation metrics across conditions, families,

    generators, domains, and the calibrated threshold.
    """
    clean_scores = condition_scores["clean"]
    clean_metrics = compute_condition_metrics(y_true, clean_scores, threshold=0.5)
    clean_auroc = clean_metrics["ai_positive_auroc"]

    # Severity level metrics (each individual condition)
    severity_metrics: dict[str, Any] = {}
    transformed_aurocs: list[float] = []
    transformed_bal_accs: list[float] = []
    transformed_tprs: list[float] = []
    transformed_tnrs: list[float] = []
    transformed_drops: list[float] = []

    # Calibrated metrics per condition
    calibrated_clean = compute_binary_metrics_at_threshold(
        y_true, clean_scores, calibrated_threshold
    )
    calibrated_condition_metrics: dict[str, Any] = {"clean": calibrated_clean}
    calibrated_transformed_tprs: list[float] = []
    calibrated_transformed_tnrs: list[float] = []
    calibrated_transformed_bal_accs: list[float] = []

    # Per-family collectors
    family_aurocs: dict[str, list[float]] = defaultdict(list)
    family_conditions: dict[str, dict[str, Any]] = defaultdict(dict)

    for cond in SINGLE_TRANSFORM_GRID:
        scores = condition_scores[cond.name]
        cond_m = compute_condition_metrics(y_true, scores, threshold=0.5)
        cond_auroc = cond_m["ai_positive_auroc"]

        drop = (
            clean_auroc - cond_auroc if clean_auroc is not None and cond_auroc is not None else None
        )
        cond_m["auroc_drop"] = _finite(drop) if drop is not None else None
        cond_m["family"] = cond.family
        cond_m["severity"] = cond.severity
        severity_metrics[cond.name] = cond_m

        # Calibrated metrics for this condition
        calibrated_m = compute_binary_metrics_at_threshold(y_true, scores, calibrated_threshold)
        calibrated_condition_metrics[cond.name] = calibrated_m

        if cond.name != "clean":
            if cond_auroc is not None:
                transformed_aurocs.append(cond_auroc)
            if drop is not None:
                transformed_drops.append(drop)
            if cond_m["balanced_accuracy"] is not None:
                transformed_bal_accs.append(cond_m["balanced_accuracy"])
            if cond_m["ai_positive_recall"] is not None:
                transformed_tprs.append(cond_m["ai_positive_recall"])
            if cond_m["authentic_recall"] is not None:
                transformed_tnrs.append(cond_m["authentic_recall"])

            family_aurocs[cond.family].append(cond_auroc)
            family_conditions[cond.family][cond.name] = cond_m

            calibrated_transformed_tprs.append(calibrated_m["ai_positive_recall"])
            calibrated_transformed_tnrs.append(calibrated_m["authentic_recall"])
            calibrated_transformed_bal_accs.append(calibrated_m["balanced_accuracy"])

    # Mean and worst metrics across transformations
    mean_metrics = {
        "mean_auroc": _finite(sum(transformed_aurocs) / len(transformed_aurocs))
        if transformed_aurocs
        else None,
        "mean_balanced_accuracy": _finite(sum(transformed_bal_accs) / len(transformed_bal_accs))
        if transformed_bal_accs
        else None,
        "mean_ai_positive_recall": _finite(sum(transformed_tprs) / len(transformed_tprs))
        if transformed_tprs
        else None,
        "mean_authentic_recall": _finite(sum(transformed_tnrs) / len(transformed_tnrs))
        if transformed_tnrs
        else None,
        "mean_auroc_drop": _finite(sum(transformed_drops) / len(transformed_drops))
        if transformed_drops
        else None,
    }

    worst_cond_name = None
    worst_cond_auroc = None
    if transformed_aurocs:
        worst_cond_auroc = min(transformed_aurocs)
        for name, m in severity_metrics.items():
            if name != "clean" and m["ai_positive_auroc"] == worst_cond_auroc:
                worst_cond_name = name
                break

    worst_metrics = {
        "worst_auroc": _finite(worst_cond_auroc) if worst_cond_auroc is not None else None,
        "worst_condition": worst_cond_name,
        "worst_auroc_drop": _finite(max(transformed_drops)) if transformed_drops else None,
        "worst_balanced_accuracy": _finite(min(transformed_bal_accs))
        if transformed_bal_accs
        else None,
        "worst_ai_positive_recall": _finite(min(transformed_tprs)) if transformed_tprs else None,
        "worst_authentic_recall": _finite(min(transformed_tnrs)) if transformed_tnrs else None,
    }

    # Family aggregations
    family_metrics: dict[str, Any] = {}
    for fam in ALLOWED_FAMILIES:
        aurocs = [a for a in family_aurocs.get(fam, []) if a is not None]
        if aurocs:
            min_a = min(aurocs)
            w_name = None
            for c_name, c_m in family_conditions[fam].items():
                if c_m["ai_positive_auroc"] == min_a:
                    w_name = c_name
                    break
            family_metrics[fam] = {
                "mean_auroc": _finite(sum(aurocs) / len(aurocs)),
                "worst_auroc": _finite(min_a),
                "worst_condition": w_name,
                "conditions": family_conditions[fam],
            }
        else:
            family_metrics[fam] = {
                "mean_auroc": None,
                "worst_auroc": None,
                "worst_condition": None,
                "conditions": {},
            }

    # Generator breakdowns (clean images, AI generator vs Authentic)
    generator_aurocs: dict[str, float] = {}
    generator_counts: dict[str, int] = defaultdict(int)
    y_true_bool = y_true.astype(bool)
    authentic_mask = ~y_true_bool

    # Collect generator per sample
    generator_labels = [
        str(row.get("generator_family") or row.get("generator") or "unknown")
        for row in metadata_rows
    ]

    for gen in sorted(set(generator_labels)):
        gen_mask = np.array([g == gen for g in generator_labels])
        pos_mask = gen_mask & y_true_bool
        generator_counts[gen] = int(np.sum(pos_mask))
        comparison_mask = pos_mask | authentic_mask

        if np.sum(pos_mask) >= 2 and np.sum(authentic_mask) >= 2:
            sub_y = y_true[comparison_mask]
            sub_score = clean_scores[comparison_mask]
            g_auroc = binary_auroc(
                torch.from_numpy(sub_y).float(), torch.from_numpy(sub_score).float()
            )
            if math.isfinite(g_auroc):
                generator_aurocs[gen] = float(round(g_auroc, 4))

    gen_auroc_vals = list(generator_aurocs.values())
    generator_metrics: dict[str, Any] = {
        "per_generator": {
            gen: {"auroc": generator_aurocs.get(gen), "n_positives": generator_counts[gen]}
            for gen in sorted(set(generator_labels))
            if generator_counts[gen] > 0
        },
        "mean_generator_auroc": _finite(sum(gen_auroc_vals) / len(gen_auroc_vals))
        if gen_auroc_vals
        else None,
        "worst_generator_auroc": _finite(min(gen_auroc_vals)) if gen_auroc_vals else None,
        "worst_generator": min(generator_aurocs, key=generator_aurocs.get)
        if generator_aurocs
        else None,
    }

    # Domain breakdowns (clean images, per dataset)
    domain_labels = [str(row.get("dataset") or "unknown") for row in metadata_rows]
    domain_metrics: dict[str, Any] = {}
    valid_domain_aurocs: dict[str, float] = {}

    for dom in sorted(set(domain_labels)):
        dom_mask = np.array([d == dom for d in domain_labels])
        dom_y = y_true[dom_mask]
        dom_scores = clean_scores[dom_mask]
        n_total = len(dom_y)
        n_pos = int(np.sum(dom_y == 1))
        n_neg = int(np.sum(dom_y == 0))

        dom_info: dict[str, Any] = {
            "n": n_total,
            "n_positive": n_pos,
            "n_authentic": n_neg,
        }

        if n_pos >= 2 and n_neg >= 2:
            dom_auroc = binary_auroc(
                torch.from_numpy(dom_y).float(), torch.from_numpy(dom_scores).float()
            )
            if math.isfinite(dom_auroc):
                dom_info["auroc"] = float(round(dom_auroc, 4))
                valid_domain_aurocs[dom] = dom_info["auroc"]
            else:
                dom_info["auroc"] = None
            dom_bm = compute_binary_metrics_at_threshold(dom_y, dom_scores, calibrated_threshold)
            dom_info["balanced_accuracy"] = dom_bm["balanced_accuracy"]
            dom_info["ai_positive_recall"] = dom_bm["ai_positive_recall"]
            dom_info["authentic_recall"] = dom_bm["authentic_recall"]
        else:
            dom_info["auroc"] = None
            dom_bm = compute_binary_metrics_at_threshold(dom_y, dom_scores, calibrated_threshold)
            dom_info["accuracy"] = dom_bm["accuracy"]

        domain_metrics[dom] = dom_info

    worst_domain_name = None
    worst_domain_auroc = None
    if valid_domain_aurocs:
        worst_domain_name = min(valid_domain_aurocs, key=valid_domain_aurocs.get)
        worst_domain_auroc = valid_domain_aurocs[worst_domain_name]

    worst_domain = {
        "name": worst_domain_name,
        "auroc": _finite(worst_domain_auroc) if worst_domain_auroc is not None else None,
        "sample_count": domain_metrics[worst_domain_name]["n"] if worst_domain_name else None,
    }

    # Summary of metrics at calibrated threshold
    calibrated_summary = {
        "threshold": float(round(calibrated_threshold, 6)),
        "satisfies_recall_constraint": satisfies_recall_constraint,
        "clean": calibrated_clean,
        "mean_transformed": {
            "mean_ai_positive_recall": _finite(
                sum(calibrated_transformed_tprs) / len(calibrated_transformed_tprs)
            )
            if calibrated_transformed_tprs
            else None,
            "mean_authentic_recall": _finite(
                sum(calibrated_transformed_tnrs) / len(calibrated_transformed_tnrs)
            )
            if calibrated_transformed_tnrs
            else None,
            "mean_balanced_accuracy": _finite(
                sum(calibrated_transformed_bal_accs) / len(calibrated_transformed_bal_accs)
            )
            if calibrated_transformed_bal_accs
            else None,
        },
        "worst_transformed": {
            "worst_ai_positive_recall": _finite(min(calibrated_transformed_tprs))
            if calibrated_transformed_tprs
            else None,
            "worst_authentic_recall": _finite(min(calibrated_transformed_tnrs))
            if calibrated_transformed_tnrs
            else None,
            "worst_balanced_accuracy": _finite(min(calibrated_transformed_bal_accs))
            if calibrated_transformed_bal_accs
            else None,
        },
        "per_condition": calibrated_condition_metrics,
    }

    return {
        "clean": clean_metrics,
        "mean": mean_metrics,
        "worst": worst_metrics,
        "family": family_metrics,
        "severity": severity_metrics,
        "generator": generator_metrics,
        "domain": domain_metrics,
        "worst_domain": worst_domain,
        "calibrated": calibrated_summary,
    }


# =========================================================================== #
# Promotion Report & Metadata Sidecar Contract
# =========================================================================== #


def create_promotion_report(
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    manifest_digest_val: str,
    calibrated_threshold: float,
    metrics: dict[str, Any],
    passed: bool,
    failed_reasons: list[str],
) -> dict[str, Any]:
    """Emit promotion report matching the exact production pipeline contract.

    Contract:
      checkpoint_path
      checkpoint_sha256
      manifest_digest
      calibrated_threshold
      metrics
      passed
      failed_reasons
    """
    return {
        "checkpoint_path": str(checkpoint_path).replace("\\", "/"),
        "checkpoint_sha256": str(checkpoint_sha256),
        "manifest_digest": str(manifest_digest_val),
        "calibrated_threshold": float(round(calibrated_threshold, 6)),
        "metrics": metrics,
        "passed": bool(passed),
        "failed_reasons": list(failed_reasons),
    }


def create_metadata_sidecar(
    manifest_digest_val: str,
    calibrated_threshold: float,
    passed: bool,
    model: torch.nn.Module | None = None,
    checkpoint_payload: dict[str, Any] | None = None,
    model_family: str = TEACHER_CANONICAL_FAMILY,
    parameter_count: int | None = None,
    input_size: list[int] | None = None,
    preprocessing_version: str = TEACHER_PREPROCESSING_VERSION,
    quantization: str = "none",
) -> dict[str, Any]:
    """Emit model artifact metadata sidecar matching the shared pipeline specification.

    Contract:
      model_family
      parameter_count
      quantization
      calibrated_threshold
      input_size
    """
    if parameter_count is None:
        if model is not None:
            parameter_count = int(sum(p.numel() for p in model.parameters()))
        elif checkpoint_payload is not None and "parameter_count" in checkpoint_payload:
            parameter_count = int(checkpoint_payload["parameter_count"])
        else:
            parameter_count = None

    if input_size is None:
        input_size = list(TEACHER_INPUT_SIZE)

    evaluation_status = "promoted" if passed else "rejected"

    return {
        "model_family": str(model_family),
        "parameter_count": int(parameter_count) if parameter_count is not None else None,
        "quantization": str(quantization),
        "calibrated_threshold": float(round(calibrated_threshold, 6)),
        "input_size": input_size,
        "preprocessing_version": str(preprocessing_version),
        "manifest_digest": str(manifest_digest_val),
        "evaluation_status": evaluation_status,
    }


# =========================================================================== #
# Inference & Evaluator Engine
# =========================================================================== #


@torch.inference_mode()
def run_condition_inference(
    model: torch.nn.Module,
    images: list[Image.Image],
    condition: RobustnessCondition,
    batch_size: int = 4,
    device: str = "cuda",
    seed_offset: int = 42,
) -> np.ndarray:
    """Run model inference for a batch of images under one robustness condition."""
    scores: list[float] = []
    autocast_device = "cuda" if "cuda" in device else "cpu"
    autocast_dtype = torch.bfloat16 if "cuda" in device else torch.float32

    for start in range(0, len(images), batch_size):
        batch_images = images[start : start + batch_size]
        transformed_batch: list[Image.Image] = []

        for idx, img in enumerate(batch_images):
            row_idx = start + idx
            if condition.spec is None:
                transformed = img
            else:
                rng = Random(seed_offset + row_idx * 37)
                transformed = apply_transform(img, condition.spec, rng)
            transformed_batch.append(transformed)

        with torch.autocast(device_type=autocast_device, dtype=autocast_dtype):
            output = model(transformed_batch)
            if hasattr(output, "ai_positive_logit"):
                probs = binary_probabilities(output.ai_positive_logit.float())
            elif hasattr(output, "probabilities"):
                probs = output.probabilities.float()
            else:
                raise RuntimeError(f"Unexpected model output type: {type(output)}")

            ai_probs = ai_generated_probability(probs)
            scores.extend(ai_probs.cpu().tolist())

    return np.array(scores, dtype=np.float32)


def evaluate_teacher_checkpoint(
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    config_path: str | Path = Path("configs/teacher_dinov3_stage2_paired_unfrozen.yaml"),
    data_root: str | Path | None = None,
    batch_size: int = 4,
    device: str | None = None,
    limit: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate one teacher checkpoint outside DDP on frozen validation IDs.

    Returns the complete evaluation dict including promotion report and metadata sidecar.
    """
    checkpoint_path = Path(checkpoint_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    config_path = Path(config_path).resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Teacher checkpoint not found at: {checkpoint_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Validation manifest not found at: {manifest_path}")

    # Compute checkpoint SHA-256
    ckpt_hash = file_sha256(checkpoint_path)

    # Load and validate manifest safety (strictly validation split, no demo data)
    raw_frame = load_manifest_frame(manifest_path)
    val_frame = validate_manifest_safety(raw_frame)

    if limit is not None and limit > 0:
        val_frame = val_frame.iloc[:limit].copy()

    val_digest = manifest_digest(val_frame)

    # Load config and runtime environment
    config = load_config(config_path)
    if data_root is None:
        data_root = Path(
            config.get("paths", {}).get("data_root") or os.environ.get("TECHJAM_DATA_ROOT", "data")
        )
    else:
        data_root = Path(data_root)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build model and load checkpoint weights
    from aigc_detector.predict import _load_checkpoint
    from aigc_detector.train import build_model

    model = build_model(config)
    _load_checkpoint(model, checkpoint_path)
    model = model.to(device).eval()

    policy = RenderPolicy(
        config.get("preprocessing", {}).get("policy", TEACHER_PREPROCESSING_VERSION)
    )

    # Stream rendered validation batches instead of retaining every PIL image.
    # The validation set is evaluated under 15 conditions; retaining all rendered
    # images simultaneously exhausts host RAM and starves the CUDA process.
    y_true_list: list[int] = []
    metadata_rows: list[dict[str, Any]] = []
    records = val_frame.to_dict(orient="records")
    for idx, row in enumerate(records):
        raw_ai_pos = row.get("ai_positive")
        if raw_ai_pos is not None and str(raw_ai_pos).strip() != "":
            ai_pos = int(raw_ai_pos)
        else:
            target = int(
                parse_provenance(row.get("provenance") or row.get("label"), str(row.get("dataset", "")))
            )
            ai_pos = int(target != int(Provenance.AUTHENTIC))
        y_true_list.append(ai_pos)
        metadata_rows.append(
            {
                "row_id": str(row.get("row_id", idx)),
                "image_path": str(row.get("image_path")),
                "dataset": str(row.get("dataset", "")),
                "generator": str(row.get("generator", "")),
                "generator_family": str(row.get("generator_family", row.get("generator", ""))),
                "duplicate_group": str(row.get("duplicate_group", "")),
            }
        )

    y_true = np.array(y_true_list, dtype=np.int64)

    def load_rendered_chunk(start: int, end: int) -> list[Image.Image]:
        chunk: list[Image.Image] = []
        for idx in range(start, end):
            row = records[idx]
            img_rel = str(row["image_path"]).replace("\\", "/")
            img_path = data_root / img_rel if not Path(img_rel).is_absolute() else Path(img_rel)
            if not img_path.is_file():
                raise FileNotFoundError(
                    f"Missing validation image at {img_path}. "
                    "Prefetch must ensure all validation images exist prior to evaluation."
                )
            with Image.open(img_path) as source:
                downloaded = source.convert("RGB").copy()
            chunk.append(render_for_model(downloaded, policy, rng=Random(seed + idx)))
        return chunk

    # Run each condition in bounded chunks to keep host memory stable.
    condition_scores: dict[str, np.ndarray] = {}
    chunk_size = 256
    for cond in SINGLE_TRANSFORM_GRID:
        score_parts: list[np.ndarray] = []
        for start in range(0, len(records), chunk_size):
            end = min(start + chunk_size, len(records))
            score_parts.append(
                run_condition_inference(
                    model=model,
                    images=load_rendered_chunk(start, end),
                    condition=cond,
                    batch_size=batch_size,
                    device=device,
                    seed_offset=seed,
                )
            )
        condition_scores[cond.name] = np.concatenate(score_parts) if score_parts else np.array([])

    # Calibrate global validation threshold on clean condition
    clean_scores = condition_scores["clean"]
    calibrated_threshold, chosen_metrics, satisfies_constraint = calibrate_threshold(
        y_true=y_true,
        y_score=clean_scores,
        min_recall=TEACHER_GATE_MIN_RECALL,
    )

    # Aggregate full metrics
    metrics = aggregate_evaluation_metrics(
        condition_scores=condition_scores,
        y_true=y_true,
        metadata_rows=metadata_rows,
        calibrated_threshold=calibrated_threshold,
        satisfies_recall_constraint=satisfies_constraint,
    )

    # Teacher promotion gate check
    clean_auroc = metrics["clean"]["ai_positive_auroc"]
    calibrated_tpr = metrics["calibrated"]["clean"]["ai_positive_recall"]
    calibrated_tnr = metrics["calibrated"]["clean"]["authentic_recall"]

    passed, failed_reasons = check_teacher_gate(
        clean_auroc=clean_auroc,
        calibrated_tpr=calibrated_tpr,
        calibrated_tnr=calibrated_tnr,
        satisfies_recall_constraint=satisfies_constraint,
    )

    # Create promotion report and metadata sidecar
    report = create_promotion_report(
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=ckpt_hash,
        manifest_digest_val=val_digest,
        calibrated_threshold=calibrated_threshold,
        metrics=metrics,
        passed=passed,
        failed_reasons=failed_reasons,
    )

    sidecar = create_metadata_sidecar(
        manifest_digest_val=val_digest,
        calibrated_threshold=calibrated_threshold,
        passed=passed,
        model=model,
    )

    return {
        "report": report,
        "metadata": sidecar,
        "condition_scores": {k: v.tolist() for k, v in condition_scores.items()},
        "y_true": y_true.tolist(),
    }


# =========================================================================== #
# CLI Entry Point
# =========================================================================== #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MechaDetect External Teacher Evaluation & Promotion Protocol"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to teacher model checkpoint (.pt or .safetensors)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("splits/production_eligible/validation.parquet"),
        help="Path to frozen validation manifest (Parquet, JSONL, or CSV)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/teacher_dinov3_stage2_paired_unfrozen.yaml"),
        help="Path to teacher training/model config YAML",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Data root directory containing validation images",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save evaluation report JSON (default outputs/eval_<ckpt_stem>.json)",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=None,
        help="Path to save promotion_report.json (if promotion requested)",
    )
    parser.add_argument(
        "--output-metadata",
        type=Path,
        default=None,
        help="Path to save metadata.json sidecar (if promotion requested)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Inference batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Computation device ('cuda' or 'cpu')",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on validation rows for quick testing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"=== Evaluating Teacher Checkpoint: {args.checkpoint} ===")
    print(f"Manifest: {args.manifest}")
    print(f"Config:   {args.config}")

    eval_result = evaluate_teacher_checkpoint(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        config_path=args.config,
        data_root=args.data_root,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
    )

    report = eval_result["report"]
    sidecar = eval_result["metadata"]

    output_path = args.output
    if output_path is None:
        ckpt_stem = args.checkpoint.stem
        output_path = Path(f"outputs/eval_{ckpt_stem}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(eval_result, indent=2), encoding="utf-8")
    print(f"Full evaluation written to: {output_path}")

    if args.output_report is not None:
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Promotion report written to: {args.output_report}")

    if args.output_metadata is not None:
        args.output_metadata.parent.mkdir(parents=True, exist_ok=True)
        args.output_metadata.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
        print(f"Metadata sidecar written to: {args.output_metadata}")

    # Summary display
    print("\n--- Evaluation Summary ---")
    clean_auroc = report["metrics"]["clean"]["ai_positive_auroc"]
    worst_auroc = report["metrics"]["worst"]["worst_auroc"]
    mean_auroc = report["metrics"]["mean"]["mean_auroc"]
    calibrated_t = report["calibrated_threshold"]
    calibrated_tpr = report["metrics"]["calibrated"]["clean"]["ai_positive_recall"]
    calibrated_tnr = report["metrics"]["calibrated"]["clean"]["authentic_recall"]
    passed = report["passed"]

    print(f"Clean AUROC:             {clean_auroc:.4f}" if clean_auroc else "Clean AUROC: N/A")
    print(
        f"Worst Transform AUROC:   {worst_auroc:.4f}"
        if worst_auroc
        else "Worst Transform AUROC: N/A"
    )
    print(
        f"Mean Transform AUROC:    {mean_auroc:.4f}" if mean_auroc else "Mean Transform AUROC: N/A"
    )
    print(f"Calibrated Threshold:    {calibrated_t:.4f}")
    print(
        f"Calibrated TPR (AI):     {calibrated_tpr:.4f}"
        if calibrated_tpr
        else "Calibrated TPR: N/A"
    )
    print(
        f"Calibrated TNR (Auth):   {calibrated_tnr:.4f}"
        if calibrated_tnr
        else "Calibrated TNR: N/A"
    )
    print(f"Promotion Gate Status:   {'PASSED' if passed else 'FAILED'}")

    if not passed:
        print("Failure Reasons:")
        for r in report["failed_reasons"]:
            print(f"  - {r}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
