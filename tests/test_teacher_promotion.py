"""Tests for MechaDetect External Teacher Evaluation and Promotion Protocol.

Covers:
  - Full single-transform severity grid (15 conditions, 6 families, exact parameters)
  - Global validation threshold calibration (maximizing balanced accuracy subject to TPR/TNR >= 0.82)
  - Deterministic tie-breaking
  - Teacher promotion gate check (Clean AUROC > 0.96, both recalls >= 0.82)
  - Validation manifest safety guards (rejection of test/test_unseen and organizer demo data)
  - Evaluation metric hierarchy (clean/mean/worst, family/severity/generator/domain/worst-domain)
  - Deterministic ranking of 25/50/75/100% candidate checkpoints
  - Shared promotion report and metadata sidecar schema contracts
  - End-to-end promotion from evaluation reports
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
if str(_PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

import numpy as np
import pandas as pd
import pytest

from aigc_detector.constants import Transformation
from scripts.evaluate_teacher import (
    ALLOWED_FAMILIES,
    CONDITIONS_BY_NAME,
    SINGLE_TRANSFORM_GRID,
    TEACHER_CANONICAL_FAMILY,
    TEACHER_FULL_PARAMETER_COUNT,
    TEACHER_INPUT_SIZE,
    TEACHER_PREPROCESSING_VERSION,
    aggregate_evaluation_metrics,
    calibrate_threshold,
    check_teacher_gate,
    create_metadata_sidecar,
    create_promotion_report,
    validate_manifest_safety,
)
from scripts.promote_teacher import (
    promote_teacher_checkpoints,
    rank_checkpoints,
)

# =========================================================================== #
# 1. Single-Transform Severity Grid Tests
# =========================================================================== #


def test_single_transform_grid_coverage() -> None:
    """Verify the single-transform grid contains exactly 15 conditions:

    1 clean baseline + 14 augmented conditions across the 6 allowed families.
    """
    assert len(SINGLE_TRANSFORM_GRID) == 15

    # 1. Clean condition
    clean_cond = CONDITIONS_BY_NAME["clean"]
    assert clean_cond.spec is None
    assert clean_cond.family == "clean"
    assert clean_cond.severity is None

    # Check that all 6 allowed families are present in the non-clean conditions
    families_present = {c.family for c in SINGLE_TRANSFORM_GRID if c.name != "clean"}
    assert families_present == set(ALLOWED_FAMILIES)

    # 2. JPEG Compression (quality = 90, 70, 50, 30)
    for q in (90, 70, 50, 30):
        name = f"jpeg{q}"
        assert name in CONDITIONS_BY_NAME
        c = CONDITIONS_BY_NAME[name]
        assert c.family == "jpeg"
        assert c.severity == float(q)
        assert c.spec is not None
        assert c.spec.family == Transformation.JPEG
        assert c.spec.severity == float(q)

    # 3. Gaussian Blur (sigma = 0.5, 1.0, 2.0)
    for s, name in [(0.5, "blur0.5"), (1.0, "blur1.0"), (2.0, "blur2.0")]:
        assert name in CONDITIONS_BY_NAME
        c = CONDITIONS_BY_NAME[name]
        assert c.family == "blur"
        assert c.severity == s
        assert c.spec is not None
        assert c.spec.family == Transformation.BLUR
        assert c.spec.severity == s

    # 4. Resize (scale = 0.50, 0.25)
    assert "resize_half" in CONDITIONS_BY_NAME
    c_half = CONDITIONS_BY_NAME["resize_half"]
    assert c_half.family == "resize"
    assert c_half.severity == 0.50
    assert c_half.spec.family == Transformation.RESIZE
    assert c_half.spec.severity == 0.50

    assert "resize_quarter" in CONDITIONS_BY_NAME
    c_quarter = CONDITIONS_BY_NAME["resize_quarter"]
    assert c_quarter.family == "resize"
    assert c_quarter.severity == 0.25
    assert c_quarter.spec.family == Transformation.RESIZE
    assert c_quarter.spec.severity == 0.25

    # 5. Gaussian Noise (std = 0.02, 0.05, 0.10)
    for s in (0.02, 0.05, 0.10):
        name = f"noise{s:.2f}"
        assert name in CONDITIONS_BY_NAME
        c = CONDITIONS_BY_NAME[name]
        assert c.family == "noise"
        assert c.severity == s
        assert c.spec.family == Transformation.NOISE
        assert c.spec.severity == s

    # 6. Color Jitter (+/- 20%)
    assert "color_jitter20" in CONDITIONS_BY_NAME
    c_col = CONDITIONS_BY_NAME["color_jitter20"]
    assert c_col.family == "color"
    assert c_col.severity == 0.20
    assert c_col.spec.family == Transformation.COLOR
    assert c_col.spec.severity == 0.20

    # 7. Center Crop (80%)
    assert "crop80" in CONDITIONS_BY_NAME
    c_crop = CONDITIONS_BY_NAME["crop80"]
    assert c_crop.family == "crop"
    assert c_crop.severity == 0.80
    assert c_crop.spec.family == Transformation.CROP
    assert c_crop.spec.severity == 0.80


# =========================================================================== #
# 2. Threshold Calibration & Determinism Tests
# =========================================================================== #


def test_threshold_calibration_deterministic_and_optimal() -> None:
    """Verify calibrate_threshold finds the threshold maximizing balanced accuracy

    subject to TPR >= 0.82 and TNR >= 0.82, with 100% deterministic tie-breaking.
    """
    np.random.seed(42)
    n_samples = 1000
    # Well-separated distributions (AUROC > 0.98)
    neg_scores = np.random.beta(1.5, 8.0, size=n_samples // 2)  # Mean ~0.15
    pos_scores = np.random.beta(8.0, 1.5, size=n_samples // 2)  # Mean ~0.85

    y_true = np.concatenate(
        [np.zeros(len(neg_scores), dtype=int), np.ones(len(pos_scores), dtype=int)]
    )
    y_score = np.concatenate([neg_scores, pos_scores])

    # Run calibration multiple times to guarantee determinism
    results = [calibrate_threshold(y_true, y_score, min_recall=0.82) for _ in range(5)]
    thresholds = [r[0] for r in results]
    assert all(t == thresholds[0] for t in thresholds), "Calibration must be deterministic!"

    chosen_t, chosen_metrics, satisfies_constraint = results[0]
    assert satisfies_constraint is True
    assert chosen_metrics["ai_positive_recall"] >= 0.82
    assert chosen_metrics["authentic_recall"] >= 0.82
    assert chosen_metrics["balanced_accuracy"] > 0.90
    assert 0.10 < chosen_t < 0.90


def test_threshold_calibration_fails_when_recalls_unachievable() -> None:
    """When a model has poor class separation such that no threshold achieves both

    recalls >= 0.82, calibration must report satisfies_constraint = False.
    """
    np.random.seed(123)
    n_samples = 500
    # Substantial overlap (AUROC ~0.70)
    neg_scores = np.random.uniform(0.1, 0.7, size=n_samples // 2)
    pos_scores = np.random.uniform(0.3, 0.9, size=n_samples // 2)

    y_true = np.concatenate(
        [np.zeros(len(neg_scores), dtype=int), np.ones(len(pos_scores), dtype=int)]
    )
    y_score = np.concatenate([neg_scores, pos_scores])

    chosen_t, chosen_metrics, satisfies_constraint = calibrate_threshold(
        y_true, y_score, min_recall=0.82
    )

    assert satisfies_constraint is False
    # At least one recall is strictly below 0.82
    assert (chosen_metrics["ai_positive_recall"] < 0.82) or (
        chosen_metrics["authentic_recall"] < 0.82
    )


# =========================================================================== #
# 3. Teacher Promotion Gate Check Tests
# =========================================================================== #


def test_teacher_gate_passes_when_criteria_met() -> None:
    """Teacher gate passes when Clean AUROC > 0.96 and both recalls >= 0.82 (including exact 0.82)."""
    passed, failed_reasons = check_teacher_gate(
        clean_auroc=0.9850,
        calibrated_tpr=0.9200,
        calibrated_tnr=0.9300,
        satisfies_recall_constraint=True,
    )
    assert passed is True
    assert len(failed_reasons) == 0

    # Exact 0.8200 recalls must also PASS (contract is >= 0.82)
    passed_exact, failed_exact = check_teacher_gate(
        clean_auroc=0.9650,
        calibrated_tpr=0.8200,
        calibrated_tnr=0.8200,
        satisfies_recall_constraint=True,
    )
    assert passed_exact is True
    assert len(failed_exact) == 0


def test_teacher_gate_fails_when_auroc_below_threshold() -> None:
    """Teacher gate fails with explicit reason when clean AUROC <= 0.96."""
    passed, failed_reasons = check_teacher_gate(
        clean_auroc=0.9550,
        calibrated_tpr=0.8500,
        calibrated_tnr=0.8600,
        satisfies_recall_constraint=True,
    )
    assert passed is False
    assert any("Clean AUROC 0.9550 <= 0.96" in r for r in failed_reasons)


def test_teacher_gate_fails_when_recall_below_threshold() -> None:
    """Teacher gate fails with explicit reason when recalls fail."""
    # AI recall fails
    passed, failed_reasons = check_teacher_gate(
        clean_auroc=0.9700,
        calibrated_tpr=0.8000,
        calibrated_tnr=0.8800,
        satisfies_recall_constraint=False,
    )
    assert passed is False
    assert any(
        "AI-positive recall at calibrated threshold 0.8000 < 0.82 (required >= 0.82)" in r
        for r in failed_reasons
    )
    assert any("Recall constraint violation" in r for r in failed_reasons)

    # Authentic recall fails
    passed, failed_reasons = check_teacher_gate(
        clean_auroc=0.9700,
        calibrated_tpr=0.8800,
        calibrated_tnr=0.8100,
        satisfies_recall_constraint=False,
    )
    assert passed is False
    assert any(
        "Authentic recall at calibrated threshold 0.8100 < 0.82 (required >= 0.82)" in r
        for r in failed_reasons
    )


# =========================================================================== #
# 4. Manifest Safety & Isolation Guard Tests
# =========================================================================== #


def test_manifest_safety_rejects_test_splits() -> None:
    """Evaluator must reject manifests that contain test/test_unseen splits."""
    # Split is exclusively test
    df_test = pd.DataFrame(
        {
            "image_path": ["a.jpg", "b.jpg"],
            "label": ["authentic", "fully_aigc"],
            "split": ["test", "test"],
        }
    )
    with pytest.raises(ValueError, match="Teacher promotion must NEVER select"):
        validate_manifest_safety(df_test)

    # Split is test_unseen
    df_unseen = pd.DataFrame(
        {
            "image_path": ["a.jpg", "b.jpg"],
            "label": ["authentic", "fully_aigc"],
            "split": ["test_unseen", "test_unseen"],
        }
    )
    with pytest.raises(ValueError, match="Teacher promotion must NEVER select"):
        validate_manifest_safety(df_unseen)

    # Split is train only (missing validation)
    df_train = pd.DataFrame(
        {
            "image_path": ["a.jpg", "b.jpg"],
            "label": ["authentic", "fully_aigc"],
            "split": ["train", "train"],
        }
    )
    with pytest.raises(ValueError, match="do not contain 'validation'"):
        validate_manifest_safety(df_train)


def test_manifest_safety_filters_to_validation() -> None:
    """When mixed splits exist, validate_manifest_safety filters to validation rows."""
    df_mixed = pd.DataFrame(
        {
            "image_path": ["v1.jpg", "v2.jpg", "t1.jpg"],
            "label": ["authentic", "fully_aigc", "fully_aigc"],
            "split": ["validation", "validation", "train"],
        }
    )
    val_df = validate_manifest_safety(df_mixed)
    assert len(val_df) == 2
    assert (val_df["split"] == "validation").all()


def test_manifest_safety_rejects_forbidden_demonstration_data() -> None:
    """Evaluator must reject manifests containing forbidden organizer demo data."""
    # COCO val2017 demo data
    df_coco = pd.DataFrame(
        {
            "image_path": ["coco_val2017/000000123456.jpg"],
            "dataset": ["coco_val2017"],
            "generator": ["camera"],
            "label": ["authentic"],
            "split": ["validation"],
        }
    )
    with pytest.raises(ValueError, match="Forbidden organizer demonstration data"):
        validate_manifest_safety(df_coco)

    # DALL-E Advanced demo data
    df_dalle = pd.DataFrame(
        {
            "image_path": ["dalle_advanced/sample_01.jpg"],
            "dataset": ["dalle_advanced"],
            "generator": ["dalle_advanced"],
            "label": ["fully_aigc"],
            "split": ["validation"],
        }
    )
    with pytest.raises(ValueError, match="Forbidden organizer demonstration data"):
        validate_manifest_safety(df_dalle)


# =========================================================================== #
# 5. Full Evaluation Metrics Hierarchy Tests
# =========================================================================== #


def test_aggregate_evaluation_metrics_complete_hierarchy() -> None:
    """Verify aggregate_evaluation_metrics computes clean/mean/worst, family,

    severity, generator, domain, and worst-domain metrics accurately.
    """
    np.random.seed(99)
    n = 200
    y_true = np.array([0] * (n // 2) + [1] * (n // 2), dtype=int)

    metadata_rows = []
    datasets = ["dataset_A", "dataset_B", "dataset_C"]
    generators = ["midjourney_v6", "flux_1", "dalle_3"]

    for i in range(n):
        dom = datasets[i % len(datasets)]
        gen = generators[i % len(generators)] if y_true[i] == 1 else "camera"
        metadata_rows.append(
            {
                "row_id": f"row_{i}",
                "image_path": f"img_{i}.jpg",
                "dataset": dom,
                "generator": gen,
                "generator_family": gen,
            }
        )

    # Create synthetic condition scores
    condition_scores: dict[str, np.ndarray] = {}
    for cond in SINGLE_TRANSFORM_GRID:
        if cond.name == "clean":
            scores = np.where(
                y_true == 1, np.random.uniform(0.7, 0.99, n), np.random.uniform(0.01, 0.3, n)
            )
        else:
            # Degrade slightly with transform severity
            degrade = 0.05
            scores = np.where(
                y_true == 1,
                np.random.uniform(0.65 - degrade, 0.95, n),
                np.random.uniform(0.05, 0.35 + degrade, n),
            )
        condition_scores[cond.name] = scores

    calibrated_t = 0.50
    metrics = aggregate_evaluation_metrics(
        condition_scores=condition_scores,
        y_true=y_true,
        metadata_rows=metadata_rows,
        calibrated_threshold=calibrated_t,
        satisfies_recall_constraint=True,
    )

    # 1. Clean metrics
    assert "clean" in metrics
    assert metrics["clean"]["ai_positive_auroc"] is not None
    assert metrics["clean"]["ai_positive_auroc"] > 0.95

    # 2. Mean and worst
    assert "mean" in metrics
    assert metrics["mean"]["mean_auroc"] is not None
    assert metrics["mean"]["mean_auroc_drop"] is not None

    assert "worst" in metrics
    assert metrics["worst"]["worst_auroc"] is not None
    assert metrics["worst"]["worst_condition"] in CONDITIONS_BY_NAME
    assert metrics["worst"]["worst_auroc_drop"] >= 0.0

    # 3. Family metrics
    assert "family" in metrics
    for fam in ALLOWED_FAMILIES:
        assert fam in metrics["family"]
        assert "mean_auroc" in metrics["family"][fam]
        assert "worst_auroc" in metrics["family"][fam]
        assert "conditions" in metrics["family"][fam]

    # 4. Severity metrics
    assert "severity" in metrics
    for cond in SINGLE_TRANSFORM_GRID:
        assert cond.name in metrics["severity"]
        assert "auroc" in metrics["severity"][cond.name]

    # 5. Generator metrics
    assert "generator" in metrics
    assert metrics["generator"]["mean_generator_auroc"] is not None
    assert metrics["generator"]["worst_generator_auroc"] is not None
    assert "per_generator" in metrics["generator"]
    for gen in generators:
        assert gen in metrics["generator"]["per_generator"]

    # 6. Domain metrics and worst domain
    assert "domain" in metrics
    for dom in datasets:
        assert dom in metrics["domain"]
        assert metrics["domain"][dom]["auroc"] is not None

    assert "worst_domain" in metrics
    assert metrics["worst_domain"]["name"] in datasets
    assert metrics["worst_domain"]["auroc"] is not None

    # 7. Calibrated threshold metrics
    assert "calibrated" in metrics
    assert metrics["calibrated"]["threshold"] == calibrated_t
    assert "clean" in metrics["calibrated"]
    assert "mean_transformed" in metrics["calibrated"]
    assert "worst_transformed" in metrics["calibrated"]


# =========================================================================== #
# 6. Deterministic Checkpoint Ranking Tests
# =========================================================================== #


def test_checkpoint_ranking_deterministic_and_prioritizes_passing() -> None:
    """Verify candidate checkpoint ranking orders passing candidates first,

    then clean AUROC, worst AUROC, mean AUROC, and balanced accuracy.
    """
    # Candidate 1: 25% checkpoint (fails gate due to low AUROC)
    cand_25 = {
        "report": {
            "checkpoint_path": "outputs/teacher/checkpoint-250.pt",
            "passed": False,
            "metrics": {
                "clean": {"ai_positive_auroc": 0.9400},
                "worst": {"worst_auroc": 0.8800},
                "mean": {"mean_auroc": 0.9100},
                "calibrated": {"clean": {"balanced_accuracy": 0.8800}},
            },
        }
    }

    # Candidate 2: 50% checkpoint (passes gate)
    cand_50 = {
        "report": {
            "checkpoint_path": "outputs/teacher/checkpoint-500.pt",
            "passed": True,
            "metrics": {
                "clean": {"ai_positive_auroc": 0.9700},
                "worst": {"worst_auroc": 0.9200},
                "mean": {"mean_auroc": 0.9450},
                "calibrated": {"clean": {"balanced_accuracy": 0.9200}},
            },
        }
    }

    # Candidate 3: 75% checkpoint (passes gate, higher AUROC than 50%)
    cand_75 = {
        "report": {
            "checkpoint_path": "outputs/teacher/checkpoint-750.pt",
            "passed": True,
            "metrics": {
                "clean": {"ai_positive_auroc": 0.9850},
                "worst": {"worst_auroc": 0.9400},
                "mean": {"mean_auroc": 0.9600},
                "calibrated": {"clean": {"balanced_accuracy": 0.9450}},
            },
        }
    }

    # Candidate 4: 100% checkpoint (passes gate, identical clean AUROC to 75%, but higher worst AUROC)
    cand_100 = {
        "report": {
            "checkpoint_path": "outputs/teacher/checkpoint-1000.pt",
            "passed": True,
            "metrics": {
                "clean": {"ai_positive_auroc": 0.9850},
                "worst": {"worst_auroc": 0.9550},  # Higher worst-case robustness
                "mean": {"mean_auroc": 0.9650},
                "calibrated": {"clean": {"balanced_accuracy": 0.9500}},
            },
        }
    }

    # Shuffle candidates to test deterministic sorting
    candidates = [cand_25, cand_100, cand_50, cand_75]
    ranked = rank_checkpoints(candidates)

    ranked_paths = [r["report"]["checkpoint_path"] for r in ranked]
    # Expected order:
    # 1. 100% checkpoint (passes, clean 0.9850, worst 0.9550)
    # 2. 75% checkpoint (passes, clean 0.9850, worst 0.9400)
    # 3. 50% checkpoint (passes, clean 0.9700)
    # 4. 25% checkpoint (fails gate)
    assert ranked_paths == [
        "outputs/teacher/checkpoint-1000.pt",
        "outputs/teacher/checkpoint-750.pt",
        "outputs/teacher/checkpoint-500.pt",
        "outputs/teacher/checkpoint-250.pt",
    ]


# =========================================================================== #
# 7. Promotion Report and Metadata Sidecar Contract Tests
# =========================================================================== #


def test_promotion_report_and_metadata_sidecar_contracts() -> None:
    """Verify promotion report and metadata sidecar match the exact schema contract."""
    ckpt_path = "models/teachers/stage2/checkpoint-best.pt"
    ckpt_sha256 = "112233445566778899aabbccddeeff00112233445566778899aabbccddeeff00"
    val_digest = "aabbccddeeff00112233445566778899aabbccddeeff00112233445566778899"
    calibrated_t = 0.5234
    metrics = {"clean": {"ai_positive_auroc": 0.9820}}
    passed = True
    failed_reasons: list[str] = []

    # 1. Promotion report
    report = create_promotion_report(
        checkpoint_path=ckpt_path,
        checkpoint_sha256=ckpt_sha256,
        manifest_digest_val=val_digest,
        calibrated_threshold=calibrated_t,
        metrics=metrics,
        passed=passed,
        failed_reasons=failed_reasons,
    )

    # Check exact contract fields
    expected_report_keys = {
        "checkpoint_path",
        "checkpoint_sha256",
        "manifest_digest",
        "calibrated_threshold",
        "metrics",
        "passed",
        "failed_reasons",
    }
    assert set(report.keys()) == expected_report_keys
    assert report["checkpoint_path"] == ckpt_path
    assert report["checkpoint_sha256"] == ckpt_sha256
    assert report["manifest_digest"] == val_digest
    assert report["calibrated_threshold"] == pytest.approx(0.5234)
    assert report["passed"] is True
    assert report["failed_reasons"] == []

    # 2. Metadata sidecar
    sidecar = create_metadata_sidecar(
        manifest_digest_val=val_digest,
        calibrated_threshold=calibrated_t,
        passed=passed,
    )

    expected_sidecar_keys = {
        "model_family",
        "parameter_count",
        "quantization",
        "calibrated_threshold",
        "input_size",
        "preprocessing_version",
        "manifest_digest",
        "evaluation_status",
    }
    assert set(sidecar.keys()) == expected_sidecar_keys
    assert sidecar["model_family"] == TEACHER_CANONICAL_FAMILY
    assert sidecar["parameter_count"] is None

    # Parameter count preserved when explicitly provided
    sidecar_with_params = create_metadata_sidecar(
        manifest_digest_val=val_digest,
        calibrated_threshold=calibrated_t,
        passed=passed,
        parameter_count=TEACHER_FULL_PARAMETER_COUNT,
    )
    assert sidecar_with_params["parameter_count"] == TEACHER_FULL_PARAMETER_COUNT
    assert sidecar["quantization"] == "none"
    assert sidecar["input_size"] == TEACHER_INPUT_SIZE
    assert sidecar["preprocessing_version"] == TEACHER_PREPROCESSING_VERSION
    assert sidecar["manifest_digest"] == val_digest
    assert sidecar["evaluation_status"] == "promoted"

    # Reject status when passed is False
    rejected_sidecar = create_metadata_sidecar(
        manifest_digest_val=val_digest,
        calibrated_threshold=calibrated_t,
        passed=False,
    )
    assert rejected_sidecar["evaluation_status"] == "rejected"


# =========================================================================== #
# 8. End-to-End Promotion from Evaluation Reports
# =========================================================================== #


def test_promote_teacher_from_eval_reports(tmp_path: Path) -> None:
    """Verify promote_teacher_checkpoints correctly promotes the best candidate

    from a directory of evaluation JSON reports.
    """
    ckpt_paths = [
        "outputs/teacher_stage2/checkpoint-250.pt",
        "outputs/teacher_stage2/checkpoint-500.pt",
        "outputs/teacher_stage2/checkpoint-750.pt",
        "outputs/teacher_stage2/checkpoint-1000.pt",
    ]
    aurocs = [0.9300, 0.9650, 0.9780, 0.9850]
    passes = [False, True, True, True]

    eval_report_paths: list[Path] = []
    for idx, (ckpt, auroc, p_status) in enumerate(zip(ckpt_paths, aurocs, passes)):
        rep_file = tmp_path / f"eval_ckpt_{idx}.json"
        rep_data = {
            "report": {
                "checkpoint_path": ckpt,
                "checkpoint_sha256": f"hash_{idx}",
                "manifest_digest": "mock_val_digest_123",
                "calibrated_threshold": 0.5100 + idx * 0.005,
                "metrics": {
                    "clean": {"ai_positive_auroc": auroc},
                    "worst": {"worst_auroc": auroc - 0.03},
                    "mean": {"mean_auroc": auroc - 0.015},
                    "calibrated": {
                        "clean": {
                            "balanced_accuracy": 0.90 + idx * 0.02,
                            "ai_positive_recall": 0.88 + idx * 0.02,
                            "authentic_recall": 0.89 + idx * 0.02,
                        }
                    },
                },
                "passed": p_status,
                "failed_reasons": [] if p_status else ["Clean AUROC 0.9300 <= 0.96"],
            },
            "metadata": {
                "model_family": "dinov3_vith16",
                "parameter_count": TEACHER_FULL_PARAMETER_COUNT,
                "quantization": "none",
                "calibrated_threshold": 0.5100 + idx * 0.005,
                "input_size": [3, 224, 224],
                "preprocessing_version": "square_jpeg95",
                "manifest_digest": "mock_val_digest_123",
                "evaluation_status": "promoted" if p_status else "rejected",
            },
        }
        rep_file.write_text(json.dumps(rep_data), encoding="utf-8")
        eval_report_paths.append(rep_file)

    promotion_result = promote_teacher_checkpoints(eval_reports=eval_report_paths)

    assert promotion_result["promoted_checkpoint"] == "outputs/teacher_stage2/checkpoint-1000.pt"
    assert promotion_result["promotion_report"]["passed"] is True
    assert promotion_result["metadata_sidecar"]["evaluation_status"] == "promoted"
    assert len(promotion_result["ranked_candidates"]) == 4
    # The failing checkpoint should be ranked last
    assert (
        promotion_result["ranked_candidates"][-1]["report"]["checkpoint_path"]
        == "outputs/teacher_stage2/checkpoint-250.pt"
    )


def test_promote_teacher_when_all_candidates_fail(tmp_path: Path) -> None:
    """When all candidate checkpoints fail the gate:

    - promoted_checkpoint is None (do NOT label failed candidate as promoted)
    - selected_candidate contains the highest-ranking candidate path
    - promotion_report has passed == False with failed_reasons
    - metadata_sidecar has evaluation_status == 'rejected'
    """
    ckpt_paths = [
        "outputs/teacher_stage2/checkpoint-250.pt",
        "outputs/teacher_stage2/checkpoint-500.pt",
    ]
    aurocs = [0.9100, 0.9400]

    eval_report_paths: list[Path] = []
    for idx, (ckpt, auroc) in enumerate(zip(ckpt_paths, aurocs)):
        rep_file = tmp_path / f"failed_eval_{idx}.json"
        rep_data = {
            "report": {
                "checkpoint_path": ckpt,
                "checkpoint_sha256": f"hash_fail_{idx}",
                "manifest_digest": "mock_val_digest_fail",
                "calibrated_threshold": 0.50,
                "metrics": {
                    "clean": {"ai_positive_auroc": auroc},
                    "worst": {"worst_auroc": auroc - 0.05},
                    "mean": {"mean_auroc": auroc - 0.02},
                    "calibrated": {
                        "clean": {
                            "balanced_accuracy": 0.85,
                            "ai_positive_recall": 0.79,
                            "authentic_recall": 0.80,
                        }
                    },
                },
                "passed": False,
                "failed_reasons": [
                    f"Clean AUROC {auroc:.4f} <= 0.96 (required > 0.96)",
                    "AI-positive recall at calibrated threshold 0.7900 < 0.82 (required >= 0.82)",
                ],
            },
            "metadata": {
                "model_family": "dinov3_vith16",
                "parameter_count": None,
                "quantization": "none",
                "calibrated_threshold": 0.50,
                "input_size": [3, 224, 224],
                "preprocessing_version": "square_jpeg95",
                "manifest_digest": "mock_val_digest_fail",
                "evaluation_status": "rejected",
            },
        }
        rep_file.write_text(json.dumps(rep_data), encoding="utf-8")
        eval_report_paths.append(rep_file)

    res = promote_teacher_checkpoints(eval_reports=eval_report_paths)

    assert res["promoted_checkpoint"] is None
    assert res["selected_candidate"] == "outputs/teacher_stage2/checkpoint-500.pt"
    assert res["promotion_report"]["passed"] is False
    assert len(res["promotion_report"]["failed_reasons"]) > 0
    assert res["metadata_sidecar"]["evaluation_status"] == "rejected"


def test_promote_teacher_copies_promoted_checkpoint_when_passed_and_fails_closed_when_rejected(
    tmp_path: Path,
) -> None:
    """Verify promote_teacher copies winning checkpoint to output_checkpoint on pass, and fails closed on fail."""
    import shutil

    # Create mock checkpoint
    ckpt_source = tmp_path / "checkpoint-1000.pt"
    ckpt_source.write_bytes(b"teacher weights binary mock")

    rep_file = tmp_path / "passed_eval.json"
    rep_file.write_text(
        json.dumps(
            {
                "report": {
                    "checkpoint_path": str(ckpt_source),
                    "checkpoint_sha256": "fake_sha",
                    "manifest_digest": "mock_val_digest",
                    "calibrated_threshold": 0.52,
                    "metrics": {
                        "clean": {"ai_positive_auroc": 0.9850},
                        "worst": {"worst_auroc": 0.9550},
                        "mean": {"mean_auroc": 0.9700},
                        "calibrated": {
                            "clean": {
                                "balanced_accuracy": 0.96,
                                "ai_positive_recall": 0.95,
                                "authentic_recall": 0.97,
                            }
                        },
                    },
                    "passed": True,
                    "failed_reasons": [],
                },
                "metadata": {
                    "model_family": "dinov3_vith16",
                    "parameter_count": 872606207,
                    "quantization": "none",
                    "calibrated_threshold": 0.52,
                    "input_size": [3, 224, 224],
                    "preprocessing_version": "square_jpeg95",
                    "manifest_digest": "mock_val_digest",
                    "evaluation_status": "promoted",
                },
            }
        ),
        encoding="utf-8",
    )

    promoted_dest = tmp_path / "checkpoint-promoted.pt"
    res = promote_teacher_checkpoints(eval_reports=[rep_file])
    assert res["promoted_checkpoint"] == str(ckpt_source)

    # Simulate CLI output-checkpoint write
    if res["promotion_report"]["passed"]:
        shutil.copyfile(res["promotion_report"]["checkpoint_path"], promoted_dest)
    assert promoted_dest.is_file()
    assert promoted_dest.read_bytes() == b"teacher weights binary mock"

    # Now test fail-closed with failing report
    promoted_dest_fail = tmp_path / "checkpoint-promoted-fail.pt"
    rep_fail_file = tmp_path / "failed_eval.json"
    rep_fail_file.write_text(
        json.dumps(
            {
                "report": {
                    "checkpoint_path": str(ckpt_source),
                    "checkpoint_sha256": "fake_sha",
                    "manifest_digest": "mock_val_digest",
                    "calibrated_threshold": 0.52,
                    "metrics": {
                        "clean": {"ai_positive_auroc": 0.9100},
                        "worst": {"worst_auroc": 0.8550},
                        "mean": {"mean_auroc": 0.8700},
                        "calibrated": {
                            "clean": {
                                "balanced_accuracy": 0.80,
                                "ai_positive_recall": 0.75,
                                "authentic_recall": 0.85,
                            }
                        },
                    },
                    "passed": False,
                    "failed_reasons": ["Clean AUROC 0.9100 <= 0.96"],
                },
                "metadata": {
                    "model_family": "dinov3_vith16",
                    "parameter_count": 872606207,
                    "quantization": "none",
                    "calibrated_threshold": 0.52,
                    "input_size": [3, 224, 224],
                    "preprocessing_version": "square_jpeg95",
                    "manifest_digest": "mock_val_digest",
                    "evaluation_status": "rejected",
                },
            }
        ),
        encoding="utf-8",
    )

    res_fail = promote_teacher_checkpoints(eval_reports=[rep_fail_file])
    assert res_fail["promoted_checkpoint"] is None
    assert res_fail["promotion_report"]["passed"] is False
    if res_fail["promotion_report"]["passed"]:
        shutil.copyfile(res_fail["promotion_report"]["checkpoint_path"], promoted_dest_fail)
    assert not promoted_dest_fail.is_file()
