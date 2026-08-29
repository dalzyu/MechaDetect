from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch
import yaml

from scripts.distill_student import (
    STUDENT_EVALUATION_CONDITIONS,
    STUDENT_METADATA,
    STUDENT_PRESETS,
    calibrate_validation_threshold,
    compute_file_sha256,
    get_student_parameter_counts,
    student_config,
    verify_checkpoint_eligibility,
    verify_teacher_promotion,
)
from scripts.launch_students_distill import (
    TRACK_SPECS,
    build_track_command,
    select_default_student_model,
)

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"


def _load_yaml(name: str) -> dict:
    return yaml.safe_load((CONFIG_ROOT / name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. Exact Parameter Counts and Architecture Specifications
# ---------------------------------------------------------------------------


def test_student_small_exact_complete_detector_parameter_count() -> None:
    """The complete MechaDetect ViT-S detector must be exactly 25,089,666 parameters."""
    counts = get_student_parameter_counts("small")
    assert counts["backbone_parameters"] == 21596544, "DINOv3 ViT-S backbone must be ~21.6M params"
    assert counts["token_adapter_parameters"] == 197888, (
        "Token adapter (384 -> 512) must be 197,888 params"
    )
    assert counts["heads_parameters"] == 3295234, "Shared provenance heads must be 3,295,234 params"
    assert counts["exact_parameter_count"] == 25089666, (
        "Complete ViT-S detector must be exactly 25,089,666 params"
    )

    meta = STUDENT_METADATA["small"]
    assert meta["model_family"] == "dinov3-vits16"
    assert meta["encoder_id"] == "facebook/dinov3-vits16-pretrain-lvd1689m"
    assert meta["encoder_dim"] == 384
    assert meta["image_size"] == 224


def test_student_base_exact_complete_detector_parameter_count() -> None:
    """The complete MechaDetect ViT-B detector must be exactly 89,350,914 parameters."""
    counts = get_student_parameter_counts("base")
    assert counts["backbone_parameters"] == 85660416, "DINOv3 ViT-B backbone must be ~85.7M params"
    assert counts["token_adapter_parameters"] == 395264, (
        "Token adapter (768 -> 512) must be 395,264 params"
    )
    assert counts["heads_parameters"] == 3295234, "Shared provenance heads must be 3,295,234 params"
    assert counts["exact_parameter_count"] == 89350914, (
        "Complete ViT-B detector must be exactly 89,350,914 params"
    )

    meta = STUDENT_METADATA["base"]
    assert meta["model_family"] == "dinov3-vitb16"
    assert meta["encoder_id"] == "facebook/dinov3-vitb16-pretrain-lvd1689m"
    assert meta["encoder_dim"] == 768
    assert meta["image_size"] == 224


def test_student_presets_canonical_contract() -> None:
    """STUDENT_PRESETS reflects the canonical 2-GPU, 2-epoch schedule with effective batch 48."""
    for variant in ("small", "base"):
        assert variant in STUDENT_PRESETS
        spec = STUDENT_PRESETS[variant]
        assert spec["epochs"] == 2
        assert spec["required_world_size"] == 2
        assert spec["physical_batch_size"] == 12
        assert (
            spec["physical_batch_size"]
            * spec["gradient_accumulation"]
            * spec["required_world_size"]
            == 48
        )


# ---------------------------------------------------------------------------
# 2. Incomplete update250 Small Artifact Guard
# ---------------------------------------------------------------------------


def test_verify_checkpoint_eligibility_rejects_incomplete_update250_artifact() -> None:
    """The existing iteration 1 update 250 small artifact must not be used as a final student."""
    with pytest.raises(ValueError, match="incomplete iteration 1 exploration checkpoint"):
        verify_checkpoint_eligibility("outputs/student_vits_checkpoint2/student_checkpoint_250.pt")

    with pytest.raises(ValueError, match="incomplete iteration 1 exploration checkpoint"):
        verify_checkpoint_eligibility(
            "C:/repos/techjam 26/outputs/student_vits_checkpoint2/student_config.yaml"
        )

    # Legitimate candidate checkpoints pass without raising
    verify_checkpoint_eligibility("outputs/student_dinov3_small/checkpoint-promoted.pt")
    verify_checkpoint_eligibility("outputs/student_dinov3_base/checkpoint-best.pt")


# ---------------------------------------------------------------------------
# 3. Canonical Configs and Batch Geometry
# ---------------------------------------------------------------------------


def test_student_distillation_configs_preserve_effective_batch_48() -> None:
    """Both student configs must enforce effective batch 48 across 2 GPUs (12 x 2 x 2 = 48)."""
    small = _load_yaml("student_dinov3_small_distill.yaml")
    base = _load_yaml("student_dinov3_base_distill.yaml")

    for cfg in (small, base):
        assert cfg["paths"]["train_manifest"] == "splits/production_eligible/train.parquet"
        assert cfg["paths"]["val_manifest"] == "splits/production_eligible/validation.parquet"
        assert cfg["training"]["epochs"] == 2, "Must specify two deterministic complete passes"
        assert cfg["training"]["required_world_size"] == 2, "Must run on a 2-GPU pool"
        assert cfg["training"]["deterministic_coverage_sampler"] is True
        assert cfg["training"]["precision"] == "bf16"
        assert cfg["training"]["optimizer"] == "adamw"
        assert cfg["seed"] == 42

        # Effective batch calculation: physical * world_size * accumulation == 48
        phys = cfg["training"]["physical_batch_size"]
        accum = cfg["training"]["gradient_accumulation"]
        ws = cfg["training"]["required_world_size"]
        assert phys * accum * ws == 48, f"Effective batch must be 48, got {phys * accum * ws}"

        # Standard loss contract
        assert cfg["loss"]["provenance_original"] == 1.0
        assert cfg["loss"]["provenance_transformed"] == 1.0
        assert cfg["loss"]["prediction_consistency"] == 1.0
        assert cfg["loss"]["feature_consistency"] == 0.5
        assert cfg["loss"]["ema_consistency"] == 2.0  # Teacher soft targets
        assert cfg["loss"]["teacher_feature_consistency"] == 0.5  # Teacher feature alignment


def test_student_distillation_hardware_and_port_isolation() -> None:
    """Small and Base tracks must use disjoint GPU sets, ports, and output directories."""
    small = _load_yaml("student_dinov3_small_distill.yaml")
    base = _load_yaml("student_dinov3_base_distill.yaml")

    assert small["training"]["cuda_visible_devices"] == "0,1"
    assert base["training"]["cuda_visible_devices"] == "2,3"
    assert small["training"]["master_port"] == 29501
    assert base["training"]["master_port"] == 29502
    assert small["paths"]["output_root"] != base["paths"]["output_root"]

    # Launcher specs match
    assert TRACK_SPECS["small"]["devices"] == "0,1"
    assert TRACK_SPECS["base"]["devices"] == "2,3"
    assert TRACK_SPECS["small"]["port"] == 29501
    assert TRACK_SPECS["base"]["port"] == 29502


def test_student_config_builder_sets_canonical_attributes() -> None:
    """student_config helper sets canonical 2-GPU, 2-epoch attributes with no max-update cap."""
    teacher_cfg = {
        "model": {"backbone_type": "dinov3"},
        "training": {"max_updates": 1000},
        "loss": {},
    }
    cfg_small = student_config(teacher_cfg, "small")
    assert cfg_small["model"]["encoder_id"] == "facebook/dinov3-vits16-pretrain-lvd1689m"
    assert cfg_small["model"]["encoder_dim"] == 384
    assert cfg_small["training"]["stage"] == "student_dinov3_small"
    assert cfg_small["training"]["required_world_size"] == 2
    assert cfg_small["training"]["epochs"] == 2
    assert "max_updates" not in cfg_small["training"]

    cfg_base = student_config(teacher_cfg, "base")
    assert cfg_base["model"]["encoder_id"] == "facebook/dinov3-vitb16-pretrain-lvd1689m"
    assert cfg_base["model"]["encoder_dim"] == 768
    assert cfg_base["training"]["stage"] == "student_dinov3_base"
    assert cfg_base["training"]["required_world_size"] == 2
    assert cfg_base["training"]["epochs"] == 2
    assert "max_updates" not in cfg_base["training"]


# ---------------------------------------------------------------------------
# 4. Launch Helper Verification
# ---------------------------------------------------------------------------


def test_build_track_command_uses_torch_distributed_run_in_production() -> None:
    """build_track_command generates torch.distributed.run command with nproc_per_node=2."""
    cmd = build_track_command(
        "small",
        teacher_config="configs/teacher.yaml",
        teacher_checkpoint="outputs/teacher/checkpoint-promoted.pt",
        manifest="splits/train.parquet",
        output_dir="outputs/student_small",
        val_manifest="splits/val.parquet",
        teacher_promotion_report="outputs/teacher/promotion_report.json",
        epochs=2,
        world_size=2,
        physical_batch_size=12,
        gradient_accumulation=2,
        dry_run=False,
    )

    cmd_str = " ".join(cmd)
    assert "-m torch.distributed.run --nproc_per_node=2 --master_port=29501" in cmd_str
    assert "--student small" in cmd_str
    assert "--teacher-config configs/teacher.yaml" in cmd_str
    assert "--teacher-checkpoint outputs/teacher/checkpoint-promoted.pt" in cmd_str
    assert "--manifest splits/train.parquet" in cmd_str
    assert "--val-manifest splits/val.parquet" in cmd_str
    assert "--teacher-promotion-report outputs/teacher/promotion_report.json" in cmd_str
    assert "--epochs 2" in cmd_str
    assert "--world-size 2" in cmd_str
    assert "--physical-batch-size 12" in cmd_str
    assert "--gradient-accumulation 2" in cmd_str


def test_build_track_command_dry_run_uses_real_two_process_ddp() -> None:
    """Dry-run preserves the production two-GPU topology while stopping after two updates."""
    cmd = build_track_command(
        "base",
        teacher_config="configs/teacher.yaml",
        teacher_checkpoint="outputs/teacher/checkpoint-promoted.pt",
        manifest="splits/train.parquet",
        output_dir="outputs/student_base",
        dry_run=True,
    )
    cmd_str = " ".join(cmd)
    assert "torch.distributed.run" in cmd_str
    assert "--nproc_per_node=2" in cmd_str
    assert "--student base" in cmd_str
    assert "--dry-run" in cmd_str


# ---------------------------------------------------------------------------
# 5. Teacher Promotion and Contract Metrics Parsing
# ---------------------------------------------------------------------------


def test_verify_teacher_promotion_parses_contract_metrics_and_checks_sha() -> None:
    """verify_teacher_promotion validates checkpoint SHA256 and extracts clean and worst AUROC."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        ckpt = tmp / "teacher.pt"
        ckpt.write_bytes(b"dummy teacher checkpoint content for testing")
        manifest = tmp / "train.parquet"
        import pandas as pd

        pd.DataFrame([{"dataset": "test", "image_path": "a.png"}]).to_parquet(manifest)

        expected_sha = compute_file_sha256(ckpt)

        # 1. Report passed matching TeacherPromotion contract
        report = tmp / "promotion_report.json"
        report.write_text(
            json.dumps(
                {
                    "passed": True,
                    "checkpoint_sha256": expected_sha,
                    "metrics": {
                        "clean": {
                            "ai_positive_auroc": 0.978,
                            "ai_positive_recall": 0.89,
                            "authentic_recall": 0.91,
                        },
                        "worst": {
                            "worst_auroc": 0.925,
                            "worst_condition": "jpeg30",
                        },
                    },
                }
            ),
            encoding="utf-8",
        )

        res = verify_teacher_promotion(ckpt, report, manifest)
        assert res["teacher_passed"] is True
        assert res["teacher_checkpoint_sha256"] == expected_sha
        assert res["teacher_clean_auroc"] == 0.978
        assert res["teacher_worst_transformed_auroc"] == 0.925

        # 2. Unpromoted teacher raises RuntimeError
        report_failed = tmp / "promotion_report_failed.json"
        report_failed.write_text(
            json.dumps(
                {
                    "passed": False,
                    "failed_reasons": ["AI-positive recall 0.79 < 0.82"],
                    "checkpoint_sha256": expected_sha,
                    "metrics": {
                        "clean": {"ai_positive_auroc": 0.95},
                        "worst": {"worst_auroc": 0.88},
                    },
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(RuntimeError, match="Teacher checkpoint failed promotion gate"):
            verify_teacher_promotion(ckpt, report_failed, manifest)

        # 3. Checkpoint SHA256 mismatch raises RuntimeError
        report_mismatch = tmp / "promotion_report_mismatch.json"
        report_mismatch.write_text(
            json.dumps(
                {
                    "passed": True,
                    "checkpoint_sha256": "different_hash_value_12345",
                    "metrics": {
                        "clean": {"ai_positive_auroc": 0.97},
                        "worst": {"worst_auroc": 0.91},
                    },
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(RuntimeError, match="Teacher checkpoint SHA256 mismatch"):
            verify_teacher_promotion(ckpt, report_mismatch, manifest)

        # 4. Missing nested metrics raises RuntimeError
        report_missing_metrics = tmp / "promotion_report_missing_metrics.json"
        report_missing_metrics.write_text(
            json.dumps(
                {
                    "passed": True,
                    "checkpoint_sha256": expected_sha,
                    "metrics": {
                        "flat_clean_auroc": 0.97,
                    },
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(RuntimeError, match="missing metrics.clean.ai_positive_auroc"):
            verify_teacher_promotion(ckpt, report_missing_metrics, manifest)


# ---------------------------------------------------------------------------
# 6. Constrained Validation Threshold Calibration
# ---------------------------------------------------------------------------


def test_calibrate_validation_threshold_maximizes_balanced_accuracy() -> None:
    """calibrate_validation_threshold finds threshold satisfying recalls >= 0.82."""
    # Synthetic scores: authentic centered around 0.2, AI-positive centered around 0.8
    torch.manual_seed(42)
    auth_scores = torch.normal(mean=0.25, std=0.10, size=(100,)).clamp(0.01, 0.99)
    ai_scores = torch.normal(mean=0.75, std=0.10, size=(100,)).clamp(0.01, 0.99)

    scores = torch.cat([auth_scores, ai_scores])
    targets = torch.cat([torch.zeros(100), torch.ones(100)])

    thresh, tpr, tnr, bal_acc, satisfied = calibrate_validation_threshold(targets, scores)
    assert satisfied is True, "Should find a threshold satisfying both recalls >= 0.82"
    assert tpr >= 0.82, f"AI-positive recall {tpr} must be >= 0.82"
    assert tnr >= 0.82, f"Authentic recall {tnr} must be >= 0.82"
    assert bal_acc >= 0.85, f"Balanced accuracy {bal_acc} should be high on well-separated data"


# ---------------------------------------------------------------------------
# 7. Default Model Selection Decision Logic
# ---------------------------------------------------------------------------


def test_select_default_student_model_picks_vit_s_by_default() -> None:
    """ViT-S is default unless ViT-B gains >= 1pp worst-transform AUROC or fixes ViT-S recall."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        small_rep = tmp / "small_promotion_report.json"
        base_rep = tmp / "base_promotion_report.json"

        # Case 1: Both pass, ViT-B gain is 0.005 (< 0.01 threshold) -> Select ViT-S
        small_rep.write_text(
            json.dumps(
                {
                    "passed": True,
                    "checkpoint_path": "outputs/small/ckpt.pt",
                    "metrics": {
                        "clean": {
                            "ai_positive_auroc": 0.965,
                            "ai_positive_recall": 0.85,
                            "authentic_recall": 0.86,
                        },
                        "worst": {"worst_auroc": 0.900},
                    },
                }
            )
        )
        base_rep.write_text(
            json.dumps(
                {
                    "passed": True,
                    "checkpoint_path": "outputs/base/ckpt.pt",
                    "metrics": {
                        "clean": {
                            "ai_positive_auroc": 0.970,
                            "ai_positive_recall": 0.86,
                            "authentic_recall": 0.87,
                        },
                        "worst": {"worst_auroc": 0.905},  # Gain: 0.005 < 0.010
                    },
                }
            )
        )

        res = select_default_student_model(small_rep, base_rep, output_dir=tmp)
        assert res["selected_model"] == "small"
        assert "default edge model" in res["decision_reason"]

        # Case 2: ViT-B gain is 0.015 (>= 0.010 threshold) -> Select ViT-B
        base_rep.write_text(
            json.dumps(
                {
                    "passed": True,
                    "checkpoint_path": "outputs/base/ckpt.pt",
                    "metrics": {
                        "clean": {
                            "ai_positive_auroc": 0.975,
                            "ai_positive_recall": 0.88,
                            "authentic_recall": 0.89,
                        },
                        "worst": {"worst_auroc": 0.915},  # Gain: 0.015 >= 0.010
                    },
                }
            )
        )
        res = select_default_student_model(small_rep, base_rep, output_dir=tmp)
        assert res["selected_model"] == "base"
        assert "exceeds 1.0pp threshold" in res["decision_reason"]

        # Case 3: ViT-S failed recall gate, ViT-B passed -> Select ViT-B
        small_rep.write_text(
            json.dumps(
                {
                    "passed": False,
                    "checkpoint_path": "outputs/small/ckpt.pt",
                    "metrics": {
                        "clean": {
                            "ai_positive_auroc": 0.965,
                            "ai_positive_recall": 0.79,
                            "authentic_recall": 0.85,
                        },
                        "worst": {"worst_auroc": 0.900},
                    },
                }
            )
        )
        base_rep.write_text(
            json.dumps(
                {
                    "passed": True,
                    "checkpoint_path": "outputs/base/ckpt.pt",
                    "metrics": {
                        "clean": {
                            "ai_positive_auroc": 0.968,
                            "ai_positive_recall": 0.84,
                            "authentic_recall": 0.85,
                        },
                        "worst": {"worst_auroc": 0.902},  # Gain: 0.002, but fixes recall!
                    },
                }
            )
        )
        res = select_default_student_model(small_rep, base_rep, output_dir=tmp)
        assert res["selected_model"] == "base"
        assert "failed recall gate" in res["decision_reason"]


# ---------------------------------------------------------------------------
# 8. Evaluation Conditions Grid
# ---------------------------------------------------------------------------


def test_student_evaluation_conditions_cover_required_transformations() -> None:
    """Evaluation conditions must include clean plus primary post-processing families."""
    conditions = STUDENT_EVALUATION_CONDITIONS
    assert "clean" in conditions
    assert "jpeg70" in conditions
    assert "blur1" in conditions
    assert "resize_half" in conditions
    assert "noise0.05" in conditions
    assert "color20" in conditions
    assert "crop80" in conditions


# ---------------------------------------------------------------------------
# 9. Gate Enforcement, Manifest Security & Isolation Tests
# ---------------------------------------------------------------------------


def test_distill_student_missing_manifest_fails_closed() -> None:
    """distill_student fails closed if manifest does not exist (no fallback searches)."""
    from scripts.distill_student import train_student

    with pytest.raises(FileNotFoundError, match="Training manifest not found"):
        train_student(
            teacher_config_path=CONFIG_ROOT / "teacher_dinov3_stage2_paired_unfrozen.yaml",
            teacher_checkpoint=Path("dummy_teacher.pt"),
            manifest_path=Path("nonexistent_manifest_12345.parquet"),
            output_dir=Path("outputs/test_student"),
            variant="small",
            dry_run=True,
        )


def test_launch_students_distill_accepts_strict_teacher_gate_flag() -> None:
    """launch_students_distill CLI accepts --strict-teacher-gate without unrecognized argument error."""
    import subprocess
    import sys

    res = subprocess.run(
        [
            sys.executable,
            "scripts/launch_students_distill.py",
            "--help",
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "--strict-teacher-gate" in res.stdout
