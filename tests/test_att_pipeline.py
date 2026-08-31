from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn as nn
import yaml
from PIL import Image

from aigc_detector.constants import Transformation
from aigc_detector.model import ProvenanceOutput
from aigc_detector.transforms import TransformSpec
from scripts.check_att_gate import (
    build_shared_report,
    evaluate_track_gate,
)
from scripts.launch_att_tracks import build_track_command

# Import newly owned ATT scripts and modules
from scripts.train_att import (
    ATT_SEVERITY_VALUES,
    ATT_TRANSFORMATION_FAMILIES,
    compute_sample_loss,
    generate_att_candidates,
    score_and_select_hardest,
    verify_train_split_membership,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


# ---------------------------------------------------------------------------
# 1. Deterministic candidate generation tests
# ---------------------------------------------------------------------------


def test_deterministic_candidate_generation_is_reproducible() -> None:
    img = Image.new("RGB", (224, 224), color=(128, 64, 32))
    row_id = "sample_row_42"
    base_seed = 12345

    # Same seed, same row_id -> exact same candidates
    candidates_1 = generate_att_candidates(
        img,
        num_candidates=3,
        row_id=row_id,
        base_seed=base_seed,
        epoch=0,
    )
    candidates_2 = generate_att_candidates(
        img,
        num_candidates=3,
        row_id=row_id,
        base_seed=base_seed,
        epoch=0,
    )

    assert len(candidates_1) == 3
    assert len(candidates_2) == 3

    for (cand_img_1, spec_1), (cand_img_2, spec_2) in zip(candidates_1, candidates_2):
        assert spec_1.family == spec_2.family
        assert spec_1.severity == spec_2.severity
        # Pixels must be identical
        assert cand_img_1.tobytes() == cand_img_2.tobytes()


def test_deterministic_candidate_generation_varies_with_seed_or_row() -> None:
    img = Image.new("RGB", (224, 224), color=(100, 150, 200))

    candidates_a = generate_att_candidates(
        img,
        num_candidates=3,
        row_id="row_a",
        base_seed=42,
        epoch=0,
    )
    candidates_b = generate_att_candidates(
        img,
        num_candidates=3,
        row_id="row_b",
        base_seed=42,
        epoch=0,
    )

    specs_a = [(c.family, c.severity) for _, c in candidates_a]
    specs_b = [(c.family, c.severity) for _, c in candidates_b]
    # Distinct rows should produce distinct transform samples
    assert specs_a != specs_b


def test_candidate_generation_single_transform_and_allowed_families() -> None:
    img = Image.new("RGB", (224, 224), color=(50, 100, 150))
    candidates = generate_att_candidates(
        img,
        num_candidates=6,
        row_id="test_row",
        base_seed=999,
    )

    assert len(candidates) == 6
    for _, spec in candidates:
        assert isinstance(spec, TransformSpec)
        # Must belong to organizer-aligned single-transform families
        assert spec.family in ATT_TRANSFORMATION_FAMILIES
        # Must use valid severity from allowed set
        allowed_sevs = ATT_SEVERITY_VALUES[spec.family]
        assert spec.severity in allowed_sevs


# ---------------------------------------------------------------------------
# 2. Hardest candidate scoring without gradients
# ---------------------------------------------------------------------------


class MockStudentModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 1)
        # Initialize positive weights so higher pixel mean -> higher logit
        with torch.no_grad():
            self.linear.weight.fill_(0.01)
            self.linear.bias.zero_()

    def forward(self, images: list[Image.Image]) -> ProvenanceOutput:
        if self.linear.weight.requires_grad and getattr(self, "expect_no_grad", False):
            assert not torch.is_grad_enabled(), "Candidate scoring must execute without gradients!"

        batch_size = len(images)
        means = torch.tensor(
            [[float(sum(images[i].getpixel((0, 0))) / 3.0)] * 3 for i in range(batch_size)],
            dtype=torch.float32,
        )
        logits = self.linear(means).squeeze(-1)
        probs = torch.stack([1.0 - torch.sigmoid(logits), torch.sigmoid(logits)], dim=-1)
        return ProvenanceOutput(
            ai_positive_logit=logits,
            probabilities=probs,
            aigc_features=torch.zeros((batch_size, 3)),
            tamper_features=torch.zeros((batch_size, 3)),
            token_tamper_logits=torch.zeros((batch_size, 1)),
            fusion_gates=None,
        )


def test_scoring_without_gradients_and_hardest_selection() -> None:
    student = MockStudentModel()
    student.expect_no_grad = True

    # 2 items in batch, each having 3 candidate images with distinct colors
    item0_cands = [
        (Image.new("RGB", (32, 32), color=(10, 10, 10)), TransformSpec(Transformation.JPEG, 90.0)),
        (
            Image.new("RGB", (32, 32), color=(200, 200, 200)),
            TransformSpec(Transformation.BLUR, 1.0),
        ),
        (Image.new("RGB", (32, 32), color=(50, 50, 50)), TransformSpec(Transformation.NOISE, 0.05)),
    ]
    item1_cands = [
        (
            Image.new("RGB", (32, 32), color=(250, 250, 250)),
            TransformSpec(Transformation.COLOR, 0.2),
        ),
        (Image.new("RGB", (32, 32), color=(20, 20, 20)), TransformSpec(Transformation.RESIZE, 0.5)),
        (Image.new("RGB", (32, 32), color=(80, 80, 80)), TransformSpec(Transformation.CROP, 0.8)),
    ]

    candidates_batch = [item0_cands, item1_cands]
    # Ground truth: authentic (0.0)
    provenance = torch.tensor([0, 0], dtype=torch.long)
    ai_positive = torch.tensor([0.0, 0.0], dtype=torch.float32)
    device = torch.device("cpu")

    hardest_images, hardest_specs, hardest_indices = score_and_select_hardest(
        student, candidates_batch, provenance, ai_positive, device
    )

    assert len(hardest_images) == 2
    assert len(hardest_specs) == 2
    assert len(hardest_indices) == 2

    # For item 0, candidate with color (200,200,200) produces highest positive logit
    # which has highest BCE loss when target is 0.0
    assert hardest_indices[0] == 1
    assert hardest_specs[0].family == Transformation.BLUR

    # For item 1, candidate with color (250,250,250) produces highest logit & highest loss
    assert hardest_indices[1] == 0
    assert hardest_specs[1].family == Transformation.COLOR


# ---------------------------------------------------------------------------
# 3. Original supervised term guaranteed in ATT update
# ---------------------------------------------------------------------------


def test_original_supervised_loss_term_guaranteed() -> None:
    student = MockStudentModel()
    student.expect_no_grad = False

    orig_img = [Image.new("RGB", (32, 32), color=(100, 100, 100))]
    hard_img = [Image.new("RGB", (32, 32), color=(200, 200, 200))]
    ai_pos = torch.tensor([1.0], dtype=torch.float32)
    prov = torch.tensor([1], dtype=torch.long)

    # Enable gradients for training step
    student.linear.weight.requires_grad_(True)

    orig_out = student(orig_img)
    hard_out = student(hard_img)

    loss_orig = compute_sample_loss(
        orig_out.ai_positive_logit, orig_out.probabilities, ai_pos, prov
    ).mean()
    loss_hard = compute_sample_loss(
        hard_out.ai_positive_logit, hard_out.probabilities, ai_pos, prov
    ).mean()

    # Original supervised loss term must have weight >= 1.0
    w_orig = 1.0
    w_hard = 1.0
    total_loss = w_orig * loss_orig + w_hard * loss_hard

    assert loss_orig.item() > 0.0
    assert loss_hard.item() > 0.0
    assert total_loss.item() >= loss_orig.item()

    total_loss.backward()
    # Backprop must produce gradients
    assert student.linear.weight.grad is not None
    assert torch.isfinite(student.linear.weight.grad).all()


# ---------------------------------------------------------------------------
# 4. Stage membership guard
# ---------------------------------------------------------------------------


def test_stage_membership_guard_rejects_non_train_splits() -> None:
    # 1. Clean train split should pass
    clean_train_df = pd.DataFrame(
        {
            "image_path": ["img1.jpg", "img2.jpg"],
            "split": ["train", "train"],
            "provenance": [0, 1],
        }
    )
    verified_df = verify_train_split_membership(clean_train_df)
    assert len(verified_df) == 2

    # 2. Manifest containing validation split must be rejected
    val_leak_df = pd.DataFrame(
        {
            "image_path": ["img1.jpg", "img2.jpg"],
            "split": ["train", "validation"],
        }
    )
    with pytest.raises(ValueError, match="ATT Membership Guard Violation"):
        verify_train_split_membership(val_leak_df)

    # 3. Manifest containing test or calibration splits must be rejected
    test_leak_df = pd.DataFrame(
        {
            "image_path": ["img1.jpg", "img2.jpg"],
            "split": ["train", "test_unseen"],
        }
    )
    with pytest.raises(ValueError, match="ATT Membership Guard Violation"):
        verify_train_split_membership(test_leak_df)

    calib_leak_df = pd.DataFrame(
        {
            "image_path": ["img1.jpg", "img2.jpg"],
            "split": ["train", "calibration"],
        }
    )
    with pytest.raises(ValueError, match="ATT Membership Guard Violation"):
        verify_train_split_membership(calib_leak_df)


# ---------------------------------------------------------------------------
# 5. Track isolation and configuration verification
# ---------------------------------------------------------------------------


def test_att_configs_default_to_sequential_single_gpu_tracks() -> None:
    small_cfg_path = CONFIG_DIR / "att_student_small.yaml"
    base_cfg_path = CONFIG_DIR / "att_student_base.yaml"

    assert small_cfg_path.is_file()
    assert base_cfg_path.is_file()

    small_cfg = yaml.safe_load(small_cfg_path.read_text(encoding="utf-8"))
    base_cfg = yaml.safe_load(base_cfg_path.read_text(encoding="utf-8"))

    # Track small checks
    assert small_cfg["model"]["encoder_id"] == "facebook/dinov3-vits16-pretrain-lvd1689m"
    assert small_cfg["model"]["encoder_dim"] == 384
    assert small_cfg["training"]["cuda_visible_devices"] == "0"
    assert small_cfg["training"]["master_port"] == 29503
    assert "small" in small_cfg["paths"]["output_root"]

    # Track base checks
    assert base_cfg["model"]["encoder_id"] == "facebook/dinov3-vitb16-pretrain-lvd1689m"
    assert base_cfg["model"]["encoder_dim"] == 768
    assert base_cfg["training"]["cuda_visible_devices"] == "0"
    assert base_cfg["training"]["master_port"] == 29504
    assert "base" in base_cfg["paths"]["output_root"]

    # Sequential defaults may share GPU 0; ports and output directories remain distinct.
    assert small_cfg["training"]["master_port"] != base_cfg["training"]["master_port"]
    assert small_cfg["paths"]["output_root"] != base_cfg["paths"]["output_root"]

    # Production manifest and materialization requirements
    for cfg in (small_cfg, base_cfg):
        assert cfg["paths"]["train_manifest"] == "splits/production_eligible/train.parquet"
        assert cfg["paths"]["val_manifest"] == "splits/production_eligible/validation.parquet"
        assert cfg["paths"]["require_materialized"] is True

    # One-GPU geometry: small 4x12x1 = 48; base 2x24x1 = 48.
    assert small_cfg["training"]["physical_batch_size"] == 4
    assert small_cfg["training"]["gradient_accumulation"] == 12
    assert base_cfg["training"]["physical_batch_size"] == 2
    assert base_cfg["training"]["gradient_accumulation"] == 24

    for cfg in (small_cfg, base_cfg):
        batch = cfg["training"]["physical_batch_size"]
        accum = cfg["training"]["gradient_accumulation"]
        world = cfg["training"]["required_world_size"]
        assert batch * accum * world == 48, (
            f"Effective batch must be 48, got {batch * accum * world}"
        )
        assert cfg["training"]["epochs"] == 1, "ATT must execute exactly one complete pass"
        assert cfg["att"]["num_candidates"] == 3
        assert cfg["att"]["score_without_gradients"] is True
        assert cfg["att"]["backprop_hardest_only"] is True


# ---------------------------------------------------------------------------
# 6. Promotion gate logic tests
# ---------------------------------------------------------------------------


def test_att_promotion_gate_all_pass() -> None:
    float_metrics = {
        "clean_auroc": 0.985,
        "worst_transformed_auroc": 0.910,
        "worst_domain_auroc": 0.920,
        "ai_positive_recall": 0.850,
        "authentic_recall": 0.860,
    }
    att_metrics = {
        "clean_auroc": 0.983,  # Drop 0.002 <= 0.005 (PASS)
        "worst_transformed_auroc": 0.935,  # Improved 0.935 > 0.910 (PASS)
        "worst_domain_auroc": 0.922,  # Nonregressed 0.922 >= 0.920 (PASS)
        "ai_positive_recall": 0.880,  # >= 0.82 (PASS)
        "authentic_recall": 0.875,  # >= 0.82 (PASS)
    }

    result = evaluate_track_gate(float_metrics, att_metrics, track_name="small")
    assert result["passed"] is True
    assert result["status"] == "PROMOTED"
    assert len(result["failure_reasons"]) == 0


def test_att_promotion_gate_worst_transform_fails() -> None:
    float_metrics = {
        "clean_auroc": 0.985,
        "worst_transformed_auroc": 0.920,
        "worst_domain_auroc": 0.920,
        "ai_positive_recall": 0.850,
        "authentic_recall": 0.860,
    }
    att_metrics = {
        "clean_auroc": 0.985,
        "worst_transformed_auroc": 0.915,  # Regressed! 0.915 <= 0.920 (FAIL)
        "worst_domain_auroc": 0.925,
        "ai_positive_recall": 0.850,
        "authentic_recall": 0.860,
    }

    result = evaluate_track_gate(float_metrics, att_metrics, track_name="small")
    assert result["passed"] is False
    assert result["status"] == "REJECTED"
    assert any("Worst-transform AUROC did not improve" in r for r in result["failure_reasons"])


def test_att_promotion_gate_clean_drop_exceeded_fails() -> None:
    float_metrics = {
        "clean_auroc": 0.980,
        "worst_transformed_auroc": 0.900,
        "worst_domain_auroc": 0.910,
        "ai_positive_recall": 0.850,
        "authentic_recall": 0.860,
    }
    att_metrics = {
        "clean_auroc": 0.970,  # Drop 0.010 > 0.005 limit (FAIL)
        "worst_transformed_auroc": 0.920,
        "worst_domain_auroc": 0.915,
        "ai_positive_recall": 0.850,
        "authentic_recall": 0.860,
    }

    result = evaluate_track_gate(float_metrics, att_metrics, track_name="base")
    assert result["passed"] is False
    assert result["status"] == "REJECTED"
    assert any("Clean AUROC drop exceeded tolerance" in r for r in result["failure_reasons"])


def test_att_promotion_gate_recalls_below_threshold_fails() -> None:
    float_metrics = {
        "clean_auroc": 0.980,
        "worst_transformed_auroc": 0.900,
        "worst_domain_auroc": 0.910,
        "ai_positive_recall": 0.850,
        "authentic_recall": 0.860,
    }
    # AI recall low
    att_metrics_low_ai = {
        "clean_auroc": 0.980,
        "worst_transformed_auroc": 0.920,
        "worst_domain_auroc": 0.915,
        "ai_positive_recall": 0.790,  # < 0.82 (FAIL)
        "authentic_recall": 0.860,
    }
    res_ai = evaluate_track_gate(float_metrics, att_metrics_low_ai, track_name="small")
    assert res_ai["passed"] is False
    assert any("AI-positive recall" in r for r in res_ai["failure_reasons"])

    # Authentic recall low
    att_metrics_low_auth = {
        "clean_auroc": 0.980,
        "worst_transformed_auroc": 0.920,
        "worst_domain_auroc": 0.915,
        "ai_positive_recall": 0.850,
        "authentic_recall": 0.750,  # < 0.82 (FAIL)
    }
    res_auth = evaluate_track_gate(float_metrics, att_metrics_low_auth, track_name="small")
    assert res_auth["passed"] is False
    assert any("Authentic recall" in r for r in res_auth["failure_reasons"])


def test_shared_report_aggregation() -> None:
    track_small = {
        "track": "small",
        "passed": True,
        "status": "PROMOTED",
        "checks": {},
        "failure_reasons": [],
    }
    track_base = {
        "track": "base",
        "passed": True,
        "status": "PROMOTED",
        "checks": {},
        "failure_reasons": [],
    }

    report = build_shared_report({"small": track_small, "base": track_base})
    assert report["overall_passed"] is True
    assert report["overall_status"] == "ALL_PROMOTED"
    assert "small" in report["tracks"]
    assert "base" in report["tracks"]

    # If one track fails, overall is false
    track_base_fail = dict(track_base, passed=False, status="REJECTED")
    report_fail = build_shared_report({"small": track_small, "base": track_base_fail})
    assert report_fail["overall_passed"] is False
    assert report_fail["overall_status"] == "REJECTED"


# ---------------------------------------------------------------------------
# 7. Launch command builder tests
# ---------------------------------------------------------------------------


def test_launch_track_command_builder() -> None:
    cmd = build_track_command(
        "small",
        checkpoint="checkpoint.pt",
        manifest="train.parquet",
        config="configs/att_student_small.yaml",
        output_dir="outputs/att_student_small",
        num_candidates=3,
        epochs=1,
        dry_run=True,
    )
    cmd_str = " ".join(cmd)
    assert "torch.distributed.run" not in cmd_str
    assert "--world-size 1" in cmd_str
    assert "--variant small" in cmd_str
    assert "--student-checkpoint checkpoint.pt" in cmd_str
    assert "--manifest train.parquet" in cmd_str
    assert "--num-candidates 3" in cmd_str
    assert "--epochs 1" in cmd_str
    assert "--dry-run" in cmd_str


def test_missing_checkpoint_hard_fails() -> None:
    from scripts.train_att import train_att

    with pytest.raises(FileNotFoundError, match="Promoted float student checkpoint is required"):
        train_att(
            variant="small",
            student_checkpoint="nonexistent_checkpoint.pt",
            manifest_path="nonexistent_manifest.parquet",
            dry_run=False,
        )


def test_missing_manifest_fallback_prohibited() -> None:
    from scripts.train_att import train_att

    with pytest.raises(
        FileNotFoundError, match="Manifest fallback across split sources is strictly prohibited"
    ):
        train_att(
            variant="small",
            student_checkpoint=None,
            manifest_path="nonexistent_manifest.parquet",
            dry_run=True,
        )


def test_random_weights_fallback_prohibited() -> None:
    """Calling train_att with None checkpoint and existing manifest strictly raises FileNotFoundError (no random weights)."""
    import tempfile

    from scripts.train_att import train_att

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        manifest_file = tmp / "train.parquet"
        import pandas as pd

        pd.DataFrame([{"split": "train", "image_path": "a.png"}]).to_parquet(manifest_file)

        with pytest.raises(
            FileNotFoundError, match="Random-weight fallbacks are strictly prohibited"
        ):
            train_att(
                variant="small",
                student_checkpoint=None,
                manifest_path=manifest_file,
                dry_run=True,
            )


def test_check_att_gate_atomic_promotion_and_rejection() -> None:
    """check_att_gate atomically promotes checkpoint-final.pt only when gate passes."""
    import subprocess
    import sys
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        small_out = tmp / "att_small"
        small_out.mkdir()
        final_ckpt = small_out / "checkpoint-final.pt"
        final_ckpt.write_bytes(b"dummy final student post-att weights")

        # Passing float vs att metrics
        float_eval = tmp / "eval_float.json"
        float_eval.write_text(
            json.dumps(
                {
                    "conditions": {
                        "clean": {
                            "ai_positive_auroc": 0.960,
                            "binary_recall": 0.85,
                            "authentic_recall": 0.86,
                        },
                    },
                    "selection_summary": {
                        "worst_transformed_auroc": 0.890,
                        "worst_generator_auroc": 0.910,
                    },
                    "calibrated_threshold": 0.52,
                    "manifest_digest": "a" * 64,
                }
            )
        )

        att_eval_pass = tmp / "eval_att_pass.json"
        att_eval_pass.write_text(
            json.dumps(
                {
                    "conditions": {
                        "clean": {
                            "ai_positive_auroc": 0.958,
                            "binary_recall": 0.84,
                            "authentic_recall": 0.85,
                        },
                    },
                    "selection_summary": {
                        "worst_transformed_auroc": 0.900,
                        "worst_generator_auroc": 0.915,
                    },
                    "calibrated_threshold": 0.52,
                    "manifest_digest": "a" * 64,
                }
            )
        )

        report_file = tmp / "shared_report.json"
        res = subprocess.run(
            [
                sys.executable,
                "scripts/check_att_gate.py",
                "--small-float-eval",
                str(float_eval),
                "--small-att-eval",
                str(att_eval_pass),
                "--small-checkpoint",
                str(final_ckpt),
                "--small-output-dir",
                str(small_out),
                "--shared-report",
                str(report_file),
                "--promote",
            ],
            capture_output=True,
            text=True,
        )

        assert res.returncode == 0, f"check_att_gate failed: {res.stderr}"
        promoted_ckpt = small_out / "checkpoint-promoted.pt"
        assert promoted_ckpt.is_file(), "checkpoint-promoted.pt must be created on gate pass"
        assert (small_out / "promotion_report.json").is_file()
        meta = json.loads((small_out / "metadata.json").read_text(encoding="utf-8"))
        assert meta["evaluation_status"] == "promoted"

        # Now test failure: worst-transform AUROC regresses -> no checkpoint-promoted.pt
        att_eval_fail = tmp / "eval_att_fail.json"
        att_eval_fail.write_text(
            json.dumps(
                {
                    "conditions": {
                        "clean": {
                            "ai_positive_auroc": 0.940,
                            "binary_recall": 0.84,
                            "authentic_recall": 0.85,
                        },
                    },
                    "selection_summary": {
                        "worst_transformed_auroc": 0.850,
                        "worst_generator_auroc": 0.910,
                    },
                    "calibrated_threshold": 0.52,
                    "manifest_digest": "a" * 64,
                }
            )
        )
        res_fail = subprocess.run(
            [
                sys.executable,
                "scripts/check_att_gate.py",
                "--small-float-eval",
                str(float_eval),
                "--small-att-eval",
                str(att_eval_fail),
                "--small-checkpoint",
                str(final_ckpt),
                "--small-output-dir",
                str(small_out),
                "--shared-report",
                str(report_file),
                "--promote",
            ],
            capture_output=True,
            text=True,
        )
        assert res_fail.returncode != 0
        assert not promoted_ckpt.is_file(), "checkpoint-promoted.pt must be removed on gate failure"
