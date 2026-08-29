from pathlib import Path

import yaml

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"


def _load(name: str) -> dict:
    return yaml.safe_load((CONFIG_ROOT / name).read_text(encoding="utf-8"))


def test_teacher_stage1_freezes_dinov3_and_uses_only_untransformed_images() -> None:
    config = _load("teacher_dinov3_stage1_clean_frozen.yaml")

    assert config["model"]["backbone_type"] == "dinov3"
    assert config["model"]["freeze_encoder"] is True
    assert config["model"]["trainable_last_layers"] == 0
    assert config["transforms"]["families"] == []
    assert config["loss"]["provenance_transformed"] == 0.0
    assert config["loss"]["prediction_consistency"] == 0.0
    assert config["loss"]["feature_consistency"] == 0.0


def test_teacher_stage2_unfreezes_all_dinov3_and_uses_single_transform_pairs() -> None:
    config = _load("teacher_dinov3_stage2_paired_unfrozen.yaml")

    assert config["model"]["backbone_type"] == "dinov3"
    assert config["model"]["freeze_encoder"] is False
    assert config["model"]["trainable_last_layers"] == 0
    assert config["transforms"]["families"] == ["jpeg", "blur", "resize", "noise", "color", "crop"]
    assert config["loss"]["provenance_transformed"] > 0.0
    assert config["loss"]["prediction_consistency"] > 0.0
    assert config["loss"]["feature_consistency"] > 0.0


def test_teacher_configs_preserve_effective_batch_across_six_gpus() -> None:
    stage1 = _load("teacher_dinov3_stage1_clean_frozen.yaml")
    stage2 = _load("teacher_dinov3_stage2_paired_unfrozen.yaml")

    for config in (stage1, stage2):
        assert config["model"]["encoder_dtype"] == "float32"
        assert config["paths"]["train_manifest"] == "splits/combined_hf_dataset/train.parquet"
        assert config["paths"]["val_manifest"] == "splits/combined_hf_dataset/validation.parquet"
        assert config["paths"]["require_materialized"] is True
        assert config["training"]["required_world_size"] == 6
        assert config["training"]["checkpoint_interval"] > 0

    assert stage1["training"]["max_updates"] == 600
    assert stage2["training"]["max_updates"] == 1250

    assert stage1["training"]["optimizer"] == "adamw"
    assert stage2["training"]["optimizer"] == "adamw"
    assert stage2["training"]["ema"]["enabled"] is True
    assert stage1["training"]["validation_interval"] > 0
    assert stage2["training"]["validation_interval"] == 0
    assert stage2["loss"]["ema_consistency"] > 0.0

    assert stage1["training"]["physical_batch_size"] * stage1["training"][
        "gradient_accumulation"
    ] * 6 == 48
    assert stage2["training"]["physical_batch_size"] * stage2["training"][
        "gradient_accumulation"
    ] * 6 == 48


def test_checkpoint2_uses_full_pool_and_decayed_adaptation_budget() -> None:
    config = _load("teacher_dinov3_checkpoint2_full_data.yaml")

    assert config["paths"]["train_manifest"] == "splits/checkpoint2_materialized/all.parquet"
    assert config["paths"]["require_materialized"] is True
    assert config["paths"]["initial_checkpoint"].endswith(
        "/teacher_stage2_paired_unfrozen/checkpoint-best.pt"
    )
    assert config["training"]["stage"] == "teacher_checkpoint2_production_full_data"
    assert config["training"]["max_updates"] == 250
    assert config["training"]["heads_lr"] == 3.0e-5
    assert config["training"]["encoder_lr"] == 1.0e-6
    assert config["training"]["physical_batch_size"] * config["training"][
        "gradient_accumulation"
    ] * config["training"]["required_world_size"] == 48
    assert config["training"]["generator_balanced_sampler"] is True
