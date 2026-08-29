from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
import torch
import yaml
from PIL import Image

from aigc_detector.constants import Transformation
from aigc_detector.dataset import PairedImageDataset
from aigc_detector.sampling import build_coverage_sampler
from aigc_detector.train import (
    restore_checkpoint,
    save_checkpoint,
)

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"


def _load_yaml(filename: str) -> dict:
    path = CONFIG_ROOT / filename
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_identical_stage1_stage2_eligible_manifest_and_ids(tmp_path: Path) -> None:
    stage1_cfg = _load_yaml("teacher_dinov3_stage1_clean_frozen.yaml")
    stage2_cfg = _load_yaml("teacher_dinov3_stage2_paired_unfrozen.yaml")

    # Both stages must target the exact same canonical eligible training and validation manifests
    assert stage1_cfg["paths"]["train_manifest"] == stage2_cfg["paths"]["train_manifest"]
    assert stage1_cfg["paths"]["train_manifest"] == "splits/production_eligible/train.parquet"
    assert stage1_cfg["paths"]["val_manifest"] == stage2_cfg["paths"]["val_manifest"]
    assert stage1_cfg["paths"]["val_manifest"] == "splits/production_eligible/validation.parquet"

    # Create dummy images and manifest
    img1 = tmp_path / "img1.png"
    img2 = tmp_path / "img2.png"
    Image.new("RGB", (32, 32), color=(10, 20, 30)).save(img1)
    Image.new("RGB", (32, 32), color=(40, 50, 60)).save(img2)

    manifest_path = tmp_path / "train.parquet"
    df = pd.DataFrame(
        [
            {
                "row_id": "row_001",
                "split": "train",
                "image_path": "img1.png",
                "label": 0,
                "ai_positive": 0,
                "dataset": "test_ds",
                "generator": "natural",
                "generator_family": "camera",
                "duplicate_group": "dup_001",
                "source_image_group": "src_001",
            },
            {
                "row_id": "row_002",
                "split": "train",
                "image_path": "img2.png",
                "label": 1,
                "ai_positive": 1,
                "dataset": "test_ds",
                "generator": "midjourney_v6",
                "generator_family": "midjourney",
                "duplicate_group": "dup_002",
                "source_image_group": "src_002",
            },
        ]
    )
    df.to_parquet(manifest_path, index=False)

    ds_stage1 = PairedImageDataset(
        manifest_path,
        data_root=tmp_path,
        expected_split="train",
        transform_families=(),
    )
    ds_stage2 = PairedImageDataset(
        manifest_path,
        data_root=tmp_path,
        expected_split="train",
        transform_families=(Transformation.JPEG, Transformation.BLUR),
    )

    assert len(ds_stage1) == len(ds_stage2) == 2
    stage1_ids = [r.row_id for r in ds_stage1.records]
    stage2_ids = [r.row_id for r in ds_stage2.records]
    assert stage1_ids == stage2_ids == ["row_001", "row_002"]

    # Verify split membership enforcement: expected_split="train" rejects "validation" rows
    df_invalid = df.copy()
    df_invalid.loc[0, "split"] = "validation"
    invalid_manifest = tmp_path / "invalid.parquet"
    df_invalid.to_parquet(invalid_manifest, index=False)

    with pytest.raises(ValueError, match="expected split 'train'"):
        PairedImageDataset(
            invalid_manifest,
            data_root=tmp_path,
            expected_split="train",
        )


def test_missing_asset_raises_without_fallback_or_fetch(tmp_path: Path) -> None:
    manifest_path = tmp_path / "train.parquet"
    df = pd.DataFrame(
        [
            {
                "row_id": "row_missing",
                "split": "train",
                "image_path": "non_existent_image.png",
                "label": 0,
                "ai_positive": 0,
                "dataset": "test_ds",
                "generator": "natural",
            }
        ]
    )
    df.to_parquet(manifest_path, index=False)

    dataset = PairedImageDataset(
        manifest_path,
        data_root=tmp_path,
        allow_missing=False,
        runtime_fetch=False,
    )

    with pytest.raises(FileNotFoundError, match="Missing image asset for row row_missing"):
        _ = dataset[0]


def test_deterministic_coverage_sampler_full_coverage_and_equal_rank_steps() -> None:
    dataset_size = 73751
    world_size = 4
    seed = 42

    samplers = [
        build_coverage_sampler(
            dataset_size,
            seed=seed,
            rank=r,
            world_size=world_size,
            epoch=0,
        )
        for r in range(world_size)
    ]

    # 1. Equal rank steps: all ranks receive identical sample counts
    expected_samples_per_rank = math.ceil(dataset_size / world_size)
    assert expected_samples_per_rank == 18438

    for sampler in samplers:
        assert len(sampler) == expected_samples_per_rank
        assert sampler.samples_per_rank == expected_samples_per_rank
    # 2. Collect draws across all ranks for epoch 0
    all_draws: list[int] = []
    per_rank_indices: list[list[int]] = []
    for sampler in samplers:
        rank_indices = list(sampler)
        assert len(rank_indices) == expected_samples_per_rank
        per_rank_indices.append(rank_indices)
        all_draws.extend(rank_indices)

    # 3. Full coverage: every unique dataset index is covered
    unique_drawn = set(all_draws)
    assert len(unique_drawn) == dataset_size
    assert unique_drawn == set(range(dataset_size))

    # 4. Minimal padding: total draws exceeds dataset size by strictly less than world_size
    total_draws = len(all_draws)
    num_padding = total_draws - dataset_size
    assert 0 <= num_padding < world_size
    assert num_padding == (expected_samples_per_rank * world_size) - dataset_size

    # 5. Coverage report
    report = samplers[0].get_coverage_report()
    assert report["dataset_size"] == dataset_size
    assert report["unique_rows_covered"] == dataset_size
    assert report["samples_per_rank"] == expected_samples_per_rank
    assert report["total_draws"] == total_draws
    assert report["num_padding"] == num_padding
    assert report["padded_repeats"] == num_padding
    assert report["missing_rows"] == 0
    assert report["world_size"] == world_size

    # 6. Resume continuity: setting start_offset resumes with exact suffix
    resume_offset = 1000
    resumed_sampler = build_coverage_sampler(
        dataset_size,
        seed=seed,
        rank=0,
        world_size=world_size,
        epoch=0,
        start_offset=resume_offset,
    )
    assert len(resumed_sampler) == expected_samples_per_rank - resume_offset
    resumed_indices = list(resumed_sampler)
    assert resumed_indices == per_rank_indices[0][resume_offset:]


def test_save_and_restore_checkpoint_metadata_and_digest_mismatch(tmp_path: Path) -> None:
    dummy_model = MagicMock()
    dummy_model.heads.state_dict.return_value = {"weight": torch.zeros(2, 2)}
    dummy_model.token_adapter.state_dict.return_value = {}
    dummy_model.spectral = None
    dummy_model.backbone.parameters.return_value = []

    dummy_opt = MagicMock()
    dummy_opt.state_dict.return_value = {"state": {}}

    dummy_sched = MagicMock()
    dummy_sched.state_dict.return_value = {"last_epoch": 1}

    manifest_file = tmp_path / "train.parquet"
    manifest_file.write_bytes(b"canonical manifest content")

    config = {"training": {"stage": "teacher_stage1_clean_frozen"}}
    cov_meta = {"coverage_pct": 50, "unique_rows": 1000}
    batch_meta = {
        "physical_batch_size": 6,
        "world_size": 4,
        "gradient_accumulation": 2,
        "effective_batch_size": 48,
    }
    ckpt_path = save_checkpoint(
        dummy_model,
        config,
        tmp_path,
        step=100,
        optimizer=dummy_opt,
        scheduler=dummy_sched,
        manifest_path=manifest_file,
        coverage_metadata=cov_meta,
        batch_metadata=batch_meta,
    )

    # Verify saved checkpoint payload contains all required metadata
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert payload["step"] == 100
    assert "git" in payload
    assert "commit" in payload["git"]
    assert "parameter_count" in payload
    assert "manifest_sha256" in payload
    assert "manifest_digest" in payload
    assert payload["manifest_sha256"] == payload["manifest_digest"]
    assert payload["coverage"] == cov_meta
    assert payload["batch_metadata"] == batch_meta

    # Restore with matching manifest succeeds
    step, epoch, micro_step = restore_checkpoint(
        ckpt_path,
        dummy_model,
        dummy_opt,
        dummy_sched,
        ema=None,
        manifest_path=manifest_file,
        device=torch.device("cpu"),
    )
    assert step == 100

    # Restore with different manifest raises RuntimeError
    other_manifest = tmp_path / "tampered_train.parquet"
    other_manifest.write_bytes(b"different manifest content")
    with pytest.raises(RuntimeError, match="different training manifest"):
        restore_checkpoint(
            ckpt_path,
            dummy_model,
            dummy_opt,
            dummy_sched,
            ema=None,
            manifest_path=other_manifest,
            device=torch.device("cpu"),
        )


def test_four_gpu_geometry_and_derived_updates() -> None:
    stage1 = _load_yaml("teacher_dinov3_stage1_clean_frozen.yaml")
    stage2 = _load_yaml("teacher_dinov3_stage2_paired_unfrozen.yaml")

    # Geometry: Stage 1 = 6 x 4 x 2 = 48; Stage 2 = 2 x 4 x 6 = 48
    s1_phys = stage1["training"]["physical_batch_size"]
    s1_world = stage1["training"]["required_world_size"]
    s1_accum = stage1["training"]["gradient_accumulation"]
    assert s1_phys * s1_world * s1_accum == 48
    assert (s1_phys, s1_world, s1_accum) == (6, 4, 2)

    s2_phys = stage2["training"]["physical_batch_size"]
    s2_world = stage2["training"]["required_world_size"]
    s2_accum = stage2["training"]["gradient_accumulation"]
    assert s2_phys * s2_world * s2_accum == 48
    assert (s2_phys, s2_world, s2_accum) == (2, 4, 6)

    # Derived updates: one pass over 73,751 rows
    dataset_rows = 73751
    effective_batch = 48
    expected_updates = math.ceil(dataset_rows / effective_batch)
    assert expected_updates == 1537

    # Stage 1: frozen backbone, no transforms
    assert stage1["model"]["freeze_encoder"] is True
    assert stage1["transforms"]["families"] == []

    # Stage 2: unfrozen backbone, 6 single-transform families
    assert stage2["model"]["freeze_encoder"] is False
    assert stage2["transforms"]["families"] == ["jpeg", "blur", "resize", "noise", "color", "crop"]

    # Coverage checkpoints at 25%, 50%, 75%, 100%
    assert stage1["training"]["coverage_checkpoints"] == [0.25, 0.50, 0.75, 1.00]
    assert stage2["training"]["coverage_checkpoints"] == [0.25, 0.50, 0.75, 1.00]

    # Rank 0 validation stalls avoided
    assert stage1["training"]["validation_interval"] == 0
    assert stage2["training"]["validation_interval"] == 0
