#!/usr/bin/env python3
"""Adversarial Transformation Training (ATT) for student models.

Performs independent ATT on promoted float student checkpoints (ViT-S or ViT-B):
1. Every train record retains downloaded-original supervised loss.
2. Deterministically generates multiple allowed single-transform candidates
   across jpeg, blur, resize, noise, color, and crop.
3. Scores candidates under torch.no_grad() (zero gradient storage).
4. Selects the hardest candidate (highest loss) per sample.
5. Backpropagates through only the selected hardest candidate + original.
6. Executes exactly one complete coverage pass over the train split only.
7. Disjoint GPU pools and master ports for track isolation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import os
from collections.abc import Sequence
from pathlib import Path
from random import Random
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

from aigc_detector.config import load_config
from aigc_detector.constants import Provenance, Transformation
from aigc_detector.dataset import load_manifest_frame, parse_provenance
from aigc_detector.model import ProvenanceOutput
from aigc_detector.predict import _load_checkpoint
from aigc_detector.preprocessing import RenderPolicy, render_for_model, render_mask_geometry
from aigc_detector.runtime import (
    cleanup_distributed,
    is_main_process,
    load_local_environment,
    setup_distributed,
)
from aigc_detector.sampling import DeterministicDistributedCoverageSampler
from aigc_detector.train import (
    build_model,
    build_optimizer,
    build_scheduler,
    restore_checkpoint,
    save_checkpoint,
)
from aigc_detector.transforms import TransformSpec, apply_transform

logger = logging.getLogger("train_att")

# Organizer-aligned candidate transformation families
ATT_TRANSFORMATION_FAMILIES = (
    Transformation.JPEG,
    Transformation.BLUR,
    Transformation.RESIZE,
    Transformation.NOISE,
    Transformation.COLOR,
    Transformation.CROP,
)

ATT_SEVERITY_VALUES: dict[Transformation, tuple[float, ...]] = {
    Transformation.JPEG: (90.0, 70.0, 50.0, 30.0),
    Transformation.BLUR: (0.5, 1.0, 2.0),
    Transformation.RESIZE: (0.5, 0.25),
    Transformation.NOISE: (0.02, 0.05, 0.10),
    Transformation.COLOR: (0.10, 0.20, 0.30),
    Transformation.CROP: (0.70, 0.80, 0.90),
}


def get_deterministic_row_seed(
    base_seed: int, epoch: int, row_id: str | int, candidate_idx: int = 0
) -> int:
    """Derive a stable 32-bit integer seed for a given row and candidate index."""
    payload = f"{base_seed}_{epoch}_{row_id}_{candidate_idx}".encode()
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:8], 16)


def generate_att_candidates(
    image: Image.Image,
    num_candidates: int,
    *,
    row_id: str | int,
    base_seed: int,
    epoch: int = 0,
    mask: Image.Image | None = None,
    allowed_families: Sequence[Transformation] = ATT_TRANSFORMATION_FAMILIES,
    severity_map: dict[Transformation, tuple[float, ...]] = ATT_SEVERITY_VALUES,
) -> list[tuple[Image.Image, TransformSpec]]:
    """Deterministically generate multiple single-transform candidates for one image.

    Each candidate applies exactly ONE content-preserving transformation to the
    downloaded-original image. No multi-transform chains are introduced.
    """
    candidates: list[tuple[Image.Image, TransformSpec]] = []
    families_list = list(allowed_families)

    # Seed an RNG deterministically for this row
    row_seed = get_deterministic_row_seed(base_seed, epoch, row_id)
    rng = Random(row_seed)

    # Sample distinct families when possible to maximize candidate diversity
    k = min(num_candidates, len(families_list))
    sampled_families = rng.sample(families_list, k=k)
    # If more candidates requested than families, sample extra with replacement
    if num_candidates > k:
        sampled_families.extend(rng.choices(families_list, k=num_candidates - k))

    for idx, family in enumerate(sampled_families):
        cand_rng = Random(get_deterministic_row_seed(base_seed, epoch, row_id, idx + 1))
        severities = severity_map.get(family, (0.5,))
        severity = cand_rng.choice(severities)
        spec = TransformSpec(family=family, severity=float(severity))
        transformed_img = apply_transform(image, spec, cand_rng, mask=mask)
        candidates.append((transformed_img, spec))

    return candidates


def verify_train_split_membership(df: pd.DataFrame) -> pd.DataFrame:
    """Verify and enforce that only 'train' split records are processed.

    Fails closed if non-train records (validation, test, test_unseen, calibration)
    are present in the training set to prevent data leakage.
    """
    if "split" not in df.columns:
        return df

    unique_splits = set(df["split"].dropna().astype(str).str.lower().unique())
    invalid_splits = unique_splits - {"train"}
    if invalid_splits:
        raise ValueError(
            f"ATT Membership Guard Violation: Found records with non-train splits: {sorted(invalid_splits)}. "
            f"Validation, test, test_unseen, and calibration data must never be mined or trained on during ATT."
        )

    return df[df["split"].astype(str).str.lower() == "train"].copy()


class ATTRawDataset(Dataset[dict[str, Any]]):
    """Dataset delivering raw downloaded originals and metadata for ATT candidate mining."""

    def __init__(
        self,
        manifest_path: Path | str,
        *,
        data_root: Path | str | None = None,
        seed: int = 42,
        render_policy: str | RenderPolicy = "square_jpeg95",
        num_candidates: int = 3,
        epoch: int = 0,
        require_materialized: bool = False,
    ) -> None:
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.data_root = Path(data_root) if data_root else None
        self.seed = seed
        self.render_policy = render_policy
        self.num_candidates = num_candidates
        self.epoch = epoch

        df = load_manifest_frame(self.manifest_path)
        df = verify_train_split_membership(df)
        self.records = df.to_dict(orient="records")

        if require_materialized:
            missing = [
                r["image_path"]
                for r in self.records
                if not self._resolve_path(r["image_path"]).is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    f"Manifest has {len(missing)} missing files; require_materialized=True"
                )

    def _resolve_path(self, path_str: str) -> Path:
        p = Path(path_str)
        if p.is_absolute():
            return p
        if self.data_root:
            return self.data_root / p
        return p

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        row_id = str(record.get("row_id", index))
        image_path = self._resolve_path(record["image_path"])

        if not image_path.is_file():
            raise FileNotFoundError(f"Missing required image file: {image_path}")

        with Image.open(image_path) as source:
            raw = source.convert("RGB").copy()

        rng = Random(self.seed + self.epoch * 100_000 + index)
        original = render_for_model(raw, self.render_policy, rng=rng)

        # Optional mask rendering
        mask_img = None
        tamper_mask_val = record.get("tamper_mask_path")
        if tamper_mask_val and pd.notna(tamper_mask_val):
            mask_path = self._resolve_path(str(tamper_mask_val))
            if mask_path.is_file():
                with Image.open(mask_path) as m_src:
                    raw_mask = m_src.convert("L").copy()
                mask_img = render_mask_geometry(
                    raw_mask,
                    self.render_policy,
                    image_size=raw.size,
                    rendered_size=original.size,
                )

        candidates = generate_att_candidates(
            original,
            self.num_candidates,
            row_id=row_id,
            base_seed=self.seed,
            epoch=self.epoch,
            mask=mask_img,
        )

        provenance = parse_provenance(record.get("provenance", 0), str(record.get("dataset", "")))
        ai_positive = float(record.get("ai_positive", int(provenance != Provenance.AUTHENTIC)))

        return {
            "row_id": row_id,
            "original": original,
            "candidates": candidates,  # list of (Image.Image, TransformSpec)
            "provenance": int(provenance),
            "ai_positive": ai_positive,
            "dataset": str(record.get("dataset", "")),
            "generator": str(record.get("generator", "")),
        }


def collate_att_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "row_ids": [b["row_id"] for b in batch],
        "original_images": [b["original"] for b in batch],
        "candidates": [b["candidates"] for b in batch],  # list of lists
        "provenance": torch.tensor([b["provenance"] for b in batch], dtype=torch.long),
        "ai_positive": torch.tensor([b["ai_positive"] for b in batch], dtype=torch.float32),
        "datasets": [b["dataset"] for b in batch],
        "generators": [b["generator"] for b in batch],
    }


def compute_sample_loss(
    ai_positive_logit: torch.Tensor,
    ai_positive_target: torch.Tensor,
) -> torch.Tensor:
    """Compute per-sample binary AI-provenance loss."""
    return F.binary_cross_entropy_with_logits(
        ai_positive_logit.view(-1), ai_positive_target.float().view(-1), reduction="none"
    )


def score_and_select_hardest(
    student: nn.Module,
    candidates_batch: list[list[tuple[Image.Image, TransformSpec]]],
    ai_positive: torch.Tensor,
) -> tuple[list[Image.Image], list[TransformSpec], list[int]]:
    """Score every candidate in one no-grad forward, then select per-row maxima."""
    counts = [len(item_candidates) for item_candidates in candidates_batch]
    if not counts or any(count == 0 for count in counts):
        raise ValueError("Every ATT record must provide at least one transformation candidate")

    flat_images = [
        image for item_candidates in candidates_batch for image, _spec in item_candidates
    ]
    flat_specs = [spec for item_candidates in candidates_batch for _image, spec in item_candidates]
    repeated_targets = torch.repeat_interleave(
        ai_positive,
        torch.tensor(counts, device=ai_positive.device),
    )

    with torch.no_grad():
        output: ProvenanceOutput = student(flat_images)
        flat_losses = compute_sample_loss(output.ai_positive_logit, repeated_targets)

    hardest_images: list[Image.Image] = []
    hardest_specs: list[TransformSpec] = []
    hardest_indices: list[int] = []
    offset = 0
    for count in counts:
        item_losses = flat_losses[offset : offset + count]
        hardest_idx = int(torch.argmax(item_losses).item())
        hardest_images.append(flat_images[offset + hardest_idx])
        hardest_specs.append(flat_specs[offset + hardest_idx])
        hardest_indices.append(hardest_idx)
        offset += count

    return hardest_images, hardest_specs, hardest_indices


def student_att_config(base_config: dict, variant: str) -> dict:
    """Build the canonical single-GPU ATT configuration."""
    config = copy.deepcopy(base_config)
    presets = {
        "small": {
            "encoder_id": "facebook/dinov3-vits16-pretrain-lvd1689m",
            "encoder_dim": 384,
            "master_port": 29503,
            "cuda_visible_devices": "0",
            "output_root": "outputs/att_student_small",
            "physical_batch_size": 4,
            "gradient_accumulation": 12,
        },
        "base": {
            "encoder_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",
            "encoder_dim": 768,
            "master_port": 29504,
            "cuda_visible_devices": "0",
            "output_root": "outputs/att_student_base",
            "physical_batch_size": 2,
            "gradient_accumulation": 24,
        },
    }
    spec = presets[variant]
    batch_sz = spec["physical_batch_size"]
    accum_sz = spec["gradient_accumulation"]
    config.setdefault("model", {}).update(
        {
            "backbone_type": "dinov3",
            "encoder_id": spec["encoder_id"],
            "encoder_revision": None,
            "encoder_dim": spec["encoder_dim"],
            "image_size": 224,
            "freeze_encoder": False,
            "trainable_last_layers": 0,
            "gradient_checkpointing": True,
            "spectral_expert": False,
            "use_token_adapter": True,
            "trunk_dim": 512,
        }
    )
    config.setdefault("training", {}).update(
        {
            "stage": f"att_student_{variant}",
            "physical_batch_size": batch_sz,
            "gradient_accumulation": accum_sz,
            "required_world_size": 1,
            "num_workers": 4,
            "optimizer": "adamw",
            "precision": "bf16",
            "encoder_lr": 1.0e-5,
            "heads_lr": 5.0e-5,
            "layerwise_lr_decay": 0.85,
            "weight_decay": 0.01,
            "gradient_clip_norm": 1.0,
            "warmup_fraction": 0.05,
            "master_port": spec["master_port"],
            "cuda_visible_devices": spec["cuda_visible_devices"],
        }
    )
    config.setdefault("loss", {}).update(
        {
            "provenance_original": 1.0,
            "provenance_transformed": 1.0,
            "prediction_consistency": 0.5,
            "feature_consistency": 0.0,
        }
    )
    config.setdefault("att", {}).update(
        {
            "num_candidates": 3,
            "score_without_gradients": True,
            "backprop_hardest_only": True,
        }
    )
    return config


def train_att(
    variant: str,
    *,
    student_checkpoint: Path | str | None = None,
    config_path: Path | str | None = None,
    manifest_path: Path | str | None = None,
    output_dir: Path | str | None = None,
    num_candidates: int = 3,
    epochs: int = 1,
    batch_size: int | None = None,
    gradient_accumulation: int | None = None,
    world_size_override: int | None = None,
    learning_rate: float | None = None,
    seed: int = 42,
    resume_checkpoint: Path | str | None = None,
    dry_run: bool = False,
) -> Path:
    """Execute one complete pass of Adversarial Transformation Training."""
    if variant not in {"small", "base"}:
        raise ValueError(f"Unknown student variant {variant!r}; choose 'small' or 'base'")

    project_root = Path(__file__).resolve().parent.parent
    load_local_environment(project_root)
    if "TECHJAM_DATA_ROOT" not in os.environ:
        os.environ["TECHJAM_DATA_ROOT"] = str(project_root / "data")
    if "TECHJAM_OUTPUT_ROOT" not in os.environ:
        os.environ["TECHJAM_OUTPUT_ROOT"] = str(project_root / "outputs")

    if config_path and Path(config_path).is_file():
        cfg = load_config(Path(config_path))
    else:
        default_cfg_file = Path(f"configs/att_student_{variant}.yaml")
        if default_cfg_file.is_file():
            cfg = load_config(default_cfg_file)
        else:
            cfg = student_att_config({}, variant)

    if batch_size is not None:
        cfg["training"]["physical_batch_size"] = batch_size
    if gradient_accumulation is not None:
        cfg["training"]["gradient_accumulation"] = gradient_accumulation
    if world_size_override is not None:
        if world_size_override <= 0:
            raise ValueError("world_size_override must be positive")
        cfg["training"]["required_world_size"] = world_size_override
    if learning_rate is not None:
        cfg["training"]["heads_lr"] = learning_rate
        cfg["training"]["encoder_lr"] = learning_rate * 0.2

    if output_dir:
        out_dir = Path(output_dir)
    else:
        out_dir = Path(cfg.get("paths", {}).get("output_root", f"outputs/att_student_{variant}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if resume_checkpoint is not None and not Path(resume_checkpoint).is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint}")

    # If student_checkpoint is specified but does not exist, fail immediately
    if student_checkpoint is not None and not Path(student_checkpoint).is_file():
        raise FileNotFoundError(
            f"Promoted float student checkpoint is required for ATT but was not found: {student_checkpoint}"
        )
    if manifest_path:
        train_manifest = Path(manifest_path)
    else:
        train_manifest = Path(cfg["paths"]["train_manifest"])

    if not train_manifest.is_file():
        raise FileNotFoundError(
            f"Train manifest not found: {train_manifest}. "
            f"Manifest fallback across split sources is strictly prohibited."
        )

    # Strictly require promoted student checkpoint or resume checkpoint
    if (student_checkpoint is None or not Path(student_checkpoint).is_file()) and resume_checkpoint is None:
        raise FileNotFoundError(
            f"Promoted float student checkpoint is required for ATT but was not found: {student_checkpoint}. "
            "Random-weight fallbacks are strictly prohibited."
        )

    # Setup distributed if launched via torchrun or multi-GPU
    rank, world_size, local_rank = setup_distributed()
    is_primary = is_main_process()
    required_ws = int(cfg["training"].get("required_world_size", 1))
    if not dry_run and torch.cuda.is_available() and world_size != required_ws:
        raise RuntimeError(
            f"ATT requires world_size={required_ws}, got {world_size}. "
            "Use launch_att_tracks.py or launch with matching torchrun processes."
        )
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    # Build student model
    student = build_model(cfg).to(device)
    if is_primary:
        print(f"Loading promoted float student checkpoint: {student_checkpoint}")
    _load_checkpoint(student, Path(student_checkpoint))

    if world_size > 1 and torch.cuda.is_available():
        student_module: nn.Module = DDP(
            student,
            device_ids=[local_rank],
            output_device=local_rank,
            gradient_as_bucket_view=True,
            find_unused_parameters=True,
        )
    else:
        student_module = student

    optimizer = build_optimizer(student, cfg)

    # Prepare dataset
    dataset = ATTRawDataset(
        manifest_path=train_manifest,
        data_root=cfg.get("paths", {}).get("data_root"),
        seed=seed,
        render_policy=cfg.get("preprocessing", {}).get("policy", "square_jpeg95"),
        num_candidates=num_candidates,
        epoch=0,
        require_materialized=bool(cfg.get("paths", {}).get("require_materialized", False)),
    )

    sampler = DeterministicDistributedCoverageSampler(
        dataset_size=len(dataset),
        seed=seed,
        rank=rank,
        world_size=world_size,
        epoch=0,
    )

    physical_batch_size = int(cfg["training"]["physical_batch_size"])
    loader = DataLoader(
        dataset,
        batch_size=physical_batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=int(cfg["training"].get("num_workers", 2)),
        collate_fn=collate_att_batch,
        pin_memory=torch.cuda.is_available(),
    )

    accum_steps = int(cfg["training"]["gradient_accumulation"])
    updates_per_epoch = math.ceil(len(loader) / accum_steps)
    total_steps = updates_per_epoch * epochs
    scheduler = build_scheduler(optimizer, total_updates=max(1, total_steps), warmup_fraction=0.05)

    w_orig = float(cfg.get("loss", {}).get("provenance_original", 1.0))
    w_hard = float(cfg.get("loss", {}).get("provenance_transformed", 1.0))
    w_cons = float(cfg.get("loss", {}).get("prediction_consistency", 0.5))

    # Tracking metrics
    hardest_family_counts: dict[str, int] = {f.name.lower(): 0 for f in ATT_TRANSFORMATION_FAMILIES}
    running_loss = 0.0
    global_update = 0
    start_epoch = 0
    resume_micro_step = 0

    if resume_checkpoint is not None:
        global_update, start_epoch, resume_micro_step = restore_checkpoint(
            Path(resume_checkpoint),
            student,
            optimizer,
            scheduler,
            None,
            train_manifest,
            device,
        )
        if is_primary:
            print(
                f"Resumed ATT from {resume_checkpoint}: update={global_update} "
                f"epoch={start_epoch} micro_step={resume_micro_step}",
                flush=True,
            )

    student.train()
    optimizer.zero_grad()

    if is_primary:
        print(
            f"Starting ATT for {variant} student: {len(dataset)} records, {epochs} epoch(s), {num_candidates} candidates/record."
        )

    coverage_milestones = {0.25, 0.50, 0.75}
    saved_milestones: set[float] = set()

    for epoch in range(start_epoch, epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
            if epoch == start_epoch and resume_micro_step > 0:
                sampler.set_start_offset(resume_micro_step)
            else:
                sampler.set_start_offset(0)
        dataset.set_epoch(epoch)

        for step, batch in enumerate(loader, start=1):
            if epoch == start_epoch and step <= resume_micro_step:
                continue
            orig_images = batch["original_images"]
            cand_batch = batch["candidates"]
            ai_pos = batch["ai_positive"].to(device)

            # 1. Score candidates without gradients and select hardest
            hardest_images, hardest_specs, _ = score_and_select_hardest(
                student, cand_batch, ai_pos
            )

            for spec in hardest_specs:
                hardest_family_counts[spec.family.name.lower()] += 1

            # One gradient-tracked forward amortizes preprocessing and DDP synchronization.
            split = len(orig_images)
            combined_out: ProvenanceOutput = student_module([*orig_images, *hardest_images])
            orig_logits = combined_out.ai_positive_logit[:split]
            hard_logits = combined_out.ai_positive_logit[split:]
            loss_orig = compute_sample_loss(orig_logits, ai_pos).mean()
            loss_hard = compute_sample_loss(hard_logits, ai_pos).mean()

            # Optional prediction consistency
            loss_cons = F.mse_loss(orig_logits, hard_logits)

            loss = (w_orig * loss_orig + w_hard * loss_hard + w_cons * loss_cons) / accum_steps
            loss.backward()

            running_loss += loss.item() * accum_steps

            if step % accum_steps == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_update += 1
                if is_primary and global_update % 25 == 0:
                    print(
                        f"ATT {variant} update={global_update}/{total_steps} "
                        f"loss={running_loss / step:.5f}",
                        flush=True,
                    )

            progress = (epoch * len(loader) + step) / max(1, epochs * len(loader))
            for m in coverage_milestones:
                if m not in saved_milestones and progress >= m:
                    saved_milestones.add(m)
                    if is_primary:
                        ckpt_name = f"checkpoint-coverage-{int(m * 100)}pct.pt"
                        cov_path = save_checkpoint(
                            student,
                            cfg,
                            out_dir,
                            global_update,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            manifest_path=train_manifest,
                            epoch=epoch,
                            micro_step=step,
                            filename=ckpt_name,
                            coverage_metadata=sampler.get_coverage_report(),
                        )
                        print(
                            f"Saved ATT coverage milestone {int(m * 100)}%: {cov_path}", flush=True
                        )
            if dry_run and global_update >= 2:
                if is_primary:
                    print("Dry run completed exactly 2 updates; stopping.")
                break

        if dry_run and global_update >= 2:
            break
    # Save in the canonical checkpoint format consumed by evaluation and export.
    final_checkpoint_path = out_dir / "checkpoint-final.pt"
    if is_primary:
        final_checkpoint_path = save_checkpoint(
            student,
            cfg,
            out_dir,
            global_update,
            optimizer=optimizer,
            scheduler=scheduler,
            manifest_path=train_manifest,
            epoch=max(0, epochs - 1),
            micro_step=len(loader),
            filename="checkpoint-final.pt",
            coverage_metadata=sampler.get_coverage_report(),
        )
        print(f"Saved post-ATT checkpoint to: {final_checkpoint_path}")

        # Write training summary
        summary = {
            "variant": variant,
            "checkpoint_path": str(final_checkpoint_path),
            "records_covered": len(dataset),
            "unique_rows_covered": sampler.dataset_size,
            "num_padded_repeats": sampler.num_padding,
            "epochs": epochs,
            "updates": global_update,
            "average_loss": running_loss / max(1, step + 1),
            "hardest_family_distribution": hardest_family_counts,
            "coverage_report": sampler.get_coverage_report(),
        }
        (out_dir / "att_training_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

    if world_size > 1:
        cleanup_distributed()

    return final_checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ATT student model")
    parser.add_argument(
        "--variant", type=str, required=True, choices=["small", "base"], help="Student variant"
    )
    parser.add_argument(
        "--student-checkpoint",
        type=Path,
        default=None,
        help="Path to promoted float student checkpoint",
    )
    parser.add_argument("--manifest", type=Path, default=None, help="Path to training manifest")
    parser.add_argument("--config", type=Path, default=None, help="Path to ATT config YAML")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory")
    parser.add_argument(
        "--num-candidates", type=int, default=3, help="Number of transform candidates per row"
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of training passes (default: 1)"
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Physical batch size per GPU")
    parser.add_argument(
        "--gradient-accumulation", type=int, default=None, help="Gradient accumulation steps"
    )
    parser.add_argument("--world-size", type=int, default=None, help="Expected DDP world size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--resume", type=Path, default=None, help="Resume ATT training from checkpoint"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Run 2 smoke iterations without full pass"
    )
    args = parser.parse_args()

    train_att(
        variant=args.variant,
        student_checkpoint=args.student_checkpoint,
        config_path=args.config,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        num_candidates=args.num_candidates,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        world_size_override=args.world_size,
        learning_rate=args.lr,
        seed=args.seed,
        resume_checkpoint=args.resume,
        dry_run=args.dry_run,
    )

if __name__ == "__main__":
    main()
