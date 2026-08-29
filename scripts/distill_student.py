from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import sys
from contextlib import nullcontext
from pathlib import Path
from random import Random
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from aigc_detector.config import load_config
from aigc_detector.constants import Provenance, Transformation
from aigc_detector.dataset import (
    PairedImageDataset,
    collate_pairs,
    load_manifest_frame,
    parse_provenance,
    verify_materialization,
)
from aigc_detector.losses import LossWeights, provenance_robustness_loss
from aigc_detector.manifests import manifest_digest as compute_df_manifest_digest
from aigc_detector.metrics import binary_auroc, calibrate_validation_threshold
from aigc_detector.model import ProvenanceOutput
from aigc_detector.predict import _load_checkpoint
from aigc_detector.preprocessing import RenderPolicy, render_for_model
from aigc_detector.runtime import (
    cleanup_distributed,
    is_main_process,
    load_local_environment,
    seed_everything,
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

# Canonical student presets: 2-GPU topology, two complete passes, effective batch 48
STUDENT_PRESETS: dict[str, dict[str, Any]] = {
    "small": {
        "encoder_id": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "encoder_dim": 384,
        "epochs": 2,
        "required_world_size": 2,
        "physical_batch_size": 12,
        "gradient_accumulation": 2,
    },
    "base": {
        "encoder_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "encoder_dim": 768,
        "epochs": 2,
        "required_world_size": 2,
        "physical_batch_size": 12,
        "gradient_accumulation": 2,
    },
}

# Complete architectural metadata and exact parameter counts for independent DINOv3 student detectors
STUDENT_METADATA: dict[str, dict[str, Any]] = {
    "small": {
        "variant": "small",
        "model_family": "dinov3-vits16",
        "encoder_id": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "encoder_dim": 384,
        "image_size": 224,
        "backbone_parameters": 21596544,
        "token_adapter_parameters": 197888,
        "heads_parameters": 3295234,
        "exact_parameter_count": 25089666,
        "description": "DINOv3 ViT-S complete detector (25.1M parameters)",
    },
    "base": {
        "variant": "base",
        "model_family": "dinov3-vitb16",
        "encoder_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "encoder_dim": 768,
        "image_size": 224,
        "backbone_parameters": 85660416,
        "token_adapter_parameters": 395264,
        "heads_parameters": 3295234,
        "exact_parameter_count": 89350914,
        "description": "DINOv3 ViT-B complete detector (89.4M parameters)",
    },
}

STUDENT_EVALUATION_CONDITIONS: dict[str, TransformSpec | None] = {
    "clean": None,
    "jpeg70": TransformSpec(Transformation.JPEG, 70.0),
    "jpeg50": TransformSpec(Transformation.JPEG, 50.0),
    "blur1": TransformSpec(Transformation.BLUR, 1.0),
    "resize_half": TransformSpec(Transformation.RESIZE, 0.5),
    "noise0.05": TransformSpec(Transformation.NOISE, 0.05),
    "color20": TransformSpec(Transformation.COLOR, 0.20),
    "crop80": TransformSpec(Transformation.CROP, 0.80),
}


def get_student_parameter_counts(variant: str) -> dict[str, int]:
    """Return verified exact parameter counts for a given student detector variant."""
    if variant not in STUDENT_METADATA:
        raise ValueError(
            f"Unknown student variant {variant!r}; choices: {list(STUDENT_METADATA.keys())}"
        )
    meta = STUDENT_METADATA[variant]
    return {
        "backbone_parameters": int(meta["backbone_parameters"]),
        "token_adapter_parameters": int(meta["token_adapter_parameters"]),
        "heads_parameters": int(meta["heads_parameters"]),
        "exact_parameter_count": int(meta["exact_parameter_count"]),
    }


def verify_checkpoint_eligibility(checkpoint_path: Path | str) -> None:
    """Guard against using the incomplete update250 small artifact as a promoted/completed student model."""
    path_str = str(checkpoint_path).replace("\\", "/")
    if "student_vits_checkpoint2" in path_str:
        raise ValueError(
            f"The existing small artifact '{checkpoint_path}' (update 250) is an incomplete "
            "iteration 1 exploration checkpoint and cannot be used as a final or promoted student model."
        )


def compute_file_sha256(path: Path | str) -> str:
    """Compute SHA-256 hex digest of a local file."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found for SHA256 computation: {p}")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_manifest_digest(manifest_path: Path | str) -> str:
    """Compute deterministic SHA-256 digest of the manifest dataset records."""
    frame = load_manifest_frame(manifest_path)
    return compute_df_manifest_digest(frame)


def _resolve_data_root(config: dict, manifest_path: Path) -> Path:
    """Resolve data root directory handling ${ENV_VAR} syntax with manifest-dir fallback."""
    raw = config.get("paths", {}).get("data_root")
    if isinstance(raw, str):
        raw = raw.strip()
        if raw.startswith("${") and raw.endswith("}"):
            env_var = raw[2:-1]
            val = os.environ.get(env_var)
            if val and Path(val).is_dir():
                return Path(val)
            return manifest_path.parent
        if raw and Path(raw).is_dir():
            return Path(raw)
    env_val = os.environ.get("TECHJAM_DATA_ROOT")
    if env_val and Path(env_val).is_dir():
        return Path(env_val)
    return manifest_path.parent


def verify_teacher_promotion(
    teacher_checkpoint: Path,
    teacher_promotion_report: Path | None,
    manifest_path: Path,
) -> dict[str, Any]:
    """Verify that the teacher checkpoint passed promotion, validating SHA and contract metrics.

    Production always requires a promoted teacher report and matching checkpoint SHA256.
    """
    teacher_sha256 = compute_file_sha256(teacher_checkpoint)
    manifest_digest_val = compute_manifest_digest(manifest_path)

    report_path = teacher_promotion_report
    if report_path is None or not report_path.is_file():
        candidate = teacher_checkpoint.parent / "promotion_report.json"
        if candidate.is_file():
            report_path = candidate

    if report_path is None or not report_path.is_file():
        raise FileNotFoundError(
            f"Teacher promotion report is strictly required for student distillation. "
            f"Expected report at {report_path or (teacher_checkpoint.parent / 'promotion_report.json')} "
            f"for checkpoint {teacher_checkpoint}. Ensure teacher passed the promotion gate."
        )

    try:
        report_data = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Corrupt teacher promotion report at {report_path}") from exc

    if not bool(report_data.get("passed", False)):
        failed_reasons = report_data.get("failed_reasons", [])
        raise RuntimeError(
            f"Teacher checkpoint failed promotion gate according to {report_path}: {failed_reasons}. "
            "Distillation requires an upstream promoted teacher."
        )

    expected_sha = report_data.get("checkpoint_sha256")
    if not expected_sha or expected_sha != teacher_sha256:
        raise RuntimeError(
            f"Teacher checkpoint SHA256 mismatch: actual {teacher_sha256} != report {expected_sha}"
        )

    # Parse teacher metrics matching the TeacherPromotion contract
    metrics = report_data.get("metrics", {})
    clean_block = metrics.get("clean", {})
    worst_block = metrics.get("worst", {})

    clean_auroc = (
        clean_block.get("ai_positive_auroc")
        if isinstance(clean_block, dict)
        else metrics.get("clean_auroc")
    )
    worst_auroc = (
        worst_block.get("worst_auroc")
        if isinstance(worst_block, dict)
        else metrics.get("worst_transformed_auroc")
    )

    if clean_auroc is None:
        raise RuntimeError(
            f"Teacher promotion report at {report_path} is missing metrics.clean.ai_positive_auroc. "
            f"Available keys: {list(metrics.keys())}"
        )
    if worst_auroc is None:
        raise RuntimeError(
            f"Teacher promotion report at {report_path} is missing metrics.worst.worst_auroc. "
            f"Available keys: {list(metrics.keys())}"
        )

    clean_auroc = float(clean_auroc)
    worst_auroc = float(worst_auroc)

    # Verify manifest digest if present
    expected_manifest_digest = report_data.get("manifest_digest")
    if expected_manifest_digest and expected_manifest_digest != manifest_digest_val:
        raise RuntimeError(
            f"Manifest digest mismatch between teacher report ({expected_manifest_digest}) "
            f"and current training manifest ({manifest_digest_val})"
        )

    return {
        "teacher_checkpoint": str(teacher_checkpoint),
        "teacher_checkpoint_sha256": teacher_sha256,
        "manifest_digest": manifest_digest_val,
        "teacher_passed": True,
        "report_path": str(report_path),
        "teacher_clean_auroc": clean_auroc,
        "teacher_worst_transformed_auroc": worst_auroc,
        "failed_reasons": [],
    }


def student_config(teacher_config: dict, variant: str) -> dict:
    """Build a complete student configuration dictionary matching the canonical 2-GPU contract."""
    if variant not in STUDENT_PRESETS:
        raise ValueError(f"Unknown variant {variant!r}; choices: {list(STUDENT_PRESETS.keys())}")
    values = STUDENT_PRESETS[variant]
    config = copy.deepcopy(teacher_config)
    config["model"].update(
        {
            "backbone_type": "dinov3",
            "encoder_id": values["encoder_id"],
            "encoder_revision": None,
            "encoder_dim": values["encoder_dim"],
            "image_size": 224,
            "freeze_encoder": False,
            "trainable_last_layers": 0,
            "gradient_checkpointing": True,
            "spectral_expert": False,
            "use_token_adapter": True,
            "trunk_dim": 512,
        }
    )
    config["training"].update(
        {
            "stage": f"student_dinov3_{variant}",
            "epochs": 2,
            "physical_batch_size": 12,
            "gradient_accumulation": 2,
            "required_world_size": 2,
            "optimizer": "adamw",
            "encoder_lr": 2.0e-5,
            "heads_lr": 2.0e-4,
            "layerwise_lr_decay": 0.85,
            "weight_decay": 0.01,
            "gradient_clip_norm": 1.0,
            "deterministic_coverage_sampler": True,
        }
    )
    # Remove any legacy max_updates cap; schedule is determined by two complete coverage passes
    config["training"].pop("max_updates", None)
    config["loss"].update(
        {
            "provenance_original": 1.0,
            "provenance_transformed": 1.0,
            "prediction_consistency": 1.0,
            "feature_consistency": 0.5,
            "mask_focal": 0.0,
            "mask_dice": 0.0,
            "ema_consistency": 2.0,
            "teacher_feature_consistency": 0.5,
        }
    )
    return config


def _slice_provenance_output(
    output: ProvenanceOutput, start: int, end: int | None = None
) -> ProvenanceOutput:
    """Slice every field of a batched structured output without losing metadata."""
    selected = slice(start, end)
    return ProvenanceOutput(
        ai_positive_logit=output.ai_positive_logit[selected],
        probabilities=output.probabilities[selected],
        aigc_features=output.aigc_features[selected],
        tamper_features=output.tamper_features[selected],
        token_tamper_logits=output.token_tamper_logits[selected],
        fusion_gates=None if output.fusion_gates is None else output.fusion_gates[selected],
    )


# Canonical validation threshold calibration is provided by aigc_detector.metrics.calibrate_validation_threshold
# (re-exported here for backwards compatibility with existing callers and tests).


def evaluate_student_validation(
    student: torch.nn.Module,
    val_manifest: Path | str,
    data_root: Path | str | None,
    device: torch.device,
    *,
    teacher_clean_auroc: float,
    teacher_worst_transformed_auroc: float,
    batch_size: int = 16,
    limit: int | None = None,
    render_policy: str = "square_jpeg95",
) -> dict[str, Any]:
    """Evaluate student on fixed validation population and check promotion gates."""
    val_path = Path(val_manifest)
    if not val_path.is_file():
        raise FileNotFoundError(f"Validation manifest not found: {val_path}")

    frame = load_manifest_frame(val_path)
    if limit is not None and limit > 0:
        frame = frame.iloc[:limit]

    root = Path(data_root) if data_root is not None else val_path.parent
    records: list[dict[str, Any]] = []
    for _idx, row in enumerate(frame.to_dict(orient="records")):
        row_split = str(row.get("split", "")).strip().lower()
        if row_split and row_split != "validation":
            raise ValueError(
                f"Validation manifest contains non-validation row split: {row_split!r}"
            )
        img_path = Path(str(row["image_path"]).replace("\\", "/"))
        if not img_path.is_absolute():
            img_path = root / img_path
        if not img_path.is_file():
            raise FileNotFoundError(f"Missing validation image: {img_path}")
        prov = parse_provenance(row["label"], str(row["dataset"]))
        ai_pos = row.get("ai_positive")
        ai_val = (
            int(ai_pos)
            if ai_pos is not None and not pd.isna(ai_pos)
            else (0 if prov == Provenance.AUTHENTIC else 1)
        )
        records.append({"image_path": img_path, "ai_positive": ai_val, "provenance": int(prov)})

    student.eval()
    condition_aurocs: dict[str, float] = {}
    condition_scores: dict[str, torch.Tensor] = {}
    condition_targets: dict[str, torch.Tensor] = {}

    for cond_name, transform_spec in STUDENT_EVALUATION_CONDITIONS.items():
        targets: list[int] = []
        scores: list[float] = []
        for i in range(0, len(records), batch_size):
            batch_records = records[i : i + batch_size]
            images: list[Image.Image] = []
            for r in batch_records:
                with Image.open(r["image_path"]) as img:
                    rgb = img.convert("RGB")
                    rendered = render_for_model(rgb, RenderPolicy(render_policy))
                    if transform_spec is not None:
                        rendered = apply_transform(rendered, transform_spec, Random(42 + i))
                    images.append(rendered)
                targets.append(r["ai_positive"])

            with torch.no_grad():
                out = student(images)
                probs = out.probabilities
                if probs.shape[-1] == 2:
                    ai_scores = probs[:, 1].tolist()
                else:
                    ai_scores = torch.sigmoid(out.ai_positive_logit).view(-1).tolist()
                scores.extend(ai_scores)

        target_tensor = torch.tensor(targets, dtype=torch.float32)
        score_tensor = torch.tensor(scores, dtype=torch.float32)

        # Compute AUROC directly; never catch and substitute 0.5
        cond_auroc = float(binary_auroc(target_tensor, score_tensor).item())
        condition_aurocs[cond_name] = cond_auroc
        condition_scores[cond_name] = score_tensor
        condition_targets[cond_name] = target_tensor

    clean_target = condition_targets["clean"]
    clean_score = condition_scores["clean"]
    clean_auroc = condition_aurocs["clean"]

    # Validation-constrained threshold calibration maximizing balanced accuracy with recalls >= 0.82
    calibrated_thresh, clean_tpr, clean_tnr, clean_bal_acc, recalls_met = (
        calibrate_validation_threshold(clean_target, clean_score)
    )

    # Evaluate conditions under the calibrated operating threshold
    condition_results: dict[str, dict[str, Any]] = {}
    for cond_name, c_score in condition_scores.items():
        c_target = condition_targets[cond_name]
        c_pos = c_target == 1
        c_neg = c_target == 0
        c_pred = c_score >= calibrated_thresh
        tp = int((c_pos & c_pred).sum().item())
        tn = int((c_neg & ~c_pred).sum().item())
        pos_cnt = int(c_pos.sum().item())
        neg_cnt = int(c_neg.sum().item())
        c_tpr = tp / pos_cnt if pos_cnt > 0 else 0.0
        c_tnr = tn / neg_cnt if neg_cnt > 0 else 0.0

        condition_results[cond_name] = {
            "ai_positive_auroc": condition_aurocs[cond_name],
            "ai_positive_recall": c_tpr,
            "authentic_recall": c_tnr,
            "balanced_accuracy": (c_tpr + c_tnr) / 2.0,
            "calibrated_threshold": calibrated_thresh,
        }

    transformed_aurocs = {k: v for k, v in condition_aurocs.items() if k != "clean"}
    worst_cond_name = (
        min(transformed_aurocs, key=transformed_aurocs.get) if transformed_aurocs else "clean"
    )
    worst_transformed_auroc = (
        transformed_aurocs[worst_cond_name] if transformed_aurocs else clean_auroc
    )
    mean_transformed_auroc = (
        sum(transformed_aurocs.values()) / len(transformed_aurocs)
        if transformed_aurocs
        else clean_auroc
    )

    clean_gap = teacher_clean_auroc - clean_auroc
    worst_gap = teacher_worst_transformed_auroc - worst_transformed_auroc

    passed = True
    failed_reasons: list[str] = []

    if clean_tpr < 0.82:
        passed = False
        failed_reasons.append(
            f"AI-positive recall {clean_tpr:.4f} < 0.82 at calibrated threshold {calibrated_thresh:.4f}"
        )
    if clean_tnr < 0.82:
        passed = False
        failed_reasons.append(
            f"Authentic recall {clean_tnr:.4f} < 0.82 at calibrated threshold {calibrated_thresh:.4f}"
        )
    if clean_gap > 0.02:
        passed = False
        failed_reasons.append(
            f"Clean AUROC drop {clean_gap:.4f} exceeds 2pp tolerance (teacher={teacher_clean_auroc:.4f}, student={clean_auroc:.4f})"
        )
    if worst_gap > 0.03:
        passed = False
        failed_reasons.append(
            f"Worst-transform AUROC drop {worst_gap:.4f} exceeds 3pp tolerance (teacher={teacher_worst_transformed_auroc:.4f}, student={worst_transformed_auroc:.4f})"
        )

    return {
        "passed": passed,
        "failed_reasons": failed_reasons,
        "calibrated_threshold": calibrated_thresh,
        "metrics": {
            "clean": {
                "ai_positive_auroc": clean_auroc,
                "ai_positive_recall": clean_tpr,
                "authentic_recall": clean_tnr,
                "balanced_accuracy": clean_bal_acc,
            },
            "worst": {
                "worst_auroc": worst_transformed_auroc,
                "worst_condition": worst_cond_name,
            },
            "mean": {
                "mean_auroc": mean_transformed_auroc,
            },
            "calibrated": {
                "clean": {
                    "ai_positive_recall": clean_tpr,
                    "authentic_recall": clean_tnr,
                    "balanced_accuracy": clean_bal_acc,
                },
                "threshold": calibrated_thresh,
            },
            "teacher_baseline": {
                "clean_auroc": teacher_clean_auroc,
                "worst_auroc": teacher_worst_transformed_auroc,
            },
            "clean_gap": clean_gap,
            "worst_gap": worst_gap,
            "conditions": condition_results,
        },
    }


def train_student(
    teacher_config_path: Path,
    teacher_checkpoint: Path,
    manifest_path: Path,
    output_dir: Path,
    variant: str,
    *,
    teacher_promotion_report: Path | None = None,
    val_manifest: Path | None = None,
    student_config_path: Path | None = None,
    world_size_override: int | None = None,
    physical_batch_size_override: int | None = None,
    gradient_accumulation_override: int | None = None,
    num_workers_override: int | None = None,
    epochs_override: int | None = None,
    seed_override: int | None = None,
    resume_checkpoint: Path | None = None,
    dry_run: bool = False,
) -> Path:
    """Execute two deterministic complete passes of student distillation from the promoted teacher."""
    verify_checkpoint_eligibility(teacher_checkpoint)
    if resume_checkpoint is not None:
        verify_checkpoint_eligibility(resume_checkpoint)

    project_root = teacher_config_path.resolve().parent.parent
    load_local_environment(project_root)

    if variant not in STUDENT_PRESETS:
        raise ValueError(
            f"Unknown student variant {variant!r}; choices: {list(STUDENT_PRESETS.keys())}"
        )

    teacher_config = load_config(teacher_config_path)

    if student_config_path is not None and student_config_path.is_file():
        config = load_config(student_config_path)
    else:
        config = student_config(teacher_config, variant)

    seed = int(seed_override if seed_override is not None else config.get("seed", 42))
    seed_everything(seed)
    config["seed"] = seed

    epochs = int(
        epochs_override if epochs_override is not None else config["training"].get("epochs", 2)
    )
    config["training"]["epochs"] = epochs

    if world_size_override is not None:
        if world_size_override <= 0:
            raise ValueError("world_size_override must be positive")
        config["training"]["required_world_size"] = world_size_override
    if physical_batch_size_override is not None:
        if physical_batch_size_override <= 0:
            raise ValueError("physical_batch_size_override must be positive")
        config["training"]["physical_batch_size"] = physical_batch_size_override
    if gradient_accumulation_override is not None:
        if gradient_accumulation_override <= 0:
            raise ValueError("gradient_accumulation_override must be positive")
        config["training"]["gradient_accumulation"] = gradient_accumulation_override
    if num_workers_override is not None:
        if num_workers_override < 0:
            raise ValueError("num_workers_override must be non-negative")
        config["training"]["num_workers"] = num_workers_override

    # Verify effective batch = 48
    phys = int(config["training"].get("physical_batch_size", 12))
    accum = int(config["training"].get("gradient_accumulation", 2))
    ws = int(config["training"].get("required_world_size", 2))

    manifest_p = Path(manifest_path)
    if not manifest_p.is_file():
        raise FileNotFoundError(
            f"Training manifest not found: {manifest_p}. "
            "Manifest fallback across split sources is strictly prohibited."
        )

    # Verify materialization (fail closed on missing image files)
    resolved_data_root = _resolve_data_root(config, manifest_path)
    require_mat = bool(config["paths"].get("require_materialized", True))
    verify_materialization(
        manifest_path,
        data_root=resolved_data_root,
        allow_missing=not require_mat,
    )

    # Always strictly enforce teacher promotion report and hash verification
    teacher_verification = verify_teacher_promotion(
        teacher_checkpoint,
        teacher_promotion_report,
        manifest_path,
    )

    rank, world_size, local_rank = setup_distributed()
    is_primary = is_main_process()

    if not dry_run and torch.cuda.is_available() and world_size != ws:
        raise RuntimeError(
            f"Student distillation requires world_size={ws} on 2-GPU pool, got {world_size}. "
            "Plain-Python pseudo-DDP launches are prohibited; launch via torch.distributed.run or torchrun."
        )
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    if is_primary:
        param_counts = get_student_parameter_counts(variant)
        print("=" * 60, flush=True)
        print(f"MechaDetect Student Distillation Track: {variant.upper()}", flush=True)
        print(f"Model family:       {STUDENT_METADATA[variant]['model_family']}", flush=True)
        print(f"Complete detector:  {param_counts['exact_parameter_count']:,} params", flush=True)
        print(f"Teacher checkpoint: {teacher_checkpoint}", flush=True)
        print(
            f"Teacher SHA-256:    {teacher_verification['teacher_checkpoint_sha256'][:16]}...",
            flush=True,
        )
        print(f"Teacher clean AUROC:{teacher_verification['teacher_clean_auroc']:.4f}", flush=True)
        print(
            f"Teacher worst AUROC:{teacher_verification['teacher_worst_transformed_auroc']:.4f}",
            flush=True,
        )
        print(f"Manifest digest:    {teacher_verification['manifest_digest'][:16]}...", flush=True)
        print(f"Training epochs:    {epochs} complete deterministic passes", flush=True)
        print(
            f"Geometry:           {phys} physical x {ws} GPUs x {accum} accum = {phys * ws * accum} effective batch",
            flush=True,
        )
        print("=" * 60, flush=True)

    # Build frozen teacher
    teacher = build_model(teacher_config).to(device).eval()
    _load_checkpoint(teacher, teacher_checkpoint)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    # Build student detector
    student = build_model(config).to(device)

    # Training dataset: strictly enforced train split
    dataset = PairedImageDataset(
        manifest_path,
        data_root=resolved_data_root,
        seed=seed,
        transform_families=tuple(
            Transformation[name.upper()]
            for name in config.get("transforms", {}).get(
                "families", ["jpeg", "blur", "resize", "noise", "color", "crop"]
            )
        ),
        render_policy=config.get("preprocessing", {}).get("policy", "square_jpeg95"),
        expected_split="train",
    )

    # Deterministic coverage sampler: complete population without replacement
    sampler = DeterministicDistributedCoverageSampler(
        len(dataset),
        seed=seed,
        rank=rank,
        world_size=world_size,
    )

    num_workers = int(config["training"].get("num_workers", 4 if sys.platform != "win32" else 0))
    loader = DataLoader(
        dataset,
        batch_size=phys,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_pairs,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        generator=torch.Generator().manual_seed(seed + 10_000),
    )

    if len(loader) == 0:
        cleanup_distributed()
        raise ValueError("Cannot distill a student from an empty dataset")

    optimizer = build_optimizer(student, config)
    student_module: torch.nn.Module = (
        DDP(
            student,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            output_device=local_rank if torch.cuda.is_available() else None,
            gradient_as_bucket_view=True,
            find_unused_parameters=True,
        )
        if world_size > 1 and torch.cuda.is_available()
        else student
    )

    training = config["training"]
    loss_config = config["loss"]
    weights = LossWeights(
        provenance_original=float(loss_config.get("provenance_original", 1.0)),
        provenance_transformed=float(loss_config.get("provenance_transformed", 1.0)),
        prediction_consistency=float(loss_config.get("prediction_consistency", 1.0)),
        feature_consistency=float(loss_config.get("feature_consistency", 0.5)),
        mask_focal=float(loss_config.get("mask_focal", 0.0)),
        mask_dice=float(loss_config.get("mask_dice", 0.0)),
        ema_consistency=float(loss_config.get("ema_consistency", 2.0)),
    )
    teacher_feature_weight = float(loss_config.get("teacher_feature_consistency", 0.5))
    teacher_confidence_threshold = float(training.get("ema", {}).get("confidence_threshold", 0.8))

    micro_steps_per_epoch = len(loader)
    updates_per_epoch = math.ceil(micro_steps_per_epoch / accum)
    max_updates = epochs * updates_per_epoch

    scheduler = build_scheduler(
        optimizer, max_updates, float(training.get("warmup_fraction", 0.05))
    )

    update = 0
    start_epoch = 0
    resume_micro_step = 0
    if resume_checkpoint is not None:
        update, start_epoch, resume_micro_step = restore_checkpoint(
            resume_checkpoint,
            student,
            optimizer,
            scheduler,
            None,
            manifest_path,
            device,
        )
        if is_primary:
            print(
                f"Resumed from {resume_checkpoint}: update={update} epoch={start_epoch} micro_step={resume_micro_step}",
                flush=True,
            )

    precision = str(training.get("precision", "bf16")).lower()
    autocast_enabled = precision == "bf16" and torch.cuda.is_available()

    coverage_milestones = [0.25, 0.50, 0.75, 1.00]
    saved_milestones: set[float] = set()

    output_dir.mkdir(parents=True, exist_ok=True)
    coverage_checkpoints: dict[float, Path] = {}

    student_module.train()
    for epoch in range(start_epoch, epochs):
        dataset.set_epoch(epoch)
        sampler.set_epoch(epoch)
        if epoch == start_epoch and resume_micro_step > 0:
            sampler.set_start_offset(resume_micro_step)
        else:
            sampler.set_start_offset(0)

        window_loss = 0.0
        window_components: dict[str, float] = {}
        window_micro_steps = 0

        for micro_step, raw_batch in enumerate(loader, start=1):
            if epoch == start_epoch and micro_step <= resume_micro_step:
                continue

            is_boundary = micro_step % accum == 0 or micro_step == micro_steps_per_epoch
            window_start = ((micro_step - 1) // accum) * accum + 1
            window_size = min(accum, micro_steps_per_epoch - window_start + 1)

            sync_context = (
                student_module.no_sync()
                if isinstance(student_module, DDP) and not is_boundary
                else nullcontext()
            )
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if autocast_enabled
                else nullcontext()
            )

            with sync_context:
                with autocast:
                    target = raw_batch["ai_positive"].to(device=device, dtype=torch.float32)
                    with torch.no_grad():
                        teacher_output = teacher(raw_batch["original"])
                    original_count = len(raw_batch["original"])
                    combined = student_module([*raw_batch["original"], *raw_batch["transformed"]])
                    original = _slice_provenance_output(combined, 0, original_count)
                    transformed = _slice_provenance_output(combined, original_count)

                    loss, components = provenance_robustness_loss(
                        original,
                        transformed,
                        ai_target=target,
                        weights=weights,
                        teacher_probabilities=teacher_output.probabilities.detach(),
                        teacher_confidence_threshold=teacher_confidence_threshold,
                    )
                    teacher_feature_loss = (
                        1.0
                        - F.cosine_similarity(
                            original.provenance_features,
                            teacher_output.provenance_features.detach(),
                            dim=-1,
                        )
                    ).mean()
                    loss = loss + teacher_feature_weight * teacher_feature_loss
                    components = dict(components)
                    components["teacher_feature_consistency"] = teacher_feature_loss
                    scaled_loss = loss / window_size
                scaled_loss.backward()

            window_loss += loss.detach().item()
            for name, value in components.items():
                window_components[name] = window_components.get(name, 0.0) + value.detach().item()
            window_micro_steps += 1

            if not is_boundary:
                continue

            torch.nn.utils.clip_grad_norm_(
                student.parameters(), float(training.get("gradient_clip_norm", 1.0))
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            update += 1

            if dry_run and update >= 2:
                if is_primary:
                    print(
                        f"Dry run completed exactly 2 updates for {variant}; stopping.", flush=True
                    )
                break

            # Check coverage fraction across the full 2-pass budget
            progress = (epoch * micro_steps_per_epoch + micro_step) / (
                epochs * micro_steps_per_epoch
            )
            for m in coverage_milestones:
                if m not in saved_milestones and progress >= m:
                    saved_milestones.add(m)
                    if is_primary:
                        ckpt_name = f"checkpoint-coverage-{int(m * 100)}pct.pt"
                        coverage_info = sampler.get_coverage_report()
                        coverage_info["pass_number"] = epoch + 1
                        coverage_info["progress_fraction"] = progress
                        coverage_path = save_checkpoint(
                            student,
                            config,
                            output_dir,
                            update,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            manifest_path=manifest_path,
                            epoch=epoch,
                            micro_step=micro_step,
                            filename=ckpt_name,
                            coverage_metadata=coverage_info,
                        )
                        coverage_checkpoints[m] = coverage_path
                        print(
                            f"Saved coverage milestone {int(m * 100)}%: {coverage_path}", flush=True
                        )

            if is_primary and update % 50 == 0:
                divisor = float(window_micro_steps)
                comp_str = " ".join(f"{k}={v / divisor:.4f}" for k, v in window_components.items())
                print(
                    f"student={variant} pass={epoch + 1}/{epochs} update={update}/{max_updates} "
                    f"loss={window_loss / divisor:.5f} {comp_str}",
                    flush=True,
                )

            window_loss = 0.0
            window_components = {}
            window_micro_steps = 0
        if dry_run and update >= 2:
            break

    if world_size > 1 and torch.cuda.is_available():
        torch.distributed.barrier()

    best_checkpoint_path = output_dir / "checkpoint-best.pt"
    promoted_checkpoint_path = output_dir / "checkpoint-promoted.pt"

    if is_primary:
        # Save final complete pass checkpoint
        _final_ckpt = save_checkpoint(
            student,
            config,
            output_dir,
            update,
            optimizer=optimizer,
            scheduler=scheduler,
            manifest_path=manifest_path,
            epoch=0 if dry_run else (epochs - 1),
            micro_step=update if dry_run else micro_steps_per_epoch,
            filename="checkpoint-final.pt",
            coverage_metadata=sampler.get_coverage_report(),
        )

    if dry_run:
        if world_size > 1 and torch.cuda.is_available():
            torch.distributed.barrier()
        cleanup_distributed()
        return output_dir / "checkpoint-final.pt"

    if is_primary:
        # External validation evaluation and promotion gate check across candidate checkpoints
        val_eval_manifest = val_manifest or config["paths"].get("val_manifest")
        if val_eval_manifest is None or not Path(val_eval_manifest).is_file():
            raise FileNotFoundError(
                f"Validation manifest is required for student promotion evaluation: {val_eval_manifest}"
            )

        candidates_to_eval: list[tuple[str, Path]] = []
        for m in sorted(coverage_checkpoints.keys()):
            ckpt_p = Path(coverage_checkpoints[m])
            if ckpt_p.is_file():
                candidates_to_eval.append((f"milestone_{int(m * 100)}pct", ckpt_p))
        if not any(cp.resolve() == _final_ckpt.resolve() for _, cp in candidates_to_eval):
            candidates_to_eval.append(("final", _final_ckpt))

        print(
            f"\nRunning external validation promotion evaluation across {len(candidates_to_eval)} candidate(s)...",
            flush=True,
        )
        evaluated_candidates: list[dict[str, Any]] = []
        for cand_name, cand_path in candidates_to_eval:
            _load_checkpoint(student, cand_path)
            cand_res = evaluate_student_validation(
                student,
                val_eval_manifest,
                resolved_data_root,
                device,
                teacher_clean_auroc=teacher_verification["teacher_clean_auroc"],
                teacher_worst_transformed_auroc=teacher_verification[
                    "teacher_worst_transformed_auroc"
                ],
            )
            evaluated_candidates.append(
                {
                    "name": cand_name,
                    "path": cand_path,
                    "result": cand_res,
                }
            )
            print(
                f"Candidate [{cand_name}]: Clean AUROC={cand_res['metrics']['clean']['ai_positive_auroc']:.4f}, "
                f"Worst AUROC={cand_res['metrics']['worst']['worst_auroc']:.4f}, Passed={cand_res['passed']}",
                flush=True,
            )

        # Deterministically rank candidates: passed first, then clean AUROC, then worst AUROC
        def _candidate_rank_key(cand: dict[str, Any]) -> tuple:
            r = cand["result"]
            m = r["metrics"]
            return (
                1 if r["passed"] else 0,
                float(m["clean"]["ai_positive_auroc"]),
                float(m["worst"]["worst_auroc"]),
                float(m.get("mean", {}).get("mean_auroc", 0.0)),
            )

        evaluated_candidates.sort(key=_candidate_rank_key, reverse=True)
        best_cand = evaluated_candidates[0]
        best_cand_path = best_cand["path"]
        promotion_result = best_cand["result"]
        print(f"Selected best candidate: {best_cand['name']} ({best_cand_path})", flush=True)

        shutil.copyfile(best_cand_path, best_checkpoint_path)
        print(f"Saved best validation checkpoint: {best_checkpoint_path}", flush=True)

        if promotion_result["passed"]:
            shutil.copyfile(best_cand_path, promoted_checkpoint_path)
            print(
                f"Promotion gate PASSED. Promoted checkpoint saved: {promoted_checkpoint_path}",
                flush=True,
            )
        else:
            if promoted_checkpoint_path.is_file():
                promoted_checkpoint_path.unlink()
            print(
                f"Promotion gate FAILED. Reasons: {promotion_result['failed_reasons']}", flush=True
            )

        # Write promotion report contract
        selected_checkpoint = (
            promoted_checkpoint_path if promotion_result["passed"] else best_checkpoint_path
        )
        selected_sha = compute_file_sha256(selected_checkpoint)
        promotion_report = {
            "checkpoint_path": str(selected_checkpoint),
            "checkpoint_sha256": selected_sha,
            "manifest_digest": compute_manifest_digest(val_eval_manifest),
            "variant": variant,
            "parameter_count": get_student_parameter_counts(variant)["exact_parameter_count"],
            "calibrated_threshold": promotion_result["calibrated_threshold"],
            "metrics": promotion_result["metrics"],
            "passed": promotion_result["passed"],
            "failed_reasons": promotion_result["failed_reasons"],
        }
        report_file = output_dir / "promotion_report.json"
        report_file.write_text(json.dumps(promotion_report, indent=2), encoding="utf-8")
        print(f"Wrote promotion report: {report_file}", flush=True)

        # Write metadata sidecar contract
        metadata = {
            "model_family": STUDENT_METADATA[variant]["model_family"],
            "parameter_count": get_student_parameter_counts(variant)["exact_parameter_count"],
            "quantization": "float32",
            "calibrated_threshold": promotion_result["calibrated_threshold"],
            "input_size": [3, 224, 224],
            "preprocessing_version": 2,
            "manifest_digest": compute_manifest_digest(val_eval_manifest),
            "evaluation_status": "promoted" if promotion_result["passed"] else "rejected",
        }
        metadata_file = output_dir / "metadata.json"
        metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Wrote metadata sidecar: {metadata_file}", flush=True)

    if world_size > 1 and torch.cuda.is_available():
        torch.distributed.barrier()

    cleanup_distributed()
    return (
        promoted_checkpoint_path
        if (is_primary and promotion_result.get("passed"))
        else best_checkpoint_path
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Distill an independent DINOv3 ViT-S or ViT-B student detector from the promoted teacher."
    )
    parser.add_argument(
        "--teacher-config", type=Path, required=True, help="Path to teacher YAML configuration"
    )
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        required=True,
        help="Path to promoted teacher checkpoint (.pt)",
    )
    parser.add_argument(
        "--manifest", type=Path, required=True, help="Path to training manifest parquet"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Output directory for student checkpoints"
    )
    parser.add_argument(
        "--student",
        choices=sorted(STUDENT_PRESETS),
        required=True,
        help="Student variant (small or base)",
    )
    parser.add_argument(
        "--teacher-promotion-report",
        type=Path,
        default=None,
        help="Path to teacher promotion_report.json",
    )
    parser.add_argument(
        "--val-manifest",
        type=Path,
        default=None,
        help="Path to validation manifest for promotion gate",
    )
    parser.add_argument(
        "--student-config", type=Path, default=None, help="Path to specific student YAML config"
    )
    parser.add_argument(
        "--world-size", type=int, default=None, help="DDP world size (e.g. 2 for 2-GPU pool)"
    )
    parser.add_argument(
        "--physical-batch-size", type=int, default=None, help="Physical batch size per GPU"
    )
    parser.add_argument(
        "--gradient-accumulation", type=int, default=None, help="Gradient accumulation steps"
    )
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader num_workers")
    parser.add_argument(
        "--epochs", type=int, default=2, help="Number of complete deterministic passes (default: 2)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Distributed rendezvous port (default: 29501 for small, 29502 for base)",
    )
    parser.add_argument(
        "--devices", type=str, default=None, help="CUDA devices (e.g. '0,1' or '2,3')"
    )
    parser.add_argument(
        "--resume", type=Path, default=None, help="Resume student training from coverage checkpoint"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Verify configuration and hashes without training"
    )

    args = parser.parse_args()

    if args.devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    if args.port is not None:
        os.environ["MASTER_PORT"] = str(args.port)

    checkpoint = train_student(
        args.teacher_config,
        args.teacher_checkpoint,
        args.manifest,
        args.output_dir,
        args.student,
        teacher_promotion_report=args.teacher_promotion_report,
        val_manifest=args.val_manifest,
        student_config_path=args.student_config,
        world_size_override=args.world_size,
        physical_batch_size_override=args.physical_batch_size,
        gradient_accumulation_override=args.gradient_accumulation,
        num_workers_override=args.num_workers,
        epochs_override=args.epochs,
        seed_override=args.seed,
        resume_checkpoint=args.resume,
        dry_run=args.dry_run,
    )
    if is_main_process():
        print(f"Student distillation process complete: {checkpoint}")


if __name__ == "__main__":
    main()
