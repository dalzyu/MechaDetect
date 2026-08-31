#!/usr/bin/env python3
"""TechJam 2026 Track 5 — Demonstration Benchmark Evaluation Harness.

Evaluates an AI-image provenance detector against the official organizer
validation dataset (WildFake demonstration subset):
  - Non-AIGC: COCO val2017 (4,998 images)
  - AIGC:     DALL·E Advanced (8,843 images)
  Total: 13,841 images

Under all official organizer robustness transformations:
  - Clean (no augmentation)
  - JPEG Compression: quality = 90, 70, 50, 30
  - Gaussian Blur:    kernel σ = 0.5, 1.0, 2.0
  - Resize:           scale 0.5× / 0.25× then bicubic upscale
  - Gaussian Noise:   σ = 0.02, 0.05, 0.10
  - Color Jitter:     brightness/contrast/saturation ±20%
  - Center Crop:      crop 80% then bicubic resize back

Usage:
  python scripts/eval_organizer_demo.py \
    --checkpoint models/teachers/iteration1/stage2/model-weights.safetensors \
    --config configs/teacher_dinov3_stage2_paired_unfrozen.yaml \
    --output outputs/organizer_demo_eval.json \
    --limit 200   # optional: for quick testing
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from aigc_detector.config import load_config
from aigc_detector.constants import Transformation
from aigc_detector.metrics import balanced_accuracy, binary_auroc, confusion_matrix, macro_f1
from aigc_detector.model import ai_generated_probability
from aigc_detector.predict import _load_checkpoint
from aigc_detector.preprocessing import RenderPolicy, render_for_model
from aigc_detector.runtime import load_local_environment
from aigc_detector.train import build_model
from aigc_detector.transforms import TransformSpec, apply_transform

load_local_environment(_PROJECT_ROOT)


# =========================================================================== #
# TechJam Organizer Augmentation Suite Specification
# =========================================================================== #

@dataclass(frozen=True)
class ConditionSpec:
    name: str
    category: str
    spec: TransformSpec | None
    real_world_analog: str


ORGANIZER_CONDITIONS: list[ConditionSpec] = [
    ConditionSpec("clean", "Baseline", None, "Unaltered input"),
    # JPEG Compression
    ConditionSpec("jpeg90", "JPEG Compression", TransformSpec(Transformation.JPEG, 90.0), "High-quality save / CDN"),
    ConditionSpec("jpeg70", "JPEG Compression", TransformSpec(Transformation.JPEG, 70.0), "Standard web publishing"),
    ConditionSpec("jpeg50", "JPEG Compression", TransformSpec(Transformation.JPEG, 50.0), "Social media re-encode / messaging"),
    ConditionSpec("jpeg30", "JPEG Compression", TransformSpec(Transformation.JPEG, 30.0), "Aggressive mobile compression"),
    # Gaussian Blur
    ConditionSpec("blur0.5", "Gaussian Blur", TransformSpec(Transformation.BLUR, 0.5), "Subtle lens / motion blur"),
    ConditionSpec("blur1.0", "Gaussian Blur", TransformSpec(Transformation.BLUR, 1.0), "Moderate out-of-focus"),
    ConditionSpec("blur2.0", "Gaussian Blur", TransformSpec(Transformation.BLUR, 2.0), "Heavy out-of-focus"),
    # Resize
    ConditionSpec("resize_half", "Resize", TransformSpec(Transformation.RESIZE, 0.50), "0.50x scale down + upscale"),
    ConditionSpec("resize_quarter", "Resize", TransformSpec(Transformation.RESIZE, 0.25), "0.25x thumbnail generation + upscale"),
    # Gaussian Noise
    ConditionSpec("noise0.02", "Gaussian Noise", TransformSpec(Transformation.NOISE, 0.02), "Low ISO sensor noise"),
    ConditionSpec("noise0.05", "Gaussian Noise", TransformSpec(Transformation.NOISE, 0.05), "Medium ISO noise / grain"),
    ConditionSpec("noise0.10", "Gaussian Noise", TransformSpec(Transformation.NOISE, 0.10), "High ISO / low-light sensor noise"),
    # Color Jitter
    ConditionSpec("color_jitter20", "Color Jitter", TransformSpec(Transformation.COLOR, 0.20), "Brightness/contrast/sat ±20%"),
    # Center Crop
    ConditionSpec("crop80", "Center Crop", TransformSpec(Transformation.CROP, 0.80), "80% crop (framing / profile picture)"),
]


# =========================================================================== #
# Dataset & Loading
# =========================================================================== #

class OrganizerDemoDataset(Dataset[dict[str, Any]]):
    """Loads images from the organizer demonstration dataset with on-the-fly transformations."""

    def __init__(
        self,
        manifest_path: Path,
        data_root: Path,
        condition: ConditionSpec,
        render_policy: str = "square_jpeg95",
        limit_per_class: int | None = None,
        limit_total: int | None = None,
        seed: int = 42,
    ) -> None:
        self.data_root = data_root
        self.condition = condition
        self.render_policy = RenderPolicy(render_policy)
        self.seed = seed

        df = pd.read_csv(manifest_path, low_memory=False)

        # Validate expected classes
        # Non-AIGC: 'authentic', AIGC: 'fully_aigc'
        df["ai_positive"] = (df["label"] == "fully_aigc").astype(int)

        if limit_per_class is not None:
            df = pd.concat(
                [
                    group.sample(n=min(limit_per_class, len(group)), random_state=seed)
                    for _, group in df.groupby("ai_positive")
                ],
                ignore_index=True,
            )
        elif limit_total is not None:
            df = df.sample(n=min(limit_total, len(df)), random_state=seed).reset_index(drop=True)

        self.records = df.to_dict(orient="records")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.records[idx]
        image_path = self.data_root / str(row["image_path"])

        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
        except Exception as err:
            raise RuntimeError(f"Failed to read image at {image_path}: {err}") from err

        # Apply transformation if specified
        if self.condition.spec is not None:
            rng = Random(self.seed + idx * 31)
            image = apply_transform(image, self.condition.spec, rng)

        return {
            "image": image,
            "ai_positive": int(row["ai_positive"]),
            "image_path": str(row["image_path"]),
            "label": str(row["label"]),
        }


def collate_demo(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": [item["image"] for item in batch],
        "ai_positives": torch.tensor([item["ai_positive"] for item in batch], dtype=torch.long),
        "image_paths": [item["image_path"] for item in batch],
        "labels": [item["label"] for item in batch],
    }


# =========================================================================== #
# Metric Evaluation
# =========================================================================== #

@torch.inference_mode()
def evaluate_condition(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    precision: str = "bf16",
    cond_label: str = "",
) -> dict[str, Any]:
    all_scores = []
    all_targets = []

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if precision == "bf16" and device.type == "cuda"
        else torch.nullcontext()
    )

    total_batches = len(loader)
    total_samples = len(loader.dataset)  # type: ignore[arg-type]
    processed_samples = 0
    t_start = time.perf_counter()

    for batch_idx, batch in enumerate(loader, start=1):
        targets = batch["ai_positives"]
        images = batch["images"]

        with autocast_ctx:
            output = model(images)
            probs = output.probabilities.float().cpu()
            scores = ai_generated_probability(probs)

        all_scores.append(scores)
        all_targets.append(targets)
        processed_samples += len(images)

        if batch_idx % 25 == 0 or batch_idx == total_batches:
            elapsed = time.perf_counter() - t_start
            rate = processed_samples / elapsed if elapsed > 0 else 0.0
            remaining = (total_samples - processed_samples) / rate if rate > 0 else 0.0
            print(
                f"  [{cond_label}] {processed_samples:>5d}/{total_samples} "
                f"({100.0 * processed_samples / total_samples:4.1f}%) | "
                f"{rate:5.1f} img/s | ETA {remaining:4.0f}s",
                flush=True,
            )

    targets_tensor = torch.cat(all_targets).long()
    scores_tensor = torch.cat(all_scores).float()

    preds_tensor = (scores_tensor >= 0.5).long()

    matrix = confusion_matrix(targets_tensor, preds_tensor, classes=2)
    clean_auroc = binary_auroc(targets_tensor.float(), scores_tensor)

    tp = int(((targets_tensor == 1) & (preds_tensor == 1)).sum())
    fn = int(((targets_tensor == 1) & (preds_tensor == 0)).sum())
    tn = int(((targets_tensor == 0) & (preds_tensor == 0)).sum())
    fp = int(((targets_tensor == 0) & (preds_tensor == 1)).sum())

    total = len(targets_tensor)
    pos_total = tp + fn
    neg_total = tn + fp

    ai_recall = (tp / pos_total) if pos_total > 0 else 0.0
    auth_recall = (tn / neg_total) if neg_total > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0

    return {
        "n_samples": total,
        "n_aigc": pos_total,
        "n_non_aigc": neg_total,
        "auroc": float(clean_auroc),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy(matrix)),
        "macro_f1": float(macro_f1(matrix)),
        "aigc_recall": float(ai_recall),
        "non_aigc_recall": float(auth_recall),
        "confusion_matrix": matrix.tolist(),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }

# =========================================================================== #
# Main Entry Point
# =========================================================================== #

def run_organizer_eval(
    checkpoint_path: Path,
    config_path: Path,
    manifest_path: Path,
    data_root: Path,
    output_json: Path | None = None,
    limit_per_class: int | None = None,
    limit_total: int | None = None,
    batch_size: int = 64,
    num_workers: int = 0,
    device_str: str = "cuda",
    conditions_filter: list[str] | None = None,
) -> dict[str, Any]:
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    config = load_config(config_path)

    print(f"Loading model on {device}...")
    model = build_model(config).to(device).eval()
    _load_checkpoint(model, checkpoint_path)
    if output_json is not None and output_json.is_file():
        try:
            results = json.loads(output_json.read_text(encoding="utf-8"))
            print(f"Resuming evaluation: found {len(results.get('conditions', {}))} completed conditions in {output_json}")
        except Exception:
            results = {}
    else:
        results = {}

    if "conditions" not in results:
        results["conditions"] = {}
    if "metadata" not in results:
        results["metadata"] = {
            "checkpoint": str(checkpoint_path.resolve()),
            "config": str(config_path.resolve()),
            "manifest": str(manifest_path.resolve()),
            "device": str(device),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "evaluation_scope": "TechJam 2026 Track 5 Organizer Demonstration Benchmark",
            "non_aigc_source": "COCO val2017 (4998)",
            "aigc_source": "DALL-E Advanced (8843)",
        }

    print("=" * 96)
    print("TECHJAM 2026 TRACK 5 — DEMONSTRATION BENCHMARK EVALUATION")
    print(f"Model:    {checkpoint_path.name}")
    print(f"Manifest: {manifest_path.name}")
    print("=" * 96)
    print(f"{'Condition':<18} | {'Category':<16} | {'AUROC':<7} | {'Acc':<7} | {'AIGC Rec':<9} | {'Non-AIGC':<9} | {'F1':<7}")
    print("-" * 96)

    precision = config.get("training", {}).get("precision", "bf16")

    selected_conditions = [
        c for c in ORGANIZER_CONDITIONS
        if conditions_filter is None or c.name in conditions_filter
    ]

    for cond_idx, condition in enumerate(selected_conditions, start=1):
        ds = OrganizerDemoDataset(
            manifest_path=manifest_path,
            data_root=data_root,
            condition=condition,
            render_policy=config.get("preprocessing", {}).get("policy", "square_jpeg95"),
            limit_per_class=limit_per_class,
            limit_total=limit_total,
        )

        if condition.name in results["conditions"]:
            m = results["conditions"][condition.name]
            if m.get("n_samples") == len(ds):
                print(
                    f"[{cond_idx}/{len(selected_conditions)}] {condition.name:<18} (cached) | "
                    f"AUROC: {m['auroc']:.4f} | Acc: {m['accuracy']:.4f} | "
                    f"AIGC Rec: {m['aigc_recall']:.4f} | Non-AIGC: {m['non_aigc_recall']:.4f}"
                )
                continue

        print(f"\n[{cond_idx}/{len(selected_conditions)}] Evaluating condition: {condition.name} ({condition.category})...")
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_demo,
            pin_memory=(device.type == "cuda"),
        )

        t_start = time.perf_counter()
        metrics = evaluate_condition(model, loader, device, precision=precision, cond_label=condition.name)
        metrics["elapsed_seconds"] = round(time.perf_counter() - t_start, 2)
        metrics["real_world_analog"] = condition.real_world_analog
        metrics["category"] = condition.category

        results["conditions"][condition.name] = metrics

        print(
            f"{condition.name:<18} | "
            f"{condition.category:<16} | "
            f"{metrics['auroc']:<7.4f} | "
            f"{metrics['accuracy']:<7.4f} | "
            f"{metrics['aigc_recall']:<9.4f} | "
            f"{metrics['non_aigc_recall']:<9.4f} | "
            f"{metrics['macro_f1']:<7.4f}"
        )
        if output_json is not None:
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("=" * 96)

    # Compute summary aggregates
    clean_metrics = results["conditions"]["clean"]
    transformed = [m for k, m in results["conditions"].items() if k != "clean"]

    mean_trans_auroc = float(np.mean([m["auroc"] for m in transformed]))
    worst_trans_auroc = float(np.min([m["auroc"] for m in transformed]))
    worst_condition = min(transformed, key=lambda m: m["auroc"])

    results["summary"] = {
        "clean_auroc": clean_metrics["auroc"],
        "clean_aigc_recall": clean_metrics["aigc_recall"],
        "clean_non_aigc_recall": clean_metrics["non_aigc_recall"],
        "mean_transformed_auroc": mean_trans_auroc,
        "worst_transformed_auroc": worst_trans_auroc,
        "worst_condition_name": [k for k, v in results["conditions"].items() if v == worst_condition][0],
        "auroc_drop": round(clean_metrics["auroc"] - mean_trans_auroc, 4),
    }

    print("\nBenchmark Summary:")
    print(f"  Clean AUROC:              {clean_metrics['auroc']:.4f}")
    print(f"  Mean Transformed AUROC:   {mean_trans_auroc:.4f}  (Δ = -{results['summary']['auroc_drop']:.4f})")
    print(f"  Worst Transformed AUROC:  {worst_trans_auroc:.4f}  [{results['summary']['worst_condition_name']}]")
    print(f"  Clean AIGC Recall:        {clean_metrics['aigc_recall']:.4f}")
    print(f"  Clean Non-AIGC Recall:    {clean_metrics['non_aigc_recall']:.4f}")

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nWrote full evaluation report to: {output_json}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate on TechJam Demonstration Benchmark")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model weights path (.pt or .safetensors)")
    parser.add_argument("--config", type=Path, default=Path("configs/teacher_dinov3_stage2_paired_unfrozen.yaml"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_PROJECT_ROOT / "metadata" / "organizer_demo_document_count.csv",
        help="Path to demo manifest (default: metadata/organizer_demo_document_count.csv)",
    )
    parser.add_argument("--output", type=Path, default=None, help="Save evaluation JSON results")
    parser.add_argument("--limit", type=int, default=None, help="Total sample limit for quick test")
    parser.add_argument("--per-class", type=int, default=None, help="Sample limit per class")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size (default: 64)")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device")
    parser.add_argument(
        "--conditions",
        type=str,
        default=None,
        help="Comma-separated condition names to evaluate (default: all 15 conditions)",
    )
    args = parser.parse_args()

    data_root = Path(os.environ.get("TECHJAM_DATA_ROOT", "E:/techjam26-runtime/data"))

    run_organizer_eval(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        manifest_path=args.manifest,
        data_root=data_root,
        output_json=args.output,
        limit_per_class=args.per_class,
        limit_total=args.limit,
        batch_size=args.batch_size,
        num_workers=args.workers,
        device_str=args.device,
        conditions_filter=[c.strip() for c in args.conditions.split(",")] if args.conditions else None,
    )


if __name__ == "__main__":
    main()
