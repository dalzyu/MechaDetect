#!/usr/bin/env python3
"""TechJam 2026 Track 5 — Ultra-High-Throughput Multi-Model Demonstration Evaluation.

Evaluates all model variants (quantized and unquantized alike) across the official
organizer demonstration benchmark in a single pipelined pass:
- Models:
    1. Quark Super Float32
    2. Quark Super Static INT8
    3. Quark Normal Float32
    4. Quark Normal Static INT8
    5. Atom Super Float32
    6. Atom Super Static INT8
    7. Atom Normal Float32
    8. Atom Normal Static INT8
- Population: 13,841 images (COCO val2017 authentic + WildFake DALL-E Advanced)
- 15 Conditions: clean, jpeg90/70/50/30, blur0.5/1.0/2.0, resize_half/quarter,
                 noise0.02/0.05/0.10, color_jitter20, crop80

Architecture for Maximum Throughput:
1. Pipelined Single-Pass I/O: Images are decoded, transformed, and normalized
   ONCE per condition across parallel CPU workers.
2. GPU Concurrent Inference: Each batch is scored across all 8 models in VRAM,
   eliminating redundant disk I/O and transform computation by 8x.
3. Hardware Acceleration: Uses ONNX Runtime with CUDAExecutionProvider on the
   NVIDIA GeForce RTX 4080 with multi-threaded pinned tensor streaming.
4. Resumability: Saves incremental condition results to disk so runs can resume.
"""

from __future__ import annotations

import argparse
import json
import logging
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

# Ensure torch CUDA DLLs are registered for ONNX Runtime CUDAExecutionProvider
_torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
if os.path.exists(_torch_lib):
    try:
        os.add_dll_directory(_torch_lib)
    except (AttributeError, OSError):
        pass
    os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")

import onnxruntime as ort

from aigc_detector.constants import Transformation
from aigc_detector.metrics import balanced_accuracy, binary_auroc, confusion_matrix, macro_f1
from aigc_detector.preprocessing import RenderPolicy, render_for_model
from aigc_detector.runtime import load_local_environment
from aigc_detector.transforms import TransformSpec, apply_transform

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_demo")

load_local_environment(_PROJECT_ROOT)


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

# ImageNet normalization parameters
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


class ParallelDemoDataset(Dataset[dict[str, Any]]):
    """Loads images, applies the condition transformation, and pre-renders pixel_values."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        data_root: Path,
        condition: ConditionSpec,
        seed: int = 42,
    ) -> None:
        self.records = records
        self.data_root = data_root
        self.condition = condition
        self.seed = seed

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

        # Pre-render to 224x224 square RGB and ImageNet normalized float32 tensor
        resized = image.resize((224, 224), Image.Resampling.BICUBIC)
        arr = np.asarray(resized, dtype=np.float32) / 255.0  # [224, 224, 3]
        tensor = np.transpose(arr, (2, 0, 1))  # [3, 224, 224]
        normalized = (tensor - _IMAGENET_MEAN) / _IMAGENET_STD

        return {
            "pixel_values": normalized,
            "ai_positive": int(row["ai_positive"]),
        }


def collate_pixel_batch(batch: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    pixel_values = np.stack([item["pixel_values"] for item in batch], axis=0)
    targets = np.array([item["ai_positive"] for item in batch], dtype=np.int64)
    return pixel_values, targets


def compute_condition_metrics(targets: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    targets_t = torch.from_numpy(targets).long()
    scores_t = torch.from_numpy(scores).float()
    preds_t = (scores_t >= 0.5).long()

    matrix = confusion_matrix(targets_t, preds_t, classes=2)
    clean_auroc = binary_auroc(targets_t.float(), scores_t)

    tp = int(((targets_t == 1) & (preds_t == 1)).sum())
    fn = int(((targets_t == 1) & (preds_t == 0)).sum())
    tn = int(((targets_t == 0) & (preds_t == 0)).sum())
    fp = int(((targets_t == 0) & (preds_t == 1)).sum())

    total = len(targets)
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


def compute_model_summary(condition_results: dict[str, Any]) -> dict[str, Any]:
    clean_res = condition_results.get("clean")
    if not clean_res:
        return {}

    clean_auroc = clean_res["auroc"]
    trans_aurocs = [
        res["auroc"] for name, res in condition_results.items() if name != "clean"
    ]

    mean_trans_auroc = float(np.mean(trans_aurocs)) if trans_aurocs else clean_auroc
    worst_trans_auroc = float(np.min(trans_aurocs)) if trans_aurocs else clean_auroc

    worst_cond = ""
    for name, res in condition_results.items():
        if name != "clean" and res["auroc"] == worst_trans_auroc:
            worst_cond = name
            break

    return {
        "clean_auroc": clean_auroc,
        "mean_transformed_auroc": mean_trans_auroc,
        "mean_delta_vs_clean": mean_trans_auroc - clean_auroc,
        "worst_transformed_auroc": worst_trans_auroc,
        "worst_condition": worst_cond,
        "clean_accuracy": clean_res["accuracy"],
        "clean_aigc_recall": clean_res["aigc_recall"],
        "clean_non_aigc_recall": clean_res["non_aigc_recall"],
        "clean_macro_f1": clean_res["macro_f1"],
    }


def generate_markdown_summary_report(
    all_models_results: dict[str, dict[str, Any]],
    output_md: Path,
) -> None:
    lines = [
        "# TechJam 2026 Track 5 — Demonstration Benchmark Evaluation",
        "",
        "## Overall Model Comparison",
        "",
        "| Model Variant | Precision | Clean AUROC | Mean Trans. AUROC | Worst Trans. AUROC | Worst Condition | Clean AIGC Rec. | Clean Non-AIGC Rec. |",
        "| :--- | :--- | :---: | :---: | :---: | :--- | :---: | :---: |",
    ]

    for model_name, data in all_models_results.items():
        summary = data.get("summary", {})
        if not summary:
            continue
        clean_auc = summary.get("clean_auroc", 0.0)
        mean_auc = summary.get("mean_transformed_auroc", 0.0)
        worst_auc = summary.get("worst_transformed_auroc", 0.0)
        worst_c = summary.get("worst_condition", "")
        aigc_rec = summary.get("clean_aigc_recall", 0.0)
        auth_rec = summary.get("clean_non_aigc_recall", 0.0)
        precision = "Static INT8" if "int8" in model_name else "Float32"

        lines.append(
            f"| **{model_name}** | {precision} | {clean_auc:.4f} | {mean_auc:.4f} | {worst_auc:.4f} | `{worst_c}` | {aigc_rec*100:.2f}% | {auth_rec*100:.2f}% |"
        )

    lines.extend(["", "---", "", "## Per-Condition AUROC Matrix", ""])
    condition_names = [c.name for c in ORGANIZER_CONDITIONS]
    header = ["| Model | " + " | ".join(condition_names) + " |"]
    divider = ["| :--- | " + " | ".join([":---:"] * len(condition_names)) + " |"]
    lines.extend(header)
    lines.extend(divider)

    for model_name, data in all_models_results.items():
        conds = data.get("conditions", {})
        row = [f"**{model_name}**"]
        for cname in condition_names:
            c_res = conds.get(cname, {})
            auc = c_res.get("auroc", 0.0)
            row.append(f"{auc:.4f}")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    output_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[REPORT] Saved markdown summary report to: {output_md}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Model Demonstration Benchmark Harness")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("metadata/organizer_demo_document_count.csv"),
        help="Path to 13841-row organizer demonstration manifest",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("E:/techjam26-runtime/data"),
        help="Root path containing images",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("outputs/models"),
        help="Directory containing ONNX models",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/demo_eval"),
        help="Directory to write reports",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Limit samples for quick test")
    parser.add_argument(
        "--conditions",
        type=str,
        nargs="+",
        default=None,
        help="Filter specific condition names",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="CUDAExecutionProvider",
        choices=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_file = args.output_dir / "evaluation_progress.json"

    # 1. Load Manifest
    df = pd.read_csv(args.manifest, low_memory=False)
    df["ai_positive"] = (df["label"] == "fully_aigc").astype(int)
    if args.limit is not None:
        df = pd.concat(
            [group.sample(n=min(args.limit // 2, len(group)), random_state=42) for _, group in df.groupby("ai_positive")],
            ignore_index=True,
        )
    records = df.to_dict(orient="records")
    print(f"Loaded {len(records)} demonstration records ({sum(r['ai_positive'] for r in records)} AIGC, {sum(1 - r['ai_positive'] for r in records)} Authentic).")

    # 2. Discover Models
    model_files = sorted(list(args.models_dir.glob("*.onnx")))
    if not model_files:
        raise FileNotFoundError(f"No ONNX models found in {args.models_dir}")

    print(f"Discovered {len(model_files)} ONNX models in {args.models_dir}:")
    for mf in model_files:
        print(f"  - {mf.name} ({mf.stat().st_size / 1e6:.1f} MB)")

    # 3. Load Sessions
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 16
    opts.log_severity_level = 3  # Quiet Memcpy warnings

    sessions: dict[str, ort.InferenceSession] = {}
    for mf in model_files:
        name = mf.stem.replace("mechadetect-", "")
        try:
            sess = ort.InferenceSession(str(mf), sess_options=opts, providers=[args.provider])
            sessions[name] = sess
            print(f"Loaded {name} with provider: {sess.get_providers()[0]}")
        except Exception as err:
            print(f"WARNING: Could not load {name} with {args.provider}: {err}. Falling back to CPU.")
            sess = ort.InferenceSession(str(mf), sess_options=opts, providers=["CPUExecutionProvider"])
            sessions[name] = sess

    # 4. Load or Initialize State
    if state_file.is_file():
        try:
            all_results = json.loads(state_file.read_text(encoding="utf-8"))
            print(f"Resuming from {state_file}...")
        except Exception:
            all_results = {name: {"conditions": {}, "summary": {}} for name in sessions}
    else:
        all_results = {name: {"conditions": {}, "summary": {}} for name in sessions}

    for name in sessions:
        if name not in all_results:
            all_results[name] = {"conditions": {}, "summary": {}}

    # 5. Conditions Loop
    target_conditions = ORGANIZER_CONDITIONS
    if args.conditions:
        target_conditions = [c for c in ORGANIZER_CONDITIONS if c.name in args.conditions]

    print("=" * 80)
    print(f"Starting Multi-Model Evaluation: {len(sessions)} models x {len(target_conditions)} conditions x {len(records)} images")
    print(f"Total passes to execute: {len(sessions) * len(target_conditions) * len(records):,}")
    print("=" * 80)

    t_global_start = time.perf_counter()

    for cond_idx, cond in enumerate(target_conditions, start=1):
        # Check if all models have completed this condition
        all_done = all(cond.name in all_results[name]["conditions"] for name in sessions)
        if all_done:
            print(f"[{cond_idx}/{len(target_conditions)}] Condition '{cond.name}' already evaluated for all models; skipping.")
            continue

        print(f"\n[{cond_idx}/{len(target_conditions)}] Evaluating Condition: {cond.name} ({cond.category}) — {cond.real_world_analog}")

        dataset = ParallelDemoDataset(records, args.data_root, cond)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers if sys.platform != "win32" else min(4, args.num_workers),
            collate_fn=collate_pixel_batch,
            pin_memory=True,
        )

        all_targets: list[np.ndarray] = []
        model_scores: dict[str, list[np.ndarray]] = {name: [] for name in sessions}

        t_cond_start = time.perf_counter()
        processed = 0

        for batch_idx, (pixel_batch, targets_batch) in enumerate(loader, start=1):
            all_targets.append(targets_batch)
            b_size = len(targets_batch)

            # Score across all models on this pre-processed batch
            for model_name, sess in sessions.items():
                if cond.name in all_results[model_name]["conditions"]:
                    continue
                probs = sess.run(["probabilities"], {"pixel_values": pixel_batch})[0]
                aigc_scores = probs[:, 1]  # P(AIGC)
                model_scores[model_name].append(aigc_scores)

            processed += b_size
            if batch_idx % 25 == 0 or batch_idx == len(loader):
                dt = time.perf_counter() - t_cond_start
                rate = processed / dt if dt > 0 else 0.0
                print(
                    f"  [{cond.name}] {processed:>5d}/{len(records)} ({100.0 * processed / len(records):4.1f}%) | "
                    f"{rate:5.1f} img/s (effective across {len(sessions)} models: {rate * len(sessions):5.1f} inf/s)",
                    flush=True,
                )

        combined_targets = np.concatenate(all_targets)

        # Compute and record metrics for each model
        for model_name in sessions:
            if cond.name in all_results[model_name]["conditions"]:
                continue
            combined_scores = np.concatenate(model_scores[model_name])
            metrics = compute_condition_metrics(combined_targets, combined_scores)
            all_results[model_name]["conditions"][cond.name] = metrics
            print(
                f"    -> {model_name:<35s} | AUROC: {metrics['auroc']:.4f} | Acc: {metrics['accuracy']:.4f} | "
                f"AIGC Rec: {metrics['aigc_recall']*100:.1f}% | Non-AIGC Rec: {metrics['non_aigc_recall']*100:.1f}%"
            )

        # Update summaries and save checkpoint
        for model_name in sessions:
            all_results[model_name]["summary"] = compute_model_summary(all_results[model_name]["conditions"])
            # Save per-model JSON
            out_file = args.output_dir / f"{model_name}_eval.json"
            out_file.write_text(json.dumps(all_results[model_name], indent=2), encoding="utf-8")

        state_file.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    # 6. Generate final markdown comparison table
    summary_md = args.output_dir / "all_models_demo_eval_summary.md"
    generate_markdown_summary_report(all_results, summary_md)

    total_time = time.perf_counter() - t_global_start
    print("=" * 80)
    print(f"DEMO BENCHMARK EVALUATION COMPLETE IN {total_time / 60:.1f} MINUTES!")
    print(f"Results written to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
