from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections.abc import Callable
from pathlib import Path
from random import Random

import pandas as pd
import torch
from PIL import Image

from aigc_detector.config import load_config
from aigc_detector.constants import PROVENANCE_NAMES, Provenance, Transformation
from aigc_detector.dataset import load_manifest_frame, parse_provenance
from aigc_detector.metrics import binary_auroc
from aigc_detector.model import ai_generated_probability, binary_probabilities
from aigc_detector.predict import _load_checkpoint
from aigc_detector.preprocessing import RenderPolicy, render_for_model
from aigc_detector.runtime import load_local_environment
from aigc_detector.train import build_model
from aigc_detector.transforms import TransformSpec, apply_transform


def _jpeg90(image: Image.Image) -> Image.Image:
    return apply_transform(image, TransformSpec(Transformation.JPEG, 90.0), Random(900))


def _alternate_resolution(image: Image.Image) -> Image.Image:
    return apply_transform(image, TransformSpec(Transformation.RESIZE, 0.8), Random(800))


ROBUSTNESS_CONDITIONS: dict[str, TransformSpec | None] = {
    "clean": None,
    "jpeg90": TransformSpec(Transformation.JPEG, 90.0),
    "jpeg70": TransformSpec(Transformation.JPEG, 70.0),
    "jpeg50": TransformSpec(Transformation.JPEG, 50.0),
    "jpeg30": TransformSpec(Transformation.JPEG, 30.0),
    "blur0.5": TransformSpec(Transformation.BLUR, 0.5),
    "blur1": TransformSpec(Transformation.BLUR, 1.0),
    "blur2": TransformSpec(Transformation.BLUR, 2.0),
    "resize_half": TransformSpec(Transformation.RESIZE, 0.5),
    "resize_quarter": TransformSpec(Transformation.RESIZE, 0.25),
    "noise0.02": TransformSpec(Transformation.NOISE, 0.02),
    "noise0.05": TransformSpec(Transformation.NOISE, 0.05),
    "noise0.10": TransformSpec(Transformation.NOISE, 0.10),
    "color20": TransformSpec(Transformation.COLOR, 0.20),
    "crop80": TransformSpec(Transformation.CROP, 0.80),
}


def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def summarize(
    target: torch.Tensor,
    probabilities: torch.Tensor,
    *,
    ai_target: torch.Tensor | None = None,
) -> dict[str, object]:
    """Summarize binary Track 5 performance using explicit AI-positive labels."""
    from aigc_detector.metrics import balanced_accuracy, binary_auroc, confusion_matrix

    binary_target = (
        ai_target.bool()
        if ai_target is not None
        else target != int(Provenance.AUTHENTIC)
    )
    binary_score = ai_generated_probability(probabilities)
    binary_prediction = binary_score >= 0.5
    matrix = confusion_matrix(binary_target, binary_prediction, classes=2)
    true_positive = int((binary_target & binary_prediction).sum())
    false_negative = int((binary_target & ~binary_prediction).sum())
    true_negative = int((~binary_target & ~binary_prediction).sum())
    false_positive = int((~binary_target & binary_prediction).sum())
    positive_total = true_positive + false_negative
    negative_total = true_negative + false_positive
    binary_recall = true_positive / positive_total if positive_total else None
    binary_specificity = true_negative / negative_total if negative_total else None
    binary_precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else None
    )
    binary_f1 = (
        2 * binary_precision * binary_recall / (binary_precision + binary_recall)
        if binary_precision is not None
        and binary_recall is not None
        and binary_precision + binary_recall
        else None
    )
    authentic_total = int((~binary_target).sum())
    return {
        "n": len(target),
        "accuracy": (true_positive + true_negative) / len(target),
        "binary_accuracy": (true_positive + true_negative) / len(target),
        "binary_balanced_accuracy": (
            (binary_recall + binary_specificity) / 2
            if binary_recall is not None and binary_specificity is not None
            else None
        ),
        "binary_recall": binary_recall,
        "binary_specificity": binary_specificity,
        "binary_precision": binary_precision,
        "binary_f1": binary_f1,
        "balanced_accuracy": balanced_accuracy(matrix),
        "ai_positive_auroc": _finite(binary_auroc(binary_target.float(), binary_score)),
        "authentic_recall": (
            true_negative / authentic_total if authentic_total else None
        ),
        "binary_confusion_matrix": matrix.tolist(),
        "confusion_matrix": matrix.tolist(),
    }


def bootstrap_auroc_ci(
    target: torch.Tensor,
    score: torch.Tensor,
    *,
    groups: list[str] | None = None,
    iterations: int = 200,
    seed: int = 42,
) -> list[float] | None:
    """Bootstrap AUROC by duplicate group, never by correlated rows."""
    generator = torch.Generator().manual_seed(seed)
    if groups is None:
        group_values = [str(index) for index in range(len(target))]
    else:
        if len(groups) != len(target):
            raise ValueError("groups must have one value per target")
        group_values = groups
    unique_groups = list(dict.fromkeys(group_values))
    group_indices = [
        torch.tensor([index for index, value in enumerate(group_values) if value == group])
        for group in unique_groups
    ]
    values = []
    for _ in range(iterations):
        selected = torch.randint(len(group_indices), (len(group_indices),), generator=generator)
        indices = torch.cat([group_indices[int(index)] for index in selected])
        value = binary_auroc(target[indices], score[indices])
        if math.isfinite(value):
            values.append(value)
    if len(values) < iterations // 2:
        return None
    values.sort()
    return [values[round(0.025 * (len(values) - 1))], values[round(0.975 * (len(values) - 1))]]


@torch.inference_mode()
def infer_batch_views(
    model: torch.nn.Module,
    images: list[Image.Image],
    *,
    three_view: bool,
) -> torch.Tensor:
    view_functions: list[Callable[[Image.Image], Image.Image]] = [lambda value: value]
    if three_view:
        view_functions.extend((_jpeg90, _alternate_resolution))
    ai_positive_logits = []
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for function in view_functions:
            output = model([function(image) for image in images])
            ai_positive_logits.append(output.ai_positive_logit.float())
    return binary_probabilities(torch.stack(ai_positive_logits).mean(0)).cpu()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/teacher_dinov3_stage2_paired_unfrozen.yaml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--three-view", action="store_true")
    parser.add_argument("--robustness", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--per-class", type=int)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--deduplicate-content",
        action="store_true",
        help="Infer each sha256 content once while preserving duplicate row weighting in metrics",
    )
    parser.add_argument(
        "--state",
        type=Path,
        help="Append completed rows here so a long evaluation can resume after interruption",
    )
    args = parser.parse_args()

    project_root = args.config.resolve().parent.parent
    load_local_environment(project_root)
    config = load_config(args.config)
    data_root = Path(
        config["paths"].get("data_root") or os.environ.get("TECHJAM_DATA_ROOT", "data")
    )
    frame = load_manifest_frame(args.manifest).fillna("")
    if args.per_class:
        frame = pd.concat(
            [
                group.sample(n=min(args.per_class, len(group)), random_state=42)
                for _, group in frame.groupby("label")
            ],
            ignore_index=True,
        ).sample(frac=1.0, random_state=42)
    if args.limit:
        frame = frame.iloc[: args.limit]
    if args.deduplicate_content:
        content_key = "sha256" if "sha256" in frame.columns else "image_path"
        frame["_eval_weight"] = frame.groupby(content_key)[content_key].transform("size")
        frame = frame.drop_duplicates(content_key, keep="first").reset_index(drop=True)
    model = build_model(config).to("cuda").eval()
    _load_checkpoint(model, args.checkpoint)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    inference_started = time.perf_counter()
    policy = RenderPolicy(config.get("preprocessing", {}).get("policy", "square_jpeg95"))
    conditions = ROBUSTNESS_CONDITIONS if args.robustness else {"clean": None}
    condition_probabilities: dict[str, list[torch.Tensor]] = {name: [] for name in conditions}
    targets: list[int] = []
    ai_targets: list[int] = []
    weights: list[int] = []
    output_rows: list[dict[str, object]] = []
    records = frame.to_dict(orient="records")
    state_path = args.state
    completed: dict[int, dict[str, object]] = {}
    state_metadata = {
        "manifest": str(args.manifest.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "config": str(args.config.resolve()),
        "conditions": list(conditions),
        "three_view": bool(args.three_view),
        "rows": len(records),
    }
    state_handle = None
    if state_path is not None and state_path.exists():
        with state_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                if "_meta" in item:
                    expected = state_metadata
                    if item["_meta"] != expected:
                        raise RuntimeError("Evaluation state does not match this manifest/config")
                else:
                    completed[int(item["_index"])] = item["row"]
        for index in sorted(completed):
            row = completed[index]
            output_rows.append(row)
            target = PROVENANCE_NAMES.index(str(row["target"]))
            targets.append(target)
            ai_targets.append(
                int(row.get("ai_positive", target != int(Provenance.AUTHENTIC)))
            )
            for condition in conditions:
                condition_probabilities[condition].append(
                    torch.tensor(row["probabilities"][condition], dtype=torch.float32)
                )
            weights.append(int(row.get("weight", 1)))
    if state_path is not None:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_handle = state_path.open("a", encoding="utf-8")
        if not completed:
            state_handle.write(
                json.dumps(
                    {
                        "_meta": state_metadata
                    }
                )
                + "\n"
            )
            state_handle.flush()
    for batch_start in range(0, len(records), args.batch_size):
        batch_rows = records[batch_start : batch_start + args.batch_size]
        batch_indices = range(batch_start, batch_start + len(batch_rows))
        if state_path is not None and all(index in completed for index in batch_indices):
            continue
        rendered_images = []
        new_state_rows = []
        for offset, row in enumerate(batch_rows):
            path = Path(str(row["image_path"]).replace("\\", "/"))
            if not path.is_absolute():
                path = data_root / path
            if not path.is_file():
                from aigc_detector.dataset import try_fetch_image_from_hub
                fetched = try_fetch_image_from_hub(str(row.get("dataset", "")), path)
                if fetched is not None and fetched.is_file():
                    path = fetched
            with Image.open(path) as source:
                downloaded = source.convert("RGB").copy()
            rendered_images.append(
                render_for_model(downloaded, policy, rng=Random(42 + batch_start + offset))
            )
        batch_condition_probabilities: dict[str, torch.Tensor] = {}
        for condition, transform in conditions.items():
            conditioned = [
                image
                if transform is None
                else apply_transform(
                    image, transform, Random(1000 + batch_start + offset)
                )
                for offset, image in enumerate(rendered_images)
            ]
            probability = infer_batch_views(
                model,
                conditioned,
                three_view=args.three_view and condition == "clean",
            )
            batch_condition_probabilities[condition] = probability
            condition_probabilities[condition].extend(list(probability))
        for offset, row in enumerate(batch_rows):
            row_probabilities = {
                condition: values[offset].tolist()
                for condition, values in batch_condition_probabilities.items()
            }
            target = int(parse_provenance(row["label"], str(row["dataset"])))
            raw_ai_positive = row.get("ai_positive", "")
            ai_positive = (
                int(raw_ai_positive)
                if str(raw_ai_positive).strip() != ""
                else int(target != int(Provenance.AUTHENTIC))
            )
            targets.append(target)
            ai_targets.append(ai_positive)
            output_row = {
                "image_path": str(row["image_path"]),
                "dataset": str(row["dataset"]),
                "generator": str(row.get("generator_family", row.get("generator", ""))),
                "target": PROVENANCE_NAMES[target],
                "ai_positive": ai_positive,
                "probabilities": row_probabilities,
                "weight": int(row.get("_eval_weight", 1)),
                "duplicate_group": str(
                    row.get("duplicate_group", row.get("sha256", row["image_path"]))
                ),
            }
            output_rows.append(output_row)
            weights.append(int(row.get("_eval_weight", 1)))
            new_state_rows.append({"_index": batch_start + offset, "row": output_row})
        progress = min(batch_start + args.batch_size, len(frame))
        if progress % 50 < args.batch_size or progress == len(frame):
            print(f"evaluated {progress}/{len(frame)}", flush=True)

    target_tensor = torch.tensor(targets)
    ai_target_tensor = torch.tensor(ai_targets, dtype=torch.long)
    weight_tensor = torch.tensor(weights, dtype=torch.long)
    expanded_indices = torch.repeat_interleave(torch.arange(len(target_tensor)), weight_tensor)
    metric_target = target_tensor[expanded_indices]
    metric_ai_target = ai_target_tensor[expanded_indices]
    metric_groups = [
        str(output_rows[index].get("duplicate_group", output_rows[index]["image_path"]))
        for index in expanded_indices.tolist()
    ]
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_started
    metrics: dict[str, object] = {"conditions": {}}
    for condition, values in condition_probabilities.items():
        probabilities = torch.stack(values)
        metric_probabilities = probabilities[expanded_indices]
        condition_metrics = summarize(
            metric_target, metric_probabilities, ai_target=metric_ai_target
        )
        condition_metrics["ai_positive_auroc_95ci"] = bootstrap_auroc_ci(
            metric_ai_target.float(),
            ai_generated_probability(metric_probabilities),
            groups=metric_groups,
        )
        metrics["conditions"][condition] = condition_metrics
    clean_auroc = metrics["conditions"]["clean"]["ai_positive_auroc"]
    for condition_metrics in metrics["conditions"].values():
        transformed_auroc = condition_metrics["ai_positive_auroc"]
        condition_metrics["ai_positive_auroc_drop"] = (
            clean_auroc - transformed_auroc
            if clean_auroc is not None and transformed_auroc is not None
            else None
        )
    clean = torch.stack(condition_probabilities["clean"])
    metric_clean = clean[expanded_indices]
    metrics["per_dataset"] = {}
    dataset_values = [str(row.get("dataset", "")) for row in output_rows]
    metric_dataset_values = [dataset_values[index] for index in expanded_indices.tolist()]
    for group in sorted(set(metric_dataset_values)):
        indices = torch.tensor([value == group for value in metric_dataset_values])
        if int(indices.sum()) >= 2:
            metrics["per_dataset"][group] = summarize(
                metric_target[indices],
                metric_clean[indices],
                ai_target=metric_ai_target[indices],
            )

    generator_values = [str(row.get("generator", "")) for row in output_rows]
    metric_generator_values = [generator_values[index] for index in expanded_indices.tolist()]
    metrics["per_ai_generator"] = {}
    authentic = metric_ai_target == 0
    for group in sorted(set(metric_generator_values)):
        generator_mask = torch.tensor(
            [value == group for value in metric_generator_values]
        )
        positives = generator_mask & ~authentic
        comparison = positives | authentic
        if positives.any() and authentic.any():
            metrics["per_ai_generator"][group] = summarize(
                metric_target[comparison],
                metric_clean[comparison],
                ai_target=metric_ai_target[comparison],
            )
    family_aurocs = [
        values["ai_positive_auroc"]
        for values in metrics["per_ai_generator"].values()
        if values["ai_positive_auroc"] is not None
    ]
    primary_transformed_aurocs = [
        values["ai_positive_auroc"]
        for name, values in metrics["conditions"].items()
        if name != "clean" and values["ai_positive_auroc"] is not None
    ]
    metrics["selection_summary"] = {
        "mean_generator_auroc": sum(family_aurocs) / len(family_aurocs) if family_aurocs else None,
        "worst_generator_auroc": min(family_aurocs) if family_aurocs else None,
        "mean_transformed_auroc": sum(primary_transformed_aurocs) / len(primary_transformed_aurocs)
        if primary_transformed_aurocs
        else None,
        "worst_transformed_auroc": min(primary_transformed_aurocs)
        if primary_transformed_aurocs
        else None,
    }
    metrics["efficiency"] = {
        "images": int(weight_tensor.sum()),
        "inferred_unique_images": len(frame),
        "views_on_clean": 3 if args.three_view else 1,
        "conditions": len(conditions),
        "inference_seconds": inference_seconds,
        "images_per_second": len(frame) / inference_seconds,
        "milliseconds_per_image": inference_seconds * 1000.0 / len(frame),
        "peak_allocated_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    metrics["three_view"] = args.three_view
    metrics["checkpoint"] = str(args.checkpoint)
    metrics["rows"] = output_rows
    if state_handle is not None:
        state_handle.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics["conditions"], indent=2))


if __name__ == "__main__":
    main()
