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
from aigc_detector.dataset import parse_provenance
from aigc_detector.metrics import balanced_accuracy, binary_auroc, confusion_matrix, macro_f1
from aigc_detector.model import hierarchical_probabilities
from aigc_detector.predict import _load_checkpoint
from aigc_detector.preprocessing import RenderPolicy, render_for_model
from aigc_detector.runtime import load_local_environment
from aigc_detector.train import build_model
from aigc_detector.transforms import TransformSpec, apply_transform_chain


def _jpeg90(image: Image.Image) -> Image.Image:
    return apply_transform_chain(image, (TransformSpec(Transformation.JPEG, 90.0),), Random(900))


def _alternate_resolution(image: Image.Image) -> Image.Image:
    return apply_transform_chain(image, (TransformSpec(Transformation.RESIZE, 0.8),), Random(800))


ROBUSTNESS_CONDITIONS: dict[str, tuple[TransformSpec, ...]] = {
    "clean": (),
    "jpeg90": (TransformSpec(Transformation.JPEG, 90.0),),
    "jpeg70": (TransformSpec(Transformation.JPEG, 70.0),),
    "jpeg50": (TransformSpec(Transformation.JPEG, 50.0),),
    "jpeg30": (TransformSpec(Transformation.JPEG, 30.0),),
    "blur0.5": (TransformSpec(Transformation.BLUR, 0.5),),
    "blur1": (TransformSpec(Transformation.BLUR, 1.0),),
    "blur2": (TransformSpec(Transformation.BLUR, 2.0),),
    "resize_half": (TransformSpec(Transformation.RESIZE, 0.5),),
    "resize_quarter": (TransformSpec(Transformation.RESIZE, 0.25),),
    "noise0.02": (TransformSpec(Transformation.NOISE, 0.02),),
    "noise0.05": (TransformSpec(Transformation.NOISE, 0.05),),
    "noise0.10": (TransformSpec(Transformation.NOISE, 0.10),),
    "color20": (TransformSpec(Transformation.COLOR, 0.20),),
    "crop80": (TransformSpec(Transformation.CROP, 0.80),),
    "crop_resize_jpeg": (
        TransformSpec(Transformation.CROP, 0.8),
        TransformSpec(Transformation.RESIZE, 0.5),
        TransformSpec(Transformation.JPEG, 70.0),
    ),
}


def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def summarize(target: torch.Tensor, probabilities: torch.Tensor) -> dict[str, object]:
    prediction = probabilities.argmax(-1)
    matrix = confusion_matrix(target, prediction)
    non_aigc = target != int(Provenance.FULLY_AIGC)
    authentic_total = int((target == int(Provenance.AUTHENTIC)).sum())
    authentic_correct = int(
        ((target == int(Provenance.AUTHENTIC)) & (prediction == int(Provenance.AUTHENTIC))).sum()
    )
    return {
        "n": len(target),
        "accuracy": float((prediction == target).float().mean()),
        "balanced_accuracy": balanced_accuracy(matrix),
        "macro_f1": macro_f1(matrix),
        "fully_aigc_auroc": _finite(
            binary_auroc((target == int(Provenance.FULLY_AIGC)).float(), probabilities[:, 2])
        ),
        "tamper_auroc": _finite(
            binary_auroc(
                (target[non_aigc] == int(Provenance.TAMPERED)).float(),
                probabilities[non_aigc, 1],
            )
        ),
        "authentic_recall": authentic_correct / authentic_total if authentic_total else None,
        "confusion_matrix": matrix.tolist(),
    }


def bootstrap_auroc_ci(
    target: torch.Tensor,
    score: torch.Tensor,
    *,
    iterations: int = 200,
    seed: int = 42,
) -> list[float] | None:
    generator = torch.Generator().manual_seed(seed)
    values = []
    for _ in range(iterations):
        indices = torch.randint(len(target), (len(target),), generator=generator)
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
    aigc_logits = []
    tamper_logits = []
    for function in view_functions:
        output = model([function(image) for image in images])
        aigc_logits.append(output.aigc_logit.float())
        tamper_logits.append(output.tamper_logit.float())
    return hierarchical_probabilities(
        torch.stack(aigc_logits).mean(0), torch.stack(tamper_logits).mean(0)
    ).cpu()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/performance_local.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--three-view", action="store_true")
    parser.add_argument("--robustness", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--per-class", type=int)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    project_root = args.config.resolve().parent.parent
    load_local_environment(project_root)
    config = load_config(args.config)
    data_root = Path(os.environ["TECHJAM_DATA_ROOT"])
    frame = pd.read_csv(args.manifest).fillna("")
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
    model = build_model(config).to("cuda").eval()
    _load_checkpoint(model, args.checkpoint)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    inference_started = time.perf_counter()
    policy = RenderPolicy(config.get("preprocessing", {}).get("policy", "square_jpeg95"))
    conditions = ROBUSTNESS_CONDITIONS if args.robustness else {"clean": ()}
    condition_probabilities: dict[str, list[torch.Tensor]] = {name: [] for name in conditions}
    targets = []
    output_rows = []
    records = frame.to_dict(orient="records")
    for batch_start in range(0, len(records), args.batch_size):
        batch_rows = records[batch_start : batch_start + args.batch_size]
        rendered_images = []
        for offset, row in enumerate(batch_rows):
            path = Path(str(row["image_path"]))
            if not path.is_absolute():
                path = data_root / path
            with Image.open(path) as source:
                downloaded = source.convert("RGB").copy()
            rendered_images.append(
                render_for_model(downloaded, policy, rng=Random(42 + batch_start + offset))
            )
        batch_condition_probabilities: dict[str, torch.Tensor] = {}
        for condition, chain in conditions.items():
            conditioned = [
                apply_transform_chain(image, chain, Random(1000 + batch_start + offset))
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
            targets.append(target)
            output_rows.append(
                {
                    "image_path": str(row["image_path"]),
                    "dataset": str(row["dataset"]),
                    "generator": str(row.get("generator_family", row.get("generator", ""))),
                    "target": PROVENANCE_NAMES[target],
                    "probabilities": row_probabilities,
                }
            )
        progress = min(batch_start + args.batch_size, len(records))
        if progress % 50 < args.batch_size or progress == len(frame):
            print(f"evaluated {progress}/{len(frame)}", flush=True)

    target_tensor = torch.tensor(targets)
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_started
    metrics: dict[str, object] = {"conditions": {}}
    for condition, values in condition_probabilities.items():
        probabilities = torch.stack(values)
        condition_metrics = summarize(target_tensor, probabilities)
        condition_metrics["fully_aigc_auroc_95ci"] = bootstrap_auroc_ci(
            (target_tensor == int(Provenance.FULLY_AIGC)).float(), probabilities[:, 2]
        )
        metrics["conditions"][condition] = condition_metrics
    clean_auroc = metrics["conditions"]["clean"]["fully_aigc_auroc"]
    for condition_metrics in metrics["conditions"].values():
        transformed_auroc = condition_metrics["fully_aigc_auroc"]
        condition_metrics["fully_aigc_auroc_drop"] = (
            clean_auroc - transformed_auroc
            if clean_auroc is not None and transformed_auroc is not None
            else None
        )
    clean = torch.stack(condition_probabilities["clean"])
    metrics["per_dataset"] = {}
    dataset_values = [str(row.get("dataset", "")) for row in output_rows]
    for group in sorted(set(dataset_values)):
        indices = torch.tensor([value == group for value in dataset_values])
        if int(indices.sum()) >= 2:
            metrics["per_dataset"][group] = summarize(target_tensor[indices], clean[indices])

    generator_values = [str(row.get("generator", "")) for row in output_rows]
    metrics["per_aigc_generator"] = {}
    non_aigc = target_tensor != int(Provenance.FULLY_AIGC)
    for group in sorted(set(generator_values)):
        generator_mask = torch.tensor([value == group for value in generator_values])
        positives = generator_mask & ~non_aigc
        comparison = positives | non_aigc
        if positives.any() and non_aigc.any():
            metrics["per_aigc_generator"][group] = summarize(
                target_tensor[comparison], clean[comparison]
            )
    family_aurocs = [
        values["fully_aigc_auroc"]
        for values in metrics["per_aigc_generator"].values()
        if values["fully_aigc_auroc"] is not None
    ]
    primary_transformed_aurocs = [
        values["fully_aigc_auroc"]
        for name, values in metrics["conditions"].items()
        if name not in {"clean", "crop_resize_jpeg"} and values["fully_aigc_auroc"] is not None
    ]
    chain_auroc = metrics["conditions"].get("crop_resize_jpeg", {}).get("fully_aigc_auroc")
    metrics["selection_summary"] = {
        "mean_generator_auroc": sum(family_aurocs) / len(family_aurocs) if family_aurocs else None,
        "worst_generator_auroc": min(family_aurocs) if family_aurocs else None,
        "mean_transformed_auroc": sum(primary_transformed_aurocs) / len(primary_transformed_aurocs)
        if primary_transformed_aurocs
        else None,
        "worst_transformed_auroc": min(primary_transformed_aurocs)
        if primary_transformed_aurocs
        else None,
        "secondary_chain_auroc": chain_auroc,
    }
    metrics["efficiency"] = {
        "images": len(frame),
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics["conditions"], indent=2))


if __name__ == "__main__":
    main()
