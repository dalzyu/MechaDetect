from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any

import torch
from PIL import Image

from aigc_detector.config import load_config
from aigc_detector.constants import PROVENANCE_NAMES, Transformation
from aigc_detector.dataset import PairedImageDataset
from aigc_detector.metrics import (
    balanced_accuracy,
    binary_auroc,
    confusion_matrix,
    macro_f1,
    multiclass_macro_auroc,
)
from aigc_detector.predict import _load_checkpoint
from aigc_detector.runtime import load_local_environment
from aigc_detector.train import build_model
from aigc_detector.transforms import TransformSpec, apply_transform


@dataclass(frozen=True)
class EvaluationCondition:
    name: str
    spec: TransformSpec | None


CONDITIONS = (
    EvaluationCondition("downloaded_original", None),
    EvaluationCondition("jpeg_90", TransformSpec(Transformation.JPEG, 90.0)),
    EvaluationCondition("jpeg_70", TransformSpec(Transformation.JPEG, 70.0)),
    EvaluationCondition("jpeg_50", TransformSpec(Transformation.JPEG, 50.0)),
    EvaluationCondition("jpeg_30", TransformSpec(Transformation.JPEG, 30.0)),
    EvaluationCondition("blur_05", TransformSpec(Transformation.BLUR, 0.5)),
    EvaluationCondition("blur_10", TransformSpec(Transformation.BLUR, 1.0)),
    EvaluationCondition("blur_20", TransformSpec(Transformation.BLUR, 2.0)),
    EvaluationCondition("resize_050", TransformSpec(Transformation.RESIZE, 0.5)),
    EvaluationCondition("resize_025", TransformSpec(Transformation.RESIZE, 0.25)),
    EvaluationCondition("noise_002", TransformSpec(Transformation.NOISE, 0.02)),
    EvaluationCondition("noise_005", TransformSpec(Transformation.NOISE, 0.05)),
    EvaluationCondition("noise_010", TransformSpec(Transformation.NOISE, 0.10)),
    EvaluationCondition("color_jitter", TransformSpec(Transformation.COLOR, 0.2)),
    EvaluationCondition("crop_080", TransformSpec(Transformation.CROP, 0.8)),
)


def _provenance_summary(target: torch.Tensor, probabilities: torch.Tensor) -> dict[str, Any]:
    prediction = probabilities.argmax(dim=-1)
    matrix = confusion_matrix(target, prediction)
    binary_mask = target != 1
    binary_target = (target[binary_mask] == 2).long()
    binary_score = probabilities[binary_mask, 2]
    return {
        "accuracy": (prediction == target).float().mean().item(),
        "macro_f1": macro_f1(matrix),
        "balanced_accuracy": balanced_accuracy(matrix),
        "macro_auroc": multiclass_macro_auroc(target, probabilities),
        "core_authentic_vs_fully_aigc_auroc": binary_auroc(binary_target, binary_score),
        "per_class_recall": {
            PROVENANCE_NAMES[index]: (
                matrix[index, index].item() / matrix[index, :].sum().item()
                if matrix[index, :].sum().item()
                else 0.0
            )
            for index in range(len(PROVENANCE_NAMES))
        },
        "confusion_matrix": matrix.tolist(),
    }


@torch.inference_mode()
def _run_condition(
    *,
    model: torch.nn.Module,
    records: list,
    condition: EvaluationCondition,
    condition_index: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    targets = []
    probabilities = []
    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        images = []
        for offset, record in enumerate(batch_records):
            with Image.open(record.image_path) as source:
                image = source.convert("RGB").copy()
            if condition.spec is not None:
                item_seed = seed + condition_index * 1_000_003 + start + offset
                image = apply_transform(image, condition.spec, Random(item_seed))
            images.append(image)
            targets.append(int(record.provenance))
        output = model(images)
        probabilities.append(output.provenance_logits.softmax(dim=-1).float().cpu())
        processed = min(start + batch_size, len(records))
        if processed % 100 == 0 or processed == len(records):
            print(f"{condition.name}: {processed}/{len(records)}", flush=True)
    return _provenance_summary(torch.tensor(targets), torch.cat(probabilities))


def _state_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".state.pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/poc.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--limit-per-class", type=int)
    args = parser.parse_args()

    project_root = args.config.resolve().parent.parent
    load_local_environment(project_root)
    config = load_config(args.config)
    model = build_model(config).to("cuda").eval()
    _load_checkpoint(model, args.checkpoint)
    dataset = PairedImageDataset(
        project_root / config["paths"]["val_manifest"],
        data_root=config["paths"]["data_root"],
        seed=int(config["seed"]),
    )
    records = dataset.records
    if args.limit_per_class is not None:
        counts = {index: 0 for index in range(len(PROVENANCE_NAMES))}
        selected = []
        for record in records:
            label = int(record.provenance)
            if counts[label] < args.limit_per_class:
                selected.append(record)
                counts[label] += 1
        records = selected

    state_path = _state_path(args.output)
    expected_checkpoint = str(args.checkpoint.resolve())
    conditions: dict[str, Any] = {}
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state["checkpoint"] != expected_checkpoint:
            raise RuntimeError("Resume state belongs to another checkpoint")
        conditions = state["conditions"]
        print(f"Resuming after: {', '.join(conditions)}", flush=True)

    for condition_index, condition in enumerate(CONDITIONS):
        if condition.name in conditions:
            continue
        conditions[condition.name] = _run_condition(
            model=model,
            records=records,
            condition=condition,
            condition_index=condition_index,
            batch_size=args.batch_size,
            seed=int(config["seed"]),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = state_path.with_suffix(state_path.suffix + ".tmp")
        torch.save(
            {"checkpoint": expected_checkpoint, "conditions": conditions},
            temporary,
        )
        temporary.replace(state_path)
        args.output.write_text(
            json.dumps(
                {
                    "status": "running",
                    "completed_conditions": list(conditions),
                    "condition_metrics": conditions,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    transformed = {
        name: metrics for name, metrics in conditions.items() if name != "downloaded_original"
    }
    result = {
        "status": "complete",
        "checkpoint": expected_checkpoint,
        "validation_images": len(records),
        "visual_token_budget": config["model"]["visual_tokens"],
        "conditions": conditions,
        "robustness_summary": {
            "average_accuracy": sum(item["accuracy"] for item in transformed.values())
            / len(transformed),
            "average_macro_auroc": sum(item["macro_auroc"] for item in transformed.values())
            / len(transformed),
            "worst_accuracy_condition": min(
                ((name, item["accuracy"]) for name, item in transformed.items()),
                key=lambda item: item[1],
            ),
        },
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["robustness_summary"], indent=2))


if __name__ == "__main__":
    main()
