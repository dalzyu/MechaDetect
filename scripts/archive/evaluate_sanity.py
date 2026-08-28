from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from aigc_detector.config import load_config
from aigc_detector.constants import PROVENANCE_NAMES
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


@torch.inference_mode()
def evaluate(
    config_path: Path,
    checkpoint_path: Path,
    limit_per_class: int | None,
    manifest_path: Path | None = None,
) -> dict:
    project_root = config_path.resolve().parent.parent
    load_local_environment(project_root)
    config = load_config(config_path)
    model = build_model(config).to("cuda").eval()
    _load_checkpoint(model, checkpoint_path)
    dataset = PairedImageDataset(
        manifest_path or project_root / config["paths"]["val_manifest"],
        data_root=config["paths"]["data_root"],
        seed=int(config["seed"]),
    )

    seen = {name: 0 for name in PROVENANCE_NAMES}
    targets = []
    probabilities = []
    paths = []
    for record in dataset.records:
        label_name = PROVENANCE_NAMES[int(record.provenance)]
        if limit_per_class is not None and seen[label_name] >= limit_per_class:
            continue
        with Image.open(record.image_path) as source:
            image = source.convert("RGB").copy()
        output = model([image])
        probabilities.append(output.provenance_logits.softmax(dim=-1)[0].float().cpu())
        targets.append(int(record.provenance))
        paths.append(str(record.image_path))
        seen[label_name] += 1

    target_tensor = torch.tensor(targets, dtype=torch.long)
    probability_tensor = torch.stack(probabilities)
    prediction_tensor = probability_tensor.argmax(dim=-1)
    matrix = confusion_matrix(target_tensor, prediction_tensor)
    binary_mask = target_tensor != 1
    binary_target = (target_tensor[binary_mask] == 2).long()
    binary_score = probability_tensor[binary_mask, 2]
    result = {
        "count": len(targets),
        "class_counts": seen,
        "accuracy": (prediction_tensor == target_tensor).float().mean().item(),
        "macro_f1": macro_f1(matrix),
        "balanced_accuracy": balanced_accuracy(matrix),
        "macro_auroc": multiclass_macro_auroc(target_tensor, probability_tensor),
        "core_authentic_vs_fully_aigc_auroc": binary_auroc(binary_target, binary_score),
        "confusion_matrix": matrix.tolist(),
        "provenance_names": list(PROVENANCE_NAMES),
        "predictions": [
            {
                "image_path": path,
                "label": PROVENANCE_NAMES[int(label)],
                "predicted": PROVENANCE_NAMES[int(prediction)],
                "provenance": {
                    name: round(float(probability[index]), 6)
                    for index, name in enumerate(PROVENANCE_NAMES)
                },
            }
            for path, label, prediction, probability in zip(
                paths, target_tensor, prediction_tensor, probability_tensor, strict=True
            )
        ],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/poc.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-per-class", type=int)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    result = evaluate(args.config, args.checkpoint, args.limit_per_class, args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    summary = {key: value for key, value in result.items() if key != "predictions"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
