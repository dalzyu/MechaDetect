from __future__ import annotations

import argparse
import json
from pathlib import Path
from random import Random

import torch
from PIL import Image

from .adaptation import load_trainable_encoder_state
from .config import load_config
from .constants import PROVENANCE_NAMES, Transformation
from .ema import load_ema_parameters
from .model import hierarchical_probabilities
from .preprocessing import RenderPolicy, render_for_model
from .runtime import load_local_environment
from .train import build_model
from .transforms import TransformSpec, apply_transform_chain

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _load_checkpoint(model: torch.nn.Module, path: Path) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    incompatible = model.heads.load_state_dict(payload["heads"], strict=False)
    if incompatible.missing_keys:
        raise RuntimeError(f"Checkpoint is missing provenance weights: {incompatible.missing_keys}")
    if payload.get("encoder_trainable"):
        if "encoder_trainable_state" in payload:
            load_trainable_encoder_state(model.backbone.encoder, payload["encoder_trainable_state"])
        else:
            model.backbone.encoder.load_state_dict(payload["encoder"])
    if model.spectral is not None:
        model.spectral.load_state_dict(payload["spectral"])
        model.aigc_gate.load_state_dict(payload["aigc_gate"])
        model.tamper_gate.load_state_dict(payload["tamper_gate"])
    if "ema" in payload:
        load_ema_parameters(model, payload["ema"])


@torch.inference_mode()
def predict_directory(
    input_dir: Path,
    output_path: Path,
    config_path: Path,
    checkpoint_path: Path,
    min_confidence: float | None,
    preprocess_inputs: bool,
    three_view: bool,
) -> None:
    project_root = config_path.resolve().parent.parent
    load_local_environment(project_root)
    config = load_config(config_path)
    model = build_model(config).to("cuda").eval()
    _load_checkpoint(model, checkpoint_path)
    policy = RenderPolicy(config.get("preprocessing", {}).get("policy", "square_jpeg95"))

    predictions = []
    paths = sorted(path for path in input_dir.rglob("*") if path.suffix.lower() in _IMAGE_SUFFIXES)
    for path in paths:
        with Image.open(path) as source:
            image = source.convert("RGB").copy()
        if preprocess_inputs:
            image = render_for_model(image, policy, rng=Random(42))
        views = [image]
        if three_view:
            views.extend(
                (
                    apply_transform_chain(
                        image,
                        (TransformSpec(Transformation.JPEG, 90.0),),
                        Random(90),
                    ),
                    apply_transform_chain(
                        image,
                        (TransformSpec(Transformation.RESIZE, 0.8),),
                        Random(80),
                    ),
                )
            )
        outputs = [model([view]) for view in views]
        provenance = (
            hierarchical_probabilities(
                torch.stack([output.aigc_logit.float() for output in outputs]).mean(0),
                torch.stack([output.tamper_logit.float() for output in outputs]).mean(0),
            )[0]
            .float()
            .cpu()
        )
        top_confidence, top_class = provenance.max(dim=0)
        if min_confidence is not None and top_confidence.item() < min_confidence:
            continue

        predictions.append(
            {
                "image_path": str(path),
                "pred": round(provenance[2].item(), 6),
                "provenance_pred": PROVENANCE_NAMES[top_class.item()],
                "provenance": {
                    name: round(provenance[index].item(), 6)
                    for index, name in enumerate(PROVENANCE_NAMES)
                },
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(predictions, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/performance_local.yaml"))
    parser.add_argument("--min-confidence", type=float)
    parser.add_argument(
        "--raw-input",
        action="store_true",
        help="Skip the preprocessing policy stored in the training config",
    )
    parser.add_argument("--three-view", action="store_true")
    args = parser.parse_args()
    predict_directory(
        args.input,
        args.output,
        args.config,
        args.checkpoint,
        args.min_confidence,
        not args.raw_input,
        args.three_view,
    )


if __name__ == "__main__":
    main()
