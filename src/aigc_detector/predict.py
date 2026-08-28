from __future__ import annotations

"""Inference and batch directory prediction pipeline for image provenance detection."""

import argparse
import json
from contextlib import nullcontext
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
    """Load model heads, adapter, and encoder weights from a saved training checkpoint."""
    payload = torch.load(path, map_location="cpu", weights_only=False)

    incompatible = model.heads.load_state_dict(payload["heads"], strict=False)
    if incompatible.missing_keys:
        raise RuntimeError(f"Checkpoint is missing provenance weights: {incompatible.missing_keys}")

    adapter_state = payload.get("token_adapter", {})
    if model.token_adapter.state_dict() or adapter_state:
        model.token_adapter.load_state_dict(adapter_state)

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
    min_confidence: float | None = None,
    preprocess_inputs: bool = True,
    three_view: bool = False,
) -> None:
    """Run batch provenance inference over all supported images in `input_dir`.

    Outputs a JSON list of predictions containing:
    - `image_path`: Relative or absolute path to the input image
    - `pred`: P(fully_aigc) probability float in [0.0, 1.0] for competition submission
    - `provenance_pred`: Predicted argmax class name ("authentic", "tampered", "fully_aigc")
    - `provenance`: Dict of individual class probabilities summing to 1.0
    """
    project_root = config_path.resolve().parent.parent
    load_local_environment(project_root)
    config = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device).eval()
    _load_checkpoint(model, checkpoint_path)

    policy = RenderPolicy(config.get("preprocessing", {}).get("policy", "square_jpeg95"))
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )

    predictions = []
    paths = sorted(path for path in input_dir.rglob("*") if path.suffix.lower() in _IMAGE_SUFFIXES)

    for path in paths:
        with Image.open(path) as source:
            image = source.convert("RGB").copy()

        if preprocess_inputs:
            image = render_for_model(image, policy, rng=Random(42))

        views = [image]
        if three_view:
            # Multi-view test-time augmentation: clean + JPEG90 + resize 0.8
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

        with autocast:
            outputs = [model([view]) for view in views]

        # Aggregate multi-view logits and compute hierarchical probabilities
        mean_aigc_logit = torch.stack([out.aigc_logit.float() for out in outputs]).mean(0)
        mean_tamper_logit = torch.stack([out.tamper_logit.float() for out in outputs]).mean(0)
        provenance = hierarchical_probabilities(mean_aigc_logit, mean_tamper_logit)[0].cpu()

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
    """CLI entrypoint for running batch image prediction."""
    parser = argparse.ArgumentParser(description="Predict provenance probabilities across an image directory.")
    parser.add_argument("--input", type=Path, required=True, help="Input directory containing images.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint path.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/teacher_dinov3_production.yaml"),
        help="Configuration YAML path.",
    )
    parser.add_argument("--min-confidence", type=float, help="Optional minimum confidence threshold.")
    parser.add_argument(
        "--raw-input",
        action="store_true",
        help="Skip the preprocessing policy stored in the training config.",
    )
    parser.add_argument(
        "--three-view",
        action="store_true",
        help="Enable multi-view test-time augmentation (clean + JPEG90 + resize0.8).",
    )
    args = parser.parse_args()

    predict_directory(
        input_dir=args.input,
        output_path=args.output,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        min_confidence=args.min_confidence,
        preprocess_inputs=not args.raw_input,
        three_view=args.three_view,
    )
    print(f"Predictions written to {args.output}")


if __name__ == "__main__":
    main()
