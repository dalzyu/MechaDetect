from __future__ import annotations

from pathlib import Path
from random import Random

import torch
from PIL import Image

from aigc_detector.config import load_config
from aigc_detector.constants import Transformation
from aigc_detector.losses import provenance_robustness_loss
from aigc_detector.runtime import load_local_environment
from aigc_detector.train import build_model
from aigc_detector.transforms import TransformSpec, apply_transform


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    load_local_environment(project_root)
    config = load_config(project_root / "configs" / "poc.yaml")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = build_model(config).to("cuda").train()

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    grayscale = Image.effect_noise((1024, 768), sigma=48)
    flipped = grayscale.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    original = Image.merge("RGB", (grayscale, flipped, grayscale))
    transformed = apply_transform(
        original,
        TransformSpec(Transformation.JPEG, 50.0),
        Random(42),
    )

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        original_output = model([original])
        transformed_output = model([transformed])
        loss, components = provenance_robustness_loss(
            original_output,
            transformed_output,
            provenance=torch.tensor([0], device="cuda"),
        )
    loss.backward()

    peak_gib = torch.cuda.max_memory_allocated() / 1024**3
    print(f"total_parameters={total_parameters:,}")
    print(f"trainable_parameters={trainable_parameters:,}")
    print(f"visual_token_budget={config['model']['visual_tokens']}")
    print(f"provenance_shape={tuple(original_output.provenance_logits.shape)}")
    print(f"loss={loss.detach().item():.6f}")
    print(
        "components="
        + ",".join(f"{name}:{value.detach().item():.6f}" for name, value in components.items())
    )
    print(f"peak_cuda_allocated_gib={peak_gib:.3f}")


if __name__ == "__main__":
    main()
