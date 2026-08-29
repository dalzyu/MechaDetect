from __future__ import annotations

import argparse
from pathlib import Path

import torch

from aigc_detector.config import load_config
from aigc_detector.dataset import PairedImageDataset, collate_pairs
from aigc_detector.losses import LossWeights, provenance_robustness_loss
from aigc_detector.preprocessing import mask_to_token_occupancy
from aigc_detector.runtime import load_local_environment
from aigc_detector.train import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/performance_local.yaml"))
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    project_root = args.config.resolve().parent.parent
    load_local_environment(project_root)
    config = load_config(args.config)
    manifest = project_root / config["paths"]["train_manifest"]
    dataset = PairedImageDataset(
        manifest,
        data_root=config["paths"]["data_root"],
        render_policy=config["preprocessing"]["policy"],
        transform_families=None,
    )
    masked_index = next(
        (
            index
            for index, record in enumerate(dataset.records)
            if record.tamper_mask_path is not None
        ),
        None,
    )
    indices = list(range(min(args.batch_size, len(dataset))))
    if masked_index is not None and masked_index not in indices:
        indices[-1] = masked_index
    batch = collate_pairs([dataset[index] for index in indices])
    model = build_model(config).to("cuda").train()
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        original = model(batch["original"])
        transformed = model(batch["transformed"])
        targets = [
            None if mask is None else mask_to_token_occupancy(mask, logits.numel(), image.size)
            for mask, logits, image in zip(
                batch["mask"], original.token_tamper_logits, batch["original"], strict=True
            )
        ]
        loss, components = provenance_robustness_loss(
            original,
            transformed,
            provenance=batch["provenance"].cuda(),
            token_mask_targets=targets,
            weights=LossWeights(mask_focal=0.5, mask_dice=0.5),
        )
    loss.backward()
    adapter_parameters = [
        parameter
        for name, parameter in model.backbone.named_parameters()
        if "lora_" in name and parameter.requires_grad
    ]
    adapter_gradients = sum(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in adapter_parameters
    )
    if adapter_parameters and adapter_gradients == 0:
        raise RuntimeError("LoRA is enabled but no adapter received a nonzero gradient")
    print(f"loss={loss.item():.6f}")
    print(f"sample_indices={indices} paths={batch['image_path']}")
    print(
        "components=" + " ".join(f"{key}={value.item():.5f}" for key, value in components.items())
    )
    print(f"probabilities={original.probabilities.detach().float().cpu().tolist()}")
    print(f"token_count={original.token_tamper_logits[0].numel()}")
    print(f"peak_vram_gib={torch.cuda.max_memory_allocated() / 2**30:.3f}")
    print(f"adapter_gradients={adapter_gradients}/{len(adapter_parameters)}")


if __name__ == "__main__":
    main()
