from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from aigc_detector.config import load_config
from aigc_detector.dataset import PairedImageDataset, collate_pairs
from aigc_detector.preprocessing import mask_to_token_occupancy
from aigc_detector.runtime import load_local_environment, resolve_project_path, seed_everything
from aigc_detector.train import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache frozen backbone tokens for the bake-off.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    project_root = args.config.resolve().parent.parent
    load_local_environment(project_root)
    config = load_config(args.config)
    seed_everything(int(config["seed"]))
    manifest = resolve_project_path(config["paths"]["train_manifest"], project_root)
    dataset = PairedImageDataset(
        manifest,
        data_root=config["paths"]["data_root"],
        seed=int(config["seed"]),
        chain_length_probabilities={0: 1.0},
        render_policy=config.get("preprocessing", {}).get("policy", "square_jpeg95"),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"].get("cache_batch_size", 1)),
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 0)),
        collate_fn=collate_pairs,
    )
    device = torch.device("cuda")
    model = build_model(config).to(device).eval()
    model.backbone.set_frozen(True)
    args.output.mkdir(parents=True, exist_ok=True)

    index = 0
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for batch in loader:
            sequences = model.backbone(batch["original"])
            for image, mask, provenance, tokens in zip(
                batch["original"], batch["mask"], batch["provenance"], sequences, strict=True
            ):
                target = (
                    None
                    if mask is None
                    else mask_to_token_occupancy(mask, tokens.shape[0], image.size).to(torch.bfloat16)
                )
                torch.save(
                    {
                        "tokens": tokens.detach().cpu().to(torch.bfloat16),
                        "provenance": int(provenance),
                        "token_mask_target": target,
                    },
                    args.output / f"{index:05d}.pt",
                )
                index += 1
            if index % 100 == 0:
                print(f"cached={index}/{len(dataset)}", flush=True)
    print(f"cache_complete={index} path={args.output}", flush=True)


if __name__ == "__main__":
    main()
