from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from aigc_detector.config import load_config
from aigc_detector.dataset import PairedImageDataset, collate_pairs
from aigc_detector.preprocessing import mask_to_token_occupancy
from aigc_detector.runtime import load_local_environment, resolve_project_path, seed_everything
from aigc_detector.train import build_model


class IndexedShard(Dataset):
    def __init__(
        self,
        dataset: PairedImageDataset,
        shard_index: int,
        num_shards: int,
        cache_dir: Path | None = None,
    ) -> None:
        self.dataset = dataset
        all_indices = list(range(shard_index, len(dataset), num_shards))
        if cache_dir is not None and cache_dir.exists():
            existing = {int(p.stem) for p in cache_dir.glob("*.pt")}
            self.indices = [idx for idx in all_indices if idx not in existing]
        else:
            self.indices = all_indices
    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        source_index = self.indices[index]
        return source_index, self.dataset[source_index]


def collate_indexed(samples):
    indices, records = zip(*samples, strict=True)
    batch = collate_pairs(list(records))
    batch["indices"] = list(indices)
    return batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache frozen backbone tokens for the bake-off.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
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
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    args.output.mkdir(parents=True, exist_ok=True)
    shard = IndexedShard(dataset, args.shard_index, args.num_shards, cache_dir=args.output)
    print(
        f"shard {args.shard_index}/{args.num_shards} missing items to compute: {len(shard)}",
        flush=True,
    )
    loader = DataLoader(
        shard,
        batch_size=int(config["training"].get("cache_batch_size", 1)),
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 0)),
        collate_fn=collate_indexed,
    )
    device = torch.device("cuda")
    model = build_model(config).to(device).eval()
    model.backbone.set_frozen(True)
    args.output.mkdir(parents=True, exist_ok=True)
    processed = 0

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for batch in loader:
            missing = [
                i
                for i, idx in enumerate(batch["indices"])
                if not (args.output / f"{idx:05d}.pt").exists()
            ]
            if not missing:
                processed += len(batch["indices"])
                continue
            images = [batch["original"][i] for i in missing]
            sequences = model.backbone(images)
            for i, tokens in zip(missing, sequences, strict=True):
                index = batch["indices"][i]
                mask = batch["mask"][i]
                provenance = batch["provenance"][i]
                image = batch["original"][i]
                target = (
                    None
                    if mask is None
                    else mask_to_token_occupancy(mask, tokens.shape[0], image.size).to(
                        torch.bfloat16
                    )
                )
                torch.save(
                    {
                        "tokens": tokens.detach().cpu().to(torch.bfloat16),
                        "provenance": int(provenance),
                        "token_mask_target": target,
                    },
                    args.output / f"{index:05d}.pt",
                )
            processed += len(batch["indices"])
            if processed % 100 == 0:
                print(
                    f"shard={args.shard_index} cached={processed}/{len(shard)}",
                    flush=True,
                )
    print(
        f"cache_shard_complete={args.shard_index}/{args.num_shards} "
        f"rows={len(shard)} path={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
