from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from .dataset import ManifestRecord, parse_provenance
from .manifests import normalize_generator


class CachedFeatureDataset(Dataset[dict[str, Any]]):
    """Frozen-backbone token cache keyed by manifest row index."""

    def __init__(self, manifest_path: str | Path, cache_root: str | Path) -> None:
        frame = pd.read_csv(manifest_path)
        self.cache_root = Path(cache_root)
        self.records: list[ManifestRecord] = []
        for row in frame.to_dict(orient="records"):
            dataset = str(row["dataset"])
            self.records.append(
                ManifestRecord(
                    image_path=Path(str(row["image_path"])),
                    provenance=parse_provenance(row["label"], dataset),
                    dataset=dataset,
                    generator=normalize_generator(row.get("generator", ""), dataset),
                    tamper_mask_path=None,
                )
            )

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        del epoch

    def __getitem__(self, index: int) -> dict[str, Any]:
        return torch.load(
            self.cache_root / f"{index:05d}.pt", map_location="cpu", weights_only=False
        )


def collate_cached_features(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "original_tokens": [sample["tokens"] for sample in samples],
        "transformed_tokens": [sample["tokens"] for sample in samples],
        "provenance": torch.tensor([sample["provenance"] for sample in samples], dtype=torch.long),
        "token_mask_targets": [sample["token_mask_target"] for sample in samples],
    }
