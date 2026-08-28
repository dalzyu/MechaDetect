from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .constants import (
    PROVENANCE_NAMES,
    SID_LABEL_TO_PROVENANCE,
    Provenance,
)
from .manifests import normalize_generator
from .preprocessing import RenderPolicy, render_for_model, render_mask_geometry
from .transforms import apply_transform_chain, sample_transform_chain


@dataclass(frozen=True)
class ManifestRecord:
    image_path: Path
    provenance: Provenance
    dataset: str
    generator: str
    tamper_mask_path: Path | None


def parse_provenance(value: Any, dataset: str = "") -> Provenance:
    if isinstance(value, str):
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "real": Provenance.AUTHENTIC,
            "authentic": Provenance.AUTHENTIC,
            "tampered": Provenance.TAMPERED,
            "full_synthetic": Provenance.FULLY_AIGC,
            "fully_synthetic": Provenance.FULLY_AIGC,
            "fully_aigc": Provenance.FULLY_AIGC,
            "aigc": Provenance.FULLY_AIGC,
            "ai": Provenance.FULLY_AIGC,
        }
        if normalized in aliases:
            return aliases[normalized]
        if normalized.isdigit():
            value = int(normalized)
        else:
            raise ValueError(f"Unknown provenance label {value!r}")

    if isinstance(value, int) or (isinstance(value, float) and value.is_integer()):
        numeric = int(value)
        if dataset.lower().replace("-", "_") in {"sid", "sid_set"}:
            return SID_LABEL_TO_PROVENANCE[numeric]
        if numeric in range(len(PROVENANCE_NAMES)):
            return Provenance(numeric)
    raise ValueError(f"Unknown provenance label {value!r} for dataset {dataset!r}")


class PairedImageDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        data_root: str | Path | None = None,
        seed: int = 42,
        chain_length_probabilities: dict[int, float] | None = None,
        render_policy: RenderPolicy | str = RenderPolicy.SQUARE_JPEG95,
    ) -> None:
        frame = pd.read_csv(manifest_path)
        required = {"image_path", "label", "dataset"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

        root = Path(data_root) if data_root is not None else Path(manifest_path).parent
        self.records = []
        for row in frame.to_dict(orient="records"):
            path = Path(str(row["image_path"]))
            if not path.is_absolute():
                path = root / path
            mask_value = row.get("tamper_mask_path", "")
            raw_mask_path = "" if pd.isna(mask_value) else str(mask_value).strip()
            mask_path = Path(raw_mask_path) if raw_mask_path else None
            if mask_path is not None and not mask_path.is_absolute():
                mask_path = root / mask_path
            self.records.append(
                ManifestRecord(
                    image_path=path,
                    provenance=parse_provenance(row["label"], str(row["dataset"])),
                    dataset=str(row["dataset"]),
                    generator=normalize_generator(row.get("generator", ""), str(row["dataset"])),
                    tamper_mask_path=mask_path,
                )
            )
        self.seed = seed
        self.epoch = 0
        self.chain_length_probabilities = chain_length_probabilities or {1: 1.0}
        self.render_policy = RenderPolicy(render_policy)

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        with Image.open(record.image_path) as source:
            raw = source.convert("RGB").copy()

        rng = Random(self.seed + self.epoch * max(1, len(self)) + index)
        original = render_for_model(raw, self.render_policy, rng=rng)
        specs = sample_transform_chain(rng, self.chain_length_probabilities)
        transformed = apply_transform_chain(original, specs, rng)

        rendered_mask = None
        if record.tamper_mask_path is not None and record.provenance is not Provenance.FULLY_AIGC:
            with Image.open(record.tamper_mask_path) as source_mask:
                raw_mask = source_mask.convert("L").copy()
            rendered_mask = render_mask_geometry(
                raw_mask,
                self.render_policy,
                image_size=raw.size,
                rendered_size=original.size,
            )

        return {
            "image_path": str(record.image_path),
            "original": original,
            "transformed": transformed,
            "provenance": int(record.provenance),
            "dataset": record.dataset,
            "generator": record.generator,
            "mask": rendered_mask,
            "transform_chain": [spec.family.name.lower() for spec in specs],
        }


def collate_pairs(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image_path": [sample["image_path"] for sample in samples],
        "original": [sample["original"] for sample in samples],
        "transformed": [sample["transformed"] for sample in samples],
        "provenance": torch.tensor([sample["provenance"] for sample in samples], dtype=torch.long),
        "dataset": [sample["dataset"] for sample in samples],
        "generator": [sample["generator"] for sample in samples],
        "mask": [sample["mask"] for sample in samples],
        "transform_chain": [sample["transform_chain"] for sample in samples],
    }
