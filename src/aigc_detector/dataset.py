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
    Transformation,
)
from .manifests import normalize_generator
from .preprocessing import RenderPolicy, render_for_model, render_mask_geometry
from .transforms import apply_transform, sample_transform


@dataclass(frozen=True)
class ManifestRecord:
    row_id: str
    split: str
    image_path: Path
    provenance: Provenance
    ai_positive: int
    dataset: str
    generator: str
    generator_family: str
    duplicate_group: str
    source_image_group: str
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


def load_manifest_frame(manifest_path: str | Path) -> pd.DataFrame:
    """Load one training manifest without silently coercing its format."""
    path = Path(manifest_path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported manifest format {path.suffix!r}; use CSV, Parquet, or JSONL")


def verify_materialization(
    manifest_path: str | Path,
    *,
    data_root: str | Path | None = None,
    limit: int | None = None,
    allow_missing: bool = False,
) -> dict[str, int]:
    """Fail closed unless every selected image and declared mask is readable."""
    frame = load_manifest_frame(manifest_path)
    if limit is not None:
        frame = frame.iloc[:limit]
    root = Path(data_root) if data_root is not None else Path(manifest_path).parent
    missing_images = 0
    missing_masks = 0
    for row in frame.to_dict(orient="records"):
        image_path = Path(str(row["image_path"]).replace("\\", "/"))
        if not image_path.is_absolute():
            image_path = root / image_path
        if not image_path.is_file():
            missing_images += 1
        mask_value = row.get("tamper_mask_path", "")
        if not pd.isna(mask_value) and str(mask_value).strip():
            mask_path = Path(str(mask_value).replace("\\", "/"))
            if not mask_path.is_absolute():
                mask_path = root / mask_path
            if not mask_path.is_file():
                missing_masks += 1
    result = {
        "rows": len(frame),
        "missing_images": missing_images,
        "missing_masks": missing_masks,
    }
    if (missing_images or missing_masks) and not allow_missing:
        raise FileNotFoundError(f"Manifest is not fully materialized: {result}")
    return result


class PairedImageDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        data_root: str | Path | None = None,
        seed: int = 42,
        transform_families: tuple[Transformation, ...] | None = None,
        render_policy: RenderPolicy | str = RenderPolicy.SQUARE_JPEG95,
        expected_split: str | None = None,
        split: str | None = None,
        runtime_fetch: bool = False,
        allow_missing: bool = False,
    ) -> None:
        frame = load_manifest_frame(manifest_path)
        required = {"image_path", "label", "dataset"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

        target_split = expected_split or split
        if target_split is not None:
            target_split = target_split.strip().lower()

        root = Path(data_root) if data_root is not None else Path(manifest_path).parent
        self.records: list[ManifestRecord] = []
        for index, row in enumerate(frame.to_dict(orient="records")):
            row_id_val = row.get("row_id")
            row_id = (
                str(row_id_val).strip()
                if (row_id_val is not None and not pd.isna(row_id_val))
                else f"{row.get('dataset', 'unknown')}_{index:06d}"
            )

            split_val = row.get("split")
            row_split = (
                str(split_val).strip().lower()
                if (split_val is not None and not pd.isna(split_val))
                else ""
            )
            if target_split is not None:
                if not row_split:
                    row_split = target_split
                elif row_split != target_split:
                    raise ValueError(
                        f"Row {row_id} has split {row_split!r}, but expected split {target_split!r}"
                    )

            path = Path(str(row["image_path"]).replace("\\", "/"))
            if not path.is_absolute():
                path = root / path
            mask_value = row.get("tamper_mask_path", "")
            raw_mask_path = "" if pd.isna(mask_value) else str(mask_value).strip()
            mask_path = Path(raw_mask_path.replace("\\", "/")) if raw_mask_path else None
            if mask_path is not None and not mask_path.is_absolute():
                mask_path = root / mask_path
            prov = parse_provenance(row["label"], str(row["dataset"]))
            raw_ai_pos = row.get("ai_positive")
            if raw_ai_pos is not None and not pd.isna(raw_ai_pos) and str(raw_ai_pos).strip() != "":
                ai_positive_val = int(raw_ai_pos)
            else:
                ai_positive_val = 0 if prov == Provenance.AUTHENTIC else 1

            gen_norm = normalize_generator(row.get("generator", ""), str(row["dataset"]))
            gen_family = row.get("generator_family")
            generator_family = (
                str(gen_family).strip()
                if (gen_family is not None and not pd.isna(gen_family))
                else gen_norm
            )
            dup_grp = row.get("duplicate_group")
            duplicate_group = (
                str(dup_grp).strip() if (dup_grp is not None and not pd.isna(dup_grp)) else row_id
            )
            src_img_grp = row.get("source_image_group")
            source_image_group = (
                str(src_img_grp).strip()
                if (src_img_grp is not None and not pd.isna(src_img_grp))
                else row_id
            )

            self.records.append(
                ManifestRecord(
                    row_id=row_id,
                    split=row_split,
                    image_path=path,
                    provenance=prov,
                    ai_positive=ai_positive_val,
                    dataset=str(row["dataset"]),
                    generator=gen_norm,
                    generator_family=generator_family,
                    duplicate_group=duplicate_group,
                    source_image_group=source_image_group,
                    tamper_mask_path=mask_path,
                )
            )
        self.seed = seed
        self.epoch = 0
        self.transform_families = transform_families
        self.render_policy = RenderPolicy(render_policy)
        self.target_split = target_split

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image_path = record.image_path
        tamper_mask_path = record.tamper_mask_path

        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image asset for row {record.row_id}: {image_path}")

        with Image.open(image_path) as source:
            raw = source.convert("RGB").copy()

        rng = Random(self.seed + self.epoch * max(1, len(self)) + index)
        original = render_for_model(raw, self.render_policy, rng=rng)

        rendered_mask = None
        if tamper_mask_path is not None and record.provenance is not Provenance.FULLY_AIGC:
            if not tamper_mask_path.is_file():
                raise FileNotFoundError(
                    f"Missing tamper mask asset for row {record.row_id}: {tamper_mask_path}"
                )
            with Image.open(tamper_mask_path) as source_mask:
                raw_mask = source_mask.convert("L").copy()
            rendered_mask = render_mask_geometry(
                raw_mask,
                self.render_policy,
                image_size=raw.size,
                rendered_size=original.size,
            )
        transform = (
            sample_transform(rng, self.transform_families) if self.transform_families else None
        )
        transformed = (
            apply_transform(original, transform, rng, mask=rendered_mask) if transform else original
        )

        return {
            "row_id": record.row_id,
            "split": record.split,
            "image_path": str(record.image_path),
            "original": original,
            "transformed": transformed,
            "provenance": int(record.provenance),
            "ai_positive": record.ai_positive,
            "dataset": record.dataset,
            "generator": record.generator,
            "generator_family": record.generator_family,
            "duplicate_group": record.duplicate_group,
            "source_image_group": record.source_image_group,
            "mask": rendered_mask,
            "transform": None if transform is None else transform.family.name.lower(),
        }


def collate_pairs(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "row_id": [sample["row_id"] for sample in samples],
        "split": [sample["split"] for sample in samples],
        "image_path": [sample["image_path"] for sample in samples],
        "original": [sample["original"] for sample in samples],
        "transformed": [sample["transformed"] for sample in samples],
        "provenance": torch.tensor([sample["provenance"] for sample in samples], dtype=torch.long),
        "ai_positive": torch.tensor(
            [sample["ai_positive"] for sample in samples], dtype=torch.float
        ),
        "dataset": [sample["dataset"] for sample in samples],
        "generator": [sample["generator"] for sample in samples],
        "generator_family": [sample["generator_family"] for sample in samples],
        "duplicate_group": [sample["duplicate_group"] for sample in samples],
        "source_image_group": [sample["source_image_group"] for sample in samples],
        "mask": [sample["mask"] for sample in samples],
        "transform": [sample["transform"] for sample in samples],
    }
