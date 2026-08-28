from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Image as DatasetImage
from datasets import load_dataset
from PIL import Image

from aigc_detector.constants import SID_LABEL_TO_PROVENANCE
from aigc_detector.runtime import load_local_environment


def _image_extension(payload: bytes) -> tuple[str, str]:
    with Image.open(BytesIO(payload)) as image:
        image_format = (image.format or "PNG").upper()
    extensions = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}
    return extensions.get(image_format, f".{image_format.lower()}"), image_format


def _write_raw_image(payload: dict[str, Any], destination_stem: Path) -> tuple[Path, str, str]:
    raw = payload.get("bytes")
    if raw is None:
        source_path = payload.get("path")
        if not source_path:
            raise ValueError("Dataset image has neither bytes nor a path")
        raw = Path(source_path).read_bytes()
    extension, image_format = _image_extension(raw)
    destination = destination_stem.with_suffix(extension)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(raw)
    return destination, image_format, hashlib.sha256(raw).hexdigest()


def _collect_split(
    *,
    split: str,
    per_class: int,
    data_root: Path,
    seed: int,
    shuffle_buffer: int,
    excluded_hashes: set[str] | None = None,
) -> list[dict[str, Any]]:
    dataset = load_dataset("saberzl/SID_Set", split=split, streaming=True)
    dataset = dataset.cast_column("image", DatasetImage(decode=False))
    dataset = dataset.cast_column("mask", DatasetImage(decode=False))
    dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)

    counts: Counter[int] = Counter()
    rows: list[dict[str, Any]] = []
    excluded_hashes = excluded_hashes or set()
    split_root = data_root / "sid_sanity" / split
    for sample in dataset:
        label = int(sample["label"])
        if counts[label] >= per_class:
            continue

        provenance = SID_LABEL_TO_PROVENANCE[label].name.lower()
        image_id = str(sample["img_id"])
        raw = sample["image"].get("bytes")
        if raw is not None and hashlib.sha256(raw).hexdigest() in excluded_hashes:
            continue
        destination, image_format, sha256 = _write_raw_image(
            sample["image"], split_root / provenance / image_id
        )

        mask_path = ""
        mask = sample.get("mask")
        if mask is not None:
            saved_mask, _, _ = _write_raw_image(mask, split_root / "masks" / image_id)
            mask_path = saved_mask.relative_to(data_root).as_posix()

        rows.append(
            {
                "image_path": destination.relative_to(data_root).as_posix(),
                "label": provenance,
                "dataset": "SID-Set",
                "official_split": split,
                "img_id": image_id,
                "source_image_group": "",
                "generator": "",
                "manipulation_family": "",
                "width": int(sample["width"]),
                "height": int(sample["height"]),
                "file_format": image_format,
                "sha256": sha256,
                "tamper_mask_path": mask_path,
            }
        )
        counts[label] += 1
        if len(rows) % 100 == 0:
            readable = {
                SID_LABEL_TO_PROVENANCE[key].name.lower(): value for key, value in counts.items()
            }
            print(f"{split}: {len(rows)} images {readable}", flush=True)
        if all(counts[label_value] >= per_class for label_value in SID_LABEL_TO_PROVENANCE):
            break

    if not all(counts[label_value] >= per_class for label_value in SID_LABEL_TO_PROVENANCE):
        raise RuntimeError(f"Split {split!r} ended before reaching {per_class} per class: {counts}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=600)
    parser.add_argument("--validation-per-class", type=int, default=150)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only-validation", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    load_local_environment(project_root)
    import os

    data_root_value = os.environ.get("TECHJAM_DATA_ROOT")
    if not data_root_value:
        raise RuntimeError("TECHJAM_DATA_ROOT is not configured")
    data_root = Path(data_root_value)

    splits_dir = project_root / "splits"
    if args.only_validation:
        train_path = splits_dir / "train.csv"
        if not train_path.exists():
            raise FileNotFoundError("--only-validation requires an existing splits/train.csv")
        train_rows = pd.read_csv(train_path).fillna("").to_dict(orient="records")
    else:
        train_rows = _collect_split(
            split="train",
            per_class=args.per_class,
            data_root=data_root,
            seed=args.seed,
            shuffle_buffer=args.shuffle_buffer,
        )
    train_hashes = {str(row["sha256"]) for row in train_rows}
    validation_rows = _collect_split(
        split="validation",
        per_class=args.validation_per_class,
        data_root=data_root,
        seed=args.seed,
        shuffle_buffer=args.shuffle_buffer,
        excluded_hashes=train_hashes,
    )

    metadata_dir = project_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train_rows + validation_rows).to_csv(metadata_dir / "sid_sanity.csv", index=False)
    pd.DataFrame(train_rows).to_csv(splits_dir / "train.csv", index=False)
    pd.DataFrame(validation_rows).to_csv(splits_dir / "val_in_domain.csv", index=False)
    print(f"Wrote {len(train_rows)} training and {len(validation_rows)} validation rows")


if __name__ == "__main__":
    main()
