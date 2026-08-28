from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image

from aigc_detector.runtime import load_local_environment


def canonicalize_image(source_path: Path, destination: Path, size: int, quality: int) -> None:
    with Image.open(source_path) as source:
        image = source.convert("RGB")
    image = image.resize((size, size), Image.Resampling.BICUBIC)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(
        destination,
        format="JPEG",
        quality=quality,
        optimize=False,
        progressive=False,
        exif=b"",
    )


def canonicalize_manifest(
    manifest_path: Path,
    *,
    data_root: Path,
    output_manifest: Path,
    split_name: str,
    size: int,
    quality: int,
) -> None:
    frame = pd.read_csv(manifest_path).fillna("")
    rows = []
    for index, row in frame.iterrows():
        source = Path(str(row["image_path"]))
        if not source.is_absolute():
            source = data_root / source
        label = str(row["label"])
        source_id = str(row.get("img_id", index))
        destination = data_root / "canonical_sid_sanity" / split_name / label / f"{source_id}.jpg"
        canonicalize_image(source, destination, size=size, quality=quality)

        item = row.to_dict()
        item["image_path"] = destination.relative_to(data_root).as_posix()
        item["source_image_path"] = str(row["image_path"])
        item["file_format"] = "JPEG"
        item["width"] = size
        item["height"] = size
        item["aspect_ratio"] = 1.0
        item["canonicalized_geometry"] = f"stretch_square_{size}"
        item["canonicalized_encoding"] = f"jpeg_q{quality}"
        rows.append(item)

        if (index + 1) % 100 == 0 or index + 1 == len(frame):
            print(f"{split_name}: {index + 1}/{len(frame)}", flush=True)

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_manifest, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, default=Path("splits/train.csv"))
    parser.add_argument("--val-manifest", type=Path, default=Path("splits/val_in_domain.csv"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--size", type=int, default=1584)
    parser.add_argument("--quality", type=int, default=95)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    load_local_environment(project_root)
    import os

    data_root = args.data_root or Path(os.environ["TECHJAM_DATA_ROOT"])
    canonicalize_manifest(
        args.train_manifest,
        data_root=data_root,
        output_manifest=project_root / "splits" / "train_canonical.csv",
        split_name="train",
        size=args.size,
        quality=args.quality,
    )
    canonicalize_manifest(
        args.val_manifest,
        data_root=data_root,
        output_manifest=project_root / "splits" / "val_canonical.csv",
        split_name="validation",
        size=args.size,
        quality=args.quality,
    )


if __name__ == "__main__":
    main()
