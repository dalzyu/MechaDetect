from __future__ import annotations

import argparse
import hashlib
import os
import zipfile
from pathlib import Path

import pandas as pd
import requests
from acquire_wildfake_subset import REPOSITORY, Source, _extract_source
from modelscope_hub import HubApi
from PIL import Image

from aigc_detector.runtime import load_local_environment

COCO_URL = "http://images.cocodataset.org/zips/val2017.zip"
DALL_E_ADVANCED = Source(
    "dalle3_advanced_demo",
    "Images/Diffusion_based/DALLE.zip",
    "label_csv_files/dalle3.csv",
    "fully_aigc",
    0,
    0,
    8843,
)


def acquire_coco(data_root: Path) -> list[dict[str, object]]:
    root = data_root / "organizer_demo" / "coco"
    archive_path = root / "val2017.zip"
    image_root = root / "val2017"
    root.mkdir(parents=True, exist_ok=True)
    if not archive_path.exists() and not image_root.exists():
        partial = archive_path.with_suffix(".zip.partial")
        with requests.get(COCO_URL, stream=True, timeout=120) as response:
            response.raise_for_status()
            with partial.open("wb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        partial.replace(archive_path)
    if not image_root.exists():
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(root)
    rows = []
    paths = sorted(image_root.glob("*.jpg"))
    for index, path in enumerate(paths, start=1):
        payload = path.read_bytes()
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
        rows.append(
            {
                "image_path": path.relative_to(data_root).as_posix(),
                "label": "authentic",
                "dataset": "OrganizerDemo-COCO-val2017",
                "official_split": "organizer_demo",
                "img_id": path.stem,
                "source_image_group": path.stem,
                "generator": "coco_val2017",
                "width": width,
                "height": height,
                "file_format": "JPEG",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "tamper_mask_path": "",
                "evaluation_only": True,
            }
        )
        if index % 500 == 0:
            print(f"coco: {index}/{len(paths)}", flush=True)
    return rows


def acquire_dalle(data_root: Path, seed: int) -> list[dict[str, object]]:
    metadata_root = data_root / "wildfake_metadata"
    labels_path = metadata_root / DALL_E_ADVANCED.labels
    if not labels_path.exists():
        HubApi().download_file(
            REPOSITORY, "dataset", DALL_E_ADVANCED.labels, local_dir=metadata_root
        )
    rows = _extract_source(DALL_E_ADVANCED, labels_path, data_root, seed)
    for row in rows:
        row["dataset"] = "OrganizerDemo-WildFake"
        row["official_split"] = "organizer_demo"
        row["generator"] = "dalle3_advanced"
        row["evaluation_only"] = True
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only", choices=("coco", "dalle"))
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    load_local_environment(project_root)
    data_root = Path(os.environ["TECHJAM_DATA_ROOT"])
    output = project_root / "metadata" / "organizer_demo.csv"
    existing = (
        pd.read_csv(output, low_memory=False).fillna("") if output.exists() else pd.DataFrame()
    )
    rows = []
    if args.only in (None, "coco"):
        rows.extend(acquire_coco(data_root))
    elif not existing.empty:
        rows.extend(existing[existing["label"] == "authentic"].to_dict(orient="records"))
    if args.only in (None, "dalle"):
        rows.extend(acquire_dalle(data_root, args.seed))
    elif not existing.empty:
        rows.extend(existing[existing["label"] == "fully_aigc"].to_dict(orient="records"))
    frame = pd.DataFrame(rows).drop_duplicates("sha256")
    frame.to_csv(output, index=False)
    print(frame.groupby(["dataset", "label"]).size())
    print(f"Wrote {len(frame)} evaluation-only rows to {output}")


if __name__ == "__main__":
    main()
