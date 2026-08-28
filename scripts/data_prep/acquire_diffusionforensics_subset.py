from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd
from datasets import load_from_disk
from huggingface_hub import snapshot_download
from PIL import Image

from aigc_detector.runtime import load_local_environment

REPOSITORY = "nebula/DF-arrow"
REVISION = "29dc74b02fa96e3024c0b71c760ddd7101b16c7d"
SUBDIRECTORY = "DiffusionForensics"


def _family(image_path: str) -> str:
    parts = image_path.replace("\\", "/").split("/")
    if len(parts) < 2:
        raise ValueError(f"Unexpected DiffusionForensics path {image_path!r}")
    return parts[1].lower()


def _select_indices(
    mapping: dict[str, int], seed: int, eval_per_generator: int
) -> dict[int, tuple[str, str]]:
    groups: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for path, index in mapping.items():
        groups[_family(path)].append((path, int(index)))
    selected: dict[int, tuple[str, str]] = {}
    for family, values in sorted(groups.items()):
        family_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{family}".encode()).digest()[:4], "big"
        )
        frame = pd.DataFrame(values, columns=["path", "index"]).sample(
            frac=1.0, random_state=family_seed
        )
        if family == "adm":
            chosen = frame
            split = "train"
        elif family == "real":
            train_count = min(6000, max(0, len(frame) - 500))
            for path, index in frame.iloc[:train_count].itertuples(index=False, name=None):
                selected[int(index)] = (str(path), "train")
            chosen = frame.iloc[train_count : train_count + 500]
            split = "validation"
        else:
            chosen = frame.iloc[:eval_per_generator]
            split = "test_unseen"
        for path, index in chosen.itertuples(index=False, name=None):
            selected[int(index)] = (str(path), split)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-per-generator", type=int, default=500)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    load_local_environment(project_root)
    data_root = Path(os.environ["TECHJAM_DATA_ROOT"])
    snapshot_root = data_root / "diffusionforensics_arrow_mirror"
    snapshot_download(
        REPOSITORY,
        repo_type="dataset",
        revision=REVISION,
        allow_patterns=[f"{SUBDIRECTORY}/*"],
        local_dir=snapshot_root,
    )
    dataset_root = snapshot_root / SUBDIRECTORY
    mapping = json.loads((dataset_root / "mapping.json").read_text(encoding="utf-8"))
    selected = _select_indices(mapping, args.seed, args.eval_per_generator)
    dataset = load_from_disk(str(dataset_root))
    rows = []
    for progress, index in enumerate(sorted(selected), start=1):
        original_path, split = selected[index]
        record = dataset[index]
        if str(record["image_path"]).replace("\\", "/") != original_path:
            raise RuntimeError(f"Mirror mapping mismatch at row {index}")
        payload = bytes(record["image"])
        sha256 = hashlib.sha256(payload).hexdigest()
        suffix = Path(original_path).suffix.lower() or ".img"
        family = _family(original_path)
        provenance = "authentic" if family == "real" else "fully_aigc"
        relative = (
            Path("diffusionforensics_subset")
            / split
            / provenance
            / family
            / f"{sha256[:20]}{suffix}"
        )
        destination = data_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            temporary = destination.with_suffix(destination.suffix + ".partial")
            temporary.write_bytes(payload)
            temporary.replace(destination)
        with Image.open(destination) as image:
            width, height = image.size
            image_format = str(image.format or "unknown")
            image.verify()
        rows.append(
            {
                "image_path": relative.as_posix(),
                "label": provenance,
                "dataset": "DiffusionForensics",
                "official_split": split,
                "img_id": str(record["md5"]),
                "source_image_group": original_path if family == "real" else "",
                "generator": family,
                "width": width,
                "height": height,
                "file_format": image_format,
                "sha256": sha256,
                "tamper_mask_path": "",
                "mirror_repo": REPOSITORY,
                "mirror_revision": REVISION,
                "original_path": original_path,
            }
        )
        if progress % 250 == 0 or progress == len(selected):
            print(f"extracted {progress}/{len(selected)}", flush=True)
    output = project_root / "metadata" / "diffusionforensics_subset.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
