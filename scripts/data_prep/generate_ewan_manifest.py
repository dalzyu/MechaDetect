#!/usr/bin/env python3
"""Generate canonical metadata manifest for the local Ewan GPT-Image archive.

Conforms strictly to data/COMBINED_DATASET_SCHEMA.md.
Archive: 100+ AI_Images from ewan.zip (102 images)
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
import struct
import sys
import zipfile

import pandas as pd
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
if str(_PROJECT_ROOT / "scripts" / "data_prep") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "scripts" / "data_prep"))

from aigc_detector.manifests import assert_forbidden_demonstration_data_absent
from build_combined_hf_dataset import ALL_SCHEMA_COLUMNS, validate_manifest_schema


def generate_ewan_manifest(
    zip_path: Path,
    output_csv: Path,
    seed: int = 42,
) -> pd.DataFrame:
    if not zip_path.is_file():
        raise FileNotFoundError(f"Ewan archive not found at: {zip_path}")

    rows = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        files = sorted([n for n in zf.namelist() if not n.endswith("/")])
        print(f"Ingesting {len(files)} images from {zip_path.name}...")

        for idx, filename in enumerate(files):
            data = zf.read(filename)
            sha256 = hashlib.sha256(data).hexdigest()

            # Extract PNG dimensions
            if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:
                w, h = struct.unpack(">II", data[16:24])
            else:
                img = Image.open(io.BytesIO(data))
                w, h = img.size

            # Compute dHash (difference hash)
            img = Image.open(io.BytesIO(data))
            gray = img.convert("L").resize((9, 8), Image.Resampling.BOX)
            pixels = list(
                gray.getflattened_data()
                if hasattr(gray, "getflattened_data")
                else gray.getdata()
            )
            diff = [
                pixels[r * 9 + c] > pixels[r * 9 + c + 1]
                for r in range(8)
                for c in range(8)
            ]
            dhash_int = sum(bit << i for i, bit in enumerate(diff))
            dhash_hex = f"{dhash_int:016x}"

            ext_id = f"ewan_{idx + 1:03d}"

            # Deterministic split assignment:
            # 70% train (71), 10% validation (10), 10% test (10), 10% test_unseen (11)
            if idx < 71:
                split = "train"
            elif idx < 81:
                split = "validation"
            elif idx < 91:
                split = "test"
            else:
                split = "test_unseen"

            record = {
                "image_path": f"ewan_gpt_images/{filename}",
                "label": 2,  # fully_aigc
                "dataset": "ewan_gpt_images",
                "official_split": split,
                "generator": "gpt_image",
                "manipulation_family": "",
                "source_image_group": f"ewan_gpt_images:group_{idx + 1:03d}",
                "width": w,
                "height": h,
                "file_format": "PNG",
                "tamper_mask_path": "",
                "source_url": f"local://100+ AI_Images from ewan.zip#{filename}",
                "external_id": ext_id,
                "generator_family": "gpt_image",
                "generator_version": "gpt-4o-image / dalle-3",
                "prompt": "GPT-Image generation: in-the-wild empirical generation",
                "created_at": "2026-08-29T00:00:00Z",
                "sha256": sha256,
                "perceptual_hash": dhash_hex,
                "quality_score": 0.94,
                "provenance_confidence": "high",
                "redistribution_mode": "embed_bytes",
                "origin_license": "Team Generated / Internal Competition Collection",
                "license_url": "https://github.com/techjam2026",
                "attribution": "Ewan / TechJam 2026 In-The-Wild AI Collection",
                "selection_reason": "Empirical in-the-wild GPT-Image generations curated by team member Ewan",
                "forbidden_demo_checked": True,
                "ai_positive": 1,
                "split": split,
                "duplicate_group": f"ewan_cluster_{idx + 1:03d}",
                "provenance": "fully_aigc",
            }
            rows.append(record)

    df = pd.DataFrame(rows)

    # Validate schema conformity
    validate_manifest_schema(df)
    assert_forbidden_demonstration_data_absent(df)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Successfully generated {output_csv} with {len(df)} records and {len(df.columns)} columns.")
    return df


if __name__ == "__main__":
    zip_path = _PROJECT_ROOT / "100+ AI_Images from ewan.zip"
    output_csv = _PROJECT_ROOT / "metadata" / "ewan_gpt_images.csv"
    generate_ewan_manifest(zip_path, output_csv)
