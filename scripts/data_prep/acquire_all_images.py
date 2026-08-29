#!/usr/bin/env python3
"""Acquire all missing images for the zye2/tj-data dataset.

Reads manifest CSVs from the HuggingFace dataset zye2/tj-data, checks
which images are missing from TECHJAM_DATA_ROOT, and downloads them
from their original source datasets.

Resumable: skips files that already exist on disk.
Saves all images as JPEG to match the manifest's .jpg extension
(except ewan_gpt_images which uses .png from the local zip).

Usage:
    uv run python scripts/data_prep/acquire_all_images.py [--dry-run] [--source SOURCE_NAME]

"""

from __future__ import annotations

import argparse
import io
import os
import sys
import zipfile
from pathlib import Path
from typing import Any, Iterator

# Ensure project root is on sys.path for aigc_detector imports
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from aigc_detector.runtime import load_local_environment

load_local_environment(_PROJECT_ROOT)

DATA_ROOT = Path(os.environ["TECHJAM_DATA_ROOT"])
HF_HOME = os.environ.get("TECHJAM_HF_HOME", "")
if HF_HOME:
    os.environ.setdefault("HF_HOME", HF_HOME)
try:
    from huggingface_hub import get_token
    _token = get_token()
    if _token:
        os.environ["HF_TOKEN"] = _token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = _token
except Exception:
    pass


# --------------------------------------------------------------------------- #
# Source configurations extracted from SOURCE_REGISTRY in build_combined_hf_dataset.py
# --------------------------------------------------------------------------- #

# HuggingFace datasets that can be streamed.
# repo: HF dataset repo ID
# split: HF split name
# image_fields: field names to try (in order) for the image column
# paired_field: if the dataset has paired images, use this field for the
#   edited/generated version (not the original)
HF_SOURCES: dict[str, dict[str, Any]] = {
    # --- embed_bytes HF sources ---
    "artic_dataset": {"repo": "links-ads/artic-dataset", "split": "train"},
    "flux_reason_6m": {"repo": "LucasFang/FLUX-Reason-6M", "split": "train"},
    "ideogram_27k": {"repo": "bitmind/ideogram-27k", "split": "train"},
    "midjourney_v6_recap": {"repo": "Photoroom/midjourney-v6-recap", "split": "train"},
    "midjourney_v5_images": {"repo": "ehristoforu/midjourney-images", "split": "train"},
    "sd3_medium_synths": {"repo": "VincHa/SD3_medium_synths", "split": "train"},
    # --- reference_only HF sources (user-approved mirrors and alternatives) ---
    "open_images_v7": {"repo": "bitmind/open-images-v7", "split": "train"},
    "authentic_glamour_portraits": {"repo": "mattymchen/celeba-hq", "split": "train"},
    "sintel_blender_open_movie": {"repo": "badigadiii/game_screenshots_11k", "split": "train"},
    "manga109_illustrations": {"repo": "hal-utokyo/Manga109-s", "split": "train"},
    "art_museums_pd": {"repo": "Mitsua/art-museums-pd-440k", "split": "train"},
    "gpt_image_edit_1_5m": {"repo": "UCSC-VLAA/GPT-Image-Edit-1.5M", "split": "train",
                            "paired_field": "edited_image"},
    "google_nano_banana_edited": {"repo": "Tungtom2004/Google_Nano_Banana_Edited_Images", "split": "train",
                                   "paired_field": "edited_image"},
    "krea2_wildcards": {"repo": "innofree/krea2-wildcards", "split": "train"},
    "nano_banana_pro_gen": {"repo": "FlameF0X/nano-banana-pro-gen-zh-en", "split": "train"},
    "ai_meme_macro_overlay": {"repo": "ideepankarsharma2003/AIGeneratedImages_Midjourney", "split": "train"},
    "authentic_classical_figure_art": {"repo": "Mitsua/art-museums-pd-440k", "split": "train"},
    "danbooru2026_aigc_wild": {"repo": "nyanko-devs/danbooru2026", "split": "train"},
    "gta5_driving_renders": {"repo": "Chris1/GTA5", "split": "train"},
    "midjourney_fantasy_environments": {"repo": "Photoroom/midjourney-v6-recap", "split": "train"},
    "tartanair2_ue5_cyberpunk": {"repo": "theairlabcmu/tartanair2", "split": "train"},
    "flux_cyberpunk_scifi": {"repo": "LucasFang/FLUX-Reason-6M", "split": "train"},
    "sdxl_photoreal_vehicles": {"repo": "diffusers-parti-prompts/sdxl-1.0", "split": "train"},
    "ai_reaction_banners": {"repo": "Goku-OpenLab/gpt-image-2-prompts-datasets", "split": "train"},
    "scam_ai_social_posts": {"repo": "Goku-OpenLab/gpt-image-2-prompts-datasets", "split": "train"},
    "game_screenshots_fantasy": {"repo": "badigadiii/game_screenshots_11k", "split": "train"},
}


BLOCKED_SOURCES: dict[str, str] = {
    "danbooru_pre2020_human": (
        "Danbooru pre-2020 human illustrations. See Gwern https://gwern.net/danbooru2021#danbooru2020. "
        "Can be acquired via Kaggle 'Tagged Anime Illustrations' or Danbooru API."
    ),
}


# --------------------------------------------------------------------------- #
# Manifest loading
# --------------------------------------------------------------------------- #

def load_manifest_paths() -> dict[str, list[str]]:
    """Load all manifest CSVs from HF and return {dataset: [image_path, ...]}."""
    import urllib.request
    import csv as csv_mod

    splits = ["train", "validation", "test", "test_unseen"]
    by_dataset: dict[str, list[str]] = {}
    for split in splits:
        local_cache = Path("E:/techjam26-runtime") / f"manifests_{split}.csv"
        print(f"  Loading manifest: {split}...", end=" ", flush=True)
        if local_cache.is_file():
            with open(local_cache, "r", encoding="utf-8") as f:
                reader = csv_mod.DictReader(f)
                count = 0
                for row in reader:
                    ds = row.get("dataset", "")
                    img_path = row.get("image_path", "")
                    if ds and img_path:
                        by_dataset.setdefault(ds, []).append(img_path)
                        count += 1
        else:
            url = f"https://huggingface.co/datasets/zye2/tj-data/resolve/main/{split}.csv"
            with urllib.request.urlopen(url) as resp:
                reader = csv_mod.DictReader(io.TextIOWrapper(resp, encoding="utf-8"))
                count = 0
                for row in reader:
                    ds = row.get("dataset", "")
                    img_path = row.get("image_path", "")
                    if ds and img_path:
                        by_dataset.setdefault(ds, []).append(img_path)
                        count += 1
        print(f"{count} rows")

    return by_dataset


def find_missing(manifest_by_dataset: dict[str, list[str]]) -> dict[str, list[str]]:
    """Check which image paths are missing from DATA_ROOT."""
    missing: dict[str, list[str]] = {}
    total_missing = 0
    total = 0

    for ds, paths in sorted(manifest_by_dataset.items()):
        miss = []
        for p in paths:
            total += 1
            full = DATA_ROOT / p
            if not full.is_file():
                miss.append(p)
        if miss:
            missing[ds] = miss
            total_missing += len(miss)

    print(f"Already on disk:       {total - total_missing}")
    print(f"Missing:                {total_missing}")
    return missing


# --------------------------------------------------------------------------- #
# HuggingFace acquisition
# --------------------------------------------------------------------------- #
IMAGE_FIELD_CANDIDATES = [
    "image", "img", "file", "edited_image", "output", "output_image",
    "generated_image", "jpg", "png", "webp", "jpeg", "content", "data",
]

def _extract_image(item: dict, config: dict[str, Any]) -> Any | None:
    """Extract a PIL Image from a dataset item, trying known field names or url."""
    from PIL import Image

    url = item.get("url")
    if url and isinstance(url, str) and url.startswith("http"):
        import urllib.request
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = resp.read()
            return Image.open(io.BytesIO(data))
        except Exception:
            return None
    # If a paired_field is specified, try it first
    fields_to_try = []
    paired = config.get("paired_field")
    if paired:
        fields_to_try.append(paired)
    fields_to_try.extend(IMAGE_FIELD_CANDIDATES)

    for field in fields_to_try:
        raw = item.get(field)
        if raw is None:
            continue
        if isinstance(raw, Image.Image):
            return raw
        if isinstance(raw, bytes):
            try:
                return Image.open(io.BytesIO(raw))
            except Exception:
                continue
        if isinstance(raw, dict) and "bytes" in raw:
            try:
                return Image.open(io.BytesIO(raw["bytes"]))
            except Exception:
                continue
        if isinstance(raw, dict) and "path" in raw:
            try:
                return Image.open(raw["path"])
            except Exception:
                continue

    return None


def acquire_hf_source(
    dataset_name: str,
    missing_paths: list[str],
    config: dict[str, Any],
    dry_run: bool = False,
) -> tuple[int, int]:
    """Stream from a HuggingFace dataset and save images to expected paths.

    Returns (saved_count, skipped_count).
    """
    from datasets import load_dataset
    from PIL import Image

    repo = config["repo"]
    split = config.get("split", "train")
    paired = config.get("paired_field")

    # Sort missing paths to determine the max index we need
    # Paths look like: {prefix}/{dataset_name}/{dataset_name}_{NNNNNN}.jpg
    # or reference_only/{dataset_name}/{dataset_name}_{NNNNNN}.jpg
    # Extract the 6-digit index from each path
    import re
    indexed = []
    for p in missing_paths:
        m = re.search(r"_(\d{6})\.", p)
        if m:
            indexed.append((int(m.group(1)), p))
        else:
            # Non-indexed path (e.g., ewan_gpt_images) — skip for HF sources
            pass

    if not indexed:
        print(f"  [{dataset_name}] No indexed paths found, skipping")
        return 0, 0

    indexed.sort()
    needed_indices = set(idx for idx, _ in indexed)
    max_idx = max(needed_indices)

    # Build a map from index to expected path
    path_by_index: dict[int, str] = {}
    for idx, p in indexed:
        # Skip if already on disk
        if not (DATA_ROOT / p).is_file():
            path_by_index[idx] = p

    if not path_by_index:
        print(f"  [{dataset_name}] All images already present")
        return 0, len(missing_paths)

    print(f"  [{dataset_name}] Streaming from {repo} (need {len(path_by_index)} images, max_idx={max_idx})")

    if dry_run:
        print(f"  [{dataset_name}] DRY RUN — skipping actual download")
        return 0, len(path_by_index)

    saved = 0
    skipped = 0
    errors = 0

    try:
        ds = load_dataset(repo, split=split, streaming=True, token=os.environ.get("HF_TOKEN"))
    except Exception as exc:
        # Try without split specification
        try:
            ds = load_dataset(repo, streaming=True, token=os.environ.get("HF_TOKEN"))
            # If it returned a DatasetDict, pick the first split
            if hasattr(ds, "keys"):
                first_split = list(ds.keys())[0]
                ds = ds[first_split]
        except Exception as exc2:
            print(f"  [{dataset_name}] ERROR: Could not load dataset: {exc2}")
            return 0, 0

    # Detect image field from first item
    image_field_detected = None
    field_check_done = False

    for i, item in enumerate(ds):
        if i > max_idx:
            break

        # Detect image field on first item
        if not field_check_done:
            field_check_done = True
            available_keys = list(item.keys()) if hasattr(item, "keys") else []
            for field in ([paired] if paired else []) + IMAGE_FIELD_CANDIDATES:
                if field in available_keys:
                    image_field_detected = field
                    break
            if image_field_detected is None:
                # Log all available keys for debugging
                print(f"  [{dataset_name}] Available fields: {available_keys}")

        # Check if this index is needed
        if i not in path_by_index:
            continue

        dest_rel = path_by_index[i]
        dest_full = DATA_ROOT / dest_rel
        dest_full.parent.mkdir(parents=True, exist_ok=True)

        # Skip if already exists (might have been saved by a concurrent run)
        if dest_full.is_file():
            skipped += 1
            continue

        img = _extract_image(item, config)
        if img is None:
            errors += 1
            if errors <= 3:
                print(f"  [{dataset_name}] WARNING: Could not extract image from item {i}")
            continue

        try:
            # Convert to RGB and save as JPEG (to match .jpg extension in manifest)
            if dest_rel.endswith(".jpg") or dest_rel.endswith(".jpeg"):
                img.convert("RGB").save(dest_full, format="JPEG", quality=95)
            else:
                # Save in the expected format
                img.save(dest_full)
            saved += 1
            if saved % 500 == 0:
                print(f"  [{dataset_name}] Progress: {saved}/{len(path_by_index)}", flush=True)
        except Exception as exc:
            errors += 1
            if errors <= 5:
                print(f"  [{dataset_name}] ERROR saving item {i}: {exc}")

    print(f"  [{dataset_name}] Done: saved={saved}, skipped={skipped}, errors={errors}")
    return saved, skipped


# --------------------------------------------------------------------------- #
# Local zip acquisition (ewan_gpt_images)
# --------------------------------------------------------------------------- #

def acquire_ewan_gpt_images(missing_paths: list[str], dry_run: bool = False) -> tuple[int, int]:
    """Extract ewan_gpt_images from the local zip file."""
    zip_path = _PROJECT_ROOT / "100+ AI_Images from ewan.zip"
    if not zip_path.is_file():
        print(f"  [ewan_gpt_images] ERROR: Zip file not found at {zip_path}")
        return 0, 0

    saved = 0
    skipped = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Get list of files in the zip
        zip_names = set(zf.namelist())

        for manifest_path in missing_paths:
            dest_full = DATA_ROOT / manifest_path
            if dest_full.is_file():
                skipped += 1
                continue

            if dry_run:
                continue

            # The manifest path is like: ewan_gpt_images/AI_Images/image_002.png
            # The zip entry is likely: AI_Images/image_002.png
            # Try both the full path and the path without the ewan_gpt_images/ prefix
            zip_entry = manifest_path
            if manifest_path.startswith("ewan_gpt_images/"):
                zip_entry = manifest_path[len("ewan_gpt_images/"):]

            if zip_entry not in zip_names:
                # Try with normalized separators
                zip_entry_norm = zip_entry.replace("\\", "/")
                if zip_entry_norm in zip_names:
                    zip_entry = zip_entry_norm
                else:
                    print(f"  [ewan_gpt_images] WARNING: {zip_entry} not found in zip")
                    continue

            dest_full.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(zip_entry) as src:
                dest_full.write_bytes(src.read())
            saved += 1

    print(f"  [ewan_gpt_images] Done: saved={saved}, skipped={skipped}")
    return saved, skipped


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire all missing images for tj-data")
    parser.add_argument("--dry-run", action="store_true", help="Check what's missing without downloading")
    parser.add_argument("--source", type=str, default=None,
                        help="Only acquire from this specific source dataset")
    args = parser.parse_args()

    print("=" * 70)
    print("TJ-Data Image Acquisition")
    print("=" * 70)
    print(f"Data root: {DATA_ROOT}")
    print()

    # Load manifests
    print("Loading manifests from HuggingFace...")
    manifest_by_dataset = load_manifest_paths()

    # Find missing
    missing = find_missing(manifest_by_dataset)

    if not missing:
        print("\nAll images are already on disk!")
        return

    # Filter to requested source if specified
    if args.source:
        if args.source not in missing:
            print(f"\nNo missing images for source '{args.source}'")
            return
        missing = {args.source: missing[args.source]}

    print(f"\nMissing sources: {len(missing)}")
    for ds, paths in sorted(missing.items(), key=lambda x: -len(x[1])):
        source_type = "HF" if ds in HF_SOURCES else ("LOCAL" if ds == "ewan_gpt_images" else "BLOCKED")
        print(f"  {ds:45s} {len(paths):6d} images  [{source_type}]")
    if args.dry_run:
        print("\nDRY RUN — no downloads performed")
        return

    # --- Acquire from HuggingFace ---
    print("\n" + "=" * 70)
    print("Acquiring from HuggingFace datasets")
    print("=" * 70)

    total_saved = 0
    total_errors = 0

    for ds_name in sorted(missing.keys()):
        if ds_name not in HF_SOURCES:
            continue

        paths = missing[ds_name]
        config = HF_SOURCES[ds_name]
        saved, skipped = acquire_hf_source(ds_name, paths, config, dry_run=args.dry_run)
        total_saved += saved

    # --- Acquire from local zip ---
    print("\n" + "=" * 70)
    print("Acquiring from local zip (ewan_gpt_images)")
    print("=" * 70)

    if "ewan_gpt_images" in missing:
        saved, skipped = acquire_ewan_gpt_images(missing["ewan_gpt_images"], dry_run=args.dry_run)
        total_saved += saved

    # --- Report blockers ---
    print("\n" + "=" * 70)
    print("BLOCKED sources (need manual acquisition)")
    print("=" * 70)

    total_blocked = 0
    for ds_name in sorted(missing.keys()):
        if ds_name in HF_SOURCES or ds_name == "ewan_gpt_images":
            continue
        if ds_name in BLOCKED_SOURCES:
            count = len(missing[ds_name])
            total_blocked += count
            print(f"\n  [{ds_name}] {count} images")
            print(f"    {BLOCKED_SOURCES[ds_name]}")
        else:
            count = len(missing[ds_name])
            total_blocked += count
            print(f"\n  [{ds_name}] {count} images — UNKNOWN source type, needs investigation")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total saved:      {total_saved}")
    print(f"  Total blocked:    {total_blocked}")
    print(f"  Total missing:    {sum(len(v) for v in missing.values())}")

    if total_blocked > 0:
        print(f"\n  ⚠ {total_blocked} images need manual acquisition.")
        print("  See the BLOCKED section above for instructions.")


if __name__ == "__main__":
    main()
