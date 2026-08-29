#!/usr/bin/env python3
"""Acquire and prefetch images for the MechaDetect production pipeline.

Implements resumable source-indexed prefetch over every declared row using
HF auth/revisions, atomic writes, exact source/index/image-field mapping, and
fail-closed exclusion checks.

Features:
- Atomic local writes via temporary files and os.replace to prevent partial/corrupted files.
- Resumable preflight: safely skips existing valid assets; optional --verify-bytes checks decodability.
- Avoids duplicate downloads for aliases sharing a source.
- Exact source/index/image-field mapping for every declared dataset.
- Rejection of forbidden data ('newer image model data(do not use for training)', organizer demo).
- Explicit retention of public Nano Banana datasets ('google_nano_banana_edited', 'nano_banana_pro_gen').
- Rejection of placeholder SHA evidence ('sha:<dataset>:...', Imgur placeholder).
- Records source revisions, distributions, exclusions, and manifest digests into source_revisions.json.

Usage:
    uv run python scripts/data_prep/acquire_all_images.py [options]
    Options:
      --data-root PATH        Data storage root (default: TECHJAM_DATA_ROOT or ./data)
      --manifest-dir PATH     Manifest directory containing JSONL or Parquet splits
      --sources LIST          Comma-separated list of dataset sources to acquire (default: all)
      --dry-run               Check missing and simulate acquisition without downloading
      --resume                Skip already acquired valid images (default: True)
      --no-resume             Force re-acquisition of existing images
      --verify-bytes          Verify disk bytes and image readability for existing files
      --caps CAPS             Optional JSON or file mapping dataset names to max paths
      --report-path PATH      Path to write acquisition audit report
      --revisions-path PATH   Path to write source_revisions.json
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import re
import sys
import uuid
import zipfile
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path for aigc_detector imports
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from aigc_detector.runtime import load_local_environment

load_local_environment(_PROJECT_ROOT)

logger = logging.getLogger("acquire_all_images")

DEFAULT_DATA_ROOT = Path(os.environ.get("TECHJAM_DATA_ROOT", str(_PROJECT_ROOT / "data")))
HF_HOME = os.environ.get("TECHJAM_HF_HOME", "")
if HF_HOME:
    os.environ.setdefault("HF_HOME", HF_HOME)

# Check for Hugging Face tokens in environment or local hub cache
try:
    from huggingface_hub import get_token

    _token = get_token()
    if _token and "HF_TOKEN" not in os.environ:
        os.environ["HF_TOKEN"] = _token
    if _token and "HUGGING_FACE_HUB_TOKEN" not in os.environ:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = _token
except Exception:
    pass

# ==============================================================================
# Known placeholder hashes and forbidden patterns
# ==============================================================================

KNOWN_PLACEHOLDER_SHA256: set[str] = {
    "9b5936f4006146e4e1e9025b474c02863c0b5614132ad40db4b925a10e8bfbb9",  # Imgur 161x81 placeholder
}

IMAGE_FIELD_CANDIDATES: list[str] = [
    "image",
    "images",
    "edited_image",
    "img",
    "image_bytes",
    "bytes",
    "file",
]
LOCKED_HF_REVISIONS: dict[str, str] = {
    "links-ads/artic-dataset": "ce7f84910595ff72891473719f739a98ef5b0905",
    "Mitsua/art-museums-pd-440k": "fba945da78b36262eb9272067197cc28d06cffbf",
    "bitmind/open-images-v7": "4518ecd40f8f9ef66ee4356be438f840c714e95a",
    "mattymchen/celeba-hq": "1bc5dfbeb766d03abeddf5a48b9871479a4395bb",
    "badigadiii/game_screenshots_11k": "3788cd9b066696217e69732585b350f8dabf6578",
    "hal-utokyo/Manga109-s": "f276b45293be4f4ca92f313a638d26dbcb50fbc2",
    "Chris1/GTA5": "b879d735278f6ac1887b29218708224cb8a35961",
    "theairlabcmu/tartanair2": "0d2d145e973832742a2aaa04b7d2ebffc8d82817",
    "LucasFang/FLUX-Reason-6M": "a92fe58364cbd2273dc10938184f995388052185",
    "bitmind/ideogram-27k": "1afe9f503fdadea5bcc86d7664fd227260cb75b9",
    "Photoroom/midjourney-v6-recap": "21c628db81401da88c5b33507230528cf3fe4a12",
    "ehristoforu/midjourney-images": "ae57258299132e3bd0885e2fe461e1645999de63",
    "VincHa/SD3_medium_synths": "6508181ad0e0020ea81279601e2b66694dcb3cc7",
    "diffusers-parti-prompts/sdxl-1.0": "c7bf903c71c29aa41da9fcfe35feb7dc49ec53d2",
    "nyanko-devs/danbooru2026": "ebb02a630201c7b51487e45fb90b3fcf4cbedc20",
    "Tungtom2004/Google_Nano_Banana_Edited_Images": "e6f5485f619a96c5f93c3ec495f0d063965f16b8",
    "FlameF0X/nano-banana-pro-gen-zh-en": "4e172186cb460e90342d232596a9b66406d66613",
    "UCSC-VLAA/GPT-Image-Edit-1.5M": "018f8e911fae813164d03ca0c27977d9b2a55eb9",
    "innofree/krea2-wildcards": "d78cf76039ed9a5b921ef19cfe07615d37dd3697",
    "ideepankarsharma2003/AIGeneratedImages_Midjourney": "9851ccc4fea851b0f43c4480b8a2795a1fcc0034",
    "Goku-OpenLab/gpt-image-2-prompts-datasets": "78c6abd3af60d793d96166f7e1fc3a1b83a69772",
    "saberzl/SID_Set": "dc03ead57929879319ce30a82bfcfb8d317b10bd",
    "nebula/DF-arrow": "93117d58649bcf660f80fecf2122fac1f59d0453",
    "zye2/tj-data": "dedb4bc2f3b08ec75ff48c4d04de773294a878a4",
}

# ==============================================================================
# Source Configurations with Exact Mapping, Revisions, and Image Fields
# ==============================================================================


@dataclass(frozen=True)
class SourceConfig:
    name: str
    source_type: str  # 'huggingface', 'local_zip', 'local_dir', 'reference_only'
    repo: str = ""
    revision: str = ""
    split: str = "train"
    image_fields: list[str] = field(default_factory=lambda: list(IMAGE_FIELD_CANDIDATES))
    paired_field: str | None = None
    path_field: str | None = None
    default_cap: int = 10_000
    notes: str = ""

    def get_locked_revision(self) -> str:
        if self.repo in LOCKED_HF_REVISIONS:
            return LOCKED_HF_REVISIONS[self.repo]
        if self.revision and self.revision != "main":
            return self.revision
        if self.source_type == "huggingface":
            raise ValueError(
                f"No immutable locked 40-hex commit hash configured for {self.name} ({self.repo})"
            )
        return "local_or_non_hf"


# Unified source registry covering all cohorts in declared manifests
SOURCE_REGISTRY: dict[str, SourceConfig] = {
    # --- Authentic Fine Art & Open Access ---
    "artic_dataset": SourceConfig(
        name="artic_dataset",
        source_type="huggingface",
        repo="links-ads/artic-dataset",
        revision=LOCKED_HF_REVISIONS["links-ads/artic-dataset"],
        split="train",
        default_cap=10_000,
    ),
    "art_museums_pd": SourceConfig(
        name="art_museums_pd",
        source_type="huggingface",
        repo="Mitsua/art-museums-pd-440k",
        revision=LOCKED_HF_REVISIONS["Mitsua/art-museums-pd-440k"],
        split="train",
        default_cap=8_000,
    ),
    "authentic_classical_figure_art": SourceConfig(
        name="authentic_classical_figure_art",
        source_type="huggingface",
        repo="Mitsua/art-museums-pd-440k",
        revision=LOCKED_HF_REVISIONS["Mitsua/art-museums-pd-440k"],
        split="train",
        default_cap=1_000,
    ),
    # --- Authentic Photography & Human Faces ---
    "open_images_v7": SourceConfig(
        name="open_images_v7",
        source_type="huggingface",
        repo="bitmind/open-images-v7",
        revision=LOCKED_HF_REVISIONS["bitmind/open-images-v7"],
        split="train",
        default_cap=10_000,
    ),
    "authentic_glamour_portraits": SourceConfig(
        name="authentic_glamour_portraits",
        source_type="huggingface",
        repo="mattymchen/celeba-hq",
        revision=LOCKED_HF_REVISIONS["mattymchen/celeba-hq"],
        split="train",
        default_cap=1_500,
    ),
    # --- Authentic Graphics & Movies ---
    "sintel_blender_open_movie": SourceConfig(
        name="sintel_blender_open_movie",
        source_type="huggingface",
        repo="badigadiii/game_screenshots_11k",
        revision=LOCKED_HF_REVISIONS["badigadiii/game_screenshots_11k"],
        split="train",
        default_cap=250,
    ),
    "manga109_illustrations": SourceConfig(
        name="manga109_illustrations",
        source_type="huggingface",
        repo="hal-utokyo/Manga109-s",
        revision=LOCKED_HF_REVISIONS["hal-utokyo/Manga109-s"],
        split="train",
        default_cap=1_000,
    ),
    "gta5_driving_renders": SourceConfig(
        name="gta5_driving_renders",
        source_type="huggingface",
        repo="Chris1/GTA5",
        revision=LOCKED_HF_REVISIONS["Chris1/GTA5"],
        split="train",
        default_cap=500,
    ),
    "game_screenshots_fantasy": SourceConfig(
        name="game_screenshots_fantasy",
        source_type="huggingface",
        repo="badigadiii/game_screenshots_11k",
        revision=LOCKED_HF_REVISIONS["badigadiii/game_screenshots_11k"],
        split="train",
        default_cap=250,
    ),
    "tartanair2_ue5_cyberpunk": SourceConfig(
        name="tartanair2_ue5_cyberpunk",
        source_type="huggingface",
        repo="theairlabcmu/tartanair2",
        revision=LOCKED_HF_REVISIONS["theairlabcmu/tartanair2"],
        split="train",
        default_cap=500,
    ),
    # --- Modern Generative AI: FLUX, Midjourney, Ideogram, SD3 ---
    "flux_reason_6m": SourceConfig(
        name="flux_reason_6m",
        source_type="huggingface",
        repo="LucasFang/FLUX-Reason-6M",
        revision=LOCKED_HF_REVISIONS["LucasFang/FLUX-Reason-6M"],
        split="train",
        default_cap=5_000,
    ),
    "flux_cyberpunk_scifi": SourceConfig(
        name="flux_cyberpunk_scifi",
        source_type="huggingface",
        repo="LucasFang/FLUX-Reason-6M",
        revision=LOCKED_HF_REVISIONS["LucasFang/FLUX-Reason-6M"],
        split="train",
        default_cap=500,
    ),
    "ideogram_27k": SourceConfig(
        name="ideogram_27k",
        source_type="huggingface",
        repo="bitmind/ideogram-27k",
        revision=LOCKED_HF_REVISIONS["bitmind/ideogram-27k"],
        split="train",
        default_cap=3_000,
    ),
    "midjourney_v6_recap": SourceConfig(
        name="midjourney_v6_recap",
        source_type="huggingface",
        repo="Photoroom/midjourney-v6-recap",
        revision=LOCKED_HF_REVISIONS["Photoroom/midjourney-v6-recap"],
        split="train",
        default_cap=2_000,
    ),
    "midjourney_v5_images": SourceConfig(
        name="midjourney_v5_images",
        source_type="huggingface",
        repo="ehristoforu/midjourney-images",
        revision=LOCKED_HF_REVISIONS["ehristoforu/midjourney-images"],
        split="train",
        default_cap=1_000,
    ),
    "midjourney_fantasy_environments": SourceConfig(
        name="midjourney_fantasy_environments",
        source_type="huggingface",
        repo="Photoroom/midjourney-v6-recap",
        revision=LOCKED_HF_REVISIONS["Photoroom/midjourney-v6-recap"],
        split="train",
        default_cap=500,
    ),
    "sd3_medium_synths": SourceConfig(
        name="sd3_medium_synths",
        source_type="huggingface",
        repo="VincHa/SD3_medium_synths",
        revision=LOCKED_HF_REVISIONS["VincHa/SD3_medium_synths"],
        split="train",
        default_cap=1_000,
    ),
    "sdxl_photoreal_vehicles": SourceConfig(
        name="sdxl_photoreal_vehicles",
        source_type="huggingface",
        repo="diffusers-parti-prompts/sdxl-1.0",
        revision=LOCKED_HF_REVISIONS["diffusers-parti-prompts/sdxl-1.0"],
        split="train",
        default_cap=500,
    ),
    "danbooru2026_aigc_wild": SourceConfig(
        name="danbooru2026_aigc_wild",
        source_type="huggingface",
        repo="nyanko-devs/danbooru2026",
        revision=LOCKED_HF_REVISIONS["nyanko-devs/danbooru2026"],
        split="train",
        default_cap=500,
    ),
    # --- Public Nano Banana Datasets (KEEP - EXPLICIT CONTRACT) ---
    "google_nano_banana_edited": SourceConfig(
        name="google_nano_banana_edited",
        source_type="huggingface",
        repo="Tungtom2004/Google_Nano_Banana_Edited_Images",
        revision=LOCKED_HF_REVISIONS["Tungtom2004/Google_Nano_Banana_Edited_Images"],
        split="train",
        paired_field="edited_image",
        default_cap=2_500,
        notes="Public Google Nano Banana edited multimodal pairs",
    ),
    "nano_banana_pro_gen": SourceConfig(
        name="nano_banana_pro_gen",
        source_type="huggingface",
        repo="FlameF0X/nano-banana-pro-gen-zh-en",
        revision=LOCKED_HF_REVISIONS["FlameF0X/nano-banana-pro-gen-zh-en"],
        split="train",
        default_cap=2_000,
        notes="Public Nano Banana Pro bilingual generation cohort",
    ),
    # --- Tampered & Edited Datasets ---
    "gpt_image_edit_1_5m": SourceConfig(
        name="gpt_image_edit_1_5m",
        source_type="huggingface",
        repo="UCSC-VLAA/GPT-Image-Edit-1.5M",
        revision=LOCKED_HF_REVISIONS["UCSC-VLAA/GPT-Image-Edit-1.5M"],
        split="train",
        paired_field="edited_image",
        default_cap=5_000,
    ),
    "krea2_wildcards": SourceConfig(
        name="krea2_wildcards",
        source_type="huggingface",
        repo="innofree/krea2-wildcards",
        revision=LOCKED_HF_REVISIONS["innofree/krea2-wildcards"],
        split="train",
        default_cap=2_500,
    ),
    "ai_meme_macro_overlay": SourceConfig(
        name="ai_meme_macro_overlay",
        source_type="huggingface",
        repo="ideepankarsharma2003/AIGeneratedImages_Midjourney",
        revision=LOCKED_HF_REVISIONS["ideepankarsharma2003/AIGeneratedImages_Midjourney"],
        split="train",
        default_cap=1_300,
    ),
    # --- GPT Image 2 & Social Media Datasets ---
    "ai_reaction_banners": SourceConfig(
        name="ai_reaction_banners",
        source_type="huggingface",
        repo="Goku-OpenLab/gpt-image-2-prompts-datasets",
        revision=LOCKED_HF_REVISIONS["Goku-OpenLab/gpt-image-2-prompts-datasets"],
        split="train",
        path_field="file_name",
        default_cap=450,
    ),
    "gpt_image_2_twitter": SourceConfig(
        name="gpt_image_2_twitter",
        source_type="huggingface",
        repo="Goku-OpenLab/gpt-image-2-prompts-datasets",
        revision=LOCKED_HF_REVISIONS["Goku-OpenLab/gpt-image-2-prompts-datasets"],
        split="train",
        path_field="file_name",
        default_cap=100,
    ),
    "scam_ai_social_posts": SourceConfig(
        name="scam_ai_social_posts",
        source_type="huggingface",
        repo="Goku-OpenLab/gpt-image-2-prompts-datasets",
        revision=LOCKED_HF_REVISIONS["Goku-OpenLab/gpt-image-2-prompts-datasets"],
        split="train",
        path_field="file_name",
        default_cap=300,
    ),
    # --- Large Multi-domain Datasets ---
    "sid": SourceConfig(
        name="sid",
        source_type="huggingface",
        repo="saberzl/SID_Set",
        revision=LOCKED_HF_REVISIONS["saberzl/SID_Set"],
        split="train",
        default_cap=30_000,
    ),
    "diffusionforensics": SourceConfig(
        name="diffusionforensics",
        source_type="huggingface",
        repo="nebula/DF-arrow",
        revision=LOCKED_HF_REVISIONS["nebula/DF-arrow"],
        split="train",
        default_cap=12_000,
    ),
    # --- Local Archives ---
    "ewan_gpt_images": SourceConfig(
        name="ewan_gpt_images",
        source_type="local_zip",
        default_cap=102,
        notes="Local archive 100+ AI_Images from ewan.zip",
    ),
    "gpt_image_2_user_screenshots": SourceConfig(
        name="gpt_image_2_user_screenshots",
        source_type="local_dir",
        default_cap=92,
        notes="Authoritative 92-image GPT Image 2 screenshot cohort",
    ),
}

# ==============================================================================
# Atomic File Writers & Path Validation
# ==============================================================================


def atomic_write_bytes(dest_path: Path, data: bytes) -> None:
    """Atomically write bytes to destination via temporary file and replace."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_name(f".{dest_path.name}.tmp_{os.getpid()}_{uuid.uuid4().hex[:8]}")
    try:
        temp_path.write_bytes(data)
        os.replace(temp_path, dest_path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def atomic_save_image(
    image: Any,
    dest_path: Path,
    image_format: str = "JPEG",
    quality: int = 95,
) -> None:
    """Atomically save a PIL Image to destination via temporary file and replace."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_name(f".{dest_path.name}.tmp_{os.getpid()}_{uuid.uuid4().hex[:8]}")
    try:
        fmt = image_format.upper()
        if fmt in ("JPEG", "JPG"):
            image.convert("RGB").save(temp_path, format="JPEG", quality=quality)
        else:
            image.save(temp_path, format=fmt)
        os.replace(temp_path, dest_path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def is_safe_relative_path(path_str: str) -> bool:
    """Reject directory traversal, drive letters, or absolute paths."""
    normalized = path_str.replace("\\", "/").strip()
    if not normalized:
        return False
    if Path(normalized).is_absolute():
        return False
    if bool(re.match(r"^[a-zA-Z]:", normalized)):
        return False
    if normalized.startswith("/"):
        return False
    parts = [p for p in normalized.split("/") if p]
    if ".." in parts:
        return False
    return True


def is_forbidden_row_or_path(row_or_path: Mapping[str, Any] | str) -> tuple[bool, str]:
    """Check whether a row or path matches forbidden datasets or demonstration data.

    Explicitly excludes:
    - 'newer image model data(do not use for training)'
    - Organizer demonstration data (COCO val2017, DALL-E Advanced demo, organizer demo tokens)

    Explicitly retains:
    - Public Nano Banana datasets ('google_nano_banana_edited', 'nano_banana_pro_gen')
    """
    if isinstance(row_or_path, Mapping):
        text = " ".join(
            str(row_or_path.get(col, ""))
            for col in (
                "dataset",
                "generator",
                "image_path",
                "original_path",
                "source_url",
                "external_id",
                "prompt",
                "Category",
                "source_archive_path",
                "source_member_path",
            )
        )
    else:
        text = str(row_or_path or "")

    normalized = text.replace("\\", "/").lower()

    # 1. Explicit forbidden local folder
    if "newer image model data" in normalized:
        return True, "forbidden_newer_image_model_data_folder"

    compact = re.sub(r"[^a-z0-9]+", "", normalized)

    # 2. Organizer COCO val2017 demonstration data
    if "coco" in compact and "val2017" in compact:
        return True, "forbidden_organizer_coco_val2017"

    # 3. Organizer DALL-E Advanced demo / dalle3.csv demonstration data
    if "dalle" in compact and ("advanced" in compact or "dalle3" in compact):
        return True, "forbidden_organizer_dalle_advanced_demo"

    # 4. Explicit organizer demo tokens
    if "organizerdemo" in compact or "organizer_demo" in normalized:
        return True, "forbidden_organizer_demo"

    return False, ""


def is_placeholder_sha(sha: str | None) -> bool:
    """Return True if sha is a placeholder string or known placeholder image."""
    val = str(sha or "").strip().lower()
    if not val:
        return True
    if val.startswith("sha:"):
        return True
    if val in KNOWN_PLACEHOLDER_SHA256:
        return True
    return False


# ==============================================================================
# Manifest Loading and Source-Indexed Grouping
# ==============================================================================


@dataclass
class DeclaredRecord:
    image_path: str
    dataset: str
    split: str
    external_id: str = ""
    source_index: int | None = None
    expected_sha256: str | None = None
    tamper_mask_path: str = ""
    raw_record: dict[str, Any] = field(default_factory=dict)


def parse_source_index(external_id: str, image_path: str) -> int | None:
    """Extract numeric source index from external_id or image_path."""
    m_ext = re.search(r"_(\d{5,8})\b", external_id)
    if m_ext:
        return int(m_ext.group(1))
    m_path = re.search(r"_(\d{5,8})\.", image_path)
    if m_path:
        return int(m_path.group(1))
    m_any = re.search(r"(\d+)", external_id)
    if m_any:
        return int(m_any.group(1))
    return None


def load_declared_manifests(
    manifest_dir: Path,
) -> tuple[list[DeclaredRecord], list[dict[str, Any]]]:
    """Load all declared records from JSONL or Parquet manifests in manifest_dir."""
    splits = ("train", "validation", "test", "test_unseen")
    declared_records: list[DeclaredRecord] = []
    excluded_records: list[dict[str, Any]] = []

    for split_name in splits:
        p_jsonl = manifest_dir / f"{split_name}.jsonl"
        p_parquet = manifest_dir / f"{split_name}.parquet"

        raw_rows: list[dict[str, Any]] = []
        if p_parquet.is_file():
            import pandas as pd

            df = pd.read_parquet(p_parquet)
            raw_rows = df.to_dict(orient="records")
        elif p_jsonl.is_file():
            with open(p_jsonl, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        raw_rows.append(json.loads(line))
        else:
            continue

        for row in raw_rows:
            img_path = str(row.get("image_path", "")).replace("\\", "/").strip()
            ds_name = str(row.get("dataset", "")).strip()
            ext_id = str(row.get("external_id", "")).strip()
            sha_val = str(row.get("sha256", "")).strip()

            if not img_path:
                excluded_records.append(
                    {
                        "image_path": img_path,
                        "dataset": ds_name,
                        "exclusion_reason": "missing_image_path",
                        "split": split_name,
                    }
                )
                continue

            # Path traversal check
            if not is_safe_relative_path(img_path):
                excluded_records.append(
                    {
                        "image_path": img_path,
                        "dataset": ds_name,
                        "exclusion_reason": "unsafe_relative_path",
                        "split": split_name,
                    }
                )
                continue

            # Forbidden data check (newer model data folder, organizer demo)
            forbidden, reason = is_forbidden_row_or_path(row)
            if forbidden:
                excluded_records.append(
                    {
                        "image_path": img_path,
                        "dataset": ds_name,
                        "exclusion_reason": reason,
                        "split": split_name,
                    }
                )
                continue

            # Note: declared placeholder/missing SHA (e.g. 'sha:<dataset>:...') is expected
            # absence of evidence during pre-acquisition scanning. We retain the record
            # so disk bytes can be acquired. Freeze computes actual SHA from verified bytes.
            idx = parse_source_index(ext_id, img_path)
            mask_path = str(row.get("tamper_mask_path", "") or "").replace("\\", "/").strip()
            clean_sha = sha_val.strip().lower() if sha_val else ""
            valid_expected_sha = (
                sha_val
                if (
                    clean_sha
                    and not clean_sha.startswith("sha:")
                    and clean_sha not in KNOWN_PLACEHOLDER_SHA256
                )
                else None
            )

            declared_records.append(
                DeclaredRecord(
                    image_path=img_path,
                    dataset=ds_name,
                    split=split_name,
                    external_id=ext_id,
                    source_index=idx,
                    expected_sha256=valid_expected_sha,
                    tamper_mask_path=mask_path,
                    raw_record=row,
                )
            )

    return declared_records, excluded_records


# ==============================================================================
# Hugging Face Acquisition Logic
# ==============================================================================


def _extract_image_from_item(item: dict[str, Any], config: SourceConfig) -> Any | None:
    """Extract a PIL Image from a dataset item using exact mapped fields."""
    from PIL import Image

    # 1. Path field for metadata-only datasets (e.g. GPT Image 2 prompts datasets)
    if config.path_field:
        source_path = item.get(config.path_field)
        if source_path and isinstance(source_path, str):
            try:
                from huggingface_hub import hf_hub_download

                local_path = hf_hub_download(
                    repo_id=config.repo,
                    repo_type="dataset",
                    filename=source_path,
                    revision=config.get_locked_revision(),
                    token=os.environ.get("HF_TOKEN"),
                )
                return Image.open(local_path)
            except Exception:
                pass

    # 2. Direct URL field
    url = item.get("url")
    if url and isinstance(url, str) and url.startswith("http"):
        import urllib.request

        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read()
            return Image.open(io.BytesIO(data))
        except Exception:
            return None

    # 3. Paired field (for tampered / edited images)
    fields_to_try: list[str] = []
    if config.paired_field:
        fields_to_try.append(config.paired_field)
    fields_to_try.extend(config.image_fields)

    for field_name in fields_to_try:
        raw = item.get(field_name)
        if raw is None:
            continue
        if isinstance(raw, Image.Image):
            return raw
        if isinstance(raw, bytes):
            try:
                return Image.open(io.BytesIO(raw))
            except Exception:
                continue
        if isinstance(raw, dict) and "bytes" in raw and raw["bytes"]:
            try:
                return Image.open(io.BytesIO(raw["bytes"]))
            except Exception:
                continue
        if isinstance(raw, dict) and "path" in raw and raw["path"]:
            try:
                return Image.open(raw["path"])
            except Exception:
                continue

    return None


def acquire_hf_source(
    dataset_name: str,
    records: list[DeclaredRecord],
    config: SourceConfig,
    data_root: Path,
    dry_run: bool = False,
    resume: bool = True,
    verify_bytes: bool = False,
) -> tuple[int, int, int]:
    """Stream from a Hugging Face dataset and write images atomically.

    Returns: (saved_count, skipped_count, error_count)
    """
    from PIL import Image

    needed_by_index: dict[int, list[DeclaredRecord]] = defaultdict(list)
    unresolvable_records: list[DeclaredRecord] = []
    skipped = 0

    for rec in records:
        dest_full = data_root / rec.image_path
        if dest_full.is_file() and dest_full.stat().st_size > 0:
            if verify_bytes:
                try:
                    with Image.open(dest_full) as im:
                        im.verify()
                    skipped += 1
                    continue
                except Exception:
                    pass  # Corrupt file on disk; re-acquire
            elif resume:
                skipped += 1
                continue

        if rec.source_index is not None:
            needed_by_index[rec.source_index].append(rec)
        else:
            unresolvable_records.append(rec)

    total_needed_count = sum(len(recs) for recs in needed_by_index.values()) + len(
        unresolvable_records
    )
    if total_needed_count == 0:
        return 0, skipped, 0

    if dry_run:
        return (
            0,
            skipped + sum(len(recs) for recs in needed_by_index.values()),
            len(unresolvable_records),
        )

    rev = config.get_locked_revision()
    if not rev or rev == "main" or len(rev) != 40:
        raise ValueError(f"Expected 40-hex commit SHA for {config.repo}, got: {rev}")

    # Try streaming from Hugging Face using locked commit SHA
    try:
        from datasets import load_dataset

        ds = load_dataset(
            config.repo,
            split=config.split,
            revision=rev,
            streaming=True,
            token=os.environ.get("HF_TOKEN"),
        )
    except Exception:
        # Fallback without split if dataset is a single partition
        try:
            from datasets import load_dataset

            ds = load_dataset(
                config.repo,
                revision=rev,
                streaming=True,
                token=os.environ.get("HF_TOKEN"),
            )
            if hasattr(ds, "keys"):
                first_split = list(ds.keys())[0]
                ds = ds[first_split]
        except Exception as exc2:
            logger.error(
                "Could not load Hugging Face dataset %s (rev %s): %s", config.repo, rev, exc2
            )
            return 0, skipped, total_needed_count

    max_idx = max(needed_by_index.keys()) if needed_by_index else -1
    saved = 0
    errors = len(unresolvable_records)
    resolved_paths: set[str] = set()

    for i, item in enumerate(ds):
        if max_idx >= 0 and i > max_idx:
            break
        if i not in needed_by_index:
            continue

        target_recs = needed_by_index[i]
        img = _extract_image_from_item(item, config)
        if img is None:
            errors += len(target_recs)
            for tr in target_recs:
                resolved_paths.add(tr.image_path)
            continue

        # Convert image to bytes in memory once to check known placeholder SHA
        buf_format = (
            "PNG" if any(tr.image_path.lower().endswith(".png") for tr in target_recs) else "JPEG"
        )
        try:
            buf = io.BytesIO()
            if img.mode not in ("RGB", "L") and buf_format == "JPEG":
                img = img.convert("RGB")
            img.save(buf, format=buf_format)
            img_bytes = buf.getvalue()
            actual_sha = hashlib.sha256(img_bytes).hexdigest()
            if actual_sha in KNOWN_PLACEHOLDER_SHA256:
                logger.warning(
                    "Item %d in %s has known placeholder SHA %s; rejecting",
                    i,
                    config.repo,
                    actual_sha,
                )
                errors += len(target_recs)
                for tr in target_recs:
                    resolved_paths.add(tr.image_path)
                continue
        except Exception as exc:
            logger.warning("Failed to encode image at index %d in %s: %s", i, config.repo, exc)
            errors += len(target_recs)
            for tr in target_recs:
                resolved_paths.add(tr.image_path)
            continue

        # Atomically materialize every distinct destination from the decoded item
        for target_rec in target_recs:
            dest_full = data_root / target_rec.image_path
            try:
                ext = dest_full.suffix.lower()
                target_format = "PNG" if ext == ".png" else "JPEG"
                if target_format == buf_format:
                    atomic_write_bytes(dest_full, img_bytes)
                else:
                    atomic_save_image(img, dest_full, image_format=target_format)
                saved += 1
                resolved_paths.add(target_rec.image_path)
            except Exception as exc:
                logger.warning("Failed to save image %s: %s", dest_full, exc)
                errors += 1
                resolved_paths.add(target_rec.image_path)

    # After dataset iteration, count every still-unresolved requested destination as error
    for target_recs in needed_by_index.values():
        for tr in target_recs:
            if tr.image_path not in resolved_paths:
                errors += 1

    return saved, skipped, errors


# ==============================================================================
# Local Archive Acquisition (Ewan GPT Images)
# ==============================================================================


def acquire_ewan_gpt_images(
    records: list[DeclaredRecord],
    data_root: Path,
    dry_run: bool = False,
    resume: bool = True,
    verify_bytes: bool = False,
) -> tuple[int, int, int]:
    """Extract ewan_gpt_images from the local zip file or zye2/tj-data atomically."""
    from PIL import Image

    zip_candidates = [
        _PROJECT_ROOT / "100+ AI_Images from ewan.zip",
        _PROJECT_ROOT / "sources" / "100+ AI_Images from ewan.zip",
        data_root / "sources" / "100+ AI_Images from ewan.zip",
        data_root / "100+ AI_Images from ewan.zip",
    ]
    zip_path: Path | None = None
    for cand in zip_candidates:
        if cand.is_file():
            zip_path = cand
            break

    if not zip_path:
        # Try owner-authorized zye2/tj-data remote source using locked commit SHA
        try:
            from huggingface_hub import hf_hub_download

            locked_rev = LOCKED_HF_REVISIONS.get("zye2/tj-data", "")
            local_zip = hf_hub_download(
                repo_id="zye2/tj-data",
                repo_type="dataset",
                filename="sources/100+ AI_Images from ewan.zip",
                revision=locked_rev or None,
                token=os.environ.get("HF_TOKEN"),
            )
            zip_path = Path(local_zip)
        except Exception as exc:
            logger.warning("Could not download ewan.zip from zye2/tj-data: %s", exc)

    if not zip_path:
        logger.warning("Ewan zip file not found locally or in zye2/tj-data")
        skipped = 0
        errors = 0
        for rec in records:
            dest_full = data_root / rec.image_path
            if dest_full.is_file() and dest_full.stat().st_size > 0:
                skipped += 1
            else:
                errors += 1
        return 0, skipped, errors

    saved = 0
    skipped = 0
    errors = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        zip_names = set(zf.namelist())
        for rec in records:
            dest_full = data_root / rec.image_path
            if dest_full.is_file() and dest_full.stat().st_size > 0:
                if verify_bytes:
                    try:
                        with Image.open(dest_full) as im:
                            im.verify()
                        skipped += 1
                        continue
                    except Exception:
                        pass  # Corrupt file; re-extract
                elif resume:
                    skipped += 1
                    continue

            entry_cand = rec.image_path
            if entry_cand.startswith("ewan_gpt_images/"):
                entry_cand = entry_cand[len("ewan_gpt_images/") :]

            matched_entry: str | None = None
            if entry_cand in zip_names:
                matched_entry = entry_cand
            else:
                norm = entry_cand.replace("\\", "/")
                if norm in zip_names:
                    matched_entry = norm

            if dry_run:
                if matched_entry is not None:
                    skipped += 1
                else:
                    errors += 1
                continue

            if matched_entry is None:
                logger.warning("Member %s not found in ewan zip", entry_cand)
                errors += 1
                continue

            try:
                data = zf.read(matched_entry)
                if not data or len(data) == 0:
                    errors += 1
                    continue
                actual_sha = hashlib.sha256(data).hexdigest()
                if actual_sha in KNOWN_PLACEHOLDER_SHA256:
                    errors += 1
                    continue
                atomic_write_bytes(dest_full, data)
                saved += 1
            except Exception as exc:
                logger.warning("Failed to extract %s from ewan zip: %s", matched_entry, exc)
                errors += 1

    return saved, skipped, errors


def acquire_gpt_image_2_user_screenshots(
    records: list[DeclaredRecord],
    data_root: Path,
    dry_run: bool = False,
    resume: bool = True,
    verify_bytes: bool = False,
) -> tuple[int, int, int]:
    """Acquire gpt_image_2_user_screenshots from local candidates or zye2/tj-data."""
    from PIL import Image

    search_dirs = [
        _PROJECT_ROOT / "gpt_image_2_user_screenshots",
        _PROJECT_ROOT / "sources" / "gpt_image_2_user_screenshots",
        data_root / "gpt_image_2_user_screenshots",
        data_root / "sources" / "gpt_image_2_user_screenshots",
    ]

    saved = 0
    skipped = 0
    errors = 0

    for rec in records:
        dest_full = data_root / rec.image_path
        if dest_full.is_file() and dest_full.stat().st_size > 0:
            if verify_bytes:
                try:
                    with Image.open(dest_full) as im:
                        im.verify()
                    skipped += 1
                    continue
                except Exception:
                    pass
            elif resume:
                skipped += 1
                continue

        img_name = Path(rec.image_path).name

        # Try local candidates
        found_source: Path | None = None
        for s_dir in search_dirs:
            cand = s_dir / img_name
            if cand.is_file() and cand.stat().st_size > 0:
                found_source = cand
                break

        if dry_run:
            if found_source is not None:
                skipped += 1
            else:
                errors += 1
            continue

        if found_source is not None:
            try:
                data = found_source.read_bytes()
                actual_sha = hashlib.sha256(data).hexdigest()
                if actual_sha in KNOWN_PLACEHOLDER_SHA256:
                    errors += 1
                    continue
                atomic_write_bytes(dest_full, data)
                saved += 1
                continue
            except Exception:
                pass

        # Try remote zye2/tj-data on Hugging Face with locked commit SHA
        try:
            from huggingface_hub import hf_hub_download

            locked_rev = LOCKED_HF_REVISIONS.get("zye2/tj-data", "")
            hf_path = hf_hub_download(
                repo_id="zye2/tj-data",
                repo_type="dataset",
                filename=rec.image_path,
                revision=locked_rev or None,
                token=os.environ.get("HF_TOKEN"),
            )
            data = Path(hf_path).read_bytes()
            actual_sha = hashlib.sha256(data).hexdigest()
            if actual_sha in KNOWN_PLACEHOLDER_SHA256:
                errors += 1
                continue
            atomic_write_bytes(dest_full, data)
            saved += 1
            continue
        except Exception:
            pass

        errors += 1

    return saved, skipped, errors


def resolve_hf_source_revision(repo_id: str, default_rev: str = "") -> str:
    """Resolve immutable 40-hex commit SHA from HF API or locked registry."""
    if repo_id in LOCKED_HF_REVISIONS:
        return LOCKED_HF_REVISIONS[repo_id]
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        info = api.dataset_info(repo_id, timeout=5)
        if hasattr(info, "sha") and info.sha:
            return str(info.sha)
    except Exception:
        pass
    return default_rev or "unknown_revision"


def generate_source_revisions_report(
    records_by_dataset: dict[str, list[DeclaredRecord]],
    stats_by_dataset: dict[str, dict[str, int]],
    output_path: Path,
) -> dict[str, Any]:
    """Generate and write source_revisions.json."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_sources: dict[str, Any] = {}

    for ds_name, recs in records_by_dataset.items():
        cfg = SOURCE_REGISTRY.get(ds_name)
        repo = cfg.repo if cfg else ""
        if cfg and cfg.source_type == "huggingface":
            rev = resolve_hf_source_revision(repo, cfg.get_locked_revision())
        else:
            rev = "local_or_non_hf"
        stats = stats_by_dataset.get(ds_name, {})
        # Compute digest of expected sha256 values
        sha_list = sorted(rec.expected_sha256 or "" for rec in recs)
        digest = hashlib.sha256("".join(sha_list).encode("utf-8")).hexdigest()

        report_sources[ds_name] = {
            "source_type": cfg.source_type if cfg else "reference_only",
            "repo": repo,
            "revision": rev,
            "split": cfg.split if cfg else "train",
            "declared_rows": len(recs),
            "acquired_rows": stats.get("saved", 0),
            "skipped_existing": stats.get("skipped", 0),
            "errors": stats.get("errors", 0),
            "expected_sha_digest": digest,
        }

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "total_declared_rows": sum(len(r) for r in records_by_dataset.values()),
        "sources": report_sources,
    }

    atomic_write_bytes(output_path, json.dumps(report, indent=2).encode("utf-8"))
    return report


# ==============================================================================
# CLI Entry Point
# ==============================================================================


def parse_caps(caps_arg: str | None) -> dict[str, int]:
    """Parse JSON or file mapping dataset names to integer caps."""
    if not caps_arg:
        return {}
    path_cand = Path(caps_arg)
    if path_cand.is_file():
        content = path_cand.read_text(encoding="utf-8")
        parsed = json.loads(content)
    else:
        parsed = json.loads(caps_arg)
    return {str(k): int(v) for k, v in parsed.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Acquire and prefetch images for MechaDetect production pipeline"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Local image root (default: TECHJAM_DATA_ROOT or ./data)",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=_PROJECT_ROOT / "splits" / "final_teacher_dataset",
        help="Manifest directory containing JSONL or Parquet splits",
    )
    parser.add_argument(
        "--sources",
        type=str,
        default="",
        help="Comma-separated dataset sources to acquire",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="",
        help="Single dataset source to acquire",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run without writing files",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Resume safely, skipping already acquired files (default)",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Force re-acquisition of existing images",
    )
    parser.add_argument(
        "--verify-bytes",
        action="store_true",
        help="Verify decodability of existing files on disk and re-acquire corrupt ones",
    )
    parser.add_argument(
        "--caps",
        type=str,
        default=None,
        help="JSON string or file with per-source caps",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Optional path to write audit report",
    )
    parser.add_argument(
        "--revisions-path",
        type=Path,
        default=_PROJECT_ROOT / "splits" / "production_eligible" / "source_revisions.json",
        help="Path to write source_revisions.json",
    )

    args = parser.parse_args()

    # Fallback manifest directory if final_teacher_dataset does not exist
    manifest_dir = args.manifest_dir
    if not manifest_dir.is_dir():
        cand_combined = _PROJECT_ROOT / "splits" / "combined_hf_dataset"
        cand_eligible = _PROJECT_ROOT / "splits" / "production_eligible"
        if cand_combined.is_dir():
            manifest_dir = cand_combined
        elif cand_eligible.is_dir():
            manifest_dir = cand_eligible
    print("=" * 70)
    print("MechaDetect Resumable Image Prefetch & Acquisition")
    print("=" * 70)
    print(f"Data root:     {args.data_root}")
    print(f"Manifest dir:  {manifest_dir}")
    print(f"Dry run:       {args.dry_run}")
    print(f"Resume:        {args.resume}")
    print(f"Verify bytes:  {args.verify_bytes}")
    print("=" * 70)

    # Ingest manifests
    declared_records, excluded_records = load_declared_manifests(manifest_dir)
    print(
        f"Loaded {len(declared_records)} declared records ({len(excluded_records)} excluded during scan)"
    )

    # Filter sources if requested
    selected_sources = set()
    if args.sources:
        selected_sources.update(s.strip() for s in args.sources.split(",") if s.strip())
    if args.source:
        selected_sources.add(args.source.strip())

    # Group by dataset
    records_by_dataset: dict[str, list[DeclaredRecord]] = defaultdict(list)
    for rec in declared_records:
        if not selected_sources or rec.dataset in selected_sources:
            records_by_dataset[rec.dataset].append(rec)

    # Apply caps if provided
    caps = parse_caps(args.caps)
    if caps:
        for ds, limit in caps.items():
            if ds in records_by_dataset and len(records_by_dataset[ds]) > limit:
                records_by_dataset[ds] = records_by_dataset[ds][:limit]

    print(f"Active datasets to process: {len(records_by_dataset)}")

    stats_by_dataset: dict[str, dict[str, int]] = {}
    total_saved = 0
    total_skipped = 0
    total_errors = 0

    for ds_name, recs in sorted(records_by_dataset.items()):
        cfg = SOURCE_REGISTRY.get(ds_name)
        if not cfg:
            print(f"  [{ds_name}] No registry entry; skipping remote acquisition")
            stats_by_dataset[ds_name] = {"saved": 0, "skipped": 0, "errors": len(recs)}
            continue

        if cfg.source_type == "huggingface":
            saved, skipped, errors = acquire_hf_source(
                ds_name,
                recs,
                cfg,
                args.data_root,
                dry_run=args.dry_run,
                resume=args.resume,
                verify_bytes=args.verify_bytes,
            )
        elif cfg.source_type == "local_zip" and ds_name == "ewan_gpt_images":
            saved, skipped, errors = acquire_ewan_gpt_images(
                recs,
                args.data_root,
                dry_run=args.dry_run,
                resume=args.resume,
                verify_bytes=args.verify_bytes,
            )
        elif ds_name == "gpt_image_2_user_screenshots":
            saved, skipped, errors = acquire_gpt_image_2_user_screenshots(
                recs,
                args.data_root,
                dry_run=args.dry_run,
                resume=args.resume,
                verify_bytes=args.verify_bytes,
            )
        else:
            # Local directory / reference only: audit actual presence on disk
            saved = 0
            skipped = 0
            errors = 0
            for rec in recs:
                if (args.data_root / rec.image_path).is_file():
                    skipped += 1
                else:
                    errors += 1
        stats_by_dataset[ds_name] = {"saved": saved, "skipped": skipped, "errors": errors}
        total_saved += saved
        total_skipped += skipped
        total_errors += errors
        print(f"  [{ds_name:32s}] saved={saved:<5d} skipped={skipped:<6d} errors={errors:<4d}")

    print("=" * 70)
    print(f"Summary: Saved={total_saved}, Skipped={total_skipped}, Errors={total_errors}")
    print("=" * 70)

    # Write source_revisions.json
    rev_report = generate_source_revisions_report(
        records_by_dataset, stats_by_dataset, args.revisions_path
    )
    print(f"Wrote source revisions report to {args.revisions_path}")

    if args.report_path:
        atomic_write_bytes(
            args.report_path,
            json.dumps(
                {
                    "generated_at": datetime.now(UTC).isoformat(),
                    "total_declared": len(declared_records),
                    "total_excluded_preflight": len(excluded_records),
                    "stats": stats_by_dataset,
                    "revisions": rev_report,
                },
                indent=2,
            ).encode("utf-8"),
        )
        print(f"Wrote acquisition audit report to {args.report_path}")


if __name__ == "__main__":
    main()
