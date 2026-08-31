#!/usr/bin/env python3
"""Build the final teacher dataset as a quality-first, varied, auditable package.

Single CLI entry point to:
1. Ingest existing candidate manifests (JSONL, Parquet, CSV) and validate against
   local materialized images under --data-root. Recompute dimensions, format,
   SHA-256, and perceptual difference hash (dHash) from disk bytes.
2. Safely import the authoritative 92-image GPT Image 2 screenshot archive
   into high-quality RGB JPEGs with deterministic naming and archive metadata.
3. Enforce quality-first domain quotas / caps, global duplicate grouping, and
   strict group-disjoint 4-split assignment (train, validation, test, test_unseen)
   with zero cross-split duplicate leakage and zero generator leakage into train.
4. Reject all forbidden data (COCO val2017, DALL-E Advanced demo, organizer demo,
   and 'newer image model data(do not use for training)') across all splits.
5. Emit JSONL + Parquet for each split and an audit_report.json in the output dir.
6. Fail closed when disk space is insufficient or when no valid local bytes exist.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import itertools
import json
import logging
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any
import zipfile

import numpy as np
import pandas as pd
from PIL import Image

# Ensure repository root and src are in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from aigc_detector.constants import PROVENANCE_NAMES
from aigc_detector.dataset import parse_provenance
from aigc_detector.manifests import (
    assert_no_group_leakage,
    difference_hash,
    duplicate_groups,
    file_sha256,
    is_held_out_generator,
    manifest_digest,
    normalize_generator,
)
from aigc_detector.runtime import load_local_environment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("build_final_teacher_dataset")

# ==============================================================================
# Canonical Schema Columns (matching data/COMBINED_DATASET_SCHEMA.md)
# ==============================================================================

ALL_SCHEMA_COLUMNS: tuple[str, ...] = (
    "image_path",
    "label",
    "dataset",
    "official_split",
    "generator",
    "manipulation_family",
    "source_image_group",
    "width",
    "height",
    "file_format",
    "tamper_mask_path",
    "source_url",
    "external_id",
    "generator_family",
    "generator_version",
    "prompt",
    "created_at",
    "sha256",
    "perceptual_hash",
    "quality_score",
    "provenance_confidence",
    "redistribution_mode",
    "origin_license",
    "license_url",
    "attribution",
    "selection_reason",
    "forbidden_demo_checked",
    "ai_positive",
    "split",
    "duplicate_group",
    "provenance",
    "domain",
    "cohort",
    "source_archive_path",
    "source_archive_sha256",
    "source_member_path",
    "source_member_sha256",
)

TARGET_SPLITS: tuple[str, ...] = ("train", "validation", "test", "test_unseen")

KNOWN_PLACEHOLDER_SHA256: set[str] = {
    "9b5936f4006146e4e1e9025b474c02863c0b5614132ad40db4b925a10e8bfbb9",  # Imgur 161x81 placeholder
}

MIN_REQUIRED_FREE_BYTES: int = 50 * 1024 * 1024  # 50 MB minimum headroom

HUMAN_TAMPERED_DATASETS: set[str] = {
    "gmorinan_memes",
    "dank_learning_templates",
    "imgflip_memes",
    "multioff_memes",
}

AI_TAMPERED_DATASETS: set[str] = {
    "sid",
    "gpt_image_edit_1_5m",
    "google_nano_banana_edited",
    "ai_meme_macro_overlay",
}


# ==============================================================================
# Parquet Engine Verification
# ==============================================================================

def verify_parquet_engine_available() -> str:
    """Ensure pyarrow or fastparquet is importable up front."""
    try:
        import pyarrow
        return f"pyarrow {pyarrow.__version__}"
    except ImportError:
        pass
    try:
        import fastparquet
        return "fastparquet"
    except ImportError:
        pass
    raise RuntimeError(
        "Parquet engine unavailable. Building the final teacher dataset requires 'pyarrow' "
        "or 'fastparquet' to emit .parquet manifests. Please run within the project virtualenv "
        "(e.g. .venv/Scripts/python.exe) or install pyarrow."
    )


# ==============================================================================
# Disk Space & Security Validation
# ==============================================================================

def check_destination_free_space(path: Path, min_required_bytes: int = MIN_REQUIRED_FREE_BYTES) -> None:
    """Check that destination volume has sufficient free space, failing closed."""
    resolved = path.resolve()
    candidate = resolved
    while not candidate.exists() and candidate.parent != candidate:
        candidate = candidate.parent
    if not candidate.exists():
        candidate = Path(resolved.anchor or "/")
    usage = shutil.disk_usage(candidate)
    if usage.free < min_required_bytes:
        raise RuntimeError(
            f"Destination '{resolved}' on volume '{candidate.anchor}' has only {usage.free} bytes free. "
            f"Required at least {min_required_bytes} bytes ({min_required_bytes // (1024 * 1024)} MB). "
            "Build failed closed to protect storage."
        )


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


def is_forbidden_path_or_source(text_or_row: Any) -> bool:
    """Reject organizer demonstration data and the forbidden newer model folder."""
    if isinstance(text_or_row, Mapping):
        text = " ".join(
            str(text_or_row.get(col, ""))
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
        text = str(text_or_row or "")

    normalized = text.replace("\\", "/").lower()

    # Reject 'newer image model data(do not use for training)'
    if "newer image model data" in normalized:
        return True

    compact = re.sub(r"[^a-z0-9]+", "", normalized)

    # 1. Organizer COCO val2017 demonstration data
    if "coco" in compact and "val2017" in compact:
        return True

    # 2. Organizer DALL-E Advanced demo / dalle3.csv demonstration data
    if "dalle" in compact and ("advanced" in compact or "dalle3" in compact):
        return True

    # 3. Explicit organizer demo tokens
    if "organizerdemo" in compact or "organizer_demo" in normalized:
        return True

    return False


# ==============================================================================
# Difference Hash & Image Quality Computation
# ==============================================================================

def compute_image_difference_hash(image: Image.Image, size: int = 8) -> int:
    """Compute difference hash (dHash) directly on a PIL Image instance."""
    grayscale = image.convert("L").resize((size + 1, size), Image.Resampling.BICUBIC)
    pixels = list(grayscale.getdata())
    value = 0
    for row in range(size):
        row_offset = row * (size + 1)
        for col in range(size):
            value = (value << 1) | int(pixels[row_offset + col] > pixels[row_offset + col + 1])
    return value


def compute_bounded_quality_score(image: Image.Image) -> float:
    """Compute a deterministic, bounded [0.0, 1.0] image quality score from pixels.

    Checks pixel variance (to catch flat/blank placeholders) and resolution/aspect ratio.
    """
    w, h = image.size
    if w < 16 or h < 16:
        return 0.0

    stat_thumb = image.convert("L").resize((64, 64), Image.Resampling.BILINEAR)
    arr = np.asarray(stat_thumb, dtype=np.float32)
    var = float(np.var(arr))
    if var < 1.0:
        return 0.0

    ar = max(w / max(h, 1), h / max(w, 1))
    if ar > 10.0:
        return 0.0

    res_score = min(min(w, h) / 512.0, 1.0)
    ar_penalty = 1.0 if ar <= 3.0 else max(0.5, 1.0 - (ar - 3.0) * 0.1)
    var_score = min(var / 400.0, 1.0)

    score = 0.50 + 0.25 * res_score + 0.15 * ar_penalty + 0.10 * var_score
    return round(float(np.clip(score, 0.10, 1.0)), 4)


# ==============================================================================
# Domain and Cohort Derivation
# ==============================================================================

def derive_domain_cohort(dataset: str, provenance: str, generator_family: str) -> tuple[str, str]:
    """Derive deterministic domain and cohort descriptors for auditing and analysis."""
    ds = dataset.lower().replace("-", "_")
    prov = provenance.lower()
    gen = generator_family.lower()

    if ds in {"gpt_image_2_user_screenshots", "gpt_image_2"}:
        return "screenshot", "gpt_image_2_user_screenshots"

    if ds in {"sid", "sid_sanity"}:
        if prov == "authentic":
            return "photoreal_camera", "sid_authentic"
        if prov == "tampered":
            return "inpainting_tampered", "sid_tampered"
        return "photoreal_synthetic", "sid_synthetic"

    if ds in {"wildfake", "wildfake_subset"}:
        return "wildfake_diverse", f"wildfake_{gen or 'unknown'}"

    if ds in {"diffusionforensics", "diffusionforensics_subset"}:
        return "forensic_diffusion", f"diffusionforensics_{gen or 'unknown'}"

    if ds in HUMAN_TAMPERED_DATASETS:
        return "meme", "meme_human_overlay"

    if ds == "ai_meme_macro_overlay":
        return "meme", "meme_ai_overlay"

    if "screenshot" in ds:
        return "screenshot", ds

    if prov == "authentic":
        return "authentic_curated", f"{ds}_authentic"
    if prov == "tampered":
        return "tampered_curated", f"{ds}_tampered"
    return "synthetic_curated", f"{ds}_{gen or 'synthetic'}"


# ==============================================================================
# Safe Atomic Writing Helpers
# ==============================================================================

def deterministic_jpeg_bytes(image: Image.Image) -> bytes:
    """Encode an RGB image to canonical quality-95 JPEG bytes without EXIF."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=95, optimize=False, progressive=False, exif=b"")
    return buf.getvalue()


def safe_atomic_save_image(image: Image.Image, destination: Path) -> None:
    """Save an image atomically, refusing to overwrite non-identical existing files."""
    expected_bytes = deterministic_jpeg_bytes(image)
    expected_sha = hashlib.sha256(expected_bytes).hexdigest()

    if destination.is_file():
        existing_sha = file_sha256(destination)
        if existing_sha == expected_sha:
            return  # Byte-identical, safe reuse
        raise FileExistsError(
            f"Destination file '{destination}' already exists with different contents "
            f"(existing={existing_sha[:16]}, expected={expected_sha[:16]}). "
            "Refusing to overwrite existing user data."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.parent / f".tmp_{destination.name}"
    try:
        temp_path.write_bytes(expected_bytes)
        temp_path.replace(destination)
    except Exception:
        if temp_path.is_file():
            temp_path.unlink(missing_ok=True)
        raise


# ==============================================================================
# Authoritative GPT Image 2 ZIP Import Engine
# ==============================================================================

@dataclass(frozen=True)
class ZipImportResult:
    records: list[dict[str, Any]]
    archive_sha256: str
    total_images_in_archive: int
    imported_count: int
    destination_dir: Path


def import_authoritative_gpt_image_2_zip(
    zip_path: Path,
    destination_dir: Path,
    data_root: Path | None = None,
    min_quality: float = 0.0,
    dry_run: bool = False,
) -> ZipImportResult:
    """Import the authoritative 92 GPT Image 2 screenshot images into RGB JPEGs.

    Preserves source archive path, member SHA-256, and archive metadata.
    """
    if not zip_path.is_file():
        raise FileNotFoundError(f"User archive not found: {zip_path}")

    archive_sha = file_sha256(zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        image_members = [
            n for n in namelist
            if not n.endswith("/")
            and not n.startswith("__MACOSX")
            and not Path(n).name.startswith(".")
            and Path(n).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        ]
        image_members.sort()
        if len(image_members) != 92:
            raise ValueError(
                f"Authoritative GPT Image 2 archive must contain exactly 92 image members; "
                f"found {len(image_members)}"
            )

        if not dry_run:
            estimated_bytes = sum(zf.getinfo(m).file_size for m in image_members) + MIN_REQUIRED_FREE_BYTES
            check_destination_free_space(destination_dir, min_required_bytes=estimated_bytes)
            destination_dir.mkdir(parents=True, exist_ok=True)

        records: list[dict[str, Any]] = []

        for idx, member_name in enumerate(image_members):
            if not is_safe_relative_path(member_name):
                raise ValueError(f"Unsafe path in ZIP archive: {member_name}")
            if is_forbidden_path_or_source(member_name):
                raise ValueError(f"Forbidden source path in archive: {member_name}")

            member_info = zf.getinfo(member_name)
            try:
                member_created_iso = datetime(*member_info.date_time, tzinfo=timezone.utc).isoformat()
            except Exception:
                member_created_iso = "2026-08-29T00:00:00+00:00"
            raw_bytes = zf.read(member_name)
            member_sha = hashlib.sha256(raw_bytes).hexdigest()

            with Image.open(io.BytesIO(raw_bytes)) as source_img:
                rgb_img = source_img.convert("RGB")
                width, height = rgb_img.size
                q_score = compute_bounded_quality_score(rgb_img)
                if q_score < min_quality:
                    raise ValueError(
                        f"Authoritative image {member_name} quality {q_score:.4f} below min_quality {min_quality:.4f}"
                    )
                dest_filename = f"gpt_image_2_screenshot_{idx + 1:03d}.jpg"
                dest_file_path = destination_dir / dest_filename

                if dry_run:
                    file_sha = hashlib.sha256(deterministic_jpeg_bytes(rgb_img)).hexdigest()
                    phash_int = compute_image_difference_hash(rgb_img)
                else:
                    safe_atomic_save_image(rgb_img, dest_file_path)
                    file_sha = file_sha256(dest_file_path)
                    phash_int = difference_hash(dest_file_path)

            phash_hex = f"{phash_int:016x}"

            # Calculate relative path if destination is under data_root
            if data_root is not None:
                try:
                    rel_image_path = dest_file_path.resolve().relative_to(data_root.resolve()).as_posix()
                except ValueError:
                    rel_image_path = dest_file_path.resolve().as_posix()
            else:
                rel_image_path = dest_file_path.resolve().as_posix()

            domain_desc, cohort_desc = derive_domain_cohort("gpt_image_2_user_screenshots", "fully_aigc", "gpt_image")

            record: dict[str, Any] = {
                "image_path": rel_image_path,
                "label": 2,
                "dataset": "gpt_image_2_user_screenshots",
                "official_split": "train",
                "generator": "gpt_image_2",
                "manipulation_family": "",
                "source_image_group": f"gpt_image_2_user_screenshots:screenshot_{idx + 1:03d}",
                "width": width,
                "height": height,
                "file_format": "JPEG",
                "tamper_mask_path": "",
                "source_url": f"archive:{zip_path.name}:{member_name}",
                "external_id": member_name,
                "generator_family": "gpt_image",
                "generator_version": "2",
                "created_at": member_created_iso,
                "sha256": file_sha,
                "perceptual_hash": phash_hex,
                "quality_score": q_score,
                "provenance_confidence": "high",
                "redistribution_mode": "embed_bytes",
                "origin_license": "User Authoritative Archive",
                "license_url": "",
                "attribution": "User authoritative GPT Image 2 screenshots",
                "selection_reason": "Authoritative user GPT Image 2 positive screenshot cohort",
                "forbidden_demo_checked": True,
                "ai_positive": 1,
                "split": "train",
                "duplicate_group": "",
                "provenance": "fully_aigc",
                "domain": domain_desc,
                "cohort": cohort_desc,
                "source_archive_path": zip_path.resolve().as_posix(),
                "source_archive_sha256": archive_sha,
                "source_member_path": member_name,
                "source_member_sha256": member_sha,
            }
            records.append(record)

    logger.info(
        "Imported %d GPT Image 2 records from %s (archive sha256=%s).",
        len(records),
        zip_path.name,
        archive_sha[:16],
    )
    return ZipImportResult(
        records=records,
        archive_sha256=archive_sha,
        total_images_in_archive=len(image_members),
        imported_count=len(records),
        destination_dir=destination_dir,
    )


# ==============================================================================
# Manifest Ingestion & Recomputation Engine
# ==============================================================================

@dataclass
class IngestionReport:
    manifest_paths: list[str]
    total_raw_rows: int
    usable_rows_count: int
    skipped_by_dataset: dict[str, dict[str, int]]
    usable_by_dataset: dict[str, int]


def normalize_data_roots(data_root: Path | Sequence[Path]) -> list[Path]:
    """Normalize one or more roots used to resolve relative manifest paths."""
    raw_roots = [data_root] if isinstance(data_root, Path) else list(data_root)
    roots = [Path(root).resolve() for root in raw_roots if str(root).strip()]
    if not roots:
        raise ValueError("At least one data root is required to ingest local images")
    return list(dict.fromkeys(roots))
def find_containing_root(path: Path, roots: Sequence[Path]) -> Path | None:
    """Return the first configured root containing path, including symlink resolution."""
    resolved_path = path.resolve()
    for root in roots:
        try:
            resolved_path.relative_to(root)
            return root
        except ValueError:
            continue
    return None


def inspect_local_record(
    rec: Mapping[str, Any],
    data_roots: Sequence[Path],
    min_quality: float,
    manifest_path: str,
) -> tuple[str, dict[str, Any]]:
    """Validate one local row and return its computed byte-level measurements."""
    ds_name = str(rec.get("dataset", "unknown")).strip()
    raw_img_path = str(rec.get("image_path", "")).strip()
    if not raw_img_path:
        return "missing_image_path_field", {}

    if is_forbidden_path_or_source(rec) or is_forbidden_path_or_source(raw_img_path):
        raise ValueError(
            f"Forbidden organizer demonstration or newer model data detected in manifest {manifest_path}: "
            f"dataset={ds_name}, path={raw_img_path}"
        )

    img_path_obj = Path(raw_img_path.replace("\\", "/"))
    full_image_path: Path | None = None
    image_root: Path | None = None
    if img_path_obj.is_absolute():
        candidate = img_path_obj.resolve()
        image_root = find_containing_root(candidate, data_roots)
        if image_root is None:
            return "absolute_path_outside_root", {}
        full_image_path = candidate
    else:
        if not is_safe_relative_path(raw_img_path):
            return "unsafe_path_traversal", {}
        for root in data_roots:
            candidate = (root / img_path_obj).resolve()
            if find_containing_root(candidate, [root]) is not None and candidate.is_file():
                full_image_path = candidate
                image_root = root
                break

    if full_image_path is None or image_root is None or not full_image_path.is_file():
        return "file_not_found_on_disk", {}

    try:
        with Image.open(full_image_path) as im:
            width, height = im.size
            file_format = (im.format or "JPEG").upper()
            quality_score = compute_bounded_quality_score(im)
    except Exception:
        return "unreadable_corrupted_image", {}

    if width < 16 or height < 16 or quality_score == 0.0:
        return "image_too_small_or_flat", {}
    if quality_score < min_quality:
        return "below_min_quality_threshold", {}

    sha256 = file_sha256(full_image_path)
    if sha256 in KNOWN_PLACEHOLDER_SHA256:
        return "known_placeholder_sha256", {}

    perceptual_hash = f"{difference_hash(full_image_path):016x}"

    raw_mask_path = str(rec.get("tamper_mask_path", "") or "").strip()
    tamper_mask_path = ""
    if raw_mask_path and raw_mask_path.lower() != "nan":
        mask_obj = Path(raw_mask_path.replace("\\", "/"))
        full_mask_path: Path | None = None
        mask_root: Path | None = None
        if mask_obj.is_absolute():
            candidate = mask_obj.resolve()
            mask_root = find_containing_root(candidate, data_roots)
            if mask_root is None:
                return "mask_outside_root", {}
            full_mask_path = candidate
        else:
            if not is_safe_relative_path(raw_mask_path):
                return "unsafe_mask_traversal", {}
            for root in data_roots:
                candidate = (root / mask_obj).resolve()
                if find_containing_root(candidate, [root]) is not None and candidate.is_file():
                    full_mask_path = candidate
                    mask_root = root
                    break

        if full_mask_path is None or mask_root is None or not full_mask_path.is_file():
            return "missing_tamper_mask_file", {}
        tamper_mask_path = full_mask_path.relative_to(mask_root).as_posix()

    return "", {
        "full_image_path": full_image_path,
        "image_root": image_root,
        "width": width,
        "height": height,
        "file_format": file_format,
        "quality_score": quality_score,
        "sha256": sha256,
        "perceptual_hash": perceptual_hash,
        "tamper_mask_path": tamper_mask_path,
    }


def ingest_candidate_manifests(
    manifest_paths: Sequence[Path],
    data_root: Path | Sequence[Path],
    min_quality: float = 0.0,
    dry_run: bool = False,
) -> tuple[list[dict[str, Any]], IngestionReport]:
    """Load manifests, resolving each relative image against all local roots."""
    usable_records: list[dict[str, Any]] = []
    skipped_stats: dict[str, Counter[str]] = {}
    usable_stats: dict[str, int] = {}
    total_raw = 0

    resolved_data_roots = normalize_data_roots(data_root)



    for m_path in manifest_paths:
        if not m_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {m_path}")

        logger.info("Ingesting manifest: %s", m_path)
        suffix = m_path.suffix.lower()
        if suffix == ".parquet":
            df = pd.read_parquet(m_path)
            raw_records = df.to_dict(orient="records")
        elif suffix in {".jsonl", ".ndjson"}:
            raw_records = []
            with open(m_path, "r", encoding="utf-8") as f:
                for line in f:
                    line_str = line.strip()
                    if line_str:
                        raw_records.append(json.loads(line_str))
        elif suffix == ".csv":
            df = pd.read_csv(m_path).fillna("")
            raw_records = df.to_dict(orient="records")
        else:
            raise ValueError(f"Unsupported manifest format: {m_path.suffix}")

        total_raw += len(raw_records)

        try:
            requested_workers = int(os.environ.get("FINAL_TEACHER_INGEST_WORKERS", "32"))
        except ValueError:
            requested_workers = 32
        worker_count = max(1, min(requested_workers, len(raw_records) or 1))
        logger.info("Validating %d rows with %d image workers", len(raw_records), worker_count)

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            inspections = executor.map(
                inspect_local_record,
                raw_records,
                itertools.repeat(resolved_data_roots),
                itertools.repeat(min_quality),
                itertools.repeat(str(m_path)),
            )
            for rec, (skip_reason, measured) in zip(raw_records, inspections, strict=True):
                ds_name = str(rec.get("dataset", "unknown")).strip()
                if ds_name not in skipped_stats:
                    skipped_stats[ds_name] = Counter()
                if skip_reason:
                    skipped_stats[ds_name][skip_reason] += 1
                    continue

                raw_img_path = str(rec.get("image_path", "")).strip()
                full_image_path = measured["full_image_path"]
                image_root = measured["image_root"]
                width = int(measured["width"])
                height = int(measured["height"])
                file_format = str(measured["file_format"])
                quality_score = float(measured["quality_score"])
                sha256 = str(measured["sha256"])
                perceptual_hash = str(measured["perceptual_hash"])
                tamper_mask_path = str(measured["tamper_mask_path"])

                # Parse label & provenance semantics
                raw_label_val = rec.get("label", rec.get("provenance", ""))
                prov_enum = parse_provenance(raw_label_val, ds_name)
                prov_str = PROVENANCE_NAMES[int(prov_enum)]
                label_int = int(prov_enum)

                # Determine binary task label ai_positive explicitly
                raw_ai_pos = rec.get("ai_positive")
                if raw_ai_pos is not None and str(raw_ai_pos).strip() in {"0", "1"}:
                    ai_positive = int(raw_ai_pos)
                elif prov_str == "authentic":
                    ai_positive = 0
                elif prov_str == "fully_aigc":
                    ai_positive = 1
                elif prov_str == "tampered":
                    ai_positive = 0 if ds_name in HUMAN_TAMPERED_DATASETS else 1
                else:
                    ai_positive = 0

                generator = normalize_generator(rec.get("generator", ""), ds_name)
                generator_family = normalize_generator(rec.get("generator_family", "") or generator, ds_name)

                # Source image group: ensure non-blank per-image group
                src_group = str(rec.get("source_image_group", "") or "").strip()
                if not src_group or src_group.lower() == "nan":
                    src_group = f"{ds_name}:{sha256[:16]}"

                # Keep paths portable; the selected root is not part of the manifest.
                portable_img_path = full_image_path.relative_to(image_root).as_posix()

                domain_desc, cohort_desc = derive_domain_cohort(ds_name, prov_str, generator_family)

                record = {
                    "image_path": portable_img_path,
                    "label": label_int,
                    "dataset": ds_name,
                    "official_split": str(rec.get("official_split", "") or rec.get("split", "") or "train"),
                    "generator": generator,
                    "manipulation_family": str(rec.get("manipulation_family", "") or ""),
                    "source_image_group": src_group,
                    "width": width,
                    "height": height,
                    "file_format": file_format,
                    "tamper_mask_path": tamper_mask_path,
                    "source_url": str(rec.get("source_url", "") or ""),
                    "external_id": str(rec.get("external_id", "") or Path(raw_img_path).stem),
                    "generator_family": generator_family,
                    "generator_version": str(rec.get("generator_version", "") or ""),
                    "created_at": str(rec.get("created_at", "") or ""),
                    "sha256": sha256,
                    "perceptual_hash": perceptual_hash,
                    "quality_score": quality_score,
                    "provenance_confidence": str(rec.get("provenance_confidence", "high") or "high"),
                    "redistribution_mode": "embed_bytes",
                    "origin_license": str(rec.get("origin_license", "Research") or "Research"),
                    "license_url": str(rec.get("license_url", "") or ""),
                    "attribution": str(rec.get("attribution", ds_name) or ds_name),
                    "selection_reason": str(rec.get("selection_reason", "Materialized candidate") or "Materialized candidate"),
                    "forbidden_demo_checked": True,
                    "ai_positive": ai_positive,
                    "split": str(rec.get("split", "") or rec.get("official_split", "") or "train"),
                    "duplicate_group": str(rec.get("duplicate_group", "") or ""),
                    "provenance": prov_str,
                    "domain": domain_desc,
                    "cohort": cohort_desc,
                    "source_archive_path": str(rec.get("source_archive_path", "") or ""),
                    "source_archive_sha256": str(rec.get("source_archive_sha256", "") or ""),
                    "source_member_path": str(rec.get("source_member_path", "") or ""),
                    "source_member_sha256": str(rec.get("source_member_sha256", "") or ""),
                }
                usable_records.append(record)
                usable_stats[ds_name] = usable_stats.get(ds_name, 0) + 1

    report = IngestionReport(
        manifest_paths=[str(p) for p in manifest_paths],
        total_raw_rows=total_raw,
        usable_rows_count=len(usable_records),
        skipped_by_dataset={k: dict(v) for k, v in skipped_stats.items()},
        usable_by_dataset=usable_stats,
    )
    return usable_records, report


# ==============================================================================
# Quality-First Quotas and Group-Aware Split Assignment
# ==============================================================================

def stable_record_key(record: Mapping[str, Any]) -> tuple[str, str, str, str]:
    """Return a stable ordering key independent of manifest input order."""
    return (
        str(record.get("dataset", "")),
        str(record.get("sha256", "")),
        str(record.get("image_path", "")),
        str(record.get("external_id", "")),
    )


def apply_domain_caps(
    records: list[dict[str, Any]],
    caps: Mapping[str, int],
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Select quality-first domain quotas, prioritizing higher quality scores and keeping groups intact."""
    if not caps:
        return sorted(records, key=stable_record_key)

    records_by_key: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        records_by_key.setdefault(str(record["dataset"]), []).append(record)

    selected_records: list[dict[str, Any]] = []

    for key in sorted(records_by_key):
        group_records = records_by_key[key]
        cap = caps.get(key)
        if cap is None and group_records:
            domain = group_records[0].get("domain")
            if domain in caps:
                cap = caps[domain]

        if cap is None:
            selected_records.extend(sorted(group_records, key=stable_record_key))
            continue
        if cap < 0:
            raise ValueError(f"Cap for {key!r} must be non-negative, got {cap}")
        if len(group_records) <= cap:
            selected_records.extend(sorted(group_records, key=stable_record_key))
            continue

        groups: dict[str, list[dict[str, Any]]] = {}
        for record in group_records:
            groups.setdefault(str(record["source_image_group"]), []).append(record)
        for rows in groups.values():
            rows.sort(key=stable_record_key)

        max_group_size = max(len(rows) for rows in groups.values())
        if max_group_size > cap:
            raise ValueError(
                f"Cap {cap} for {key!r} is smaller than an indivisible source group "
                f"of {max_group_size} rows"
            )

        group_items: list[tuple[float, int, str, list[dict[str, Any]]]] = []
        for group_key, rows in groups.items():
            average_quality = float(
                np.mean([float(row.get("quality_score", 0.0) or 0.0) for row in rows])
            )
            tie_breaker = int.from_bytes(
                hashlib.sha256(f"{seed}:{group_key}".encode()).digest()[:4], "big"
            )
            group_items.append((average_quality, tie_breaker, group_key, rows))

        # Prefer the highest-quality complete groups; hashes make ties deterministic.
        group_items.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)

        accumulated: list[dict[str, Any]] = []
        for _, _, _, rows in group_items:
            if len(accumulated) + len(rows) <= cap:
                accumulated.extend(rows)

        selected_records.extend(accumulated)
        logger.info(
            "Quality-first capped '%s' from %d to %d rows (min quality: %.4f, max: %.4f).",
            key,
            len(group_records),
            len(accumulated),
            min(float(row["quality_score"]) for row in accumulated) if accumulated else 0.0,
            max(float(row["quality_score"]) for row in accumulated) if accumulated else 0.0,
        )
    return selected_records


def assign_group_disjoint_splits(
    records: list[dict[str, Any]],
    seed: int = 42,
) -> pd.DataFrame:
    """Cluster duplicate groups and assign group-disjoint 4-way splits.

    Guarantees:
      1. Every image in a duplicate group belongs to the exact same split.
      2. No cross-split duplicate SHA-256 or perceptual hash leakage.
      3. test_unseen is strictly reserved for synthetic generator families wholly absent from train.
      4. Forbidden demonstration data is rejected across ALL splits.
    """
    df = pd.DataFrame(records).copy()
    if df.empty:
        return df

    sha_values = [str(val) for val in df["sha256"]]
    phash_ints = [int(str(val), 16) for val in df["perceptual_hash"]]
    source_groups = [str(val) for val in df["source_image_group"]]

    # Cluster duplicate groups across all candidate rows
    df["duplicate_group"] = duplicate_groups(
        sha_values,
        phash_ints,
        source_groups,
        max_hamming_distance=4,
    )
    conflicting_groups = (
        df.groupby("duplicate_group")["ai_positive"].nunique().loc[lambda values: values > 1]
    )
    if not conflicting_groups.empty:
        conflicting_ids = set(str(group_id) for group_id in conflicting_groups.index)
        quarantined_rows = int(df["duplicate_group"].isin(conflicting_ids).sum())
        logger.warning(
            "Quarantining %d rows in %d cross-label duplicate groups before split assignment.",
            quarantined_rows,
            len(conflicting_ids),
        )
        df = df[~df["duplicate_group"].isin(conflicting_ids)].copy()
        df.attrs["quarantined_conflicting_duplicate_groups"] = len(conflicting_ids)
        df.attrs["quarantined_conflicting_rows"] = quarantined_rows

    # Normalize generator and generator family
    generators = df["generator"] if "generator" in df else [""] * len(df)
    df["generator_family"] = [
        normalize_generator(gen, ds)
        for gen, ds in zip(generators, df["dataset"], strict=True)
    ]

    # Assign candidate split
    synthetic = df["provenance"].astype(str).str.lower().isin({"fully_aigc", "aigc", "ai"})
    splits = []

    for is_synth, gen_fam, official, dup_grp in zip(
        synthetic, df["generator_family"], df["official_split"], df["duplicate_group"], strict=True
    ):
        official_clean = str(official).strip().lower()
        is_held = is_held_out_generator(gen_fam, seed)

        if is_synth and is_held:
            splits.append("test_unseen")
        elif official_clean == "test_unseen" and is_synth and is_held:
            splits.append("test_unseen")
        elif official_clean in {"validation", "val", "dev"}:
            splits.append("validation")
        elif official_clean in {"test", "testing"}:
            splits.append("test")
        else:
            # Deterministic hash bucket based on group and seed
            group_hash = int.from_bytes(
                hashlib.sha256(f"{seed}:{dup_grp}".encode()).digest()[:4],
                "big",
            )
            bucket = group_hash % 100
            if bucket < 70:
                splits.append("train")
            elif bucket < 85:
                splits.append("validation")
            else:
                splits.append("test")

    df["split"] = splits

    # Enforce strict group disjointness (all rows in a duplicate_group get the same split)
    priority = {"train": 0, "validation": 1, "test": 2, "test_unseen": 3}
    for _, indices in df.groupby("duplicate_group").groups.items():
        grp_idx = list(indices)
        best_split = min(
            (df.loc[idx, "split"] for idx in grp_idx),
            key=lambda item: priority.get(item, 99),
        )
        df.loc[grp_idx, "split"] = best_split

    # Reserve test_unseen strictly for generator families wholly absent from train
    trained_synthetic = df.loc[(df["split"] == "train") & synthetic, "generator_family"].astype(str)
    trained_families = set(trained_synthetic.tolist())

    for _, indices in df.groupby("duplicate_group").groups.items():
        grp_idx = list(indices)
        if not (df.loc[grp_idx, "split"] == "test_unseen").any():
            continue
        grp_synthetic = synthetic.loc[grp_idx]
        grp_families = set(
            df.loc[[idx for idx in grp_idx if grp_synthetic.loc[idx]], "generator_family"]
            .astype(str)
            .tolist()
        )
        # If any generator family in the group was seen in train, demote group to test
        if grp_families & trained_families or not grp_synthetic.any():
            df.loc[grp_idx, "split"] = "test"

    # Remap any rogue split labels into the canonical 4 target splits
    valid_splits = set(TARGET_SPLITS)
    df.loc[~df["split"].isin(valid_splits), "split"] = "test"

    # Strict Assertions
    assert_no_group_leakage(df)

    # Assert no SHA-256 collision across splits
    for s1, s2 in itertools.combinations(TARGET_SPLITS, 2):
        s1_shas = set(df.loc[df["split"] == s1, "sha256"])
        s2_shas = set(df.loc[df["split"] == s2, "sha256"])
        leakage = s1_shas & s2_shas
        if leakage:
            raise ValueError(
                f"Cross-split duplicate SHA256 leakage detected between {s1} and {s2}: {len(leakage)} hashes"
            )

    # Assert no generator family leakage into train from test_unseen
    unseen_families = set(
        df.loc[(df["split"] == "test_unseen") & synthetic, "generator_family"].astype(str)
    )
    overlap = trained_families & unseen_families
    if overlap:
        raise ValueError(
            f"Strict unseen generator leakage: families {overlap} appear in both train and test_unseen"
        )

    # Assert forbidden demonstration data absent across ALL splits
    for idx, row in df.iterrows():
        if is_forbidden_path_or_source(row.to_dict()):
            raise ValueError(
                f"Forbidden demonstration data found in row {idx} (split={row['split']}): {row['image_path']}"
            )

    return df


# ==============================================================================
# Packaging and Export Engine
# ==============================================================================

def export_final_dataset_package(
    frame: pd.DataFrame,
    output_dir: Path,
    ingestion_report: IngestionReport | None = None,
    zip_result: ZipImportResult | None = None,
    seed: int = 42,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Export the 4 target splits to JSONL + Parquet and generate audit_report.json."""
    if not dry_run:
        check_destination_free_space(output_dir, min_required_bytes=MIN_REQUIRED_FREE_BYTES)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure all canonical columns exist
    export_df = frame.copy()
    for col in ALL_SCHEMA_COLUMNS:
        if col not in export_df.columns:
            export_df[col] = ""

    export_df = export_df[list(ALL_SCHEMA_COLUMNS)]

    split_dfs: dict[str, pd.DataFrame] = {
        split: export_df[export_df["split"] == split].copy()
        for split in TARGET_SPLITS
    }

    if not dry_run:
        for split_name, s_df in split_dfs.items():
            jsonl_path = output_dir / f"{split_name}.jsonl"
            parquet_path = output_dir / f"{split_name}.parquet"

            # Write JSONL atomically
            records = s_df.to_dict(orient="records")
            temp_jsonl = output_dir / f".tmp_{jsonl_path.name}"
            with open(temp_jsonl, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
            temp_jsonl.replace(jsonl_path)

            # Write Parquet atomically
            temp_parquet = output_dir / f".tmp_{parquet_path.name}"
            s_df.to_parquet(temp_parquet, index=False, engine="auto")
            temp_parquet.replace(parquet_path)

            logger.info("Wrote %d rows to %s and %s", len(s_df), jsonl_path.name, parquet_path.name)

    # Compute comprehensive audit report
    digest = manifest_digest(export_df)
    train_syn = export_df.loc[(export_df["split"] == "train") & (export_df["provenance"] == "fully_aigc"), "generator_family"]
    unseen_syn = export_df.loc[(export_df["split"] == "test_unseen") & (export_df["provenance"] == "fully_aigc"), "generator_family"]

    counts_by_domain_cohort = {
        ds: int(count) for ds, count in export_df["dataset"].value_counts().items()
    }
    if "cohort" in frame.columns:
        counts_by_domain_cohort["user_screenshot_cohort"] = int((frame["cohort"] == "gpt_image_2_user_screenshots").sum())

    audit_report: dict[str, Any] = {
        "manifest_sha256": digest,
        "total_rows": int(len(export_df)),
        "split_counts": {split: int(len(s_df)) for split, s_df in split_dfs.items()},
        "counts_by_dataset": {k: int(v) for k, v in export_df["dataset"].value_counts().items()},
        "counts_by_provenance": {k: int(v) for k, v in export_df["provenance"].value_counts().items()},
        "counts_by_ai_positive": {str(k): int(v) for k, v in export_df["ai_positive"].value_counts().items()},
        "counts_by_generator": {k: int(v) for k, v in export_df["generator"].value_counts().items() if k},
        "counts_by_domain": {k: int(v) for k, v in export_df["domain"].value_counts().items() if k},
        "counts_by_cohort": {k: int(v) for k, v in export_df["cohort"].value_counts().items() if k},
        "counts_by_domain_cohort": counts_by_domain_cohort,
        "counts_by_split_and_dataset": {
            split: {k: int(v) for k, v in s_df["dataset"].value_counts().items()}
            for split, s_df in split_dfs.items()
        },
        "counts_by_split_and_provenance": {
            split: {k: int(v) for k, v in s_df["provenance"].value_counts().items()}
            for split, s_df in split_dfs.items()
        },
        "counts_by_split_and_ai_positive": {
            split: {str(k): int(v) for k, v in s_df["ai_positive"].value_counts().items()}
            for split, s_df in split_dfs.items()
        },
        "counts_by_split_and_domain": {
            split: {k: int(v) for k, v in s_df["domain"].value_counts().items()}
            for split, s_df in split_dfs.items()
        },
        "train_generator_families": sorted(set(train_syn.dropna().astype(str).tolist())),
        "unseen_generator_families": sorted(set(unseen_syn.dropna().astype(str).tolist())),
        "generator_family_leakage_detected": bool(set(train_syn) & set(unseen_syn)),
        "duplicate_groups_count": int(export_df["duplicate_group"].nunique()) if "duplicate_group" in export_df else 0,
        "quarantined_conflicting_duplicate_groups": int(frame.attrs.get("quarantined_conflicting_duplicate_groups", 0)),
        "quarantined_conflicting_rows": int(frame.attrs.get("quarantined_conflicting_rows", 0)),
        "duplicate_hash_leakage_detected": False,
        "forbidden_demo_check_passed": True,
        "newer_model_data_excluded": True,
        "gpt_image_2_zip_metadata": {
            "archive_sha256": zip_result.archive_sha256 if zip_result else "",
            "total_images_in_archive": zip_result.total_images_in_archive if zip_result else 0,
            "imported_images": zip_result.imported_count if zip_result else 0,
            "cohort_name": "gpt_image_2_user_screenshots",
            "cohort_type": "narrow_screenshot_positive",
            "destination_dir": str(zip_result.destination_dir) if zip_result else "",
        } if zip_result else None,
        "skipped_summary": ingestion_report.skipped_by_dataset if ingestion_report else {},
        "seed": seed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    if not dry_run:
        audit_path = output_dir / "audit_report.json"
        temp_audit = output_dir / f".tmp_{audit_path.name}"
        with open(temp_audit, "w", encoding="utf-8") as f:
            json.dump(audit_report, f, indent=2, ensure_ascii=False)
        temp_audit.replace(audit_path)
        logger.info("Saved audit report to %s", audit_path.name)

    return audit_report


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def parse_caps_argument(caps_arg: str | None) -> dict[str, int]:
    """Parse caps from a JSON string, a file path, or comma-separated key=value pairs."""
    if not caps_arg:
        return {}
    caps_str = caps_arg.strip()
    if Path(caps_str).is_file():
        with open(caps_str, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if caps_str.endswith((".yaml", ".yml")):
                try:
                    import yaml
                    return {str(k): int(v) for k, v in yaml.safe_load(content).items()}
                except ImportError:
                    pass
            return {str(k): int(v) for k, v in json.loads(content).items()}

    if caps_str.startswith("{"):
        return {str(k): int(v) for k, v in json.loads(caps_str).items()}

    result = {}
    for item in caps_str.split(","):
        item_clean = item.strip()
        if not item_clean:
            continue
        if "=" not in item_clean:
            raise ValueError(f"Invalid cap specifier '{item_clean}'. Expected 'dataset=1000' or JSON.")
        k, v = item_clean.split("=", 1)
        result[k.strip()] = int(v.strip())
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the final teacher dataset as a quality-first, varied, auditable package.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        nargs="*",
        type=Path,
        default=[],
        help="One or more candidate manifest paths (.parquet, .jsonl, .csv) referencing materialized local images.",
    )
    parser.add_argument(
        "--data-root",
        action="append",
        type=Path,
        default=None,
        help="Root for local images; repeat to search multiple materialized roots.",
    )
    parser.add_argument(
        "--zip",
        type=Path,
        default=None,
        help="Path to user-provided gpt-image-2-synthetic-images.zip archive.",
    )
    parser.add_argument(
        "--zip-destination",
        type=Path,
        default=None,
        help="Destination directory where extracted GPT Image 2 RGB JPEGs should be stored.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("splits/final_teacher_dataset"),
        help="Output directory to emit train, validation, test, test_unseen JSONL+Parquet and audit_report.json.",
    )
    parser.add_argument(
        "--caps",
        type=str,
        default=None,
        help="Caps per dataset for variety: JSON string (e.g. '{\"sid\": 10000}'), key=val pairs, or path to caps file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling and group-aware split assignment.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run: validate manifests and inspect ZIP, reporting usable versus skipped rows without writing any files.",
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.0,
        help="Minimum quality threshold (default: 0.0).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML configuration file specifying default arguments.",
    )
    return parser


def main() -> None:
    # 1. Verify parquet engine up front
    parquet_engine = verify_parquet_engine_available()
    logger.info("Verified Parquet engine: %s", parquet_engine)

    parser = build_parser()
    args = parser.parse_args()

    # Load environment variables if .env exists
    load_local_environment(_PROJECT_ROOT)

    # Load config file if provided
    if args.config:
        if not args.config.is_file():
            raise FileNotFoundError(f"Config file not found: {args.config}")
        try:
            import yaml
            with open(args.config, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
                for k, v in cfg.items():
                    dest_k = k.replace("-", "_")
                    if not hasattr(args, dest_k):
                        continue
                    current = getattr(args, dest_k)
                    if current not in (None, parser.get_default(dest_k)):
                        continue
                    if dest_k == "manifest" and isinstance(v, list):
                        setattr(args, dest_k, [Path(x) for x in v])
                    elif dest_k == "data_root" and v:
                        values = v if isinstance(v, list) else [v]
                        setattr(args, dest_k, [Path(x) for x in values])
                    elif dest_k in {"zip", "zip_destination", "output_dir"} and v:
                        setattr(args, dest_k, Path(v))
                    else:
                        setattr(args, dest_k, v)
        except ImportError:
            logger.warning("PyYAML not installed; cannot parse YAML config file %s", args.config)

    # Resolve one or more data roots. Explicit CLI roots take precedence over .env.
    data_roots = args.data_root
    if not data_roots:
        env_root = os.environ.get("TECHJAM_DATA_ROOT", "").strip()
        data_roots = [Path(env_root)] if env_root else []
    if not data_roots and args.manifest:
        logger.warning("No --data-root supplied and TECHJAM_DATA_ROOT not set; using current working directory.")
        data_roots = [Path.cwd()]
    caps = parse_caps_argument(args.caps)

    # 2. Ingest candidate manifests
    candidate_records: list[dict[str, Any]] = []
    ingestion_report: IngestionReport | None = None

    if args.manifest:
        manifest_paths = [p.resolve() if p.is_absolute() else (_PROJECT_ROOT / p).resolve() for p in args.manifest]
        candidate_records, ingestion_report = ingest_candidate_manifests(
            manifest_paths=manifest_paths,
            data_root=data_roots,
            min_quality=args.min_quality,
            dry_run=args.dry_run,
        )
        logger.info(
            "Manifest ingestion complete: %d usable rows out of %d inspected.",
            len(candidate_records),
            ingestion_report.total_raw_rows,
        )

    # 3. Import authoritative GPT Image 2 ZIP if provided
    zip_result: ZipImportResult | None = None
    if args.zip:
        zip_path = args.zip.resolve() if args.zip.is_absolute() else (_PROJECT_ROOT / args.zip).resolve()
        zip_dest = (
            args.zip_destination.resolve()
            if args.zip_destination and args.zip_destination.is_absolute()
            else (
                _PROJECT_ROOT / args.zip_destination
                if args.zip_destination
                else data_roots[0] / "gpt_image_2_user_screenshots"
            ).resolve()
        )
        zip_base_root = find_containing_root(zip_dest, data_roots) or data_roots[0]

        zip_result = import_authoritative_gpt_image_2_zip(
            zip_path=zip_path,
            destination_dir=zip_dest,
            data_root=zip_base_root,
            min_quality=args.min_quality,
            dry_run=args.dry_run,
        )
        candidate_records.extend(zip_result.records)

    # 4. Fail closed if no valid bytes exist
    if not candidate_records:
        error_msg = (
            "Error: No valid local image bytes were selected. "
            f"Inspected {ingestion_report.total_raw_rows if ingestion_report else 0} manifest rows "
            "(all missing or unreadable on disk), and no valid archive images were ingested. "
            "Build failed closed."
        )
        logger.error(error_msg)
        if ingestion_report and ingestion_report.skipped_by_dataset:
            logger.error("Skipped summary by dataset: %s", json.dumps(ingestion_report.skipped_by_dataset, indent=2))
        sys.exit(1)

    # 5. Apply quality-first domain quotas / caps
    selected_records = apply_domain_caps(candidate_records, caps, seed=args.seed)
    logger.info("Selected %d records across %d datasets.", len(selected_records), len(set(r['dataset'] for r in selected_records)))

    # 6. Global duplicate grouping and group-disjoint split assignment
    split_frame = assign_group_disjoint_splits(selected_records, seed=args.seed)

    # 7. Packaging & Audit
    report = export_final_dataset_package(
        frame=split_frame,
        output_dir=args.output_dir.resolve() if args.output_dir.is_absolute() else (_PROJECT_ROOT / args.output_dir).resolve(),
        ingestion_report=ingestion_report,
        zip_result=zip_result,
        seed=args.seed,
        dry_run=args.dry_run,
    )

    # Summary output
    print("\n================ FINAL TEACHER DATASET BUILD REPORT ================")
    print(f"Mode: {'DRY RUN (no files written)' if args.dry_run else 'PRODUCTION BUILD'}")
    print(f"Manifest SHA-256: {report['manifest_sha256']}")
    print(f"Total Rows:       {report['total_rows']}")
    print(f"Split Counts:     {report['split_counts']}")
    print(f"Datasets:         {report['counts_by_dataset']}")
    print(f"Provenance:       {report['counts_by_provenance']}")
    print(f"AI Positive:      {report['counts_by_ai_positive']}")
    print(f"Domains:          {report['counts_by_domain']}")
    print(f"Cohorts:          {report['counts_by_cohort']}")
    print(f"Unseen Families:  {report['unseen_generator_families']}")
    print("Verification:     Group Leakage Checked=True, Cross-Split SHA Checked=True")
    print(f"                  Forbidden Demo Checked={report['forbidden_demo_check_passed']}")
    print(f"                  Newer Model Data Excluded={report['newer_model_data_excluded']}")
    if ingestion_report:
        print(f"Ingestion:        {ingestion_report.usable_rows_count} usable / {ingestion_report.total_raw_rows} raw")
        if any(ingestion_report.skipped_by_dataset.values()):
            print("Skipped Breakdown:")
            for ds, reasons in ingestion_report.skipped_by_dataset.items():
                if reasons:
                    print(f"  {ds}: {reasons}")
    if zip_result:
        print(f"User ZIP:         {zip_result.imported_count} images imported from {zip_result.archive_sha256[:16]}")
    print("====================================================================\n")


if __name__ == "__main__":
    main()
