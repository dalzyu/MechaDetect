#!/usr/bin/env python3
"""Prepare a metadata-first Hugging Face transparency package for the teacher dataset.

Public and source datasets contribute only provenance/index/metadata rows.
Private team-generated Ewan images and authoritative GPT Image 2 screenshots are
projected with explicit payload policies ('metadata_only' or 'private_upload') and
optionally staged as image files under strict privacy gates.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import io
import json
import logging
from pathlib import Path
import re
import sys
from typing import Any
import zipfile

import pandas as pd
from PIL import Image

# Ensure repository root and src are on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from aigc_detector.manifests import file_sha256, manifest_digest
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("prepare_hf_transparency")

# ==============================================================================
# Canonical Schema Definition
# ==============================================================================

CANONICAL_TRANSPARENCY_COLUMNS: tuple[str, ...] = (
    "row_id",
    "dataset",
    "source_url",
    "external_id",
    "source_index",
    "source_image_group",
    "duplicate_group",
    "split",
    "official_split",
    "domain",
    "cohort",
    "label",
    "ai_positive",
    "generator",
    "generator_family",
    "generator_version",
    "manipulation_family",
    "provenance",
    "width",
    "height",
    "file_format",
    "sha256",
    "perceptual_hash",
    "quality_score",
    "provenance_confidence",
    "origin_license",
    "license_url",
    "attribution",
    "redistribution_mode",
    "selection_reason",
    "image_path",
    "image_payload_policy",
    "forbidden_demo_checked",
    "prompt",
    "tamper_mask_path",
    "source_archive_path",
    "source_archive_sha256",
    "source_member_path",
    "source_member_sha256",
    "created_at",
)

PRIVATE_DATASET_NAMES: set[str] = {
    "ewan_gpt_images",
    "gpt_image_2_user_screenshots",
}

PRIVATE_COHORT_NAMES: set[str] = {
    "gpt_image_2_user_screenshots",
    "ewan_gpt_images_gpt_image",
}

HUMAN_TAMPERED_DATASETS: set[str] = {
    "gmorinan_memes",
    "dank_learning_templates",
    "imgflip_memes",
    "multioff_memes",
}


# ==============================================================================
# Safety & Forbidden Content Validation
# ==============================================================================

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
    if isinstance(text_or_row, (Mapping, pd.Series)) or hasattr(text_or_row, "get"):
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
                "cohort",
            )
        )
    elif text_or_row is None:
        text = ""
    else:
        text = str(text_or_row)

    normalized = text.replace("\\", "/").lower()

    # Reject 'newer image model data(do not use for training)'
    if "newer image model data" in normalized or "newer_image_model" in normalized:
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


def assert_forbidden_data_absent(df: pd.DataFrame) -> None:
    """Assert that no forbidden demonstration or newer-model data appears in the frame."""
    violations: list[int] = []
    for idx, row in df.iterrows():
        if is_forbidden_path_or_source(row):
            violations.append(int(idx))
    if violations:
        raise ValueError(
            f"Forbidden organizer demonstration or newer-model data detected in {len(violations)} rows "
            f"(sample row indices: {violations[:5]}). Aborting to protect compliance."
        )


def strip_and_assert_no_image_bytes(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure no image bytes, bytearrays, numpy arrays, or PIL image objects exist in any column."""
    import numpy as np

    columns_to_drop = [
        col for col in df.columns
        if col.lower() in {"image", "bytes", "data", "image_bytes", "pil_image", "thumbnail", "raw_bytes"}
        and col not in {"image_path", "image_payload_policy"}
    ]
    if columns_to_drop:
        logger.info("Dropping raw binary/image columns from manifest: %s", columns_to_drop)
        df = df.drop(columns=columns_to_drop)

    for col in df.columns:
        if col in {"image_path", "image_payload_policy"}:
            continue
        if df[col].dtype == object:
            for idx, val in enumerate(df[col]):
                if isinstance(val, (bytes, bytearray, memoryview, Image.Image, np.ndarray)):
                    raise ValueError(
                        f"Column '{col}' at row {idx} contains binary or PIL Image object ({type(val).__name__})! "
                        "Transparency manifests must be strictly metadata-only."
                    )
    return df


# ==============================================================================
# Domain & Cohort Derivation
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
# Deterministic Row ID and Projection
# ==============================================================================

def compute_stable_row_id(row: Mapping[str, Any]) -> str:
    """Compute a deterministic 32-hex ID from immutable row identity fields."""
    split = str(row.get("split", "")).strip()
    dataset = str(row.get("dataset", "")).strip()
    image_path = str(row.get("image_path", "")).strip().replace("\\", "/")
    sha = str(row.get("sha256", "")).strip()
    ext_id = str(row.get("external_id", "")).strip()
    source_index = str(row.get("source_index", "")).strip()
    source_member = str(row.get("source_member_path", "")).strip().replace("\\", "/")
    raw_key = f"{split}:{dataset}:{source_index}:{image_path}:{sha}:{ext_id}:{source_member}"
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:32]
def _parse_bool(val: Any, default: bool = True) -> bool:
    """Safely parse boolean values avoiding truthy string pitfalls like bool('False')."""
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in {"false", "0", "no", "f", "off"}:
        return False
    if s in {"true", "1", "yes", "t", "on"}:
        return True
    return default


def compute_stable_manifest_digest(frame: pd.DataFrame) -> str:
    """Compute deterministic SHA-256 digest across all stable row identity attributes."""
    sort_cols = [
        c for c in ["split", "dataset", "image_path", "row_id", "sha256", "external_id"]
        if c in frame.columns
    ]
    sorted_df = frame.sort_values(sort_cols).reset_index(drop=True)
    records = sorted_df.fillna("").to_dict(orient="records")
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def classify_payload_policy(
    row: Mapping[str, Any],
    ewan_archive_supplied: bool,
    gpt2_archive_supplied: bool,
) -> tuple[str, bool]:
    """Classify the row payload policy.

    Returns:
        (policy: 'private_upload' | 'metadata_only', is_missing_private_payload: bool)
    """
    dataset = str(row.get("dataset", "")).strip().lower()
    cohort = str(row.get("cohort", "")).strip().lower()

    is_ewan = (dataset in {"ewan_gpt_images", "ewan"} or "ewan" in cohort)
    is_gpt2 = (
        dataset == "gpt_image_2_user_screenshots"
        or cohort == "gpt_image_2_user_screenshots"
    )
    if is_ewan:
        if ewan_archive_supplied:
            return "private_upload", False
        return "metadata_only", True
    if is_gpt2:
        if gpt2_archive_supplied:
            return "private_upload", False
        return "metadata_only", True
    return "metadata_only", False


def project_metadata_manifest(
    raw_df: pd.DataFrame,
    ewan_archive_supplied: bool,
    gpt2_archive_supplied: bool,
    ewan_archive_path: Path | None = None,
    gpt2_archive_path: Path | None = None,
    ewan_archive_sha256: str = "",
    gpt2_archive_sha256: str = "",
) -> tuple[pd.DataFrame, int]:
    """Project input DataFrame into canonical transparency schema deterministically.

    Returns:
        (projected_df: pd.DataFrame, missing_private_payload_count: int)
    """
    assert_forbidden_data_absent(raw_df)
    clean_df = strip_and_assert_no_image_bytes(raw_df)

    rows: list[dict[str, Any]] = []
    missing_private_count = 0

    for _, raw_row in clean_df.iterrows():
        row_dict = dict(raw_row)
        def _str_val(val: Any) -> str:
            if val is None:
                return ""
            value = str(val).strip()
            return "" if value.lower() in {"nan", "none"} else value

        dataset = str(row_dict.get("dataset", "")).strip()
        prov = str(row_dict.get("provenance", "")).strip()
        gen_family = str(row_dict.get("generator_family", "")).strip()
        gen = str(row_dict.get("generator", "")).strip()

        # Derive domain and cohort if missing
        existing_domain = str(row_dict.get("domain", "")).strip()
        existing_cohort = str(row_dict.get("cohort", "")).strip()
        if not existing_domain or not existing_cohort:
            derived_dom, derived_coh = derive_domain_cohort(dataset, prov, gen_family or gen)
            domain = existing_domain or derived_dom
            cohort = existing_cohort or derived_coh
        else:
            domain = existing_domain
            cohort = existing_cohort

        # Policy classification
        temp_mapping = dict(row_dict)
        temp_mapping["domain"] = domain
        temp_mapping["cohort"] = cohort
        policy, is_missing = classify_payload_policy(
            temp_mapping,
            ewan_archive_supplied=ewan_archive_supplied,
            gpt2_archive_supplied=gpt2_archive_supplied,
        )
        if is_missing:
            missing_private_count += 1

        # Integer label and ai_positive
        raw_label = row_dict.get("label")
        try:
            label_val = int(raw_label)
        except (ValueError, TypeError):
            if prov == "authentic":
                label_val = 0
            elif prov == "tampered":
                label_val = 1
            else:
                label_val = 2

        raw_ai_pos = row_dict.get("ai_positive")
        try:
            ai_pos_val = int(raw_ai_pos)
        except (ValueError, TypeError):
            ai_pos_val = 1 if label_val in {1, 2} else 0

        # Quality score
        raw_quality = row_dict.get("quality_score")
        try:
            quality_val = round(float(raw_quality), 4)
        except (ValueError, TypeError):
            quality_val = 1.0

        # Width and height
        try:
            w_val = int(row_dict.get("width", 0))
        except (ValueError, TypeError):
            w_val = 0
        try:
            h_val = int(row_dict.get("height", 0))
        except (ValueError, TypeError):
            h_val = 0

        # Stable created_at (NEVER current timestamp)
        created_at_val = str(row_dict.get("created_at", "")).strip()
        if created_at_val in {"nan", "None"}:
            created_at_val = ""

        # Archive / member provenance enrichment for private rows
        raw_arch_path = str(row_dict.get("source_archive_path", "")).strip()
        source_archive_path = (
            Path(raw_arch_path).name
            if raw_arch_path and raw_arch_path.lower() not in {"nan", "none"}
            else ""
        )
        source_archive_sha = str(row_dict.get("source_archive_sha256", "")).strip()
        source_member_path = str(row_dict.get("source_member_path", "")).strip().replace("\\", "/")
        source_member_sha = str(row_dict.get("source_member_sha256", "")).strip()

        if dataset == "ewan_gpt_images" and ewan_archive_supplied and ewan_archive_path:
            if not source_archive_path:
                source_archive_path = ewan_archive_path.name
            if not source_archive_sha:
                source_archive_sha = ewan_archive_sha256
            if not source_member_path:
                img_p = str(row_dict.get("image_path", "")).replace("\\", "/").strip()
                if img_p.startswith("ewan_gpt_images/"):
                    source_member_path = img_p[len("ewan_gpt_images/"):]
                else:
                    source_member_path = img_p

        elif dataset == "gpt_image_2_user_screenshots" and gpt2_archive_supplied and gpt2_archive_path:
            if not source_archive_path:
                source_archive_path = gpt2_archive_path.name
            if not source_archive_sha:
                source_archive_sha = gpt2_archive_sha256
            if not source_member_path:
                source_member_path = str(row_dict.get("external_id", "")).strip().replace("\\", "/")

        # Source index extraction preserving numeric 0.
        raw_src_idx = row_dict.get("source_index")
        if raw_src_idx is None:
            raw_src_idx = row_dict.get("_source_index")
        if raw_src_idx is not None and _str_val(raw_src_idx):
            source_idx_val = _str_val(raw_src_idx)
        else:
            source_idx_val = ""
            for candidate in (
                row_dict.get("external_id"),
                row_dict.get("source_member_path"),
                row_dict.get("image_path"),
            ):
                candidate_value = _str_val(candidate)
                match = re.search(
                    r"(?:^|[/_.-])(?:image[_-])?(\d+)(?:[^0-9]|$)",
                    candidate_value.lower(),
                )
                if match:
                    source_idx_val = str(int(match.group(1)))
                    break
            if not source_idx_val:
                source_idx_val = _str_val(row_dict.get("external_id"))

        image_path_val = _str_val(row_dict.get("image_path")).replace("\\", "/")
        if not is_safe_relative_path(image_path_val):
            image_name = image_path_val.rsplit("/", 1)[-1]
            image_path_val = f"{dataset}/{image_name}" if image_name else ""
        if source_member_path and not is_safe_relative_path(source_member_path):
            source_member_path = source_member_path.rsplit("/", 1)[-1]

        projected: dict[str, Any] = {
            "dataset": dataset,
            "source_url": _str_val(row_dict.get("source_url")),
            "external_id": _str_val(row_dict.get("external_id")),
            "source_index": source_idx_val,
            "source_image_group": _str_val(row_dict.get("source_image_group")),
            "duplicate_group": _str_val(row_dict.get("duplicate_group")),
            "split": _str_val(row_dict.get("split") or row_dict.get("official_split") or "train"),
            "official_split": _str_val(row_dict.get("official_split") or row_dict.get("split") or "train"),
            "domain": domain,
            "cohort": cohort,
            "label": label_val,
            "ai_positive": ai_pos_val,
            "generator": _str_val(row_dict.get("generator")),
            "generator_family": gen_family,
            "generator_version": _str_val(row_dict.get("generator_version")),
            "manipulation_family": _str_val(row_dict.get("manipulation_family")),
            "provenance": prov,
            "width": w_val,
            "height": h_val,
            "file_format": _str_val(row_dict.get("file_format")),
            "sha256": _str_val(row_dict.get("sha256")),
            "perceptual_hash": _str_val(row_dict.get("perceptual_hash")),
            "quality_score": quality_val,
            "provenance_confidence": _str_val(row_dict.get("provenance_confidence") or "high"),
            "origin_license": _str_val(row_dict.get("origin_license")),
            "license_url": _str_val(row_dict.get("license_url")),
            "attribution": _str_val(row_dict.get("attribution")),
            "redistribution_mode": "embed_bytes" if policy == "private_upload" else "reference_only",
            "selection_reason": _str_val(row_dict.get("selection_reason")),
            "image_path": image_path_val,
            "image_payload_policy": policy,
            "forbidden_demo_checked": _parse_bool(row_dict.get("forbidden_demo_checked"), default=True),
            "prompt": _str_val(row_dict.get("prompt")),
            "tamper_mask_path": _str_val(row_dict.get("tamper_mask_path")),
            "source_archive_path": source_archive_path,
            "source_archive_sha256": source_archive_sha,
            "source_member_path": source_member_path,
            "source_member_sha256": source_member_sha,
            "created_at": created_at_val,
        }

        # Deterministic stable row_id
        projected["row_id"] = compute_stable_row_id(projected)
        rows.append(projected)

    projected_df = pd.DataFrame(rows)

    # Sort deterministically
    sort_cols = [c for c in ["split", "dataset", "image_path", "row_id"] if c in projected_df.columns]
    projected_df = projected_df.sort_values(by=sort_cols).reset_index(drop=True)

    # Reorder columns to match canonical schema exactly
    ordered_cols = [c for c in CANONICAL_TRANSPARENCY_COLUMNS if c in projected_df.columns]
    remaining_cols = [c for c in projected_df.columns if c not in CANONICAL_TRANSPARENCY_COLUMNS]
    final_df = projected_df[ordered_cols + remaining_cols]

    return final_df, missing_private_count


# ==============================================================================
# Private Archive Staging Engine
# ==============================================================================

def deterministic_jpeg_bytes(image: Image.Image) -> bytes:
    """Encode an RGB image to canonical quality-95 JPEG bytes without EXIF."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=95, optimize=False, progressive=False, exif=b"")
    return buf.getvalue()


def resolve_safe_staging_target(base_dir: Path, rel_path: Path | str) -> Path:
    """Validate that rel_path is safe and stays within base_dir."""
    norm_str = str(rel_path).replace("\\", "/").strip()
    if not is_safe_relative_path(norm_str):
        raise ValueError(f"Unsafe relative path for staging: '{rel_path}'")
    target_full = (base_dir / norm_str).resolve()
    if not target_full.is_relative_to(base_dir.resolve()):
        raise ValueError(f"Target path '{rel_path}' escapes staging directory '{base_dir}'")
    return target_full


def stage_file_safely(
    data_bytes: bytes,
    dest_path: Path,
    expected_sha256: str | None = None,
) -> tuple[bool, str]:
    """Write bytes to dest_path atomically without clobbering mismatched existing files.

    Returns:
        (written: bool, sha256_hex: str)
    Raises:
        FileExistsError: If dest_path exists with different sha256.
        ValueError: If expected_sha256 does not match data_bytes.
    """
    actual_sha = hashlib.sha256(data_bytes).hexdigest()
    if expected_sha256 and actual_sha.lower() != expected_sha256.lower():
        raise ValueError(
            f"Content hash mismatch for '{dest_path.name}': computed {actual_sha}, expected {expected_sha256}"
        )

    if dest_path.is_file():
        existing_sha = file_sha256(dest_path)
        if existing_sha.lower() == actual_sha.lower():
            return False, actual_sha
        raise FileExistsError(
            f"Refusing to overwrite existing file '{dest_path}' with mismatched contents. "
            f"Existing SHA-256: {existing_sha}, New SHA-256: {actual_sha}"
        )

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.parent / f".tmp_{dest_path.name}_{actual_sha[:8]}"
    try:
        temp_path.write_bytes(data_bytes)
        temp_path.replace(dest_path)
    except Exception:
        if temp_path.is_file():
            temp_path.unlink(missing_ok=True)
        raise
    return True, actual_sha

def stage_private_images_from_archives(
    staged_images_dir: Path,
    ewan_zip_path: Path | None,
    gpt2_zip_path: Path | None,
    transparency_df: pd.DataFrame | None = None,
    allow_partial_archive: bool = False,
) -> list[dict[str, Any]]:
    """Stage private images from ZIP archives into caller-supplied directory.

    Requires private gate (enforced by caller).
    Rejects any forbidden files or directory traversals.
    Preserves member SHA-256 and archive metadata.
    """
    staged_images_dir.mkdir(parents=True, exist_ok=True)
    staged_records: list[dict[str, Any]] = []

    # Map manifest rows by member name or image_path for format matching
    manifest_by_member: dict[str, dict[str, Any]] = {}
    manifest_by_image_path: dict[str, dict[str, Any]] = {}
    if transparency_df is not None:
        for _, r in transparency_df.iterrows():
            m_path = str(r.get("source_member_path", "")).strip()
            if m_path:
                manifest_by_member[m_path] = dict(r)
            img_p = str(r.get("image_path", "")).strip().replace("\\", "/")
            if img_p:
                manifest_by_image_path[img_p] = dict(r)

    # 1. Ewan Archive Staging
    if ewan_zip_path and ewan_zip_path.is_file():
        archive_sha = file_sha256(ewan_zip_path)
        logger.info("Staging Ewan private images from '%s' (sha256=%s)...", ewan_zip_path.name, archive_sha[:16])

        with zipfile.ZipFile(ewan_zip_path, "r") as zf:
            members = sorted([
                n for n in zf.namelist()
                if not n.endswith("/")
                and not n.startswith("__MACOSX")
                and not Path(n).name.startswith(".")
                and Path(n).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
            ])
            if len(members) != len(set(members)):
                raise ValueError(f"Duplicate entry detected in Ewan archive: {ewan_zip_path.name}")
            if not allow_partial_archive and len(members) != 102:
                raise ValueError(
                    f"Authoritative Ewan archive '{ewan_zip_path.name}' must contain exactly 102 image members; "
                    f"found {len(members)}. (Pass --allow-partial-archive to override in tests)."
                )

            for member in members:
                if not is_safe_relative_path(member):
                    raise ValueError(f"Unsafe path in Ewan archive: {member}")
                if is_forbidden_path_or_source(member):
                    raise ValueError(f"Forbidden source path in Ewan archive: {member}")

                info = zf.getinfo(member)
                if ((info.external_attr >> 16) & 0o170000) == 0o120000:
                    raise ValueError(f"Symlink rejected in Ewan archive: {member}")

                data = zf.read(member)
                member_sha = hashlib.sha256(data).hexdigest()

                # Target path in staged directory: ewan_gpt_images/AI_Images/image_001.png
                target_rel = Path("ewan_gpt_images") / member
                target_full = resolve_safe_staging_target(staged_images_dir, target_rel)

                written, final_sha = stage_file_safely(data, target_full, expected_sha256=member_sha)

                staged_records.append({
                    "path": target_rel.as_posix(),
                    "sha256": final_sha,
                    "size_bytes": len(data),
                    "type": "private_image",
                    "dataset": "ewan_gpt_images",
                    "source_archive": ewan_zip_path.name,
                    "source_archive_sha256": archive_sha,
                    "source_member": member,
                    "source_member_sha256": member_sha,
                    "written_new": written,
                })

    # 2. GPT Image 2 Archive Staging
    if gpt2_zip_path and gpt2_zip_path.is_file():
        archive_sha = file_sha256(gpt2_zip_path)
        logger.info("Staging GPT Image 2 private images from '%s' (sha256=%s)...", gpt2_zip_path.name, archive_sha[:16])

        with zipfile.ZipFile(gpt2_zip_path, "r") as zf:
            members = sorted([
                n for n in zf.namelist()
                if not n.endswith("/")
                and not n.startswith("__MACOSX")
                and not Path(n).name.startswith(".")
                and Path(n).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
            ])
            if len(members) != len(set(members)):
                raise ValueError(f"Duplicate entry detected in GPT Image 2 archive: {gpt2_zip_path.name}")
            if not allow_partial_archive and len(members) != 92:
                raise ValueError(
                    f"Authoritative GPT Image 2 archive '{gpt2_zip_path.name}' must contain exactly 92 image members; "
                    f"found {len(members)}. (Pass --allow-partial-archive to override in tests)."
                )

            for idx, member in enumerate(members):
                if not is_safe_relative_path(member):
                    raise ValueError(f"Unsafe path in GPT Image 2 archive: {member}")
                if is_forbidden_path_or_source(member):
                    raise ValueError(f"Forbidden source path in GPT Image 2 archive: {member}")

                info = zf.getinfo(member)
                if ((info.external_attr >> 16) & 0o170000) == 0o120000:
                    raise ValueError(f"Symlink rejected in GPT Image 2 archive: {member}")

                data = zf.read(member)
                member_sha = hashlib.sha256(data).hexdigest()

                # Check if there is a matching manifest row
                matching_row = manifest_by_member.get(member)
                if not matching_row:
                    dest_jpg_name = f"gpt_image_2_user_screenshots/gpt_image_2_screenshot_{idx + 1:03d}.jpg"
                    matching_row = manifest_by_image_path.get(dest_jpg_name)

                # Keep staged paths relative to this package. Builder manifests may contain
                # absolute runtime paths, which are sanitized during projection.
                if matching_row and str(matching_row.get("image_path", "")).lower().endswith((".jpg", ".jpeg")):
                    with Image.open(io.BytesIO(data)) as img:
                        rgb_img = img.convert("RGB")
                        jpeg_bytes = deterministic_jpeg_bytes(rgb_img)
                    expected_sha = matching_row.get("sha256") or None
                    candidate_rel = str(matching_row.get("image_path", "")).replace("\\", "/")
                    if not (
                        is_safe_relative_path(candidate_rel)
                        and candidate_rel.startswith("gpt_image_2_user_screenshots/")
                    ):
                        candidate_rel = (
                            f"gpt_image_2_user_screenshots/gpt_image_2_screenshot_{idx + 1:03d}.jpg"
                        )
                    target_rel = Path(candidate_rel)
                    target_full = resolve_safe_staging_target(staged_images_dir, target_rel)
                    written, final_sha = stage_file_safely(jpeg_bytes, target_full, expected_sha256=expected_sha)

                    staged_records.append({
                        "path": target_rel.as_posix(),
                        "sha256": final_sha,
                        "size_bytes": len(jpeg_bytes),
                        "type": "private_image",
                        "dataset": "gpt_image_2_user_screenshots",
                        "source_archive": gpt2_zip_path.name,
                        "source_archive_sha256": archive_sha,
                        "source_member": member,
                        "source_member_sha256": member_sha,
                        "written_new": written,
                    })
                else:
                    target_rel = Path("gpt_image_2_user_screenshots") / member
                    target_full = resolve_safe_staging_target(staged_images_dir, target_rel)
                    written, final_sha = stage_file_safely(data, target_full, expected_sha256=member_sha)

                    staged_records.append({
                        "path": target_rel.as_posix(),
                        "sha256": final_sha,
                        "size_bytes": len(data),
                        "type": "private_image",
                        "dataset": "gpt_image_2_user_screenshots",
                        "source_archive": gpt2_zip_path.name,
                        "source_archive_sha256": archive_sha,
                        "source_member": member,
                        "source_member_sha256": member_sha,
                        "written_new": written,
                    })
    logger.info("Staged %d private image files into '%s'.", len(staged_records), staged_images_dir)
    return staged_records


# ==============================================================================
# Manifest Ingestion
# ==============================================================================

def ingest_manifest_sources(manifest_targets: Sequence[Path]) -> pd.DataFrame:
    """Ingest one or more manifest files or directories containing manifests.

    Supports .parquet, .jsonl, .csv.
    Combines them into a single DataFrame deterministically.
    """
    files_to_read: list[Path] = []

    for target in manifest_targets:
        if not target.exists():
            raise FileNotFoundError(f"Manifest path not found: {target}")

        if target.is_dir():
            # Look for parquet files first
            parquets = sorted(target.glob("*.parquet"))
            if parquets:
                files_to_read.extend(parquets)
            else:
                jsonls = sorted(target.glob("*.jsonl"))
                if jsonls:
                    files_to_read.extend(jsonls)
                else:
                    csvs = sorted(target.glob("*.csv"))
                    if csvs:
                        files_to_read.extend(csvs)
                    else:
                        raise ValueError(f"No .parquet, .jsonl, or .csv files found in directory '{target}'")
        elif target.is_file():
            files_to_read.append(target)

    if not files_to_read:
        raise ValueError("No manifest files specified to ingest.")

    logger.info("Ingesting %d manifest file(s): %s", len(files_to_read), [f.name for f in files_to_read])

    dfs: list[pd.DataFrame] = []
    for path in files_to_read:
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            df = pd.read_parquet(path)
        elif suffix in {".jsonl", ".ndjson"}:
            df = pd.read_json(path, lines=True)
        elif suffix == ".csv":
            df = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported manifest format: {path.name}. Expected .parquet, .jsonl, or .csv")

        # If split is missing, try inferring from filename
        if "split" not in df.columns:
            stem = path.stem.lower()
            if stem in {"train", "validation", "val", "test", "test_unseen"}:
                df["split"] = "validation" if stem == "val" else stem

        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    logger.info("Loaded combined manifest with %d raw rows.", len(combined_df))
    return combined_df


# ==============================================================================
# HF Hub Upload Engine
# ==============================================================================

def execute_hf_upload(
    repo_id: str,
    output_dir: Path,
    staged_images_dir: Path | None,
    staged_records: list[dict[str, Any]] | None,
    is_private: bool,
    hf_token: str | None = None,
) -> None:
    """Safely upload transparency files and staged private images to Hugging Face Hub.

    Strictly requires is_private=True and valid credentials.
    """
    if not repo_id:
        raise ValueError("HF upload requested, but repo_id is empty.")
    if not is_private:
        raise ValueError(
            "CRITICAL: Refusing to upload transparency package without private=True. "
            "Private team assets must never be uploaded to public repositories."
        )

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for upload. Please install it or use project virtualenv."
        ) from exc

    api = HfApi(token=hf_token)

    # 1. Verify authentication
    try:
        user_info = api.whoami()
        username = user_info.get("name") or user_info.get("fullname") or "authenticated_user"
        logger.info("Authenticated to Hugging Face as user '%s'.", username)
    except Exception as exc:
        raise RuntimeError(
            f"Hugging Face authentication check failed: {exc}. "
            "Please log in via 'huggingface-cli login' or supply a valid HF token."
        ) from exc

    # 2. Check / Create repository with private=True
    try:
        repo_info = api.repo_info(repo_id=repo_id, repo_type="dataset")
        if not repo_info.private:
            raise RuntimeError(
                f"Repository '{repo_id}' exists on Hugging Face but is PUBLIC! "
                "Aborting upload to prevent exposure of private team assets."
            )
        logger.info("Confirmed destination repository '%s' is private.", repo_id)
    except Exception as exc:
        if "404" in str(exc) or "RepositoryNotFoundError" in type(exc).__name__:
            logger.info("Repository '%s' does not exist. Creating private dataset repository...", repo_id)
            api.create_repo(repo_id=repo_id, repo_type="dataset", private=True)
            logger.info("Created private dataset repository '%s'.", repo_id)
        else:
            raise

    # 3. Upload metadata files
    for filename in ("transparency.parquet", "transparency.jsonl", "upload_manifest.json"):
        file_path = output_dir / filename
        if file_path.is_file():
            logger.info("Uploading %s to %s...", filename, repo_id)
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Upload {filename}",
            )

    # 4. Upload exact verified staged private images
    if staged_images_dir and staged_records:
        logger.info(
            "Uploading %d verified staged private image(s) to %s/private_images...",
            len(staged_records),
            repo_id,
        )
        for rec in staged_records:
            rel_path = rec["path"]
            file_to_upload = resolve_safe_staging_target(staged_images_dir, rel_path)
            if file_to_upload.is_file():
                api.upload_file(
                    path_or_fileobj=file_to_upload,
                    path_in_repo=f"private_images/{rel_path}",
                    repo_id=repo_id,
                    repo_type="dataset",
                    commit_message=f"Upload private image {Path(rel_path).name}",
                )
    logger.info("Successfully uploaded transparency package to '%s' (private).", repo_id)


# ==============================================================================
# Main Transparency Packaging Pipeline
# ==============================================================================

def prepare_hf_transparency_package(
    manifest_paths: Sequence[Path],
    output_dir: Path,
    gpt2_zip: Path | None = None,
    ewan_zip: Path | None = None,
    include_private_images: bool = False,
    staged_images_dir: Path | None = None,
    is_private: bool = False,
    upload: bool = False,
    repo_id: str | None = None,
    hf_token: str | None = None,
    allow_partial_archive: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Build the metadata-first HF transparency package, optional staging, and upload manifest."""
    # Gate 1: Check private image staging privacy requirements
    if include_private_images and not is_private:
        raise ValueError(
            "CRITICAL: --include-private-images stages private team images and requires an explicit privacy gate. "
            "Pass --private or --private-repo to confirm the target destination is private. "
            "Aborting to prevent private asset exposure."
        )

    # Gate 2: Check upload requirements
    if upload:
        if not repo_id:
            raise ValueError("Upload requested (--upload), but --repo-id was not provided.")
        if not is_private:
            raise ValueError(
                "Upload requested (--upload), but target repository is not marked private. "
                "Specify --private to confirm."
            )

    # Verify archives
    ewan_supplied = False
    ewan_archive_sha = ""
    if ewan_zip:
        if not ewan_zip.is_file():
            raise FileNotFoundError(f"Specified Ewan archive not found: {ewan_zip}")
        ewan_supplied = True
        ewan_archive_sha = file_sha256(ewan_zip)

    gpt2_supplied = False
    gpt2_archive_sha = ""
    if gpt2_zip:
        if not gpt2_zip.is_file():
            raise FileNotFoundError(f"Specified GPT Image 2 archive not found: {gpt2_zip}")
        gpt2_supplied = True
        gpt2_archive_sha = file_sha256(gpt2_zip)

    # Ingest manifests
    raw_df = ingest_manifest_sources(manifest_paths)

    # Project metadata deterministically
    transparency_df, missing_private_count = project_metadata_manifest(
        raw_df=raw_df,
        ewan_archive_supplied=ewan_supplied,
        gpt2_archive_supplied=gpt2_supplied,
        ewan_archive_path=ewan_zip,
        gpt2_archive_path=gpt2_zip,
        ewan_archive_sha256=ewan_archive_sha,
        gpt2_archive_sha256=gpt2_archive_sha,
    )

    total_rows = len(transparency_df)
    policy_counts = dict(transparency_df["image_payload_policy"].value_counts())
    metadata_only_count = int(policy_counts.get("metadata_only", 0))
    private_upload_count = int(policy_counts.get("private_upload", 0))

    logger.info(
        "Projected %d transparency rows: %d metadata_only, %d private_upload (missing private payloads: %d).",
        total_rows,
        metadata_only_count,
        private_upload_count,
        missing_private_count,
    )

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Write transparency.parquet atomically
    parquet_path = output_dir / "transparency.parquet"
    if not dry_run:
        temp_parquet = output_dir / f".tmp_{parquet_path.name}"
        transparency_df.to_parquet(temp_parquet, index=False, engine="pyarrow")
        temp_parquet.replace(parquet_path)
        logger.info("Saved %s (%d rows)", parquet_path.name, total_rows)

    # Write transparency.jsonl atomically
    jsonl_path = output_dir / "transparency.jsonl"
    if not dry_run:
        temp_jsonl = output_dir / f".tmp_{jsonl_path.name}"
        with open(temp_jsonl, "w", encoding="utf-8") as f:
            for record in transparency_df.to_dict(orient="records"):
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        temp_jsonl.replace(jsonl_path)
        logger.info("Saved %s (%d rows)", jsonl_path.name, total_rows)

    # Stage private images if requested
    staged_records: list[dict[str, Any]] = []
    actual_staged_dir: Path | None = None
    if include_private_images and (ewan_supplied or gpt2_supplied):
        actual_staged_dir = staged_images_dir or (output_dir / "private_images")
        if not dry_run:
            staged_records = stage_private_images_from_archives(
                staged_images_dir=actual_staged_dir,
                ewan_zip_path=ewan_zip if ewan_supplied else None,
                gpt2_zip_path=gpt2_zip if gpt2_supplied else None,
                transparency_df=transparency_df,
                allow_partial_archive=allow_partial_archive,
            )

    # Compute upload manifest
    parquet_sha = file_sha256(parquet_path) if parquet_path.is_file() else ""
    parquet_size = parquet_path.stat().st_size if parquet_path.is_file() else 0
    jsonl_sha = file_sha256(jsonl_path) if jsonl_path.is_file() else ""
    jsonl_size = jsonl_path.stat().st_size if jsonl_path.is_file() else 0

    files_list: list[dict[str, Any]] = [
        {
            "path": "transparency.parquet",
            "sha256": parquet_sha,
            "size_bytes": parquet_size,
            "type": "metadata_parquet",
            "rows": total_rows,
        },
        {
            "path": "transparency.jsonl",
            "sha256": jsonl_sha,
            "size_bytes": jsonl_size,
            "type": "metadata_jsonl",
            "rows": total_rows,
        },
    ]

    for rec in staged_records:
        files_list.append({
            "path": f"private_images/{rec['path']}",
            "sha256": rec["sha256"],
            "size_bytes": rec["size_bytes"],
            "type": rec["type"],
            "dataset": rec["dataset"],
            "source_archive": rec["source_archive"],
            "source_member": rec["source_member"],
        })

    example_cmd_parts = [
        "uv run python scripts/data_prep/prepare_hf_transparency.py",
        f'--manifest "{output_dir.as_posix()}"',
        "--repo-id <YOUR_PRIVATE_HF_REPO>",
        "--private --upload",
    ]
    if ewan_zip:
        example_cmd_parts.append(f'--ewan-zip "{ewan_zip.as_posix()}"')
    if gpt2_zip:
        example_cmd_parts.append(f'--gpt2-zip "{gpt2_zip.as_posix()}"')
    if staged_images_dir:
        example_cmd_parts.append(f'--staged-images-dir "{staged_images_dir.as_posix()}"')
    if include_private_images:
        example_cmd_parts.append("--include-private-images")

    # One line works in cmd.exe, PowerShell, and POSIX shells.
    safe_example_cmd = " ".join(example_cmd_parts)

    upload_manifest = {
        "manifest_version": "1.0.0",
        "transparency_manifest_digest": compute_stable_manifest_digest(transparency_df),
        "output_directory": output_dir.resolve().as_posix(),
        "repo_privacy_requirements": {
            "private_required": True,
            "repo_type": "dataset",
            "public_upload_forbidden": True,
            "rationale": (
                "The transparency package contains private team-generated Ewan images and "
                "authoritative user GPT Image 2 screenshots. Uploads to Hugging Face MUST "
                "be strictly private (private=True). Public dataset uploads are forbidden."
            ),
        },
        "row_counts": {
            "total": total_rows,
            "by_payload_policy": {
                "metadata_only": metadata_only_count,
                "private_upload": private_upload_count,
            },
            "missing_private_payload_count": missing_private_count,
            "by_split": {str(k): int(v) for k, v in transparency_df["split"].value_counts().items()},
            "by_dataset": {str(k): int(v) for k, v in transparency_df["dataset"].value_counts().items()},
            "by_provenance": {str(k): int(v) for k, v in transparency_df["provenance"].value_counts().items()},
        },
        "files": files_list,
        "safe_example_command": safe_example_cmd,
    }

    manifest_path = output_dir / "upload_manifest.json"
    if not dry_run:
        temp_manifest = output_dir / f".tmp_{manifest_path.name}"
        with open(temp_manifest, "w", encoding="utf-8") as f:
            json.dump(upload_manifest, f, indent=2, ensure_ascii=False)
        temp_manifest.replace(manifest_path)
        logger.info("Saved %s", manifest_path.name)

    # Upload if requested (Never run in agent automated runs)
    if upload and not dry_run:
        execute_hf_upload(
            repo_id=str(repo_id),
            output_dir=output_dir,
            staged_images_dir=actual_staged_dir,
            staged_records=staged_records if include_private_images else None,
            is_private=is_private,
            hf_token=hf_token,
        )

    return upload_manifest


# CLI Entry Point
# ==============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a metadata-first Hugging Face transparency package for the teacher dataset. "
            "Public/source datasets contribute only metadata rows; private team Ewan images and "
            "user GPT Image 2 screenshots are optionally staged under strict private repository gates."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        nargs="+",
        type=Path,
        required=True,
        help="One or more manifest files (.parquet, .jsonl, .csv) or directories containing them.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("hf_transparency_package"),
        help="Directory where transparency.parquet, transparency.jsonl, and upload_manifest.json will be written.",
    )
    parser.add_argument(
        "--gpt2-zip",
        "--zip",
        dest="gpt2_zip",
        type=Path,
        default=None,
        help="Path to gpt-image-2-synthetic-images.zip archive.",
    )
    parser.add_argument(
        "--ewan-zip",
        type=Path,
        default=None,
        help="Path to 100+ AI_Images from ewan.zip archive.",
    )
    parser.add_argument(
        "--auto-detect-archives",
        action="store_true",
        help="Auto-detect Ewan and GPT Image 2 archives in project root or current working directory if not explicitly passed.",
    )
    parser.add_argument(
        "--include-private-images",
        action="store_true",
        help="Stage actual private image files from archives into the private images directory (requires --private).",
    )
    parser.add_argument(
        "--staged-images-dir",
        type=Path,
        default=None,
        help="Caller-supplied directory for staging private images (defaults to <output-dir>/private_images).",
    )
    parser.add_argument(
        "--private",
        "--private-repo",
        dest="private",
        action="store_true",
        help="Explicit privacy gate confirming the destination repository is private. Mandatory for --include-private-images and --upload.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload the package to Hugging Face Hub (requires --repo-id and --private; do not execute during agent work).",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Target Hugging Face dataset repository ID (e.g. username/private-transparency-dataset).",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Optional Hugging Face authentication token (falls back to HF_TOKEN env var).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform projection and validation without writing files to disk or uploading.",
    )
    parser.add_argument(
        "--allow-partial-archive",
        action="store_true",
        help="Allow archives with member counts differing from authoritative 102 (Ewan) or 92 (GPT Image 2).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    gpt2_zip = args.gpt2_zip
    ewan_zip = args.ewan_zip

    if args.auto_detect_archives:
        if gpt2_zip is None:
            for cand in [Path("gpt-image-2-synthetic-images.zip"), _PROJECT_ROOT / "gpt-image-2-synthetic-images.zip"]:
                if cand.is_file():
                    gpt2_zip = cand
                    logger.info("Auto-detected GPT Image 2 archive at: %s", cand)
                    break
        if ewan_zip is None:
            for cand in [Path("100+ AI_Images from ewan.zip"), _PROJECT_ROOT / "100+ AI_Images from ewan.zip"]:
                if cand.is_file():
                    ewan_zip = cand
                    logger.info("Auto-detected Ewan archive at: %s", cand)
                    break

    try:
        prepare_hf_transparency_package(
            manifest_paths=args.manifest,
            output_dir=args.output_dir,
            gpt2_zip=gpt2_zip,
            ewan_zip=ewan_zip,
            include_private_images=args.include_private_images,
            staged_images_dir=args.staged_images_dir,
            is_private=args.private,
            upload=args.upload,
            repo_id=args.repo_id,
            hf_token=args.hf_token,
            dry_run=args.dry_run,
            allow_partial_archive=args.allow_partial_archive,
        )
    except Exception as exc:
        logger.error("Preparation failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
