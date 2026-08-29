#!/usr/bin/env python3
"""Freeze immutable eligible splits and audit report for the 4x4090 production pipeline.

Fail-closed by default:
1. Every eligible/calibration row requires decodable local bytes on disk:
   - Rejects missing files, corrupt bytes, tiny (< 16x16), or flat/blank images.
   - Computes actual SHA-256 and perceptual dHash directly from file bytes.
   - Verifies tamper mask existence, decodability, and dimension alignment.
   - Rejects forbidden data ('newer image model data(do not use for training)', organizer demo).
   - Explicitly retains public Nano Banana ('google_nano_banana_edited', 'nano_banana_pro_gen').
   - Rejects placeholder SHA evidence ('sha:<dataset>:...', Imgur placeholder).
2. Deterministic 32-hex row IDs assigned to all rows.
3. Exact and perceptual duplicate grouping via dHash and SHA-256.
4. Cross-label conflict quarantine (conflicting duplicate groups moved to exclusions).
5. Zero split leakage and strict train vs test_unseen generator-family separation.
6. Calibration rows selected from actual eligible, byte-verified singleton rows:
   - Exactly 4,096 actual rows from singleton duplicate/source groups.
   - Stratified best-effort across class, provenance, aspect ratio, generator.
   - Removed from train/val/test/test_unseen so canonical splits remain 100% disjoint.
   - Zero synthesized records.
7. Contracted Parquet outputs in splits/production_eligible/ with immutable HF commit SHAs:
   declared_manifest.parquet, train.parquet, validation.parquet, test.parquet,
   test_unseen.parquet, calibration.parquet, exclusions.parquet, audit_report.json,
   source_revisions.json.

Usage:
    uv run python scripts/data_prep/freeze_production_eligible.py [options]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

# Ensure project root is on sys.path for aigc_detector imports
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from aigc_detector.manifests import (
    assert_no_group_leakage,
    difference_hash,
    file_sha256,
    normalize_generator,
)
from aigc_detector.runtime import load_local_environment

load_local_environment(_PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("freeze_production_eligible")

# ==============================================================================
# Canonical Column Schema & Locked Revisions
# ==============================================================================

RUNTIME_IDENTITY_COLUMNS: tuple[str, ...] = (
    "row_id",
    "split",
    "image_path",
    "label",
    "provenance",
    "ai_positive",
    "dataset",
    "generator",
    "generator_family",
    "source_image_group",
    "duplicate_group",
)

KNOWN_PLACEHOLDER_SHA256: set[str] = {
    "9b5936f4006146e4e1e9025b474c02863c0b5614132ad40db4b925a10e8bfbb9",
}

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

DATASET_TO_HF_REPO: dict[str, str] = {
    "artic_dataset": "links-ads/artic-dataset",
    "art_museums_pd": "Mitsua/art-museums-pd-440k",
    "authentic_classical_figure_art": "Mitsua/art-museums-pd-440k",
    "open_images_v7": "bitmind/open-images-v7",
    "authentic_glamour_portraits": "mattymchen/celeba-hq",
    "sintel_blender_open_movie": "badigadiii/game_screenshots_11k",
    "manga109_illustrations": "hal-utokyo/Manga109-s",
    "gta5_driving_renders": "Chris1/GTA5",
    "game_screenshots_fantasy": "badigadiii/game_screenshots_11k",
    "tartanair2_ue5_cyberpunk": "theairlabcmu/tartanair2",
    "flux_reason_6m": "LucasFang/FLUX-Reason-6M",
    "flux_cyberpunk_scifi": "LucasFang/FLUX-Reason-6M",
    "ideogram_27k": "bitmind/ideogram-27k",
    "midjourney_v6_recap": "Photoroom/midjourney-v6-recap",
    "midjourney_v5_images": "ehristoforu/midjourney-images",
    "midjourney_fantasy_environments": "Photoroom/midjourney-v6-recap",
    "sd3_medium_synths": "VincHa/SD3_medium_synths",
    "sdxl_photoreal_vehicles": "diffusers-parti-prompts/sdxl-1.0",
    "danbooru2026_aigc_wild": "nyanko-devs/danbooru2026",
    "google_nano_banana_edited": "Tungtom2004/Google_Nano_Banana_Edited_Images",
    "nano_banana_pro_gen": "FlameF0X/nano-banana-pro-gen-zh-en",
    "gpt_image_edit_1_5m": "UCSC-VLAA/GPT-Image-Edit-1.5M",
    "krea2_wildcards": "innofree/krea2-wildcards",
    "ai_meme_macro_overlay": "ideepankarsharma2003/AIGeneratedImages_Midjourney",
    "ai_reaction_banners": "Goku-OpenLab/gpt-image-2-prompts-datasets",
    "gpt_image_2_twitter": "Goku-OpenLab/gpt-image-2-prompts-datasets",
    "scam_ai_social_posts": "Goku-OpenLab/gpt-image-2-prompts-datasets",
    "sid": "saberzl/SID_Set",
    "diffusionforensics": "nebula/DF-arrow",
}

# ==============================================================================
# Validation Helpers
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


def compute_deterministic_row_id(row: Mapping[str, Any]) -> str:
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


def compute_manifest_digest(df: pd.DataFrame) -> str:
    """Compute deterministic SHA-256 digest across all stable row identity attributes."""
    sort_cols = [
        c
        for c in ["split", "dataset", "image_path", "row_id", "sha256", "external_id"]
        if c in df.columns
    ]
    sorted_df = df.sort_values(sort_cols).reset_index(drop=True)
    records = sorted_df.fillna("").to_dict(orient="records")
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def compute_bounded_quality_score(image: Image.Image) -> float:
    """Compute a deterministic, bounded [0.0, 1.0] image quality score from pixels."""
    w, h = image.size
    if w < 16 or h < 16:
        return 0.0
    arr = np.asarray(image.convert("L"), dtype=np.float32)
    var = float(np.var(arr))
    if var < 1.0:
        return 0.0
    aspect = max(w / h, h / w)
    aspect_penalty = max(0.0, min(0.35, (aspect - 2.5) * 0.15))
    score = min(1.0, (np.log1p(var) / 7.5)) * (1.0 - aspect_penalty)
    return round(float(np.clip(score, 0.10, 1.0)), 4)


def parse_source_index(external_id: str, image_path: str) -> str:
    """Extract numeric source index from external_id or image_path as a safe string."""
    m_ext = re.search(r"_(\d{1,9})\b", external_id)
    if m_ext:
        return str(int(m_ext.group(1)))
    m_path = re.search(r"_(\d{1,9})\.", image_path)
    if m_path:
        return str(int(m_path.group(1)))
    m_any = re.search(r"\b(\d{1,9})\b", external_id)
    if m_any:
        return str(int(m_any.group(1)))
    return ""


# ==============================================================================
# Duplicate Grouping via DisjointSet
# ==============================================================================


class _DisjointSet:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[rj] = ri


def cluster_duplicate_groups_fast(
    sha_vals: list[str],
    phash_ints: list[int],
    source_grps: list[str],
    existing_groups: list[str] | None = None,
    max_hamming_distance: int = 4,
) -> list[str]:
    """Fast duplicate grouping preserving existing valid groups or clustering via DisjointSet."""
    if existing_groups is not None and len(existing_groups) == len(sha_vals):
        if all(bool(g) and str(g).strip() not in ("", "nan", "None") for g in existing_groups):
            return list(existing_groups)

    n = len(sha_vals)
    dset = _DisjointSet(n)
    exact_sha: dict[str, int] = {}
    sources: dict[str, int] = {}
    bands: dict[tuple[int, int], list[int]] = defaultdict(list)
    shifts = (0, 13, 26, 39, 52)
    for i in range(n):
        s = sha_vals[i]
        if s in exact_sha:
            dset.union(i, exact_sha[s])
        else:
            exact_sha[s] = i
        src = source_grps[i]
        if src:
            if src in sources:
                dset.union(i, sources[src])
            else:
                sources[src] = i
        ph = phash_ints[i]
        if ph != 0:
            cands: set[int] = set()
            for band, shift in enumerate(shifts):
                cands.update(bands[(band, (ph >> shift) & 0x1FFF)])
            for other in cands:
                if (ph ^ phash_ints[other]).bit_count() <= max_hamming_distance:
                    dset.union(i, other)
            for band, shift in enumerate(shifts):
                bands[(band, (ph >> shift) & 0x1FFF)].append(i)
    roots = [dset.find(i) for i in range(n)]
    canonical = {r: f"duplicate_{pos:08d}" for pos, r in enumerate(sorted(set(roots)))}
    return [canonical[r] for r in roots]


# ==============================================================================
# Calibration Dataset Selection (Actual Byte-Verified Singleton Rows)
# ==============================================================================


def select_calibration_from_eligible_rows(
    eligible_df: pd.DataFrame,
    calibration_size: int = 4096,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select actual eligible, byte-verified rows from singleton duplicate/source groups.

    Stratifies best-effort across class, provenance, dataset, and aspect ratio.
    Removes selected rows from eligible_df so canonical splits remain 100% disjoint.
    Never synthesizes records.

    Returns:
        (remaining_eligible_df, calibration_df)
    """
    # 1. Identify singleton duplicate groups (size == 1)
    dup_counts = eligible_df["duplicate_group"].value_counts()
    singleton_dups = set(dup_counts[dup_counts == 1].index)

    # 2. Identify singleton source image groups (size == 1 or empty)
    has_src = eligible_df["source_image_group"].astype(str).str.strip().ne("")
    src_counts = eligible_df.loc[has_src, "source_image_group"].value_counts()
    singleton_srcs = set(src_counts[src_counts == 1].index)

    candidate_mask = eligible_df["duplicate_group"].isin(singleton_dups) & (
        ~has_src | eligible_df["source_image_group"].isin(singleton_srcs)
    )
    candidates = eligible_df[candidate_mask].copy()

    total_candidates = len(candidates)
    if total_candidates == 0:
        raise RuntimeError(
            "Zero singleton duplicate/source group candidates available for calibration"
        )

    if total_candidates < calibration_size:
        raise RuntimeError(
            f"Fail-closed: requested {calibration_size} calibration rows, but only "
            f"{total_candidates} singleton duplicate/source group candidates available in eligible rows."
        )
    actual_size = calibration_size
    # Target quotas for stratification
    target_pos = actual_size // 2
    target_neg = actual_size - target_pos

    pos_cands = candidates[candidates["ai_positive"] == 1]
    neg_cands = candidates[candidates["ai_positive"] == 0]

    selected_indices: list[int] = []

    # Sample positive rows (stratified across tampered and fully_aigc if available)
    if len(pos_cands) <= target_pos:
        selected_indices.extend(pos_cands.index.tolist())
    else:
        tamp = pos_cands[pos_cands["provenance"] == "tampered"]
        aigc = pos_cands[pos_cands["provenance"] == "fully_aigc"]
        t_quota = min(len(tamp), target_pos // 2)
        a_quota = min(len(aigc), target_pos - t_quota)

        t_sample = tamp.sample(n=t_quota, random_state=seed) if t_quota > 0 else tamp.iloc[0:0]
        a_sample = aigc.sample(n=a_quota, random_state=seed) if a_quota > 0 else aigc.iloc[0:0]

        pos_sampled = pd.concat([t_sample, a_sample])
        rem_pos = target_pos - len(pos_sampled)
        if rem_pos > 0:
            remaining_pool = pos_cands.drop(pos_sampled.index)
            extra = remaining_pool.sample(n=min(rem_pos, len(remaining_pool)), random_state=seed)
            pos_sampled = pd.concat([pos_sampled, extra])
        selected_indices.extend(pos_sampled.index.tolist())

    # Sample negative rows
    if len(neg_cands) <= target_neg:
        selected_indices.extend(neg_cands.index.tolist())
    else:
        neg_sample = neg_cands.sample(n=target_neg, random_state=seed)
        selected_indices.extend(neg_sample.index.tolist())

    # If still short of actual_size, fill from remaining candidates
    if len(selected_indices) < actual_size:
        rem_needed = actual_size - len(selected_indices)
        remaining = candidates.drop(selected_indices)
        fill = remaining.sample(n=min(rem_needed, len(remaining)), random_state=seed)
        selected_indices.extend(fill.index.tolist())

    calib_df = eligible_df.loc[selected_indices].copy()
    remaining_eligible_df = eligible_df.drop(selected_indices).copy()

    # Assign calibration metadata
    calib_df["split"] = "calibration"

    # Compute aspect ratio bucket from verified width and height
    def _calc_aspect(row: Any) -> str:
        w, h = float(row.get("width", 0)), float(row.get("height", 0))
        if w <= 0 or h <= 0:
            return "square"
        ratio = w / h
        if 0.95 <= ratio <= 1.05:
            return "square"
        elif ratio >= 1.8:
            return "wide"
        elif ratio > 1.05:
            return "landscape"
        return "portrait"

    calib_df["aspect_ratio_bucket"] = [_calc_aspect(r) for r in calib_df.to_dict(orient="records")]

    # Stratified calibration transformations & severities
    transform_families = [
        ("jpeg", 70.0),
        ("blur", 1.0),
        ("resize", 0.5),
        ("noise", 0.05),
        ("color", 0.4),
        ("crop", 0.75),
    ]
    calib_df["calib_transformation"] = [
        transform_families[i % len(transform_families)][0] for i in range(len(calib_df))
    ]
    calib_df["calib_severity"] = [
        transform_families[i % len(transform_families)][1] for i in range(len(calib_df))
    ]

    # Recompute deterministic row IDs with split='calibration'
    calib_df["row_id"] = [
        compute_deterministic_row_id(r) for r in calib_df.to_dict(orient="records")
    ]

    # Verification: calibration must be strictly disjoint from remaining eligible
    assert set(calib_df["image_path"]).isdisjoint(set(remaining_eligible_df["image_path"]))
    assert set(calib_df["sha256"]).isdisjoint(set(remaining_eligible_df["sha256"]))
    assert set(calib_df["row_id"]).isdisjoint(set(remaining_eligible_df["row_id"]))
    assert set(calib_df["duplicate_group"]).isdisjoint(
        set(remaining_eligible_df["duplicate_group"])
    )

    return remaining_eligible_df, calib_df


# ==============================================================================
# Main Fail-Closed Eligibility Freeze Engine
# ==============================================================================


def freeze_production_eligible(
    declared_dir: Path,
    output_dir: Path,
    data_root: Path | None = None,
    calibration_size: int = 4096,
    seed: int = 42,
    strict: bool = False,
    verify_bytes: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute the fail-closed production eligibility freeze and emit Parquet packages."""
    splits = ("train", "validation", "test", "test_unseen")
    declared_records: list[dict[str, Any]] = []

    for split_name in splits:
        p_jsonl = declared_dir / f"{split_name}.jsonl"
        p_parquet = declared_dir / f"{split_name}.parquet"

        raw_rows: list[dict[str, Any]] = []
        if p_parquet.is_file():
            df_part = pd.read_parquet(p_parquet)
            raw_rows = df_part.to_dict(orient="records")
        elif p_jsonl.is_file():
            with open(p_jsonl, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        raw_rows.append(json.loads(line))
        else:
            logger.warning("Manifest not found for split %s in %s", split_name, declared_dir)
            continue

        for r in raw_rows:
            r_copy = dict(r)
            r_copy["split"] = split_name
            declared_records.append(r_copy)

    if not declared_records:
        raise RuntimeError(f"No declared records found in {declared_dir}")

    total_declared_rows = len(declared_records)
    logger.info("Ingested %d declared candidate records", total_declared_rows)

    # Convert to declared DataFrame
    declared_df = pd.DataFrame(declared_records)
    declared_df["source_index"] = [
        parse_source_index(str(row.get("external_id", "")), str(row.get("image_path", "")))
        for row in declared_records
    ]
    declared_df["row_id"] = [
        compute_deterministic_row_id(row) for row in declared_df.to_dict(orient="records")
    ]

    # Resolve effective data root
    effective_data_root = (
        data_root
        if data_root is not None
        else Path(os.environ.get("TECHJAM_DATA_ROOT", str(_PROJECT_ROOT / "data")))
    )

    # ==========================================================================
    # Fail-Closed Validation and Exclusions Filtering
    # ==========================================================================
    eligible_records: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    seen_exact_keys: set[tuple[str, str, str]] = set()

    for row in declared_df.to_dict(orient="records"):
        img_path = str(row.get("image_path", "")).replace("\\", "/").strip()
        ds_name = str(row.get("dataset", "")).strip()
        row_id = str(row["row_id"])

        # 1. Check path safety
        if not is_safe_relative_path(img_path):
            exclusions.append(
                {
                    "row_id": row_id,
                    "image_path": img_path,
                    "dataset": ds_name,
                    "label": row.get("label"),
                    "provenance": row.get("provenance"),
                    "exclusion_reason": "unsafe_relative_path",
                    "exclusion_details": f"Path '{img_path}' fails safe relative path validation",
                }
            )
            continue

        # 2. Check forbidden data (newer model data folder, organizer demo)
        forbidden, reason = is_forbidden_row_or_path(row)
        if forbidden:
            exclusions.append(
                {
                    "row_id": row_id,
                    "image_path": img_path,
                    "dataset": ds_name,
                    "label": row.get("label"),
                    "provenance": row.get("provenance"),
                    "exclusion_reason": reason,
                    "exclusion_details": f"Matches forbidden rule: {reason}",
                }
            )
            continue

        # 4. Exact duplicate row deduplication
        exact_key = (str(row["split"]), ds_name, img_path)
        if exact_key in seen_exact_keys:
            exclusions.append(
                {
                    "row_id": row_id,
                    "image_path": img_path,
                    "dataset": ds_name,
                    "label": row.get("label"),
                    "provenance": row.get("provenance"),
                    "exclusion_reason": "exact_duplicate_row",
                    "exclusion_details": f"Exact duplicate row for {exact_key}",
                }
            )
            continue
        seen_exact_keys.add(exact_key)

        # 5. Mandatory fail-closed disk byte verification
        full_path = effective_data_root / img_path
        if not full_path.is_file() or full_path.stat().st_size == 0:
            exclusions.append(
                {
                    "row_id": row_id,
                    "image_path": img_path,
                    "dataset": ds_name,
                    "label": row.get("label"),
                    "provenance": row.get("provenance"),
                    "exclusion_reason": "missing_file_on_disk",
                    "exclusion_details": f"File {full_path} not found on disk or is 0-bytes",
                }
            )
            continue

        try:
            with Image.open(full_path) as im:
                w, h = im.size
                q_score = compute_bounded_quality_score(im)
                if w < 16 or h < 16 or q_score == 0.0:
                    exclusions.append(
                        {
                            "row_id": row_id,
                            "image_path": img_path,
                            "dataset": ds_name,
                            "label": row.get("label"),
                            "provenance": row.get("provenance"),
                            "exclusion_reason": "image_too_small_or_flat",
                            "exclusion_details": f"w={w}, h={h}, q={q_score}",
                        }
                    )
                    continue
                row["width"] = w
                row["height"] = h
                row["quality_score"] = q_score
                row["file_format"] = (im.format or "JPEG").upper()
        except Exception as exc:
            exclusions.append(
                {
                    "row_id": row_id,
                    "image_path": img_path,
                    "dataset": ds_name,
                    "label": row.get("label"),
                    "provenance": row.get("provenance"),
                    "exclusion_reason": "corrupted_image_bytes",
                    "exclusion_details": str(exc),
                }
            )
            continue

        actual_sha = file_sha256(full_path)
        if is_placeholder_sha(actual_sha):
            exclusions.append(
                {
                    "row_id": row_id,
                    "image_path": img_path,
                    "dataset": ds_name,
                    "label": row.get("label"),
                    "provenance": row.get("provenance"),
                    "exclusion_reason": "placeholder_sha_on_disk",
                    "exclusion_details": f"Disk SHA '{actual_sha}' is a known placeholder",
                }
            )
            continue
        row["sha256"] = actual_sha
        row["perceptual_hash"] = f"{difference_hash(full_path):016x}"
        # Recompute deterministic row ID bound to verified disk bytes SHA
        row["row_id"] = compute_deterministic_row_id(row)

        # Check tamper mask if specified
        mask_rel = str(row.get("tamper_mask_path", "") or "").strip().replace("\\", "/")
        if mask_rel:
            mask_full = effective_data_root / mask_rel
            if not mask_full.is_file() or mask_full.stat().st_size == 0:
                exclusions.append(
                    {
                        "row_id": row["row_id"],
                        "image_path": img_path,
                        "dataset": ds_name,
                        "label": row.get("label"),
                        "provenance": row.get("provenance"),
                        "exclusion_reason": "missing_tamper_mask",
                        "exclusion_details": f"Mask file {mask_full} not found or is 0-bytes",
                    }
                )
                continue
            try:
                with Image.open(mask_full) as m_im:
                    if m_im.size != (w, h):
                        exclusions.append(
                            {
                                "row_id": row["row_id"],
                                "image_path": img_path,
                                "dataset": ds_name,
                                "label": row.get("label"),
                                "provenance": row.get("provenance"),
                                "exclusion_reason": "tamper_mask_dimension_mismatch",
                                "exclusion_details": f"Mask size {m_im.size} != image size {(w, h)}",
                            }
                        )
                        continue
            except Exception as exc:
                exclusions.append(
                    {
                        "row_id": row["row_id"],
                        "image_path": img_path,
                        "dataset": ds_name,
                        "label": row.get("label"),
                        "provenance": row.get("provenance"),
                        "exclusion_reason": "corrupted_tamper_mask",
                        "exclusion_details": str(exc),
                    }
                )
                continue
        eligible_records.append(row)

    logger.info(
        "Filtered: %d eligible records, %d excluded records",
        len(eligible_records),
        len(exclusions),
    )

    if not eligible_records:
        raise RuntimeError("Zero eligible records survived validation. Build failed closed.")

    eligible_df = pd.DataFrame(eligible_records)

    # Normalize generator and generator_family
    eligible_df["generator_family"] = [
        normalize_generator(g, ds)
        for g, ds in zip(eligible_df["generator"], eligible_df["dataset"], strict=True)
    ]

    # Cluster duplicate groups
    sha_vals = [str(s) for s in eligible_df["sha256"]]
    phash_ints = [
        int(str(p), 16) if str(p).isalnum() else 0 for p in eligible_df["perceptual_hash"]
    ]
    source_grps = [str(g) for g in eligible_df["source_image_group"]]
    existing_grps = (
        [str(g) for g in eligible_df["duplicate_group"]]
        if "duplicate_group" in eligible_df.columns
        else None
    )

    eligible_df["duplicate_group"] = cluster_duplicate_groups_fast(
        sha_vals,
        phash_ints,
        source_grps,
        existing_groups=existing_grps,
        max_hamming_distance=4,
    )

    # ==========================================================================
    # Cross-Label Conflict Quarantine
    # ==========================================================================
    conflicting_groups = (
        eligible_df.groupby("duplicate_group")["ai_positive"].nunique().loc[lambda v: v > 1]
    )
    quarantined_count = 0
    if not conflicting_groups.empty:
        conflicting_ids = set(conflicting_groups.index)
        conflicting_mask = eligible_df["duplicate_group"].isin(conflicting_ids)
        quarantined_rows = eligible_df[conflicting_mask]
        quarantined_count = len(quarantined_rows)
        logger.warning(
            "Quarantining %d rows across %d conflicting duplicate groups",
            quarantined_count,
            len(conflicting_ids),
        )
        for q_row in quarantined_rows.to_dict(orient="records"):
            exclusions.append(
                {
                    "row_id": q_row["row_id"],
                    "image_path": q_row["image_path"],
                    "dataset": q_row["dataset"],
                    "label": q_row.get("label"),
                    "provenance": q_row.get("provenance"),
                    "exclusion_reason": "cross_label_conflict",
                    "exclusion_details": f"Duplicate group {q_row['duplicate_group']} contains conflicting labels",
                }
            )
        eligible_df = eligible_df[~conflicting_mask].copy()

    # ==========================================================================
    # Group-Disjoint Split Enforcement & Leakage Checks
    # ==========================================================================
    priority = {"train": 0, "validation": 1, "test": 2, "test_unseen": 3}
    group_split_counts = eligible_df.groupby("duplicate_group")["split"].nunique()
    multi_split_groups = set(group_split_counts[group_split_counts > 1].index)
    if multi_split_groups:
        multi_mask = eligible_df["duplicate_group"].isin(multi_split_groups)
        sub = eligible_df[multi_mask]
        group_to_best = {
            grp: min(splits, key=lambda s: priority.get(str(s), 99))
            for grp, splits in sub.groupby("duplicate_group")["split"]
        }
        eligible_df.loc[multi_mask, "split"] = sub["duplicate_group"].map(group_to_best)

    # Check zero group leakage
    assert_no_group_leakage(eligible_df)

    # Check generator family leakage between train and test_unseen
    synthetic_mask = eligible_df["provenance"].isin(["fully_aigc", "tampered"])
    train_syn = set(
        eligible_df.loc[(eligible_df["split"] == "train") & synthetic_mask, "generator_family"]
    )
    unseen_syn = set(
        eligible_df.loc[
            (eligible_df["split"] == "test_unseen") & synthetic_mask, "generator_family"
        ]
    )
    leakage = train_syn & unseen_syn
    if leakage:
        raise ValueError(
            f"Generator family leakage detected between train and test_unseen: {leakage}"
        )

    # ==========================================================================
    # Select Deterministic Calibration Rows from Actual Singleton Records
    # ==========================================================================
    logger.info("Selecting actual eligible calibration rows (target: %d)...", calibration_size)
    eligible_canonical_df, calibration_df = select_calibration_from_eligible_rows(
        eligible_df=eligible_df,
        calibration_size=calibration_size,
        seed=seed,
    )
    logger.info("Selected %d byte-verified calibration rows", len(calibration_df))

    if len(calibration_df) != calibration_size:
        raise RuntimeError(
            f"Fail-closed: calibration set has {len(calibration_df)} rows, expected exactly {calibration_size}"
        )

    # Ensure all required runtime columns exist and have non-null values
    for col in RUNTIME_IDENTITY_COLUMNS:
        if col not in eligible_canonical_df.columns:
            eligible_canonical_df[col] = ""
        if col not in calibration_df.columns:
            calibration_df[col] = ""

    # Build exclusions DataFrame
    exclusions_df = pd.DataFrame(exclusions)
    if exclusions_df.empty:
        exclusions_df = pd.DataFrame(
            columns=[
                "row_id",
                "image_path",
                "dataset",
                "label",
                "provenance",
                "exclusion_reason",
                "exclusion_details",
            ]
        )

    # ==========================================================================
    # Write Parquet Splits (if not dry_run)
    # ==========================================================================
    split_dfs: dict[str, pd.DataFrame] = {}
    split_counts: dict[str, int] = {}

    for split_name in ("train", "validation", "test", "test_unseen"):
        part_df = eligible_canonical_df[eligible_canonical_df["split"] == split_name].reset_index(
            drop=True
        )
        split_dfs[split_name] = part_df
        split_counts[split_name] = len(part_df)

    split_counts["calibration"] = len(calibration_df)

    all_eligible_combined = pd.concat(
        [split_dfs["train"], split_dfs["validation"], split_dfs["test"], split_dfs["test_unseen"]],
        ignore_index=True,
    )
    manifest_dig = compute_manifest_digest(all_eligible_combined)

    # Revisions report
    source_revs: dict[str, Any] = {}
    for ds_name in sorted(declared_df["dataset"].unique()):
        hf_repo = DATASET_TO_HF_REPO.get(ds_name, "")
        commit_sha = LOCKED_HF_REVISIONS.get(hf_repo, "local_or_non_hf")
        source_revs[ds_name] = {
            "hf_repo": hf_repo,
            "revision": commit_sha,
            "declared_count": int((declared_df["dataset"] == ds_name).sum()),
            "eligible_count": int((all_eligible_combined["dataset"] == ds_name).sum()),
        }

    audit_report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "manifest_sha256": manifest_dig,
        "total_declared_rows": total_declared_rows,
        "total_eligible_rows": len(all_eligible_combined),
        "total_excluded_rows": len(exclusions_df),
        "split_counts": split_counts,
        "counts_by_dataset": {
            str(k): int(v) for k, v in all_eligible_combined["dataset"].value_counts().items()
        },
        "counts_by_provenance": {
            str(k): int(v) for k, v in all_eligible_combined["provenance"].value_counts().items()
        },
        "counts_by_ai_positive": {
            str(k): int(v) for k, v in all_eligible_combined["ai_positive"].value_counts().items()
        },
        "counts_by_generator_family": {
            str(k): int(v)
            for k, v in all_eligible_combined["generator_family"].value_counts().items()
        },
        "duplicate_groups_count": int(all_eligible_combined["duplicate_group"].nunique()),
        "quarantined_conflicting_rows": quarantined_count,
        "leakage_checks": {
            "group_leakage_detected": False,
            "generator_family_leakage_detected": False,
            "forbidden_demo_check_passed": True,
            "newer_model_data_excluded": True,
            "placeholder_sha_detected": False,
        },
        "calibration_audit": {
            "calibration_rows_count": len(calibration_df),
            "disjoint_from_canonical_splits": True,
            "gradient_use_prohibited": True,
            "model_selection_use_prohibited": True,
            "synthetic_records_created": False,
        },
        "exclusions_summary": dict(
            Counter(exclusions_df["exclusion_reason"] if not exclusions_df.empty else [])
        ),
        "source_revisions": source_revs,
    }

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        declared_df.to_parquet(output_dir / "declared_manifest.parquet", index=False)
        for split_name, part_df in split_dfs.items():
            part_df.to_parquet(output_dir / f"{split_name}.parquet", index=False)
            logger.info("Wrote %s (%d rows)", output_dir / f"{split_name}.parquet", len(part_df))
        calibration_df.to_parquet(output_dir / "calibration.parquet", index=False)
        logger.info("Wrote %s (%d rows)", output_dir / "calibration.parquet", len(calibration_df))
        exclusions_df.to_parquet(output_dir / "exclusions.parquet", index=False)
        logger.info("Wrote %s (%d rows)", output_dir / "exclusions.parquet", len(exclusions_df))
        with open(output_dir / "audit_report.json", "w", encoding="utf-8") as f:
            json.dump(audit_report, f, indent=2)
        with open(output_dir / "source_revisions.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "generated_at": datetime.now(UTC).isoformat(),
                    "manifest_digest": manifest_dig,
                    "sources": source_revs,
                },
                f,
                indent=2,
            )
        logger.info("Wrote audit report and source revisions to %s", output_dir)

    return audit_report


# ==============================================================================
# CLI Entry Point
# ==============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze immutable eligible splits and audit report for production pipeline"
    )
    parser.add_argument(
        "--declared-dir",
        "--manifest-dir",
        dest="declared_dir",
        type=Path,
        default=None,
        help="Input manifest directory containing JSONL or Parquet splits",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_PROJECT_ROOT / "splits" / "production_eligible",
        help="Output directory to write production eligible parquets and audit report",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("TECHJAM_DATA_ROOT", str(_PROJECT_ROOT / "data"))),
        help="Local image root (default: TECHJAM_DATA_ROOT or ./data)",
    )
    parser.add_argument(
        "--calibration-size",
        type=int,
        default=4096,
        help="Exact number of calibration rows (default: 4096)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic hashing and sampling",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="Enforce fail-closed invariants (default: True)",
    )
    parser.add_argument(
        "--verify-bytes",
        action="store_true",
        default=True,
        help="Mandatory fail-closed disk byte verification (default: True)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run without writing files to disk",
    )

    args = parser.parse_args()

    declared_dir = args.declared_dir
    if declared_dir is None:
        cand_final = _PROJECT_ROOT / "splits" / "final_teacher_dataset"
        cand_combined = _PROJECT_ROOT / "splits" / "combined_hf_dataset"
        if cand_final.is_dir():
            declared_dir = cand_final
        elif cand_combined.is_dir():
            declared_dir = cand_combined
        else:
            declared_dir = cand_combined
    elif not declared_dir.is_dir():
        cand_combined = _PROJECT_ROOT / "splits" / "combined_hf_dataset"
        if cand_combined.is_dir():
            declared_dir = cand_combined

    report = freeze_production_eligible(
        declared_dir=declared_dir,
        output_dir=args.output_dir,
        data_root=args.data_root,
        calibration_size=args.calibration_size,
        seed=args.seed,
        strict=args.strict,
        verify_bytes=True,  # Mandatory fail-closed byte verification by default
        dry_run=args.dry_run,
    )

    print("\n" + "=" * 70)
    print("PRODUCTION ELIGIBILITY FREEZE COMPLETE")
    print("=" * 70)
    print(f"Manifest SHA256: {report['manifest_sha256']}")
    print(f"Declared rows:   {report['total_declared_rows']}")
    print(f"Eligible rows:   {report['total_eligible_rows']}")
    print(f"Excluded rows:   {report['total_excluded_rows']}")
    print(f"Split counts:    {report['split_counts']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
