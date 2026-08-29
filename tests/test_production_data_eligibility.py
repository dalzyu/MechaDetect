"""Focused tests for production data prefetch, eligibility freeze, and calibration.

Verifies:
- Resumable source-indexed prefetch CLI and atomic write operations.
- Fail-closed eligibility freeze with byte, dimension, hash, and mask validations.
- Rejection of forbidden data ('newer image model data(do not use for training)', organizer demo).
- Explicit retention of public Nano Banana datasets ('google_nano_banana_edited', 'nano_banana_pro_gen').
- Rejection of placeholder SHA evidence ('sha:<dataset>:...', Imgur placeholder).
- Deterministic 32-hex row ID assignment and stable manifest digest computation.
- Cross-label duplicate group quarantine.
- Static INT8 calibration: selected from actual byte-verified eligible rows, strictly disjoint from canonical splits.
- Contracted Parquet schema: presence of runtime identity columns in all splits.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DATA_PREP_DIR = _PROJECT_ROOT / "scripts" / "data_prep"
if str(_DATA_PREP_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_PREP_DIR))
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from acquire_all_images import (
    KNOWN_PLACEHOLDER_SHA256,
    LOCKED_HF_REVISIONS,
    SOURCE_REGISTRY,
    atomic_save_image,
    atomic_write_bytes,
    is_forbidden_row_or_path,
    is_placeholder_sha,
    is_safe_relative_path,
    load_declared_manifests,
)
from freeze_production_eligible import (
    RUNTIME_IDENTITY_COLUMNS,
    compute_deterministic_row_id,
    compute_manifest_digest,
    freeze_production_eligible,
    select_calibration_from_eligible_rows,
)

# ==============================================================================
# 1. Path Safety, Forbidden Rules, and Nano Banana Retention
# ==============================================================================


def test_safe_relative_path_enforcement() -> None:
    assert is_safe_relative_path("images/train/sample_001.jpg")
    assert is_safe_relative_path("reference_only/sub/pic.png")
    assert not is_safe_relative_path("../outside.jpg")
    assert not is_safe_relative_path("images/../../outside.jpg")
    assert not is_safe_relative_path("/root/abs.jpg")
    assert not is_safe_relative_path("C:/windows/path.jpg")
    assert not is_safe_relative_path("D:\\windows\\path.jpg")
    assert not is_safe_relative_path("")


def test_forbidden_rejection_and_public_nano_banana_retention() -> None:
    # 1. Explicit forbidden local folder: 'newer image model data(do not use for training)'
    forbidden_row = {
        "dataset": "custom_images",
        "image_path": "newer image model data(do not use for training)/img_01.jpg",
    }
    is_forbidden, reason = is_forbidden_row_or_path(forbidden_row)
    assert is_forbidden
    assert reason == "forbidden_newer_image_model_data_folder"

    # 2. Organizer COCO val2017 demonstration data
    coco_demo = {
        "dataset": "coco_test",
        "image_path": "coco/val2017/000000001.jpg",
    }
    is_forbidden, reason = is_forbidden_row_or_path(coco_demo)
    assert is_forbidden
    assert reason == "forbidden_organizer_coco_val2017"

    # 3. Organizer DALL-E Advanced demo
    dalle_demo = {
        "dataset": "dalle_eval",
        "image_path": "dalle/advanced/image_01.png",
    }
    is_forbidden, reason = is_forbidden_row_or_path(dalle_demo)
    assert is_forbidden
    assert reason == "forbidden_organizer_dalle_advanced_demo"

    # 4. Public Nano Banana datasets MUST BE KEPT (explicit contract)
    nano_edited = {
        "dataset": "google_nano_banana_edited",
        "image_path": "google_nano_banana_edited/google_nano_banana_edited_000001.jpg",
    }
    is_forbidden, _ = is_forbidden_row_or_path(nano_edited)
    assert not is_forbidden

    nano_gen = {
        "dataset": "nano_banana_pro_gen",
        "image_path": "nano_banana_pro_gen/nano_banana_pro_gen_000001.jpg",
    }
    is_forbidden, _ = is_forbidden_row_or_path(nano_gen)
    assert not is_forbidden


# ==============================================================================
# 2. Placeholder SHA Rejection
# ==============================================================================


def test_placeholder_sha_rejection() -> None:
    # 'sha:<dataset>:...' pattern rejection
    assert is_placeholder_sha("sha:artic_dataset:42:1")
    assert is_placeholder_sha("sha:sid:1234")
    assert is_placeholder_sha("")
    assert is_placeholder_sha(None)

    # Imgur known placeholder hash rejection
    for placeholder in KNOWN_PLACEHOLDER_SHA256:
        assert is_placeholder_sha(placeholder)

    # Valid hex sha256 should NOT be flagged as placeholder
    valid_sha = hashlib.sha256(b"authentic image bytes").hexdigest()
    assert not is_placeholder_sha(valid_sha)


def test_declared_placeholder_sha_retained_in_acquire_manifests(tmp_path: Path) -> None:
    """Verify declared placeholder sha:* rows are NOT excluded during pre-acquisition scan."""
    manifest_dir = tmp_path / "declared"
    manifest_dir.mkdir()
    row = {
        "image_path": "artic_dataset/sample_001.jpg",
        "dataset": "artic_dataset",
        "external_id": "artic_000001",
        "sha256": "sha:artic_dataset:42:1",
        "tamper_mask_path": "",
    }
    with open(manifest_dir / "train.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")

    declared_records, excluded_records = load_declared_manifests(manifest_dir)
    assert len(declared_records) == 1
    assert len(excluded_records) == 0
    assert declared_records[0].image_path == "artic_dataset/sample_001.jpg"
    # expected_sha256 is None because placeholder string is absent evidence
    assert declared_records[0].expected_sha256 is None


# ==============================================================================
# 3. Atomic File Operations
# ==============================================================================


def test_atomic_write_and_cleanup(tmp_path: Path) -> None:
    dest_bytes = tmp_path / "subdir" / "data.bin"
    payload = b"deterministic bytes payload"
    atomic_write_bytes(dest_bytes, payload)

    assert dest_bytes.is_file()
    assert dest_bytes.read_bytes() == payload
    # Verify no temp files left behind
    temp_files = list(dest_bytes.parent.glob(".*tmp*"))
    assert len(temp_files) == 0

    # Test atomic image save
    img = Image.new("RGB", (64, 64), color="blue")
    dest_img = tmp_path / "subdir" / "image.jpg"
    atomic_save_image(img, dest_img, image_format="JPEG")

    assert dest_img.is_file()
    with Image.open(dest_img) as loaded:
        assert loaded.size == (64, 64)
        assert loaded.format == "JPEG"

    temp_img_files = list(dest_img.parent.glob(".*tmp*"))
    assert len(temp_img_files) == 0


# ==============================================================================
# 4. Deterministic Row IDs and Manifest Digests
# ==============================================================================


def test_deterministic_row_id_uniqueness() -> None:
    rec1 = {
        "split": "train",
        "dataset": "artic_dataset",
        "image_path": "artic_dataset/artic_dataset_000001.jpg",
        "sha256": "abc123def456",
        "external_id": "artic_dataset_000001",
        "source_index": "1",
        "source_member_path": "",
    }
    rec2 = dict(rec1)
    rec3 = dict(rec1, source_index="2", external_id="artic_dataset_000002")

    # Identical record yields identical 32-hex ID
    id1 = compute_deterministic_row_id(rec1)
    id2 = compute_deterministic_row_id(rec2)
    id3 = compute_deterministic_row_id(rec3)

    assert len(id1) == 32
    assert id1 == id2
    assert id1 != id3


def test_compute_manifest_digest_stability() -> None:
    rows = [
        {
            "split": "train",
            "dataset": "dsA",
            "image_path": "a.jpg",
            "row_id": "id1",
            "sha256": "sha1",
            "external_id": "ext1",
        },
        {
            "split": "train",
            "dataset": "dsB",
            "image_path": "b.jpg",
            "row_id": "id2",
            "sha256": "sha2",
            "external_id": "ext2",
        },
    ]
    df1 = pd.DataFrame(rows)
    # Permute order of rows
    df2 = pd.DataFrame([rows[1], rows[0]])

    digest1 = compute_manifest_digest(df1)
    digest2 = compute_manifest_digest(df2)

    assert digest1 == digest2
    assert len(digest1) == 64


# ==============================================================================
# 5. Fail-Closed Byte Verification & Cross-Label Conflict Quarantine
# ==============================================================================


def test_fail_closed_byte_verification_and_quarantine(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "declared_manifests"
    manifest_dir.mkdir(parents=True)
    out_dir = tmp_path / "production_eligible"
    data_root = tmp_path / "data"

    # Create dummy images on disk
    # img_0: authentic
    # img_1: tampered (shares duplicate_group with img_0 -> cross-label conflict!)
    # img_clean: clean authentic singleton
    # img_missing: declared in manifest but missing on disk -> must be excluded!
    img0_path = data_root / "ds" / "img_0.jpg"
    img1_path = data_root / "ds" / "img_1.jpg"
    clean_path = data_root / "ds" / "img_clean.jpg"
    placeholder_sha_path = data_root / "ds" / "img_placeholder_sha.jpg"
    known_placeholder_path = data_root / "ds" / "img_known_placeholder.jpg"

    import numpy as np

    rng = np.random.default_rng(42)
    for p in (img0_path, img1_path, clean_path, placeholder_sha_path):
        p.parent.mkdir(parents=True, exist_ok=True)
        arr = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        atomic_save_image(img, p)

    # Create known placeholder byte file matching KNOWN_PLACEHOLDER_SHA256
    known_placeholder_path.parent.mkdir(parents=True, exist_ok=True)
    ph_arr = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    atomic_save_image(Image.fromarray(ph_arr), known_placeholder_path)
    known_ph_hash = hashlib.sha256(known_placeholder_path.read_bytes()).hexdigest()
    from freeze_production_eligible import KNOWN_PLACEHOLDER_SHA256 as FREEZE_PLACEHOLDERS

    FREEZE_PLACEHOLDERS.add(known_ph_hash)
    records = [
        {
            "image_path": "ds/img_0.jpg",
            "label": 0,
            "provenance": "authentic",
            "ai_positive": 0,
            "dataset": "ds",
            "generator": "authentic",
            "generator_family": "authentic",
            "source_image_group": "group_conflict",
            "duplicate_group": "dup_conflict_01",
            "width": 64,
            "height": 64,
            "file_format": "JPEG",
            "tamper_mask_path": "",
            "sha256": "initial_sha",
            "perceptual_hash": "0000000000000001",
            "quality_score": 1.0,
            "external_id": "ds_0",
            "split": "train",
        },
        {
            "image_path": "ds/img_1.jpg",
            "label": 1,
            "provenance": "tampered",
            "ai_positive": 1,
            "dataset": "ds",
            "generator": "tamper_gen",
            "generator_family": "tamper_gen",
            "source_image_group": "group_conflict",
            "duplicate_group": "dup_conflict_01",
            "width": 64,
            "height": 64,
            "file_format": "JPEG",
            "tamper_mask_path": "",
            "sha256": "initial_sha",
            "perceptual_hash": "0000000000000001",
            "quality_score": 1.0,
            "external_id": "ds_1",
            "split": "train",
        },
        {
            "image_path": "ds/img_clean.jpg",
            "label": 0,
            "provenance": "authentic",
            "ai_positive": 0,
            "dataset": "ds",
            "generator": "authentic",
            "generator_family": "authentic",
            "source_image_group": "group_clean",
            "duplicate_group": "dup_clean_02",
            "width": 64,
            "height": 64,
            "file_format": "JPEG",
            "tamper_mask_path": "",
            "sha256": "initial_sha",
            "perceptual_hash": "0000000000000002",
            "quality_score": 1.0,
            "external_id": "ds_clean",
            "split": "train",
        },
        {
            "image_path": "ds/img_missing.jpg",
            "label": 0,
            "provenance": "authentic",
            "ai_positive": 0,
            "dataset": "ds",
            "generator": "authentic",
            "generator_family": "authentic",
            "source_image_group": "group_missing",
            "duplicate_group": "dup_missing_03",
            "width": 64,
            "height": 64,
            "file_format": "JPEG",
            "tamper_mask_path": "",
            "sha256": "initial_sha",
            "perceptual_hash": "0000000000000003",
            "quality_score": 1.0,
            "external_id": "ds_missing",
            "split": "train",
        },
        {
            "image_path": "ds/img_placeholder_sha.jpg",
            "label": 0,
            "provenance": "authentic",
            "ai_positive": 0,
            "dataset": "ds",
            "generator": "authentic",
            "generator_family": "authentic",
            "source_image_group": "group_placeholder_sha",
            "duplicate_group": "dup_placeholder_sha",
            "width": 64,
            "height": 64,
            "file_format": "JPEG",
            "tamper_mask_path": "",
            "sha256": "sha:artic_dataset:42:1",  # Declared placeholder!
            "perceptual_hash": "0000000000000004",
            "quality_score": 1.0,
            "external_id": "ds_placeholder_sha",
            "split": "train",
        },
        {
            "image_path": "ds/img_known_placeholder.jpg",
            "label": 0,
            "provenance": "authentic",
            "ai_positive": 0,
            "dataset": "ds",
            "generator": "authentic",
            "generator_family": "authentic",
            "source_image_group": "group_known_ph",
            "duplicate_group": "dup_known_ph",
            "width": 64,
            "height": 64,
            "file_format": "JPEG",
            "tamper_mask_path": "",
            "sha256": "some_declared_sha",
            "perceptual_hash": "0000000000000005",
            "quality_score": 1.0,
            "external_id": "ds_known_ph",
            "split": "train",
        },
    ]

    for s in ("train", "validation", "test", "test_unseen"):
        split_records = records if s == "train" else []
        with open(manifest_dir / f"{s}.jsonl", "w", encoding="utf-8") as f:
            for r in split_records:
                r_s = dict(r, split=s)
                f.write(json.dumps(r_s) + "\n")

    report = freeze_production_eligible(
        declared_dir=manifest_dir,
        output_dir=out_dir,
        data_root=data_root,
        calibration_size=1,
        seed=42,
        strict=False,
        verify_bytes=True,  # Mandatory fail-closed byte verification
    )

    excl_df = pd.read_parquet(out_dir / "exclusions.parquet")
    reasons = set(excl_df["exclusion_reason"])

    # 1. Missing file must be excluded
    assert "missing_file_on_disk" in reasons

    # 2. Conflicting duplicate group must be quarantined
    assert "cross_label_conflict" in reasons
    assert report["quarantined_conflicting_rows"] == 2

    # 3. Both clean row and placeholder-sha row survived into output
    assert out_dir.joinpath("calibration.parquet").is_file()
    assert out_dir.joinpath("train.parquet").is_file()
    calib_df = pd.read_parquet(out_dir / "calibration.parquet")
    train_df = pd.read_parquet(out_dir / "train.parquet")
    assert len(calib_df) == 1
    assert len(train_df) == 1

    # Verify placeholder declared SHA was replaced by actual verified disk bytes SHA-256
    all_out = pd.concat([calib_df, train_df], ignore_index=True)
    for _, out_row in all_out.iterrows():
        assert not out_row["sha256"].startswith("sha:"), (
            "Placeholder sha string survived into output parquet!"
        )
        assert len(out_row["sha256"]) == 64, f"Invalid SHA-256 length: {out_row['sha256']}"
        disk_file = data_root / out_row["image_path"]
        actual_bytes_sha = hashlib.sha256(disk_file.read_bytes()).hexdigest()
        assert out_row["sha256"] == actual_bytes_sha, "Output SHA does not match actual disk bytes!"
    # 4. Known placeholder bytes on disk must be rejected
    assert "placeholder_sha_on_disk" in reasons
    FREEZE_PLACEHOLDERS.discard(known_ph_hash)


# ==============================================================================
# 6. Static INT8 Calibration Disjointness and Zero Synthesis
# ==============================================================================


def test_calibration_selection_from_actual_eligible_rows() -> None:
    # Create test pool of 100 byte-verified rows
    rows = []
    for i in range(100):
        prov = "authentic" if i % 2 == 0 else ("tampered" if i % 4 == 1 else "fully_aigc")
        ai_pos = 0 if prov == "authentic" else 1
        rows.append(
            {
                "row_id": f"row_{i:06d}",
                "split": "train" if i < 70 else ("validation" if i < 85 else "test"),
                "image_path": f"dataset/img_{i:06d}.jpg",
                "label": ai_pos,
                "provenance": prov,
                "ai_positive": ai_pos,
                "dataset": f"dataset_{i % 5}",
                "generator": f"gen_{i % 3}",
                "generator_family": f"gen_fam_{i % 3}",
                "source_image_group": f"group_{i:06d}",
                "duplicate_group": f"dup_{i:06d}",  # all singletons
                "width": 512,
                "height": 512,
                "file_format": "JPEG",
                "tamper_mask_path": "",
                "sha256": f"sha256_{i:06d}",
                "perceptual_hash": f"{i:016x}",
                "quality_score": 1.0,
                "external_id": f"ext_{i:06d}",
                "source_index": str(i),
            }
        )

    eligible_df = pd.DataFrame(rows)

    rem_df, calib_df = select_calibration_from_eligible_rows(
        eligible_df=eligible_df,
        calibration_size=20,
        seed=42,
    )

    # 1. Exact count selected
    assert len(calib_df) == 20
    assert len(rem_df) == 80

    # 2. Strict disjointness: removed from eligible splits
    assert set(calib_df["image_path"]).isdisjoint(set(rem_df["image_path"]))
    assert set(calib_df["sha256"]).isdisjoint(set(rem_df["sha256"]))
    assert set(calib_df["row_id"]).isdisjoint(set(rem_df["row_id"]))
    assert set(calib_df["duplicate_group"]).isdisjoint(set(rem_df["duplicate_group"]))

    # 3. Stratification: balanced positive and negative
    counts = dict(calib_df["ai_positive"].value_counts())
    assert counts[0] == 10
    assert counts[1] == 10

    # 4. Zero synthesis: all image_paths originated from eligible_df
    assert set(calib_df["image_path"]).issubset(set(eligible_df["image_path"]))
    assert set(calib_df["sha256"]).issubset(set(eligible_df["sha256"]))

    # 5. Metadata fields assigned
    assert "aspect_ratio_bucket" in calib_df.columns
    assert "calib_transformation" in calib_df.columns
    assert "calib_severity" in calib_df.columns
    assert (calib_df["split"] == "calibration").all()

    # 6. Fail-closed: raises RuntimeError if requested calibration size exceeds available candidates
    import pytest

    with pytest.raises(RuntimeError, match="Fail-closed"):
        select_calibration_from_eligible_rows(
            eligible_df=eligible_df,
            calibration_size=200,
            seed=42,
        )


# ==============================================================================
# 7. Production Eligible Schema Contract Verification on Materialized Package
# ==============================================================================


def test_production_eligible_schema_contract_on_output(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "declared"
    manifest_dir.mkdir(parents=True)
    out_dir = tmp_path / "production_eligible"
    data_root = tmp_path / "data"

    # Create dummy images
    records = []
    for i in range(12):
        rel = f"cohort/img_{i:02d}.jpg"
        p = data_root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        atomic_save_image(Image.fromarray(arr), p)
        split_name = (
            "train" if i < 6 else ("validation" if i < 8 else ("test" if i < 10 else "test_unseen"))
        )
        if i % 2 == 0:
            gen_name = "authentic"
        elif split_name == "test_unseen":
            gen_name = "unseen_generator"
        else:
            gen_name = "train_generator"
        records.append(
            {
                "image_path": rel,
                "label": 0 if i % 2 == 0 else 1,
                "provenance": "authentic" if i % 2 == 0 else "fully_aigc",
                "ai_positive": 0 if i % 2 == 0 else 1,
                "dataset": "cohort",
                "generator": gen_name,
                "generator_family": gen_name,
                "source_image_group": f"grp_{i}",
                "duplicate_group": f"dup_{i}",
                "width": 32,
                "height": 32,
                "file_format": "JPEG",
                "tamper_mask_path": "",
                "sha256": f"dummy_sha_{i}",
                "perceptual_hash": f"{i:016x}",
                "quality_score": 1.0,
                "external_id": f"cohort_{i}",
                "split": split_name,
            }
        )

    for s in ("train", "validation", "test", "test_unseen"):
        split_recs = [r for r in records if r["split"] == s]
        with open(manifest_dir / f"{s}.jsonl", "w", encoding="utf-8") as f:
            for r in split_recs:
                f.write(json.dumps(r) + "\n")

    freeze_production_eligible(
        declared_dir=manifest_dir,
        output_dir=out_dir,
        data_root=data_root,
        calibration_size=2,
        seed=42,
        strict=False,
        verify_bytes=True,
    )

    expected_parquets = (
        "declared_manifest.parquet",
        "train.parquet",
        "validation.parquet",
        "test.parquet",
        "test_unseen.parquet",
        "calibration.parquet",
        "exclusions.parquet",
    )

    for p_name in expected_parquets:
        p_path = out_dir / p_name
        assert p_path.is_file(), f"Missing required parquet: {p_name}"

        df = pd.read_parquet(p_path)
        if p_name != "exclusions.parquet":
            for col in RUNTIME_IDENTITY_COLUMNS:
                assert col in df.columns, f"Missing {col} in {p_name}"
                assert not df[col].isna().any(), f"NaN in {col} of {p_name}"

    # Verify audit_report.json and source_revisions.json
    assert out_dir.joinpath("audit_report.json").is_file()
    assert out_dir.joinpath("source_revisions.json").is_file()


# ==============================================================================
# 8. Source Registry Mapping Completeness & Immutable Revisions
# ==============================================================================


def test_source_registry_completeness_and_immutable_revisions() -> None:
    # Public Nano Banana datasets must be present with correct configurations
    assert "google_nano_banana_edited" in SOURCE_REGISTRY
    assert (
        SOURCE_REGISTRY["google_nano_banana_edited"].repo
        == "Tungtom2004/Google_Nano_Banana_Edited_Images"
    )
    assert SOURCE_REGISTRY["google_nano_banana_edited"].paired_field == "edited_image"

    assert "nano_banana_pro_gen" in SOURCE_REGISTRY
    assert SOURCE_REGISTRY["nano_banana_pro_gen"].repo == "FlameF0X/nano-banana-pro-gen-zh-en"

    # Verify every registered HF repo has an immutable 40-hex commit SHA (never "main")
    for repo, sha in LOCKED_HF_REVISIONS.items():
        assert len(sha) == 40, f"Revision for {repo} must be 40-hex commit SHA, got {sha}"
        assert all(c in "0123456789abcdefABCDEF" for c in sha), f"Invalid hex in {sha}"


# ==============================================================================
# 9. CLI Invocations and Dry-Run Acceptance
# ==============================================================================


def test_acquire_all_images_cli_dry_run(tmp_path: Path) -> None:
    script_path = _DATA_PREP_DIR / "acquire_all_images.py"
    revisions_path = tmp_path / "test_source_revisions.json"
    cmd = [
        sys.executable,
        str(script_path),
        "--dry-run",
        "--sources",
        "google_nano_banana_edited,nano_banana_pro_gen",
        "--revisions-path",
        str(revisions_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "MechaDetect" in res.stdout or "Acquisition" in res.stdout
    assert revisions_path.is_file()
    with open(revisions_path, encoding="utf-8") as f:
        data = json.load(f)
    assert "google_nano_banana_edited" in data["sources"]
    assert "nano_banana_pro_gen" in data["sources"]
    # Revisions must be immutable 40-hex SHAs, never 'main'
    assert len(data["sources"]["google_nano_banana_edited"]["revision"]) == 40
    assert len(data["sources"]["nano_banana_pro_gen"]["revision"]) == 40


def test_freeze_production_eligible_cli_dry_run(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "declared"
    manifest_dir.mkdir(parents=True)
    data_root = tmp_path / "data"
    script_path = _DATA_PREP_DIR / "freeze_production_eligible.py"

    rng = np.random.default_rng(42)
    splits = ("train", "validation", "test", "test_unseen")
    for idx, s in enumerate(splits):
        rel = f"ds/sample_{s}.jpg"
        p = data_root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        arr = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
        atomic_save_image(Image.fromarray(arr), p)

        # Provide 2 rows per split so calibration can take 1 without emptying the split
        rows = []
        for r_i in range(2):
            r_rel = f"ds/sample_{s}_{r_i}.jpg"
            r_p = data_root / r_rel
            atomic_save_image(Image.fromarray(arr), r_p)
            rows.append(
                {
                    "image_path": r_rel,
                    "label": 0,
                    "provenance": "authentic",
                    "ai_positive": 0,
                    "dataset": "ds",
                    "generator": "authentic",
                    "generator_family": "authentic",
                    "source_image_group": f"g_{s}_{r_i}",
                    "duplicate_group": f"d_{s}_{r_i}",
                    "width": 32,
                    "height": 32,
                    "file_format": "JPEG",
                    "tamper_mask_path": "",
                    "sha256": f"dummy_{s}_{r_i}",
                    "perceptual_hash": f"{idx * 10 + r_i:016x}",
                    "quality_score": 1.0,
                    "external_id": f"ext_{s}_{r_i}",
                    "split": s,
                }
            )

        with open(manifest_dir / f"{s}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    cmd = [
        sys.executable,
        str(script_path),
        "--declared-dir",
        str(manifest_dir),
        "--data-root",
        str(data_root),
        "--dry-run",
        "--calibration-size",
        "2",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "PRODUCTION ELIGIBILITY FREEZE COMPLETE" in res.stdout


def test_local_source_missing_records_explicit_errors(tmp_path: Path) -> None:
    """Verify local sources record explicit errors when assets are absent, not silent skips."""
    from acquire_all_images import (
        DeclaredRecord,
        acquire_ewan_gpt_images,
        acquire_gpt_image_2_user_screenshots,
    )

    empty_data_root = tmp_path / "empty_data"
    empty_data_root.mkdir()
    recs = [
        DeclaredRecord(
            image_path="ewan_gpt_images/sample_001.png",
            dataset="ewan_gpt_images",
            split="train",
            external_id="ewan_001",
            source_index=1,
        ),
        DeclaredRecord(
            image_path="ewan_gpt_images/sample_002.png",
            dataset="ewan_gpt_images",
            split="train",
            external_id="ewan_002",
            source_index=2,
        ),
    ]
    # With non-existent zip path and no remote auth, must record 2 errors, not silent skips
    saved, skipped, errors = acquire_ewan_gpt_images(
        recs, empty_data_root, dry_run=False, resume=True
    )
    assert saved == 0
    assert skipped == 0
    assert errors == 2

    user_recs = [
        DeclaredRecord(
            image_path="gpt_image_2_user_screenshots/shot_001.png",
            dataset="gpt_image_2_user_screenshots",
            split="train",
            external_id="shot_001",
            source_index=1,
        ),
    ]
    saved, skipped, errors = acquire_gpt_image_2_user_screenshots(
        user_recs, empty_data_root, dry_run=False, resume=True
    )
    assert saved == 0
    assert errors == 1


def test_unverified_corrupt_or_flat_rows_never_enter_output_splits(tmp_path: Path) -> None:
    """Verify corrupted, flat, or tiny images are quarantined to exclusions and never enter splits."""
    manifest_dir = tmp_path / "declared_bad"
    manifest_dir.mkdir(parents=True)
    out_dir = tmp_path / "out_bad"
    data_root = tmp_path / "data_bad"

    # 1. Corrupted image bytes (invalid header)
    corrupt_p = data_root / "bad" / "corrupt.jpg"
    corrupt_p.parent.mkdir(parents=True, exist_ok=True)
    corrupt_p.write_bytes(b"NOT_A_VALID_IMAGE_HEADER_DATA_STREAM")

    # 2. Flat image (0 variance)
    flat_p = data_root / "bad" / "flat.jpg"
    atomic_save_image(Image.new("RGB", (64, 64), color=(128, 128, 128)), flat_p)

    # 3. Tiny image (<16x16)
    tiny_p = data_root / "bad" / "tiny.jpg"
    atomic_save_image(Image.new("RGB", (8, 8), color="red"), tiny_p)

    # 4. Valid clean image
    clean_p = data_root / "bad" / "clean.jpg"
    rng = np.random.default_rng(123)
    atomic_save_image(Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)), clean_p)

    records = [
        {
            "image_path": "bad/corrupt.jpg",
            "label": 0,
            "provenance": "authentic",
            "ai_positive": 0,
            "dataset": "ds",
            "generator": "auth",
            "generator_family": "auth",
            "source_image_group": "g1",
            "duplicate_group": "d1",
            "width": 64,
            "height": 64,
            "file_format": "JPEG",
            "tamper_mask_path": "",
            "sha256": "sha_corrupt",
            "perceptual_hash": "0000000000000001",
            "quality_score": 1.0,
            "external_id": "bad_1",
            "split": "train",
        },
        {
            "image_path": "bad/flat.jpg",
            "label": 0,
            "provenance": "authentic",
            "ai_positive": 0,
            "dataset": "ds",
            "generator": "auth",
            "generator_family": "auth",
            "source_image_group": "g2",
            "duplicate_group": "d2",
            "width": 64,
            "height": 64,
            "file_format": "JPEG",
            "tamper_mask_path": "",
            "sha256": "sha_flat",
            "perceptual_hash": "0000000000000002",
            "quality_score": 1.0,
            "external_id": "bad_2",
            "split": "train",
        },
        {
            "image_path": "bad/tiny.jpg",
            "label": 0,
            "provenance": "authentic",
            "ai_positive": 0,
            "dataset": "ds",
            "generator": "auth",
            "generator_family": "auth",
            "source_image_group": "g3",
            "duplicate_group": "d3",
            "width": 8,
            "height": 8,
            "file_format": "JPEG",
            "tamper_mask_path": "",
            "sha256": "sha_tiny",
            "perceptual_hash": "0000000000000003",
            "quality_score": 1.0,
            "external_id": "bad_3",
            "split": "train",
        },
        {
            "image_path": "bad/clean.jpg",
            "label": 0,
            "provenance": "authentic",
            "ai_positive": 0,
            "dataset": "ds",
            "generator": "auth",
            "generator_family": "auth",
            "source_image_group": "g4",
            "duplicate_group": "d4",
            "width": 64,
            "height": 64,
            "file_format": "JPEG",
            "tamper_mask_path": "",
            "sha256": "sha_clean",
            "perceptual_hash": "0000000000000004",
            "quality_score": 1.0,
            "external_id": "clean_4",
            "split": "train",
        },
    ]

    for s in ("train", "validation", "test", "test_unseen"):
        with open(manifest_dir / f"{s}.jsonl", "w", encoding="utf-8") as f:
            recs_to_write = records if s == "train" else []
            for r in recs_to_write:
                f.write(json.dumps(r) + "\n")

    freeze_production_eligible(
        declared_dir=manifest_dir,
        output_dir=out_dir,
        data_root=data_root,
        calibration_size=1,
        seed=42,
    )

    excl = pd.read_parquet(out_dir / "exclusions.parquet")
    excl_reasons = set(excl["exclusion_reason"])
    assert "corrupted_image_bytes" in excl_reasons
    assert "image_too_small_or_flat" in excl_reasons

    calib = pd.read_parquet(out_dir / "calibration.parquet")
    train = pd.read_parquet(out_dir / "train.parquet")
    all_eligible_paths = set(calib["image_path"]).union(set(train["image_path"]))
    assert "bad/clean.jpg" in all_eligible_paths
    assert "bad/corrupt.jpg" not in all_eligible_paths
    assert "bad/flat.jpg" not in all_eligible_paths
    assert "bad/tiny.jpg" not in all_eligible_paths
