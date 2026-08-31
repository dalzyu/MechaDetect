"""Focused tests for the Hugging Face transparency package generator."""
from __future__ import annotations

from pathlib import Path
import io
import sys
import zipfile

import pandas as pd
from PIL import Image
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT / "scripts" / "data_prep") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "scripts" / "data_prep"))

from prepare_hf_transparency import (
    _parse_bool,
    compute_stable_manifest_digest,
    compute_stable_row_id,
    prepare_hf_transparency_package,
    project_metadata_manifest,
    resolve_safe_staging_target,
    stage_file_safely,
    assert_forbidden_data_absent,
)


def _create_synthetic_manifest() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "dataset": "diffusionforensics",
            "image_path": "diffusionforensics/adm/sample_001.png",
            "split": "train",
            "official_split": "train",
            "generator": "adm",
            "generator_family": "adm",
            "provenance": "fully_aigc",
            "label": 2,
            "ai_positive": 1,
            "width": 256,
            "height": 256,
            "file_format": "PNG",
            "sha256": "aaaa1111222233334444555566667777888899990000aaaabbbbccccddddeeee",
            "external_id": "df_001",
            "source_url": "https://huggingface.co/datasets/nebula/DF-arrow",
            "created_at": "2026-01-01T00:00:00Z",
        },
        {
            "dataset": "ewan_gpt_images",
            "image_path": "ewan_gpt_images/AI_Images/image_001.png",
            "split": "train",
            "official_split": "train",
            "generator": "gpt_image",
            "generator_family": "gpt_image",
            "provenance": "fully_aigc",
            "label": 2,
            "ai_positive": 1,
            "width": 1536,
            "height": 1024,
            "file_format": "PNG",
            "sha256": "bbbb1111222233334444555566667777888899990000aaaabbbbccccddddeeee",
            "external_id": "ewan_001",
            "source_url": "local://100+ AI_Images from ewan.zip#AI_Images/image_001.png",
            "created_at": "2026-08-29T00:00:00Z",
        },
        {
            "dataset": "gpt_image_2_user_screenshots",
            "image_path": "gpt_image_2_user_screenshots/gpt_image_2_screenshot_001.jpg",
            "split": "train",
            "official_split": "train",
            "generator": "gpt_image_2",
            "generator_family": "gpt_image",
            "provenance": "fully_aigc",
            "label": 2,
            "ai_positive": 1,
            "width": 1024,
            "height": 1024,
            "file_format": "JPEG",
            "sha256": "cccc1111222233334444555566667777888899990000aaaabbbbccccddddeeee",
            "external_id": "AI_Images/image_001.png",
            "source_url": "archive:gpt-image-2-synthetic-images.zip:AI_Images/image_001.png",
            "created_at": "2026-08-29T12:00:00Z",
        },
    ])


def _create_dummy_zip(path: Path, members: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        for m in members:
            # Create a tiny 16x16 PNG
            img = Image.new("RGB", (16, 16), color=(100, 150, 200))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            zf.writestr(m, buf.getvalue())


def test_deterministic_projection_no_image_bytes(tmp_path: Path) -> None:
    raw_df = _create_synthetic_manifest()
    # Intentionally add a dummy bytes column that should be stripped
    raw_df["image_bytes"] = [b"123", b"456", b"789"]

    manifest_file = tmp_path / "test_manifest.jsonl"
    raw_df.to_json(manifest_file, orient="records", lines=True)

    out_dir = tmp_path / "output"
    res = prepare_hf_transparency_package(
        manifest_paths=[manifest_file],
        output_dir=out_dir,
        is_private=False,
    )

    # 1. Output files exist
    parquet_path = out_dir / "transparency.parquet"
    jsonl_path = out_dir / "transparency.jsonl"
    manifest_path = out_dir / "upload_manifest.json"
    assert parquet_path.is_file()
    assert jsonl_path.is_file()
    assert manifest_path.is_file()

    # 2. Read back parquet and verify schema
    p_df = pd.read_parquet(parquet_path)
    assert len(p_df) == 3
    assert "row_id" in p_df.columns
    assert "image_payload_policy" in p_df.columns
    assert "source_index" in p_df.columns

    # Verify no binary/image columns exist
    for col in p_df.columns:
        assert col not in {"image", "bytes", "image_bytes", "pil_image"}
        sample = p_df[col].dropna()
        if len(sample) > 0:
            assert not isinstance(sample.iloc[0], (bytes, bytearray, Image.Image))

    # 3. Deterministic repeatability
    out_dir_2 = tmp_path / "output_2"
    res_2 = prepare_hf_transparency_package(
        manifest_paths=[manifest_file],
        output_dir=out_dir_2,
        is_private=False,
    )
    p_df_2 = pd.read_parquet(out_dir_2 / "transparency.parquet")
    assert list(p_df["row_id"]) == list(p_df_2["row_id"])
    assert res["transparency_manifest_digest"] == res_2["transparency_manifest_digest"]


def test_payload_policy_without_and_with_archives(tmp_path: Path) -> None:
    raw_df = _create_synthetic_manifest()
    manifest_file = tmp_path / "manifest.jsonl"
    raw_df.to_json(manifest_file, orient="records", lines=True)

    # Case A: Without archives supplied
    out_a = tmp_path / "out_a"
    res_a = prepare_hf_transparency_package(
        manifest_paths=[manifest_file],
        output_dir=out_a,
        is_private=False,
    )
    df_a = pd.read_parquet(out_a / "transparency.parquet")
    assert len(df_a) == 3
    assert res_a["row_counts"]["missing_private_payload_count"] == 2
    assert res_a["row_counts"]["by_payload_policy"]["metadata_only"] == 3
    assert res_a["row_counts"]["by_payload_policy"]["private_upload"] == 0

    # Case B: With archives supplied
    ewan_zip = tmp_path / "ewan.zip"
    gpt2_zip = tmp_path / "gpt2.zip"
    _create_dummy_zip(ewan_zip, ["AI_Images/image_001.png"])
    _create_dummy_zip(gpt2_zip, ["AI_Images/image_001.png"])

    out_b = tmp_path / "out_b"
    res_b = prepare_hf_transparency_package(
        manifest_paths=[manifest_file],
        output_dir=out_b,
        ewan_zip=ewan_zip,
        gpt2_zip=gpt2_zip,
        is_private=True,
    )
    df_b = pd.read_parquet(out_b / "transparency.parquet")
    assert res_b["row_counts"]["missing_private_payload_count"] == 0
    assert res_b["row_counts"]["by_payload_policy"]["metadata_only"] == 1
    assert res_b["row_counts"]["by_payload_policy"]["private_upload"] == 2

    # Public dataset row is always metadata_only
    df_row = df_b[df_b["dataset"] == "diffusionforensics"].iloc[0]
    assert df_row["image_payload_policy"] == "metadata_only"

    # Private rows are private_upload
    ewan_row = df_b[df_b["dataset"] == "ewan_gpt_images"].iloc[0]
    assert ewan_row["image_payload_policy"] == "private_upload"
    assert ewan_row["source_archive_path"] == "ewan.zip"
    assert len(ewan_row["source_archive_sha256"]) == 64


def test_private_staging_privacy_gate_and_no_clobber(tmp_path: Path) -> None:
    raw_df = _create_synthetic_manifest()
    manifest_file = tmp_path / "manifest.jsonl"
    raw_df.to_json(manifest_file, orient="records", lines=True)

    ewan_zip = tmp_path / "ewan.zip"
    _create_dummy_zip(ewan_zip, ["AI_Images/image_001.png"])

    # 1. Staging without private gate MUST fail closed
    with pytest.raises(ValueError, match="requires an explicit privacy gate"):
        prepare_hf_transparency_package(
            manifest_paths=[manifest_file],
            output_dir=tmp_path / "out_fail",
            ewan_zip=ewan_zip,
            include_private_images=True,
            is_private=False,  # Private gate NOT passed
        )

    # 2. Staging with mismatched member count without allow_partial_archive MUST fail closed
    with pytest.raises(ValueError, match="must contain exactly 102 image members"):
        prepare_hf_transparency_package(
            manifest_paths=[manifest_file],
            output_dir=tmp_path / "out_count_fail",
            ewan_zip=ewan_zip,
            include_private_images=True,
            is_private=True,
            allow_partial_archive=False,
        )

    # 3. Staging with private gate and allow_partial_archive=True succeeds
    out_ok = tmp_path / "out_ok"
    staged_dir = out_ok / "private_images"
    res = prepare_hf_transparency_package(
        manifest_paths=[manifest_file],
        output_dir=out_ok,
        ewan_zip=ewan_zip,
        include_private_images=True,
        staged_images_dir=staged_dir,
        is_private=True,
        allow_partial_archive=True,
    )
    staged_file = staged_dir / "ewan_gpt_images" / "AI_Images" / "image_001.png"
    assert staged_file.is_file()
    assert len(res["files"]) >= 2

    # 3. Hash-safe no-clobber behavior:
    # Existing identical file is safely reused
    written, sha = stage_file_safely(staged_file.read_bytes(), staged_file)
    assert not written  # Reused without rewrite

    # Mismatched existing file raises FileExistsError
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        stage_file_safely(b"different_content", staged_file)


def test_forbidden_content_rejected() -> None:
    # Newer image model data
    forbidden_df = pd.DataFrame([
        {
            "dataset": "newer image model data(do not use for training)",
            "image_path": "forbidden/img.png",
            "split": "train",
        }
    ])
    with pytest.raises(ValueError, match="Forbidden organizer demonstration or newer-model data"):
        assert_forbidden_data_absent(forbidden_df)

    # COCO val2017 demonstration data
    coco_df = pd.DataFrame([
        {
            "dataset": "coco_val2017",
            "image_path": "coco/val2017/000000001.jpg",
            "split": "train",
        }
    ])
    with pytest.raises(ValueError, match="Forbidden organizer demonstration or newer-model data"):
        assert_forbidden_data_absent(coco_df)


def test_upload_gates_fail_closed(tmp_path: Path) -> None:
    manifest_file = tmp_path / "manifest.jsonl"
    _create_synthetic_manifest().to_json(manifest_file, orient="records", lines=True)

    # 1. Upload without repo-id fails
    with pytest.raises(ValueError, match="--repo-id was not provided"):
        prepare_hf_transparency_package(
            manifest_paths=[manifest_file],
            output_dir=tmp_path / "out_up1",
            upload=True,
            repo_id=None,
            is_private=True,
        )

    # 2. Upload without private gate fails
    with pytest.raises(ValueError, match="target repository is not marked private"):
        prepare_hf_transparency_package(
            manifest_paths=[manifest_file],
            output_dir=tmp_path / "out_up2",
            upload=True,
            repo_id="test_org/test_repo",
            is_private=False,
        )


def test_edge_cases_and_hardening(tmp_path: Path) -> None:
    # 1. Test _parse_bool handles string variants safely
    assert _parse_bool("False") is False
    assert _parse_bool("false") is False
    assert _parse_bool("0") is False
    assert _parse_bool(False) is False
    assert _parse_bool("True") is True
    assert _parse_bool("1") is True
    assert _parse_bool(True) is True

    # 2. Test source_index = 0 is preserved (not dropped to external_id)
    # and source_archive_path does not leak absolute local machine paths
    raw_df = pd.DataFrame([
        {
            "dataset": "diffusionforensics",
            "image_path": "df/sample.png",
            "split": "train",
            "source_index": 0,
            "external_id": "ext_999",
            "source_archive_path": "C:/Users/secret_user/sensitive/data/archive.zip",
            "source_archive_sha256": "abcdef123456",
        }
    ])
    projected_df, _ = project_metadata_manifest(
        raw_df=raw_df,
        ewan_archive_supplied=False,
        gpt2_archive_supplied=False,
    )
    assert projected_df.iloc[0]["source_index"] == "0"
    assert projected_df.iloc[0]["source_archive_path"] == "archive.zip"
    assert projected_df.iloc[0]["quality_score"] == 1.0

    private_df = pd.DataFrame([
        {
            "dataset": "gpt_image_2_user_screenshots",
            "image_path": "D:/runtime/final/gpt_image_2_screenshot_001.jpg",
            "split": "test",
            "source_index": 0,
            "external_id": "AI_Images/image_001.png",
            "source_member_path": "AI_Images/image_001.png",
            "quality_score": 0.81234,
        }
    ])
    private_projection, _ = project_metadata_manifest(
        raw_df=private_df,
        ewan_archive_supplied=False,
        gpt2_archive_supplied=True,
        gpt2_archive_path=Path("gpt-image-2-synthetic-images.zip"),
        gpt2_archive_sha256="f" * 64,
    )
    private_row = private_projection.iloc[0]
    assert private_row["image_path"] == "gpt_image_2_user_screenshots/gpt_image_2_screenshot_001.jpg"
    assert private_row["image_payload_policy"] == "private_upload"
    assert private_row["redistribution_mode"] == "embed_bytes"
    assert private_row["quality_score"] == 0.8123
    changed_index = dict(private_row)
    changed_index["source_index"] = "1"
    assert compute_stable_row_id(private_row) != compute_stable_row_id(changed_index)

    # 3. Test resolve_safe_staging_target blocks directory traversal
    base_dir = tmp_path / "staging"
    base_dir.mkdir()
    with pytest.raises(ValueError, match="Unsafe relative path|escapes staging directory"):
        resolve_safe_staging_target(base_dir, "../escaped.jpg")
    with pytest.raises(ValueError, match="Unsafe relative path|escapes staging directory"):
        resolve_safe_staging_target(base_dir, "subdir/../../escaped.jpg")

    # 4. Test compute_stable_manifest_digest remains stable regardless of duplicate keys
    df1 = pd.DataFrame([
        {"split": "train", "dataset": "ds", "image_path": "img.png", "sha256": "111", "external_id": "a"},
        {"split": "train", "dataset": "ds", "image_path": "img.png", "sha256": "222", "external_id": "b"},
    ])
    df2 = pd.DataFrame([
        {"split": "train", "dataset": "ds", "image_path": "img.png", "sha256": "222", "external_id": "b"},
        {"split": "train", "dataset": "ds", "image_path": "img.png", "sha256": "111", "external_id": "a"},
    ])
    digest1 = compute_stable_manifest_digest(df1)
    digest2 = compute_stable_manifest_digest(df2)
    assert digest1 == digest2


def test_gpt2_staging_sanitizes_absolute_manifest_path(tmp_path: Path) -> None:
    raw_df = pd.DataFrame([
        {
            "dataset": "gpt_image_2_user_screenshots",
            "image_path": "D:/runtime/final/gpt_image_2_screenshot_001.jpg",
            "split": "train",
            "external_id": "AI_Images/image_001.png",
            "source_member_path": "AI_Images/image_001.png",
            "sha256": "",
        }
    ])
    manifest_file = tmp_path / "manifest.jsonl"
    raw_df.to_json(manifest_file, orient="records", lines=True)
    gpt2_zip = tmp_path / "renamed-authoritative.zip"
    _create_dummy_zip(gpt2_zip, ["AI_Images/image_001.png"])

    out_dir = tmp_path / "out"
    prepare_hf_transparency_package(
        manifest_paths=[manifest_file],
        output_dir=out_dir,
        gpt2_zip=gpt2_zip,
        include_private_images=True,
        staged_images_dir=out_dir / "private_images",
        is_private=True,
        allow_partial_archive=True,
    )
    assert (
        out_dir
        / "private_images"
        / "gpt_image_2_user_screenshots"
        / "gpt_image_2_screenshot_001.jpg"
    ).is_file()
