"""Focused tests for the strict final teacher dataset builder."""
from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "data_prep"))

from build_final_teacher_dataset import (  # noqa: E402
    apply_domain_caps,
    assign_group_disjoint_splits,
    import_authoritative_gpt_image_2_zip,
    ingest_candidate_manifests,
    is_safe_relative_path,
)


def _record(
    *,
    name: str,
    label: int,
    ai_positive: int,
    group: str,
    quality: float,
    generator: str = "authentic",
    provenance: str = "authentic",
) -> dict[str, object]:
    return {
        "image_path": f"{name}.jpg",
        "label": label,
        "dataset": "fixture",
        "official_split": "train",
        "generator": generator,
        "manipulation_family": "",
        "source_image_group": group,
        "width": 64,
        "height": 64,
        "file_format": "JPEG",
        "tamper_mask_path": "",
        "source_url": "fixture",
        "external_id": name,
        "generator_family": generator,
        "generator_version": "",
        "prompt": "",
        "created_at": "",
        "sha256": (name * 64)[:64],
        "perceptual_hash": "0",
        "quality_score": quality,
        "provenance_confidence": "high",
        "redistribution_mode": "embed_bytes",
        "origin_license": "fixture",
        "license_url": "",
        "attribution": "fixture",
        "selection_reason": "fixture",
        "forbidden_demo_checked": True,
        "ai_positive": ai_positive,
        "split": "train",
        "duplicate_group": "",
        "provenance": provenance,
        "domain": "fixture",
        "cohort": "fixture",
    }


def test_path_safety_rejects_absolute_and_traversal_paths() -> None:
    assert is_safe_relative_path("source/image.jpg")
    assert not is_safe_relative_path("../outside.jpg")
    assert not is_safe_relative_path("C:/outside.jpg")
    assert not is_safe_relative_path("/outside.jpg")


def test_quality_caps_are_deterministic_and_keep_source_groups() -> None:
    records = [
        _record(name="low_a", label=0, ai_positive=0, group="low", quality=0.70),
        _record(name="low_b", label=0, ai_positive=0, group="low", quality=0.70),
        _record(name="high_a", label=0, ai_positive=0, group="high", quality=0.99),
        _record(name="high_b", label=0, ai_positive=0, group="high", quality=0.99),
        _record(name="middle", label=0, ai_positive=0, group="middle", quality=0.80),
    ]
    first = apply_domain_caps(records, {"fixture": 2}, seed=7)
    second = apply_domain_caps(list(reversed(records)), {"fixture": 2}, seed=7)
    assert [row["image_path"] for row in first] == [row["image_path"] for row in second]
    assert {row["source_image_group"] for row in first} == {"high"}


def test_split_quarantines_conflicting_binary_labels_in_one_duplicate_group() -> None:
    rows = [
        _record(name="negative", label=0, ai_positive=0, group="same", quality=0.9),
        _record(
            name="positive",
            label=2,
            ai_positive=1,
            group="same",
            quality=0.9,
            generator="gpt_image_2",
            provenance="fully_aigc",
        ),
    ]
    frame = assign_group_disjoint_splits(rows, seed=42)
    assert frame.empty
    assert frame.attrs["quarantined_conflicting_duplicate_groups"] == 1
    assert frame.attrs["quarantined_conflicting_rows"] == 2


def test_authoritative_zip_rejects_wrong_image_count(tmp_path: Path) -> None:
    archive = tmp_path / "wrong.zip"
    payload = io.BytesIO()
    Image.new("RGB", (32, 32), (20, 30, 40)).save(payload, format="PNG")
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("AI_Images/image_001.png", payload.getvalue())

    with pytest.raises(ValueError, match="exactly 92"):
        import_authoritative_gpt_image_2_zip(
            archive,
            tmp_path / "images",
            dry_run=True,
        )


def test_ingestion_resolves_relative_paths_across_multiple_roots(tmp_path: Path) -> None:
    root_a = tmp_path / "root_a"
    root_b = tmp_path / "root_b"
    root_a.mkdir()
    root_b.mkdir()
    rows = []
    for root, name in ((root_a, "a.jpg"), (root_b, "b.jpg")):
        image = Image.effect_noise((32, 32), 80)
        image.save(root / name, format="JPEG", quality=95)
        rows.append(
            {
                "dataset": "fixture",
                "image_path": name,
                "label": 0,
                "ai_positive": 0,
                "provenance": "authentic",
                "generator": "authentic",
                "generator_family": "authentic",
                "source_image_group": name,
            }
        )

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    records, report = ingest_candidate_manifests(
        [manifest],
        [root_a, root_b],
        min_quality=0.70,
    )

    assert report.total_raw_rows == 2
    assert report.usable_rows_count == 2
    assert {record["image_path"] for record in records} == {"a.jpg", "b.jpg"}
