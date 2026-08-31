"""Tests for the balanced GPT Image 2 Stage 2 patch."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "data_prep"))

from curate_gpt_image2_cohort import (  # noqa: E402
    MANIFEST_COLUMNS,
    build_balanced_manifest,
    write_manifest,
)


def _positive_records(count: int = 2_000) -> list[dict[str, object]]:
    return [
        {
            "id": f"GI2_{index:05d}",
            "file_name": f"gpt-image-2/images/{index % 4}/GI2_{index:05d}_0.jpg",
            "model_info": {"name": "gpt-image-2", "version": "1.0"},
            "spec": {"width": 1024, "height": 1024},
            "raw_p": f"fixture prompt {index}",
        }
        for index in range(count)
    ]


def _authentic_records(count: int = 2_000) -> list[dict[str, object]]:
    return [
        {
            "image_path": f"open_images_v7/open_images_v7_{index:06d}.jpg",
            "label": 0,
            "dataset": "open_images_v7",
            "generator": "authentic",
            "generator_family": "authentic",
            "file_format": "JPEG",
            "width": 640,
            "height": 480,
            "ai_positive": 0,
        }
        for index in range(count)
    ]


def test_manifest_is_exactly_balanced() -> None:
    rows = build_balanced_manifest(_positive_records(), _authentic_records(), seed=7)
    assert len(rows) == 4_000
    assert sum(row["ai_positive"] == 1 for row in rows) == 2_000
    assert sum(row["ai_positive"] == 0 for row in rows) == 2_000
    assert len({row["image_path"] for row in rows}) == 4_000


def test_manifest_has_standard_schema_and_valid_generator_metadata(tmp_path: Path) -> None:
    rows = build_balanced_manifest(_positive_records(), _authentic_records(), seed=11)
    output = tmp_path / "gpt_image_2_balanced_patch.jsonl"
    write_manifest(rows, output)
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert records
    assert set(MANIFEST_COLUMNS).issubset(records[0])
    expected = {"image_path", "label", "dataset", "official_split", "generator", "ai_positive"}
    assert expected.issubset(records[0])
    positives = [row for row in records if row["ai_positive"] == 1]
    negatives = [row for row in records if row["ai_positive"] == 0]
    assert all(
        row["label"] == "fully_aigc"
        and row["dataset"] == "gpt_image_2_twitter"
        and row["generator"] == "gpt_image_2"
        and row["generator_family"] == "gpt_image_2"
        for row in positives
    )
    assert all(
        row["label"] == "authentic"
        and row["generator"] == "authentic"
        and row["generator_family"] == "authentic"
        for row in negatives
    )


def test_positive_source_rows_are_filtered_and_deduplicated() -> None:
    positive = _positive_records(3)
    positive.append(dict(positive[0]))
    positive.append({"id": "other", "file_name": "other.jpg", "model_info": {"name": "dall-e-3"}})
    rows = build_balanced_manifest(positive, _authentic_records(3), positive_count=3, negative_count=3)
    assert [row["ai_positive"] for row in rows] == [1, 1, 1, 0, 0, 0]
    assert {row["generator"] for row in rows[:3]} == {"gpt_image_2"}
