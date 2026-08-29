from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DATA_PREP_DIR = _PROJECT_ROOT / "scripts" / "data_prep"
if str(_DATA_PREP_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_PREP_DIR))

from generate_pathological_stress_set import (  # noqa: E402
    STRESS_TYPES,
    generate_pathological_stress_set,
)


def test_pathological_stress_set_is_deterministic_and_evaluation_only(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = generate_pathological_stress_set(first_dir, seed=7, size=32, samples_per_type=1)
    second = generate_pathological_stress_set(second_dir, seed=7, size=32, samples_per_type=1)

    assert {row["stress_type"] for row in first} == set(STRESS_TYPES)
    assert [row["sha256"] for row in first] == [row["sha256"] for row in second]
    assert (first_dir / "manifest.jsonl").is_file()

    manifest_rows = [
        json.loads(line) for line in (first_dir / "manifest.jsonl").read_text().splitlines()
    ]
    for row in manifest_rows:
        assert row["expected_behavior"] == "abstain_or_low_confidence"
        assert not ({"label", "ai_positive", "provenance"} & row.keys())
        image_path = first_dir / row["image_path"]
        assert hashlib.sha256(image_path.read_bytes()).hexdigest() == row["sha256"]
        with Image.open(image_path) as image:
            image.verify()
