from pathlib import Path

import pandas as pd
from PIL import Image

from aigc_detector.constants import Provenance, Transformation
from aigc_detector.dataset import (
    PairedImageDataset,
    collate_pairs,
    parse_provenance,
    verify_materialization,
)


def test_sid_label_mapping() -> None:
    assert parse_provenance(0, "SID-Set") is Provenance.AUTHENTIC
    assert parse_provenance(1, "SID-Set") is Provenance.FULLY_AIGC
    assert parse_provenance(2, "SID-Set") is Provenance.TAMPERED


def test_paired_dataset_and_collation(tmp_path: Path) -> None:
    Image.new("RGB", (32, 24), color=(50, 100, 150)).save(tmp_path / "image.png")
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([{"image_path": "image.png", "label": 2, "dataset": "SID-Set"}]).to_csv(
        manifest, index=False
    )

    dataset = PairedImageDataset(
        manifest,
        data_root=tmp_path,
        transform_families=(Transformation.JPEG,),
    )
    sample = dataset[0]
    assert sample["provenance"] == int(Provenance.TAMPERED)
    assert sample["original"].size == sample["transformed"].size
    assert sample["transform"] == "jpeg"
    assert sample["generator"] == "sid_set_unknown"

    batch = collate_pairs([sample])
    assert batch["provenance"].shape == (1,)
    assert batch["mask"] == [None]

def test_verify_materialization_allow_missing(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([
        {"image_path": "missing.jpg", "label": 0, "dataset": "SID-Set"}
    ]).to_csv(manifest, index=False)

    # Fail closed by default
    import pytest
    with pytest.raises(FileNotFoundError):
        verify_materialization(manifest, data_root=tmp_path)

    # Permitted when allow_missing=True
    result = verify_materialization(manifest, data_root=tmp_path, allow_missing=True)
    assert result["rows"] == 1
    assert result["missing_images"] == 1


def test_paired_dataset_allow_missing_fallback(tmp_path: Path) -> None:
    # Create one materialized authentic image
    Image.new("RGB", (32, 24), color=(10, 20, 30)).save(tmp_path / "real.jpg")
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([
        {"image_path": "real.jpg", "label": 0, "dataset": "SID-Set", "ai_positive": 0},
        {"image_path": "missing.jpg", "label": 0, "dataset": "SID-Set", "ai_positive": 0},
    ]).to_csv(manifest, index=False)

    dataset = PairedImageDataset(
        manifest,
        data_root=tmp_path,
        allow_missing=True,
        runtime_fetch=False,
    )
    assert len(dataset) == 2
    sample0 = dataset[0]
    sample1 = dataset[1]
    assert sample0["ai_positive"] == 0
    assert sample1["ai_positive"] == 0
    assert sample0["original"].size == sample1["original"].size

