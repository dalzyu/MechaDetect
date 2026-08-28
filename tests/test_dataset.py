from pathlib import Path

import pandas as pd
from PIL import Image

from aigc_detector.constants import Provenance
from aigc_detector.dataset import PairedImageDataset, collate_pairs, parse_provenance


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
        chain_length_probabilities={2: 1.0},
    )
    sample = dataset[0]
    assert sample["provenance"] == int(Provenance.TAMPERED)
    assert sample["original"].size == sample["transformed"].size
    assert len(sample["transform_chain"]) == 2
    assert sample["generator"] == "sid_set_unknown"

    batch = collate_pairs([sample])
    assert batch["provenance"].shape == (1,)
    assert batch["mask"] == [None]
