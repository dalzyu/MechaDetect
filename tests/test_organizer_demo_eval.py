from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from aigc_detector.constants import Transformation
from scripts.eval_organizer_demo import (
    ORGANIZER_CONDITIONS,
    ConditionSpec,
    OrganizerDemoDataset,
    evaluate_condition,
    collate_demo,
)
from aigc_detector.config import load_config
from aigc_detector.predict import _load_checkpoint
from aigc_detector.train import build_model
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEMO_MANIFEST = _PROJECT_ROOT / "metadata" / "organizer_demo_document_count.csv"


def test_organizer_demo_manifest_specification() -> None:
    """Verify demonstration manifest strictly adheres to TechJam info document specs:
    - Non-AIGC: COCO val2017 = 4,998 images
    - AIGC:     DALL-E Advanced = 8,843 images
    - Total:    13,841 images
    """
    assert _DEMO_MANIFEST.is_file(), f"Manifest missing at {_DEMO_MANIFEST}"
    df = pd.read_csv(_DEMO_MANIFEST, low_memory=False)

    coco_count = (df["label"] == "authentic").sum()
    dalle_count = (df["label"] == "fully_aigc").sum()

    assert coco_count == 4998, f"Expected 4998 COCO val2017 rows, got {coco_count}"
    assert dalle_count == 8843, f"Expected 8843 DALL-E Advanced rows, got {dalle_count}"
    assert len(df) == 13841, f"Expected 13841 total rows, got {len(df)}"


def test_organizer_robustness_transform_grid_coverage() -> None:
    """Verify all 6 categories and specific parameters from the TechJam info document
    are defined in the evaluation conditions.
    """
    condition_map = {c.name: c for c in ORGANIZER_CONDITIONS}

    # Baseline
    assert "clean" in condition_map
    assert condition_map["clean"].spec is None

    # 1. JPEG Compression (quality = 90, 70, 50, 30)
    for q in (90, 70, 50, 30):
        name = f"jpeg{q}"
        assert name in condition_map
        spec = condition_map[name].spec
        assert spec.family == Transformation.JPEG
        assert spec.severity == float(q)

    # 2. Gaussian Blur (kernel σ = 0.5, 1.0, 2.0)
    for s, name in [(0.5, "blur0.5"), (1.0, "blur1.0"), (2.0, "blur2.0")]:
        assert name in condition_map
        spec = condition_map[name].spec
        assert spec.family == Transformation.BLUR
        assert spec.severity == s

    # 3. Resize (scale 0.5x, 0.25x)
    assert "resize_half" in condition_map
    assert condition_map["resize_half"].spec.family == Transformation.RESIZE
    assert condition_map["resize_half"].spec.severity == 0.50

    assert "resize_quarter" in condition_map
    assert condition_map["resize_quarter"].spec.family == Transformation.RESIZE
    assert condition_map["resize_quarter"].spec.severity == 0.25

    # 4. Gaussian Noise (σ = 0.02, 0.05, 0.10)
    for s in (0.02, 0.05, 0.10):
        name = f"noise{s:.2f}"
        assert name in condition_map
        spec = condition_map[name].spec
        assert spec.family == Transformation.NOISE
        assert spec.severity == s

    # 5. Color Jitter (±20%)
    assert "color_jitter20" in condition_map
    assert condition_map["color_jitter20"].spec.family == Transformation.COLOR
    assert condition_map["color_jitter20"].spec.severity == 0.20

    # 6. Center Crop (80%)
    assert "crop80" in condition_map
    assert condition_map["crop80"].spec.family == Transformation.CROP
    assert condition_map["crop80"].spec.severity == 0.80

    # Total 15 conditions (1 clean + 14 augmented)
    assert len(ORGANIZER_CONDITIONS) == 15


@pytest.mark.skipif(
    not Path(os.environ.get("TECHJAM_DATA_ROOT", "E:/techjam26-runtime/data")).is_dir(),
    reason="Local data runtime directory not found",
)
def test_organizer_eval_smoke_end_to_end() -> None:
    """Smoke test: run evaluation over a small balanced subset (4 samples)
    under clean and a transformation, verifying metric computation and integrity.
    """
    data_root = Path(os.environ.get("TECHJAM_DATA_ROOT", "E:/techjam26-runtime/data"))
    config_path = _PROJECT_ROOT / "configs" / "teacher_dinov3_stage2_paired_unfrozen.yaml"
    ckpt_path = _PROJECT_ROOT / "models" / "teachers" / "iteration1" / "stage2" / "model-weights.safetensors"

    if not ckpt_path.is_file():
        pytest.skip(f"Teacher weights not found at {ckpt_path}")

    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(config).to(device).eval()
    _load_checkpoint(model, ckpt_path)

    # Test clean and one transformed condition on 4 samples (2 authentic, 2 AIGC)
    for cond_name in ["clean", "jpeg50", "blur1.0"]:
        cond_spec = next(c for c in ORGANIZER_CONDITIONS if c.name == cond_name)
        ds = OrganizerDemoDataset(
            manifest_path=_DEMO_MANIFEST,
            data_root=data_root,
            condition=cond_spec,
            limit_per_class=2,
            seed=42,
        )
        assert len(ds) == 4

        loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_demo)
        metrics = evaluate_condition(model, loader, device, precision="bf16" if device.type == "cuda" else "fp32")

        assert metrics["n_samples"] == 4
        assert metrics["n_aigc"] == 2
        assert metrics["n_non_aigc"] == 2
        assert 0.0 <= metrics["auroc"] <= 1.0
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["aigc_recall"] <= 1.0
        assert 0.0 <= metrics["non_aigc_recall"] <= 1.0
        assert 0.0 <= metrics["macro_f1"] <= 1.0
