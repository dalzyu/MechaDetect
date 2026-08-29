from random import Random

import pytest
from PIL import Image, ImageChops

from aigc_detector.constants import Transformation
from aigc_detector.preprocessing import infer_token_grid, mask_to_token_occupancy
from aigc_detector.transforms import (
    TransformSpec,
    apply_transform,
    sample_transform,
)


@pytest.mark.parametrize(
    ("family", "severity"),
    [
        (Transformation.JPEG, 50.0),
        (Transformation.BLUR, 1.0),
        (Transformation.RESIZE, 0.5),
        (Transformation.NOISE, 0.05),
        (Transformation.COLOR, 0.2),
        (Transformation.CROP, 0.8),
    ],
)
def test_transform_preserves_shape_and_changes_pixels(family, severity) -> None:
    image = Image.linear_gradient("L").resize((64, 48)).convert("RGB")
    transformed = apply_transform(image, TransformSpec(family, severity), Random(42))
    assert transformed.mode == "RGB"
    assert transformed.size == image.size
    assert ImageChops.difference(image, transformed).getbbox() is not None


def test_sample_transform_selects_one_family() -> None:
    transform = sample_transform(Random(42))
    assert transform.family in tuple(Transformation)


def test_mask_occupancy_matches_square_gemma_grid() -> None:
    mask = Image.new("L", (99, 99), 0)
    for y in range(50):
        for x in range(99):
            mask.putpixel((x, y), 255)
    occupancy = mask_to_token_occupancy(mask, 1089, (1584, 1584))
    assert infer_token_grid(1089, 1.0) == (33, 33)
    assert occupancy.shape == (1089,)
    assert 0.45 < occupancy.mean().item() < 0.55
