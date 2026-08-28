from __future__ import annotations

from enum import StrEnum
from io import BytesIO
from math import log
from random import Random

import numpy as np
import torch
from PIL import Image
from torch import Tensor


class RenderPolicy(StrEnum):
    SQUARE_JPEG95 = "square_jpeg95"
    ASPECT_JPEG95 = "aspect_jpeg95"
    ASPECT_RANDOMIZED = "aspect_randomized"


def _encode_decode(image: Image.Image, format_name: str, **options: object) -> Image.Image:
    buffer = BytesIO()
    image.save(buffer, format=format_name, exif=b"", **options)
    buffer.seek(0)
    with Image.open(buffer) as encoded:
        return encoded.convert("RGB").copy()


def _aspect_resize(image: Image.Image, max_edge: int, resample: Image.Resampling) -> Image.Image:
    width, height = image.size
    scale = max_edge / max(width, height)
    return image.resize((max(1, round(width * scale)), max(1, round(height * scale))), resample)


def render_for_model(
    image: Image.Image,
    policy: RenderPolicy | str,
    *,
    rng: Random | None = None,
    size: int = 1584,
) -> Image.Image:
    policy = RenderPolicy(policy)
    rng = rng or Random(0)
    image = image.convert("RGB")
    if policy is RenderPolicy.SQUARE_JPEG95:
        rendered = image.resize((size, size), Image.Resampling.BICUBIC)
        return _encode_decode(rendered, "JPEG", quality=95, optimize=False, progressive=False)

    resampling = (
        rng.choice([Image.Resampling.BILINEAR, Image.Resampling.BICUBIC, Image.Resampling.LANCZOS])
        if policy is RenderPolicy.ASPECT_RANDOMIZED
        else Image.Resampling.BICUBIC
    )
    rendered = _aspect_resize(image, size, resampling)
    if policy is RenderPolicy.ASPECT_JPEG95:
        return _encode_decode(rendered, "JPEG", quality=95, optimize=False, progressive=False)
    draw = rng.random()
    if draw < 0.25:
        return _encode_decode(rendered, "PNG", optimize=False)
    if draw < 0.75:
        return _encode_decode(rendered, "JPEG", quality=rng.randint(75, 100), optimize=False)
    if draw < 0.875:
        return _encode_decode(rendered, "WEBP", quality=rng.randint(80, 100), method=4)
    return _encode_decode(rendered, "JPEG", quality=rng.randint(75, 100), optimize=True)


def canonicalize_for_model(
    image: Image.Image,
    *,
    size: int = 1584,
    jpeg_quality: int = 95,
) -> Image.Image:
    """Normalize geometry/encoding to match the controlled training derivative."""
    if jpeg_quality == 95:
        return render_for_model(image, RenderPolicy.SQUARE_JPEG95, size=size)
    normalized = image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    return _encode_decode(
        normalized, "JPEG", quality=jpeg_quality, optimize=False, progressive=False
    )


def infer_token_grid(token_count: int, aspect_ratio: float) -> tuple[int, int]:
    """Infer Gemma's pooled row-major token grid from count and rendered aspect."""
    candidates = []
    for height in range(1, int(token_count**0.5) + 1):
        if token_count % height == 0:
            width = token_count // height
            candidates.extend(((height, width), (width, height)))
    return min(candidates, key=lambda shape: abs(log((shape[1] / shape[0]) / aspect_ratio)))


def mask_to_token_occupancy(
    mask: Image.Image, token_count: int, image_size: tuple[int, int]
) -> Tensor:
    width, height = image_size
    grid_height, grid_width = infer_token_grid(token_count, width / height)
    grayscale = mask.convert("L").resize((grid_width, grid_height), Image.Resampling.BOX)
    array = np.asarray(grayscale, dtype=np.float32).flatten()
    values = torch.from_numpy(array).div_(255.0)
    if values.numel() != token_count:
        raise RuntimeError(f"Mask occupancy tokens ({values.numel()}) does not match expected token count ({token_count})")
    return values


def render_mask_geometry(
    mask: Image.Image,
    policy: RenderPolicy | str,
    *,
    image_size: tuple[int, int],
    rendered_size: tuple[int, int],
) -> Image.Image:
    """Apply the deterministic geometry used by a rendered image to a binary/soft mask."""
    policy = RenderPolicy(policy)
    if policy is RenderPolicy.SQUARE_JPEG95:
        return mask.convert("L").resize(rendered_size, Image.Resampling.BOX)
    expected_ratio = image_size[0] / image_size[1]
    actual_ratio = rendered_size[0] / rendered_size[1]
    if abs(expected_ratio - actual_ratio) > 0.02:
        raise ValueError("Rendered image geometry is inconsistent with the source mask")
    return mask.convert("L").resize(rendered_size, Image.Resampling.BOX)
