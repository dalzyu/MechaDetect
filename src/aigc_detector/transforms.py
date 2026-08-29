from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from random import Random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from .constants import SEVERITY_VALUES, Transformation


@dataclass(frozen=True)
class TransformSpec:
    family: Transformation
    severity: float


def sample_transform(
    rng: Random, families: tuple[Transformation, ...] | None = None
) -> TransformSpec:
    candidates = families or tuple(Transformation)
    family = rng.choice(candidates)
    if family in SEVERITY_VALUES:
        severity = rng.choice(SEVERITY_VALUES[family])
    elif family is Transformation.COLOR:
        severity = 0.2
    elif family is Transformation.CROP:
        severity = 0.8
    else:  # pragma: no cover - enum exhaustiveness guard
        raise ValueError(f"Unsupported transformation family: {family}")
    return TransformSpec(family=family, severity=float(severity))


def apply_transform(
    image: Image.Image,
    spec: TransformSpec,
    rng: Random,
    *,
    mask: Image.Image | None = None,
    min_mask_retention: float = 0.5,
) -> Image.Image:
    """Apply one content-preserving transform.

    A localized edit must remain visible when crop augmentation is sampled;
    otherwise the positive label becomes contradictory.  The crop sampler
    therefore retries deterministic random windows and finally centers the
    window on the mask centroid.
    """
    image = image.convert("RGB")
    width, height = image.size

    if spec.family is Transformation.JPEG:
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=int(spec.severity), optimize=False)
        buffer.seek(0)
        with Image.open(buffer) as encoded:
            return encoded.convert("RGB").copy()

    if spec.family is Transformation.BLUR:
        return image.filter(ImageFilter.GaussianBlur(radius=spec.severity))

    if spec.family is Transformation.RESIZE:
        down_width = max(1, round(width * spec.severity))
        down_height = max(1, round(height * spec.severity))
        down = image.resize((down_width, down_height), Image.Resampling.BICUBIC)
        return down.resize((width, height), Image.Resampling.BICUBIC)

    if spec.family is Transformation.NOISE:
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        noise_rng = np.random.default_rng(rng.randrange(2**63))
        noisy = np.clip(pixels + noise_rng.normal(0.0, spec.severity, pixels.shape), 0.0, 1.0)
        return Image.fromarray(np.rint(noisy * 255.0).astype(np.uint8), mode="RGB")

    if spec.family is Transformation.COLOR:
        magnitude = spec.severity
        brightness = rng.uniform(1.0 - magnitude, 1.0 + magnitude)
        contrast = rng.uniform(1.0 - magnitude, 1.0 + magnitude)
        saturation = rng.uniform(1.0 - magnitude, 1.0 + magnitude)
        result = ImageEnhance.Brightness(image).enhance(brightness)
        result = ImageEnhance.Contrast(result).enhance(contrast)
        return ImageEnhance.Color(result).enhance(saturation)

    if spec.family is Transformation.CROP:
        keep = spec.severity
        crop_width = max(1, round(width * keep))
        crop_height = max(1, round(height * keep))
        crop_mask = None
        total_mask_area = 0
        if mask is not None:
            crop_mask = np.asarray(mask.convert("L"), dtype=np.uint8) > 0
            total_mask_area = int(crop_mask.sum())

        def retained(left: int, top: int) -> int:
            if crop_mask is None or total_mask_area == 0:
                return total_mask_area
            return int(crop_mask[top : top + crop_height, left : left + crop_width].sum())

        max_left = width - crop_width
        max_top = height - crop_height
        for _ in range(16):
            left = rng.randint(0, max_left) if max_left else 0
            top = rng.randint(0, max_top) if max_top else 0
            if total_mask_area == 0 or retained(left, top) / total_mask_area >= min_mask_retention:
                cropped = image.crop((left, top, left + crop_width, top + crop_height))
                return cropped.resize((width, height), Image.Resampling.BICUBIC)

        if total_mask_area:
            ys, xs = np.nonzero(crop_mask)
            center_x, center_y = int(xs.mean()), int(ys.mean())
            left = min(max(center_x - crop_width // 2, 0), max_left)
            top = min(max(center_y - crop_height // 2, 0), max_top)
        else:
            left = max_left // 2
            top = max_top // 2
        cropped = image.crop((left, top, left + crop_width, top + crop_height))
        return cropped.resize((width, height), Image.Resampling.BICUBIC)

    raise ValueError(f"Unsupported transformation family: {spec.family}")
