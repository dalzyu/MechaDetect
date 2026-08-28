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


_CHAIN_ORDER = {
    Transformation.CROP: 0,
    Transformation.RESIZE: 1,
    Transformation.COLOR: 2,
    Transformation.BLUR: 3,
    Transformation.NOISE: 4,
    Transformation.JPEG: 5,
}


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


def sample_transform_chain(
    rng: Random,
    chain_length_probabilities: dict[int, float],
    families: tuple[Transformation, ...] | None = None,
) -> tuple[TransformSpec, ...]:
    candidates = families or tuple(Transformation)
    lengths = tuple(sorted(chain_length_probabilities))
    weights = tuple(chain_length_probabilities[length] for length in lengths)
    length = rng.choices(lengths, weights=weights, k=1)[0]
    if not 0 <= length <= len(candidates):
        raise ValueError(f"Invalid transform-chain length {length}")
    if length == 0:
        return ()
    selected_families = rng.sample(list(candidates), k=length)
    specs = tuple(sample_transform(rng, (family,)) for family in selected_families)
    return tuple(sorted(specs, key=lambda spec: _CHAIN_ORDER[spec.family]))


def apply_transform(image: Image.Image, spec: TransformSpec, rng: Random) -> Image.Image:
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
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        cropped = image.crop((left, top, left + crop_width, top + crop_height))
        return cropped.resize((width, height), Image.Resampling.BICUBIC)

    raise ValueError(f"Unsupported transformation family: {spec.family}")


def apply_transform_chain(
    image: Image.Image, specs: tuple[TransformSpec, ...], rng: Random
) -> Image.Image:
    result = image
    for spec in specs:
        result = apply_transform(result, spec, rng)
    return result
