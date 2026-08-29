#!/usr/bin/env python3
"""Generate deterministic evaluation-only pathological image inputs.

These content-free procedural images have no defensible binary provenance label.
They are intentionally excluded from supervised manifests and exist to expose
unsafe model confidence on blank, near-blank, noise, and aliasing inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

STRESS_TYPES = (
    "black",
    "white",
    "mid_gray",
    "near_flat",
    "gaussian_noise",
    "salt_pepper_noise",
    "checkerboard",
    "gradient",
    "alpha_edge",
)
EXPECTED_BEHAVIOR = "abstain_or_low_confidence"


def _stress_array(stress_type: str, size: int, rng: np.random.Generator) -> np.ndarray:
    shape = (size, size, 3)
    if stress_type == "black":
        return np.zeros(shape, dtype=np.uint8)
    if stress_type == "white":
        return np.full(shape, 255, dtype=np.uint8)
    if stress_type == "mid_gray":
        return np.full(shape, 128, dtype=np.uint8)
    if stress_type == "near_flat":
        return np.clip(128 + rng.integers(-2, 3, size=shape), 0, 255).astype(np.uint8)
    if stress_type == "gaussian_noise":
        return np.clip(rng.normal(127.5, 48.0, size=shape), 0, 255).astype(np.uint8)
    if stress_type == "salt_pepper_noise":
        values = rng.choice(np.array([0, 255], dtype=np.uint8), size=(size, size, 1))
        return np.repeat(values, 3, axis=2)
    if stress_type == "checkerboard":
        yy, xx = np.indices((size, size))
        values = (((xx // 2) + (yy // 2)) % 2 * 255).astype(np.uint8)
        return np.repeat(values[:, :, None], 3, axis=2)
    if stress_type == "gradient":
        x = np.linspace(0, 255, size, dtype=np.uint8)
        values = np.broadcast_to(x, (size, size))
        return np.stack((values, np.flipud(values.T), np.fliplr(values)), axis=2)
    if stress_type == "alpha_edge":
        rgb = np.full(shape, 255, dtype=np.uint8)
        alpha = np.zeros((size, size, 1), dtype=np.uint8)
        alpha[:, size // 2 :] = 255
        return np.concatenate((rgb, alpha), axis=2)
    raise ValueError(f"Unsupported stress type: {stress_type}")


def generate_pathological_stress_set(
    output_dir: Path,
    *,
    seed: int = 42,
    size: int = 224,
    samples_per_type: int = 4,
) -> list[dict[str, object]]:
    if size < 16:
        raise ValueError("size must be at least 16 pixels")
    if samples_per_type < 1:
        raise ValueError("samples_per_type must be positive")

    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for type_index, stress_type in enumerate(STRESS_TYPES):
        for sample_index in range(samples_per_type):
            sample_seed = seed + type_index * 1_000_003 + sample_index
            rng = np.random.default_rng(sample_seed)
            array = _stress_array(stress_type, size, rng)
            mode = "RGBA" if array.shape[2] == 4 else "RGB"
            relative_path = Path("images") / f"{stress_type}_{sample_index:03d}.png"
            destination = output_dir / relative_path
            temporary = destination.with_name(
                f".{destination.name}.tmp_{os.getpid()}_{uuid.uuid4().hex[:8]}"
            )
            try:
                Image.fromarray(array, mode=mode).save(temporary, format="PNG", optimize=False)
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
            payload = destination.read_bytes()
            rows.append(
                {
                    "image_path": relative_path.as_posix(),
                    "stress_type": stress_type,
                    "width": size,
                    "height": size,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "seed": sample_seed,
                    "expected_behavior": EXPECTED_BEHAVIOR,
                }
            )

    manifest = output_dir / "manifest.jsonl"
    manifest_payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
    ).encode("utf-8")
    temporary_manifest = manifest.with_name(
        f".{manifest.name}.tmp_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    )
    try:
        temporary_manifest.write_bytes(manifest_payload)
        os.replace(temporary_manifest, manifest)
    finally:
        temporary_manifest.unlink(missing_ok=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--samples-per-type", type=int, default=4)
    args = parser.parse_args()
    rows = generate_pathological_stress_set(
        args.output_dir,
        seed=args.seed,
        size=args.size,
        samples_per_type=args.samples_per_type,
    )
    print(f"Generated {len(rows)} evaluation-only pathological inputs in {args.output_dir}")


if __name__ == "__main__":
    main()
