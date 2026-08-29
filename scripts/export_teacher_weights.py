#!/usr/bin/env python3
"""Export model-only, publishable teacher weights from a training checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

SECTIONS = (
    "heads",
    "token_adapter",
    "spectral",
    "aigc_gate",
    "tamper_gate",
    "encoder_trainable_state",
)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_weights(source: Path, destination: Path, dtype: torch.dtype) -> None:
    print(f"loading {source}", flush=True)
    payload: dict[str, Any] = torch.load(
        source,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    print("checkpoint loaded; extracting model tensors", flush=True)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a checkpoint mapping, got {type(payload).__name__}")

    tensors: dict[str, torch.Tensor] = {}
    for section in SECTIONS:
        values = payload.get(section)
        if not isinstance(values, dict):
            continue
        for name, value in values.items():
            if not isinstance(value, torch.Tensor):
                continue
            tensor = value.detach().cpu().contiguous()
            if tensor.is_floating_point():
                tensor = tensor.to(dtype=dtype)
            tensors[f"{section}.{name}"] = tensor
    print(f"extracted {len(tensors)} tensors; hashing source", flush=True)
    if not tensors:
        raise ValueError(f"No model tensors found in {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": "model-only teacher weights",
        "source_checkpoint": str(source),
        "source_sha256": sha256_file(source),
        "source_size_bytes": str(source.stat().st_size),
        "step": str(payload.get("step", "unknown")),
        "manifest_sha256": str(payload.get("manifest_sha256", "unknown")),
        "dtype": str(dtype).removeprefix("torch."),
        "tensor_count": str(len(tensors)),
    }
    tensor_bytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
    print(f"extracted_bytes={tensor_bytes}", flush=True)
    save_file(tensors, str(destination), metadata=metadata)

    sidecar = destination.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                **metadata,
                "source_checkpoint": str(source),
                "tensor_names": sorted(tensors),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"weights": str(destination), **metadata}, sort_keys=True))


def parse_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    args = parser.parse_args()
    export_weights(args.source, args.destination, parse_dtype(args.dtype))


if __name__ == "__main__":
    main()
