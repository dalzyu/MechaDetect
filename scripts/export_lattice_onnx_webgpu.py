#!/usr/bin/env python3
"""Export the two MechaDetect Lattice checkpoints for browser inference.

The Lattice checkpoints contain the DINOv3 ViT-H+/16 encoder (872,606,180
parameters) plus an optional image-domain spectral expert.  The browser
contract is the already-supported ``ProvenanceModel.forward_tensor`` path:
``pixel_values`` -> DINO tokens -> token adapter -> provenance head -> binary
probabilities.  ``forward_tensor`` intentionally does not consume the PIL
image sequence needed by ``SpectralExpert.forward``, so this exporter marks the
spectral branch as omitted rather than claiming full-model parity.

PyTorch's dynamo ONNX exporter is used because the model is larger than the
2-GiB protobuf limit.  It first writes one temporary external-data file, then
rewrites the ONNX external-data records into deterministic <=1-GB shards.  The
shards are relocatable with the protobuf and are suitable for ORT Web's
``externalData`` descriptors.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable

# Resolve the repository and the vendored ONNX/ORT packages before importing
# project modules.  The runtime loader supplies credentials/cache locations;
# no credential value is ever logged or written by this script.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
_ONNX_PACKAGE_DIR = REPO_ROOT / ".runtime" / "onnx_packages"
if _ONNX_PACKAGE_DIR.exists() and str(_ONNX_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_ONNX_PACKAGE_DIR))

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx import TensorProto
from PIL import Image

from aigc_detector.predict import _load_checkpoint
from aigc_detector.runtime import load_local_environment
from aigc_detector.train import build_model

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

OUTPUT_DIR = REPO_ROOT / "outputs" / "models"
WEB_MODEL_DIR = REPO_ROOT / "web" / "model"
DEFAULT_SHARD_BYTES = 1_000_000_000
EXPORT_OPSET = 18
EXPECTED_PARAMETER_COUNT = 872_606_180
EXPECTED_ENCODER_ID = "facebook/dinov3-vith16plus-pretrain-lvd1689m"
EXPECTED_ENCODER_REVISION = "c807c9eeea853df70aec4069e6f56b28ddc82acc"
HF_MODEL_REPO = "zye2/mechadetect-models"


def _hf_resolve_url(repo_id: str, revision: str, filename: str) -> str:
    return f"https://huggingface.co/{repo_id}/resolve/{revision}/models/{filename}"
EXPECTED_MANIFEST_DIGESTS = {
    "normal": "2d3a0c29aa405f5680dbe0d86ae463e18b92d4f08f23159a79ba59cb6fd9e729",
    "super": "f99f12b2c5610dc64be513e3adef353cac5ce7e5b8b5f634e885aebcea861e9e",
}
DEFAULT_CHECKPOINTS = {
    "normal": Path(r"C:/techjam26-exfil/checkpoints/mechadetect-lattice-checkpoint-final.pt"),
    "super": Path(r"C:/techjam26-exfil/checkpoints/mechadetect-lattice-super-checkpoint-final.pt"),
}

# The spectral module is retained and loaded into the PyTorch model, but is
# not reachable from forward_tensor.  This is an explicit experimental
# disposition, not a silent omission.
SPECTRAL_DISPOSITION = {
    "spectral_expert_omitted": True,
    "spectral_export_status": "omitted_from_browser_graph",
    "spectral_export_reason": (
        "SpectralExpert.forward consumes a Sequence[PIL.Image] and performs image conversion, "
        "fixed residual convolutions, torch.fft.rfft2/fftshift, radial-bin construction, and "
        "ConvNeXt fusion. The supported tensor browser path is ProvenanceModel.forward_tensor, "
        "which accepts only pixel_values and intentionally excludes this branch."
    ),
    "parity_scope": "ProvenanceModel.forward_tensor (spectral-free browser path)",
}


class LatticeWebGPUExportWrapper(nn.Module):
    """Expose the exact spectral-free tensor path as a browser contract."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Keep this expression in lockstep with the existing student exporter:
        # the model emits the AI-positive logit and the browser receives
        # [authentic_probability, aigc_probability].
        ai_positive_logit = self.model.forward_tensor(pixel_values)
        p_aigc = torch.sigmoid(ai_positive_logit)
        p_auth = 1.0 - p_aigc
        return torch.stack((p_auth, p_aigc), dim=-1)


class _DynamicMultiheadAttention(nn.Module):
    """Batch-dynamic equivalent of the encoder's eval-time cross attention.

    ``torch.export`` in the pinned PyTorch release specializes
    ``nn.MultiheadAttention`` when its learned query is expanded from a
    parameter.  The resulting ONNX graph accepts only the example batch.
    This implementation keeps the exact packed Q/K/V and output projection
    weights, while expressing the attention with ordinary matmul/softmax
    operations that retain the batch symbol.
    """

    def __init__(self, attention: nn.MultiheadAttention) -> None:
        super().__init__()
        if attention.in_proj_weight is None or attention.in_proj_bias is None:
            raise ValueError("Lattice export requires packed MultiheadAttention projections")
        if attention.bias_k is not None or attention.bias_v is not None or attention.add_zero_attn:
            raise ValueError("Unsupported MultiheadAttention options for Lattice export")
        self.embed_dim = attention.embed_dim
        self.num_heads = attention.num_heads
        self.head_dim = attention.head_dim
        self.in_proj_weight = attention.in_proj_weight
        self.in_proj_bias = attention.in_proj_bias
        self.out_proj = attention.out_proj

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, None]:
        if need_weights:
            raise ValueError("Lattice export attention does not return attention weights")
        query_length = query.shape[1]
        key_length = key.shape[1]
        q = F.linear(
            query,
            self.in_proj_weight[: self.embed_dim],
            self.in_proj_bias[: self.embed_dim],
        )
        k = F.linear(
            key,
            self.in_proj_weight[self.embed_dim : 2 * self.embed_dim],
            self.in_proj_bias[self.embed_dim : 2 * self.embed_dim],
        )
        v = F.linear(
            value,
            self.in_proj_weight[2 * self.embed_dim :],
            self.in_proj_bias[2 * self.embed_dim :],
        )
        q = q.reshape(query.shape[0], query_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(key.shape[0], key_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(value.shape[0], key_length, self.num_heads, self.head_dim).transpose(1, 2)
        attention_weights = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim**-0.5)
        attention_weights = attention_weights.softmax(dim=-1)
        pooled = torch.matmul(attention_weights, v).transpose(1, 2)
        pooled = pooled.reshape(query.shape[0], query_length, self.embed_dim)
        return self.out_proj(pooled), None


class _DynamicLearnedQueryPool(nn.Module):
    """LearnedQueryPool variant whose query batch is expressed by broadcast."""

    def __init__(self, pool: nn.Module) -> None:
        super().__init__()
        self.queries = pool.queries
        self.attention = _DynamicMultiheadAttention(pool.attention)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() == 2:
            query = self.queries.to(dtype=tokens.dtype).unsqueeze(0)
            tokens_batch = tokens.unsqueeze(0)
            pooled, _ = self.attention(query, tokens_batch, tokens_batch, need_weights=False)
            return pooled.squeeze(0)
        # Adding a zero-shaped slice makes the batch dimension data-dependent;
        # unlike expand(batch, ...), torch.export keeps it symbolic.
        query = self.queries.to(dtype=tokens.dtype).unsqueeze(0) + tokens[:, :1, :] * 0.0
        pooled, _ = self.attention(query, tokens, tokens, need_weights=False)
        return pooled


class _DynamicDINOv3Embeddings(nn.Module):
    """DINOv3 embeddings with symbolic batch broadcasts for prefix tokens."""

    def __init__(self, embeddings: nn.Module) -> None:
        super().__init__()
        self.config = embeddings.config
        self.cls_token = embeddings.cls_token
        self.mask_token = embeddings.mask_token
        self.register_tokens = embeddings.register_tokens
        self.patch_embeddings = embeddings.patch_embeddings

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        target_dtype = self.patch_embeddings.weight.dtype
        patch_embeddings = self.patch_embeddings(pixel_values.to(dtype=target_dtype))
        patch_embeddings = patch_embeddings.flatten(2).transpose(1, 2)
        if bool_masked_pos is not None:
            mask_token = self.mask_token.to(patch_embeddings.dtype)
            patch_embeddings = torch.where(bool_masked_pos.unsqueeze(-1), mask_token, patch_embeddings)
        # The zero terms are mathematically neutral, but keep the prefix
        # tensors batch-shaped for torch.export/ONNX rather than specializing
        # Parameter.expand(batch_size, ... ) to the example batch.
        cls_token = self.cls_token + patch_embeddings[:, :1, :] * 0.0
        register_tokens = self.register_tokens + (
            patch_embeddings[:, : self.config.num_register_tokens, :] * 0.0
        )
        return torch.cat([cls_token, register_tokens, patch_embeddings], dim=1)


def _prepare_dynamic_export_model(model: nn.Module) -> nn.Module:
    """Replace only exporter-sensitive broadcasts; weights and forward scope stay unchanged."""

    encoder = model.backbone.encoder
    encoder.embeddings = _DynamicDINOv3Embeddings(encoder.embeddings)
    model.heads.aigc_queries = _DynamicLearnedQueryPool(model.heads.aigc_queries)
    model.heads.tamper_query = _DynamicLearnedQueryPool(model.heads.tamper_query)
    return model


def compute_file_sha256(path: Path) -> str:
    """Hash a file incrementally without retaining model data in memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_checkpoint_metadata(checkpoint_path: Path) -> tuple[dict[str, Any], int, str]:
    """Read checkpoint config/count/digest using meta tensors without loading weights."""

    # map_location='meta' parses payload metadata while avoiding a second
    # multi-gigabyte allocation before the real model is constructed.
    payload = torch.load(checkpoint_path, map_location="meta", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        raise ValueError(f"Checkpoint {checkpoint_path} has no embedded config dictionary")
    digest = str(payload.get("manifest_digest") or payload.get("manifest_sha256") or "")
    return payload["config"], int(payload.get("parameter_count", -1)), digest




def inspect_lattice_checkpoint(checkpoint_path: Path, variant: str) -> dict[str, Any]:
    """Validate the embedded architecture/config before allocating the model."""

    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Lattice checkpoint not found: {checkpoint_path}")
    config, checkpoint_parameter_count, digest = _read_checkpoint_metadata(checkpoint_path)
    model_config = config.get("model")
    if not isinstance(model_config, dict):
        raise ValueError(f"Checkpoint {checkpoint_path} config has no model section")

    expected = {
        "backbone_type": "dinov3",
        "encoder_id": EXPECTED_ENCODER_ID,
        "encoder_revision": EXPECTED_ENCODER_REVISION,
        "image_size": 224,
        "encoder_dim": 1280,
        "trunk_dim": 512,
        "branch_dim": 256,
        "use_token_adapter": True,
        "spectral_expert": True,
    }
    mismatches = {
        key: {"expected": value, "actual": model_config.get(key)}
        for key, value in expected.items()
        if model_config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Lattice {variant} config mismatch: {json.dumps(mismatches, sort_keys=True)}")

    expected_digest = EXPECTED_MANIFEST_DIGESTS[variant]
    if digest != expected_digest:
        raise ValueError(
            f"Lattice {variant} manifest digest mismatch: expected {expected_digest}, got {digest or '<empty>'}"
        )

    # The checkpoint stores this count as a useful independent architecture
    # assertion; the instantiated model count is checked again after loading.
    if checkpoint_parameter_count != EXPECTED_PARAMETER_COUNT:
        raise ValueError(
            f"Lattice {variant} checkpoint parameter_count={checkpoint_parameter_count}, "
            f"expected {EXPECTED_PARAMETER_COUNT}"
        )

    return {
        "config": config,
        "model_config": model_config,
        "manifest_digest": digest,
        "checkpoint_parameter_count": checkpoint_parameter_count,
    }


def _configure_cached_hf_assets() -> None:
    """Avoid metadata writes when the configured H+ cache is already complete."""

    hf_home = os.environ.get("HF_HOME") or os.environ.get("TECHJAM_HF_HOME")
    if not hf_home:
        return
    cache_roots = [Path(hf_home) / "hub", Path(hf_home)]
    for cache_root in cache_roots:
        snapshot = (
            cache_root
            / "models--facebook--dinov3-vith16plus-pretrain-lvd1689m"
            / "snapshots"
            / EXPECTED_ENCODER_REVISION
        )
        if all(
            (snapshot / filename).is_file()
            for filename in ("config.json", "preprocessor_config.json", "model.safetensors")
        ):
            # DINOv3VisionBackbone passes HF_HOME as transformers' cache_dir,
            # so point it at the actual hub root (not HF_HOME/hub/hub).
            os.environ["HF_HOME"] = str(cache_root)
            # HF_HUB_OFFLINE prevents a full cache from trying to create
            # .no_exist entries on a volume with no remaining free space.
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            return


def load_lattice_model(checkpoint_path: Path, variant: str) -> tuple[nn.Module, dict[str, Any]]:
    """Build the exact embedded architecture and load the existing checkpoint loader."""

    load_local_environment(REPO_ROOT)
    _configure_cached_hf_assets()
    inspection = inspect_lattice_checkpoint(checkpoint_path, variant)
    config = inspection["config"]
    LOGGER.info("Building DINOv3 ViT-H+/16 Lattice %s model", variant)
    model = build_model(config)
    LOGGER.info("Loading checkpoint/EMA weights from %s", checkpoint_path)
    _load_checkpoint(model, checkpoint_path)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    actual_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if actual_parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            f"Instantiated Lattice {variant} parameter count {actual_parameter_count} "
            f"does not match expected {EXPECTED_PARAMETER_COUNT}"
        )
    return model, inspection


def _external_entry(tensor: TensorProto) -> dict[str, str]:
    values = {entry.key: entry.value for entry in tensor.external_data}
    if "location" not in values or "length" not in values:
        raise ValueError(f"External initializer {tensor.name!r} lacks location/length metadata")
    return values


def _all_external_tensors(model: onnx.ModelProto) -> list[TensorProto]:
    """Return external graph initializers, including any nested function tensors."""

    tensors: list[TensorProto] = []
    tensors.extend(model.graph.initializer)
    for graph in model.functions:
        # FunctionProto does not normally carry initializers, but keeping this
        # branch makes layout validation explicit if the exporter changes.
        tensors.extend(getattr(graph, "initializer", []))
    return [tensor for tensor in tensors if tensor.data_location == TensorProto.EXTERNAL]


def _set_external_location(tensor: TensorProto, location: str, offset: int, length: int) -> None:
    del tensor.external_data[:]
    tensor.data_location = TensorProto.EXTERNAL
    for key, value in (("location", location), ("offset", offset), ("length", length)):
        entry = tensor.external_data.add()
        entry.key = key
        entry.value = str(value)


def _copy_range(source: Any, destination: Any, offset: int, length: int) -> None:
    source.seek(offset)
    remaining = length
    while remaining:
        chunk = source.read(min(8 * 1024 * 1024, remaining))
        if not chunk:
            raise IOError(f"Unexpected end of external-data source at offset {offset}")
        destination.write(chunk)
        remaining -= len(chunk)


def split_external_data(
    onnx_path: Path,
    *,
    max_shard_bytes: int = DEFAULT_SHARD_BYTES,
) -> list[Path]:
    """Split PyTorch's temporary external file into deterministic relocatable shards."""

    if max_shard_bytes <= 0:
        raise ValueError("max_shard_bytes must be positive")
    onnx_path = Path(onnx_path)
    model = onnx.load_model(str(onnx_path), load_external_data=False)
    tensors = _all_external_tensors(model)
    if not tensors:
        raise RuntimeError(f"Exported model {onnx_path} contains no external initializers")

    source_locations = {_external_entry(tensor)["location"] for tensor in tensors}
    if len(source_locations) != 1:
        raise RuntimeError(f"Expected one temporary external file, found {sorted(source_locations)}")
    source_path = onnx_path.parent / next(iter(source_locations))
    if not source_path.is_file():
        raise FileNotFoundError(f"Exporter external-data file not found: {source_path}")

    records = []
    for tensor in tensors:
        entry = _external_entry(tensor)
        records.append((int(entry.get("offset", "0")), int(entry["length"]), tensor))
    records.sort(key=lambda item: item[0])

    total_bytes = sum(length for _, length, _ in records)
    shard_count = max(1, math.ceil(total_bytes / max_shard_bytes))
    stem = onnx_path.name
    shard_paths = [
        onnx_path.with_name(f"{stem}.data-{index:05d}-of-{shard_count:05d}")
        for index in range(shard_count)
    ]
    for shard_path in shard_paths:
        if shard_path.exists():
            shard_path.unlink()

    # Every tensor is far smaller than the shard cap.  Start a new shard only
    # at an initializer boundary; no initializer is ever split across files.
    shard_index = 0
    shard_offset = 0
    with source_path.open("rb") as source:
        destination = shard_paths[shard_index].open("wb")
        try:
            for source_offset, length, tensor in records:
                if length > max_shard_bytes:
                    raise RuntimeError(
                        f"Initializer {tensor.name!r} is {length} bytes, larger than shard cap {max_shard_bytes}"
                    )
                if shard_offset and shard_offset + length > max_shard_bytes:
                    destination.close()
                    shard_index += 1
                    if shard_index >= len(shard_paths):
                        # Padding/alignment can make the conservative count one
                        # short; extend deterministically if that occurs.
                        shard_paths.append(
                            onnx_path.with_name(
                                f"{stem}.data-{shard_index:05d}-of-{shard_count + 1:05d}"
                            )
                        )
                    destination = shard_paths[shard_index].open("wb")
                    shard_offset = 0
                _copy_range(source, destination, source_offset, length)
                _set_external_location(
                    tensor,
                    shard_paths[shard_index].name,
                    shard_offset,
                    length,
                )
                shard_offset += length
        finally:
            destination.close()

    # If an alignment edge case extended the shard list, rename all files to a
    # single stable total count before updating the protobuf references.
    if len(shard_paths) != shard_count:
        final_count = len(shard_paths)
        renamed: list[Path] = []
        for index, old_path in enumerate(shard_paths):
            new_path = onnx_path.with_name(f"{stem}.data-{index:05d}-of-{final_count:05d}")
            if old_path != new_path:
                old_path.replace(new_path)
            renamed.append(new_path)
        shard_paths = renamed
        # Locations were written before the final count was known.
        for tensor in tensors:
            entry = _external_entry(tensor)
            old_name = entry["location"]
            index = next(i for i, path in enumerate(shard_paths) if path.name.startswith(old_name.rsplit("-of-", 1)[0]))
            _set_external_location(tensor, shard_paths[index].name, int(entry["offset"]), int(entry["length"]))

    onnx.save_model(model, str(onnx_path), save_as_external_data=False)
    source_path.unlink()

    # Verify all references before returning.  This catches accidental stale
    # locations and boundary errors without loading any initializer bytes.
    reloaded = onnx.load_model(str(onnx_path), load_external_data=False)
    reloaded_tensors = _all_external_tensors(reloaded)
    expected_names = {path.name for path in shard_paths}
    referenced_names = set()
    for tensor in reloaded_tensors:
        entry = _external_entry(tensor)
        location = entry["location"]
        if location not in expected_names:
            raise RuntimeError(f"Initializer {tensor.name!r} references unexpected shard {location!r}")
        shard_size = (onnx_path.parent / location).stat().st_size
        if int(entry.get("offset", "0")) + int(entry["length"]) > shard_size:
            raise RuntimeError(f"Initializer {tensor.name!r} exceeds shard {location!r} bounds")
        referenced_names.add(location)
    if referenced_names != expected_names:
        raise RuntimeError(f"Unused or missing external shards: expected {expected_names}, used {referenced_names}")
    return shard_paths


def _embed_metadata_without_loading_weights(onnx_path: Path, metadata: dict[str, Any]) -> None:
    """Embed metadata while preserving external references and avoiding 3.5-GB reads."""

    model = onnx.load_model(str(onnx_path), load_external_data=False)
    existing = {prop.key for prop in model.metadata_props}
    for key, value in metadata.items():
        if value is None or key in {"artifact_sha256", "artifact_size_bytes"}:
            continue
        serialized = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
        if key in existing:
            for prop in model.metadata_props:
                if prop.key == key:
                    prop.value = serialized
                    break
        else:
            prop = model.metadata_props.add()
            prop.key = key
            prop.value = serialized
    onnx.save_model(model, str(onnx_path), save_as_external_data=False)


def _human_size(size_bytes: int) -> str:
    value = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TB"


def _external_metadata(onnx_path: Path, shard_paths: Iterable[Path]) -> list[dict[str, Any]]:
    model = onnx.load_model(str(onnx_path), load_external_data=False)
    locations = sorted({
        _external_entry(tensor)["location"]
        for tensor in _all_external_tensors(model)
    })
    by_name = {path.name: path for path in shard_paths}
    if set(locations) != set(by_name):
        raise RuntimeError(f"External shard metadata mismatch: refs={locations}, files={sorted(by_name)}")
    return [
        {
            "location": name,
            "filename": name,
            "size_bytes": by_name[name].stat().st_size,
            "sha256": compute_file_sha256(by_name[name]),
        }
        for name in locations
    ]


def verify_pytorch_vs_onnx_parity(
    wrapper: nn.Module,
    onnx_path: Path,
    *,
    device: torch.device,
    tolerance: float = 1e-4,
    num_samples: int = 3,
) -> dict[str, Any]:
    """Run the required synthetic wrapper-vs-ONNX numerical parity check."""

    wrapper.eval()
    generator = torch.Generator(device=device).manual_seed(42)
    sample_inputs = torch.randn(num_samples, 3, 224, 224, generator=generator, device=device)
    with torch.inference_mode():
        pytorch_outputs = wrapper(sample_inputs).detach().cpu().numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_outputs = session.run(["probabilities"], {"pixel_values": sample_inputs.detach().cpu().numpy()})[0]
    if onnx_outputs.shape != (num_samples, 2):
        raise RuntimeError(f"Unexpected ONNX parity output shape {onnx_outputs.shape}")

    difference = np.abs(pytorch_outputs - onnx_outputs)
    max_abs_diff = float(np.max(difference))
    mean_abs_diff = float(np.mean(difference))
    result = {
        "passed": bool(max_abs_diff <= tolerance),
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "tolerance": tolerance,
        "num_samples": num_samples,
        "scope": SPECTRAL_DISPOSITION["parity_scope"],
    }
    if not result["passed"]:
        raise RuntimeError(
            f"PyTorch wrapper to ONNX parity failed: max_abs_diff={max_abs_diff:.6e} > {tolerance:.6e}"
        )
    LOGGER.info(
        "Parity passed: max_abs_diff=%.6e mean_abs_diff=%.6e (%d samples)",
        max_abs_diff,
        mean_abs_diff,
        num_samples,
    )
    return result


def _preprocess_real_image(image_path: Path) -> np.ndarray:
    image = Image.open(image_path).convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    pixels = (pixels - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)) / np.asarray(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    return np.transpose(pixels, (2, 0, 1))[None, ...].astype(np.float32, copy=False)


def run_onnxruntime_probe(onnx_path: Path, image_path: Path) -> dict[str, Any]:
    """Run one real-image ONNX Runtime inference probe and fail on bad output."""

    image_path = Path(image_path).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Probe image not found: {image_path}")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    pixels = _preprocess_real_image(image_path)
    outputs = session.run(["probabilities"], {"pixel_values": pixels})[0]
    if outputs.shape != (1, 2) or not np.isfinite(outputs).all():
        raise RuntimeError(f"ONNX Runtime probe returned invalid output shape/values: {outputs.shape}")
    probabilities = [float(value) for value in outputs[0]]
    if abs(sum(probabilities) - 1.0) > 1e-4:
        raise RuntimeError(f"ONNX Runtime probe probabilities do not sum to one: {probabilities}")
    result = {
        "passed": True,
        "runtime": "onnxruntime",
        "providers": ["CPUExecutionProvider"],
        "image": image_path.name,
        "input_shape": list(pixels.shape),
        "output_shape": list(outputs.shape),
        "probabilities": probabilities,
    }
    LOGGER.info("Real-image ORT probe passed: probabilities=%s", probabilities)
    return result


def _build_metadata(
    *,
    variant: str,
    checkpoint_path: Path,
    inspection: dict[str, Any],
    onnx_path: Path,
    shard_metadata: list[dict[str, Any]],
    parity: dict[str, Any],
    probe: dict[str, Any],
) -> dict[str, Any]:
    config = inspection["config"]
    model_config = inspection["model_config"]
    artifact_size = onnx_path.stat().st_size
    artifact_sha = compute_file_sha256(onnx_path)
    total_bundle_size = artifact_size + sum(item["size_bytes"] for item in shard_metadata)
    return {
        "schema_version": 2,
        "artifact_id": f"mechadetect-lattice-{variant}-float32",
        "model_family": "dinov3-vith16plus",
        "family": "Lattice",
        "scope": "Normal" if variant == "normal" else "Super",
        "variant": variant,
        "stage": str(config.get("training", {}).get("stage", "teacher_stage2")),
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "quantization": "float32",
        "precision": "float32",
        "opset_version": EXPORT_OPSET,
        "input_contract": {
            "name": "pixel_values",
            "dtype": "float32",
            "shape": ["batch", 3, 224, 224],
            "layout": "NCHW",
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
        "output_contract": {
            "name": "probabilities",
            "dtype": "float32",
            "shape": ["batch", 2],
            "columns": ["authentic_probability", "aigc_probability"],
        },
        "preprocessing": {
            "policy": config.get("preprocessing", {}).get("policy", "square_jpeg95"),
            "version": str(config.get("preprocessing", {}).get("version", 2)),
            "browser_resize": "224x224 RGB canvas resize",
        },
        "manifest_digest": inspection["manifest_digest"],
        "evaluation_status": "experimental",
        "calibrated_threshold": None,
        "threshold_status": "uncalibrated",
        "temporary_ui_threshold": 0.5,
        "threshold_note": "0.5 is a temporary UI display threshold only; no calibrated promotion threshold is claimed.",
        "checkpoint_file": checkpoint_path.name,
        "checkpoint_sha256": compute_file_sha256(checkpoint_path),
        "checkpoint_parameter_count": inspection["checkpoint_parameter_count"],
        "architecture": {
            "backbone_type": model_config["backbone_type"],
            "encoder_id": model_config["encoder_id"],
            "encoder_revision": model_config["encoder_revision"],
            "encoder_dim": model_config["encoder_dim"],
            "encoder_layers": 32,
            "patch_size": 16,
            "image_size": model_config["image_size"],
            "prefix_tokens_removed": 5,
            "trunk_dim": model_config["trunk_dim"],
            "branch_dim": model_config["branch_dim"],
            "use_token_adapter": model_config["use_token_adapter"],
            "spectral_expert_configured": model_config["spectral_expert"],
        },
        **SPECTRAL_DISPOSITION,
        "browser_path_status": "experimental_spectral_free_forward_tensor",
        "hardware_class": "workstation-class",
        "recommended_execution_provider": "WebGPU",
        "browser_warning": (
            "Very large workstation-class Float32 Lattice artifact. WebGPU is preferred. "
            "The browser graph follows forward_tensor and omits the configured spectral expert; "
            "parity is verified only against that spectral-free path."
        ),
        "artifact_path": f"outputs/models/{onnx_path.name}",
        "artifact_sha256": artifact_sha,
        "artifact_size_bytes": artifact_size,
        "bundle_size_bytes": total_bundle_size,
        "external_data": shard_metadata,
        "external_data_total_bytes": sum(item["size_bytes"] for item in shard_metadata),
        "parity_verified": bool(parity["passed"]),
        "parity": parity,
        "onnxruntime_probe": probe,
        "repository": "zye2/mechadetect-models",
        "repository_path": f"models/{onnx_path.name}",
    }


def _write_sidecar(metadata: dict[str, Any], onnx_path: Path) -> Path:
    sidecar_path = onnx_path.with_name(f"{onnx_path.stem}.metadata.json")
    with sidecar_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return sidecar_path




def _catalog_entry_from_metadata(metadata: dict[str, Any], remote_revision: str) -> dict[str, Any]:
    variant = metadata["variant"]
    name_variant = "Normal" if variant == "normal" else "Super"
    size_label = f"{_human_size(metadata['bundle_size_bytes'])} · WebGPU preferred"
    entry: dict[str, Any] = {
        "id": metadata["artifact_id"],
        "name": f"Lattice {name_variant} Float32 · Workstation",
        "path": _hf_resolve_url(metadata["repository"], remote_revision, Path(metadata["artifact_path"]).name),
        "family": "Lattice",
        "scope": name_variant,
        "model_family": metadata["model_family"],
        "variant": variant,
        "stage": metadata["stage"],
        "parameter_count": metadata["parameter_count"],
        "quantization": "float32",
        "precision_label": "Float32 · Workstation",
        "size_bytes": metadata["artifact_size_bytes"],
        "size_label": size_label,
        "calibrated_threshold": None,
        "temporary_ui_threshold": metadata["temporary_ui_threshold"],
        "threshold_status": metadata["threshold_status"],
        "preprocessing_version": metadata["preprocessing"]["version"],
        "manifest_digest": metadata["manifest_digest"],
        "evaluation_status": metadata["evaluation_status"],
        "is_experimental_int8": False,
        "experimental_browser_path": True,
        "browser_path_status": metadata["browser_path_status"],
        "hardware_class": metadata["hardware_class"],
        "recommended_execution_provider": metadata["recommended_execution_provider"],
        "spectral_expert_omitted": True,
        "browser_warning": metadata["browser_warning"],
        "opset_version": metadata["opset_version"],
        "checkpoint_repository": None,
        "checkpoint_revision": None,
        "checkpoint_file": metadata["checkpoint_file"],
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "artifact_sha256": metadata["artifact_sha256"],
        "artifact_size_bytes": metadata["artifact_size_bytes"],
        "bundle_size_bytes": metadata["bundle_size_bytes"],
        "external_data": [
            {
                "path": item["location"],
                "url": _hf_resolve_url(metadata["repository"], remote_revision, item["filename"]),
                "size_bytes": item["size_bytes"],
                "sha256": item["sha256"],
            }
            for item in metadata["external_data"]
        ],
        "parity_verified": metadata["parity_verified"],
        "max_abs_diff_pytorch_vs_onnx": metadata["parity"]["max_abs_diff"],
        "mean_abs_diff_pytorch_vs_onnx": metadata["parity"]["mean_abs_diff"],
        "parity_scope": metadata["parity_scope"],
        "onnxruntime_probe": metadata["onnxruntime_probe"],
    }
    entry["artifact_revision"] = remote_revision
    entry["artifact_repository"] = metadata["repository"]
    entry["artifact_repository_path"] = metadata["repository_path"]
    return entry


def update_web_catalog(metadata_list: list[dict[str, Any]], remote_revision: str) -> Path:
    """Keep only usable Float32 artifacts and append the two real Lattice entries."""

    catalog_path = WEB_MODEL_DIR / "metadata.json"
    if catalog_path.is_file():
        with catalog_path.open(encoding="utf-8") as handle:
            catalog = json.load(handle)
    else:
        catalog = {"default_model": "atom-super-float32", "students": []}
    existing = []
    repository = metadata_list[0]["repository"]
    for source_entry in catalog.get("students", []):
        if source_entry.get("quantization") != "float32" or str(source_entry.get("id", "")).startswith("mechadetect-lattice-"):
            continue
        entry = dict(source_entry)
        filename = str(entry["path"]).rsplit("/", 1)[-1]
        entry["path"] = _hf_resolve_url(repository, remote_revision, filename)
        entry["artifact_repository"] = repository
        entry["artifact_revision"] = remote_revision
        entry["artifact_repository_path"] = f"models/{filename}"
        existing.append(entry)
    lattice_entries = sorted(
        (_catalog_entry_from_metadata(metadata, remote_revision) for metadata in metadata_list),
        key=lambda entry: (entry["scope"] != "Normal", entry["id"]),
    )
    combined = existing + lattice_entries
    if len(combined) != 6:
        raise RuntimeError(f"Expected final six usable Float32 catalog entries, found {len(combined)}")
    catalog = {"default_model": "atom-super-float32", "students": combined}
    with catalog_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(catalog, handle, indent=2)
        handle.write("\n")
    LOGGER.info("Updated web catalog: %d usable Float32 entries", len(combined))
    return catalog_path


def ensure_hf_repo(api: Any, repo_id: str) -> None:
    try:
        api.repo_info(repo_id=repo_id, repo_type="model")
    except Exception:
        api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)


def upload_artifacts(
    artifact_files: list[Path],
    *,
    repo_id: str = HF_MODEL_REPO,
) -> dict[str, Any]:
    """Upload the complete Lattice bundle and verify every remote path."""

    load_local_environment(REPO_ROOT)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("HF authentication is unavailable after loading the local environment")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    ensure_hf_repo(api, repo_id)
    names = [path.name for path in artifact_files]
    commit_info = api.upload_folder(
        folder_path=str(OUTPUT_DIR),
        path_in_repo="models",
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=names,
        commit_message="Publish Lattice Float32 browser artifacts with external data",
    )
    revision = getattr(commit_info, "oid", None) or getattr(commit_info, "commit_id", None)
    remote_paths = [f"models/{name}" for name in names]
    listed = set(api.list_repo_files(repo_id=repo_id, repo_type="model", revision=revision))
    missing = [path for path in remote_paths if path not in listed]
    if missing:
        raise RuntimeError(f"Hugging Face upload completed but remote paths are missing: {missing}")
    LOGGER.info("Uploaded and verified %d files to %s at revision %s", len(remote_paths), repo_id, revision or "<unknown>")
    return {"repo_id": repo_id, "revision": revision, "remote_paths": remote_paths}


def export_variant(
    variant: str,
    checkpoint_path: Path,
    *,
    output_dir: Path = OUTPUT_DIR,
    shard_bytes: int = DEFAULT_SHARD_BYTES,
    probe_image: Path | None = None,
) -> tuple[dict[str, Any], list[Path]]:
    """Export, split, verify, and sidecar one Lattice variant."""

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"mechadetect-lattice-{variant}-float32.onnx"
    model, inspection = load_lattice_model(checkpoint_path, variant)
    # Keep calling ProvenanceModel.forward_tensor, but replace only the two
    # exporter-sensitive attention/prefix-token broadcasts with mathematically
    # identical batch-dynamic modules.
    model = _prepare_dynamic_export_model(model)
    wrapper = LatticeWebGPUExportWrapper(model).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Export device: %s", device)
    wrapper.to(device)
    dummy_input = torch.randn(1, 3, 224, 224, device=device, dtype=torch.float32)

    # The dynamo exporter ignores/partially applies ``dynamic_axes`` for
    # intermediate tensors.  In this graph that leaves the attention/GEMM
    # reshapes specialized to the sample batch of one, so ORT rejects a
    # legitimate batch of two even when the input annotation is dynamic.
    # ``dynamic_shapes`` carries the batch symbol through torch.export and
    # keeps both the input and final probabilities contract truly batched.
    batch_dim = torch.export.Dim("batch_size", min=1)
    LOGGER.info("Exporting %s to %s (opset=%d, external_data=True, dynamic batch)", variant, output_path, EXPORT_OPSET)
    torch.onnx.export(
        wrapper,
        (dummy_input,),
        str(output_path),
        input_names=["pixel_values"],
        output_names=["probabilities"],
        opset_version=EXPORT_OPSET,
        dynamic_shapes=({0: batch_dim},),
        external_data=True,
        dynamo=True,
        optimize=True,
        verify=False,
    )
    shard_paths = split_external_data(output_path, max_shard_bytes=shard_bytes)

    # Keep the wrapper on the same device for the synthetic parity check. ORT
    # reads the relocatable shards itself, so this probe also proves the split
    # references are valid.
    parity = verify_pytorch_vs_onnx_parity(wrapper, output_path, device=device)
    probe_path = probe_image or (REPO_ROOT / "web" / "samples" / "sample_authentic_imagenet.jpg")
    probe = run_onnxruntime_probe(output_path, probe_path)
    shard_metadata = _external_metadata(output_path, shard_paths)
    metadata = _build_metadata(
        variant=variant,
        checkpoint_path=Path(checkpoint_path).resolve(),
        inspection=inspection,
        onnx_path=output_path,
        shard_metadata=shard_metadata,
        parity=parity,
        probe=probe,
    )
    # Embed metadata without loading the multi-gigabyte initializers.  The
    # protobuf grows when metadata is first added, so recalculate the bundle
    # size after each write until the embedded value and on-disk size agree.
    external_size = sum(item["size_bytes"] for item in shard_metadata)
    for _ in range(4):
        before_size = output_path.stat().st_size
        metadata["bundle_size_bytes"] = before_size + external_size
        _embed_metadata_without_loading_weights(output_path, metadata)
        after_size = output_path.stat().st_size
        if after_size == before_size:
            break
    final_size = output_path.stat().st_size
    final_bundle_size = final_size + external_size
    if metadata["bundle_size_bytes"] != final_bundle_size:
        metadata["bundle_size_bytes"] = final_bundle_size
        _embed_metadata_without_loading_weights(output_path, metadata)
        final_size = output_path.stat().st_size
        metadata["bundle_size_bytes"] = final_size + external_size
    metadata["artifact_sha256"] = compute_file_sha256(output_path)
    metadata["artifact_size_bytes"] = final_size

    sidecar_path = _write_sidecar(metadata, output_path)
    files = [output_path, *shard_paths, sidecar_path]
    LOGGER.info(
        "Lattice %s complete: protobuf=%s bundle=%s shards=%d",
        variant,
        _human_size(output_path.stat().st_size),
        _human_size(metadata["bundle_size_bytes"]),
        len(shard_paths),
    )

    # Release the model before the next 872M-parameter variant is built.
    del wrapper, model, dummy_input
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metadata, files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["normal", "super", "all"], default="all")
    parser.add_argument("--normal-checkpoint", type=Path, default=DEFAULT_CHECKPOINTS["normal"])
    parser.add_argument("--super-checkpoint", type=Path, default=DEFAULT_CHECKPOINTS["super"])
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--shard-bytes", type=int, default=DEFAULT_SHARD_BYTES)
    parser.add_argument("--probe-image", type=Path, default=None)
    parser.add_argument("--upload", action="store_true", help="Upload generated artifacts to Hugging Face")
    parser.add_argument("--repo-id", default=HF_MODEL_REPO)
    args = parser.parse_args()

    load_local_environment(REPO_ROOT)
    variants = ["normal", "super"] if args.variant == "all" else [args.variant]
    checkpoints = {"normal": args.normal_checkpoint, "super": args.super_checkpoint}
    all_metadata: list[dict[str, Any]] = []
    all_files: list[Path] = []
    all_file_groups: list[list[Path]] = []
    for variant in variants:
        metadata, files = export_variant(
            variant,
            checkpoints[variant],
            output_dir=args.output_dir,
            shard_bytes=args.shard_bytes,
            probe_image=args.probe_image,
        )
        all_metadata.append(metadata)
        all_files.extend(files)
        all_file_groups.append(files)

    upload_result: dict[str, Any] | None = None
    catalog_path: Path | None = None
    if args.upload:
        upload_result = upload_artifacts(all_files, repo_id=args.repo_id)
        revision = upload_result.get("revision")
        if not revision:
            raise RuntimeError("Hugging Face upload returned no immutable revision")
        catalog_path = update_web_catalog(all_metadata, revision)
    else:
        LOGGER.info("Skipping web catalog update because artifacts were not uploaded")
    print(json.dumps({
        "artifacts": [
            {
                "variant": metadata["variant"],
                "files": [path.name for path in files],
                "sizes": {path.name: path.stat().st_size for path in files},
                "parity": metadata["parity"],
                "onnxruntime_probe": metadata["onnxruntime_probe"],
            }
            for metadata, files in zip(all_metadata, all_file_groups, strict=True)
        ],
        "upload": upload_result,
        "catalog": str(catalog_path) if catalog_path else None,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
