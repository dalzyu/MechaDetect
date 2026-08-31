"""Batch AIGC prediction for the TechJam Track 5 browser models."""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parent
CATALOG_PATH = ROOT / "web" / "model" / "metadata.json"
CACHE_DIR = ROOT / ".cache" / "mechadetect"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".avif"}
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _catalog_model(model_id: str) -> dict:
    with CATALOG_PATH.open(encoding="utf-8") as handle:
        catalog = json.load(handle)
    for model in catalog.get("students", []):
        if model.get("id") == model_id:
            return model
    available = ", ".join(str(m.get("id")) for m in catalog.get("students", []))
    raise ValueError(f"Unknown model {model_id!r}. Available models: {available}")


def _read_menu(title: str, options: tuple[tuple[str, str], ...]) -> str:
    print(title)
    for index, (label, _) in enumerate(options, start=1):
        suffix = " [default]" if index == 1 else ""
        print(f"  {index}. {label}{suffix}")
    while True:
        try:
            choice = input("Choose a model [1]: ").strip()
        except EOFError:
            choice = ""
        if not choice:
            return options[0][1]
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1][1]
        print(f"Enter a number from 1 to {len(options)}.", file=sys.stderr)


def _choose_model() -> str:
    choice = _read_menu(
        "Super models:",
        (
            ("Atom Super — 89.4M parameters", "atom-super-float32"),
            ("Quark Super — 25.1M parameters", "quark-super-float32"),
            ("More — show Normal variants", "more"),
        ),
    )
    if choice != "more":
        return choice
    return _read_menu(
        "Normal models:",
        (
            ("Atom Normal — 89.4M parameters", "atom-normal-float32"),
            ("Quark Normal — 25.1M parameters", "quark-normal-float32"),
        ),
    )




def _token() -> str | None:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


def _download(url: str, destination: Path) -> None:
    """Download atomically so an interrupted first run cannot poison the cache."""
    if destination.is_file() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url)
    token = _token()
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.part")
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open(
            "wb"
        ) as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        os.replace(temporary, destination)
    except urllib.error.HTTPError as error:
        temporary.unlink(missing_ok=True)
        if error.code in (401, 403):
            raise RuntimeError(
                "HuggingFace denied model download; set HF_TOKEN (the model repository is private)."
            ) from error
        raise RuntimeError(f"Could not download {url} (HTTP {error.code}).") from error
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _model_files(model: dict) -> Path:
    model_url = model.get("path")
    if not model_url:
        raise ValueError(f"Catalog entry {model.get('id')!r} has no model URL")
    model_name = unquote(urlparse(model_url).path).rsplit("/", 1)[-1]
    if not model_name:
        raise ValueError(f"Invalid model URL in catalog: {model_url!r}")
    model_path = CACHE_DIR / model_name
    _download(model_url, model_path)
    for shard in model.get("external_data", []):
        shard_name = Path(str(shard.get("path", ""))).name
        shard_url = shard.get("url")
        if not shard_name or not shard_url:
            raise ValueError(f"Invalid external_data entry for {model.get('id')!r}")
        shard_path = CACHE_DIR / shard_name
        _download(str(shard_url), shard_path)
    return model_path


def _images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image format: {input_path}")
        return [input_path]
    if input_path.is_dir():
        return sorted(
            p
            for p in input_path.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
    raise ValueError(f"Input path does not exist: {input_path}")


def _preprocess(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32) / 255.0
    pixels = (pixels - MEAN) / STD
    return np.transpose(pixels, (2, 0, 1))[None, ...].astype(np.float32, copy=False)


def _session(model_path: Path) -> ort.InferenceSession:
    available = set(ort.get_available_providers())
    providers = [
        name
        for name in ("WebGPUExecutionProvider", "CUDAExecutionProvider")
        if name in available
    ]
    if "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")
    try:
        return ort.InferenceSession(str(model_path), providers=providers or None)
    except Exception as error:
        if "CPUExecutionProvider" in available and providers != ["CPUExecutionProvider"]:
            print(
                f"Accelerated provider unavailable ({error}); retrying on CPU.",
                file=sys.stderr,
            )
            return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        raise


def predict(input_path: Path, output_path: Path, model_id: str, threshold: float | None) -> None:
    paths = _images(input_path)
    if not paths:
        raise ValueError(f"No supported images found under {input_path}")
    if threshold is not None and not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    model = _catalog_model(model_id)
    model_path = _model_files(model)
    session = _session(model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    calibrated = model.get("calibrated_threshold")
    if threshold is None:
        threshold = (
            calibrated
            if calibrated is not None
            else model.get("temporary_ui_threshold", 0.5)
        )
    results: list[dict[str, str | float]] = []
    print(f"{'image_path':<60} {'pred':>8}  verdict")
    print(f"{'-' * 60} {'-' * 8}  {'-' * 8}")
    for path in paths:
        values = np.asarray(
            session.run([output_name], {input_name: _preprocess(path)})[0]
        ).reshape(-1)
        if values.size < 2 or not np.isfinite(values[:2]).all():
            raise RuntimeError(
                f"Unexpected output from model for {path}: shape {values.shape}"
            )
        pred = float(np.clip(values[1], 0.0, 1.0))
        verdict = "AIGC" if pred >= threshold else "Original"
        results.append({"image_path": str(path), "pred": pred})
        print(f"{str(path):<60} {pred:8.6f}  {verdict}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict AIGC confidence for images using an ONNX model."
    )
    parser.add_argument(
        "--input", type=Path, required=True, help="Image directory or single image file."
    )
    parser.add_argument("--output", type=Path, required=True, help="JSON output path.")
    parser.add_argument(
        "--model",
        default=None,
        help="Catalog model ID. Omit for the interactive chooser; Atom Super is the default.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="AIGC verdict threshold (default: catalog value or 0.5).",
    )
    args = parser.parse_args()
    try:
        model_id = args.model or _choose_model()
        predict(args.input, args.output, model_id, args.threshold)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
