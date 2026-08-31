#!/usr/bin/env python3
"""Curate a deterministic GPT Image 2 versus authentic cohort."""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GPT_REPO = "Goku-OpenLab/gpt-image-2-prompts-datasets"
GPT_SOURCE_URL = f"https://huggingface.co/datasets/{GPT_REPO}"
DATASET_NAME = "gpt_image_2_twitter"
DEFAULT_OUTPUT = PROJECT_ROOT / "splits" / "gpt_image_2_balanced_patch.jsonl"
MANIFEST_COLUMNS = (
    "image_path", "label", "dataset", "official_split", "generator",
    "manipulation_family", "source_image_group", "width", "height", "file_format",
    "tamper_mask_path", "source_url", "external_id", "generator_family",
    "generator_version", "prompt", "created_at", "sha256", "perceptual_hash",
    "quality_score", "provenance_confidence", "redistribution_mode", "origin_license",
    "license_url", "attribution", "selection_reason", "forbidden_demo_checked",
    "ai_positive", "split", "duplicate_group", "provenance",
)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _model_info(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("model_info", {})
    return value if isinstance(value, Mapping) else {}


def _image_name(row: Mapping[str, Any]) -> str:
    value = row.get("file_name")
    if isinstance(value, str) and value.strip():
        return value.strip().replace("\\", "/")
    media = row.get("media", {})
    if isinstance(media, Mapping):
        images = media.get("images", [])
        if isinstance(images, Sequence) and not isinstance(images, (str, bytes)):
            for image in images:
                if isinstance(image, str) and image.strip():
                    return image.strip().replace("\\", "/")
    return ""


def _is_gpt_image_2(row: Mapping[str, Any]) -> bool:
    model = str(_model_info(row).get("name", "")).strip().lower()
    return not model or model in {"gpt-image-2", "gpt_image_2", "gpt image 2"}


def iter_gpt_image2_rows(dataset: Iterable[Mapping[str, Any]]) -> Iterator[dict[str, Any]]:
    """Yield distinct GPT Image 2 source rows."""
    seen: set[str] = set()
    for source_index, raw in enumerate(dataset):
        if not isinstance(raw, Mapping) or not _is_gpt_image_2(raw):
            continue
        image_name = _image_name(raw)
        if not image_name or image_name in seen:
            continue
        seen.add(image_name)
        row = dict(raw)
        # Acquisition streams the same source and uses this index to avoid
        # downloading the first N rows instead of the sampled N rows.
        row["_source_index"] = source_index
        yield row


def reservoir_sample(records: Iterable[Mapping[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    """Sample at most ``count`` records from a stream without loading it all."""
    if count < 0:
        raise ValueError("count must be non-negative")
    rng = random.Random(seed)
    reservoir: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        copied = dict(record)
        if index < count:
            reservoir.append(copied)
        elif count and rng.randrange(index + 1) < count:
            reservoir[rng.randrange(count)] = copied
    if len(reservoir) < count:
        raise ValueError(f"source contains {len(reservoir)} distinct records; need {count}")
    return reservoir


def _positive_manifest_row(row: Mapping[str, Any], index: int) -> dict[str, Any]:
    image_name = _image_name(row)
    if not image_name:
        raise ValueError("GPT Image 2 row has no file_name/media.images value")
    model = _model_info(row)
    spec = row.get("spec", {})
    spec = spec if isinstance(spec, Mapping) else {}
    i18n = row.get("i18n", {})
    i18n = i18n if isinstance(i18n, Mapping) else {}
    english = i18n.get("en", {})
    english = english if isinstance(english, Mapping) else {}
    prompt = row.get("raw_p") or english.get("p") or ""
    external_id = str(row.get("id") or Path(image_name).stem or f"gpt_image_2_{index:06d}")
    return {
        "image_path": f"{DATASET_NAME}/{DATASET_NAME}_{int(row.get('_source_index', index)):06d}.jpg",
        "label": "fully_aigc", "dataset": DATASET_NAME, "official_split": "train",
        "generator": "gpt_image_2", "manipulation_family": "",
        "source_image_group": f"{DATASET_NAME}:{external_id}",
        "width": _as_int(spec.get("width")), "height": _as_int(spec.get("height")),
        "file_format": Path(image_name).suffix.lstrip(".").upper() or "JPG",
        "tamper_mask_path": "", "source_url": GPT_SOURCE_URL, "external_id": external_id,
        "generator_family": "gpt_image_2", "generator_version": str(model.get("version", "1.0")),
        "prompt": str(prompt), "created_at": str(row.get("date", "")), "sha256": "",
        "perceptual_hash": "", "quality_score": "", "provenance_confidence": "high",
        "redistribution_mode": "reference_only", "origin_license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": "Goku-OpenLab, GPT Image 2 Prompt Dataset",
        "selection_reason": "Deterministic GPT Image 2 generated image sampled for Stage 2 retraining",
        "forbidden_demo_checked": True, "ai_positive": 1, "split": "train",
        "duplicate_group": f"{DATASET_NAME}:{external_id}", "provenance": "fully_aigc",
    }


def _is_authentic(row: Mapping[str, Any]) -> bool:
    label = str(row.get("label", "")).strip().lower()
    ai_positive = row.get("ai_positive")
    if ai_positive not in (None, ""):
        try:
            return int(ai_positive) == 0
        except (TypeError, ValueError):
            return False
    return label in {"0", "authentic", "real", "human"}


def _authentic_key(row: Mapping[str, Any]) -> str:
    return str(row.get("image_path") or row.get("external_id") or row.get("sha256") or "").strip()


def _negative_manifest_row(row: Mapping[str, Any], index: int) -> dict[str, Any]:
    source_path = _authentic_key(row)
    if not source_path:
        raise ValueError("authentic source row has no image_path/external_id/sha256")
    external_id = str(row.get("external_id") or Path(source_path).stem or f"authentic_{index:06d}")
    return {
        "image_path": source_path.replace("\\", "/"), "label": "authentic",
        "dataset": str(row.get("dataset") or "authentic_pool"), "official_split": "train",
        "generator": "authentic", "manipulation_family": "",
        "source_image_group": str(row.get("source_image_group") or f"authentic:{external_id}"),
        "width": _as_int(row.get("width")), "height": _as_int(row.get("height")),
        "file_format": str(row.get("file_format") or "").upper(), "tamper_mask_path": "",
        "source_url": str(row.get("source_url") or ""), "external_id": external_id,
        "generator_family": "authentic", "generator_version": str(row.get("generator_version") or ""),
        "prompt": "", "created_at": str(row.get("created_at") or ""),
        "sha256": str(row.get("sha256") or ""), "perceptual_hash": str(row.get("perceptual_hash") or ""),
        "quality_score": row.get("quality_score", ""),
        "provenance_confidence": str(row.get("provenance_confidence") or "medium"),
        "redistribution_mode": str(row.get("redistribution_mode") or "reference_only"),
        "origin_license": str(row.get("origin_license") or ""),
        "license_url": str(row.get("license_url") or ""), "attribution": str(row.get("attribution") or ""),
        "selection_reason": "Distinct authentic image sampled from the existing authentic pool",
        "forbidden_demo_checked": True, "ai_positive": 0, "split": "train",
        "duplicate_group": str(row.get("duplicate_group") or f"authentic:{external_id}"),
        "provenance": "authentic",
    }


def build_balanced_manifest(positive_records: Iterable[Mapping[str, Any]], authentic_records: Iterable[Mapping[str, Any]], *, positive_count: int = 2_000, negative_count: int = 2_000, seed: int = 42) -> list[dict[str, Any]]:
    """Build exactly balanced, distinct positive/negative manifest rows."""
    positives = reservoir_sample(iter_gpt_image2_rows(positive_records), positive_count, seed)
    unique_authentic: dict[str, dict[str, Any]] = {}
    for row in authentic_records:
        if _is_authentic(row):
            key = _authentic_key(row)
            if key:
                unique_authentic.setdefault(key, dict(row))
    negatives = reservoir_sample(unique_authentic.values(), negative_count, seed + 1)
    rows = ([_positive_manifest_row(row, i) for i, row in enumerate(positives)] +
            [_negative_manifest_row(row, i) for i, row in enumerate(negatives)])
    if len(rows) != positive_count + negative_count or sum(row["ai_positive"] for row in rows) != positive_count:
        raise AssertionError("curation did not produce the requested class balance")
    return rows


def load_manifest_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Load JSONL or CSV authentic-pool manifests."""
    paths = list(paths)
    records: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            continue
        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8", newline="") as handle:
                records.extend(dict(row) for row in csv.DictReader(handle))
        else:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        value = json.loads(line)
                        if isinstance(value, Mapping):
                            records.append(dict(value))
    if not records:
        raise FileNotFoundError("no authentic records found in: " + ", ".join(map(str, paths)))
    return records


def default_authentic_paths() -> list[Path]:
    combined = PROJECT_ROOT / "splits" / "combined_hf_dataset"
    return [combined / f"{split}.jsonl" for split in ("train", "validation", "test", "test_unseen")]


def write_manifest(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    """Write JSONL, or CSV when the output has a .csv suffix."""
    output.parent.mkdir(parents=True, exist_ok=True)
    normalized = [{column: row.get(column, "") for column in MANIFEST_COLUMNS} for row in rows]
    if output.suffix.lower() == ".csv":
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_COLUMNS))
            writer.writeheader(); writer.writerows(normalized)
        return
    with output.open("w", encoding="utf-8") as handle:
        for row in normalized:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def curate_gpt_image2_cohort(output: Path = DEFAULT_OUTPUT, *, authentic_paths: Iterable[Path] | None = None, positive_count: int = 2_000, negative_count: int = 2_000, seed: int = 42) -> list[dict[str, Any]]:
    """Query Hugging Face and write the balanced cohort manifest."""
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("datasets is required; run this script with uv") from exc
    dataset = load_dataset(GPT_REPO, split="train", streaming=True)
    authentic = load_manifest_records(list(authentic_paths or default_authentic_paths()))
    rows = build_balanced_manifest(dataset, authentic, positive_count=positive_count, negative_count=negative_count, seed=seed)
    write_manifest(rows, output)
    return rows

create_balanced_manifest = build_balanced_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--authentic-manifest", type=Path, action="append", dest="authentic_paths")
    parser.add_argument("--positive-count", type=int, default=2_000)
    parser.add_argument("--negative-count", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = curate_gpt_image2_cohort(args.output, authentic_paths=args.authentic_paths, positive_count=args.positive_count, negative_count=args.negative_count, seed=args.seed)
    print(f"Wrote {len(rows)} rows ({sum(row['ai_positive'] for row in rows)} AI-positive) to {args.output}")


if __name__ == "__main__":
    main()
