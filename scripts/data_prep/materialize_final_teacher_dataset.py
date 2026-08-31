#!/usr/bin/env python3
"""Copy the final teacher dataset's referenced bytes into one portable root.

The final manifests intentionally retain relative ``image_path`` and
``tamper_mask_path`` values.  This utility copies those assets from one or more
read-only source roots into a single destination that the training runtime can
use through its existing ``data_root`` setting.  It is fail-closed: unsafe
paths, missing assets, target collisions, and image hash mismatches abort the
materialization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET_SPLITS: tuple[str, ...] = ("train", "validation", "test", "test_unseen")
DEFAULT_OUTPUT_ROOT = _PROJECT_ROOT / ".runtime" / "final_teacher_data" / "images"
DEFAULT_MIN_FREE_BYTES = 512 * 1024 * 1024
CHUNK_SIZE = 1024 * 1024

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("materialize_final_teacher_dataset")


@dataclass(frozen=True)
class Asset:
    relative_path: str
    source_path: Path
    kind: str
    expected_sha256: str | None
    size_bytes: int


def _normalise_relative_path(raw: Any, *, field: str) -> str | None:
    """Return a safe portable path or ``None`` for an empty manifest field."""
    if raw is None or pd.isna(raw):
        return None
    value = str(raw).replace("\\", "/").strip()
    if not value or value.lower() == "nan":
        return None
    if value.startswith("/") or re.match(r"^[A-Za-z]:/", value):
        raise ValueError(f"{field} must be relative for a portable package: {value}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"Unsafe {field} path: {value}")
    return path.as_posix()


def _resolve_source(raw: Any, *, field: str, roots: list[Path]) -> Path:
    """Resolve one relative manifest path under the configured source roots."""
    value = str(raw).replace("\\", "/").strip()
    if not value:
        raise ValueError(f"Empty required {field} path")
    candidate_rel = _normalise_relative_path(value, field=field)
    if candidate_rel is None:
        raise ValueError(f"Empty required {field} path")
    for root in roots:
        candidate = (root / PurePosixPath(candidate_rel)).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"{field} '{value}' was not found below any source root: "
        + ", ".join(str(root) for root in roots)
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_rows(manifest_dir: Path, splits: Iterable[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for split in splits:
        path = manifest_dir / f"{split}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"Missing final manifest: {path}")
        frame = pd.read_parquet(path)
        if "image_path" not in frame.columns:
            raise ValueError(f"Manifest {path} has no image_path column")
        frames.append(frame)
    if not frames:
        raise ValueError("At least one split is required")
    return pd.concat(frames, ignore_index=True)


def _add_asset(
    assets: dict[str, Asset],
    *,
    relative_path: str,
    source_path: Path,
    kind: str,
    expected_sha256: str | None,
) -> None:
    existing = assets.get(relative_path)
    if existing is not None:
        if existing.source_path.resolve() != source_path.resolve():
            raise ValueError(
                f"Target path collision for {relative_path}: "
                f"{existing.source_path} versus {source_path}"
            )
        if (
            expected_sha256
            and existing.expected_sha256
            and expected_sha256 != existing.expected_sha256
        ):
            raise ValueError(f"Conflicting expected hashes for {relative_path}")
        return
    assets[relative_path] = Asset(
        relative_path=relative_path,
        source_path=source_path,
        kind=kind,
        expected_sha256=expected_sha256,
        size_bytes=source_path.stat().st_size,
    )


def _copy_one(asset: Asset, output_root: Path) -> tuple[str, int, bool]:
    destination = output_root / PurePosixPath(asset.relative_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_digest = _sha256(asset.source_path)
    if asset.expected_sha256 and source_digest != asset.expected_sha256:
        raise ValueError(
            f"Source SHA-256 mismatch for {asset.relative_path}: "
            f"expected {asset.expected_sha256}, got {source_digest}"
        )

    if destination.is_file():
        if _sha256(destination) != source_digest:
            raise ValueError(f"Existing destination differs from source: {destination}")
        return asset.relative_path, asset.size_bytes, True
    if destination.exists():
        raise ValueError(f"Destination is not a regular file: {destination}")

    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with asset.source_path.open("rb") as source, temporary.open("wb") as target:
            for chunk in iter(lambda: source.read(CHUNK_SIZE), b""):
                target.write(chunk)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return asset.relative_path, asset.size_bytes, False


def materialize(
    *,
    manifest_dir: Path,
    roots: list[Path],
    output_root: Path,
    splits: tuple[str, ...] = TARGET_SPLITS,
    workers: int = 32,
    minimum_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Materialize all image and tamper-mask assets referenced by final splits."""
    if workers < 1:
        raise ValueError("workers must be positive")
    roots = [root.resolve() for root in roots]
    frame = _load_rows(manifest_dir.resolve(), splits)
    assets: dict[str, Asset] = {}
    for row in frame.to_dict(orient="records"):
        image_relative = _normalise_relative_path(row.get("image_path"), field="image_path")
        if image_relative is None:
            raise ValueError("Final manifest contains an empty image_path")
        image_source = _resolve_source(row["image_path"], field="image_path", roots=roots)
        expected = str(row.get("sha256", "") or "").strip().lower() or None
        if expected and not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"Invalid image SHA-256 for {image_relative}: {expected}")
        _add_asset(
            assets,
            relative_path=image_relative,
            source_path=image_source,
            kind="image",
            expected_sha256=expected,
        )

        mask_value = row.get("tamper_mask_path")
        mask_relative = _normalise_relative_path(mask_value, field="tamper_mask_path")
        if mask_relative is not None:
            mask_source = _resolve_source(mask_value, field="tamper_mask_path", roots=roots)
            _add_asset(
                assets,
                relative_path=mask_relative,
                source_path=mask_source,
                kind="tamper_mask",
                expected_sha256=None,
            )

    total_bytes = sum(asset.size_bytes for asset in assets.values())
    output_root = output_root.resolve()
    output_root.parent.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output_root.parent)
    if usage.free < minimum_free_bytes:
        raise RuntimeError(
            f"Destination volume has only {usage.free} bytes free; "
            f"required at least {minimum_free_bytes} bytes"
        )
    existing_bytes = sum(
        asset.size_bytes
        for asset in assets.values()
        if (output_root / PurePosixPath(asset.relative_path)).is_file()
    )
    required_new_bytes = max(0, total_bytes - existing_bytes)
    if usage.free < required_new_bytes + minimum_free_bytes:
        raise RuntimeError(
            f"Insufficient free space for materialization: need {required_new_bytes} new bytes "
            f"plus {minimum_free_bytes} bytes headroom, have {usage.free}"
        )

    results: list[tuple[str, int, bool]] = []
    if not dry_run:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(lambda asset: _copy_one(asset, output_root), assets.values()):
                results.append(result)
        logger.info("Materialized %d assets into %s", len(results), output_root)
    else:
        logger.info("Dry run: %d assets would be materialized into %s", len(assets), output_root)

    image_count = sum(asset.kind == "image" for asset in assets.values())
    mask_count = sum(asset.kind == "tamper_mask" for asset in assets.values())
    report: dict[str, Any] = {
        "report_version": "1.0.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "manifest_dir": str(manifest_dir.resolve()),
        "source_roots": [str(root) for root in roots],
        "output_root": str(output_root),
        "splits": list(splits),
        "rows": int(len(frame)),
        "unique_image_assets": image_count,
        "unique_tamper_mask_assets": mask_count,
        "unique_assets": len(assets),
        "source_bytes": total_bytes,
        "required_new_bytes": required_new_bytes,
        "workers": workers,
        "dry_run": dry_run,
        "sha256_verified_images": image_count if not dry_run else 0,
        "assets_copied": sum(not reused for _, _, reused in results),
        "assets_reused": sum(reused for _, _, reused in results),
    }
    if not dry_run:
        report_path = output_root.parent / "materialization_report.json"
        temporary = report_path.with_name(f".{report_path.name}.tmp")
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(report_path)
        logger.info("Saved materialization report to %s", report_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize final teacher image and tamper-mask bytes into one root.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest-dir", type=Path, default=Path("splits/final_teacher_dataset"))
    parser.add_argument("--data-root", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--split",
        dest="splits",
        nargs="+",
        default=list(TARGET_SPLITS),
        choices=TARGET_SPLITS,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("FINAL_TEACHER_MATERIALIZE_WORKERS", "32")),
    )
    parser.add_argument("--minimum-free-bytes", type=int, default=DEFAULT_MIN_FREE_BYTES)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest_dir = (
        args.manifest_dir if args.manifest_dir.is_absolute() else _PROJECT_ROOT / args.manifest_dir
    )
    output_root = (
        args.output_root if args.output_root.is_absolute() else _PROJECT_ROOT / args.output_root
    )
    roots = [root if root.is_absolute() else _PROJECT_ROOT / root for root in args.data_root]
    report = materialize(
        manifest_dir=manifest_dir,
        roots=roots,
        output_root=output_root,
        splits=tuple(args.splits),
        workers=args.workers,
        minimum_free_bytes=args.minimum_free_bytes,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
