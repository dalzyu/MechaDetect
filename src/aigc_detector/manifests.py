from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

GENERATOR_ALIASES = {
    "stable_diffusion": "stable_diffusion_1",
    "stable_diffusion_v1": "stable_diffusion_1",
    "stable_diffusion_1_4": "stable_diffusion_1",
    "stable_diffusion_1_5": "stable_diffusion_1",
    "sd": "stable_diffusion_1",
    "sd_1": "stable_diffusion_1",
    "sd_v1": "stable_diffusion_1",
    "sd14": "stable_diffusion_1",
    "sd15": "stable_diffusion_1",
    "stable_diffusion_v2": "stable_diffusion_2",
    "sd_2": "stable_diffusion_2",
    "sd_v2": "stable_diffusion_2",
    "dall_e_2": "dalle_2",
    "dall_e_3": "dalle_3",
}


def normalize_generator(value: Any, dataset: str = "") -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    if not normalized:
        dataset_name = re.sub(r"[^a-z0-9]+", "_", dataset.strip().lower()).strip("_")
        return f"{dataset_name or 'unknown'}_unknown"
    return GENERATOR_ALIASES.get(normalized, normalized)


def is_held_out_generator(family: str, seed: int = 42, modulus: int = 5) -> bool:
    digest = hashlib.sha256(f"seed{seed}:{family}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % modulus == 0


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def difference_hash(path: str | Path, size: int = 8) -> int:
    with Image.open(path) as source:
        gray = source.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    value = 0
    for row in range(size):
        offset = row * (size + 1)
        for column in range(size):
            value = (value << 1) | int(pixels[offset + column] > pixels[offset + column + 1])
    return value


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def duplicate_groups(
    sha256_values: list[str],
    perceptual_hashes: list[int],
    source_groups: list[str],
    *,
    max_hamming_distance: int = 4,
) -> list[str]:
    """Cluster exact, linked-source, and near duplicates using four 16-bit LSH bands."""
    if not (len(sha256_values) == len(perceptual_hashes) == len(source_groups)):
        raise ValueError("Duplicate inputs must have equal lengths")
    sets = _DisjointSet(len(sha256_values))
    exact: dict[str, int] = {}
    sources: dict[str, int] = {}
    bands: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, (sha256, phash, source) in enumerate(
        zip(sha256_values, perceptual_hashes, source_groups, strict=True)
    ):
        if sha256 in exact:
            sets.union(index, exact[sha256])
        else:
            exact[sha256] = index
        if source:
            if source in sources:
                sets.union(index, sources[source])
            else:
                sources[source] = index
        candidates: set[int] = set()
        for band in range(4):
            key = (band, (phash >> (band * 16)) & 0xFFFF)
            candidates.update(bands[key])
        for other in candidates:
            if (phash ^ perceptual_hashes[other]).bit_count() <= max_hamming_distance:
                sets.union(index, other)
        for band in range(4):
            bands[(band, (phash >> (band * 16)) & 0xFFFF)].append(index)
    roots = [sets.find(index) for index in range(len(sha256_values))]
    canonical = {
        root: f"duplicate_{position:08d}" for position, root in enumerate(sorted(set(roots)))
    }
    return [canonical[root] for root in roots]


def enforce_group_splits(frame: pd.DataFrame) -> pd.DataFrame:
    priority = {"train": 0, "calibration": 1, "validation": 2, "test": 3, "test_unseen": 4}
    result = frame.copy()
    for _, indices in result.groupby("duplicate_group").groups.items():
        chosen = max(
            (str(result.at[index, "split"]) for index in indices), key=priority.__getitem__
        )
        result.loc[list(indices), "split"] = chosen
    return result


def assign_splits(frame: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    result = frame.copy()
    generators = result["generator"] if "generator" in result else [""] * len(result)
    result["generator_family"] = [
        normalize_generator(generator, dataset)
        for generator, dataset in zip(generators, result["dataset"], strict=True)
    ]
    splits = []
    for row in result.to_dict(orient="records"):
        official = str(row.get("official_split", "train")).strip().lower()
        family = str(row["generator_family"])
        provenance = str(row["label"]).strip().lower()
        dataset = re.sub(r"[^a-z0-9]+", "", str(row["dataset"]).lower())
        forced_adm_training = dataset in {"diffusionforensics", "dire"} and family == "adm"
        if official == "test_unseen":
            splits.append("test_unseen")
        elif (
            provenance in {"fully_aigc", "aigc", "ai"}
            and is_held_out_generator(family, seed)
            and not forced_adm_training
        ):
            splits.append("test_unseen")
        elif official in {"test", "testing"}:
            splits.append("test")
        elif official in {"validation", "val", "dev"}:
            splits.append("validation")
        else:
            splits.append("train")
    result["split"] = splits
    if "duplicate_group" in result:
        result = enforce_group_splits(result)
    return normalize_unseen_split_semantics(result)


def normalize_unseen_split_semantics(frame: pd.DataFrame) -> pd.DataFrame:
    """Reserve test_unseen for synthetic generator families wholly absent from train."""
    result = frame.copy()
    synthetic = result["label"].astype(str).str.lower().isin({"fully_aigc", "aigc", "ai"})
    trained_families = set(
        result.loc[(result["split"] == "train") & synthetic, "generator_family"].astype(str)
    )
    if "duplicate_group" in result:
        for _, indices in result.groupby("duplicate_group").groups.items():
            group_indices = list(indices)
            if not (result.loc[group_indices, "split"] == "test_unseen").any():
                continue
            group_synthetic = synthetic.loc[group_indices]
            group_families = set(
                result.loc[
                    [index for index in group_indices if group_synthetic.loc[index]],
                    "generator_family",
                ]
                .astype(str)
                .tolist()
            )
            if group_families & trained_families:
                result.loc[group_indices, "split"] = "test_seen"
            elif not group_synthetic.any():
                result.loc[group_indices, "split"] = "test"
    else:
        claimed_unseen = result["split"] == "test_unseen"
        seen_synthetic = (
            claimed_unseen & synthetic & result["generator_family"].isin(trained_families)
        )
        result.loc[seen_synthetic, "split"] = "test_seen"
        result.loc[claimed_unseen & ~synthetic, "split"] = "test"
    return result


def manifest_digest(frame: pd.DataFrame) -> str:
    records = frame.sort_values(["dataset", "image_path"]).fillna("").to_dict(orient="records")
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def assert_no_group_leakage(frame: pd.DataFrame) -> None:
    if "duplicate_group" not in frame:
        raise ValueError("Manifest has no duplicate_group column")
    counts = frame.groupby("duplicate_group")["split"].nunique()
    leaking = counts[counts > 1]
    if not leaking.empty:
        raise ValueError(f"{len(leaking)} duplicate groups cross splits")


def assert_forbidden_demonstration_data_absent(frame: pd.DataFrame) -> None:
    """Reject the organizer's COCO-val2017 and DALL-E-Advanced demo data from train."""
    training = frame[frame["split"] == "train"] if "split" in frame else frame
    violations = []
    for index, row in training.iterrows():
        text = " ".join(
            str(row.get(column, ""))
            for column in ("dataset", "generator", "image_path", "original_path", "Category")
        ).lower()
        compact = re.sub(r"[^a-z0-9]+", "", text)
        is_coco_validation = "coco" in compact and "val2017" in compact
        is_dalle_advanced = "dalle" in compact and (
            "advanced" in compact or str(row.get("IsAdvanced", "0")) in {"1", "1.0"}
        )
        if is_coco_validation or is_dalle_advanced:
            violations.append(int(index))
    if violations:
        raise ValueError(
            f"Forbidden organizer demonstration data appears in {len(violations)} training rows"
        )
