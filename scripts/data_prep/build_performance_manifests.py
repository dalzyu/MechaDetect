from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from aigc_detector.constants import PROVENANCE_NAMES
from aigc_detector.dataset import parse_provenance
from aigc_detector.manifests import (
    assert_forbidden_demonstration_data_absent,
    assert_no_group_leakage,
    assign_splits,
    difference_hash,
    duplicate_groups,
    file_sha256,
    manifest_digest,
)
from aigc_detector.runtime import load_local_environment

TRAIN_QUOTAS = {
    ("sid", "authentic"): 10_000,
    ("sid", "tampered"): 10_000,
    ("sid", "fully_aigc"): 10_000,
    ("wildfake", "authentic"): 7_500,
    ("wildfake", "fully_aigc"): 7_500,
    ("diffusionforensics", "authentic"): 7_500,
    ("diffusionforensics", "fully_aigc"): 7_500,
}


def dataset_key(value: str) -> str:
    normalized = "".join(character for character in value.lower() if character.isalnum())
    aliases = {
        "sidset": "sid",
        "sid": "sid",
        "wildfake": "wildfake",
        "diffusionforensics": "diffusionforensics",
        "dire": "diffusionforensics",
    }
    if normalized not in aliases:
        raise ValueError(f"Unknown training dataset {value!r}")
    return aliases[normalized]


def _resolve_path(value: str, data_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else data_root / path


def standardize(frame: pd.DataFrame, data_root: Path, compute_hashes: bool) -> pd.DataFrame:
    required = {"image_path", "label", "dataset"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Input metadata missing columns {sorted(missing)}")
    result = frame.fillna("").copy()
    result["label"] = [
        PROVENANCE_NAMES[int(parse_provenance(label, dataset))]
        for label, dataset in zip(result["label"], result["dataset"], strict=True)
    ]
    for column in ("generator", "official_split", "source_image_group", "tamper_mask_path"):
        if column not in result:
            result[column] = ""
    result.loc[result["official_split"] == "", "official_split"] = "train"

    sha_values: list[str] = []
    phash_values: list[int] = []
    for row in result.to_dict(orient="records"):
        path = _resolve_path(str(row["image_path"]), data_root)
        sha = str(row.get("sha256", ""))
        phash = str(row.get("perceptual_hash", ""))
        if not sha or not phash:
            if not compute_hashes:
                raise ValueError("Missing hashes; rerun with --compute-hashes")
            if not path.is_file():
                raise FileNotFoundError(path)
        sha_values.append(sha or file_sha256(path))
        phash_values.append(int(phash, 16) if phash else difference_hash(path))
    result["sha256"] = sha_values
    result["perceptual_hash"] = [f"{value:016x}" for value in phash_values]
    result["duplicate_group"] = duplicate_groups(
        sha_values,
        phash_values,
        [str(value) for value in result["source_image_group"]],
    )
    return assign_splits(result)


def select_training_pool(
    frame: pd.DataFrame, seed: int, allow_shortfall: bool
) -> tuple[pd.DataFrame, list[str]]:
    selected = []
    warnings = []
    training = frame[frame["split"] == "train"].copy()
    training["dataset_key"] = training["dataset"].map(dataset_key)
    for (dataset, label), target in TRAIN_QUOTAS.items():
        candidates = training[(training["dataset_key"] == dataset) & (training["label"] == label)]
        if len(candidates) < target:
            message = f"{dataset}/{label}: need {target}, found {len(candidates)}"
            if not allow_shortfall:
                raise RuntimeError(message)
            warnings.append(message)
        selected.append(candidates.sample(n=min(target, len(candidates)), random_state=seed))
    pool = pd.concat(selected, ignore_index=True) if selected else training.iloc[:0]
    return pool.drop(columns="dataset_key"), warnings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("splits/performance"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compute-hashes", action="store_true")
    parser.add_argument("--allow-shortfall", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    load_local_environment(project_root)
    data_root = Path(os.environ["TECHJAM_DATA_ROOT"])
    frames = [pd.read_csv(path) for path in args.inputs]
    complete = standardize(pd.concat(frames, ignore_index=True), data_root, args.compute_hashes)
    assert_no_group_leakage(complete)
    assert_forbidden_demonstration_data_absent(complete)
    training, warnings = select_training_pool(complete, args.seed, args.allow_shortfall)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {"train": training}
    for split in ("validation", "test", "test_seen", "test_unseen"):
        outputs[split] = complete[complete["split"] == split].copy()
    for name, frame in outputs.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)

    registry = (
        complete[["dataset", "generator_family", "split"]]
        .drop_duplicates()
        .sort_values(["dataset", "generator_family"])
        .to_dict(orient="records")
    )
    report = {
        "seed": args.seed,
        "manifest_sha256": manifest_digest(complete),
        "selected_training_rows": len(training),
        "all_rows": len(complete),
        "warnings": warnings,
        "counts": {
            name: {
                f"{dataset}/{label}": int(count)
                for (dataset, label), count in frame.groupby(["dataset", "label"]).size().items()
            }
            for name, frame in outputs.items()
        },
        "generator_registry": registry,
    }
    with (output_dir / "manifest-report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({key: value for key, value in report.items() if key != "counts"}, indent=2))


if __name__ == "__main__":
    main()
