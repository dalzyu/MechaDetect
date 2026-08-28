from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from aigc_detector.manifests import assert_no_group_leakage


def balanced_sample(
    frame: pd.DataFrame,
    group_column: str,
    total: int,
    seed: int,
) -> pd.DataFrame:
    groups = []
    for group_name, group in frame.groupby(group_column):
        group_seed = (
            seed + int.from_bytes(hashlib.sha256(str(group_name).encode()).digest()[:4], "big")
        ) % (2**32 - 1)
        groups.append(group.sample(frac=1.0, random_state=group_seed).index.tolist())
    positions = [0] * len(groups)
    selected = []
    while len(selected) < min(total, len(frame)):
        progress = False
        for group_index, indices in enumerate(groups):
            position = positions[group_index]
            if position < len(indices):
                selected.append(indices[position])
                positions[group_index] += 1
                progress = True
                if len(selected) >= total:
                    break
        if not progress:
            break
    return frame.loc[selected].copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path("splits/performance"))
    parser.add_argument("--probe-per-class", type=int, default=240)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    frames = {
        name: pd.read_csv(args.directory / f"{name}.csv", low_memory=False).fillna("")
        for name in ("train", "validation", "test", "test_unseen")
    }
    combined = pd.concat(frames.values(), ignore_index=True)
    assert_no_group_leakage(combined)
    synthetic = combined["label"] == "fully_aigc"
    train_families = set(
        combined.loc[(combined["split"] == "train") & synthetic, "generator_family"]
    )
    positives = frames["test_unseen"][frames["test_unseen"]["label"] == "fully_aigc"]
    unseen_families = set(positives["generator_family"])
    if train_families & unseen_families:
        raise RuntimeError("Strict unseen generator families overlap training")
    negatives = combined[
        (combined["label"] == "authentic") & (combined["split"] != "train")
    ].drop_duplicates("duplicate_group")
    full = pd.concat((positives, negatives), ignore_index=True).sample(
        frac=1.0, random_state=args.seed
    )
    full.to_csv(args.directory / "strict_unseen_eval.csv", index=False)

    probe_positives = balanced_sample(
        positives, "generator_family", args.probe_per_class, args.seed
    )
    probe_negatives = balanced_sample(negatives, "dataset", args.probe_per_class, args.seed + 1)
    probe = pd.concat((probe_positives, probe_negatives), ignore_index=True).sample(
        frac=1.0, random_state=args.seed
    )
    probe.to_csv(args.directory / "strict_unseen_probe.csv", index=False)
    print(f"full: {len(full)} ({len(positives)} synthetic, {len(negatives)} authentic)")
    print(f"probe: {len(probe)}")
    print(probe.groupby(["label", "dataset", "generator_family"]).size())


if __name__ == "__main__":
    main()
