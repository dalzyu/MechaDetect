from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

TARGETS = {"authentic": 3000, "tampered": 3000, "fully_aigc": 6000}


def balanced_select(frame: pd.DataFrame, label: str, target: int, seed: int) -> pd.DataFrame:
    candidates = frame[frame["label"] == label].copy()
    generator_column = "generator_family" if "generator_family" in candidates else "generator"
    if len(candidates) < target:
        raise RuntimeError(f"Need {target} {label} rows, found {len(candidates)}")
    groups = []
    for generator, group in candidates.groupby(generator_column):
        group_seed = (
            seed + int.from_bytes(hashlib.sha256(str(generator).encode()).digest()[:4], "big")
        ) % (2**32 - 1)
        groups.append(group.sample(frac=1.0, random_state=group_seed).index.tolist())
    positions = [0] * len(groups)
    selected_indices = []
    while len(selected_indices) < target:
        made_progress = False
        for group_index, indices in enumerate(groups):
            position = positions[group_index]
            if position < len(indices):
                selected_indices.append(indices[position])
                positions[group_index] += 1
                made_progress = True
                if len(selected_indices) == target:
                    break
        if not made_progress:
            break
    result = candidates.loc[selected_indices].copy()
    if len(result) != target:
        raise RuntimeError(f"Balanced selector produced {len(result)} of {target} {label} rows")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("splits/performance/train.csv"))
    parser.add_argument("--output", type=Path, default=Path("splits/performance/ablation12k.csv"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    frame = pd.read_csv(args.input).fillna("")
    selected = [
        balanced_select(frame, label, target, args.seed) for label, target in TARGETS.items()
    ]
    output = pd.concat(selected, ignore_index=True).sample(frac=1.0, random_state=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(output.groupby(["label", "dataset", "generator_family"]).size())
    print(f"Wrote {len(output)} rows to {args.output}")


if __name__ == "__main__":
    main()
