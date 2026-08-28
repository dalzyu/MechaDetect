from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from aigc_detector.manifests import (
    assert_no_group_leakage,
    normalize_unseen_split_semantics,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path("splits/performance"))
    args = parser.parse_args()
    paths = [
        args.directory / f"{name}.csv"
        for name in ("train", "validation", "test", "test_seen", "test_unseen")
        if (args.directory / f"{name}.csv").exists()
    ]
    frame = pd.concat((pd.read_csv(path).fillna("") for path in paths), ignore_index=True)
    normalized = normalize_unseen_split_semantics(frame)
    assert_no_group_leakage(normalized)
    for split in ("train", "validation", "test", "test_seen", "test_unseen"):
        subset = normalized[normalized["split"] == split]
        subset.to_csv(args.directory / f"{split}.csv", index=False)
        print(f"{split}: {len(subset)}")


if __name__ == "__main__":
    main()
