from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from aigc_detector.dataset import load_manifest_frame


def build_composite_ood(
    test_unseen: Path,
    canonical_test: Path,
    output: Path,
) -> pd.DataFrame:
    """Pair every unseen AI-positive row with every canonical test negative."""
    unseen = load_manifest_frame(test_unseen).fillna("")
    test = load_manifest_frame(canonical_test).fillna("")
    if "ai_positive" not in unseen or "ai_positive" not in test:
        raise ValueError("Both manifests must contain the explicit ai_positive column")
    positives = unseen[unseen["ai_positive"].astype(int) == 1].copy()
    negatives = test[test["ai_positive"].astype(int) == 0].copy()
    if positives.empty or negatives.empty:
        raise ValueError("Composite OOD requires non-empty positive and negative cohorts")
    positives["ood_cohort"] = "test_unseen_ai_positive"
    negatives["ood_cohort"] = "canonical_test_authentic_negative"
    result = pd.concat((positives, negatives), ignore_index=True)
    result["split"] = "ood_composite"
    result.to_csv(output, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the locked composite OOD evaluation cohort.")
    parser.add_argument("--test-unseen", type=Path, required=True)
    parser.add_argument("--canonical-test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = build_composite_ood(args.test_unseen, args.canonical_test, args.output)
    print(
        f"composite_ood rows={len(result)} positives={int(result.ai_positive.sum())} "
        f"negatives={int((result.ai_positive == 0).sum())} output={args.output}"
    )


if __name__ == "__main__":
    main()
