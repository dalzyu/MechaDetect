from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from aigc_detector.metrics import binary_auprc, binary_auroc


def audit(manifest_path: Path) -> dict:
    frame = pd.read_csv(manifest_path)
    binary = frame[frame.label.isin(["authentic", "fully_aigc"])].copy()
    target = torch.tensor((binary.label == "fully_aigc").astype(int).to_numpy())
    format_score = torch.tensor((binary.file_format.str.upper() == "PNG").astype(float).to_numpy())
    square_score = torch.tensor(
        (((binary.width / binary.height) - 1.0).abs() < 0.01).astype(float).to_numpy()
    )
    return {
        "manifest": str(manifest_path),
        "rows": len(frame),
        "format_by_label": pd.crosstab(frame.label, frame.file_format).to_dict(),
        "square_by_label": pd.crosstab(
            frame.label, ((frame.width / frame.height) - 1.0).abs() < 0.01
        ).to_dict(),
        "authentic_vs_fully_aigc": {
            "format_png_rule_auroc": binary_auroc(target, format_score),
            "format_png_rule_auprc": binary_auprc(target, format_score),
            "square_rule_auroc": binary_auroc(target, square_score),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.manifest), indent=2, default=str))


if __name__ == "__main__":
    main()
