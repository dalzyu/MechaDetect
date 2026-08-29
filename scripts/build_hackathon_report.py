from __future__ import annotations

import argparse
import json
from pathlib import Path


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parameters", type=int, default=840_592_640)
    args = parser.parse_args()
    report = json.loads(args.evaluation.read_text(encoding="utf-8"))
    conditions = report["conditions"]
    efficiency = report["efficiency"]

    lines = [
        "# Robust AIGC Detection Evaluation",
        "",
        "## Clean and transformed performance",
        "",
        "| Condition | AI-positive AUROC | AUROC drop | Binary accuracy | "
        "Binary balanced accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in conditions.items():
        lines.append(
            f"| {name} | {percent(values['ai_positive_auroc'])} | "
            f"{percent(values['ai_positive_auroc_drop'])} | "
            f"{percent(values['binary_accuracy'])} | "
            f"{percent(values['binary_balanced_accuracy'])} |"
        )

    lines.extend(
        [
            "",
            "## Generator generalisation",
            "",
            "Each AI generator/edit family is ranked against the same authentic pool.",
            "",
            "| Generator | AUROC | Accuracy |",
            "|---|---:|---:|",
        ]
    )
    for name, values in report["per_ai_generator"].items():
        lines.append(
            f"| {name} | {percent(values['ai_positive_auroc'])} | "
            f"{percent(values['binary_accuracy'])} |"
        )

    def ai_score(row: dict[str, object]) -> float:
        probabilities = row["probabilities"]["clean"]
        return float(probabilities[1])

    rows = report.get("rows", [])
    false_positives = sorted(
        (row for row in rows if row["target"] == "authentic"),
        key=ai_score,
        reverse=True,
    )[:5]
    false_negatives = sorted(
        (row for row in rows if row["target"] != "authentic"),
        key=ai_score,
    )[:5]
    lines.extend(
        [
            "",
            "## Error analysis",
            "",
            "### Highest-risk false positives",
            "",
            "| Image | Target | P(AI-generated or edited) |",
            "|---|---|---:|",
        ]
    )
    for row in false_positives:
        lines.append(
            f"| `{row['image_path']}` | {row['target']} | "
            f"{percent(ai_score(row))} |"
        )
    lines.extend(
        [
            "",
            "### Highest-risk false negatives",
            "",
            "| Image | Generator | P(AI-generated or edited) |",
            "|---|---|---:|",
        ]
    )
    for row in false_negatives:
        lines.append(
            f"| `{row['image_path']}` | {row['generator']} | "
            f"{percent(ai_score(row))} |"
        )

    lines.extend(
        [
            "",
            "## Feasibility",
            "",
            f"- Parameters: {args.parameters:,} "
            f"({args.parameters / 1e9:.3f}B; organizer limit: 2B).",
            f"- Inference: {efficiency['milliseconds_per_image']:.1f} ms/image "
            f"({efficiency['images_per_second']:.2f} images/s).",
            f"- Peak allocated VRAM: {efficiency['peak_allocated_vram_gib']:.2f} GiB.",
            f"- Clean inference views: {efficiency['views_on_clean']}.",
            "",
            "## Limitations and trade-offs",
            "",
            "- Detection is probabilistic and should not be treated as proof of provenance.",
            "- Severe blur and rescaling can erase generator evidence; "
            "condition-specific drops are shown above.",
            "- Three-view mode improves stability but costs roughly three times "
            "the Gemma inference work.",
            "- Thresholds must be calibrated to the deployment's false-positive tolerance.",
            "- The private organizer test set remains unseen and was not used for selection.",
            "",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
