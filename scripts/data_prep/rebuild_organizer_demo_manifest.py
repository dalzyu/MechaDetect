from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd

from aigc_detector.runtime import load_local_environment


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    load_local_environment(project_root)
    data_root = Path(os.environ["TECHJAM_DATA_ROOT"])
    unique = pd.read_csv(project_root / "metadata" / "organizer_demo.csv", low_memory=False)
    coco = unique[unique["label"] == "authentic"].copy().sort_values("image_path")
    dalle_unique = unique[unique["label"] == "fully_aigc"].copy()
    labels = pd.read_csv(data_root / "wildfake_metadata" / "label_csv_files" / "dalle3.csv")

    row_by_md5 = {}
    for row in dalle_unique.to_dict(orient="records"):
        path = data_root / str(row["image_path"])
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        row_by_md5[digest] = row
    rebuilt = []
    for source_path in labels["Image_path"]:
        basename_hash = Path(str(source_path)).stem.lower()
        if basename_hash not in row_by_md5:
            raise RuntimeError(f"No local DALL-E content for {source_path}")
        row = dict(row_by_md5[basename_hash])
        row["img_id"] = str(source_path)
        row["source_image_group"] = basename_hash
        row["original_path"] = str(source_path)
        rebuilt.append(row)
    dalle = pd.DataFrame(rebuilt)
    if len(dalle) != 8843:
        raise RuntimeError(f"Expected 8843 DALL-E rows, rebuilt {len(dalle)}")

    upstream = pd.concat((coco, dalle), ignore_index=True)
    document_count = pd.concat((coco.iloc[:4998], dalle), ignore_index=True)
    upstream.to_csv(project_root / "metadata" / "organizer_demo_upstream_full.csv", index=False)
    document_count.to_csv(
        project_root / "metadata" / "organizer_demo_document_count.csv", index=False
    )
    print(
        f"upstream_full={len(upstream)} (COCO={len(coco)}, DALL-E={len(dalle)})",
        flush=True,
    )
    print(
        f"document_count={len(document_count)} (COCO={len(coco.iloc[:4998])}, DALL-E={len(dalle)})",
        flush=True,
    )
    print(
        "Note: the organizer document does not identify the two omitted COCO files; "
        "document_count uses the first 4998 lexicographically.",
        flush=True,
    )


if __name__ == "__main__":
    main()
