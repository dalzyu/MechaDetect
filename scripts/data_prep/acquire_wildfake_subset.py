from __future__ import annotations

import argparse
import hashlib
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import fsspec
import pandas as pd
from modelscope_hub import HubApi
from PIL import Image

from aigc_detector.runtime import load_local_environment

REPOSITORY = "hy2628982280/WildFake"
RESOLVE_ROOT = f"https://modelscope.cn/datasets/{REPOSITORY}/resolve/master"


@dataclass(frozen=True)
class Source:
    name: str
    archive: str
    labels: str
    provenance: str
    train: int
    validation: int
    test_unseen: int = 0


SOURCES = (
    Source(
        "ddim",
        "Images/Diffusion_based/DDIM.zip",
        "label_csv_files/ddim.csv",
        "fully_aigc",
        3750,
        500,
    ),
    Source(
        "ddpm",
        "Images/Diffusion_based/DDPM.zip",
        "label_csv_files/ddpm.csv",
        "fully_aigc",
        3750,
        500,
    ),
    Source(
        "vqdm",
        "Images/Diffusion_based/VQDM.zip",
        "label_csv_files/vqdm.csv",
        "fully_aigc",
        0,
        0,
        1000,
    ),
    Source("afhq", "Images/Real/afhq.zip", "label_csv_files/real_afhq.csv", "authentic", 3750, 500),
    Source(
        "celebahq",
        "Images/Real/celebahq.zip",
        "label_csv_files/real_celebahq.csv",
        "authentic",
        3750,
        500,
    ),
)


def _source_seed(name: str, seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:4], "big")


def _candidate_names(csv_path: str, source: Source) -> list[str]:
    path = csv_path.replace("\\", "/").removeprefix("./")
    candidates = [path]
    for prefix in ("Images/", "Diffusion_based/", "Real/"):
        if path.startswith(prefix):
            candidates.append(path.removeprefix(prefix))
    if source.name == "afhq":
        candidates.extend(
            candidate.replace("afhq/afhq/", "afhq/afhq_v2/") for candidate in list(candidates)
        )
    return candidates


def _map_zip_entries(
    selected: pd.DataFrame, archive: zipfile.ZipFile, source: Source
) -> list[tuple[pd.Series, zipfile.ZipInfo]]:
    files = [info for info in archive.infolist() if not info.is_dir()]
    exact = {info.filename.replace("\\", "/").lower(): info for info in files}
    basenames: dict[str, list[zipfile.ZipInfo]] = {}
    for info in files:
        basenames.setdefault(Path(info.filename).name.lower(), []).append(info)
    mapped = []
    for _, row in selected.iterrows():
        info = next(
            (
                exact[candidate.lower()]
                for candidate in _candidate_names(str(row["Image_path"]), source)
                if candidate.lower() in exact
            ),
            None,
        )
        if info is None:
            matches = basenames.get(Path(str(row["Image_path"])).name.lower(), [])
            if len(matches) == 1:
                info = matches[0]
        if info is None:
            raise KeyError(f"Could not find {row['Image_path']!r} in {source.archive}")
        mapped.append((row, info))
    return sorted(mapped, key=lambda item: item[1].header_offset)


def _select_rows(frame: pd.DataFrame, source: Source, seed: int) -> pd.DataFrame:
    total = source.train + source.validation + source.test_unseen
    if len(frame) < total:
        raise RuntimeError(f"{source.name}: need {total} rows, found {len(frame)}")
    selected = frame.sample(n=total, random_state=_source_seed(source.name, seed)).copy()
    selected["official_split"] = (
        ["train"] * source.train
        + ["validation"] * source.validation
        + ["test_unseen"] * source.test_unseen
    )
    return selected


def _extract_source(
    source: Source,
    labels_path: Path,
    data_root: Path,
    seed: int,
) -> list[dict[str, object]]:
    selected = _select_rows(pd.read_csv(labels_path), source, seed)
    url = f"{RESOLVE_ROOT}/{source.archive}"
    remote = fsspec.open(url, "rb", block_size=8 * 1024 * 1024, cache_type="readahead").open()
    rows = []
    try:
        with zipfile.ZipFile(remote) as archive:
            mapped = _map_zip_entries(selected, archive, source)
            for index, (record, info) in enumerate(mapped, start=1):
                payload = archive.read(info)
                digest = hashlib.sha256(payload).hexdigest()
                with Image.open(io.BytesIO(payload)) as image:
                    width, height = image.size
                    image_format = str(image.format or "unknown")
                    image.verify()
                suffix = Path(info.filename).suffix.lower() or ".img"
                split = str(record["official_split"])
                relative = (
                    Path("wildfake_subset")
                    / split
                    / source.provenance
                    / source.name
                    / f"{digest[:20]}{suffix}"
                )
                destination = data_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    temporary = destination.with_suffix(destination.suffix + ".partial")
                    temporary.write_bytes(payload)
                    temporary.replace(destination)
                rows.append(
                    {
                        "image_path": relative.as_posix(),
                        "label": source.provenance,
                        "dataset": "WildFake",
                        "official_split": split,
                        "img_id": digest[:20],
                        "source_image_group": "",
                        "generator": source.name,
                        "width": width,
                        "height": height,
                        "file_format": image_format,
                        "sha256": digest,
                        "tamper_mask_path": "",
                    }
                )
                if index % 100 == 0 or index == len(mapped):
                    print(f"{source.name}: {index}/{len(mapped)}", flush=True)
    finally:
        remote.close()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sources", nargs="*", choices=[source.name for source in SOURCES])
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    load_local_environment(project_root)
    import os

    data_root = Path(os.environ["TECHJAM_DATA_ROOT"])
    metadata_root = data_root / "wildfake_metadata"
    metadata_root.mkdir(parents=True, exist_ok=True)
    api = HubApi()
    chosen = [source for source in SOURCES if not args.sources or source.name in args.sources]
    output = project_root / "metadata" / "wildfake_subset.csv"
    existing = pd.read_csv(output).fillna("") if output.exists() else pd.DataFrame()
    chosen_names = {source.name for source in chosen}
    if not existing.empty:
        existing = existing[~existing["generator"].isin(chosen_names)]
    rows = existing.to_dict(orient="records")
    for source in chosen:
        labels_path = metadata_root / source.labels
        if not labels_path.exists():
            api.download_file(REPOSITORY, "dataset", source.labels, local_dir=metadata_root)
        rows.extend(_extract_source(source, labels_path, data_root, args.seed))
        pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
