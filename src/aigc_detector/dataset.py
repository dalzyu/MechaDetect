from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .constants import (
    PROVENANCE_NAMES,
    SID_LABEL_TO_PROVENANCE,
    Provenance,
    Transformation,
)
from .manifests import normalize_generator
from .preprocessing import RenderPolicy, render_for_model, render_mask_geometry
from .transforms import apply_transform, sample_transform


@dataclass(frozen=True)
class ManifestRecord:
    image_path: Path
    provenance: Provenance
    dataset: str
    generator: str
    tamper_mask_path: Path | None
    ai_positive: int = 0


def parse_provenance(value: Any, dataset: str = "") -> Provenance:
    if isinstance(value, str):
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "real": Provenance.AUTHENTIC,
            "authentic": Provenance.AUTHENTIC,
            "tampered": Provenance.TAMPERED,
            "full_synthetic": Provenance.FULLY_AIGC,
            "fully_synthetic": Provenance.FULLY_AIGC,
            "fully_aigc": Provenance.FULLY_AIGC,
            "aigc": Provenance.FULLY_AIGC,
            "ai": Provenance.FULLY_AIGC,
        }
        if normalized in aliases:
            return aliases[normalized]
        if normalized.isdigit():
            value = int(normalized)
        else:
            raise ValueError(f"Unknown provenance label {value!r}")

    if isinstance(value, int) or (isinstance(value, float) and value.is_integer()):
        numeric = int(value)
        if dataset.lower().replace("-", "_") in {"sid", "sid_set"}:
            return SID_LABEL_TO_PROVENANCE[numeric]
        if numeric in range(len(PROVENANCE_NAMES)):
            return Provenance(numeric)
    raise ValueError(f"Unknown provenance label {value!r} for dataset {dataset!r}")


def load_manifest_frame(manifest_path: str | Path) -> pd.DataFrame:
    """Load one training manifest without silently coercing its format."""
    path = Path(manifest_path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported manifest format {path.suffix!r}; use CSV, Parquet, or JSONL")


HF_SOURCE_MAP: dict[str, dict[str, Any]] = {
    "artic_dataset": {"repo": "links-ads/artic-dataset", "split": "train", "img_key": "image"},
    "flux_reason_6m": {"repo": "LucasFang/FLUX-Reason-6M", "split": "train", "img_key": "image"},
    "flux_cyberpunk_scifi": {"repo": "LucasFang/FLUX-Reason-6M", "split": "train", "img_key": "image"},
    "ideogram_27k": {"repo": "bitmind/ideogram-27k", "split": "train", "img_key": "image"},
    "art_museums_pd": {"repo": "Mitsua/art-museums-pd-440k", "split": "train", "img_key": "jpg"},
    "authentic_classical_figure_art": {"repo": "Mitsua/art-museums-pd-440k", "split": "train", "img_key": "jpg"},
    "gpt_image_edit_1_5m": {"repo": "UCSC-VLAA/GPT-Image-Edit-1.5M", "split": "train", "img_key": "png"},
    "google_nano_banana_edited": {"repo": "Tungtom2004/Google_Nano_Banana_Edited_Images", "split": "train", "img_key": "image"},
    "krea2_wildcards": {"repo": "innofree/krea2-wildcards", "split": "train", "img_key": "image"},
    "nano_banana_pro_gen": {"repo": "FlameF0X/nano-banana-pro-gen-zh-en", "split": "train", "img_key": "image"},
    "ai_meme_macro_overlay": {"repo": "ideepankarsharma2003/AIGeneratedImages_Midjourney", "split": "train", "img_key": "image"},
    "danbooru2026_aigc_wild": {"repo": "nyanko-devs/danbooru2026", "split": "train", "img_key": "webp"},
    "gta5_driving_renders": {"repo": "Chris1/GTA5", "split": "train", "img_key": "image"},
    "sdxl_photoreal_vehicles": {"repo": "diffusers-parti-prompts/sdxl-1.0", "split": "train", "img_key": "image"},
    "game_screenshots_fantasy": {"repo": "badigadiii/game_screenshots_11k", "split": "train", "img_key": "image"},
    "midjourney_v5_images": {"repo": "ehristoforu/midjourney-images", "split": "train", "img_key": "image"},
    "sd3_medium_synths": {"repo": "VincHa/SD3_medium_synths", "split": "train", "img_key": "image"},
    "tartanair2_ue5_cyberpunk": {"repo": "theairlabcmu/tartanair2", "split": "train", "img_key": "image"},
    "open_images_v7": {"repo": "bitmind/open-images-v7", "split": "train", "img_key": "url"},
    "authentic_glamour_portraits": {"repo": "mattymchen/celeba-hq", "split": "train", "img_key": "image"},
    "sintel_blender_open_movie": {"repo": "badigadiii/game_screenshots_11k", "split": "train", "img_key": "image"},
    "manga109_illustrations": {"repo": "hal-utokyo/Manga109-s", "split": "train", "img_key": "image"},
}


def try_fetch_image_from_hub(dataset_name: str, dest_path: Path) -> Path | None:
    """Attempt dynamic on-demand retrieval of an image from its Hugging Face source."""
    if dest_path.is_file():
        return dest_path

    cfg = HF_SOURCE_MAP.get(dataset_name)
    if not cfg:
        return None

    match = re.search(r"_(\d{6})\.", dest_path.name)
    if not match:
        return None
    target_idx = int(match.group(1))

    try:
        from datasets import load_dataset
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        ds = load_dataset(cfg["repo"], split=cfg.get("split", "train"), streaming=True, token=token)
        for i, item in enumerate(ds):
            if i == target_idx:
                img_val = item.get(cfg["img_key"])
                if img_val is None:
                    for k in ["image", "img", "file", "jpg", "png", "webp"]:
                        if k in item and item[k] is not None:
                            img_val = item[k]
                            break
                if img_val is None:
                    return None
                if isinstance(img_val, str) and img_val.startswith("http"):
                    import urllib.request
                    try:
                        req = urllib.request.Request(img_val, headers={"User-Agent": "Mozilla/5.0"})
                        with urllib.request.urlopen(req, timeout=10) as resp:
                            pil_img = Image.open(io.BytesIO(resp.read()))
                    except Exception:
                        return None
                elif isinstance(img_val, Image.Image):
                    pil_img = img_val
                elif isinstance(img_val, bytes):
                    pil_img = Image.open(io.BytesIO(img_val))
                elif isinstance(img_val, dict) and "bytes" in img_val:
                    pil_img = Image.open(io.BytesIO(img_val["bytes"]))
                elif isinstance(img_val, dict) and "path" in img_val:
                    pil_img = Image.open(img_val["path"])
                else:
                    return None

                dest_path.parent.mkdir(parents=True, exist_ok=True)
                pil_img.convert("RGB").save(dest_path, format="JPEG", quality=95)
                return dest_path
            if i > target_idx:
                break
    except Exception:
        return None
    return None


def verify_materialization(
    manifest_path: str | Path,
    *,
    data_root: str | Path | None = None,
    limit: int | None = None,
    allow_missing: bool = False,
) -> dict[str, int]:
    """Fail closed unless every selected image and declared mask is readable."""
    frame = load_manifest_frame(manifest_path)
    if limit is not None:
        frame = frame.iloc[:limit]
    root = Path(data_root) if data_root is not None else Path(manifest_path).parent
    missing_images = 0
    missing_masks = 0
    for row in frame.to_dict(orient="records"):
        image_path = Path(str(row["image_path"]).replace("\\", "/"))
        if not image_path.is_absolute():
            image_path = root / image_path
        if not image_path.is_file():
            missing_images += 1
        mask_value = row.get("tamper_mask_path", "")
        if not pd.isna(mask_value) and str(mask_value).strip():
            mask_path = Path(str(mask_value).replace("\\", "/"))
            if not mask_path.is_absolute():
                mask_path = root / mask_path
            if not mask_path.is_file():
                missing_masks += 1
    result = {
        "rows": len(frame),
        "missing_images": missing_images,
        "missing_masks": missing_masks,
    }
    if (missing_images or missing_masks) and not allow_missing:
        raise FileNotFoundError(
            f"Manifest is not fully materialized: {result}"
        )
    return result


class PairedImageDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        data_root: str | Path | None = None,
        seed: int = 42,
        transform_families: tuple[Transformation, ...] | None = None,
        render_policy: RenderPolicy | str = RenderPolicy.SQUARE_JPEG95,
        runtime_fetch: bool = True,
        allow_missing: bool = True,
    ) -> None:
        frame = load_manifest_frame(manifest_path)
        required = {"image_path", "label", "dataset"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

        root = Path(data_root) if data_root is not None else Path(manifest_path).parent
        self.records = []
        for row in frame.to_dict(orient="records"):
            path = Path(str(row["image_path"]).replace("\\", "/"))
            if not path.is_absolute():
                path = root / path
            mask_value = row.get("tamper_mask_path", "")
            raw_mask_path = "" if pd.isna(mask_value) else str(mask_value).strip()
            mask_path = Path(raw_mask_path.replace("\\", "/")) if raw_mask_path else None
            if mask_path is not None and not mask_path.is_absolute():
                mask_path = root / mask_path
            prov = parse_provenance(row["label"], str(row["dataset"]))
            raw_ai_pos = row.get("ai_positive")
            if raw_ai_pos is not None and not pd.isna(raw_ai_pos) and str(raw_ai_pos).strip() != "":
                ai_positive_val = int(raw_ai_pos)
            else:
                ai_positive_val = 0 if prov == Provenance.AUTHENTIC else 1
            self.records.append(
                ManifestRecord(
                    image_path=path,
                    provenance=prov,
                    dataset=str(row["dataset"]),
                    generator=normalize_generator(row.get("generator", ""), str(row["dataset"])),
                    tamper_mask_path=mask_path,
                    ai_positive=ai_positive_val,
                )
            )
        self.seed = seed
        self.epoch = 0
        self.transform_families = transform_families
        self.render_policy = RenderPolicy(render_policy)
        self.runtime_fetch = runtime_fetch
        self.allow_missing = allow_missing
        self._fallback_by_key: dict[tuple[int, int], list[ManifestRecord]] = {}
        for rec in self.records:
            if rec.image_path.is_file():
                self._fallback_by_key.setdefault((int(rec.provenance), rec.ai_positive), []).append(rec)

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image_path = record.image_path
        tamper_mask_path = record.tamper_mask_path

        if not image_path.is_file():
            if self.runtime_fetch:
                fetched = try_fetch_image_from_hub(record.dataset, image_path)
                if fetched is not None and fetched.is_file():
                    image_path = fetched
            if not image_path.is_file() and self.allow_missing:
                candidates = self._fallback_by_key.get((int(record.provenance), record.ai_positive))
                if candidates:
                    fallback_rec = candidates[index % len(candidates)]
                    image_path = fallback_rec.image_path
                    tamper_mask_path = fallback_rec.tamper_mask_path

        with Image.open(image_path) as source:
            raw = source.convert("RGB").copy()

        rng = Random(self.seed + self.epoch * max(1, len(self)) + index)
        original = render_for_model(raw, self.render_policy, rng=rng)

        rendered_mask = None
        if tamper_mask_path is not None and tamper_mask_path.is_file() and record.provenance is not Provenance.FULLY_AIGC:
            with Image.open(tamper_mask_path) as source_mask:
                raw_mask = source_mask.convert("L").copy()
            rendered_mask = render_mask_geometry(
                raw_mask,
                self.render_policy,
                image_size=raw.size,
                rendered_size=original.size,
            )
        transform = (
            sample_transform(rng, self.transform_families)
            if self.transform_families
            else None
        )
        transformed = (
            apply_transform(original, transform, rng, mask=rendered_mask)
            if transform
            else original
        )

        return {
            "image_path": str(record.image_path),
            "original": original,
            "transformed": transformed,
            "provenance": int(record.provenance),
            "ai_positive": record.ai_positive,
            "dataset": record.dataset,
            "generator": record.generator,
            "mask": rendered_mask,
            "transform": None if transform is None else transform.family.name.lower(),
        }


def collate_pairs(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image_path": [sample["image_path"] for sample in samples],
        "original": [sample["original"] for sample in samples],
        "transformed": [sample["transformed"] for sample in samples],
        "provenance": torch.tensor([sample["provenance"] for sample in samples], dtype=torch.long),
        "ai_positive": torch.tensor([sample["ai_positive"] for sample in samples], dtype=torch.float),
        "dataset": [sample["dataset"] for sample in samples],
        "generator": [sample["generator"] for sample in samples],
        "mask": [sample["mask"] for sample in samples],
        "transform": [sample["transform"] for sample in samples],
    }
