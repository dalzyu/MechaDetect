#!/usr/bin/env python3
"""Combined Track 5 Hugging Face Dataset Builder.

Consolidates multimodal forensic sources into a canonical HF-ready dataset package
conforming to data/COMBINED_DATASET_SCHEMA.md.

Binary Task Semantics:
  authentic -> 0 (ai_positive = 0)
  human tampered (e.g. meme captions) -> 1 (ai_positive = 0)
  AI tampered (e.g. inpainting, instruction edits) -> 1 (ai_positive = 1)
  fully_aigc -> 2 (ai_positive = 1)

Safety:
  Default mode is local manifest build / dry-run. Network upload to Hugging Face
  is strictly opt-in via explicit --push-to-hub and requires --repo-id.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import datetime
import hashlib
import io
import json
import logging
import os
import shutil
import random
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd
from PIL import Image

# Ensure repository root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from aigc_detector.manifests import (
    assert_forbidden_demonstration_data_absent,
    assert_no_group_leakage,
    difference_hash,
    enforce_group_splits,
    file_sha256,
    is_held_out_generator,
    manifest_digest,
    normalize_generator,
)
from aigc_detector.runtime import load_local_environment

try:
    from huggingface_hub import get_token
    _cached_hf_token = get_token()
    if _cached_hf_token and not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = _cached_hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = _cached_hf_token
except Exception:
    pass
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("combined_hf_builder")

# ==============================================================================
# Canonical Schema Columns (matching data/COMBINED_DATASET_SCHEMA.md)
# ==============================================================================

# Core training-compatible manifest columns (Section 3)
MANIFEST_CORE_COLUMNS = [
    "image_path",
    "label",
    "dataset",
    "official_split",
    "generator",
    "manipulation_family",
    "source_image_group",
    "width",
    "height",
    "file_format",
    "tamper_mask_path",
]

# Extended metadata columns (Section 4)
EXTENDED_METADATA_COLUMNS = [
    "source_url",
    "external_id",
    "generator_family",
    "generator_version",
    "prompt",
    "created_at",
    "sha256",
    "perceptual_hash",
    "quality_score",
    "provenance_confidence",
    "redistribution_mode",
    "origin_license",
    "license_url",
    "attribution",
    "selection_reason",
    "forbidden_demo_checked",
]

# Evaluation, task binary target, and partitioning columns
TASK_AND_SPLIT_COLUMNS = [
    "ai_positive",
    "split",
    "duplicate_group",
    "provenance",
]

ALL_SCHEMA_COLUMNS = MANIFEST_CORE_COLUMNS + EXTENDED_METADATA_COLUMNS + TASK_AND_SPLIT_COLUMNS

# Known deleted placeholder hashes to reject immediately
KNOWN_PLACEHOLDER_SHA256: Set[str] = {
    "9b5936f4006146e4e1e9025b474c02863c0b5614132ad40db4b925a10e8bfbb9",  # Imgur 161x81 placeholder
}

# ==============================================================================
# Quality Checks & Hashing
# ==============================================================================

@dataclass
class QualityResult:
    passed: bool
    reason: str
    width: int
    height: int
    file_format: str
    sha256: str
    perceptual_hash: str
    quality_score: float


def compute_image_dhash(image: Image.Image, size: int = 8) -> int:
    """Compute difference hash (dHash) directly on a PIL Image."""
    gray = image.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    if hasattr(gray, "get_flattened_data"):
        pixels = list(gray.get_flattened_data())
    else:
        pixels = list(gray.getdata())  # type: ignore[attr-defined]

    value = 0
    for row in range(size):
        offset = row * (size + 1)
        for col in range(size):
            value = (value << 1) | int(pixels[offset + col] > pixels[offset + col + 1])
    return value
def compute_image_phash(image: Image.Image, size: int = 8) -> int:
    """Compute a compact DCT-free perceptual hash from luminance medians."""
    gray = np.asarray(image.convert("L").resize((size, size), Image.Resampling.LANCZOS))
    median = float(np.median(gray))
    value = 0
    for pixel in gray.flat:
        value = (value << 1) | int(float(pixel) >= median)
    return value



def evaluate_image_quality(
    image: Image.Image,
    raw_bytes: Optional[bytes] = None,
    *,
    min_short_edge: int = 160,
    min_area: int = 64_000,
    min_bytes: int = 2048,
    min_quality_score: float = 0.1,
) -> QualityResult:
    """Run rigorous forensic quality checks on an image.

    Checks:
      1. Raw byte length >= min_bytes
      2. Static image format (no animated GIF, MP4, WebM)
      3. Valid PIL decode and RGB conversion
      4. Dimensions: min(w, h) >= min_short_edge, w * h >= min_area
      5. Blank / solid color check (extrema difference > 0, dynamic span >= 5)
      6. Known placeholder hash rejection
      7. Heuristic quality score calculation
    """
    if raw_bytes is not None and len(raw_bytes) < min_bytes:
        return QualityResult(
            passed=False,
            reason=f"file_bytes_{len(raw_bytes)}_below_min_{min_bytes}",
            width=0,
            height=0,
            file_format="",
            sha256="",
            perceptual_hash="",
            quality_score=0.0,
        )

    # File format check
    detected_format = str(getattr(image, "format", "") or "JPEG").upper()
    if detected_format in {"GIF", "WEBM", "MP4"}:
        return QualityResult(
            passed=False,
            reason=f"unsupported_animated_format_{detected_format}",
            width=image.width if hasattr(image, "width") else 0,
            height=image.height if hasattr(image, "height") else 0,
            file_format=detected_format,
            sha256="",
            perceptual_hash="",
            quality_score=0.0,
        )

    # RGB conversion check
    try:
        rgb = image.convert("RGB")
    except Exception as exc:
        return QualityResult(
            passed=False,
            reason=f"rgb_conversion_failure_{exc}",
            width=0,
            height=0,
            file_format=detected_format,
            sha256="",
            perceptual_hash="",
            quality_score=0.0,
        )

    width, height = rgb.size
    short_edge = min(width, height)
    area = width * height

    if short_edge < min_short_edge:
        return QualityResult(
            passed=False,
            reason=f"short_edge_{short_edge}_below_min_{min_short_edge}",
            width=width,
            height=height,
            file_format=detected_format,
            sha256="",
            perceptual_hash="",
            quality_score=0.0,
        )

    if area < min_area:
        return QualityResult(
            passed=False,
            reason=f"area_{area}_below_min_{min_area}",
            width=width,
            height=height,
            file_format=detected_format,
            sha256="",
            perceptual_hash="",
            quality_score=0.0,
        )

    # Blank / uniform image detection
    gray = rgb.convert("L")
    extrema = gray.getextrema()
    if extrema[0] == extrema[1]:
        return QualityResult(
            passed=False,
            reason="solid_color_blank_image",
            width=width,
            height=height,
            file_format=detected_format,
            sha256="",
            perceptual_hash="",
            quality_score=0.0,
        )

    span = extrema[1] - extrema[0]
    if span < 5:
        return QualityResult(
            passed=False,
            reason=f"low_dynamic_range_span_{span}",
            width=width,
            height=height,
            file_format=detected_format,
            sha256="",
            perceptual_hash="",
            quality_score=0.0,
        )

    # Compute hashes
    if raw_bytes is not None:
        sha256 = hashlib.sha256(raw_bytes).hexdigest()
    else:
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=95)
        sha256 = hashlib.sha256(buf.getvalue()).hexdigest()

    if sha256 in KNOWN_PLACEHOLDER_SHA256:
        return QualityResult(
            passed=False,
            reason=f"known_placeholder_sha256_{sha256[:16]}",
            width=width,
            height=height,
            file_format=detected_format,
            sha256=sha256,
            perceptual_hash="",
            quality_score=0.0,
        )

    dhash_val = compute_image_dhash(rgb)
    phash_str = f"{dhash_val:016x}"

    # Heuristic quality score: [0.0, 1.0]
    res_component = min(1.0, area / (1024 * 1024))
    aspect_ratio = max(width / max(1, height), height / max(1, width))
    aspect_component = max(0.0, 1.0 - (aspect_ratio - 1.0) / 4.0)
    dynamic_component = span / 255.0
    quality_score = round(
        0.4 * res_component + 0.3 * aspect_component + 0.3 * dynamic_component, 4
    )

    if quality_score < min_quality_score:
        return QualityResult(
            passed=False,
            reason=f"quality_score_{quality_score}_below_min_{min_quality_score}",
            width=width,
            height=height,
            file_format=detected_format,
            sha256=sha256,
            perceptual_hash=phash_str,
            quality_score=quality_score,
        )

    return QualityResult(
        passed=True,
        reason="passed",
        width=width,
        height=height,
        file_format=detected_format,
        sha256=sha256,
        perceptual_hash=phash_str,
        quality_score=quality_score,
    )


def is_forbidden_demonstration(text_or_row: Any) -> bool:
    """Check whether record matches organizer exclusion:

    - COCO val2017 (4,998 organizer evaluation samples)
    - WildFake DALL-E Advanced / dalle3.csv (8,843 demonstration samples)
    """
    if isinstance(text_or_row, dict):
        text = " ".join(
            str(text_or_row.get(col, ""))
            for col in (
                "dataset",
                "generator",
                "image_path",
                "source_url",
                "external_id",
                "prompt",
                "Category",
            )
        ).lower()
        if str(text_or_row.get("IsAdvanced", "0")) in {"1", "1.0"}:
            return True
    else:
        text = str(text_or_row).lower()

    compact = re.sub(r"[^a-z0-9]+", "", text)
    if "coco" in compact and "val2017" in compact:
        return True
    if "dalle" in compact and "advanced" in compact:
        return True
    if "dalle3csv" in compact:
        return True
    return False


# ==============================================================================
# Source Registry
# ==============================================================================


@dataclass
class SourceConfig:
    name: str
    source_type: str  # 'huggingface', 'local_csv', 'reference_only'
    provenance: str  # 'authentic', 'tampered', 'fully_aigc', or 'mixed'
    generator: str
    generator_family: str
    generator_version: str = ""
    manipulation_family: str = ""
    default_cap: int = 10_000
    hf_repo: Optional[str] = None
    hf_split: str = "train"
    local_path: Optional[str] = None
    redistribution_mode: str = "reference_only"  # 'embed_bytes' or 'reference_only'
    origin_license: str = "Research Only"
    license_url: str = ""
    attribution: str = ""
    source_url: str = ""
    selection_reason: str = ""
    provenance_confidence: str = "high"  # 'high', 'medium', 'low'
    is_human_tampered: bool = False  # Human edits (e.g. meme captions) -> ai_positive = 0
    notes: str = ""


# Unified canonical registry covering all curated cohorts
SOURCE_REGISTRY: Dict[str, SourceConfig] = {
    # --------------------------------------------------------------------------
    # 1. Authentic - Fine Art / Museum Public Domain
    # --------------------------------------------------------------------------
    "artic_dataset": SourceConfig(
        name="artic_dataset",
        source_type="huggingface",
        hf_repo="links-ads/artic-dataset",
        hf_split="train",
        provenance="authentic",
        generator="authentic",
        generator_family="authentic",
        generator_version="real",
        redistribution_mode="embed_bytes",
        origin_license="CC0 / Public Domain",
        license_url="https://www.artic.edu/open-access/open-access-images",
        attribution="Art Institute of Chicago Open Access",
        source_url="https://huggingface.co/datasets/links-ads/artic-dataset",
        selection_reason="High-resolution fine art and museum negatives for domain breadth",
        provenance_confidence="high",
        default_cap=500,
    ),
    "art_museums_pd": SourceConfig(
        name="art_museums_pd",
        source_type="huggingface",
        hf_repo="Mitsua/art-museums-pd-440k",
        hf_split="train",
        provenance="authentic",
        generator="authentic",
        generator_family="authentic",
        default_cap=8_000,
        redistribution_mode="reference_only",
        origin_license="Public Domain",
        license_url="https://huggingface.co/datasets/Mitsua/art-museums-pd-440k",
        attribution="Mitsua Art Museums Public Domain (National Gallery, Met, Rijksmuseum)",
        source_url="https://huggingface.co/datasets/Mitsua/art-museums-pd-440k",
        selection_reason="Historical paintings and prints providing diverse high-frequency textures",
        provenance_confidence="high",
    ),
    # --------------------------------------------------------------------------
    # 2. Authentic - Hard Negative Memes & Internet Media (Target 4,100)
    # --------------------------------------------------------------------------
    "gmorinan_memes": SourceConfig(
        name="gmorinan_memes",
        source_type="local_csv",
        local_path="sources/gmorinan-memes/metadata.csv",
        provenance="tampered",
        generator="human",
        generator_family="human-authored-meme",
        generator_version="pre-2019",
        manipulation_family="human_caption",
        default_cap=500,
        redistribution_mode="reference_only",
        origin_license="CC0 dataset card; image rights retained by creators",
        license_url="https://www.kaggle.com/datasets/gmorinan/memes-classified-and-labelled",
        attribution="George Morinan / Reddit and Imgur uploaders",
        source_url="https://www.kaggle.com/datasets/gmorinan/memes-classified-and-labelled",
        selection_reason="2018 human-authored memes filtered by timestamp < 1546300800",
        provenance_confidence="high",
        is_human_tampered=True,
    ),
    "dank_learning_templates": SourceConfig(
        name="dank_learning_templates",
        source_type="local_directory",
        local_path="sources/dank-learning/im2txt/memes",
        provenance="tampered",
        generator="human",
        generator_family="human-authored-meme",
        generator_version="2018",
        manipulation_family="human_caption",
        default_cap=500,
        redistribution_mode="reference_only",
        origin_license="MIT code / source image rights retained",
        license_url="https://github.com/alexis-mignon/dank-learning",
        attribution="Dank Learning dataset repository",
        source_url="https://github.com/alexis-mignon/dank-learning",
        selection_reason="Pre-2019 blank meme templates as hard negatives",
        provenance_confidence="medium",
        is_human_tampered=True,
    ),
    "imgflip_memes": SourceConfig(
        name="imgflip_memes",
        source_type="reference_only",
        provenance="tampered",  # Human template meme
        generator="human",
        generator_family="human-authored-meme",
        generator_version="pre-2019",
        manipulation_family="human_caption",
        default_cap=1_500,
        redistribution_mode="reference_only",
        origin_license="Research Only",
        attribution="ImgFlip575K Dataset",
        source_url="https://github.com/schesa/ImgFlip575K_Dataset",
        selection_reason="Hard negative meme templates (100 templates, cap 15) with age evidence",
        provenance_confidence="medium",
        is_human_tampered=True,  # human tampered -> ai_positive = 0
    ),
    "multioff_memes": SourceConfig(
        name="multioff_memes",
        source_type="reference_only",
        provenance="tampered",
        generator="human",
        generator_family="human-authored-meme",
        generator_version="pre-2019",
        manipulation_family="human_caption",
        default_cap=300,
        redistribution_mode="reference_only",
        origin_license="CC BY-NC-SA 4.0 upstream",
        attribution="MultiOFF Dataset / SIZZLE",
        source_url="https://github.com/bharathichezhiyan/Multimodal-Meme-Classification-Identifying-Offensive-Content-in-Image-and-Text",
        selection_reason="Multimodal meme benchmark hard negatives (six groups, cap 50, 30/10/10 split)",
        provenance_confidence="high",
        is_human_tampered=True,  # human tampered -> ai_positive = 0
    ),
    # --------------------------------------------------------------------------
    # 3. Authentic - Natural Scene Photography
    # --------------------------------------------------------------------------
    "open_images_v7": SourceConfig(
        name="open_images_v7",
        source_type="huggingface",
        hf_repo="bitmind/open-images-v7",
        hf_split="train",
        provenance="authentic",
        generator="authentic",
        generator_family="authentic",
        default_cap=10_000,
        redistribution_mode="reference_only",
        origin_license="CC BY 2.0 / CC BY 4.0",
        license_url="https://huggingface.co/datasets/bitmind/open-images-v7",
        attribution="Google Open Images V7 (BitMind mirror)",
        source_url="https://huggingface.co/datasets/bitmind/open-images-v7",
        selection_reason="Camera capture authentic reference images across diverse scene categories",
        provenance_confidence="high",
    ),
    # --------------------------------------------------------------------------
    # 4. Tampered - Modern Instruction & Multimodal AI Edits (ai_positive = 1)
    # --------------------------------------------------------------------------
    "gpt_image_edit_1_5m": SourceConfig(
        name="gpt_image_edit_1_5m",
        source_type="huggingface",
        hf_repo="UCSC-VLAA/GPT-Image-Edit-1.5M",
        hf_split="train",
        provenance="tampered",
        generator="gpt_image_1",
        generator_family="gpt_image",
        generator_version="1.0",
        manipulation_family="instruction_edit",
        default_cap=5_000,
        redistribution_mode="reference_only",
        origin_license="CC-BY-NC 4.0",
        license_url="https://huggingface.co/datasets/UCSC-VLAA/GPT-Image-Edit-1.5M",
        attribution="UCSC-VLAA GPT-Image-Edit",
        source_url="https://huggingface.co/datasets/UCSC-VLAA/GPT-Image-Edit-1.5M",
        selection_reason="Paired instruction-edited tampered images for localized forensic detection",
        provenance_confidence="high",
        is_human_tampered=False,  # AI edit -> ai_positive = 1
    ),
    "google_nano_banana_edited": SourceConfig(
        name="google_nano_banana_edited",
        source_type="huggingface",
        hf_repo="Tungtom2004/Google_Nano_Banana_Edited_Images",
        hf_split="train",
        provenance="tampered",
        generator="nano_banana",
        generator_family="nano_banana",
        generator_version="1.0",
        manipulation_family="nano_edit",
        default_cap=2_500,
        redistribution_mode="reference_only",
        origin_license="Research Only",
        attribution="Tungtom2004 Google Nano Banana Edited Images",
        source_url="https://huggingface.co/datasets/Tungtom2004/Google_Nano_Banana_Edited_Images",
        selection_reason="Multimodal mobile foundation model edited image pairs",
        provenance_confidence="medium",
        is_human_tampered=False,  # AI edit -> ai_positive = 1
    ),
    # --------------------------------------------------------------------------
    # 5. Fully AIGC - Modern State-of-the-Art Generators (ai_positive = 1)
    # --------------------------------------------------------------------------
    "flux_reason_6m": SourceConfig(
        name="flux_reason_6m",
        source_type="huggingface",
        hf_repo="LucasFang/FLUX-Reason-6M",
        hf_split="train",
        provenance="fully_aigc",
        generator="flux_1_dev",
        generator_family="flux",
        generator_version="1-dev",
        default_cap=5_000,
        redistribution_mode="embed_bytes",
        origin_license="FLUX.1-dev Non-Commercial License",
        license_url="https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md",
        attribution="LucasFang / Black Forest Labs FLUX.1",
        source_url="https://huggingface.co/datasets/LucasFang/FLUX-Reason-6M",
        selection_reason="Flow-matching diffusion architecture representations",
        provenance_confidence="high",
    ),
    "ideogram_27k": SourceConfig(
        name="ideogram_27k",
        source_type="huggingface",
        hf_repo="bitmind/ideogram-27k",
        hf_split="train",
        provenance="fully_aigc",
        generator="ideogram_v2",
        generator_family="ideogram",
        generator_version="v2",
        default_cap=3_000,
        redistribution_mode="embed_bytes",
        origin_license="BitMind Community License",
        attribution="BitMind Ideogram 27K",
        source_url="https://huggingface.co/datasets/bitmind/ideogram-27k",
        selection_reason="Typography-heavy photorealistic generative representations",
        provenance_confidence="high",
    ),
    "krea2_wildcards": SourceConfig(
        name="krea2_wildcards",
        source_type="huggingface",
        hf_repo="innofree/krea2-wildcards",
        hf_split="train",
        provenance="fully_aigc",
        generator="krea_2",
        generator_family="krea",
        generator_version="2.0",
        default_cap=2_500,
        redistribution_mode="reference_only",
        origin_license="Community / Research",
        attribution="Innofree Krea 2 Wildcards",
        source_url="https://huggingface.co/datasets/innofree/krea2-wildcards",
        selection_reason="Diverse wildcard-prompted Krea 2 generations",
        provenance_confidence="medium",
    ),
    "nano_banana_pro_gen": SourceConfig(
        name="nano_banana_pro_gen",
        source_type="huggingface",
        hf_repo="FlameF0X/nano-banana-pro-gen-zh-en",
        hf_split="train",
        provenance="fully_aigc",
        generator="nano_banana_pro",
        generator_family="nano_banana",
        generator_version="pro",
        default_cap=2_000,
        redistribution_mode="reference_only",
        origin_license="Research Only",
        attribution="FlameF0X Nano Banana Pro",
        source_url="https://huggingface.co/datasets/FlameF0X/nano-banana-pro-gen-zh-en",
        selection_reason="Bilingual prompt generation cohort",
        provenance_confidence="medium",
    ),
    # --------------------------------------------------------------------------
    # 6. Core Benchmark Datasets (WildFake, DiffusionForensics, SID)
    # --------------------------------------------------------------------------
    "wildfake": SourceConfig(
        name="wildfake",
        source_type="local_csv",
        local_path="metadata/wildfake_metadata.csv",
        provenance="mixed",
        generator="wildfake_generator",
        generator_family="wildfake",
        default_cap=15_000,
        redistribution_mode="reference_only",
        origin_license="Academic Research Only",
        attribution="WildFake Benchmark (hy2628982280)",
        source_url="https://modelscope.cn/datasets/hy2628982280/WildFake",
        selection_reason="Standardized diffusion benchmark with strict exclusion of DALL-E Advanced",
        provenance_confidence="high",
    ),
    "diffusionforensics": SourceConfig(
        name="diffusionforensics",
        source_type="local_csv",
        local_path="metadata/diffusionforensics_subset.csv",
        provenance="mixed",
        generator="diffusionforensics_generator",
        generator_family="diffusionforensics",
        default_cap=12_000,
        redistribution_mode="reference_only",
        origin_license="Academic Research Only",
        attribution="DiffusionForensics / DIRE (Wang et al.)",
        source_url="https://huggingface.co/datasets/nebula/DF-arrow",
        selection_reason="Multi-generator diffusion artifacts with ADM training anchor",
        provenance_confidence="high",
    ),
    "sid": SourceConfig(
        name="sid",
        source_type="local_csv",
        local_path="metadata/sid_metadata.csv",
        provenance="mixed",
        generator="sid_generator",
        generator_family="sid",
        default_cap=30_000,
        redistribution_mode="reference_only",
        origin_license="Organizer Provided",
        attribution="Synthetic Image Detection (SID) Challenge",
        source_url="SID Challenge Dataset",
        selection_reason="Primary in-domain competition distribution with balanced subtypes",
        provenance_confidence="high",
    ),
    # --------------------------------------------------------------------------
    # 7. SOTA Generative Cohorts: Midjourney v5/v6, SD 3/3.5, and Ewan GPT-Image
    # --------------------------------------------------------------------------
    "midjourney_v6_recap": SourceConfig(
        name="midjourney_v6_recap",
        source_type="huggingface",
        hf_repo="Photoroom/midjourney-v6-recap",
        hf_split="train",
        provenance="fully_aigc",
        generator="midjourney_6",
        generator_family="midjourney",
        generator_version="6.0",
        default_cap=2_000,
        redistribution_mode="embed_bytes",
        origin_license="MIT",
        license_url="https://huggingface.co/datasets/Photoroom/midjourney-v6-recap",
        attribution="Photoroom / CortexLM Midjourney v6 Recaptioned",
        source_url="https://huggingface.co/datasets/Photoroom/midjourney-v6-recap",
        selection_reason="Diverse Midjourney v6 high-fidelity photographic, macro, and typography generations",
        provenance_confidence="high",
    ),
    "midjourney_v5_images": SourceConfig(
        name="midjourney_v5_images",
        source_type="huggingface",
        hf_repo="ehristoforu/midjourney-images",
        hf_split="train",
        provenance="fully_aigc",
        generator="midjourney_5",
        generator_family="midjourney",
        generator_version="5.2",
        default_cap=1_000,
        redistribution_mode="embed_bytes",
        origin_license="MIT",
        license_url="https://huggingface.co/datasets/ehristoforu/midjourney-images",
        attribution="ehristoforu Midjourney V5/V6 Archive",
        source_url="https://huggingface.co/datasets/ehristoforu/midjourney-images",
        selection_reason="Diverse Midjourney v5 stylistic, architectural, and cinematic portrait generations",
        provenance_confidence="high",
    ),
    "sd3_medium_synths": SourceConfig(
        name="sd3_medium_synths",
        source_type="huggingface",
        hf_repo="VincHa/SD3_medium_synths",
        hf_split="train",
        provenance="fully_aigc",
        generator="sd_3_medium",
        generator_family="stable_diffusion_3",
        generator_version="3-medium",
        default_cap=1_000,
        redistribution_mode="embed_bytes",
        origin_license="Open Source Research / Academic",
        license_url="https://huggingface.co/datasets/VincHa/SD3_medium_synths",
        attribution="VincHa Stable Diffusion 3 Medium Synths",
        source_url="https://huggingface.co/datasets/VincHa/SD3_medium_synths",
        selection_reason="Multi-subject Stable Diffusion 3 synthetic generations covering landscapes, portraits, and macros",
        provenance_confidence="high",
    ),
    "ewan_gpt_images": SourceConfig(
        name="ewan_gpt_images",
        source_type="local_csv",
        local_path="metadata/ewan_gpt_images.csv",
        provenance="fully_aigc",
        generator="gpt_image",
        generator_family="gpt_image",
        generator_version="gpt-4o-image / dalle-3",
        default_cap=102,
        redistribution_mode="embed_bytes",
        origin_license="Team Generated / Internal Competition Collection",
        license_url="https://github.com/techjam2026",
        attribution="Ewan / TechJam 2026 In-The-Wild AI Collection",
        source_url="local://100+ AI_Images from ewan.zip",
        selection_reason="Empirical in-the-wild GPT-Image generations curated by team member Ewan",
        provenance_confidence="high",
    ),
    # --------------------------------------------------------------------------
    # 8. Counterbalance: AI-Generated Memes & Synthetic Text Overlays
    # --------------------------------------------------------------------------
    "ai_meme_macro_overlay": SourceConfig(
        name="ai_meme_macro_overlay",
        source_type="huggingface",
        hf_repo="ideepankarsharma2003/AIGeneratedImages_Midjourney",
        hf_split="train",
        provenance="tampered",
        generator="midjourney_text_overlay",
        generator_family="midjourney",
        default_cap=1_300,
        redistribution_mode="reference_only",
        origin_license="Creative Commons / Research",
        attribution="AIGeneratedImages_Midjourney Meme Overlays",
        source_url="https://huggingface.co/datasets/ideepankarsharma2003/AIGeneratedImages_Midjourney",
        selection_reason="Counterbalance authentic pre-2019 memes with AI-generated text macros",
        provenance_confidence="high",
    ),
    "ai_reaction_banners": SourceConfig(
        name="ai_reaction_banners",
        source_type="reference_only",
        provenance="tampered",
        generator="gpt_image_2",
        generator_family="gpt_image",
        default_cap=450,
        redistribution_mode="reference_only",
        origin_license="Research Only",
        attribution="Goku-OpenLab GPT Image Reaction Banners",
        source_url="https://huggingface.co/datasets/Goku-OpenLab/gpt-image-2-prompts-datasets",
        selection_reason="Top banner text overlays matching reaction macro distribution",
        provenance_confidence="medium",
    ),
    "scam_ai_social_posts": SourceConfig(
        name="scam_ai_social_posts",
        source_type="reference_only",
        provenance="fully_aigc",
        generator="gpt_image_2",
        generator_family="gpt_image",
        default_cap=300,
        redistribution_mode="reference_only",
        origin_license="Research Only",
        attribution="Synthetic Social UI Posts",
        source_url="https://huggingface.co/datasets/Goku-OpenLab/gpt-image-2-prompts-datasets",
        selection_reason="Fake social media card balance against authentic Twitter/Reddit screenshots",
        provenance_confidence="medium",
    ),
    # --------------------------------------------------------------------------
    # 9. Stylized 2D Anime, Manga & Digital Illustration (Balanced Pairs)
    # --------------------------------------------------------------------------
    "danbooru2026_aigc_wild": SourceConfig(
        name="danbooru2026_aigc_wild",
        source_type="reference_only",
        provenance="fully_aigc",
        generator="midjourney_niji",
        generator_family="danbooru_aigc",
        default_cap=500,
        redistribution_mode="reference_only",
        origin_license="Rights retained by uploaders",
        source_url="https://huggingface.co/datasets/nyanko-devs/danbooru2026",
        selection_reason="In-the-wild community anime AI generations across Niji, NovelAI, and WebUI",
        provenance_confidence="medium",
    ),
    "danbooru_pre2020_human": SourceConfig(
        name="danbooru_pre2020_human",
        source_type="reference_only",
        provenance="authentic",
        generator="authentic",
        generator_family="authentic",
        default_cap=500,
    ),
    "pepper_and_carrot": SourceConfig(
        name="pepper_and_carrot",
        source_type="reference_only",
        provenance="authentic",
        generator="authentic",
        generator_family="authentic",
        default_cap=500,
        redistribution_mode="reference_only",
        origin_license="CC BY-SA 4.0",
        license_url="https://creativecommons.org/licenses/by-sa/4.0/",
        attribution="Pepper&Carrot open-source comic by David Revoy",
        source_url="https://www.peppercarrot.com/",
        selection_reason="Public human-authored color manga/comic panels",
        provenance_confidence="high",
    ),
    # --------------------------------------------------------------------------
    # 10. 3D CGI, Game Rendering & Virtual Worlds (DirectX/UE5 vs AI Concept Art)
    # --------------------------------------------------------------------------
    "tartanair2_ue5_cyberpunk": SourceConfig(
        name="tartanair2_ue5_cyberpunk",
        source_type="reference_only",
        provenance="authentic",
        generator="authentic",
        generator_family="authentic",
        default_cap=500,
        redistribution_mode="reference_only",
        origin_license="BSD-3-Clause",
        attribution="Carnegie Mellon University / TartanAir2 Unreal Engine 5",
        source_url="https://huggingface.co/datasets/theairlabcmu/tartanair2",
        selection_reason="Human-built Unreal Engine 5 ray-traced urban & cyberpunk rendering",
        provenance_confidence="high",
    ),
    "gta5_driving_renders": SourceConfig(
        name="gta5_driving_renders",
        source_type="reference_only",
        provenance="authentic",
        generator="authentic",
        generator_family="authentic",
        default_cap=500,
        redistribution_mode="reference_only",
        origin_license="Research Use Only",
        attribution="GTA5 / RAGE Engine Synthetic Benchmark",
        source_url="https://huggingface.co/datasets/Chris1/GTA5",
        selection_reason="DirectX rasterized vehicles, asphalt, and architectural shaders",
        provenance_confidence="high",
    ),
    "game_screenshots_fantasy": SourceConfig(
        name="game_screenshots_fantasy",
        source_type="reference_only",
        provenance="authentic",
        generator="authentic",
        generator_family="authentic",
        default_cap=250,
        redistribution_mode="reference_only",
        origin_license="Public Domain / Fair Use",
        attribution="In-Game Captures (Witcher 3, Elden Ring)",
        source_url="https://huggingface.co/datasets/badigadiii/game_screenshots_11k",
        selection_reason="Authentic 3D video game fantasy landscapes, foliage shaders, and god-rays",
        provenance_confidence="high",
    ),
    "sintel_blender_open_movie": SourceConfig(
        name="sintel_blender_open_movie",
        source_type="huggingface",
        hf_repo="badigadiii/game_screenshots_11k",
        hf_split="train",
        provenance="authentic",
        generator="authentic",
        generator_family="authentic",
        default_cap=250,
        redistribution_mode="reference_only",
        origin_license="Public Domain / Fair Use",
        attribution="In-Game Captures & 3D Renders (badigadiii)",
        source_url="https://huggingface.co/datasets/badigadiii/game_screenshots_11k",
        selection_reason="Authentic 3D CGI and video game engine renders",
        provenance_confidence="high",
    ),
    "flux_cyberpunk_scifi": SourceConfig(
        name="flux_cyberpunk_scifi",
        source_type="reference_only",
        provenance="fully_aigc",
        generator="flux_1_dev",
        generator_family="flux",
        default_cap=500,
        redistribution_mode="reference_only",
        origin_license="FLUX.1-dev Non-Commercial",
        source_url="https://huggingface.co/datasets/LucasFang/FLUX-Reason-6M",
        selection_reason="AI concept art matching Cyberpunk UE5 neon specular reflections",
        provenance_confidence="high",
    ),
    "sdxl_photoreal_vehicles": SourceConfig(
        name="sdxl_photoreal_vehicles",
        source_type="reference_only",
        provenance="fully_aigc",
        generator="sdxl_1_0",
        generator_family="stable_diffusion_xl",
        default_cap=500,
        redistribution_mode="reference_only",
        origin_license="OpenRAIL++",
        source_url="https://huggingface.co/datasets/diffusers-parti-prompts/sdxl-1.0",
        selection_reason="AI vehicle and highway generations paired against GTA 5 rendering",
        provenance_confidence="high",
    ),
    "midjourney_fantasy_environments": SourceConfig(
        name="midjourney_fantasy_environments",
        source_type="reference_only",
        provenance="fully_aigc",
        generator="midjourney_6",
        generator_family="midjourney",
        default_cap=500,
        redistribution_mode="reference_only",
        origin_license="Research",
        source_url="https://huggingface.co/datasets/Photoroom/midjourney-v6-recap",
        selection_reason="AI fantasy worldbuilding matching Witcher/Elden Ring game environments",
        provenance_confidence="high",
    ),
    # --------------------------------------------------------------------------
    # 11. Human Anatomy, Skin Texture & Glamour (Authentic vs Uncensored Diffusion)
    # --------------------------------------------------------------------------
    "authentic_classical_figure_art": SourceConfig(
        name="authentic_classical_figure_art",
        source_type="reference_only",
        provenance="authentic",
        generator="authentic",
        generator_family="authentic",
        default_cap=1_000,
        redistribution_mode="reference_only",
        origin_license="Public Domain / CC0",
        attribution="Museum Classical Figure Art & Sculpture Collections",
        source_url="https://huggingface.co/datasets/Mitsua/art-museums-pd-440k",
        selection_reason="Human anatomy, musculature, and classical figure studies pre-2020",
        provenance_confidence="high",
    ),
    "authentic_glamour_portraits": SourceConfig(
        name="authentic_glamour_portraits",
        source_type="huggingface",
        hf_repo="mattymchen/celeba-hq",
        hf_split="train",
        provenance="authentic",
        generator="authentic",
        generator_family="authentic",
        default_cap=1_500,
        redistribution_mode="reference_only",
        origin_license="CelebA-HQ Non-commercial Research",
        attribution="CelebA-HQ High-Fashion Studio Portraits",
        source_url="https://huggingface.co/datasets/mattymchen/celeba-hq",
        selection_reason="Authentic human glamour, cosmetics, studio flash, and skin pore textures",
        provenance_confidence="high",
    ),
}
# ==============================================================================
# Acquisition & Normalization Engine
# ==============================================================================


def normalize_record_schema(
    record: Dict[str, Any],
    source: SourceConfig,
    data_root: Path,
) -> Dict[str, Any]:
    """Ensure record contains every schema column with strictly typed values."""
    # Determine provenance string and numeric label
    raw_prov = str(record.get("provenance", "") or record.get("label", "") or source.provenance)
    raw_prov = raw_prov.strip().lower()
    if raw_prov in {"0", "authentic", "real"}:
        prov_str = "authentic"
        label_int = 0
    elif raw_prov in {"1", "tampered", "edited"}:
        prov_str = "tampered"
        label_int = 1
    elif raw_prov in {"2", "fully_aigc", "aigc", "synthetic", "full_synthetic"}:
        prov_str = "fully_aigc"
        label_int = 2
    else:
        # Fallback to source default
        prov_str = source.provenance if source.provenance != "mixed" else "authentic"
        label_int = 0 if prov_str == "authentic" else (1 if prov_str == "tampered" else 2)

    # Binary task semantics: Track 5 binary label
    # AI positive = 1 for fully_aigc and AI-tampered; 0 for authentic and human-tampered
    if source.is_human_tampered:
        ai_positive = 0
    elif prov_str == "authentic":
        ai_positive = 0
    else:
        ai_positive = 1

    generator = str(record.get("generator", "") or source.generator or "")
    generator_family = str(record.get("generator_family", "") or source.generator_family or "")
    if not generator_family:
        generator_family = normalize_generator(generator, source.name)

    manipulation_family = str(
        record.get("manipulation_family", "") or source.manipulation_family or ""
    )
    if prov_str == "authentic" and not source.is_human_tampered:
        manipulation_family = ""

    image_path = str(record.get("image_path", "") or "")
    raw_group = str(record.get("source_image_group", "") or "").strip()
    if raw_group and raw_group.lower() not in {"nan", "none"}:
        if not raw_group.startswith(f"{source.name}:"):
            source_image_group = f"{source.name}:{raw_group}"
        else:
            source_image_group = raw_group
    else:
        source_image_group = ""
    width = int(record.get("width", 0) or 0)
    height = int(record.get("height", 0) or 0)
    file_format = str(record.get("file_format", "JPEG") or "JPEG").upper()
    sha256 = str(record.get("sha256", "") or "").strip()
    resolved_path = Path(image_path) if Path(image_path).is_absolute() else data_root / image_path
    if not sha256:
        if resolved_path.is_file():
            try:
                sha256 = file_sha256(resolved_path)
            except Exception:
                sha256 = hashlib.sha256(f"{source.name}:{image_path}:{record.get('external_id', '')}".encode()).hexdigest()
        else:
            sha256 = hashlib.sha256(f"{source.name}:{image_path}:{record.get('external_id', '')}".encode()).hexdigest()

    perceptual_hash = str(record.get("perceptual_hash", "") or "").strip()
    if not perceptual_hash or perceptual_hash == "0" * 16:
        if resolved_path.is_file():
            try:
                phash_val = difference_hash(resolved_path)
                perceptual_hash = f"{phash_val:016x}"
            except Exception:
                phash_val = int.from_bytes(hashlib.sha256(f"phash:{source.name}:{sha256}:{image_path}".encode()).digest()[:8], "big")
                perceptual_hash = f"{phash_val:016x}"
        else:
            phash_val = int.from_bytes(hashlib.sha256(f"phash:{source.name}:{sha256}:{image_path}".encode()).digest()[:8], "big")
            perceptual_hash = f"{phash_val:016x}"
    perceptual_hash_2 = str(record.get("perceptual_hash_2", "") or "").strip()
    if not perceptual_hash_2 and resolved_path.is_file():
        try:
            with Image.open(resolved_path) as hash_image:
                perceptual_hash_2 = f"{compute_image_phash(hash_image):016x}"
        except Exception:
            perceptual_hash_2 = ""

    quality_score = float(record.get("quality_score", 0.85) or 0.85)

    created_at = str(
        record.get("created_at", "") or datetime.datetime.now(datetime.timezone.utc).isoformat()
    )

    out = {
        # Core training-compatible manifest columns
        "image_path": image_path,
        "label": label_int,
        "dataset": source.name,
        "official_split": str(record.get("official_split", "train") or "train"),
        "generator": generator,
        "manipulation_family": manipulation_family,
        "source_image_group": source_image_group,
        "width": width,
        "height": height,
        "file_format": file_format,
        "tamper_mask_path": str(record.get("tamper_mask_path", "") or ""),
        # Extended metadata
        "source_url": str(record.get("source_url", "") or source.source_url),
        "external_id": str(record.get("external_id", "") or record.get("img_id", "") or ""),
        "generator_family": generator_family,
        "generator_version": str(
            record.get("generator_version", "") or source.generator_version or ""
        ),
        "prompt": str(record.get("prompt", "") or ""),
        "created_at": created_at,
        "sha256": sha256,
        "perceptual_hash": perceptual_hash,
        "perceptual_hash_2": perceptual_hash_2,
        "quality_score": quality_score,
        "provenance_confidence": str(
            record.get("provenance_confidence", "") or source.provenance_confidence or "high"
        ),
        "redistribution_mode": str(
            record.get("redistribution_mode", "") or source.redistribution_mode or "reference_only"
        ),
        "origin_license": str(
            record.get("origin_license", "") or source.origin_license or "Research Only"
        ),
        "license_url": str(record.get("license_url", "") or source.license_url or ""),
        "attribution": str(record.get("attribution", "") or source.attribution or ""),
        "selection_reason": str(
            record.get("selection_reason", "") or source.selection_reason or ""
        ),
        "forbidden_demo_checked": True,
        # Task & Split semantics
        "ai_positive": ai_positive,
        "split": str(record.get("split", record.get("official_split", "train")) or "train"),
        "duplicate_group": str(record.get("duplicate_group", "") or ""),
        "provenance": prov_str,
    }
    return out


def acquire_huggingface_source(
    source: SourceConfig,
    cap: int,
    seed: int,
    data_root: Path,
    dry_run: bool = False,
    min_short_edge: int = 160,
    min_area: int = 64_000,
    min_bytes: int = 2048,
    embed_images: bool = False,
) -> List[Dict[str, Any]]:
    """Acquire or inspect rows from a Hugging Face dataset."""
    if not source.hf_repo:
        raise ValueError(f"Source {source.name} has no hf_repo specified")

    logger.info(
        "Loading Hugging Face source: %s (%s, cap=%d, dry_run=%s)",
        source.name,
        source.hf_repo,
        cap,
        dry_run,
    )

    if dry_run:
        # In dry run mode: generate representative metadata entries without pulling images
        logger.info(
            "Dry run mode: creating %d synthetic schema rows for source %s", cap, source.name
        )
        records = []
        source_seed = (seed + int.from_bytes(hashlib.sha256(source.name.encode()).digest()[:4], "big")) % (2**32 - 1)
        rng = random.Random(source_seed)
        for i in range(cap):
            ext_id = f"{source.name}_{i:06d}"
            dummy_hash = hashlib.sha256(f"sha:{source.name}:{seed}:{i}".encode()).hexdigest()
            dhash_int = int.from_bytes(hashlib.sha256(f"phash:{source.name}:{seed}:{i}".encode()).digest()[:8], "big")
            native_path = data_root / f"{source.name}/{ext_id}.jpg"
            native_w, native_h = 0, 0
            if native_path.is_file():
                try:
                    with Image.open(native_path) as probe:
                        native_w, native_h = probe.size
                except Exception:
                    pass
            row_dict = {
                "image_path": f"{source.name}/{ext_id}.jpg",
                "external_id": ext_id,
                "width": native_w,
                "height": native_h,
                "file_format": "JPEG",
                "sha256": dummy_hash,
                "perceptual_hash": f"{dhash_int:016x}",
                "quality_score": 0.95,
                "source_image_group": f"group_{i // 5:05d}"
                if source.provenance == "tampered"
                else "",
                "official_split": "train",
            }
            records.append(normalize_record_schema(row_dict, source, data_root))
        return records

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("huggingface datasets library is required for live acquisition") from exc

    records = []
    skipped_forbidden = 0
    skipped_quality = 0

    try:
        ds = load_dataset(source.hf_repo, split=source.hf_split, streaming=True)
    except Exception as exc:
        logger.warning(
            "Streaming failed for %s (%s), trying non-streaming or builder fallback: %s",
            source.name,
            source.hf_repo,
            exc,
        )
        raise

    for i, item in enumerate(ds):
        if len(records) >= cap:
            break

        # Check forbidden demo data
        if is_forbidden_demonstration(item):
            skipped_forbidden += 1
            continue

        # Extract image
        raw_img = item.get("image") or item.get("img") or item.get("file")
        if raw_img is None:
            continue

        raw_bytes = None
        if isinstance(raw_img, bytes):
            raw_bytes = raw_img
            try:
                pil_img = Image.open(io.BytesIO(raw_bytes))
            except Exception:
                skipped_quality += 1
                continue
        elif isinstance(raw_img, Image.Image):
            pil_img = raw_img
        elif isinstance(raw_img, dict) and "bytes" in raw_img:
            raw_bytes = raw_img["bytes"]
            try:
                pil_img = Image.open(io.BytesIO(raw_bytes))
            except Exception:
                skipped_quality += 1
                continue
        else:
            continue

        # Run quality filter
        q_res = evaluate_image_quality(
            pil_img,
            raw_bytes=raw_bytes,
            min_short_edge=min_short_edge,
            min_area=min_area,
            min_bytes=min_bytes,
        )
        if not q_res.passed:
            skipped_quality += 1
            continue

        ext_id = str(item.get("id") or item.get("img_id") or f"{source.name}_{i:06d}")
        prompt_text = str(
            item.get("prompt") or item.get("text") or item.get("caption") or item.get("title") or ""
        )

        dest_relative = f"{source.name}/{ext_id}.{q_res.file_format.lower()}"
        if embed_images:
            dest_full = data_root / dest_relative
            dest_full.parent.mkdir(parents=True, exist_ok=True)
            if not dest_full.exists():
                pil_img.convert("RGB").save(dest_full, format=q_res.file_format)

        row_dict = {
            "image_path": dest_relative,
            "external_id": ext_id,
            "prompt": prompt_text,
            "width": q_res.width,
            "height": q_res.height,
            "file_format": q_res.file_format,
            "sha256": q_res.sha256,
            "perceptual_hash": q_res.perceptual_hash,
            "quality_score": q_res.quality_score,
            "official_split": str(item.get("split", "train")),
        }
        records.append(normalize_record_schema(row_dict, source, data_root))

    logger.info(
        "Acquired %d rows from %s (skipped: %d forbidden, %d quality)",
        len(records),
        source.name,
        skipped_forbidden,
        skipped_quality,
    )
    return records


def acquire_local_source(
    source: SourceConfig,
    cap: int,
    seed: int,
    data_root: Path,
    allow_missing: bool = False,
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    candidates = [
        data_root / (source.local_path or f"metadata/{source.name}.csv"),
        _PROJECT_ROOT / (source.local_path or f"metadata/{source.name}.csv"),
        Path(os.environ.get("TECHJAM_RUNTIME_ROOT", "")) / (source.local_path or ""),
        _PROJECT_ROOT / f"metadata/{source.name}_subset.csv",
        _PROJECT_ROOT / f"metadata/{source.name}_sanity.csv",
    ]
    csv_path = None
    for cand in candidates:
        if cand.is_file():
            csv_path = cand
            break
    if csv_path is None:
        csv_path = candidates[1]
    if not csv_path.is_file():
        if dry_run or allow_missing:
            logger.warning(
                "Local manifest %s for source %s not found. Dry-run/allow-missing synthesizing %d metadata rows.",
                csv_path,
                source.name,
                min(cap, 500) if dry_run else 0,
            )
            if not dry_run:
                return []
            records = []
            source_seed = (seed + int.from_bytes(hashlib.sha256(source.name.encode()).digest()[:4], "big")) % (2**32 - 1)
            rng = random.Random(source_seed)
            sample_cap = min(cap, 500)
            for i in range(sample_cap):
                ext_id = f"{source.name}_{i:06d}"
                dummy_hash = hashlib.sha256(f"sha:{source.name}:{seed}:{i}".encode()).hexdigest()
                dhash_int = int.from_bytes(hashlib.sha256(f"phash:{source.name}:{seed}:{i}".encode()).digest()[:8], "big")
                prov = "authentic" if (i % 2 == 0) else "fully_aigc"
                row_dict = {
                    "image_path": f"{source.name}/{ext_id}.jpg",
                    "external_id": ext_id,
                    "provenance": prov,
                    "width": 1024,
                    "height": 1024,
                    "file_format": "JPEG",
                    "sha256": dummy_hash,
                    "perceptual_hash": f"{dhash_int:016x}",
                    "quality_score": 0.90,
                    "source_image_group": f"local_group_{i // 4:04d}",
                    "official_split": "train",
                }
                records.append(normalize_record_schema(row_dict, source, data_root))
            return records
        raise FileNotFoundError(f"Local manifest not found: {csv_path}")

    logger.info("Reading local manifest: %s", csv_path)
    df = pd.read_csv(csv_path, low_memory=False).fillna("")

    # Build one filename index so local acquisition is O(files + rows).
    image_index: dict[str, Path] = {}
    if source.name == "gmorinan_memes":
        image_index = {
            path.name: path
            for path in csv_path.parent.rglob("*")
            if path.is_file()
        }
    materialized_dir = data_root / "materialized" / source.name
    if source.name == "gmorinan_memes":
        materialized_dir.mkdir(parents=True, exist_ok=True)
    # Filter forbidden demonstration data
    valid_rows = []
    skipped_forbidden = 0
    for row in df.to_dict(orient="records"):
        if source.name == "gmorinan_memes":
            if int(row.get("timestamp", 0) or 0) >= 1546300800:
                continue
            image_name = str(row.get("image", "")).strip()
            match = image_index.get(image_name)
            if match is None:
                continue
            materialized_path = materialized_dir / image_name
            if not materialized_path.exists():
                shutil.copy2(match, materialized_path)
            row["image_path"] = str(materialized_path.relative_to(data_root))
            row["provenance"] = "tampered"
            row["official_split"] = "train"
            row["external_id"] = str(row.get("idRdt", image_name))
            row["source_image_group"] = str(row.get("label", image_name))
            row["source_url"] = str(row.get("url", ""))
        if is_forbidden_demonstration(row):
            skipped_forbidden += 1
            continue
        valid_rows.append(row)

    if skipped_forbidden > 0:
        logger.info(
            "Filtered %d forbidden demonstration rows from %s", skipped_forbidden, source.name
        )

    if not valid_rows:
        return []

    valid_df = pd.DataFrame(valid_rows)
    if len(valid_df) > cap:
        valid_df = valid_df.sample(n=cap, random_state=seed)

    records = []
    for row in valid_df.to_dict(orient="records"):
        records.append(normalize_record_schema(row, source, data_root))
    return records


def acquire_local_directory(
    source: SourceConfig,
    cap: int,
    seed: int,
    data_root: Path,
) -> List[Dict[str, Any]]:
    """Materialize a local image directory into portable relative paths."""
    runtime_root = Path(os.environ.get("TECHJAM_RUNTIME_ROOT", ""))
    source_dir = runtime_root / (source.local_path or "")
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Local image directory not found: {source_dir}")
    candidates = sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if len(candidates) > cap:
        candidates = sorted(random.Random(seed).sample(candidates, cap))
    materialized_dir = data_root / "materialized" / source.name
    materialized_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for path in candidates:
        target = materialized_dir / path.name
        if not target.exists():
            shutil.copy2(path, target)
        records.append(
            normalize_record_schema(
                {
                    "image_path": str(target.relative_to(data_root)),
                    "provenance": source.provenance,
                    "official_split": "train",
                    "external_id": path.stem,
                    "source_image_group": path.stem,
                },
                source,
                data_root,
            )
        )
    return records


def acquire_reference_only_source(
    source: SourceConfig,
    cap: int,
    seed: int,
    data_root: Path,
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    """Generate reference-only rows only for explicit dry-run planning.

    Production manifests must contain materialized bytes; synthetic metadata is
    never an acceptable substitute for training data.
    """
    if not dry_run:
        raise RuntimeError(
            f"Source {source.name} is reference-only and has no materialized acquisition adapter"
        )
    logger.info(
        "Generating dry-run reference_only rows for %s (cap=%d, seed=%d)",
        source.name,
        cap,
        seed,
    )
    records = []
    source_seed = (seed + int.from_bytes(hashlib.sha256(source.name.encode()).digest()[:4], "big")) % (2**32 - 1)
    rng = random.Random(source_seed)

    for i in range(cap):
        ext_id = f"{source.name}_{i:06d}"
        dummy_hash = hashlib.sha256(f"sha:{source.name}:{seed}:{i}".encode()).hexdigest()
        dhash_int = int.from_bytes(hashlib.sha256(f"phash:{source.name}:{seed}:{i}".encode()).digest()[:8], "big")

        # Source image groups link related templates or labels
        if source.name == "gmorinan_memes":
            group = f"label_{i % 115:03d}"
            manip_fam = "human_caption"
        elif source.name == "imgflip_memes":
            group = f"template_{i % 100:03d}"
            manip_fam = "human_caption"
        elif source.name == "multioff_memes":
            group = f"format_bucket_{i % 6:02d}"
            manip_fam = "human_caption"
        else:
            group = f"group_{i // 10:04d}"
            manip_fam = source.manipulation_family

        native_path = data_root / f"reference_only/{source.name}/{ext_id}.jpg"
        native_w = 0
        native_h = 0
        if (not native_w or not native_h) and native_path.is_file():
            try:
                with Image.open(native_path) as probe:
                    native_w, native_h = probe.size
            except Exception:
                pass

        row_dict = {
            "image_path": f"reference_only/{source.name}/{ext_id}.jpg",
            "external_id": ext_id,
            "provenance": source.provenance,
            "manipulation_family": manip_fam,
            "source_image_group": group,
            "width": native_w,
            "height": native_h,
            "file_format": "JPEG",
            "sha256": dummy_hash,
            "perceptual_hash": f"{dhash_int:016x}",
            "quality_score": 0.88,
            "official_split": "train",
            "redistribution_mode": "reference_only",
        }
        records.append(normalize_record_schema(row_dict, source, data_root))

    return records


# ==============================================================================
# Duplicate Grouping & Split Assignment
# ==============================================================================
class _FastDisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root
def fast_duplicate_groups(
    sha256_values: list[str],
    perceptual_hashes: list[int],
    source_groups: list[str],
    perceptual_hashes_2: list[int] | None = None,
    max_hamming_distance: int = 4,
    max_candidates: int | None = None,
) -> list[str]:
    """Cluster exact, source-linked, and dual perceptual-hash duplicates.

    ``max_candidates`` is retained only for API compatibility and is ignored:
    dropping candidates makes duplicate leakage nondeterministic.
    """
    del max_candidates
    n = len(sha256_values)
    second = perceptual_hashes_2 or [0] * n
    if len(perceptual_hashes) != n or len(second) != n or len(source_groups) != n:
        raise ValueError("duplicate inputs must have equal lengths")
    sets = _FastDisjointSet(n)
    exact: dict[str, int] = {}
    sources: dict[str, int] = {}
    bands: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    first_shifts = (0, 13, 26, 39, 52)
    second_shifts = tuple(range(0, 64, 8))

    for index, (sha256, first_hash, source) in enumerate(
        zip(sha256_values, perceptual_hashes, source_groups, strict=True)
    ):
        if sha256 in exact:
            sets.union(index, exact[sha256])
        else:
            exact[sha256] = index
        if source:
            if source in sources:
                sets.union(index, sources[source])
            else:
                sources[source] = index
        candidates: set[int] = set()
        for band, shift in enumerate(first_shifts):
            candidates.update(bands[(0, band, (first_hash >> shift) & 0x1FFF)])
        for band, shift in enumerate(second_shifts):
            candidates.update(bands[(1, band, (second[index] >> shift) & 0xFF)])
        for other in candidates:
            first_match = (first_hash ^ perceptual_hashes[other]).bit_count() <= max_hamming_distance
            second_match = (
                second[index] != 0
                and second[other] != 0
                and (second[index] ^ second[other]).bit_count() <= 6
            )
            if first_match or second_match:
                sets.union(index, other)
        for band, shift in enumerate(first_shifts):
            bands[(0, band, (first_hash >> shift) & 0x1FFF)].append(index)
        for band, shift in enumerate(second_shifts):
            bands[(1, band, (second[index] >> shift) & 0xFF)].append(index)

    roots = [sets.find(i) for i in range(n)]
    canonical = {root: f"dup_{order:08d}" for order, root in enumerate(sorted(set(roots)))}
    return [canonical[root] for root in roots]


def assign_group_disjoint_splits(
    frame: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """Cluster duplicate groups and assign group-disjoint splits.

    Guarantees:
      1. All images in the same duplicate_group or source_image_group belong to the same split.
      2. test_unseen is strictly reserved for synthetic generator families wholly absent from train.
      3. assert_no_group_leakage() passes.
      4. assert_forbidden_demonstration_data_absent() passes.
    """
    logger.info("Computing global duplicate groups for %d rows...", len(frame))
    result = frame.copy()
    sha_values = [str(val) for val in result["sha256"]]
    phash_ints = []
    phash_ints_2 = []
    for phash_str, phash_str_2, sha_str, pth in zip(
        result["perceptual_hash"],
        result.get("perceptual_hash_2", pd.Series([""] * len(result))),
        sha_values,
        result["image_path"],
        strict=True,
    ):
        try:
            phash_ints.append(int(str(phash_str), 16))
        except ValueError:
            phash_ints.append(
                int.from_bytes(hashlib.sha256(f"{sha_str}:{pth}".encode()).digest()[:8], "big")
            )
        try:
            phash_ints_2.append(int(str(phash_str_2), 16))
        except ValueError:
            phash_ints_2.append(0)
    source_groups = [str(val) if pd.notna(val) else "" for val in result["source_image_group"]]

    result["duplicate_group"] = fast_duplicate_groups(
        sha_values, phash_ints, source_groups, phash_ints_2
    )
    # Normalize generator family
    generators = result["generator"] if "generator" in result else [""] * len(result)
    result["generator_family"] = [
        normalize_generator(gen, ds)
        for gen, ds in zip(generators, result["dataset"], strict=True)
    ]

    # Assign initial candidate split per row
    splits = []
    for row in result.to_dict(orient="records"):
        official = str(row.get("official_split", "train")).strip().lower()
        family = str(row["generator_family"])
        provenance = str(row["provenance"]).strip().lower()
        dataset = re.sub(r"[^a-z0-9]+", "", str(row["dataset"]).lower())

        forced_adm_training = dataset in {"diffusionforensics", "dire"} and family == "adm"
        if official == "test_unseen":
            splits.append("test_unseen")
        elif (
            provenance in {"fully_aigc", "aigc", "ai"}
            and is_held_out_generator(family, seed)
            and not forced_adm_training
        ):
            splits.append("test_unseen")
        elif official in {"test", "testing"}:
            splits.append("test")
        elif official in {"validation", "val", "dev"}:
            splits.append("validation")
        else:
            # Deterministic group partition based on duplicate_group hash
            group_hash = int.from_bytes(
                hashlib.sha256(f"{seed}:{row['duplicate_group']}".encode()).digest()[:4],
                "big",
            )
            bucket = group_hash % 100
            if bucket < 70:
                splits.append("train")
            elif bucket < 85:
                splits.append("validation")
            else:
                splits.append("test")

    result["split"] = splits

    # Enforce strict group-disjoint constraint across duplicate groups
    result = enforce_group_splits(result)

    # Reserve test_unseen strictly for generator families absent from train
    synthetic = result["provenance"].astype(str).str.lower().isin({"fully_aigc", "aigc", "ai"})
    trained_families = set(
        result.loc[(result["split"] == "train") & synthetic, "generator_family"].astype(str)
    )

    for _, indices in result.groupby("duplicate_group").groups.items():
        group_indices = list(indices)
        if not (result.loc[group_indices, "split"] == "test_unseen").any():
            continue
        group_synthetic = synthetic.loc[group_indices]
        group_families = set(
            result.loc[
                [idx for idx in group_indices if group_synthetic.loc[idx]], "generator_family"
            ]
            .astype(str)
            .tolist()
        )
        if group_families & trained_families:
            result.loc[group_indices, "split"] = "test"
        elif not group_synthetic.any():
            result.loc[group_indices, "split"] = "test"

    # Assertions
    assert_no_group_leakage(result)
    assert_forbidden_demonstration_data_absent(result)

    logger.info("Split assignment completed. Split counts: %s", Counter(result["split"]))
    return result


# ==============================================================================
# Validation & Export Engine
# ==============================================================================


def validate_manifest_schema(frame: pd.DataFrame) -> None:
    """Verify that DataFrame strictly conforms to data/COMBINED_DATASET_SCHEMA.md."""
    missing_cols = set(ALL_SCHEMA_COLUMNS) - set(frame.columns)
    if missing_cols:
        raise ValueError(f"Manifest missing canonical schema columns: {sorted(missing_cols)}")

    # Check non-empty rows
    if len(frame) == 0:
        raise ValueError("Cannot export empty manifest")

    # Verify label values
    valid_labels = {0, 1, 2}
    actual_labels = set(frame["label"].unique())
    if not actual_labels.issubset(valid_labels):
        raise ValueError(
            f"Invalid numeric labels found: {actual_labels - valid_labels}. Expected subset of {valid_labels}"
        )

    # Verify binary ai_positive values
    valid_ai_pos = {0, 1}
    actual_ai_pos = set(frame["ai_positive"].unique())
    if not actual_ai_pos.issubset(valid_ai_pos):
        raise ValueError(f"Invalid ai_positive values: {actual_ai_pos - valid_ai_pos}")

    # Verify forbidden demo checked
    if not frame["forbidden_demo_checked"].all():
        raise ValueError("forbidden_demo_checked must be True for all records")

    logger.info("Schema validation passed successfully (%d columns verified).", len(frame.columns))


def export_dataset_package(
    frame: pd.DataFrame,
    output_dir: Path,
    dataset_name: str = "track5-combined",
    embed_images: bool = False,
    data_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Export combined dataset into manifests, Parquet, JSONL, and Hugging Face DatasetDict."""
    output_dir.mkdir(parents=True, exist_ok=True)
    validate_manifest_schema(frame)

    logger.info("Exporting combined dataset package to: %s", output_dir)

    # 1. Write CSV, JSONL, and Parquet manifests
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "combined_manifest.csv", index=False)
    split_frames: Dict[str, pd.DataFrame] = {}
    for split_name in ("train", "validation", "test", "test_unseen"):
        split_df = frame[frame["split"] == split_name].copy()
        split_frames[split_name] = split_df
        split_df.to_csv(output_dir / f"{split_name}.csv", index=False)
        # Parquet export in data/ for automatic Hugging Face datasets resolution
        try:
            split_df.to_parquet(data_dir / f"{split_name}.parquet", index=False)
            split_df.to_parquet(output_dir / f"{split_name}.parquet", index=False)
        except Exception as exc:
            logger.warning("Could not export %s.parquet (parquet engine missing): %s", split_name, exc)
        # JSONL export
        split_df.to_json(
            output_dir / f"{split_name}.jsonl", orient="records", lines=True, date_format="iso"
        )
    # 2. Build and save Hugging Face DatasetDict
    try:
        from datasets import Dataset, DatasetDict

        hf_splits = {}
        for split_name, split_df in split_frames.items():
            if len(split_df) > 0:
                hf_splits[split_name] = Dataset.from_pandas(split_df, preserve_index=False)
        ds_dict = DatasetDict(hf_splits)
        ds_dict_path = output_dir / "hf_dataset"
        ds_dict.save_to_disk(str(ds_dict_path))
        logger.info("Saved Hugging Face DatasetDict to: %s", ds_dict_path)
    except Exception as exc:
        logger.warning("Could not serialize HF DatasetDict to disk: %s", exc)

    # 3. Create Dataset Card (README.md)
    readme_content = f"""---
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train.parquet
      - split: validation
        path: data/validation.parquet
      - split: test
        path: data/test.parquet
      - split: test_unseen
        path: data/test_unseen.parquet
tags:
  - image-forensics
  - multimodal-vision
pretty_name: TJ Multimodal Research Dataset
size_categories:
  - 100K<n<1M
---

# TJ Research Dataset — Canonical Manifests

Canonical consolidated multimodal research dataset.
Conforms to `data/COMBINED_DATASET_SCHEMA.md`.

## Task Target
Binary classification:
- `authentic` -> 0 (`ai_positive = 0`)
- `tampered` (human memes) -> 1 (`ai_positive = 0`)
- `tampered` (AI edits) -> 1 (`ai_positive = 1`)
- `fully_aigc` -> 2 (`ai_positive = 1`)

## Split Breakdown
- Total rows: {len(frame)}
- Train: {len(split_frames.get('train', []))}
- Validation: {len(split_frames.get('validation', []))}
- Test: {len(split_frames.get('test', []))}
- Test Unseen: {len(split_frames.get('test_unseen', []))}

## Exclusions
Strictly excludes COCO val2017 and WildFake DALL-E Advanced per competition contract.
"""
    (output_dir / "README.md").write_text(readme_content, encoding="utf-8")

    # 4. Generate Audit Report
    generator_registry = (
        frame[["dataset", "generator_family", "split"]]
        .drop_duplicates()
        .sort_values(["dataset", "generator_family"])
        .to_dict(orient="records")
    )
    counts_by_split = {
        split_name: {
            f"{ds}/{prov}": int(cnt)
            for (ds, prov), cnt in split_df.groupby(["dataset", "provenance"]).size().items()
        }
        for split_name, split_df in split_frames.items()
    }
    ai_positive_counts = {
        split_name: {
            f"ai_positive_{pos}": int(cnt)
            for pos, cnt in split_df.groupby("ai_positive").size().items()
        }
        for split_name, split_df in split_frames.items()
    }

    report = {
        "manifest_sha256": manifest_digest(frame),
        "total_rows": len(frame),
        "split_counts": {k: len(v) for k, v in split_frames.items()},
        "counts_by_split_and_source": counts_by_split,
        "ai_positive_distribution": ai_positive_counts,
        "duplicate_groups_count": int(frame["duplicate_group"].nunique()),
        "quality_score_mean": float(frame["quality_score"].mean()),
        "quality_score_min": float(frame["quality_score"].min()),
        "quality_score_max": float(frame["quality_score"].max()),
        "forbidden_demo_check_passed": bool(frame["forbidden_demo_checked"].all()),
        "generator_registry": generator_registry,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with (output_dir / "audit_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Audit report written to %s", output_dir / "audit_report.json")
    return report


# ==============================================================================
# Push to Hub Interface
# ==============================================================================


def push_dataset_to_hub(
    output_dir: Path,
    repo_id: str,
    private: bool = False,
) -> None:
    """Push the generated dataset package to Hugging Face Hub.

    Safety: MUST be explicitly invoked via --push-to-hub.
    """
    if not repo_id:
        raise ValueError("--push-to-hub requires --repo-id (e.g. --repo-id zye2/techjam-track5-combined)")

    logger.info("Initiating upload to Hugging Face Hub: repo_id=%s (private=%s)", repo_id, private)
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
        api.upload_folder(
            folder_path=str(output_dir),
            repo_id=repo_id,
            repo_type="dataset",
            ignore_patterns=["*.partial", "*.tmp", "hf_dataset/**"],
        )
        logger.info("Successfully pushed combined dataset to: https://huggingface.co/datasets/%s", repo_id)
    except Exception as exc:
        logger.error("Failed to push dataset to Hugging Face Hub: %s", exc)
        raise


# ==============================================================================
# CLI and Main Flow
# ==============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Consolidated Hugging Face Dataset Builder for TechJam Track 5.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Source selection and control
    parser.add_argument(
        "--sources",
        type=str,
        default="",
        help="Comma-separated list of registered sources to include (default: all registered)",
    )
    parser.add_argument(
        "--exclude-sources",
        type=str,
        default="",
        help="Comma-separated list of registered sources to exclude",
    )
    parser.add_argument(
        "--max-samples-per-source",
        type=int,
        default=0,
        help="Optional ceiling on samples per source (0 uses default per-source caps)",
    )
    parser.add_argument(
        "--source-caps",
        type=str,
        default="",
        help="JSON string or key=val pairs overriding per-source caps (e.g. 'flux_reason_6m=1000,artic_dataset=1000')",
    )
    parser.add_argument(
        "--inspect-sources",
        "--list-sources",
        action="store_true",
        help="Print table of all registered sources and configurations, then exit 0",
    )
    # Execution mode
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode: validate configs, simulate metadata extraction, check schemas, without heavy network downloads",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Allow proceeding with available sources if some local manifests or paths are absent",
    )
    # Paths & environment
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Local data directory (defaults to TECHJAM_DATA_ROOT or E:/techjam26-runtime)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("splits/combined_hf"),
        help="Directory to write output manifests, Parquet/JSONL packages, and audit report",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed for sampling and group hash partitioning",
    )
    parser.add_argument(
        "--embed-images",
        action="store_true",
        help="Download and embed raw image bytes for embed_bytes sources into data root",
    )
    # Quality filters
    parser.add_argument(
        "--min-short-edge",
        type=int,
        default=160,
        help="Minimum short-edge dimension in pixels (default: 160)",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=64_000,
        help="Minimum pixel area width * height (default: 64,000)",
    )
    parser.add_argument(
        "--min-bytes",
        type=int,
        default=2048,
        help="Minimum raw file size in bytes (default: 2,048)",
    )
    # Push to Hub (strictly opt-in)
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Explicit flag to upload resulting package to Hugging Face Hub (default: False)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="",
        help="Hugging Face repo id to push to (required if --push-to-hub is specified)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create/push to private Hugging Face repository",
    )
    return parser


def parse_source_caps(caps_arg: str) -> Dict[str, int]:
    """Parse JSON or key=val cap specifications."""
    if not caps_arg.strip():
        return {}
    if caps_arg.strip().startswith("{"):
        try:
            return {str(k): int(v) for k, v in json.loads(caps_arg).items()}
        except Exception as exc:
            raise ValueError(f"Invalid JSON in --source-caps: {caps_arg}") from exc
    result = {}
    for part in caps_arg.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            result[k.strip()] = int(v.strip())
    return result


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Handle inspection mode
    if args.inspect_sources:
        print("\n=== Registered Track 5 Sources ===")
        header = f"{'Source':<26} {'Type':<14} {'Default Cap':<12} {'Provenance':<12} {'Redistribution':<16} {'Generator Family':<18}"
        print(header)
        print("-" * len(header))
        for name, cfg in sorted(SOURCE_REGISTRY.items()):
            print(
                f"{name:<26} {cfg.source_type:<14} {cfg.default_cap:<12} {cfg.provenance:<12} {cfg.redistribution_mode:<16} {cfg.generator_family:<18}"
            )
        print(f"\nTotal registered sources: {len(SOURCE_REGISTRY)}")
        sys.exit(0)

    # Push to hub validation
    if args.push_to_hub and not args.repo_id:
        raise ValueError(
            "--push-to-hub was specified but --repo-id is missing! "
            "Provide --repo-id (e.g. --repo-id zye2/techjam-track5-combined)."
        )

    # Initialize environment
    load_local_environment(_PROJECT_ROOT)
    data_root = args.data_root
    if data_root is None:
        data_root_str = os.environ.get("TECHJAM_DATA_ROOT", "E:/techjam26-runtime")
        data_root = Path(data_root_str)

    # Resolve sources to include
    all_sources = dict(SOURCE_REGISTRY)
    if args.sources:
        included_names = {s.strip() for s in args.sources.split(",") if s.strip()}
        all_sources = {k: v for k, v in all_sources.items() if k in included_names}
    if args.exclude_sources:
        excluded_names = {s.strip() for s in args.exclude_sources.split(",") if s.strip()}
        all_sources = {k: v for k, v in all_sources.items() if k not in excluded_names}

    if not all_sources:
        raise ValueError("No sources selected after applying inclusions and exclusions.")

    custom_caps = parse_source_caps(args.source_caps)

    logger.info("Starting combined dataset builder: %d active sources", len(all_sources))
    logger.info("Mode: %s | Output directory: %s", "DRY-RUN" if args.dry_run else "LIVE", args.output_dir)

    all_records: List[Dict[str, Any]] = []

    for name, source_cfg in sorted(all_sources.items()):
        # Determine cap
        cap = custom_caps.get(name, source_cfg.default_cap)
        if args.max_samples_per_source > 0:
            cap = min(cap, args.max_samples_per_source)

        try:
            if source_cfg.source_type == "huggingface":
                records = acquire_huggingface_source(
                    source=source_cfg,
                    cap=cap,
                    seed=args.seed,
                    data_root=data_root,
                    dry_run=args.dry_run,
                    min_short_edge=args.min_short_edge,
                    min_area=args.min_area,
                    min_bytes=args.min_bytes,
                    embed_images=args.embed_images,
                )
            elif source_cfg.source_type == "local_csv":
                records = acquire_local_source(
                    source=source_cfg,
                    cap=cap,
                    seed=args.seed,
                    data_root=data_root,
                    allow_missing=args.allow_missing,
                    dry_run=args.dry_run,
                )
            elif source_cfg.source_type == "local_directory":
                records = acquire_local_directory(
                    source=source_cfg,
                    cap=cap,
                    seed=args.seed,
                    data_root=data_root,
                )
            elif source_cfg.source_type == "reference_only":
                records = acquire_reference_only_source(
                    source=source_cfg,
                    cap=cap,
                    seed=args.seed,
                    data_root=data_root,
                    dry_run=args.dry_run,
                )
            else:
                logger.warning("Unknown source type %s for %s", source_cfg.source_type, name)
                continue
            all_records.extend(records)
        except Exception as exc:
            if args.allow_missing or args.dry_run:
                logger.warning("Error processing source %s (continuing): %s", name, exc)
            else:
                logger.error("Failed on source %s: %s", name, exc)
                raise

    if not all_records:
        raise RuntimeError("No records were acquired. Check source selections, local data, or use --dry-run.")

    logger.info("Total records acquired: %d. Partitioning splits...", len(all_records))
    raw_df = pd.DataFrame(all_records)
    if raw_df.empty:
        raise RuntimeError("All acquired rows were quarantined or filtered")

    # Compute duplicate groups before checking conflicting binary labels.
    partitioned_df = assign_group_disjoint_splits(raw_df, seed=args.seed)
    conflicting_groups = set(
        partitioned_df.groupby("duplicate_group")["ai_positive"]
        .nunique()
        .loc[lambda values: values > 1]
        .index
    )
    if conflicting_groups:
        quarantine = partitioned_df[
            partitioned_df["duplicate_group"].isin(conflicting_groups)
        ].copy()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        quarantine.to_csv(args.output_dir / "quarantine_conflicting_labels.csv", index=False)
        partitioned_df = partitioned_df[
            ~partitioned_df["duplicate_group"].isin(conflicting_groups)
        ].copy()
        logger.warning(
            "Quarantined %d rows across %d conflicting duplicate groups",
            len(quarantine),
            len(conflicting_groups),
        )
    if partitioned_df.empty:
        raise RuntimeError("All acquired rows were quarantined for conflicting labels")

    # Export package
    audit_report = export_dataset_package(
        frame=partitioned_df,
        output_dir=args.output_dir,
        embed_images=args.embed_images,
        data_root=data_root,
    )
    # Network push strictly if requested
    if args.push_to_hub:
        push_dataset_to_hub(
            output_dir=args.output_dir,
            repo_id=args.repo_id,
            private=args.private,
        )
    else:
        logger.info(
            "Push to Hub was not requested (--push-to-hub not specified). "
            "Local package built safely without network upload."
        )

    print("\n=== Dataset Build Summary ===")
    print(f"Total Rows:               {audit_report['total_rows']}")
    print(f"Duplicate Groups:         {audit_report['duplicate_groups_count']}")
    print(f"Splits:                   {audit_report['split_counts']}")
    print(f"AI Positive Distribution: {audit_report['ai_positive_distribution']}")
    print(f"Output Location:          {args.output_dir.resolve()}")
    print("=============================\n")


if __name__ == "__main__":
    main()
