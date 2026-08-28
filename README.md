# TechJam Robust Image Provenance Detector

Prototype for robust three-class image provenance: `authentic`, `tampered`, and
`fully_aigc`. Transformations are training augmentations and evaluation conditions;
the model does not attempt to infer edit history.

## Current machine

- NVIDIA RTX 4080, 16 GB VRAM
- Python 3.11
- PyTorch 2.10.0 with CUDA 13.0
- Large runtime assets should use `E:/techjam26-runtime`, not the repository drive

## Current implementation

The performance-first path is operational:

- pinned 569.5M-parameter Gemma 4 vision encoder cached on `E:`;
- zero missing, unexpected, or mismatched checkpoint tensors;
- separate learned-query fully-AIGC and patch-aware tamper experts;
- optional pretrained ConvNeXt-Tiny over RGB + fixed residuals + radial FFT;
- hierarchical probabilities that sum to one and exclude fully-AIGC samples from tamper loss;
- SID fractional token-mask supervision with focal BCE + soft Dice;
- frozen, last-layer, and all-attention LoRA adaptation paths;
- generator-balanced 50/25/25 sampling and chained post-processing augmentation;
- exact, linked-source, and perceptual near-duplicate grouping before split assignment;
- one-view/three-view evaluation with per-dataset, per-generator, and robustness metrics.

The full local SID selection now contains 10,000 images per provenance class for
training and 1,000 per class for official validation. A pinned DiffusionForensics
mirror contributes 6,000 ADM + 6,000 authentic training images, 500 validation
images, and 5,700 images from unseen generator families. WildFake is acquired by
HTTP-range extraction of selected ZIP members so its multi-million-image release is
never downloaded wholesale.

Full-resolution GPU gates pass on the RTX 4080:

- frozen Gemma + spectral expert: 1.97 GiB peak allocated VRAM;
- all-layer rank-8 LoRA + spectral expert: 4.21 GiB peak allocated VRAM;
- all 108 LoRA attention projections receive the expected first-step gradients;
- real 16-image/two-view optimizer update and resumable checkpoint save pass.

The first raw-source run scored 92% but was invalidated by a severe dataset
shortcut: authentic images were JPEG/non-square while fully-AIGC images were
PNG/square. A trivial format rule nearly reproduced the binary AUROC.

That result is retained only as the obsolete baseline. The replacement model is
currently being trained and must be judged on the leakage-safe combined manifests;
no performance claim is made from a smoke checkpoint.

## Environment

```powershell
Copy-Item .env.example .env
uv venv E:/techjam26-runtime/.venv --python 3.11 --system-site-packages --seed
uv pip install --python E:/techjam26-runtime/.venv/Scripts/python.exe `
    transformers==5.10.1 python-dotenv pytest ruff
uv pip install --python E:/techjam26-runtime/.venv/Scripts/python.exe --no-deps -e .
```

The current system `transformers` package is too old for Gemma 4. This project pins
`transformers==5.10.1`, the first supported runtime used by the official examples.

## Verified commands

```powershell
$python = 'E:/techjam26-runtime/.venv/Scripts/python.exe'

# Unit tests and static checks
& $python -m pytest -q
& $python -m ruff check src tests scripts

# Build the selected SID pool
& $python scripts/prepare_sid.py `
    --per-class 10000 `
    --validation-per-class 1000 `
    --shuffle-buffer 10000

# Selectively acquire WildFake without downloading its full archives
& $python scripts/acquire_wildfake_subset.py

# Acquire the pinned DiffusionForensics subset
& $python scripts/acquire_diffusionforensics_subset.py

# Build leakage-safe combined manifests
& $python scripts/build_performance_manifests.py `
    metadata/sid_sanity.csv `
    metadata/wildfake_subset.csv `
    metadata/diffusionforensics_subset.csv `
    --output-dir splits/performance `
    --compute-hashes `
    --allow-shortfall

# Full combined-model forward/backward integration tests
& $python scripts/smoke_performance_model.py --config configs/performance_local.yaml
& $python scripts/smoke_performance_model.py --config configs/performance_lora_smoke.yaml

# Training; use --max-steps for a bounded test
& $python -m aigc_detector.train --config configs/performance_local.yaml --max-steps 1
```

## Configuration

The current performance configuration is `configs/performance_local.yaml`.
Machine-local roots are supplied by:

```text
TECHJAM_DATA_ROOT
TECHJAM_HF_HOME
TECHJAM_OUTPUT_ROOT
```

## Safety rails

- Always use 1120 visual tokens for the main experiment.
- Keep SID's official splits intact.
- Never put private evaluation images into training or public source control.
- Keep the required `pred` output as `P(fully_aigc)`.
- Use transformations only to train and evaluate provenance robustness.
- Do not infer transformation families, severities, or edit history.
