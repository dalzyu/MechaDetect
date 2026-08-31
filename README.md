# MechaDetect

MechaDetect estimates whether an image is authentic or AI-positive. “AI-positive”
covers both fully generated images and images whose semantic content was changed
with a generative model. Ordinary JPEG recompression, resizing, blur, noise,
colour adjustment, and cropping do not change the label.

The repository contains the data controls, training code, browser export path,
and evaluation tools used for TechJam 2026 Track 5. A score is evidence for
screening; it is not proof of authorship or ownership.

---

## 1. Backbone decision

A controlled four-GPU tournament on 12,000 generator-balanced images selected
DINOv3 ViT-H+/16 (`facebook/dinov3-vith16plus-pretrain-lvd1689m`) as the teacher backbone.

### 3-Way Tournament Summary

| Dimension | DINOv3 ViT-H+/16 (Winner) | PE-Spatial-G/14 (Runner-Up) | Gemma 4 Tower (Eliminated) |
| :--- | :---: | :---: | :---: |
| **Parameters (Ceiling < 2B)** | **840.6M** | 1,851.9M | **569.5M** |
| **Tokens per Image** | **196** ($14 \times 14$ grid at 224px) | 1,024 ($32 \times 32$ grid at 448px) | 1,120 soft tokens |
| **Strict-Unseen AUROC (480 img)** | **0.9794** [0.9689, 0.9899] | **0.9946** [0.9899, 0.9981] | 0.9548 [0.9331, 0.9721] |
| **TechJam Proxy AUROC (COCO vs DALL·E 3)** | **0.9978** [0.9955, 0.9995] | 0.9538 [0.9344, 0.9692] | 0.9235 [0.8942, 0.9455] |
| **DALL·E 3 Generator Recall** | **96.5%** (193/200 correct) | 48.5% (82 misclassified as tampered) | 65.5% (64 misclassified) |
| **Aspect Ratio Shortcut Accuracy (Chance=33%)**| **54.66%** (Near-chance / invariant) | **86.61%** (Severe shortcut leak) | **80.53%** (Severe shortcut leak) |
| **Adaptation Speed (Time per Update)** | **~1.8 seconds** | ~35.0 seconds (19.4× slower) | ~14.0 seconds |

Full empirical findings, 16-condition robustness tables, and failure analyses:
- Comprehensive findings: [`docs/backbone_bakeoff_findings.md`](docs/backbone_bakeoff_findings.md)
- Executive decision & checkpoint handoff: [`docs/backbone_bakeoff_decision.md`](docs/backbone_bakeoff_decision.md)
- Production teacher plan: [`docs/teacher_training_plan.md`](docs/teacher_training_plan.md)
- Consolidated teacher training run record and published weights: [`docs/training_run_consolidated.md`](docs/training_run_consolidated.md)

---

## 2. Model Architecture

MechaDetect uses a unified architecture across both its teacher and student models.
The prediction objective is binary: authentic images are negative ($y=0$), while
fully synthetic images and localized generative edits share a single positive label
($y=1$).

### Architectural Topology

```text
[ Input RGB Image: 224 × 224 × 3 ]
       │
       ▼
[ Vision Backbone: DINOv3 ViT (Patch Size 16) ]
       │  1 CLS + 4 register prefix tokens stripped
       │  Output: B × 196 patch tokens × encoder_dim
       ▼
[ Token Adapter: LayerNorm + Linear(encoder_dim → 512) ]
       │  Output: B × 196 adapted tokens × 512
       ├───────────────────────────────────────────────┐
       ▼                                               ▼
[ Global AI-Evidence Head ]                 [ Edit Localization Head ]
• 4 learned query vectors                   • Token classifier (512 → 1 score)
• Multi-head cross-attention (4 heads)      • Softmax attention pooling (512)
• Mean (512) + Std (512) summary pooling    • Top-5% patch pooling (k=10 tokens, 512)
• Concat 6 × 512 = 3072 dims                • 1 learned global query (512)
• Projection: 3072 → 256 (GELU + Dropout)   • Concat 3 × 512 = 1536 dims
       │                                    • Projection: 1536 → 256 (GELU + Dropout)
       │                                               │
       └───────────────────────┬───────────────────────┘
                               ▼
             [ Binary AI-Positive Classifier ]
             • Concat global + local features (256 + 256 = 512 dims)
             • Linear(512 → 1) → ai_positive_logit
             • P(AI-positive) = sigmoid(logit)
             • P(authentic) = 1 - P(AI-positive)
```

### Model Family & Training Lineage

The architecture is parameterized across three scales:

| Variant | Role | Backbone Identifier | Encoder Dim | Parameters | Training Data Scope |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Normal Teacher** | Primary Distillation Source | `facebook/dinov3-vith16plus-pretrain-lvd1689m` | 1280 | 872.6M | Canonical `train` split (51.1k rows) |
| **Super Teacher** | Full-Data Adaptation | `facebook/dinov3-vith16plus-pretrain-lvd1689m` | 1280 | 872.6M | All eligible rows (87.8k rows) |
| **Quark** | Post-ATT Base Student (Normal) | `facebook/dinov3-vitb16-pretrain-lvd1689m` | 768 | 89.4M | Canonical `train` split (51.1k rows) |
| **Quark Super** | Post-ATT Base Student (Super) | `facebook/dinov3-vitb16-pretrain-lvd1689m` | 768 | 89.4M | All eligible rows (87.8k rows) |
| **Atom** | Post-ATT Small Student (Normal) | `facebook/dinov3-vits16-pretrain-lvd1689m` | 384 | 25.1M | Canonical `train` split (51.1k rows) |
| **Atom Super** | Post-ATT Small Student (Super) | `facebook/dinov3-vits16-pretrain-lvd1689m` | 384 | 25.1M | All eligible rows (87.8k rows) |

* **Teacher Variants:**
  * **Normal Teacher (Stage 2):** Trained on the canonical `train` split (51.1k rows), providing the primary source for student distillation.
  * **Super Teacher (Full-Data):** Trained across all eligible rows (87.8k rows) for maximum data coverage.
* **Student Variants:**
  * **Normal Students (`Atom`, `Quark`):** Distilled and hardened with Adversarial Transformation Training (ATT) on the canonical `train` split.
  * **Super Students (`Atom Super`, `Quark Super`):** Distilled and hardened with Adversarial Transformation Training (ATT) across all available rows (`train_super_all.parquet`).
  * **Adversarial Transformation Training (ATT):** Both Normal and Super students undergo ATT. During training, the model evaluates 3 candidate perturbations (JPEG, blur, resize, noise, color, crop) per row under `torch.no_grad()` and optimizes against the most challenging candidate alongside the original image to maximize corruption robustness.
* **Browser Runtime & WebGPU Export:** The browser catalog resolves six immutable Float32 ONNX artifacts from `zye2/mechadetect-models` on Hugging Face; model weights are not stored in this Git repository. ONNX Runtime Web executes them with WebGPU and WebAssembly fallback.
* **Optional Dual-Stream Spectral Expert:** For training experiments on spatial residuals, an optional frequency-domain ConvNeXt-Tiny stream processes RGB plus fixed high-pass spatial residuals (`conv2d` with discrete derivative kernels) and a 32-bin radial 2D FFT energy projection, gated via learned sigmoid parameters. Production student ONNX exports omit this branch to minimize client memory and enable pure browser execution.

### ONNX Precision and Deployment Formats

The public browser release uses Float32 ONNX artifacts:

| Model Variant | Precision | Opset | File Size | Primary Target / Runtime | Status |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **Atom Super** | **Float32** | 17 | **96.1 MB** | **Browser (WebGPU / WASM)** | **Production Default** |
| **Atom (Normal)** | Float32 | 17 | 96.1 MB | Browser baseline | Available |
| **Quark Super** | Float32 | 17 | 341.3 MB | Desktop / Server edge | Available |
| **Quark (Normal)** | Float32 | 17 | 341.3 MB | Desktop / Server edge | Available |

Float32 exports use vectorized batched token extraction (`forward_batched_tokens`). Static INT8 artifacts are excluded from the release because the current post-training quantization policy does not satisfy the numerical-quality gate. See [Static INT8 Release Evaluation](docs/int8_release_evaluation.md) for complete results, the diagnosed activation-range failure, the PTQ-versus-QAT tradeoff, and the requirements for a future INT8 release.

---

## 3. Training Data & Transparency

The training manifests, split distributions, and source audit reports are publicly hosted and versioned on Hugging Face:

👉 [**`zye2/tj-data` Dataset Repository**](https://huggingface.co/datasets/zye2/tj-data)

### Dataset Structure & Split Allocation

The declared dataset package contains **122,344 total records** across 29 active forensic and natural cohorts. Preflight verification quarantined 30,455 unmaterializable or conflicting records into `exclusions.parquet`, leaving **87,793 verified, decodable, and clean eligible records**.

| Manifest / Split | Row Count | Percentage | Class Balance (AI-Pos / Auth) | Dataset Scope & Usage |
| :--- | :---: | :---: | :---: | :--- |
| **`train.parquet`** | **51,107** | 58.2% | 28,594 / 22,513 | Canonical training split used by Normal models (`Atom`, `Quark`, Normal Teacher) |
| **`train_super_all.parquet`** | **87,793** | 100.0% | 49,120 / 38,673 | Full eligible training split used by Super models (`Atom Super`, `Quark Super`, Super Teacher) |
| `validation.parquet` | 14,617 | 16.6% | 8,187 / 6,430 | Validation split for threshold calibration and model promotion gates |
| `test.parquet` | 11,129 | 12.7% | 6,233 / 4,896 | In-distribution generalization test benchmark |
| `test_unseen.parquet` | 10,940 | 12.5% | 10,932 / 8 | Out-of-distribution benchmark evaluating unseen generator families |
| `calibration.parquet` | 4,096 | 4.7% | 2,294 / 1,802 | Strictly isolated split used exclusively for static INT8 PTQ calibration |
| `exclusions.parquet` | 30,455 | — | — | Quarantined records (missing remote bytes, unaligned masks, cross-label conflicts) |

### Cohort Composition

The dataset consists of authentic negative anchors and generative positive cohorts across 29 collections:

* **Authentic Negatives (38.7k images):**
  * *Photography & Portraits (21.8k):* Real camera captures from SID (9.4k), CelebA-HQ human portraits (4.8k), AFHQ animal faces (3.3k), DiffusionForensics natural photo anchors (4.1k), and Open Images v7 (0.2k).
  * *Art & Illustrations (15.9k):* Public domain museum scans (Art Museums PD 7.6k, Artic 6.3k, Classical Figure Art 1.0k) and hand-drawn Manga109 illustrations (1.0k).
  * *3D CGI & Gaming (1.0k):* Video game captures (GTA 5 0.5k, fantasy gaming 0.25k) and 3D Blender animation (Sintel 0.25k).
* **Generative Positives (49.1k images):**
  * Midjourney v5/v6/Niji (6.6k), FLUX.1 [dev] (5.3k), GPT-Image-Edit (5.0k), Stable Diffusion SD 1.x/2.x/3/XL (4.9k), diffusion baselines (DDPM, DDIM, ADM, LDM, 15.0k), Ideogram v2 (2.9k), Krea 2 (2.4k), Google Nano Banana edited/pro (1.9k), and specialized engines (DALL·E 2, VQDM, Danbooru 2026 AIGC, 5.2k).

### Benchmark Isolation & Zero-Leakage Audit

* **Zero Sample Overlap:** 0 identical SHA-256 hashes across train, validation, test, and calibration splits.
* **Zero Duplicate Group Leakage:** All 75,168 perceptual difference hash (dHash) and SHA-256 duplicate clusters are strictly isolated within individual splits.
* **Strict Unseen Generator Separation:** 0 AI generator leakage into train. The 15 positive generator families in `test_unseen` (FLUX.1 [dev], DALL·E 2, SDXL 1.0, SD v1/v2, IF, LDM, IDDPM, PNDM, VQDM, etc.) are 100% absent from the training splits.
* **Organizer Demo Exclusion:** 0 overlap (0 file paths, 0 SHA-256 hashes) with the official TechJam demonstration evaluation dataset ($13,841$ images: $4,998$ COCO val2017 authentic + $8,843$ WildFake DALL-E Advanced images).
* **Forbidden Cohort Rejection:** The forbidden directory `newer image model data(do not use for training)` is completely blocked by fail-closed preflight guards.

---

## 4. Quickstart & Usage

### 4.1 Installation

This repository uses [uv](https://docs.astral.sh/uv/) to create the project
virtual environment and install the exact versions recorded in `uv.lock`.
Install uv once, then run the matching setup sequence for your shell:

**Windows PowerShell**

```powershell
irm https://astral.sh/uv/install.ps1 | iex
uv venv --python 3.11
uv sync --locked --dev
Copy-Item .env.example .env
notepad .env
```

**POSIX shell (Linux, including the training cluster)**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.11
uv sync --locked --dev
cp .env.example .env
$EDITOR .env
```

`uv sync` creates (or updates) `.venv`; `--locked` refuses to change
`uv.lock`, and `--dev` installs the test/lint tools from the `dev` dependency
group. You do not need to activate the environment when using `uv run`, which
always executes against this project environment. If you prefer activation:

```powershell
.venv\Scripts\Activate.ps1
```

```bash
source .venv/bin/activate
```

The lockfile selects the CUDA 13.0 PyTorch wheels (`torch==2.13.0` and
`torchvision==0.28.0`) from the explicit PyTorch index. An NVIDIA driver
compatible with those wheels is therefore required; this project does not
provide a CPU or macOS dependency fallback.

Run Python tools reproducibly through uv:

```bash
uv run python -m aigc_detector.predict --help
uv run pytest tests/ -q
```

On PowerShell, the same `uv run ...` commands apply.

### 4.2 Environment Setup

Copy `.env.example` to `.env` and set machine-local storage paths:
```bash
TECHJAM_DATA_ROOT=E:/techjam26-runtime/data
TECHJAM_HF_HOME=E:/techjam26-runtime/huggingface
TECHJAM_OUTPUT_ROOT=E:/techjam26-runtime/outputs
```

### 4.3 Preparing the Production Eligible Manifests

Data prefetch and manifest generation are performed prior to DDP:

```bash
# 1. Prefetch source images to local NVMe
uv run python scripts/data_prep/acquire_all_images.py \
  --data-root "$TECHJAM_DATA_ROOT" \
  --resume

# 2. Freeze immutable eligible Parquet splits and 4,096-row calibration split
uv run python scripts/data_prep/freeze_production_eligible.py \
  --data-root "$TECHJAM_DATA_ROOT" \
  --output-dir splits/production_eligible \
  --calibration-size 4096 \
  --strict \
  --verify-bytes
```

The builder creates `train.parquet`, `validation.parquet`, `test.parquet`, `test_unseen.parquet`,
`calibration.parquet`, `exclusions.parquet`, and `audit_report.json`. Production loaders
fail closed and forbid missing-image fallback substitution.

### 4.4 Training

The maintained path starts on one CUDA GPU and keeps an effective record batch
of 48 through gradient accumulation:

| Track | Physical batch | Accumulation | Effective batch |
|---|---:|---:|---:|
| Teacher Stage 1 | 6 | 8 | 48 |
| Teacher Stage 2 | 2 | 24 | 48 |
| Atom / ViT-S distillation | 12 | 4 | 48 |
| Quark / ViT-B distillation | 3 | 16 | 48 |
| Atom ATT | 4 | 12 | 48 |
| Quark ATT | 2 | 24 | 48 |

Prepare the workstation and run the resumable teacher → students → ATT flow:

```bash
bash cluster_setup.sh
bash scripts/launch_production.sh
```

`GPU_DEVICES=0` is the default. An explicit list enables DDP while the launcher
reduces accumulation to preserve the same effective batch:

```bash
GPU_DEVICES=0,1 bash scripts/launch_production.sh
```

Both student tracks and both ATT tracks run sequentially unless their launchers
receive `--parallel-tracks` with disjoint device pools. This avoids loading two
large models onto a one-GPU workstation.

Run a teacher stage directly with plain Python:

```bash
uv run python -m aigc_detector.train \
  --config configs/teacher_dinov3_stage1_clean_frozen.yaml
```

Resume only with the same world size, physical batch, and accumulation used to
create the checkpoint:

```bash
uv run python -m aigc_detector.train \
  --config configs/teacher_dinov3_stage2_paired_unfrozen.yaml \
  --resume /absolute/path/to/checkpoint-step-N.pt
```

The former `orchestrate_4x4090.sh` state machine remains available for the
original four-RTX-4090 Vast workflow. It is an explicit specialist path, not the
default training command.

The operational contracts are documented in
[`docs/production_training_and_delivery_plan.md`](docs/production_training_and_delivery_plan.md)
and [`docs/teacher_training_plan.md`](docs/teacher_training_plan.md).

### 4.5 Evaluating a Checkpoint


Run clean evaluation and the complete single-transform robustness grid on a
candidate production checkpoint:

```bash
uv run python scripts/evaluate_performance.py \
  --manifest splits/production_eligible/test_unseen.parquet \
  --checkpoint /path/to/checkpoint-step-N.pt \
  --config configs/teacher_dinov3_stage2_paired_unfrozen.yaml \
  --output outputs/teacher-clean-unseen.json \
  --batch-size 1

uv run python scripts/evaluate_performance.py \
  --manifest splits/production_eligible/test_unseen.parquet \
  --checkpoint /path/to/checkpoint-step-N.pt \
  --config configs/teacher_dinov3_stage2_paired_unfrozen.yaml \
  --output outputs/teacher-robustness-unseen.json \
  --robustness \
  --batch-size 1
```

### 4.6 Predicting on an Image Directory
Generate Track 5 predictions (`pred = P(AI-generated or AI-edited)`) for submissions:
```bash
uv run python -m aigc_detector.predict \
  --input-dir /path/to/images \
  --output predictions.json \
  --config configs/teacher_dinov3_stage2_paired_unfrozen.yaml \
  --checkpoint /path/to/checkpoint-step-N.pt
```

### 4.7 Model Export

Export a trained student checkpoint to opset 17 Float32 ONNX:

```bash
uv run python scripts/export_onnx_webgpu.py \
  --checkpoint outputs/att_student_small/checkpoint-final.pt \
  --variant small \
  --output outputs/models/mechadetect-atom-super-post-att-float32.onnx
```

### 4.8 Repository checks
Project checks are available under `tests/`; training does not invoke them.

---

## 5. Repository Layout

```text
techjam26/
├── configs/                       # Production and experiment configurations
│   ├── teacher_dinov3_stage1_clean_frozen.yaml
│   ├── teacher_dinov3_stage2_paired_unfrozen.yaml
│   ├── student_dinov3_small_distill.yaml
│   ├── student_dinov3_base_distill.yaml
│   ├── att_student_small.yaml
│   └── att_student_base.yaml
├── docs/                          # Architecture decisions, empirical findings, and plans
│   ├── production_training_and_delivery_plan.md # Authoritative delivery record
│   ├── training_dataset_specification.md        # Dataset caps and provenance rules
│   ├── teacher_training_plan.md                 # Teacher stage specifications
│   └── backbone_bakeoff_findings.md             # 3-way empirical bake-off report
├── scripts/                       # Training, evaluation, and export entry points
│   ├── launch_production.sh       # Resumable 1-GPU production pipeline orchestrator
│   ├── launch_students_distill.py # Sequential student distillation runner
│   ├── launch_att_tracks.py       # Sequential ATT track runner
│   ├── distill_student.py         # Knowledge distillation training engine
│   ├── train_att.py               # Adversarial transformation training engine
│   ├── evaluate_teacher.py        # Checkpoint evaluation and metrics reporting
│   ├── promote_teacher.py         # Validation gating and promotion verification
│   └── export_onnx_webgpu.py      # ONNX export with parity verification
├── src/aigc_detector/             # Reusable core library package
│   ├── model.py                   # ProvenanceModel, ProvenanceHead, backbones, adapters
│   ├── train.py                   # Optimization loop, LLRD, EMA, checkpoint save/restore
│   ├── predict.py                 # Binary batch inference and submission formatting
│   ├── dataset.py                 # PairedImageDataset, CachedFeatureDataset
│   ├── sampling.py                # DeterministicDistributedCoverageSampler
│   ├── preprocessing.py           # Standardized geometry and image normalization
│   ├── transforms.py              # Perturbation pipeline (JPEG, blur, noise, resize, crop)
│   ├── losses.py                  # BCE, consistency, and mask losses
│   ├── metrics.py                 # AUROC, AUPRC, balanced accuracy, confusion matrix
│   ├── runtime.py                 # Distributed runtime setup and environment loading
│   └── static_int8.py             # Calibration and PTQ static quantization engine
├── web/                           # Client-side WebGPU browser application
│   ├── index.html                 # UI layout and screening interface
│   ├── app.js                     # ONNX Runtime Web WebGPU/WASM controller
│   ├── serve.py                   # Static server with COOP/COEP isolation headers
│   └── model/
│       └── metadata.json          # Remote Hugging Face model catalog
└── tests/                         # Unit and integration test suite
```

---

## 6. Track 5 Submission Interface

### Solution and stack

The model uses PyTorch for optimization and DDP, Hugging Face Transformers for
the pinned DINOv3 encoder and image processor, torchvision for the optional
ConvNeXt spectral branch, Pillow/NumPy for deterministic transformations, and
pandas/PyYAML for manifests and configuration.

Training data is drawn from SID, WildFake, and DiffusionForensics under their
respective licenses and usage terms. The production split is grouped by exact
and perceptual duplicate identity and separates unseen generator families.
COCO val2017 and DALL-E Advanced organizer demonstration data is explicitly
blocked from training.

### Required prediction format

`python -m aigc_detector.predict` recursively reads an image directory and
writes a JSON array. `pred` is the combined probability that an image is fully
AI-generated or AI-edited:

```json
[
  {
    "image_path": "relative/path/example.jpg",
    "pred": 0.8731
  }
]
```

The manifests retain fully generated and tampered subtypes for diagnostics and
optional localization supervision. The image-level model is binary, so the
exported score covers both.

### Organizer Demonstration Benchmark Results

The completed organizer benchmark uses `metadata/organizer_demo_document_count.csv`: **13,841 images** consisting of **4,998 authentic COCO val2017 images** and **8,843 AIGC WildFake DALL-E Advanced images**. The demo set was excluded from the production-eligible training manifest by path and SHA-256 audit. Each artifact was evaluated on all 13,841 images under 15 conditions: clean, JPEG quality 90/70/50/30, Gaussian blur sigma 0.5/1.0/2.0, resize to 0.50x/0.25x followed by upscaling, Gaussian noise sigma 0.02/0.05/0.10, color jitter ±20%, and center crop 80%.

| Model | Format | Clean AUROC | Mean transformed AUROC | Worst transformed AUROC | Worst condition | Clean AIGC recall | Clean authentic recall |
| :--- | :--- | ---: | ---: | ---: | :--- | ---: | ---: |
| Atom (Normal) | Float32 | 0.9921 | 0.9871 | 0.9691 | `resize_quarter` | 97.60% | 92.60% |
| **Atom Super** | **Float32** | **0.9947** | **0.9931** | **0.9870** | `resize_quarter` | **99.39%** | **83.77%** |
| Quark (Normal) | Float32 | 0.9973 | 0.9945 | 0.9876 | `resize_half` | 98.94% | 94.82% |
| **Quark Super** | **Float32** | **0.9980** | **0.9967** | **0.9928** | `resize_quarter` | **99.66%** | **93.86%** |

Full per-condition AUROC:

| Model | clean | jpeg90 | jpeg70 | jpeg50 | jpeg30 | blur0.5 | blur1.0 | blur2.0 | resize_half | resize_quarter | noise0.02 | noise0.05 | noise0.10 | color_jitter20 | crop80 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Atom (Normal) Float32 | 0.9921 | 0.9939 | 0.9963 | 0.9960 | 0.9950 | 0.9911 | 0.9826 | 0.9763 | 0.9786 | 0.9691 | 0.9912 | 0.9889 | 0.9803 | 0.9919 | 0.9885 |
| Atom Super Float32 | 0.9947 | 0.9955 | 0.9968 | 0.9967 | 0.9963 | 0.9947 | 0.9922 | 0.9887 | 0.9907 | 0.9870 | 0.9943 | 0.9928 | 0.9885 | 0.9947 | 0.9942 |
| Quark (Normal) Float32 | 0.9973 | 0.9978 | 0.9992 | 0.9991 | 0.9988 | 0.9977 | 0.9935 | 0.9901 | 0.9876 | 0.9884 | 0.9965 | 0.9938 | 0.9884 | 0.9972 | 0.9954 |
| Quark Super Float32 | 0.9980 | 0.9976 | 0.9986 | 0.9989 | 0.9989 | 0.9979 | 0.9965 | 0.9958 | 0.9939 | 0.9928 | 0.9975 | 0.9961 | 0.9940 | 0.9979 | 0.9980 |

Quark Super Float32 reached **0.9980 clean AUROC** and **0.9928 worst transformed AUROC**. Atom Super Float32 reached **0.9870 worst transformed AUROC** while remaining the browser-sized release.

### Inference Speed & Footprint

Measured independently per ONNX artifact with ONNX Runtime graph optimizations enabled. GPU results use the NVIDIA GeForce RTX 4080 and CUDAExecutionProvider with batch size 64; CPU results use the Intel Core i9-13900KF with 16 intra-op threads and batch size 32. Single-image latency is measured at batch size 1. Inputs are synthetic 224 × 224 RGB tensors, so image decode, resize, upload, and browser scheduling are not included.

| Model | Format | File size | GPU p50 batch 1 | GPU throughput batch 64 | CPU p50 batch 1 | CPU throughput batch 32 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Atom (Normal) | Float32 | 96.1 MB | 6.32 ms | 1,495.2 img/s | 23.01 ms | 69.3 img/s |
| **Atom Super** | **Float32** | **96.1 MB** | **6.50 ms** | **1,554.6 img/s** | **22.07 ms** | **72.3 img/s** |
| Quark (Normal) | Float32 | 341.3 MB | 6.95 ms | 538.9 img/s | 48.16 ms | 22.7 img/s |
| Quark Super | Float32 | 341.3 MB | 7.00 ms | 534.7 img/s | 49.26 ms | 22.5 img/s |

The Atom family is the lightweight deployment path: the Float32 artifact is 96.1 MB and reaches roughly 1.5k images/s batched on this GPU. Reproduce the measurements with:

```bash
uv run python scripts/benchmark_model_speed.py \
  --models-dir outputs/models \
  --output-json outputs/benchmark_results.json
```

### Delivered artifacts and evaluation status

Teacher, Atom, Quark, Atom Super, and Quark Super checkpoints have been trained and delivered. The web demo defaults to the Atom Super Float32 export for optimal WebGPU performance and numerical consistency across browsers.

---

## 7. Known Limitations

- This is a probabilistic detector, not a cryptographic provenance proof,
  watermark verifier, copyright decision, or authorship certificate.
- The optional edit-localization output requires localized-edit coverage and
  mask quality. It is not a second image-level provenance classifier.
- Heavy crop, resize, blur, or JPEG processing can remove evidence; unusual
  authentic post-processing can resemble generative artifacts.
- The DINOv3 teacher is for training and distillation, not constrained deployment.

---

## 8. Team Contributions

Contributor identities and role allocation were not provided in this
repository context. The submitting team must add each member's name and exact
contributions before Track 5 submission; this repository does not invent or
attribute work without confirmation.
