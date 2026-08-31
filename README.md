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
| **Atom** | Post-ATT Base Student (Normal) | `facebook/dinov3-vitb16-pretrain-lvd1689m` | 768 | 89.4M | Canonical `train` split (51.1k rows) |
| **Atom Super** | Post-ATT Base Student (Super) | `facebook/dinov3-vitb16-pretrain-lvd1689m` | 768 | 89.4M | All eligible rows (87.8k rows) |
| **Quark** | Post-ATT Small Student (Normal) | `facebook/dinov3-vits16-pretrain-lvd1689m` | 384 | 25.1M | Canonical `train` split (51.1k rows) |
| **Quark Super** | Post-ATT Small Student (Super) | `facebook/dinov3-vits16-pretrain-lvd1689m` | 384 | 25.1M | All eligible rows (87.8k rows) |

* **Teacher Variants:**
  * **Normal Teacher (Stage 2):** Trained on the canonical `train` split (51.1k rows), providing the primary source for student distillation.
  * **Super Teacher (Full-Data):** Trained across all eligible rows (87.8k rows) for maximum data coverage.
* **Student Variants:**
  * **Normal Students (`Quark`, `Atom`):** Distilled and hardened with Adversarial Transformation Training (ATT) on the canonical `train` split.
  * **Super Students (`Quark Super`, `Atom Super`):** Distilled and hardened with Adversarial Transformation Training (ATT) across all available rows (`train_super_all.parquet`).
  * **Adversarial Transformation Training (ATT):** Both Normal and Super students undergo ATT. During training, the model evaluates 3 candidate perturbations (JPEG, blur, resize, noise, color, crop) per row under `torch.no_grad()` and optimizes against the most challenging candidate alongside the original image to maximize corruption robustness.
* **Browser Runtime & WebGPU Export:** The browser catalog resolves six immutable Float32 ONNX artifacts from `zye2/mechadetect-models` on Hugging Face; model weights are not stored in this Git repository. ONNX Runtime Web executes them with WebGPU and WebAssembly fallback.
* **Optional Dual-Stream Spectral Expert:** For training experiments on spatial residuals, an optional frequency-domain ConvNeXt-Tiny stream processes RGB plus fixed high-pass spatial residuals (`conv2d` with discrete derivative kernels) and a 32-bin radial 2D FFT energy projection, gated via learned sigmoid parameters. Production student ONNX exports omit this branch to minimize client memory and enable pure browser execution.

### ONNX Precision and Deployment Formats

The public browser release uses Float32 ONNX artifacts:

| Model Variant | Precision | Opset | File Size | Primary Target / Runtime | Status |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **Quark Super** | **Float32** | 17 | **96.1 MB** | **Browser (WebGPU / WASM)** | **Production Default** |
| **Quark (Normal)** | Float32 | 17 | 96.1 MB | Browser baseline | Available |
| **Atom Super** | Float32 | 17 | 341.3 MB | Desktop / Server edge | Available |
| **Atom (Normal)** | Float32 | 17 | 341.3 MB | Desktop / Server edge | Available |

Float32 exports use vectorized batched token extraction (`forward_batched_tokens`). Static INT8 artifacts are excluded from the release because the current post-training quantization policy does not satisfy the numerical-quality gate. See [Static INT8 Release Evaluation](docs/int8_release_evaluation.md) for complete results, the diagnosed activation-range failure, the PTQ-versus-QAT tradeoff, and the requirements for a future INT8 release.

---

## 3. Training Data & Transparency

The training manifests, split distributions, and source audit reports are publicly hosted and versioned on Hugging Face:

👉 [**`zye2/tj-data` Dataset Repository**](https://huggingface.co/datasets/zye2/tj-data)

### Dataset Structure & Split Allocation

The declared dataset package contains **122,344 total records** across 29 active forensic and natural cohorts. Preflight verification quarantined 30,455 unmaterializable or conflicting records into `exclusions.parquet`, leaving **87,793 verified, decodable, and clean eligible records**.

| Manifest / Split | Row Count | Percentage | Class Balance (AI-Pos / Auth) | Dataset Scope & Usage |
| :--- | :---: | :---: | :---: | :--- |
| **`train.parquet`** | **51,107** | 58.2% | 28,594 / 22,513 | Canonical training split used by Normal models (`Quark`, `Atom`, Normal Teacher) |
| **`train_super_all.parquet`** | **87,793** | 100.0% | 49,120 / 38,673 | Full eligible training split used by Super models (`Quark Super`, `Atom Super`, Super Teacher) |
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

### 4.4 End-to-end training

Open [`train.ipynb`](train.ipynb) from the repository root. It is the maintained
training path and runs the complete single-GPU workflow:

1. download the pinned manifest package and acquire source images;
2. freeze leak-free train, validation, test, and calibration splits;
3. train the frozen-backbone and unfrozen teacher stages;
4. evaluate and promote the teacher;
5. distill Quark (ViT-S) and Atom (ViT-B) sequentially;
6. run Adversarial Transformation Training on both students;
7. evaluate clean and transformed images, apply the ATT gate, and export ONNX.

Launch Jupyter without adding notebook packages to the locked training
environment:

```bash
uv run --with jupyter jupyter lab train.ipynb
```

The notebook defaults to the complete run. Set `SMOKE_TEST = True` in its first
configuration cell to exercise every training path with two updates. A single
NVIDIA GPU is sufficient; the complete teacher run is long.

All tracks preserve effective batch 48:

| Track | Physical batch | Accumulation |
|---|---:|---:|
| Teacher Stage 1 | 6 | 8 |
| Teacher Stage 2 | 2 | 24 |
| Quark / ViT-S distillation | 12 | 4 |
| Atom / ViT-B distillation | 3 | 16 |
| Quark ATT | 4 | 12 |
| Atom ATT | 2 | 24 |

The underlying CLIs remain directly usable. For example:

```bash
uv run python -m aigc_detector.train \
  --config configs/teacher_dinov3_stage1_clean_frozen.yaml
```

Resume only with the same world size, physical batch, and accumulation used to
create the checkpoint.

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

### 4.6 Predicting on an image directory

The submission entry point downloads the selected Float32 ONNX artifact from
Hugging Face on first use, then writes exactly the required `image_path` and
`pred` fields:

```bash
uv run python predict.py \
  --input /path/to/images \
  --output predictions.json
```

Quark Super is the default because it is the smallest model. Use
`--model atom-super-float32` for the larger student. Private model access
requires `HF_TOKEN`.

### 4.7 Model Export

Export a trained student checkpoint to opset 17 Float32 ONNX:

```bash
uv run python scripts/export_onnx_webgpu.py \
  --checkpoint outputs/att_student_small/checkpoint-final.pt \
  --variant small \
  --output outputs/models/mechadetect-quark-super-post-att-float32.onnx
```

### 4.8 Repository checks
Project checks are available under `tests/`; training does not invoke them.

---

## 5. Repository layout

```text
techjam26/
├── train.ipynb                     # Complete one-GPU training workflow
├── predict.py                      # Required directory-to-JSON inference CLI
├── configs/                        # Teacher, student, and ATT configurations
├── scripts/
│   ├── data_prep/
│   │   ├── acquire_all_images.py
│   │   └── freeze_production_eligible.py
│   ├── distill_student.py
│   ├── train_att.py
│   ├── evaluate_teacher.py
│   ├── evaluate_performance.py
│   ├── promote_teacher.py
│   ├── check_att_gate.py
│   └── export_onnx_webgpu.py
├── src/aigc_detector/              # Model, data, losses, metrics, and training
├── web/                            # Existing client-side WebGPU/WASM demo
└── tests/
```

## 6. Track 5 expected deliverables

### Project description and technical stack

MechaDetect estimates the probability that an image is fully AI-generated or
semantically edited by a generative model. Training uses PyTorch, CUDA, and
Hugging Face Transformers with a DINOv3 ViT-H+/16 teacher; Quark and Atom are
distilled ViT-S and ViT-B students. Pillow and NumPy implement deterministic
JPEG, blur, resize, noise, colour, and crop transformations. Pandas and PyArrow
store the audited manifests. ONNX Runtime provides portable inference.

Development uses Git, VS Code or Jupyter, `uv`, Python 3.11, and NVIDIA CUDA.
Training data combines public or properly licensed image-forensics sources,
including SID, WildFake, and DiffusionForensics. Exact hashes, perceptual
duplicate groups, source revisions, and split membership are recorded. The
COCO val2017 and WildFake DALL-E Advanced organizer demonstration set is
explicitly blocked from training.

### Required prediction format

[`predict.py`](predict.py) recursively reads an image directory or accepts one
image. It outputs a JSON array with only the required fields; `pred` is
$P(\text{AI-generated or AI-edited})$:

```json
[
  {
    "image_path": "images/example.jpg",
    "pred": 0.8731
  }
]
```

### Robustness evaluation summary

The organizer demonstration benchmark contains 13,841 held-out images: 4,998
authentic COCO val2017 images and 8,843 WildFake DALL-E Advanced images. All
four Float32 students were evaluated on every image under clean input and 14
transform conditions.

| Model | Clean AUROC | Mean transformed AUROC | Worst transformed AUROC | Worst condition |
| :--- | ---: | ---: | ---: | :--- |
| Quark Normal | 0.9921 | 0.9871 | 0.9691 | quarter resize |
| **Quark Super** | **0.9947** | **0.9931** | **0.9870** | quarter resize |
| Atom Normal | 0.9973 | 0.9945 | 0.9876 | half resize |
| **Atom Super** | **0.9980** | **0.9967** | **0.9928** | quarter resize |

| Condition | Quark Super AUROC | Atom Super AUROC |
| :--- | ---: | ---: |
| Clean | 0.9947 | 0.9980 |
| JPEG quality 30 | 0.9963 | 0.9989 |
| Gaussian blur 2.0 | 0.9887 | 0.9958 |
| Quarter resize | 0.9870 | 0.9928 |
| Gaussian noise 0.10 | 0.9885 | 0.9940 |
| Colour jitter 20% | 0.9947 | 0.9979 |
| Centre crop 80% | 0.9942 | 0.9980 |

### Error analysis

At the fixed operating threshold, the main failure mode is false positives on
authentic images after strong blur or downsampling—not missed synthetic
images. On clean input, Quark Super produced 811 false positives and 54 false
negatives; Atom Super reduced these to 307 false positives and 30 false
negatives. Quarter resize increased false positives to 1,418 for Quark Super
and 1,197 for Atom Super, while false negatives remained low at 60 and 15.

This indicates that aggressive resampling removes camera/detail evidence and
pushes authentic images toward the AI-positive side of the threshold. The
larger Atom model ranks transformed examples more reliably, but costs 341 MB
instead of 96 MB. Quark is therefore the practical browser default; Atom is
the accuracy-first option. Applications should treat uncertain scores as
review signals rather than proof and should calibrate the threshold for their
false-positive tolerance.

### Submission checklist

- **Written description:** the overview, method, stack, data, results, error
  analysis, and limitations are present in this README for transfer to Devpost.
- **Public repository:** source, configs, [`train.ipynb`](train.ipynb), and the
  required [`predict.py`](predict.py) interface are included.
- **Demo video:** record the existing web demo and `predict.py` end-to-end,
  upload it publicly to YouTube, and link it from Devpost. This external step
  cannot be completed inside the repository.
- **Robustness summary:** included above.
- **Error analysis:** included above.

## 7. Known limitations

- The output is a probabilistic screening score, not cryptographic provenance,
  proof of authorship, a watermark check, or a copyright decision.
- Strong resize, crop, or blur can remove evidence and increase false positives
  on authentic images.
- Generator families and post-processing pipelines continue to evolve; the
  model requires monitoring and periodic evaluation on new held-out sources.
- The 872.6M-parameter teacher is practical for offline training but not
  constrained deployment. Quark exists specifically for the browser path.
- The model repository currently requires Hugging Face authentication. A public
  browser deployment requires approved model licensing and redistribution terms.

## 8. Team Contributions

Contributor identities and role allocation were not provided in this
repository context. The submitting team must add each member's name and exact
contributions before Track 5 submission; this repository does not invent or
attribute work without confirmation.
