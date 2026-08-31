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
DINOv3 ViT-H+/16 (`facebook/dinov3-vith16plus-pretrain-lvd1689m`) as the teacher
backbone. The table below records that historical comparison; it is not a claim
that the delivered MechaDetect students beat the current state of the art.

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

The image-level model is binary. A global AI-evidence branch and a token-aware
edit-localization branch feed one shared AI-positive classifier:

```
[ Input RGB Image ]
       │
       ▼
[ Vision Backbone: DINOv3 ViT-H+/16 ]
       │  (Output: B × N × 1280 patch tokens)
       ▼
[ Token Adapter: LayerNorm + Linear(1280 → 512) ]
       │  (Output: B × N × 512 adapted tokens)
       ├───────────────────────────────────────────────┐
       ▼                                               ▼
[ Global AI-Evidence Head ]                 [ Edit Localization Head ]
• 4 learned query vectors                   • Token-level linear classifier
• Multi-head cross-attention                • Top-5% patch pooling
• Mean + Std summary pooling                • Softmax attention pooling
       │                                               │
       └───────────────────────┬───────────────────────┘
                               ▼
             [ Binary AI-Positive Classifier ]
             • ai_positive_logit
             • P(AI-positive) = sigmoid(logit)
             • P(authentic) = 1 - P(AI-positive)
             • Fully generated and AI-edited share this target
```

The edit-localization branch may receive patch-mask supervision where masks
exist. It is not trained to distinguish fully generated images from edited
images.

### Optional Dual-Stream Spectral Expert
For frequency-domain residual detection, an optional ConvNeXt-Tiny stream processes RGB + fixed high-pass spatial residuals (`conv2d` with discrete derivative kernels) augmented with a 32-bin radial 2D FFT energy projection, dynamically fused via learned sigmoid gates.

---

## 3. Quickstart & Usage

### 3.1 Installation

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

### 3.2 Environment Setup

Copy `.env.example` to `.env` and set machine-local storage paths:
```bash
TECHJAM_DATA_ROOT=E:/techjam26-runtime/data
TECHJAM_HF_HOME=E:/techjam26-runtime/huggingface
TECHJAM_OUTPUT_ROOT=E:/techjam26-runtime/outputs
```

### 3.3 Preparing the Production Eligible Manifests

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

### 3.4 Training

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

### 3.5 Evaluating a Checkpoint


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

### 3.6 Predicting on an Image Directory
Generate Track 5 predictions (`pred = P(AI-generated or AI-edited)`) for submissions:
```bash
uv run python -m aigc_detector.predict \
  --input-dir /path/to/images \
  --output predictions.json \
  --config configs/teacher_dinov3_stage2_paired_unfrozen.yaml \
  --checkpoint /path/to/checkpoint-step-N.pt
```

### 3.7 Repository checks

Project checks are available under `tests/`; training does not invoke them.

---

## 4. Repository Layout

```text
techjam26/
├── configs/                       # Production and experiment configurations
│   ├── bakeoff/                    # Completed backbone tournament configs
│   ├── teacher_dinov3_stage1_clean_frozen.yaml
│   ├── teacher_dinov3_stage2_paired_unfrozen.yaml
│   └── ...
├── docs/                          # Tournament decisions, findings, and design history
│   ├── backbone_bakeoff_findings.md # Complete 3-way empirical report
│   ├── backbone_bakeoff_decision.md # Executive winner decision & catalog
│   ├── teacher_training_plan.md   # Authoritative two-stage teacher plan
│   └── archive/                   # Historical execution and PoC plans
├── outputs/bakeoff/               # Exfiltrated evaluation results and raw JSON metrics
├── scripts/                       # Essential CLI tools (training, eval, prediction)
│   ├── model.py                   # Backbones, binary AI head, edit localization
│   ├── train.py                   # Optimization loop, layerwise decay, EMA, checkpointing
│   ├── predict.py                 # Binary batch inference and submission export
│   ├── losses.py                  # Binary BCE, focal BCE, Dice, consistency losses
│   ├── transforms.py              # Perturbation pipeline (JPEG, blur, noise, resize, crop)
│   ├── sampling.py                # Generator-balanced stratified sampling
│   ├── preprocessing.py           # Standardized geometry and image normalization
│   └── metrics.py                 # AUROC, AUPRC, balanced accuracy, confusion matrix
└── tests/                         # Unit and integration test suite
```

---

## 5. Track 5 Submission Interface

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

### Delivered artifacts and evaluation status

Teacher, Atom, Quark, Atom Super, and Quark Super checkpoints have been trained
and delivered. Four float32 ONNX exports and four static-INT8 candidates were
also produced. The browser defaults to the Atom Super float32 export because its
WebGPU and WebAssembly outputs agreed; the INT8 candidates remain experimental
after a provider-level numerical disagreement.

The final MechaDetect models have **not** yet been compared with external
state-of-the-art detectors on the TechJam evaluation set. Do not present the
backbone bake-off, internal parity checks, or browser runtime checks as that
comparison. TechJam-set and SoTA results must be published only after a
controlled evaluation using the same rows, transforms, thresholds, and metrics.

---

## 6. Known Limitations

- This is a probabilistic detector, not a cryptographic provenance proof,
  watermark verifier, copyright decision, or authorship certificate.
- The optional edit-localization output requires localized-edit coverage and
  mask quality. It is not a second image-level provenance classifier.
- Heavy crop, resize, blur, or JPEG processing can remove evidence; unusual
  authentic post-processing can resemble generative artifacts.
- The DINOv3 teacher is for training and distillation, not constrained deployment.
- The static-INT8 exports are experimental; the browser release uses float32.
- Current reports cover training completion, ONNX parity, graph structure, and
  browser-provider consistency. They do not establish TechJam-set accuracy or a
  state-of-the-art ranking.

---

## 7. Team Contributions

Contributor identities and role allocation were not provided in this
repository context. The submitting team must add each member's name and exact
contributions before Track 5 submission; this repository does not invent or
attribute work without confirmation.
