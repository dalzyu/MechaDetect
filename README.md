# TechJam 2026: Robust Image Provenance Detection

Research and training code for binary Track 5 AI-provenance detection:
1. **`authentic`**: Human-created imagery; content-preserving transformations do not alter this class.
2. **AI-positive**: Both AI-edited/tampered imagery and fully generated imagery.

The internal dataset metadata preserves `tampered` and `fully_aigc` subtype
labels for provenance reporting and optional edit-mask localization, but the
image-level model objective treats both as the same positive class.

---

## 1. Backbone Bake-Off Winner: DINOv3 ViT-H+/16

Following a 4-hour controlled tournament across 4× NVIDIA RTX 4090s on 12,000 generator-balanced images, **DINOv3 ViT-H+/16** (`facebook/dinov3-vith16plus-pretrain-lvd1689m`) was selected as the winning vision backbone.

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

### 3.3 Preparing the Full Training Split

Build the leakage-controlled production manifests from the SID, WildFake, and
DiffusionForensics metadata files:

```bash
uv run python scripts/data_prep/build_performance_manifests.py \
  /path/to/sid_metadata.csv \
  /path/to/wildfake_metadata.csv \
  /path/to/diffusionforensics_metadata.csv \
  --output-dir splits/performance \
  --compute-hashes
```

The builder creates `train.csv`, validation/test splits, and
`manifest-report.json`. It rejects duplicate groups crossing splits and rejects
the organizer's prohibited COCO val2017 and DALL-E Advanced demonstration
samples from training. The trainer checks the prohibition again at launch.

### 3.4 Training the DINOv3 Teacher on Six GPUs

Training uses one process per GPU through PyTorch DistributedDataParallel.
Run these commands on Linux with six CUDA devices visible.

Stage 1 freezes DINOv3 and trains the task-specific layers on clean originals.
Its effective batch is `8 records/GPU × 1 accumulation × 6 GPUs = 48`:

```bash
uv run torchrun --standalone --nproc-per-node=6 \
  -m aigc_detector.train \
  --config configs/teacher_dinov3_stage1_clean_frozen.yaml
```

Choose the Stage 1 checkpoint using clean balanced accuracy and macro-F1.
Stage 2 starts a new AdamW optimizer from those task-specific weights,
unfreezes DINOv3, enables the EMA teacher, and trains original/transformed
pairs. Its effective batch is `1 record/GPU × 8 accumulation × 6 GPUs = 48`:

```bash
uv run torchrun --standalone --nproc-per-node=6 \
  -m aigc_detector.train \
  --config configs/teacher_dinov3_stage2_paired_unfrozen.yaml \
  --initial-checkpoint /absolute/path/to/checkpoint-step-N.pt
```

Resume an interrupted stage—including optimizer, scheduler, EMA, sampler
position, and manifest identity—with:

```bash
uv run torchrun --standalone --nproc-per-node=6 \
  -m aigc_detector.train \
  --config configs/teacher_dinov3_stage2_paired_unfrozen.yaml \
  --resume /absolute/path/to/checkpoint-step-N.pt
```

Rank 0 alone writes logs, clean validation metrics, resolved config, and atomic
checkpoints. For a one-GPU smoke run, replace `torchrun ...` with `python` and
add `--max-steps 2 --stage teacher-smoke`.

The complete operational plan—including hardware checks, failure rules,
checkpoint selection, error analysis, Track 5 deliverables, and the future
INT8 PTQ student—is in
[`docs/teacher_training_plan.md`](docs/teacher_training_plan.md).

### 3.5 Evaluating a Checkpoint

Run clean evaluation and the complete single-transform robustness grid on a
candidate production checkpoint:

```bash
uv run python scripts/evaluate_performance.py \
  --manifest splits/performance/test_unseen.csv \
  --checkpoint /path/to/checkpoint-step-N.pt \
  --config configs/teacher_dinov3_stage2_paired_unfrozen.yaml \
  --output outputs/teacher-clean-unseen.json \
  --batch-size 1

uv run python scripts/evaluate_performance.py \
  --manifest splits/performance/test_unseen.csv \
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

### 3.7 Running Tests
```bash
uv run python -m pytest tests/ -q
```

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

The internal model distinguishes fully generated images from locally
AI-edited images for diagnostics and localization. Track 5 treats both as the
same positive class, so the exported score is their probability sum.

### Artifacts still produced after training

The final selected checkpoint must be accompanied by:

1. clean, seen-generator, unseen-generator, and transformation-grid results;
2. false-positive and false-negative analysis with threshold tradeoffs;
3. the future float student and calibrated INT8 PTQ export comparison; and
4. a demo video showing setup, directory inference, output JSON, robustness
   results, and known limitations.

These results are intentionally not prefilled: they must come from the selected
full-scale checkpoint.

---

## 6. Known Limitations

- This is a probabilistic detector, not a cryptographic provenance proof,
  watermark verifier, copyright decision, or authorship certificate.
- The optional edit-localization output requires localized-edit coverage and
  mask quality. It is not a second image-level provenance classifier.
- Heavy crop, resize, blur, or JPEG processing can remove evidence; unusual
  authentic post-processing can resemble generative artifacts.
- The full DINOv3 teacher is intended for training and distillation, not
  resource-constrained deployment. Deployment uses the separately validated
  INT8 student.
- The production teacher metrics, error analysis, and INT8 student comparison
  remain pending until their corresponding runs are complete.

---

## 7. Team Contributions

Contributor identities and role allocation were not provided in this
repository context. The submitting team must add each member's name and exact
contributions before Track 5 submission; this repository does not invent or
attribute work without confirmation.
