# TechJam 2026: Robust Image Provenance Detection

Production-ready architecture for robust three-class image provenance detection:
1. **`authentic`**: Pristine real-world scene capture. Content-preserving transformations do not alter this class.
2. **`tampered`**: Authentic scene containing localized semantic additions, removals, splices, or generative inpainting.
3. **`fully_aigc`**: Visual content synthesized end-to-end by a generative model (GAN or diffusion).

Transformations (JPEG, blur, resize, noise, color, crop) are used as robustness augmentations and evaluation stress-tests; the model directly predicts semantic provenance rather than inferring edit history.

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
| **Compound Robustness (Crop+Resize+JPEG)** | **0.9712 AUROC** | 0.9400 AUROC | 0.8140 AUROC |
| **Aspect Ratio Shortcut Accuracy (Chance=33%)**| **54.66%** (Near-chance / invariant) | **86.61%** (Severe shortcut leak) | **80.53%** (Severe shortcut leak) |
| **Adaptation Speed (Time per Update)** | **~1.8 seconds** | ~35.0 seconds (19.4× slower) | ~14.0 seconds |

Full empirical findings, 16-condition robustness tables, and failure analyses:
- Comprehensive findings: [`docs/backbone_bakeoff_findings.md`](docs/backbone_bakeoff_findings.md)
- Executive decision & checkpoint handoff: [`docs/backbone_bakeoff_decision.md`](docs/backbone_bakeoff_decision.md)

---

## 2. Model Architecture

The architecture decouples global generative artifact detection from localized patch tampering:

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
[ AIGC Query Pool Head ]                    [ Token-Aware Tamper Head ]
• 4 learned query vectors                   • Token-level linear classifier
• Multi-head cross-attention over tokens    • Top-5% patch pooling
• Mean + Std summary pooling                • Softmax attention pooling
• MLP projection → aigc_logit               • MLP projection → tamper_logit
       │                                               │
       └───────────────────────┬───────────────────────┘
                               ▼
            [ Hierarchical Probabilities Engine ]
            • P(fully_aigc) = sigmoid(aigc_logit)
            • P(tampered)   = (1 - P(fully_aigc)) * sigmoid(tamper_logit)
            • P(authentic)  = (1 - P(fully_aigc)) * (1 - sigmoid(tamper_logit))
            • Probabilities sum to 1.0; fully-AIGC samples do not incur tamper loss
```

### Optional Dual-Stream Spectral Expert
For frequency-domain residual detection, an optional ConvNeXt-Tiny stream processes RGB + fixed high-pass spatial residuals (`conv2d` with discrete derivative kernels) augmented with a 32-bin radial 2D FFT energy projection, dynamically fused via learned sigmoid gates.

---

## 3. Quickstart & Usage

### 3.1 Installation
```bash
git clone https://github.com/dalzyu/techjam26.git
cd techjam26
pip install -e .
```

Requires PyTorch $\ge 2.0$, torchvision, and `transformers>=5.10.1`.

### 3.2 Environment Setup
Copy `.env.example` to `.env` and set machine-local storage paths:
```bash
TECHJAM_DATA_ROOT=E:/techjam26-runtime/data
TECHJAM_HF_HOME=E:/techjam26-runtime/huggingface
TECHJAM_OUTPUT_ROOT=E:/techjam26-runtime/outputs
```

### 3.3 Training the Winning DINOv3 Model
To train the adapted DINOv3 model (final 25% transformer blocks unfrozen with layerwise learning rate decay 0.85):
```bash
python -m aigc_detector.train --config configs/bakeoff_adapt_dinov3.yaml
```

To run full teacher training with synthetic multi-transform augmentation and EMA guidance:
```bash
python -m aigc_detector.train --config configs/teacher_dinov3_production.yaml
```

### 3.4 Evaluating a Checkpoint
Run evaluation on clean probe images or across the 16-condition single-transform robustness grid:
```bash
# Clean evaluation on the strict-unseen benchmark (480 images)
python scripts/evaluate_performance.py \
  --manifest manifests/strict_unseen_probe.csv \
  --checkpoint outputs/bakeoff/adapt_dinov3/checkpoint-step-188.pt \
  --config configs/bakeoff_adapt_dinov3.yaml \
  --output outputs/eval_results.json \
  --batch-size 8

# 16-condition single-transform robustness evaluation
python scripts/evaluate_performance.py \
  --manifest manifests/strict_unseen_probe.csv \
  --checkpoint outputs/bakeoff/adapt_dinov3/checkpoint-step-188.pt \
  --config configs/bakeoff_adapt_dinov3.yaml \
  --output outputs/eval_robustness.json \
  --limit 120 \
  --robustness \
  --batch-size 8
```

### 3.5 Predicting on an Image Directory
Generate predictions (`pred = P(fully_aigc)`) for test submissions:
```bash
python -m aigc_detector.predict \
  --input-dir /path/to/images \
  --output predictions.json \
  --config configs/bakeoff_adapt_dinov3.yaml \
  --checkpoint outputs/bakeoff/adapt_dinov3/checkpoint-step-188.pt
```

### 3.6 Running Tests
```bash
python -m pytest tests/ -q
```

---

## 4. Repository Layout

```text
techjam26/
├── configs/                       # Production and experiment configurations
│   ├── bakeoff_adapt_dinov3.yaml  # Winning DINOv3 adaptation config (8 layers unfrozen)
│   ├── teacher_dinov3_production.yaml # Production robust teacher training config
│   └── ...
├── docs/                          # Tournament decisions, findings, and design history
│   ├── backbone_bakeoff_findings.md # Complete 3-way empirical report
│   ├── backbone_bakeoff_decision.md # Executive winner decision & catalog
│   └── archive/                   # Historical execution and PoC plans
├── outputs/bakeoff/               # Exfiltrated evaluation results and raw JSON metrics
├── scripts/                       # Essential CLI tools (training, eval, prediction)
│   ├── evaluate_performance.py    # Standardized provenance & robustness evaluator
│   ├── cache_backbone_features.py # Fast multi-GPU token extraction
│   └── probe_feature_shortcuts.py # Linear shortcut audit probe (aspect ratio & dataset)
├── src/aigc_detector/             # Modular core library
│   ├── model.py                   # Backbones, token adapter, dual-task provenance heads
│   ├── train.py                   # Optimization loop, layerwise decay, EMA, checkpointing
│   ├── predict.py                 # Batch inference and submission export
│   ├── losses.py                  # Hierarchical cross-entropy, focal BCE, Dice loss
│   ├── transforms.py              # Perturbation pipeline (JPEG, blur, noise, resize, crop)
│   ├── sampling.py                # Generator-balanced stratified sampling
│   ├── preprocessing.py           # Standardized geometry and image normalization
│   └── metrics.py                 # AUROC, AUPRC, balanced accuracy, confusion matrix
└── tests/                         # Unit and integration test suite
```
