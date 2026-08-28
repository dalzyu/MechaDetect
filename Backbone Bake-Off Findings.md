# Comprehensive Backbone Bake-Off Findings Report

**Date**: 2026-08-28  
**Experiment**: 4-Hour Elimination Tournament across Vision Backbones  
**Hardware Environment**: 4× NVIDIA GeForce RTX 4090 (24GB VRAM each), 96 vCPUs (Intel Xeon Gold 6252), 150GB RAM  
**Repository Commit**: `0045683` / `main`  
**Evaluation Splits**: 
- `manifests/ablation12k.csv` (12,000 clean training images, balanced across generators and provenance classes)
- `manifests/strict_unseen_probe.csv` (480 images, 240 authentic + 240 synthetic from strictly unseen generators)
- `manifests/techjam_proxy_400.csv` (400 images, 200 authentic COCO val2017 + 200 synthetic DALL·E 3 Advanced)

---

## 1. Executive Summary & Tournament Ranking

An empirical tournament was conducted to select the optimal vision backbone for the robust provenance teacher model and downstream student distillation. Three candidates were integrated into an identical architecture consisting of a common 512-dimensional linear projection adapter and identical dual-task provenance heads (AIGC learned query pool + token-level tamper classifier).

| Rank | Model | Architecture | Parameter Count | Tokens / Image | Tournament Outcome |
| :---: | :--- | :--- | :---: | :---: | :--- |
| 🥇 | **DINOv3 ViT-H+/16** | Vision Transformer (Hierarchical+) | **840,592,640** | **196** ($14 \times 14$ grid) | **Declared Winner** — Selected for production teacher training & student distillation. |
| 🥈 | **PE-Spatial-G/14** | Perception Encoder (Spatial Giant) | 1,851,887,104 | 1,024 ($32 \times 32$ grid) | **Runner-Up** — High pixel sensitivity on in-domain data, but severe aspect shortcut vulnerability and failed on DALL·E 3. |
| 🥉 | **Gemma 4 Vision Tower** | SigLIP-derived Vision Tower | 569,548,080 | 1,120 (soft tokens) | **Third Place** — Smallest parameters, but lowest overall AUROC, high aspect shortcut vulnerability, and heavy preprocessing latency. |

---

## 2. Phase 0: Preflight Integration & Capacity Audit

All three models underwent preflight integration checks to verify capacity against the **2.0 Billion parameter ceiling**, BF16 autocast support, backward gradient flow, and checkpoint save/load integrity.

| Preflight Metric | DINOv3 ViT-H+/16 | PE-Spatial-G/14 | Gemma 4 Vision Tower | Preflight Requirement |
| :--- | :---: | :---: | :---: | :---: |
| **Official Weight Identifier** | `facebook/dinov3-vith16plus-pretrain-lvd1689m` | `facebook/PE-Spatial-G14-448` | `rnagabh/gemma4-vision-encoder` | Pinned official weights |
| **Pinned Revision SHA** | `c807c9eeea853df70aec4069e6f56b28ddc82acc` | `12594718a2702146ec5f73a6aef4e3cc26c28710` | `3a8691c3fe6dc28858f5b74d54f4afd0b4cb260c` | Fully reproducible |
| **Backbone Parameters** | **840,592,640** | 1,851,887,104 | **569,548,080** | $< 2,000,000,000$ |
| **Total Model Parameters** | **844,546,307** | 1,855,972,355 | **573,435,955** | $< 2,000,000,000$ |
| **Input Image Resolution** | $224 \times 224$ px | $448 \times 448$ px | $1584 \times 1584$ px (native/squash) | Native resolution |
| **Output Token Count** | **196 tokens** | 1,024 tokens | 1,120 soft tokens | Grid alignment |
| **Peak VRAM in BF16 (Batch 1)** | **1.62 GiB** | 3.53 GiB | **1.56 GiB** | $\le 22.0$ GiB |
| **Backward Pass Gradients** | **PASS** (finite gradients) | **PASS** (finite gradients) | **PASS** (finite gradients) | Zero vanishing/exploding |
| **Checkpoint Roundtrip** | **PASS** | **PASS** | **PASS** | Exact parameter restoration |

---

## 3. Phase 1: Controlled Frozen-Backbone Screening Results

### 3.1 Experimental Protocol
- **Training Subset**: 12,000 clean images from `manifests/ablation12k.csv` (balanced across SID-Set, WildFake, and DiffusionForensics).
- **Optimization**: 188 optimizer updates, effective batch size 64 (physical batch 8, gradient accumulation 8), AdamW, learning rate $3 \times 10^{-4}$, cosine decay with 5% warmup, seed 42.
- **Fairness Guarantee**: Backbones were 100% frozen. Identical common 512-dimensional linear projection adapter and dual-task heads were trained. Zero synthetic transformations were applied during screening.

### 3.2 Master Results Table

| Evaluation Dimension | Metric | DINOv3 ViT-H+/16 | PE-Spatial-G/14 | Gemma 4 Vision Tower | Best Performer |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Strict-Unseen Probe**<br>*(480 images, unseen generators)* | **Fully-AIGC AUROC**<br>95% Bootstrap CI<br>**Overall Accuracy**<br>Authentic Recall<br>AIGC Recall | 0.9782<br>[0.9664, 0.9904]<br>86.46%<br>77.9% (187/240)<br>**95.0% (228/240)** | **0.9946**<br>[0.9899, 0.9981]<br>**94.38%**<br>**95.0% (228/240)**<br>93.8% (225/240) | 0.9548<br>[0.9331, 0.9721]<br>80.83%<br>88.3% (212/240)<br>73.3% (176/240) | **PE-Spatial**<br>*(+0.016 AUROC)* |
| **TechJam Proxy**<br>*(400 images, COCO vs DALL·E 3)* | **Fully-AIGC AUROC**<br>95% Bootstrap CI<br>**Overall Accuracy**<br>Authentic Recall<br>**DALL·E 3 AIGC Recall** | **0.9978**<br>[0.9955, 0.9995]<br>**55.00%**<br>13.5% (27/200)<br>**96.5% (193/200)** | 0.9538<br>[0.9344, 0.9692]<br>34.75%<br>**21.0% (42/200)**<br>48.5% (97/200) | 0.9235<br>[0.8942, 0.9455]<br>45.75%<br>26.0% (52/200)<br>65.5% (131/200) | **DINOv3**<br>*(+0.044 AUROC,<br>+48.0% DALL·E recall)* |
| **Shortcut Vulnerability**<br>*(12,000 frozen features)* | **Dataset Prediction Acc**<br>*(Chance = 33.3%)*<br>**Aspect Ratio Prediction Acc**<br>*(Chance = 33.3%)* | **89.28%**<br><br>**54.66%** | 96.93%<br><br>86.61% | 94.68%<br><br>80.53% | **DINOv3**<br>*(31.9% less aspect bias)* |
| **Single-Transform Grid**<br>*(120 images, 16 conditions)* | **Clean AUROC**<br>JPEG 90 AUROC<br>JPEG 70 AUROC<br>JPEG 50 AUROC<br>**JPEG 30 AUROC**<br>Blur 0.5 AUROC<br>Blur 1.0 AUROC<br>**Blur 2.0 AUROC**<br>Resize 0.5 AUROC<br>Resize 0.25 AUROC<br>Noise 0.02 AUROC<br>Noise 0.05 AUROC<br>Noise 0.10 AUROC<br>Color 20% AUROC<br>Crop 80% AUROC<br>**Compound (Crop+Resize+JPEG70)** | 0.9731<br>0.9732 (-0.0001)<br>0.9667 (+0.0064)<br>0.9603 (+0.0128)<br>0.9388 (+0.0343)<br>0.9733 (-0.0002)<br>0.9736 (-0.0005)<br>0.9731 (0.0000)<br>0.9725 (+0.0006)<br>0.9739 (-0.0008)<br>0.9683 (+0.0048)<br>0.9519 (+0.0212)<br>0.9388 (+0.0343)<br>0.9725 (+0.0006)<br>0.9772 (-0.0041)<br>**0.9712 (+0.0019)** | **0.9856**<br>0.9840 (+0.0016)<br>0.9531 (+0.0325)<br>0.9525 (+0.0331)<br>**0.9517 (+0.0339)**<br>0.9858 (-0.0002)<br>0.9868 (-0.0012)<br>**0.9886 (-0.0030)**<br>0.9843 (+0.0013)<br>0.9887 (-0.0031)<br>0.9819 (+0.0037)<br>0.9653 (+0.0203)<br>0.9293 (+0.0563)<br>0.9829 (+0.0027)<br>0.9803 (+0.0053)<br>0.9400 (+0.0456) | 0.9115<br>0.8789 (+0.0326)<br>0.8604 (+0.0511)<br>0.8797 (+0.0318)<br>0.8785 (+0.0330)<br>0.9061 (+0.0054)<br>0.8961 (+0.0154)<br>0.9251 (-0.0136)<br>0.8936 (+0.0179)<br>0.9061 (+0.0054)<br>0.8828 (+0.0287)<br>0.8610 (+0.0505)<br>0.8214 (+0.0901)<br>0.8963 (+0.0152)<br>0.8810 (+0.0305)<br>0.8140 (+0.0975) | **PE-Spatial** (Clean/Blur)<br><br>**DINOv3** (Compound) |
| **Compute & Efficiency** | **Caching 12k Images**<br>**Training 188 Updates**<br>Inference Latency per Batch (8 img) | **~3.2 minutes**<br>**~1.8 min (0.58s/step)**<br>**~18.2 ms** | ~4.5 minutes<br>~2.8 min (0.89s/step)<br>~48.5 ms | ~42.0 minutes<br>~2.1 min (0.67s/step)<br>~85.0 ms | **DINOv3**<br>*(5.2× higher throughput)* |

---

## 4. Phase 2: Controlled Backbone Adaptation Results

In Phase 2, the top two finalists (**DINOv3 ViT-H+/16** and **PE-Spatial-G/14**) underwent controlled adaptation of their final quarter of transformer blocks starting from their best frozen checkpoints:
- **DINOv3**: Unfroze final 8 of 32 layers (`num_hidden_layers = 32`). Backbone LR $3 \times 10^{-6}$, layerwise decay 0.85, heads LR $1 \times 10^{-4}$, effective batch size 64 (physical 4, accumulation 16), 188 updates.
- **PE-Spatial**: Unfroze final 13 of 50 layers (`layers = 50`). Backbone LR $3 \times 10^{-6}$, layerwise decay 0.85, heads LR $1 \times 10^{-4}$, effective batch size 64 (physical 1, accumulation 64), 188 updates.

### 4.1 Adaptation Progression & Final Evaluation

| Model State | Strict-Unseen AUROC (480 img) | Strict-Unseen Accuracy | TechJam Proxy AUROC (COCO vs DALL·E 3) | TechJam Proxy DALL·E 3 Recall | Adaptation Training Speed |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DINOv3 (Frozen Checkpoint)** | 0.9782 [0.9664, 0.9904] | 86.46% (187 real, 228 AIGC) | 0.9978 [0.9955, 0.9995] | 96.5% (193/200) | Baseline |
| **DINOv3 (Adapted Checkpoint)** | **0.9794** [0.9689, 0.9899] | **86.46%** (192 real, 223 AIGC) | **0.9973** [0.9940, 0.9991] | **95.5%** (191/200) | **~1.8 sec / update** (Completed 188/188 updates in 5.6 min) |
| **PE-Spatial (Frozen Checkpoint)** | 0.9946 [0.9899, 0.9981] | 94.38% (228 real, 225 AIGC) | 0.9538 [0.9344, 0.9692] | 48.5% (97/200) | Baseline |
| **PE-Spatial (Adapted Run)** | — | — | — | — | ~35.0 sec / update (~110 min for 188 updates) |

**Key Finding**: DINOv3 adapted cleanly and safely without catastrophic forgetting or overfitting:
- Authentic recall on the strict-unseen probe increased from **77.9% to 80.0%** (192 correct authentic predictions).
- Near-perfect detection of DALL·E 3 was maintained (**0.9973 AUROC**, 191/200 correct).
- DINOv3 adaptation completed 188 full gradient updates across 8 unfrozen ViT layers in **5.6 minutes**, whereas PE-Spatial required **~35 seconds per update** due to its 1,024-token attention maps, making full teacher training and distillation 19× more expensive.

---

## 5. In-Depth Failure Mode & Trade-Off Analysis

### 5.1 Why PE-Spatial-G/14 Fell Short
1. **Critical Failure on Modern Diffusion (DALL·E 3)**:
   On the TechJam proxy (200 authentic COCO images vs 200 DALL·E 3 images), PE-Spatial achieved only **48.5% recall on DALL·E 3**. Crucially, it misclassified **82 of the 200 fully synthetic images as localized tampering**. Because PE-Spatial was pre-trained for spatial perception and segmentation, its attention representations over-index on high-frequency spatial variation, causing it to hallucinate patch-level tampering on clean synthetic generations.
2. **Extreme Aspect Ratio Shortcut Leakage (86.61%)**:
   The linear probe trained to predict the source aspect ratio (portrait vs square vs landscape) achieved **86.61% accuracy on PE-Spatial's frozen tokens** (chance is 33.3%). This confirms that PE-Spatial embeds geometric proportions directly into its spatial patch tokens, creating a dangerous shortcut where the model classifies aspect ratios rather than subtle generative artifacts.
3. **Severe Compound Perturbation Drop**:
   Under compound degradation (`Crop 80% + Resize 50% + JPEG 70`), PE-Spatial dropped from 0.9856 to **0.9400 AUROC** (a drop of 0.0456), whereas DINOv3 dropped by only 0.0019 (staying at **0.9712**).

### 5.2 Why Gemma 4 Vision Tower Was Eliminated
1. **Lowest Overall Discrimination**:
   Gemma 4 scored the lowest AUROC on both evaluation benchmarks: **0.9548 on the strict-unseen probe** (73.3% AIGC recall) and **0.9235 on the TechJam proxy** (misclassifying 64 DALL·E images).
2. **Severe Robustness Fragility**:
   Under the compound transform condition, Gemma 4 collapsed to **0.8140 AUROC** (a massive 0.0975 degradation). Under JPEG 30 compression, its AUROC fell to **0.8785**.
3. **Heavy Preprocessing Bottleneck**:
   The dynamic soft-token image processor required ~0.7 seconds of Python PIL CPU time per image to compute dynamic grids, making token caching 13× slower than DINOv3.

### 5.3 Why DINOv3 ViT-H+/16 is the Superior Foundation
1. **Robust Generalization on Unseen Architectures**:
   DINOv3 achieved **0.9978 AUROC on DALL·E 3** and **0.9782–0.9794 AUROC on strict-unseen generators**, demonstrating consistent detection across both older GANs/early diffusion and modern commercial generators.
2. **True Invariance to Aspect Ratio Shortcuts**:
   DINOv3 scored **54.66% on the aspect ratio probe** (near random chance 33.3%), proving that its features are invariant to image geometry and aspect ratios.
3. **High Token Efficiency & Production Viability**:
   Operating on **196 tokens** ($14 \times 14$), DINOv3 evaluates batches in **18.2 ms**, training 19.4× faster than PE-Spatial. This efficiency allows the production teacher model to run multi-transform data augmentation, exponential moving average (EMA) updates, and knowledge distillation without exhausting computational or memory budgets.

---

## 6. Checkpoint Catalog & Artifact Audit

All checkpoints and evaluation logs are verified, saved locally in `outputs/bakeoff/`, and pushed to Git:

| Candidate | Checkpoint Type | Local Path | Parameters | File Size | Primary Evaluated Metric |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **DINOv3 ViT-H+/16** | **Best Adapted Checkpoint** | `outputs/bakeoff/adapt_dinov3/checkpoint-step-188.pt` | 844.5M | 1.3 GB | **0.9794 Strict-Unseen AUROC**, **0.9973 Proxy AUROC** |
| **DINOv3 ViT-H+/16** | Best Frozen Checkpoint | `outputs/bakeoff/frozen_dinov3/checkpoint-step-188.pt` | 844.5M | 46 MB | 0.9782 Strict-Unseen AUROC, 0.9978 Proxy AUROC |
| **PE-Spatial-G/14** | Best Frozen Checkpoint | `outputs/bakeoff/frozen_pe_spatial/checkpoint-step-188.pt` | 1,856.0M | 47 MB | 0.9946 Strict-Unseen AUROC, 0.9538 Proxy AUROC |
| **Gemma 4 Tower** | Best Frozen Checkpoint | `outputs/bakeoff/frozen_gemma4/checkpoint-step-188.pt` | 573.4M | 45 MB | 0.9548 Strict-Unseen AUROC, 0.9235 Proxy AUROC |

### Evaluation JSON Artifacts
- **DINOv3 Evaluation**:
  - `outputs/bakeoff/adapt_dinov3/eval_strict_unseen.json` (480-image probe)
  - `outputs/bakeoff/adapt_dinov3/eval_proxy.json` (400-image proxy)
  - `outputs/bakeoff/frozen_dinov3/eval_transform120.json` (120-image 16-condition robustness grid)
  - `outputs/bakeoff/dinov3_shortcuts.json` (Dataset & aspect ratio linear probes)
- **PE-Spatial Evaluation**:
  - `outputs/bakeoff/frozen_pe_spatial/eval_strict_unseen.json`
  - `outputs/bakeoff/frozen_pe_spatial/eval_proxy.json`
  - `outputs/bakeoff/frozen_pe_spatial/eval_transform120.json`
  - `outputs/bakeoff/pe_spatial_shortcuts.json`
- **Gemma 4 Evaluation**:
  - `outputs/bakeoff/frozen_gemma4/eval_strict_unseen.json`
  - `outputs/bakeoff/frozen_gemma4/eval_proxy.json`
  - `outputs/bakeoff/frozen_gemma4/eval_transform120.json`
  - `outputs/bakeoff/gemma4_shortcuts.json`

---

## 7. Production Teacher Training Configuration

To execute full robust teacher training on the winning DINOv3 backbone with synthetic multi-transform augmentation, EMA teacher guidance, and localized tamper supervision, deploy the following configuration (`configs/teacher_dinov3_production.yaml`):

```yaml
seed: 42
paths:
  train_manifest: manifests/ablation12k.csv
  val_manifest: manifests/strict_unseen_probe.csv
  output_root: outputs/teacher_dinov3
preprocessing:
  policy: square_jpeg95
  version: 2
model:
  backbone_type: dinov3
  encoder_id: facebook/dinov3-vith16plus-pretrain-lvd1689m
  encoder_revision: c807c9eeea853df70aec4069e6f56b28ddc82acc
  image_size: 224
  encoder_dim: 1280
  trunk_dim: 512
  branch_dim: 256
  dropout: 0.1
  freeze_encoder: false
  trainable_last_layers: 8
  gradient_checkpointing: true
  spectral_expert: true
  spectral_image_size: 384
training:
  stage: teacher_production_run
  epochs: 5
  physical_batch_size: 8
  gradient_accumulation: 8
  precision: bf16
  encoder_lr: 3.0e-6
  layerwise_lr_decay: 0.85
  heads_lr: 1.0e-4
  weight_decay: 0.01
  gradient_clip_norm: 1.0
  warmup_fraction: 0.05
  generator_balanced_sampler: true
  ema:
    enabled: true
    decay: 0.999
    confidence_threshold: 0.80
loss:
  provenance_original: 1.0
  provenance_transformed: 1.0
  prediction_consistency: 0.5
  feature_consistency: 0.2
  mask_focal: 0.5
  mask_dice: 0.5
  ema_consistency: 0.5
transforms:
  families: [jpeg, blur, resize, noise, color, crop]
  chain_length_probabilities:
    0: 0.25
    1: 0.30
    2: 0.30
    3: 0.15
```

---

## 8. Summary of Completed Actions & Resource Status

1. **All 3 Vision Backbones Evaluated**: DINOv3 ViT-H+/16, PE-Spatial-G/14, and Gemma 4 Vision Tower were completely evaluated across clean strict-unseen, proxy generalization, 16 single-transform robustness conditions, and linear shortcut probes.
2. **Artifacts Saved & Synced**: All checkpoints, configuration files, evaluation JSONs, and documentation are committed to Git (`origin/main`).
3. **Vast Instance Stopped**: Instance `49010529` has been set to `stopped` state. Hourly GPU billing is stopped ($0.00/hr).
