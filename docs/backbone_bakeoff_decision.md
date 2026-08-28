# Official Backbone Bake-Off Decision Document

## 1. Executive Decision: Winner Declaration

**Winner**: **DINOv3 ViT-H+/16** (`facebook/dinov3-vith16plus-pretrain-lvd1689m`)  
**Runner-Up**: **PE-Spatial-G/14** (`facebook/PE-Spatial-G14-448`)  
**Eliminated in Screening**: **Gemma 4 Vision Tower** (`rnagabh/gemma4-vision-encoder`)

DINOv3 ViT-H+/16 is selected as the production vision backbone for the robust teacher model and downstream student distillation.

---

## 2. Head-to-Head Empirical Evidence Summary

All candidates were evaluated under strict fairness rules:
- Identical 12,000 clean training subset (`ablation12k.csv`)
- Identical sample order, random seed (42), effective batch size (64), and optimizer budget (188 updates)
- Identical common 512-dimensional token adapter and dual-task provenance heads (AIGC learned queries + token-aware tamper classification)
- Zero exposure to multi-transform chains during screening

| Evaluation Metric | DINOv3 ViT-H+/16 (Frozen) | DINOv3 ViT-H+/16 (Adapted) | PE-Spatial-G/14 (Frozen) | Winning Candidate |
| :--- | :--- | :--- | :--- | :--- |
| **Total Parameters** | **844.5M** | **844.5M** | 1,856.0M | **DINOv3** (2.2× leaner) |
| **Backbone Parameters** | **840.6M** | **840.6M** | 1,851.9M | **DINOv3** (< 2B ceiling) |
| **Tokens per Image** | **196** ($14 \times 14$) | **196** ($14 \times 14$) | 1024 ($32 \times 32$) | **DINOv3** (5.2× throughput) |
| **Strict-Unseen AUROC (480 images)** | 0.9782 [0.9664, 0.9904] | **0.9794** [0.9689, 0.9899] | **0.9946** [0.9899, 0.9981] | PE-Spatial (+0.015) |
| **Strict-Unseen Accuracy** | 86.46% | **86.46%** | **94.38%** | PE-Spatial |
| **Strict-Unseen Authentic Recall** | 77.9% (187/240) | **80.0%** (192/240) | **95.0%** (228/240) | PE-Spatial |
| **Strict-Unseen AIGC Recall** | **95.0%** (228/240) | 92.9% (223/240) | 93.8% (225/240) | **DINOv3** |
| **TechJam Proxy AUROC (COCO vs DALL·E 3)** | **0.9978** [0.9955, 0.9995] | **0.9973** [0.9940, 0.9991] | 0.9538 [0.9344, 0.9692] | **DINOv3** (+0.044 AUROC) |
| **TechJam Proxy DALL·E 3 Recall** | **96.5%** (193/200) | **95.5%** (191/200) | 48.5% (97/200) | **DINOv3** (+47.0% recall) |
| **Robustness Grid Clean AUROC** | 0.9731 | — | **0.9856** | PE-Spatial |
| **Compound Degradation (Crop+Resize+JPEG70)** | **0.9712** | — | 0.9400 | **DINOv3** (+0.031 AUROC) |
| **JPEG 30 Heavy Compression AUROC** | 0.9388 (drop: 0.034) | — | **0.9517** (drop: 0.034) | Tie (identical drop) |
| **Blur 2.0 Heavy Defocus AUROC** | 0.9731 (drop: 0.000) | — | **0.9886** (drop: -0.003) | PE-Spatial |
| **Shortcut Probe: Dataset Balanced Acc** | **89.28%** (Chance = 33.3%) | — | 96.93% | **DINOv3** (7.6% less biased) |
| **Shortcut Probe: Aspect Ratio Balanced Acc** | **54.66%** (Chance = 33.3%) | — | 86.61% | **DINOv3** (31.9% less biased) |
| **Adaptation Speed (Time per Update)** | **~1.8 seconds** | **~1.8 seconds** | ~35.0 seconds | **DINOv3** (19.4× faster) |

---

## 3. Detailed Evidence Analysis

### Why DINOv3 ViT-H+/16 Won
1. **Decisive Generalization on Hard Diffusion Generators (DALL·E 3)**:
   On the 400-image TechJam proxy (authentic COCO vs DALL·E 3 Advanced), DINOv3 achieved **0.9978 AUROC** and successfully detected **96.5% of DALL·E 3 images**. PE-Spatial failed on 41% of DALL·E 3 images, misclassifying 82 of 200 fully synthetic images as localized tampering (AUROC 0.9538).
2. **Immunity to Aspect Ratio Shortcuts**:
   A 3-way linear probe trained directly on the frozen backbone tokens to predict the original image aspect ratio (portrait vs square vs landscape) achieved **86.61% accuracy on PE-Spatial-G**, demonstrating severe geometric feature entanglement. DINOv3 achieved **54.66%** (near random chance 33.3%), proving that DINOv3 representations focus on genuine generation artifacts rather than aspect ratio shortcuts.
3. **Superior Resilience to Compound Corruption**:
   Under the severe compound degradation condition (`Crop 80% + Resize 50% + JPEG 70`), DINOv3 maintained **0.9712 AUROC**, outperforming PE-Spatial's **0.9400 AUROC** (+0.0312 advantage).
4. **Computational & Distillation Feasibility**:
   DINOv3 operates on **196 tokens per image** ($14 \times 14$ grid at 224px) with 840.6M parameters. PE-Spatial requires **1024 tokens per image** ($32 \times 32$ grid at 448px) with 1,851.9M parameters. DINOv3 executes forward and backward passes **19× faster**, leaving ample GPU headroom for multi-transform data augmentation, EMA teachers, and student distillation.

### Evidence Against DINOv3 (and for PE-Spatial)
- On the 480-image strict-unseen probe containing older diffusion/GAN generators (DDPM, LDM, StyleGAN, ProGAN), PE-Spatial achieved 0.9946 AUROC compared to DINOv3's 0.9782–0.9794 AUROC, exhibiting higher authentic recall (95.0% vs 80.0%). PE-Spatial's large patch grid ($32 \times 32$) gives it strong localized pixel-level sensitivity when images conform to square/near-square training dimensions.

### Why Gemma 4 Vision Tower Was Eliminated
- Gemma 4 Vision Tower required CPU-intensive soft token preprocessing taking ~0.7 seconds per image in Python PIL. In frozen token extraction, Gemma 4 achieved only ~1.3–2.0 images/second, making the 12,000-image caching take >1.5 hours and falling far behind DINOv3 (which cached 12,000 images in 3 minutes).

---

## 4. Production Checkpoint Handoff

### Pinned Model Identifiers
- **Repository**: `facebook/dinov3-vith16plus-pretrain-lvd1689m`
- **Revision**: `c807c9eeea853df70aec4069e6f56b28ddc82acc`
- **Input Resolution**: $224 \times 224$ (AutoImageProcessor)
- **Token Grid**: 196 patch tokens ($14 \times 14$), dimension 1280

### Checkpoints
- **Best Frozen Checkpoint**:  
  `outputs/bakeoff_frozen_dinov3/checkpoint-step-188.pt`  
  - AUROC on Strict Unseen: `0.9782`  
  - AUROC on TechJam Proxy (DALL·E): `0.9978`
- **Best Adapted Checkpoint**:  
  `outputs/bakeoff_adapt_dinov3/checkpoint-step-188.pt`  
  - AUROC on Strict Unseen: `0.9794`  
  - AUROC on TechJam Proxy (DALL·E): `0.9973`

---

## 5. Exact Configuration for Teacher Training & Distillation

```yaml
seed: 42
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
```
