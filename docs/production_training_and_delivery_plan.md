# Production Training, Distillation, and WebGPU Edge Delivery Plan

**Project:** TechJam 2026 — Track 5: AI-Generated & Tampered Image Detection  
**Core Objective:** High-Accuracy, Adversarially Robust AIGC Provenance Detection on the Edge  
**Primary Edge Deliverable:** Zero-Server-Cost Client-Side WebGPU Detection Engine with Interactive Adversarial Stress-Testing  
**Canonical Dataset:** [`zye2/tj-data`](https://huggingface.co/datasets/zye2/tj-data) (127,902 rows across 38 balanced cohorts)  

---

> **Executed-run correction:** The 127,902-row / 38-cohort figures below are the canonical metadata target, not the materialized payload count achieved in the first run. Teacher Iteration 1 trained on 22,483 rows; the production adaptation used 41,035 executable rows from five source families. The complete run record, weight catalogue, hashes, failures, and promotion status are in [`docs/training_run_consolidated.md`](training_run_consolidated.md).

## 1. Executive Summary & Architectural Overview

The core deliverable of this project is a **tri-tier suite of edge vision models running locally in the browser via WebGPU** through `onnxruntime-web`.

To achieve both **scientific evaluation integrity** and **maximum real-world edge performance**, the project uses:
1. **An All-DINOv3 Model Family:** Every model—from the 840M cluster teacher down to the 21M mobile student—shares Meta's `lvd1689m` pretraining corpus (1.689 billion images) and $16 \times 16$ patch geometry. This enables direct, native feature-cosine distillation without projection loss or positional interpolation artifacts.
2. **A Two-Checkpoint Phasing Strategy:**
   - **Checkpoint 1 (The Audit Checkpoint):** Trained strictly on `train` (77k), validated on `val` (21k), and evaluated on `test` (16k) and `test_unseen` (13k). Generates leak-free AUROC, ablation, and robustness metrics for the competition report.
   - **Checkpoint 2 (The Production Checkpoint):** Initialized from Checkpoint 1 and fine-tuned on 100% of the curated data (all 127,902 rows) for a locked step budget ($0.20 \times N^*$) with a $3\times$ decayed learning rate and locked 50/50 balanced sampling. Deployed as the live weights in the WebGPU client.
3. **Paired Transformation Consistency & EMA Self-Distillation:** During training, every image is accompanied by an on-the-fly perturbed twin drawn from the organizer's exact transformation grid (JPEG, blur, resize, noise, color, crop). Mathematical consistency penalties force the network's representation space to be invariant to compression and blur.

---

## 2. Model Family & Tier Specification

| Model Tier | Checkpoint Identifier | Parameters | FP16 Size | INT8 Size | Target WebGPU Hardware | Target Latency / FPS |
|---|---|---:|---:|---:|---|---:|
| **Teacher (Cluster)** | `facebook/dinov3-vith16plus-pretrain-lvd1689m` | **840.6M** | 1.68 GB | — | Multi-GPU Cluster (CMP 170HX / RTX 4090) | ~90 ms (Offline Anchor) |
| **Tier 1 (Pro Edge)** | `facebook/dinov3-convnext-large-pretrain-lvd1689m` | **198.0M** | 396 MB | **198 MB** | Discrete GPU / Apple M-Series (M1/M2/M3/M4) | ~35 ms (28 FPS) |
| **Tier 2 (Balanced)** | `facebook/dinov3-vitb16-pretrain-lvd1689m` | **86.0M** | 172 MB | **86 MB** | Laptop Integrated GPU (Intel Iris / AMD Radeon) | ~18 ms (55 FPS) |
| **Tier 3 (Ultra Edge)** | `facebook/dinov3-vits16-pretrain-lvd1689m` | **21.0M** | 42 MB | **21 MB** | Mobile WebGPU / Low-Power Chromebooks | **~6 ms (120+ FPS)** |

### Architectural Rationale: Why DINOv3 $\rightarrow$ DINOv3 Distillation Wins
- **Zero Token Dimension Mismatch:** The teacher (`vith16plus`) and the transformer students (`vitb16`, `vits16`) share the exact identical $16 \times 16$ patch tokenizer. At $224 \times 224$ input resolution, both produce exactly $14 \times 14 = 196$ visual tokens.
- **Direct Feature Cosine Distillation:** Student patch embeddings map directly into teacher embeddings with zero positional interpolation.
- **ConvNeXt for WebGPU Graph Simplicity:** Tier 1 uses ConvNeXt-Large rather than a transformer because pure convolutions translate into streamlined WGSL compute shaders with zero multi-head attention matrix multiplication overhead.

---

## 3. The 127,902-Row Canonical Dataset (`zye2/tj-data`)

The dataset is live on Hugging Face at [`zye2/tj-data`](https://huggingface.co/datasets/zye2/tj-data). It consolidates 38 distinct cohorts into 31 standardized columns, strictly filtered against organizer exclusions and near-duplicate leakage.

### Balanced Semantic Breakdown

$$\text{Total Rows: } \mathbf{127{,}902} \quad \Big| \quad \text{Authentic Negatives } (a=0): \mathbf{61{,}600 \text{ (48.2\%)}} \quad \Big| \quad \text{AI Positives } (a=1): \mathbf{66{,}302 \text{ (51.8\%)}}$$

| Strategic Cohort | Sub-Sources Included | Rows | Provenance | Binary Target (`ai_positive`) | Role in Anti-Shortcut Training |
|---|---|---:|---|:---:|---|
| **Authentic Memes & Screenshots** | `gmorinan_memes`, `imgflip_memes`, `multioff_memes` | 4,100 | `tampered` (human) | **0** | Neutralizes the *"Impact font / white text banner / UI = AI"* shortcut. Strictly pre-2019 verified. |
| **AI Memes & Social Overlays** | `ai_meme_macro_overlay`, `yesbut_satire_comics`, `ai_reaction_banners`, `scam_ai_social_posts` | 2,700 | `tampered` / `fully_aigc` | **1** | Counterbalances authentic memes. Prevents the inverse shortcut (*"meme = authentic"*). |
| **SOTA Modern Generators** | `flux_reason_6m`, `ideogram_27k`, `krea2_wildcards`, `nano_banana_pro_gen` | 12,500 | `fully_aigc` | **1** | Exposes the network to flow-matching (FLUX), typography SOTA (Ideogram), and modern diffusion. |
| **Midjourney & SD3 Cohort** | `midjourney_v6_recap`, `midjourney_v5_images`, `sd3_medium_synths` | 4,000 | `fully_aigc` | **1** | Closes the critical consumer gap; covers Midjourney's unique micro-textures and SD3 MMDiT noise. |
| **Ewan Empirical Archive** | `ewan_gpt_images` (`100+ AI_Images from ewan.zip`) | 102 | `fully_aigc` | **1** | Lossless in-the-wild PNG generations from ChatGPT / DALL-E 3 web interfaces. |
| **Instruction-Based AI Edits** | `gpt_image_edit_1_5m`, `google_nano_banana_edited` | 7,500 | `tampered` | **1** | Trains detection of partial inpainting, object insertion, and background swaps. |
| **Stylized 2D Anime & Manga** | `novelai_artist_comparison`, `animagine_xl` vs. `danbooru_pre2020`, `manga109` | 4,000 | `fully_aigc` / `authentic` | **Pairs (0 & 1)** | Prevents camera-sensor reliance; balances human lineart/manga against NovelAI/Animagine diffusion. |
| **3D CGI & Gaming Captures** | `tartanair2_ue5`, `gta5`, `game_screenshots` vs. `flux_cyberpunk`, `sdxl_vehicles` | 3,000 | `authentic` / `fully_aigc` | **Pairs (0 & 1)** | Neutralizes the *"ray-traced lighting / polygon edge = AI"* trap. |
| **Anatomy & Glamour Skin** | Classical figure art, high-fashion portraits vs. Civitai photorealism / Pony XL | 5,000 | `authentic` / `fully_aigc` | **Pairs (0 & 1)** | Prevents bare skin / cosmetic smoothing from triggering false positives. 100% HF ToS compliant. |
| **Core Benchmarks** | `sid`, `wildfake`, `diffusionforensics` | 57,000 | Mixed | **Pairs (0 & 1)** | Baseline historical calibration and in-domain competition distribution anchor. |
| **Authentic Photography & Art** | `artic_dataset`, `open_images_v7`, `art_museums_pd` | 28,000 | `authentic` | **0** | Clean DSLR photography, museum scans, and real-world natural scenes. |

### Exclusion and Leakage Controls
1. **Forbidden Demonstrations:** 100% filtered against COCO val2017 (4,998 images) and WildFake DALL-E Advanced (8,843 images). All 100,008 UltraEdit rows were discarded due to unverified upstream COCO lineage.
2. **Group-Disjoint Splitting:** 16-bit LSH perceptual difference hashing (`dHash`) clusters near-duplicate meme templates, prompt variants, and video trajectories into single disjoint groups. No duplicate cluster crosses between splits.

---

## 4. Checkpoint Strategy & Phasing

```
[ Step 1: Checkpoint 1 (The Audit Model) ]
├── Training Pool: 'train' split (77,106 rows)
├── Validation Pool: 'validation' split (21,001 rows) — monitored every 100 steps
├── Early Stopping: Pin peak AUROC step N* (expected: ~600 steps in Stage 1, ~1,200 in Stage 2)
└── Freeze & Evaluate:
      ├── Evaluate on 'test' (16,773 rows) -> in-domain test score
      ├── Evaluate on 'test_unseen' (13,022 rows) -> strict zero-shot OOD generator generalization
      └── Generate: outputs/audit_evaluation_report.json (unbiased hackathon proof)

[ Step 2: Checkpoint 2 (The Production Full-Data Model) ]
├── Initialization: Load weights from Checkpoint 1 at step N*
├── Training Pool: Pool ALL 127,902 rows (train + validation + test + test_unseen)
├── Sampler: Enforce generator_balanced_weights (50% authentic / 50% AI-positive)
├── Duration: Exactly 0.20 * N* steps (controlled adaptation; prevents catastrophic forgetting)
├── Learning Rate: Decayed 3x (heads: 3.0e-5, backbone: 1.0e-6)
└── Export: Ready for student distillation and WebGPU conversion
```

---

## 5. Mathematical Loss Formulations

The optimization loss unites classification, geometric consistency, teacher distillation, and localized mask supervision:

$$\mathcal{L}_{\text{total}} = w_1 \mathcal{L}_{\text{cls}}(x) + w_2 \mathcal{L}_{\text{cls}}(T(x)) + w_3 \mathcal{L}_{\text{pred}} + w_4 \mathcal{L}_{\text{feat}} + w_5 \mathcal{L}_{\text{EMA}} + w_6 \mathcal{L}_{\text{mask}}$$

### 1. Binary AI Classification Loss ($\mathcal{L}_{\text{cls}}$)
Supervises clean original $x$ and transformed $T(x)$ against the binary target $y \in \{0.0, 1.0\}$:
$$\mathcal{L}_{\text{cls}} = \text{BCEWithLogits}(z_{\text{ai}}, y) = -y \log \sigma(z_{\text{ai}}) - (1 - y) \log (1 - \sigma(z_{\text{ai}}))$$
*Crucial fix:* Grounded directly on `batch["ai_positive"]`, ensuring pre-2019 memes ($y=0.0$) are trained as authentic negatives even though categorized under `tampered` taxonomy.

### 2. Prediction Consistency Loss ($\mathcal{L}_{\text{pred}}$)
Penalizes probability drift when an image undergoes compression or blurring:
$$\mathcal{L}_{\text{pred}} = \text{MSE}\big(P(x), P(T(x))\big) = \big(\sigma(z_{\text{clean}}) - \sigma(z_{\text{trans}})\big)^2$$

### 3. Feature Embedding Consistency Loss ($\mathcal{L}_{\text{feat}}$)
Enforces that backbone representations are geometrically invariant to post-processing:
$$\mathcal{L}_{\text{feat}} = 1 - \frac{f(x) \cdot f(T(x))}{\|f(x)\|_2 \, \|f(T(x))\|_2}$$

### 4. Confidence-Gated Distillation Loss ($\mathcal{L}_{\text{distill}}$ / $\mathcal{L}_{\text{EMA}}$)
Transfers knowledge from teacher to student, gated strictly on teacher confidence ($\ge 0.80$):
$$\mathcal{L}_{\text{distill}} = \mathbb{I}\big(\max P_{\text{teacher}} \ge 0.80\big) \cdot D_{\text{KL}}\big(P_{\text{student}}(T(x)) \;\big\|\; P_{\text{teacher}}(x)\big)$$
*Crucial fix:* Computed using exact `F.logsigmoid` on student logits ($[\text{logsigmoid}(-z), \text{logsigmoid}(z)]$) rather than clamping probabilities, guaranteeing non-zero analytical gradients across extreme tails ($|z| > 15$).

### 5. Localized Tamper Mask Supervision ($\mathcal{L}_{\text{mask}}$)
Applies Focal BCE ($\gamma=2.0$) and soft Dice loss to token-level logits against fractional patch occupancy:
$$\mathcal{L}_{\text{mask}} = \text{FocalBCE}(z_{\text{patch}}, m) + \Big(1 - \frac{2 \sum \sigma(z_{\text{patch}}) m + \epsilon}{\sum \sigma(z_{\text{patch}}) + \sum m + \epsilon}\Big)$$

---

## 6. Two-Stage Teacher Training Execution

### Stage 1: Clean Frozen Alignment Anchor
- **Configuration:** `configs/teacher_dinov3_stage1_clean_frozen.yaml`
- **Backbone:** DINOv3 ViT-H+/16 frozen. Patch tokens pre-extracted and cached.
- **Batch Size:** 8 physical $\times$ 6 GPUs $\times$ 1 accum = **48 effective samples/step**.
- **Budget:** 600 steps ($28{,}800$ samples seen).
- **Command:**
  ```bash
  torchrun --standalone --nproc-per-node=6 \
    -m aigc_detector.train \
    --config configs/teacher_dinov3_stage1_clean_frozen.yaml \
    --stage teacher_stage1_clean_frozen
  ```

### Stage 2: End-to-End Paired Robustness Training
- **Configuration:** `configs/teacher_dinov3_stage2_paired_unfrozen.yaml`
- **Initialization:** `--initial-checkpoint outputs/teacher_stage1/checkpoint-step-600.pt`
- **Backbone:** Unfrozen with Layer-Wise Learning Rate Decay (LLRD = 0.85).
- **Transforms:** Active on-the-fly pairing (JPEG, blur, resize, noise, color, crop).
- **Budget:** 1,250 steps ($\sim 60{,}000$ paired samples seen).
- **Command:**
  ```bash
  torchrun --standalone --nproc-per-node=6 \
    -m aigc_detector.train \
    --config configs/teacher_dinov3_stage2_paired_unfrozen.yaml \
    --initial-checkpoint outputs/teacher_stage1/checkpoint-step-600.pt
  ```

---

## 7. Teacher Iteration 1 Audit and Iteration 2 Retraining Decision

### Decision

**Do not start student distillation from Teacher Iteration 1.** The first teacher proved that paired corruption training can preserve ranking under common post-processing, but it also exposed dataset, evaluation, selection, resume, and utilization failures. Teacher Iteration 2 must pass the fixed audit gate below before Checkpoint 2 is promoted for distillation.

The metrics in this section describe the immutable Stage 2 audit checkpoint at update 1,250. They do not use the later Checkpoint 2 adaptation.

### Preliminary Audit Evidence

| Checkpoint / Probe | Rows | Clean AUROC | AI-Positive Recall @ 0.5 | Authentic Recall @ 0.5 | Interpretation |
|---|---:|---:|---:|---:|---|
| Stage 1 validation probe | 1,024 | 0.9798 | 0.9456 | 0.8824 | Strong frozen-backbone anchor. |
| Stage 2 validation audit sample | 300 | 0.9577 | 0.7411 | 0.9709 | Large threshold shift toward predicting authentic; AI recall is unacceptable. |
| Stage 2 test audit sample | 300 | 0.9590 | 0.7754 | 0.9469 | Ranking remains useful, but the default threshold misses too many AI-positive samples. |
| Stage 2 unseen-positive sample | 115 | undefined | 0.9304 | undefined | Positive-only split can measure recall, not AUROC, specificity, or balanced accuracy. |

Stage 1 and Stage 2 numbers were produced by different sampled probes, so their absolute difference is **not** a controlled model-to-model delta. Changing the evaluation sample between stages was itself an audit mistake. The signal is still actionable: the Stage 2 checkpoint has robust ranking but poor AI-positive recall at the operating threshold.

The Stage 2 test AUROC changed only from 0.9590 clean to 0.9544 at JPEG-50, 0.9583 at blur-\(1.0\), 0.9586 at \(0.5\times\) resize, and 0.9551 at noise-\(0.02\). The unseen-positive recall changed from 0.9304 clean to 0.8783 at JPEG-50, 0.9304 at blur-\(1.0\), 0.9217 at \(0.5\times\) resize, 0.9043 at noise-\(0.02\), and 0.8957 at crop-80. Transformation robustness is therefore not the first problem to optimize in Iteration 2; calibration, positive recall, data coverage, and trustworthy selection are.

### Iteration 1 Weaknesses and Mistakes

1. **The executed data scope did not match the 127,902-row plan.** The audit teacher trained on 22,483 materialized training rows from five datasets. After adding the complete Gmorinan archive, the conflict-free pool contains 41,035 materialized rows across five datasets with zero missing paths—not 127,902 rows across 38 cohorts. The Hugging Face repository publishes all canonical metadata, but many `reference_only` cohorts do not include image payloads.
2. **Stage 2 had no live model selection.** Distributed rank-0 validation exceeded the NCCL watchdog while the other ranks waited, so `validation_interval` was disabled. `checkpoint-best.pt` consequently means final state, not the best observed \(N^*\).
3. **The evaluation protocol changed between stages.** Stage 1, Stage 2 validation, test, and unseen probes used different sample shapes. The unseen split contained no authentic negatives, making AUROC and specificity mathematically undefined.
4. **Full unfreezing shifted the operating point.** Stage 2 retained good AUROC and corruption stability but moved strongly toward authentic predictions, reducing AI-positive recall to 0.7411 on the sampled validation audit.
5. **Mask supervision was nominal rather than effective.** `mask_focal` and `mask_dice` stayed at zero throughout the observed Stage 2 run. The next run must either guarantee mask-bearing samples or disable those losses explicitly.
6. **Resume semantics were coupled to loader geometry.** Changing physical batch size changed loader length while the checkpoint retained `epoch` and `micro_step`; two resumes performed no updates because the saved offset exceeded the new epoch.
7. **A configuration key typo changed the effective batch.** `gradient_accumulation_steps` was added while the trainer reads `gradient_accumulation`. A short discarded run therefore used effective batch 96 instead of 48.
8. **Checkpoint 2 initially ignored its teacher checkpoint.** `initial_checkpoint` was placed under `training`; the trainer reads it under `paths`. The random-start run was detected from its 1.44 first-step loss and discarded after two updates. Configuration validation did not catch the misplaced key.
9. **Checkpoint storage was unsafe.** Full model, optimizer, scheduler, RNG, and EMA checkpoints were approximately 13.96 GB each. Saving every 100 updates filled the 128 GB instance disk and interrupted training.
10. **GPU duty cycle was low.** Six GPUs reached 100% SM during compute bursts but commonly drew only 50–60 W between bursts. CPU/Pillow transforms, `AutoImageProcessor`, Python per-image head work, and PCIe DDP synchronization dominated average utilization.

### Teacher Iteration 2 Execution Contract

#### A. Data and Audit Gates Before Renting GPUs
- Materialize and hash every intended image. If all 127,902 payloads are unavailable, publish the exact executable row count, source count, class balance, and split counts; never label a smaller run “127,902-row full data.”
- Freeze one validation probe and one test probe as immutable row-ID lists. Reuse the same rows, duplicate policy, transforms, and bootstrap groups for every checkpoint comparison.
- Build a composite unseen audit with the held-out AI-positive generators plus disjoint authentic negatives. Report the original positive-only recall separately.
- Record per-source and per-generator recall. Aggregate AUROC alone must not hide one failed generator family.

#### B. Configuration and Resume Preflight
- Reject unknown configuration keys and validate field placement before launch.
- Assert `physical_batch_size × gradient_accumulation × world_size = 48`.
- Load the initial checkpoint and verify a known weight tensor digest before the first optimizer step.
- Run a six-rank, two-update smoke test that exercises checkpoint save, resume, validation, and the exact production loader geometry.
- Do not change physical batch size mid-epoch. Resume only with identical loader geometry or reset the sampler epoch/offset explicitly.

#### C. Iteration 2 Optimization Schedule
- Rebuild the Stage 1 frozen anchor for 600 updates when the materialized dataset changes.
- Start Stage 2 by unfreezing only the last eight DINOv3 transformer layers for 300 updates.
- Use head LR \(3.0 \times 10^{-5}\), encoder LR \(1.0 \times 10^{-6}\), LLRD 0.85, AdamW weight decay 0.01, BF16/TF32, and effective batch 48.
- Evaluate the immutable validation probe every 50 updates. Continue to full unfreezing for at most 600 additional updates only when robust mean AUROC improves without violating either recall guard.
- Keep the current realistic transform suite and severities. Do not add stronger adversarial transformations while clean AI-positive recall is the dominant failure.
- Require AI-positive recall \(\ge 0.85\), authentic recall \(\ge 0.85\), and clean AUROC within 0.5 percentage points of the best checkpoint. Select the highest robust-probe mean AUROC satisfying all three guards.
- Calibrate the binary operating threshold on validation only, then freeze it before test and unseen evaluation. Report both threshold-free AUROC and thresholded recalls.
- Enable mask losses only when the batch sampler guarantees measurable mask coverage; log mask-bearing samples per update.

#### D. Throughput and Checkpointing
- Move JPEG/blur/resize/noise/color/crop generation to batched tensor transforms and feed normalized tensors directly to the backbone.
- Vectorize the provenance head so batch work does not loop over images in Python.
- Benchmark physical batches 2 and 4 with effective batch fixed at 48; choose by median samples/second and update time, not one-second utilization snapshots.
- Save lightweight model-selection weights every 50 updates and a full resumable optimizer checkpoint every 250 updates. Maintain at most two full resume checkpoints locally and exfiltrate them asynchronously.

#### E. Promotion Gate
- Produce full validation, test, composite-unseen, per-generator, and corruption reports with duplicate-group bootstrap intervals.
- Promote Teacher Iteration 2 only if it beats the Iteration 1 audit checkpoint on the frozen comparison set, passes both recall guards, and has no critical corruption or generator-family regression.
- Only after that promotion: train Checkpoint 2 for exactly \(0.20N^*\) updates on the verified full available pool, audit it for catastrophic forgetting, and unlock student distillation.

---

## 8. Tri-Tier Student Distillation Execution

**Blocked pending the Teacher Iteration 2 promotion gate.** Once the retrained Stage 2 Teacher Checkpoint 1 is locked and its Checkpoint 2 adaptation passes the post-adaptation audit, run `scripts/distill_student.py` for each tier:

### Distillation Hyperparameters
- **Optimizer:** AdamW, $\beta_1=0.9, \beta_2=0.999$, weight decay 0.01.
- **Student Head LR:** $2.0 \times 10^{-4}$ | **Student Backbone LR:** $2.0 \times 10^{-5}$ (LLRD = 0.85).
- **Loss Weights:** BCE = 1.0, KL Distillation = 2.0, Prediction Consistency = 1.0, Feature Cosine = 0.5.
- **Duration:** 1,000 steps per student tier.

```bash
# Tier 3: Ultra Edge (DINOv3 ViT-S/16 — 21M)
python scripts/distill_student.py \
  --student-model facebook/dinov3-vits16-pretrain-lvd1689m \
  --teacher-checkpoint outputs/teacher_stage2/checkpoint-best.pt \
  --output-dir outputs/student_tier3_vits \
  --steps 1000

# Tier 2: Balanced (DINOv3 ViT-B/16 — 86M)
python scripts/distill_student.py \
  --student-model facebook/dinov3-vitb16-pretrain-lvd1689m \
  --teacher-checkpoint outputs/teacher_stage2/checkpoint-best.pt \
  --output-dir outputs/student_tier2_vitb \
  --steps 1000

# Tier 1: Pro Edge (DINOv3 ConvNeXt-L — 198M)
python scripts/distill_student.py \
  --student-model facebook/dinov3-convnext-large-pretrain-lvd1689m \
  --teacher-checkpoint outputs/teacher_stage2/checkpoint-best.pt \
  --output-dir outputs/student_tier1_convnext \
  --steps 1000
```

---

## 9. Pre-PTQ Adversarial Transformation Hardening

Adversarial transformation hardening is a **metric-triggered fallback**, not an automatic stage. The main distillation run already MUST use paired clean/transformed inputs and consistency losses. A second robustness-only pass is justified only when the distilled FP model fails the held-out corruption gate; running it unconditionally wastes compute and can trade clean accuracy for narrow corruption robustness.

### Entry Gate
- Evaluate the distilled FP checkpoint on clean, JPEG-50, blur-\(1.0\), downscale-\(0.5\times\), and noise-\(0.02\) probes.
- Enter hardening when robust-probe mean AUROC trails clean AUROC by more than 1.0 percentage point, or any critical corruption probe has AI-positive or authentic recall below 0.80.
- When the checkpoint passes both conditions, skip this phase and export the distilled checkpoint directly.

### Conditional Training Contract
- **Initialization:** Load the best distilled checkpoint for the same student tier. Never restart from the public backbone.
- **Training Pool:** Use the Checkpoint 2 full-data pool with generator-balanced sampling (50% authentic / 50% AI-positive).
- **Paired Inputs:** Train on each clean image \(x\) and one deployment-realistic adversarial transform \(T(x)\).
- **Transform Search:** Sample three candidates per image from JPEG quality \(15\text{--}90\), Gaussian blur \(\sigma=0.1\text{--}3.0\), downscale/upsample \(0.35\text{--}0.95\times\), Gaussian noise \(\sigma=0.005\text{--}0.05\), retained-area crop \(0.60\text{--}0.95\), and brightness/contrast/saturation \(0.70\text{--}1.30\). Train against the candidate that maximizes classification loss plus clean/transformed prediction divergence.
- **Clean Replay:** Keep 25% of minibatch pairs untransformed to prevent clean-distribution regression.
- **Loss:** Preserve student BCE, teacher KL distillation, prediction consistency, and feature cosine losses. The production teacher predicts the clean image; the student learns from both \(x\) and the selected \(T(x)\).
- **Budget:** At most 200 optimizer steps per student, with early stopping on the held-out robust probes.
- **Learning Rate:** Student heads \(2.0 \times 10^{-5}\), backbone \(2.0 \times 10^{-6}\), cosine decay, no warmup restart.
- **Selection Gate:** Maximize robust-probe mean AUROC while requiring clean AUROC to remain within 0.5 percentage points of the distilled checkpoint.
- **Output:** Save `checkpoint-adversarial-best.pt` only when it passes the selection gate. Export the hardened checkpoint when it exists; otherwise export the original distilled checkpoint.

This is transformation-based hardening against realistic browser and social-media post-processing, not imperceptible \(\ell_p\)-bounded PGD. After PTQ, repeat the same probes. If INT8 alone causes more than 0.5 percentage points of AUROC regression, use quantization-aware training; do not repeat adversarial hardening.

---

## 10. WebGPU Export & INT8 PTQ Pipeline

Export the selected student checkpoint (hardened when the entry gate triggered, otherwise distilled) using `forward_tensor(pixel_values: Tensor)` to avoid Python batch loops and unsupported WebGPU operators:

### 1. Vectorized ONNX Export
```python
import torch

dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)
torch.onnx.export(
    student_model,
    dummy_input,
    "edge_models/model_tier3_vits.onnx",
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["ai_positive_logit"],
    dynamic_axes={"input": {0: "batch_size"}, "ai_positive_logit": {0: "batch_size"}},
)
```

### 2. Static INT8 Post-Training Quantization (PTQ)
Run calibration over 1,500 diverse, class-balanced samples from `zye2/tj-data`, split evenly between clean images and the deployment transformations used by the robustness probes:
```bash
python scripts/export_webgpu.py \
  --onnx-model edge_models/model_tier3_vits.onnx \
  --calibration-manifest splits/combined_hf_dataset/validation.csv \
  --quantization int8 \
  --output edge_models/model_tier3_vits_int8.onnx
```

---

## 11. Interactive WebGPU Demo Client (`demo/`)

The demo runs 100% client-side inside the browser using WebGPU execution providers.

### File Structure:
```text
demo/
├── index.html                 # Sleek UI with upload zone, sliders, and tier dropdown
├── css/style.css              # Modern dark-mode forensic styling
├── js/
│   ├── app.js                 # UI events, canvas rendering, adversarial slider drivers
│   ├── webgpu_detector.js     # onnxruntime-web session management and inference runner
│   └── transforms.js          # In-browser image perturbation engine (JPEG, blur, noise, crop)
└── models/
    ├── model_tier3_vits_int8.onnx   # 21 MB (Default load)
    ├── model_tier2_vitb_int8.onnx   # 86 MB
    └── model_tier1_convnext_int8.onnx # 198 MB
```

### WebGPU Worker Core (`webgpu_detector.js`):
```javascript
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/esm/ort.webgpu.min.js";

ort.env.wasm.numThreads = 4;
ort.env.webgpu.powerPreference = "high-performance";

export class WebGPUAIGCDetector {
    async initialize(modelPath = "./models/model_tier3_vits_int8.onnx") {
        this.session = await ort.InferenceSession.create(modelPath, {
            executionProviders: ["webgpu"]
        });
    }

    async predict(rgbFloat32Tensor) {
        const tensor = new ort.Tensor("float32", rgbFloat32Tensor, [1, 3, 224, 224]);
        const start = performance.now();
        const results = await this.session.run({ input: tensor });
        const latency = performance.now() - start;

        const logit = results.ai_positive_logit.data[0];
        const probability = 1.0 / (1.0 + Math.exp(-logit));
        return { probability, latency };
    }
}
```

### Interactive Features That Impress Judges:
1. **Zero Cloud Latency:** Displays live inference latency (e.g. `6.2 ms | 161 FPS`).
2. **Interactive Perturbation Stress-Test:** Sliders for live JPEG compression quality ($100 \rightarrow 15$), Gaussian blur radius ($0 \rightarrow 4\text{px}$), and noise. Judges see competing web models flip from Fake to Real as soon as compression is applied, while our consistency-trained model remains rock-solid.
3. **Multi-Tier Switcher:** Seamlessly switches between the 21M mobile model and the 198M Pro model with live VRAM readout.

---

## 12. Execution Timeline

| Milestone | Status | Observable Verification Deliverable |
|---|---|---|
| **Data Publishing** | **Metadata done; payload hydration incomplete** | [`zye2/tj-data`](https://huggingface.co/datasets/zye2/tj-data) publishes 127,902 metadata rows; executable training reports exact materialized coverage separately. |
| **Teacher Iteration 1 Stage 1** | **Done** | Frozen anchor at update 600; validation AUROC 0.9798. |
| **Teacher Iteration 1 Stage 2** | **Done** | Immutable audit checkpoint at update 1,250 plus sampled clean/corruption reports. |
| **Iteration 1 Audit Review** | **Done—retraining required** | Weaknesses and corrective contract documented in Section 7. |
| **Teacher Iteration 2 Preflight** | **Next** | Frozen manifests, strict config validation, exact checkpoint warm-start proof, and six-rank save/resume/validation smoke test. |
| **Teacher Iteration 2 Training** | **Pending** | Progressive-unfreeze checkpoint passing AUROC and dual-recall promotion gates. |
| **Production Checkpoint 2** | **Blocked on Iteration 2** | \(0.20N^*\) full-available-pool adaptation plus post-adaptation regression audit. |
| **Tri-Tier Distillation** | **Blocked** | Starts only from the promoted Iteration 2 production teacher. |
| **Adversarial Student Hardening Gate** | **Blocked** | Runs per student only when its FP robustness probe fails. |
| **ONNX, INT8, and WebGPU Demo** | **Blocked** | Starts after student promotion and post-PTQ parity checks. |
