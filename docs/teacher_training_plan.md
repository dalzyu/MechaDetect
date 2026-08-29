# DINOv3 Teacher Training Plan (4x RTX 4090)

**Target Topology:** 4× NVIDIA GeForce RTX 4090 (24 GB VRAM each), NCCL DDP
**Authoritative Plan Reference:** [`docs/production_4x4090_implementation_plan.md`](production_4x4090_implementation_plan.md)
**Historical Run Reference:** [`docs/training_run_consolidated.md`](training_run_consolidated.md)

---

## 1. Teacher Architecture & Identity

- **Vision Backbone:** DINOv3 ViT-H+/16 (`facebook/dinov3-vith16plus-pretrain-lvd1689m`, revision `c807c9eeea853df70aec4069e6f56b28ddc82acc`).
- **Backbone Parameters:** **840.6M** parameters.
- **Complete MechaDetect Teacher:** **872.6M** parameters (872,606,207 saved parameter values including token adapter, global evidence head, localization head, and spectral expert).
- **Target Label:** Binary image-level `ai_positive` (authentic = 0, tampered/fully-generated = 1).

---

## 2. Immutable Training Data Contract

Both Stage 1 and Stage 2 consume the identical immutable eligible training split:
- **Manifest:** `splits/production_eligible/train.parquet`
- **Validation Population:** `splits/production_eligible/validation.parquet`
- **Data Integrity:** Prefetched prior to DDP; zero remote network fetching during training; fail-closed loaders; no missing-row substitution.
- **Calibration Isolation:** 4,096 calibration rows (`splits/production_eligible/calibration.parquet`) are preserved exclusively for static INT8 PTQ and never seen by the teacher.

---

## 3. Stage 1: Frozen Backbone Representation Anchor

Stage 1 trains only task-specific layers (token adapter, attention heads, and classifier) from the pinned pretrained DINOv3 weights on untransformed downloaded-original views.

### Contract
- **Config:** `configs/teacher_dinov3_stage1_clean_frozen.yaml`
- **Backbone:** Complete DINOv3 backbone frozen.
- **Data View:** Downloaded-original view only (no synthetic post-processing perturbations).
- **Schedule:** One complete deterministic pass across all eligible training rows.
- **Batch Geometry:** Physical batch 6 × 4 GPUs × 2 gradient accumulation steps = **effective batch 48**.

### Stage 1 Execution
```bash
uv run torchrun --standalone --nproc-per-node=4 \
  -m aigc_detector.train \
  --config configs/teacher_dinov3_stage1_clean_frozen.yaml
```

### Stage 1 External Promotion Gate
Checkpoints are evaluated at ~25%, 50%, 75%, and 100% coverage via external post-pass evaluation on the validation split:
```bash
uv run python scripts/promote_teacher.py \
  --checkpoints "outputs/teacher_stage1_clean_frozen/*.pt" \
  --manifest splits/production_eligible/validation.parquet \
  --output-report outputs/teacher_stage1_clean_frozen/promotion_report.json \
  --output-metadata outputs/teacher_stage1_clean_frozen/metadata.json
```
- **Promotion Gate:** Clean AUROC > 0.96 and both class recalls $\ge 0.82$ at the calibrated threshold.
- Promoted artifact (`checkpoint-promoted.pt`) is immediately uploaded to private Hugging Face repo `zye2/mechadetect-models`.

---

## 4. Stage 2: Full Backbone Unfrozen Paired Adaptation

Stage 2 warm-starts from the promoted Stage 1 checkpoint with a fresh optimizer and scheduler, unfreezing the complete backbone to learn robust invariant representations across paired content-preserving transformations.

### Contract
- **Config:** `configs/teacher_dinov3_stage2_paired_unfrozen.yaml`
- **Initial Checkpoint:** Promoted Stage 1 checkpoint (`outputs/teacher_stage1_clean_frozen/checkpoint-promoted.pt`).
- **Backbone:** Fully unfrozen with Layer-wise Learning Rate Decay (LLRD 0.85). Low encoder learning rate ($1 \times 10^{-5}$).
- **Data View:** Downloaded-original paired with exactly one allowed single post-processing transformation per sample (JPEG, blur, resize, noise, color, crop).
- **Objectives:** Supervised classification on both views, prediction consistency loss, feature consistency loss, and confidence-gated EMA distillation. Mask losses disabled.
- **Schedule:** One complete deterministic pass across all eligible training rows.
- **Batch Geometry:**
  - *Primary:* Physical batch 2 × 4 GPUs × 6 gradient accumulation steps = **effective batch 48**.
  - *OOM Fallback:* Physical batch 1 × 4 GPUs × 12 gradient accumulation steps = **effective batch 48**.

### Stage 2 Execution
```bash
uv run torchrun --standalone --nproc-per-node=4 \
  -m aigc_detector.train \
  --config configs/teacher_dinov3_stage2_paired_unfrozen.yaml \
  --initial-checkpoint outputs/teacher_stage1_clean_frozen/checkpoint-promoted.pt
```

### Stage 2 External Promotion Gate
```bash
uv run python scripts/promote_teacher.py \
  --checkpoints "outputs/teacher_stage2_paired_unfrozen/*.pt" \
  --manifest splits/production_eligible/validation.parquet \
  --output-report outputs/teacher_stage2_paired_unfrozen/promotion_report.json \
  --output-metadata outputs/teacher_stage2_paired_unfrozen/metadata.json
```
- **Promotion Gate:** Clean AUROC > 0.96 and both class recalls $\ge 0.82$ at the calibrated threshold.
- Promoted artifact (`checkpoint-promoted.pt`) is immediately uploaded to `zye2/mechadetect-models`.

---

## 5. Checkpoint 2 Status: Demoted

Checkpoint 2 has been demoted from the forward canonical pipeline. Downstream student distillation proceeds directly from the promoted Stage 2 teacher.

---

## 6. Full Orchestration

To run the complete teacher pipeline with automated hardware checks, smoke runs, promotion gates, and Hub uploads:
```bash
bash orchestrate_4x4090.sh --stage teacher-stage1
```
