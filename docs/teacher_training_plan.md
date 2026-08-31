# DINOv3 Teacher Training Plan

**Default topology:** one CUDA GPU with BF16 support
**Optional topology:** explicit DDP through `torchrun`
**Historical run reference:** [`docs/training_run_consolidated.md`](training_run_consolidated.md)

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
- **Data Integrity:** Prefetched before training; no network fetches in the loader; fail-closed decoding; no missing-row substitution.
- **Calibration Isolation:** 4,096 calibration rows (`splits/production_eligible/calibration.parquet`) are preserved exclusively for static INT8 PTQ and never seen by the teacher.

---

## 3. Stage 1: Frozen Backbone Representation Anchor

Stage 1 trains only task-specific layers (token adapter, attention heads, and classifier) from the pinned pretrained DINOv3 weights on untransformed downloaded-original views.

### Contract
- **Config:** `configs/teacher_dinov3_stage1_clean_frozen.yaml`
- **Backbone:** Complete DINOv3 backbone frozen.
- **Data View:** Downloaded-original view only (no synthetic post-processing perturbations).
- **Schedule:** One complete deterministic pass across all eligible training rows.
- **Default batch geometry:** physical batch 6 × 1 GPU × 8 accumulation = **effective batch 48**.

### Stage 1 execution

```bash
uv run python -m aigc_detector.train \
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
- **Default batch geometry:** physical batch 2 × 1 GPU × 24 accumulation = **effective batch 48**.
- **One-GPU OOM fallback:** physical batch 1 × 1 GPU × 48 accumulation = **effective batch 48**.

### Stage 2 execution

```bash
uv run python -m aigc_detector.train \
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

## 6. Maintained orchestration

[`train.ipynb`](../train.ipynb) is the maintained one-GPU workflow for both
teacher stages, external evaluation and promotion, student distillation, ATT,
and ONNX export:

```bash
uv run --with jupyter jupyter lab train.ipynb
```

Do not resume a checkpoint under a different world size or batch geometry.

The completed August 2026 delivery skipped teacher evaluation and promotion at
operator request. Those checkpoints are final training outputs, not
promotion-gated or state-of-the-art-evaluated models.

## 7. Local runtime verification

On September 1, 2026, the maintained trainer loaded the pinned DINOv3
ViT-H+/16 weights and ran Teacher Stage 1 on an RTX 4080 using a temporary
three-image manifest. With physical batch 1, accumulation 1, BF16, and the
encoder frozen, it completed 29 real forward, backward, and optimizer updates.
The process was intentionally cancelled after 2 minutes 3 seconds.

This is an execution-path check only. It proves model construction, pretrained
weight loading, manifest materialization, preprocessing, CUDA execution, loss
calculation, backpropagation, and optimizer stepping. It does not constitute a
trained checkpoint, quality result, promotion result, or complete execution of
[`train.ipynb`](../train.ipynb).
