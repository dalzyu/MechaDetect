# MechaDetect Production Training and Delivery Plan (4x RTX 4090)

**Deadline:** September 1, 2026
**Status:** Active
**Hardware Target:** 4× NVIDIA GeForce RTX 4090 (24 GB VRAM each) on Vast.ai
**Working Branch:** `training/production-4x4090`
**Authoritative Plan Reference:** [`docs/production_4x4090_implementation_plan.md`](production_4x4090_implementation_plan.md)
**Historical Run Reference:** [`docs/training_run_consolidated.md`](training_run_consolidated.md)

---

## 1. Executive Summary & Architecture

MechaDetect is a robust binary AI-provenance detector (Track 5) distinguishing authentic human imagery (`ai_positive = 0`) from both fully generated and AI-edited imagery (`ai_positive = 1`).

The production pipeline deploys an 872.6M-parameter foundation teacher (DINOv3 ViT-H+/16 backbone + multi-scale evidence/localization heads) and distills it concurrently into two independent edge students:
1. **Edge Primary (Tier 2):** Complete 25.1M-parameter ViT-S detector (`facebook/dinov3-vits16-pretrain-lvd1689m`) optimized for WebGPU and browser deployment.
2. **Accuracy Primary (Tier 1):** Complete ViT-B detector (`facebook/dinov3-vitb16-pretrain-lvd1689m`, ~86M backbone) for high-capacity desktop / edge deployment.

### Exact Model Identities & Parameter Counts

| Model Role | Base Backbone | Complete Detector Params | Target Runtime |
|---|---|---:|---|
| **Teacher** | DINOv3 ViT-H+/16 (840.6M) | **872.6M** (872,606,207 values) | Training / Distillation Only |
| **Student (Edge Primary)** | DINOv3 ViT-S/16 (~21M) | **25.1M** (25,089,666 values) | WebGPU & WASM (Static INT8) |
| **Student (Accuracy Primary)** | DINOv3 ViT-B/16 (~86M) | Measured (~92M complete) | High-throughput Edge |

---

## 2. Immutable Dataset Specification (`splits/production_eligible/`)

Data is managed through a prefetch-before-DDP architecture. Production loaders are strictly local and fail-closed:
- **No missing-row substitution:** The legacy `_fallback_by_key` replacement is deleted. Missing or corrupt samples raise errors immediately.
- **No training-time network fetching:** All assets must be downloaded and verified in preflight.
- **Calibration Split Disjointness:** Exactly 4,096 calibration rows (`calibration.parquet`) are preserved exclusively for static INT8 PTQ and never seen during gradient training or validation.
- **Package Layout:**
  - `splits/production_eligible/declared_manifest.parquet`
  - `splits/production_eligible/train.parquet`
  - `splits/production_eligible/validation.parquet`
  - `splits/production_eligible/test.parquet`
  - `splits/production_eligible/test_unseen.parquet`
  - `splits/production_eligible/calibration.parquet` (4,096 rows)
  - `splits/production_eligible/exclusions.parquet`
  - `splits/production_eligible/audit_report.json`
  - `splits/production_eligible/source_revisions.json`

---

## 3. 4x RTX 4090 Gated Training Pipeline

Execution is managed by `orchestrate_4x4090.sh` through a 21-stage resumable state machine with budget guards and artifact uploads after every passed gate:

```text
[Preflight & Verification]
         │
         ▼
[Acquire Images & Freeze Manifests] ──► Upload Manifests to zye2/tj-data
         │
         ▼
[Teacher Stage 1 (Frozen Backbone, Full Train Split)]
         │
         ▼
[Teacher Stage 1 Promotion Gate] ──► Upload Stage 1 to zye2/mechadetect-models
         │ (clean AUROC > 0.96, both recalls >= 0.82)
         ▼
[Teacher Stage 2 (Unfrozen Backbone, Paired Transforms)]
         │
         ▼
[Teacher Stage 2 Promotion Gate] ──► Upload Stage 2 to zye2/mechadetect-models
         │ (clean AUROC > 0.96, both recalls >= 0.82)
         ├──────────────────────────────────────────┐
         ▼                                          ▼
[ViT-S Distillation (GPUs 0,1)]            [ViT-B Distillation (GPUs 2,3)]
         │                                          │
         ├──────────────────────────────────────────┘
         ▼
[Student Float Promotion Gate] ──► Upload Float Students to zye2/mechadetect-models
         │ (recalls >= 0.82, AUROC within 2pp clean / 3pp robust of teacher)
         ├──────────────────────────────────────────┐
         ▼                                          ▼
[ViT-S ATT Hardening (GPUs 0,1)]           [ViT-B ATT Hardening (GPUs 2,3)]
         │                                          │
         ├──────────────────────────────────────────┘
         ▼
[ATT Promotion Gate] ──► Upload ATT Models to zye2/mechadetect-models
         │ (worst-transform AUROC improves, clean loss <= 0.5pp, recalls >= 0.82)
         ▼
[ONNX Opset 17 Export]
         │
         ▼
[Static INT8 PTQ (4,096 Calibration Rows)]
         │
         ▼
[Graph Inspection (Strict Anti-INT4 Gate)]
         │
         ▼
[Runtime Verification (WebGPU & forced WASM)]
         │
         ▼
[Consolidated Upload & Pipeline Completion Receipt]
```

### Note on Checkpoint 2 Demotion
Checkpoint 2 has been demoted from the forward canonical pipeline. A second 200-update run over the same manifest is not a distinct phase. The canonical pipeline moves directly from promoted Stage 2 into concurrent student distillation.

---

## 4. Multi-GPU Topology and Batch Geometry

All stages preserve the canonical effective record batch of **48**:

| Pipeline Stage | GPU Allocation | Physical Batch / GPU | Gradient Accumulation | Effective Batch |
|---|---|---:|---:|---:|
| **Teacher Stage 1** | GPUs 0, 1, 2, 3 (4-GPU DDP) | 6 | 2 | 48 (6 × 4 × 2) |
| **Teacher Stage 2 (Primary)** | GPUs 0, 1, 2, 3 (4-GPU DDP) | 2 | 6 | 48 (2 × 4 × 6) |
| **Teacher Stage 2 (OOM Fallback)** | GPUs 0, 1, 2, 3 (4-GPU DDP) | 1 | 12 | 48 (1 × 4 × 12) |
| **ViT-S Distillation** | GPUs 0, 1 (2-GPU DDP, Port 29501) | 6 | 4 | 48 (6 × 2 × 4) |
| **ViT-B Distillation** | GPUs 2, 3 (2-GPU DDP, Port 29502) | 3 | 8 | 48 (3 × 2 × 8) |
| **ViT-S ATT Hardening** | GPUs 0, 1 (2-GPU DDP, Port 29503) | 4 | 6 | 48 (4 × 2 × 6) |
| **ViT-B ATT Hardening** | GPUs 2, 3 (2-GPU DDP, Port 29504) | 2 | 12 | 48 (2 × 2 × 12) |

---

## 5. Adversarial Transformation Training (ATT) Contract

- **Candidate Set:** Allowed content-preserving single transformations:
  - JPEG compression (quality: 40–90)
  - Gaussian blur (radius: 0.5–2.5)
  - Bilinear/bicubic resizing (scale: 0.5–1.5)
  - Gaussian noise ($\sigma$: 0.02–0.10)
  - Color adjustment (brightness/contrast/saturation: 0.7–1.3)
  - Center/random cropping (area: 0.75–0.95)
- **Loss Contract:** Retains untransformed downloaded-original supervised loss in every update. The student evaluates multiple candidates without gradient retention and backpropagates through the candidate with the highest loss.
- **ATT Promotion Gate:**
  - Worst-transformation AUROC improves over float baseline.
  - Worst-domain AUROC does not regress.
  - Clean AUROC regression $\le 0.5$ percentage points.
  - Both class recalls remain $\ge 0.82$.

---

## 6. Static INT8 PTQ & Graph Verification Contract

- **Opset:** ONNX opset 17.
- **Quantization Mode:** Calibrated static INT8 (QDQ / QLinear) using the 4,096 calibration rows.
- **Strict Anti-INT4 Gate:** Models containing INT4 `MatMulNBits` or dynamic-only quantizations are strictly rejected by `scripts/quantize_webgpu_nbits.py --inspect-model`.
- **Runtime Verification:** Verified across desktop Chrome/Edge WebGPU and forced-WASM CPU fallback.

---

## 7. Vast.ai Budget Guard & Cost Projection

- **Reserve Requirement:** The pipeline enforces a mandatory **$5.00** balance reserve at all times.
- **Cost Projection:** Prior to launching full stages, measured throughput from 2-update smoke tests is used to project stage costs.
- **Fail Closed:** If balance cannot be determined via Vast CLI or API (and `--explicit-balance` is not passed), the pipeline fails closed.
- **Exfiltration on Stop:** If a stage is projected to breach the $5 reserve, execution stops gracefully and uploads all currently promoted artifacts before shutdown.

---

## 8. CLI Command Reference

```bash
# 1. Fresh Instance Setup
bash cluster_setup.sh

# 2. Complete Gated Production Run
bash orchestrate_4x4090.sh

# 3. Targeted Stage Execution / Resume
bash orchestrate_4x4090.sh --stage teacher-stage1
bash orchestrate_4x4090.sh --stage students-distill --explicit-balance 35.00
```
