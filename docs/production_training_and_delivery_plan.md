# MechaDetect Production Training and Delivery Record

**Delivery date:** August 31, 2026
**Status:** Training and artifact delivery complete; TechJam/SoTA evaluation pending
**Default hardware:** one CUDA GPU with BF16 support
**Default branch:** `main`
**Historical four-GPU design:** [`docs/production_4x4090_implementation_plan.md`](production_4x4090_implementation_plan.md)
**Historical teacher run:** [`docs/training_run_consolidated.md`](training_run_consolidated.md)

---

## 1. Executive Summary & Architecture

MechaDetect is a robust binary AI-provenance detector (Track 5) distinguishing authentic human imagery (`ai_positive = 0`) from both fully generated and AI-edited imagery (`ai_positive = 1`).

The training path uses an 872.6M-parameter DINOv3 ViT-H+/16 teacher and two
independent students. Atom is the 25.1M-parameter ViT-S browser model. Quark is
the higher-capacity ViT-B model. On one GPU, their distillation and ATT stages
run sequentially; explicit disjoint device pools allow parallel execution.

### Exact Model Identities & Parameter Counts

| Model Role | Base Backbone | Complete Detector Params | Target Runtime |
|---|---|---:|---|
| **Teacher** | DINOv3 ViT-H+/16 (840.6M) | **872.6M** (872,606,207 values) | Training / Distillation Only |
| **Atom / ViT-S** | DINOv3 ViT-S/16 (~21M) | **25.1M** (25,089,666 values) | Browser WebGPU/WASM |
| **Quark / ViT-B** | DINOv3 ViT-B/16 (~86M) | ~89M complete | Desktop / higher-capacity edge |

---

## 2. Immutable Dataset Specification (`splits/production_eligible/`)

Data is prefetched before optimization. Training loaders are local and fail closed:
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

## 3. Maintained Training and Delivery Flow

`scripts/launch_production.sh` is the maintained resumable training entry point.
It defaults to GPU 0. The older `orchestrate_4x4090.sh` state machine remains an
explicit Vast/RTX-4090 workflow and is not the default.
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
[Atom Distillation]                            [Quark Distillation]
         │                                          │
         ├──────────────────────────────────────────┘
         ▼
[Student Float Promotion Gate] ──► Upload Float Students to zye2/mechadetect-models
         │ (recalls >= 0.82, AUROC within 2pp clean / 3pp robust of teacher)
         ├──────────────────────────────────────────┐
         ▼                                          ▼
[Atom ATT Hardening]                           [Quark ATT Hardening]
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

### Checkpoint 2

Checkpoint 2 is a historical ablation, not a production phase. The maintained
flow moves directly from the final Stage 2 teacher into student distillation.

---

## 4. Default Batch Geometry

The checked-in configs target one GPU and preserve an effective record batch of
48. An explicit multi-GPU device list reduces accumulation by the same factor.

| Pipeline stage | Default devices | Physical batch | Accumulation | Effective batch |
|---|---|---:|---:|---:|
| **Teacher Stage 1** | GPU 0 | 6 | 8 | 48 |
| **Teacher Stage 2** | GPU 0 | 2 | 24 | 48 |
| **Teacher Stage 2 OOM fallback** | GPU 0 | 1 | 48 | 48 |
| **Atom distillation** | GPU 0 | 12 | 4 | 48 |
| **Quark distillation** | GPU 0 | 3 | 16 | 48 |
| **Atom ATT** | GPU 0 | 4 | 12 | 48 |
| **Quark ATT** | GPU 0 | 2 | 24 | 48 |

Student and ATT launchers run both tracks sequentially by default.
`--parallel-tracks` requires disjoint device pools. Resume requires the same
world size and batch geometry as the checkpoint.

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
- **Quantization mode:** calibrated static INT8 QDQ using all 4,096 calibration rows.
- **Graph result:** all four INT8 exports passed the static-quantization structure checks.
- **Runtime result:** Atom Super float32 agreed between WebGPU and forced WASM and is the browser default. The INT8 candidate executed on both providers but produced materially different probabilities, so INT8 remains experimental.

---

## 7. Historical Vast Run Controls

The four-RTX-4090 orchestrator retains its balance reserve, cost projection,
stage receipts, and upload logic for reproducibility. Those controls are not
required by the local one-GPU launcher. The completed Vast instance was
terminated after artifact exfiltration.

---

## 8. Commands and Evaluation Status

```bash
# Prepare dependencies, storage, images, and frozen manifests
bash cluster_setup.sh

# Default: one GPU, sequential student and ATT tracks
bash scripts/launch_production.sh

# Explicit two-GPU DDP
GPU_DEVICES=0,1 bash scripts/launch_production.sh

# Historical four-RTX-4090 state machine
bash orchestrate_4x4090.sh
```

The completed checkpoints and ONNX exports have not yet been benchmarked against
external state-of-the-art detectors on the TechJam evaluation set. Training
completion, ONNX parity, graph inspection, and browser-provider checks are not a
substitute for that comparison. Publish a SoTA claim only after all detectors
run on the same TechJam rows, transformations, and metric implementation.
