# Production Training Dataset Specification: Eligible Package (`splits/production_eligible/`)

**Project:** TechJam 2026 — Track 5: AI-Generated & Tampered Image Detection  
**Canonical Package:** `splits/production_eligible/`  
**Remote Transparency Repo:** `zye2/tj-data`  
**Maintained Workflow:** [`train.ipynb`](../train.ipynb)
**Historical Run Reference:** [`docs/training_run_consolidated.md`](training_run_consolidated.md)

---

## 1. Executive Summary & Production Data Architecture

For the 4× RTX 4090 production pipeline, data ingestion adheres to strict integrity and reproducibility guarantees:
1. **Prefetch Before DDP:** All image payloads must be acquired and verified on local NVMe prior to distributed training (`python scripts/data_prep/acquire_all_images.py`). No dynamic network streaming during DDP execution.
2. **Fail Closed:** If any requested image file is missing or corrupt at runtime, the dataset raises an immediate exception.
3. **Zero Missing-Row Substitution:** The legacy `_fallback_by_key` mechanism has been removed. Missing rows are never substituted with another image sharing provenance/label.
4. **Immutable Frozen Manifests:** Manifests are partitioned and frozen into immutable Parquet files under `splits/production_eligible/`, accompanied by cryptographic digests, source revisions, and an integrity audit report.
5. **Strictly Disjoint Static Calibration Split:** Exactly 4,096 calibration samples are selected for static INT8 post-training quantization (PTQ), strictly disjoint from train, validation, test, and test-unseen splits.

---

## 2. Package Structure (`splits/production_eligible/`)

| File Name | Format | Role / Membership |
|---|---|---|
| `declared_manifest.parquet` | Parquet | Full declared population across all strategic cohorts |
| `train.parquet` | Parquet | Immutable eligible training split (consumed by Stage 1, Stage 2, Distillation, ATT) |
| `validation.parquet` | Parquet | Validation split for threshold calibration and promotion gates |
| `test.parquet` | Parquet | In-distribution benchmark evaluation split |
| `test_unseen.parquet` | Parquet | Strict out-of-distribution / unseen generator family test split |
| `calibration.parquet` | Parquet | **4,096 rows** strictly disjoint from all canonical splits (static INT8 PTQ only) |
| `exclusions.parquet` | Parquet | Quarantined records (corrupt bytes, missing sources, leakage/duplicate conflicts) |
| `audit_report.json` | JSON | Machine-readable breakdown: declared vs eligible counts, distributions, digests |
| `source_revisions.json` | JSON | Pinned Hugging Face source git revisions and commit SHAs |

---

## 3. Runtime Identity & Schema

Every record loaded during training, evaluation, and quantization carries complete identity metadata:

| Column | Type | Description |
|---|---|---|
| `row_id` | string | Deterministic unique row identifier |
| `split` | string | Split membership (`train`, `validation`, `test`, `test_unseen`, `calibration`) |
| `image_path` | string | Standardized relative path under `$TECHJAM_DATA_ROOT` |
| `provenance` | string | Forensic subtype: `authentic`, `tampered`, or `fully_aigc` |
| `ai_positive` | int (0/1) | Primary binary classification target (0 = authentic, 1 = tampered or fully_aigc) |
| `dataset` | string | Cohort identifier (e.g. `sid`, `wildfake`, `diffusionforensics`, `open_images_v7`) |
| `generator` | string | Underlying generative model or camera sensor (e.g. `flux`, `dalle3`, `midjourney_v6`) |
| `generator_family` | string | Generator architecture family for cross-domain evaluation |
| `source_image_group`| string | Grouping key for related/edited versions of an authentic image |
| `duplicate_group` | string | Exact/perceptual duplicate cluster ID (strictly isolated across splits) |

---

## 4. Provenance Hierarchy & Target Mapping

$$\begin{aligned}
\text{authentic} &\longrightarrow \text{provenance: authentic} \quad (\text{ai\_positive } = 0) \\
\text{tampered} &\longrightarrow \text{provenance: tampered} \quad (\text{ai\_positive } = 1 \text{ for AI edits; } 0 \text{ for human memes}) \\
\text{fully\_aigc} &\longrightarrow \text{provenance: fully\_aigc} \quad (\text{ai\_positive } = 1)
\end{aligned}$$

---

## 5. Offline Prefetch & Manifest Freeze Commands

### Step 1: Prefetch All Images
```bash
# Resumable atomic prefetch of all declared assets to local NVMe
uv run python scripts/data_prep/acquire_all_images.py \
  --data-root "$TECHJAM_DATA_ROOT" \
  --resume
```

### Step 2: Freeze Eligible Splits & Calibration
```bash
# Decode bytes, compute SHA-256/dHash, verify dimensions, isolate calibration, and freeze Parquet splits
uv run python scripts/data_prep/freeze_production_eligible.py \
  --data-root "$TECHJAM_DATA_ROOT" \
  --output-dir splits/production_eligible \
  --calibration-size 4096 \
  --strict \
  --verify-bytes
```

### Step 3: Hub Exfiltration
Manifests, audit reports, and source revisions are published to the public dataset repository:
```bash
uv run python scripts/upload_promoted_artifacts.py \
  --stage manifests \
  --files splits/production_eligible/* \
  --repo-id zye2/tj-data \
  --repo-type dataset
```
