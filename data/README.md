# Data

Large datasets do not belong in this Git repository. Runtime assets are managed on
fast local NVMe mounts (e.g. `/workspace/techjam26-runtime/data` on Vast or `E:/techjam26-runtime/data` locally),
configured via `.env` copied from `.env.cluster` or `.env.example`.

## 1. Production Eligible Package (`splits/production_eligible/`)

The authoritative dataset specification for the 4x RTX 4090 production pipeline is frozen into immutable Parquet manifests under `splits/production_eligible/`:

```text
splits/production_eligible/
  declared_manifest.parquet   # Full declared population across all cohorts
  train.parquet               # Eligible training split (consumed by Stage 1, Stage 2, Distillation, ATT)
  validation.parquet          # Validation split (promotion gates & threshold calibration)
  test.parquet                # In-distribution test split
  test_unseen.parquet         # Strict out-of-distribution / unseen generator test split
  calibration.parquet         # 4,096 rows strictly disjoint from every canonical split (static INT8 PTQ)
  exclusions.parquet          # Quarantined rows (corrupted, missing, placeholder, or leakage conflicts)
  audit_report.json           # Integrity audit report: counts, distributions, manifest digests
  source_revisions.json       # Pinned Hugging Face source git revisions
```

### Runtime Identity Columns
Every runtime record carries:
```text
row_id, split, image_path, provenance, ai_positive, dataset, generator, generator_family, source_image_group, duplicate_group
```

### Binary Target and Provenance Hierarchy
Canonical source-provenance labels are:
- `authentic` -> Binary target `ai_positive = 0`
- `tampered`  -> Binary target `ai_positive = 1`
- `fully_aigc` -> Binary target `ai_positive = 1`

### Data Integrity Rules
1. **Prefetch before DDP and Strict Freeze:** All images must be downloaded and verified prior to distributed training (`python scripts/data_prep/acquire_all_images.py --data-root <dir> --resume`), followed by fail-closed byte-verified manifest freezing (`python scripts/data_prep/freeze_production_eligible.py --data-root <dir> --output-dir splits/production_eligible --calibration-size 4096 --strict --verify-bytes`). No per-row remote fetching during DDP.
2. **Fail Closed:** If an image is missing or cannot be decoded, the loader raises an error immediately.
3. **No Row Substitution:** The legacy `_fallback_by_key` substitution is eliminated. Missing rows are never substituted with another image.
4. **Disjoint Calibration Split:** Exactly 4,096 calibration rows are reserved for static INT8 PTQ. These rows are strictly disjoint from train, validation, test, and test-unseen splits, and must never be used for gradient training.
5. **Transparency Package:** The dataset audit and eligible manifests are exfiltrated to the public Hugging Face dataset repo `zye2/tj-data`.
