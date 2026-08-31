# Historical Four-RTX-4090 Implementation Plan

Created: 2026-08-30
Status: retained as the design record for `orchestrate_4x4090.sh`

The maintained training entry point is now `scripts/launch_production.sh`, which
defaults to one GPU and scales accumulation for explicit DDP device lists. The
August 2026 delivery completed training and artifact upload, but skipped model
evaluation/promotion at operator request. Static INT8 artifacts remain
experimental; Atom Super float32 is the browser default. Read
`production_training_and_delivery_plan.md` for current commands and status.

## Objective

Prepare one reproducible `training/production-4x4090` branch that can execute the complete gated MechaDetect pipeline on a 4× RTX 4090 Vast instance:

1. Prefetch and validate all reasonably acquirable declared data.
2. Freeze immutable eligible train, validation, test, test-unseen, calibration, and exclusion manifests.
3. Train and promote a new Stage 1 teacher.
4. Train and promote a new Stage 2 teacher.
5. Distil the promoted teacher concurrently into independent ViT-S and ViT-B students.
6. Promote each float student independently.
7. Run independent Adversarial Transformation Training (ATT) for each promoted student.
8. Promote each ATT result independently.
9. Export selected students to ONNX opset 17 and apply calibrated static INT8 PTQ.
10. Verify desktop WebGPU and forced-WASM runtime paths.
11. Upload data records and promoted artifacts after every gate.

No downstream stage runs unless its upstream promotion gate passes.

## Locked decisions

### Compute and budget

- Vast topology: 4× RTX 4090.
- Persistent disk: 200 GB minimum; 300 GB preferred.
- Effective record batch: 48.
- Spend available Vast credit while retaining a $5 reserve.
- Project stage cost from measured smoke throughput; do not launch a stage projected to consume the reserve.
- Upload promoted artifacts immediately after every gate.

### Data

- Current declared package: 122,344 rows total and 73,751 train rows. Runtime logic must recompute counts rather than hard-code them.
- Prefetch before DDP. Do not perform per-sample remote Hugging Face scans during training.
- Acquisition is best effort. Record exact exclusions by source and reason, inspect coverage/source diversity, then freeze eligibility.
- Production loaders are local-only and fail closed.
- Never substitute a different image for a missing selected row.
- Public Nano Banana Hugging Face cohorts remain eligible.
- The explicitly forbidden local `newer image model data(do not use for training)` directory remains excluded.
- Ewan and user GPT Image 2 payloads are owner-authorized for public redistribution.
- Stage 1 and Stage 2 consume the exact same eligible training row IDs.
- PTQ uses 4,096 additional calibration-only rows disjoint from train, validation, test, and test-unseen.

### Teacher

- Run a new Stage 1 from the pinned DINOv3 pretrained backbone.
- Stage 1: one deterministic complete eligible-population pass, complete backbone frozen, downloaded-original view only, no project-added post-processing transformation.
- Run a new Stage 2 from the externally promoted new Stage 1 checkpoint with a fresh optimizer/scheduler.
- Stage 2: one deterministic complete eligible-population pass, complete backbone unfrozen, downloaded-original plus exactly one allowed post-processing transformation per pair.
- Demote Checkpoint 2 from the canonical forward pipeline.
- Save restartable checkpoints at approximately 25%, 50%, 75%, and 100% coverage.
- Select checkpoints with an external post-pass evaluator; do not stall DDP ranks for rank-0-only validation.
- Calibrate one global validation threshold maximizing balanced accuracy subject to both class recalls ≥0.82, then freeze it.
- Teacher gate: clean AUROC >0.96 and both class recalls >0.82.

### Students

- Run ViT-S on GPUs 0,1 and ViT-B on GPUs 2,3 with separate rendezvous ports, output directories, checkpoints, and logs.
- Both tracks use the same promoted teacher hash, manifest digest, row IDs, preprocessing, seeds, effective record batch, and record-draw budget.
- Each student receives two complete ordinary distillation passes.
- Float-student gate: both recalls ≥0.82, clean AUROC within 2pp of teacher, and worst-transformation AUROC within 3pp of teacher.

### ATT

- Each promoted float student receives one complete independent ATT pass.
- Candidate families: JPEG, blur, resize, noise, color adjustment, and crop.
- Single transformations only; no chains in the primary run.
- Retain downloaded-original supervised loss in every ATT update.
- Default to three deterministic candidates per row; candidate count remains configurable.
- Promote ATT only when worst-transformation AUROC improves, worst-domain AUROC does not regress, clean AUROC loss is ≤0.5pp, and both recalls remain ≥0.82.

### Export and PTQ

- Export each selected post-ATT student to ONNX opset 17.
- Apply calibrated static INT8 independently per track.
- Always produce the smallest INT8 artifact.
- If INT8 fails quality gates, label it experimental and keep the promoted float/ATT artifact as default.
- Verify the graph is static INT8, not INT4 `MatMulNBits` or dynamic-only quantization.

### Default deployment model

ViT-S remains default unless ViT-B gains at least 1pp worst-transformation AUROC or fixes a class-recall failure that ViT-S cannot keep above 0.82.

### Runtime and repositories

- Release-gating runtimes: desktop Chrome/Edge WebGPU and forced WASM fallback.
- Data records and transparency package: `zye2/tj-data`.
- Promoted models/reports: `zye2/mechadetect-models`, created private before launch.
- Delivery branch: `training/production-4x4090`.

## Phase 0 — Isolate a reproducible branch

1. Create `training/production-4x4090` without resetting unrelated user changes.
2. Classify dirty paths into production source/config, historical evidence, generated reports, model artifacts, runtime/cache, and secrets.
3. Commit only reviewed source, configs, tests, compact manifests/audits, lockfiles, launch scripts, plans, and handoff.
4. Exclude `.env`, `.runtime`, `.venv`, local outputs, staged private images, caches, and unselected binaries.
5. Record the final Git SHA in every run report.
6. Push the branch before renting Vast.

Acceptance:

- Fresh clone and `uv sync --locked --dev` succeed.
- No secret is tracked.
- Historical Iteration 1 reports remain factual.

## Phase 1 — Source-indexed prefetch and immutable eligibility

### Acquisition

Replace placeholder-oriented and training-time fetch behavior with a resumable preflight command:

```text
declared manifests → source registry → exact source/index retrieval → atomic local write → validation → eligibility
```

For each declared row:

- resolve repository, revision, split, source index, and intended image field;
- retrieve the exact source record;
- write atomically into the Vast data root;
- avoid duplicate downloads for aliases sharing a source;
- resume safely;
- record repository revision and acquisition result.

### Validation and freeze

For each acquired asset:

- decode actual bytes;
- verify dimensions and declared masks;
- reject corrupt, tiny, flat, or known-placeholder images;
- compute SHA-256 and perceptual dHash;
- normalize safe relative paths;
- reject forbidden data;
- assign deterministic row IDs;
- build exact/perceptual duplicate groups;
- quarantine cross-label conflicts;
- check split leakage and train/test-unseen generator-family separation.

Produce:

```text
splits/production_eligible/
  declared_manifest.parquet
  train.parquet
  validation.parquet
  test.parquet
  test_unseen.parquet
  calibration.parquet
  exclusions.parquet
  audit_report.json
  source_revisions.json
```

The audit records declared versus eligible counts, exclusions, distributions, duplicate/conflict counts, split membership, actual byte hashes, source revisions, and manifest digests. Placeholder `sha:<dataset>:...` values never survive into eligible manifests.

### Calibration

Acquire 4,096 additional rows whose source IDs are absent from every canonical split. Stratify across class, provenance, dataset, generator, aspect-ratio bucket, and transformation family/severity. Freeze IDs and prohibit gradient/model-selection use.

## Phase 2 — Loader and membership integrity

Files: `src/aigc_detector/dataset.py`, manifest helpers, tests.

1. Delete `_fallback_by_key` substitution.
2. Disable production runtime fetching and `allow_missing`.
3. Carry `row_id`, `split`, `generator_family`, and `duplicate_group` through runtime records.
4. Enforce stage membership:
   - Stage 1/2/distillation/ATT: train only;
   - promotion: validation only;
   - PTQ: calibration only;
   - final claims: test/test-unseen only.
5. Save manifest/population digests in checkpoints.
6. Reject resume when the manifest digest differs.

## Phase 3 — Deterministic coverage-preserving DDP

1. Replace weighted replacement sampling in primary passes with seeded no-replacement distributed shuffling.
2. Add deterministic minimal padding so all ranks execute equal microstep counts.
3. Report unique rows, padded repeats, missing rows, and per-rank draws.
4. Require complete unique-row coverage before promotion.
5. Preserve resume coverage continuity.

Four-GPU geometry:

- Stage 1: physical batch 6 × 4 GPUs × accumulation 2 = 48.
- Stage 2 primary: physical batch 2 × 4 GPUs × accumulation 6 = 48.
- Stage 2 OOM fallback: physical batch 1 × 4 GPUs × accumulation 12 = 48.

At the current declared 73,751 train rows, one pass is `ceil(73,751 / 48) = 1,537` optimizer updates. Compute from the final eligible count at launch.

## Phase 4 — Canonical teacher configs and training

### Stage 1

- Pinned DINOv3 ViT-H+/16.
- Complete backbone frozen.
- Task-specific layers only.
- Downloaded-original view only.
- One complete pass.
- No fixed 600-update cap.

### Stage 2

- Warm-start from promoted Stage 1.
- Fresh optimizer/scheduler.
- Complete backbone unfrozen.
- Low encoder LR and layer-wise decay.
- Downloaded-original plus one allowed post-processing transformation.
- Classification on both views, prediction/feature consistency, confidence-gated EMA consistency.
- Mask losses disabled.
- One complete pass.
- No fixed 1,000-update cap or Checkpoint 2 phase.

Run two-update smoke tests first. Measure memory, step time, data wait, optimizer time, EMA overhead, and throughput. Change only physical batch/accumulation on OOM; do not silently reduce the model or loss contract.

## Phase 5 — External teacher evaluation and promotion

Evaluate 25/50/75/100% checkpoints after the pass on fixed validation IDs and the complete allowed single-transform severity grid. Emit clean, mean/worst transformation, family, severity, generator, domain, and worst-domain metrics.

Promotion report contract:

```text
checkpoint_path
checkpoint_sha256
manifest_digest
calibrated_threshold
metrics
passed
failed_reasons
```

Stop before distillation if no teacher checkpoint passes.

## Phase 6 — Concurrent student distillation

Refactor `scripts/distill_student.py` and add independent configs/launchers.

- Small: `facebook/dinov3-vits16-pretrain-lvd1689m`; measure exact complete parameter count.
- Base: `facebook/dinov3-vitb16-pretrain-lvd1689m`; measure exact complete parameter count.
- GPU 0,1 small; GPU 2,3 base.
- Separate ports, outputs, logs, and checkpoints.
- Ground-truth supervision on original/transformed views, teacher soft targets, teacher feature alignment, and consistency.
- Two complete coverage passes per track.
- External validation-selected promotion; never label terminal-loss checkpoint as best.

## Phase 7 — Independent ATT

Add ATT training/evaluation scripts and small/base configs.

For every train record:

1. retain downloaded-original supervised loss;
2. deterministically generate multiple allowed single-transform candidates and severities;
3. score candidates without retaining gradients;
4. select the highest eligible loss for that student;
5. backpropagate through only the selected candidate.

Run small/base ATT concurrently on disjoint 2-GPU pools. Mine hardness independently. Never mine validation, test, unseen, or calibration rows.

## Phase 8 — Student ONNX and static INT8

1. Generalize export for selected ViT-S/ViT-B checkpoints.
2. Emit model metadata: family, exact params, quantization, threshold, input size, preprocessing version, manifest digest, evaluation status.
3. Verify PyTorch float versus ONNX float parity.
4. Calibrate static INT8 from the 4,096-row disjoint manifest.
5. Quantize supported conv/linear operators; preserve sensitive operations in float as required.
6. Verify INT8 graph structure and reject INT4/dynamic-only claims.
7. Evaluate float, ONNX float, and INT8 on identical rows/seeds.
8. Use collision-proof track/stage artifact names.

## Phase 9 — Browser runtime integration

Current browser code hard-codes Checkpoint 2, INT4 language, and threshold 0.5. Replace that with model-selection metadata.

1. Load artifact path, model identity, quantization, calibrated threshold, preprocessing, manifest digest, and status from metadata.
2. Use the calibrated threshold.
3. Preserve WebGPU-first and WASM fallback.
4. Add forced-WASM release testing.
5. Remove stale/unverified performance and Checkpoint 2 claims.
6. Benchmark size, cold/warm load, latency, throughput, peak memory, and provider compatibility.

## Phase 10 — Resumable Vast orchestration

Create a gated state machine:

```text
preflight
acquire
freeze-manifests
teacher-stage1-smoke
teacher-stage1
teacher-stage1-eval
teacher-stage2-smoke
teacher-stage2
teacher-eval
students-smoke
students-distill
students-eval
att-smoke
att
att-eval
export-float
calibrate-int8
evaluate-int8
runtime-benchmark
upload
complete
```

Preflight verifies:

- exactly four RTX 4090 GPUs;
- BF16;
- disk ≥200 GB;
- intended Git SHA;
- lockfile installation;
- HF auth and repo access;
- private model repo existence/creation;
- forbidden-data absence;
- smoke command availability.

Each stage writes started/completed markers, config/manifest/checkpoint hashes, reports, and upload receipts. Resume skips only hash-verified completed stages. Failed promotion stops downstream execution.

Budget guard:

- query Vast balance where supported;
- reserve $5;
- project the next stage from measured smoke throughput;
- fail closed when balance cannot be read unless explicit balance is supplied;
- upload before stopping.

## Phase 11 — Verification

Required focused tests:

- deterministic row IDs and acquisition;
- missing image raises, no substitution;
- atomic/resumable prefetch;
- exclusion recording;
- duplicate conflict and leakage guards;
- Stage 1/2 row-set identity;
- calibration disjointness;
- deterministic DDP coverage and equal rank lengths;
- Stage 1 freeze/no-transform contract;
- Stage 2 unfreeze/one-transform contract;
- external validation promotion;
- isolated concurrent student jobs;
- two-pass student coverage;
- deterministic hardest-candidate ATT with original loss;
- correct student ONNX checkpoint identity;
- ONNX parity and static INT8 graph verification;
- browser metadata threshold, WebGPU, and forced WASM paths.

Required smoke sequence before full launch:

1. One acquisition probe per source adapter.
2. Two-update Stage 1.
3. Two-update Stage 2.
4. Two-update ViT-S distillation.
5. Two-update ViT-B distillation.
6. Two-update ViT-S ATT.
7. Two-update ViT-B ATT.
8. One student ONNX export.
9. One static INT8 calibration/export.
10. Browser WebGPU load.
11. Browser forced-WASM load.

## Phase 12 — Documentation and branch release

Align forward-looking docs and configs with executable behavior and `CONTEXT.md`. Preserve `docs/training_run_consolidated.md` as historical evidence. Use the correct parameter identities: 840.6M backbone, 872.6M complete teacher, approximately 25.1M complete ViT-S, measured complete ViT-B count. Remove 1.88B and unverified accuracy/latency claims.

Final branch acceptance:

- fresh clone works;
- no secrets/runtime junk;
- exact commands and recovery procedure documented;
- targeted tests pass;
- all stage smoke paths pass on suitable hardware;
- exact commit SHA recorded;
- branch pushed as `training/production-4x4090`.
