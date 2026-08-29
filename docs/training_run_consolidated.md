# Teacher Training Run Consolidation

**Project:** TechJam 2026 Track 5 — AI-generated and tampered-image detection  
**Run:** Vast instance `49060153`, six NVIDIA CMP 170HX devices  
**Run window:** 2026-08-28/29 (UTC+08)  
**Status:** Teacher Iteration 1 complete; student distillation intentionally blocked; Vast instance stopped after artifact recovery.

This document is the factual record of the training run. It supersedes optimistic planned counts where the executed materialized data, configuration, or audit protocol differed. The forward-looking corrective plan remains in [`production_training_and_delivery_plan.md`](production_training_and_delivery_plan.md).

## 1. Executive outcome

The run produced three usable teacher states:

1. **Stage 1 frozen anchor**, update 600.
2. **Stage 2 audit teacher**, update 1,250, initialized from Stage 1 and trained with paired transformations.
3. **Checkpoint 2 production adaptation**, update 250, initialized from Stage 2 and trained on the full *available executable* pool.

The Stage 2 teacher retained useful ranking performance under common corruptions, but its default 0.5 operating point under-recalled AI-positive images. The audit did not justify distillation. Stage 2 validation was disabled after a distributed validation/NCCL watchdog failure, so no checkpoint was selected by a live validation metric. Checkpoint 2 was therefore retained as an experimental production adaptation, not promoted as a deployment teacher.

**No ViT-S, ViT-B, or ConvNeXt student was trained.**

## 2. Model definition

All teacher states use the same model family:

- **Backbone:** DINOv3 ViT-H+/16.
- **Hugging Face identifier:** `facebook/dinov3-vith16plus-pretrain-lvd1689m`.
- **Pinned revision:** `c807c9eeea853df70aec4069e6f56b28ddc82acc`.
- **Backbone parameters:** approximately 840.6M.
- **Input:** 224 px square RGB render under `square_jpeg95` preprocessing.
- **Backbone representation:** 1,280-dimensional patch tokens, 196 tokens at 224 px.
- **Task path:** token adapter, global AI-evidence head, token-level edit-localization head, spectral expert, and shared binary AI-positive classifier.
- **Target:** `ai_positive = 0` for authentic/human-created negatives and `ai_positive = 1` for fully generated or AI-edited/tampered images. Subtype labels remain metadata.
- **Training precision:** fp32 master weights with BF16 autocast on the CMP devices.
- **Distributed execution:** one process per GPU, six-rank PyTorch DDP/NCCL.

The frozen DINOv3 backbone was selected by the earlier backbone bake-off. See [`backbone_bakeoff_decision.md`](backbone_bakeoff_decision.md) and [`backbone_bakeoff_findings.md`](backbone_bakeoff_findings.md).

## 3. Data actually used

### 3.1 Planned versus executable data

The project plan described `zye2/tj-data` as **127,902 metadata rows across 38 cohorts**. That repository contains the canonical metadata, but its image payload coverage was not sufficient to execute a 127,902-image training run on the rented host.

The repository also contains a 123,252-row candidate manifest at [`splits/checkpoint2_materialized/audit_report.json`](../splits/checkpoint2_materialized/audit_report.json). That report records the post-purge metadata candidate after removing four source cohorts, not the exact strict training pool used by the audit teacher or the remote production adaptation.

The strict conflict-free executable audit materialization was:

| Split | Rows |
|---|---:|
| Train | 22,483 |
| Validation | 8,217 |
| Test | 5,189 |
| Test unseen | 3,075 |
| **Total** | **38,964** |

Its audit report is [`splits/vast_materialized_dataset/audit_report.json`](../splits/vast_materialized_dataset/audit_report.json). The Stage 1 and Stage 2 audit teacher used the **22,483-row train split**.

The complete Gmorinan archive contributed 2,300 authentic meme rows. After duplicate/conflict filtering and union with the strict materialized pool, the remote Checkpoint 2 adaptation used **41,035 executable rows**, with:

- **23,324 AI-positive rows**;
- **17,711 AI-negative rows**;
- **5 source dataset families**;
- **0 missing image paths in the runtime preflight**.

This is the full available executable pool for this run. It is **not** the planned 127,902-image dataset. The raw archives remain local reference material; they were not uploaded to the repository because of size, licensing, and provenance concerns.

### 3.2 Data controls

The builder and runtime checks enforced:

- forbidden organizer demonstration filtering;
- duplicate-group and conflict handling;
- materialized-path requirements;
- explicit `ai_positive` labels rather than inferring the target from the broad provenance subtype;
- deterministic split manifests and manifest digests where available.

An important limitation remains: the executed unseen audit contained AI-positive rows only. It cannot produce an AUROC, specificity, balanced accuracy, or authentic-recall estimate without disjoint AI-negative rows.

## 4. Hardware and runtime

| Property | Executed value |
|---|---|
| Instance | Vast `49060153` |
| GPUs | 6 × NVIDIA CMP 170HX, 40 GB HBM2 each |
| Interconnect | PCIe; no NVLink |
| DDP backend | NCCL |
| Disk | 128 GB class local disk |
| Instance rate | approximately `$1.47556/hour` |
| Final state | `actual_status=exited`, `cur_state=stopped` |

The training jobs were stopped before teardown. Only file transfer remained during artifact recovery. The instance was stopped immediately after hashes were verified.

## 5. Stage 1 — clean frozen anchor

**Configuration:** [`configs/teacher_dinov3_stage1_clean_frozen.yaml`](../configs/teacher_dinov3_stage1_clean_frozen.yaml)

### Contract

- DINOv3 backbone frozen.
- Train token adapter, provenance heads, tamper head, spectral branch, and fusion gates.
- Clean originals only; no transformed view.
- Physical batch 8 × accumulation 1 × 6 ranks = **48 effective records/update**.
- Maximum 600 updates.
- Validation sample: 1,024 rows.
- Generator-balanced sampling enabled.

### Executed result

- Completed: **600/600 updates**.
- Training split: **22,483 rows**.
- Validation probe: **1,024 rows**.
- Best checkpoint metadata reports `step=600`.
- Checkpoint manifest digest: `a99b064a47158427499951f347a77dc93d6d60fca9e77159880825c0e561fe78`.
- Full resume checkpoint local SHA-256: `c191edcb1da3c9ae1a199cf5c7732d6304a1d87a19b1d977764d302055175301`.
- Compact model-only publication weights are under [`models/teachers/iteration1/stage1/`](../models/teachers/iteration1/stage1/).

The recorded Stage 1 validation result was:

| Metric | Value |
|---|---:|
| Accuracy | 0.9062 |
| Balanced accuracy | 0.9140 |
| Macro-F1 | 0.9026 |
| AI-positive AUROC | 0.9798 |
| AI-positive recall | 0.9456 |
| AI-negative/authentic recall | 0.8824 |
| Robust-probe mean AUROC | 0.9795 |

These metrics came from a different sampled protocol than the later Stage 2 audit and must not be treated as a controlled paired comparison.

## 6. Stage 2 — paired robustness teacher

**Configuration:** [`configs/teacher_dinov3_stage2_paired_unfrozen.yaml`](../configs/teacher_dinov3_stage2_paired_unfrozen.yaml)

### Contract

- Initialize task weights from Stage 1.
- Unfreeze the DINOv3 backbone.
- Layer-wise learning-rate decay: 0.85.
- Pair each image with one JPEG, blur, resize, noise, color, or crop transformation.
- Train original and transformed classification, prediction consistency, feature consistency, mask losses, and confidence-gated EMA consistency.
- Physical batch 2 × accumulation 4 × 6 ranks = **48 effective records/update**.
- Maximum 1,250 updates.
- EMA enabled after update 250.

### Executed result

- Completed: **1,250/1,250 updates**.
- Initial checkpoint: Stage 1 update 600.
- Full final audit checkpoint local SHA-256: `6c363d57feb82e5643ee6cf21b2312d5bda261f129b30cc0168679de70dba4b0`.
- Full checkpoint size: **13,963,071,401 bytes**.
- Compact model-only publication weights are under [`models/teachers/iteration1/stage2/`](../models/teachers/iteration1/stage2/).
- Audit JSON reports are under [`artifacts/vast_49060153/final/`](../artifacts/vast_49060153/final/).

### Audit metrics

The audit used limited, deduplicated probes with approximately 300 rows for validation and test, and 115 AI-positive rows for unseen. These are preliminary audit samples, not full split metrics.

| Probe | Rows | Clean AUROC | AI-positive recall @ 0.5 | AI-negative/authentic recall @ 0.5 |
|---|---:|---:|---:|---:|
| Validation audit | 300 | 0.9577 | 0.7411 | 0.9709 |
| Test audit | 300 | 0.9590 | 0.7754 | 0.9469 |
| Unseen positive-only | 115 | undefined | 0.9304 | undefined |

The test confusion matrix for clean, JPEG-50, blur-1, resize-half, and noise-0.02 was `[[107, 6], [42, 145]]` or an equivalent class-count arrangement. The AI-negative/authentic recall is therefore:

\[
\frac{107}{107 + 6} = \frac{107}{113} = 0.9469026549.
\]

The repeated 94.69% value is a real equal error count across those conditions, not a hard-coded evaluator value. Crop-80 had `[[109, 4], [39, 148]]`, giving 96.46% AI-negative/authentic recall.

The complete measured robustness values are preserved in:

- [`stage2_final_validation_robustness.json`](../artifacts/vast_49060153/final/stage2_final_validation_robustness.json)
- [`stage2_final_test_robustness.json`](../artifacts/vast_49060153/final/stage2_final_test_robustness.json)
- [`stage2_final_unseen_robustness.json`](../artifacts/vast_49060153/final/stage2_final_unseen_robustness.json)

### Interpretation

- Ranking remained useful: clean test AUROC was 0.9590.
- Corruption AUROC degradation was modest: JPEG-50 0.9544, blur-1 0.9583, resize-half 0.9586, noise-0.02 0.9551.
- Thresholded AI-positive recall was the primary weakness: 0.7411 on the sampled validation audit and 0.7754 on the sampled test audit.
- AI-negative/authentic recall was comparatively high, indicating an operating-point shift toward predicting the negative class.
- Stage 1 and Stage 2 probes were not frozen and identical, so the magnitude of the apparent regression is not a controlled causal comparison.
- The positive-only unseen split cannot establish unseen discrimination quality by itself.

## 7. Checkpoint 2 — production adaptation

**Configuration:** [`configs/teacher_dinov3_checkpoint2_full_data.yaml`](../configs/teacher_dinov3_checkpoint2_full_data.yaml)

### Intended contract

Checkpoint 2 was intended to start from the accepted Stage 2 teacher, use a lower learning rate, and run a controlled 0.20\(N^*\) adaptation on the full verified available pool. It was not eligible for deployment because the Stage 2 checkpoint had not passed the Iteration 1 audit gate.

### Operational execution

The first launch exposed a configuration-placement bug: `initial_checkpoint` was placed under `training`, while the trainer reads it under `paths`. The resulting random-start run began with loss approximately 1.446 and was discarded after two updates.

The corrected launch used:

- Warm start from Stage 2 `checkpoint-best.pt`.
- Runtime pool: 41,035 available executable rows.
- Physical batch 4 × accumulation 2 × 6 ranks = **48 effective records/update**.
- Encoder LR: `1.0e-6`.
- Head LR: `3.0e-5`.
- One epoch / maximum 250 updates.
- Paired transformation losses and EMA enabled.
- Live validation disabled; this was an adaptation run, not a model-selection run.

### Executed result

- Completed: **250/250 updates**.
- Final checkpoint local and remote SHA-256: `3e8875b97d5a60eda217f27b5a8dc454ec44c4ff976caebc8c3f1f0256d655fd`.
- Full checkpoint size: **13,963,064,729 bytes**.
- Compact model-only publication weights are under [`models/teachers/iteration1/checkpoint2/`](../models/teachers/iteration1/checkpoint2/).
- Status: **experimental; not promoted for distillation**.

No post-adaptation audit was run before teardown. The checkpoint must not be described as a validated production model.

## 8. Mistakes, failures, and recovery actions

### Training and distributed execution

1. **DDP unused-parameter failure.** Some model branches were inactive for particular batches. The run required `find_unused_parameters=True` to avoid distributed reduction failure.
2. **Rank-0-only validation watchdog failure.** Rank 0 spent too long validating while other ranks waited, triggering the NCCL watchdog. Stage 2 live validation was disabled rather than redesigned during this run.
3. **No true Stage 2 best checkpoint.** Because validation was disabled, `checkpoint-best.pt` is a final-state fallback, not a validation-selected optimum.
4. **Resume/loader geometry mismatch.** Changing physical batch size altered loader micro-step counts while retaining saved epoch/micro-step offsets. Two resumes made no progress because the saved offset exceeded the new loader length.
5. **Configuration key mismatch.** `gradient_accumulation_steps` was introduced but the trainer reads `gradient_accumulation`; one discarded run used effective batch 96 instead of the intended 48.
6. **Initial-checkpoint placement error.** Checkpoint 2 initially started from random task weights because the key was under the wrong configuration section. A first-loss smoke check caught and discarded it.

### Storage and utilization

7. **Checkpoint disk exhaustion.** Full resumable checkpoints were approximately 13.96 GB. Saving every 100 updates filled the 128 GB disk at step 800. Older checkpoints were deleted only after local copies were verified, and checkpoint intervals were increased.
8. **Low average GPU duty cycle.** GPU compute bursts reached high SM utilization, but average power commonly remained around 50–60 W. CPU/Pillow transforms, `AutoImageProcessor`, Python per-image head work, and PCIe synchronization dominated the gaps.
9. **Mask objective was inactive.** `mask_focal` and `mask_dice` remained zero in the observed Stage 2 logs. The next run must guarantee mask-bearing batches or disable those losses explicitly.

### Evaluation and data accounting

10. **Path separator bug.** Windows manifest backslashes caused Linux evaluation file-not-found errors. The evaluator now normalizes backslashes to forward slashes.
11. **Mixed evaluation state risk.** Resume metadata initially omitted checkpoint/config/condition identity. The evaluator now records manifest, checkpoint, config, conditions, three-view mode, and rows.
12. **Sample protocol changed across stages.** Stage 1 and Stage 2 used different sampled probes. The next audit must freeze row IDs and reuse them for every comparison.
13. **Positive-only unseen audit.** The unseen set had no AI-negative rows, so AUROC and specificity were undefined.
14. **Canonical-versus-executable count confusion.** Metadata row counts, candidate manifests, strict conflict-free materialization, and the final runtime pool were different populations. All future reports must state each count and never call 41,035 rows “the full 127,902-row dataset.”

## 9. Notable weights and publication layout

The repository distinguishes **full resume checkpoints** from **model-only publication weights**:

- Full `.pt` checkpoints retain optimizer, scheduler, RNG, EMA, and training metadata. They are large and remain in the ignored local artifact store.
- `.safetensors` files contain only model tensors converted to FP16 for practical publication and deployment transfer. They do not support optimizer resume and are not claimed to be byte-identical to the full FP32 resume checkpoint.
- Each publication weight has a JSON sidecar containing source path, source SHA-256, source size, update number, manifest digest, dtype, and tensor names.

| Model state | Full local checkpoint | Published model-only weights | Role |
|---|---|---|---|
| Stage 1 update 600 | `artifacts/vast_49060153/checkpoints/checkpoint-best.pt` | `models/teachers/iteration1/stage1/model-weights.safetensors` | Frozen task-specific anchor |
| Stage 2 update 1,250 | `artifacts/vast_49060153/final/checkpoint-step-1250.pt` | `models/teachers/iteration1/stage2/model-weights.safetensors` | Audited paired-robustness teacher; not accepted for distillation |
| Checkpoint 2 update 250 | `artifacts/vast_49060153/final/checkpoint-step-250.pt` | `models/teachers/iteration1/checkpoint2/model-weights.safetensors` | Experimental full-available-pool adaptation |

Published file details:

| Publication file | Size | SHA-256 |
|---|---:|---|
| `stage1/model-weights.safetensors` | 64,051,374 bytes | `dd067ad34bc654a798a2651cc37a676d88b02c556eab4dc8d76dd5fb5f37b397` |
| `stage2/model-weights.safetensors` | 1,745,317,942 bytes | `0e83234bdd1c407dea1e794124b5b6c8dc3807b5325a9daddb8035d31ada7a3f` |
| `checkpoint2/model-weights.safetensors` | 1,745,317,942 bytes | `74a6752a790ec196ec72b722d565a588da19a6a456b5b52baba696f95758053f` |

The intermediate Stage 2 checkpoints at updates 700 and 1,000 were pruned from the local artifact cache after the final checkpoint was copied and hash-verified. They were not publication candidates.

## 10. Repository organization

Training code, configs, data builders, audits, and tests are tracked in their existing domain directories. The new canonical weight layout is:

```text
models/
└── teachers/
    └── iteration1/
        ├── stage1/
        │   ├── model-weights.safetensors
        │   └── model-weights.json
        ├── stage2/
        │   ├── model-weights.safetensors
        │   └── model-weights.json
        └── checkpoint2/
            ├── model-weights.safetensors
            └── model-weights.json
```

Large model-only files use Git LFS. Raw archives, generated image payloads, full resume checkpoints, caches, and private runtime data remain excluded from ordinary Git history. The compact model-only files are the weights intended to travel with the repository.

## 11. Next run gate

Student distillation remains blocked until Teacher Iteration 2 satisfies the promotion contract in [`production_training_and_delivery_plan.md`](production_training_and_delivery_plan.md):

- materialize and hash the actual intended payloads;
- freeze identical validation and test row IDs;
- add disjoint authentic negatives to the unseen audit;
- reject unknown or misplaced config keys;
- prove warm-start identity before update 1;
- prove effective batch and save/resume round-trip with the exact loader geometry;
- use progressive unfreezing and live validation every 50 updates;
- require AI-positive recall ≥ 0.85, AI-negative/authentic recall ≥ 0.85, and clean AUROC within 0.005 of the best checkpoint;
- select by robust mean AUROC subject to those guards;
- enable mask losses only with measured mask coverage;
- complete a post-adaptation regression audit before any student job.

Until those conditions are met, the Stage 1 and Stage 2 weights are research artifacts, not validated deployment models.
