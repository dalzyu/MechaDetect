# Backbone Bake-Off Plan

## Authoritative four-hour execution plan

Constraint: one remote instance with 8× RTX 4090 24GB for four hours. This
section overrides the larger experiment below.

The four-hour run is an elimination tournament, not an exhaustive paper-quality
study. Use one seed, the existing fixed 12k generator-balanced subset, native-resolution
inputs (equalized via gradient accumulation), and a smaller but identical evaluation suite for every candidate.

### GPU allocation

Frozen screening runs all three backbones concurrently:

~~~
PE-Spatial-G: GPUs 0–2
DINOv3 ViT-H+: GPUs 3–4
Gemma 4 vision tower: GPUs 5–6
GPU 7: evaluation/preprocessing reserve
~~~

For the adaptation final, allocate four GPUs to each of the top two models.
Consumer 4090s have limited interconnect bandwidth, so avoid communication-heavy
full-model FSDP unless profiling proves it faster.

### 0:00–0:20 — preflight

- Start from a machine image with all official weights and containers already
  cached. Downloads do not belong inside the four-hour window.
- Verify each backbone, common 512-dimensional adapter, mask alignment, BF16,
  checkpoint save/load, and one optimizer update.
- Measure exact parameter count. Reject any configuration at or above 2B.
- Pick the largest per-GPU batch that passes twice with at least 2GB headroom.

If a backbone cannot pass integration within 20 minutes, record the failure and
continue with the working candidates rather than consuming the experiment window.

### 0:20–1:10 — frozen screening

- Run all three models in parallel.
- Freeze every backbone parameter.
- Optimization: run one forward pass to extract and cache the frozen backbone tokens. Train the adapter/head directly from these cached tensors to bypass backbone computation during epochs.
- Disable the ConvNeXt/spectral expert.
- Train the identical adapter and provenance heads on the same 12k clean images.
- Use the exact same shuffled sample IDs, effective batch, update count, loss, LR,
  warmup, and seed.
- Save a final checkpoint; intermediate checkpoints are unnecessary for this short
  stage unless preemption is expected.

### 1:10–1:45 — frozen evaluation and elimination

Evaluate all three in parallel on the same fixed suite:

- 480-image strict-unseen probe.
- 400-image TechJam proxy: 200 COCO val2017 + 200 DALL·E Advanced.
- 120-image balanced single-transform grid containing every official family and
  severity.
- Shortcut probes: train a linear classifier on the frozen features to predict the source dataset and original aspect ratio. 
- No multi-transform chains.

Save every prediction for paired bootstrap analysis. Eliminate the lowest-ranked
backbone. Do not tune thresholds on these evaluation images.

### 1:45–3:10 — controlled backbone adaptation for top two

Fully unfreezing all of PE-Spatial-G is too risky and communication-heavy within
this hardware/time budget. Instead, unfreeze the final 25% of transformer blocks
for both finalists:

- Gemma: final 7 of 27 blocks.
- PE-Spatial-G: final 13 of 50 blocks.
- DINOv3 H+: final quarter of its blocks from the pinned official config.

Continue from each frozen checkpoint on the same 12k clean subset. Use a single
predeclared schedule: backbone LR 3e-6, adapter/head LR 1e-4, layer-wise LR decay
0.85, BF16, activation checkpointing, and identical update count. This measures
whether each representation adapts beneficially without attempting an expensive
full-teacher training run.
### 3:10–3:50 — adapted evaluation and winner selection
### 2:45–3:25 — adapted evaluation and winner selection

Rerun the identical evaluation suite for both finalists. Compare:

- Absolute adapted score.
- Frozen-to-adapted change.
- Clean/transform trade-off.
- Worst generator and worst single transform.
- Peak VRAM and throughput.

Use the predeclared score:

~~~
45% strict-unseen AUROC
30% TechJam proxy AUROC
15% mean single-transform AUROC
10% worst-generator AUROC
~~~

A model cannot win if its worst-generator AUROC is below 70%, it violates the 2B
limit, or adaptation materially damages clean performance.

### 3:50–4:00 — handoff

- Write the winner, runner-up, metrics, parameter count, throughput, and rationale.
- Export the winning backbone adapter/config and resolved environment.
- Do not spend this window distilling or running adversarial training.
- Immediately start the full clean-teacher run after the backbone decision on the
  compute available for final training.

### Expected outcome and limitations

This plan can produce a defensible backbone decision in four hours if weights and
environments are cached beforehand. It cannot provide three-seed confidence, full
backbone unfreezing of all candidates, or full-dataset transform evaluation. Those
are deliberately traded away to protect the September 1 delivery deadline.

## Objective

Choose the vision transformer that will become the foundation of the final
teacher and later the 100–200M distilled student.

Compare:

1. Gemma 4 vision tower only (570M parameters extracted from the 31B model), 1120-token maximum.
2. DINOv3 ViT-H+/16.
3. PE-Spatial-G/14 at native 448px.

Use PE-Spatial-G rather than PE-Core-G. Core-G is optimized for global CLIP
classification and retrieval. Spatial-G preserves stronger token-level spatial
features, which better fits localized manipulation and forensic evidence while
still supporting global pooling.

## Non-negotiable fairness rules

- Use the same immutable train, calibration, validation, strict-unseen, and
  TechJam demonstration manifests.
- Never train on COCO val2017, DALL·E Advanced, private images, or any evaluation
  derivative.
- Use identical sample order, class/generator sampler, random seeds, clean input
  render, labels, optimizer-update count, effective batch, and evaluation code.
- Disable the ConvNeXt/spectral expert during the primary bake-off so it cannot
  hide differences between ViTs.
- Project every backbone's tokens to a common 512-dimensional interface, then use
  the same learned-query AIGC head and token-aware tamper head.
- Save per-image predictions, not only aggregate metrics, for paired statistical
  comparison.
- Pin the exact model revision, license, preprocessing implementation, code
  commit, and manifest hash for every run.

## Phase 0: integration and capacity audit

For each backbone:

1. Verify official weights and tensor-key coverage.
2. Record total/trainable parameters, native token layout, patch size, supported
   input resolution, throughput, and peak VRAM.
3. Confirm the complete submission model remains below TechJam's 2B-parameter
   limit. PE-Spatial-G is close enough to the limit that this must be measured,
   not estimated.
4. Run a 16-image forward/backward/checkpoint/resume smoke test.
5. Verify token order and SID mask-to-token alignment.

Do not begin the expensive race until all three pass the same integration tests.

## Phase 1: controlled frozen-backbone race

### Input and training

- Train on the full 56,505-row leakage-safe public training pool.
- Clean downloaded images only, after metadata removal and a shared render pipeline to destroy codec shortcuts.
- Primary controlled input: native resolution for each backbone (e.g., Gemma 1120 tokens).
- Freeze every backbone parameter.
- Use gradient accumulation to guarantee identical effective batch sizes, optimizer updates, LR grid, warmup, weight decay, sampler order, and seeds.
- Run three seeds per backbone if the remote budget permits. At minimum, run one
  screening seed and repeat the top two with two additional seeds.

### Frozen evaluation

After training is locked, run the full evaluation suite once per model:

- Seen-dataset validation.
- Strict unseen-generator evaluation.
- Full evaluation-only COCO val2017 versus DALL·E Advanced proxy.
- Every organizer-listed single transform and severity.
- Shortcut probes: measure linear-probe accuracy for dataset identity and original aspect ratio.
- Chain stress test, reported separately and excluded from primary ranking.
- One-view inference for the primary comparison; three-view results are secondary.

Calibrate the binary threshold only on the calibration split. Never tune it on an
evaluation set.

## Phase 2: full-backbone fine-tuning race

Start each model from its own best frozen checkpoint.

- Unfreeze the complete vision backbone; do not use LoRA in this experiment.
- Keep the common head architecture unchanged.
- Use BF16, DDP with gradient accumulation (avoid PCIe-bound FSDP), activation checkpointing, EMA, and layer-wise LR decay.
- Use the same clean training data, sample order, update budget, and seeds.
- Preserve a frozen-checkpoint copy so the exact frozen-to-unfrozen delta is
  measurable.

Then rerun the exact full evaluation suite. Report absolute performance and the
change caused by unfreezing for each backbone.


## Metrics and decision rule

Primary metrics:

- Clean strict-unseen fully-AIGC AUROC.
- TechJam demonstration AUROC and binary balanced accuracy.
- Mean organizer single-transform AUROC.
- Worst unseen-generator AUROC.
- Worst single-transform AUROC and binary balanced accuracy.
- Clean-to-transformed performance drop.

Secondary metrics:

- Tamper AUROC and SID area-bucket results.
- Calibration error and false-positive rate at the chosen operating point.
- Throughput, latency, peak VRAM, and parameter count.
- Multi-transform chain stress results.

Internal ranking score:

~~~
40% clean strict-unseen AUROC
25% TechJam demonstration AUROC
20% mean single-transform AUROC
10% worst-generator AUROC
 5% inverse dataset-probe accuracy (anti-shortcut penalty)
~~~

Do not select a winner from the weighted score alone. A backbone is ineligible if
it violates the 2B limit, materially regresses clean performance after unfreezing,
or has a catastrophic generator/transform failure hidden by its mean.

## Required output artifacts


- Resolved config and pinned weight revision.
- Training curves and checkpoint/resume proof.
- Per-image predictions for every evaluation condition.
- Bootstrap confidence intervals and paired significance comparisons.
- Per-dataset, per-generator, and per-transform tables.
- Confusion matrices and representative errors.
- Parameter, throughput, latency, and VRAM profile.

Final deliverable: one decision document naming the winning ViT, the evidence for
and against it, the best frozen checkpoint, the best fully fine-tuned checkpoint,
and the exact configuration to carry into robust teacher training and student
distillation.
