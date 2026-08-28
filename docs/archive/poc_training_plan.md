# Performance-First Teacher PoC Execution Plan

## Summary

The local RTX 4080 run is an architecture and training-policy search, not the
final teacher. It will decide preprocessing, pooling, forensic fusion,
fine-tuning, and robustness policy before the full teacher is trained on a rented
GPU cluster.

Locked decisions:

- Train on SID, WildFake, and DiffusionForensics.
- Use a 60,000-image selected pool.
- Use hierarchical provenance outputs.
- Use SID manipulation masks as training-only supervision.
- Race Gemma-only experts against Gemma plus a lightweight spectral expert.
- Benchmark FIRE separately and fuse it only if it adds complementary signal.
- Use three-view inference for the performance-first teacher.
- Keep private data evaluation-only and deferred.
- Do not build transformation detection.
- Use organizer-aligned single-transform robustness as the primary target.
- Distil the proven robust teacher into a 100–200M parameter student.
- Do not use RL on the detector; worst-transform mining may be tested directly.

Candidate selection score:

~~~
50% strict held-out fully-AIGC AUROC
25% robustness
20% authentic-vs-tampered AUROC
5% inference efficiency
~~~

## 1. Data and leakage controls

### 60k source pool

| Dataset | Authentic | Tampered | Fully AIGC | Total |
|---|---:|---:|---:|---:|
| SID | 10,000 | 10,000 | 10,000 | 30,000 |
| WildFake | 7,500 | 0 | 7,500 | 15,000 |
| DiffusionForensics | 7,500 | 0 | 7,500 ADM | 15,000 |

Use the official releases:

- SID-Set
- WildFake
- DiffusionForensics/DIRE

Use ADM-generated DiffusionForensics images for training. Keep other
DiffusionForensics generator families for evaluation.

### Generator-family split

Build one canonical generator registry across all datasets.

- Normalize aliases such as SD-v1, stable-diffusion, and sd_1.
- Assign held-out families with SHA256("seed42:" + family) modulo 5 equals zero.
- Never use Python's process-dependent hash.
- Exclude held-out families from every training source.
- Preserve WildFake official train/test membership.
- Treat SID samples without generator metadata as sid_unknown.
- Report cross-dataset and strict unseen-generator results separately.

### Deduplication

Before sampling:

- SHA-256 exact duplicate removal.
- Perceptual-hash near-duplicate clustering.
- Source-image and tampered-derivative grouping.
- Cross-dataset duplicate search.
- Immutable train, calibration, validation, and test manifests.

No cluster or linked source family may cross a split.

### Shortcut probes

For each provenance class, measure whether frozen features predict:

- Dataset identity.
- Original codec.
- Original aspect bucket.
- Original resolution bucket.
- Canonical rendering recipe.

Record these probe scores beside every model result. Do not accept a candidate
whose provenance gain is explained by higher dataset-ID accuracy.

## 2. Preprocessing race

The fixed 1584-square JPEG95 pipeline remains a baseline because it removes the
known SID square/codec shortcut. It is not assumed to preserve the most forensic
signal.

Use the same fixed 12,000-image subset for all candidates:

- 3,000 authentic.
- 3,000 tampered.
- 6,000 fully AIGC.

All candidates strip source metadata and use identical class and dataset
balancing.

### Candidate A: square baseline

- Decode to RGB.
- Stretch to 1584 by 1584.
- Encode as JPEG quality 95.
- Use Gemma max_soft_tokens equal to 1120.

### Candidate B: aspect-preserving fixed render

- Decode to RGB.
- Preserve aspect ratio.
- Resize with a common JPEG95 rendering policy.
- Use Gemma native adaptive layout with the 1120-token ceiling.
- Balance aspect buckets across labels.

### Candidate C: aspect-preserving matched randomized render

Use the same rendering distribution for every class and dataset:

~~~
25% lossless RGB render
50% JPEG with quality sampled continuously from 75 through 100
25% WebP or an alternate JPEG implementation
~~~

Randomize bilinear, bicubic, and Lanczos interpolation. Hold out one codec and
one interpolation implementation for evaluation only.

For each candidate:

1. Run frozen Gemma on the same subset.
2. Cache FP16 token tensors sequentially.
3. Train identical lightweight heads.
4. Evaluate strict held-out generators and SID tampering.
5. Run shortcut probes.
6. Retain only metrics, hashes, and the winning cache.

Select by:

~~~
60% strict held-out fully-AIGC AUROC
25% authentic-vs-tampered AUROC
15% inverse dataset-probe accuracy
~~~

## 3. Architecture race

All candidates use the winning preprocessing and the same 12k subset.

### Candidate 0: current baseline

Gemma mean and channel-wise maximum pooling with hierarchical binary heads.
This is a reference point only.

### Candidate 1: task-specific Gemma experts

Fully-AIGC expert:

- Four learned query tokens.
- One four-head cross-attention block over Gemma tokens.
- Query outputs plus token mean and standard deviation.
- Independent projection and binary fully-AIGC logit.

Tamper expert:

- Per-token tamper logits.
- Top-five-percent token embedding average.
- Attention-weighted token embedding.
- One global learned query.
- Independent projection and binary tamper logit.

Do not force both tasks through one shared forensic trunk.

### Candidate 2: Gemma plus spectral expert

Add a small forensic branch operating at 384 to 512 pixels:

- RGB channels.
- Three fixed SRM/high-pass residual channels.
- A 32-bin radial log-FFT magnitude vector.
- Six-channel ImageNet-initialized ConvNeXt-Tiny.
- ConvNeXt output projected to 256 dimensions.
- FFT vector projected through 32 to 128 to 128.

Fuse Gemma and spectral embeddings with independent gates for the fully-AIGC
and tamper decisions.

### Candidate 3: optional FIRE late fusion

Evaluate the official FIRE checkpoint unchanged. FIRE is a diffusion-specific
binary detector; it is not a replacement for the three-class model.

Add its calibrated logit only if:

- Fusion improves mean held-out AUROC by at least 2 points, or worst-family
  AUROC by at least 3 points.
- Fusion does not materially worsen blur robustness.
- Its errors are complementary to Gemma plus the spectral expert.

Keep FIRE frozen during the local PoC.

### SID mask supervision

For SID authentic and tampered images:

- Apply the exact image geometry to masks.
- Convert masks to fractional occupancy per Gemma token.
- Train a token-level tamper head.
- Use focal BCE plus soft Dice.
- Report tamper results by manipulated-area bucket.
- Keep patch scores out of public inference.

## 4. Hierarchical output

Use two logits:

~~~
a = sigmoid(fully_AIGC_logit)
t = sigmoid(tamper_logit)

P(fully_AIGC) = a
P(tampered) = (1 - a) * t
P(authentic) = (1 - a) * (1 - t)
~~~

The probabilities must sum to one.

Public JSON:

~~~
{
  "image_path": "example.jpg",
  "pred": 0.72,
  "provenance_pred": "fully_aigc",
  "provenance": {
    "authentic": 0.08,
    "tampered": 0.20,
    "fully_aigc": 0.72
  }
}
~~~

The required scalar pred remains the fully-AIGC probability.

## 5. Fine-tuning race

Use a fixed hard and generator-balanced 12k subset with identical update budgets.

Compare:

### Frozen Gemma

Train queries, provenance heads, patch head, spectral expert, and fusion only.

### Last-four-layer SFT

- Unfreeze the final four Gemma layers.
- Encoder learning rate: 1e-5.
- Keep earlier layers frozen.

### All-layer LoRA

~~~
rank: 8
alpha: 16
dropout: 0.05
targets: q, k, v, and output attention projections
all 27 Gemma layers
base weights frozen
learning rate: 1e-4
~~~

Use the same seeds, update counts, and validation intervals.

Keep LoRA only if it improves worst-family AUROC by at least 2 points.
Otherwise prefer last-four-layer tuning.

## 6. Local RTX 4080 schedule

The local objective is decision quality within 3 to 5 days.

### Day 1: data and preprocessing

- Acquire selected public data.
- Build the 60k manifests.
- Deduplicate and assign generator holdouts.
- Produce mask-aligned derivatives.
- Run shortcut probes.
- Start all three preprocessing candidates.

### Day 2: preprocessing and Gemma heads

- Finish the preprocessing comparison.
- Select the winning rendering policy.
- Cache one 12k token set at a time.
- Train Candidate 0 and Candidate 1.

### Day 3: forensic fusion

- Train Candidate 2.
- Evaluate pretrained FIRE.
- Test Candidate 3 only if complementarity is plausible.
- Repeat the top two candidates with additional seeds.

### Day 4: adaptation race

- Compare frozen, last-four-layer, and all-layer LoRA.
- Use strict held-out generators for selection.
- Freeze the winning architecture and adaptation strategy.

### Day 5: robustness race

Use a 20k to 30k hard subset containing:

- Misclassified samples.
- Eligible SID tampered images.
- Highest-loss authentic images.
- Highest-loss fully-AIGC images.
- Generator-balanced fill.

Compare:

Recipe 1:

~~~
supervised original loss
supervised transformed loss
probability consistency
~~~

Recipe 2:

~~~
supervised original loss
supervised transformed loss
EMA clean-view teacher
confidence-gated KL consistency
supervised contrastive clean/transformed pairing
~~~

Use EMA decay 0.999 and teacher confidence threshold 0.80. Do not force raw
feature MSE across crops or severe blur.

## 7. Robustness policy

Organizer update on 28 August 2026: the evaluator will most likely apply one
transformation at a time. Multiple transformations may help generalization, but
that has not been demonstrated. Therefore the primary training and selection
protocol is:

~~~
25% clean views
75% exactly one transformation
0% multi-transformation chains
~~~

Balance single transformations across every organizer-listed family and severity.
Do not let easy JPEG90 or mild blur dominate the sampler.

Single operations:

- Random-position crop.
- Absolute resize targets 256, 512, 768, and 1024.
- Bilinear, bicubic, and Lanczos interpolation.
- JPEG quality sampled continuously from 30 through 95.
- Continuous blur, noise, color, and sharpening.

Realistic chains are a secondary ablation only:

~~~
resize -> sharpen or blur -> JPEG
crop -> resize -> JPEG
noise -> denoise or sharpen -> JPEG
screenshot-like resize -> re-encode
~~~

Run the chain-heavy recipe separately. Keep it only if it improves mean and
worst-case single-transform results without reducing clean performance. Do not
mix chain-heavy results into the primary model-selection score.

Evaluate held-out codec and interpolation implementations.

For performance-first inference, average calibrated logits over:

1. Winning primary render.
2. JPEG90 render.
3. Mild alternate-resolution render.

Report one-view and three-view results separately.

## 8. Evaluation gates

### Public selection data

- SID official validation.
- WildFake official test and strict family holdouts.
- DiffusionForensics non-ADM generators.
- Held-out rendering implementations.
- Every organizer-listed single perturbation and severity.
- Chained perturbations as a separately labelled secondary stress test.

The two private images are informational only. The unfinished private
collection is reserved for final teacher evaluation.

### Exact benchmark separation

Maintain:

1. Combined-data hackathon teacher results.
2. Exact DiffusionForensics/FIRE-protocol results.

Do not claim the combined model directly beats FIRE from non-identical training
data.

DiffusionForensics target:

~~~
average AUROC >= 95%
preferred >= 98%
~~~

Modern held-out generators:

~~~
mean AUROC >= 78%
worst-family AUROC >= 70%
~~~

Three-class provenance:

~~~
authentic recall >= 75%
authentic-vs-tampered AUROC >= 80%
macro-F1 >= 75%
~~~

Robustness:

~~~
average single-transform AUROC drop <= 10 points
average transformed binary balanced accuracy >= 80%
worst single-transform AUROC >= 80%
~~~

Report confidence intervals, per-dataset results, per-generator results,
confusion matrices, and severe-blur performance.

## 9. Cluster-teacher handoff

The local PoC must output:

- Winning preprocessing policy.
- Winning architecture.
- FIRE fusion decision.
- Winning fine-tuning method.
- Winning robustness recipe.
- Generator registry and immutable manifests.
- Resolved configs.
- Throughput and VRAM measurements.
- Learning curves.
- Checkpoint/resume verification.
- Cluster scaling estimate.

Reference cluster configuration:

~~~
8 H100 80GB or equivalent
DDP
BF16
global effective batch 256
gradient checkpointing as needed
full 60k clean pass
hard/diverse robustness pass
early stopping on held-out generators
~~~

After validating the 60k recipe, scale toward 200k balanced images only if
held-out learning curves remain unsaturated.

## 10. Student distillation and adversarial hardening

Distillation begins only after the teacher passes the clean unseen-generator and
single-transform gates. Target a 100–200M parameter student. Train it on clean and
single-transform views using:

- Ground-truth binary and hierarchical provenance losses.
- Teacher AIGC and tamper logits with temperature-scaled KL.
- Teacher forensic embeddings after learned projection.
- Clean/transformed consistency.
- SID patch supervision where available.

After ordinary distillation converges, optionally apply adversarial
worst-single-transform mining. For each image, sample valid organizer-listed
single transformations, choose the one producing the highest current loss, and
train on it. This is direct min-max training, not reinforcement learning. Keep the
adversarial stage only if it preserves clean accuracy and improves the held-out
single-transform grid.

## 11. Checkpointing and tests

Save every 100 to 250 optimizer updates and at every evaluation point.

Each checkpoint stores model or adapter weights, spectral and fusion weights,
optimizer and scheduler state, EMA state, sampler state, all RNG states,
manifest hash, preprocessing version, generator-registry version, candidate
name, and seed.

Required tests:

- Hierarchical probabilities sum to one.
- Fully-AIGC samples are excluded from tamper loss.
- SID masks align with Gemma tokens.
- Fractional mask occupancy is correct.
- The sampler matches the 50/25/25 target.
- SHA-256 holdout assignment is stable.
- No duplicate group crosses splits.
- Shortcut probes are reproducible.
- SRM and FFT branches use matched renders.
- LoRA touches only intended projections.
- EMA and confidence-gated KL behave correctly.
- Checkpoint resume reproduces the next batch and learning rate.
- One-view and three-view inference produce valid JSON.

## Assumptions

- This is a performance-first teacher PoC, not a deployment model.
- Dual experts and three-view inference are allowed.
- Private modern-generator data will be added later and remains evaluation-only.
- The local RTX 4080 is for decisions; the final teacher uses a rented cluster.
- Transformation detection, RL on the detector, domain-adversarial training, and
  public localization output remain out of scope.
- Multi-transform chains remain a secondary ablation unless they prove beneficial
  on the organizer-aligned single-transform protocol.
