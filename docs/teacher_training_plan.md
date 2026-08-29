# DINOv3 Teacher Training and Delivery Plan

## 1. Locked decisions

- **Teacher backbone:** DINOv3 ViT-H+/16,
  `facebook/dinov3-vith16plus-pretrain-lvd1689m`, pinned to revision
  `c807c9eeea853df70aec4069e6f56b28ddc82acc`.
- **Why:** the controlled bake-off is complete. DINOv3 had the best combined
  organizer-proxy performance, unseen-generator generalization, shortcut
  resistance, and training speed. The bake-off will not be rerun.
- **Teacher training:** two stages: frozen clean training, then end-to-end
  paired robustness training.
- **Challenge label:** authentic images are negative; both fully generated and
  AI-edited/tampered images are positive. The production head has one binary
  AI-positive logit; subtype labels remain metadata for audits and optional
  edit-mask localization only.
- **Student deployment:** train the future student in fp32/bf16, then apply
  post-training INT8 quantization (PTQ). Quantization-aware training is not the
  default.
- **Competition ceiling:** the DINOv3 teacher is below the Track 5 limit of
  two billion parameters.

## 2. Hardware and software contract

The production target is one Linux host with six unlocked NVIDIA CMP 170HX
cards, each exposing 40 GB of HBM2. Treat them as independent CUDA devices
connected through PCIe; do not assume NVLink. The launch path uses PyTorch
DistributedDataParallel (DDP) with the NCCL backend.

Each process owns exactly one GPU:

```text
torchrun
├── rank 0 -> cuda:0 -> logs, validation, checkpoints
├── rank 1 -> cuda:1
├── rank 2 -> cuda:2
├── rank 3 -> cuda:3
├── rank 4 -> cuda:4
└── rank 5 -> cuda:5
```

The code keeps fp32 master weights and uses bf16 autocast for forward/backward
compute. Loading the backbone itself as bf16 is intentionally disabled in the
production configs: direct bf16 parameters lose optimizer precision during
full fine-tuning.

Before the long run, verify the unlocked cards outside this repository:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.device_count()); print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]); print(torch.cuda.is_bf16_supported())"
```

Required result: six visible GPUs and bf16 support. Also verify the first
checkpoint can be written atomically to `TECHJAM_OUTPUT_ROOT`; full-stage
optimizer checkpoints are large.

## 3. Data contract

Build the production split once from the source metadata:

```bash
python scripts/data_prep/build_performance_manifests.py \
  /path/to/sid_metadata.csv \
  /path/to/wildfake_metadata.csv \
  /path/to/diffusionforensics_metadata.csv \
  --output-dir splits/performance \
  --compute-hashes
```

Expected files:

- `splits/performance/train.csv`: generator-balanced production training pool;
- `splits/performance/validation.csv`: clean model-selection split;
- `splits/performance/test_seen.csv`: held-out images from seen families;
- `splits/performance/test_unseen.csv`: generator families absent from train;
- `splits/performance/manifest-report.json`: counts, warnings, generator
  registry, and the manifest digest.

Do not use `--allow-shortfall` for the final run unless the resulting warning
is reviewed and recorded. The builder rejects duplicate groups crossing
splits and rejects the organizer demonstration data. Training repeats the
forbidden-data check at launch, including for a manually supplied manifest.

**Never train on the Track 5 demonstration subset:** COCO val2017 or DALL-E
Advanced. Those samples may be used only for organizer-aligned evaluation.

## 4. Batch and synchronization design

Both stages use an effective global batch of 48 training records:

| Stage | Records/GPU/micro-step | Accumulation | GPUs | Effective batch |
|---|---:|---:|---:|---:|
| Stage 1 | 8 | 1 | 6 | 48 |
| Stage 2 | 1 | 8 | 6 | 48 |

Formula:

$$
B_{\text{effective}} =
B_{\text{per GPU}} \times N_{\text{accumulation}} \times N_{\text{GPUs}}.
$$

In Stage 2, each record contains an original and one transformed view, so a
micro-step forwards two images per GPU. DDP gradient synchronization is
disabled with `no_sync()` for the first seven micro-steps and occurs only on
the eighth. This is important on the CMP system's narrow PCIe links.

The balanced sampler creates one deterministic global epoch, rounds it up to
an equal multiple of six, and gives each rank a distinct equal-length shard.
Unequal rank lengths are not allowed because they can deadlock DDP.

## 5. Stage 1: clean frozen-backbone training

Purpose: learn stable task-specific features before changing DINOv3.

- Keep the complete DINOv3 backbone frozen.
- Train the token adapter, provenance heads, token tamper head, spectral
  branch, and fusion gates.
- Use downloaded originals only.
- Disable transformed-view and consistency losses.
- Validate on clean images every 1,250 optimizer updates.
- Save an atomic resume checkpoint every 1,250 updates.

Launch:

```bash
torchrun --standalone --nproc-per-node=6 \
  -m aigc_detector.train \
  --config configs/teacher_dinov3_stage1_clean_frozen.yaml
```

If the six-GPU host is not yet available, the same config can be smoke-tested
on one GPU:

```bash
python -m aigc_detector.train \
  --config configs/teacher_dinov3_stage1_clean_frozen.yaml \
  --max-steps 2 \
  --stage teacher_stage1_smoke
```

Select the Stage 1 handoff checkpoint from clean validation metrics. Reject a
checkpoint if aggregate accuracy is high but balanced accuracy or macro-F1
shows class collapse.

## 6. Stage 2: end-to-end paired robustness training

Purpose: make the teacher invariant to the organizer-listed,
content-preserving transformations.

- Start task-specific weights from the selected Stage 1 checkpoint.
- Reload the same pinned pretrained DINOv3 backbone and unfreeze it fully.
- Enable activation checkpointing.
- Keep fp32 master weights; use bf16 autocast.
- Pair each original with one JPEG, blur, resize, noise, color, or crop
  transformation.
- Optimize classification on both views, prediction consistency, feature
  consistency, mask supervision, and confidence-gated EMA consistency.
- Use AdamW for the fully trainable teacher. The 40 GB available on each card
  accommodates replicated fp32 master weights, gradients, optimizer moments,
  EMA parameters, DDP buckets, and checkpointed activations.
- Start EMA consistency after update 250 so the teacher target is not the
  randomly initialized task head. EMA teacher inference uses a functional
  parameter substitution and does not allocate a temporary copy of the live
  DINOv3 weights on every micro-step.
- Use layer-wise learning-rate decay; embeddings receive the smallest encoder
  learning rate.

Launch with the selected Stage 1 checkpoint:

```bash
torchrun --standalone --nproc-per-node=6 \
  -m aigc_detector.train \
  --config configs/teacher_dinov3_stage2_paired_unfrozen.yaml \
  --initial-checkpoint /absolute/path/to/checkpoint-step-N.pt
```

Resume an interrupted run with optimizer, scheduler, EMA, epoch, micro-step,
manifest digest, and RNG state:

```bash
torchrun --standalone --nproc-per-node=6 \
  -m aigc_detector.train \
  --config configs/teacher_dinov3_stage2_paired_unfrozen.yaml \
  --resume /absolute/path/to/checkpoint-step-N.pt
```

Do not use both `--resume` and `--initial-checkpoint` for the same purpose.
`--initial-checkpoint` starts a new stage with a fresh optimizer;
`--resume` continues the interrupted stage.

## 7. Monitoring and failure rules

Rank 0 prints averaged loss components and clean validation metrics. Watch:

- total loss and original/transformed binary classification losses;
- prediction and feature consistency;
- mask focal and Dice losses;
- clean binary accuracy, balanced accuracy, F1, and AI-positive AUROC;
- GPU memory, utilization, temperature, and throttling in `nvidia-smi`;
- NCCL errors or a rank that stops progressing.

Stop and diagnose rather than continuing when:

- any loss becomes NaN or infinite;
- one rank exits or hangs;
- balanced accuracy collapses while raw accuracy rises;
- transformed performance improves by sacrificing a material amount of clean
  performance;
- a checkpoint cannot be atomically written;
- the manifest digest differs during resume.

If memory is insufficient, reduce `physical_batch_size` first and increase
`gradient_accumulation` to retain an effective batch near 48. Do not change
the encoder to bf16 master weights as a memory workaround.

## 8. Final teacher selection

Do not select the final model from in-training clean validation alone.
Evaluate every candidate checkpoint on:

1. clean validation;
2. seen-generator test data;
3. unseen-generator test data;
4. every single-transform severity in the organizer-aligned robustness grid.

Example:

```bash
python scripts/evaluate_performance.py \
  --manifest splits/performance/test_unseen.csv \
  --checkpoint /path/to/checkpoint-step-N.pt \
  --config configs/teacher_dinov3_stage2_paired_unfrozen.yaml \
  --output outputs/teacher-test-unseen-robustness.json \
  --robustness \
  --batch-size 1
```

Choose the checkpoint with the best robust AI-positive performance subject to no
material regression in clean balanced accuracy, F1, or authentic recall.
Preserve the raw JSON outputs used for the decision.

## 9. Error analysis

For the final teacher, manually inspect at least:

- 50 highest-confidence false positives;
- 50 highest-confidence false negatives;
- failures by generator family;
- failures by provenance class;
- the worst transformation family and severity;
- crop failures where semantic evidence leaves the frame;
- JPEG/resize failures that may erase synthetic frequency artifacts;
- authentic images with heavy natural processing or unusual aspect ratios.

Report counts and representative examples. State the operating-threshold
tradeoff: lowering the AI-positive threshold improves recall but increases
false accusations on authentic images.

## 10. Student plan: bf16 training, then INT8 PTQ

Start student work only after the final teacher passes the clean and robust
selection gates.

1. Train the 100–200M parameter student in fp32-master/bf16-compute mode.
2. Use binary ground-truth labels plus binary teacher soft targets on clean and
   transformed views.
3. Freeze the chosen float student checkpoint.
4. Reserve a calibration set that was not used for gradient training. It must
   cover authentic negatives, both AI-positive subtypes, generator families,
   aspect ratios, and transformation families.
5. Apply static INT8 PTQ to supported linear/convolution operators. Keep
   numerically sensitive normalization, softmax, and probability operations in
   floating point when the deployment backend requires it.
6. Benchmark the exported artifact on the actual deployment runtime, not only
   fake-quantized PyTorch.
7. Compare float and INT8 clean/robust metrics, model size, peak memory,
   throughput, and latency.
8. Accept INT8 only if metric loss remains inside the predeclared tolerance.
   If PTQ misses the tolerance, revisit calibration first; QAT is a documented
   fallback, not the default path.

## 11. Track 5 delivery checklist

The final submission is not complete until all items below exist:

- [x] Written solution, architecture, tools, model choice, libraries, and data
  protocol in the repository.
- [x] Structured source code and a directory prediction CLI that writes JSON
  entries containing `image_path` and `pred`.
- [x] Pinned teacher model below the two-billion-parameter ceiling.
- [x] Training-data guard for the forbidden organizer demo subset.
- [ ] Final clean and transformation robustness summary generated from the
  selected teacher.
- [ ] Final false-positive/false-negative analysis and threshold tradeoff.
- [ ] INT8 student artifact and float-versus-INT8 benchmark.
- [ ] Public repository URL verified from a clean clone.
- [ ] Demo video showing setup, directory inference, output JSON, robustness
  results, and limitations.
- [ ] Exact team-member contributions added to the README.

Unchecked items depend on the completed training run, named team members, or
video production; they must not be marked complete without the corresponding
artifact.
