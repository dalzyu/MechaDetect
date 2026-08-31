# Static INT8 Release Evaluation

## Release decision

The current Static INT8 ONNX artifacts are **not included in the public MechaDetect release or browser model selector**. They reduced storage and improved CPU throughput, but they did not preserve detection quality. The Float32 ONNX artifacts remain the supported deployment models.

This is not an inherent failure of INT8. It is a failure of this specific post-training quantization policy and quality-gating path.

## Evaluation setup

All four Float32 student models and their matching Static INT8 exports were evaluated on the organizer demonstration set:

- 13,841 total images
- 4,998 authentic COCO val2017 images
- 8,843 AIGC WildFake DALL-E Advanced images
- 15 conditions per model: clean, four JPEG levels, three blur levels, two resize levels, three noise levels, color jitter, and center crop
- 207,615 image-condition samples per model
- 1,660,920 total model-image evaluations across eight artifacts

Float32 and INT8 pairs used the same source checkpoint, input contract, preprocessing, output contract, and evaluation records. The input was normalized Float32 RGB in `[batch, 3, 224, 224]`; the output was `[P(authentic), P(AIGC)]`.

## Detection-quality results

| Model | Format | Clean AUROC | Mean transformed AUROC | Worst transformed AUROC | Worst condition | Clean AIGC recall | Clean authentic recall |
| :--- | :--- | ---: | ---: | ---: | :--- | ---: | ---: |
| Atom Normal | Float32 | 0.9921 | 0.9871 | 0.9691 | `resize_quarter` | 97.60% | 92.60% |
| Atom Normal | Static INT8 | 0.5942 | 0.5422 | 0.3785 | `resize_quarter` | 76.44% | 34.73% |
| Atom Super | Float32 | 0.9947 | 0.9931 | 0.9870 | `resize_quarter` | 99.39% | 83.77% |
| Atom Super | Static INT8 | 0.7141 | 0.6793 | 0.5380 | `resize_quarter` | 87.55% | 35.57% |
| Quark Normal | Float32 | 0.9973 | 0.9945 | 0.9876 | `resize_half` | 98.94% | 94.82% |
| Quark Normal | Static INT8 | 0.6780 | 0.6735 | 0.6200 | `resize_half` | 38.81% | 84.13% |
| Quark Super | Float32 | 0.9980 | 0.9967 | 0.9928 | `resize_quarter` | 99.66% | 93.86% |
| Quark Super | Static INT8 | 0.7125 | 0.7138 | 0.6644 | `resize_half` | 63.12% | 67.07% |

The quality loss is systematic across architecture size, training scope, and image transformation. It is not an isolated threshold problem: AUROC measures ranking quality independently of the decision threshold.

### Per-condition AUROC

| Model | clean | jpeg90 | jpeg70 | jpeg50 | jpeg30 | blur0.5 | blur1.0 | blur2.0 | resize_half | resize_quarter | noise0.02 | noise0.05 | noise0.10 | color_jitter20 | crop80 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Atom Normal Float32 | 0.9921 | 0.9939 | 0.9963 | 0.9960 | 0.9950 | 0.9911 | 0.9826 | 0.9763 | 0.9786 | 0.9691 | 0.9912 | 0.9889 | 0.9803 | 0.9919 | 0.9885 |
| Atom Normal INT8 | 0.5942 | 0.5929 | 0.5812 | 0.5703 | 0.5311 | 0.5823 | 0.5474 | 0.4699 | 0.5423 | 0.3785 | 0.5849 | 0.5437 | 0.4902 | 0.5905 | 0.5860 |
| Atom Super Float32 | 0.9947 | 0.9955 | 0.9968 | 0.9967 | 0.9963 | 0.9947 | 0.9922 | 0.9887 | 0.9907 | 0.9870 | 0.9943 | 0.9928 | 0.9885 | 0.9947 | 0.9942 |
| Atom Super INT8 | 0.7141 | 0.7122 | 0.7089 | 0.7003 | 0.6711 | 0.7107 | 0.6944 | 0.6216 | 0.6890 | 0.5380 | 0.7085 | 0.6798 | 0.6549 | 0.7107 | 0.7098 |
| Quark Normal Float32 | 0.9973 | 0.9978 | 0.9992 | 0.9991 | 0.9988 | 0.9977 | 0.9935 | 0.9901 | 0.9876 | 0.9884 | 0.9965 | 0.9938 | 0.9884 | 0.9972 | 0.9954 |
| Quark Normal INT8 | 0.6780 | 0.6841 | 0.6804 | 0.6902 | 0.6858 | 0.6605 | 0.6251 | 0.6633 | 0.6200 | 0.6239 | 0.6759 | 0.7037 | 0.7699 | 0.6756 | 0.6704 |
| Quark Super Float32 | 0.9980 | 0.9976 | 0.9986 | 0.9989 | 0.9989 | 0.9979 | 0.9965 | 0.9958 | 0.9939 | 0.9928 | 0.9975 | 0.9961 | 0.9940 | 0.9979 | 0.9980 |
| Quark Super INT8 | 0.7125 | 0.7165 | 0.7202 | 0.7140 | 0.7206 | 0.6928 | 0.6685 | 0.7394 | 0.6644 | 0.6675 | 0.7210 | 0.7343 | 0.8160 | 0.7134 | 0.7046 |

## Storage and runtime results

These measurements used ONNX Runtime 1.29.0. GPU results used `CUDAExecutionProvider` on an NVIDIA GeForce RTX 4080 with batch size 64. CPU results used `CPUExecutionProvider` on an Intel Core i9-13900KF with 16 intra-op threads and batch size 32. Batch-1 latency excludes image decoding, resizing, browser upload, and UI scheduling.

| Model | Format | File size | GPU p50 batch 1 | GPU throughput batch 64 | CPU p50 batch 1 | CPU throughput batch 32 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Atom Normal | Float32 | 96.1 MB | 6.32 ms | 1,495.2 img/s | 23.01 ms | 69.3 img/s |
| Atom Normal | Static INT8 | 51.1 MB | 5.65 ms | 1,265.6 img/s | 21.46 ms | 77.2 img/s |
| Atom Super | Float32 | 96.1 MB | 6.50 ms | 1,554.6 img/s | 22.07 ms | 72.3 img/s |
| Atom Super | Static INT8 | 51.1 MB | 5.53 ms | 1,266.4 img/s | 18.88 ms | 79.6 img/s |
| Quark Normal | Float32 | 341.3 MB | 6.95 ms | 538.9 img/s | 48.16 ms | 22.7 img/s |
| Quark Normal | Static INT8 | 173.3 MB | 6.31 ms | 460.5 img/s | 38.46 ms | 28.9 img/s |
| Quark Super | Float32 | 341.3 MB | 7.00 ms | 534.7 img/s | 49.26 ms | 22.5 img/s |
| Quark Super | Static INT8 | 173.3 MB | 6.41 ms | 456.9 img/s | 38.67 ms | 28.7 img/s |

INT8 achieved its storage goal and improved CPU throughput, but it reduced batched CUDA throughput on this runtime. These speed results used the CUDA execution provider, not TensorRT. Provider-specific speed does not explain the accuracy loss: the same numerical failure reproduces on CPU.

## Root cause

### What was quantized

The Static INT8 exporter used ONNX Runtime post-training static quantization:

- QDQ graph format
- signed INT8 activations and weights
- default per-tensor activation quantization
- default MinMax calibration
- `MatMul`, `Gemm`, and `Conv` selected for quantization

Each matching Float32 graph has 2,424 nodes. Each INT8 graph retains those nodes and adds 56 `QuantizeLinear` plus 87 `DequantizeLinear` nodes. The checkpoint identities, graph inputs, graph outputs, and non-quantized initializers match between each Float32/INT8 pair.

The policy quantized all 24 transformer MLP matrix multiplications, all 12 exported GELU activation boundaries, the patch convolution, the token adapter, and both feature projections. QDQ was also inserted immediately after 15 LayerNorm outputs. The final classifier, sigmoid, and probability output remained Float32.

### Why the ranges failed

Transformer MLP activations had highly asymmetric distributions and large positive outliers. Per-tensor MinMax calibration assigned one scale to each entire activation tensor. This made the quantization step too coarse around the dense near-zero region and clipped negative GELU values.

Measured examples from real organizer images:

- Atom Normal layer 2 GELU boundary: scale 0.9094, zero-point -128; approximately 75.0% of observed values were negative and clipped.
- Atom Normal layer 6 GELU boundary: scale 1.1793, zero-point -128; approximately 87.9% were negative and clipped.
- Quark Normal layer 2 GELU boundary: scale 5.0271, zero-point -128; approximately 99.1% were negative and clipped.
- Quark layer 2 MLP up/down boundaries used scales 8.306 and 20.888 while typical runtime signal RMS was approximately 1.15.

Every individual QDQ operation was structurally legal. The problem was repeated loss of activation information through residual transformer blocks. Small local errors accumulated into large feature and final-score changes.

### Isolation experiment

An in-memory graph ablation preserved the quantized weights but bypassed selected activation QDQ boundaries:

- Atom Normal original INT8 versus Float32 on eight real images: mean absolute probability difference 0.237; maximum 0.860.
- Bypassing only head QDQ: effectively unchanged; the failure remained.
- Bypassing the 48 encoder MLP activation QDQ boundaries: mean absolute difference fell to 0.0149; maximum 0.0618.
- Quark Normal included a positive image whose Float32 score was 0.9940 and INT8 score was 0.0181. Bypassing the MLP activation QDQ boundaries restored it to 0.9941.

This identifies repeated encoder MLP activation quantization as the dominant failure. The final classifier, CUDA provider, source checkpoint identity, and shared image preprocessing are ruled out as primary causes.

## Why PTQ versus QAT matters

We used **post-training quantization (PTQ)**. PTQ observes a calibration sample after training, chooses quantization ranges, and converts the trained graph without changing learned weights. It is fast and inexpensive, but the model never learns to tolerate the resulting rounding and clipping.

**Quantization-aware training (QAT)** simulates quantization during fine-tuning. The model can adjust weights and activation distributions around quantization noise. That makes QAT a strong candidate for transformer blocks whose activation ranges are difficult to calibrate after training.

However, “PTQ bad, QAT good” is too broad:

- Conservative mixed-precision PTQ may work if encoder MLP activations remain Float32.
- Per-channel weight quantization and percentile or distribution-aware activation calibration may reduce error.
- Weight-only quantization could reduce storage without repeatedly quantizing residual activations.
- QAT can still fail if the quantization policy or target runtime is unsuitable.

The immediate failure was the chosen **per-tensor MinMax activation PTQ policy across every transformer MLP**, not PTQ as a category.

## Why the artifacts were withdrawn

The release quality gate allows at most 0.005 AUROC loss relative to Float32. The observed clean AUROC losses were approximately 0.28–0.40, far beyond that limit. Smaller files and modest CPU speed gains do not compensate for a detector that no longer ranks authentic and AIGC images reliably.

The current INT8 files are therefore excluded from:

- the public browser selector;
- the supported model inventory;
- performance or accuracy claims for the release;
- automatic promotion or packaging as usable artifacts.

## Requirements for a future INT8 release

A replacement INT8 artifact must complete all of the following:

1. Start with a conservative mixed-precision policy: keep encoder MLP activation boundaries, LayerNorm outputs, token adapter, feature projections, and final classifier in Float32.
2. Quantize only empirically safe operations, then widen coverage one group at a time.
3. Evaluate per-channel weights, percentile/distribution-aware calibration, weight-only quantization, and QAT rather than assuming one method will work.
4. Make the graph verifier reject QDQ placement after protected normalization and projection outputs.
5. Apply the calibration manifest’s intended transform distribution rather than silently reading every calibration image as clean.
6. Require Float32-versus-INT8 numerical parity on real images before full evaluation.
7. Require no more than 0.005 AUROC loss on the complete 13,841-image organizer benchmark, including all 15 conditions.
8. Benchmark the exact release provider separately: WebGPU, WebAssembly, CPU, CUDA, or TensorRT. A provider name alone is not evidence of acceleration.
9. Publish only after accuracy, graph integrity, storage, latency, and throughput gates all pass.

Until those conditions are met, Float32 is the supported MechaDetect deployment format.
