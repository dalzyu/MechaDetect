# Robust Image Provenance Detection

This project classifies an image's provenance while remaining robust to
content-preserving post-processing and redistribution.

## Language

**Authentic image**:
An image captured from a real-world scene with no deliberate content-level manipulation. Ordinary post-processing does not change this provenance class.
_Avoid_: Real image, clean image

**Tampered image**:
An authentic or mixed-origin image whose semantic content was deliberately added, removed, replaced, or generatively edited. Content-preserving redistribution alone is not tampering.
_Avoid_: Edited image, transformed image, fake image

**Fully AIGC image**:
An image whose visual content was generated end-to-end by a generative model, even if it was subsequently post-processed.
_Avoid_: AI image, synthetic image, fake image

**Downloaded original**:
The dataset file before this project's transformation pipeline modifies it. It is not assumed to be pristine or free from earlier post-processing.
_Avoid_: Clean image, raw image

**Post-processing transformation**:
A content-preserving operation such as JPEG compression, blur, resizing, noise, color adjustment, or cropping. It can be applied to any provenance class without changing that class.
_Avoid_: Tampering, editing

## Organizer evaluation protocol

As of 28 August 2026, the TechJam organizer expects evaluation to most likely
apply one post-processing transformation at a time. Primary model selection uses
clean images and the complete single-transform severity grid. Multi-transform
chains are a separately reported research ablation and must not replace the
single-transform benchmark.
