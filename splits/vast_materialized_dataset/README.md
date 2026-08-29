---
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train.parquet
      - split: validation
        path: data/validation.parquet
      - split: test
        path: data/test.parquet
      - split: test_unseen
        path: data/test_unseen.parquet
tags:
  - image-forensics
  - multimodal-vision
pretty_name: TJ Multimodal Research Dataset
size_categories:
  - 100K<n<1M
---

# TJ Research Dataset — Canonical Manifests

Canonical consolidated multimodal research dataset.
Conforms to `data/COMBINED_DATASET_SCHEMA.md`.

## Task Target
Binary classification:
- `authentic` -> 0 (`ai_positive = 0`)
- `tampered` (human memes) -> 1 (`ai_positive = 0`)
- `tampered` (AI edits) -> 1 (`ai_positive = 1`)
- `fully_aigc` -> 2 (`ai_positive = 1`)

## Split Breakdown
- Total rows: 38964
- Train: 22483
- Validation: 8217
- Test: 5189
- Test Unseen: 3075

## Exclusions
Strictly excludes COCO val2017 and WildFake DALL-E Advanced per competition contract.
