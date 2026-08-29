# Combined Track 5 Dataset — Canonical Package

Canonical consolidated dataset for TechJam Track 5 (AI-Generated & Tampered Image Detection).

## Schema
Conforms to `data/COMBINED_DATASET_SCHEMA.md`.

## Task Target
Binary classification:
- `authentic` -> 0 (`ai_positive = 0`)
- `tampered` (human memes) -> 1 (`ai_positive = 0`)
- `tampered` (AI edits) -> 1 (`ai_positive = 1`)
- `fully_aigc` -> 2 (`ai_positive = 1`)

## Split Breakdown
- Total rows: 190
- Train: 103
- Validation: 19
- Test: 40
- Test Unseen: 28

## Exclusions
Strictly excludes COCO val2017 and WildFake DALL-E Advanced per competition contract.
