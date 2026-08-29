#!/usr/bin/env bash
# cluster_setup.sh — run once on a fresh instance before training.
# Usage: bash cluster_setup.sh
# Assumes: repo is cloned, .env is populated, uv is installed.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "=== Loading environment ==="
if [[ ! -f .env ]]; then
  echo "ERROR: .env not found. Copy .env.cluster to .env and fill in HF_TOKEN."
  exit 1
fi
set -a; source .env; set +a

echo "=== Creating runtime directories ==="
mkdir -p "$TECHJAM_DATA_ROOT"
mkdir -p "$TECHJAM_HF_HOME"
mkdir -p "$TECHJAM_OUTPUT_ROOT"

echo "=== Installing Python dependencies ==="
uv sync --locked --dev

echo "=== Verifying GPU access ==="
uv run python -c "
import torch
n = torch.cuda.device_count()
print(f'GPUs visible: {n}')
for i in range(n):
    print(f'  cuda:{i} — {torch.cuda.get_device_name(i)}')
bf16 = torch.cuda.is_bf16_supported()
print(f'BF16 supported: {bf16}')
if n < 6:
    print('WARNING: expected 6 GPUs, got', n)
if not bf16:
    print('ERROR: BF16 required')
    exit(1)
"

echo "=== Downloading missing images from HuggingFace ==="
echo "This will take a while. Images land at: $TECHJAM_DATA_ROOT"
uv run python scripts/data_prep/acquire_all_images.py

echo "=== Setup complete. Ready to train. ==="
echo "Run: bash cluster_train_stage2.sh"
