#!/usr/bin/env bash
# ==============================================================================
# cluster_setup.sh — Run once on a fresh 4x RTX 4090 instance before training.
# ==============================================================================
# Usage:
#   bash cluster_setup.sh
#
# Assumes:
#   - Repository cloned: git clone ... && cd "techjam 26"
#   - Branch checked out: git checkout training/production-4x4090
#   - uv is installed (curl -LsSf https://astral.sh/uv/install.sh | sh)
#   - .env is configured (copied from .env.cluster with HF_TOKEN filled in)
# ==============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "=== MechaDetect 4x RTX 4090 Cluster Setup ==="
echo "Working directory: $REPO_ROOT"

# 1. Environment Loading
echo "=== Loading environment ==="
if [[ ! -f .env ]]; then
  if [[ -f .env.cluster ]]; then
    echo "ERROR: .env not found. Copying .env.cluster to .env..."
    cp .env.cluster .env
    echo "Created .env from .env.cluster. Please edit .env and set HF_TOKEN, then re-run."
    exit 1
  else
    echo "ERROR: Neither .env nor .env.cluster found. Please configure environment."
    exit 1
  fi
fi

set -a
source .env
set +a

# 2. Runtime Directories
echo "=== Creating runtime directories ==="
export TECHJAM_RUNTIME_ROOT="${TECHJAM_RUNTIME_ROOT:-/workspace/techjam26-runtime}"
export TECHJAM_DATA_ROOT="${TECHJAM_DATA_ROOT:-$TECHJAM_RUNTIME_ROOT/data}"
export TECHJAM_HF_HOME="${TECHJAM_HF_HOME:-$TECHJAM_RUNTIME_ROOT/huggingface}"
export TECHJAM_OUTPUT_ROOT="${TECHJAM_OUTPUT_ROOT:-$TECHJAM_RUNTIME_ROOT/outputs}"

mkdir -p "$TECHJAM_DATA_ROOT"
mkdir -p "$TECHJAM_HF_HOME"
mkdir -p "$TECHJAM_OUTPUT_ROOT"
mkdir -p "$TECHJAM_OUTPUT_ROOT/upload_receipts"
mkdir -p "$TECHJAM_OUTPUT_ROOT/exported"

# 3. Dependency Installation
echo "=== Installing Python dependencies via uv ==="
uv sync --locked --dev
echo "=== Installing browser benchmark dependencies ==="
npm ci

# 4. Hardware & Environment Preflight
echo "=== Verifying 4x RTX 4090 and BF16 ==="
uv run python -c "
import sys
import torch
n = torch.cuda.device_count()
print(f'CUDA GPUs visible: {n}')
if n != 4:
    print(f'ERROR: Expected exactly 4 GPUs, found {n}.', file=sys.stderr)
    sys.exit(1)

for i in range(n):
    name = torch.cuda.get_device_name(i)
    print(f'  cuda:{i} — {name}')
    if '4090' not in name:
        print(f'ERROR: cuda:{i} is not an RTX 4090 ({name})', file=sys.stderr)
        sys.exit(1)

bf16 = torch.cuda.is_bf16_supported()
print(f'BF16 supported: {bf16}')
if not bf16:
    print('ERROR: BF16 required for 4x4090 production pipeline.', file=sys.stderr)
    sys.exit(1)
"

# 5. Disk Space Verification
echo "=== Verifying output storage disk space ==="
AVAIL_GB=$(df -BG "$TECHJAM_OUTPUT_ROOT" | awk 'NR==2 {gsub("G",""); print $4}')
echo "Available disk on $TECHJAM_OUTPUT_ROOT: ${AVAIL_GB} GB"
if (( AVAIL_GB < 200 )); then
  echo "ERROR: Less than 200 GB available (${AVAIL_GB} GB). Production run requires >= 200 GB."
  exit 1
fi

# 6. Git Verification
echo "=== Verifying Git branch and commit ==="
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
CURRENT_SHA=$(git rev-parse HEAD)
echo "Current branch: $CURRENT_BRANCH"
echo "Current commit: $CURRENT_SHA"
if [[ "$CURRENT_BRANCH" != "training/production-4x4090" ]]; then
  echo "WARNING: Not on branch training/production-4x4090 (currently on $CURRENT_BRANCH)."
fi

# 7. Hugging Face Hub Access Verification
echo "=== Verifying Hugging Face repository access ==="
uv run python -c "
import os, sys
token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
if not token:
    print('ERROR: HF_TOKEN is empty. Set HF_TOKEN in .env', file=sys.stderr)
    sys.exit(1)

from huggingface_hub import HfApi
api = HfApi(token=token)
user_info = api.whoami()
print(f'Authenticated as HF user: {user_info.get(\"name\")}')

model_repo = os.environ.get('TECHJAM_MODEL_REPO', 'zye2/mechadetect-models')
data_repo = os.environ.get('TECHJAM_DATA_REPO', 'zye2/tj-data')

try:
    api.repo_info(repo_id=model_repo, repo_type='model')
    print(f'Verified access to private model repo: {model_repo}')
except Exception:
    print(f'Creating private model repo: {model_repo}')
    api.create_repo(repo_id=model_repo, repo_type='model', private=True, exist_ok=True)
    print(f'Created private model repo: {model_repo}')

try:
    api.repo_info(repo_id=data_repo, repo_type='dataset')
    print(f'Verified access to public dataset repo: {data_repo}')
except Exception:
    print(f'Creating public dataset repo: {data_repo}')
    try:
        api.create_repo(repo_id=data_repo, repo_type='dataset', private=False, exist_ok=True)
        print(f'Created public dataset repo: {data_repo}')
    except Exception as e:
        print(f'ERROR: Failed verifying/creating dataset repo {data_repo}: {e}', file=sys.stderr)
        sys.exit(1)

# Verify Vast balance & hourly rate
bal_str = os.environ.get('TECHJAM_EXPLICIT_BALANCE')
rate_str = os.environ.get('VAST_HOURLY_RATE')
if not rate_str:
    print('ERROR: VAST_HOURLY_RATE not set in .env. Required for budget guard.', file=sys.stderr)
    sys.exit(1)
if bal_str:
    bal = float(bal_str)
    if bal < 5.00:
        print(f'ERROR: Balance (${bal:.2f}) violates mandatory $5.00 reserve.', file=sys.stderr)
        sys.exit(1)
"

# 8. Prefetch / Download Images
echo "=== Prefetching source images into local storage ==="
echo "Images will land at: $TECHJAM_DATA_ROOT"
uv run python scripts/data_prep/acquire_all_images.py \
  --data-root "$TECHJAM_DATA_ROOT" \
  --resume \
  --verify-bytes \
  --report-path "$TECHJAM_OUTPUT_ROOT/acquisition_report.json" \
  --revisions-path "$REPO_ROOT/splits/production_eligible/source_revisions.json"

# 9. Freeze Eligible Manifests
echo "=== Freezing production-eligible manifests ==="
uv run python scripts/data_prep/freeze_production_eligible.py \
  --data-root "$TECHJAM_DATA_ROOT" \
  --output-dir "$REPO_ROOT/splits/production_eligible" \
  --calibration-size 4096 \
  --strict \
  --verify-bytes

echo "=== Setup complete! Instance is ready for 4x4090 execution. ==="
echo "To launch the complete gated production pipeline, run:"
echo "    bash orchestrate_4x4090.sh"
