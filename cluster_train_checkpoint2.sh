#!/usr/bin/env bash
# cluster_train_checkpoint2.sh — run after Stage 2 is promoted.
# Usage: bash cluster_train_checkpoint2.sh [--resume /path/to/checkpoint.pt]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

set -a; source .env; set +a

RESUME_ARG=""
if [[ "${1:-}" == "--resume" && -n "${2:-}" ]]; then
  RESUME_ARG="--resume $2"
  echo "=== Resuming from: $2 ==="
fi

echo "=== Verifying Stage 2 checkpoint ==="
STAGE2_CKPT="$TECHJAM_OUTPUT_ROOT/teacher_stage2_paired_unfrozen/checkpoint-best.pt"
if [[ -z "$RESUME_ARG" && ! -f "$STAGE2_CKPT" ]]; then
  echo "ERROR: $STAGE2_CKPT not found. Run and promote Stage 2 first."
  exit 1
fi

echo "=== Launching Checkpoint 2 (6-GPU DDP) ==="
echo "Config: configs/teacher_dinov3_checkpoint2_full_data.yaml"
echo "Output: $TECHJAM_OUTPUT_ROOT/teacher_checkpoint2_production_full_data/"
echo ""

uv run torchrun \
  --standalone \
  --nproc-per-node=6 \
  -m aigc_detector.train \
  --config configs/teacher_dinov3_checkpoint2_full_data.yaml \
  $RESUME_ARG
