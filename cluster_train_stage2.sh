#!/usr/bin/env bash
# cluster_train_stage2.sh — launch Stage 2 teacher training.
# Run after cluster_setup.sh completes successfully.
# Usage: bash cluster_train_stage2.sh [--resume /path/to/checkpoint.pt]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

set -a; source .env; set +a

RESUME_ARG=""
if [[ "${1:-}" == "--resume" && -n "${2:-}" ]]; then
  RESUME_ARG="--resume $2"
  echo "=== Resuming from: $2 ==="
fi

echo "=== Warm-start checkpoint ==="
STAGE1_CKPT="$TECHJAM_OUTPUT_ROOT/teacher_stage1_clean_frozen/checkpoint-best.pt"
if [[ -z "$RESUME_ARG" ]]; then
  if [[ ! -f "$STAGE1_CKPT" ]]; then
    # Fall back to the published safetensors from Iteration 1
    STAGE1_CKPT="$REPO_ROOT/models/teachers/iteration1/stage1/model-weights.safetensors"
    echo "Full checkpoint not found — using published safetensors: $STAGE1_CKPT"
  fi
  if [[ ! -f "$STAGE1_CKPT" ]]; then
    echo "ERROR: No Stage 1 checkpoint found. Run Stage 1 first or copy checkpoint-best.pt."
    exit 1
  fi
  echo "Stage 1 checkpoint: $STAGE1_CKPT"
fi

echo "=== Verifying output disk space ==="
AVAIL_GB=$(df -BG "$TECHJAM_OUTPUT_ROOT" | awk 'NR==2 {gsub("G",""); print $4}')
echo "Available on output disk: ${AVAIL_GB}GB"
if (( AVAIL_GB < 80 )); then
  echo "WARNING: Less than 80GB available. Full resume checkpoints are ~14GB each."
  echo "Reduce checkpoint_interval or provision more disk before continuing."
fi

echo "=== Launching Stage 2 (6-GPU DDP) ==="
echo "Config: configs/teacher_dinov3_stage2_paired_unfrozen.yaml"
echo "Output: $TECHJAM_OUTPUT_ROOT/teacher_stage2_paired_unfrozen/"
echo ""

uv run torchrun \
  --standalone \
  --nproc-per-node=6 \
  -m aigc_detector.train \
  --config configs/teacher_dinov3_stage2_paired_unfrozen.yaml \
  $RESUME_ARG
