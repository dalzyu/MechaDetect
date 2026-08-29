#!/usr/bin/env bash
# ==============================================================================
# cluster_train_stage2.sh — Launch Stage 2 Teacher Training on 4x RTX 4090
# ==============================================================================
# Run after Stage 1 completes and passes the promotion gate.
# Usage:
#   bash cluster_train_stage2.sh [--resume /path/to/checkpoint.pt] [--initial-checkpoint /path/to/checkpoint.pt]
# ==============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

if [[ ! -f .env ]]; then
  echo "ERROR: .env not found. Copy .env.cluster to .env and configure it."
  exit 1
fi
set -a; source .env; set +a

export TECHJAM_RUNTIME_ROOT="${TECHJAM_RUNTIME_ROOT:-/workspace/techjam26-runtime}"
export TECHJAM_DATA_ROOT="${TECHJAM_DATA_ROOT:-$TECHJAM_RUNTIME_ROOT/data}"
export TECHJAM_OUTPUT_ROOT="${TECHJAM_OUTPUT_ROOT:-$TECHJAM_RUNTIME_ROOT/outputs}"

RESUME_ARG=""
INIT_CKPT_ARG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume)
      RESUME_ARG="--resume $2"
      shift 2
      ;;
    --initial-checkpoint)
      INIT_CKPT_ARG="--initial-checkpoint $2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [[ -z "$RESUME_ARG" && -z "$INIT_CKPT_ARG" ]]; then
  echo "=== Locating Promoted Stage 1 Checkpoint ==="
  STAGE1_PROMOTED="$TECHJAM_OUTPUT_ROOT/teacher_stage1_clean_frozen/checkpoint-promoted.pt"
  STAGE1_REPORT="$TECHJAM_OUTPUT_ROOT/teacher_stage1_clean_frozen/promotion_report.json"

  if [[ -f "$STAGE1_PROMOTED" && -f "$STAGE1_REPORT" ]]; then
    INIT_CKPT_ARG="--initial-checkpoint $STAGE1_PROMOTED"
    echo "Found verified promoted Stage 1 checkpoint: $STAGE1_PROMOTED"
  else
    echo "ERROR: Stage 1 promoted checkpoint or promotion report missing."
    echo "Expected: $STAGE1_PROMOTED and $STAGE1_REPORT"
    echo "Run and promote Teacher Stage 1 first (orchestrate_4x4090.sh) or pass --initial-checkpoint explicitly."
    exit 1
  fi
fi

echo "=== Verifying Output Storage Space ==="
AVAIL_GB=$(df -BG "$TECHJAM_OUTPUT_ROOT" | awk 'NR==2 {gsub("G",""); print $4}')
echo "Available on output disk: ${AVAIL_GB} GB"
if (( AVAIL_GB < 60 )); then
  echo "WARNING: Less than 60 GB available. Full resume checkpoints are ~14 GB each."
fi

echo "=== Launching Teacher Stage 2 (4-GPU RTX 4090 DDP) ==="
echo "Config: configs/teacher_dinov3_stage2_paired_unfrozen.yaml"
echo "Output: $TECHJAM_OUTPUT_ROOT/teacher_stage2_paired_unfrozen/"
echo ""

exec uv run torchrun \
  --standalone \
  --nproc-per-node=4 \
  -m aigc_detector.train \
  --config configs/teacher_dinov3_stage2_paired_unfrozen.yaml \
  $INIT_CKPT_ARG \
  $RESUME_ARG
