#!/usr/bin/env bash
# ==============================================================================
# cluster_train_checkpoint2.sh — historical Checkpoint 2 adaptation
#
# Checkpoint 2 is not part of the forward production pipeline. Use
# scripts/launch_production.sh for the maintained teacher → students → ATT flow.
# This wrapper remains for explicit ablation work and defaults to GPU 0.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "=============================================================================="
echo "NOTICE: Checkpoint 2 is a historical ablation, not the production pipeline."
echo "Maintained pipeline: bash scripts/launch_production.sh"
echo "=============================================================================="

ALLOW_LEGACY=false
RESUME_PATH=""
while (($#)); do
  case "$1" in
    --legacy-run|--force)
      ALLOW_LEGACY=true
      shift
      ;;
    --resume)
      if (($# < 2)); then
        echo "--resume requires a checkpoint path" >&2
        exit 2
      fi
      RESUME_PATH="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "$ALLOW_LEGACY" != "true" ]]; then
  echo ""
  echo "Run the maintained production pipeline with:"
  echo "    bash scripts/launch_production.sh"
  echo ""
  echo "Run this historical ablation explicitly with:"
  echo "    bash cluster_train_checkpoint2.sh --legacy-run"
  exit 1
fi

if [[ ! -f .env ]]; then
  echo "ERROR: .env not found."
  exit 1
fi
set -a; source .env; set +a

export TECHJAM_OUTPUT_ROOT="${TECHJAM_OUTPUT_ROOT:-$REPO_ROOT/.runtime/outputs}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
  echo "This historical wrapper accepts one CUDA device; use scripts/launch_production.sh for DDP" >&2
  exit 2
fi
STAGE2_CKPT="$TECHJAM_OUTPUT_ROOT/teacher_stage2/checkpoint-promoted.pt"
if [[ -z "$RESUME_PATH" && ! -f "$STAGE2_CKPT" ]]; then
  echo "Stage 2 checkpoint not found: $STAGE2_CKPT" >&2
  exit 1
fi

echo "=== Launching historical Checkpoint 2 adaptation on GPU $CUDA_VISIBLE_DEVICES ==="
COMMAND=(
  uv run python -m aigc_detector.train
  --config configs/teacher_dinov3_checkpoint2_full_data.yaml
)
if [[ -n "$RESUME_PATH" ]]; then
  COMMAND+=(--resume "$RESUME_PATH")
else
  COMMAND+=(--initial-checkpoint "$STAGE2_CKPT")
fi
exec "${COMMAND[@]}"
