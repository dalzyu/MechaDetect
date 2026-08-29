#!/usr/bin/env bash
# ==============================================================================
# orchestrate_4x4090.sh — Top-level runner for MechaDetect 4x RTX 4090 Pipeline
# ==============================================================================
# Usage:
#   bash orchestrate_4x4090.sh [OPTIONS]
#
# Examples:
#   bash orchestrate_4x4090.sh                        # Run complete pipeline from current stage
#   bash orchestrate_4x4090.sh --stage preflight       # Run preflight verification
#   bash orchestrate_4x4090.sh --stage teacher-stage1  # Start from Teacher Stage 1
#   bash orchestrate_4x4090.sh --explicit-balance 25.50 # Supply balance override
#   bash orchestrate_4x4090.sh --force                 # Ignore state file and force re-run
# ==============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "=== MechaDetect 4x RTX 4090 Production Orchestrator ==="
echo "Working directory: $REPO_ROOT"

# Check environment configuration
if [[ ! -f .env ]]; then
  if [[ -f .env.cluster ]]; then
    echo "ERROR: .env not found. Initializing from .env.cluster..."
    cp .env.cluster .env
    echo "Created .env from .env.cluster. Please review and fill in HF_TOKEN, then re-run."
    exit 1
  else
    echo "ERROR: Neither .env nor .env.cluster found. Please configure environment."
    exit 1
  fi
fi

echo "=== Loading environment (.env) ==="
set -a
source .env
set +a

# Export default runtime paths if not set
export TECHJAM_REPO_ROOT="$REPO_ROOT"
export TECHJAM_RUNTIME_ROOT="${TECHJAM_RUNTIME_ROOT:-/workspace/techjam26-runtime}"
export TECHJAM_DATA_ROOT="${TECHJAM_DATA_ROOT:-$TECHJAM_RUNTIME_ROOT/data}"
export TECHJAM_HF_HOME="${TECHJAM_HF_HOME:-$TECHJAM_RUNTIME_ROOT/huggingface}"
export TECHJAM_OUTPUT_ROOT="${TECHJAM_OUTPUT_ROOT:-$TECHJAM_RUNTIME_ROOT/outputs}"

# Create directories
mkdir -p "$TECHJAM_DATA_ROOT"
mkdir -p "$TECHJAM_HF_HOME"
mkdir -p "$TECHJAM_OUTPUT_ROOT"
mkdir -p "$TECHJAM_OUTPUT_ROOT/upload_receipts"
mkdir -p "$TECHJAM_OUTPUT_ROOT/exported"

echo "=== Executing 4x4090 State Machine ==="
exec uv run python scripts/orchestrate_4x4090.py "$@"
