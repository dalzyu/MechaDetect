#!/usr/bin/env bash
set -euo pipefail

# Resumable end-to-end training. One visible GPU is the default:
#   bash scripts/launch_production.sh
#
# Multi-GPU DDP is explicit:
#   GPU_DEVICES=0,1 bash scripts/launch_production.sh
#
# All topologies preserve an effective record batch of 48.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

UV_BIN="${UV_BIN:-uv}"
GPU_DEVICES="${GPU_DEVICES:-0}"
EFFECTIVE_BATCH_SIZE="${EFFECTIVE_BATCH_SIZE:-48}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_HOURS="${MAX_HOURS:-0}"

IFS=',' read -r -a DEVICE_IDS <<< "$GPU_DEVICES"
GPU_COUNT=0
for device in "${DEVICE_IDS[@]}"; do
  if [[ -n "${device//[[:space:]]/}" ]]; then
    GPU_COUNT=$((GPU_COUNT + 1))
  fi
done
if (( GPU_COUNT < 1 )); then
  echo "GPU_DEVICES must contain at least one CUDA device" >&2
  exit 2
fi
if ! [[ "$EFFECTIVE_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "EFFECTIVE_BATCH_SIZE must be a positive integer" >&2
  exit 2
fi
if ! [[ "$NUM_WORKERS" =~ ^[0-9]+$ ]]; then
  echo "NUM_WORKERS must be a non-negative integer" >&2
  exit 2
fi
if ! [[ "$MAX_HOURS" =~ ^[0-9]+$ ]]; then
  echo "MAX_HOURS must be a non-negative integer; 0 disables the deadline" >&2
  exit 2
fi

export TECHJAM_REPO_ROOT="${TECHJAM_REPO_ROOT:-$ROOT_DIR}"
export TECHJAM_RUNTIME_ROOT="${TECHJAM_RUNTIME_ROOT:-$ROOT_DIR/.runtime}"
export TECHJAM_DATA_ROOT="${TECHJAM_DATA_ROOT:-$TECHJAM_RUNTIME_ROOT/data}"
export TECHJAM_OUTPUT_ROOT="${TECHJAM_OUTPUT_ROOT:-$TECHJAM_RUNTIME_ROOT/outputs}"
export TECHJAM_HF_HOME="${TECHJAM_HF_HOME:-$TECHJAM_RUNTIME_ROOT/huggingface}"
export HF_HOME="${HF_HOME:-$TECHJAM_HF_HOME}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$TECHJAM_HF_HOME/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$TECHJAM_HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$TECHJAM_HF_HOME/datasets}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

if [[ -z "${HF_TOKEN:-}" && -f "$ROOT_DIR/.env" ]]; then
  HF_TOKEN="$("$UV_BIN" run python - "$ROOT_DIR/.env" <<'PY'
import os
import sys
from dotenv import load_dotenv

load_dotenv(sys.argv[1], override=False)
print(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "")
PY
)"
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required for the pinned gated DINOv3 backbone" >&2
  exit 1
fi
export HF_TOKEN
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"

OUTPUT_ROOT="$TECHJAM_OUTPUT_ROOT"
DATA_ROOT="$TECHJAM_DATA_ROOT"
STATE_DIR="$OUTPUT_ROOT/production_pipeline_state"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-$ROOT_DIR/splits/production_eligible/train.parquet}"
VAL_MANIFEST="${VAL_MANIFEST:-$ROOT_DIR/splits/production_eligible/validation.parquet}"

mkdir -p "$DATA_ROOT" "$OUTPUT_ROOT" "$TECHJAM_HF_HOME" "$STATE_DIR"
TRAIN_MANIFEST="$("$UV_BIN" run python -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$TRAIN_MANIFEST")"
VAL_MANIFEST="$("$UV_BIN" run python -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$VAL_MANIFEST")"
if [[ ! -f "$TRAIN_MANIFEST" || ! -f "$VAL_MANIFEST" ]]; then
  echo "Frozen train and validation manifests are required" >&2
  echo "TRAIN_MANIFEST=$TRAIN_MANIFEST" >&2
  echo "VAL_MANIFEST=$VAL_MANIFEST" >&2
  exit 1
fi

if (( MAX_HOURS > 0 )); then
  DEADLINE=$((SECONDS + MAX_HOURS * 3600))
else
  DEADLINE=0
fi

accumulation_for() {
  local physical_batch="$1"
  local divisor=$((physical_batch * GPU_COUNT))
  if (( EFFECTIVE_BATCH_SIZE % divisor != 0 )); then
    echo "Effective batch $EFFECTIVE_BATCH_SIZE is not divisible by ${physical_batch}x${GPU_COUNT}" >&2
    exit 2
  fi
  echo $((EFFECTIVE_BATCH_SIZE / divisor))
}
find_resume_checkpoint() {
  local stage_dir="$1"
  "$UV_BIN" run python - "$stage_dir" <<'PY'
import sys
from pathlib import Path
stage_dir = Path(sys.argv[1])
if not stage_dir.is_dir():
    sys.exit(0)
candidates = list(stage_dir.glob("checkpoint-coverage-*.pt"))
if not candidates:
    candidates = [
        p for p in stage_dir.glob("checkpoint-*.pt")
        if p.is_file() and p.name not in ("checkpoint-promoted.pt", "checkpoint-final.pt", "checkpoint-best.pt", "checkpoint-100pct.pt")
    ]
if candidates:
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    print(str(latest.resolve()))
PY
}

make_teacher_config() {
  local source="$1" target="$2" stage="$3" physical_batch="$4" initial_checkpoint="${5:-}"
  local accumulation
  accumulation="$(accumulation_for "$physical_batch")"
  "$UV_BIN" run python - "$source" "$target" "$stage" "$physical_batch" "$accumulation" \
    "$GPU_COUNT" "$NUM_WORKERS" "$DATA_ROOT" "$OUTPUT_ROOT" "$TRAIN_MANIFEST" \
    "$VAL_MANIFEST" "$initial_checkpoint" <<'PY'
import sys
from pathlib import Path
import yaml

(
    source,
    target,
    stage,
    physical_batch,
    accumulation,
    world_size,
    num_workers,
    data_root,
    output_root,
    train_manifest,
    val_manifest,
    initial_checkpoint,
) = sys.argv[1:]
config = yaml.safe_load(Path(source).read_text(encoding="utf-8"))
paths = config.setdefault("paths", {})
paths.update(
    data_root=str(Path(data_root).resolve()),
    output_root=str(Path(output_root).resolve()),
    train_manifest=str(Path(train_manifest).resolve()),
    val_manifest=str(Path(val_manifest).resolve()),
    require_materialized=True,
)
if initial_checkpoint:
    paths["initial_checkpoint"] = str(Path(initial_checkpoint).resolve())
training = config.setdefault("training", {})
training.update(
    stage=stage,
    required_world_size=int(world_size),
    physical_batch_size=int(physical_batch),
    gradient_accumulation=int(accumulation),
    num_workers=int(num_workers),
    validation_workers=int(num_workers),
)
Path(target).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY
}

run_stage() {
  local name="$1"
  shift
  local marker="$STATE_DIR/${name}.done"
  local log_file="$STATE_DIR/${name}.log"
  if [[ -f "$marker" ]]; then
    echo "[TRAIN] resume: $name already complete"
    return 0
  fi
  echo "[TRAIN] START $name"
  if (( DEADLINE > 0 )); then
    local remaining=$((DEADLINE - SECONDS))
    if (( remaining <= 0 )); then
      echo "[TRAIN] MAX_HOURS reached before $name" >&2
      exit 124
    fi
    timeout --signal=TERM --kill-after=120 "${remaining}s" "$@" 2>&1 | tee "$log_file"
  else
    "$@" 2>&1 | tee "$log_file"
  fi
  date -u +%Y-%m-%dT%H:%M:%SZ > "$marker"
  echo "[TRAIN] DONE $name"
}

if (( GPU_COUNT == 1 )); then
  TEACHER_LAUNCH=(env "CUDA_VISIBLE_DEVICES=$GPU_DEVICES" "$UV_BIN" run python -m aigc_detector.train)
else
  TEACHER_LAUNCH=(
    env "CUDA_VISIBLE_DEVICES=$GPU_DEVICES" "$UV_BIN" run torchrun
    --standalone "--nproc-per-node=$GPU_COUNT" -m aigc_detector.train
  )
fi

STAGE1_CONFIG="$STATE_DIR/teacher_stage1.yaml"
STAGE2_CONFIG="$STATE_DIR/teacher_stage2.yaml"
STAGE1_DIR="$OUTPUT_ROOT/teacher_stage1"
STAGE2_DIR="$OUTPUT_ROOT/teacher_stage2"
make_teacher_config \
  configs/teacher_dinov3_stage1_clean_frozen.yaml \
  "$STAGE1_CONFIG" teacher_stage1 6
make_teacher_config \
  configs/teacher_dinov3_stage2_paired_unfrozen.yaml \
  "$STAGE2_CONFIG" teacher_stage2 2 "$STAGE1_DIR/checkpoint-promoted.pt"

if [[ -f "$STAGE1_DIR/checkpoint-100pct.pt" ]]; then
  echo "[TRAIN] $STAGE1_DIR/checkpoint-100pct.pt exists; teacher-stage1 already completed training"
  touch "$STATE_DIR/teacher-stage1.done"
fi
STAGE1_RESUME="$(find_resume_checkpoint "$STAGE1_DIR")"
STAGE1_ARGS=(--config "$STAGE1_CONFIG")
if [[ -n "$STAGE1_RESUME" ]]; then
  echo "[TRAIN] teacher-stage1 resuming from $STAGE1_RESUME"
  STAGE1_ARGS+=(--resume "$STAGE1_RESUME")
fi
run_stage teacher-stage1 "${TEACHER_LAUNCH[@]}" "${STAGE1_ARGS[@]}"
  "$UV_BIN" run python scripts/evaluate_teacher.py \
  --checkpoint "$STAGE1_DIR/checkpoint-100pct.pt" \
  --manifest "$VAL_MANIFEST" --config "$STAGE1_CONFIG" --data-root "$DATA_ROOT" \
  --output "$STAGE1_DIR/evaluation.json" --device cuda --batch-size 8
run_stage teacher-stage1-promote \
  "$UV_BIN" run python scripts/promote_teacher.py \
  --checkpoints "$STAGE1_DIR/checkpoint-100pct.pt" \
  --eval-reports "$STAGE1_DIR/evaluation.json" \
  --manifest "$VAL_MANIFEST" --config "$STAGE1_CONFIG" --data-root "$DATA_ROOT" \
  --output-report "$STAGE1_DIR/promotion_report.json" \
  --output-metadata "$STAGE1_DIR/metadata.json" \
  --output-checkpoint "$STAGE1_DIR/checkpoint-promoted.pt" \
  --device cuda --batch-size 8

if [[ -f "$STAGE2_DIR/checkpoint-100pct.pt" ]]; then
  echo "[TRAIN] $STAGE2_DIR/checkpoint-100pct.pt exists; teacher-stage2 already completed training"
  touch "$STATE_DIR/teacher-stage2.done"
fi
STAGE2_RESUME="$(find_resume_checkpoint "$STAGE2_DIR")"
STAGE2_ARGS=(--config "$STAGE2_CONFIG")
if [[ -n "$STAGE2_RESUME" ]]; then
  echo "[TRAIN] teacher-stage2 resuming from $STAGE2_RESUME"
  STAGE2_ARGS+=(--resume "$STAGE2_RESUME")
else
  STAGE2_ARGS+=(--initial-checkpoint "$STAGE1_DIR/checkpoint-promoted.pt")
fi
run_stage teacher-stage2 "${TEACHER_LAUNCH[@]}" "${STAGE2_ARGS[@]}"
run_stage teacher-stage2-eval \
  "$UV_BIN" run python scripts/evaluate_teacher.py \
  --checkpoint "$STAGE2_DIR/checkpoint-100pct.pt" \
  --manifest "$VAL_MANIFEST" --config "$STAGE2_CONFIG" --data-root "$DATA_ROOT" \
  --output "$STAGE2_DIR/evaluation.json" --device cuda --batch-size 8
run_stage teacher-stage2-promote \
  "$UV_BIN" run python scripts/promote_teacher.py \
  --checkpoints "$STAGE2_DIR/checkpoint-100pct.pt" \
  --eval-reports "$STAGE2_DIR/evaluation.json" \
  --manifest "$VAL_MANIFEST" --config "$STAGE2_CONFIG" --data-root "$DATA_ROOT" \
  --output-report "$STAGE2_DIR/promotion_report.json" \
  --output-metadata "$STAGE2_DIR/metadata.json" \
  --output-checkpoint "$STAGE2_DIR/checkpoint-promoted.pt" \
  --device cuda --batch-size 8

TEACHER="$STAGE2_DIR/checkpoint-promoted.pt"
TEACHER_REPORT="$STAGE2_DIR/promotion_report.json"
for track in small base; do
  track_dir="$OUTPUT_ROOT/student_dinov3_${track}"
  if [[ -f "$track_dir/checkpoint-promoted.pt" ]]; then
    touch "$STATE_DIR/student-${track}.done"
  fi
  run_stage "student-$track" \
    "$UV_BIN" run python scripts/launch_students_distill.py \
    --track "$track" \
    --teacher-config "$STAGE2_CONFIG" \
    --teacher-checkpoint "$TEACHER" \
    --teacher-promotion-report "$TEACHER_REPORT" \
    --manifest "$TRAIN_MANIFEST" \
    --val-manifest "$VAL_MANIFEST" \
    --resume auto \
    "--${track}-devices" "$GPU_DEVICES" \
    "--${track}-output-dir" "$track_dir"
done

for track in small base; do
  track_dir="$OUTPUT_ROOT/att_student_${track}"
  if [[ -f "$track_dir/checkpoint-final.pt" ]]; then
    touch "$STATE_DIR/att-${track}.done"
  fi
  run_stage "att-$track" \
    "$UV_BIN" run python scripts/launch_att_tracks.py \
    --track "$track" \
    "--${track}-checkpoint" "$OUTPUT_ROOT/student_dinov3_${track}/checkpoint-promoted.pt" \
    --train-manifest "$TRAIN_MANIFEST" \
    --resume auto \
    "--${track}-devices" "$GPU_DEVICES" \
    "--${track}-output" "$track_dir"
done

echo "[TRAIN] Complete teacher, student-distillation, and ATT stages"
