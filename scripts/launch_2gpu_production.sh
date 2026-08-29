#!/usr/bin/env bash
set -euo pipefail

# Resumable adaptation of the canonical production plan for a visible 2-GPU host.
# Effective batch stays 48; promotion/evaluation gates remain unchanged.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${TECHJAM_OUTPUT_ROOT:-$ROOT_DIR/outputs}"
DATA_ROOT="${TECHJAM_DATA_ROOT:-$ROOT_DIR/data}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-$ROOT_DIR/splits/production_eligible/train.parquet}"
VAL_MANIFEST="${VAL_MANIFEST:-$ROOT_DIR/splits/production_eligible/validation.parquet}"
STATE_DIR="$OUTPUT_ROOT/2gpu_pipeline_state"
MAX_HOURS="${MAX_HOURS:-16}"
mkdir -p "$STATE_DIR"

if ! [[ "$MAX_HOURS" =~ ^[0-9]+([.][0-9]+)?$ ]] || (( ${MAX_HOURS%.*} < 1 )); then
  echo "MAX_HOURS must be a positive number" >&2
  exit 2
fi
DEADLINE=$((SECONDS + ${MAX_HOURS%.*} * 3600))

make_teacher_config() {
  local source="$1" target="$2" stage="$3" batch="$4" accumulation="$5"
  "$PYTHON_BIN" - "$source" "$target" "$stage" "$batch" "$accumulation" <<'PY'
import sys
from pathlib import Path
import yaml
source, target, stage, batch, accumulation = sys.argv[1:]
config = yaml.safe_load(Path(source).read_text())
training = config.setdefault("training", {})
training["stage"] = stage
training["required_world_size"] = 2
training["physical_batch_size"] = int(batch)
training["gradient_accumulation"] = int(accumulation)
training["num_workers"] = min(8, int(training.get("num_workers", 4)))
Path(target).write_text(yaml.safe_dump(config, sort_keys=False))
PY
}

run_stage() {
  local name="$1"; shift
  local marker="$STATE_DIR/${name}.done"
  if [[ -f "$marker" ]]; then
    echo "[2GPU] resume: $name already complete"
    return 0
  fi
  local remaining=$((DEADLINE - SECONDS))
  if (( remaining <= 0 )); then
    echo "[2GPU] max-hours budget reached before $name" >&2
    exit 124
  fi
  echo "[2GPU] START $name (${remaining}s remaining)"
  if timeout --signal=TERM --kill-after=120 "${remaining}s" "$@" 2>&1 | tee "$OUTPUT_ROOT/2gpu_pipeline_state/${name}.log"; then
    date -u +%Y-%m-%dT%H:%M:%SZ > "$marker"
    echo "[2GPU] DONE $name"
  else
    echo "[2GPU] FAILED $name" >&2
    exit 1
  fi
}

if [[ ! -f "$TRAIN_MANIFEST" || ! -f "$VAL_MANIFEST" ]]; then
  echo "Frozen train/validation manifests are required before training" >&2
  exit 1
fi

STAGE1_CONFIG="$STATE_DIR/teacher_stage1_2gpu.yaml"
STAGE2_CONFIG="$STATE_DIR/teacher_stage2_2gpu.yaml"
make_teacher_config configs/teacher_dinov3_stage1_clean_frozen.yaml "$STAGE1_CONFIG" teacher_stage1_2gpu 6 4
make_teacher_config configs/teacher_dinov3_stage2_paired_unfrozen.yaml "$STAGE2_CONFIG" teacher_stage2_2gpu 2 12

run_stage teacher-stage1-smoke \
  torchrun --standalone --nproc-per-node=2 -m aigc_detector.train \
  --config "$STAGE1_CONFIG" --max-steps 2 --stage teacher_stage1_2gpu_smoke
run_stage teacher-stage1 \
  torchrun --standalone --nproc-per-node=2 -m aigc_detector.train \
  --config "$STAGE1_CONFIG"
run_stage teacher-stage1-eval \
  "$PYTHON_BIN" scripts/evaluate_teacher.py \
  --checkpoint "$OUTPUT_ROOT/teacher_stage1_2gpu/checkpoint-100pct.pt" \
  --manifest "$VAL_MANIFEST" --config "$STAGE1_CONFIG" --data-root "$DATA_ROOT" \
  --output "$OUTPUT_ROOT/teacher_stage1_2gpu/evaluation.json" \
  --device cuda --batch-size 8
run_stage teacher-stage1-promote \
  "$PYTHON_BIN" scripts/promote_teacher.py \
  --checkpoints "$OUTPUT_ROOT/teacher_stage1_2gpu/checkpoint-100pct.pt" \
  --eval-reports "$OUTPUT_ROOT/teacher_stage1_2gpu/evaluation.json" \
  --manifest "$VAL_MANIFEST" --config "$STAGE1_CONFIG" --data-root "$DATA_ROOT" \
  --output-report "$OUTPUT_ROOT/teacher_stage1_2gpu/promotion_report.json" \
  --output-metadata "$OUTPUT_ROOT/teacher_stage1_2gpu/metadata.json" \
  --output-checkpoint "$OUTPUT_ROOT/teacher_stage1_2gpu/checkpoint-promoted.pt" \
  --device cuda --batch-size 8

run_stage teacher-stage2-smoke \
  torchrun --standalone --nproc-per-node=2 -m aigc_detector.train \
  --config "$STAGE2_CONFIG" --initial-checkpoint "$OUTPUT_ROOT/teacher_stage1_2gpu/checkpoint-promoted.pt" \
  --max-steps 2 --stage teacher_stage2_2gpu_smoke
run_stage teacher-stage2 \
  torchrun --standalone --nproc-per-node=2 -m aigc_detector.train \
  --config "$STAGE2_CONFIG" --initial-checkpoint "$OUTPUT_ROOT/teacher_stage1_2gpu/checkpoint-promoted.pt"
run_stage teacher-stage2-eval \
  "$PYTHON_BIN" scripts/evaluate_teacher.py \
  --checkpoint "$OUTPUT_ROOT/teacher_stage2_2gpu/checkpoint-100pct.pt" \
  --manifest "$VAL_MANIFEST" --config "$STAGE2_CONFIG" --data-root "$DATA_ROOT" \
  --output "$OUTPUT_ROOT/teacher_stage2_2gpu/evaluation.json" \
  --device cuda --batch-size 8
run_stage teacher-stage2-promote \
  "$PYTHON_BIN" scripts/promote_teacher.py \
  --checkpoints "$OUTPUT_ROOT/teacher_stage2_2gpu/checkpoint-100pct.pt" \
  --eval-reports "$OUTPUT_ROOT/teacher_stage2_2gpu/evaluation.json" \
  --manifest "$VAL_MANIFEST" --config "$STAGE2_CONFIG" --data-root "$DATA_ROOT" \
  --output-report "$OUTPUT_ROOT/teacher_stage2_2gpu/promotion_report.json" \
  --output-metadata "$OUTPUT_ROOT/teacher_stage2_2gpu/metadata.json" \
  --output-checkpoint "$OUTPUT_ROOT/teacher_stage2_2gpu/checkpoint-promoted.pt" \
  --device cuda --batch-size 8

TEACHER="$OUTPUT_ROOT/teacher_stage2_2gpu/checkpoint-promoted.pt"
TEACHER_REPORT="$OUTPUT_ROOT/teacher_stage2_2gpu/promotion_report.json"
run_stage student-small \
  env CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  scripts/distill_student.py --teacher-config "$STAGE2_CONFIG" --teacher-checkpoint "$TEACHER" \
  --teacher-promotion-report "$TEACHER_REPORT" --manifest "$TRAIN_MANIFEST" --val-manifest "$VAL_MANIFEST" \
  --output-dir "$OUTPUT_ROOT/student_dinov3_small_2gpu" --student small \
  --student-config configs/student_dinov3_small_distill.yaml --world-size 2 \
  --physical-batch-size 12 --gradient-accumulation 2 --num-workers 8 --epochs 2 --devices 0,1 --port 29501
run_stage student-base \
  env CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  scripts/distill_student.py --teacher-config "$STAGE2_CONFIG" --teacher-checkpoint "$TEACHER" \
  --teacher-promotion-report "$TEACHER_REPORT" --manifest "$TRAIN_MANIFEST" --val-manifest "$VAL_MANIFEST" \
  --output-dir "$OUTPUT_ROOT/student_dinov3_base_2gpu" --student base \
  --student-config configs/student_dinov3_base_distill.yaml --world-size 2 \
  --physical-batch-size 3 --gradient-accumulation 8 --num-workers 8 --epochs 2 --devices 0,1 --port 29502

run_stage att-small \
  env CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  scripts/train_att.py --variant small --student-checkpoint "$OUTPUT_ROOT/student_dinov3_small_2gpu/checkpoint-promoted.pt" \
  --manifest "$TRAIN_MANIFEST" --config configs/att_student_small.yaml --output-dir "$OUTPUT_ROOT/att_student_small_2gpu" \
  --num-candidates 3 --batch-size 4 --gradient-accumulation 6 --epochs 1 --seed 42
run_stage att-base \
  env CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  scripts/train_att.py --variant base --student-checkpoint "$OUTPUT_ROOT/student_dinov3_base_2gpu/checkpoint-promoted.pt" \
  --manifest "$TRAIN_MANIFEST" --config configs/att_student_base.yaml --output-dir "$OUTPUT_ROOT/att_student_base_2gpu" \
  --num-candidates 3 --batch-size 2 --gradient-accumulation 12 --epochs 1 --seed 42

echo "[2GPU] Complete teacher, student, and ATT training stages"
