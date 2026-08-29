#!/usr/bin/env bash
# MechaDetect 4x4090 Production Launcher for Student Distillation Tracks
# Runs ViT-S on GPUs 0,1 and ViT-B on GPUs 2,3 with hardware and network isolation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

TEACHER_CHECKPOINT="${1:-outputs/teacher_stage2_paired_unfrozen/checkpoint-promoted.pt}"
TEACHER_CONFIG="${2:-configs/teacher_dinov3_stage2_paired_unfrozen.yaml}"
TEACHER_REPORT="${3:-outputs/teacher_stage2_paired_unfrozen/promotion_report.json}"
TRAIN_MANIFEST="${4:-splits/production_eligible/train.parquet}"
VAL_MANIFEST="${5:-splits/production_eligible/validation.parquet}"

echo "=================================================================="
echo "MechaDetect 4x4090 Student Distillation: Isolated Dual-Track Launch"
echo "Teacher Checkpoint: ${TEACHER_CHECKPOINT}"
echo "Teacher Config:     ${TEACHER_CONFIG}"
echo "Promotion Report:   ${TEACHER_REPORT}"
echo "Training Manifest:  ${TRAIN_MANIFEST}"
echo "Validation Manifest:${VAL_MANIFEST}"
echo "=================================================================="

# Check Python environment
PYTHON_BIN="${PYTHON:-python}"

${PYTHON_BIN} scripts/launch_students_distill.py \
    --track both \
    --teacher-checkpoint "${TEACHER_CHECKPOINT}" \
    --teacher-config "${TEACHER_CONFIG}" \
    --teacher-promotion-report "${TEACHER_REPORT}" \
    --manifest "${TRAIN_MANIFEST}" \
    --val-manifest "${VAL_MANIFEST}" \
    --epochs 2 \
    --seed 42
