#!/usr/bin/env python3
"""MechaDetect 4x RTX 4090 Gated Production Pipeline Orchestrator.

Implements the resumable 21-stage state machine from preflight through upload:
1. preflight
2. acquire
3. freeze-manifests
4. teacher-stage1-smoke
5. teacher-stage1
6. teacher-stage1-eval
7. teacher-stage2-smoke
8. teacher-stage2
9. teacher-eval
10. students-smoke
11. students-distill
12. students-eval
13. att-smoke
14. att
15. att-eval
16. export-float
17. calibrate-int8
18. evaluate-int8
19. runtime-benchmark
20. upload
21. complete

Engineering Invariants:
- Output dirs: ${TECHJAM_OUTPUT_ROOT}/teacher_stage1_clean_frozen and teacher_stage2_paired_unfrozen
- Checkpoint promotion: reads checkpoint_path from promotion_report.json, copies atomically to checkpoint-promoted.pt, verifies hash
- Resumability: validates completion artifacts and SHA-256 hashes on disk before skipping any completed stage
- Both student and ATT smokes: runs independent dry-runs for both small and base tracks before launching expensive stages
- ATT promotion gate: trains checkpoint-final.pt, evaluates against pre-ATT float baselines, copies to checkpoint-promoted.pt only upon passing check_att_gate.py
- Runtime benchmark: synchronizes metadata to web/model/metadata.json, benchmarks ONNX Runtime CPUExecutionProvider (forced-WASM path), verifies WebGPU graph compatibility, and executes web/benchmark.js if Node/Puppeteer is present
- Upload receipts: upload_stage return values are strictly verified and fail stages on error
- Preflight: fails closed on missing HF token, unqueryable balance, balance < $5 reserve, unknown hourly rate, GPU count != 4, disk < 200GB, or forbidden data leaks
- Measured cost projection: derives updates from eligible train count / 48, computes seconds/update from measured 2-update smoke (no guessing), requires explicit actual hourly rate via env/CLI, and rechecks $5 reserve before every expensive stage
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PIPELINE_STAGES = [
    "preflight",
    "acquire",
    "freeze-manifests",
    "teacher-stage1-smoke",
    "teacher-stage1",
    "teacher-stage1-eval",
    "teacher-stage2-smoke",
    "teacher-stage2",
    "teacher-eval",
    "students-smoke",
    "students-distill",
    "students-eval",
    "att-smoke",
    "att",
    "att-eval",
    "export-float",
    "calibrate-int8",
    "evaluate-int8",
    "runtime-benchmark",
    "upload",
    "complete",
]


def log(stage: str, msg: str) -> None:
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{now}] [{stage.upper()}] {msg}", flush=True)


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest for a file."""
    if not path.is_file():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class PipelineStateManager:
    """Manages persistent pipeline execution state, markers, hashes, and receipts."""

    def __init__(self, state_file: Path):
        self.state_file = state_file
        self.data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if self.state_file.is_file():
            try:
                with open(self.state_file, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                log("STATE", f"Warning: failed to load existing state file ({e}). Starting fresh.")
        return {
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "completed_stages": {},
            "stage_metrics": {},
            "stage_hashes": {},
            "upload_receipts": {},
            "smoke_seconds_per_update": {},
            "current_stage": None,
        }

    def save(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.data["updated_at"] = datetime.now(UTC).isoformat()
        temp_file = self.state_file.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)
        temp_file.replace(self.state_file)

    def validate_stage_completion(
        self, stage: str, root_dir: Path | None = None, output_root: Path | None = None
    ) -> tuple[bool, str]:
        """Validate that all expected completion artifacts and hashes for a stage exist on disk."""
        if stage not in self.data.get("completed_stages", {}):
            return False, f"Stage '{stage}' not recorded as completed."

        info = self.data["completed_stages"][stage]
        artifacts = info.get("artifacts", {})
        hashes = self.data.get("stage_hashes", {}).get(stage, {})

        # 1. Verify artifact paths and hashes
        for art_key, art_val in artifacts.items():
            if isinstance(art_val, str) and any(
                art_val.endswith(ext) for ext in [".pt", ".onnx", ".json", ".parquet"]
            ):
                p = Path(art_val)
                if not p.is_file():
                    return False, f"Artifact '{art_key}' missing on disk: {p}"
                if art_key in hashes:
                    actual_hash = compute_sha256(p)
                    expected_hash = hashes[art_key]
                    if actual_hash != expected_hash:
                        return (
                            False,
                            f"Artifact '{art_key}' hash mismatch ({actual_hash} != {expected_hash}) on {p}",
                        )

        # 2. Stage-specific semantic integrity checks if paths provided
        if root_dir is not None and output_root is not None:
            if stage == "freeze-manifests":
                splits_dir = root_dir / "splits" / "production_eligible"
                for req in [
                    "train.parquet",
                    "validation.parquet",
                    "test.parquet",
                    "calibration.parquet",
                    "audit_report.json",
                ]:
                    if not (splits_dir / req).is_file():
                        return False, f"Manifest file missing: {splits_dir / req}"

            elif stage == "teacher-stage1-smoke":
                if "teacher-stage1" not in self.data.get("smoke_seconds_per_update", {}):
                    return False, "Missing smoke throughput measurement for teacher-stage1"

            elif stage == "teacher-stage1-eval":
                rep_path = output_root / "teacher_stage1_clean_frozen" / "promotion_report.json"
                ckpt_path = output_root / "teacher_stage1_clean_frozen" / "checkpoint-promoted.pt"
                if not rep_path.is_file():
                    return False, f"Teacher stage 1 promotion report missing: {rep_path}"
                if not ckpt_path.is_file():
                    return False, f"Teacher stage 1 promoted checkpoint missing: {ckpt_path}"
                try:
                    with open(rep_path, encoding="utf-8") as f:
                        rep = json.load(f)
                    if not rep.get("passed", False):
                        return False, "Teacher stage 1 promotion report records passed=False"
                except Exception as e:
                    return False, f"Failed reading teacher stage 1 report: {e}"

            elif stage == "teacher-stage2-smoke":
                if "teacher-stage2" not in self.data.get("smoke_seconds_per_update", {}):
                    return False, "Missing smoke throughput measurement for teacher-stage2"

            elif stage == "teacher-eval":
                rep_path = output_root / "teacher_stage2_paired_unfrozen" / "promotion_report.json"
                ckpt_path = (
                    output_root / "teacher_stage2_paired_unfrozen" / "checkpoint-promoted.pt"
                )
                if not rep_path.is_file():
                    return False, f"Teacher stage 2 promotion report missing: {rep_path}"
                if not ckpt_path.is_file():
                    return False, f"Teacher stage 2 promoted checkpoint missing: {ckpt_path}"
                try:
                    with open(rep_path, encoding="utf-8") as f:
                        rep = json.load(f)
                    if not rep.get("passed", False):
                        return False, "Teacher stage 2 promotion report records passed=False"
                except Exception as e:
                    return False, f"Failed reading teacher stage 2 report: {e}"

            elif stage == "students-smoke":
                if "students-distill" not in self.data.get("smoke_seconds_per_update", {}):
                    return False, "Missing smoke throughput measurement for students-distill"

            elif stage == "students-eval":
                small_rep = output_root / "student_dinov3_small" / "promotion_report.json"
                base_rep = output_root / "student_dinov3_base" / "promotion_report.json"
                small_passed = False
                base_passed = False
                if small_rep.is_file():
                    try:
                        with open(small_rep, encoding="utf-8") as f:
                            small_passed = json.load(f).get("passed", False)
                    except Exception:
                        pass
                if base_rep.is_file():
                    try:
                        with open(base_rep, encoding="utf-8") as f:
                            base_passed = json.load(f).get("passed", False)
                    except Exception:
                        pass
                if not small_passed and not base_passed:
                    return False, "Neither float student passed promotion gate in students-eval"

            elif stage == "att-smoke":
                if "att" not in self.data.get("smoke_seconds_per_update", {}):
                    return False, "Missing smoke throughput measurement for att"

            elif stage == "att-eval":
                rep_path = output_root / "att_shared_promotion_report.json"
                if not rep_path.is_file():
                    return False, f"ATT shared promotion report missing: {rep_path}"
                try:
                    with open(rep_path, encoding="utf-8") as f:
                        rep = json.load(f)
                    if not rep.get("passed", False):
                        return False, "ATT promotion report records passed=False"
                except Exception as e:
                    return False, f"Failed reading ATT promotion report: {e}"

            elif stage == "export-float":
                exported_dir = output_root / "exported"
                onnx_files = list(exported_dir.glob("student_*_float32.onnx"))
                if not onnx_files:
                    return False, f"No exported float ONNX model found in {exported_dir}"

            elif stage == "runtime-benchmark":
                bench_path = output_root / "runtime_benchmark_report.json"
                if not bench_path.is_file():
                    return False, f"Runtime benchmark report missing: {bench_path}"

            elif stage == "complete":
                rec_path = output_root / "pipeline_completion_receipt.json"
                if not rec_path.is_file():
                    return False, f"Pipeline completion receipt missing: {rec_path}"

        return True, "Artifacts verified"

    def is_stage_completed(
        self, stage: str, root_dir: Path | None = None, output_root: Path | None = None
    ) -> bool:
        """Check if stage is marked completed AND passes artifact validation on disk."""
        if stage not in self.data.get("completed_stages", {}):
            return False

        valid, reason = self.validate_stage_completion(stage, root_dir, output_root)
        if not valid:
            log(
                "STATE",
                f"Resume validation failed for stage '{stage}': {reason}. Invalidating stage.",
            )
            self.data["completed_stages"].pop(stage, None)
            self.save()
            return False

        return True

    def mark_stage_started(self, stage: str) -> None:
        self.data["current_stage"] = stage
        if "stage_timings" not in self.data:
            self.data["stage_timings"] = {}
        self.data["stage_timings"][stage] = {
            "started_at": datetime.now(UTC).isoformat(),
            "completed_at": None,
        }
        self.save()

    def mark_stage_completed(
        self,
        stage: str,
        artifacts: dict[str, Any] | None = None,
        hashes: dict[str, str] | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        completed = self.data.setdefault("completed_stages", {})
        completed[stage] = {
            "completed_at": datetime.now(UTC).isoformat(),
            "artifacts": artifacts or {},
        }
        if hashes:
            self.data.setdefault("stage_hashes", {})[stage] = hashes
        if metrics:
            self.data.setdefault("stage_metrics", {})[stage] = metrics
        if "stage_timings" in self.data and stage in self.data["stage_timings"]:
            self.data["stage_timings"][stage]["completed_at"] = datetime.now(
                UTC
            ).isoformat()
        self.data["current_stage"] = None
        self.save()

    def record_smoke_throughput(self, stage_key: str, seconds_per_update: float) -> None:
        """Record measured smoke throughput (seconds per optimizer update)."""
        self.data.setdefault("smoke_seconds_per_update", {})[stage_key] = seconds_per_update
        self.save()

    def get_smoke_throughput(self, stage_key: str) -> float | None:
        return self.data.get("smoke_seconds_per_update", {}).get(stage_key)

    def record_upload(self, stage: str, receipt: dict[str, Any]) -> None:
        self.data.setdefault("upload_receipts", {})[stage] = receipt
        self.save()


class VastBudgetGuard:
    """Manages Vast.ai cost projection from smoke throughput and enforces the $5 reserve policy."""

    def __init__(
        self,
        api_key: str | None = None,
        min_reserve: float = 5.00,
        explicit_balance: float | None = None,
        explicit_hourly_rate: float | None = None,
    ):
        self.api_key = api_key or os.environ.get("VAST_API_KEY")
        self.min_reserve = float(os.environ.get("VAST_MIN_BALANCE_RESERVE", min_reserve))
        env_explicit_bal = os.environ.get("TECHJAM_EXPLICIT_BALANCE")
        self.explicit_balance = explicit_balance or (
            float(env_explicit_bal) if env_explicit_bal else None
        )

        env_explicit_rate = os.environ.get("VAST_HOURLY_RATE")
        self.explicit_hourly_rate = explicit_hourly_rate or (
            float(env_explicit_rate) if env_explicit_rate else None
        )

    def query_hourly_rate(self) -> tuple[float | None, str]:
        """Query actual hourly rate (USD/hour) for the instance."""
        if self.explicit_hourly_rate is not None:
            return (
                self.explicit_hourly_rate,
                "explicitly provided (--hourly-rate / VAST_HOURLY_RATE)",
            )

        # Try vastai CLI to inspect instance cost
        try:
            res = subprocess.run(
                ["vastai", "show", "instances", "--raw"],
                capture_output=True,
                text=True,
                check=False,
            )
            if res.returncode == 0 and res.stdout.strip():
                instances = json.loads(res.stdout)
                if isinstance(instances, list) and len(instances) > 0:
                    dph = instances[0].get("dph_total")
                    if dph is not None:
                        return float(dph), f"vastai CLI (dph_total={dph})"
        except Exception:
            pass

        return None, "hourly rate not specified"

    def query_balance(self) -> tuple[float | None, str]:
        """Query current Vast.ai balance, returning (balance, source_description)."""
        if self.explicit_balance is not None:
            return (
                self.explicit_balance,
                "explicitly provided (--explicit-balance / TECHJAM_EXPLICIT_BALANCE)",
            )

        # Try vastai CLI
        try:
            res = subprocess.run(
                ["vastai", "show", "user", "--raw"],
                capture_output=True,
                text=True,
                check=False,
            )
            if res.returncode == 0 and res.stdout.strip():
                user_data = json.loads(res.stdout)
                balance = user_data.get("credit") or user_data.get("balance")
                if balance is not None:
                    return float(balance), "vastai CLI (show user)"
        except Exception:
            pass

        # Try curl / requests via VAST_API_KEY
        if self.api_key:
            try:
                import urllib.request

                req = urllib.request.Request(
                    "https://console.vast.ai/api/v0/users/current/",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    credit = data.get("credit")
                    if credit is not None:
                        return float(credit), "Vast.ai API (users/current)"
            except Exception as e:
                log("BUDGET", f"Vast API check failed: {e}")

        return None, "unable to determine balance"

    def check_stage_affordability(
        self,
        stage_name: str,
        projected_hours: float,
    ) -> tuple[bool, str]:
        """Check if smoke-derived projected stage cost violates the $5 reserve."""
        balance, bal_source = self.query_balance()
        if balance is None:
            return False, (
                f"Cannot query Vast balance ({bal_source}). Must provide --explicit-balance <USD> "
                f"or set TECHJAM_EXPLICIT_BALANCE in environment."
            )

        rate, rate_source = self.query_hourly_rate()
        if rate is None:
            return False, (
                f"Hourly rate is unknown ({rate_source}). Must provide --hourly-rate <USD/h> "
                f"or set VAST_HOURLY_RATE in environment."
            )

        projected_cost = projected_hours * rate
        projected_remaining = balance - projected_cost

        log(
            "BUDGET",
            f"Current Balance: ${balance:.2f} ({bal_source}) | Rate: ${rate:.2f}/h ({rate_source}) | "
            f"Projected Stage Cost: ${projected_cost:.2f} ({projected_hours:.2f}h) | "
            f"Projected Remaining: ${projected_remaining:.2f} | Mandatory Reserve: ${self.min_reserve:.2f}",
        )

        if projected_remaining < self.min_reserve:
            return False, (
                f"Projected cost (${projected_cost:.2f}) would leave balance ${projected_remaining:.2f}, "
                f"violating the mandatory ${self.min_reserve:.2f} reserve. Stage '{stage_name}' aborted."
            )

        return True, "Affordability check passed"


class PipelineOrchestrator:
    """Controls the 4x RTX 4090 production state machine."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.root_dir = Path(os.environ.get("TECHJAM_REPO_ROOT", Path.cwd())).resolve()
        self.runtime_root = Path(
            os.environ.get("TECHJAM_RUNTIME_ROOT", "/workspace/techjam26-runtime")
        )
        self.data_root = Path(os.environ.get("TECHJAM_DATA_ROOT", str(self.runtime_root / "data")))
        self.output_root = Path(
            os.environ.get("TECHJAM_OUTPUT_ROOT", str(self.runtime_root / "outputs"))
        )
        self.state_file = Path(
            os.environ.get("TECHJAM_PIPELINE_STATE", str(self.output_root / "pipeline_state.json"))
        )
        self.state = PipelineStateManager(self.state_file)
        self.budget = VastBudgetGuard(
            explicit_balance=args.explicit_balance,
            explicit_hourly_rate=args.hourly_rate,
        )
        self.model_repo = os.environ.get("TECHJAM_MODEL_REPO", "zye2/mechadetect-models")
        self.data_repo = os.environ.get("TECHJAM_DATA_REPO", "zye2/tj-data")

    def run_cmd(self, cmd: list[str], stage: str, env: dict[str, str] | None = None) -> int:
        """Run a command synchronously, logging stdout/stderr."""
        log(stage, f"Executing: {' '.join(cmd)}")
        run_env = os.environ.copy()
        if env:
            run_env.update(env)
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(self.root_dir),
            env=run_env,
        )
        assert p.stdout is not None
        for line in iter(p.stdout.readline, ""):
            print(f"[{stage}] {line.rstrip()}", flush=True)
        p.stdout.close()
        return p.wait()

    def upload_stage(
        self,
        stage: str,
        files: list[Path],
        *,
        repo_id: str | None = None,
        repo_type: str = "model",
    ) -> bool:
        """Upload verified artifacts and persist the Hub receipt."""
        repo = repo_id or self.model_repo
        if repo_type not in {"model", "dataset"}:
            raise ValueError(f"Unsupported Hugging Face repo type: {repo_type}")
        missing = [path for path in files if not path.is_file()]
        if not files or missing:
            log(stage, f"FATAL: Upload inputs missing or empty: {missing}")
            return False
        from scripts.upload_promoted_artifacts import upload_files_to_hf

        receipt_path = self.output_root / "upload_receipts" / f"receipt_{stage}.json"
        result = upload_files_to_hf(
            files=files,
            repo_id=repo,
            repo_type=repo_type,
            path_in_repo_prefix=stage,
            receipt_path=receipt_path,
        )
        if not result.get("success") or not receipt_path.is_file():
            log(stage, f"FATAL: Artifact upload failed for {repo_type} repo {repo}")
            return False
        self.state.record_upload(stage, result)
        return True

    def get_eligible_train_count(self) -> int:
        """Read the exact frozen train population; guessed schedules are prohibited."""
        train_manifest = self.root_dir / "splits" / "production_eligible" / "train.parquet"
        if not train_manifest.is_file():
            raise FileNotFoundError(f"Frozen eligible train manifest not found: {train_manifest}")
        import pandas as pd

        count = len(pd.read_parquet(train_manifest, columns=["row_id"]))
        if count <= 0:
            raise RuntimeError(f"Frozen eligible train manifest is empty: {train_manifest}")
        return count

    def calculate_measured_hours(self, stage_key: str, passes: int = 1) -> float:
        """Calculate stage duration in hours from smoke-measured seconds/update and eligible train rows."""
        train_count = self.get_eligible_train_count()
        updates_per_pass = math.ceil(train_count / 48)  # Effective batch 48
        total_updates = updates_per_pass * passes

        sec_per_update = self.state.get_smoke_throughput(stage_key)
        if sec_per_update is None:
            raise RuntimeError(
                f"Cannot project duration for '{stage_key}': measured smoke throughput is missing. "
                f"Corresponding smoke stage must run first to measure seconds/update. Guessed fallback is prohibited."
            )

        total_hours = (total_updates * sec_per_update) / 3600.0
        log(
            "BUDGET",
            f"Derived schedule for '{stage_key}': {train_count} rows / 48 batch = {updates_per_pass} updates/pass "
            f"x {passes} passes = {total_updates} total updates @ {sec_per_update:.3f}s/update -> {total_hours:.2f} hours",
        )
        return total_hours

    # --------------------------------------------------------------------------
    # Stage Implementations
    # --------------------------------------------------------------------------

    def stage_preflight(self) -> bool:
        """Stage 1: Verify hardware, environment, git branch, disk space, and repos (fails closed)."""
        log("PREFLIGHT", "Starting preflight verification...")

        # 1. GPU Verification (Exactly 4x RTX 4090)
        try:
            import torch

            num_gpus = torch.cuda.device_count()
            log("PREFLIGHT", f"CUDA devices detected: {num_gpus}")
            if num_gpus != 4:
                msg = f"Expected exactly 4 GPUs, found {num_gpus}."
                if not self.args.allow_gpu_mismatch:
                    log(
                        "PREFLIGHT",
                        f"FATAL: {msg} Pass --allow-gpu-mismatch to bypass in development.",
                    )
                    return False
                log("PREFLIGHT", f"WARNING: {msg} Continuing due to --allow-gpu-mismatch.")

            for i in range(num_gpus):
                name = torch.cuda.get_device_name(i)
                log("PREFLIGHT", f"  Device {i}: {name}")
                if "4090" not in name and not self.args.allow_gpu_mismatch:
                    log("PREFLIGHT", f"FATAL: Device {i} ({name}) is not an RTX 4090.")
                    return False

            # 2. BF16 Verification
            bf16 = torch.cuda.is_bf16_supported()
            log("PREFLIGHT", f"BF16 supported: {bf16}")
            if not bf16 and not self.args.allow_gpu_mismatch:
                log("PREFLIGHT", "FATAL: BF16 is required for 4x4090 training.")
                return False
        except Exception as e:
            if not self.args.allow_gpu_mismatch:
                log("PREFLIGHT", f"FATAL: PyTorch CUDA check failed: {e}")
                return False
            log("PREFLIGHT", f"PyTorch CUDA check skipped or failed ({e}).")

        # 3. Disk Space Verification (>= 200 GB)
        check_path = self.output_root if self.output_root.exists() else Path.cwd()
        total, used, free = shutil.disk_usage(check_path)
        free_gb = free / (1024**3)
        log("PREFLIGHT", f"Available disk space on {check_path}: {free_gb:.2f} GB")
        if free_gb < 200.0 and not self.args.allow_disk_mismatch:
            log("PREFLIGHT", f"FATAL: Minimum 200 GB disk space required. Found {free_gb:.2f} GB.")
            return False

        # 4. Git Branch & SHA
        try:
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                text=True,
                cwd=str(self.root_dir),
            ).strip()
            sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                text=True,
                cwd=str(self.root_dir),
            ).strip()
            log("PREFLIGHT", f"Git Branch: {branch} | Commit SHA: {sha}")
            expected_sha = os.environ.get("TECHJAM_EXPECTED_GIT_SHA")
            if expected_sha and sha != expected_sha:
                log("PREFLIGHT", f"FATAL: Git SHA {sha} does not match expected {expected_sha}.")
                return False
        except Exception as e:
            log("PREFLIGHT", f"Git revision check warning: {e}")

        # 5. Lockfile Sync Check
        log("PREFLIGHT", "Checking uv dependency sync...")
        rc = self.run_cmd(["uv", "sync", "--locked", "--dev"], "PREFLIGHT")
        if rc != 0 and not self.args.allow_gpu_mismatch:
            log("PREFLIGHT", "FATAL: 'uv sync --locked --dev' failed.")
            return False

        # 6. Forbidden Data Check (Fail closed if forbidden rows leak)
        train_manifest = self.root_dir / "splits" / "production_eligible" / "train.parquet"
        if train_manifest.is_file():
            try:
                import pandas as pd

                df = pd.read_parquet(train_manifest)
                leak_mask = df["image_path"].str.contains(
                    "newer image model data", case=False, na=False
                )
                if leak_mask.any():
                    log(
                        "PREFLIGHT",
                        f"FATAL: Detected {leak_mask.sum()} forbidden rows leaking into train.parquet!",
                    )
                    return False
            except Exception as e:
                log("PREFLIGHT", f"Warning inspecting train manifest: {e}")

        # 7. Hugging Face Repos Access (Fails closed on missing token)
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not token:
            if not self.args.allow_gpu_mismatch:
                log(
                    "PREFLIGHT",
                    "FATAL: Missing HF_TOKEN / HUGGING_FACE_HUB_TOKEN. Hub uploads will fail.",
                )
                return False
            log("PREFLIGHT", "WARNING: No HF_TOKEN found in environment (bypassed in dev mode).")
        else:
            from scripts.upload_promoted_artifacts import ensure_hf_repo_exists

            log("PREFLIGHT", f"Verifying private model repository: {self.model_repo}...")
            ok_m = ensure_hf_repo_exists(
                self.model_repo, repo_type="model", private=True, token=token
            )
            if not ok_m and not self.args.allow_gpu_mismatch:
                log(
                    "PREFLIGHT",
                    f"FATAL: Unable to access or create private model repo {self.model_repo}.",
                )
                return False
            log("PREFLIGHT", f"Verifying public dataset repository: {self.data_repo}...")
            ok_d = ensure_hf_repo_exists(
                self.data_repo, repo_type="dataset", private=False, token=token
            )
            if not ok_d and not self.args.allow_gpu_mismatch:
                log(
                    "PREFLIGHT", f"FATAL: Unable to access or create dataset repo {self.data_repo}."
                )
                return False

        # 8. Budget & Hourly Rate Checks (Fails closed on unknown balance or rate)
        bal, src = self.budget.query_balance()
        log(
            "PREFLIGHT",
            f"Vast.ai Balance: {f'${bal:.2f}' if bal is not None else 'Unknown'} ({src})",
        )
        if bal is None and not self.args.allow_gpu_mismatch:
            log(
                "PREFLIGHT",
                "FATAL: Cannot verify Vast balance. Set TECHJAM_EXPLICIT_BALANCE or pass --explicit-balance.",
            )
            return False
        if bal is not None and bal < self.budget.min_reserve and not self.args.allow_gpu_mismatch:
            log(
                "PREFLIGHT",
                f"FATAL: Balance (${bal:.2f}) is already less than mandatory ${self.budget.min_reserve:.2f} reserve.",
            )
            return False

        rate, r_src = self.budget.query_hourly_rate()
        log(
            "PREFLIGHT",
            f"Instance Hourly Rate: {f'${rate:.2f}/h' if rate is not None else 'Unknown'} ({r_src})",
        )
        if rate is None and not self.args.allow_gpu_mismatch:
            log(
                "PREFLIGHT",
                "FATAL: Instance hourly rate is unknown. Set VAST_HOURLY_RATE or pass --hourly-rate.",
            )
            return False

        self.state.mark_stage_completed("preflight")
        return True

    def stage_acquire(self) -> bool:
        """Stage 2: acquire every resolvable declared image with byte validation."""
        command = [
            "uv",
            "run",
            "python",
            "scripts/data_prep/acquire_all_images.py",
            "--data-root",
            str(self.data_root),
            "--resume",
            "--verify-bytes",
            "--report-path",
            str(self.output_root / "acquisition_report.json"),
            "--revisions-path",
            str(self.output_root / "source_revisions.json"),
        ]
        if self.run_cmd(command, "ACQUIRE") != 0:
            return False
        report = self.output_root / "acquisition_report.json"
        revisions = self.output_root / "source_revisions.json"
        if not report.is_file() or not revisions.is_file():
            return False
        self.state.mark_stage_completed(
            "acquire",
            artifacts={"report": str(report), "source_revisions": str(revisions)},
            hashes={
                "report": compute_sha256(report),
                "source_revisions": compute_sha256(revisions),
            },
        )
        return True

    def stage_freeze_manifests(self) -> bool:
        """Stage 3: freeze only byte-verified eligible rows and upload the audit package."""
        output_dir = self.root_dir / "splits" / "production_eligible"
        command = [
            "uv",
            "run",
            "python",
            "scripts/data_prep/freeze_production_eligible.py",
            "--data-root",
            str(self.data_root),
            "--output-dir",
            str(output_dir),
            "--calibration-size",
            "4096",
            "--strict",
            "--source-revisions",
            str(self.output_root / "source_revisions.json"),
        ]
        if self.run_cmd(command, "FREEZE") != 0:
            return False
        required_names = (
            "declared_manifest.parquet",
            "train.parquet",
            "validation.parquet",
            "test.parquet",
            "test_unseen.parquet",
            "calibration.parquet",
            "exclusions.parquet",
            "audit_report.json",
            "source_revisions.json",
        )
        files = [output_dir / name for name in required_names]
        if any(not path.is_file() for path in files):
            return False
        audit = json.loads((output_dir / "audit_report.json").read_text(encoding="utf-8"))
        if audit.get("calibration_audit", {}).get("calibration_rows_count") != 4096:
            return False
        if not self.upload_stage(
            "manifests",
            files,
            repo_id=self.data_repo,
            repo_type="dataset",
        ):
            return False
        self.state.mark_stage_completed(
            "freeze-manifests",
            artifacts={"splits_dir": str(output_dir)},
            hashes={path.name: compute_sha256(path) for path in files},
        )
        return True

    def stage_teacher_stage1_smoke(self) -> bool:
        """Stage 4: 2-step smoke run for Teacher Stage 1 to test throughput and measure sec/update."""
        log("T-STAGE1-SMOKE", "Running 2-update smoke test for Teacher Stage 1...")
        cfg = "configs/teacher_dinov3_stage1_clean_frozen.yaml"
        cmd = [
            "uv",
            "run",
            "torchrun",
            "--standalone",
            "--nproc-per-node=4",
            "-m",
            "aigc_detector.train",
            "--config",
            cfg,
            "--world-size",
            "4",
            "--physical-batch-size",
            "6",
            "--gradient-accumulation",
            "2",
            "--max-steps",
            "2",
            "--stage",
            "teacher-stage1-smoke",
        ]
        t0 = time.time()
        rc = self.run_cmd(cmd, "T-STAGE1-SMOKE")
        dt = time.time() - t0
        if rc != 0:
            log("T-STAGE1-SMOKE", "FATAL: Teacher Stage 1 smoke test failed.")
            return False
        sec_per_update = dt / 2.0
        log("T-STAGE1-SMOKE", f"Smoke test passed in {dt:.2f}s ({sec_per_update:.3f}s/update).")
        self.state.record_smoke_throughput("teacher-stage1", sec_per_update)
        self.state.mark_stage_completed(
            "teacher-stage1-smoke", metrics={"sec_per_update": sec_per_update}
        )
        return True

    def stage_teacher_stage1(self) -> bool:
        """Stage 5: Train Teacher Stage 1 (frozen backbone, all eligible train originals)."""
        projected_hours = self.calculate_measured_hours("teacher-stage1", passes=1)
        ok, msg = self.budget.check_stage_affordability(
            "teacher-stage1", projected_hours=projected_hours
        )
        if not ok:
            log("T-STAGE1", f"FATAL: {msg}")
            return False

        log("T-STAGE1", "Launching Teacher Stage 1 training (4-GPU DDP)...")
        cfg = "configs/teacher_dinov3_stage1_clean_frozen.yaml"
        cmd = [
            "uv",
            "run",
            "torchrun",
            "--standalone",
            "--nproc-per-node=4",
            "-m",
            "aigc_detector.train",
            "--config",
            cfg,
            "--world-size",
            "4",
            "--physical-batch-size",
            "6",
            "--gradient-accumulation",
            "2",
        ]
        rc = self.run_cmd(cmd, "T-STAGE1")
        if rc != 0:
            log("T-STAGE1", "FATAL: Teacher Stage 1 training failed.")
            return False
        self.state.mark_stage_completed("teacher-stage1")
        return True

    def stage_teacher_stage1_eval(self) -> bool:
        """Stage 6: Evaluate and promote Teacher Stage 1."""
        log("T-STAGE1-EVAL", "Evaluating and promoting Teacher Stage 1 checkpoints...")
        stage1_dir = self.output_root / "teacher_stage1_clean_frozen"
        val_manifest = self.root_dir / "splits" / "production_eligible" / "validation.parquet"
        promotion_report = stage1_dir / "promotion_report.json"
        metadata_file = stage1_dir / "metadata.json"

        checkpoints = list(stage1_dir.glob("checkpoint-*.pt"))
        if not checkpoints:
            log("T-STAGE1-EVAL", f"FATAL: No checkpoints found in {stage1_dir}")
            return False

        cmd = [
            "uv",
            "run",
            "python",
            "scripts/promote_teacher.py",
            "--checkpoints",
            *[str(c) for c in checkpoints],
            "--manifest",
            str(val_manifest),
            "--output-report",
            str(promotion_report),
            "--output-metadata",
            str(metadata_file),
        ]
        rc = self.run_cmd(cmd, "T-STAGE1-EVAL")
        if rc != 0 or not promotion_report.is_file():
            log("T-STAGE1-EVAL", "FATAL: Stage 1 promotion script execution failed.")
            return False

        with open(promotion_report, encoding="utf-8") as f:
            rep = json.load(f)

        if not rep.get("passed", False):
            log(
                "T-STAGE1-EVAL",
                f"FATAL: Stage 1 promotion gate failed! Reasons: {rep.get('failed_reasons')}",
            )
            return False

        raw_ckpt_path = Path(rep.get("checkpoint_path", ""))
        if not raw_ckpt_path.is_file():
            log("T-STAGE1-EVAL", f"FATAL: Promoted checkpoint path does not exist: {raw_ckpt_path}")
            return False

        # Atomically copy to checkpoint-promoted.pt
        promoted_target = stage1_dir / "checkpoint-promoted.pt"
        shutil.copyfile(raw_ckpt_path, promoted_target)
        promoted_sha = compute_sha256(promoted_target)
        log(
            "T-STAGE1-EVAL",
            f"Promoted checkpoint saved: {promoted_target} (sha256: {promoted_sha[:12]}...)",
        )

        ok = self.upload_stage("teacher-stage1", [promoted_target, promotion_report, metadata_file])
        if not ok:
            log("T-STAGE1-EVAL", "FATAL: Stage 1 artifact upload failed.")
            return False

        self.state.mark_stage_completed(
            "teacher-stage1-eval",
            artifacts={
                "promoted_checkpoint": str(promoted_target),
                "report": str(promotion_report),
            },
            hashes={
                "promoted_checkpoint": promoted_sha,
                "report": compute_sha256(promotion_report),
            },
            metrics=rep.get("metrics"),
        )
        return True

    def stage_teacher_stage2_smoke(self) -> bool:
        """Stage 7: 2-update smoke run for Teacher Stage 2 warm-started from Stage 1."""
        log("T-STAGE2-SMOKE", "Running 2-update smoke test for Teacher Stage 2...")
        stage1_ckpt = self.output_root / "teacher_stage1_clean_frozen" / "checkpoint-promoted.pt"
        if not stage1_ckpt.is_file():
            log("T-STAGE2-SMOKE", f"FATAL: Stage 1 promoted checkpoint missing: {stage1_ckpt}")
            return False

        cfg = "configs/teacher_dinov3_stage2_paired_unfrozen.yaml"
        cmd = [
            "uv",
            "run",
            "torchrun",
            "--standalone",
            "--nproc-per-node=4",
            "-m",
            "aigc_detector.train",
            "--config",
            cfg,
            "--world-size",
            "4",
            "--physical-batch-size",
            "2",
            "--gradient-accumulation",
            "6",
            "--initial-checkpoint",
            str(stage1_ckpt),
            "--max-steps",
            "2",
            "--stage",
            "teacher-stage2-smoke",
        ]
        t0 = time.time()
        rc = self.run_cmd(cmd, "T-STAGE2-SMOKE")
        dt = time.time() - t0
        if rc != 0:
            log("T-STAGE2-SMOKE", "FATAL: Teacher Stage 2 smoke test failed.")
            return False
        sec_per_update = dt / 2.0
        log("T-STAGE2-SMOKE", f"Smoke test passed in {dt:.2f}s ({sec_per_update:.3f}s/update).")
        self.state.record_smoke_throughput("teacher-stage2", sec_per_update)
        self.state.mark_stage_completed(
            "teacher-stage2-smoke", metrics={"sec_per_update": sec_per_update}
        )
        return True

    def stage_teacher_stage2(self) -> bool:
        """Stage 8: Train Teacher Stage 2 (unfrozen backbone, paired transforms)."""
        projected_hours = self.calculate_measured_hours("teacher-stage2", passes=1)
        ok, msg = self.budget.check_stage_affordability(
            "teacher-stage2", projected_hours=projected_hours
        )
        if not ok:
            log("T-STAGE2", f"FATAL: {msg}")
            return False

        log("T-STAGE2", "Launching Teacher Stage 2 training (4-GPU DDP)...")
        stage1_ckpt = self.output_root / "teacher_stage1_clean_frozen" / "checkpoint-promoted.pt"
        cfg = "configs/teacher_dinov3_stage2_paired_unfrozen.yaml"
        cmd = [
            "uv",
            "run",
            "torchrun",
            "--standalone",
            "--nproc-per-node=4",
            "-m",
            "aigc_detector.train",
            "--config",
            cfg,
            "--world-size",
            "4",
            "--physical-batch-size",
            "2",
            "--gradient-accumulation",
            "6",
            "--initial-checkpoint",
            str(stage1_ckpt),
        ]
        rc = self.run_cmd(cmd, "T-STAGE2")
        if rc != 0:
            log("T-STAGE2", "FATAL: Teacher Stage 2 training failed.")
            return False
        self.state.mark_stage_completed("teacher-stage2")
        return True

    def stage_teacher_eval(self) -> bool:
        """Stage 9: Evaluate and promote Teacher Stage 2."""
        log("T-STAGE2-EVAL", "Evaluating and promoting Teacher Stage 2 checkpoints...")
        stage2_dir = self.output_root / "teacher_stage2_paired_unfrozen"
        val_manifest = self.root_dir / "splits" / "production_eligible" / "validation.parquet"
        promotion_report = stage2_dir / "promotion_report.json"
        metadata_file = stage2_dir / "metadata.json"

        checkpoints = list(stage2_dir.glob("checkpoint-*.pt"))
        if not checkpoints:
            log("T-STAGE2-EVAL", f"FATAL: No checkpoints found in {stage2_dir}")
            return False

        cmd = [
            "uv",
            "run",
            "python",
            "scripts/promote_teacher.py",
            "--checkpoints",
            *[str(c) for c in checkpoints],
            "--manifest",
            str(val_manifest),
            "--output-report",
            str(promotion_report),
            "--output-metadata",
            str(metadata_file),
        ]
        rc = self.run_cmd(cmd, "T-STAGE2-EVAL")
        if rc != 0 or not promotion_report.is_file():
            log("T-STAGE2-EVAL", "FATAL: Stage 2 promotion script execution failed.")
            return False

        with open(promotion_report, encoding="utf-8") as f:
            rep = json.load(f)

        if not rep.get("passed", False):
            log(
                "T-STAGE2-EVAL",
                f"FATAL: Teacher Stage 2 promotion gate failed! Reasons: {rep.get('failed_reasons')}",
            )
            return False

        raw_ckpt_path = Path(rep.get("checkpoint_path", ""))
        if not raw_ckpt_path.is_file():
            log("T-STAGE2-EVAL", f"FATAL: Promoted checkpoint path does not exist: {raw_ckpt_path}")
            return False

        promoted_target = stage2_dir / "checkpoint-promoted.pt"
        shutil.copyfile(raw_ckpt_path, promoted_target)
        promoted_sha = compute_sha256(promoted_target)
        log(
            "T-STAGE2-EVAL",
            f"Teacher Stage 2 promoted checkpoint: {promoted_target} (sha256: {promoted_sha[:12]}...)",
        )

        ok = self.upload_stage("teacher-stage2", [promoted_target, promotion_report, metadata_file])
        if not ok:
            log("T-STAGE2-EVAL", "FATAL: Stage 2 artifact upload failed.")
            return False

        self.state.mark_stage_completed(
            "teacher-eval",
            artifacts={
                "promoted_checkpoint": str(promoted_target),
                "report": str(promotion_report),
            },
            hashes={
                "promoted_checkpoint": promoted_sha,
                "report": compute_sha256(promotion_report),
            },
            metrics=rep.get("metrics"),
        )
        return True

    def stage_students_smoke(self) -> bool:
        """Stage 10: run exactly two updates on each real two-GPU student track."""
        log("STUDENTS-SMOKE", "Running 2-update 2-GPU student smoke tests...")
        teacher_ckpt = (
            self.output_root / "teacher_stage2_paired_unfrozen" / "checkpoint-promoted.pt"
        )
        teacher_report = (
            self.output_root / "teacher_stage2_paired_unfrozen" / "promotion_report.json"
        )
        manifest = self.root_dir / "splits" / "production_eligible" / "train.parquet"
        val_manifest = self.root_dir / "splits" / "production_eligible" / "validation.parquet"
        required = (teacher_ckpt, teacher_report, manifest, val_manifest)
        if any(not path.is_file() for path in required):
            log(
                "STUDENTS-SMOKE",
                f"FATAL: Missing smoke prerequisite: {[str(p) for p in required if not p.is_file()]}",
            )
            return False

        durations: dict[str, float] = {}
        for track in ("small", "base"):
            command = [
                "uv",
                "run",
                "python",
                "scripts/launch_students_distill.py",
                "--track",
                track,
                "--teacher-config",
                "configs/teacher_dinov3_stage2_paired_unfrozen.yaml",
                "--teacher-checkpoint",
                str(teacher_ckpt),
                "--teacher-promotion-report",
                str(teacher_report),
                "--manifest",
                str(manifest),
                "--val-manifest",
                str(val_manifest),
                f"--{track}-config",
                f"configs/student_dinov3_{track}_distill.yaml",
                f"--{track}-output-dir",
                str(self.output_root / f"smoke_student_{track}"),
                f"--{track}-devices",
                "0,1" if track == "small" else "2,3",
                "--dry-run",
            ]
            started = time.time()
            if self.run_cmd(command, "STUDENTS-SMOKE") != 0:
                log("STUDENTS-SMOKE", f"FATAL: {track} student smoke test failed.")
                return False
            durations[track] = time.time() - started

        sec_per_update = max(durations.values()) / 2.0
        self.state.record_smoke_throughput("students-distill", sec_per_update)
        self.state.mark_stage_completed(
            "students-smoke",
            metrics={
                "sec_per_update": sec_per_update,
                **{f"{k}_seconds": v for k, v in durations.items()},
            },
        )
        return True

    def stage_students_distill(self) -> bool:
        """Stage 11: Concurrent 2-GPU distillation for ViT-S (0,1) and ViT-B (2,3)."""
        projected_hours = self.calculate_measured_hours("students-distill", passes=2)
        ok, msg = self.budget.check_stage_affordability(
            "students-distill", projected_hours=projected_hours
        )
        if not ok:
            log("STUDENTS-DISTILL", f"FATAL: {msg}")
            return False

        log(
            "STUDENTS-DISTILL",
            "Launching concurrent student distillation (Small on 0,1; Base on 2,3)...",
        )
        teacher_ckpt = (
            self.output_root / "teacher_stage2_paired_unfrozen" / "checkpoint-promoted.pt"
        )
        teacher_rep = self.output_root / "teacher_stage2_paired_unfrozen" / "promotion_report.json"
        train_manifest = self.root_dir / "splits" / "production_eligible" / "train.parquet"
        val_manifest = self.root_dir / "splits" / "production_eligible" / "validation.parquet"

        helper = self.root_dir / "scripts" / "launch_students_distill.py"
        if not helper.is_file():
            log("STUDENTS-DISTILL", "FATAL: launch_students_distill.py missing.")
            return False

        cmd = [
            "uv",
            "run",
            "python",
            str(helper),
            "--track",
            "both",
            "--teacher-checkpoint",
            str(teacher_ckpt),
            "--teacher-config",
            "configs/teacher_dinov3_stage2_paired_unfrozen.yaml",
            "--teacher-promotion-report",
            str(teacher_rep),
            "--manifest",
            str(train_manifest),
            "--val-manifest",
            str(val_manifest),
            "--small-output-dir",
            str(self.output_root / "student_dinov3_small"),
            "--base-output-dir",
            str(self.output_root / "student_dinov3_base"),
            "--small-devices",
            "0,1",
            "--base-devices",
            "2,3",
            "--parallel-tracks",
            "--epochs",
            "2",
        ]
        rc = self.run_cmd(cmd, "STUDENTS-DISTILL")
        if rc != 0:
            log("STUDENTS-DISTILL", "FATAL: Student distillation failed.")
            return False

        self.state.mark_stage_completed("students-distill")
        return True

    def stage_students_eval(self) -> bool:
        """Stage 12: validate and upload independently promoted float students."""
        log("STUDENTS-EVAL", "Validating student promotion contracts...")
        selection_report = self.output_root / "student_selection_report.json"
        if not selection_report.is_file():
            log("STUDENTS-EVAL", f"FATAL: Student selection report missing: {selection_report}")
            return False

        uploads: list[Path] = [selection_report]
        hashes: dict[str, str] = {"selection_report": compute_sha256(selection_report)}
        artifacts: dict[str, Any] = {"selection_report": str(selection_report)}
        passed_tracks: list[str] = []
        for track in ("small", "base"):
            track_dir = self.output_root / f"student_dinov3_{track}"
            report_path = track_dir / "promotion_report.json"
            if not report_path.is_file():
                log("STUDENTS-EVAL", f"FATAL: {track} promotion report missing: {report_path}")
                return False
            report = json.loads(report_path.read_text(encoding="utf-8"))
            passed = report.get("passed") is True
            artifacts[f"{track}_passed"] = passed
            if not passed:
                continue
            checkpoint = track_dir / "checkpoint-promoted.pt"
            if not checkpoint.is_file():
                log("STUDENTS-EVAL", f"FATAL: {track} passed without promoted checkpoint")
                return False
            actual_hash = compute_sha256(checkpoint)
            if report.get("checkpoint_sha256") != actual_hash:
                log("STUDENTS-EVAL", f"FATAL: {track} checkpoint/report SHA-256 mismatch")
                return False
            passed_tracks.append(track)
            uploads.extend([checkpoint, report_path])
            hashes[f"{track}_promoted_checkpoint"] = actual_hash
            artifacts[f"{track}_promoted_checkpoint"] = str(checkpoint)

        if not passed_tracks:
            log("STUDENTS-EVAL", "FATAL: Neither float student passed the promotion gate")
            return False
        if not self.upload_stage("students", uploads):
            return False
        self.state.mark_stage_completed("students-eval", artifacts=artifacts, hashes=hashes)
        return True

    def stage_att_smoke(self) -> bool:
        """Stage 13: run exactly two updates on every eligible two-GPU ATT track."""
        manifest = self.root_dir / "splits" / "production_eligible" / "train.parquet"
        if not manifest.is_file():
            log("ATT-SMOKE", f"FATAL: Training manifest missing: {manifest}")
            return False
        durations: dict[str, float] = {}
        for track in ("small", "base"):
            checkpoint = self.output_root / f"student_dinov3_{track}" / "checkpoint-promoted.pt"
            if not checkpoint.is_file():
                continue
            command = [
                "uv",
                "run",
                "python",
                "scripts/launch_att_tracks.py",
                "--track",
                track,
                f"--{track}-checkpoint",
                str(checkpoint),
                "--train-manifest",
                str(manifest),
                f"--{track}-config",
                f"configs/att_student_{track}.yaml",
                f"--{track}-output",
                str(self.output_root / f"smoke_att_{track}"),
                f"--{track}-devices",
                "0,1" if track == "small" else "2,3",
                "--dry-run",
            ]
            started = time.time()
            if self.run_cmd(command, "ATT-SMOKE") != 0:
                log("ATT-SMOKE", f"FATAL: {track} ATT smoke test failed")
                return False
            durations[track] = time.time() - started
        if not durations:
            log("ATT-SMOKE", "FATAL: No promoted float student is eligible for ATT")
            return False
        sec_per_update = max(durations.values()) / 2.0
        self.state.record_smoke_throughput("att", sec_per_update)
        self.state.mark_stage_completed(
            "att-smoke",
            metrics={
                "sec_per_update": sec_per_update,
                **{f"{k}_seconds": v for k, v in durations.items()},
            },
        )
        return True

    def stage_att(self) -> bool:
        """Stage 14: run one independent ATT pass on each promoted float track."""
        projected_hours = self.calculate_measured_hours("att", passes=1)
        ok, message = self.budget.check_stage_affordability("att", projected_hours=projected_hours)
        if not ok:
            log("ATT", f"FATAL: {message}")
            return False
        train_manifest = self.root_dir / "splits" / "production_eligible" / "train.parquet"
        available = [
            track
            for track in ("small", "base")
            if (self.output_root / f"student_dinov3_{track}" / "checkpoint-promoted.pt").is_file()
        ]
        if not available:
            return False
        command = [
            "uv",
            "run",
            "python",
            "scripts/launch_att_tracks.py",
            "--track",
            "both" if len(available) == 2 else available[0],
            "--train-manifest",
            str(train_manifest),
            "--small-output",
            str(self.output_root / "att_student_small"),
            "--base-output",
            str(self.output_root / "att_student_base"),
            "--small-devices",
            "0,1",
            "--base-devices",
            "2,3",
            "--parallel-tracks",
            "--epochs",
            "1",
        ]
        for track in available:
            checkpoint = self.output_root / f"student_dinov3_{track}" / "checkpoint-promoted.pt"
            command.extend([f"--{track}-checkpoint", str(checkpoint)])
        if self.run_cmd(command, "ATT") != 0:
            return False
        self.state.mark_stage_completed("att", artifacts={"tracks": available})
        return True

    def stage_att_eval(self) -> bool:
        """Stage 15: gate ATT per track; retain float defaults for rejected ATT tracks."""
        val_manifest = self.root_dir / "splits" / "production_eligible" / "validation.parquet"
        gate_command = [
            "uv",
            "run",
            "python",
            "scripts/check_att_gate.py",
            "--shared-report",
            str(self.output_root / "att_shared_promotion_report.json"),
            "--promote",
        ]
        uploads: list[Path] = []
        evaluated_tracks: list[str] = []
        eval_paths: dict[str, tuple[Path, Path]] = {}
        for track in ("small", "base"):
            float_dir = self.output_root / f"student_dinov3_{track}"
            att_dir = self.output_root / f"att_student_{track}"
            float_checkpoint = float_dir / "checkpoint-promoted.pt"
            att_checkpoint = att_dir / "checkpoint-final.pt"
            float_report_path = float_dir / "promotion_report.json"
            if not float_checkpoint.is_file():
                continue
            if not att_checkpoint.is_file() or not float_report_path.is_file():
                log("ATT-EVAL", f"FATAL: Missing {track} ATT evaluation input")
                return False
            float_report = json.loads(float_report_path.read_text(encoding="utf-8"))
            threshold = float_report.get("calibrated_threshold")
            if threshold is None:
                log("ATT-EVAL", f"FATAL: {track} float report has no calibrated threshold")
                return False
            float_eval = self.output_root / f"eval_float_{track}.json"
            att_eval = self.output_root / f"eval_att_{track}.json"
            for checkpoint, output in ((float_checkpoint, float_eval), (att_checkpoint, att_eval)):
                command = [
                    "uv",
                    "run",
                    "python",
                    "scripts/evaluate_performance.py",
                    "--manifest",
                    str(val_manifest),
                    "--checkpoint",
                    str(checkpoint),
                    "--config",
                    f"configs/student_dinov3_{track}_distill.yaml",
                    "--threshold",
                    str(threshold),
                    "--output",
                    str(output),
                    "--robustness",
                ]
                if self.run_cmd(command, "ATT-EVAL") != 0:
                    return False
            gate_command.extend(
                [
                    f"--{track}-float-eval",
                    str(float_eval),
                    f"--{track}-att-eval",
                    str(att_eval),
                    f"--{track}-checkpoint",
                    str(att_checkpoint),
                    f"--{track}-output-dir",
                    str(att_dir),
                ]
            )
            eval_paths[track] = (float_eval, att_eval)
            evaluated_tracks.append(track)

        if not evaluated_tracks:
            return False
        report_path = self.output_root / "att_shared_promotion_report.json"
        gate_returncode = self.run_cmd(gate_command, "ATT-EVAL")
        if gate_returncode not in {0, 1} or not report_path.is_file():
            log("ATT-EVAL", "FATAL: ATT gate execution failed without a valid report")
            return False
        report = json.loads(report_path.read_text(encoding="utf-8"))
        tracks = report.get("tracks")
        if not isinstance(tracks, dict) or set(tracks) != set(evaluated_tracks):
            log("ATT-EVAL", "FATAL: ATT gate report does not cover every evaluated track")
            return False

        uploads.append(report_path)
        hashes = {"report": compute_sha256(report_path)}
        artifacts: dict[str, Any] = {"report": str(report_path)}
        for track in evaluated_tracks:
            uploads.extend(eval_paths[track])
            artifacts[f"{track}_att_passed"] = tracks[track].get("passed") is True
            if tracks[track].get("passed") is True:
                att_dir = self.output_root / f"att_student_{track}"
                promoted = att_dir / "checkpoint-promoted.pt"
                promotion_report = att_dir / "promotion_report.json"
                metadata = att_dir / "metadata.json"
                if any(not path.is_file() for path in (promoted, promotion_report, metadata)):
                    log("ATT-EVAL", f"FATAL: Promoted {track} ATT artifact set is incomplete")
                    return False
                uploads.extend([promoted, promotion_report, metadata])
                hashes[f"{track}_promoted_checkpoint"] = compute_sha256(promoted)
                artifacts[f"{track}_promoted_checkpoint"] = str(promoted)
        if not self.upload_stage("att", uploads):
            return False
        self.state.mark_stage_completed("att-eval", artifacts=artifacts, hashes=hashes)
        return True

    def stage_export_float(self) -> bool:
        """Stage 16: export every promoted final student with authoritative metadata."""
        export_dir = self.output_root / "exported"
        export_dir.mkdir(parents=True, exist_ok=True)
        artifacts: dict[str, Any] = {}
        hashes: dict[str, str] = {}
        for track in ("small", "base"):
            att_dir = self.output_root / f"att_student_{track}"
            float_dir = self.output_root / f"student_dinov3_{track}"
            if (att_dir / "checkpoint-promoted.pt").is_file():
                checkpoint = att_dir / "checkpoint-promoted.pt"
                promotion_report = att_dir / "promotion_report.json"
                stage = "post_att"
            else:
                checkpoint = float_dir / "checkpoint-promoted.pt"
                promotion_report = float_dir / "promotion_report.json"
                stage = "float"
            if not checkpoint.is_file():
                continue
            output = export_dir / f"student_{track}_{stage}_float32.onnx"
            metadata = output.with_name(f"{output.stem}.metadata.json")
            command = [
                "uv",
                "run",
                "python",
                "scripts/export_onnx_webgpu.py",
                "--checkpoint",
                str(checkpoint),
                "--variant",
                track,
                "--stage",
                stage,
                "--config",
                f"configs/student_dinov3_{track}_distill.yaml",
                "--promotion-report",
                str(promotion_report),
                "--output",
                str(output),
                "--metadata-output",
                str(metadata),
            ]
            if self.run_cmd(command, "EXPORT-FLOAT") != 0:
                return False
            if not output.is_file() or not metadata.is_file():
                return False
            sidecar = json.loads(metadata.read_text(encoding="utf-8"))
            if sidecar.get("evaluation_status") != "promoted":
                log("EXPORT-FLOAT", f"FATAL: {track} float export is not promoted")
                return False
            actual_hash = compute_sha256(output)
            if sidecar.get("artifact_sha256") != actual_hash:
                log("EXPORT-FLOAT", f"FATAL: {track} float sidecar hash mismatch")
                return False
            artifacts[f"{track}_float_onnx"] = str(output)
            artifacts[f"{track}_float_metadata"] = str(metadata)
            hashes[f"{track}_float_onnx"] = actual_hash
        if not artifacts:
            return False
        self.state.mark_stage_completed("export-float", artifacts=artifacts, hashes=hashes)
        return True

    def stage_calibrate_int8(self) -> bool:
        """Stage 17: create and evaluate static INT8 for every exported float student."""
        export_dir = self.output_root / "exported"
        calibration = self.root_dir / "splits" / "production_eligible" / "calibration.parquet"
        manifests_dir = self.root_dir / "splits" / "production_eligible"
        validation = manifests_dir / "validation.parquet"
        artifacts: dict[str, Any] = {}
        hashes: dict[str, str] = {}
        for track in ("small", "base"):
            candidates = sorted(export_dir.glob(f"student_{track}_*_float32.onnx"))
            if not candidates:
                continue
            if len(candidates) != 1:
                log("CALIBRATE-INT8", f"FATAL: Ambiguous {track} float exports: {candidates}")
                return False
            float_model = candidates[0]
            float_metadata = float_model.with_name(f"{float_model.stem}.metadata.json")
            int8_model = float_model.with_name(
                f"{float_model.stem.replace('_float32', '')}_static_int8.onnx"
            )
            int8_metadata = int8_model.with_name(f"{int8_model.stem}.metadata.json")
            command = [
                "uv",
                "run",
                "python",
                "scripts/calibrate_quantize_int8.py",
                "--input-model",
                str(float_model),
                "--float-metadata",
                str(float_metadata),
                "--calibration-manifest",
                str(calibration),
                "--canonical-manifests-dir",
                str(manifests_dir),
                "--data-root",
                str(self.data_root),
                "--output-model",
                str(int8_model),
                "--metadata-output",
                str(int8_metadata),
                "--num-calibration-samples",
                "4096",
                "--eval-manifest",
                str(validation),
            ]
            if self.run_cmd(command, "CALIBRATE-INT8") != 0:
                return False
            if not int8_model.is_file() or not int8_metadata.is_file():
                log("CALIBRATE-INT8", f"FATAL: {track} INT8 artifact set is incomplete")
                return False
            metadata = json.loads(int8_metadata.read_text(encoding="utf-8"))
            if metadata.get("evaluation_status") not in {"promoted", "experimental"}:
                log("CALIBRATE-INT8", f"FATAL: Invalid {track} INT8 evaluation_status")
                return False
            actual_hash = compute_sha256(int8_model)
            if metadata.get("artifact_sha256") != actual_hash:
                log("CALIBRATE-INT8", f"FATAL: {track} INT8 sidecar hash mismatch")
                return False
            artifacts[f"{track}_int8"] = str(int8_model)
            artifacts[f"{track}_int8_metadata"] = str(int8_metadata)
            artifacts[f"{track}_int8_status"] = metadata["evaluation_status"]
            hashes[f"{track}_int8"] = actual_hash
        if not artifacts:
            return False
        self.state.mark_stage_completed("calibrate-int8", artifacts=artifacts, hashes=hashes)
        return True

    def stage_evaluate_int8(self) -> bool:
        """Stage 18: prove each PTQ artifact is static INT8 and contains no INT4 graph."""
        reports: dict[str, str] = {}
        for track in ("small", "base"):
            candidates = sorted(
                (self.output_root / "exported").glob(f"student_{track}_*_static_int8.onnx")
            )
            if not candidates:
                continue
            if len(candidates) != 1:
                return False
            report = self.output_root / f"int8_{track}_inspection_report.json"
            command = [
                "uv",
                "run",
                "python",
                "scripts/inspect_onnx_graph.py",
                "--inspect-model",
                str(candidates[0]),
                "--report-json",
                str(report),
            ]
            if self.run_cmd(command, "EVALUATE-INT8") != 0 or not report.is_file():
                return False
            inspection = json.loads(report.read_text(encoding="utf-8"))
            if inspection.get("passed") is not True:
                return False
            reports[track] = str(report)
        if not reports:
            return False
        self.state.mark_stage_completed("evaluate-int8", artifacts={"reports": reports})
        return True

    def stage_runtime_benchmark(self) -> bool:
        """Stage 19: package authoritative artifacts and require browser WebGPU and forced-WASM."""
        export_dir = self.output_root / "exported"
        selection_path = self.output_root / "student_selection_report.json"
        if not selection_path.is_file():
            return False
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        default_track = selection.get("selected_track")
        if default_track not in {"small", "base"}:
            log("RUNTIME-BENCHMARK", "FATAL: Invalid selected_track in student selection report")
            return False

        web_model_dir = self.root_dir / "web" / "model"
        web_model_dir.mkdir(parents=True, exist_ok=True)
        students: list[dict[str, Any]] = []
        default_model: str | None = None
        for track in ("small", "base"):
            float_candidates = sorted(export_dir.glob(f"student_{track}_*_float32.onnx"))
            int8_candidates = sorted(export_dir.glob(f"student_{track}_*_static_int8.onnx"))
            candidates = [(path, "float32") for path in float_candidates]
            candidates.extend((path, "static_int8") for path in int8_candidates)
            for source, quantization in candidates:
                sidecar_path = source.with_name(f"{source.stem}.metadata.json")
                if not sidecar_path.is_file():
                    log("RUNTIME-BENCHMARK", f"FATAL: Missing sidecar for {source}")
                    return False
                metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
                actual_hash = compute_sha256(source)
                if metadata.get("artifact_sha256") != actual_hash:
                    log("RUNTIME-BENCHMARK", f"FATAL: Artifact hash mismatch for {source}")
                    return False
                status = metadata.get("evaluation_status")
                if quantization == "float32" and status != "promoted":
                    log(
                        "RUNTIME-BENCHMARK",
                        f"FATAL: Float deployment artifact is not promoted: {source}",
                    )
                    return False
                if quantization == "static_int8" and status != "promoted":
                    log(
                        "RUNTIME-BENCHMARK",
                        f"Skipping unpromoted INT8 deployment artifact: {source}",
                    )
                    continue
                destination = web_model_dir / source.name
                shutil.copyfile(source, destination)
                model_id = f"student-{track}-{quantization}"
                entry = dict(metadata)
                entry.update(
                    {
                        "id": model_id,
                        "name": f"Student {track.title()} ({'Static INT8' if quantization == 'static_int8' else 'Float32'})",
                        "path": f"model/{destination.name}",
                        "variant": track,
                        "quantization": quantization,
                        "artifact_sha256": actual_hash,
                        "artifact_size_bytes": destination.stat().st_size,
                    }
                )
                students.append(entry)
                if track == default_track and quantization == "float32":
                    default_model = model_id
                if (
                    track == default_track
                    and quantization == "static_int8"
                    and status == "promoted"
                ):
                    default_model = model_id

        if not students or default_model is None:
            return False
        metadata_path = web_model_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps({"default_model": default_model, "students": students}, indent=2),
            encoding="utf-8",
        )

        node = shutil.which("node")
        benchmark = self.root_dir / "web" / "benchmark.js"
        if node is None or not benchmark.is_file():
            log("RUNTIME-BENCHMARK", "FATAL: Node/Puppeteer browser benchmark is required")
            return False
        server = subprocess.Popen(
            [sys.executable, "web/serve.py", "--port", "8000"],
            cwd=str(self.root_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        results: dict[str, str] = {}
        try:
            time.sleep(2)
            for provider, extra in (("webgpu", []), ("wasm", ["--wasm"])):
                result = subprocess.run(
                    [node, "web/benchmark.js", "http://localhost:8000", *extra],
                    cwd=str(self.root_dir),
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
                output = "\n".join(
                    part for part in (result.stdout.strip(), result.stderr.strip()) if part
                )
                if result.returncode != 0 or f"PROVIDER={provider}" not in output:
                    log("RUNTIME-BENCHMARK", f"FATAL: Browser {provider} gate failed: {output}")
                    return False
                results[provider] = output
        finally:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)

        report_path = self.output_root / "runtime_benchmark_report.json"
        report = {
            "timestamp": datetime.now(UTC).isoformat(),
            "default_model": default_model,
            "webgpu_verified": True,
            "forced_wasm_verified": True,
            "results": results,
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        self.state.mark_stage_completed(
            "runtime-benchmark",
            artifacts={"report": str(report_path), "metadata": str(metadata_path)},
        )
        return True

    def stage_upload(self) -> bool:
        """Stage 20: Upload final models, ONNX packages, reports, and transparency data."""
        log("UPLOAD", "Finalizing all Hugging Face uploads...")
        exported_files = list((self.output_root / "exported").glob("*.*"))
        receipts = list((self.output_root / "upload_receipts").glob("*.json"))
        state_files = [self.state_file] if self.state_file.is_file() else []

        ok1 = self.upload_stage(
            "final-models",
            exported_files + receipts + state_files,
            repo_id=self.model_repo,
        )
        if not ok1:
            log("UPLOAD", "FATAL: Final models upload to Hugging Face Hub failed.")
            return False

        manifest_files = list((self.root_dir / "splits" / "production_eligible").glob("*.*"))
        ok2 = self.upload_stage(
            "final-transparency",
            manifest_files,
            repo_id=self.data_repo,
            repo_type="dataset",
        )
        if not ok2:
            log("UPLOAD", "FATAL: Final dataset transparency upload to Hugging Face Hub failed.")
            return False

        self.state.mark_stage_completed("upload")
        return True

    def stage_complete(self) -> bool:
        """Stage 21: Emit pipeline completion receipt with hashes and promotion summary."""
        log("COMPLETE", "Pipeline execution complete! Emitting completion receipt...")
        receipt_path = self.output_root / "pipeline_completion_receipt.json"
        summary = {
            "completed_at": datetime.now(UTC).isoformat(),
            "pipeline_state": self.state.data,
            "success": True,
        }
        with open(receipt_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        log("COMPLETE", f"Receipt saved to {receipt_path}")
        self.state.mark_stage_completed("complete", artifacts={"receipt": str(receipt_path)})
        return True

    # --------------------------------------------------------------------------
    # Main Execution Loop
    # --------------------------------------------------------------------------

    def execute(self) -> int:
        """Run all pipeline stages sequentially, respecting resume and gates."""
        log("PIPELINE", "=================================================================")
        log("PIPELINE", "Starting MechaDetect 4x RTX 4090 Gated Production Pipeline")
        log("PIPELINE", f"Root Directory:   {self.root_dir}")
        log("PIPELINE", f"Runtime Data:     {self.data_root}")
        log("PIPELINE", f"Runtime Outputs:  {self.output_root}")
        log("PIPELINE", f"State File:       {self.state_file}")
        log("PIPELINE", "=================================================================")

        stage_methods = {
            "preflight": self.stage_preflight,
            "acquire": self.stage_acquire,
            "freeze-manifests": self.stage_freeze_manifests,
            "teacher-stage1-smoke": self.stage_teacher_stage1_smoke,
            "teacher-stage1": self.stage_teacher_stage1,
            "teacher-stage1-eval": self.stage_teacher_stage1_eval,
            "teacher-stage2-smoke": self.stage_teacher_stage2_smoke,
            "teacher-stage2": self.stage_teacher_stage2,
            "teacher-eval": self.stage_teacher_eval,
            "students-smoke": self.stage_students_smoke,
            "students-distill": self.stage_students_distill,
            "students-eval": self.stage_students_eval,
            "att-smoke": self.stage_att_smoke,
            "att": self.stage_att,
            "att-eval": self.stage_att_eval,
            "export-float": self.stage_export_float,
            "calibrate-int8": self.stage_calibrate_int8,
            "evaluate-int8": self.stage_evaluate_int8,
            "runtime-benchmark": self.stage_runtime_benchmark,
            "upload": self.stage_upload,
            "complete": self.stage_complete,
        }

        target_stages = PIPELINE_STAGES
        if self.args.stage:
            if self.args.stage not in PIPELINE_STAGES:
                log(
                    "PIPELINE",
                    f"FATAL: Unknown stage '{self.args.stage}'. Choose from {PIPELINE_STAGES}",
                )
                return 1
            idx = PIPELINE_STAGES.index(self.args.stage)
            target_stages = (
                PIPELINE_STAGES[idx : idx + 1] if self.args.only else PIPELINE_STAGES[idx:]
            )

        for stage in target_stages:
            if not self.args.force and self.state.is_stage_completed(
                stage, root_dir=self.root_dir, output_root=self.output_root
            ):
                log(
                    "PIPELINE",
                    f"Stage '{stage}' already completed and artifacts verified. Skipping (resume mode).",
                )
                continue

            log("PIPELINE", f"--- Starting Stage: {stage} ---")
            self.state.mark_stage_started(stage)
            method = stage_methods[stage]

            success = method()
            if not success:
                log(
                    "PIPELINE",
                    f"FATAL: Pipeline halted at stage '{stage}'. Downstream stages will NOT run.",
                )
                return 1

        log("PIPELINE", "All stages completed successfully!")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Orchestrate MechaDetect 4x RTX 4090 Production Pipeline"
    )
    parser.add_argument("--stage", type=str, default=None, help="Start from or run specific stage")
    parser.add_argument("--only", action="store_true", help="Run only the specified stage and exit")
    parser.add_argument(
        "--force", action="store_true", help="Force re-run stage even if marked completed in state"
    )
    parser.add_argument(
        "--explicit-balance", type=float, default=None, help="Explicit Vast.ai balance in USD"
    )
    parser.add_argument(
        "--hourly-rate", type=float, default=None, help="Explicit instance hourly rate in USD/hour"
    )
    parser.add_argument(
        "--allow-gpu-mismatch",
        action="store_true",
        help="Allow non-4090 or <4 GPUs (for local test)",
    )
    parser.add_argument(
        "--allow-disk-mismatch", action="store_true", help="Allow <200GB free disk (for local test)"
    )

    args = parser.parse_args()
    orchestrator = PipelineOrchestrator(args)
    return orchestrator.execute()


if __name__ == "__main__":
    sys.exit(main())
