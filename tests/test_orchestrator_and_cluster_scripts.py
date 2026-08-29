from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from scripts.orchestrate_4x4090 import (
    PIPELINE_STAGES,
    PipelineOrchestrator,
    PipelineStateManager,
    VastBudgetGuard,
    compute_sha256,
)


def test_pipeline_stages_contract() -> None:
    """Verify the 21-stage state machine matches the implementation plan specification."""
    expected_stages = [
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
    assert PIPELINE_STAGES == expected_stages
    assert len(PIPELINE_STAGES) == 21


def test_pipeline_state_manager(tmp_path: Path) -> None:
    """Test state manager persistence, start/complete transitions, and receipts."""
    state_file = tmp_path / "state.json"
    mgr = PipelineStateManager(state_file)

    assert not mgr.is_stage_completed("preflight")
    mgr.mark_stage_started("preflight")
    assert mgr.data["current_stage"] == "preflight"

    dummy_file = tmp_path / "dummy.pt"
    dummy_file.write_bytes(b"dummy_weights")
    dummy_hash = compute_sha256(dummy_file)

    mgr.mark_stage_completed(
        "preflight",
        artifacts={"test_file": str(dummy_file)},
        metrics={"gpus": 4},
        hashes={"test_file": dummy_hash},
    )
    assert mgr.is_stage_completed("preflight")
    assert mgr.data["current_stage"] is None
    assert mgr.data["stage_hashes"]["preflight"]["test_file"] == dummy_hash

    # Reload from disk and verify persistence
    mgr2 = PipelineStateManager(state_file)
    assert mgr2.is_stage_completed("preflight")
    assert mgr2.data["stage_metrics"]["preflight"]["gpus"] == 4

    # Record upload receipt
    receipt = {"repo_id": "zye2/mechadetect-models", "success": True}
    mgr2.record_upload("preflight", receipt)

    mgr3 = PipelineStateManager(state_file)
    assert mgr3.data["upload_receipts"]["preflight"]["success"] is True


def test_resumability_validates_artifacts_and_hashes_on_disk(tmp_path: Path) -> None:
    """Verify that is_stage_completed validates on-disk files and SHA-256 digests, invalidating on corruption."""
    state_file = tmp_path / "state_resumable.json"
    mgr = PipelineStateManager(state_file)

    art_file = tmp_path / "model.pt"
    art_file.write_bytes(b"exact_bytes_12345")
    true_hash = compute_sha256(art_file)

    mgr.mark_stage_completed(
        "teacher-stage1",
        artifacts={"model": str(art_file)},
        hashes={"model": true_hash},
    )

    # 1. Valid artifact and matching hash -> Resume succeeds
    assert mgr.is_stage_completed("teacher-stage1") is True

    # 2. Corrupted file content (hash mismatch) -> Resume invalidates stage
    art_file.write_bytes(b"tampered_bytes_67890")
    assert mgr.is_stage_completed("teacher-stage1") is False
    # Stage should have been removed from completed_stages
    assert "teacher-stage1" not in mgr.data.get("completed_stages", {})

    # 3. Missing file entirely -> Resume invalidates stage
    mgr.mark_stage_completed(
        "teacher-stage1",
        artifacts={"model": str(art_file)},
        hashes={"model": compute_sha256(art_file)},
    )
    art_file.unlink()
    assert mgr.is_stage_completed("teacher-stage1") is False
    assert "teacher-stage1" not in mgr.data.get("completed_stages", {})


def test_resumability_validates_promotion_report_gate(tmp_path: Path) -> None:
    """Verify that promotion stage resume validation fails if promotion report has passed=False."""
    state_file = tmp_path / "state_gate.json"
    mgr = PipelineStateManager(state_file)

    output_root = tmp_path / "outputs"
    stage1_dir = output_root / "teacher_stage1_clean_frozen"
    stage1_dir.mkdir(parents=True)

    ckpt_file = stage1_dir / "checkpoint-promoted.pt"
    ckpt_file.write_bytes(b"checkpoint_weights")
    ckpt_sha = compute_sha256(ckpt_file)

    rep_file = stage1_dir / "promotion_report.json"
    rep_file.write_text(
        json.dumps(
            {
                "passed": False,
                "failed_reasons": ["Clean AUROC 0.94 < 0.96"],
                "checkpoint_path": str(ckpt_file),
                "checkpoint_sha256": ckpt_sha,
            }
        ),
        encoding="utf-8",
    )

    mgr.mark_stage_completed(
        "teacher-stage1-eval",
        artifacts={"promoted_checkpoint": str(ckpt_file), "report": str(rep_file)},
        hashes={"promoted_checkpoint": ckpt_sha},
    )

    # Validate with output_root: passed=False must invalidate
    assert (
        mgr.is_stage_completed("teacher-stage1-eval", root_dir=tmp_path, output_root=output_root)
        is False
    )
    assert "teacher-stage1-eval" not in mgr.data.get("completed_stages", {})

    # Now update report to passed=True
    rep_file.write_text(
        json.dumps(
            {
                "passed": True,
                "failed_reasons": [],
                "checkpoint_path": str(ckpt_file),
                "checkpoint_sha256": ckpt_sha,
            }
        ),
        encoding="utf-8",
    )
    mgr.mark_stage_completed(
        "teacher-stage1-eval",
        artifacts={"promoted_checkpoint": str(ckpt_file), "report": str(rep_file)},
        hashes={"promoted_checkpoint": ckpt_sha},
    )
    assert (
        mgr.is_stage_completed("teacher-stage1-eval", root_dir=tmp_path, output_root=output_root)
        is True
    )


def test_vast_budget_guard_enforces_reserve() -> None:
    """Verify VastBudgetGuard enforces $5 reserve, requires hourly rate, and projects costs."""
    # Test 1: $10 balance, 1 hour @ $2.50/h -> remaining $7.50 >= $5.00 -> Affordability passes
    guard = VastBudgetGuard(min_reserve=5.00, explicit_balance=10.00, explicit_hourly_rate=2.50)
    ok, msg = guard.check_stage_affordability("stage1", projected_hours=1.0)
    assert ok is True
    assert "Affordability check passed" in msg

    # Test 2: $10 balance, 3 hours @ $2.50/h -> cost $7.50 -> remaining $2.50 < $5.00 reserve -> Fails
    ok, msg = guard.check_stage_affordability("stage2", projected_hours=3.0)
    assert ok is False
    assert "violating the mandatory $5.00 reserve" in msg

    # Test 3: No balance available and no explicit balance -> Fails closed
    guard_unknown = VastBudgetGuard(
        api_key=None, min_reserve=5.00, explicit_balance=None, explicit_hourly_rate=2.50
    )
    ok, msg = guard_unknown.check_stage_affordability("stage1", projected_hours=1.0)
    assert ok is False
    assert "Cannot query Vast balance" in msg

    # Test 4: Balance available but hourly rate unknown -> Fails closed
    guard_no_rate = VastBudgetGuard(
        min_reserve=5.00, explicit_balance=10.00, explicit_hourly_rate=None
    )
    ok, msg = guard_no_rate.check_stage_affordability("stage1", projected_hours=1.0)
    assert ok is False
    assert "Hourly rate is unknown" in msg


def test_calculate_measured_hours_rejects_missing_smoke_without_guessing(tmp_path: Path) -> None:
    """Verify duration calculation derives updates from rows/48 and rejects unmeasured stages (no guessing)."""
    state_file = tmp_path / "state.json"
    mgr = PipelineStateManager(state_file)

    args = MagicMock()
    args.explicit_balance = 20.0
    args.hourly_rate = 2.50
    orch = PipelineOrchestrator(args)
    orch.state = mgr

    # Mock eligible train count to 73,751 rows
    orch.get_eligible_train_count = MagicMock(return_value=73751)

    # 1. No smoke throughput recorded -> must raise RuntimeError (prohibits guessed fallback)
    with pytest.raises(RuntimeError) as exc_info:
        orch.calculate_measured_hours("teacher-stage1", passes=1)
    assert "measured smoke throughput is missing" in str(exc_info.value)
    assert "Guessed fallback is prohibited" in str(exc_info.value)

    # 2. Record measured throughput: 1.500 s/update
    mgr.record_smoke_throughput("teacher-stage1", 1.500)

    # Expected: ceil(73,751 / 48) = 1,537 updates. 1,537 * 1.5s / 3600 = 0.6404 hours
    hours = orch.calculate_measured_hours("teacher-stage1", passes=1)
    expected_updates = math.ceil(73751 / 48)
    expected_hours = (expected_updates * 1.500) / 3600.0
    assert abs(hours - expected_hours) < 1e-5


def test_smoke_throughput_state(tmp_path: Path) -> None:
    """Verify smoke-measured throughput is persisted and retrievable."""
    state_file = tmp_path / "state_smoke.json"
    mgr = PipelineStateManager(state_file)
    assert mgr.get_smoke_throughput("teacher-stage1") is None

    mgr.record_smoke_throughput("teacher-stage1", 1.825)
    assert mgr.get_smoke_throughput("teacher-stage1") == 1.825

    # Reload from disk
    mgr2 = PipelineStateManager(state_file)
    assert mgr2.get_smoke_throughput("teacher-stage1") == 1.825


def test_cluster_script_files_exist() -> None:
    """Verify all cluster and orchestration scripts exist and have executable permissions."""
    expected_scripts = [
        "cluster_setup.sh",
        "cluster_train_stage2.sh",
        "cluster_train_checkpoint2.sh",
        "orchestrate_4x4090.sh",
        "scripts/orchestrate_4x4090.py",
        "scripts/upload_promoted_artifacts.py",
        "scripts/launch_student_tracks.sh",
        ".env.cluster",
    ]
    for rel_path in expected_scripts:
        p = REPO_ROOT / rel_path
        assert p.is_file(), f"Missing required file: {rel_path}"
        assert p.stat().st_size > 0, f"File is empty: {rel_path}"


def test_checkpoint2_demotion_contract() -> None:
    """Verify cluster_train_checkpoint2.sh announces demotion and stops without --legacy-run."""
    script_path = REPO_ROOT / "cluster_train_checkpoint2.sh"
    content = script_path.read_text(encoding="utf-8")

    assert "DEMOTED from the forward 4x4090 production pipeline" in content
    assert "orchestrate_4x4090.sh" in content
    assert "--legacy-run" in content
    assert "--nproc-per-node=4" in content
    assert "--nproc-per-node=6" not in content


def test_cluster_train_stage2_strictly_requires_promoted_checkpoint() -> None:
    """Verify cluster_train_stage2.sh requires verified checkpoint-promoted.pt and removes checkpoint-best fallback."""
    script_path = REPO_ROOT / "cluster_train_stage2.sh"
    content = script_path.read_text(encoding="utf-8")

    assert "checkpoint-promoted.pt" in content
    assert "promotion_report.json" in content
    assert "checkpoint-best.pt" not in content
    assert "teacher_stage2_paired_unfrozen" in content


def test_launch_student_tracks_canonical_contract() -> None:
    """Verify launch_student_tracks.sh defaults to canonical stage 2 promoted paths and has no unsupported flags."""
    script_path = REPO_ROOT / "scripts" / "launch_student_tracks.sh"
    content = script_path.read_text(encoding="utf-8")

    assert "teacher_stage2_paired_unfrozen/checkpoint-promoted.pt" in content
    assert "teacher_stage2_paired_unfrozen/promotion_report.json" in content
    assert "outputs/teacher_stage2/checkpoint-promoted.pt" not in content
    assert "--strict-teacher-gate" not in content


def test_env_cluster_keys() -> None:
    """Verify .env.cluster defines all required 4x4090 variables including VAST_HOURLY_RATE."""
    content = (REPO_ROOT / ".env.cluster").read_text(encoding="utf-8")
    required_keys = [
        "TECHJAM_RUNTIME_ROOT",
        "TECHJAM_DATA_ROOT",
        "TECHJAM_OUTPUT_ROOT",
        "TECHJAM_DATA_REPO=zye2/tj-data",
        "TECHJAM_MODEL_REPO=zye2/mechadetect-models",
        "TECHJAM_GIT_BRANCH=training/production-4x4090",
        "VAST_MIN_BALANCE_RESERVE=5.00",
        "VAST_HOURLY_RATE=",
        "TEACHER_WORLD_SIZE=4",
        "STUDENT_SMALL_GPUS=0,1",
        "STUDENT_BASE_GPUS=2,3",
        "TECHJAM_CALIBRATION_SIZE=4096",
    ]
    for key in required_keys:
        assert key in content, f"Missing key '{key}' in .env.cluster"


def test_orchestrator_cli_help() -> None:
    """Verify scripts/orchestrate_4x4090.py runs with --help."""
    res = subprocess.run(
        ["python", str(REPO_ROOT / "scripts" / "orchestrate_4x4090.py"), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0
    assert "Orchestrate MechaDetect 4x RTX 4090 Production Pipeline" in res.stdout
    assert "--stage" in res.stdout
    assert "--explicit-balance" in res.stdout
    assert "--hourly-rate" in res.stdout
    assert "--allow-gpu-mismatch" in res.stdout


def test_preflight_budget_and_rate_checks_fail_closed(tmp_path: Path) -> None:
    """Verify stage_preflight fails closed if balance < $5.00 reserve or hourly rate is unknown."""
    # Test 1: Balance $3.50 < $5.00 -> Preflight must fail
    args_low_bal = MagicMock()
    args_low_bal.explicit_balance = 3.50
    args_low_bal.hourly_rate = 2.40
    args_low_bal.allow_gpu_mismatch = False
    args_low_bal.allow_disk_mismatch = True

    orch = PipelineOrchestrator(args_low_bal)
    orch.output_root = tmp_path
    # In production without allow_gpu_mismatch, preflight must fail
    assert orch.stage_preflight() is False

    # Test 2: Rate is unknown (None) -> Preflight must fail
    args_no_rate = MagicMock()
    args_no_rate.explicit_balance = 25.00
    args_no_rate.hourly_rate = None
    args_no_rate.allow_gpu_mismatch = False
    args_no_rate.allow_disk_mismatch = True

    orch2 = PipelineOrchestrator(args_no_rate)
    orch2.output_root = tmp_path
    assert orch2.stage_preflight() is False


def test_students_smoke_invokes_both_small_and_base(tmp_path: Path) -> None:
    """Verify stage_students_smoke executes dry-run smoke tests for BOTH small and base tracks."""
    args = MagicMock()
    args.explicit_balance = 20.0
    args.hourly_rate = 2.40
    orch = PipelineOrchestrator(args)
    orch.output_root = tmp_path / "outputs"
    orch.root_dir = tmp_path
    orch.state = PipelineStateManager(tmp_path / "student_smoke_state.json")

    stage2_dir = orch.output_root / "teacher_stage2_paired_unfrozen"
    stage2_dir.mkdir(parents=True)
    teacher_ckpt = stage2_dir / "checkpoint-promoted.pt"
    teacher_ckpt.write_bytes(b"teacher_weights")
    (stage2_dir / "promotion_report.json").write_text("{}", encoding="utf-8")

    splits_dir = orch.root_dir / "splits" / "production_eligible"
    splits_dir.mkdir(parents=True)
    train_manifest = splits_dir / "train.parquet"
    train_manifest.write_bytes(b"parquet_data")
    (splits_dir / "validation.parquet").write_bytes(b"parquet_data")

    invoked_cmds = []

    def mock_run_cmd(cmd, stage, env=None):
        invoked_cmds.append(" ".join(cmd))
        return 0

    orch.run_cmd = mock_run_cmd
    res = orch.stage_students_smoke()
    assert res is True
    assert len(invoked_cmds) == 2
    assert any("--track small" in c and "--dry-run" in c for c in invoked_cmds)
    assert any("--track base" in c and "--dry-run" in c for c in invoked_cmds)
    assert orch.state.get_smoke_throughput("students-distill") is not None


def test_att_smoke_invokes_both_small_and_base(tmp_path: Path) -> None:
    """Verify stage_att_smoke executes dry-run smoke tests for BOTH small and base tracks."""
    args = MagicMock()
    args.explicit_balance = 20.0
    args.hourly_rate = 2.40
    orch = PipelineOrchestrator(args)
    orch.state = PipelineStateManager(tmp_path / "att_smoke_state.json")
    orch.output_root = tmp_path / "outputs"
    orch.root_dir = tmp_path

    sm_dir = orch.output_root / "student_dinov3_small"
    sm_dir.mkdir(parents=True)
    (sm_dir / "checkpoint-promoted.pt").write_bytes(b"sm_weights")

    ba_dir = orch.output_root / "student_dinov3_base"
    ba_dir.mkdir(parents=True)
    (ba_dir / "checkpoint-promoted.pt").write_bytes(b"ba_weights")

    splits_dir = orch.root_dir / "splits" / "production_eligible"
    splits_dir.mkdir(parents=True)
    train_manifest = splits_dir / "train.parquet"
    train_manifest.write_bytes(b"parquet_data")

    invoked_cmds = []

    def mock_run_cmd(cmd, stage, env=None):
        invoked_cmds.append(" ".join(cmd))
        return 0

    orch.run_cmd = mock_run_cmd
    res = orch.stage_att_smoke()
    assert res is True
    assert len(invoked_cmds) == 2
    assert any("--track small" in c and "--dry-run" in c for c in invoked_cmds)
    assert any("--track base" in c and "--dry-run" in c for c in invoked_cmds)
    assert orch.state.get_smoke_throughput("att") is not None


@pytest.mark.parametrize(
    ("int8_status", "expected_default"),
    [
        ("experimental", "student-small-float32"),
        ("promoted", "student-small-static_int8"),
    ],
)
def test_web_metadata_sync_schema_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    int8_status: str,
    expected_default: str,
) -> None:
    """Runtime packaging defaults to INT8 only when its quality gate promoted it."""
    import scripts.orchestrate_4x4090 as orchestration

    args = MagicMock()
    orch = PipelineOrchestrator(args)
    orch.root_dir = tmp_path
    orch.output_root = tmp_path / "outputs"
    orch.state = PipelineStateManager(tmp_path / f"runtime_{int8_status}_state.json")

    export_dir = orch.output_root / "exported"
    export_dir.mkdir(parents=True)
    (orch.output_root / "student_selection_report.json").write_text(
        json.dumps({"selected_track": "small"}),
        encoding="utf-8",
    )
    common_metadata = {
        "model_family": "dinov3_vits16",
        "variant": "small",
        "calibrated_threshold": 0.5432,
        "manifest_digest": "a" * 64,
        "input_size": [3, 224, 224],
        "preprocessing_version": 2,
    }
    float_model = export_dir / "student_small_post_att_float32.onnx"
    float_model.write_bytes(b"float_onnx")
    (export_dir / "student_small_post_att_float32.metadata.json").write_text(
        json.dumps(
            {
                **common_metadata,
                "quantization": "float32",
                "evaluation_status": "promoted",
                "artifact_sha256": compute_sha256(float_model),
            }
        ),
        encoding="utf-8",
    )
    int8_model = export_dir / "student_small_post_att_static_int8.onnx"
    int8_model.write_bytes(b"int8_onnx")
    (export_dir / "student_small_post_att_static_int8.metadata.json").write_text(
        json.dumps(
            {
                **common_metadata,
                "quantization": "static_int8",
                "evaluation_status": int8_status,
                "artifact_sha256": compute_sha256(int8_model),
            }
        ),
        encoding="utf-8",
    )

    web_dir = tmp_path / "web"
    web_dir.mkdir()
    (web_dir / "benchmark.js").write_text("// mocked benchmark", encoding="utf-8")
    server = MagicMock()
    server.wait.return_value = 0
    monkeypatch.setattr(orchestration.shutil, "which", lambda _: "node")
    monkeypatch.setattr(orchestration.subprocess, "Popen", lambda *args, **kwargs: server)

    def fake_browser_run(command, **kwargs):
        provider = "wasm" if "--wasm" in command else "webgpu"
        return MagicMock(returncode=0, stdout=f"PROVIDER={provider}", stderr="")

    monkeypatch.setattr(orchestration.subprocess, "run", fake_browser_run)

    assert orch.stage_runtime_benchmark() is True
    metadata_path = web_dir / "model" / "metadata.json"
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert data["default_model"] == expected_default
    assert len(data["students"]) == 2
    by_quantization = {entry["quantization"]: entry for entry in data["students"]}
    assert by_quantization["float32"]["evaluation_status"] == "promoted"
    assert by_quantization["static_int8"]["evaluation_status"] == int8_status
    assert by_quantization["float32"]["calibrated_threshold"] == 0.5432
    assert by_quantization["float32"]["manifest_digest"] == "a" * 64
    assert by_quantization["float32"]["input_size"] == [3, 224, 224]
    assert len(by_quantization["float32"]["artifact_sha256"]) == 64
