#!/usr/bin/env python3
"""Upload promoted artifacts, reports, and manifests to Hugging Face Hub.

Upload targets:
- Private models & reports repo: zye2/mechadetect-models (private=True)
- Dataset transparency & manifests repo: zye2/tj-data

Implements upload receipts with content SHA-256 digests and supports stage-by-stage
or final consolidated uploads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def compute_file_sha256(path: Path) -> str:
    """Compute hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def ensure_hf_repo_exists(
    repo_id: str,
    repo_type: str = "model",
    private: bool = True,
    token: str | None = None,
) -> bool:
    """Ensure Hugging Face repository exists, creating it as private if missing."""
    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print(
            f"[upload] WARNING: No HF_TOKEN or HUGGING_FACE_HUB_TOKEN set. Cannot verify {repo_id}"
        )
        return False

    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        try:
            api.repo_info(repo_id=repo_id, repo_type=repo_type)
            print(f"[upload] Verified repo exists: {repo_id} ({repo_type})")
            return True
        except Exception:
            print(f"[upload] Repo {repo_id} not found. Creating repo (private={private})...")
            api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)
            print(f"[upload] Successfully created repo: {repo_id} ({repo_type}, private={private})")
            return True
    except Exception as exc:
        print(f"[upload] ERROR ensuring repo {repo_id} exists: {exc}")
        return False


def upload_files_to_hf(
    files: list[Path],
    repo_id: str,
    repo_type: str = "model",
    path_in_repo_prefix: str = "",
    receipt_path: Path | None = None,
    commit_message: str | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    """Upload a list of files to Hugging Face Hub and record receipt."""
    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    timestamp = datetime.now(UTC).isoformat()
    receipt: dict[str, Any] = {
        "timestamp": timestamp,
        "repo_id": repo_id,
        "repo_type": repo_type,
        "files": [],
        "success": False,
    }

    if not files:
        print(f"[upload] No files specified for upload to {repo_id}")
        receipt["success"] = True
        return receipt

    # Verify all files exist locally
    valid_files: list[Path] = []
    for f in files:
        p = Path(f)
        if p.is_file():
            valid_files.append(p)
        else:
            print(f"[upload] WARNING: File does not exist, skipping: {p}")

    if not valid_files:
        print(f"[upload] No existing files to upload to {repo_id}")
        return receipt

    print(f"[upload] Preparing to upload {len(valid_files)} files to {repo_id} ({repo_type})...")
    for vf in valid_files:
        sha256 = compute_file_sha256(vf)
        size_bytes = vf.stat().st_size
        dest_path = (
            f"{path_in_repo_prefix.rstrip('/')}/{vf.name}" if path_in_repo_prefix else vf.name
        )
        receipt["files"].append(
            {
                "local_path": str(vf),
                "dest_path": dest_path,
                "sha256": sha256,
                "size_bytes": size_bytes,
            }
        )
        print(
            f"  - {vf.name} ({size_bytes / 1024 / 1024:.2f} MB, sha256: {sha256[:12]}...) -> {dest_path}"
        )

    if not token:
        print(f"[upload] ERROR: HF token missing. Cannot upload to {repo_id}.")
        return receipt

    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        ok_repo = ensure_hf_repo_exists(
            repo_id=repo_id,
            repo_type=repo_type,
            private=(repo_type == "model"),
            token=token,
        )
        if not ok_repo:
            print(f"[upload] ERROR: Repository verification/creation failed for {repo_id}")
            receipt["error"] = f"Failed verifying or creating repository {repo_id}"
            if receipt_path:
                receipt_path = Path(receipt_path)
                receipt_path.parent.mkdir(parents=True, exist_ok=True)
                with open(receipt_path, "w", encoding="utf-8") as f:
                    json.dump(receipt, f, indent=2)
            return receipt

        commit_msg = (
            commit_message
            or f"Upload artifacts for {path_in_repo_prefix or 'production run'} at {timestamp}"
        )
        for item in receipt["files"]:
            local_p = Path(item["local_path"])
            dest_p = item["dest_path"]
            print(f"[upload] Uploading {local_p} -> {dest_p}...")
            api.upload_file(
                path_or_fileobj=str(local_p),
                path_in_repo=dest_p,
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=f"{commit_msg}: {dest_p}",
            )

        receipt["success"] = True
        print(f"[upload] Successfully uploaded all {len(valid_files)} files to {repo_id}")
    except Exception as exc:
        print(f"[upload] ERROR during upload to {repo_id}: {exc}")
        receipt["error"] = str(exc)

    if receipt_path:
        receipt_path = Path(receipt_path)
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(receipt_path, "w", encoding="utf-8") as f:
            json.dump(receipt, f, indent=2)
        print(f"[upload] Saved upload receipt to {receipt_path}")

    return receipt


def upload_stage_artifacts(
    stage_name: str,
    artifact_paths: list[str | Path],
    model_repo: str = "zye2/mechadetect-models",
    data_repo: str = "zye2/tj-data",
    receipts_dir: Path | None = None,
) -> bool:
    """Upload artifacts for a specific pipeline stage."""
    receipts_dir = receipts_dir or Path("outputs/upload_receipts")
    receipts_dir.mkdir(parents=True, exist_ok=True)
    receipt_file = receipts_dir / f"receipt_{stage_name}.json"

    file_paths = [Path(p) for p in artifact_paths if p]
    if not file_paths:
        print(f"[upload] No artifact paths supplied for stage {stage_name}")
        return True

    res = upload_files_to_hf(
        files=file_paths,
        repo_id=model_repo,
        repo_type="model",
        path_in_repo_prefix=stage_name,
        receipt_path=receipt_file,
        commit_message=f"Promoted artifacts for stage {stage_name}",
    )
    return bool(res.get("success", False))


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload promoted artifacts to Hugging Face Hub")
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        help="Stage name (e.g. stage1, stage2, students, att, export, manifests)",
    )
    parser.add_argument(
        "--files", type=str, nargs="+", required=True, help="List of file paths to upload"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Target HF repo ID (default: zye2/mechadetect-models for models or zye2/tj-data for manifests)",
    )
    parser.add_argument(
        "--repo-type", type=str, choices=["model", "dataset"], default="model", help="HF repo type"
    )
    parser.add_argument("--prefix", type=str, default="", help="Prefix path inside the repository")
    parser.add_argument(
        "--receipt-dir",
        type=str,
        default="outputs/upload_receipts",
        help="Directory for upload receipts",
    )
    parser.add_argument("--commit-message", type=str, default=None, help="Custom commit message")

    args = parser.parse_args()

    default_model_repo = os.environ.get("TECHJAM_MODEL_REPO", "zye2/mechadetect-models")
    default_data_repo = os.environ.get("TECHJAM_DATA_REPO", "zye2/tj-data")

    repo_id = args.repo_id
    if not repo_id:
        if args.repo_type == "dataset" or args.stage == "manifests":
            repo_id = default_data_repo
            repo_type = "dataset"
        else:
            repo_id = default_model_repo
            repo_type = "model"
    else:
        repo_type = args.repo_type

    prefix = args.prefix if args.prefix else args.stage
    receipts_dir = Path(args.receipt_dir)
    receipt_file = receipts_dir / f"receipt_{args.stage}.json"

    file_paths = [Path(f) for f in args.files]
    res = upload_files_to_hf(
        files=file_paths,
        repo_id=repo_id,
        repo_type=repo_type,
        path_in_repo_prefix=prefix,
        receipt_path=receipt_file,
        commit_message=args.commit_message or f"Promoted artifacts for stage {args.stage}",
    )

    return 0 if res.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
