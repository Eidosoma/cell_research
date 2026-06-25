#!/usr/bin/env python3
"""Record S01 repository, environment, hardware, and entry-point provenance."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STEP_ID = "S01"
STEP_NUMBER = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_cmd(
    args: list[str],
    cwd: Path | None = None,
    timeout: int = 120,
    check: bool = False,
) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        if check:
            raise
        return {
            "args": args,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "ok": False,
        }
    except subprocess.TimeoutExpired as exc:
        if check:
            raise
        return {
            "args": args,
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": f"timed out after {timeout}s\n{exc.stderr or ''}",
            "ok": False,
        }

    if check and completed.returncode != 0:
        raise RuntimeError(
            f"command failed: {' '.join(args)}\n{completed.stderr}"
        )
    return {
        "args": args,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "ok": completed.returncode == 0,
    }


def sha256_bytes(data: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(data)
    return digest.hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def git_stdout(repo_root: Path, *args: str, timeout: int = 120) -> str:
    result = run_cmd(["git", *args], cwd=repo_root, timeout=timeout, check=True)
    return result["stdout"].strip()


def list_tracked_files(repo_root: Path) -> list[str]:
    output = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=repo_root
    )
    return [item.decode() for item in output.split(b"\0") if item]


def write_file_tree(repo_root: Path, out_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for rel_path in list_tracked_files(repo_root):
        full_path = repo_root / rel_path
        if not full_path.is_file():
            continue
        records.append(
            {
                "path": rel_path,
                "sizeBytes": full_path.stat().st_size,
                "sha256": sha256_path(full_path),
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["path", "sizeBytes", "sha256"], delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(records)
    return records


def git_archive_sha256(repo_root: Path, ref: str) -> str | None:
    result = subprocess.run(
        ["git", "archive", "--format=tar", ref],
        cwd=repo_root,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return sha256_bytes(result.stdout)


def scan_entry_points(repo_root: Path, out_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(repo_root.rglob("*.py")):
        if ".git" in path.parts:
            continue
        rel = path.relative_to(repo_root).as_posix()
        text = path.read_text(errors="replace")
        lines = text.splitlines()
        has_main = any('if __name__ == "__main__"' in line for line in lines)
        npy_writes = sorted(
            {
                line.strip()
                for line in lines
                if "np.save" in line or ".npy" in line
            }
        )
        hardcoded_abs = sorted(
            {
                line.strip()
                for line in lines
                if "/Users/" in line or "Workspace/research" in line
            }
        )
        imports = sorted(
            {
                line.strip()
                for line in lines
                if line.startswith("import ") or line.startswith("from ")
            }
        )
        records.append(
            {
                "path": rel,
                "hasMain": has_main,
                "npyWriteOrLoadLines": len(npy_writes),
                "hardcodedAbsolutePathLines": len(hardcoded_abs),
                "importLines": len(imports),
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "path",
                "hasMain",
                "npyWriteOrLoadLines",
                "hardcodedAbsolutePathLines",
                "importLines",
            ],
        )
        writer.writeheader()
        writer.writerows(records)
    return records


def collect_package_freeze(out_path: Path) -> dict[str, Any]:
    result = run_cmd([sys.executable, "-m", "pip", "freeze"], timeout=120)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result["stdout"])
    return {
        "command": " ".join(result["args"]),
        "ok": result["ok"],
        "returncode": result["returncode"],
        "path": str(out_path),
        "sha256": sha256_path(out_path),
        "lineCount": len([line for line in result["stdout"].splitlines() if line]),
        "stderr": result["stderr"],
    }


def collect_hardware() -> dict[str, Any]:
    nproc = run_cmd(["nproc"], timeout=10)
    nvidia = run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        timeout=20,
    )
    cuda_version = run_cmd(["nvcc", "--version"], timeout=20)
    meminfo = ""
    meminfo_path = Path("/proc/meminfo")
    if meminfo_path.exists():
        meminfo = "\n".join(meminfo_path.read_text().splitlines()[:8])
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "pythonExecutable": sys.executable,
        "pythonVersion": sys.version,
        "osCpuCount": os.cpu_count(),
        "nproc": nproc["stdout"].strip() if nproc["ok"] else None,
        "meminfoHead": meminfo,
        "gpuVisible": nvidia["ok"] and bool(nvidia["stdout"].strip()),
        "nvidiaSmi": nvidia,
        "nvccVersion": cuda_version,
    }


def collect_container_context() -> dict[str, Any]:
    interesting_env = {
        key: value
        for key, value in os.environ.items()
        if any(token in key.upper() for token in ["CONTAINER", "IMAGE", "DIGEST"])
    }
    cgroup = ""
    cgroup_path = Path("/proc/self/cgroup")
    if cgroup_path.exists():
        cgroup = cgroup_path.read_text(errors="replace")
    return {
        "imageOrDigestEnv": interesting_env,
        "procSelfCgroup": cgroup,
        "containerDigestExposed": bool(interesting_env),
    }


def dependency_files(repo_root: Path) -> list[str]:
    names = {
        "requirements.txt",
        "requirements-dev.txt",
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "environment.yml",
        "environment.yaml",
        "Pipfile",
        "poetry.lock",
    }
    return sorted(
        path.relative_to(repo_root).as_posix()
        for path in repo_root.rglob("*")
        if path.is_file() and path.name in names
    )


def input_attachment_records(workspace_root: Path) -> list[dict[str, Any]]:
    base = workspace_root / "input-attachments"
    if not base.exists():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(base.rglob("*")):
        if path.is_file():
            records.append(
                {
                    "path": path.relative_to(workspace_root).as_posix(),
                    "sizeBytes": path.stat().st_size,
                    "sha256": sha256_path(path),
                }
            )
    return records


def write_repository_audit(
    path: Path,
    manifest: dict[str, Any],
    entry_points: list[dict[str, Any]],
    file_records: list[dict[str, Any]],
) -> None:
    git_info = manifest["repository"]
    caveats = manifest["caveatsOrBlockers"]
    main_scripts = [row for row in entry_points if row["hasMain"]]
    hardcoded = [row for row in entry_points if row["hardcodedAbsolutePathLines"]]
    npy_related = [row for row in entry_points if row["npyWriteOrLoadLines"]]

    lines = [
        "# S01 Repository Audit",
        "",
        f"- Research step ID: {STEP_ID}",
        "- Completion status: audit complete",
        "- Artifacts written: environment_manifest.json, repository_file_tree.tsv, package_freeze.txt, runnable_entry_points.csv, repository_audit.md, provenance/run_manifest.json",
        "- Validation result: git metadata, tracked-file hashes, archive SHA256, package freeze, hardware report, GPU visibility, and entry-point scan were recorded.",
        f"- Caveats or blockers: {'; '.join(caveats) if caveats else 'None for the audit stage.'}",
        "- Recommended next action: run S01 smoke tests, then proceed to S02 only after Chief Scientist instruction.",
        "",
        "## Repository",
        "",
        f"- Path: `{git_info['path']}`",
        f"- Branch: `{git_info['branch']}`",
        f"- Current commit: `{git_info['head']}`",
        f"- Dirty state: `{git_info['dirtyState']}`",
        f"- Source archive ref: `{git_info['sourceArchiveRef']}`",
        f"- Source archive SHA256: `{git_info['sourceArchiveSha256']}`",
        f"- Current HEAD archive SHA256: `{git_info['currentHeadArchiveSha256']}`",
        f"- Tracked file count: {len(file_records)}",
        "",
        "## Remotes",
        "",
    ]
    for remote in git_info["remotes"]:
        lines.append(f"- `{remote}`")
    if not git_info["remotes"]:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Entry Points",
            "",
            f"- Python files scanned: {len(entry_points)}",
            f"- Scripts with `if __name__ == \"__main__\"`: {len(main_scripts)}",
            f"- Scripts with `.npy` load/save references: {len(npy_related)}",
            f"- Scripts with author-local absolute paths: {len(hardcoded)}",
            "",
            "| Script | Main | NPY refs | Absolute path refs |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for row in sorted(entry_points, key=lambda item: item["path"]):
        if row["hasMain"] or row["npyWriteOrLoadLines"] or row["hardcodedAbsolutePathLines"]:
            lines.append(
                f"| `{row['path']}` | {row['hasMain']} | "
                f"{row['npyWriteOrLoadLines']} | {row['hardcodedAbsolutePathLines']} |"
            )

    lines.extend(
        [
            "",
            "## Dependency Files",
            "",
        ]
    )
    deps = manifest["dependencyFiles"]
    if deps:
        lines.extend(f"- `{item}`" for item in deps)
    else:
        lines.append("- None found in the repository.")

    lines.extend(
        [
            "",
            "## S01 Interpretation",
            "",
            "- The checkout is the active experiment working copy for the shared GitHub folder, not a mounted read-only upstream directory.",
            "- Cell-view, Frozen Cell, Probe, metric, and mixed-Algotype code paths are identifiable.",
            "- Traditional-sort analysis scripts reference expected `.npy` outputs, but no clean original traditional-sort generator was found during S01.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="/workspace/cell-research")
    parser.add_argument("--workspace-root", default="/workspace")
    parser.add_argument("--artifacts-dir", default=os.environ.get("ARTIFACTS_DIR", "/artifacts"))
    parser.add_argument("--step-id", default=STEP_ID)
    parser.add_argument("--source-archive-ref", default="HEAD")
    args = parser.parse_args()

    if args.step_id != STEP_ID:
        raise SystemExit(f"this script is scoped to {STEP_ID}, got {args.step_id}")

    repo_root = Path(args.repo_root).resolve()
    workspace_root = Path(args.workspace_root).resolve()
    artifacts_dir = Path(args.artifacts_dir).resolve()
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    provenance_dir = artifacts_dir / "provenance"
    step_dir.mkdir(parents=True, exist_ok=True)
    provenance_dir.mkdir(parents=True, exist_ok=True)

    file_tree_path = step_dir / "repository_file_tree.tsv"
    package_freeze_path = step_dir / "package_freeze.txt"
    entry_points_path = step_dir / "runnable_entry_points.csv"
    repository_audit_path = step_dir / "repository_audit.md"
    environment_manifest_path = step_dir / "environment_manifest.json"
    run_manifest_path = provenance_dir / "run_manifest.json"

    file_records = write_file_tree(repo_root, file_tree_path)
    entry_points = scan_entry_points(repo_root, entry_points_path)
    package_freeze = collect_package_freeze(package_freeze_path)
    dirty_state = git_stdout(repo_root, "status", "--short")
    remotes = [
        line
        for line in git_stdout(repo_root, "remote", "-v").splitlines()
        if line.strip()
    ]
    source_archive_sha = git_archive_sha256(repo_root, args.source_archive_ref)
    current_archive_sha = git_archive_sha256(repo_root, "HEAD")
    tracked_pycache = [
        record["path"] for record in file_records if "__pycache__/" in record["path"]
    ]
    deps = dependency_files(repo_root)

    caveats: list[str] = []
    if not deps:
        caveats.append("No dependency manifest was found in the repository.")
    if dirty_state:
        caveats.append("The repository had uncommitted changes at audit time.")
    if tracked_pycache:
        caveats.append(f"{len(tracked_pycache)} tracked __pycache__ files are present in the archive.")
    if source_archive_sha is None:
        caveats.append(f"Could not create git archive for {args.source_archive_ref}.")

    artifacts_written = [
        str(environment_manifest_path),
        str(repository_audit_path),
        str(file_tree_path),
        str(package_freeze_path),
        str(entry_points_path),
        str(run_manifest_path),
    ]

    manifest: dict[str, Any] = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": True,
        "status": "audit_complete",
        "artifactsWritten": artifacts_written,
        "validationResult": "Repository metadata, archive checksums, tracked-file checksums, entry points, package freeze, hardware, CPU, GPU, and attachment hashes recorded.",
        "caveatsOrBlockers": caveats,
        "recommendedNextAction": "Run S01 smoke tests, then stop before S02 until the Chief Scientist workflow gives the next instruction.",
        "generatedAt": utc_now(),
        "repository": {
            "path": str(repo_root),
            "branch": git_stdout(repo_root, "branch", "--show-current"),
            "head": git_stdout(repo_root, "rev-parse", "HEAD"),
            "sourceArchiveRef": args.source_archive_ref,
            "sourceArchiveSha256": source_archive_sha,
            "currentHeadArchiveSha256": current_archive_sha,
            "dirtyState": dirty_state or "clean",
            "remotes": remotes,
            "trackedFileCount": len(file_records),
            "trackedPycacheFiles": tracked_pycache,
            "fileTreePath": str(file_tree_path),
            "entryPointsPath": str(entry_points_path),
        },
        "dependencyFiles": deps,
        "packageFreeze": package_freeze,
        "hardware": collect_hardware(),
        "container": collect_container_context(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "prefix": sys.prefix,
            "basePrefix": sys.base_prefix,
            "pathHead": sys.path[:8],
        },
        "inputAttachments": input_attachment_records(workspace_root),
    }

    write_json(environment_manifest_path, manifest)
    write_repository_audit(repository_audit_path, manifest, entry_points, file_records)

    run_manifest: dict[str, Any] = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": True,
        "status": "s01_environment_audit_complete",
        "artifactsWritten": artifacts_written,
        "validationResult": manifest["validationResult"],
        "caveatsOrBlockers": caveats,
        "recommendedNextAction": manifest["recommendedNextAction"],
        "generatedAt": utc_now(),
        "experimentId": "E01",
        "sourceRepository": manifest["repository"],
        "environmentManifest": str(environment_manifest_path),
        "packageFreeze": package_freeze,
        "hardware": manifest["hardware"],
    }
    write_json(run_manifest_path, run_manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
