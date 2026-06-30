#!/usr/bin/env python3
"""Rebuild and document the S01 source/runtime environment.

This script intentionally keeps the downloaded public repository under
/cache and writes only compact manifests/logs under ARTIFACTS_DIR.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import venv
from datetime import datetime, timezone
from pathlib import Path


STEP_ID = "S01"
EXPERIMENT_ID = "E01"
DEFAULT_REPO_URL = "https://github.com/Zhangtaining/cell_research.git"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> dict[str, object]:
    started = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError as exc:
        result = {
            "command": cmd,
            "cwd": str(cwd) if cwd else None,
            "returncode": 127,
            "stdout": "",
            "stderr": repr(exc),
            "elapsedSeconds": time.time() - started,
        }
        if check:
            raise RuntimeError(json.dumps(result, indent=2)) from exc
        return result
    elapsed = time.time() - started
    result = {
        "command": cmd,
        "cwd": str(cwd) if cwd else None,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "elapsedSeconds": elapsed,
    }
    if check and proc.returncode != 0:
        raise RuntimeError(json.dumps(result, indent=2))
    return result


def sha256_bytes(data: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(data)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def remove_cache_child(path: Path, cache_root: Path) -> None:
    resolved = path.resolve()
    root = cache_root.resolve()
    if root not in resolved.parents and resolved != root:
        raise ValueError(f"Refusing to remove path outside cache root: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def tracked_file_manifest(repo_dir: Path) -> list[dict[str, object]]:
    files = run(["git", "ls-files", "-z"], cwd=repo_dir)["stdout"].split("\0")
    entries: list[dict[str, object]] = []
    for name in sorted(f for f in files if f):
        path = repo_dir / name
        entries.append(
            {
                "path": name,
                "sizeBytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return entries


def dependency_manifests(repo_dir: Path) -> list[str]:
    names = {
        "requirements.txt",
        "requirements-dev.txt",
        "pyproject.toml",
        "setup.py",
        "environment.yml",
        "environment.yaml",
        "Pipfile",
    }
    return sorted(str(path.relative_to(repo_dir)) for path in repo_dir.rglob("*") if path.name in names)


def static_network_import_scan(repo_dir: Path) -> list[dict[str, object]]:
    network_markers = ("requests", "urllib", "httpx", "socket", "ftplib")
    hits: list[dict[str, object]] = []
    for path in sorted(repo_dir.rglob("*.py")):
        rel = path.relative_to(repo_dir)
        try:
            for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith(("import ", "from ")) and any(marker in stripped for marker in network_markers):
                    hits.append({"path": str(rel), "line": lineno, "text": stripped})
        except OSError as exc:
            hits.append({"path": str(rel), "error": repr(exc)})
    return hits


def package_versions(python_exe: Path) -> dict[str, object]:
    pip_list = run([str(python_exe), "-m", "pip", "list", "--format=json"], check=False)
    version_probe = (
        "import json, platform, sys\n"
        "mods = ['numpy','scipy','pandas','matplotlib','sklearn']\n"
        "out = {'python': sys.version, 'platform': platform.platform(), 'imports': {}}\n"
        "for name in mods:\n"
        "    try:\n"
        "        mod = __import__(name)\n"
        "        out['imports'][name] = getattr(mod, '__version__', 'available')\n"
        "    except Exception as exc:\n"
        "        out['imports'][name] = {'error': repr(exc)}\n"
        "print(json.dumps(out, sort_keys=True))\n"
    )
    import_probe = run([str(python_exe), "-c", version_probe], check=False)
    parsed_pip: object
    try:
        parsed_pip = json.loads(pip_list["stdout"]) if pip_list["returncode"] == 0 else []
    except json.JSONDecodeError:
        parsed_pip = []
    parsed_imports: object
    try:
        parsed_imports = json.loads(import_probe["stdout"]) if import_probe["returncode"] == 0 else {}
    except json.JSONDecodeError:
        parsed_imports = {}
    return {
        "pipListReturnCode": pip_list["returncode"],
        "pipList": parsed_pip,
        "importProbeReturnCode": import_probe["returncode"],
        "importProbe": parsed_imports,
        "importProbeStderr": import_probe["stderr"],
    }


def hardware_metadata() -> dict[str, object]:
    nvidia = run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        check=False,
    )
    cpu_model = ""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpuModel": cpu_model,
        "osCpuCount": os.cpu_count(),
        "nproc": run(["nproc"], check=False)["stdout"].strip(),
        "pythonExecutable": sys.executable,
        "pythonVersion": sys.version,
        "gitVersion": run(["git", "--version"], check=False)["stdout"].strip(),
        "gpuQueryReturnCode": nvidia["returncode"],
        "gpuQueryStdout": nvidia["stdout"].strip(),
        "gpuQueryStderr": nvidia["stderr"].strip(),
    }


def create_venv(venv_dir: Path) -> Path:
    if not venv_dir.exists():
        builder = venv.EnvBuilder(system_site_packages=True, with_pip=True, clear=False)
        builder.create(venv_dir)
    python_exe = venv_dir / "bin" / "python"
    if not python_exe.exists():
        raise FileNotFoundError(f"Expected venv Python not found: {python_exe}")
    return python_exe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--work-dir", default="/cache/e01_s01")
    parser.add_argument("--artifacts-dir", default=os.environ.get("ARTIFACTS_DIR", "/artifacts"))
    parser.add_argument("--force", action="store_true", help="Remove and rebuild /cache/e01_s01 child outputs.")
    args = parser.parse_args()

    started = time.time()
    work_dir = Path(args.work_dir)
    artifacts_dir = Path(args.artifacts_dir)
    cache_root = work_dir
    original_repo = work_dir / "original_repo"
    patched_repo = work_dir / "patched_repo"
    venv_dir = work_dir / "venv"
    code_dir = artifacts_dir / "code"
    logs_dir = artifacts_dir / "logs"
    research_dir = artifacts_dir / "research_steps" / STEP_ID
    code_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    research_dir.mkdir(parents=True, exist_ok=True)

    if args.force:
        work_dir.mkdir(parents=True, exist_ok=True)
        for child in (original_repo, patched_repo, venv_dir):
            remove_cache_child(child, cache_root)

    work_dir.mkdir(parents=True, exist_ok=True)

    commands: list[dict[str, object]] = []
    if not original_repo.exists():
        commands.append(run(["git", "clone", args.repo_url, str(original_repo)]))
    else:
        commands.append(run(["git", "status", "--short"], cwd=original_repo))

    if not patched_repo.exists():
        commands.append(run(["git", "clone", str(original_repo), str(patched_repo)]))
    else:
        commands.append(run(["git", "status", "--short"], cwd=patched_repo))

    commit = run(["git", "rev-parse", "HEAD"], cwd=original_repo)["stdout"].strip()
    branch = run(["git", "branch", "--show-current"], cwd=original_repo, check=False)["stdout"].strip()
    remote_url = run(["git", "remote", "get-url", "origin"], cwd=original_repo, check=False)["stdout"].strip()
    remote_head = run(["git", "ls-remote", "--symref", args.repo_url, "HEAD"], check=False)
    latest_commit_iso = run(["git", "show", "-s", "--format=%cI", "HEAD"], cwd=original_repo)["stdout"].strip()
    archive = subprocess.run(
        ["git", "archive", "--format=tar", "HEAD"],
        cwd=str(original_repo),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    ).stdout
    archive_sha256 = sha256_bytes(archive)

    patch_path = code_dir / "e01_s01_compatibility.patch"
    patch_text = run(["git", "diff", "--no-ext-diff"], cwd=patched_repo)["stdout"]
    patch_path.write_text(patch_text, encoding="utf-8")

    venv_python = create_venv(venv_dir)
    dep_manifests = dependency_manifests(original_repo)
    install_commands: list[dict[str, object]] = []
    if dep_manifests:
        for manifest in dep_manifests:
            if Path(manifest).name.startswith("requirements"):
                install_commands.append(
                    run([str(venv_python), "-m", "pip", "install", "-r", manifest], cwd=patched_repo, check=False)
                )

    original_manifest = {
        "schema": "eidosoma.e01_s01.original_repo_manifest.v1",
        "researchStepId": STEP_ID,
        "experimentId": EXPERIMENT_ID,
        "retrievedAtUtc": utc_now(),
        "repoUrlRequested": args.repo_url,
        "repoUrlResolved": remote_url,
        "remoteHead": {
            "returncode": remote_head["returncode"],
            "stdout": remote_head["stdout"],
            "stderr": remote_head["stderr"],
        },
        "originalRepoPath": str(original_repo),
        "patchedRepoPath": str(patched_repo),
        "commit": commit,
        "commitDate": latest_commit_iso,
        "branch": branch,
        "archive": {
            "method": "git archive --format=tar HEAD",
            "sha256": archive_sha256,
            "sizeBytes": len(archive),
        },
        "trackedFiles": tracked_file_manifest(original_repo),
        "dependencyManifests": dep_manifests,
        "dependencyInstallCommands": install_commands,
        "compatibilityPatch": {
            "path": str(patch_path),
            "sha256": sha256_file(patch_path),
            "appliedPatchCount": 0 if patch_text == "" else 1,
            "note": "No compatibility patch was required for S01 smoke tests." if patch_text == "" else "Patch diff captured.",
        },
        "staticNetworkImportScan": static_network_import_scan(original_repo),
    }

    original_manifest_path = code_dir / "original_repo_manifest.json"
    write_json(original_manifest_path, original_manifest)

    run_manifest_path = artifacts_dir / "run_manifest.json"
    run_manifest = {
        "schema": "eidosoma.e01.run_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "createdAtUtc": utc_now(),
        "updatedAtUtc": utc_now(),
        "status": "setup_complete",
        "repository": {
            "url": args.repo_url,
            "commit": commit,
            "commitDate": latest_commit_iso,
            "branch": branch,
            "archiveSha256": archive_sha256,
            "originalRepoPath": str(original_repo),
            "patchedRepoPath": str(patched_repo),
        },
        "workspaceRepository": {
            "path": str(Path.cwd()),
            "commit": run(["git", "rev-parse", "HEAD"], cwd=Path.cwd(), check=False)["stdout"].strip(),
            "branch": run(["git", "branch", "--show-current"], cwd=Path.cwd(), check=False)["stdout"].strip(),
            "statusShort": run(["git", "status", "--short"], cwd=Path.cwd(), check=False)["stdout"].splitlines(),
        },
        "runtime": {
            "venvPath": str(venv_dir),
            "venvPython": str(venv_python),
            "systemSitePackages": True,
            "packageVersions": package_versions(venv_python),
            "hardware": hardware_metadata(),
            "threadPolicy": {
                "maxCpuCoresAllowedByPlan": 8,
                "s01Execution": "Serial setup; smoke tests use one small multithreaded cell-view run per Algotype.",
            },
        },
        "seedPolicy": {
            "s01SmokeTests": "Fixed seed in scripts/e01_s01_smoke_tests.py; paper-scale seeds deferred to S03.",
        },
        "commands": commands,
        "artifacts": {
            "originalRepoManifest": str(original_manifest_path),
            "compatibilityPatch": str(patch_path),
        },
        "checksums": {
            str(original_manifest_path): sha256_file(original_manifest_path),
            str(patch_path): sha256_file(patch_path),
        },
        "wallTimeSeconds": time.time() - started,
    }
    write_json(run_manifest_path, run_manifest)

    setup_log_path = logs_dir / "e01_s01_setup_environment.log"
    setup_log_path.write_text(
        "\n".join(
            [
                f"S01 setup completed at {utc_now()}",
                f"Repository: {args.repo_url}",
                f"Commit: {commit}",
                f"Archive SHA256: {archive_sha256}",
                f"Original repo: {original_repo}",
                f"Patched repo: {patched_repo}",
                f"Venv Python: {venv_python}",
                f"Dependency manifests: {dep_manifests if dep_manifests else 'none'}",
                f"Compatibility patch: {patch_path} ({'empty' if patch_text == '' else 'non-empty'})",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
