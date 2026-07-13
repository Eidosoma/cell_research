#!/usr/bin/env python3
"""Package compact S04 evidence without copying the frozen public source."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys
from typing import Any, Sequence

from legacy_adapter import FROZEN_PUBLIC_COMMIT, FROZEN_PUBLIC_TREE, sha256_file, verify_source_tree


def run(command: Sequence[str], cwd: Path | None = None) -> str:
    completed = subprocess.run(
        list(command), cwd=cwd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    return completed.stdout.strip()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def docker_json_probe(image: str, source: Path | None = None) -> dict[str, Any]:
    probe_code = (
        "import json, pathlib, platform, sys; import numpy; "
        "p=pathlib.Path('/opt/legacy-adapter'); "
        "print(json.dumps({'python':sys.version,'executable':sys.executable,"
        "'platform':platform.platform(),'numpy':numpy.__version__,"
        "'imageAdapterFiles':sorted(x.name for x in p.iterdir()),"
        "'historicalSourceExistsWithoutMount':pathlib.Path('/historical-src').exists()}))"
    )
    command = [
        "docker", "run", "--rm", "--network", "none", "--read-only", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges", "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m",
    ]
    if source is not None:
        command.extend(["--mount", f"type=bind,src={source},dst=/historical-src,readonly"])
    command.extend(["--entrypoint", "python", image, "-c", probe_code])
    return json.loads(run(command))


def readonly_mount_probe(image: str, source: Path) -> dict[str, Any]:
    probe_code = (
        "import errno,json,pathlib; p=pathlib.Path('/historical-src/.s04_write_probe'); "
        "r={'path':str(p),'writeSucceeded':False}; "
        "\ntry:\n p.write_bytes(b'forbidden')\n r['writeSucceeded']=True"
        "\nexcept OSError as e:\n r.update({'errno':e.errno,'error':str(e),'readOnlyFilesystem':e.errno==errno.EROFS})"
        "\nprint(json.dumps(r))"
    )
    command = [
        "docker", "run", "--rm", "--network", "none", "--read-only", "--user", "0:0",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--pids-limit", "32",
        "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m",
        "--mount", f"type=bind,src={source},dst=/historical-src,readonly",
        "--entrypoint", "python", image, "-c", probe_code,
    ]
    return json.loads(run(command))


def build_manifest(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return {
        "schema": "e01-s04-artifact-manifest-v1",
        "researchStepId": "S04",
        "historicalSourceBytesIncluded": False,
        "files": rows,
        "fileCount": len(rows),
    }


def package(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    release = args.release.resolve()
    repo = args.repo.resolve()
    source = args.source.resolve()
    checksums = args.checksums.resolve()
    output.mkdir(parents=True, exist_ok=True)
    release.mkdir(parents=True, exist_ok=True)

    host = json.loads(args.host_validation.read_text(encoding="utf-8"))
    container = json.loads(args.container_validation.read_text(encoding="utf-8"))
    if not host.get("success") or not container.get("success"):
        raise RuntimeError("host and container validation must pass before packaging")

    packaged_at = datetime.now(timezone.utc).isoformat()
    source_post = verify_source_tree(source, checksums)
    mode_summary = {
        "sourceDirectoryMode": oct(stat.S_IMODE(source.stat().st_mode)),
        "manifestMode": oct(stat.S_IMODE(checksums.stat().st_mode)),
        "allManifestFilesHaveNoWriteBits": not source_post["modeWritableFiles"],
        "modeWritableFiles": source_post["modeWritableFiles"],
        "note": "os.access reports root effective access on the host; mode bits and container read-only mount are the relevant immutability controls.",
    }

    image_inspect = json.loads(run(["docker", "image", "inspect", args.image]))[0]
    image_probe = docker_json_probe(args.image)
    readonly_probe = readonly_mount_probe(args.image, source)
    source_after_write_probe = verify_source_tree(source, checksums)
    if readonly_probe["writeSucceeded"] or not source_after_write_probe["contentValid"]:
        raise RuntimeError("read-only source containment validation failed")

    host_runtime = json.loads(
        run(
            [
                str(args.host_python),
                "-c",
                "import json,numpy,platform,sys; print(json.dumps({'python':sys.version,'executable':sys.executable,'numpy':numpy.__version__,'platform':platform.platform()}))",
            ]
        )
    )
    docker_version = run(["docker", "version"])
    repo_files = [
        repo / "historical_backend" / name
        for name in [
            "Dockerfile", "README.md", "legacy_adapter.py", "validate_backend.py",
            "package_evidence.py", "requirements.lock",
        ]
    ]
    repo_file_rows = [
        {
            "path": path.relative_to(repo).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in repo_files
    ]

    environment = {
        "schema": "e01-s04-environment-v1",
        "researchStepId": "S04",
        "packagedAtUtc": packaged_at,
        "host": {
            "packagerPython": sys.version,
            "platform": platform.platform(),
            "cpuCount": os.cpu_count(),
            "isolatedLegacyRuntime": host_runtime,
            "isolatedEnvironmentPath": str(args.host_python.parent.parent),
        },
        "oci": {
            "imageTag": args.image,
            "imageId": image_inspect["Id"],
            "created": image_inspect["Created"],
            "architecture": image_inspect["Architecture"],
            "os": image_inspect["Os"],
            "configuredUser": image_inspect["Config"]["User"],
            "rootFilesystemLayers": image_inspect["RootFS"]["Layers"],
            "baseImage": "python:3.11.9-slim-bookworm",
            "baseImageDigest": "sha256:8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822c6dfc60c317",
            "runtimeProbe": image_probe,
            "sourceBytesPresentWithoutMount": image_probe["historicalSourceExistsWithoutMount"],
            "dockerVersionText": docker_version,
        },
        "dependencies": {
            "python": "3.11.9",
            "numpy": "1.23.5",
            "numpyWheelSha256": "58f545efd1108e647604a1b5aa809591ccd2540f468a880bedb97247e72db387",
            "basis": "Reconstruction choice: CPython 3.10/3.11 bytecode was present at frozen HEAD, and NumPy 1.23.5 is the final line that preserves the driver's warned ragged np.array conversion. Exact publication versions were not recorded.",
        },
        "adapterFiles": repo_file_rows,
    }
    write_json(output / "environment_provenance.json", environment)

    source_integrity = {
        "schema": "e01-s04-source-integrity-v1",
        "researchStepId": "S04",
        "publicCommit": FROZEN_PUBLIC_COMMIT,
        "publicTree": FROZEN_PUBLIC_TREE,
        "publicationSnapshotClaimed": False,
        "checksumLedger": str(checksums),
        "checksumLedgerSha256": sha256_file(checksums),
        "hostPreExecution": host["precheck"],
        "hostPostExecution": host["postcheck"],
        "containerPreExecution": container["precheck"],
        "containerPostExecution": container["postcheck"],
        "finalPostPackagingProbe": source_after_write_probe,
        "modeSummary": mode_summary,
        "containerReadOnlyMountProbe": readonly_probe,
        "unchanged": all(
            check["contentValid"]
            for check in [host["precheck"], host["postcheck"], container["precheck"], container["postcheck"], source_after_write_probe]
        ),
    }
    write_json(output / "source_integrity_validation.json", source_integrity)

    known_toys = {
        "schema": "e01-s04-known-toys-v1",
        "researchStepId": "S04",
        "evidenceProfile": "C-frozen-public-commit",
        "publicationSnapshotClaimed": False,
        "cleanRoomParityClaimed": False,
        "host": host["knownToys"],
        "container": container["knownToys"],
        "allPassed": all(row["passed"] for row in host["knownToys"] + container["knownToys"]),
        "interpretation": "S03 fixture relations name overlapping toy ideas only; these are full C-profile thread executions, not one-step R-profile oracle checks.",
    }
    write_json(output / "known_toy_results.json", known_toys)

    scheduler = {
        "schema": "e01-s04-scheduler-variability-v1",
        "researchStepId": "S04",
        "host": host["schedulerSummary"],
        "container": container["schedulerSummary"],
        "variabilityDetected": (
            host["schedulerSummary"]["schedulerVariabilityDetected"]
            and container["schedulerSummary"]["schedulerVariabilityDetected"]
        ),
        "interpretation": "Fixed input and seed did not fix interleavings. Swap count remained invariant at the inversion count, while trace order and comparison counts varied.",
        "limitation": "Thirty runs on one Linux host and one OCI runtime do not characterize other kernels, CPU topologies, or historical machines.",
    }
    write_json(output / "scheduler_variability.json", scheduler)

    raw_output = container["rawOutputReadability"]
    raw_output["historicalRawDataRecovered"] = False
    raw_output["substitutionForHistoricalData"] = False
    write_json(output / "raw_output_readability.json", raw_output)
    smoke_target = output / "smoke_outputs" / "generated_raw_smoke"
    smoke_target.mkdir(parents=True, exist_ok=True)
    for name in ["run.json", "sorting_steps.npy", "cell_types.npy"]:
        shutil.copy2(args.container_validation.parent / "generated_raw_smoke" / name, smoke_target / name)

    patches = [
        {
            "id": "A01", "kind": "adapter", "change": "Mount frozen source externally and verify all 91 S02 checksums before import.",
            "touchesOriginalBytes": False, "behavioralScope": "provenance and containment only",
        },
        {
            "id": "A02", "kind": "adapter", "change": "Instantiate original cell/group/probe classes without importing GUI-heavy experiment drivers.",
            "touchesOriginalBytes": False, "behavioralScope": "toy input sizing and orchestration; policy methods remain original",
        },
        {
            "id": "A03", "kind": "environment", "change": "Use reconstructed CPython 3.11.9 and NumPy 1.23.5 environment.",
            "touchesOriginalBytes": False, "behavioralScope": "runtime reconstruction chosen to preserve warned ragged-array conversion; exact publication versions are unknown",
        },
        {
            "id": "A04", "kind": "safety", "change": "Add a wall-clock safety timeout and bounded post-stop joins.",
            "touchesOriginalBytes": False, "behavioralScope": "containment only; timeout is labeled adapter_safety_timeout and never historical completion",
        },
        {
            "id": "A05", "kind": "validation", "change": "Allow an optional global random.seed override for controlled repeated runs.",
            "touchesOriginalBytes": False, "behavioralScope": "validation-only random-stream initialization; source random calls are unchanged",
        },
        {
            "id": "A06", "kind": "adapter", "change": "Expose exact with-replacement frozen-index draws plus an explicit-index toy option.",
            "touchesOriginalBytes": False, "behavioralScope": "explicit-index mode is a toy convenience and was not used in no-fault validation",
        },
        {
            "id": "A07", "kind": "output", "change": "Write S04-generated JSON and non-pickled numeric NumPy smoke arrays.",
            "touchesOriginalBytes": False, "behavioralScope": "safe adapter output only; not recovered or substituted historical data",
        },
        {
            "id": "A08", "kind": "containment", "change": "Run as UID 0 only to read S02 mode-0400 files, with all capabilities dropped, no network, no privilege escalation, and read-only mounts/rootfs.",
            "touchesOriginalBytes": False, "behavioralScope": "container access boundary only; image defaults to UID 65532",
        },
        {
            "id": "A09", "kind": "preservation-decision", "change": "Do not insert the paper-described pre-lock Bernoulli gate because it is absent from frozen public HEAD.",
            "touchesOriginalBytes": False, "behavioralScope": "preserves C-profile unconditional lock acquisition; paper contradiction remains explicit",
        },
        {
            "id": "A10", "kind": "adapter", "change": "Re-express the frozen driver is_sorted and no_cells_should_move monitor predicates in the external adapter.",
            "touchesOriginalBytes": False, "behavioralScope": "predicate text is logic-equivalent; avoids top-level Tk/visualization imports",
        },
        {
            "id": "A11", "kind": "containment", "change": "Set PYTHONDONTWRITEBYTECODE=1 to prevent import cache writes into the frozen tree.",
            "touchesOriginalBytes": False, "behavioralScope": "filesystem side effects only",
        },
    ]
    patch_ledger = {
        "schema": "e01-s04-patch-ledger-v1",
        "researchStepId": "S04",
        "patchesAppliedToOriginalSource": 0,
        "originalSourceBytesModified": False,
        "adapterEntries": patches,
        "adapterEntryCount": len(patches),
    }
    write_json(output / "patch_ledger.json", patch_ledger)
    patch_lines = [
        "# S04 compatibility adapter / patch ledger", "",
        "No patch was applied to the frozen source. Every compatibility or containment action is external:", "",
        "| ID | Kind | External change | Behavioral scope |", "| --- | --- | --- | --- |",
    ]
    patch_lines.extend(
        f"| {row['id']} | {row['kind']} | {row['change']} | {row['behavioralScope']} |"
        for row in patches
    )
    (output / "patch_ledger.md").write_text("\n".join(patch_lines) + "\n", encoding="utf-8")

    missing = {
        "schema": "e01-s04-missing-evidence-v1",
        "researchStepId": "S04",
        "items": [
            {"id": "M01", "missing": "exact publication commit or tag", "impact": "C execution cannot be attributed as the publication snapshot"},
            {"id": "M02", "missing": "publication Python, NumPy, OS, kernel, and hardware versions", "impact": "historical timing cannot be reconstructed exactly"},
            {"id": "M03", "missing": "dependency lock or environment file", "impact": "CPython 3.11.9 / NumPy 1.23.5 are documented reconstruction choices"},
            {"id": "M04", "missing": "all historical .npy output bytes", "impact": "raw parity cannot be tested; S04 smoke arrays are not substitutes"},
            {"id": "M05", "missing": "license or reuse grant", "impact": "source and image must remain separate; no original bytes are packaged"},
            {"id": "M06", "missing": "historical scheduler trace or seed records", "impact": "exact event replay is impossible"},
            {"id": "M07", "missing": "code for the paper's pre-lock random attempt gate at frozen HEAD", "impact": "paper and C scheduler semantics remain contradictory"},
            {"id": "M08", "missing": "stuck-fault implementation", "impact": "S04 cannot execute that paper condition faithfully"},
            {"id": "M09", "missing": "traditional-controller generators and exact figure recipes", "impact": "S04 validates only recovered cell-view policy classes"},
            {"id": "M10", "missing": "cross-kernel/cross-machine scheduler evidence", "impact": "observed variability is bounded to the tested host/runtime"},
        ],
    }
    write_json(output / "missing_evidence.json", missing)

    license_boundary = {
        "schema": "e01-s04-license-boundary-v1",
        "researchStepId": "S04",
        "s02LicenseReview": "/artifacts/research_steps/S02/license_review.json",
        "s02LawfulManifest": "/artifacts/research_steps/S02/lawful_source_manifest.json",
        "licenseGrantFound": False,
        "historicalSourceBytesInS04Artifacts": False,
        "historicalSourceBytesInOciImage": False,
        "ociSourceDelivery": "runtime-only read-only bind mount from S02 cache quarantine",
        "redistributionDecision": "Do not package or redistribute frozen public source bytes or archives.",
    }
    write_json(output / "license_and_source_boundary.json", license_boundary)

    oci_validation = {
        "schema": "e01-s04-oci-validation-v1",
        "researchStepId": "S04",
        "imageId": image_inspect["Id"],
        "defaultUser": image_inspect["Config"]["User"],
        "imageProbe": image_probe,
        "readOnlySourceMountProbe": readonly_probe,
        "sourceAbsentFromImage": not image_probe["historicalSourceExistsWithoutMount"],
        "sourceMountRejectedWrite": readonly_probe.get("readOnlyFilesystem", False),
        "containerValidationSuccess": container["success"],
        "runtimeHardening": [
            "--network none", "--read-only", "--cap-drop ALL", "--security-opt no-new-privileges",
            "--pids-limit 128", "read-only source and checksum bind mounts", "bounded tmpfs",
        ],
    }
    write_json(output / "oci_validation.json", oci_validation)

    validation = {
        "schema": "e01-s04-validation-summary-v1",
        "researchStepId": "S04",
        "hostChecks": host["checks"],
        "containerChecks": container["checks"],
        "sourceUnchanged": source_integrity["unchanged"],
        "knownToysPassed": known_toys["allPassed"],
        "rawOutputsReadableWithoutPickle": container["checks"]["rawOutputsReadableWithoutPickle"],
        "historicalStyleRaggedOutputReadableWithTrustedPickle": container["checks"]["historicalStyleRaggedOutputReadableWithTrustedPickle"],
        "schedulerVariabilityAssessed": scheduler["variabilityDetected"],
        "sourceAbsentFromImage": oci_validation["sourceAbsentFromImage"],
        "readOnlyMountEnforced": oci_validation["sourceMountRejectedWrite"],
    }
    validation["success"] = all(
        [
            all(validation["hostChecks"].values()), all(validation["containerChecks"].values()),
            validation["sourceUnchanged"], validation["knownToysPassed"],
            validation["rawOutputsReadableWithoutPickle"], validation["schedulerVariabilityAssessed"],
            validation["historicalStyleRaggedOutputReadableWithTrustedPickle"],
            validation["sourceAbsentFromImage"], validation["readOnlyMountEnforced"],
        ]
    )
    write_json(output / "validation_summary.json", validation)

    recipe_dir = output / "legacy_execution_recipe"
    recipe_dir.mkdir(parents=True, exist_ok=True)
    recipe_manifest = {
        "schema": "e01-s04-recipe-manifest-v1",
        "researchStepId": "S04",
        "repository": str(repo),
        "recipePath": "historical_backend/Dockerfile",
        "adapterPath": "historical_backend/legacy_adapter.py",
        "validationPath": "historical_backend/validate_backend.py",
        "lockPath": "historical_backend/requirements.lock",
        "repositoryFiles": repo_file_rows,
        "baseImageDigest": environment["oci"]["baseImageDigest"],
        "builtImageId": environment["oci"]["imageId"],
        "historicalSourceIncluded": False,
        "sourceRuntimeMount": str(source),
        "checksumRuntimeMount": str(checksums),
    }
    write_json(recipe_dir / "recipe_manifest.json", recipe_manifest)
    (recipe_dir / "README.md").write_text(
        "# S04 legacy execution recipe index\n\n"
        "The executable recipe is repository-backed at `historical_backend/`. This artifact directory "
        "records its paths and hashes without copying repository code or the unlicensed historical source. "
        "Build only that directory as the Docker context. Supply the S02 quarantine at runtime as a read-only "
        "mount; see the repository README for the validated hardened command.\n",
        encoding="utf-8",
    )

    commands = [
        "uv python install 3.11.9",
        "uv venv --python 3.11.9 /cache/e01_s04/venv311",
        "uv pip install --python /cache/e01_s04/venv311/bin/python --require-hashes -r historical_backend/requirements.lock",
        "PYTHONDONTWRITEBYTECODE=1 /cache/e01_s04/venv311/bin/python historical_backend/validate_backend.py --source /cache/e01_s02/historical-worktree --checksums /artifacts/research_steps/S02/source_tree_checksums.sha256 --output /cache/e01_s04/host-output --repetitions 30",
        "docker build --pull=false -t e01-s04-legacy-adapter:py311 /workspace/cell-research/historical_backend",
        "docker run --rm --network none --read-only --user 0:0 --cap-drop ALL --security-opt no-new-privileges --pids-limit 128 --tmpfs /tmp:rw,noexec,nosuid,size=64m --mount type=bind,src=/cache/e01_s02/historical-worktree,dst=/historical-src,readonly --mount type=bind,src=/artifacts/research_steps/S02/source_tree_checksums.sha256,dst=/provenance/source_tree_checksums.sha256,readonly --mount type=bind,src=/cache/e01_s04/container-output,dst=/output --entrypoint python e01-s04-legacy-adapter:py311 /opt/legacy-adapter/validate_backend.py --source /historical-src --checksums /provenance/source_tree_checksums.sha256 --output /output --repetitions 30",
        "python3 -m unittest discover -s tests -v",
    ]
    (output / "validation_commands.log").write_text("\n".join(commands) + "\n", encoding="utf-8")

    release_manifest = {
        "schema": "e01-historical-backend-release-v1",
        "researchStepId": "S04",
        "classification": "executable frozen-public-commit backend, not an exact publication snapshot",
        "repositoryCommitAtPackaging": run(["git", "rev-parse", "HEAD"], cwd=repo),
        "repositoryPaths": [row["path"] for row in repo_file_rows],
        "recipeManifest": "/artifacts/research_steps/S04/legacy_execution_recipe/recipe_manifest.json",
        "validationSummary": "/artifacts/research_steps/S04/validation_summary.json",
        "patchLedger": "/artifacts/research_steps/S04/patch_ledger.json",
        "historicalSourceIncluded": False,
        "lawfulRuntimeSource": str(source),
    }
    write_json(release / "manifest.json", release_manifest)
    (release / "README.md").write_text(
        "# Historical backend release index\n\n"
        "This is a lawful pointer release. It contains no frozen public source or OCI image layer. "
        "The repository-backed adapter and OCI recipe execute the checksum-verified S02 quarantine "
        "through a read-only runtime mount. The result is attributable to frozen public HEAD only, "
        "not to an exact publication snapshot.\n",
        encoding="utf-8",
    )

    write_json(output / "artifact_manifest.json", build_manifest(output))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-validation", required=True, type=Path)
    parser.add_argument("--container-validation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--checksums", required=True, type=Path)
    parser.add_argument("--host-python", required=True, type=Path)
    parser.add_argument("--image", default="e01-s04-legacy-adapter:py311")
    args = parser.parse_args(argv)
    package(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
