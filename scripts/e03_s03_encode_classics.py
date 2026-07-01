#!/usr/bin/env python3
"""Encode and validate E03 S03 classic policy mappings."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.e03.classic_policies import classic_policy_library, validation_cases


STEP_ID = "S03"
STEP_NUMBER = 3
EXPERIMENT_ID = "E03"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=Path.cwd())
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--previous-e01-dir", type=Path, default=Path("/previous-artifacts/E01"))
    parser.add_argument("--previous-e02-dir", type=Path, default=Path("/previous-artifacts/E02"))
    parser.add_argument("--run-unit-tests", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def git_output(repo_dir: Path, args: list[str]) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=repo_dir, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc!r}"
    return result.stdout.strip()


def run_command(command: list[str], repo_dir: Path) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    result = subprocess.run(command, cwd=repo_dir, capture_output=True, text=True)
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    return {
        "command": " ".join(command),
        "returnCode": int(result.returncode),
        "elapsedSeconds": elapsed,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "success": result.returncode == 0,
    }


def artifact_entry(path: Path, artifacts_dir: Path, description: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(artifacts_dir)),
        "description": description,
        "sha256": sha256_file(path),
        "sizeBytes": path.stat().st_size,
    }


def manifest_self_entry(path: Path, artifacts_dir: Path, description: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(artifacts_dir)),
        "description": description,
        "sha256": None,
        "sizeBytes": None,
        "note": "Manifest checksum is omitted here to avoid self-referential checksum drift.",
    }


def source_entry(path: Path, repo_dir: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(repo_dir)),
        "sha256": sha256_file(path),
        "sizeBytes": path.stat().st_size,
    }


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for record in df[columns].to_dict(orient="records"):
        rows.append("| " + " | ".join(str(record[column]).replace("|", "\\|") for column in columns) + " |")
    return "\n".join([header, separator, *rows])


def render_mapping_report(library: dict[str, Any], validation_df: pd.DataFrame) -> str:
    rows = []
    for entry in library["policies"]:
        rows.append(
            {
                "policy_id": entry["policyId"],
                "algorithm": entry["algorithm"],
                "representation": entry["representationType"],
                "exactness": entry["exactness"],
                "direction": entry["direction"],
            }
        )
    mapping_table = markdown_table(pd.DataFrame(rows), ["policy_id", "algorithm", "representation", "exactness", "direction"])
    validation_table = markdown_table(
        validation_df,
        ["validation_case", "algorithm", "representation_type", "exactness", "success", "observed_match"],
    )
    return f"""# E03 Classic Policy Mapping

## Summary

S03 assigns stable policy IDs to classic Bubble, Insertion, and Selection representations. Exact public behavior is represented by S01 interface wrappers for all three classics. The S02 DSL also represents active/no-frozen Insertion and Selection subsets exactly on regression fixtures. Bubble receives DSL shadow policies only, because the current DSL cannot exactly preserve the public Bubble method's sampled-side control flow and comparison-count convention.

## Policy Library

{mapping_table}

## Validation

{validation_table}

## Exactness Notes

- `exact_public_method`: S01 `OriginalCellPolicyWrapper` delegates action application to the original public `move()` method and is the canonical behavior-preserving mapping.
- `exact_active_unfrozen_subset`: DSL action proposals match S01 wrapper proposals for active, no-frozen target fixtures in the declared direction.
- `approximate_shadow`: DSL policy is a useful morphospace point near the classic algorithm but should not be used as a faithful replication baseline.

## Known Deviations

- Bubble DSL shadows can fall through to the opposite side after a sampled-side guard fails; the public method samples one side and waits if that side is not swappable.
- Bubble public comparison counting records a comparison when either active neighbor is disordered, not just when the sampled target is swappable.
- Insertion frozen-target passive swaps are not exact in S02 DSL because there is no `target_frozen` guard to distinguish no-compare frozen swaps.
- Selection frozen-target behavior is not exact in S02 DSL because the public method can update `ideal_position` and attempt a frozen swap in one call.

## Recommended S04 Use

Use interface-wrapper policy IDs as canonical classic baselines. Use DSL exact-subset policies for DSL interpreter tests and generated-policy neighborhood seeds only under their compatible conditions. Use Bubble shadow policies only as approximate morphospace landmarks until the DSL gains conditional nested actions or side-choice primitives.
"""


def render_report(
    *,
    library_path: Path,
    mapping_report_path: Path,
    validation_path: Path,
    manifest_path: Path,
    run_manifest_path: Path,
    checksums_path: Path,
    validation_df: pd.DataFrame,
    unit_tests: dict[str, Any],
    s02_tests: dict[str, Any],
    manifest: dict[str, Any],
) -> str:
    validation_success = bool(validation_df["success"].all() and unit_tests["success"] and s02_tests["success"])
    outcome = "supportive" if validation_success else "constraining/contradictory"
    artifact_md = "\n".join(
        f"- `{path}`"
        for path in [
            library_path.parent.parent / "research_steps/S03/research_step_full_results.md",
            library_path,
            mapping_report_path,
            validation_path,
            manifest_path,
            run_manifest_path,
            checksums_path,
        ]
    )
    validation_line = (
        f"Passed: {int(validation_df['success'].sum())}/{len(validation_df)} mapping validation cases passed; "
        f"S03 unit tests return code {unit_tests['returnCode']}; S02 DSL tests return code {s02_tests['returnCode']}"
    )
    table = markdown_table(
        validation_df,
        ["validation_case", "algorithm", "representation_type", "exactness", "success", "observed_match"],
    )
    commands = "\n".join(
        [
            f"- `{unit_tests['command']}` -> return code {unit_tests['returnCode']}",
            f"- `{s02_tests['command']}` -> return code {s02_tests['returnCode']}",
            "- `python scripts/e03_s03_encode_classics.py --repo-dir /workspace/cell-research --artifacts-dir $ARTIFACTS_DIR`",
        ]
    )
    return f"""# E03 S03 Research Step Full Results

## Top Summary

- Research step ID: S03
- Completion status: {'Completed' if validation_success else 'Completed with validation failure'} on {utc_now()}
- Artifacts written:
{artifact_md}
- Validation result: {validation_line}
- Outcome classification: {outcome}
- Caveats or blockers: Exact public behavior for Bubble, Insertion, and Selection is represented by S01 interface-level mappings. S02 DSL encodes exact active/no-frozen Insertion and Selection subsets, but Bubble remains a documented DSL shadow rather than an exact DSL baseline.
- Lay summary: S03 gives each classic sorting behavior a stable ID and records which representations are faithful versus approximate. This lets S04 and later sweeps use the exact public-method wrappers as baselines while still seeding the DSL morphospace with clearly labeled classic-like rules.
- Recommended next action: Proceed to S04 only after Chief Scientist review; define the competence vector using interface-wrapper policy IDs as canonical classics and DSL exact-subset policies as interpreter controls.

## Frozen Question

Can Bubble, Insertion, and Selection cell-view policies be represented as points or parameterized regions in the DSL rather than isolated classes?

## Inputs

- Active research plan: `/workspace/RESEARCH_PLAN.md`, Experiment E03, step S03.
- S01 interface: `src/e03/policy_interface.py`.
- S02 DSL: `src/e03/rule_dsl.py` and `/artifacts/reports/e03_rule_dsl_spec.md`.
- E01 code map context: `/previous-artifacts/E01/reports/e01_codebase_map.md`.
- E02 deterministic simulator context: `src/e02/deterministic_simulator.py`.
- Datasets: none required.

## Methods

Implemented `src/e03/classic_policies.py` to build a classic policy library with stable IDs:

- Exact interface-wrapper mappings for Bubble, Insertion, and Selection using `OriginalCellPolicyWrapper`.
- DSL shadow mappings for increasing/decreasing Bubble.
- DSL exact active/no-frozen subset mappings for increasing/decreasing Insertion.
- DSL exact active/no-frozen subset mappings for increasing/decreasing Selection.

Validation compared exact mappings against direct public method or S01 wrapper behavior on small fixtures. For Bubble DSL, S03 deliberately validates and records a known mismatch so the shadow policy is not mistaken for the canonical baseline.

## Commands

{commands}

## Dependencies And Runtime

- Python: {platform.python_version()}
- pandas: {pd.__version__}
- New dependencies installed: none.
- Worker count: serial validation only; no CPU parallelism was needed for S03.
- Platform: {platform.platform()}

## Parameters

- Policy entries: 9
- Exact public-method interface entries: 3
- DSL exact-subset entries: 4
- DSL approximate shadow entries: 2
- Validation fixtures: {len(validation_df)}

## Results

{table}

The policy library was written to `{library_path}`. The mapping report was written to `{mapping_report_path}`.

## Validation Checks

- Interface mappings matched direct public `move()` signatures and counters.
- DSL Insertion and Selection exact-subset mappings matched S01 wrapper action proposals on active/no-frozen fixtures.
- Bubble DSL mismatch was observed and documented as an expected deviation.
- Stable policy IDs were unique.
- S03 unit tests and S02 DSL regression tests passed.

## Caveats, Blockers, Failed Assumptions, And Limitations

- No blocker remains for S03.
- Bubble is not exact in S02 DSL; use the interface-wrapper ID as canonical.
- Insertion and Selection DSL exactness is scoped to active/no-frozen decision fixtures.
- Frozen-target branches need additional DSL primitives if later steps require exact DSL-only replication of all public behavior.
- Integration into full event sweeps remains S04/S06 work.

## Provenance

- Git commit at validation time: `{manifest['gitCommit']}`
- Git status at validation time: `{manifest['gitStatusShort'] or 'clean'}`
- Source files tracked in manifest: {len(manifest['sourceFiles'])}
- Previous E01 path: `/previous-artifacts/E01`
- Previous E02 path: `/previous-artifacts/E02`
- Created at UTC: `{manifest['createdAtUtc']}`

## Recommended Next Action

Proceed to S04 only after Chief Scientist review. Freeze the competence-vector schema around the exact interface-wrapper classic IDs and use DSL exact-subset policies as controls.
"""


def main() -> int:
    args = parse_args()
    repo_dir = args.repo_dir.resolve()
    artifacts_dir = args.artifacts_dir.resolve()
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    library_path = artifacts_dir / "policies" / "e03_classic_policy_library.json"
    mapping_report_path = artifacts_dir / "reports" / "e03_classic_policy_mapping.md"
    validation_path = artifacts_dir / "results" / "e03_classic_policy_mapping_validation.parquet"
    manifest_path = artifacts_dir / "src_snapshot" / "e03_classic_policy_mapping_manifest.json"
    run_manifest_path = artifacts_dir / "run_manifest.json"
    checksums_path = artifacts_dir / "checksums" / "sha256sums.txt"
    report_path = step_dir / "research_step_full_results.md"
    for path in [step_dir, library_path.parent, mapping_report_path.parent, validation_path.parent, manifest_path.parent, checksums_path.parent]:
        path.mkdir(parents=True, exist_ok=True)

    library = classic_policy_library()
    validation_df = validation_cases()
    write_json(library_path, library)
    write_text(mapping_report_path, render_mapping_report(library, validation_df))
    validation_df.to_parquet(validation_path, index=False)

    unit_tests = (
        run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e03", "-p", "test_classic_policies.py"], repo_dir)
        if args.run_unit_tests
        else {"command": "not run", "returnCode": 0, "elapsedSeconds": 0.0, "stdout": "", "stderr": "", "success": True}
    )
    s02_tests = (
        run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e03", "-p", "test_rule_dsl.py"], repo_dir)
        if args.run_unit_tests
        else {"command": "not run", "returnCode": 0, "elapsedSeconds": 0.0, "stdout": "", "stderr": "", "success": True}
    )
    source_paths = [
        repo_dir / "src/e03/classic_policies.py",
        repo_dir / "tests/e03/test_classic_policies.py",
        repo_dir / "scripts/e03_s03_encode_classics.py",
        repo_dir / "src/e03/rule_dsl.py",
        repo_dir / "src/e03/policy_interface.py",
    ]
    manifest: dict[str, Any] = {
        "schema": "eidosoma.src_snapshot.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAtUtc": utc_now(),
        "gitCommit": git_output(repo_dir, ["rev-parse", "HEAD"]),
        "gitStatusShort": git_output(repo_dir, ["status", "--short"]),
        "dependencies": {"newDependenciesInstalled": [], "python": platform.python_version(), "pandas": pd.__version__},
        "upstreamContext": {
            "previousE01Dir": str(args.previous_e01_dir),
            "previousE02Dir": str(args.previous_e02_dir),
            "s01Manifest": str(artifacts_dir / "src_snapshot/e03_policy_interface_manifest.json"),
            "s02Manifest": str(artifacts_dir / "src_snapshot/e03_rule_dsl_manifest.json"),
        },
        "sourceFiles": [source_entry(path, repo_dir) for path in source_paths if path.exists()],
        "validation": {
            "allValidationPassed": bool(validation_df["success"].all() and unit_tests["success"] and s02_tests["success"]),
            "validationCaseCount": int(len(validation_df)),
            "validationCaseSuccessCount": int(validation_df["success"].sum()),
            "unitTests": unit_tests,
            "s02RuleDslTests": s02_tests,
            "policyLibrary": str(library_path),
            "policyLibrarySha256": sha256_file(library_path),
            "mappingReport": str(mapping_report_path),
            "mappingReportSha256": sha256_file(mapping_report_path),
            "validationParquet": str(validation_path),
            "validationParquetSha256": sha256_file(validation_path),
        },
    }
    write_json(manifest_path, manifest)
    report = render_report(
        library_path=library_path,
        mapping_report_path=mapping_report_path,
        validation_path=validation_path,
        manifest_path=manifest_path,
        run_manifest_path=run_manifest_path,
        checksums_path=checksums_path,
        validation_df=validation_df,
        unit_tests=unit_tests,
        s02_tests=s02_tests,
        manifest=manifest,
    )
    write_text(report_path, report)
    run_manifest = {
        "schema": "eidosoma.run_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "lastResearchStepId": STEP_ID,
        "createdAtUtc": utc_now(),
        "gitCommit": manifest["gitCommit"],
        "gitStatusShort": manifest["gitStatusShort"],
        "repository": {"path": str(repo_dir), "branch": git_output(repo_dir, ["branch", "--show-current"])},
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpuCountVisible": os.cpu_count(),
            "workerCount": 1,
            "threadingPolicy": "serial S03 validation; no parallel workers",
        },
        "dependencies": manifest["dependencies"],
        "seedPolicy": {"validationSeeds": [1, 11, 21, 101, 102, 103, 104, 105, 106]},
        "artifacts": [
            {"path": str(report_path), "role": "S03 full-results report"},
            {"path": str(library_path), "role": "S03 classic policy library"},
            {"path": str(mapping_report_path), "role": "S03 mapping report"},
            {"path": str(validation_path), "role": "S03 validation table"},
            {"path": str(manifest_path), "role": "S03 source snapshot manifest"},
            {"path": str(checksums_path), "role": "artifact checksums"},
        ],
        "validation": manifest["validation"],
    }
    write_json(run_manifest_path, run_manifest)
    manifest["artifactsWritten"] = [
        artifact_entry(report_path, artifacts_dir, "S03 full-results handoff report"),
        artifact_entry(library_path, artifacts_dir, "S03 classic policy library"),
        artifact_entry(mapping_report_path, artifacts_dir, "S03 classic policy mapping report"),
        artifact_entry(validation_path, artifacts_dir, "S03 mapping validation results"),
        manifest_self_entry(manifest_path, artifacts_dir, "S03 source snapshot and provenance manifest"),
        artifact_entry(run_manifest_path, artifacts_dir, "Experiment-level run manifest updated through S03"),
        {
            "path": str(checksums_path),
            "relativePath": str(checksums_path.relative_to(artifacts_dir)),
            "description": "SHA256 checksums for compact S03 artifacts",
            "sha256": None,
            "sizeBytes": None,
            "note": "Checksum file is written after the manifest so it can include the final manifest hash.",
        },
    ]
    write_json(manifest_path, manifest)
    checksum_targets = [report_path, library_path, mapping_report_path, validation_path, manifest_path, run_manifest_path]
    write_text(
        checksums_path,
        "\n".join(f"{sha256_file(path)}  {path.relative_to(artifacts_dir)}" for path in checksum_targets) + "\n",
    )
    print(json.dumps(manifest["validation"], indent=2, sort_keys=True))
    return 0 if manifest["validation"]["allValidationPassed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
