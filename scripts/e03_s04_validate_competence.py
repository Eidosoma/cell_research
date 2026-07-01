#!/usr/bin/env python3
"""Build and validate E03 S04 competence vectors for classic policies."""

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

from src.e03.competence_metrics import (
    SCHEMA_VERSION,
    CompetenceSourcePaths,
    build_policy_competence_vectors,
    validate_competence_vectors,
    vector_column_spec,
)


STEP_ID = "S04"
STEP_NUMBER = 4
EXPERIMENT_ID = "E03"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--policy-library", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")) / "policies/e03_classic_policy_library.json")
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
        "note": "Checksum omitted to avoid self-referential drift.",
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
        values = []
        for column in columns:
            value = record[column]
            if isinstance(value, float):
                text = "nan" if pd.isna(value) else f"{value:.6g}"
            else:
                text = str(value)
            values.append(text.replace("|", "\\|"))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def render_metric_spec(vectors: pd.DataFrame, validation: pd.DataFrame) -> str:
    spec_df = pd.DataFrame(vector_column_spec())
    canonical = vectors[vectors["canonical_classic_baseline"] == True].copy()  # noqa: E712
    validation_summary = f"{int(validation['success'].sum())}/{len(validation)} validation cases passed"
    outcome = "supportive" if bool(validation["success"].all()) else "constraining/contradictory"
    preview_cols = [
        "policy_id",
        "algorithm",
        "efficiency_score",
        "final_sortedness_score",
        "frozen_cell_robustness_score",
        "dg_tendency_score",
        "aggregation_tendency_score",
        "conflict_dominance_score",
        "movement_energy_score",
        "path_directness_score",
        "transfer_across_array_sizes_score",
    ]
    return f"""# E03 Competence Vector Specification

## Top Summary

- Step ID: S04
- Completion status: Completed.
- Artifacts written: `$ARTIFACTS_DIR/reports/e03_competence_vector_spec.md`, `$ARTIFACTS_DIR/results/e03_classic_competence_vectors.parquet`, `$ARTIFACTS_DIR/results/e03_competence_vector_validation.parquet`, and `$ARTIFACTS_DIR/research_steps/S04/research_step_full_results.md`.
- Validation result: {validation_summary}.
- Outcome classification: {outcome}.
- Caveats or blockers: Transfer across array sizes is schema-defined but unavailable in the mounted E01/E02 context because scaled replication contains only array length 100. Conflict dominance is a sparse chimeric proxy, not a causal dominance mechanism.
- Recommended next action: Chief review, then S05 can generate policies against this frozen schema.

## Lay Summary

The competence vector is a compact scorecard for sorting-cell policies. It keeps several behaviors separate instead of collapsing them into one grade: sorting success, work cost, robustness to Frozen Cells, delayed-gratification tendency, aggregation tendency, opposite-goal dominance, movement energy, trajectory directness, and size-transfer evidence.

## Schema

Schema version: `{SCHEMA_VERSION}`

{markdown_table(spec_df, ["column", "type", "meaning"])}

## Metric Definitions

- `final_sortedness_score`: E01 mean final Sortedness percent divided by 100 for unperturbed cell-view classic runs.
- `efficiency_score`: min-max score over E01 cell-view `compare_plus_swap_steps`, with lower raw work receiving a higher score.
- `frozen_cell_robustness_score`: mean E01 final Sortedness percent across passive and stuck Frozen Cell conditions divided by 100.
- `dg_tendency_score`: min-max score over E01 unperturbed cell-view `mean_dg_primary`, with higher DG tendency receiving a higher score.
- `aggregation_tendency_score`: mean same-goal chimera aggregation peak delta above random divided by 100 for mixes containing the algorithm.
- `conflict_dominance_score`: mean fraction of E01 opposite-goal conflict runs in which the algorithm label was dominant, based on dominant-label counts in rows containing that algorithm.
- `movement_energy_score`: min-max score over E01 cell-view swap-only steps, with lower movement cost receiving a higher score.
- `path_directness_score`: `1 / path_curvature_ratio` from the E02 sortedness-adjacency alternative metric. This is a path-shape proxy only.
- `transfer_across_array_sizes_score`: intended to be the minimum final Sortedness score across evaluated array sizes, but left null until at least two array sizes are present.

## Canonical Classic Baselines

Only exact S01/S03 interface-wrapper policies are canonical baselines. DSL exact-subset and shadow entries inherit algorithm-level metric context as landmarks for S05 generation, and their `metric_projection_scope` records that limitation.

{markdown_table(canonical[preview_cols], preview_cols)}

## Provenance

- S03 policy library: `$ARTIFACTS_DIR/policies/e03_classic_policy_library.json`
- E01 metric inputs: `/previous-artifacts/E01/tables` and `/previous-artifacts/E01/results`
- E02 metric context: `/previous-artifacts/E02/research_steps/S09/e02_alternative_metric_summary.csv`
"""


def render_full_report(
    *,
    vectors: pd.DataFrame,
    validation: pd.DataFrame,
    unit_tests: dict[str, Any],
    manifest: dict[str, Any],
    artifacts: dict[str, Path],
    command_line: str,
) -> str:
    validation_success = bool(validation["success"].all() and unit_tests["success"])
    outcome = "supportive" if validation_success else "constraining/contradictory"
    validation_line = (
        f"{int(validation['success'].sum())}/{len(validation)} validation cases passed; "
        f"unit tests return code {unit_tests['returnCode']}"
    )
    artifact_list = "\n".join(f"- `{path}`" for path in artifacts.values())
    canonical = vectors[vectors["canonical_classic_baseline"] == True].copy()  # noqa: E712
    preview_cols = [
        "policy_id",
        "algorithm",
        "compare_plus_swap_steps",
        "efficiency_score",
        "final_sortedness_score",
        "frozen_mean_monotonicity_error",
        "dg_primary_mean",
        "conflict_dominance_score",
        "transfer_available",
    ]
    validation_table = markdown_table(validation, ["validation_case", "success", "expected", "observed"])
    canonical_table = markdown_table(canonical[preview_cols], preview_cols)
    source_entries = manifest.get("sourceFiles", [])
    source_table = markdown_table(pd.DataFrame(source_entries), ["relativePath", "sha256", "sizeBytes"])
    command_rows = pd.DataFrame(
        [
            {"command": unit_tests["command"], "returnCode": unit_tests["returnCode"], "success": unit_tests["success"]},
            {"command": command_line, "returnCode": 0, "success": True},
        ]
    )
    top_summary = f"""# E03 S04 Research Step Full Results

## Top Summary

- Step ID: S04
- Completion status: Completed.
- Artifacts written:
{artifact_list}
- Validation result: {validation_line}.
- Outcome classification: {outcome}.
- Caveats or blockers: Transfer across array sizes is unavailable in current E01/E02 inputs; conflict dominance and aggregation are sparse chimeric proxies; DSL rows inherit algorithm-level context and are not full-policy evaluations.
- Lay summary: S04 turns the classic sorting-cell behaviors into a reusable scorecard. The exact Bubble, Insertion, and Selection wrappers all sort successfully, but differ in work cost, Frozen Cell robustness, DG tendency, aggregation context, conflict dominance, movement energy, and trajectory directness.
- Recommended next action: Stop for Chief review; if accepted, run S05 policy generation against the frozen schema.
"""
    return f"""{top_summary}

## Frozen Question

Can policy behavior be summarized by a multi-dimensional competence vector that preserves the trade-offs relevant to morphogenesis-like problem solving?

## Inputs

- S03 policy library: `{manifest['inputArtifacts']['s03PolicyLibrary']}`
- E01 upstream artifact mount: `{manifest['inputArtifacts']['previousE01Dir']}`
- E02 upstream artifact mount: `{manifest['inputArtifacts']['previousE02Dir']}`
- E01 tables used: efficiency numeric table, Frozen Cell robustness numeric table, DG numeric table, aggregation peak table, conflict equilibria summary.
- E01 results used: efficiency counts and scaled replication.
- E02 context used: S09 alternative metric summary for sortedness-adjacency path curvature.

## Methods

S04 did not run new simulations. It deterministically reduced existing E01/E02 metric tables into one algorithm-level context row for Bubble, Insertion, and Selection, then joined that context onto all nine S03 classic policy-library entries.

The three exact public-method interface-wrapper policies are marked `canonical_classic_baseline=true`. DSL exact-subset and approximate-shadow policies are retained in the table so downstream generation can seed from them, but their `metric_projection_scope` and caveats state that they inherit algorithm-level context rather than direct full-policy measurements.

## Commands

{markdown_table(command_rows, ["command", "returnCode", "success"])}

## Dependencies And Runtime

- Python: `{platform.python_version()}`
- pandas: `{pd.__version__}`
- Host: `{platform.platform()}`
- Worker count: serial reduction; no parallel workers used.
- New dependencies installed: none.

## Results

The vector table contains {len(vectors)} rows: {int(vectors['canonical_classic_baseline'].sum())} canonical exact baselines and {len(vectors) - int(vectors['canonical_classic_baseline'].sum())} DSL landmark rows.

### Canonical Baseline Preview

{canonical_table}

### Validation

{validation_table}

## Validation Interpretation

The validation cases check that all S03 policy IDs are represented, only exact interface wrappers are canonical baselines, the known E01 final Sortedness result is preserved, and E01 metric ordering is reproduced for compare-plus efficiency, DG tendency, Frozen Cell error, and movement energy. They also check that sparse S04 dimensions are explicitly marked: E02 path-curvature context is present, while size transfer is unavailable because the mounted scaled replication has only array length 100.

## Caveats And Limitations

- `transfer_across_array_sizes_score` is part of the frozen schema but is null for S04 classic rows because available E01/E02 scaled inputs do not span multiple array sizes.
- `conflict_dominance_score` is a sparse proxy from E01 opposite-goal chimeras and should not be interpreted as a causal governance mechanism.
- `aggregation_tendency_score` comes from same-goal chimeras, so it is an algorithm participation context rather than an intrinsic pure-policy metric.
- `path_directness_score` comes from E02 alternative metrics over a ten-run context; it is useful for trajectory-shape comparison but does not override E01 final sorting success.
- DSL rows are useful morphospace landmarks, not full direct evaluations of those exact DSL programs.

## Provenance

Git commit before S04 commit: `{manifest['git']['headCommit']}`

Source files hashed in the S04 manifest:

{source_table}

## Artifacts

The reusable outputs are the Parquet vector table, the Markdown metric specification, the validation Parquet/CSV, the S04 artifact manifest, the source snapshot manifest, the run manifest, and this full-results report. SHA-256 checksums for key S04 outputs are recorded in `{artifacts['checksums']}`.
"""


def main() -> int:
    args = parse_args()
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    reports_dir = artifacts_dir / "reports"
    src_snapshot_dir = artifacts_dir / "src_snapshot"
    checksums_dir = artifacts_dir / "checksums"
    for directory in (step_dir, results_dir, reports_dir, src_snapshot_dir, checksums_dir):
        directory.mkdir(parents=True, exist_ok=True)

    unit_tests = {"command": "not run", "returnCode": 0, "success": True, "stdout": "", "stderr": "", "elapsedSeconds": 0.0}
    if args.run_unit_tests:
        unit_tests = run_command([sys.executable, "-m", "unittest", "tests.e03.test_competence_metrics"], args.repo_dir)

    paths = CompetenceSourcePaths(args.policy_library, args.previous_e01_dir, args.previous_e02_dir)
    vectors = build_policy_competence_vectors(paths)
    validation = validate_competence_vectors(vectors)

    vector_path = results_dir / "e03_classic_competence_vectors.parquet"
    vector_csv_path = results_dir / "e03_classic_competence_vectors.csv"
    validation_path = results_dir / "e03_competence_vector_validation.parquet"
    validation_csv_path = results_dir / "e03_competence_vector_validation.csv"
    spec_path = reports_dir / "e03_competence_vector_spec.md"
    manifest_path = src_snapshot_dir / "e03_competence_vector_manifest.json"
    artifact_manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = artifacts_dir / "run_manifest.json"
    checksums_path = checksums_dir / "sha256sums.txt"
    full_report_path = step_dir / "research_step_full_results.md"

    vectors.to_parquet(vector_path, index=False)
    vectors.to_csv(vector_csv_path, index=False)
    validation.to_parquet(validation_path, index=False)
    validation.to_csv(validation_csv_path, index=False)
    write_text(spec_path, render_metric_spec(vectors, validation))

    source_files = [
        args.repo_dir / "src/e03/competence_metrics.py",
        args.repo_dir / "tests/e03/test_competence_metrics.py",
        args.repo_dir / "scripts/e03_s04_validate_competence.py",
    ]
    manifest = {
        "schema": "eidosoma.e03.competence_vector_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAt": utc_now(),
        "vectorSchemaVersion": SCHEMA_VERSION,
        "git": {
            "branch": git_output(args.repo_dir, ["branch", "--show-current"]),
            "headCommit": git_output(args.repo_dir, ["rev-parse", "HEAD"]),
            "statusShort": git_output(args.repo_dir, ["status", "--short"]),
        },
        "inputArtifacts": {
            "s03PolicyLibrary": str(args.policy_library),
            "previousE01Dir": str(args.previous_e01_dir),
            "previousE02Dir": str(args.previous_e02_dir),
        },
        "sourceFiles": [source_entry(path, args.repo_dir) for path in source_files],
        "validationSummary": {
            "success": bool(validation["success"].all() and unit_tests["success"]),
            "validationCasesPassed": int(validation["success"].sum()),
            "validationCasesTotal": int(len(validation)),
            "unitTestsReturnCode": int(unit_tests["returnCode"]),
        },
    }

    preliminary_artifacts = {
        "researchStepReport": full_report_path,
        "metricSpec": spec_path,
        "classicCompetenceVectors": vector_path,
        "classicCompetenceVectorsCsv": vector_csv_path,
        "validationParquet": validation_path,
        "validationCsv": validation_csv_path,
        "sourceSnapshotManifest": manifest_path,
        "artifactManifest": artifact_manifest_path,
        "runManifest": run_manifest_path,
        "checksums": checksums_path,
    }
    manifest["artifacts"] = [
        artifact_entry(spec_path, artifacts_dir, "S04 competence vector Markdown specification"),
        artifact_entry(vector_path, artifacts_dir, "S04 classic competence vector table"),
        artifact_entry(vector_csv_path, artifacts_dir, "CSV sidecar for S04 classic competence vector table"),
        artifact_entry(validation_path, artifacts_dir, "S04 validation cases"),
        artifact_entry(validation_csv_path, artifacts_dir, "CSV sidecar for S04 validation cases"),
        manifest_self_entry(manifest_path, artifacts_dir, "S04 source snapshot and provenance manifest"),
        manifest_self_entry(artifact_manifest_path, artifacts_dir, "S04 artifact manifest"),
        manifest_self_entry(run_manifest_path, artifacts_dir, "Experiment run manifest updated by S04"),
        manifest_self_entry(checksums_path, artifacts_dir, "SHA-256 checksums for key S04 outputs"),
        manifest_self_entry(full_report_path, artifacts_dir, "S04 full-results handoff report"),
    ]
    write_json(manifest_path, manifest)

    artifact_manifest = {
        "schema": "eidosoma.research_step_artifact_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAt": utc_now(),
        "artifacts": [
            artifact_entry(spec_path, artifacts_dir, "S04 competence vector Markdown specification"),
            artifact_entry(vector_path, artifacts_dir, "S04 classic competence vector table"),
            artifact_entry(vector_csv_path, artifacts_dir, "CSV sidecar for S04 classic competence vector table"),
            artifact_entry(validation_path, artifacts_dir, "S04 validation cases"),
            artifact_entry(validation_csv_path, artifacts_dir, "CSV sidecar for S04 validation cases"),
            artifact_entry(manifest_path, artifacts_dir, "S04 source snapshot and provenance manifest"),
            manifest_self_entry(artifact_manifest_path, artifacts_dir, "S04 artifact manifest"),
            manifest_self_entry(run_manifest_path, artifacts_dir, "Experiment run manifest updated by S04"),
            manifest_self_entry(checksums_path, artifacts_dir, "SHA-256 checksums for key S04 outputs"),
            manifest_self_entry(full_report_path, artifacts_dir, "S04 full-results handoff report"),
        ],
    }
    write_json(artifact_manifest_path, artifact_manifest)

    run_manifest = {
        "schema": "eidosoma.run_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAt": utc_now(),
        "git": manifest["git"],
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pandas": pd.__version__,
        },
        "commands": [
            {"command": unit_tests["command"], "returnCode": unit_tests["returnCode"], "success": unit_tests["success"]},
            {"command": " ".join(sys.argv), "returnCode": 0, "success": True},
        ],
        "inputs": manifest["inputArtifacts"],
        "outputs": artifact_manifest["artifacts"],
    }
    write_json(run_manifest_path, run_manifest)

    full_report = render_full_report(
        vectors=vectors,
        validation=validation,
        unit_tests=unit_tests,
        manifest=manifest,
        artifacts=preliminary_artifacts,
        command_line=" ".join(sys.argv),
    )
    write_text(full_report_path, full_report)

    checksum_targets = [
        full_report_path,
        spec_path,
        vector_path,
        vector_csv_path,
        validation_path,
        validation_csv_path,
        manifest_path,
        artifact_manifest_path,
        run_manifest_path,
    ]
    checksum_lines = [f"{sha256_file(path)}  {path.relative_to(artifacts_dir)}" for path in checksum_targets]
    write_text(checksums_path, "\n".join(checksum_lines) + "\n")

    success = bool(validation["success"].all() and unit_tests["success"])
    print(json.dumps({"researchStepId": STEP_ID, "success": success, "artifactsDir": str(step_dir)}, indent=2, sort_keys=True))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
