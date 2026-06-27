#!/usr/bin/env python3
"""Execute E03 S04 competence-vector schema validation and artifact packaging.

S04 defines the run-level competence-vector schema, implements metric utilities
and toy examples, validates available classic-policy values against S01/E01/E02
contexts where possible, writes artifacts, and stops before S05.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True

from morphospace import (  # noqa: E402
    COMPETENCE_VECTOR_VERSION,
    PolicyEventSimulator,
    competence_metric_specs,
    competence_schema,
    compute_competence_vector,
    run_metric_unit_cases,
    summarize_competence_vectors,
    toy_competence_examples,
    vector_from_summary_record,
)


EXPERIMENT_ID = "E03"
STEP_ID = "S04"
STEP_NUMBER = 4
STATUS = "completed"
OUTCOME_CLASSIFICATION = "supportive"
DEFAULT_ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
DEFAULT_E02_CONTEXT = Path("/previous-artifacts/E02/results/e02_input_distribution_original_context.csv")
DEFAULT_E02_DG = Path("/previous-artifacts/E02/results/e02_dg_observed.csv")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_command(args: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> dict[str, Any]:
    merged_env = os.environ.copy()
    merged_env["PYTHONDONTWRITEBYTECODE"] = "1"
    if env:
        merged_env.update(env)
    try:
        proc = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            env=merged_env,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return {
            "args": args,
            "returncode": proc.returncode,
            "ok": proc.returncode == 0,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except Exception as exc:  # pragma: no cover - defensive provenance path
        return {
            "args": args,
            "returncode": None,
            "ok": False,
            "stdout": "",
            "stderr": repr(exc),
        }


def get_git_metadata() -> dict[str, Any]:
    commit = run_command(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
    branch = run_command(["git", "branch", "--show-current"], cwd=REPO_ROOT)
    status = run_command(["git", "status", "--short"], cwd=REPO_ROOT)
    remote = run_command(["git", "remote", "-v"], cwd=REPO_ROOT)
    return {
        "commit": commit["stdout"].strip() if commit["ok"] else "unknown",
        "branch": branch["stdout"].strip() if branch["ok"] else "unknown",
        "dirtyStatus": status["stdout"].strip(),
        "remote": remote["stdout"].strip(),
    }


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return json_ready(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def clean(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            if not math.isfinite(value):
                return ""
            return f"{value:.4f}".rstrip("0").rstrip(".")
        return str(value).replace("\n", " ").replace("|", "\\|")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(clean(item) for item in row) + " |")
    return "\n".join(lines)


def artifact_preview(artifacts_written: list[str], limit: int = 20) -> str:
    preview = "\n".join(f"- `{path}`" for path in artifacts_written[:limit])
    if len(artifacts_written) > limit:
        preview += f"\n- ... {len(artifacts_written) - limit} additional artifact path(s) in status.json"
    return preview


def metric_specs_df() -> pd.DataFrame:
    rows = []
    for spec in competence_metric_specs():
        row = spec.to_dict()
        row["uncertaintyColumnsJson"] = json.dumps(row.pop("uncertaintyColumns"), separators=(",", ":"))
        rows.append(row)
    return pd.DataFrame(rows)


def classic_competence_examples() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for algorithm in ["bubble", "insertion", "selection"]:
        result = PolicyEventSimulator(
            [6, 2, 4, 1, 5, 3],
            algorithm,
            scheduler_seed=123,
            tie_breaker_seed=456,
            condition_id=f"S04_classic_{algorithm}_six_cell",
            implementation="s04_competence_examples",
            research_step_id=STEP_ID,
        ).run(max_activations=200000)
        rows.append(
            compute_competence_vector(
                result,
                policy_id=f"classic_{algorithm}",
                policy_family="classic",
                task_id=f"S04_classic_{algorithm}_six_cell",
                task_family="sorting",
                task_panel="classic_toy_validation",
                input_profile="manual_unique",
                replicate_index=0,
            )
        )
    return pd.DataFrame(rows)


def run_repo_unit_tests(step_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    cmd = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"]
    result = run_command(cmd, cwd=REPO_ROOT)
    log_path = step_dir / "repo_unit_test_log.txt"
    log_path.write_text(
        "$ " + " ".join(cmd) + "\n\nSTDOUT\n" + result["stdout"] + "\n\nSTDERR\n" + result["stderr"],
        encoding="utf-8",
    )
    row = {
        "validationFamily": "repo_unit_tests",
        "conditionId": "repo_unit_tests",
        "success": result["ok"],
        "validationDetail": f"Repository unittest discovery return code {result['returncode']}.",
        "sourcePath": str(log_path),
    }
    payload = {
        "command": cmd,
        "returnCode": result["returncode"],
        "success": result["ok"],
        "logPath": str(log_path),
    }
    return row, payload


def validate_schema_and_metrics(
    *,
    s01_summary_path: Path,
    s01_status_path: Path,
    s02_status_path: Path,
    s03_status_path: Path,
    e02_context_path: Path,
    e02_dg_path: Path,
    unit_cases: pd.DataFrame,
    toy_df: pd.DataFrame,
    classic_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    schema = competence_schema()
    required_metric_ids = {
        "completion_success",
        "final_sortedness_percent",
        "final_monotonicity_error",
        "swap_count",
        "comparison_count",
        "delayed_gratification",
        "aggregation_peak",
        "dominance_score",
        "transfer_score",
        "failure_or_oscillation",
    }
    observed_ids = {item["metricId"] for item in schema["metricSpecs"]}
    rows.append(
        {
            "validationFamily": "schema",
            "conditionId": "required_metric_ids_present",
            "success": required_metric_ids.issubset(observed_ids),
            "validationDetail": f"Observed {len(observed_ids)} metric specs in {COMPETENCE_VECTOR_VERSION}.",
        }
    )
    rows.append(
        {
            "validationFamily": "schema",
            "conditionId": "missing_and_uncertainty_contract_present",
            "success": bool(schema.get("missingValuePolicy"))
            and bool(schema.get("uncertaintyFieldContract", {}).get("columns")),
            "validationDetail": "Schema includes missing-value and uncertainty-field contracts.",
        }
    )

    rows.append(
        {
            "validationFamily": "unit_cases",
            "conditionId": "metric_unit_cases",
            "success": bool(unit_cases["passed"].all()),
            "validationDetail": f"{int(unit_cases['passed'].sum())} of {len(unit_cases)} metric unit cases passed.",
        }
    )
    rows.append(
        {
            "validationFamily": "toy_examples",
            "conditionId": "toy_vectors_unique_and_complete",
            "success": len(toy_df) == 3 and toy_df["vectorId"].is_unique and toy_df["finalSortednessScore"].notna().all(),
            "validationDetail": "Toy examples produced three unique competence vectors with required final scores.",
        }
    )
    rows.append(
        {
            "validationFamily": "classic_examples",
            "conditionId": "classic_toy_vectors_sort",
            "success": len(classic_df) == 3
            and classic_df["completed"].all()
            and (classic_df["finalSortednessPercent"] == 100.0).all(),
            "validationDetail": "Classic Bubble, Insertion, and Selection toy vectors sort the six-cell array.",
        }
    )

    if s01_summary_path.exists():
        s01 = pd.read_csv(s01_summary_path)
        checked = 0
        mismatches = 0
        for _, source_row in s01.head(12).iterrows():
            vector = vector_from_summary_record(source_row.to_dict(), source_artifact_path=str(s01_summary_path))
            checked += 1
            if not (
                math.isclose(float(vector["finalSortednessPercent"]), float(source_row["final_sortedness_percent"]))
                and int(vector["finalMonotonicityError"]) == int(source_row["final_monotonicity_error"])
                and int(vector["swapCount"]) == int(source_row["swap_count"])
                and int(vector["comparisonCount"]) == int(source_row["comparison_count"])
            ):
                mismatches += 1
        rows.append(
            {
                "validationFamily": "s01_replay_ingest",
                "conditionId": "s01_policy_interface_replay_values",
                "success": checked > 0 and mismatches == 0,
                "validationDetail": f"Checked {checked} S01 replay rows; mismatches={mismatches}.",
                "sourcePath": str(s01_summary_path),
            }
        )
    else:
        rows.append(
            {
                "validationFamily": "s01_replay_ingest",
                "conditionId": "s01_policy_interface_replay_values",
                "success": False,
                "validationDetail": "S01 policy-interface replay summary was unavailable.",
                "sourcePath": str(s01_summary_path),
            }
        )

    if e02_context_path.exists():
        e02 = pd.read_csv(e02_context_path)
        e02 = e02[e02["algorithm"].isin(["bubble", "insertion", "selection"])].head(24)
        checked = 0
        mismatches = 0
        for _, source_row in e02.iterrows():
            vector = vector_from_summary_record(source_row.to_dict(), source_artifact_path=str(e02_context_path))
            checked += 1
            if not (
                math.isclose(float(vector["finalSortednessPercent"]), float(source_row["finalSortednessPercent"]))
                and int(vector["finalMonotonicityError"]) == int(source_row["finalMonotonicityError"])
                and int(vector["swapCount"]) == int(source_row["swapCount"])
                and int(vector["comparisonCount"]) == int(source_row["comparisonCount"])
            ):
                mismatches += 1
        rows.append(
            {
                "validationFamily": "e01_e02_context_ingest",
                "conditionId": "e02_original_context_values",
                "success": checked > 0 and mismatches == 0,
                "validationDetail": f"Checked {checked} E02 original-context classic rows; mismatches={mismatches}.",
                "sourcePath": str(e02_context_path),
            }
        )
    else:
        rows.append(
            {
                "validationFamily": "e01_e02_context_ingest",
                "conditionId": "e02_original_context_values",
                "success": False,
                "validationDetail": "E02 original-context values were unavailable.",
                "sourcePath": str(e02_context_path),
            }
        )

    if e02_dg_path.exists():
        dg = pd.read_csv(e02_dg_path).head(24)
        checked = 0
        mismatches = 0
        for _, source_row in dg.iterrows():
            vector = vector_from_summary_record(source_row.to_dict(), source_artifact_path=str(e02_dg_path))
            checked += 1
            if not (
                vector["delayedGratification"] is not None
                and math.isclose(float(vector["delayedGratification"]), float(source_row["delayedGratification"]))
                and int(vector["dgEventCount"]) == int(source_row["dgEventCount"])
            ):
                mismatches += 1
        rows.append(
            {
                "validationFamily": "e02_dg_ingest",
                "conditionId": "e02_observed_dg_values",
                "success": checked > 0 and mismatches == 0,
                "validationDetail": f"Checked {checked} E02 DG observed rows; mismatches={mismatches}.",
                "sourcePath": str(e02_dg_path),
            }
        )
    else:
        rows.append(
            {
                "validationFamily": "e02_dg_ingest",
                "conditionId": "e02_observed_dg_values",
                "success": False,
                "validationDetail": "E02 observed DG values were unavailable.",
                "sourcePath": str(e02_dg_path),
            }
        )

    for step_id, path in [("S01", s01_status_path), ("S02", s02_status_path), ("S03", s03_status_path)]:
        status = read_json(path) if path.exists() else {}
        rows.append(
            {
                "validationFamily": "upstream_boundary",
                "conditionId": f"{step_id.lower()}_completed_anchor",
                "success": bool(status.get("success")),
                "validationDetail": f"{step_id} status confirms required upstream boundary completed successfully.",
                "sourcePath": str(path),
            }
        )
    return pd.DataFrame(rows)


def validation_counts(validation_df: pd.DataFrame) -> pd.DataFrame:
    return (
        validation_df.groupby("validationFamily", dropna=False)["success"]
        .agg(total="count", passed="sum")
        .reset_index()
        .assign(failed=lambda df: df["total"] - df["passed"])
    )


def render_schema_md(
    metric_df: pd.DataFrame,
    artifacts_written: list[str],
    validation_result: str,
    caveats: list[str],
    recommended_next_action: str,
) -> str:
    rows = metric_df[["metricId", "column", "family", "direction", "normalization", "missingPolicy"]].values.tolist()
    return f"""# E03 S04 Competence Vector Schema

- Research step ID: {STEP_ID}
- Completion status: {STATUS}
- Artifacts written:
{artifact_preview(artifacts_written)}
- Validation result: {validation_result}
- Caveats or blockers: {"; ".join(caveats)}
- Recommended next action: {recommended_next_action}

## Frozen Question

A multidimensional competence vector can summarize policy behavior across sorting, perturbation, chimera, and transfer tasks.

## Schema Contract

- Version: `{COMPETENCE_VECTOR_VERSION}`
- Missing-value handling: nullable fields use `missingMetricReasonsJson` to distinguish not-applicable metrics from unavailable source data.
- Normalization: raw metric columns are preserved, while normalized score columns are bounded higher-is-better proxies intended for morphospace screening.
- Uncertainty: replicate summaries report N, sample SD, SEM, and normal-approximation 95% intervals over nonmissing values.

## Metric Specs

{markdown_table(["Metric ID", "Column", "Family", "Direction", "Normalization", "Missing policy"], rows)}

## Scope Notes

Sorting and efficiency metrics are required for run-level vectors. DG, aggregation, curvature, dominance, transfer, and repair fields are context-sensitive: they are present when the task panel supplies the needed trace or comparison data and otherwise remain explicitly missing.
"""


def render_validation_report(validation_df: pd.DataFrame, artifacts_written: list[str], caveats: list[str]) -> str:
    validation_passed = bool(validation_df["success"].all())
    validation_result = "passed" if validation_passed else "failed"
    failed = validation_df[~validation_df["success"].astype(bool)]
    failure_text = "None"
    if not failed.empty:
        failure_text = markdown_table(
            ["Family", "Condition", "Detail"],
            failed[["validationFamily", "conditionId", "validationDetail"]].head(20).values.tolist(),
        )
    rows = validation_counts(validation_df)[["validationFamily", "passed", "total", "failed"]].values.tolist()
    return f"""# S04 Validation Report

- Research step ID: {STEP_ID}
- Completion status: {STATUS}
- Artifacts written:
{artifact_preview(artifacts_written)}
- Validation result: {validation_result}; {int(validation_df["success"].sum())} of {len(validation_df)} checks passed.
- Caveats or blockers: {"; ".join(caveats)}
- Recommended next action: stop before S05 for Chief Scientist review; if approved, generate the policy corpus in S05 using this schema.

## Validation Counts

{markdown_table(["Validation family", "Passed", "Total", "Failed"], rows)}

## Failed Checks

{failure_text}
"""


def render_summary(validation_df: pd.DataFrame, artifacts_written: list[str], caveats: list[str]) -> str:
    validation_passed = bool(validation_df["success"].all())
    validation_result = "passed" if validation_passed else "failed"
    outcome = OUTCOME_CLASSIFICATION if validation_passed else "constraining/contradictory"
    return f"""# S04 Summary

- Research step ID: {STEP_ID}
- Completion status: {STATUS}
- Artifacts written: `{artifacts_written[0]}` and {len(artifacts_written) - 1} additional files listed in `status.json` and `artifact_manifest.json`.
- Validation result: {validation_result}; {int(validation_df["success"].sum())} of {len(validation_df)} checks passed.
- Outcome classification: {outcome}
- Caveats or blockers: {"; ".join(caveats)}
- Lay summary: S04 defines a reusable competence-vector schema for later morphospace sweeps. It preserves raw classic metrics, adds bounded higher-is-better normalized scores, records missing reasons for context-specific fields, and includes uncertainty fields for replicate summaries. Toy and classic examples validate the schema against S01/S02/S03 boundaries plus available E01/E02 metric tables.
- Recommended next action: stop before S05 for Chief Scientist review. If approved, S05 should generate the policy corpus and require every candidate to emit this competence-vector schema during sample execution.
"""


def collect_artifacts(paths: list[Path]) -> list[dict[str, Any]]:
    artifacts = []
    seen: set[Path] = set()
    for path in paths:
        if path.exists() and path.is_file() and path.resolve() not in seen:
            seen.add(path.resolve())
            artifacts.append({"path": str(path), "sizeBytes": path.stat().st_size, "sha256": sha256_path(path)})
    return sorted(artifacts, key=lambda item: item["path"])


def copy_code_artifacts(step_dir: Path) -> list[Path]:
    code_dir = step_dir / "code"
    copied: list[Path] = []
    package_dst = code_dir / "morphospace"
    if package_dst.exists():
        shutil.rmtree(package_dst)
    shutil.copytree(REPO_ROOT / "morphospace", package_dst, ignore=shutil.ignore_patterns("__pycache__"))
    copied.extend(sorted(path for path in package_dst.rglob("*.py")))

    script_dst = code_dir / "scripts" / "e03_s04_competence_vector.py"
    script_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO_ROOT / "scripts" / "e03_s04_competence_vector.py", script_dst)
    copied.append(script_dst)

    test_dst = code_dir / "tests" / "test_e03_competence.py"
    test_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO_ROOT / "tests" / "test_e03_competence.py", test_dst)
    copied.append(test_dst)
    return copied


def write_run_manifest(provenance_dir: Path, artifacts_written: list[str], validation_passed: bool) -> Path:
    provenance_dir.mkdir(parents=True, exist_ok=True)
    path = provenance_dir / "run_manifest.json"
    manifest = read_json(path) if path.exists() else {"schema": "eidosoma.run_manifest.v1", "experimentId": EXPERIMENT_ID}
    manifest.update(
        {
            "experimentId": EXPERIMENT_ID,
            "lastResearchStepId": STEP_ID,
            "lastStepNumber": STEP_NUMBER,
            "updatedAt": utc_now(),
            "git": get_git_metadata(),
            "platform": platform.platform(),
            "python": sys.version,
        }
    )
    manifest.setdefault("researchSteps", {})
    manifest["researchSteps"][STEP_ID] = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "status": STATUS,
        "success": validation_passed,
        "artifactsWritten": artifacts_written,
        "validationResult": "passed" if validation_passed else "failed",
        "completedAt": utc_now(),
        "outcomeClassification": OUTCOME_CLASSIFICATION if validation_passed else "constraining/contradictory",
    }
    write_json(path, manifest)
    return path


def write_table(df: pd.DataFrame, csv_path: Path, parquet_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--e02-context", type=Path, default=DEFAULT_E02_CONTEXT)
    parser.add_argument("--e02-dg", type=Path, default=DEFAULT_E02_DG)
    args = parser.parse_args()

    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    provenance_dir = artifacts_dir / "provenance"
    s01_summary_path = artifacts_dir / "research_steps" / "S01" / "policy_interface_replay_summary.csv"
    s01_status_path = artifacts_dir / "research_steps" / "S01" / "status.json"
    s02_status_path = artifacts_dir / "research_steps" / "S02" / "status.json"
    s03_status_path = artifacts_dir / "research_steps" / "S03" / "status.json"
    step_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    started_at = utc_now()
    unit_success, unit_cases = run_metric_unit_cases()
    metric_df = metric_specs_df()
    toy_df = toy_competence_examples()
    classic_df = classic_competence_examples()
    summary_input = pd.concat(
        [frame.dropna(axis=1, how="all") for frame in [toy_df, classic_df]],
        ignore_index=True,
    )
    uncertainty_df = summarize_competence_vectors(
        summary_input,
        group_columns=("policyId", "taskPanel"),
    )
    validation_df = validate_schema_and_metrics(
        s01_summary_path=s01_summary_path,
        s01_status_path=s01_status_path,
        s02_status_path=s02_status_path,
        s03_status_path=s03_status_path,
        e02_context_path=args.e02_context,
        e02_dg_path=args.e02_dg,
        unit_cases=unit_cases,
        toy_df=toy_df,
        classic_df=classic_df,
    )
    repo_row, repo_command = run_repo_unit_tests(step_dir)
    validation_df = pd.concat([validation_df, pd.DataFrame([repo_row])], ignore_index=True)

    schema_json = step_dir / "competence_vector_schema.json"
    schema_md = step_dir / "competence_vector_schema.md"
    metric_csv = step_dir / "competence_metric_specs.csv"
    metric_parquet = step_dir / "competence_metric_specs.parquet"
    metric_results_csv = results_dir / "e03_s04_competence_metric_specs.csv"
    metric_results_parquet = results_dir / "e03_s04_competence_metric_specs.parquet"
    unit_cases_csv = step_dir / "competence_metric_unit_cases.csv"
    toy_csv = step_dir / "toy_competence_examples.csv"
    toy_parquet = step_dir / "toy_competence_examples.parquet"
    toy_results_csv = results_dir / "e03_s04_toy_competence_examples.csv"
    toy_results_parquet = results_dir / "e03_s04_toy_competence_examples.parquet"
    classic_csv = step_dir / "classic_policy_competence_examples.csv"
    classic_parquet = step_dir / "classic_policy_competence_examples.parquet"
    classic_results_csv = results_dir / "e03_s04_classic_policy_competence_examples.csv"
    classic_results_parquet = results_dir / "e03_s04_classic_policy_competence_examples.parquet"
    uncertainty_csv = step_dir / "competence_uncertainty_examples.csv"
    uncertainty_parquet = step_dir / "competence_uncertainty_examples.parquet"
    uncertainty_results_csv = results_dir / "e03_s04_competence_uncertainty_examples.csv"
    uncertainty_results_parquet = results_dir / "e03_s04_competence_uncertainty_examples.parquet"
    validation_csv = step_dir / "competence_vector_validation.csv"
    validation_parquet = step_dir / "competence_vector_validation.parquet"
    validation_results_csv = results_dir / "e03_s04_competence_vector_validation.csv"
    validation_results_parquet = results_dir / "e03_s04_competence_vector_validation.parquet"
    validation_report_path = step_dir / "validation_report.md"
    summary_path = step_dir / "summary.md"
    run_manifest_path = provenance_dir / "run_manifest.json"
    status_path = step_dir / "status.json"
    artifact_manifest_path = step_dir / "artifact_manifest.json"

    write_json(schema_json, competence_schema())
    write_table(metric_df, metric_csv, metric_parquet)
    write_table(metric_df, metric_results_csv, metric_results_parquet)
    unit_cases.to_csv(unit_cases_csv, index=False)
    write_table(toy_df, toy_csv, toy_parquet)
    write_table(toy_df, toy_results_csv, toy_results_parquet)
    write_table(classic_df, classic_csv, classic_parquet)
    write_table(classic_df, classic_results_csv, classic_results_parquet)
    write_table(uncertainty_df, uncertainty_csv, uncertainty_parquet)
    write_table(uncertainty_df, uncertainty_results_csv, uncertainty_results_parquet)
    write_table(validation_df, validation_csv, validation_parquet)
    write_table(validation_df, validation_results_csv, validation_results_parquet)

    copied_code = copy_code_artifacts(step_dir)
    artifacts_written_paths = [
        schema_json,
        schema_md,
        metric_csv,
        metric_parquet,
        metric_results_csv,
        metric_results_parquet,
        unit_cases_csv,
        toy_csv,
        toy_parquet,
        toy_results_csv,
        toy_results_parquet,
        classic_csv,
        classic_parquet,
        classic_results_csv,
        classic_results_parquet,
        uncertainty_csv,
        uncertainty_parquet,
        uncertainty_results_csv,
        uncertainty_results_parquet,
        validation_csv,
        validation_parquet,
        validation_results_csv,
        validation_results_parquet,
        step_dir / "repo_unit_test_log.txt",
        *copied_code,
        validation_report_path,
        summary_path,
        run_manifest_path,
        status_path,
        artifact_manifest_path,
    ]
    artifacts_written = [str(path) for path in artifacts_written_paths]
    validation_passed = bool(unit_success and validation_df["success"].all())
    validation_result = (
        "passed: schema contract, metric unit cases, toy vectors, classic examples, S01 replay ingest, E01/E02 context ingest, E02 DG ingest, upstream anchors, and repository tests passed"
        if validation_passed
        else "failed: one or more S04 competence-vector validations failed"
    )
    caveats = [
        "Efficiency, energy, oscillation, and path-directness scores are bounded computational proxies, not physical or causal measurements.",
        "Dominance, compatibility, transfer, and repair fields are schema-ready but remain missing until later task panels supply those measurements.",
        "Trajectory curvature requires explicit state trajectories; S01/S04 simulator trace rows do not yet store full value states for every event.",
        "No policy corpus, generated policy sweep, GPU simulator, or S05 artifact was started.",
    ]
    recommended_next_action = "Stop before S05 for Chief Scientist review; if approved, generate policy corpus records that conform to the S04 competence schema."

    schema_md.write_text(
        render_schema_md(metric_df, artifacts_written, validation_result, caveats, recommended_next_action),
        encoding="utf-8",
    )
    validation_report_path.write_text(render_validation_report(validation_df, artifacts_written, caveats), encoding="utf-8")
    summary_path.write_text(render_summary(validation_df, artifacts_written, caveats), encoding="utf-8")
    write_run_manifest(provenance_dir, artifacts_written, validation_passed)

    status = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": validation_passed,
        "status": STATUS,
        "artifactsWritten": artifacts_written,
        "validationResult": validation_result,
        "caveatsOrBlockers": caveats,
        "recommendedNextAction": recommended_next_action,
        "outcomeClassification": OUTCOME_CLASSIFICATION if validation_passed else "constraining/contradictory",
        "startedAt": started_at,
        "completedAt": utc_now(),
        "workerCount": 1,
        "threadEnvironment": {
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        },
        "repoUnitTestCommand": repo_command,
        "sourceArtifacts": {
            "s01SummaryPath": str(s01_summary_path),
            "e02ContextPath": str(args.e02_context),
            "e02DgPath": str(args.e02_dg),
        },
        "git": get_git_metadata(),
    }
    write_json(status_path, status)

    artifact_manifest = {
        "schema": "eidosoma.e03.s04.artifact_manifest.v1",
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "generatedAt": utc_now(),
        "artifacts": collect_artifacts([Path(path) for path in artifacts_written]),
    }
    write_json(artifact_manifest_path, artifact_manifest)
    artifact_manifest["artifacts"] = collect_artifacts([Path(path) for path in artifacts_written])
    write_json(artifact_manifest_path, artifact_manifest)

    print(json.dumps({"success": validation_passed, "statusPath": str(status_path)}, indent=2))
    return 0 if validation_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
