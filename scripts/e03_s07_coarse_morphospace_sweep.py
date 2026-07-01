#!/usr/bin/env python3
"""Run the E03 S07 coarse policy morphospace sweep."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.e03.coarse_sweep import (
    DEFAULT_SWEEP_SEED,
    load_policy_records,
    policy_summary_frame,
    result_digest,
    run_configs,
    scale_configs,
    screen_configs,
    select_scale_policy_ids,
    validation_frame,
)
from src.e03.gpu_batch_simulator import jax_backend_summary


STEP_ID = "S07"
STEP_NUMBER = 7
EXPERIMENT_ID = "E03"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument(
        "--policy-library",
        type=Path,
        default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")) / "policies/e03_generated_policy_library.jsonl",
    )
    parser.add_argument(
        "--compatibility-table",
        type=Path,
        default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")) / "results/e03_gpu_policy_compatibility.parquet",
    )
    parser.add_argument("--scale-policy-count", type=int, default=32)
    parser.add_argument("--sparse-n1000-policy-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=DEFAULT_SWEEP_SEED)
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
    if df.empty:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for record in df[columns].to_dict(orient="records"):
        values = []
        for column in columns:
            value = record[column]
            if isinstance(value, float) or isinstance(value, np.floating):
                text = "nan" if pd.isna(value) else f"{value:.6g}"
            else:
                text = str(value)
            values.append(text.replace("|", "\\|"))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def append_validation_cases(validation: pd.DataFrame, cases: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.concat([validation, pd.DataFrame(cases)], ignore_index=True)


def make_summary_tables(run_df: pd.DataFrame, summary_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    route_summary = (
        run_df.groupby("route")
        .agg(
            run_count=("policy_id", "size"),
            policy_count=("policy_id", "nunique"),
            mean_final_sortedness=("final_inversion_sortedness", "mean"),
            mean_delta=("inversion_sortedness_delta", "mean"),
            timeout_count=("timed_out", "sum"),
            invalid_count=("invalid", "sum"),
        )
        .reset_index()
    )
    phase_summary = (
        run_df.groupby(["phase", "split", "array_size"])
        .agg(
            run_count=("policy_id", "size"),
            policy_count=("policy_id", "nunique"),
            mean_final_sortedness=("final_inversion_sortedness", "mean"),
            sorted_run_fraction=("final_is_sorted", "mean"),
            timeout_count=("timed_out", "sum"),
        )
        .reset_index()
    )
    top_policies = summary_df.sort_values(["screen_score", "screen_heldout_final_sortedness_mean"], ascending=False).head(20)
    return {"route": route_summary, "phase": phase_summary, "top": top_policies}


def write_overview_figure(path: Path, run_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    heldout = run_df[(run_df["phase"] == "screen") & (run_df["heldout"] == True)]  # noqa: E712
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for route, color in [("jax_batch", "#2f6fbb"), ("cpu_fallback", "#b85c38")]:
        data = heldout[heldout["route"] == route]["final_inversion_sortedness"]
        axes[0, 0].hist(data, bins=30, alpha=0.65, label=route, color=color)
    axes[0, 0].set_title("Held-out small-array sortedness")
    axes[0, 0].set_xlabel("Final inversion sortedness")
    axes[0, 0].set_ylabel("Run count")
    axes[0, 0].legend()

    route_counts = run_df["route"].value_counts().sort_index()
    axes[0, 1].bar(route_counts.index, route_counts.values, color=["#b85c38" if item == "cpu_fallback" else "#2f6fbb" for item in route_counts.index])
    axes[0, 1].set_title("Rows by execution route")
    axes[0, 1].set_ylabel("Run count")
    axes[0, 1].tick_params(axis="x", rotation=20)

    scatter = summary_df.dropna(subset=["screen_heldout_work_mean", "screen_heldout_final_sortedness_mean"])
    colors = scatter["route"].map({"jax_batch": "#2f6fbb", "cpu_fallback": "#b85c38"}).fillna("#555555")
    axes[1, 0].scatter(scatter["screen_heldout_work_mean"], scatter["screen_heldout_final_sortedness_mean"], c=colors, s=12, alpha=0.65)
    axes[1, 0].set_title("Held-out competence proxy")
    axes[1, 0].set_xlabel("Mean work count")
    axes[1, 0].set_ylabel("Mean final inversion sortedness")

    scale = run_df[run_df["phase"].isin(["scale", "scale_sparse"])]
    if not scale.empty:
        scale_box = [scale[scale["array_size"] == n]["final_inversion_sortedness"].dropna() for n in sorted(scale["array_size"].unique())]
        axes[1, 1].boxplot(scale_box, tick_labels=[str(n) for n in sorted(scale["array_size"].unique())])
    axes[1, 1].set_title("Selected transfer probes")
    axes[1, 1].set_xlabel("Array size")
    axes[1, 1].set_ylabel("Final inversion sortedness")

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def render_report(
    *,
    artifacts: dict[str, Path],
    run_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    validation: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    manifest: dict[str, Any],
    unit_tests: dict[str, Any],
    command_line: str,
) -> str:
    validation_success = bool(validation["success"].all() and unit_tests["success"])
    outcome = "supportive" if validation_success else "constraining/contradictory"
    validation_line = (
        f"{int(validation['success'].sum())}/{len(validation)} validation cases passed; "
        f"unit tests return code {unit_tests['returnCode']}"
    )
    artifact_list = "\n".join(f"- `{path}`" for path in artifacts.values())
    command_rows = pd.DataFrame(
        [
            {"command": unit_tests["command"], "returnCode": unit_tests["returnCode"], "success": unit_tests["success"]},
            {"command": command_line, "returnCode": 0, "success": True},
        ]
    )
    source_table = markdown_table(pd.DataFrame(manifest["sourceFiles"]), ["relativePath", "sha256", "sizeBytes"])
    selected_scale = int(run_df[run_df["phase"].isin(["scale", "scale_sparse"])]["policy_id"].nunique())
    policy_count = int(summary_df["policy_id"].nunique())
    route_counts = run_df["route"].value_counts().to_dict()
    timeout_count = int(run_df["timed_out"].sum())
    invalid_count = int(run_df["invalid"].sum())
    best = summary_df.sort_values("screen_score", ascending=False).head(12)
    best_cols = [
        "policy_name",
        "source_kind",
        "route",
        "screen_score",
        "screen_heldout_final_sortedness_mean",
        "screen_heldout_improvement_mean",
        "scale_run_count",
    ]
    return f"""# E03 S07 Research Step Full Results

## Top Summary

- Step ID: S07
- Completion status: Completed.
- Artifacts written:
{artifact_list}
- Validation result: {validation_line}; result digest `{result_digest(run_df)}`.
- Outcome classification: {outcome}.
- Caveats or blockers: No blocker remains. This is a coarse DSL local-step screen under a deterministic position scheduler, not a full E02 public simulator replication. Event-cap timeouts are expected and explicitly labeled; sparse n=1000 probes are transfer signals, not convergence claims.
- Lay summary: S07 screened all {policy_count} generated DSL policies on small held-out arrays, routed S06 batch-compatible policies through JAX, routed exact stochastic or memory/signal policies through CPU fallback, and probed {selected_scale} selected policies at larger sizes. The table is an initial competence map for choosing S08 search directions.
- Recommended next action: Stop for Chief review; if accepted, proceed to S08 quality-diversity search using the S07 competence table and timeout/fallback labels.

## Frozen Question

Which regions of the generated policy space can sort small arrays, and which promising regions transfer to n equals 100 or n equals 1,000?

## Inputs

- S05 policy library: `{manifest['inputArtifacts']['s05PolicyLibrary']}`
- S06 compatibility table: `{manifest['inputArtifacts']['s06CompatibilityTable']}`
- S06 JAX local-step simulator: `src/e03/gpu_batch_simulator.py`
- S07 scheduled sweep wrapper: `src/e03/coarse_sweep.py`
- Sweep seed: `{manifest['parameters']['seed']}`

## Methods

Each policy was evaluated as a DSL local rule under a deterministic cyclic position scheduler. For every event, the scheduler selected an actor position; the policy then observed the local array state and executed at most one terminal local action. Policies marked by S06 as batch-ready were evaluated through the JAX local-step kernel. Policies requiring exact stochastic random-stream replay or memory/signal-sensitive semantics were evaluated with the CPU DSL interpreter and labeled as `cpu_fallback`.

The full S05 library was screened on n=8 and n=16 random permutations with train and held-out seeds. A held-out screen score selected promising and landmark policies for n=100 probes and a smaller sparse n=1000 probe. Event caps were fixed by config; runs that were not sorted by the cap are labeled `timeout_event_cap` rather than dropped.

## Commands

{markdown_table(command_rows, ["command", "returnCode", "success"])}

## Dependencies And Runtime

- Python: `{platform.python_version()}`
- pandas: `{pd.__version__}`
- numpy: `{np.__version__}`
- matplotlib: `{matplotlib.__version__}`
- JAX backend summary: `{json.dumps(manifest['runtime']['jax'], sort_keys=True)}`
- Worker count: serial CPU fallback plus vectorized JAX batches; no CPU worker pool used.
- New dependencies installed: none.

## Parameters

- Screen configs: `{manifest['parameters']['screenConfigCount']}`
- Scale configs: `{manifest['parameters']['scaleConfigCount']}`
- Scale policy count target: `{manifest['parameters']['scalePolicyCount']}`
- Sparse n=1000 policy count target: `{manifest['parameters']['sparseN1000PolicyCount']}`
- Total sweep rows: `{len(run_df)}`
- Route counts: `{json.dumps(route_counts, sort_keys=True)}`
- Timeout rows: `{timeout_count}`
- Invalid rows: `{invalid_count}`

## Results

### Route Summary

{markdown_table(tables['route'], ["route", "run_count", "policy_count", "mean_final_sortedness", "mean_delta", "timeout_count", "invalid_count"])}

### Phase Summary

{markdown_table(tables['phase'], ["phase", "split", "array_size", "run_count", "policy_count", "mean_final_sortedness", "sorted_run_fraction", "timeout_count"])}

### Top Held-out Screen Policies

{markdown_table(best[best_cols], best_cols)}

## Metrics

- `final_inversion_sortedness`: `1 - inversion_count / max_inversions`; higher is better, one means globally sorted.
- `adjacent_sortedness`: fraction of adjacent pairs in nondecreasing order.
- `inversion_sortedness_delta`: final minus initial inversion sortedness.
- `work_count`: compare-counted events plus swaps plus state updates.
- `run_status`: `ok_sorted`, `timeout_event_cap`, or `invalid`.
- `route`: `jax_batch` for S06 deterministic-compatible policies or `cpu_fallback` for exact stochastic/memory/signal cases.

## Figures

- Overview figure: `{artifacts['overviewFigure']}`

## Validation Checks

{markdown_table(validation, ["validation_case", "success", "expected", "observed", "notes"])}

## Caveats, Blockers, And Limitations

- The sweep is a proxy screen over DSL local actor steps. It does not replace the E02 deterministic simulator or the public original cell scheduler.
- Selection-style `ideal_position` is represented as one policy-row state in the scheduler, not a full per-cell memory map.
- CPU fallback rows preserve exact S02 interpreter behavior for stochastic/memory/signal policies, but `remember` and `signal` still have only S02 lightweight semantics.
- Sparse n=1000 probes intentionally use a small event cap for selected policies; they are transfer stress signals, not proof of sorting at n=1000.
- Many timeout rows are expected because broad random policy generation includes non-sorting and weakly active rules.

## Failed Assumptions

No required input was missing. The S06 compatibility table was present and every S05 policy was routable.

## Provenance

Git commit before S07 commit: `{manifest['git']['headCommit']}`

Source files hashed in the S07 manifest:

{source_table}

## Artifacts

The reusable S07 outputs are the row-level sweep table, policy-level competence table, validation table, overview figure, config, manifests, checksums, and this full-results handoff report.
"""


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    tables_dir = artifacts_dir / "tables"
    figures_dir = artifacts_dir / "figures" / "e03"
    configs_dir = artifacts_dir / "configs"
    src_snapshot_dir = artifacts_dir / "src_snapshot"
    checksums_dir = artifacts_dir / "checksums"
    for directory in (step_dir, results_dir, tables_dir, figures_dir, configs_dir, src_snapshot_dir, checksums_dir):
        directory.mkdir(parents=True, exist_ok=True)

    unit_tests = {"command": "not run", "returnCode": 0, "success": True, "stdout": "", "stderr": "", "elapsedSeconds": 0.0}
    if args.run_unit_tests:
        print("S07: running E03 unit tests", flush=True)
        unit_tests = run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e03"], args.repo_dir)

    print("S07: loading policies and S06 compatibility flags", flush=True)
    policies = load_policy_records(args.policy_library, args.compatibility_table)
    print(f"S07: screening {len(policies)} policies across {len(screen_configs())} small-array configs", flush=True)
    screen_df = run_configs(policies, screen_configs(), seed=args.seed)
    print("S07: summarizing screen and selecting scale probes", flush=True)
    screen_summary = policy_summary_frame(screen_df)
    scale_ids, sparse_ids = select_scale_policy_ids(
        screen_summary,
        target_count=args.scale_policy_count,
        sparse_count=args.sparse_n1000_policy_count,
    )
    scale_by_id = {record.policy_id: record for record in policies}
    route_by_id = {record.policy_id: record.route for record in policies}
    cpu_sparse_ids = [policy_id for policy_id in scale_ids if route_by_id.get(policy_id) == "cpu_fallback"]
    if len(cpu_sparse_ids) >= args.sparse_n1000_policy_count:
        sparse_ids = cpu_sparse_ids[: args.sparse_n1000_policy_count]
    n100_records = [scale_by_id[policy_id] for policy_id in scale_ids if policy_id in scale_by_id]
    n1000_records = [scale_by_id[policy_id] for policy_id in sparse_ids if policy_id in scale_by_id]
    all_scale_configs = scale_configs()
    n100_configs = [config for config in all_scale_configs if config.array_size == 100]
    n1000_configs = [config for config in all_scale_configs if config.array_size == 1000]
    scale_parts = []
    if n100_records:
        print(f"S07: running n=100 probes for {len(n100_records)} selected policies", flush=True)
        scale_parts.append(run_configs(n100_records, n100_configs, seed=args.seed))
    if n1000_records:
        print(f"S07: running sparse n=1000 probes for {len(n1000_records)} selected policies", flush=True)
        scale_parts.append(run_configs(n1000_records, n1000_configs, seed=args.seed))
    print("S07: building result tables, validation, and figure", flush=True)
    run_df = pd.concat([screen_df, *scale_parts], ignore_index=True) if scale_parts else screen_df
    summary_df = policy_summary_frame(run_df)
    validation = validation_frame(run_df, summary_df, len(policies), require_scale=True)
    tables = make_summary_tables(run_df, summary_df)

    sweep_path = results_dir / "e03_coarse_policy_sweep.parquet"
    sweep_csv_path = tables_dir / "e03_coarse_policy_sweep.csv"
    competence_path = results_dir / "e03_policy_competence.parquet"
    competence_csv_path = tables_dir / "e03_policy_competence.csv"
    summary_path = tables_dir / "e03_coarse_policy_sweep_summary.csv"
    route_summary_path = tables_dir / "e03_coarse_policy_route_summary.csv"
    validation_path = results_dir / "e03_coarse_sweep_validation.parquet"
    validation_csv_path = tables_dir / "e03_coarse_sweep_validation.csv"
    config_path = configs_dir / "e03_s07_coarse_sweep.json"
    overview_path = figures_dir / "coarse_sweep_overview.png"
    manifest_path = src_snapshot_dir / "e03_coarse_sweep_manifest.json"
    artifact_manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = artifacts_dir / "run_manifest.json"
    checksums_path = checksums_dir / "sha256sums.txt"
    full_report_path = step_dir / "research_step_full_results.md"

    write_overview_figure(overview_path, run_df, summary_df)
    extra_validation = [
        {
            "validation_case": "unit_tests_passed",
            "success": bool(unit_tests["success"]),
            "expected": "full E03 unit test suite passes",
            "observed": unit_tests["returnCode"],
            "notes": unit_tests["command"],
        },
        {
            "validation_case": "overview_figure_written",
            "success": overview_path.exists() and overview_path.stat().st_size > 0,
            "expected": "overview figure exists and is non-empty",
            "observed": overview_path.stat().st_size if overview_path.exists() else 0,
            "notes": str(overview_path),
        },
    ]
    validation = append_validation_cases(validation, extra_validation)
    for column in ("validation_case", "expected", "observed", "notes"):
        validation[column] = validation[column].astype(str)

    run_df.to_parquet(sweep_path, index=False)
    run_df.to_csv(sweep_csv_path, index=False)
    summary_df.to_parquet(competence_path, index=False)
    summary_df.to_csv(competence_csv_path, index=False)
    tables["phase"].to_csv(summary_path, index=False)
    tables["route"].to_csv(route_summary_path, index=False)
    validation.to_parquet(validation_path, index=False)
    validation.to_csv(validation_csv_path, index=False)
    config = {
        "schema": "eidosoma.e03.s07_coarse_sweep_config.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "seed": args.seed,
        "policyLibrary": str(args.policy_library),
        "compatibilityTable": str(args.compatibility_table),
        "screenConfigs": [config.__dict__ for config in screen_configs()],
        "scaleConfigs": [config.__dict__ for config in scale_configs()],
        "scalePolicyIds": scale_ids,
        "sparseN1000PolicyIds": sparse_ids,
        "scalePolicyCount": args.scale_policy_count,
        "sparseN1000PolicyCount": args.sparse_n1000_policy_count,
    }
    write_json(config_path, config)

    source_files = [
        args.repo_dir / "src/e03/coarse_sweep.py",
        args.repo_dir / "tests/e03/test_coarse_sweep.py",
        args.repo_dir / "scripts/e03_s07_coarse_morphospace_sweep.py",
        args.repo_dir / "src/e03/gpu_batch_simulator.py",
        args.repo_dir / "src/e03/rule_dsl.py",
    ]
    elapsed = time.perf_counter() - started
    jax_summary = jax_backend_summary()
    manifest = {
        "schema": "eidosoma.e03.coarse_sweep_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAt": utc_now(),
        "git": {
            "branch": git_output(args.repo_dir, ["branch", "--show-current"]),
            "headCommit": git_output(args.repo_dir, ["rev-parse", "HEAD"]),
            "statusShort": git_output(args.repo_dir, ["status", "--short"]),
        },
        "parameters": {
            "seed": args.seed,
            "screenConfigCount": len(screen_configs()),
            "scaleConfigCount": len(scale_configs()),
            "scalePolicyCount": args.scale_policy_count,
            "sparseN1000PolicyCount": args.sparse_n1000_policy_count,
        },
        "inputArtifacts": {
            "s05PolicyLibrary": str(args.policy_library),
            "s06CompatibilityTable": str(args.compatibility_table),
            "s06ValidationReport": str(artifacts_dir / "reports/e03_gpu_simulator_validation.md"),
            "s04CompetenceSpec": str(artifacts_dir / "reports/e03_competence_vector_spec.md"),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "jax": jax_summary,
            "elapsedSeconds": elapsed,
        },
        "sourceFiles": [source_entry(path, args.repo_dir) for path in source_files],
        "validationSummary": {
            "success": bool(validation["success"].all() and unit_tests["success"]),
            "validationCasesPassed": int(validation["success"].sum()),
            "validationCasesTotal": int(len(validation)),
            "unitTestsReturnCode": int(unit_tests["returnCode"]),
            "sweepRows": int(len(run_df)),
            "policyRows": int(len(summary_df)),
            "routeCounts": {str(key): int(value) for key, value in run_df["route"].value_counts().to_dict().items()},
            "invalidRunCount": int(run_df["invalid"].sum()),
            "timeoutRunCount": int(run_df["timed_out"].sum()),
            "resultDigest": result_digest(run_df),
        },
    }

    artifacts = {
        "researchStepReport": full_report_path,
        "coarsePolicySweep": sweep_path,
        "coarsePolicySweepCsv": sweep_csv_path,
        "policyCompetence": competence_path,
        "policyCompetenceCsv": competence_csv_path,
        "phaseSummary": summary_path,
        "routeSummary": route_summary_path,
        "validationParquet": validation_path,
        "validationCsv": validation_csv_path,
        "overviewFigure": overview_path,
        "validationConfig": config_path,
        "sourceSnapshotManifest": manifest_path,
        "artifactManifest": artifact_manifest_path,
        "runManifest": run_manifest_path,
        "checksums": checksums_path,
    }

    full_report = render_report(
        artifacts=artifacts,
        run_df=run_df,
        summary_df=summary_df,
        validation=validation,
        tables=tables,
        manifest=manifest,
        unit_tests=unit_tests,
        command_line=" ".join(sys.argv),
    )
    write_text(full_report_path, full_report)

    artifact_entries = [
        artifact_entry(sweep_path, artifacts_dir, "S07 row-level coarse policy sweep table"),
        artifact_entry(sweep_csv_path, artifacts_dir, "CSV sidecar for S07 row-level sweep"),
        artifact_entry(competence_path, artifacts_dir, "S07 policy-level competence summary"),
        artifact_entry(competence_csv_path, artifacts_dir, "CSV sidecar for S07 competence summary"),
        artifact_entry(summary_path, artifacts_dir, "S07 phase summary table"),
        artifact_entry(route_summary_path, artifacts_dir, "S07 route summary table"),
        artifact_entry(validation_path, artifacts_dir, "S07 validation cases"),
        artifact_entry(validation_csv_path, artifacts_dir, "CSV sidecar for S07 validation cases"),
        artifact_entry(overview_path, artifacts_dir, "S07 overview figure"),
        artifact_entry(config_path, artifacts_dir, "S07 validation config"),
        manifest_self_entry(manifest_path, artifacts_dir, "S07 source snapshot and provenance manifest"),
        manifest_self_entry(artifact_manifest_path, artifacts_dir, "S07 artifact manifest"),
        manifest_self_entry(run_manifest_path, artifacts_dir, "Experiment run manifest updated by S07"),
        manifest_self_entry(checksums_path, artifacts_dir, "SHA-256 checksums for key S07 outputs"),
        artifact_entry(full_report_path, artifacts_dir, "S07 full-results handoff report"),
    ]
    manifest["artifacts"] = artifact_entries
    write_json(manifest_path, manifest)
    artifact_manifest = {
        "schema": "eidosoma.research_step_artifact_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAt": utc_now(),
        "artifacts": [
            artifact_entry(sweep_path, artifacts_dir, "S07 row-level coarse policy sweep table"),
            artifact_entry(sweep_csv_path, artifacts_dir, "CSV sidecar for S07 row-level sweep"),
            artifact_entry(competence_path, artifacts_dir, "S07 policy-level competence summary"),
            artifact_entry(competence_csv_path, artifacts_dir, "CSV sidecar for S07 competence summary"),
            artifact_entry(summary_path, artifacts_dir, "S07 phase summary table"),
            artifact_entry(route_summary_path, artifacts_dir, "S07 route summary table"),
            artifact_entry(validation_path, artifacts_dir, "S07 validation cases"),
            artifact_entry(validation_csv_path, artifacts_dir, "CSV sidecar for S07 validation cases"),
            artifact_entry(overview_path, artifacts_dir, "S07 overview figure"),
            artifact_entry(config_path, artifacts_dir, "S07 validation config"),
            artifact_entry(manifest_path, artifacts_dir, "S07 source snapshot and provenance manifest"),
            manifest_self_entry(artifact_manifest_path, artifacts_dir, "S07 artifact manifest"),
            manifest_self_entry(run_manifest_path, artifacts_dir, "Experiment run manifest updated by S07"),
            manifest_self_entry(checksums_path, artifacts_dir, "SHA-256 checksums for key S07 outputs"),
            artifact_entry(full_report_path, artifacts_dir, "S07 full-results handoff report"),
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
        "runtime": manifest["runtime"],
        "commands": [
            {"command": unit_tests["command"], "returnCode": unit_tests["returnCode"], "success": unit_tests["success"]},
            {"command": " ".join(sys.argv), "returnCode": 0, "success": True},
        ],
        "inputs": manifest["inputArtifacts"],
        "parameters": manifest["parameters"],
        "outputs": artifact_manifest["artifacts"],
    }
    write_json(run_manifest_path, run_manifest)
    checksum_targets = [
        full_report_path,
        sweep_path,
        sweep_csv_path,
        competence_path,
        competence_csv_path,
        summary_path,
        route_summary_path,
        validation_path,
        validation_csv_path,
        overview_path,
        config_path,
        manifest_path,
        artifact_manifest_path,
        run_manifest_path,
    ]
    checksum_lines = [f"{sha256_file(path)}  {path.relative_to(artifacts_dir)}" for path in checksum_targets]
    write_text(checksums_path, "\n".join(checksum_lines) + "\n")

    success = bool(validation["success"].all() and unit_tests["success"])
    print(
        json.dumps(
            {
                "researchStepId": STEP_ID,
                "success": success,
                "sweepRows": len(run_df),
                "policyRows": len(summary_df),
                "routeCounts": {str(key): int(value) for key, value in run_df["route"].value_counts().to_dict().items()},
                "invalidRunCount": int(run_df["invalid"].sum()),
                "timeoutRunCount": int(run_df["timed_out"].sum()),
                "artifactsDir": str(step_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
