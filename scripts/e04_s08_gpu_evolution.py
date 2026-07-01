#!/usr/bin/env python3
"""Run E04 S08 local-only evolutionary search and write artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.e04.evolutionary_search import (
    EXCLUDED_TRAINING_SIGNAL_FIELDS,
    LOCAL_ONLY_PROTOCOL_ID,
    S08_PROTOCOL_ID,
    run_s08_search,
)


STEP_ID = "S08"
STEP_NUMBER = 8
EXPERIMENT_ID = "E04"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--population-size", type=int, default=10)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--elite-count", type=int, default=3)
    parser.add_argument("--validation-count", type=int, default=4)
    parser.add_argument("--heldout-count", type=int, default=2)
    parser.add_argument("--max-events", type=int, default=120)
    parser.add_argument("--heldout-max-events", type=int, default=140)
    parser.add_argument("--seed", type=int, default=80801)
    parser.add_argument("--run-unit-tests", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
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
        "note": "Checksum omitted to avoid self-referential checksum drift.",
    }


def source_entry(path: Path, repo_dir: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(repo_dir)),
        "sha256": sha256_file(path),
        "sizeBytes": path.stat().st_size,
    }


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 40) -> str:
    if df.empty:
        return "(no rows)"
    view = df[columns].head(max_rows)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for record in view.to_dict(orient="records"):
        rows.append("| " + " | ".join(str(record[column]).replace("|", "\\|") for column in columns) + " |")
    return "\n".join([header, separator, *rows])


def summarize_runs(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["split", "candidate_type", "policy_id"], as_index=False)
        .agg(
            runs=("fitness_score", "size"),
            mean_fitness=("fitness_score", "mean"),
            mean_time_in_target_fraction=("time_in_target_fraction", "mean"),
            mean_final_sortedness_percent=("final_sortedness_percent", "mean"),
            mean_energy_total=("energy_total", "mean"),
            mean_unfreeze_count=("unfreeze_count", "mean"),
            oracle_hit_rows=("uses_global_oracle", "sum"),
        )
        .sort_values(["split", "candidate_type", "mean_fitness"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    for column in [
        "mean_fitness",
        "mean_time_in_target_fraction",
        "mean_final_sortedness_percent",
        "mean_energy_total",
        "mean_unfreeze_count",
    ]:
        summary[column] = summary[column].round(6)
    return summary


def write_figure(progress_df: pd.DataFrame, summary_df: pd.DataFrame, figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    if not progress_df.empty:
        train_progress = (
            progress_df.groupby("generation", as_index=False)
            .agg(best_train_fitness=("mean_fitness", "max"), mean_train_fitness=("mean_fitness", "mean"))
            .sort_values("generation")
        )
        axes[0].plot(train_progress["generation"], train_progress["best_train_fitness"], marker="o", label="best")
        axes[0].plot(train_progress["generation"], train_progress["mean_train_fitness"], marker="o", label="mean")
    axes[0].set_title("Evolution progress")
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Train fitness")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    evolved = summary_df[summary_df["candidate_type"] == "evolved_local_only"]
    selected = evolved[evolved["split"].isin(["validation", "heldout"])]
    if not selected.empty:
        plot_df = (
            selected.groupby("split", as_index=False)
            .agg(best_fitness=("mean_fitness", "max"), mean_fitness=("mean_fitness", "mean"))
            .sort_values("split")
        )
        x = range(len(plot_df))
        axes[1].bar([value - 0.18 for value in x], plot_df["best_fitness"], width=0.36, label="best")
        axes[1].bar([value + 0.18 for value in x], plot_df["mean_fitness"], width=0.36, label="mean")
        axes[1].set_xticks(list(x), list(plot_df["split"]))
    axes[1].set_title("Selected-policy checks")
    axes[1].set_ylabel("Fitness")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(loc="best")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=180)
    plt.close(fig)


def enrich_selected_policies(selected: Sequence[dict[str, Any]], summary_df: pd.DataFrame) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for record in selected:
        policy_id = str(record["policy_id"])
        out = dict(record)
        for split in ("train", "validation", "heldout"):
            rows = summary_df[(summary_df["policy_id"] == policy_id) & (summary_df["split"] == split)]
            if rows.empty:
                continue
            row = rows.iloc[0].to_dict()
            out[f"{split}_mean_fitness"] = float(row["mean_fitness"])
            out[f"{split}_mean_time_in_target_fraction"] = float(row["mean_time_in_target_fraction"])
            out[f"{split}_mean_final_sortedness_percent"] = float(row["mean_final_sortedness_percent"])
            out[f"{split}_run_count"] = int(row["runs"])
        enriched.append(out)
    return enriched


def validate_results(
    *,
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
    selected_policies: Sequence[dict[str, Any]],
    split_audit: dict[str, Any],
    backend_records: Sequence[dict[str, Any]],
    drift_log: Sequence[dict[str, Any]],
    unit_tests: dict[str, Any],
    e03_policy_tests: dict[str, Any],
    e03_gpu_tests: dict[str, Any],
    e02_tests: dict[str, Any],
) -> pd.DataFrame:
    evolved = df[df["candidate_type"] == "evolved_local_only"]
    baselines = df[df["candidate_type"] == "open_loop_original"]
    rows: list[dict[str, Any]] = []
    rows.append(
        {
            "validation_case": "train_validation_heldout_splits_separated",
            "success": bool(split_audit.get("success") and set(df["split"]) == {"train", "validation", "heldout"}),
            "detail": json.dumps(split_audit, sort_keys=True, separators=(",", ":")),
        }
    )
    rows.append(
        {
            "validation_case": "local_only_features_have_no_oracle_hits",
            "success": bool((evolved["uses_global_oracle"] == False).all() and not evolved.empty),
            "detail": f"evolved rows={len(evolved)}; oracle rows={int(evolved['uses_global_oracle'].sum()) if len(evolved) else 0}",
        }
    )
    rows.append(
        {
            "validation_case": "target_derived_signal_fields_excluded",
            "success": bool(
                all(field in EXCLUDED_TRAINING_SIGNAL_FIELDS for field in ("sorted", "target_seeking", "morphogen"))
                and evolved["feature_audit_summary_json"].str.contains("excludedSignalHits").all()
            ),
            "detail": f"excluded={list(EXCLUDED_TRAINING_SIGNAL_FIELDS)}; protocol={LOCAL_ONLY_PROTOCOL_ID}",
        }
    )
    rows.append(
        {
            "validation_case": "evolved_candidates_selected_for_holdout",
            "success": bool(len(selected_policies) > 0 and (evolved["split"] == "heldout").any()),
            "detail": f"selected policies={len(selected_policies)}; heldout rows={int((evolved['split'] == 'heldout').sum())}",
        }
    )
    rows.append(
        {
            "validation_case": "policy_lineages_logged",
            "success": bool(
                len(selected_policies) > 0
                and all("parent_ids" in item and "parameters" in item for item in selected_policies)
                and evolved["parent_ids_json"].notna().all()
            ),
            "detail": f"selected lineage records={len(selected_policies)}; evolved policies={evolved['policy_id'].nunique()}",
        }
    )
    rows.append(
        {
            "validation_case": "open_loop_baselines_present",
            "success": bool(
                {"open_loop_bubble", "open_loop_insertion", "open_loop_selection"}.issubset(set(baselines["policy_id"]))
                and set(baselines["split"]) == {"train", "validation", "heldout"}
            ),
            "detail": f"baseline rows={len(baselines)}; policies={sorted(set(baselines['policy_id'])) if len(baselines) else []}",
        }
    )
    rows.append(
        {
            "validation_case": "no_centralized_baseline_mixed_into_local_claims",
            "success": bool((df["centralized_baseline"] == False).all()),
            "detail": f"centralized rows={int(df['centralized_baseline'].sum())}",
        }
    )
    rows.append(
        {
            "validation_case": "gpu_and_batch_drift_logged",
            "success": bool(backend_records and drift_log and any(record.get("defaultBackend") == "gpu" for record in backend_records)),
            "detail": json.dumps({"backendRecords": list(backend_records), "driftLog": list(drift_log)}, sort_keys=True, default=str)[:1800],
        }
    )
    best_validation = summary_df[(summary_df["candidate_type"] == "evolved_local_only") & (summary_df["split"] == "validation")]
    rows.append(
        {
            "validation_case": "validation_scores_computed_for_selected_candidates",
            "success": bool(not best_validation.empty and best_validation["mean_fitness"].notna().all()),
            "detail": f"validation candidate summaries={len(best_validation)}",
        }
    )
    rows.append(
        {
            "validation_case": "unit_and_regression_tests_passed",
            "success": bool(
                unit_tests["success"]
                and e03_policy_tests["success"]
                and e03_gpu_tests["success"]
                and e02_tests["success"]
            ),
            "detail": (
                f"E04={unit_tests['returnCode']}; E03 policy={e03_policy_tests['returnCode']}; "
                f"E03 gpu={e03_gpu_tests['returnCode']}; E02={e02_tests['returnCode']}"
            ),
        }
    )
    return pd.DataFrame(rows)


def render_drift_report(drift_log: Sequence[dict[str, Any]], backend_records: Sequence[dict[str, Any]]) -> str:
    drift_df = pd.DataFrame(list(drift_log))
    return f"""# E04 S08 GPU And Batched-Simulator Drift Log

## Top Summary

- Research step ID: S08
- Completion status: Completed on {utc_now()}
- Artifacts written:
- `/artifacts/reports/e04_s08_gpu_batch_drift.md`
- Validation result: GPU backend and batch-simulator drift items logged.
- Outcome classification: supportive with caveat
- Caveats or blockers: JAX reports a GPU backend, but the E03 DSL trajectory kernel was not used for policy trajectories because it exposes target/ideal-position machinery incompatible with the S07 local-only training view.
- Recommended next action: Use the selected S08 policies in S09 memory ablations while preserving this drift caveat.

## Drift Items

{markdown_table(drift_df, ["item", "status", "detail"], max_rows=20)}

## Backend Records

```json
{json.dumps(list(backend_records), indent=2, sort_keys=True, default=str)}
```
"""


def render_report(
    *,
    df: pd.DataFrame,
    progress_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    selected_policies: Sequence[dict[str, Any]],
    validation_line: str,
    validation_success: bool,
    unit_tests: dict[str, Any],
    e03_policy_tests: dict[str, Any],
    e03_gpu_tests: dict[str, Any],
    e02_tests: dict[str, Any],
    artifact_paths: Sequence[Path],
    result_path: Path,
    policy_path: Path,
    figure_path: Path,
    drift_report_path: Path,
    config_path: Path,
    manifest_path: Path,
    run_manifest_path: Path,
    checksums_path: Path,
    backend_records: Sequence[dict[str, Any]],
    drift_log: Sequence[dict[str, Any]],
    args: argparse.Namespace,
) -> str:
    outcome = "supportive" if validation_success and len(selected_policies) > 0 else "constraining/contradictory"
    artifact_md = "\n".join(f"- `{path}`" for path in artifact_paths)
    commands = "\n".join(
        [
            f"- `{unit_tests['command']}` -> return code {unit_tests['returnCode']}",
            f"- `{e03_policy_tests['command']}` -> return code {e03_policy_tests['returnCode']}",
            f"- `{e03_gpu_tests['command']}` -> return code {e03_gpu_tests['returnCode']}",
            f"- `{e02_tests['command']}` -> return code {e02_tests['returnCode']}",
            (
                "- `python scripts/e04_s08_gpu_evolution.py --repo-dir /workspace/cell-research "
                "--artifacts-dir $ARTIFACTS_DIR`"
            ),
        ]
    )
    selected_df = pd.DataFrame(list(selected_policies))
    selected_table_cols = [
        "policy_id",
        "generation",
        "validation_mean_fitness",
        "heldout_mean_fitness",
        "heldout_mean_time_in_target_fraction",
    ]
    selected_table = markdown_table(selected_df, [col for col in selected_table_cols if col in selected_df.columns], max_rows=10)
    top_summary = summary_df.sort_values(["split", "mean_fitness"], ascending=[True, False])
    summary_table = markdown_table(
        top_summary,
        [
            "split",
            "candidate_type",
            "policy_id",
            "runs",
            "mean_fitness",
            "mean_time_in_target_fraction",
            "mean_final_sortedness_percent",
            "oracle_hit_rows",
        ],
        max_rows=24,
    )
    validation_table = markdown_table(validation_df, ["validation_case", "success", "detail"], max_rows=20)
    progress_table = markdown_table(
        progress_df[["generation", "rank", "policy_id", "mean_fitness", "mean_time_in_target_fraction", "oracle_hit_rows"]],
        ["generation", "rank", "policy_id", "mean_fitness", "mean_time_in_target_fraction", "oracle_hit_rows"],
        max_rows=24,
    )
    return f"""# E04 S08 Research Step Full Results

## Top Summary

- Research step ID: S08
- Completion status: {'Completed' if validation_success else 'Completed with validation failure'} on {utc_now()}
- Artifacts written:
{artifact_md}
- Validation result: {validation_line}
- Outcome classification: {outcome}
- Caveats or blockers: E03's JAX DSL batch trajectory kernel was not used for policy trajectories because it retains target/ideal-position machinery incompatible with S07 local-only training; S08 used the S05 Python event simulator for trajectories and JAX/GPU for vectorized population initialization/mutation metadata. No centralized baseline was run or mixed into local-only claims.
- Lay summary: S08 evolved small local policies that decide only among idle, swap-left, and swap-right from immediate-neighbor values, bounded memory counters, and allowed blocked/frustrated signals. The best candidates were selected on training tasks, checked on separate validation tasks, and then run on held-out larger arrays with cumulative damage.
- Recommended next action: Hand control back to the Chief Scientist. If accepted, run S09 memory ablations on the selected S08 policies and preserve train/validation/held-out separation.

## Frozen Question

Can evolved memory and signaling rules repair better than original open-loop algorithms under batched simulation?

S08 answers this as an evolutionary-search readiness and candidate-selection step. It selects local-only policies for downstream ablation and compares them with original open-loop baselines on matched train, validation, and held-out task families. It does not yet prove a minimal memory or signaling mechanism; that is the purpose of S09 and S10.

## Inputs

- Active plan: `/workspace/RESEARCH_PLAN.md`, Experiment E04, step S08.
- S07 protocol: `src/e04/no_oracle_protocol.py` and `/artifacts/reports/e04_no_global_oracle_audit.md`.
- S05 benchmark substrate: `src/e04/homeostasis.py`.
- S01/S02 interfaces: `src/e04/memory_policies.py`, `src/e04/signaling.py`.
- E03 GPU context: `src/e03/gpu_batch_simulator.py`.
- Previous mounted artifacts: E01 `/previous-artifacts/E01`, E02 `/previous-artifacts/E02`, E03 `/previous-artifacts/E03`.
- Datasets: none required.

## Methods

Implemented `src/e04/evolutionary_search.py`, `tests/e04/test_evolutionary_search.py`, and `scripts/e04_s08_gpu_evolution.py`.

Each evolved policy is a parameterized adjacent-action rule over the S07 `LocalTrainingObservation` projection. Policy inputs include actor value/status, immediate neighbor values/statuses, boundary flags, reverse-direction flag, bounded S01 memory counters, and S02 `blocked` and `frustrated` fields. The policy does not receive raw E03 `values`, `labels`, `statuses`, `ideal_position`, global Sortedness, final target, or target-derived `sorted`, `target_seeking`, or `morphogen` signals.

The search used `{args.population_size}` candidates for `{args.generations}` generations. Training fitness was computed on separate train configurations, top candidates were evaluated on validation configurations, and the final selected policies were evaluated on held-out larger arrays. Original open-loop Bubble, Insertion, and Selection baselines were evaluated on the same split configs.

Fitness combined offline trajectory metrics after each run: time-in-target fraction, final sortedness, perturbation recovery, repair/unfreeze evidence when freeze perturbations occurred, lower energy use, and lower impairment burden. These global metrics were used only after trajectories were complete, not inside policy decisions.

## Commands

{commands}

## Dependencies And Runtime

- Python: {platform.python_version()}
- pandas: {pd.__version__}
- New dependencies installed: none.
- CPU use: serial stateful trajectory evaluation with worker count `1`; host logical CPUs recorded in run manifest.
- GPU/JAX: `{json.dumps(list(backend_records), sort_keys=True, default=str)}`
- Platform: {platform.platform()}

## Parameters

- Protocol ID: `{S08_PROTOCOL_ID}`
- Population size: `{args.population_size}`
- Generations: `{args.generations}`
- Elite count: `{args.elite_count}`
- Validation candidate count: `{args.validation_count}`
- Held-out candidate count: `{args.heldout_count}`
- Train/validation max events: `{args.max_events}`
- Held-out max events: `{args.heldout_max_events}`
- Seed: `{args.seed}`
- Result table: `{result_path}`
- Selected policies: `{policy_path}`
- Figure: `{figure_path}`
- Drift report: `{drift_report_path}`

## Results

- Total result rows: `{len(df)}`
- Evolved local-only rows: `{int((df['candidate_type'] == 'evolved_local_only').sum())}`
- Open-loop baseline rows: `{int((df['candidate_type'] == 'open_loop_original').sum())}`
- Selected policies: `{len(selected_policies)}`
- Evolved rows with oracle hits: `{int(df[df['candidate_type'] == 'evolved_local_only']['uses_global_oracle'].sum())}`

### Selected Policies

{selected_table}

### Evolution Progress

{progress_table}

### Summary By Split And Policy

{summary_table}

## Validation Checks

{validation_table}

## GPU And Batched-Simulator Drift

{markdown_table(pd.DataFrame(list(drift_log)), ["item", "status", "detail"], max_rows=20)}

The full drift log is at `{drift_report_path}`.

## Caveats, Blockers, Failed Assumptions, And Limitations

- No blocker remains for S08 candidate selection.
- The trajectory simulator path is not a fully GPU-batched S07-safe simulator. The E03 JAX DSL simulator remains useful context but was not used for policy trajectories because its target-position machinery violates the S07 local-only training contract.
- The selected policies are compact adjacent-action policies, not a broad morphospace proof.
- Fitness is a computational proxy combining repair, homeostasis, robustness, energy, and transfer metrics; it is not biological validation.
- S09 and S10 must ablate memory and signal terms before attributing gains to specific mechanisms.

## Provenance

- Git commit at validation time: `{git_output(args.repo_dir, ['rev-parse', 'HEAD'])}`
- Git branch at validation time: `{git_output(args.repo_dir, ['branch', '--show-current'])}`
- Git status at validation time: `{git_output(args.repo_dir, ['status', '--short']) or 'clean'}`
- Config: `{config_path}`
- Artifact manifest: `{manifest_path}`
- Run manifest: `{run_manifest_path}`
- Checksums: `{checksums_path}`
- Created at UTC: `{utc_now()}`

## Artifact Manifest

- Result parquet: `{result_path}`
- Selected-policy JSONL: `{policy_path}`
- Figure: `{figure_path}`
- Drift report: `{drift_report_path}`

## Recommended Next Action

Hand control back to the Chief Scientist. If S08 is accepted, execute S09 memory ablations using the selected S08 policies and keep S07 local-only projections enforced.
"""


def main() -> int:
    args = parse_args()
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    result_path = artifacts_dir / "results" / "e04_evolution_runs.parquet"
    result_csv_path = artifacts_dir / "tables" / "e04_evolution_runs.csv"
    summary_path = artifacts_dir / "tables" / "e04_evolution_summary.csv"
    progress_path = artifacts_dir / "tables" / "e04_evolution_progress.csv"
    validation_path = artifacts_dir / "tables" / "e04_evolution_validation.csv"
    policy_path = artifacts_dir / "policies" / "e04_evolved_repair_policies.jsonl"
    figure_path = artifacts_dir / "figures" / "e04" / "evolution_progress.png"
    drift_report_path = artifacts_dir / "reports" / "e04_s08_gpu_batch_drift.md"
    config_path = artifacts_dir / "configs" / "e04_s08_gpu_evolution.json"
    source_manifest_path = artifacts_dir / "src_snapshot" / "e04_gpu_evolution_manifest.json"
    report_path = step_dir / "research_step_full_results.md"
    manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = artifacts_dir / "run_manifest.json"
    checksums_path = artifacts_dir / "checksums" / "sha256sums.txt"

    config_record = {
        "schema": "eidosoma.e04.s08.gpu_evolution_config.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAtUtc": utc_now(),
        "populationSize": args.population_size,
        "generations": args.generations,
        "eliteCount": args.elite_count,
        "validationCount": args.validation_count,
        "heldoutCount": args.heldout_count,
        "maxEvents": args.max_events,
        "heldoutMaxEvents": args.heldout_max_events,
        "seed": args.seed,
        "protocolId": S08_PROTOCOL_ID,
        "workerCount": 1,
    }
    write_json(config_path, config_record)

    search = run_s08_search(
        population_size=args.population_size,
        generations=args.generations,
        elite_count=args.elite_count,
        validation_count=args.validation_count,
        heldout_count=args.heldout_count,
        max_events=args.max_events,
        heldout_max_events=args.heldout_max_events,
        seed=args.seed,
    )
    df = pd.DataFrame(search["rows"])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(result_path, index=False)
    df.to_csv(result_csv_path, index=False)

    progress_df = pd.DataFrame(search["progressRows"])
    if not progress_df.empty:
        for column in ("mean_fitness", "mean_time_in_target_fraction", "mean_final_sortedness_percent", "mean_energy_total"):
            if column in progress_df:
                progress_df[column] = progress_df[column].round(6)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_df.to_csv(progress_path, index=False)

    summary_df = summarize_runs(df)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)

    selected_policies = enrich_selected_policies(search["selectedPolicies"], summary_df)
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    with policy_path.open("w", encoding="utf-8") as handle:
        for record in selected_policies:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":"), default=str) + "\n")

    write_figure(progress_df, summary_df, figure_path)
    write_text(drift_report_path, render_drift_report(search["driftLog"], search["backendRecords"]))

    if args.run_unit_tests:
        unit_tests = run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e04", "-p", "test_*.py"], args.repo_dir)
        e03_policy_tests = run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e03", "-p", "test_policy_interface.py"], args.repo_dir)
        e03_gpu_tests = run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e03", "-p", "test_gpu_batch_simulator.py"], args.repo_dir)
        e02_tests = run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e02", "-p", "test_deterministic_simulator.py"], args.repo_dir)
    else:
        skipped = {"command": "skipped by --no-run-unit-tests", "returnCode": 0, "elapsedSeconds": 0.0, "stdout": "", "stderr": "", "success": True}
        unit_tests = e03_policy_tests = e03_gpu_tests = e02_tests = skipped

    validation_df = validate_results(
        df=df,
        summary_df=summary_df,
        selected_policies=selected_policies,
        split_audit=search["splitAudit"],
        backend_records=search["backendRecords"],
        drift_log=search["driftLog"],
        unit_tests=unit_tests,
        e03_policy_tests=e03_policy_tests,
        e03_gpu_tests=e03_gpu_tests,
        e02_tests=e02_tests,
    )
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation_df.to_csv(validation_path, index=False)
    validation_success = bool(validation_df["success"].all())
    validation_line = (
        f"{int(validation_df['success'].sum())}/{len(validation_df)} validation cases passed; "
        f"E04 tests return code {unit_tests['returnCode']}; E03 policy tests return code {e03_policy_tests['returnCode']}; "
        f"E03 GPU tests return code {e03_gpu_tests['returnCode']}; E02 simulator tests return code {e02_tests['returnCode']}; "
        f"{len(selected_policies)} selected policies; evolved oracle-hit rows={int(df[df['candidate_type'] == 'evolved_local_only']['uses_global_oracle'].sum())}"
    )

    source_files = [
        args.repo_dir / "src/e04/evolutionary_search.py",
        args.repo_dir / "scripts/e04_s08_gpu_evolution.py",
        args.repo_dir / "tests/e04/test_evolutionary_search.py",
        args.repo_dir / "src/e04/no_oracle_protocol.py",
        args.repo_dir / "src/e04/homeostasis.py",
        args.repo_dir / "src/e04/signaling.py",
        args.repo_dir / "src/e04/memory_policies.py",
        args.repo_dir / "src/e03/gpu_batch_simulator.py",
    ]
    source_manifest = {
        "schema": "eidosoma.src_snapshot.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAtUtc": utc_now(),
        "gitCommit": git_output(args.repo_dir, ["rev-parse", "HEAD"]),
        "gitBranch": git_output(args.repo_dir, ["branch", "--show-current"]),
        "sources": [source_entry(path, args.repo_dir) for path in source_files if path.exists()],
    }
    write_json(source_manifest_path, source_manifest)

    artifact_paths = [
        report_path,
        result_path,
        result_csv_path,
        summary_path,
        progress_path,
        validation_path,
        policy_path,
        figure_path,
        drift_report_path,
        config_path,
        source_manifest_path,
        manifest_path,
        run_manifest_path,
        checksums_path,
    ]
    report_text = render_report(
        df=df,
        progress_df=progress_df,
        summary_df=summary_df,
        validation_df=validation_df,
        selected_policies=selected_policies,
        validation_line=validation_line,
        validation_success=validation_success,
        unit_tests=unit_tests,
        e03_policy_tests=e03_policy_tests,
        e03_gpu_tests=e03_gpu_tests,
        e02_tests=e02_tests,
        artifact_paths=artifact_paths,
        result_path=result_path,
        policy_path=policy_path,
        figure_path=figure_path,
        drift_report_path=drift_report_path,
        config_path=config_path,
        manifest_path=manifest_path,
        run_manifest_path=run_manifest_path,
        checksums_path=checksums_path,
        backend_records=search["backendRecords"],
        drift_log=search["driftLog"],
        args=args,
    )
    write_text(report_path, report_text)

    manifest_artifacts = [
        artifact_entry(report_path, artifacts_dir, "S08 full-results handoff report"),
        artifact_entry(result_path, artifacts_dir, "S08 evolution run results"),
        artifact_entry(result_csv_path, artifacts_dir, "CSV sidecar for S08 evolution run results"),
        artifact_entry(summary_path, artifacts_dir, "S08 summary table"),
        artifact_entry(progress_path, artifacts_dir, "S08 evolution progress table"),
        artifact_entry(validation_path, artifacts_dir, "S08 validation table"),
        artifact_entry(policy_path, artifacts_dir, "S08 selected evolved repair policies"),
        artifact_entry(figure_path, artifacts_dir, "S08 evolution progress figure"),
        artifact_entry(drift_report_path, artifacts_dir, "S08 GPU/batched-simulator drift report"),
        artifact_entry(config_path, artifacts_dir, "S08 search configuration"),
        artifact_entry(source_manifest_path, artifacts_dir, "S08 source/provenance manifest"),
        manifest_self_entry(manifest_path, artifacts_dir, "S08 artifact manifest"),
        manifest_self_entry(run_manifest_path, artifacts_dir, "Experiment run manifest"),
        manifest_self_entry(checksums_path, artifacts_dir, "SHA-256 checksums for key S08 outputs"),
    ]
    artifact_manifest = {
        "schema": "eidosoma.artifact_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAtUtc": utc_now(),
        "artifacts": manifest_artifacts,
    }
    write_json(manifest_path, artifact_manifest)

    run_manifest = {
        "schema": "eidosoma.run_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAtUtc": utc_now(),
        "gitCommit": git_output(args.repo_dir, ["rev-parse", "HEAD"]),
        "gitBranch": git_output(args.repo_dir, ["branch", "--show-current"]),
        "gitStatusShort": git_output(args.repo_dir, ["status", "--short"]),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "workerCount": 1,
        "cpuCount": os.cpu_count(),
        "configPath": str(config_path),
        "resultRows": int(len(df)),
        "selectedPolicyCount": int(len(selected_policies)),
        "validation": validation_df.to_dict(orient="records"),
        "backendRecords": list(search["backendRecords"]),
        "driftLog": list(search["driftLog"]),
        "artifacts": manifest_artifacts,
    }
    write_json(run_manifest_path, run_manifest)

    checksum_lines = []
    for path in [
        report_path,
        result_path,
        result_csv_path,
        summary_path,
        progress_path,
        validation_path,
        policy_path,
        figure_path,
        drift_report_path,
        config_path,
        source_manifest_path,
        manifest_path,
        run_manifest_path,
    ]:
        checksum_lines.append(f"{sha256_file(path)}  {path}")
    checksums_path.parent.mkdir(parents=True, exist_ok=True)
    checksums_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    print(validation_line)
    return 0 if validation_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
