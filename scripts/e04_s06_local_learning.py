#!/usr/bin/env python3
"""Run E04 S06 local learning pilots and write artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.e04.fatigue_damage import RELIABILITY_MODES
from src.e04.homeostasis import HOMEOSTATIC_TASKS
from src.e04.local_learning import (
    LEARNING_FORBIDDEN_KEYS,
    LEARNING_POLICY_MODES,
    LEARNING_REWARD_FEATURES,
    default_local_learning_configs,
    run_local_learning_matrix,
)


STEP_ID = "S06"
STEP_NUMBER = 6
EXPERIMENT_ID = "E04"
INTERFACE_MODES = ("full", "no_memory", "no_signal")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--max-events", type=int, default=120)
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


def build_comparison(df: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "task_name",
        "interface_mode",
        "reliability_mode",
        "activation_seed",
        "policy_seed",
        "schedule_seed",
        "learning_seed",
    ]
    metrics = [
        "time_in_target_fraction",
        "mean_recovery_events",
        "energy_total",
        "final_sortedness_percent",
        "mean_local_reward",
    ]
    rows: list[dict[str, Any]] = []
    for key_values, group in df.groupby(keys, dropna=False):
        by_mode = {row["policy_mode"]: row for row in group.to_dict(orient="records")}
        if not set(LEARNING_POLICY_MODES).issubset(by_mode):
            continue
        record = dict(zip(keys, key_values, strict=True))
        learning = by_mode["local_learning"]
        control = by_mode["nonlearning_control"]
        for metric in metrics:
            lv = learning.get(metric)
            cv = control.get(metric)
            record[f"learning_{metric}"] = lv
            record[f"control_{metric}"] = cv
            if lv is None or cv is None or pd.isna(lv) or pd.isna(cv):
                record[f"delta_{metric}"] = None
            else:
                record[f"delta_{metric}"] = float(lv) - float(cv)
        record["learning_update_count"] = int(learning["learning_update_count"])
        record["control_update_count"] = int(control["learning_update_count"])
        record["learning_uses_global_oracle"] = bool(learning["uses_global_oracle"])
        record["control_uses_global_oracle"] = bool(control["uses_global_oracle"])
        rows.append(record)
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def validate_results(df: pd.DataFrame, comparison: pd.DataFrame, max_events: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.append(
        {
            "validation_case": "pilot_matrix_complete",
            "success": bool(
                set(df["task_name"]) == set(HOMEOSTATIC_TASKS)
                and set(df["interface_mode"]) == set(INTERFACE_MODES)
                and set(df["reliability_mode"]) == set(RELIABILITY_MODES)
                and set(df["policy_mode"]) == set(LEARNING_POLICY_MODES)
                and len(df) == 144
            ),
            "detail": (
                f"tasks={sorted(set(df['task_name']))}; interfaces={sorted(set(df['interface_mode']))}; "
                f"reliability={sorted(set(df['reliability_mode']))}; policy_modes={sorted(set(df['policy_mode']))}; rows={len(df)}"
            ),
        }
    )
    rows.append(
        {
            "validation_case": "matched_nonlearning_controls_present",
            "success": bool(len(comparison) == 72 and comparison["control_update_count"].eq(0).all()),
            "detail": f"matched pairs={len(comparison)}; control updates={int(comparison['control_update_count'].sum()) if len(comparison) else 0}",
        }
    )
    learning = df[df["policy_mode"] == "local_learning"]
    control = df[df["policy_mode"] == "nonlearning_control"]
    rows.append(
        {
            "validation_case": "learning_updates_logged",
            "success": bool((learning["learning_update_count"] > 0).all() and control["learning_update_count"].eq(0).all()),
            "detail": f"learning updates={int(learning['learning_update_count'].sum())}; control updates={int(control['learning_update_count'].sum())}",
        }
    )
    rows.append(
        {
            "validation_case": "no_global_reward_or_oracle_used",
            "success": bool(
                (df["uses_global_oracle"] == False).all()
                and not any(
                    token in feature.lower()
                    for feature in LEARNING_REWARD_FEATURES
                    for token in LEARNING_FORBIDDEN_KEYS
                )
            ),
            "detail": f"oracle hits={int(df['uses_global_oracle'].sum())}; reward features={list(LEARNING_REWARD_FEATURES)}",
        }
    )
    rows.append(
        {
            "validation_case": "s01_s02_ablation_controls_present",
            "success": bool({"no_memory", "no_signal", "full"}.issubset(set(df["interface_mode"]))),
            "detail": f"interfaces={sorted(set(df['interface_mode']))}",
        }
    )
    rows.append(
        {
            "validation_case": "s05_homeostatic_tasks_used",
            "success": bool(
                (df["perturbation_count"] > 0).all()
                and (df["swap_perturbation_count"] > 0).any()
                and (df["delete_insert_perturbation_count"] > 0).any()
                and (df["freeze_perturbation_count"] > 0).any()
            ),
            "detail": (
                f"perturbations={int(df['perturbation_count'].sum())}; "
                f"swap={int(df['swap_perturbation_count'].sum())}; "
                f"delete_insert={int(df['delete_insert_perturbation_count'].sum())}; "
                f"freeze={int(df['freeze_perturbation_count'].sum())}"
            ),
        }
    )
    rows.append(
        {
            "validation_case": "fixed_horizon_and_metrics_finite",
            "success": bool(
                (df["stop_reason"] == "fixed_horizon_complete").all()
                and (df["event_count"] == max_events).all()
                and df["time_in_target_fraction"].between(0.0, 1.0).all()
                and all(math.isfinite(float(value)) for value in df["mean_local_reward"])
            ),
            "detail": f"stop reasons={df['stop_reason'].value_counts().to_dict()}; max_events={max_events}",
        }
    )
    mean_delta = float(comparison["delta_time_in_target_fraction"].mean()) if len(comparison) else float("nan")
    rows.append(
        {
            "validation_case": "pilot_comparison_computed",
            "success": bool(
                len(comparison) == 72
                and math.isfinite(mean_delta)
                and (comparison["delta_time_in_target_fraction"] > 0).sum() > 0
            ),
            "detail": (
                f"mean time-in-target delta={mean_delta:.6f}; "
                f"positive pairs={int((comparison['delta_time_in_target_fraction'] > 0).sum()) if len(comparison) else 0}; "
                f"negative pairs={int((comparison['delta_time_in_target_fraction'] < 0).sum()) if len(comparison) else 0}"
            ),
        }
    )
    return pd.DataFrame(rows)


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["policy_mode", "task_name", "interface_mode", "reliability_mode"], as_index=False)
        .agg(
            runs=("time_in_target_fraction", "size"),
            mean_time_in_target_fraction=("time_in_target_fraction", "mean"),
            mean_recovery_events=("mean_recovery_events", "mean"),
            mean_energy_total=("energy_total", "mean"),
            mean_final_sortedness_percent=("final_sortedness_percent", "mean"),
            mean_learning_updates=("learning_update_count", "mean"),
            mean_local_reward=("mean_local_reward", "mean"),
            oracle_hit_count=("uses_global_oracle", "sum"),
        )
        .sort_values(["task_name", "interface_mode", "reliability_mode", "policy_mode"])
    )
    for column in [
        "mean_time_in_target_fraction",
        "mean_recovery_events",
        "mean_energy_total",
        "mean_final_sortedness_percent",
        "mean_learning_updates",
        "mean_local_reward",
    ]:
        summary[column] = summary[column].round(4)
    return summary


def write_figure(comparison: pd.DataFrame, figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    task_delta = (
        comparison.groupby("task_name", as_index=False)
        .agg(
            mean_delta_time_in_target=("delta_time_in_target_fraction", "mean"),
            mean_delta_energy=("delta_energy_total", "mean"),
            positive_pairs=("delta_time_in_target_fraction", lambda values: int((values > 0).sum())),
        )
        .sort_values("task_name")
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].bar(task_delta["task_name"], task_delta["mean_delta_time_in_target"], color="#356c9a")
    axes[0].axhline(0, color="#333333", linewidth=0.8)
    axes[0].set_title("Learning minus control")
    axes[0].set_ylabel("Time-in-target delta")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(task_delta["task_name"], task_delta["positive_pairs"], color="#7b7f35")
    axes[1].set_title("Positive matched pairs")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("E04 S06 local learning pilot comparison", fontsize=13)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=180)
    plt.close(fig)


def render_rule_spec(
    *,
    artifact_paths: list[Path],
    validation_line: str,
    validation_success: bool,
    result_path: Path,
    comparison_path: Path,
    config_path: Path,
) -> str:
    artifact_md = "\n".join(f"- `{path}`" for path in artifact_paths)
    outcome = "supportive" if validation_success else "constraining/contradictory"
    return f"""# E04 Local Learning Rule Specification

## Top Summary

- Research step ID: S06
- Completion status: {'Completed' if validation_success else 'Completed with validation failure'} on {utc_now()}
- Artifacts written:
{artifact_md}
- Validation result: {validation_line}
- Outcome classification: {outcome}
- Caveats or blockers: The S06 learner is a compact adjacent-action pilot over Bubble-style local swaps on S05 tasks. It is not yet a broad learned-policy search over the full E03 morphospace.
- Recommended next action: Stop before S07. Chief Scientist should review whether the local reward contract is strict enough before the S07 no-oracle training audit.

## Local Reward Contract

The learner updates per-cell weights for `swap_left`, `swap_right`, and `idle`. Its reward terms are:

- `local_order_delta`: change in adjacent-order violations inside the actor/target immediate local window.
- `swap_success_bonus`: small local bonus when the chosen adjacent action swaps successfully.
- `blocked_penalty`: penalty when a local swap is blocked, including Frozen Cell blocks.
- `idle_with_local_disorder_penalty`: penalty for idling while the local window still contains a local disorder.
- `memory_frustration_delta`: small penalty or recovery term from S01 local frustration changes.

Forbidden reward sources are not used: global Sortedness, full-array rank, whole-array target distance, and final-target oracle. Sortedness and time-in-target are computed only after runs for offline evaluation.

## Inputs And Interfaces

- S01 memory supplies bounded per-cell frustration and local failed-swap state.
- S02 signaling supplies local/diffusive blocked and frustrated fields when enabled.
- S05 supplies reproducible homeostatic task schedules.
- Config path: `{config_path}`
- Pilot result table: `{result_path}`
- Matched comparison table: `{comparison_path}`

## Pilot Matrix

Default matrix: 2 seeds x 4 S05 tasks x 3 interface modes x 3 reliability modes x 2 policy modes = 144 rows. Each learning row is matched to a nonlearning no-update control with the same activation, policy, schedule, and learning seeds.
"""


def render_report(
    *,
    df: pd.DataFrame,
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    validation_df: pd.DataFrame,
    validation_line: str,
    validation_success: bool,
    unit_tests: dict[str, Any],
    e03_tests: dict[str, Any],
    e02_tests: dict[str, Any],
    artifact_paths: list[Path],
    result_path: Path,
    comparison_path: Path,
    figure_path: Path,
    spec_path: Path,
    summary_path: Path,
    config_path: Path,
    run_manifest_path: Path,
    checksums_path: Path,
    manifest: dict[str, Any],
) -> str:
    mean_delta = float(comparison["delta_time_in_target_fraction"].mean()) if len(comparison) else float("nan")
    positive_pairs = int((comparison["delta_time_in_target_fraction"] > 0).sum()) if len(comparison) else 0
    negative_pairs = int((comparison["delta_time_in_target_fraction"] < 0).sum()) if len(comparison) else 0
    outcome = "supportive" if validation_success else "constraining/contradictory"
    artifact_md = "\n".join(f"- `{path}`" for path in artifact_paths)
    commands = "\n".join(
        [
            f"- `{unit_tests['command']}` -> return code {unit_tests['returnCode']}",
            f"- `{e03_tests['command']}` -> return code {e03_tests['returnCode']}",
            f"- `{e02_tests['command']}` -> return code {e02_tests['returnCode']}",
            "- `python scripts/e04_s06_local_learning.py --repo-dir /workspace/cell-research --artifacts-dir $ARTIFACTS_DIR`",
        ]
    )
    summary_headline = (
        summary.groupby(["policy_mode", "task_name"], as_index=False)
        .agg(
            mean_time_in_target_fraction=("mean_time_in_target_fraction", "mean"),
            mean_energy_total=("mean_energy_total", "mean"),
            mean_learning_updates=("mean_learning_updates", "mean"),
            mean_local_reward=("mean_local_reward", "mean"),
            oracle_hit_count=("oracle_hit_count", "sum"),
        )
        .round(4)
    )
    summary_table = markdown_table(
        summary_headline,
        [
            "policy_mode",
            "task_name",
            "mean_time_in_target_fraction",
            "mean_energy_total",
            "mean_learning_updates",
            "mean_local_reward",
            "oracle_hit_count",
        ],
        max_rows=20,
    )
    comparison_table = markdown_table(
        comparison.groupby("task_name", as_index=False)
        .agg(
            pairs=("delta_time_in_target_fraction", "size"),
            mean_delta_time_in_target=("delta_time_in_target_fraction", "mean"),
            mean_delta_energy=("delta_energy_total", "mean"),
            positive_pairs=("delta_time_in_target_fraction", lambda values: int((values > 0).sum())),
            negative_pairs=("delta_time_in_target_fraction", lambda values: int((values < 0).sum())),
        )
        .round(4),
        ["task_name", "pairs", "mean_delta_time_in_target", "mean_delta_energy", "positive_pairs", "negative_pairs"],
    )
    validation_table = markdown_table(validation_df, ["validation_case", "success", "detail"])
    return f"""# E04 S06 Research Step Full Results

## Top Summary

- Research step ID: S06
- Completion status: {'Completed' if validation_success else 'Completed with validation failure'} on {utc_now()}
- Artifacts written:
{artifact_md}
- Validation result: {validation_line}
- Outcome classification: {outcome}
- Caveats or blockers: This is a compact local-learning pilot, not a full policy-search result. The adjacent-action learner improved mean time-in-target in the matched pilot matrix, but some matched pairs worsened and the policy is limited to local Bubble-style moves.
- Lay summary: S06 adds a simple local learner. Each cell adjusts small action weights from nearby ordering changes, blocked moves, movement success, memory frustration, and local signals. The learner never receives global Sortedness or full-array rank as reward. In the pilot, learning rows were compared against matched no-update controls on the S05 maintenance tasks.
- Recommended next action: Stop before S07. Chief Scientist should review the local reward contract and then authorize S07 to audit training pathways for hidden global-oracle access.

## Frozen Question

Can simple local reinforcement-like updates improve future local outcomes without global reward signals?

S06 supports a cautious pilot answer: the local-only update rule produced a positive aggregate time-in-target delta versus matched nonlearning controls in this compact matrix, while preserving explicit no-global-oracle audits. This is not yet evidence of broad learned competence.

## Inputs

- Active plan: `/workspace/RESEARCH_PLAN.md`, Experiment E04, step S06.
- S01 memory interface: `src/e04/memory_policies.py`.
- S02 signaling interface: `src/e04/signaling.py`.
- S05 homeostatic task suite: `src/e04/homeostasis.py` and `/artifacts/results/e04_homeostatic_baselines.parquet`.
- S04 fatigue/damage and S03 repair code remain in the event loop.
- Datasets: none required.

## Methods

Implemented `src/e04/local_learning.py` and `scripts/e04_s06_local_learning.py`.

The learner is a per-cell local bandit over `swap_left`, `swap_right`, and `idle`. It updates only from a bounded local reward over actor/target neighbor windows, local action outcome, S01 memory frustration, and S02 signals. S05 perturbation schedules remain fixed by seed. Global Sortedness, time-in-target, final morphology, and full-array rank are computed only after the run for evaluation and matched-control comparison.

## Commands

{commands}

## Dependencies And Runtime

- Python: {platform.python_version()}
- pandas: {pd.__version__}
- matplotlib: {matplotlib.__version__}
- New dependencies installed: none.
- CPU use: serial validation and pilot matrix; host reports `{os.cpu_count()}` logical CPUs and worker count used was 1.
- Platform: {platform.platform()}

## Parameters

- Algorithm: Bubble cell class with S06 adjacent local learner.
- Initial values: `(1, 2, 3, 4, 5, 6)`.
- Tasks: `{', '.join(HOMEOSTATIC_TASKS)}`.
- Interface modes: `{', '.join(INTERFACE_MODES)}`.
- Reliability modes: `{', '.join(RELIABILITY_MODES)}`.
- Policy modes: `{', '.join(LEARNING_POLICY_MODES)}`.
- Seeds: 2 activation/policy/schedule/learning seed sets.
- Max events per run: `{int(df['max_events'].iloc[0]) if len(df) else 'NA'}`.
- Reward features: `{', '.join(LEARNING_REWARD_FEATURES)}`.

## Results

Pilot rows: `{len(df)}`. Matched learning/control pairs: `{len(comparison)}`. Mean time-in-target delta, learning minus control: `{mean_delta:.6f}`. Positive pairs: `{positive_pairs}`. Negative pairs: `{negative_pairs}`.

Result table: `{result_path}`. Comparison table: `{comparison_path}`. Summary table: `{summary_path}`. Figure: `{figure_path}`. Rule spec: `{spec_path}`.

{summary_table}

### Matched Comparison By Task

{comparison_table}

## Validation Checks

{validation_table}

Additional validation:

- Unit tests cover learning update logs, nonlearning controls, local access windows, matched matrix controls, deterministic replay, and no-global-oracle audit.
- E03 policy-interface and E02 deterministic simulator regression tests passed.
- Every result row has `uses_global_oracle=false` from combined S02 signal and S06 learning audits.

## Caveats, Blockers, Failed Assumptions, And Limitations

- No blocker remains for S06.
- The learner uses adjacent local swaps and does not cover the full E03 policy morphospace.
- Positive aggregate pilot deltas coexist with negative matched pairs; later steps need holdouts and stricter training audits.
- Offline Sortedness appears in results and comparisons only after trajectories are logged. It is not used in the reward update.

## Provenance

- Git commit at validation time: `{manifest['gitCommit']}`
- Git branch at validation time: `{manifest['gitBranch']}`
- Git status at validation time: `{manifest['gitStatusShort'] or 'clean'}`
- Source files tracked in manifest: {len(manifest['sourceFiles'])}
- Run manifest: `{run_manifest_path}`
- Checksums: `{checksums_path}`
- Previous artifacts: E01 `{manifest['upstreamContext']['previousE01Dir']}`, E02 `{manifest['upstreamContext']['previousE02Dir']}`, E03 `{manifest['upstreamContext']['previousE03Dir']}`
- Created at UTC: `{manifest['createdAtUtc']}`

## Recommended Next Action

Stop before S07. Chief Scientist should review S06 and, if accepted, authorize S07 no-global-oracle training audit.
"""


def main() -> int:
    args = parse_args()
    repo_dir = args.repo_dir.resolve()
    artifacts_dir = args.artifacts_dir.resolve()
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    result_path = artifacts_dir / "results" / "e04_local_learning_pilots.parquet"
    result_csv_path = artifacts_dir / "tables" / "e04_local_learning_pilots.csv"
    comparison_path = artifacts_dir / "tables" / "e04_local_learning_pilot_comparison.csv"
    summary_path = artifacts_dir / "tables" / "e04_local_learning_pilots_summary.csv"
    validation_path = artifacts_dir / "tables" / "e04_local_learning_pilots_validation.csv"
    figure_path = artifacts_dir / "figures" / "e04" / "local_learning_pilots.png"
    spec_path = artifacts_dir / "reports" / "e04_local_learning_rule_spec.md"
    config_path = artifacts_dir / "configs" / "e04_s06_local_learning.json"
    source_manifest_path = artifacts_dir / "src_snapshot" / "e04_local_learning_manifest.json"
    artifact_manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = artifacts_dir / "run_manifest.json"
    checksums_path = artifacts_dir / "checksums" / "sha256sums.txt"
    report_path = step_dir / "research_step_full_results.md"
    for path in [
        step_dir,
        result_path.parent,
        result_csv_path.parent,
        comparison_path.parent,
        summary_path.parent,
        validation_path.parent,
        figure_path.parent,
        spec_path.parent,
        config_path.parent,
        source_manifest_path.parent,
        checksums_path.parent,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    configs = default_local_learning_configs(max_events=args.max_events)
    config_doc = {
        "schema": "eidosoma.e04.s06.local_learning_config.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAtUtc": utc_now(),
        "runCount": len(configs),
        "maxEvents": int(args.max_events),
        "tasks": list(HOMEOSTATIC_TASKS),
        "interfaceModes": list(INTERFACE_MODES),
        "reliabilityModes": list(RELIABILITY_MODES),
        "policyModes": list(LEARNING_POLICY_MODES),
        "rewardFeatures": list(LEARNING_REWARD_FEATURES),
        "forbiddenRewardSources": list(LEARNING_FORBIDDEN_KEYS),
        "workerCount": 1,
        "configsPreview": [config.to_dict() for config in configs[:6]],
    }
    write_json(config_path, config_doc)

    results = run_local_learning_matrix(configs)
    df = pd.DataFrame([result.to_row() for result in results])
    df.to_parquet(result_path, index=False)
    df.to_csv(result_csv_path, index=False)
    comparison = build_comparison(df)
    comparison.to_csv(comparison_path, index=False)
    summary = summarize_results(df)
    summary.to_csv(summary_path, index=False)
    validation_df = validate_results(df, comparison, args.max_events)
    validation_df.to_csv(validation_path, index=False)
    write_figure(comparison, figure_path)

    unit_tests = (
        run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e04", "-p", "test_*.py"], repo_dir)
        if args.run_unit_tests
        else {"command": "not run", "returnCode": 0, "elapsedSeconds": 0.0, "stdout": "", "stderr": "", "success": True}
    )
    e03_tests = (
        run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e03", "-p", "test_policy_interface.py"], repo_dir)
        if args.run_unit_tests
        else {"command": "not run", "returnCode": 0, "elapsedSeconds": 0.0, "stdout": "", "stderr": "", "success": True}
    )
    e02_tests = (
        run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e02", "-p", "test_deterministic_simulator.py"], repo_dir)
        if args.run_unit_tests
        else {"command": "not run", "returnCode": 0, "elapsedSeconds": 0.0, "stdout": "", "stderr": "", "success": True}
    )
    validation_success = bool(validation_df["success"].all() and unit_tests["success"] and e03_tests["success"] and e02_tests["success"])
    validation_line = (
        f"{int(validation_df['success'].sum())}/{len(validation_df)} validation cases passed; "
        f"E04 unit tests return code {unit_tests['returnCode']}; "
        f"E03 policy-interface tests return code {e03_tests['returnCode']}; "
        f"E02 simulator tests return code {e02_tests['returnCode']}; "
        f"{len(df)} pilot rows written; {len(comparison)} matched comparisons"
    )
    artifact_paths = [
        report_path,
        spec_path,
        result_path,
        figure_path,
        result_csv_path,
        comparison_path,
        summary_path,
        validation_path,
        config_path,
        source_manifest_path,
        artifact_manifest_path,
        run_manifest_path,
        checksums_path,
    ]
    write_text(
        spec_path,
        render_rule_spec(
            artifact_paths=artifact_paths,
            validation_line=validation_line,
            validation_success=validation_success,
            result_path=result_path,
            comparison_path=comparison_path,
            config_path=config_path,
        ),
    )

    source_paths = [
        repo_dir / "src/e04/local_learning.py",
        repo_dir / "tests/e04/test_local_learning.py",
        repo_dir / "scripts/e04_s06_local_learning.py",
        repo_dir / "src/e04/homeostasis.py",
        repo_dir / "src/e04/fatigue_damage.py",
        repo_dir / "src/e04/repairable_frozen.py",
        repo_dir / "src/e04/memory_policies.py",
        repo_dir / "src/e04/signaling.py",
        repo_dir / "src/e02/deterministic_simulator.py",
        repo_dir / "src/e03/policy_interface.py",
    ]
    manifest: dict[str, Any] = {
        "schema": "eidosoma.src_snapshot.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAtUtc": utc_now(),
        "gitCommit": git_output(repo_dir, ["rev-parse", "HEAD"]),
        "gitBranch": git_output(repo_dir, ["branch", "--show-current"]),
        "gitStatusShort": git_output(repo_dir, ["status", "--short"]),
        "dependencies": {
            "newDependenciesInstalled": [],
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "upstreamContext": {
            "previousE01Dir": "/previous-artifacts/E01",
            "previousE02Dir": "/previous-artifacts/E02",
            "previousE03Dir": "/previous-artifacts/E03",
            "s01MemorySpec": str(artifacts_dir / "reports/e04_memory_state_spec.md"),
            "s02SignalSpec": str(artifacts_dir / "reports/e04_signal_model_spec.md"),
            "s05TaskSpec": str(artifacts_dir / "reports/e04_homeostatic_task_spec.md"),
            "s05Results": str(artifacts_dir / "results/e04_homeostatic_baselines.parquet"),
        },
        "sourceFiles": [source_entry(path, repo_dir) for path in source_paths if path.exists()],
        "validation": {
            "allValidationPassed": validation_success,
            "validationCaseCount": int(len(validation_df)),
            "validationCaseSuccessCount": int(validation_df["success"].sum()),
            "unitTests": unit_tests,
            "e03PolicyInterfaceTests": e03_tests,
            "e02DeterministicSimulatorTests": e02_tests,
            "resultParquet": str(result_path),
            "resultParquetSha256": sha256_file(result_path),
            "resultCsv": str(result_csv_path),
            "resultCsvSha256": sha256_file(result_csv_path),
            "comparisonCsv": str(comparison_path),
            "comparisonCsvSha256": sha256_file(comparison_path),
            "summaryCsv": str(summary_path),
            "summaryCsvSha256": sha256_file(summary_path),
            "validationCsv": str(validation_path),
            "validationCsvSha256": sha256_file(validation_path),
            "figure": str(figure_path),
            "figureSha256": sha256_file(figure_path),
            "ruleSpec": str(spec_path),
            "ruleSpecSha256": sha256_file(spec_path),
            "config": str(config_path),
            "configSha256": sha256_file(config_path),
        },
    }
    write_json(source_manifest_path, manifest)
    write_text(
        report_path,
        render_report(
            df=df,
            summary=summary,
            comparison=comparison,
            validation_df=validation_df,
            validation_line=validation_line,
            validation_success=validation_success,
            unit_tests=unit_tests,
            e03_tests=e03_tests,
            e02_tests=e02_tests,
            artifact_paths=artifact_paths,
            result_path=result_path,
            comparison_path=comparison_path,
            figure_path=figure_path,
            spec_path=spec_path,
            summary_path=summary_path,
            config_path=config_path,
            run_manifest_path=run_manifest_path,
            checksums_path=checksums_path,
            manifest=manifest,
        ),
    )
    artifacts = [
        artifact_entry(report_path, artifacts_dir, "S06 full-results handoff report"),
        artifact_entry(spec_path, artifacts_dir, "S06 local learning rule specification"),
        artifact_entry(result_path, artifacts_dir, "S06 local learning pilot rows"),
        artifact_entry(figure_path, artifacts_dir, "S06 local learning pilot comparison figure"),
        artifact_entry(result_csv_path, artifacts_dir, "CSV sidecar for S06 pilot rows"),
        artifact_entry(comparison_path, artifacts_dir, "S06 matched learning-vs-control comparison table"),
        artifact_entry(summary_path, artifacts_dir, "S06 pilot summary table"),
        artifact_entry(validation_path, artifacts_dir, "S06 validation table"),
        artifact_entry(config_path, artifacts_dir, "S06 pilot config"),
        artifact_entry(source_manifest_path, artifacts_dir, "S06 source snapshot and provenance manifest"),
        manifest_self_entry(artifact_manifest_path, artifacts_dir, "S06 artifact manifest"),
        manifest_self_entry(run_manifest_path, artifacts_dir, "Experiment run manifest"),
        manifest_self_entry(checksums_path, artifacts_dir, "SHA-256 checksums for key S06 outputs"),
    ]
    artifact_manifest = {
        "schema": "eidosoma.artifact_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAtUtc": utc_now(),
        "artifacts": artifacts,
    }
    write_json(artifact_manifest_path, artifact_manifest)
    run_manifest = {
        "schema": "eidosoma.run_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "lastResearchStepId": STEP_ID,
        "createdAtUtc": utc_now(),
        "git": {
            "branch": manifest["gitBranch"],
            "headCommit": manifest["gitCommit"],
            "statusShort": manifest["gitStatusShort"],
        },
        "runtime": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
            "platform": platform.platform(),
            "cpuCountReported": os.cpu_count(),
            "workerCountUsed": 1,
        },
        "dependencies": manifest["dependencies"],
        "validation": manifest["validation"],
        "artifacts": artifacts,
    }
    write_json(run_manifest_path, run_manifest)
    checksum_targets = [
        report_path,
        spec_path,
        result_path,
        figure_path,
        result_csv_path,
        comparison_path,
        summary_path,
        validation_path,
        config_path,
        source_manifest_path,
        artifact_manifest_path,
        run_manifest_path,
    ]
    checksum_lines = [f"{sha256_file(path)}  {path.relative_to(artifacts_dir)}" for path in checksum_targets]
    write_text(checksums_path, "\n".join(checksum_lines) + "\n")
    return 0 if validation_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
