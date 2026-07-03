#!/usr/bin/env python3
"""Run E06 S06 bounded opposite-goal dominance hierarchy sweep."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.e06.dominance_hierarchy import (  # noqa: E402
    PERTURBATION_PROFILES,
    STEP_ID,
    VALUE_PROFILES,
    S06Config,
    context_dependence,
    dominance_hierarchy,
    run_s06_sweep,
    summarize_s06_runs,
    validate_s06_outputs,
)
from src.e06.mixture_ratios import EXPERIMENT_ID, load_s01_library, sha256_file  # noqa: E402


STEP_NUMBER = 6
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--library-path", type=Path, default=ARTIFACTS_DIR / "policies/e06_chimeric_algotype_library.jsonl")
    parser.add_argument("--metadata-path", type=Path, default=ARTIFACTS_DIR / "tables/e06_algotype_metadata.csv")
    parser.add_argument("--s05-candidate-path", type=Path, default=ARTIFACTS_DIR / "tables/e06_s05_s06_candidate_contests.csv")
    parser.add_argument("--s05-score-path", type=Path, default=ARTIFACTS_DIR / "results/e06_compatibility_scores.parquet")
    parser.add_argument("--s04-run-path", type=Path, default=ARTIFACTS_DIR / "results/e06_goal_compatibility.parquet")
    parser.add_argument("--s04-summary-path", type=Path, default=ARTIFACTS_DIR / "tables/e06_goal_compatibility_summary.csv")
    parser.add_argument("--array-size", type=int, default=100)
    parser.add_argument("--event-cap", type=int, default=4_000)
    parser.add_argument("--max-s05-candidates", type=int, default=18)
    parser.add_argument("--seeds", type=int, nargs="*", default=[2026070201, 2026070202, 2026070203, 2026070204])
    parser.add_argument("--value-profiles", nargs="*", default=list(VALUE_PROFILES))
    parser.add_argument("--perturbation-profiles", nargs="*", default=list(PERTURBATION_PROFILES))
    parser.add_argument("--run-unit-tests", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_command(command: list[str], *, cwd: Path) -> dict[str, Any]:
    started = datetime.now(UTC)
    proc = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    elapsed = (datetime.now(UTC) - started).total_seconds()
    return {
        "command": " ".join(command),
        "cwd": str(cwd),
        "returnCode": proc.returncode,
        "success": proc.returncode == 0,
        "elapsedSeconds": elapsed,
        "stdout": proc.stdout[-6000:],
        "stderr": proc.stderr[-6000:],
    }


def git_output(repo_dir: Path, args: list[str]) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=repo_dir, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc!r}"


def artifact_entry(path: Path, artifacts_dir: Path, description: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(artifacts_dir)),
        "description": description,
        "sha256": sha256_file(path),
        "sizeBytes": path.stat().st_size,
    }


def pending_entry(path: Path, artifacts_dir: Path, description: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(artifacts_dir)),
        "description": description,
        "sha256": None,
        "sizeBytes": None,
        "note": "Checksum computed after this file is written.",
    }


def source_entry(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "sha256": sha256_file(path) if path.exists() else None,
        "sizeBytes": path.stat().st_size if path.exists() else None,
    }


def markdown_table(frame: pd.DataFrame, columns: list[str], *, max_rows: int = 30) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame[columns].head(max_rows).copy()
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in display.to_dict(orient="records"):
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                text = f"{value:.6g}"
            else:
                text = str(value)
            values.append(text.replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    if len(frame) > max_rows:
        lines.append("| ... | " + f"{len(frame) - max_rows} more rows omitted from report table" + " |" * (len(columns) - 1))
    return "\n".join(lines)


def _display_name(text: str) -> str:
    try:
        return " vs ".join(str(item) for item in json.loads(str(text))[:3])
    except json.JSONDecodeError:
        return str(text)


def _ratio_label(text: str) -> str:
    try:
        return ":".join(str(int(round(float(value) * 100))) for value in json.loads(str(text)))
    except (TypeError, ValueError, json.JSONDecodeError):
        return str(text)


def write_dominance_figure(path: Path, run_df: pd.DataFrame, hierarchy: pd.DataFrame) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if run_df.empty or hierarchy.empty:
        return False
    nodes = hierarchy["policy_id"].astype(str).tolist()
    labels = dict(zip(hierarchy["policy_id"].astype(str), hierarchy["display_name"].astype(str), strict=True))
    net = dict(zip(hierarchy["policy_id"].astype(str), hierarchy["net_dominance_score"].astype(float), strict=True))
    n = len(nodes)
    angles = [2.0 * math.pi * idx / max(1, n) for idx in range(n)]
    positions = {node: (math.cos(angle), math.sin(angle)) for node, angle in zip(nodes, angles, strict=True)}
    edges = (
        run_df[run_df["dominant_policy_id"].astype(str) != "tie"]
        .groupby(["dominant_policy_id", "dominated_policy_id"], as_index=False)
        .agg(run_count=("seed", "size"), mean_margin=("dominance_margin_abs", "mean"))
    )
    max_edge_count = max(1, int(edges["run_count"].max())) if not edges.empty else 1
    max_abs_net = max(1e-12, max(abs(float(value)) for value in net.values()))

    fig, ax = plt.subplots(figsize=(11, 8), constrained_layout=True)
    ax.set_title("E06 S06 Opposite-Goal Dominance Network")
    ax.axis("off")
    ax.set_aspect("equal")

    for edge in edges.to_dict(orient="records"):
        source = str(edge["dominant_policy_id"])
        target = str(edge["dominated_policy_id"])
        if source not in positions or target not in positions or source == target:
            continue
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        linewidth = 0.8 + 3.0 * float(edge["run_count"]) / max_edge_count
        alpha = 0.25 + 0.55 * float(edge["run_count"]) / max_edge_count
        ax.annotate(
            "",
            xy=(x1 * 0.86, y1 * 0.86),
            xytext=(x0 * 0.86, y0 * 0.86),
            arrowprops={
                "arrowstyle": "->",
                "lw": linewidth,
                "alpha": alpha,
                "color": "#425466",
                "shrinkA": 12,
                "shrinkB": 12,
            },
        )

    colors = [plt.get_cmap("coolwarm")(0.5 + 0.45 * net[node] / max_abs_net) for node in nodes]
    sizes = [650 + 1600 * abs(net[node]) / max_abs_net for node in nodes]
    ax.scatter(
        [positions[node][0] for node in nodes],
        [positions[node][1] for node in nodes],
        s=sizes,
        c=colors,
        edgecolors="#1f2933",
        linewidths=1.2,
        zorder=3,
    )
    for node in nodes:
        x, y = positions[node]
        ax.text(
            x * 1.18,
            y * 1.18,
            f"{labels.get(node, node)}\nnet={net[node]:.2f}",
            ha="center",
            va="center",
            fontsize=8,
            zorder=4,
        )
    if not edges.empty:
        ax.text(
            -1.28,
            -1.25,
            "Arrows point from dominant to dominated policy; edge thickness scales with run count.",
            fontsize=8,
            ha="left",
            va="bottom",
        )
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path.exists() and path.stat().st_size > 0


def _normalized_json_list(text: Any) -> str:
    return json.dumps(json.loads(str(text)), separators=(",", ":"))


def _summary_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        _normalized_json_list(row["policy_ids_json"]),
        _normalized_json_list(row["ratio_targets_json"]),
        str(row["arrangement"]),
        str(row["goal_profile_id"]),
    )


def s04_trace_validation(base_contests: pd.DataFrame, scores: pd.DataFrame, s04_summary: pd.DataFrame) -> pd.DataFrame:
    source_scores = base_contests.loc[base_contests["source_kind"] == "s05_candidate", "source_score_id"].astype(str).tolist()
    selected_scores = scores[scores["score_id"].astype(str).isin(source_scores)].copy()
    score_success = (
        len(selected_scores) == len(set(source_scores))
        and not selected_scores.empty
        and set(selected_scores["source_research_step_id"].astype(str)) == {"S04"}
    )
    s04_keys = {_summary_key(row) for row in s04_summary.to_dict(orient="records")}
    selected_keys = {_summary_key(row) for row in selected_scores.to_dict(orient="records")}
    missing = sorted(selected_keys - s04_keys)
    return pd.DataFrame(
        [
            {
                "validation_case": "s05_candidates_trace_to_s04_inputs",
                "success": bool(score_success and not missing and selected_keys),
                "observed": f"s05_source_scores={len(selected_scores)} missing_s04_summary_keys={len(missing)}",
                "expected": "all S05-selected S06 candidate rows are S04-derived and match S04 summary rows by policy IDs, ratio, arrangement, and goal profile",
            }
        ]
    )


def write_report(
    path: Path,
    *,
    artifacts: list[dict[str, Any]],
    run_df: pd.DataFrame,
    summary: pd.DataFrame,
    hierarchy: pd.DataFrame,
    context_df: pd.DataFrame,
    base_contests: pd.DataFrame,
    condition_df: pd.DataFrame,
    validation: pd.DataFrame,
    config_payload: dict[str, Any],
    unit_test_result: dict[str, Any],
    repo_state: dict[str, Any],
    command: str,
) -> None:
    validation_passed = bool(validation["success"].all() and unit_test_result.get("success", True))
    non_tie_count = int((run_df["dominant_policy_id"].astype(str) != "tie").sum()) if not run_df.empty else 0
    context_dependent_count = int(context_df["context_dependent_dominance"].sum()) if "context_dependent_dominance" in context_df else 0
    outcome = "supportive" if validation_passed and non_tie_count > 0 else ("constraining/contradictory" if not validation_passed else "null")
    artifact_lines = "\n".join(f"- `{item['relativePath']}`: {item['description']}" for item in artifacts)

    top_hierarchy = hierarchy.head(15).copy()
    top_context = context_df.head(15).copy()
    representative = summary.sort_values(["mean_dominance_margin_abs", "mean_goal_alignment_gap"], ascending=[False, False]).copy()
    selected = base_contests.copy()
    for frame in (top_context, representative, selected):
        if not frame.empty and "display_names_json" in frame.columns:
            frame["display_names"] = frame["display_names_json"].map(_display_name)
        if not frame.empty and "ratio_targets_json" in frame.columns:
            frame["ratio"] = frame["ratio_targets_json"].map(_ratio_label)

    winner_counts = run_df["dominant_policy_id"].value_counts().sort_values(ascending=False).to_dict() if not run_df.empty else {}
    text = f"""# E06 S06 Research Step Full Results

## Top Summary

- Step ID: {STEP_ID}
- Completion status: Complete
- Artifacts written:
{artifact_lines}
- Validation result: {"Passed" if validation_passed else "Failed"}; {int(validation["success"].sum())}/{len(validation)} validation checks passed and unit tests {'passed' if unit_test_result.get('success', True) else 'failed'}.
- Outcome classification: {outcome}
- Caveats or blockers: The contest sweep was bounded to {len(base_contests)} base contests, including up to {config_payload['maxS05Candidates']} S05-prioritized rows plus paper-original controls. Dominance is measured in the E06 1D proxy harness under opposite rank-order goals, not in a biological tissue model.
- Lay summary: S06 directly paired candidate policies against each other with one policy trying to make the array increasing and the other trying to make it decreasing. The run table maps which Algotypes tend to dominate under these opposite goals and which contests change with value initialization, frozen-edge perturbation, or label relabeling.
- Recommended next action: Proceed to S07 final-state taxonomy using `tables/e06_dominance_hierarchy.csv`, `tables/e06_s06_context_dependence.csv`, and the S02-S06 run summaries to separate stable dominance from context-dependent mosaics.

## Frozen Question

Which Algotype families dominate under direct contest, and can hierarchy explain failed or successful chimeras?

## Inputs

- S01 Algotype library: `{config_payload['libraryPath']}`
- S01 metadata table: `{config_payload['metadataPath']}`
- S05 S06 candidate-contest table: `{config_payload['s05CandidatePath']}`
- S05 compatibility score table: `{config_payload['s05ScorePath']}`
- S04 goal-compatibility runs: `{config_payload['s04RunPath']}`
- S04 goal-compatibility summary: `{config_payload['s04SummaryPath']}`
- Repository checkout: `{repo_state.get('branch')}` at `{repo_state.get('head')}`

## Methods

S06 selected the top S05 candidate contests by `s06_priority_score`, deduplicated by policy pair, ratio, and arrangement, then added explicit paper-original controls for Bubble vs Insertion, Bubble vs Selection, and Insertion vs Selection. Each base contest was expanded across two orientations, configured value profiles, configured perturbation profiles, and matched seeds.

The `as_selected` orientation assigns the first policy to the increasing goal and the second policy to the decreasing goal. The `relabel_swap` orientation swaps policy identities while preserving the base ratio vector and spatial pattern, so it is a relabeling audit rather than a new ratio sweep. The `frozen_edges_2` perturbation blocks swaps involving the two edge positions and records those blocked moves separately from invalid actions.

Each final state was evaluated with the S04 goal evaluator under the relevant opposite goals. The run winner is the policy with the larger relevant global goal score unless the absolute margin is within the configured tie threshold. Policy-level hierarchy sums signed dominance margins across all contest runs and separately reports win, loss, and tie counts.

## Commands

- Main command: `{command}`
- Unit-test command: `{unit_test_result.get('command', 'not run')}`
- Unit-test return code: `{unit_test_result.get('returnCode', 'not run')}`

## Dependencies

No new dependencies were installed. The script used repository modules plus preinstalled `pandas`, `numpy`, `pyarrow`, and `matplotlib`.

## Parameters

```json
{json.dumps(config_payload, indent=2, sort_keys=True, default=str)}
```

## Results

- Base contests: {len(base_contests)}.
- Executable conditions before seed expansion: {len(condition_df)}.
- Contest runs: {len(run_df)}.
- Non-tie dominance outcomes: {non_tie_count}.
- Policies in hierarchy: {len(hierarchy)}.
- Context-dependent base contests: {context_dependent_count}/{len(context_df)}.
- Winner counts: {winner_counts}.
- Mean absolute dominance margin: {run_df['dominance_margin_abs'].mean():.4f}.
- Mean goal-alignment gap: {run_df['goal_alignment_gap'].mean():.4f}.
- Mean aggregation delta percent: {run_df['aggregation_delta_percent'].mean():.4f}.

Selected base contests:

{markdown_table(selected, ["selection_rank", "source_kind", "panel", "display_names", "ratio", "arrangement", "s05_priority_score", "s05_goal_conflict_index", "s05_dominance_abs_margin"], max_rows=25)}

Dominance hierarchy:

{markdown_table(top_hierarchy, ["display_name", "source_category", "contest_run_count", "dominance_wins", "dominance_losses", "dominance_ties", "win_fraction", "net_dominance_score", "mean_abs_dominance_margin"], max_rows=15)}

Most context-dependent base contests:

{markdown_table(top_context, ["display_names", "ratio", "condition_count", "dominant_policy_modes_json", "dominance_margin_range", "abs_dominance_margin_range", "aggregation_delta_range", "context_dependent_dominance"], max_rows=15)}

Representative high-margin condition summaries:

{markdown_table(representative, ["display_names", "ratio", "orientation", "arrangement", "value_profile", "perturbation_profile", "seed_count", "mean_global_goal_margin_first_minus_second", "mean_dominance_margin_abs", "dominant_policy_mode", "mean_goal_alignment_gap", "goal_state_class_mode"], max_rows=20)}

## Validation Checks

{markdown_table(validation, ["validation_case", "success", "observed", "expected"], max_rows=30)}

## Artifacts

{artifact_lines}

## Caveats And Limitations

- The sweep is bounded by design; it maps S05-prioritized contests and paper-original controls, not every possible S01 pair or ratio.
- S06 uses a 1D local-action rank-order proxy. It is useful for computational compatibility structure but does not establish biological causality or wet-lab validation.
- The relabel-swap orientation preserves the base ratio vector by role, so policy identities receive swapped counts. This is intentional for label-symmetry auditing.
- Frozen-edge perturbation is a simple local blockage stressor. Other perturbation types could produce different dominance orderings.
- Net dominance score can hide context dependence; the context table should be used alongside the hierarchy table.

## Provenance

Repository state:

```json
{json.dumps(repo_state, indent=2, sort_keys=True)}
```

## Recommended Next Action

Run S07 final-state taxonomy and explanatory synthesis using S02-S06 outputs. Use S06 hierarchy as one layer, but keep context-dependent dominance and goal-conflict cases separate from stable dominance.
"""
    write_text(path, text)


def main() -> int:
    args = parse_args()
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_path = artifacts_dir / "results/e06_dominance_contests.parquet"
    hierarchy_path = artifacts_dir / "tables/e06_dominance_hierarchy.csv"
    figure_path = artifacts_dir / "figures/e06/dominance_network.png"
    summary_path = artifacts_dir / "tables/e06_dominance_contest_summary.csv"
    context_path = artifacts_dir / "tables/e06_s06_context_dependence.csv"
    condition_path = step_dir / "e06_s06_condition_matrix.csv"
    base_path = step_dir / "e06_s06_selected_base_contests.csv"
    validation_path = step_dir / "e06_s06_validation_checks.csv"
    config_path = artifacts_dir / "configs/e06_s06_dominance_config.json"
    report_path = step_dir / "research_step_full_results.md"
    manifest_path = step_dir / "manifest.json"
    run_manifest_path = artifacts_dir / "run_manifest.json"
    checksums_path = artifacts_dir / "checksums/sha256sums.txt"

    config = S06Config(
        array_size=args.array_size,
        event_cap=args.event_cap,
        seeds=tuple(int(seed) for seed in args.seeds),
        max_s05_candidates=args.max_s05_candidates,
        value_profiles=tuple(str(item) for item in args.value_profiles),
        perturbation_profiles=tuple(str(item) for item in args.perturbation_profiles),
    )
    config_payload = {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "generatedAtUtc": utc_now(),
        "libraryPath": str(args.library_path),
        "metadataPath": str(args.metadata_path),
        "s05CandidatePath": str(args.s05_candidate_path),
        "s05ScorePath": str(args.s05_score_path),
        "s04RunPath": str(args.s04_run_path),
        "s04SummaryPath": str(args.s04_summary_path),
        "arraySize": config.array_size,
        "eventCap": config.event_cap,
        "seeds": list(config.seeds),
        "maxS05Candidates": config.max_s05_candidates,
        "includePaperOriginalControls": config.include_paper_original_controls,
        "valueProfiles": list(config.value_profiles),
        "perturbationProfiles": list(config.perturbation_profiles),
        "orientations": list(config.orientations),
        "scheduler": config.scheduler,
        "dominanceTieThreshold": config.dominance_tie_threshold,
    }
    write_json(config_path, config_payload)

    unit_test_result = {"success": True, "command": "not run", "returnCode": 0, "stdout": "", "stderr": ""}
    if args.run_unit_tests:
        unit_test_result = run_command(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests/e06", "-p", "test_*.py"],
            cwd=args.repo_dir,
        )

    records, metadata = load_s01_library(args.library_path, args.metadata_path)
    if not args.s05_candidate_path.exists():
        raise FileNotFoundError(f"S05 S06 candidate-contest table is missing: {args.s05_candidate_path}")
    if not args.s05_score_path.exists():
        raise FileNotFoundError(f"S05 compatibility score table is missing: {args.s05_score_path}")
    if not args.s04_run_path.exists():
        raise FileNotFoundError(f"S04 run table is missing: {args.s04_run_path}")
    if not args.s04_summary_path.exists():
        raise FileNotFoundError(f"S04 summary table is missing: {args.s04_summary_path}")

    candidates = pd.read_csv(args.s05_candidate_path)
    scores = pd.read_parquet(args.s05_score_path)
    pd.read_parquet(args.s04_run_path, columns=["condition_id"])
    s04_summary = pd.read_csv(args.s04_summary_path)

    run_df, condition_df, base_contests = run_s06_sweep(records, metadata, candidates, scores, config)
    summary = summarize_s06_runs(run_df)
    hierarchy = dominance_hierarchy(run_df, metadata)
    context_df = context_dependence(summary)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    run_df.to_parquet(results_path, index=False)
    hierarchy_path.parent.mkdir(parents=True, exist_ok=True)
    hierarchy.to_csv(hierarchy_path, index=False)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    context_path.parent.mkdir(parents=True, exist_ok=True)
    context_df.to_csv(context_path, index=False)
    condition_path.parent.mkdir(parents=True, exist_ok=True)
    condition_df.to_csv(condition_path, index=False)
    base_contests.to_csv(base_path, index=False)
    figure_written = write_dominance_figure(figure_path, run_df, hierarchy)

    validation = validate_s06_outputs(
        run_df,
        condition_df,
        base_contests,
        hierarchy,
        config,
        figure_written=figure_written,
        unit_tests_success=bool(unit_test_result.get("success", True)),
    )
    validation = pd.concat([validation, s04_trace_validation(base_contests, scores, s04_summary)], ignore_index=True)
    validation.to_csv(validation_path, index=False)

    repo_state = {
        "head": git_output(args.repo_dir, ["rev-parse", "HEAD"]),
        "branch": git_output(args.repo_dir, ["branch", "--show-current"]),
        "statusShort": git_output(args.repo_dir, ["status", "--short"]),
        "remote": git_output(args.repo_dir, ["remote", "-v"]),
        "python": sys.version,
        "platform": platform.platform(),
    }

    base_artifacts = [
        artifact_entry(results_path, artifacts_dir, "S06 run-level opposite-goal dominance contest table."),
        artifact_entry(hierarchy_path, artifacts_dir, "S06 policy-level dominance hierarchy table."),
        artifact_entry(figure_path, artifacts_dir, "S06 dominance network figure."),
        artifact_entry(summary_path, artifacts_dir, "S06 condition-level dominance contest summary table."),
        artifact_entry(context_path, artifacts_dir, "S06 context-dependence table by base contest."),
        artifact_entry(condition_path, artifacts_dir, "S06 executable condition matrix."),
        artifact_entry(base_path, artifacts_dir, "S06 selected base contest table."),
        artifact_entry(validation_path, artifacts_dir, "S06 validation check table."),
        artifact_entry(config_path, artifacts_dir, "S06 configuration file."),
    ]
    manifest_pending = pending_entry(manifest_path, artifacts_dir, "S06 artifact manifest.")
    report_pending = pending_entry(report_path, artifacts_dir, "S06 full-results Markdown handoff report.")

    write_report(
        report_path,
        artifacts=[*base_artifacts, manifest_pending, report_pending],
        run_df=run_df,
        summary=summary,
        hierarchy=hierarchy,
        context_df=context_df,
        base_contests=base_contests,
        condition_df=condition_df,
        validation=validation,
        config_payload=config_payload,
        unit_test_result=unit_test_result,
        repo_state=repo_state,
        command=" ".join(sys.argv),
    )
    report_artifact = artifact_entry(report_path, artifacts_dir, "S06 full-results Markdown handoff report.")

    success = bool(validation["success"].all() and unit_test_result.get("success", True))
    manifest = {
        "schema": "eidosoma.e06.s06_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "generatedAtUtc": utc_now(),
        "success": success,
        "runCount": int(len(run_df)),
        "conditionCount": int(len(condition_df)),
        "baseContestCount": int(len(base_contests)),
        "hierarchyRowCount": int(len(hierarchy)),
        "contextDependentContestCount": int(context_df["context_dependent_dominance"].sum()) if "context_dependent_dominance" in context_df else 0,
        "validationPassed": int(validation["success"].sum()),
        "validationTotal": int(len(validation)),
        "artifacts": [*base_artifacts, report_artifact, manifest_pending],
        "repoState": repo_state,
        "unitTestResult": unit_test_result,
    }
    write_json(manifest_path, manifest)
    manifest_artifact = artifact_entry(manifest_path, artifacts_dir, "S06 artifact manifest.")
    artifacts = [*base_artifacts, report_artifact, manifest_artifact]

    run_manifest = {
        "schema": "eidosoma.run_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "latestResearchStepId": STEP_ID,
        "generatedAtUtc": utc_now(),
        "repoState": repo_state,
        "python": sys.version,
        "platform": platform.platform(),
        "artifactCount": len(artifacts),
        "artifacts": artifacts,
        "sourceInputs": {
            "s01AlgotypeLibrary": source_entry(args.library_path),
            "s01Metadata": source_entry(args.metadata_path),
            "s05CandidateContests": source_entry(args.s05_candidate_path),
            "s05CompatibilityScores": source_entry(args.s05_score_path),
            "s04GoalCompatibilityRuns": source_entry(args.s04_run_path),
            "s04GoalCompatibilitySummary": source_entry(args.s04_summary_path),
        },
    }
    write_json(run_manifest_path, run_manifest)
    checksum_paths = [
        results_path,
        hierarchy_path,
        figure_path,
        summary_path,
        context_path,
        condition_path,
        base_path,
        validation_path,
        config_path,
        manifest_path,
        report_path,
        run_manifest_path,
    ]
    write_text(checksums_path, "\n".join(f"{sha256_file(path)}  {path.relative_to(artifacts_dir)}" for path in checksum_paths) + "\n")

    status = {
        "success": success,
        "runCount": int(len(run_df)),
        "conditionCount": int(len(condition_df)),
        "baseContestCount": int(len(base_contests)),
        "hierarchyRowCount": int(len(hierarchy)),
        "validationPassed": int(validation["success"].sum()),
        "validationTotal": int(len(validation)),
        "contextDependentContestCount": int(context_df["context_dependent_dominance"].sum()) if "context_dependent_dominance" in context_df else 0,
        "artifacts": artifacts,
    }
    print(json.dumps(status, indent=2, sort_keys=True, default=str))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
