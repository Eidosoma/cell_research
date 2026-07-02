#!/usr/bin/env python3
"""Run E04 S14 centralized/global repair baseline comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.e04.centralized_repair import (  # noqa: E402
    CENTRALIZED_COMPARATOR_GROUP,
    LOCAL_COMPARATOR_GROUP,
    classify_s14_outcome,
    run_s14_comparison,
    validate_s14_outputs,
    write_s14_figure,
)


DEFAULT_ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
S14_DIR = DEFAULT_ARTIFACTS_DIR / "research_steps" / "S14"
RESULT_PATH = DEFAULT_ARTIFACTS_DIR / "results" / "e04_centralized_vs_local_repair.parquet"
PAIRED_PATH = DEFAULT_ARTIFACTS_DIR / "results" / "e04_centralized_vs_local_repair_matched_deltas.parquet"
SUMMARY_PATH = DEFAULT_ARTIFACTS_DIR / "results" / "e04_centralized_vs_local_repair_summary.csv"
FIGURE_PATH = DEFAULT_ARTIFACTS_DIR / "figures" / "e04" / "centralized_vs_local.png"
REPORT_PATH = S14_DIR / "research_step_full_results.md"
STATUS_PATH = S14_DIR / "status.json"
MANIFEST_PATH = S14_DIR / "manifest.json"
VALIDATION_PATH = S14_DIR / "validation_checks.json"


def run_command(command: list[str], *, cwd: Path = REPO_ROOT) -> dict[str, Any]:
    """Run a command and capture a compact provenance record."""

    started = datetime.now(UTC).isoformat()
    proc = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    return {
        "command": command,
        "cwd": str(cwd),
        "started_at_utc": started,
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_state() -> dict[str, Any]:
    commands = {
        "head": ["git", "rev-parse", "HEAD"],
        "branch": ["git", "branch", "--show-current"],
        "status_short": ["git", "status", "--short"],
        "remote": ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
    }
    state: dict[str, Any] = {}
    for name, command in commands.items():
        result = run_command(command)
        state[name] = {
            "returncode": result["returncode"],
            "stdout": result["stdout"].strip(),
            "stderr": result["stderr"].strip(),
        }
    return state


def artifact_record(path: Path, kind: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "kind": kind,
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "NA"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def markdown_table(df: pd.DataFrame) -> str:
    """Render a compact GitHub-flavored Markdown table without tabulate."""

    if df.empty:
        return "_No rows._"
    display = df.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: _fmt(value))
        else:
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else str(value))
    headers = [str(column) for column in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display.iterrows():
        cells = [str(row[column]).replace("|", "\\|") for column in display.columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(
    *,
    df: pd.DataFrame,
    paired: pd.DataFrame,
    summary: pd.DataFrame,
    checks: dict[str, Any],
    outcome: str,
    outcome_reason: str,
    commands: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    central = df[df["comparison_group"] == CENTRALIZED_COMPARATOR_GROUP]
    local = df[df["comparison_group"] == LOCAL_COMPARATOR_GROUP]
    mean_delta_repair_score = paired["delta_comparison_repair_score_centralized_minus_local"].mean()
    mean_delta_fitness = paired["delta_fitness_score_centralized_minus_local"].mean()
    mean_delta_energy = paired["delta_energy_total_centralized_minus_local"].mean()
    mean_delta_repair = paired["delta_repair_quality_score_centralized_minus_local"].mean()
    mean_retention = paired["comparison_repair_score_retention_local_vs_centralized"].mean()
    delta_table = (
        paired[
            [
                "split",
                "generalization_axis",
                "delta_fitness_score_centralized_minus_local",
                "delta_comparison_repair_score_centralized_minus_local",
                "delta_repair_quality_score_centralized_minus_local",
                "delta_energy_total_centralized_minus_local",
                "comparison_repair_score_retention_local_vs_centralized",
            ]
        ]
        .groupby(["split", "generalization_axis"])
        .mean(numeric_only=True)
        .reset_index()
    )
    git = repo_state()

    report = f"""# E04 S14 Full Results: Centralized Global-State Repair Baseline

## Chief Scientist Handoff

- Research step ID: S14.
- Completion status: complete.
- Outcome classification: {outcome}.
- Artifacts written: `{REPORT_PATH}`, `{RESULT_PATH}`, `{PAIRED_PATH}`, `{SUMMARY_PATH}`, `{FIGURE_PATH}`, `{VALIDATION_PATH}`, `{MANIFEST_PATH}`, `{STATUS_PATH}`.
- Validation result: {"pass" if checks.get("all_passed") else "fail"}; centralized rows are explicitly global-access baselines and ineligible for local-only claims.
- Main result: centralized-global rows and best local S13/S09-supported neighbor-memory rows were matched by frozen S13 config, seed set, and schedule hash for {len(paired)} paired comparisons.
- Caveats or blockers: centralized rows intentionally access forbidden target/global fields and therefore must not be included in local-only evidence claims.
- Lay summary: a top-down controller with whole-array knowledge was compared against the best local memory policies on the same repair problems to estimate how much performance is left on the table when only local information is allowed.
- Recommended next action: proceed to S15 with centralized/global rows kept as non-local ceiling controls only.

## Frozen Question

Does a centralized controller with global array state and target-order access outperform the best S13/S09 local memory policies on repair quality, energy, recovery speed, and robustness, and can those global-access rows be cleanly separated from local-only claims?

## Inputs

- S13 overfitting/transfer results: `{args.s13_results}`.
- S13 predefined held-out/train configs: `{args.s13_configs}`.
- Local comparator: one best eligible `neighbor_memory` S13/S09-supported row per S13 config, selected by `fitness_score`.
- Centralized comparator: `centralized_global_state_repair_v1`, which reads full array values, all statuses, frozen positions, target sorted order, and the full perturbation schedule.

## Methods

The centralized baseline replays every frozen S13 schedule. At each event it applies the same perturbation schedule, globally repairs all currently frozen cells, then globally reorders the array to target sorted order when needed. Energy is counted as the adjacent-swap-equivalent inversion work. This makes the baseline a deliberate global-access ceiling rather than a local policy.

Rows copied from S13 retain the S07 projected local-only policy contract and are labeled `comparison_group={LOCAL_COMPARATOR_GROUP}`. Centralized rows are labeled `comparison_group={CENTRALIZED_COMPARATOR_GROUP}`, `centralized_baseline=True`, `uses_global_oracle=True`, `global_state_access=True`, and `eligible_for_local_only_claims=False`.

The inherited `fitness_score` keeps the S08 oracle penalty for rows with `uses_global_oracle=True`. Direct S14 central/local comparisons therefore use `comparison_repair_score`, which removes that penalty while preserving the same repair, target-time, final-sortedness, and energy terms.

## Results

- Paired configs: {len(paired)}.
- Mean local comparison repair score: {_fmt(local["comparison_repair_score"].mean())}.
- Mean centralized comparison repair score: {_fmt(central["comparison_repair_score"].mean())}.
- Mean central-minus-local comparison repair score delta: {_fmt(mean_delta_repair_score)}.
- Mean local-vs-central comparison repair score retention: {_fmt(mean_retention)}.
- Mean central-minus-local oracle-penalized fitness delta: {_fmt(mean_delta_fitness)}.
- Mean central-minus-local repair-quality delta: {_fmt(mean_delta_repair)}.
- Mean central-minus-local energy delta: {_fmt(mean_delta_energy)}.
- Outcome reason: {outcome_reason}

### Group Summary

{markdown_table(summary)}

### Matched Delta Summary

{markdown_table(delta_table)}

## Validation Checks

{json.dumps(checks, indent=2, sort_keys=True)}

## Commands

{json.dumps(commands, indent=2, sort_keys=True)}

## Dependencies And Parameters

- Python: `{sys.version.split()[0]}`.
- pandas: `{pd.__version__}`.
- Worker count: serial S14 evaluation; the problem size is 16 matched configs and does not benefit materially from multiprocessing.
- New dependencies installed: none.
- Centralized access fields: full array values, all statuses, global frozen positions, target sorted order, full perturbation schedule.

## Provenance

- Repository path: `{REPO_ROOT}`.
- Branch: `{git["branch"]["stdout"]}`.
- HEAD commit: `{git["head"]["stdout"]}`.
- Upstream: `{git["remote"]["stdout"]}`.
- Git status short at report write:

```text
{git["status_short"]["stdout"] or "(clean)"}
```

## Artifacts And Checksums

{json.dumps(artifacts, indent=2, sort_keys=True)}

## Caveats, Failed Assumptions, And Limitations

The centralized baseline is not biologically local. It is useful only as a global-access ceiling and as a stress test of how close local S13/S09 mechanisms come to a top-down controller. Its repair action globally unfreezes frozen cells and globally reorders the array after scheduled perturbations, so energy comparisons should be read as adjacent-swap-equivalent controller work, not as equivalent local metabolic costs.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--s13-results",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR / "results" / "e04_overfitting_and_transfer.parquet",
    )
    parser.add_argument(
        "--s13-configs",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR
        / "configs"
        / "e04_s13_predefined_heldout_configs.json",
    )
    parser.add_argument("--skip-tests", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    S14_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    commands: list[dict[str, Any]] = []
    if not args.skip_tests:
        test_result = run_command(
            [
                sys.executable,
                "-m",
                "unittest",
                "tests.e04.test_centralized_repair",
            ]
        )
        commands.append(test_result)
        if test_result["returncode"] != 0:
            STATUS_PATH.write_text(
                json.dumps(
                    {
                        "research_step_id": "S14",
                        "status": "blocked",
                        "reason": "unit_tests_failed",
                        "command": test_result,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            return test_result["returncode"]

    commands.append(
        {
            "command": [
                sys.executable,
                "scripts/e04_s14_centralized_repair_baseline.py",
                "--s13-results",
                str(args.s13_results),
                "--s13-configs",
                str(args.s13_configs),
                "--skip-tests" if args.skip_tests else "<tests-enabled>",
            ],
            "cwd": str(REPO_ROOT),
            "started_at_utc": datetime.now(UTC).isoformat(),
            "finished_at_utc": datetime.now(UTC).isoformat(),
            "returncode": 0,
            "stdout": "current process invocation recorded for provenance",
            "stderr": "",
        }
    )

    df, paired, summary = run_s14_comparison(
        s13_results_path=args.s13_results,
        s13_configs_path=args.s13_configs,
    )
    checks = validate_s14_outputs(df, paired)
    outcome, outcome_reason = classify_s14_outcome(paired)
    df.to_parquet(RESULT_PATH, index=False)
    paired.to_parquet(PAIRED_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    write_s14_figure(summary, FIGURE_PATH)

    VALIDATION_PATH.write_text(json.dumps(checks, indent=2, sort_keys=True), encoding="utf-8")
    status = {
        "research_step_id": "S14",
        "status": "complete" if checks.get("all_passed") else "blocked",
        "outcome_classification": outcome,
        "validation_passed": bool(checks.get("all_passed")),
        "result_rows": int(len(df)),
        "paired_comparisons": int(len(paired)),
        "centralized_rows_are_local_claim_eligible": False,
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    STATUS_PATH.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")

    artifacts = [
        artifact_record(RESULT_PATH, "machine_readable_full_results"),
        artifact_record(PAIRED_PATH, "machine_readable_matched_deltas"),
        artifact_record(SUMMARY_PATH, "summary_table"),
        artifact_record(FIGURE_PATH, "figure"),
        artifact_record(VALIDATION_PATH, "validation_checks"),
        artifact_record(STATUS_PATH, "machine_readable_status"),
    ]
    manifest = {
        "research_step_id": "S14",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "inputs": {
            "s13_results": str(args.s13_results),
            "s13_configs": str(args.s13_configs),
            "s13_results_sha256": sha256_file(args.s13_results),
            "s13_configs_sha256": sha256_file(args.s13_configs),
        },
        "artifacts": artifacts,
        "validation": checks,
        "outcome_classification": outcome,
        "repo_state": repo_state(),
    }
    write_report(
        df=df,
        paired=paired,
        summary=summary,
        checks=checks,
        outcome=outcome,
        outcome_reason=outcome_reason,
        commands=commands,
        artifacts=artifacts,
        args=args,
    )
    artifacts.append(artifact_record(REPORT_PATH, "full_results_report"))
    manifest["artifacts"] = artifacts
    manifest["manifest_path"] = str(MANIFEST_PATH)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    if not checks.get("all_passed"):
        return 2
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
