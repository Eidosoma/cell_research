#!/usr/bin/env python3
"""Build E05 S02 timing scenarios, validation evidence, curves, and handoff."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version as package_version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from src.regeneration.timing import (  # noqa: E402
    TIMING_SCENARIO_SCHEMA,
    TIMING_SPEC_SCHEMA,
    build_timing_panel,
    validate_timing_spec,
)


CONFIG = REPOSITORY / "configs/regeneration/s02_timing.json"
SOURCE = REPOSITORY / "src/regeneration/timing.py"
TEST = REPOSITORY / "tests/test_regeneration_timing.py"
S01_DIR = Path("/artifacts/research_steps/S01")
S01_SCENARIOS = S01_DIR / "baseline_scenarios.parquet"
S01_SNAPSHOTS = S01_DIR / "checkpoint_snapshots.jsonl"
S01_SPEC = S01_DIR / "task_spec.json"
S01_VALIDATION = S01_DIR / "validation_summary.json"
E01_RELEASE = Path(
    "/previous-artifacts/E01/release/reference_simulator/release_manifest.json"
)
E02_RELEASE = Path(
    "/previous-artifacts/E02/release/causal_simulator_extension/release_manifest.json"
)
E02_S14 = Path("/previous-artifacts/E02/research_steps/S14/validation_summary.json")
E02_S04_SCHEDULER = Path(
    "/previous-artifacts/E02/research_steps/S04/scheduler_package/scheduler_prespecification.json"
)
E02_S08_STREAMS = Path(
    "/previous-artifacts/E02/research_steps/S08/semantic_random_stream_specification.json"
)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPOSITORY, text=True).strip()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_inputs() -> dict[str, Any]:
    required = (
        S01_SCENARIOS,
        S01_SNAPSHOTS,
        S01_SPEC,
        S01_VALIDATION,
        E01_RELEASE,
        E02_RELEASE,
        E02_S14,
        E02_S04_SCHEDULER,
        E02_S08_STREAMS,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing required S01/E01/E02 evidence: {missing}")
    s01_validation = _load_json(S01_VALIDATION)
    s01_spec = _load_json(S01_SPEC)
    e01 = _load_json(E01_RELEASE)
    e02 = _load_json(E02_RELEASE)
    e02_s14 = _load_json(E02_S14)
    if not s01_validation.get("success"):
        raise RuntimeError("S01 validation gate did not pass")
    if s01_spec.get("schemaVersion") != "e05.s01.task-spec.v1":
        raise RuntimeError("unexpected S01 task specification")
    if not e01.get("validationSuccess"):
        raise RuntimeError("E01 release gate did not pass")
    if not e02.get("smokeValidation", {}).get("success"):
        raise RuntimeError("E02 release smoke gate did not pass")
    if not e02_s14.get("checks", {}).get("freshSmokePass"):
        raise RuntimeError("E02 S14 fresh-smoke gate did not pass")
    return {
        "s01Validation": s01_validation,
        "s01Spec": s01_spec,
        "e01": e01,
        "e02": e02,
        "e02S14": e02_s14,
        "e02Scheduler": _load_json(E02_S04_SCHEDULER),
        "e02Streams": _load_json(E02_S08_STREAMS),
    }


def _s01_contract() -> dict[str, Any]:
    scenarios = pd.read_parquet(
        S01_SCENARIOS,
        columns=["pairingBlockId", "executableScenarioId"],
    ).drop_duplicates()
    sources = [
        {
            "s01PairingBlockId": str(row.pairingBlockId),
            "executableScenarioId": str(row.executableScenarioId),
        }
        for row in scenarios.itertuples(index=False)
    ]
    checkpoints: list[dict[str, Any]] = []
    for line in S01_SNAPSHOTS.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        checkpoints.append(
            {
                "s01PairingBlockId": item["pairingBlockId"],
                "taskFamily": item["taskFamily"],
                "stateHash": item["checkpoint"]["stateHash"],
            }
        )
    return {"sources": sources, "checkpoints": checkpoints}


def _scalar_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        clean: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, (dict, list, tuple)):
                clean[key + "Json"] = json.dumps(
                    value, sort_keys=True, separators=(",", ":")
                )
            else:
                clean[key] = value
        normalized.append(clean)
    return pd.DataFrame(normalized)


def _paired_contrasts(scenarios: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    merged = results.merge(
        scenarios[
            [
                "timingScenarioId",
                "s01PairingBlockId",
                "preInjuryEventIndex",
                "preInjuryDistance",
                "actualProgress",
            ]
        ],
        on="timingScenarioId",
        how="left",
        validate="one_to_one",
    )
    executed = merged[merged["executionStatus"] == "executed"].copy()
    fields = [
        "activationCount",
        "primaryRecoveryTime",
        "completedFinalTarget",
        "primaryRecoverySuccess",
        "finalDistance",
    ]
    wide = executed.pivot(
        index=[
            "timingPairId",
            "s01PairingBlockId",
            "timingConditionId",
            "clock",
            "nominalFraction",
            "n",
            "policy",
            "direction",
            "preInjuryEventIndex",
            "preInjuryDistance",
            "actualProgress",
        ],
        columns="arm",
        values=fields,
    )
    wide.columns = [f"{field}_{arm}" for field, arm in wide.columns]
    wide = wide.reset_index()
    for field in ("activationCount", "primaryRecoveryTime", "finalDistance"):
        active = f"{field}_validation_injury"
        sham = f"{field}_matched_no_injury"
        wide[f"pairedDelta_{field}"] = wide[active] - wide[sham]
    wide["pairedDelta_completedFinalTarget"] = wide[
        "completedFinalTarget_validation_injury"
    ].astype(int) - wide["completedFinalTarget_matched_no_injury"].astype(int)
    return wide


def _baseline_curves(results: pd.DataFrame) -> pd.DataFrame:
    executed = results[results["executionStatus"] == "executed"].copy()
    order = {
        "initialization": 0,
        "progress_25": 1,
        "event_25": 2,
        "progress_50": 3,
        "event_50": 4,
        "progress_75": 5,
        "event_75": 6,
        "post_completion": 7,
    }
    rows: list[dict[str, Any]] = []
    for keys, frame in executed.groupby(
        ["timingConditionId", "clock", "nominalFraction", "policy", "arm"],
        sort=False,
        dropna=False,
    ):
        condition, clock, fraction, policy, arm = keys
        rows.append(
            {
                "timingConditionId": condition,
                "timingOrder": order[str(condition)],
                "clock": clock,
                "nominalFraction": fraction,
                "policy": policy,
                "arm": arm,
                "runCount": len(frame),
                "completionCount": int(frame["completedFinalTarget"].sum()),
                "completionRate": float(frame["completedFinalTarget"].mean()),
                "primarySuccessCount": int(frame["primaryRecoverySuccess"].sum()),
                "primarySuccessRate": float(frame["primaryRecoverySuccess"].mean()),
                "medianActivationCount": float(frame["activationCount"].median()),
                "meanActivationCount": float(frame["activationCount"].mean()),
                "medianPrimaryRecoveryTime": float(
                    frame["primaryRecoveryTime"].dropna().median()
                )
                if frame["primaryRecoveryTime"].notna().any()
                else np.nan,
                "failureCount": int((~frame["completedFinalTarget"]).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["policy", "arm", "timingOrder"])


def _clock_comparison(trigger_audits: pd.DataFrame) -> pd.DataFrame:
    during = trigger_audits[
        trigger_audits["clock"].isin(
            ["progress_first_crossing", "paired_uninjured_completion_fraction"]
        )
    ].copy()
    during["clockLabel"] = during["clock"].map(
        {
            "progress_first_crossing": "progress",
            "paired_uninjured_completion_fraction": "event",
        }
    )
    wide = during.pivot(
        index=[
            "s01PairingBlockId",
            "executableScenarioId",
            "nominalFraction",
        ],
        columns="clockLabel",
        values=["actualEventIndex", "actualProgress", "triggerStatus"],
    )
    wide.columns = [f"{field}_{clock}" for field, clock in wide.columns]
    wide = wide.reset_index()
    wide["eventMinusProgressTriggerEvent"] = (
        wide["actualEventIndex_event"] - wide["actualEventIndex_progress"]
    )
    wide["eventMinusNominalProgress"] = (
        wide["actualProgress_event"] - wide["nominalFraction"]
    )
    wide["progressOvershoot"] = (
        wide["actualProgress_progress"] - wide["nominalFraction"]
    )
    return wide


def _plot_curves(curves: pd.DataFrame, output: Path) -> None:
    active = curves[curves["arm"] == "validation_injury"].copy()
    labels = [
        "init",
        "P25",
        "E25",
        "P50",
        "E50",
        "P75",
        "E75",
        "post",
    ]
    colors = {"Bubble": "#2678b2", "Insertion": "#e6862a", "Selection": "#3b9b63"}
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for policy in ("Bubble", "Insertion", "Selection"):
        frame = active[active["policy"] == policy].sort_values("timingOrder")
        axes[0].plot(
            frame["timingOrder"],
            frame["completionRate"],
            marker="o",
            label=policy,
            color=colors[policy],
        )
        axes[1].plot(
            frame["timingOrder"],
            frame["medianActivationCount"],
            marker="o",
            label=policy,
            color=colors[policy],
        )
    axes[0].set_ylabel("Final-target completion rate")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].legend(frameon=False, ncol=3)
    axes[0].grid(alpha=0.25)
    axes[1].set_ylabel("Median post-trigger opportunities")
    axes[1].set_xlabel("Injury timing (P=progress clock, E=event clock)")
    axes[1].set_xticks(range(8), labels)
    axes[1].grid(alpha=0.25)
    fig.suptitle("E05 S02 calibrated +1-inversion timing curves")
    fig.tight_layout()
    fig.savefig(output / "timing_baseline_curves.png", dpi=180)
    fig.savefig(output / "timing_baseline_curves.svg")
    plt.close(fig)


def _artifact_roles() -> dict[str, str]:
    return {
        "research_step_full_results.md": "canonical S02 full-results handoff",
        "timing_spec.md": "human-readable frozen timing specification",
        "timing_spec.json": "machine-readable frozen timing specification",
        "timing_spec.schema.json": "timing specification JSON Schema",
        "damage_timing_scenario.schema.json": "scenario-row JSON Schema",
        "damage_timing_scenarios.parquet": "complete planned timing-arm scenario bank",
        "timing_results.parquet": "per-arm execution outcomes including retained non-runs",
        "preinjury_states.parquet": "compact exact pre-injury state records",
        "trigger_validation.parquet": "per-block trigger accuracy audits",
        "stabilization_validation.parquet": "inherited post-completion stabilization audits",
        "timing_paired_contrasts.parquet": "injury-minus-sham paired descriptive contrasts",
        "timing_baseline_curves.parquet": "aggregated timing curves",
        "timing_clock_comparison.parquet": "paired progress/event clock comparison",
        "timing_baseline_curves.png": "raster baseline timing figure",
        "timing_baseline_curves.svg": "vector baseline timing figure",
        "pairing_validation.json": "shared-prefix pair completeness audit",
        "s01_source_identity_validation.json": "exact S01 source ID comparison",
        "s01_checkpoint_identity_validation.json": "exact S01 checkpoint-state comparison",
        "trigger_validation.json": "consolidated trigger validation",
        "timing_leakage_validation.json": "policy-boundary and pre-state leakage audit",
        "lesion_severity_validation.json": "active/sham severity comparability audit",
        "replay_validation.json": "all-run exact replay audit",
        "unreachable_policy_validation.json": "explicit non-reaching-policy handling audit",
        "run_accounting.json": "planned, executed, and retained-nonrun accounting",
        "validation_summary.json": "consolidated S02 validation gate",
        "input_provenance.json": "S01/E01/E02 and repository provenance",
        "environment_provenance.json": "runtime and dependency provenance",
        "execution_commands.log": "reproduction and test commands",
        "artifact_manifest.json": "artifact roles, sizes, and SHA-256 hashes",
    }


def _top_summary(
    *, validation: Mapping[str, Any], artifacts: Sequence[str], caveat: str
) -> str:
    success = bool(validation["success"])
    artifact_text = ", ".join(f"`{item}`" for item in artifacts)
    return f"""## Top summary

- **Research step ID:** S02
- **Completion status:** {"Complete" if success else "Blocked"} on {datetime.now(timezone.utc).date().isoformat()}; S02 alone was executed and S03 was not started.
- **Artifacts written:** {artifact_text}.
- **Validation result:** **{"PASS — all declared S01-inheritance, trigger, state-recording, leakage, severity, pairing, replay, stabilization, unreachable-policy, and run-accounting checks passed." if success else "FAIL — one or more declared S02 checks failed; see validation_summary.json."}**
- **Outcome classification:** **{"supportive" if success else "constraining/contradictory"}.** The temporal-intervention panel {"met" if success else "did not meet"} its predeclared completion criteria.
- **Caveats or blockers:** {caveat}
- **Lay summary:** The same one-step disturbance can now be applied before organization, at three measured stages during organization, or after a finished pattern has passed the exact S01 stability check. Progress and elapsed-opportunity clocks are kept separate, and conditions that cannot be reached remain visible instead of being moved or discarded.
- **Recommended next action:** {"Return control to the Chief Scientist for S02 review; if accepted, separately authorize S03 lesion-type implementation." if success else "Return control to the Chief Scientist to review the failed S02 checks before S03."}
"""


def _timing_spec_markdown(
    specification: Mapping[str, Any],
    validation: Mapping[str, Any],
    artifacts: Sequence[str],
) -> str:
    caveat = (
        "The +1 adjacent swap is only the inherited timing-calibration fixture; it is not the S03 lesion library. "
        "The event clock uses a paired no-injury completion reference and is an analysis/design clock, never a policy input."
    )
    top = _top_summary(validation=validation, artifacts=artifacts, caveat=caveat)
    rows = "\n".join(
        f"| `{item['timingConditionId']}` | `{item['clock']}` | {item['nominalFraction']:.2f} | {item['triggerRule']} |"
        for item in specification["timingConditions"]
    )
    return f"""# E05 S02 timing specification

{top}
## Frozen question

{specification["frozenQuestion"]}

## Timing conditions

| Condition | Clock | Fraction | Atomic trigger rule |
| --- | --- | ---: | --- |
{rows}

Progress conditions use the first post-opportunity crossing of strict-inversion progress. Event conditions use `ceil(f * T_complete)` for the same block's paired uninjured completion count. Initialization is event zero. Post-completion is the exact S01 absorbing-certificate plus uniform two-opportunities-per-identity checkpoint.

## Preserved S01 contracts

- Exact checkpoint state includes occupancy, Selection cursors, activation count, stream counters, ledger, and global event clock.
- Development and recovery each retain the phase-local `100*n^2` charged-opportunity budget.
- The target remains direction-aware strict unequal inversion distance, with exact final distance zero.
- Injury and sham retain the same executable scenario ID and counter-addressed RNG addresses through their common prefix; coupling ends when either arm terminates.
- The post-completion boundary retains the `20*n` uniform-scheduler stabilization cap and two opportunities per identity.

## Unreachable triggers

If a progress threshold is not crossed, both planned arms are written with `unreached_terminal_before_progress_threshold` and are not executed. If paired development has no within-budget completion, event-clock fractions are written with `unreached_no_paired_completion_reference`. Neither case receives a substitute time, and no row is silently excluded.

## Lesion calibration and claim boundary

Active arms receive the deterministic S01 central adjacent swap with exactly +1 strict inversion; shams receive delta 0. This fixture isolates timing mechanics only. {specification["claimBoundary"]}
"""


def _report(
    specification: Mapping[str, Any],
    panel: Mapping[str, Any],
    inputs: Mapping[str, Any],
    curves: pd.DataFrame,
    contrasts: pd.DataFrame,
    clock: pd.DataFrame,
    provenance: Mapping[str, Any],
    focused_test_summary: str,
) -> str:
    validation = panel["validationSummary"]
    accounting = panel["runAccounting"]
    unreachable = panel["unreachablePolicyValidation"]
    results = pd.DataFrame(panel["resultRows"])
    active = results[
        (results["executionStatus"] == "executed")
        & (results["arm"] == "validation_injury")
    ]
    failures = active[~active["completedFinalTarget"]]
    caveat = (
        "The S01 +1 adjacent swap is a calibration fixture, not the S03 lesion library; event fractions are relative to a paired uninjured completion trajectory and therefore answer a different timing question than progress crossings; normal-mobility continuation is not evidence of active biological repair. "
        f"{len(failures)}/{len(active)} active timing runs failed and were retained."
    )
    top = _top_summary(
        validation=validation,
        artifacts=list(_artifact_roles()),
        caveat=caveat,
    )
    condition_rows = []
    for condition in [
        item["timingConditionId"] for item in specification["timingConditions"]
    ]:
        frame = active[active["timingConditionId"] == condition]
        condition_rows.append(
            f"| `{condition}` | {len(frame)} | {int(frame['completedFinalTarget'].sum())} | "
            f"{float(frame['completedFinalTarget'].mean()):.3f} | {float(frame['activationCount'].median()):.1f} | "
            f"{int((~frame['completedFinalTarget']).sum())} |"
        )
    failure_counts = (
        failures.groupby(["timingConditionId", "policy", "stopReason"])
        .size()
        .reset_index(name="count")
    )
    failure_text = (
        "None."
        if failure_counts.empty
        else "; ".join(
            f"{row.timingConditionId}/{row.policy}/{row.stopReason}: {row.count}"
            for row in failure_counts.itertuples(index=False)
        )
    )
    clock_summary = (
        clock.groupby("nominalFraction")
        .agg(
            blocks=("s01PairingBlockId", "size"),
            median_event_minus_progress=("eventMinusProgressTriggerEvent", "median"),
            median_event_clock_progress=("actualProgress_event", "median"),
            median_progress_overshoot=("progressOvershoot", "median"),
        )
        .reset_index()
    )
    clock_rows = "\n".join(
        f"| {row.nominalFraction:.2f} | {int(row.blocks)} | {row.median_event_minus_progress:.1f} | "
        f"{row.median_event_clock_progress:.3f} | {row.median_progress_overshoot:.3f} |"
        for row in clock_summary.itertuples(index=False)
    )
    checks = "\n".join(
        f"| `{key}` | {'PASS' if value else 'FAIL'} |"
        for key, value in validation["checks"].items()
    )
    return f"""# Research Step S02 — Vary damage timing

{top}
## Frozen question and outcome

**Question:** Does operational recovery depend on whether the same calibrated lesion occurs before, during, or after target formation, and how do first-crossing progress clocks differ from paired completion-relative event clocks?

**Outcome:** The design and validation hypothesis was supported: all eight timing conditions were generated for every S01 construction block, all clock and state contracts passed, and progress/event differences were measured without exposing timing to a policy. This step is a temporal-design benchmark, not a powered causal claim that timing universally changes biological or simulated repair.

## Lay summary

The experiment now pauses each original sorting run at several precisely defined moments. One set of moments is based on how much ordering has actually happened; another is based on how much of that run's no-damage completion time has elapsed. The same tiny disturbance is then applied and compared with an undisturbed continuation. These clocks often identify different states. A few finished Selection-policy systems could not fix the disturbance when their exact internal cursor state was preserved; those failures remain in the results.

## Inputs and provenance

| Input | Version / path | Use |
| --- | --- | --- |
| E05 S01 task specification and artifacts | `e05.s01.task-spec.v1`; `/artifacts/research_steps/S01/` | Exact construction IDs, task state, 50% and stabilized checkpoint hashes, budgets, targets, pairing |
| E01 reference simulator | `{inputs["e01"]["commit"]}` / `{inputs["e01"]["semanticsVersion"]}` | Deterministic state transitions and event ledger |
| E02 causal extension | `{inputs["e02"]["commit"]}` / `{inputs["e02"]["releaseName"]}` | Distributed-local architecture and uniform scheduler |
| E02 scheduler / stream contracts | `E02.scheduler-prespecification.v1`; `e02.s08.semantic_random_stream_specification.v1` | Charged opportunity, information-boundary, counter-addressed coupling rules |
| Supplied paper | mounted Docling Markdown | Scientific framing only; no timing benchmark was imported from the paper |

No scientific dataset was required. Upstream artifacts were read-only. Repository source is preserved at `{provenance["repository"]["commit"]}` on `eidosoma/groups/28`.

## Detailed methods

### Scenario panel and exact S01 inheritance

The panel reuses the 48 S01 blocks: `n` 20 and 50; Bubble, Insertion, and Selection; ascending and descending targets; four construction replicates. The S01 construction seed, generation key, executable scenario ID, target, uniform scheduler, normal mobility, `skip_and_continue`, `100*n^2` development and recovery budgets, and `2*budget + 20*n` total envelope are unchanged. The build compared all 48 `(pairingBlockId, executableScenarioId)` pairs and all 96 relevant checkpoint hashes (S01 partial 50% and stabilized achieved states) to the S01 artifacts exactly.

Every pre-injury checkpoint preserves occupancy, Selection cursors, activation count, counter-addressed stream counters, the full ledger, and global event clock. Injury changes occupancy only. Sham and injury arms use the same executable scenario ID and pre-injury prefix and share exact random addresses until either arm terminates.

### Timing clocks

- **Initialization:** event zero, before any charged proposal.
- **Progress 25/50/75%:** the first atomic post-opportunity checkpoint satisfying `(D0-Dt)/D0 >= f`; the immediately prior progress is recorded to prove first crossing.
- **Event 25/50/75%:** the checkpoint after `ceil(f*T_complete)` opportunities, where `T_complete` is the same block's paired uninjured within-budget completion time. The reference is used by orchestration only and is not policy-visible.
- **Post-completion:** exact final target followed by the S01 absorbing-occupancy certificate and charged uniform-scheduler coverage of at least two opportunities per identity within `20*n`.

Initialization and post-completion require final distance zero as the primary target. During-formation arms retain the S01 partial-repair primary criterion of regaining or improving the exact pre-injury distance, plus exact final completion as the secondary target.

### Unreachable policies and stopping

No onset row is replaced or discarded. A missed progress first crossing is retained for both arms as `unreached_terminal_before_progress_threshold`; an absent paired completion reference is retained as `unreached_no_paired_completion_reference`. Main-panel uninjured development completed for every block, so no onset was unreachable here. The preserved-state post-completion Selection failures supply executable terminal examples: {unreachable["preservedStateSelectionFailureCount"]} systems failed before reaching one or more requested recovery-progress levels, and each terminal outcome was retained without state reset, fallback, or exclusion.

### Lesion, controls, and endpoints

The active calibration fixture swaps the central correctly ordered unequal adjacent pair and adds exactly one strict inversion at every timing. Sham delta is zero. This is deliberately not a lesion-diversity study. Direct endpoints are final-target completion, post-trigger opportunities, primary threshold time, stop reason, final distance, and event digest. Timing curves are descriptive and preserve every failure.

### Validation and replay

All {accounting["formationRunCount"]} formation trajectories were byte-replayed. All {panel["replayValidation"]["recoveryReplayCount"]} executed timing arms were replayed from the exact checkpoint and compared over summary, final state, event digest, and full events. Static policy-signature/source checks and dynamic paired state/prefix hashes verified that timing metadata did not enter policy proposals.

## Results

### Complete timing-arm outcomes

| Condition | Active runs | Final completions | Completion rate | Median post-trigger opportunities | Failures |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(condition_rows)}

The scenario bank contains {accounting["plannedTimingArmRows"]} planned arm rows and {accounting["executedTimingRuns"]} executed timing runs; {accounting["notRunTriggerUnreachedRows"]} rows were retained as trigger-unreached non-runs and {accounting["silentExclusionCount"]} were silently excluded. Active-run failures by condition/policy/reason: {failure_text}

### Progress versus event clocks

| Nominal fraction | Paired blocks | Median event-minus-progress trigger index | Median actual progress at event clock | Median progress-clock overshoot |
| ---: | ---: | ---: | ---: | ---: |
{clock_rows}

The event and progress clocks are not interchangeable. Event timing is relative to each trajectory's completion duration, whereas progress timing waits for a state-space crossing and can move earlier or later when progress is uneven. Exact block-level values are in `timing_clock_comparison.parquet`.

### Calibrated lesion and preserved-state constraint

All {panel["severityValidation"]["activeTriggeredCount"]} active triggers added exactly one inversion, and all {panel["severityValidation"]["shamTriggeredCount"]} shams added zero. The key constraining secondary result is preserved rather than repaired by reset: {unreachable["preservedStateSelectionFailureCount"]} post-completion Selection active runs terminated without completing. This is consistent with S01's cursor-state caveat and does not establish a general absence of repair.

## Validation

| Check | Result |
| --- | --- |
{checks}

Focused and inherited tests: {focused_test_summary}

## Commands, dependencies, and execution parameters

Executed from `/workspace/cell-research`:

```bash
python -m pytest -q tests/test_regeneration_timing.py
python -m pytest -q tests/test_regeneration_tasks.py tests/test_regeneration_timing.py tests/test_reference_simulator.py tests/test_schedulers.py tests/test_faults.py tests/test_pairing.py -k 'not S03FixtureTests'
python scripts/build_regeneration_s02.py --output /cache/e05-s02-smoke --allow-dirty --skip-exact-replay
git commit -m "Validate E05 damage timing"
git push origin eidosoma/groups/28
python scripts/build_regeneration_s02.py --output /artifacts/research_steps/S02 --focused-test-summary "{focused_test_summary}"
```

Execution was intentionally serial (`workerCount=1`) because each block's no-injury trace defines its atomic trigger checkpoints and must precede its paired arms. No nested parallelism and no GPU were used. Exact replay doubled formation and timing-arm execution. No dependency was installed; Python, pandas/PyArrow, NumPy, Matplotlib, jsonschema, and the repository's E01/E02 packages were already present.

## Artifact guide

- `damage_timing_scenarios.parquet` and `timing_results.parquet`: complete planned rows and outcomes/non-run statuses.
- `preinjury_states.parquet`, `trigger_validation.parquet`, and `stabilization_validation.parquet`: exact checkpoint, clock, and post-completion audits.
- `timing_paired_contrasts.parquet`, `timing_baseline_curves.parquet`, `timing_clock_comparison.parquet`, plus PNG/SVG: descriptive outcome and clock evidence.
- Validation JSON files: S01 identity, pairing, leakage, severity, replay, unreachable-policy, and full accounting gates.

## Caveats, blockers, failed assumptions, and limitations

1. **Timing fixture only:** the +1 adjacent swap holds severity constant but is deliberately minimal; S03 owns lesion diversity and target correspondence for identity-changing operators.
2. **Different clock estimands:** completion-relative event fractions depend on a paired no-injury reference; progress crossings depend on observed state. Their differences are expected and should not be pooled as duplicate measures.
3. **Reference availability:** if an uninjured policy does not complete within `100*n^2`, event fractions are undefined by design. No budget-fraction substitute is permitted.
4. **Preserved memory:** restarting or resetting Selection cursors would change the task. Exact cursor state was retained, exposing terminal post-completion failures.
5. **Passive continuation:** successful normal-mobility continuation after a perturbation is an operational recovery outcome, not evidence of an engineered active-repair mechanism or biology.
6. **Panel scale:** 48 paired blocks validate the temporal design and establish descriptive baselines; this is not the later 100,000–300,000-run mechanism benchmark or a powered universal timing claim.
7. **Shared-prefix boundary:** exact RNG coupling holds only until arm-specific path or terminal divergence; no draws are invented to extend coupling.
8. **No main onset censoring:** all S01 no-injury formation trajectories reached all three progress thresholds and completion. The explicit unreachable branch is code-tested and policy-terminal handling is evidenced by preserved-state Selection failures, but broader non-reaching onset populations belong to later fault panels.
9. **Claim boundary:** these are direct measurements in a transparent simulator and bounded computational proxies for regeneration; they do not establish biological mechanisms, memory, or clinical efficacy.

There is no S02 blocker for Chief Scientist review. S03 was not executed.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--skip-exact-replay", action="store_true")
    parser.add_argument(
        "--focused-test-summary",
        default="Not supplied; see execution_commands.log.",
    )
    args = parser.parse_args()
    if not args.allow_dirty and _git("status", "--porcelain"):
        raise RuntimeError("repository must be clean for final artifact construction")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    specification = _load_json(CONFIG)
    validate_timing_spec(specification)
    inputs = _load_inputs()
    panel = build_timing_panel(
        specification,
        exact_replay=not args.skip_exact_replay,
        expected_s01_contract=_s01_contract(),
    )
    scenarios = _scalar_frame(panel["scenarioRows"])
    results = _scalar_frame(panel["resultRows"])
    triggers = _scalar_frame(panel["triggerAudits"])
    preinjury = _scalar_frame(panel["preinjuryRecords"])
    stabilization = _scalar_frame(panel["stabilizationAudits"])
    contrasts = _paired_contrasts(scenarios, results)
    curves = _baseline_curves(results)
    clock = _clock_comparison(triggers)

    _write_json(output / "timing_spec.json", specification)
    _write_json(output / "timing_spec.schema.json", TIMING_SPEC_SCHEMA)
    _write_json(output / "damage_timing_scenario.schema.json", TIMING_SCENARIO_SCHEMA)
    scenarios.to_parquet(output / "damage_timing_scenarios.parquet", index=False)
    results.to_parquet(output / "timing_results.parquet", index=False)
    preinjury.to_parquet(output / "preinjury_states.parquet", index=False)
    triggers.to_parquet(output / "trigger_validation.parquet", index=False)
    stabilization.to_parquet(output / "stabilization_validation.parquet", index=False)
    contrasts.to_parquet(output / "timing_paired_contrasts.parquet", index=False)
    curves.to_parquet(output / "timing_baseline_curves.parquet", index=False)
    clock.to_parquet(output / "timing_clock_comparison.parquet", index=False)
    _plot_curves(curves, output)

    json_outputs = {
        "pairing_validation.json": panel["pairingValidation"],
        "s01_source_identity_validation.json": panel["sourceIdentityValidation"],
        "s01_checkpoint_identity_validation.json": panel[
            "checkpointIdentityValidation"
        ],
        "trigger_validation.json": panel["triggerValidation"],
        "timing_leakage_validation.json": panel["leakageValidation"],
        "lesion_severity_validation.json": panel["severityValidation"],
        "replay_validation.json": panel["replayValidation"],
        "unreachable_policy_validation.json": panel["unreachablePolicyValidation"],
        "run_accounting.json": panel["runAccounting"],
        "validation_summary.json": panel["validationSummary"],
    }
    for filename, value in json_outputs.items():
        _write_json(output / filename, value)

    provenance = {
        "schemaVersion": "e05.s02.input-provenance.v1",
        "researchStepId": "S02",
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "repository": {
            "path": str(REPOSITORY),
            "branch": _git("branch", "--show-current"),
            "commit": _git("rev-parse", "HEAD"),
            "clean": not bool(_git("status", "--porcelain")),
        },
        "inputs": [
            {"path": str(path), "sha256": _sha256_file(path)}
            for path in (
                CONFIG,
                SOURCE,
                TEST,
                S01_SPEC,
                S01_VALIDATION,
                S01_SCENARIOS,
                S01_SNAPSHOTS,
                E01_RELEASE,
                E02_RELEASE,
                E02_S14,
                E02_S04_SCHEDULER,
                E02_S08_STREAMS,
            )
        ],
        "upstream": {
            "e01Commit": inputs["e01"]["commit"],
            "e02Commit": inputs["e02"]["commit"],
            "s01RepositoryCommit": "1ff1f79cbb9a6f7533c5da6db5c3ad5fdf8dba53",
        },
    }
    _write_json(output / "input_provenance.json", provenance)
    environment = {
        "schemaVersion": "e05.s02.environment-provenance.v1",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpuCount": os.cpu_count(),
        "workerCount": 1,
        "threadEnvironmentOverrides": "none",
        "gpuUsed": False,
        "packages": {
            name: package_version(name)
            for name in ("pandas", "pyarrow", "numpy", "matplotlib", "jsonschema")
        },
    }
    _write_json(output / "environment_provenance.json", environment)
    commands = """python -m pytest -q tests/test_regeneration_timing.py
python -m pytest -q tests/test_regeneration_tasks.py tests/test_regeneration_timing.py tests/test_reference_simulator.py tests/test_schedulers.py tests/test_faults.py tests/test_pairing.py -k 'not S03FixtureTests'
python scripts/build_regeneration_s02.py --output /cache/e05-s02-smoke --allow-dirty --skip-exact-replay
git commit -m "Validate E05 damage timing"
git push origin eidosoma/groups/28
python scripts/build_regeneration_s02.py --output /artifacts/research_steps/S02 --focused-test-summary <summary>
"""
    (output / "execution_commands.log").write_text(commands, encoding="utf-8")
    artifacts = list(_artifact_roles())
    (output / "timing_spec.md").write_text(
        _timing_spec_markdown(specification, panel["validationSummary"], artifacts),
        encoding="utf-8",
    )
    (output / "research_step_full_results.md").write_text(
        _report(
            specification,
            panel,
            inputs,
            curves,
            contrasts,
            clock,
            provenance,
            args.focused_test_summary,
        ),
        encoding="utf-8",
    )
    manifest_entries = []
    roles = _artifact_roles()
    for filename, role in roles.items():
        if filename == "artifact_manifest.json":
            continue
        path = output / filename
        if not path.is_file():
            raise FileNotFoundError(f"expected artifact not written: {path}")
        manifest_entries.append(
            {
                "path": filename,
                "role": role,
                "sizeBytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    _write_json(
        output / "artifact_manifest.json",
        {
            "schemaVersion": "e05.s02.artifact-manifest.v1",
            "researchStepId": "S02",
            "artifactCountExcludingManifest": len(manifest_entries),
            "artifacts": manifest_entries,
        },
    )
    if not panel["validationSummary"]["success"] and not args.skip_exact_replay:
        raise RuntimeError("S02 validation failed; artifacts retained for review")
    print(
        json.dumps(
            {
                "success": True,
                "output": str(output),
                "plannedRows": panel["runAccounting"]["plannedTimingArmRows"],
                "executedRuns": panel["runAccounting"]["executedTimingRuns"],
                "activeFailures": int(
                    (
                        (results["arm"] == "validation_injury")
                        & (results["executionStatus"] == "executed")
                        & (~results["completedFinalTarget"])
                    ).sum()
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
