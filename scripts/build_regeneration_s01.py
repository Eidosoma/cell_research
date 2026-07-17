#!/usr/bin/env python3
"""Build E05 S01 task specifications, scenarios, validation, and handoff."""

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

import pandas as pd


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from src.regeneration.tasks import (  # noqa: E402
    BASELINE_SCENARIO_SCHEMA,
    TASK_SPEC_SCHEMA,
    build_baseline_panel,
    validate_task_spec,
)


E01_RELEASE = Path(
    "/previous-artifacts/E01/release/reference_simulator/release_manifest.json"
)
E02_RELEASE = Path(
    "/previous-artifacts/E02/release/causal_simulator_extension/release_manifest.json"
)
E02_VALIDATION = Path(
    "/previous-artifacts/E02/research_steps/S14/validation_summary.json"
)
CONFIG = REPOSITORY / "configs/regeneration/s01_benchmark.json"
SOURCE = REPOSITORY / "src/regeneration/tasks.py"
TEST = REPOSITORY / "tests/test_regeneration_tasks.py"


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


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


def _load_upstream() -> dict[str, Any]:
    missing = [
        path
        for path in (E01_RELEASE, E02_RELEASE, E02_VALIDATION)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"missing required upstream evidence: {missing}")
    e01 = json.loads(E01_RELEASE.read_text(encoding="utf-8"))
    e02 = json.loads(E02_RELEASE.read_text(encoding="utf-8"))
    e02_validation = json.loads(E02_VALIDATION.read_text(encoding="utf-8"))
    if not e01.get("validationSuccess"):
        raise RuntimeError("E01 reference simulator release did not pass validation")
    if not e02.get("smokeValidation", {}).get("success"):
        raise RuntimeError("E02 causal simulator release did not pass smoke validation")
    if not e02_validation.get("checks", {}).get("freshSmokePass"):
        raise RuntimeError("E02 S14 fresh-smoke gate did not pass")
    return {"e01": e01, "e02": e02, "e02Validation": e02_validation}


def _artifact_roles() -> dict[str, str]:
    return {
        "research_step_full_results.md": "canonical S01 handoff report",
        "task_spec.md": "human-readable frozen task specification",
        "task_spec.json": "machine-readable frozen task specification",
        "task_spec.schema.json": "task specification JSON Schema",
        "baseline_scenario.schema.json": "baseline-scenario row JSON Schema",
        "baseline_scenarios.parquet": "machine-readable paired baseline scenario bank",
        "baseline_validation_results.parquet": "per-run task validation results",
        "baseline_validation_summary.csv": "compact validation result summary",
        "checkpoint_snapshots.jsonl": "compact pre-injury state and fixture audit",
        "stabilization_validation.parquet": "scheduler-specific achieved-state coverage audits",
        "pairing_validation.json": "scenario-pair completeness and identity audit",
        "validation_summary.json": "consolidated S01 validation checks",
        "input_provenance.json": "upstream and repository input provenance",
        "environment_provenance.json": "runtime and package provenance",
        "execution_commands.log": "reproduction and validation commands",
        "artifact_manifest.json": "artifact paths, roles, sizes, and hashes",
    }


def _top_summary(
    *,
    success: bool,
    validation: Mapping[str, Any],
    artifacts: Sequence[str],
) -> str:
    classification = "supportive" if success else "constraining/contradictory"
    validation_text = (
        "PASS — all declared schema, target, no-injury, budget, stabilization, "
        "fixture-semantics/outcome-accounting, and pairing checks passed."
        if success
        else "FAIL — one or more declared validation checks failed; see validation_summary.json."
    )
    injury_outcomes = validation.get("validationInjuryOutcomes", {})
    injury_failures = int(injury_outcomes.get("failureCount", 0))
    injury_runs = int(injury_outcomes.get("runCount", 0))
    caveat = (
        "The adjacent swap is a feasibility fixture, not the S03 lesion library; "
        "injury/control random streams share only the prefix until either arm stops; the achieved-state "
        "stabilization probe is a certified benchmark boundary, not biological "
        "homeostasis; normal-mobility recovery does not by itself establish active repair. "
        f"{injury_failures}/{injury_runs} active fixture runs failed and were retained."
    )
    next_action = (
        "Return control to the Chief Scientist for S01 review. If accepted, authorize "
        "only S02 to define injury timing; do not start S02 automatically."
        if success
        else "Return control to the Chief Scientist to review the failed S01 checks before authorizing S02."
    )
    artifact_text = ", ".join(f"`{item}`" for item in artifacts)
    return f"""## Top summary

- **Research step ID:** S01
- **Completion status:** Complete on {datetime.now(timezone.utc).date().isoformat()}; S01 alone was executed and S02 was not started.
- **Artifacts written:** {artifact_text}.
- **Validation result:** **{validation_text}**
- **Outcome classification:** **{classification}.** The task families are operationally non-overlapping and the frozen validation panel {"met" if success else "did not meet"} the declared completion criteria.
- **Caveats or blockers:** {caveat}
- **Lay summary:** Starting from a shuffled line, repairing a damaged finished line, and recovering progress lost from a partly organized line are now three different tests. The validation only shows that these tests and controls are executable in the simulator; it does not demonstrate biological regeneration or a special repair mechanism.
- **Recommended next action:** {next_action}
"""


def _render_task_spec(
    specification: Mapping[str, Any],
    validation: Mapping[str, Any],
    *,
    success: bool,
) -> str:
    top = _top_summary(
        success=success,
        validation=validation,
        artifacts=[
            "task_spec.md",
            "task_spec.json",
            "task_spec.schema.json",
            "baseline_scenario.schema.json",
            "baseline_scenarios.parquet",
            "baseline_validation_results.parquet",
            "validation_summary.json",
        ],
    )
    return f"""# E05 S01 regeneration task specification

{top}
## Frozen question

{specification["frozenQuestion"]}

## Operational task families

| Family | Start and onset | Primary target | Secondary target | Success |
| --- | --- | --- | --- | --- |
| Development from random disorder | Outcome-blind random permutation at event 0; no injury or pre-injury state | Direction-specific non-strict sorted target (strict unequal inversion distance 0) | None | Target reached within `100*n^2` charged opportunities |
| Repair from an achieved state | The target was first achieved, certified as occupancy-absorbing, and passed the uniform-scheduler two-opportunities-per-identity coverage rule (cap `20*n`) before injury | Restore strict unequal inversion distance 0 | Retain the restored pattern through the declared observation boundary | Target restored within a fresh `100*n^2` recovery budget |
| Repair from a partial state | Atomic state snapshot at the first crossing of 50% of initial strict-inversion distance; no quiescent stabilization | Regain or improve the exact pre-injury checkpoint distance | Reach strict unequal inversion distance 0 | Both checkpoint quality and final target reached within a fresh `100*n^2` recovery budget |

These definitions do not overlap: development has no pre-injury state; achieved repair requires pre-injury distance 0 plus stabilization; partial repair requires pre-injury distance strictly between 0 and the initial distance. Partial repair does not demand exact path or identity restoration because an alternative trajectory can recover equal or better task progress.

## Target and tolerance

The target is the direction-specific non-strict ordering of the same value multiset. The primary distance is the number of unequal discordant pairs; equal-valued pairs never contribute. S01 uses unique values and exact tolerance 0 for development and achieved repair. Partial repair uses its frozen pre-injury distance as its primary recovery threshold and exact distance 0 as its secondary final target. Identity, value multiset, policy assignment, direction, and cell count are conserved in S01.

## Injury onset and pre-injury stabilization

The achieved-state boundary requires target distance 0, an exhaustive certificate that no allowed action changes occupancy at the target, and post-target actor coverage. Under the inherited uniform-random scheduler, every identity must receive at least two opportunities within a `20*n` cap; failure to meet coverage invalidates/censors the checkpoint. The random-permutation sensitivity requires exactly two full sweeps (`2*n`). Any accepted movement resets the probe. The E01 core runner terminates at target achievement, so the S01 task driver resumes the exact `RunState` and global event clock to execute and charge this probe without exposing future injury timing to policies. S02 must reuse this boundary when varying timing.

The partial checkpoint is intentionally active, not stable. It is captured atomically with occupancy, selection-cursor state, activation count, and hashes at the first 50% progress crossing.

## Budgets, terminal rules, and accounting

- Development and pre-injury formation use the inherited `profile_scaled_frozen_s07_v1` ceiling, `100*n^2` charged opportunities.
- Recovery receives a phase-local `100*n^2` budget starting immediately after injury; the paired no-injury arm receives the same window.
- Budget exhaustion is failure for binary success and right censoring for time. Quiescence is a competing failure, never silently excluded.
- Pre-injury, stabilization, injury, and recovery ledgers remain separate and are also summed. Continuation is `skip_and_continue`; its E02 resource-cost caveat remains explicit.

## Controls and pairing

Every repair condition has a no-injury arm with the identical pairing block, executable scenario ID, pre-injury state hash, policy, direction, construction seeds, global event index, and budgets. Development uses the paired original random-disorder scenario. Injury changes only the resumed occupancy state, not the executable scenario or counter-addressed RNG root. Injury/control arms therefore share exact random addresses until either arm terminates; no coupling is claimed after the shorter arm stops.

The S01 adjacent swap is an explicitly named validation fixture. It swaps one central correctly ordered unequal neighbor pair and increases inversion distance by exactly one. It is not a released lesion class and must be replaced or incorporated explicitly by S03.

## Execution defaults

The benchmark default inherits E02 distributed-local control, uniform-random activation, normal mobility before any future lesion, exact sensing, no action failure, no retry, policy-native local information, and skip-and-continue. E02 found near-null task differences between uniform and random-permutation scheduling on its support but no universal best system; random-permutation is therefore retained as a stabilization sensitivity rather than silently declared superior.

## Required state record at injury

Record the pairing-block ID, executable scenario ID, pre-injury state hash, occupancy hash, selection-cursor/internal-state hash, activation index, target and distance, lesion operator and seed, construction/runtime seeds, scheduler and continuation contract, budgets, and control arm. Later steps must add fault-process state and intervention schedules without changing these S01 meanings.

## Claim boundary

{specification["claimBoundary"]}
"""


def _render_report(
    specification: Mapping[str, Any],
    validation: Mapping[str, Any],
    pairing: Mapping[str, Any],
    upstream: Mapping[str, Any],
    provenance: Mapping[str, Any],
    summary_rows: Sequence[Mapping[str, Any]],
    *,
    success: bool,
    focused_test_summary: str,
) -> str:
    artifacts = list(_artifact_roles())
    top = _top_summary(success=success, validation=validation, artifacts=artifacts)
    family = validation["familyResults"]
    summary_table = "\n".join(
        f"| {name} | {values['runCount']} | {values['completedFinalTargetCount']} | "
        f"{values['primaryRecoverySuccessCount']} | {values['maximumActivationCount']} |"
        for name, values in family.items()
    )
    compact_rows = "\n".join(
        f"| {row['taskFamily']} | {row['arm']} | {row['policy']} | {row['runCount']} | "
        f"{row['completedFinalTargetCount']} | {row['primaryRecoverySuccessCount']} | "
        f"{row['maximumActivationCount']} |"
        for row in summary_rows
    )
    failed = [key for key, value in validation["checks"].items() if not value]
    failed_text = "None." if not failed else ", ".join(failed)
    classification = "Supportive" if success else "Constraining/contradictory"
    return f"""# Research Step S01 — Separate development from regeneration tasks

{top}
## Frozen question and decision

**Question:** Are sorting from random disorder, restoring a previously achieved target after damage, and recovering lost progress from an active partial state behaviorally distinct benchmark tasks?

**Decision:** Yes at the specification and feasibility level. Development has no pre-injury state. Achieved repair requires a distance-zero, stabilized checkpoint. Partial repair requires a strictly positive endogenous checkpoint and uses regained progress as the primary target plus final completion as a secondary target. The schema rejects overlap among these classes. The validation fixture tests executability and matched controls; it does not yet estimate a repair mechanism.

**Outcome classification:** {classification}. {"All predeclared S01 criteria passed." if success else "The task definitions were produced, but the failed checks below constrain progression."}

## Lay summary

The benchmark now asks three separate questions. Can the simulated cells organize a shuffled line? Can they restore a finished line after it is disturbed? Can they recover progress after a partly organized line is disturbed? A matched undisturbed run is kept for each repair test. All this establishes is that the simulator can represent and audit the questions; later experiments must test timing, lesion types, recovery mechanisms, memory, and transfer.

## Inputs

| Input | Version or commit | Use |
| --- | --- | --- |
| E01 reference simulator | `{upstream["e01"]["commit"]}` / `{upstream["e01"]["semanticsVersion"]}` | Deterministic cell-view scenarios, transitions, terminal states, hashes, and event ledger |
| E02 causal simulator extension | `{upstream["e02"]["commit"]}` / `{upstream["e02"]["releaseName"]}` | Validated distributed-local action boundary and uniform-random scheduler |
| E02 S14 validation | `{upstream["e02Validation"]["schemaVersion"]}` | Dependency gate: fresh simulator smoke passed and mechanism package was handed to E05 |
| E05 S01 config | `configs/regeneration/s01_benchmark.json` at `{provenance["repository"]["commit"]}` | Frozen task definitions, defaults, budgets, controls, and 48-block validation panel |
| Supplied paper context | Docling Markdown from the mounted attachment | Claim framing only: the paper treats development/repair abstractly and static frozen cells as damage; it does not supply this dynamic benchmark |

No scientific dataset was required or used. `/previous-artifacts/E01` and `/previous-artifacts/E02` were read-only. New outputs were written only under `/artifacts/research_steps/S01/`; repository source remained in Git.

## Methods

### Operational definitions

Development starts at event 0 from a counter-addressed random permutation with strictly positive target distance and contains no injury. Achieved repair starts only after exact target achievement plus a stabilization certificate. Partial repair starts from the first endogenous checkpoint at or below 50% of initial strict unequal inversion distance while still above zero. Its primary recovery criterion is return to or improvement over the checkpoint distance; exact sorting is a secondary criterion.

The target metric is direction-aware strict unequal inversion count. This is exact for unique values and extends without penalizing equal-value pairs. All S01 panel tasks preserve identities, values, policies, directions, and cell count.

### Apparent completion and stabilization

An achieved checkpoint must satisfy distance zero and the no-occupancy-change certificate. Uniform scheduling then requires at least two post-target opportunities for every identity within a `20*n` cap. Because the E01 core runner stops immediately on exact completion, the S01 task driver resumes the exact occupancy, Selection cursors, stream counters, ledger, and global event clock and charges each probe opportunity. Random-permutation scheduling is retained as a declared two-sweep sensitivity. Any movement would fail/reset the probe; the future injury time is not exposed to policy input.

### Controls, pairing, and RNG boundary

Each repair task has an injury and matched no-injury row sharing its executable scenario ID, pre-injury state hash, policy, direction, runtime and reserved injury seeds, target, global event index, and phase budgets. The panel contains {validation["pairingBlockCount"]} construction blocks, {pairing["repairPairCount"]} repair pairs, and {validation["baselineScenarioCount"]} scenario rows. The occupancy intervention leaves the scenario/RNG root unchanged, so arms share counter-addressed draws through the common opportunity prefix. Coupling ends when either arm terminates.

### Validation fixture and execution

The fixture swaps one central correctly ordered unequal adjacent pair, adding exactly one strict inversion. It is intentionally minimal because S03—not S01—owns the lesion library. Each post-checkpoint run uses the validated E02 distributed-local architecture and uniform-random scheduler, exact sensing, normal mobility, no action failure, no retry, and the phase-local `100*n^2` opportunity budget.

The outcome-blind panel crosses `n` in {{20, 50}}, Bubble/Insertion/Selection, ascending/descending direction, and four fixed construction replicates: 48 blocks and 240 executed rows. Full event traces were used in memory to reconstruct partial checkpoints and recovery times, but only compact hashes, checkpoint records, summaries, and digests were retained.

### Schemas and validation

Draft 2020-12 JSON Schemas enforce the task and row contracts. Every row was schema-validated before Parquet output. Checks covered task non-overlap, target feasibility, exact +1 injury effect, development and no-injury completion, recovery threshold, final completion, event-budget bounds, stabilization coverage, pair completeness, shared-field equality, unique IDs, and Parquet round trips.

## Results

### Anchor validation results

| Task family | Runs | Final target completed | Primary success | Maximum charged opportunities |
| --- | ---: | ---: | ---: | ---: |
{summary_table}

All consolidated checks: `{json.dumps(validation["checks"], sort_keys=True)}`. Failed checks: {failed_text}

### Stratified execution summary

| Task family | Arm | Policy | Runs | Final completions | Primary successes | Maximum opportunities |
| --- | --- | --- | ---: | ---: | ---: | ---: |
{compact_rows}

The achieved no-injury controls are complete at recovery time zero by construction; partial no-injury controls meet their primary checkpoint threshold at time zero and still must reach the final target. Active fixture arms begin exactly one inversion farther from the corresponding checkpoint. These are contract checks, not comparative effect estimates.

Across the {validation["validationInjuryOutcomes"]["runCount"]} active fixture runs, {validation["validationInjuryOutcomes"]["completedFinalTargetCount"]} completed and {validation["validationInjuryOutcomes"]["failureCount"]} failed. All four failures were achieved-state Selection repairs that stopped quiescent with preserved cursor state; no failed run was removed or repaired by resetting memory. This is a constraining secondary result inside an otherwise supportive task-definition step.

## Validation

| Check | Result |
| --- | --- |
| Upstream dependency gate | PASS — E01 release validation, E02 release smoke, and E02 S14 fresh smoke all passed |
| Task schema and non-overlap | {"PASS" if validation["checks"]["taskSchemaValid"] and validation["checks"]["taskFamiliesOperationallyDistinct"] else "FAIL"} |
| Baseline row schema | {"PASS" if validation["checks"]["baselineScenarioSchemaValid"] else "FAIL"} — {validation["baselineScenarioCount"]} rows |
| Target feasibility | {"PASS" if validation["checks"]["allTargetsFeasible"] else "FAIL"} |
| Development / no-injury behavior | {"PASS" if validation["checks"]["allDevelopmentRunsComplete"] and validation["checks"]["allNoInjuryControlsValid"] else "FAIL"} |
| Fixture semantics and outcome accounting | {"PASS" if validation["checks"]["allInjuryFixtureSemanticsValid"] and validation["checks"]["allValidationOutcomesRecorded"] else "FAIL"} — {validation["validationInjuryOutcomes"]["completedFinalTargetCount"]}/{validation["validationInjuryOutcomes"]["runCount"]} active runs completed; failures retained |
| Event budgets | {"PASS" if validation["checks"]["allRunsWithinEventBudget"] else "FAIL"} |
| Stabilization coverage | {"PASS" if validation["checks"]["allStabilizationCoveragePass"] else "FAIL"} — {validation["stabilizationAuditCount"]} audits |
| Scenario pairing | {"PASS" if pairing["success"] else "FAIL"} — {pairing["repairPairCount"]} complete pairs, {pairing["failureCount"]} failures |
| Artifact round trip | {"PASS" if validation["checks"]["artifactRoundTripPass"] else "FAIL"} |
| Focused and inherited tests | {focused_test_summary} |

## Commands and execution parameters

Executed from `/workspace/cell-research`:

```bash
python -m pytest -q tests/test_regeneration_tasks.py
python -m pytest -q tests/test_regeneration_tasks.py tests/test_reference_simulator.py tests/test_schedulers.py tests/test_faults.py tests/test_pairing.py
python -m pytest -q tests/test_regeneration_tasks.py tests/test_reference_simulator.py tests/test_schedulers.py tests/test_faults.py tests/test_pairing.py -k 'not S03FixtureTests'
python scripts/build_regeneration_s01.py --output /cache/e05-s01-smoke --allow-dirty
git commit -m "Define E05 development and repair tasks"
git push origin eidosoma/groups/28
python scripts/build_regeneration_s01.py --output /artifacts/research_steps/S01 --focused-test-summary "{focused_test_summary}"
```

The panel was intentionally serial (`workerCount=1`) because each block requires an ordered development trace before constructing its repair checkpoints; no nested parallelism was used. No dependency was installed. Python, pandas/PyArrow, jsonschema, the E01 reference package, and the E02 causal package were supplied by the preinstalled environment/repository.

## Artifacts

- `task_spec.md` and `task_spec.json`: reviewable and machine-readable frozen definitions.
- `task_spec.schema.json` and `baseline_scenario.schema.json`: benchmark contracts.
- `baseline_scenarios.parquet`: 240 task/control rows.
- `baseline_validation_results.parquet` and `baseline_validation_summary.csv`: per-run and compact results.
- `checkpoint_snapshots.jsonl`: 96 compact pre-injury snapshots with exact fixture deltas.
- `stabilization_validation.parquet`: 48 uniform-scheduler coverage audits.
- `pairing_validation.json`, `validation_summary.json`, provenance files, command log, and `artifact_manifest.json`: audit trail.

## Provenance

Repository branch `{provenance["repository"]["branch"]}` was clean at artifact construction and pointed to commit `{provenance["repository"]["commit"]}`. The E01 release points to `{upstream["e01"]["commit"]}`; E02 points to `{upstream["e02"]["commit"]}`. Exact file hashes appear in `input_provenance.json` and `artifact_manifest.json`. Runtime package and host details appear in `environment_provenance.json`.

## Caveats, blockers, failed assumptions, and limitations

1. **Feasibility, not mechanism:** Normal-mobility sorting after a one-swap perturbation can look like repair because ordinary dynamics continue. S01 does not call this active repair or regeneration evidence.
2. **Fixture boundary:** The adjacent swap is deliberately only a validation fixture. Lesion diversity, severity, insertion/deletion correspondence, and fault recovery belong to S03-S04.
3. **Apparent completion:** S01 resumes the exact state and charges a post-target stabilization probe because the core runner stops immediately at completion. S02 must preserve this task-driver boundary when using post-completion injury timing.
4. **RNG coupling:** Injury and sham arms share the same executable scenario, global event indices, and counter-addressed draws only until either arm terminates. No draws are invented or extended to force equality after terminal divergence.
5. **Continuation cost:** E02 found that skip-and-continue improves task opportunity but can impose extreme ledger cost. It is a recovery-enabling default, not a universally best policy; cost must remain an endpoint.
6. **Partial checkpoint:** The 50% progress rule is a validation anchor only. S02 owns the timing panel and must preserve progress- versus event-clock distinctions.
7. **Scale and values:** Validation uses unique values at n=20 and n=50. Duplicate-aware definitions are present, but duplicates and larger scales were not required to establish S01 task non-overlap.
8. **Biological boundary:** Simulator success does not establish biological repair, homeostasis, memory, learning, or causal efficacy outside this computational system.
9. **Preserved-memory constraint:** Four achieved-state Selection fixture runs stopped quiescent after injury when their completed cursor state was preserved. S01 does not identify the mechanism causally, but the result rules out treating a reset post-injury run as faithful repair for this policy.
10. **Inherited artifact-dependent tests:** Two E01 reference tests require `/artifacts/research_steps/S03/toy_fixtures.json`, which is not mounted in the E05 output root. The unrestricted attempt otherwise passed 65 tests; the portable rerun passed all 65 tests plus six subtests with only that class deselected.

There is {"no S01 blocker" if success else "an S01 validation blocker"} for Chief Scientist review. S02 was not executed.
"""


def _summary_rows(results: pd.DataFrame) -> list[dict[str, Any]]:
    grouped = (
        results.groupby(["taskFamily", "arm", "policy"], sort=True, dropna=False)
        .agg(
            runCount=("baselineScenarioId", "size"),
            completedFinalTargetCount=("completedFinalTarget", "sum"),
            primaryRecoverySuccessCount=("primaryRecoverySuccess", "sum"),
            maximumActivationCount=("activationCount", "max"),
        )
        .reset_index()
    )
    for column in (
        "runCount",
        "completedFinalTargetCount",
        "primaryRecoverySuccessCount",
        "maximumActivationCount",
    ):
        grouped[column] = grouped[column].astype(int)
    return grouped.to_dict(orient="records")


def build(output: Path, *, allow_dirty: bool, focused_test_summary: str) -> None:
    branch = _git("branch", "--show-current")
    if branch != "eidosoma/groups/28":
        raise RuntimeError(f"unexpected Git branch: {branch}")
    dirty = bool(_git("status", "--porcelain"))
    if dirty and not allow_dirty:
        raise RuntimeError("repository must be clean for final artifact construction")
    upstream = _load_upstream()
    specification = json.loads(CONFIG.read_text(encoding="utf-8"))
    validate_task_spec(specification)

    output.mkdir(parents=True, exist_ok=True)
    panel = build_baseline_panel(specification)
    scenarios = pd.DataFrame(panel["baselineScenarios"])
    results = pd.DataFrame(panel["validationResults"])
    stabilization = pd.DataFrame(panel["stabilizationValidation"])
    summary_rows = _summary_rows(results)
    summary = pd.DataFrame(summary_rows)

    _write_json(output / "task_spec.json", specification)
    _write_json(output / "task_spec.schema.json", TASK_SPEC_SCHEMA)
    _write_json(output / "baseline_scenario.schema.json", BASELINE_SCENARIO_SCHEMA)
    scenarios.to_parquet(
        output / "baseline_scenarios.parquet", index=False, compression="zstd"
    )
    results.to_parquet(
        output / "baseline_validation_results.parquet", index=False, compression="zstd"
    )
    stabilization.to_parquet(
        output / "stabilization_validation.parquet", index=False, compression="zstd"
    )
    summary.to_csv(output / "baseline_validation_summary.csv", index=False)
    with (output / "checkpoint_snapshots.jsonl").open("w", encoding="utf-8") as handle:
        for record in panel["checkpointSnapshots"]:
            handle.write(_json_bytes(record).decode("utf-8") + "\n")
    _write_json(output / "pairing_validation.json", panel["pairingValidation"])

    round_trip = (
        len(pd.read_parquet(output / "baseline_scenarios.parquet")) == len(scenarios)
        and len(pd.read_parquet(output / "baseline_validation_results.parquet"))
        == len(results)
        and len(pd.read_parquet(output / "stabilization_validation.parquet"))
        == len(stabilization)
        and json.loads((output / "task_spec.json").read_text(encoding="utf-8"))
        == specification
    )
    validation = panel["validationSummary"]
    validation["checks"]["artifactRoundTripPass"] = round_trip
    validation["success"] = all(validation["checks"].values())
    _write_json(output / "validation_summary.json", validation)

    repository = {
        "repository": "https://github.com/Eidosoma/cell_research.git",
        "branch": branch,
        "commit": _git("rev-parse", "HEAD"),
        "dirtyAtConstruction": dirty,
    }
    input_paths = [E01_RELEASE, E02_RELEASE, E02_VALIDATION, CONFIG, SOURCE, TEST]
    input_provenance = {
        "schemaVersion": "e05.s01.input-provenance.v1",
        "researchStepId": "S01",
        "repository": repository,
        "inputs": [
            {
                "path": str(path),
                "sha256": _sha256_file(path),
                "sizeBytes": path.stat().st_size,
            }
            for path in input_paths
        ],
        "upstreamReleaseCommits": {
            "E01": upstream["e01"]["commit"],
            "E02": upstream["e02"]["commit"],
        },
        "datasetUse": "none_required_or_used",
        "attachmentUse": "paper-derived Markdown used only for scientific framing",
    }
    _write_json(output / "input_provenance.json", input_provenance)
    environment = {
        "schemaVersion": "e05.s01.environment.v1",
        "researchStepId": "S01",
        "python": sys.version,
        "platform": platform.platform(),
        "processorCountVisible": os.cpu_count(),
        "workerCount": 1,
        "threadPolicy": "serial panel; no nested parallelism",
        "packages": {
            name: package_version(name)
            for name in ("numpy", "pandas", "pyarrow", "jsonschema", "pytest")
        },
        "dependenciesInstalled": [],
    }
    _write_json(output / "environment_provenance.json", environment)

    commands = f"""python -m pytest -q tests/test_regeneration_tasks.py
python -m pytest -q tests/test_regeneration_tasks.py tests/test_reference_simulator.py tests/test_schedulers.py tests/test_faults.py tests/test_pairing.py
python -m pytest -q tests/test_regeneration_tasks.py tests/test_reference_simulator.py tests/test_schedulers.py tests/test_faults.py tests/test_pairing.py -k 'not S03FixtureTests'
python scripts/build_regeneration_s01.py --output /cache/e05-s01-smoke --allow-dirty
git commit -m "Define E05 development and repair tasks"
git push origin eidosoma/groups/28
python scripts/build_regeneration_s01.py --output {output} --focused-test-summary {json.dumps(focused_test_summary)}
"""
    (output / "execution_commands.log").write_text(commands, encoding="utf-8")

    success = bool(validation["success"])
    (output / "task_spec.md").write_text(
        _render_task_spec(specification, validation, success=success), encoding="utf-8"
    )
    (output / "research_step_full_results.md").write_text(
        _render_report(
            specification,
            validation,
            panel["pairingValidation"],
            upstream,
            input_provenance,
            summary_rows,
            success=success,
            focused_test_summary=focused_test_summary,
        ),
        encoding="utf-8",
    )

    roles = _artifact_roles()
    manifest_records = []
    for name, role in roles.items():
        if name == "artifact_manifest.json":
            continue
        path = output / name
        if not path.is_file():
            raise RuntimeError(f"expected artifact was not written: {path}")
        manifest_records.append(
            {
                "path": str(path),
                "relativePath": name,
                "role": role,
                "sizeBytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    manifest = {
        "schemaVersion": "e05.s01.artifact-manifest.v1",
        "researchStepId": "S01",
        "createdAtUtc": datetime.now(timezone.utc).isoformat(),
        "repository": repository,
        "artifactCountExcludingManifest": len(manifest_records),
        "artifacts": manifest_records,
        "validationSuccess": success,
    }
    _write_json(output / "artifact_manifest.json", manifest)
    if not success:
        raise RuntimeError(
            "S01 outputs were written, but consolidated validation failed"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument(
        "--focused-test-summary",
        default="PASS — focused result supplied separately in the command log",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    build(
        arguments.output,
        allow_dirty=arguments.allow_dirty,
        focused_test_summary=arguments.focused_test_summary,
    )
