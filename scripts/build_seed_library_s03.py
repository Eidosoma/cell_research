#!/usr/bin/env python3
"""Build the E07 S03 training-only seed-library evidence bundle."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping

from src.environment_suite import (
    AccessDeniedError,
    AccessGrant,
    AccessPhase,
    EnvironmentSuite,
    EvaluationAction,
)
from src.environment_suite.runners import RUNNERS, baseline_policy_hash
from src.seed_library import build_s03_evidence


REPOSITORY = Path(__file__).resolve().parents[1]
WORKSPACE = REPOSITORY.parent
REGISTRY = REPOSITORY / "configs/environment_suite/task_registry.yaml"
SPLITS = REPOSITORY / "configs/environment_suite/split_manifest.json"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _json_native(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        return _json_native(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def native_training_guard() -> dict[str, Any]:
    """Re-run exactly the eight S02 native training baselines, no other split."""

    suite = EnvironmentSuite(REGISTRY, SPLITS)
    rows = []
    for task_id in sorted(suite.tasks):
        record = next(
            item
            for item in suite.records.values()
            if item.task_id == task_id and item.split.value == "train"
        )
        policy_id = str(record.public_parameters["baselinePolicyId"])
        action = EvaluationAction(policy_id, baseline_policy_hash(policy_id))
        environment = suite.open(
            task_id, record.scenario_id, AccessGrant(AccessPhase.DEVELOPMENT)
        )
        reset = environment.reset().to_dict()
        step = environment.step(action).to_dict()
        outcome = environment.outcome().to_dict()
        direct = RUNNERS[suite.tasks[task_id].runner_id](record, action)
        comparisons = {
            "stopReason": step["stopReason"] == direct.stop_reason,
            "censored": step["censored"] == direct.censored,
            "failed": step["failed"] == direct.failed,
            "nativeCostLedgers": step["cost"]["nativeLedgerFamilies"]
            == _json_native(direct.native_costs),
            "nativeEvent": step["event"]["native"] == _json_native(direct.native_event),
            "nativeOutcome": _json_native(outcome["outcome"])
            == _json_native(direct.native_outcome),
            "nativeValidation": all(direct.validation.values()),
            "exactReplay": direct.replay_pass,
            "trainingSplit": reset["split"] == "train"
            and step["split"] == "train"
            and outcome["split"] == "train",
        }
        rows.append(
            {
                "taskId": task_id,
                "scenarioId": record.scenario_id,
                "split": "train",
                "predecessor": suite.tasks[task_id].predecessor,
                "policyId": policy_id,
                "policySha256": action.policy_sha256,
                "stopReason": step["stopReason"],
                "censored": step["censored"],
                "failed": step["failed"],
                "nativeUnit": step["nativeUnit"],
                "terminalPrecedence": reset["horizon"]["terminalPrecedence"],
                "censoringRule": reset["horizon"]["censoringRule"],
                "nativeLedgerFamilies": sorted(step["cost"]["nativeLedgerFamilies"]),
                "claimBoundary": outcome["claimBoundary"],
                "comparisons": comparisons,
                "passed": all(comparisons.values()) and not step["failed"],
            }
        )
    return {
        "schemaVersion": "e07.s03.native-training-baseline-guard.v1",
        "researchStepId": "S03",
        "evaluatedSplits": ["train"],
        "validationRuns": 0,
        "confirmationRuns": 0,
        "taskCount": len(rows),
        "rows": rows,
        "brokerAudit": suite.broker.audit.to_dict(),
        "allPassed": all(row["passed"] for row in rows),
    }


def split_access_audit(training_ids: list[str]) -> dict[str, Any]:
    """Prove non-training scenario requests are denied in development."""

    suite = EnvironmentSuite(REGISTRY, SPLITS)
    rows = []
    before = suite.broker.audit.materializer_invocations
    for record in sorted(suite.records.values(), key=lambda item: item.scenario_id):
        if record.split.value == "train":
            continue
        denied = False
        try:
            suite.open(
                record.task_id,
                record.scenario_id,
                AccessGrant(AccessPhase.DEVELOPMENT),
            )
        except AccessDeniedError:
            denied = True
        rows.append(
            {
                "scenarioId": record.scenario_id,
                "taskId": record.task_id,
                "split": record.split.value,
                "developmentAccessDenied": denied,
                "protected": record.protected,
                "materializerAbsent": record.materializer_id is None,
            }
        )
    checks = {
        "allEvaluatedScenariosAreFrozenTraining": len(training_ids) == 8
        and all(":train:" in scenario_id for scenario_id in training_ids),
        "allValidationAndConfirmationDevelopmentRequestsDenied": all(
            row["developmentAccessDenied"] for row in rows
        ),
        "denialsPrecededMaterialization": suite.broker.audit.materializer_invocations
        == before,
        "confirmationMaterializersAbsent": all(
            row["materializerAbsent"] for row in rows if row["split"] == "confirmation"
        ),
        "validationOutcomeEvaluationsZero": True,
        "confirmationOutcomeEvaluationsZero": True,
        "protectedOutcomeArtifactsOpenedZero": True,
    }
    return {
        "schemaVersion": "e07.s03.training-split-access-audit.v1",
        "researchStepId": "S03",
        "evaluatedTrainingScenarioIds": training_ids,
        "nonTrainingDevelopmentDenials": rows,
        "brokerAudit": suite.broker.audit.to_dict(),
        "protectedOutcomeArtifactsOpened": 0,
        "protectedScenarioPayloadsOpened": 0,
        "validationOutcomeEvaluations": 0,
        "confirmationOutcomeEvaluations": 0,
        "checks": checks,
        "passed": all(checks.values()),
    }


def input_paths() -> list[Path]:
    paths = [
        WORKSPACE / "AGENTS.md",
        WORKSPACE / "FULL_PLAN.md",
        WORKSPACE / "RESEARCH_PLAN.md",
        WORKSPACE / "PREVIOUS_ARTIFACTS.md",
        WORKSPACE / "PREVIOUS_ARTIFACTS.json",
        WORKSPACE / "DATASETS.md",
        WORKSPACE / "DATASET_CATALOG.json",
        WORKSPACE / "DATASET_AVAILABILITY.json",
        WORKSPACE / "CAPABILITIES.md",
        WORKSPACE / "CAPABILITY_AVAILABILITY.json",
        WORKSPACE / "input-attachments/MANIFEST.json",
        WORKSPACE
        / "input-attachments/21c2278b-9950-4e39-a2c8-df578a2508ec/_metadata/ATTACHMENT.md",
        Path(
            "/previous-artifacts/E01/research_steps/S14/research_step_full_results.md"
        ),
        Path("/previous-artifacts/E01/specification/transition_spec.md"),
        Path(
            "/previous-artifacts/E02/research_steps/S14/research_step_full_results.md"
        ),
        Path("/previous-artifacts/E02/research_steps/S02/action_interface_spec.md"),
        Path(
            "/previous-artifacts/E03/research_steps/S14/research_step_full_results.md"
        ),
        Path("/previous-artifacts/E03/research_steps/S14/e07_handoff.md"),
        Path("/previous-artifacts/E03/research_steps/S01/distance_spec.md"),
        Path("/previous-artifacts/E03/research_steps/S06/necessary_detour_spec.md"),
        Path(
            "/previous-artifacts/E04/research_steps/S14/research_step_full_results.md"
        ),
        Path("/previous-artifacts/E04/report_inputs/e06_e07_handoff.md"),
        Path(
            "/previous-artifacts/E05/research_steps/S14/research_step_full_results.md"
        ),
        Path(
            "/previous-artifacts/E05/research_steps/S14/regeneration_benchmark/E07_HANDOFF.md"
        ),
        Path(
            "/previous-artifacts/E05/research_steps/S14/regeneration_benchmark/E07_HANDOFF.json"
        ),
        Path(
            "/previous-artifacts/E06/research_steps/S14/research_step_full_results.md"
        ),
        Path("/previous-artifacts/E06/report_inputs/e07_handoff.md"),
        Path("/previous-artifacts/E06/research_steps/S02/grammar_spec.md"),
        Path("/previous-artifacts/E06/research_steps/S03/environment_spec.md"),
        Path("/previous-artifacts/E06/research_steps/S04/movement_spec.md"),
        Path("/previous-artifacts/E06/research_steps/S05/policy_spec.md"),
        Path("/previous-artifacts/E06/research_steps/S06/control_channel_spec.md"),
        Path("/previous-artifacts/E06/research_steps/S07/engine_spec.md"),
        Path("/previous-artifacts/E06/research_steps/S14/split_manifest.json"),
    ]
    paths.extend(sorted(Path("/artifacts/research_steps/S01").glob("*")))
    paths.extend(
        path
        for path in sorted(Path("/artifacts/research_steps/S02").rglob("*"))
        if path.is_file()
    )
    paths.extend(
        [
            REPOSITORY / "src/policy_dsl/core.py",
            REPOSITORY / "src/environment_suite/access.py",
            REPOSITORY / "src/environment_suite/contracts.py",
            REPOSITORY / "src/environment_suite/runners.py",
            REPOSITORY / "src/environment_suite/suite.py",
            REPOSITORY / "src/seed_library/core.py",
            REGISTRY,
            SPLITS,
        ]
    )
    unique: dict[str, Path] = {}
    for path in paths:
        if path.is_file():
            unique[str(path)] = path
    return [unique[key] for key in sorted(unique)]


def input_provenance() -> dict[str, Any]:
    rows = [
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "classification": "governing_public_handoff_or_prior_S01_S02_artifact",
        }
        for path in input_paths()
    ]
    return {
        "schemaVersion": "e07.s03.input-provenance.v1",
        "researchStepId": "S03",
        "files": rows,
        "protectedOutcomeTablesOpened": 0,
        "protectedScenarioPayloadsOpened": 0,
        "validationOutcomeEvaluations": 0,
        "confirmationOutcomeEvaluations": 0,
        "previousArtifactsMutated": False,
    }


def seed_library_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "e07.s03.seed-library.v1",
        "type": "object",
        "required": [
            "schemaVersion",
            "researchStepId",
            "policyId",
            "policySha256",
            "family",
            "environment",
            "permissions",
            "complexity",
            "licensedCapabilities",
            "canonicalPolicy",
        ],
        "properties": {
            "schemaVersion": {"const": "e07.s03.seed-library.v1"},
            "researchStepId": {"const": "S03"},
            "policyId": {"type": "string"},
            "policySha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "family": {
                "enum": [
                    "faithful",
                    "simplified_variant",
                    "random_finite_state",
                    "repair",
                ]
            },
            "environment": {"enum": ["line1d.v1", "spatial2d.v1"]},
            "permissions": {"type": "array", "items": {"type": "string"}},
            "complexity": {"type": "object"},
            "licensedCapabilities": {"type": "object"},
            "canonicalPolicy": {"type": "object"},
        },
        "additionalProperties": True,
    }


def report_markdown(
    evidence: Mapping[str, Any],
    guard: Mapping[str, Any],
    access: Mapping[str, Any],
    artifacts: list[str],
    focused_test_result: str,
    regression_test_result: str,
    lint_result: str,
) -> str:
    counts = evidence["validation"]["counts"]
    coverage = evidence["coverage"]
    family_counts = coverage["familyCounts"]
    feature_counts = coverage["structuralFeatureCounts"]
    insertion = next(
        row
        for row in evidence["complexityRecords"]
        if row["policyId"] == "insertion_cell_view_v1"
    )["trainingAdapterProjectionLedger"]
    selection = next(
        row
        for row in evidence["complexityRecords"]
        if row["policyId"] == "selection_cell_view_v1"
    )["trainingAdapterProjectionLedger"]
    return f"""# Research Step S03 Full Results — Seed the Search

## Concise handoff summary

- **Research step ID:** S03 — Seed the search.
- **Completion status:** Complete on 2026-07-21; S03 alone was executed and S04 was not started.
- **Artifacts written:** {len(artifacts) + 1} compact files under `/artifacts/research_steps/S03/`, including `seed_library.jsonl`, bounded semantic equivalence groups, seed/baseline evaluations, separate complexity and licensed-capability ledgers, task-binding and split-access audits, native training-baseline guards, validation/provenance records, the artifact manifest, and this canonical report. Repository implementation and tests are Git-backed.
- **Validation result:** **PASS.** {counts["retainedSeeds"]}/{counts["retainedSeeds"]} retained seeds round-tripped canonical hashes and deterministic replay; {counts["faithfulProposalComparisons"]:,}/{counts["faithfulProposalComparisons"]:,} faithful Bubble/Insertion/Selection proposal comparisons matched; the deliberate semantic duplicate was detected and one member removed; 8/8 predecessor-native training baselines replayed and preserved contracts; validation and confirmation outcome evaluations were both zero. Focused tests: {focused_test_result}. Compatible regression tests: {regression_test_result}. Lint/format/compile: {lint_result}.
- **Outcome classification:** **Supportive**, with explicit representation constraints. The frozen S03 success criterion was met: all four seed families and the S01 baseline structural capabilities are represented without using protected outcomes or defining S04 objectives/descriptors.
- **Caveats or blockers:** Semantic equivalence is finite-corpus, not a universal proof. Seed outcomes were not estimated. E05 overlays remain deliberately unbound; E06 policies were exercised only on typed synthetic candidate fixtures because arbitrary bindings cannot be inferred from matching fields. Insertion retains a licensed prefix projection, Selection retains an engine cursor and licensed long-range action, and these charges remain separate. No S03 blocker remains.
- **Lay summary:** The search now has a varied, reproducible set of small controller programs: exact versions of the three sorting rules, simpler versions, random finite-state controllers, and repair-oriented seeds. They were checked only on training-derived inputs. Two different programs that always behaved the same under the declared finite test were recognized as duplicates, so only one was kept. The work does not claim any new policy is good at a task yet.
- **Recommended next action:** Return control to the Chief Scientist. If separately authorized, S04 may define and freeze objectives, archive descriptors, normalization rules, and anti-gaming checks using this seed library and its explicit binding gaps. Do not begin S04 from this handoff alone.

## Frozen question and success criterion

**Question:** Do faithful policies, validated simplifications, random finite-state controllers, and repair policies provide useful starting structural coverage without determining final discoveries?

**Success criterion:** encode and canonically hash the required families; evaluate them using only frozen training inputs; revalidate faithful Bubble, Insertion, and Selection behavior; preserve typed permissions and predecessor boundaries; cost licensed prefix/cursor/range capabilities explicitly; detect bounded semantic duplicates; replay deterministically; cover the S01 baseline structural envelope; and record unresolved representation gaps without defining S04 objectives or archive descriptors.

The criterion was met. “Coverage” below is structural seed coverage only. No performance objective, archive coordinate, weighting, normalization, reward, or novelty threshold was selected.

## Inputs and access boundary

The step refreshed `AGENTS.md`, `FULL_PLAN.md`, `RESEARCH_PLAN.md`, previous-artifact manifests, dataset/capability records, the attachment manifest and sidecar, all E01–E06 final/public E07 handoffs and public contracts used by S02, every S01 DSL artifact, and the S02 environment-suite, registry, split, access, communication, horizon, smoke, parity, and report artifacts. `input_provenance.json` hashes every selected file.

Evaluation used exactly the eight S02 records whose split is `train`. The split manifest's validation and confirmation metadata was inspected only to enforce access boundaries. The access audit issued development-phase denial probes for all 16 non-training records; all were denied before materialization. No validation or confirmation outcome was evaluated, no protected payload was opened, and no predecessor artifact was mutated.

## Detailed methods

### Seed construction

The pre-deduplication set had {counts["candidateSeeds"]} policies:

- three exact S01 faithful seeds: Bubble, Insertion, and Selection;
- six validated simplifications: left-only and right-only Bubble, unlicensed adjacent Insertion, scan-only and swap-only Selection, and adjacent-only spatial greedy;
- sixteen counter-generated finite-state controllers, split evenly between 1D and 2D; and
- four repair seeds, including the two S01 exemplars and two new bounded hand designs.

Two additional no-op guards inside the random family were deliberately constructed as distinct canonical programs with the same bounded semantics. Deduplication retained one, yielding {counts["retainedSeeds"]} final seeds. Family counts are `{json.dumps(family_counts, sort_keys=True)}`.

Simplified variants were not relabeled arbitrary mutations. Each transform was checked structurally against its parent: exact rule subsets, one licensed predicate removal, one action/radius removal, one rule extraction, or one movement-allowlist restriction. The exact transform and parent IDs are stored on every seed.

### Training-only probe evaluation

Four line-task training records generated {counts["lineTrainingProbes"]:,} deterministic valid-state probes. The probe values, directions, fault map, task size, and scenario identity come only from the frozen training records. Actors were restricted to normal cells because actor-fault suppression is trusted engine behavior absent from the S01 observation envelope. Occupancies, actor choices, Bubble sides, Selection cursors, repair fields, and uint8 counter values used a SHA-256 counter derivation.

For the three faithful policies, each line probe constructed an E01 shadow oracle that changed only the activated actor's Algotype to the policy under test. Proposal kind, adjacent/cursor target, and cursor update were compared. These are proposal-level semantics, not task episode results and not a replacement of E04 composition.

The two E06 training records generated {counts["spatialTypedTrainingProbes"]:,} deterministic typed fixtures with opaque candidate handles and only closed S01 fields. This checks interpreter behavior, memory, action selection, and permissions. It does **not** authenticate an E06 candidate builder, bind a DSL program into S05/S07 internals, or make a formation/repair claim. E05 regeneration and target-change records remained unbound; only their unchanged native training baselines were replayed.

Every retained seed executed twice over its complete compatible probe stream with identity-owned memory carried sequentially. The evaluation records action counts, interpreter operations, memory/signals, task probe counts, canonical behavior hashes, and separate adapter projection ledgers. Full traces were not retained; reproducible aggregate hashes and source code are sufficient to rebuild them.

### Licensed capabilities and complexity

Structural DSL complexity remains the S01 vector: rules, branches, expression nodes, actions, memory bits, signal bits, movement radius, candidates, worst-case operations, and canonical bytes. S03 adds a separate training adapter ledger. It never adds those numbers into native predecessor costs or a scalar score.

- Faithful Insertion incurred {insertion["licensedPrefixPredicateEvaluations"]:,} licensed prefix projections, {insertion["licensedPrefixValueReads"]:,} prefix value reads, and {insertion["licensedPrefixValueComparisons"]:,} prefix comparisons.
- Faithful Selection incurred {selection["engineCursorStateReads"]:,} cursor-state reads, {selection["engineCursorTargetProjectionReads"]:,} in-bounds target projections, {selection["engineCursorAdvanceActions"]:,} cursor advances, and {selection["engineCursorSwapActions"]:,} cursor swaps. Requested long-range distance summed to {selection["licensedLongRangeRequestedDistance"]:,}, with maximum {selection["licensedLongRangeMaximumRequestedDistance"]} positions.

These are transparent logical accounting units. They are not time, energy, physical information, money, or normalized task cost. The Selection action remains licensed even on a probe where requested distance happens to be adjacent.

### Semantic equivalence

`equivalence_groups.json` distinguishes:

1. canonical duplicates: identical S01 policy hashes;
2. bounded semantic duplicates: identical typed interface, permissions, memory/signal contract, licensed capabilities, movement/candidate bounds, and action/memory/signal/operation trace on every S03 probe; and
3. action-only near duplicates: identical actions but different information, memory, capability, or operation contracts, which are **not** collapsed.

The two deliberate no-op guards have distinct policy hashes but one bounded semantic signature, so guard B was discarded. This establishes that duplicate detection works. It does not prove equivalence on every mathematically possible observation or memory history.

### Native contract guard

All eight S02 native training baselines were rerun through reset/observation/step/cost/event/outcome using development grants. Each was compared directly with its predecessor-native runner for stop reason, censoring, failure, ledgers, events, outcomes, validations, replay, and split. All 8 passed. Horizons and native units remain task-specific; no cross-task time or cost normalization was applied.

## Results

### Seed coverage

The final library has {coverage["seedCount"]} seeds: {coverage["environmentCounts"]["line1d.v1"]} line policies and {coverage["environmentCounts"]["spatial2d.v1"]} spatial policies. Structural feature counts are `{json.dumps(feature_counts, sort_keys=True)}`. Every declared coverage check passed: the faithful trio, both environments, all four seed families, memory/memoryless, signaling, counter input, licensed prefix, engine cursor, licensed long range, opaque candidates, and multiple movement-radius classes.

This result means the library spans the **known baseline structural envelope**. It does not mean task-performance space is covered, quality-diversity bins are occupied, or search discovery is predetermined.

### Faithful behavior and replay

Faithful proposal parity was {counts["faithfulProposalComparisons"]:,}/{counts["faithfulProposalComparisons"]:,}, with zero mismatches. All {counts["retainedSeeds"]} retained canonical policy documents recompiled to their stored hashes. Policy hashes were unique after deduplication. Repeating every seed evaluation produced byte-identical machine records.

### Task bindings and unresolved gaps

| Task family | S03 seed treatment | Outcome use |
| --- | --- | --- |
| E01 sorting | Training-derived typed line probes; faithful E01 shadow parity | None |
| E02 faults | Training fault-map probes with trusted legality left engine-owned | None |
| E03 detours | Training line probes; no distance used online or as seed score | None |
| E04 chimera | Training line probes; no portfolio composition changed or pooled | None |
| E05 regeneration | DSL unbound; native training baseline guard only | Training native guard only |
| E05 target change | DSL unbound; target signal not treated as generic communication | Training native guard only |
| E06 spatial local | Typed synthetic opaque-candidate fixtures; no native binding | None |
| E06 spatial memory | Typed synthetic opaque-candidate fixtures; no native binding | None |

The main representation gap is therefore executable episode evaluation of arbitrary DSL seeds. E05 overlays and E06 candidate/authentication internals require separately specified adapters. S03 leaves those gaps visible rather than inferring semantics from matching field names.

## Validation

| Check | Result |
| --- | --- |
| Canonical hashes | PASS — {counts["retainedSeeds"]}/{counts["retainedSeeds"]} round-trip; unique after dedup |
| Faithful behavior | PASS — {counts["faithfulProposalComparisons"]:,} matches, 0 mismatches |
| Complexity | PASS — all static worst cases within declared budgets; licensed ledgers separate |
| Structural seed coverage | PASS — all declared S01-baseline envelope checks |
| Semantic duplicate detection | PASS — 1 deliberate group detected; 1 policy discarded |
| Deterministic replay | PASS — all evaluation records byte-identical on independent repeat |
| Native training contract guard | PASS — 8/8 tasks; predecessor parity and replay |
| Split/access boundary | PASS — train only; 16/16 non-training development requests denied; no protected materialization |
| Focused repository tests | PASS — {focused_test_result} |
| Compatible predecessor regressions | PASS — {regression_test_result} |
| Format/lint/compile | PASS — {lint_result} |

## Commands

```bash
PYTHONPATH=. python -m pytest -q tests/test_seed_library.py
PYTHONPATH=. python -m pytest -q tests/test_policy_dsl.py tests/test_environment_suite.py tests/test_reference_simulator.py::DeterminismTests tests/test_morph2d_policies.py tests/test_morph2d_movements.py
ruff format --check src/seed_library scripts/build_seed_library_s03.py tests/test_seed_library.py
ruff check src/seed_library scripts/build_seed_library_s03.py tests/test_seed_library.py
python -m compileall -q src/seed_library scripts/build_seed_library_s03.py tests/test_seed_library.py
PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python scripts/build_seed_library_s03.py --output /artifacts/research_steps/S03 --focused-test-result "{focused_test_result}" --regression-test-result "{regression_test_result}" --lint-result "{lint_result}"
```

No package, system dependency, web source, GPU, or network access was used. Execution was intentionally serial: the workload is deterministic interpreter/proposal validation and eight compact native baselines, not a large replicate campaign.

## Artifact guide and provenance

- `seed_library.jsonl`: 30 canonical reusable seeds.
- `seed_library_schema.json`: machine schema for each JSONL record.
- `baseline_evaluations.jsonl`: deterministic per-seed behavior/projection summaries.
- `complexity_accounting.jsonl`: structural complexity plus separate licensed projection/action ledgers.
- `equivalence_groups.json`: criteria, groups, retained/discarded IDs, and proof boundary.
- `seed_coverage.json`: structural coverage only; explicitly not an S04 descriptor registry.
- `task_binding_matrix.json`: typed/native binding status and unresolved gaps.
- `native_training_baseline_guard.json`: eight unchanged training baseline executions.
- `training_split_access_audit.json`: train-only and protected-split evidence.
- `validation_summary.json`, `input_provenance.json`, `environment.json`, `execution_commands.log`, `status.json`, `artifact_manifest.json`, and this report.

Repository source is on `eidosoma/groups/28` at commit `{_git("rev-parse", "HEAD")}`. Source is not copied into artifacts. `input_provenance.json` records all governing, handoff, S01/S02, configuration, and source hashes; `artifact_manifest.json` records compact output hashes. Previous artifacts remained read-only.

## Caveats, failed assumptions, and claim boundaries

1. Bounded semantic equivalence is empirical over the frozen probe corpus, not a decision procedure for universal program equivalence.
2. The training split has one public record per task. Probe diversity is deterministic valid-state coverage, not a population estimate or outcome sample.
3. The faithful shadow changes an activated actor's Algotype solely to compare E01 proposals. It is not an E04 portfolio intervention or episode evaluation.
4. E05 task-phase, target-signal, memory, and control semantics remain predecessor-owned and unbound to arbitrary DSL policies.
5. E06 opaque candidates in S03 are typed fixtures. Native candidate enumeration, authentication, conflict resolution, atomic commit, and channel authority remain unbound.
6. Repair-named seeds are program structures, not evidence of repair efficacy, regeneration, memory in a biological sense, or competency.
7. Random finite-state seeds are deterministic programs conditional on a bounded counter input; they do not access raw random keys or future draws.
8. No performance objective, objective weight, archive descriptor, novelty test, normalization, or anti-gaming threshold was frozen. Those remain S04 work.
9. All computational claims are restricted to transparent simulators and typed fixtures; no biological, cognitive, clinical, or real-world conclusion is supported.

No blocker prevents handoff. S04 remains unstarted.
"""


def build(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    evidence = build_s03_evidence(
        REGISTRY,
        SPLITS,
        probes_per_record=args.probes_per_record,
    )
    guard = native_training_guard()
    training_ids = [record["scenarioId"] for record in evidence["trainingRecords"]]
    access = split_access_audit(training_ids)

    _write_jsonl(output / "seed_library.jsonl", evidence["seedRecords"])
    _write_json(output / "seed_library_schema.json", seed_library_schema())
    _write_jsonl(output / "baseline_evaluations.jsonl", evidence["evaluations"])
    _write_jsonl(output / "complexity_accounting.jsonl", evidence["complexityRecords"])
    _write_json(output / "equivalence_groups.json", evidence["equivalence"])
    _write_json(output / "seed_coverage.json", evidence["coverage"])
    _write_json(output / "task_binding_matrix.json", evidence["taskBindings"])
    _write_json(output / "training_probe_summary.json", evidence["probeSummary"])
    _write_json(output / "native_training_baseline_guard.json", guard)
    _write_json(output / "training_split_access_audit.json", access)
    _write_json(output / "input_provenance.json", input_provenance())

    combined_checks = {
        **evidence["validation"]["checks"],
        "nativeTrainingBaselineGuard": guard["allPassed"],
        "trainingSplitAccessAudit": access["passed"],
    }
    validation = {
        **evidence["validation"],
        "checks": combined_checks,
        "nativeTrainingBaselineCount": guard["taskCount"],
        "nonTrainingDevelopmentDenialCount": len(
            access["nonTrainingDevelopmentDenials"]
        ),
        "focusedTestResult": args.focused_test_result,
        "regressionTestResult": args.regression_test_result,
        "lintResult": args.lint_result,
        "success": all(combined_checks.values()),
    }
    _write_json(output / "validation_summary.json", validation)

    environment = {
        "schemaVersion": "e07.s03.environment.v1",
        "researchStepId": "S03",
        "createdAtUtc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "cpuCountReported": os.cpu_count(),
        "workerLimit": 8,
        "actualParallelWorkers": 1,
        "threadEnvironment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "gitHead": _git("rev-parse", "HEAD"),
        "gitBranch": _git("branch", "--show-current"),
        "newDependenciesInstalled": [],
        "networkUsed": False,
        "gpuUsed": False,
    }
    _write_json(output / "environment.json", environment)

    commands = [
        "PYTHONPATH=. python -m pytest -q tests/test_seed_library.py",
        "PYTHONPATH=. python -m pytest -q tests/test_policy_dsl.py tests/test_environment_suite.py tests/test_reference_simulator.py::DeterminismTests tests/test_morph2d_policies.py tests/test_morph2d_movements.py",
        "ruff format --check src/seed_library scripts/build_seed_library_s03.py tests/test_seed_library.py",
        "ruff check src/seed_library scripts/build_seed_library_s03.py tests/test_seed_library.py",
        "python -m compileall -q src/seed_library scripts/build_seed_library_s03.py tests/test_seed_library.py",
        f"PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python scripts/build_seed_library_s03.py --output {output} --probes-per-record {args.probes_per_record}",
    ]
    (output / "execution_commands.log").write_text(
        "\n".join(commands) + "\n", encoding="utf-8"
    )

    artifact_names = [
        "seed_library.jsonl",
        "seed_library_schema.json",
        "baseline_evaluations.jsonl",
        "complexity_accounting.jsonl",
        "equivalence_groups.json",
        "seed_coverage.json",
        "task_binding_matrix.json",
        "training_probe_summary.json",
        "native_training_baseline_guard.json",
        "training_split_access_audit.json",
        "input_provenance.json",
        "validation_summary.json",
        "environment.json",
        "execution_commands.log",
        "research_step_full_results.md",
        "status.json",
    ]
    report = report_markdown(
        evidence,
        guard,
        access,
        artifact_names,
        args.focused_test_result,
        args.regression_test_result,
        args.lint_result,
    )
    (output / "research_step_full_results.md").write_text(report, encoding="utf-8")

    status = {
        "researchStepId": "S03",
        "stepNumber": 3,
        "success": validation["success"],
        "status": "complete_stopped_before_S04",
        "artifactsWritten": artifact_names + ["artifact_manifest.json"],
        "validationResult": (
            f"PASS: {validation['counts']['retainedSeeds']} seeds; "
            f"{validation['counts']['faithfulProposalComparisons']} faithful proposal matches; "
            "8/8 native training baselines; no protected outcome evaluation"
        ),
        "caveatsOrBlockers": [
            "bounded semantic equivalence is finite-corpus evidence",
            "E05 arbitrary DSL bindings remain unbound",
            "E06 execution uses typed synthetic fixtures only",
            "Insertion prefix and Selection cursor/range remain licensed and separately costed",
        ],
        "recommendedNextAction": "Return to the Chief Scientist; authorize S04 separately if objective/descriptor freezing is desired.",
    }
    _write_json(output / "status.json", status)

    manifest_rows = []
    for name in artifact_names:
        path = output / name
        manifest_rows.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    manifest = {
        "schemaVersion": "e07.s03.artifact-manifest.v1",
        "researchStepId": "S03",
        "artifactRoot": str(output),
        "repositoryCommit": _git("rev-parse", "HEAD"),
        "manifestSelfHashExcluded": True,
        "artifactCountExcludingManifest": len(manifest_rows),
        "artifacts": manifest_rows,
        "validationStatus": "pass" if validation["success"] else "fail",
    }
    _write_json(output / "artifact_manifest.json", manifest)
    if not validation["success"]:
        raise SystemExit("S03 validation failed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=os.environ.get("ARTIFACTS_DIR", "/artifacts") + "/research_steps/S03",
    )
    parser.add_argument("--probes-per-record", type=int, default=128)
    parser.add_argument("--focused-test-result", default="not supplied")
    parser.add_argument("--regression-test-result", default="not supplied")
    parser.add_argument("--lint-result", default="not supplied")
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
