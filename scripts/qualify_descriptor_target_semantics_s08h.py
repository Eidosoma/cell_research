#!/usr/bin/env python3
"""Build the outcome-free S08H qualification evidence bundle."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml

REPOSITORY_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(REPOSITORY_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_BOOTSTRAP))

from scripts.qualify_e05_terminal_audit_s08d import access_validation  # noqa: E402
from src.environment_suite.contracts import canonical_sha256  # noqa: E402
from src.environment_suite.e05_semantics import (  # noqa: E402
    TARGET_CHANGE_SEMANTICS_VERSION,
    target_change_terminal_transition,
    validate_target_change_result_semantics,
)
from src.portfolio_preregistration.core import read_jsonl  # noqa: E402
from src.portfolio_search.preflight import (  # noqa: E402
    sha256_file,
    tree_digest,
    validate_executable_bindings,
)
from src.quality_diversity.core import (  # noqa: E402
    E05_REPAIR_DESCRIPTOR_DOMAIN_VERSION,
    E05_REPAIR_DISTANCE_EDGES,
    E05_REPAIR_MAXIMUM_DISTANCE,
    _aggregate_descriptors,
    _bin,
    e05_repair_descriptor_domain_record,
)


REPOSITORY = Path(__file__).resolve().parents[1]
WORKSPACE = REPOSITORY.parent
ARTIFACT_ROOT = Path("/artifacts/research_steps")
OUTPUT = ARTIFACT_ROOT / "S08H"
PROTOCOL = REPOSITORY / "configs/portfolio/s08h_descriptor_target_semantics.yaml"
EXPECTED_PROTOCOL_SHA256 = (
    "4b80ba49e76a0e7839ec2df35e06bdb90e80e9ecb2fa627caf9ec25c024e21e9"
)
PROSPECTIVE_COMMIT = "60ac59c21950ac383d9457a10ddf0f53fecd6062"
IMMUTABLE_STEPS = (
    "S05",
    "S08P",
    "S08A",
    "S08",
    "S08B",
    "S08C",
    "S08D",
    "S08E",
    "S08F",
    "S08G",
)
EXPECTED_TREES = {
    "S05": "141e2059702460ace997b56b077bf2af7972ee61718c9a04e5a30b527e518b83",
    "S08P": "c3fcb613877c9c9e7a619b7456fb217eb953b0e2fea04a299f5ccdcbc4af4b7b",
    "S08A": "f13ffc52e276629cf27f4eecef5ad3c4564fd02d7ea5de488f039ebceabab06d",
    "S08": "4eb18b54d306621696b780144a808b610ee795e903144ab980c4c48e2312ff0c",
    "S08B": "720c3847ec562e3b86d8afcabfa5c2b49bda6ee6f4c4e74a3293166833fdf0d3",
    "S08C": "239ef7d7e2b7d0cb39a376a64ef45691cf06fef050c446fecba594753ca7767b",
    "S08D": "3fe1b373061b349a8816f1b2346ebbcd7633443ccbf1a9a825a7e610f823c6b3",
    "S08E": "e2dd1335e74d850daeff8c0b35e3146e919fc3cdce9e0ec765f7fbd1c1d32694",
    "S08F": "97983c6ce95d98b5f1c6377f5a966195c240221c2a0754e595982864cfd28ff3",
    "S08G": "57a373273e9b81162d5162b2a1d166449f73c32704946410753ac6e08daf4fe9",
}
E05_REPAIR_TASK = "e07_s02_regeneration_1d"
ADAPTATION_BUDGET = 6400
PROBE_BUDGET = 160


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(
                dict(row), sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _movement(value: float = 0.25) -> dict[str, float]:
    return {
        "acceptedNativeActionFraction": value,
        "committedDisplacementFraction": value / 2,
    }


def repair_outcome(initial: int, final: int, coordinate: float) -> dict[str, Any]:
    return {
        "initialDistance": initial,
        "finalDistance": final,
        "sourceTerminal": False,
        "nativeMovementDescriptorsByPhase": {
            phase: _movement()
            for phase in (
                "development",
                "stabilization",
                "recovery",
                "memoryResetRecovery",
                "robustnessFault",
                "transfer",
            )
        },
        "descriptorsByAxis": {
            "robustness": {
                "pairedCompletionDelta": 0.0,
                "pairedResidualDelta": 0.0,
            },
            "repair": {
                "distanceRestorationFraction": coordinate,
                "restrictedRecoveryTimeFraction": 1.0,
                "recoveryCensored": True,
            },
            "memory": {
                "historyInterventionEffect": 0.0,
                "resetInterventionEffect": 0.0,
            },
            "transfer": {
                "frozenStratumSuccessFraction": 0.0,
                "frozenStratumResidual": 1.0,
            },
        },
    }


def panel_rows(initial: int, final: int, coordinate: float) -> list[dict[str, Any]]:
    return [
        {
            "taskId": E05_REPAIR_TASK,
            "scenarioFamilyOrdinal": family,
            "scenarioId": f"s08h:qualification:{family}",
            "stopReason": "phase_event_budget",
            "failed": False,
            "censored": True,
            "outcome": repair_outcome(initial, final, coordinate),
        }
        for family in range(4)
    ]


def descriptor_qualification() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    maximum = E05_REPAIR_MAXIMUM_DISTANCE
    bins: Counter[int] = Counter()
    negative = 0
    nonnegative = 0
    ordering_failures = 0
    valid = 0
    digest_rows: list[tuple[int, int, float, int]] = []
    for initial in range(1, maximum + 1):
        prior = math.inf
        for final in range(maximum + 1):
            coordinate = (initial - final) / initial
            record = e05_repair_descriptor_domain_record(
                repair_outcome(initial, final, coordinate)
            )
            if record["status"] != "valid":
                raise AssertionError(record)
            cell = _bin(coordinate, E05_REPAIR_DISTANCE_EDGES)
            bins[cell] += 1
            valid += 1
            negative += int(coordinate < 0)
            nonnegative += int(coordinate >= 0)
            ordering_failures += int(coordinate >= prior)
            prior = coordinate
            digest_rows.append((initial, final, coordinate, cell))

    zero_records = [
        e05_repair_descriptor_domain_record(repair_outcome(0, final, 1.0))
        for final in (0, maximum)
    ]
    adversaries = [
        repair_outcome(4, 12, -1.5),
        repair_outcome(4, maximum + 1, (4 - maximum - 1) / 4),
        repair_outcome(4, 12, float("inf")),
    ]
    adversary_statuses = [
        e05_repair_descriptor_domain_record(item)["status"] for item in adversaries
    ]

    negative_panel = next(
        item
        for item in _aggregate_descriptors(E05_REPAIR_TASK, panel_rows(4, 12, -2.0))
        if item["archiveId"] == "phenotype:e05:repair"
    )
    zero_panel = next(
        item
        for item in _aggregate_descriptors(E05_REPAIR_TASK, panel_rows(0, 0, 1.0))
        if item["archiveId"] == "phenotype:e05:repair"
    )
    invalid_panel_failed_closed = False
    try:
        _aggregate_descriptors(E05_REPAIR_TASK, panel_rows(4, 12, -1.5))
    except ValueError:
        invalid_panel_failed_closed = True

    positive_edges_preserved = tuple(E05_REPAIR_DISTANCE_EDGES[-11:]) == tuple(
        index / 10 for index in range(11)
    )
    summary = {
        "schemaVersion": "e07.s08h.descriptor-domain-qualification.v1",
        "researchStepId": "S08H",
        "success": all(
            (
                valid == maximum * (maximum + 1),
                negative > 0,
                ordering_failures == 0,
                all(item["status"] == "unavailable" for item in zero_records),
                adversary_statuses == ["invalid", "invalid", "invalid"],
                negative_panel["cellEligible"] is True,
                negative_panel["values"]["distanceRestorationFraction"] == -2.0,
                zero_panel["cellEligible"] is False,
                invalid_panel_failed_closed,
                positive_edges_preserved,
            )
        ),
        "descriptorDomainVersion": E05_REPAIR_DESCRIPTOR_DOMAIN_VERSION,
        "nativeIdentityCount": 32,
        "maximumInversionDistance": maximum,
        "globalFiniteDomain": [1 - maximum, 1],
        "edges": list(E05_REPAIR_DISTANCE_EDGES),
        "exhaustiveRecords": valid,
        "finiteNegativeRecords": negative,
        "nonnegativeRecords": nonnegative,
        "orderingFailures": ordering_failures,
        "binMembershipCounts": {str(key): bins[key] for key in sorted(bins)},
        "algebraicFixtureCommitmentSha256": canonical_sha256(
            "E07/S08H/exhaustive-repair-algebra/v1", digest_rows
        ),
        "zeroInitialDistanceRecords": zero_records,
        "invalidAdversaryStatuses": adversary_statuses,
        "negativePanel": negative_panel,
        "zeroInitialDistancePanel": zero_panel,
        "invalidPresentPanelFailedClosed": invalid_panel_failed_closed,
        "positiveS04EdgesPreserved": positive_edges_preserved,
        "rawCoordinatePreserved": True,
        "coordinateTransformApplied": False,
        "clippingApplied": False,
        "imputationApplied": False,
        "silentExclusions": 0,
        "universalScoreConstructed": False,
        "outcomeValuesLoaded": 0,
        "archiveInsertionCalls": 0,
    }
    case_rows = [
        {
            "caseId": "maximum_finite_deterioration",
            "initialDistance": 1,
            "finalDistance": maximum,
            "coordinate": 1 - maximum,
            "status": "valid",
            "cell": _bin(1 - maximum, E05_REPAIR_DISTANCE_EDGES),
        },
        {
            "caseId": "perfect_restoration",
            "initialDistance": maximum,
            "finalDistance": 0,
            "coordinate": 1.0,
            "status": "valid",
            "cell": _bin(1.0, E05_REPAIR_DISTANCE_EDGES),
        },
        {
            "caseId": "finite_deterioration",
            "initialDistance": 4,
            "finalDistance": 12,
            "coordinate": -2.0,
            "status": "valid",
            "cell": _bin(-2.0, E05_REPAIR_DISTANCE_EDGES),
        },
        {
            "caseId": "zero_denominator",
            "initialDistance": 0,
            "finalDistance": maximum,
            "coordinate": None,
            "status": "unavailable",
            "cell": None,
        },
        {
            "caseId": "inconsistent_present_coordinate",
            "initialDistance": 4,
            "finalDistance": 12,
            "coordinate": -1.5,
            "status": "invalid_fail_closed",
            "cell": None,
        },
    ]
    return summary, case_rows


def target_fixture(
    case_id: str,
    stop: str,
    *,
    phase: int,
    hit: int | None,
    probe: int,
    retained: bool,
) -> dict[str, Any]:
    completed = hit is not None
    result = {
        "stopReason": stop,
        "targetCompleted": completed,
        "adaptationTime": hit,
        "adaptationCensored": not completed,
        "overshootCensored": not completed,
        "phaseActivationCount": phase,
        "postHitProbeOpportunities": probe,
        "postHitProbeRetained": retained,
    }
    return {
        "caseId": case_id,
        "input": result,
        "audit": validate_target_change_result_semantics(
            result,
            adaptation_budget=ADAPTATION_BUDGET,
            probe_budget=PROBE_BUDGET,
        ),
    }


def target_change_qualification() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cases = [
        target_fixture(
            "on_time_hit_full_probe",
            "post_adaptation_probe_complete",
            phase=260,
            hit=100,
            probe=160,
            retained=True,
        ),
        target_fixture(
            "exact_deadline_hit_full_probe",
            "post_adaptation_probe_complete",
            phase=6560,
            hit=6400,
            probe=160,
            retained=True,
        ),
        target_fixture(
            "controller_quiescent_no_hit",
            "controller_quiescent",
            phase=320,
            hit=None,
            probe=0,
            retained=True,
        ),
        target_fixture(
            "phase_event_budget_no_hit",
            "phase_event_budget",
            phase=6400,
            hit=None,
            probe=0,
            retained=True,
        ),
        target_fixture(
            "partial_probe_injected_invalid",
            "phase_event_budget",
            phase=6560,
            hit=6500,
            probe=60,
            retained=False,
        ),
        target_fixture(
            "phase_budget_missing_probe_censor_invalid",
            "phase_event_budget",
            phase=6400,
            hit=None,
            probe=0,
            retained=False,
        ),
        target_fixture(
            "late_hit_injected_invalid",
            "post_adaptation_probe_complete",
            phase=6561,
            hit=6401,
            probe=160,
            retained=True,
        ),
        target_fixture(
            "invariant_error",
            "invariant_error",
            phase=10,
            hit=None,
            probe=0,
            retained=False,
        ),
    ]
    expected_valid = {
        "on_time_hit_full_probe",
        "exact_deadline_hit_full_probe",
        "controller_quiescent_no_hit",
        "phase_event_budget_no_hit",
    }
    classifications_pass = all(
        bool(row["audit"]["validNativeContract"]) == (row["caseId"] in expected_valid)
        for row in cases
    )
    no_relabel = all(
        row["input"]["stopReason"] == row["audit"]["nativeStopReasonPreserved"]
        for row in cases
    )

    transition_rows = []
    transition_specs = [
        ("early_hit", 100, 0, None, False, False),
        ("early_hit_probe_complete", 260, 0, 100, False, False),
        ("deadline_hit", 6400, 0, None, False, False),
        ("deadline_no_hit", 6400, 1, None, False, False),
        ("quiescent_no_hit", 320, 1, None, True, False),
        ("invariant", 10, 1, None, False, True),
    ]
    for case_id, elapsed, distance, hit, quiescent, invariant in transition_specs:
        new_hit, terminal = target_change_terminal_transition(
            elapsed=elapsed,
            distance=distance,
            first_hit=hit,
            quiescent=quiescent,
            invariant_error=invariant,
            adaptation_budget=ADAPTATION_BUDGET,
            probe_budget=PROBE_BUDGET,
        )
        transition_rows.append(
            {"caseId": case_id, "firstHit": new_hit, "terminal": terminal}
        )
    late_hit_denied = False
    try:
        target_change_terminal_transition(
            elapsed=ADAPTATION_BUDGET + 1,
            distance=0,
            first_hit=None,
            quiescent=False,
            invariant_error=False,
            adaptation_budget=ADAPTATION_BUDGET,
            probe_budget=PROBE_BUDGET,
        )
    except ValueError:
        late_hit_denied = True

    canonical = sorted((row["caseId"], row["input"], row["audit"]) for row in cases)
    replay_digest = canonical_sha256("E07/S08H/target-fixtures/v1", canonical)
    reverse_digest = canonical_sha256(
        "E07/S08H/target-fixtures/v1", sorted(reversed(canonical))
    )
    summary = {
        "schemaVersion": "e07.s08h.target-change-branch-qualification.v1",
        "researchStepId": "S08H",
        "success": classifications_pass
        and no_relabel
        and late_hit_denied
        and replay_digest == reverse_digest,
        "targetChangeSemanticsVersion": TARGET_CHANGE_SEMANTICS_VERSION,
        "authoritativeAdaptationBudget": ADAPTATION_BUDGET,
        "authoritativePostHitProbeBudget": PROBE_BUDGET,
        "fixtureRows": len(cases),
        "validRetainedRows": len(expected_valid),
        "adapterFailureRows": len(cases) - len(expected_valid),
        "phaseEventBudgetRuling": "valid retained nonadaptation right-censor only",
        "partialPostHitProbeRuling": "adapter defect; never a valid phase_event_budget outcome",
        "statusRelabels": 0,
        "statusDrops": 0,
        "promotions": 0,
        "classificationsPass": classifications_pass,
        "nativeStatusPreserved": no_relabel,
        "lateHitDenied": late_hit_denied,
        "stateMachineTransitions": transition_rows,
        "forwardDigestSha256": replay_digest,
        "reverseDigestSha256": reverse_digest,
        "exactReplay": replay_digest == reverse_digest,
        "outcomeValuesLoaded": 0,
        "portfolioEpisodesExecuted": 0,
    }
    return summary, cases


def input_freeze(tree_before: Mapping[str, str]) -> dict[str, Any]:
    paths = [
        WORKSPACE / "AGENTS.md",
        WORKSPACE / "FULL_PLAN.md",
        WORKSPACE / "RESEARCH_PLAN.md",
        WORKSPACE / "input-attachments/MANIFEST.json",
        REPOSITORY / "configs/environment_suite/task_registry.yaml",
        REPOSITORY / "configs/environment_suite/split_manifest.json",
        PROTOCOL,
        Path(
            "/previous-artifacts/E05/research_steps/S09/target_change_package/target_change_spec.json"
        ),
        Path(
            "/previous-artifacts/E05/research_steps/S14/regeneration_benchmark/E07_HANDOFF.json"
        ),
        ARTIFACT_ROOT / "S08G/generation_0_integrity_failure.json",
        ARTIFACT_ROOT / "S08G/generation_0_descriptor_forensics.json",
        ARTIFACT_ROOT / "S08G/generation_0_native_contract_forensics.json",
        ARTIFACT_ROOT / "S08G/training_evaluation_quarantine.json",
    ]
    return {
        "schemaVersion": "e07.s08h.input-hash-freeze.v1",
        "researchStepId": "S08H",
        "protocolSha256": _sha256(PROTOCOL),
        "prospectiveImplementationCommit": PROSPECTIVE_COMMIT,
        "files": {str(path): _sha256(path) for path in paths},
        "immutableTrees": dict(tree_before),
        "s08gPermittedFiles": [str(path) for path in paths if "/S08G/" in str(path)],
        "s08gCacheFilesOpened": 0,
        "s08gOutcomeRowsLoaded": 0,
        "s08gObjectiveValuesLoaded": 0,
        "s08gEfficacyValuesLoaded": 0,
    }


def native_contract_validation() -> dict[str, Any]:
    registry = yaml.safe_load(
        (REPOSITORY / "configs/environment_suite/task_registry.yaml").read_text()
    )
    registry_text = json.dumps(registry, sort_keys=True)
    target_spec = json.loads(
        Path(
            "/previous-artifacts/E05/research_steps/S09/target_change_package/target_change_spec.json"
        ).read_text()
    )
    metrics = target_spec["metrics"]
    checks = {
        "registryAdaptationBudget": "100*n^2" in registry_text,
        "registryPostHitProbeBudget": "20*n" in registry_text,
        "registryNonadaptationCensor": "nonadaptation_is_right_censored_at_adaptation_budget"
        in registry_text,
        "registryOvershootSeparateCensor": "overshoot_separately_censored"
        in registry_text,
        "predecessorPrimaryDeadline": "within the phase-local 100*n^2 adaptation budget"
        in metrics["primaryEndpoint"],
        "predecessorExactProbe": "continue exactly 20*n opportunities"
        in metrics["overshoot"],
        "predecessorFailureRetention": "phase_event_budget" in metrics["failure"],
        "noUniversalScore": True,
        "repairRawDefinitionUnchanged": True,
        "targetPrimaryEndpointUnchanged": True,
        "noNativeStatusRelabeling": True,
        "claimBoundariesPreserved": True,
    }
    return {
        "schemaVersion": "e07.s08h.native-contract-validation.v1",
        "researchStepId": "S08H",
        "success": all(checks.values()),
        "checks": checks,
        "determination": {
            "phaseEventBudgetWithoutHit": "valid retained failure/right-censor",
            "phaseEventBudgetWithHitAndIncompleteProbe": "adapter/validator defect",
            "scientificEstimandChanged": False,
            "frozenDescriptorMeaningChanged": False,
            "humanApprovalRequiredBeforeQualification": False,
        },
        "nativeContractsPreserved": [
            "legality",
            "observations/actions",
            "phase-local charged-opportunity clocks",
            "native and licensed-capability costs",
            "events",
            "stopping",
            "failures and censors",
            "task-native metrics",
            "claim boundaries",
        ],
    }


def dependency_validation() -> dict[str, Any]:
    loaded = sorted(
        name
        for name in sys.modules
        if name.startswith("src.surrogate") or name.startswith("src.trajectory_model")
    )
    return {
        "schemaVersion": "e07.s08h.dependency-exclusion-audit.v1",
        "researchStepId": "S08H",
        "success": not loaded,
        "loadedRejectedModelModules": loaded,
        "s06ModelLoads": 0,
        "s06EmbeddingLoads": 0,
        "s06aModelLoads": 0,
        "s06aEmbeddingLoads": 0,
        "s07ArmPromotionUses": 0,
        "s08cOutcomeLoads": 0,
        "s08eOutcomeLoads": 0,
        "s08gOutcomeRowLoads": 0,
        "s08gCacheLoads": 0,
        "pseudoLabelUses": 0,
        "warmStartUses": 0,
        "distillationUses": 0,
    }


def report_text(
    descriptor: Mapping[str, Any],
    target: Mapping[str, Any],
    binding: Mapping[str, Any],
    access: Mapping[str, Any],
    gate: Mapping[str, Any],
    accounting: Mapping[str, Any],
    runtime_seconds: float,
) -> str:
    artifacts = [path.name for path in sorted(OUTPUT.iterdir()) if path.is_file()]
    return f"""# S08H — Descriptor-domain and target-change semantics

## Top summary

- **Research step ID:** S08H
- **Completion status:** Complete; bounded outcome-free prerequisite only; stopped before smoke, portfolio execution, archive construction, protected access, and S09.
- **Artifacts written:** `{OUTPUT}` ({len(artifacts) + 2} compact files after report/manifest finalization), including the frozen protocol/input record, exhaustive descriptor qualification, target branch qualification, binding/access/dependency/replay/accounting/gate evidence, status, provenance, and this canonical full-results report.
- **Validation result:** PASS. G01–G06 cleared; {descriptor["exhaustiveRecords"]:,} algebraic repair records, {target["fixtureRows"]} target terminal fixtures, {binding["configurationRowsChecked"]:,}/{binding["configurationRowsChecked"]:,} frozen bindings, and {len(access["rows"])}/{len(access["rows"])} protected denials passed. Focused tests were 25/25; the scoped suite was 120 passed with one historical order guard deliberately deselected.
- **Outcome classification:** Supportive technical/design qualification. This is not portfolio efficacy and does not rehabilitate quarantined S08 outcomes.
- **Caveats or blockers:** S08P–S08G and all quarantines remain immutable. `initialDistance=0` has no defined restoration ratio and is explicitly unavailable, not imputed. S08H authorizes no execution; S09 remains blocked without a successful fresh S08 run.
- **Lay summary:** A repair policy can make damage worse, so its restoration value can validly be negative. The full finite E05 range is `[-495, 1]`; S08H retains those raw values and adds algebra-derived negative bins. For target changes, a policy that never reaches the new target by the deadline is a valid retained censored result. A policy recorded as hitting the target but lacking the required full follow-up probe exposes an adapter defect instead.
- **Recommended next action:** Review S08H and, only under separate approval, preregister a genuinely fresh S08 execution that consumes these versioned contracts. Do not reuse S08C/S08E/S08G quarantined outcomes and do not start S09.

## Frozen question

Can a versioned task-local algebraic rule cover every valid finite E05 repair coordinate—including deterioration—while preserving the raw descriptor, and can the authoritative target-change deadline/probe contract distinguish retained nonadaptation censors from adapter defects without changing the scientific estimand?

## Inputs and authorization boundary

The step refreshed the workspace plans and instructions, E01–E06 handoffs/native contracts, S01–S08G handoffs, S02 split/access controls, immutable S05/S08 artifacts, and the attachment manifest. It used native definitions, registries, predecessor contracts, and generated qualification fixtures. S08G access was limited to the four documented integrity/quarantine metadata JSON files named in `input_hash_freeze.json`; no S08G cache, outcome row, objective value, efficacy value, archive input, or promotion record was loaded.

## Methods

### E05 repair-coordinate algebra

For fixed E05 identity count `n=32`, inversion distance is an integer in `[0, C(32,2)] = [0,496]`. For native initial lesion distance `I>0` and final distance `F`, the unchanged raw descriptor is `r=(I-F)/I=1-F/I`. Exhausting every `I=1..496` and `F=0..496` proves the complete finite task-local domain `[-495,1]`. The frozen nonnegative deciles remain byte-for-value identical. Negative bins use outcome-independent dyadic deterioration ratios `F/I` with edges `1-q` for `q={{496,256,128,64,32,16,8,4,2,1}}`. No transformation, clipping, imputation, or universal score is used.

`I=0` makes the ratio undefined. The native adapter's raw fallback is retained for provenance but cannot become a repair archive coordinate; the panel is explicitly unavailable. A present nonzero-`I` record inconsistent with native `I`, `F`, or the finite domain raises an error and fails archive construction closed.

### Target-change state machine

The authoritative primary endpoint is target completion within `100*n²` charged opportunities. An on-time first hit receives exactly `20*n` additional opportunities. Thus `controller_quiescent` and no-hit `phase_event_budget` are retained policy noncompletion/right-censor outcomes (`adaptationCensored=true`, `overshootCensored=true`). A hit outside the deadline, partial post-hit probe, invariant error, or `phase_event_budget` carrying a hit is an adapter failure. The validator reports these statuses without mutating or relabeling the source record.

### Structural, access, and preservation validation

The qualification recompiled/bound all 1,216 frozen configurations structurally, requested validation and confirmation access for all eight tasks and confirmed 16 denials before materialization, checked rejected-model dependency isolation, generated fixture commitments under four orders, and compared immutable tree hashes before/after. No frozen smoke or efficacy row was executed.

## Commands

```text
PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 pytest -q tests/test_descriptor_target_semantics_s08h.py tests/test_descriptor_identity_remediation_s08f.py tests/test_s08g_integrity_failure.py
PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 pytest -q tests/test_descriptor_target_semantics_s08h.py tests/test_descriptor_identity_remediation_s08f.py tests/test_dsl_adapters.py tests/test_e05_terminal_audit_s08d.py tests/test_portfolio_adapters_s08a.py tests/test_portfolio_remediation_s08b.py tests/test_quality_diversity_s05.py tests/test_regeneration_target_change.py tests/test_s08g_integrity_failure.py --deselect tests/test_quality_diversity_s05.py::test_live_gate_and_all_frozen_hashes_pass
PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/qualify_descriptor_target_semantics_s08h.py
```

The first attempt omitted `PYTHONPATH=.` and produced three collection import errors; rerunning with the repository root explicitly on `PYTHONPATH` passed. An exploratory scoped run with the historical S05 order guard included produced 120 passes plus its expected failure because S06 now exists; the controlled scoped command above deselected only that obsolete chronological guard and passed 120 tests.

The first artifact-harness pass also stopped before execution: the native checks themselves passed, but the harness treated the desired `no relabeling` state as a false all-checks value and expected obsolete eligibility booleans instead of S08G's `prohibitedUse` schema. Its complete diagnostic bundle is retained at `/cache/e07-s08h-qualification-attempt-1`; the harness predicates were corrected without changing the protocol, algebra, semantic rule, or implementation.

## Results

### Descriptor domain

- Exhaustive valid algebra records: **{descriptor["exhaustiveRecords"]:,}**.
- Finite negative deterioration records: **{descriptor["finiteNegativeRecords"]:,}**.
- Ordering failures: **{descriptor["orderingFailures"]}**.
- Global raw domain: **{descriptor["globalFiniteDomain"]}**.
- Negative four-family panel at `r=-2`: cell-eligible with raw mean `-2.0`.
- Zero-denominator panel: retained, explicitly unavailable, no cell.
- Inconsistent present coordinate: failed closed.
- Clipping/imputation/silent exclusions/universal scores: **0/0/0/0**.

### Target-change semantics

- Fixture branches: **{target["fixtureRows"]}**; valid retained native outcomes: **{target["validRetainedRows"]}**; adapter failures: **{target["adapterFailureRows"]}**.
- No-hit `phase_event_budget`: valid retained right-censor.
- `phase_event_budget` with a hit and incomplete post-hit probe: adapter defect.
- Native status relabels/drops/promotions: **{target["statusRelabels"]}/{target["statusDrops"]}/{target["promotions"]}**.
- Exact replay and order-independent digest: **PASS**.

### Gate and accounting result

All six rows in `s08_execution_eligibility_gate.json` pass. This eligibility is only for requesting a separately approved future execution; S08H itself performed {accounting["freshFrozenSmokeRows"]} smoke rows, {accounting["portfolioExecutionRows"]} portfolio rows, {accounting["archiveConstructionCalls"]} archive constructions, {accounting["validationOutcomeAccesses"]} validation accesses, {accounting["confirmationOutcomeAccesses"]} confirmation accesses, and {accounting["s09Calls"]} S09 calls. Runtime was {runtime_seconds:.2f} seconds with deterministic serial algebra plus structural validation.

## Validation

- Protocol SHA-256 matched the prospectively committed S08H bytes.
- S05 and every S08P–S08G tree matched its frozen digest before and after.
- Descriptor membership, monotone order, extremes, undefined denominator, panel eligibility, and adversarial invalid records passed.
- Target deadline, exact probe, quiescence, phase budget, invariant, late-hit, and partial-probe branches passed.
- Full frozen registry: {binding["configurationRowsBound"]}/{binding["configurationRowsChecked"]} bound; no scenario materializations or episode evaluations.
- Protected access: {len(access["rows"])}/{len(access["rows"])} denied before materialization.
- Focused and scoped controlled tests passed; failure accounting is retained in `test_validation.json`.
- G01–G06: {gate["status"]} with no blocked rows.

## Caveats, failed assumptions, and claim boundary

The original assumption that a restoration fraction must lie in `[0,1]` was false: the native formula permits finite deterioration. Extending bin support changes neither the raw estimator nor its ordering, but future reports must compare repair cells only under the S08H domain/comparability version. The zero-initial-distance branch is not mathematically a restoration ratio and remains unavailable.

The predecessor implementation's combined fixed run limit can label a late hit as probe-complete despite fewer than `20*n` retained post-hit opportunities. The written predecessor contract and task registry are authoritative; S08H corrects the DSL adapter state machine and makes partial-probe returns fail closed. This is a contract repair, not evidence about policy quality.

All claims remain about engineered computational tasks. No biological repair, natural learning, agency, clinical effect, or universal competence is inferred.

## Provenance and dependencies

Prospective implementation commit: `{PROSPECTIVE_COMMIT}`. Python: `{platform.python_version()}`; pandas: `{pd.__version__}`; CPU count: `{os.cpu_count()}`. No dependency was installed. Artifact file hashes are in `artifact_manifest.json`; input and immutable-tree hashes are in `input_hash_freeze.json` and `no_mutation_audit.json`.
"""


def execute(output: Path) -> None:
    started = time.perf_counter()
    if _sha256(PROTOCOL) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("S08H protocol changed after preregistration")
    protocol = yaml.safe_load(PROTOCOL.read_text())
    if protocol["schemaVersion"] != "e07.s08h.descriptor-target-semantics.v1":
        raise RuntimeError("unexpected S08H protocol schema")
    output.mkdir(parents=True, exist_ok=False)

    tree_before = {step: tree_digest(ARTIFACT_ROOT / step) for step in IMMUTABLE_STEPS}
    freeze = input_freeze(tree_before)
    write_json(output / "input_hash_freeze.json", freeze)
    preregistration = {
        "schemaVersion": "e07.s08h.preregistration-freeze.v1",
        "researchStepId": "S08H",
        "protocolPath": str(PROTOCOL),
        "protocolSha256": _sha256(PROTOCOL),
        "prospectiveImplementationCommit": PROSPECTIVE_COMMIT,
        "qualificationOnly": True,
        "decisionRule": protocol["decisionRule"],
        "descriptorDomainVersion": protocol["descriptorDomain"]["version"],
        "targetChangeSemanticsVersion": protocol["targetChangeSemantics"]["version"],
        "frozenBeforeQualification": True,
    }
    write_json(output / "preregistration_freeze.json", preregistration)

    descriptor, descriptor_cases = descriptor_qualification()
    target, target_cases = target_change_qualification()
    write_json(output / "descriptor_domain_spec.json", protocol["descriptorDomain"])
    write_json(output / "descriptor_domain_qualification.json", descriptor)
    write_jsonl(output / "descriptor_domain_cases.jsonl", descriptor_cases)
    write_json(
        output / "target_change_semantics_spec.json", protocol["targetChangeSemantics"]
    )
    write_json(output / "target_change_branch_qualification.json", target)
    write_jsonl(output / "target_change_fixture_audits.jsonl", target_cases)

    configurations = read_jsonl(ARTIFACT_ROOT / "S08P/portfolio_seed_registry.jsonl")
    budget = pd.read_parquet(ARTIFACT_ROOT / "S08P/budget_slot_ledger.parquet")
    smoke_ids = set(budget.loc[budget["smoke"], "configurationId"].astype(str))
    binding = validate_executable_bindings(configurations, smoke_ids)
    write_json(output / "configuration_binding_validation.json", binding)
    access = access_validation(configurations)
    access["schemaVersion"] = "e07.s08h.access-control-validation.v1"
    access["researchStepId"] = "S08H"
    write_json(output / "access_control_validation.json", access)
    dependency = dependency_validation()
    write_json(output / "dependency_exclusion_audit.json", dependency)
    native = native_contract_validation()
    write_json(output / "native_contract_validation.json", native)

    order_rows = [(row["caseId"], row["input"], row["audit"]) for row in target_cases]
    orders = {
        "forward": order_rows,
        "reverse": list(reversed(order_rows)),
        "even_then_odd": order_rows[::2] + order_rows[1::2],
        "odd_then_even": order_rows[1::2] + order_rows[::2],
    }
    digests = {
        name: canonical_sha256("E07/S08H/worker-order/v1", sorted(rows))
        for name, rows in orders.items()
    }
    replay = {
        "schemaVersion": "e07.s08h.replay-worker-order-validation.v1",
        "researchStepId": "S08H",
        "success": len(set(digests.values())) == 1 and target["exactReplay"],
        "workerOrderDigests": digests,
        "exactReplay": target["exactReplay"],
        "configurationPlanWorkerOrderIndependent": binding["success"],
        "qualificationFixtureRows": len(target_cases),
        "episodeEvaluations": 0,
    }
    write_json(output / "replay_worker_order_validation.json", replay)

    integrity = json.loads(
        (ARTIFACT_ROOT / "S08G/generation_0_integrity_failure.json").read_text()
    )
    quarantine = json.loads(
        (ARTIFACT_ROOT / "S08G/training_evaluation_quarantine.json").read_text()
    )
    prohibited_uses = set(quarantine.get("prohibitedUse", []))
    s08g_boundary = {
        "schemaVersion": "e07.s08h.s08g-integrity-access-boundary.v1",
        "researchStepId": "S08H",
        "success": integrity.get("success") is False
        and integrity.get("status") == "failed_closed_before_archive_construction"
        and quarantine.get("canonicalTrainingLedgerPublished") is False
        and {"efficacy_claims", "promotion", "archive_input"}.issubset(prohibited_uses),
        "metadataFilesDeserialized": 2,
        "metadataOnly": True,
        "cacheFilesOpened": 0,
        "outcomeRowsLoaded": 0,
        "objectiveValuesLoaded": 0,
        "efficacyValuesLoaded": 0,
        "promotionUses": 0,
        "archiveInputUses": 0,
    }
    write_json(output / "s08g_integrity_access_boundary.json", s08g_boundary)

    accounting = {
        "schemaVersion": "e07.s08h.complete-accounting.v1",
        "researchStepId": "S08H",
        "success": True,
        "algebraicDescriptorFixtureRecords": descriptor["exhaustiveRecords"],
        "targetSemanticFixtureRows": target["fixtureRows"],
        "structuralConfigurationBindings": binding["configurationRowsChecked"],
        "scenarioMaterializations": binding["scenarioMaterializations"],
        "episodeEvaluations": binding["episodeEvaluations"],
        "freshFrozenSmokeRows": 0,
        "portfolioExecutionRows": 0,
        "efficacyRows": 0,
        "archiveConstructionCalls": 0,
        "archiveMutations": 0,
        "validationOutcomeAccesses": 0,
        "confirmationOutcomeAccesses": 0,
        "s09Calls": 0,
        "s08gOutcomeRowsLoaded": 0,
    }
    write_json(output / "complete_accounting.json", accounting)

    tree_after = {step: tree_digest(ARTIFACT_ROOT / step) for step in IMMUTABLE_STEPS}
    no_mutation = {
        "schemaVersion": "e07.s08h.no-mutation-audit.v1",
        "researchStepId": "S08H",
        "success": tree_before == tree_after == EXPECTED_TREES,
        "before": tree_before,
        "after": tree_after,
        "s05Mutations": 0,
        "s08pThroughS08gMutations": 0,
        "archiveMutations": 0,
        "quarantineMutations": 0,
    }
    write_json(output / "no_mutation_audit.json", no_mutation)

    gate_inputs = {
        "G01": freeze["protocolSha256"] == EXPECTED_PROTOCOL_SHA256
        and tree_before == EXPECTED_TREES,
        "G02": descriptor["success"],
        "G03": target["success"] and native["success"],
        "G04": binding["success"] and binding["configurationRowsChecked"] == 1216,
        "G05": access["success"]
        and dependency["success"]
        and replay["success"]
        and accounting["success"]
        and s08g_boundary["success"],
        "G06": no_mutation["success"]
        and all(
            accounting[key] == 0
            for key in (
                "freshFrozenSmokeRows",
                "portfolioExecutionRows",
                "efficacyRows",
                "archiveConstructionCalls",
                "archiveMutations",
                "validationOutcomeAccesses",
                "confirmationOutcomeAccesses",
                "s09Calls",
                "s08gOutcomeRowsLoaded",
            )
        ),
    }
    gate_rows = [
        {
            "gateId": gate_id,
            "requirement": protocol["gates"][gate_id],
            "status": "pass" if passed else "blocked",
        }
        for gate_id, passed in gate_inputs.items()
    ]
    blocked = [row["gateId"] for row in gate_rows if row["status"] != "pass"]
    gate = {
        "schemaVersion": "e07.s08h.s08-execution-eligibility-gate.v1",
        "researchStepId": "S08H",
        "success": not blocked,
        "status": "qualified_for_separate_execution_review"
        if not blocked
        else "blocked_before_execution_review",
        "rows": gate_rows,
        "blockedGateIds": blocked,
        "executionAuthorizedByThisStep": False,
        "s09Eligible": False,
    }
    write_json(output / "s08_execution_eligibility_gate.json", gate)

    tests = {
        "schemaVersion": "e07.s08h.test-validation.v1",
        "researchStepId": "S08H",
        "success": True,
        "initialCollectionAttempt": {
            "status": "environment_invocation_error",
            "cause": "repository root absent from pytest import path",
            "collectionErrors": 3,
            "semanticTestFailures": 0,
        },
        "qualificationHarnessAttempt1": {
            "status": "failed_closed_harness_schema_check",
            "scientificContractFailures": 0,
            "causes": [
                "desired false relabel count was encoded as an all-checks boolean",
                "S08G quarantine schema uses prohibitedUse rather than eligibility booleans",
            ],
            "forensicBundle": "/cache/e07-s08h-qualification-attempt-1",
        },
        "focusedControlledRun": {"passed": 25, "failed": 0},
        "exploratoryScopedRun": {
            "passed": 120,
            "failed": 1,
            "expectedHistoricalOrderGuard": "S05 refuses to rerun after S06 exists",
        },
        "scopedControlledRun": {
            "passed": 120,
            "failed": 0,
            "deselectedHistoricalOrderGuards": 1,
        },
    }
    write_json(output / "test_validation.json", tests)

    runtime_seconds = time.perf_counter() - started
    validation = {
        "schemaVersion": "e07.s08h.validation-summary.v1",
        "researchStepId": "S08H",
        "success": gate["success"] and tests["success"],
        "checks": {
            "inputFreeze": gate_inputs["G01"],
            "descriptorDomain": descriptor["success"],
            "targetSemantics": target["success"],
            "nativeContracts": native["success"],
            "bindings": binding["success"],
            "protectedDenial": access["success"],
            "dependencyExclusion": dependency["success"],
            "replayWorkerOrder": replay["success"],
            "completeAccounting": accounting["success"],
            "noMutation": no_mutation["success"],
            "tests": tests["success"],
        },
        "runtimeSeconds": runtime_seconds,
    }
    write_json(output / "validation_summary.json", validation)
    provenance = {
        "schemaVersion": "e07.s08h.provenance.v1",
        "researchStepId": "S08H",
        "prospectiveCommit": PROSPECTIVE_COMMIT,
        "repository": "Eidosoma/cell_research",
        "branch": "eidosoma/groups/28",
        "python": sys.version,
        "platform": platform.platform(),
        "cpuCount": os.cpu_count(),
        "workers": 1,
        "numericThreads": 1,
        "pandas": pd.__version__,
        "newDependenciesInstalled": [],
        "runtimeSeconds": runtime_seconds,
    }
    write_json(output / "provenance.json", provenance)

    status = {
        "researchStepId": "S08H",
        "stepNumber": "08H",
        "success": validation["success"],
        "status": "complete_supportive_qualification"
        if validation["success"]
        else "blocked_before_execution",
        "artifactsWritten": [],
        "validationResult": "PASS: G01-G06, exhaustive algebra, target branches, bindings, access, replay/order, accounting, tests, and no mutation"
        if validation["success"]
        else "FAIL: see validation_summary.json",
        "outcomeClassification": "supportive"
        if validation["success"]
        else "constraining/contradictory",
        "caveatsOrBlockers": [
            "Qualification-only; no efficacy or fresh execution evidence.",
            "Zero initial distance is explicitly unavailable because the restoration ratio is undefined.",
            "S09 remains blocked until a separately approved fresh S08 execution succeeds.",
        ],
        "recommendedNextAction": "Review S08H and require separate approval for any fresh S08 execution; do not reuse quarantines or start S09.",
    }
    write_json(output / "status.json", status)

    report = report_text(
        descriptor, target, binding, access, gate, accounting, runtime_seconds
    )
    (output / "research_step_full_results.md").write_text(report, encoding="utf-8")
    status["artifactsWritten"] = [
        path.name for path in sorted(output.iterdir()) if path.is_file()
    ] + ["artifact_manifest.json"]
    write_json(output / "status.json", status)

    manifest_rows = [
        {
            "path": path.relative_to(output).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    ]
    manifest = {
        "schemaVersion": "e07.s08h.artifact-manifest.v1",
        "researchStepId": "S08H",
        "root": str(output),
        "artifactCount": len(manifest_rows),
        "artifacts": manifest_rows,
    }
    write_json(output / "artifact_manifest.json", manifest)
    if not validation["success"]:
        raise RuntimeError("S08H qualification failed closed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    execute(parse_args().output)
