"""E05 S14 regeneration-benchmark release and validation helpers.

S14 packages previously frozen evidence.  It does not expose a composition
search API and does not pool the five competency axes into a scalar score.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from typing import Any

import jsonschema

from causal_simulator.architectures import ArchitectureExecutionContract
from causal_simulator.schedulers import (
    SchedulerExecutionContract,
    SchedulerFamily,
    run_scheduled_architecture,
)
from reference_simulator.model import LEDGER_FIELDS, canonical_json_bytes

from .chimeric import (
    build_composition_scenario,
    exact_replay_native,
    run_native_recovery,
)
from .tasks import Checkpoint, _checkpoint_hash, stabilize_achieved_checkpoint
from .transfer import apply_transfer_lesion


BENCHMARK_VERSION = "E05-regeneration-benchmark-v1"
SPEC_SCHEMA_VERSION = "e05.s14.regeneration-benchmark-spec.v1"

CORE_S13_PORTFOLIOS = frozenset(
    {"pure_bubble", "pure_insertion", "chimera_bubble_insertion"}
)
OUTCOME_FIELDS = frozenset(
    {
        "success",
        "stopReason",
        "phaseActivationCount",
        "restrictedTime",
        "initialDistance",
        "finalDistance",
        "maximumDistance",
        "distanceOvershoot",
        "normalizedErrorAuc",
        "resultDigest",
        "exactReplayPass",
        "validationPass",
        "trajectoryExecuted",
        "executionStatus",
    }
)

SPEC_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": True,
    "required": [
        "schemaVersion",
        "researchStepId",
        "benchmarkVersion",
        "frozenAtUtc",
        "implementationStateAtFreeze",
        "frozenQuestion",
        "suiteAndCompetencyContract",
        "coreExtendedSplit",
        "immutableInputs",
        "taskRelease",
        "e07UsageSplit",
        "metricContract",
        "smokeAndReproduction",
        "validation",
        "releaseLayout",
        "classificationRule",
        "claimBoundary",
    ],
    "properties": {
        "schemaVersion": {"const": SPEC_SCHEMA_VERSION},
        "researchStepId": {"const": "S14"},
        "benchmarkVersion": {"const": BENCHMARK_VERSION},
    },
}


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def validate_benchmark_spec(specification: Mapping[str, Any]) -> None:
    """Validate the pre-packaging S14 contract and its no-search boundary."""

    jsonschema.Draft202012Validator(SPEC_SCHEMA).validate(specification)
    suite = specification["suiteAndCompetencyContract"]
    if len(suite["benchmarkSuites"]) != 4:
        raise ValueError("S14 must retain exactly four benchmark suites")
    if len(suite["competencyAxes"]) != 5 or suite["aggregateScorePermitted"]:
        raise ValueError("S14 must report five separate, nonaggregated competency axes")
    split = specification["coreExtendedSplit"]
    if set(split["core"]["s13Portfolios"]) != CORE_S13_PORTFOLIOS:
        raise ValueError("S14 core S13 portfolios changed")
    if split["core"]["s13PlacementMaps"] != [0]:
        raise ValueError("S14 core must use only frozen placement map 0")
    membership = set(split["membershipInputs"])
    prohibited = set(split["prohibitedMembershipInputs"])
    if membership & prohibited:
        raise ValueError("S14 split uses a prohibited outcome input")
    if not split["assignmentFrozenBeforePackaging"]:
        raise ValueError("S14 split was not frozen before packaging")
    if "No composition" not in split["noCompositionSearch"]:
        raise ValueError("S14 must prohibit outcome-driven composition search")
    e07 = specification["e07UsageSplit"]
    if not e07["outcomeInputsProhibited"]:
        raise ValueError("S14 E07 usage split must prohibit outcome inputs")
    expected = specification["validation"]["expectedCounts"]
    required_counts = {
        "coreS13Sources": 96,
        "coreS13MainRows": 768,
        "extendedS13Sources": 256,
        "extendedS13MainRows": 2048,
        "allS13MainRows": 2816,
        "s13ThresholdAssignments": 2016,
        "s13ThresholdResultRows": 672,
        "s13StoppedThresholdRows": 1344,
        "s13SourceCompetingTerminals": 64,
        "s12MatchedPairs": 2496,
        "s12AssignedRuns": 4992,
        "s12ChangedTaskFailureOrCensorRuns": 2120,
        "s12SubgroupPrecisionRows": 137,
    }
    if expected != required_counts:
        raise ValueError("S14 frozen accounting totals changed")


def s13_partition(row: Mapping[str, Any]) -> str:
    """Return the outcome-blind package partition for one S13 main row."""

    portfolio = str(row["portfolioId"])
    placement_map = int(row["placementMap"])
    return (
        "core"
        if portfolio in CORE_S13_PORTFOLIOS and placement_map == 0
        else "extended"
    )


def split_is_outcome_invariant(row: Mapping[str, Any]) -> bool:
    """Prove that arbitrary outcome-field mutations do not change membership."""

    baseline = s13_partition(row)
    mutated = dict(row)
    for index, field in enumerate(sorted(OUTCOME_FIELDS)):
        mutated[field] = f"mutated-outcome-{index}"
    return s13_partition(mutated) == baseline


def _checkpoint_from_development(scenario: Any, result: Any) -> Checkpoint:
    occupancy = tuple(result.summary["finalOccupancy"])
    cursors = dict(result.final_state["selectionCursors"])
    activation_count = int(result.summary["activationCount"])
    streams = {
        str(key): int(value)
        for key, value in result.final_state["streamCounters"].items()
    }
    ledger = {
        field: int(result.final_state["ledger"][field]) for field in LEDGER_FIELDS
    }
    return Checkpoint(
        occupancy,
        tuple(sorted(cursors.items())),
        activation_count,
        tuple(sorted(streams.items())),
        tuple((field, ledger[field]) for field in LEDGER_FIELDS),
        0,
        _checkpoint_hash(
            scenario.scenario_id,
            occupancy,
            cursors,
            activation_count,
            streams,
            ledger,
        ),
    )


def run_fresh_core_smoke(specification: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild and exactly replay the frozen S13 pure-Bubble core fixture."""

    validate_benchmark_spec(specification)
    fixture = specification["smokeAndReproduction"]["freshScenario"]
    n = int(fixture["n"])
    direction = str(fixture["direction"])
    replicate = int(fixture["replicateOrdinal"])
    portfolio = str(fixture["portfolioId"])
    placement_map = int(fixture["placementMap"])
    task_id = str(fixture["taskId"])
    development_budget = 100 * n * n
    recovery_budget = 100 * n * n
    scenario, metadata = build_composition_scenario(
        n=n,
        portfolio_id=portfolio,
        direction=direction,
        replicate=replicate,
        placement_map=placement_map,
        max_activations=2 * development_budget + 40 * n,
    )
    architecture = ArchitectureExecutionContract.distributed_local()
    scheduler = SchedulerExecutionContract(
        SchedulerFamily.UNIFORM_RANDOM_ACTIVATION
    )
    development = run_scheduled_architecture(
        scenario, architecture, scheduler, trace_mode="digest"
    )
    development_replay = run_scheduled_architecture(
        scenario, architecture, scheduler, trace_mode="digest"
    )
    development_replay_pass = (
        development.result.to_json_bytes()
        == development_replay.result.to_json_bytes()
    )
    if not development.result.summary["completed"]:
        raise AssertionError("fresh S14 core source did not complete development")
    if int(development.result.summary["activationCount"]) > development_budget:
        raise AssertionError("fresh S14 core source exceeded the frozen budget")
    checkpoint = _checkpoint_from_development(scenario, development.result)
    checkpoint, stabilization = stabilize_achieved_checkpoint(scenario, checkpoint)
    checkpoint_replay, stabilization_replay = stabilize_achieved_checkpoint(
        scenario, _checkpoint_from_development(scenario, development.result)
    )
    stabilization_replay_pass = (
        checkpoint.to_dict() == checkpoint_replay.to_dict()
        and stabilization == stabilization_replay
    )
    if not stabilization["success"]:
        raise AssertionError("fresh S14 core source failed stabilization")
    lesion = apply_transfer_lesion(
        scenario,
        checkpoint,
        lesion_type=task_id,
        location="central",
    )
    recovery = run_native_recovery(
        scenario,
        checkpoint,
        postinjury_occupancy=lesion["postOccupancy"],
        lesion_state_hash=lesion["lesionStateHash"],
        recovery_budget=recovery_budget,
    )
    exact_replay_native(
        scenario,
        checkpoint,
        postinjury_occupancy=lesion["postOccupancy"],
        lesion_state_hash=lesion["lesionStateHash"],
        recovery_budget=recovery_budget,
        expected=recovery,
    )
    return {
        "schemaVersion": "e05.s14.fresh-smoke.v1",
        "researchStepId": "S14",
        "fixture": dict(fixture),
        "sourceScenarioId": scenario.scenario_id,
        "sourceCheckpointHash": checkpoint.state_hash,
        "baseDrawPairingId": metadata["baseDrawPairingId"],
        "policyAssignmentSha256": metadata["policyAssignmentSha256"],
        "developmentActivationCount": int(
            development.result.summary["activationCount"]
        ),
        "developmentEventDigest": development.result.event_digest,
        "stabilizationOpportunities": int(stabilization["opportunities"]),
        "lesionStateHash": lesion["lesionStateHash"],
        "lesionWindowLength": int(lesion["windowLength"]),
        "lesionPostDistance": int(lesion["postDistance"]),
        "success": bool(recovery["success"]),
        "stopReason": str(recovery["stopReason"]),
        "phaseActivationCount": int(recovery["phaseActivationCount"]),
        "restrictedTime": int(recovery["restrictedTime"]),
        "initialDistance": int(recovery["initialDistance"]),
        "finalDistance": int(recovery["finalDistance"]),
        "resultDigest": str(recovery["resultDigest"]),
        "developmentReplayPass": development_replay_pass,
        "stabilizationReplayPass": stabilization_replay_pass,
        "recoveryReplayPass": True,
        "allRuntimeValidationPass": all(recovery["validation"].values()),
    }


def required_columns_present(
    available: Sequence[str], required: Sequence[str]
) -> bool:
    return set(required).issubset(set(available))
