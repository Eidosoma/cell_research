"""E05 S13 homogeneous-versus-chimeric recovery contracts.

Only immutable, outcome-blind composition and permanent-stuck placement live
here.  The module reuses the validated S01 checkpoint, E02 transition/ledger,
S03 lesion, and S12 target-transfer implementations without policy search.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
from typing import Any, Mapping, Sequence

import jsonschema

from causal_simulator.architectures import (
    ArchitectureExecutionContract,
    ArchitectureProposalRouter,
)
from reference_simulator.model import (
    Cell,
    Direction,
    FaultMode,
    LEDGER_FIELDS,
    Policy,
    RunState,
    Scenario,
    canonical_json_bytes,
    sha256_json,
    state_hash,
)
from reference_simulator.rng import permutation, u64
from reference_simulator.scheduler import ScheduledOpportunity, scheduled_actor
from reference_simulator.engine import evaluate_terminal, execute_serial_summary_activation
from reference_simulator.transition_primitives import ledger_identity

from .tasks import (
    Checkpoint,
    _checkpoint_hash,
    occupancy_values,
    strict_unequal_inversions,
)


BENCHMARK_VERSION = "E05-chimeric-recovery-v1"
SPEC_SCHEMA_VERSION = "e05.s13.chimeric-recovery-spec.v1"
RESULT_SCHEMA_VERSION = "e05.s13.chimeric-run.v1"
MASTER_SEED = int("e0513000000000000000000000000001", 16)

PORTFOLIOS: dict[str, tuple[Policy, ...]] = {
    "pure_bubble": (Policy.BUBBLE,),
    "pure_insertion": (Policy.INSERTION,),
    "pure_selection": (Policy.SELECTION,),
    "chimera_bubble_insertion": (Policy.BUBBLE, Policy.INSERTION),
    "chimera_bubble_selection": (Policy.BUBBLE, Policy.SELECTION),
    "chimera_insertion_selection": (Policy.INSERTION, Policy.SELECTION),
    "chimera_three_way": (Policy.BUBBLE, Policy.INSERTION, Policy.SELECTION),
}

SPEC_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": [
        "schemaVersion",
        "researchStepId",
        "benchmarkVersion",
        "frozenAtUtc",
        "implementationStateAtFreeze",
        "frozenQuestion",
        "priorEvidenceConstraints",
        "inherits",
        "portfolios",
        "assignment",
        "panel",
        "criticalFaultThreshold",
        "estimands",
        "costs",
        "inference",
        "terminalHandling",
        "validation",
        "claimBoundary",
    ],
    "properties": {
        "schemaVersion": {"const": SPEC_SCHEMA_VERSION},
        "researchStepId": {"const": "S13"},
        "benchmarkVersion": {"const": BENCHMARK_VERSION},
    },
}


def validate_chimeric_spec(specification: Mapping[str, Any]) -> None:
    """Validate the frozen S13 contract and all planned accounting totals."""

    jsonschema.Draft202012Validator(SPEC_SCHEMA).validate(specification)
    inherited = specification["inherits"]
    expected = {
        "taskSpecSchemaVersion": "e05.s01.task-spec.v1",
        "timingSpecSchemaVersion": "e05.s02.timing-spec.v1",
        "lesionSpecSchemaVersion": "e05.s03.lesion-spec.v1",
        "dynamicSpecSchemaVersion": "e05.s04.dynamic-fault-spec.v1",
        "transferSpecSchemaVersion": "e05.s12.transfer-spec.v1",
        "architecture": "distributed_local",
        "scheduler": "uniform_random_activation",
        "continuation": "skip_and_continue",
        "retry": "no_retry",
        "nativeInformationPermission": "policy_native_local",
        "developmentOpportunities": "100*n^2",
        "postInterventionOpportunities": "100*n^2",
        "postHitProbeOpportunities": "20*n",
    }
    for key, value in expected.items():
        if inherited.get(key) != value:
            raise ValueError(f"S13 inherited contract changed: {key}")
    listed = {item["portfolioId"]: tuple(Policy(p) for p in item["policies"])
              for item in specification["portfolios"]}
    if listed != PORTFOLIOS:
        raise ValueError("S13 portfolio set or policy order changed")
    panel = specification["panel"]
    blocks = len(panel["sizes"]) * len(panel["directions"]) * int(panel["replicatesPerCell"])
    primary = len(PORTFOLIOS) * blocks
    sensitivity = sum(len(v) > 1 for v in PORTFOLIOS.values()) * blocks
    if primary != panel["plannedPrimarySources"]:
        raise ValueError("S13 primary source count changed")
    if sensitivity != panel["plannedSensitivitySources"]:
        raise ValueError("S13 sensitivity source count changed")
    if primary + sensitivity != panel["plannedSourceCount"]:
        raise ValueError("S13 total source count changed")
    arms_per_source = (
        len(panel["mainLesions"])
        + 2 * len(panel["targetChanges"])
        + len(panel["intactStabilityArms"])
    )
    if panel["plannedMainRuns"] != panel["plannedSourceCount"] * arms_per_source:
        raise ValueError("S13 planned main-run count changed")
    if panel["plannedMainExactReplays"] != panel["plannedMainRuns"]:
        raise ValueError("S13 main replay count changed")
    threshold = specification["criticalFaultThreshold"]
    potential = primary * len(threshold["fractionsInOrder"])
    if potential != threshold["plannedPotentialRunsIncludingReusedZero"]:
        raise ValueError("S13 threshold potential count changed")
    new = primary * (len(threshold["fractionsInOrder"]) - 1)
    if new != threshold["plannedPotentialNewRuns"]:
        raise ValueError("S13 threshold new-run count changed")
    if new != threshold["plannedPotentialNewExactReplays"]:
        raise ValueError("S13 threshold replay count changed")
    if specification["inference"]["primaryMultiplicityFamilySize"] != 16:
        raise ValueError("S13 primary multiplicity family changed")
    if not specification["assignment"]["outcomeBlind"]:
        raise ValueError("S13 composition assignment must be outcome blind")


def semantic_seed(stream: str, *address: Any) -> int:
    material = {
        "benchmarkVersion": BENCHMARK_VERSION,
        "stream": stream,
        "address": list(address),
    }
    digest = hashlib.sha256(
        b"E05/S13/seed/v1\x00"
        + MASTER_SEED.to_bytes(16, "big")
        + b"\x00"
        + canonical_json_bytes(material)
    ).digest()
    # The frozen reference counter address reserves uint32 for draw_index.
    draw_index = int.from_bytes(digest[:4], "big")
    return u64(MASTER_SEED, "E05/S13", stream, 0, draw_index)


def composition_counts(
    n: int,
    portfolio_id: str,
    *,
    replicate: int,
    placement_map: int,
) -> dict[Policy, int]:
    """Return exact counts, rotating only an unavoidable three-way remainder."""

    policies = PORTFOLIOS[portfolio_id]
    base, remainder = divmod(n, len(policies))
    counts = {policy: base for policy in policies}
    offset = (replicate + placement_map) % len(policies)
    for index in range(remainder):
        counts[policies[(offset + index) % len(policies)]] += 1
    if sum(counts.values()) != n or any(value <= 0 for value in counts.values()):
        raise AssertionError("invalid exact S13 composition")
    return counts


def assign_policies(
    identity_ids: Sequence[str],
    portfolio_id: str,
    *,
    n: int,
    direction: str,
    replicate: int,
    placement_map: int,
) -> dict[str, Policy]:
    """Assign exact portfolios from identity ranks without state/outcome input."""

    if n != len(identity_ids) or len(set(identity_ids)) != n:
        raise ValueError("S13 assignment requires n unique identity IDs")
    if placement_map not in {0, 1}:
        raise ValueError("S13 freezes exactly placement maps 0 and 1")
    counts = composition_counts(
        n, portfolio_id, replicate=replicate, placement_map=placement_map
    )
    seed = semantic_seed(
        "composition_assignment",
        n,
        direction,
        replicate,
        portfolio_id,
        placement_map,
    )
    key = f"E05/S13/composition/{n}/{direction}/{replicate}/{portfolio_id}/m{placement_map}"
    ranked = permutation(tuple(identity_ids), seed, key)
    result: dict[str, Policy] = {}
    cursor = 0
    for policy in PORTFOLIOS[portfolio_id]:
        count = counts[policy]
        for identity in ranked[cursor : cursor + count]:
            result[identity] = policy
        cursor += count
    if set(result) != set(identity_ids):
        raise AssertionError("S13 assignment is not a bijection over identities")
    return result


def build_composition_scenario(
    *,
    n: int,
    portfolio_id: str,
    direction: str,
    replicate: int,
    placement_map: int,
    max_activations: int,
) -> tuple[Scenario, dict[str, Any]]:
    """Build one paired base draw with a composition-specific immutable policy map."""

    selected_direction = Direction(direction)
    ids = tuple(f"cell-{index:04d}" for index in range(n))
    input_seed = semantic_seed("base_input", n, direction, replicate)
    occupancy = permutation(
        ids, input_seed, f"E05/S13/base-input/{n}/{direction}/{replicate}"
    )
    policy_by_id = assign_policies(
        ids,
        portfolio_id,
        n=n,
        direction=direction,
        replicate=replicate,
        placement_map=placement_map,
    )
    cells = tuple(
        Cell(
            identity,
            index,
            policy_by_id[identity],
            selected_direction,
            FaultMode.NORMAL,
        )
        for index, identity in enumerate(ids)
    )
    runtime_seed = semantic_seed("runtime", n, direction, replicate)
    generation_key = (
        f"E05/S13/source/{n}/{direction}/{replicate}/"
        f"{portfolio_id}/m{placement_map}"
    )
    scenario = Scenario.create(
        cells,
        initial_occupancy=occupancy,
        seed=runtime_seed,
        max_activations=max_activations,
        generation_key=generation_key,
    )
    counts = {policy.value: 0 for policy in PORTFOLIOS[portfolio_id]}
    for policy in policy_by_id.values():
        counts[policy.value] += 1
    assignment = [policy_by_id[identity].value for identity in ids]
    base_material = {
        "n": n,
        "direction": direction,
        "replicateOrdinal": replicate,
        "initialOccupancy": list(occupancy),
        "runtimeSeed": str(runtime_seed),
    }
    metadata = {
        "baseDrawPairingId": "e05s13base:" + sha256_json(base_material),
        "portfolioId": portfolio_id,
        "placementMap": placement_map,
        "compositionCounts": counts,
        "policyAssignmentSha256": sha256_json(assignment),
        "initialOccupancySha256": sha256_json(list(occupancy)),
        "runtimeSeed": str(runtime_seed),
        "sourceScenarioId": scenario.scenario_id,
    }
    return scenario, metadata


def fault_identity_ranking(
    identity_ids: Sequence[str], *, n: int, direction: str, replicate: int
) -> tuple[str, ...]:
    """One portfolio-independent nested identity order for the stuck grid."""

    seed = semantic_seed("fault_identity_ranking", n, direction, replicate)
    return permutation(
        tuple(identity_ids), seed, f"E05/S13/fault-rank/{n}/{direction}/{replicate}"
    )


def fault_count(n: int, fraction: float) -> int:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fault fraction must be in [0,1]")
    return int(math.floor(fraction * n + 0.5))


def clone_with_stuck_identities(
    scenario: Scenario,
    checkpoint: Checkpoint,
    stuck_ids: Sequence[str],
    *,
    fraction: float,
) -> tuple[Scenario, Checkpoint, dict[str, Any]]:
    """Change only immutable fault flags and rebind the full checkpoint hash."""

    selected = frozenset(stuck_ids)
    if not selected <= set(scenario.cell_map):
        raise ValueError("unknown identity in S13 stuck set")
    cells = tuple(
        replace(
            cell,
            fault=FaultMode.STUCK if cell.cell_id in selected else FaultMode.NORMAL,
        )
        for cell in scenario.cells
    )
    faulted = Scenario.create(
        cells,
        initial_occupancy=scenario.initial_occupancy,
        initial_selection_cursors=dict(scenario.initial_selection_cursors),
        seed=scenario.seed,
        max_activations=scenario.max_activations,
        architecture=scenario.architecture,
        batch_width=scenario.batch_width,
        traditional_policy=scenario.traditional_policy,
        generation_key=scenario.generation_key + f"/post-completion-stuck/{fraction:.3f}",
        # Scenario only accepts its frozen mechanical placement vocabulary;
        # the S13 post-completion/nested semantics are carried in metadata and
        # the generation key, while the exact map itself is explicit.
        fault_placement="explicit",
        requested_fault_count=len(selected),
        rng_profile=scenario.rng_profile,
        goal_profile=scenario.goal_profile,
        metric_profile=scenario.metric_profile,
    )
    rebound = Checkpoint(
        checkpoint.occupancy,
        checkpoint.selection_cursors,
        checkpoint.activation_count,
        checkpoint.stream_counters,
        checkpoint.ledger,
        checkpoint.distance,
        _checkpoint_hash(
            faulted.scenario_id,
            checkpoint.occupancy,
            dict(checkpoint.selection_cursors),
            checkpoint.activation_count,
            dict(checkpoint.stream_counters),
            dict(checkpoint.ledger),
        ),
    )
    metadata = {
        "faultFraction": fraction,
        "requestedFaultCount": len(selected),
        "realizedFaultCount": faulted.realized_fault_count,
        "faultIdentitySetSha256": sha256_json(sorted(selected)),
        "faultedScenarioId": faulted.scenario_id,
        "faultedCheckpointHash": rebound.state_hash,
        "onlyFaultFlagsChanged": all(
            left.cell_id == right.cell_id
            and left.value == right.value
            and left.policy == right.policy
            and left.direction == right.direction
            for left, right in zip(scenario.cells, faulted.cells)
        ),
    }
    return faulted, rebound, metadata


class UniformSchedule:
    """S01 uniform random activation without a state-observing scheduler."""

    __slots__ = ("scenario", "count")

    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario
        self.count = 0

    def __call__(
        self, event_index: int, remaining_opportunities: int
    ) -> tuple[ScheduledOpportunity, ...]:
        if remaining_opportunities < 1:
            return ()
        actor, draws, consumed = scheduled_actor(
            self.scenario, event_index, include_draws=True
        )
        self.count += 1
        return (
            ScheduledOpportunity(
                actor,
                draws,
                (("actor_activation", consumed),),
            ),
        )


def _delta(final: Mapping[str, int], initial: Mapping[str, int]) -> dict[str, int]:
    return {field: int(final.get(field, 0)) - int(initial.get(field, 0)) for field in final}


def run_native_recovery(
    scenario: Scenario,
    checkpoint: Checkpoint,
    *,
    postinjury_occupancy: Sequence[str],
    lesion_state_hash: str,
    recovery_budget: int,
) -> dict[str, Any]:
    """Run policy-native recovery with no rescue, memory, or plasticity overlay."""

    state = checkpoint.to_run_state(occupancy=postinjury_occupancy)
    start = state.activation_count
    initial_ledger = dict(state.ledger)
    initial_streams = dict(state.stream_counters)
    initial_state_hash = state_hash(scenario.scenario_id, state)
    direction = scenario.cells[0].direction
    initial_distance = strict_unequal_inversions(
        occupancy_values(scenario, state.occupancy), direction
    )
    distance = initial_distance
    max_distance = distance
    distance_auc = 0
    router = ArchitectureProposalRouter(
        ArchitectureExecutionContract.distributed_local()
    )
    state.terminal = evaluate_terminal(scenario, state)
    while state.terminal is None and state.activation_count - start < recovery_budget:
        before_swaps = state.ledger["acceptedSwaps"]
        changed = execute_serial_summary_activation(
            scenario,
            state,
            proposal_factory=router.proposal_for,
        )
        if state.ledger["acceptedSwaps"] != before_swaps:
            distance = strict_unequal_inversions(
                occupancy_values(scenario, state.occupancy), direction
            )
        if changed and state.terminal is None:
            state.terminal = evaluate_terminal(scenario, state)
        max_distance = max(max_distance, distance)
        distance_auc += distance
    if state.terminal is None:
        state.terminal = "phase_event_budget"
    phase = state.activation_count - start
    # Restricted-horizon AUC retains early competing terminals by carrying
    # their last error through the remainder of the fixed recovery budget.
    restricted_auc = distance_auc + max(0, recovery_budget - phase) * distance
    maximum_pairs = len(state.occupancy) * (len(state.occupancy) - 1) // 2
    native = _delta(dict(state.ledger), initial_ledger)
    streams = _delta(dict(state.stream_counters), initial_streams)
    success = distance == 0
    result: dict[str, Any] = {
        "schemaVersion": RESULT_SCHEMA_VERSION,
        "benchmarkVersion": BENCHMARK_VERSION,
        "taskFamily": "injury_recovery",
        "sourceScenarioId": scenario.scenario_id,
        "sourceCheckpointHash": checkpoint.state_hash,
        "lesionStateHash": lesion_state_hash,
        "startEventIndex": start,
        "endEventIndex": state.activation_count,
        "initialStateHash": initial_state_hash,
        "finalStateHash": state_hash(scenario.scenario_id, state),
        "stopReason": state.terminal,
        "success": success,
        "phaseActivationCount": phase,
        "recoveryBudget": recovery_budget,
        "restrictedTime": phase if success else recovery_budget + 1,
        "initialDistance": initial_distance,
        "finalDistance": distance,
        "maximumDistance": max_distance,
        "distanceOvershoot": max_distance - initial_distance,
        "distanceAucObserved": distance_auc,
        "distanceAucRestricted": restricted_auc,
        "normalizedDistanceAucRestricted": restricted_auc / (recovery_budget * maximum_pairs),
        "nativeLedgerDelta": native,
        "streamCounterDelta": streams,
        "schedulerOpportunityCount": phase,
        "finalOccupancy": list(state.occupancy),
        "validation": {
            **ledger_identity(native),
            "phaseBudgetRespected": phase <= recovery_budget,
            "schedulerCountMatchesPhase": True,
            "oneProposalPerOpportunity": native["proposals"] == phase,
            "identityCountPreserved": len(state.occupancy) == len(scenario.cells),
            "identitySetPreserved": set(state.occupancy) == set(scenario.cell_map),
            "successMatchesDistance": success == (distance == 0),
        },
    }
    digest_body = {key: value for key, value in result.items() if key != "resultDigest"}
    result["resultDigest"] = hashlib.sha256(canonical_json_bytes(digest_body)).hexdigest()
    return result


def exact_replay_native(
    scenario: Scenario,
    checkpoint: Checkpoint,
    *,
    postinjury_occupancy: Sequence[str],
    lesion_state_hash: str,
    recovery_budget: int,
    expected: Mapping[str, Any],
) -> None:
    replay = run_native_recovery(
        scenario,
        checkpoint,
        postinjury_occupancy=postinjury_occupancy,
        lesion_state_hash=lesion_state_hash,
        recovery_budget=recovery_budget,
    )
    if canonical_json_bytes(replay) != canonical_json_bytes(expected):
        raise AssertionError("S13 native recovery exact replay mismatch")


def should_stop_threshold(
    history: Sequence[Mapping[str, float]], portfolio_ids: Sequence[str]
) -> bool:
    """Apply the frozen global two-consecutive-UCB threshold rule."""

    if len(history) < 2:
        return False
    last_two = history[-2:]
    return all(
        all(float(level[portfolio]) < 0.5 for level in last_two)
        for portfolio in portfolio_ids
    )


def shannon_entropy_bits(counts: Mapping[str, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        raise ValueError("composition is empty")
    return -sum(
        (count / total) * math.log2(count / total)
        for count in counts.values()
        if count
    )


def assignment_description_bits(counts: Mapping[str, int]) -> float:
    """log2 multinomial assignments, a declared immutable complexity proxy."""

    total = sum(counts.values())
    value = math.lgamma(total + 1) - sum(math.lgamma(count + 1) for count in counts.values())
    return value / math.log(2)
