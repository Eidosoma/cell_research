"""Frozen S09 baseline formation scenarios, metrics, and CPU execution.

S09 keeps the S07 CPU oracle authoritative.  Global S01/S02 evaluation is a
read-only offline audit callback and never changes a policy decision, state,
stop rule, or budget.  Production condition sizes remain below the frozen S07
GPU crossover, so the exact CPU fallback is used rather than widening GPU
control-plane authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .elapsed_clock import (
    build_first_completion_projection,
    genesis_commitment,
    parse_and_validate_elapsed_clock_record,
    summarize_elapsed_clock_records,
)
from .episode_origin_clock import (
    EPISODE_ORIGIN_CLOCK_RECORD_VERSION,
    build_episode_origin_first_completion_projection,
    episode_origin_genesis_commitment,
    parse_and_validate_episode_origin_clock_record,
    summarize_episode_origin_clock_records,
)
from .engine import (
    EngineContext,
    EpisodeDefinition,
    canonical_episode_result_bytes,
    load_engine_context,
    run_cpu_episode,
)
from .environments import Environment, parse_environment_spec
from .grammar import RelationalGrammar, score_grid
from .movements import (
    MovementState,
    initial_movement_state,
    movement_state_sha256,
    validate_state_against_environment,
)
from .targets import (
    Grid,
    TargetDefinition,
    evaluate_success,
    exact_equivalence_orbit,
    load_target_catalog,
)


ROOT = Path(__file__).resolve().parents[2]
BASELINE_CATALOG_VERSION = "e06.s09.baseline-catalog.v1"
RUN_RESULT_VERSION = "e06.s09.baseline-run.v1"
MASTER_SEED_HEX = "0xe0609000000000000000000000000001"
UNIVERSAL_POLICIES = {
    "greedy_neighbor_satisfaction_v1",
    "exploration_v1",
    "memory_based_recovery_v1",
    "conflict_avoidance_v1",
}
GRAMMAR_POLICIES = {
    "greedy_neighbor_satisfaction_v1",
    "memory_based_recovery_v1",
    "conflict_avoidance_v1",
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _sha256(domain: str, value: Any) -> str:
    return hashlib.sha256(
        domain.encode("ascii") + b"\x00" + _canonical_bytes(value)
    ).hexdigest()


def _rank(domain: str, address: str, value: str) -> bytes:
    return hashlib.sha256(
        domain.encode("ascii")
        + b"\x00"
        + address.encode("utf-8")
        + b"\x00"
        + value.encode("utf-8")
    ).digest()


def _rectangle_expected(rows: int, columns: int, vacancy_count: int) -> dict[str, Any]:
    degree_histogram: Counter[int] = Counter()
    for row in range(rows):
        for column in range(columns):
            degree = 0
            for delta_row, delta_column in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                other = row + delta_row, column + delta_column
                degree += int(0 <= other[0] < rows and 0 <= other[1] < columns)
            degree_histogram[degree] += 1
    boundary_count = sum(
        row in {0, rows - 1} or column in {0, columns - 1}
        for row in range(rows)
        for column in range(columns)
    )
    return {
        "siteCount": rows * columns,
        "occupiableSiteCount": rows * columns,
        "edgeCount": rows * (columns - 1) + (rows - 1) * columns,
        "componentCount": 1,
        "degreeHistogram": {
            str(key): value for key, value in sorted(degree_histogram.items())
        },
        "vacancyCount": vacancy_count,
        "obstacleCount": 0,
        "fixedBoundaryCount": 0,
        "naturalBoundarySiteCount": boundary_count,
        "signaledSiteCount": boundary_count,
    }


def build_target_environment(
    target: TargetDefinition, grammar: RelationalGrammar
) -> Environment:
    """Compile a calibrated unobstructed bounded square target environment."""

    rows, columns = target.shape
    vacancy_count = int(target.cell_type_counts[target.vacancy_label])
    raw = {
        "environmentId": f"s09_target_{target.target_id}",
        "title": f"S09 calibrated baseline for {target.title}",
        "geometry": "square",
        "boundaryMode": "bounded",
        "occupancyMode": ("vacancy_enabled" if vacancy_count else "fully_occupied"),
        "vacancyLabel": target.vacancy_label,
        "obstacleToken": "#",
        "generator": {"kind": "rectangle", "rows": rows, "columns": columns},
        "obstacles": [],
        "fixedBoundary": {"mode": "none", "sites": [], "token": None},
        "boundarySignal": {
            "mode": "domain_edge_flags",
            "observable": True,
            "includeObstacleContact": False,
            "includeFixedRole": False,
        },
        "initialState": {
            "defaultToken": target.vacancy_label,
            "overrides": {},
            "rows": ["".join(row) for row in target.grid],
        },
        "targetBinding": {
            "targetId": target.target_id,
            "grammarId": grammar.grammar_id,
            "evaluationProfile": "s01_s02_conjunctive_square_v1",
        },
        "requireConnected": True,
        "notes": (
            "S09 target-derived fixture; target completion is calibrated only for "
            "this unobstructed bounded square topology."
        ),
        "expected": _rectangle_expected(rows, columns, vacancy_count),
    }
    return parse_environment_spec(raw)


@lru_cache(maxsize=1)
def load_baseline_assets() -> tuple[
    EngineContext,
    Mapping[str, TargetDefinition],
    Mapping[str, RelationalGrammar],
    Mapping[str, Environment],
]:
    """Load repository-backed contracts once per worker process."""

    context = load_engine_context(
        ROOT / "configs/morphologies/engine_catalog.yaml",
        environment_catalog=ROOT / "configs/morphologies/environment_catalog.yaml",
        policy_catalog=ROOT / "configs/morphologies/policy_catalog.yaml",
        grammar_catalog=ROOT / "configs/morphologies/grammar_catalog.yaml",
        channel_catalog=ROOT / "configs/morphologies/control_channel_catalog.yaml",
    )
    _, target_values = load_target_catalog(
        ROOT / "configs/morphologies/target_catalog.yaml"
    )
    targets = {item.target_id: item for item in target_values}
    grammars = dict(context.grammars)
    environments = {
        target_id: build_target_environment(
            target,
            next(item for item in grammars.values() if item.target_id == target_id),
        )
        for target_id, target in targets.items()
    }
    context = replace(
        context,
        environments={
            **context.environments,
            **{item.environment_id: item for item in environments.values()},
        },
    )
    return context, targets, grammars, environments


def eligible_policies(target_id: str, catalog: Mapping[str, Any]) -> tuple[str, ...]:
    policies = list(catalog["policyEligibility"]["universalByTarget"])
    for policy_id, target_ids in catalog["policyEligibility"][
        "targetRestricted"
    ].items():
        if target_id in target_ids:
            policies.append(policy_id)
    return tuple(sorted(policies))


def _coordinate_sites(environment: Environment) -> tuple[str, ...]:
    return tuple(
        site.site_id
        for site in sorted(
            environment.occupiable_sites, key=lambda item: item.coordinate
        )
    )


def state_grid(environment: Environment, state: MovementState) -> Grid:
    rows, columns = (
        int(environment.generator["rows"]),
        int(environment.generator["columns"]),
    )
    by_coordinate = {
        site.coordinate: state.occupant_map[site.site_id].token
        for site in environment.occupiable_sites
    }
    return tuple(
        tuple(by_coordinate[(row, column)] for column in range(columns))
        for row in range(rows)
    )


def _state_with_assignment(
    environment: Environment,
    initial: MovementState,
    assignment: Mapping[str, Any],
) -> MovementState:
    state = MovementState(
        environment_id=initial.environment_id,
        environment_sha256=initial.environment_sha256,
        transition_index=0,
        occupancy=tuple(sorted(assignment.items())),
    )
    validate_state_against_environment(environment, state)
    return state


def _random_state(
    environment: Environment, initial: MovementState, address: str
) -> MovementState:
    sites = _coordinate_sites(environment)
    occupants = [initial.occupant_map[site] for site in sites]
    ranked = sorted(
        occupants,
        key=lambda item: _rank(
            "E06/S09/random-permutation/v1", address, item.occupant_id
        ),
    )
    return _state_with_assignment(
        environment, initial, dict(zip(sites, ranked, strict=True))
    )


def _block_scrambled_state(
    environment: Environment, initial: MovementState, address: str
) -> MovementState:
    by_block: dict[tuple[int, int], list[tuple[tuple[int, int], str]]] = defaultdict(
        list
    )
    for site in environment.occupiable_sites:
        row, column = site.coordinate
        by_block[(row // 2, column // 2)].append(((row % 2, column % 2), site.site_id))
    by_shape: dict[tuple[tuple[int, int], ...], list[tuple[int, int]]] = defaultdict(
        list
    )
    for block, entries in by_block.items():
        signature = tuple(sorted(offset for offset, _ in entries))
        by_shape[signature].append(block)
    assignment: dict[str, Any] = {}
    for signature, blocks in sorted(by_shape.items()):
        sources = sorted(
            blocks,
            key=lambda item: _rank(
                "E06/S09/block-source/v1", address, f"{item[0]}:{item[1]}"
            ),
        )
        targets = sorted(
            blocks,
            key=lambda item: _rank(
                "E06/S09/block-target/v1", address, f"{item[0]}:{item[1]}"
            ),
        )
        for source_block, target_block in zip(sources, targets, strict=True):
            source_by_offset = dict(by_block[source_block])
            target_by_offset = dict(by_block[target_block])
            for offset in signature:
                assignment[target_by_offset[offset]] = initial.occupant_map[
                    source_by_offset[offset]
                ]
    return _state_with_assignment(environment, initial, assignment)


def _partially_correct_state(
    environment: Environment,
    initial: MovementState,
    address: str,
    *,
    fraction: float,
    minimum: int,
) -> MovementState:
    desired = max(minimum, math.ceil(fraction * len(initial.occupancy) / 2))
    edges = [
        edge
        for edge in environment.edges
        if initial.occupant_map[edge[0]].token != initial.occupant_map[edge[1]].token
    ]
    ranked = sorted(
        edges,
        key=lambda item: _rank(
            "E06/S09/partial-edge/v1", address, f"{item[0]}|{item[1]}"
        ),
    )
    selected = []
    reserved: set[str] = set()
    for edge in ranked:
        if reserved.intersection(edge):
            continue
        selected.append(edge)
        reserved.update(edge)
        if len(selected) == desired:
            break
    if len(selected) < desired:
        raise ValueError("insufficient disjoint cross-token edges for partial start")
    assignment = initial.occupant_map
    for first, second in selected:
        assignment[first], assignment[second] = assignment[second], assignment[first]
    return _state_with_assignment(environment, initial, assignment)


def make_initial_state(
    environment: Environment,
    target: TargetDefinition,
    start_family: str,
    pairing_key: str,
    catalog: Mapping[str, Any],
) -> MovementState:
    """Generate a deterministic, count-preserving, initially incomplete state."""

    initial = initial_movement_state(environment)
    for attempt in range(32):
        address = f"{pairing_key}:attempt:{attempt}"
        if start_family == "random":
            state = _random_state(environment, initial, address)
        elif start_family == "block_scrambled":
            state = _block_scrambled_state(environment, initial, address)
        elif start_family == "partially_correct":
            parameters = catalog["initialStateFamilies"][start_family]["swapCount"]
            state = _partially_correct_state(
                environment,
                initial,
                address,
                fraction=float(parameters["fractionOfSites"]),
                minimum=int(parameters["minimum"]),
            )
        else:
            raise ValueError(f"unknown S09 start family: {start_family}")
        if not evaluate_success(state_grid(environment, state), target)["success"]:
            return state
    raise ValueError("failed to generate an initially incomplete S09 state")


def scenario_identity(
    split: str,
    target_id: str,
    start_family: str,
    replicate: int,
    policy_id: str,
) -> dict[str, Any]:
    address = {
        "split": split,
        "targetId": target_id,
        "startFamily": start_family,
        "replicate": int(replicate),
    }
    seeded_address = {"masterSeedHex": MASTER_SEED_HEX, "address": address}
    pairing_digest = _sha256("E06/S09/pairing-block/v1", seeded_address)
    scenario_id = f"s09-{split[:4]}-{pairing_digest[:24]}"
    run_id = "run1:" + _sha256(
        "E06/S09/run/v1",
        {**seeded_address, "policyId": policy_id},
    )
    seed_digest = _sha256("E06/S09/seed/v1", seeded_address)
    return {
        "scenarioId": scenario_id,
        "pairingBlockId": "pb1:" + pairing_digest,
        "runId": run_id,
        "seedHex": "0x" + seed_digest[:32],
        "seedDecimal": str(int(seed_digest[:32], 16)),
    }


class TargetMetricTracker:
    """Offline S01/S02 tracker with a fast exact-orbit distance screen."""

    def __init__(
        self,
        environment: Environment,
        target: TargetDefinition,
        grammar: RelationalGrammar,
    ) -> None:
        self.environment = environment
        self.target = target
        self.grammar = grammar
        self.orbit_grids = exact_equivalence_orbit(target)
        tokens = sorted(target.cell_type_counts)
        self.token_codes = {token: index for index, token in enumerate(tokens)}
        self.orbit = np.asarray(
            [
                [self.token_codes[token] for row in grid for token in row]
                for grid in self.orbit_grids
            ],
            dtype=np.int8,
        )
        self.site_count = target.shape[0] * target.shape[1]
        self.initial_mismatch: int | None = None
        self.minimum_mismatch = self.site_count
        self.minimum_transition = -1
        self.ever_s01 = False
        self.first_s01_transition: int | None = None
        self.ever_conjunctive = False
        self.first_conjunctive_transition: int | None = None
        self.observation_count = 0
        self.last_state: MovementState | None = None
        self._clock_mode: str | None = None
        self._authenticated_clock_records: list[dict[str, Any]] = []
        self._first_completion_record_commitment: str | None = None

    def _mismatch(self, grid: Grid) -> int:
        candidate = np.fromiter(
            (self.token_codes[token] for row in grid for token in row),
            dtype=np.int8,
            count=self.site_count,
        )
        return int(np.count_nonzero(self.orbit != candidate, axis=1).min())

    def _observe_at_elapsed(
        self,
        elapsed_transition: int,
        state: MovementState,
        *,
        completion_record_commitment: str | None,
    ) -> None:
        grid = state_grid(self.environment, state)
        mismatch = self._mismatch(grid)
        if self.initial_mismatch is None:
            self.initial_mismatch = mismatch
        if mismatch < self.minimum_mismatch:
            self.minimum_mismatch = mismatch
            self.minimum_transition = elapsed_transition
        if mismatch <= int(self.target.success["maxMismatches"]):
            global_audit = evaluate_success(
                grid, self.target, equivalence_orbit=self.orbit_grids
            )
            if global_audit["success"]:
                if not self.ever_s01:
                    self.ever_s01 = True
                    self.first_s01_transition = elapsed_transition
                local = score_grid(grid, self.grammar)
                if local["accepted"] and not self.ever_conjunctive:
                    self.ever_conjunctive = True
                    self.first_conjunctive_transition = elapsed_transition
                    self._first_completion_record_commitment = (
                        completion_record_commitment
                    )
        self.observation_count += 1
        self.last_state = state

    def observe(
        self,
        transition_index: int,
        state: MovementState,
        _summary: Mapping[str, Any],
    ) -> None:
        """Legacy observation-label path retained for predecessor compatibility."""

        if self._clock_mode in {
            "authenticated_elapsed",
            "authenticated_episode_origin",
        }:
            raise ValueError("cannot mix legacy and authenticated clock observations")
        self._clock_mode = "legacy_observation_label"
        self._observe_at_elapsed(
            transition_index,
            state,
            completion_record_commitment=None,
        )

    def observe_authenticated(
        self,
        clock_record: Mapping[str, Any],
        state: MovementState,
        _summary: Mapping[str, Any],
    ) -> None:
        """Observe one engine-authenticated elapsed-clock record."""

        if self._clock_mode == "legacy_observation_label":
            raise ValueError("cannot mix authenticated and legacy clock observations")
        expected_ordinal = len(self._authenticated_clock_records)
        episode_origin_mode = (
            clock_record.get("schemaVersion")
            == EPISODE_ORIGIN_CLOCK_RECORD_VERSION
        )
        desired_mode = (
            "authenticated_episode_origin"
            if episode_origin_mode
            else "authenticated_elapsed"
        )
        if self._clock_mode not in {None, desired_mode}:
            raise ValueError("cannot mix authenticated clock schemas")
        self._clock_mode = desired_mode
        if episode_origin_mode:
            if expected_ordinal == 0:
                scenario_id = clock_record.get("scenarioId")
                horizon = clock_record.get("horizonTransitions")
                origin = clock_record.get("episodeOriginTransitionIndex")
                origin_hash = clock_record.get("episodeOriginMovementStateSha256")
                if (
                    not isinstance(scenario_id, str)
                    or type(horizon) is not int
                    or type(origin) is not int
                    or not isinstance(origin_hash, str)
                ):
                    raise ValueError(
                        "authenticated episode-origin genesis metadata is invalid"
                    )
                expected_previous = episode_origin_genesis_commitment(
                    scenario_id=scenario_id,
                    horizon_transitions=horizon,
                    episode_origin_transition_index=origin,
                    episode_origin_movement_state_sha256=origin_hash,
                )
            else:
                previous = self._authenticated_clock_records[-1]
                scenario_id = str(previous["scenarioId"])
                horizon = int(previous["horizonTransitions"])
                origin = int(previous["episodeOriginTransitionIndex"])
                origin_hash = str(previous["episodeOriginMovementStateSha256"])
                expected_previous = str(previous["recordCommitmentSha256"])
            parsed = parse_and_validate_episode_origin_clock_record(
                clock_record,
                expected_scenario_id=scenario_id,
                expected_horizon=horizon,
                expected_ordinal=expected_ordinal,
                expected_elapsed=expected_ordinal,
                expected_origin_transition_index=origin,
                expected_origin_movement_state_sha256=origin_hash,
                expected_previous_commitment=expected_previous,
                state=state,
            )
        else:
            if expected_ordinal == 0:
                scenario_id = clock_record.get("scenarioId")
                horizon = clock_record.get("horizonTransitions")
                if not isinstance(scenario_id, str) or type(horizon) is not int:
                    raise ValueError("authenticated clock genesis metadata is invalid")
                expected_previous = genesis_commitment(
                    scenario_id=scenario_id,
                    horizon_transitions=horizon,
                )
            else:
                previous = self._authenticated_clock_records[-1]
                scenario_id = str(previous["scenarioId"])
                horizon = int(previous["horizonTransitions"])
                expected_previous = str(previous["recordCommitmentSha256"])
            parsed = parse_and_validate_elapsed_clock_record(
                clock_record,
                expected_scenario_id=scenario_id,
                expected_horizon=horizon,
                expected_ordinal=expected_ordinal,
                expected_elapsed=expected_ordinal,
                expected_previous_commitment=expected_previous,
                state=state,
            )
        canonical = parsed.to_mapping()
        self._authenticated_clock_records.append(canonical)
        self._observe_at_elapsed(
            parsed.elapsed_transition,
            state,
            completion_record_commitment=parsed.record_commitment_sha256,
        )

    def finalize(self) -> dict[str, Any]:
        if self.initial_mismatch is None or self.last_state is None:
            raise ValueError("metric tracker did not observe an episode")
        final_grid = state_grid(self.environment, self.last_state)
        global_audit = evaluate_success(
            final_grid, self.target, equivalence_orbit=self.orbit_grids
        )
        local = score_grid(final_grid, self.grammar)
        result = {
            "initialS01MismatchCount": self.initial_mismatch,
            "initialS01MismatchFraction": self.initial_mismatch / self.site_count,
            "minimumS01MismatchCount": self.minimum_mismatch,
            "minimumS01MismatchFraction": self.minimum_mismatch / self.site_count,
            "minimumS01MismatchTransition": self.minimum_transition,
            "terminalS01MismatchCount": global_audit["mismatchCount"],
            "terminalS01MismatchFraction": global_audit["mismatchFraction"],
            "terminalS01GlobalSuccess": bool(global_audit["success"]),
            "terminalS01GeometryMatch": bool(global_audit["geometryMatch"]),
            "terminalS01ComponentMatch": bool(global_audit["componentMatch"]),
            "terminalS01TopologyMatch": bool(global_audit["topologyMatch"]),
            "terminalVacancyHoles": int(global_audit["vacancyHoles"]),
            "everS01GlobalSuccess": self.ever_s01,
            "firstS01GlobalSuccessTransition": self.first_s01_transition,
            "terminalS02GrammarAccepted": bool(local["accepted"]),
            "terminalS02SoftScore": float(local["softScore"]),
            "terminalS02RelationalScore": float(local["relationalScore"]),
            "terminalS02HardViolationCount": int(local["hardViolationCount"]),
            "conjunctiveCompletionByBudget": self.ever_conjunctive,
            "firstCompletionTransition": self.first_conjunctive_transition,
            "terminalConjunctiveCompletion": bool(
                global_audit["success"] and local["accepted"]
            ),
            "localGlobalDiscordance": bool(
                bool(local["accepted"]) != bool(global_audit["success"])
            ),
            "metricObservationCount": self.observation_count,
            "finalGridRowsJson": json.dumps(
                ["".join(row) for row in final_grid], separators=(",", ":")
            ),
        }
        if self._clock_mode == "authenticated_elapsed":
            first = self._authenticated_clock_records[0]
            clock_summary = summarize_elapsed_clock_records(
                self._authenticated_clock_records,
                scenario_id=str(first["scenarioId"]),
                horizon_transitions=int(first["horizonTransitions"]),
                require_complete=True,
            )
            projection = build_first_completion_projection(
                clock_summary=clock_summary,
                first_completion_transition=self.first_conjunctive_transition,
                first_completion_record_commitment_sha256=(
                    self._first_completion_record_commitment
                ),
            )
            result.update(
                {
                    "clockMode": "authenticated_elapsed_transition",
                    "authenticatedElapsedClockSummary": clock_summary,
                    "authenticatedFirstCompletionProjection": projection,
                    "rawObservationLabelsUsedAsScientificTime": False,
                }
            )
        elif self._clock_mode == "authenticated_episode_origin":
            first = self._authenticated_clock_records[0]
            clock_summary = summarize_episode_origin_clock_records(
                self._authenticated_clock_records,
                scenario_id=str(first["scenarioId"]),
                horizon_transitions=int(first["horizonTransitions"]),
                episode_origin_transition_index=int(
                    first["episodeOriginTransitionIndex"]
                ),
                episode_origin_movement_state_sha256=str(
                    first["episodeOriginMovementStateSha256"]
                ),
                require_complete=True,
            )
            projection = build_episode_origin_first_completion_projection(
                clock_summary=clock_summary,
                first_completion_transition=self.first_conjunctive_transition,
                first_completion_record_commitment_sha256=(
                    self._first_completion_record_commitment
                ),
            )
            result.update(
                {
                    "clockMode": "authenticated_episode_origin_elapsed_transition",
                    "authenticatedElapsedClockSummary": clock_summary,
                    "authenticatedFirstCompletionProjection": projection,
                    "rawObservationLabelsUsedAsScientificTime": False,
                    "rawLifetimeMovementProvenanceRetained": True,
                }
            )
        return result


def _episode_definition(
    scenario_id: str,
    environment: Environment,
    grammar: RelationalGrammar,
    policy_id: str,
    event_budget: int,
    catalog: Mapping[str, Any],
) -> EpisodeDefinition:
    relation_id = grammar.grammar_id if policy_id in GRAMMAR_POLICIES else None
    parameters: dict[str, Any] = {}
    mode = "none"
    policy_parameters = catalog["policyEligibility"]["policyParameters"]
    if policy_id == "memory_based_recovery_v1":
        parameters["memoryInitialBestLocalUtility"] = int(
            policy_parameters["memoryInitialBestLocalUtility"]
        )
    elif policy_id == "boundary_seeking_v1":
        mode = "boundary_signal"
        parameters.update(
            {
                "boundaryDirection": str(policy_parameters["boundaryDirection"]),
                "boundaryTokens": list(
                    policy_parameters["boundaryActorTokensByTarget"][grammar.target_id]
                ),
            }
        )
    elif policy_id == "gradient_following_v1":
        mode = "static_gradient"
        parameters.update(
            {
                "gradientAxis": str(policy_parameters["gradientAxis"]),
                "gradientDirection": str(policy_parameters["gradientDirection"]),
            }
        )
    return EpisodeDefinition(
        scenario_id=scenario_id,
        environment_id=environment.environment_id,
        policy_id=policy_id,
        relation_grammar_id=relation_id,
        channel_mode=mode,
        transitions=int(event_budget),
        actor_batch_size=int(catalog["simulation"]["actorBatchSize"]),
        parameters=parameters,
    )


def run_baseline_once(specification: Mapping[str, Any]) -> tuple[dict[str, Any], Any]:
    """Execute one complete fixed-budget S09 CPU-oracle run."""

    context, targets, grammars, environments = load_baseline_assets()
    target = targets[str(specification["targetId"])]
    grammar = grammars[str(specification["grammarId"])]
    environment = environments[target.target_id]
    catalog = specification["catalog"]
    identity = scenario_identity(
        str(specification["split"]),
        target.target_id,
        str(specification["startFamily"]),
        int(specification["replicate"]),
        str(specification["policyId"]),
    )
    initial = make_initial_state(
        environment,
        target,
        str(specification["startFamily"]),
        identity["pairingBlockId"],
        catalog,
    )
    tracker = TargetMetricTracker(environment, target, grammar)
    definition = _episode_definition(
        identity["scenarioId"],
        environment,
        grammar,
        str(specification["policyId"]),
        int(specification["eventBudget"]),
        catalog,
    )
    started = time.perf_counter()
    result = run_cpu_episode(
        context,
        definition,
        include_selected_traces=bool(specification.get("retainTrace", False)),
        initial_state_override=initial,
        state_audit=tracker.observe,
    )
    elapsed = time.perf_counter() - started
    metrics = tracker.finalize()
    final_state = tracker.last_state
    assert final_state is not None
    initial_ids = {item.occupant_id for _, item in initial.occupancy}
    final_ids = {item.occupant_id for _, item in final_state.occupancy}
    initial_tokens = Counter(item.token for _, item in initial.occupancy)
    final_tokens = Counter(item.token for _, item in final_state.occupancy)
    invariant_success = (
        initial_ids == final_ids
        and initial_tokens == final_tokens
        and movement_state_sha256(final_state) == result["finalState"]["stateSha256"]
        and len(result["transitionSummaries"]) == int(specification["eventBudget"])
    )
    episode_bytes = canonical_episode_result_bytes(result)
    initial_grid = state_grid(environment, initial)
    movement = result["movementLedger"]
    observation = result["observationLedger"]
    channel = result["channelLedger"]
    row = {
        "schemaVersion": RUN_RESULT_VERSION,
        "phase": str(specification["phase"]),
        "split": str(specification["split"]),
        **identity,
        "targetId": target.target_id,
        "grammarId": grammar.grammar_id,
        "environmentId": environment.environment_id,
        "startFamily": str(specification["startFamily"]),
        "replicate": int(specification["replicate"]),
        "policyId": str(specification["policyId"]),
        "channelMode": definition.channel_mode,
        "backend": "canonical_s07_cpu_oracle_fallback",
        "eventBudgetTransitions": int(specification["eventBudget"]),
        "actorBatchSize": definition.actor_batch_size,
        "runStatus": "completed",
        "stopReason": "fixed_event_budget",
        "censored": not metrics["conjunctiveCompletionByBudget"],
        "failed": False,
        "errorType": None,
        "errorMessage": None,
        "initialStateSha256": movement_state_sha256(initial),
        "finalStateSha256": result["finalState"]["stateSha256"],
        "episodeSha256": result["episodeSha256"],
        "episodeCanonicalBytesSha256": hashlib.sha256(episode_bytes).hexdigest(),
        "initialGridRowsJson": json.dumps(
            ["".join(item) for item in initial_grid], separators=(",", ":")
        ),
        **metrics,
        "acceptedMovements": int(movement["acceptedMovements"]),
        "submittedProposals": int(movement["submittedProposals"]),
        "conflictLosses": int(movement["conflictLosses"]),
        "invalidProposals": int(movement["invalidProposals"]),
        "totalGraphDisplacement": int(movement["totalGraphDisplacement"]),
        "observationCommunicatedBitsUpperBound": int(
            observation["communicatedBitsUpperBound"]
        ),
        "observationUtilityEvaluations": int(observation["utilityEvaluations"]),
        "channelConfigurationBits": int(channel["configurationBits"]),
        "channelTotalInformationBits": int(channel["totalInformationBits"]),
        "invariantSuccess": invariant_success,
        "permissionAuditSuccess": all(
            value is False for value in result["permissionAudit"].values()
        ),
        "wallSeconds": elapsed,
        "traceSelected": bool(specification.get("retainTrace", False)),
    }
    row["metricSummarySha256"] = _sha256(
        "E06/S09/metric-summary/v1",
        {
            key: row[key]
            for key in sorted(row)
            if key.startswith(
                ("initialS01", "minimumS01", "terminalS01", "terminalS02")
            )
            or key
            in {
                "conjunctiveCompletionByBudget",
                "firstCompletionTransition",
                "terminalConjunctiveCompletion",
                "localGlobalDiscordance",
            }
        },
    )
    trace = result if bool(specification.get("retainTrace", False)) else None
    return row, trace


def run_condition_task(task: Mapping[str, Any]) -> dict[str, Any]:
    """Run one condition chunk; suitable for an eight-process pool."""

    rows = []
    traces = []
    for replicate in task["replicates"]:
        specification = {**task, "replicate": int(replicate)}
        specification.pop("replicates", None)
        identity = scenario_identity(
            str(specification["split"]),
            str(specification["targetId"]),
            str(specification["startFamily"]),
            int(replicate),
            str(specification["policyId"]),
        )
        specification["retainTrace"] = identity["runId"] in set(task["traceRunIds"])
        try:
            row, trace = run_baseline_once(specification)
            rows.append(row)
            if trace is not None:
                traces.append(
                    {
                        "runId": row["runId"],
                        "reason": "preregistered_one_percent_condition_sample",
                        "episode": trace,
                    }
                )
        except Exception as error:  # Preserve every failed intended run.
            rows.append(
                {
                    "schemaVersion": RUN_RESULT_VERSION,
                    "phase": str(specification["phase"]),
                    "split": str(specification["split"]),
                    **identity,
                    "targetId": str(specification["targetId"]),
                    "grammarId": str(specification["grammarId"]),
                    "environmentId": None,
                    "startFamily": str(specification["startFamily"]),
                    "replicate": int(replicate),
                    "policyId": str(specification["policyId"]),
                    "channelMode": None,
                    "backend": "canonical_s07_cpu_oracle_fallback",
                    "eventBudgetTransitions": int(specification["eventBudget"]),
                    "actorBatchSize": int(
                        specification["catalog"]["simulation"]["actorBatchSize"]
                    ),
                    "runStatus": "failed",
                    "stopReason": "execution_error",
                    "censored": False,
                    "failed": True,
                    "errorType": type(error).__name__,
                    "errorMessage": str(error),
                    "traceSelected": True,
                }
            )
            traces.append(
                {
                    "runId": identity["runId"],
                    "reason": "execution_failure",
                    "errorType": type(error).__name__,
                    "errorMessage": str(error),
                    "specification": {
                        key: value
                        for key, value in specification.items()
                        if key != "catalog"
                    },
                }
            )
    return {
        "conditionId": str(task["conditionId"]),
        "rows": rows,
        "traces": traces,
    }
