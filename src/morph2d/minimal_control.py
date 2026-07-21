"""Frozen E06 S14 minimal-control and spatial-transfer execution layer.

S14 searches only the calibrated exact-start terminal-retention endpoint.  The
module keeps formation/repair audits and topology-transfer diagnostics outside
optimization, preserves the S06 controller boundary, and prices schedules,
queries, action, source work, computation, and foregone local opportunity as
separate components.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, deque
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .baseline import _rectangle_expected, load_baseline_assets
from .engine import EpisodeDefinition, canonical_episode_result_bytes, run_cpu_episode
from .environments import Environment, parse_environment_spec
from .hybrid_control import canonical_bytes, run_hybrid_once
from .movements import (
    MovementState,
    initial_movement_state,
    make_proposal,
    movement_state_sha256,
    parse_movement_state,
    resolve_batch,
)


ROOT = Path(__file__).resolve().parents[2]
CATALOG_VERSION = "e06.s14.minimal-control-catalog.v1"
RUN_VERSION = "e06.s14.minimal-control-run.v1"
TRANSFER_RUN_VERSION = "e06.s14.transfer-diagnostic-run.v1"
MASTER_SEED_HEX = "0xe0614000000000000000000000000001"
DIRECT_BASE_ARMS = {"central_only", "combined"}


def _sha256(domain: str, value: Any) -> str:
    return hashlib.sha256(
        domain.encode("ascii") + b"\x00" + canonical_bytes(value)
    ).hexdigest()


@lru_cache(maxsize=1)
def load_minimal_control_catalog() -> Mapping[str, Any]:
    raw = yaml.safe_load(
        (ROOT / "configs/morphologies/minimal_control_catalog.yaml").read_text(
            encoding="utf-8"
        )
    )
    validate_minimal_control_catalog(raw)
    return raw


def _all_policy_records(catalog: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    policies = catalog["interventionPolicies"]
    return [*policies["searchCandidates"], *policies["mandatoryControls"]]


def validate_minimal_control_catalog(catalog: Mapping[str, Any]) -> None:
    required = {
        "schemaVersion",
        "researchStepId",
        "title",
        "frozenQuestion",
        "claimBoundary",
        "canonicalContracts",
        "seedContract",
        "metricDirections",
        "topologySuccessCriteria",
        "splits",
        "interventionPolicies",
        "budgets",
        "optimization",
        "endpointRules",
        "transfer",
        "censoring",
        "validation",
        "release",
    }
    if set(catalog) != required:
        raise ValueError("S14 catalog top-level schema mismatch")
    if (
        catalog["schemaVersion"] != CATALOG_VERSION
        or catalog["researchStepId"] != "S14"
    ):
        raise ValueError("S14 catalog version mismatch")
    policies = _all_policy_records(catalog)
    identifiers = [str(item["policyId"]) for item in policies]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("S14 policy IDs must be unique")
    if len(catalog["interventionPolicies"]["searchCandidates"]) != int(
        catalog["optimization"]["maximumCandidatePolicies"]
    ):
        raise ValueError("S14 search size differs from frozen maximum")
    eligible = set(map(int, catalog["interventionPolicies"]["eligibleEpochs"]))
    for item in policies:
        if str(item["baseArm"]) not in {"local_only", *DIRECT_BASE_ARMS}:
            raise ValueError("S14 policy uses unsupported authority")
        epochs = tuple(map(int, item["directEpochs"]))
        if tuple(sorted(set(epochs))) != epochs or set(epochs) - eligible:
            raise ValueError("S14 direct epochs are invalid")
        if not 1 <= int(item["candidateLimit"]) <= 6:
            raise ValueError("S14 candidate content limit is invalid")
        if item["actionRelay"] not in {"enabled", "disconnected"}:
            raise ValueError("S14 action relay is invalid")
        if item["baseArm"] == "local_only" and epochs:
            raise ValueError("local-only control cannot schedule direct queries")
    if catalog["budgets"]["scalarCostForbidden"] is not True:
        raise ValueError("S14 cannot collapse cost axes")
    if catalog["topologySuccessCriteria"]["transferGate"] != (
        "diagnostic_only_require_complete_accounting_invariants_replay_and_no_claim_promotion"
    ):
        raise ValueError("S14 transfer claim boundary changed")


def policy_catalog(catalog: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(item["policyId"]): item for item in _all_policy_records(catalog)}


def scenario_identity(
    split: str,
    scenario_key: str,
    challenge_id: str,
    replicate: int,
    policy_id: str,
) -> dict[str, Any]:
    address = {
        "split": str(split),
        "scenarioKey": str(scenario_key),
        "challengeId": str(challenge_id),
        "replicate": int(replicate),
    }
    seeded = {"masterSeedHex": MASTER_SEED_HEX, "address": address}
    pairing = _sha256("E06/S14/pairing-block/v1", seeded)
    seed = _sha256("E06/S14/seed/v1", seeded)
    return {
        "scenarioId": f"s14-{str(split)[:4]}-{pairing[:24]}",
        "pairingBlockId": "pb1:" + pairing,
        "runId": "run1:"
        + _sha256("E06/S14/run/v1", {**seeded, "policyId": str(policy_id)}),
        "seedHex": "0x" + seed[:32],
        "seedDecimal": str(int(seed[:32], 16)),
    }


def schedule_configuration_bits(policy: Mapping[str, Any]) -> int:
    if str(policy["baseArm"]) == "local_only":
        return 0
    return 4 + 3 * len(policy["directEpochs"]) + 3 + 1


def _s14_budget_fields(
    row: Mapping[str, Any], policy: Mapping[str, Any]
) -> dict[str, Any]:
    schedule_bits = schedule_configuration_bits(policy)
    query_count = int(row["directRecipientQuerySlots"])
    scheduled_count = len(policy["directEpochs"])
    checks = {
        "queryCountWithinSchedule": query_count <= scheduled_count,
        "controllerInputWithinPerQueryCeiling": int(row["controllerInputBits"])
        <= query_count * 352,
        "addressWithinPerQueryCeiling": int(row["addressBits"]) <= query_count * 7,
        "attemptWithinPerQueryCeiling": int(row["actuationAttempts"]) <= query_count,
        "overrideWithinPerQueryCeiling": int(row["overrideActionUnits"]) <= query_count,
        "directDisplacementWithinPerQueryCeiling": int(
            row["directMovementGraphDisplacement"]
        )
        <= query_count * 6,
        "baseEnvelope": bool(row["budgetEnvelopeSuccess"]),
        "permissionIsolation": bool(row["permissionAuditSuccess"]),
    }
    return {
        "scheduleConfigurationBits": schedule_bits,
        "channelConfigurationBits": int(row["configurationBits"]),
        "totalConfigurationBitsIncludingSchedule": int(row["configurationBits"])
        + schedule_bits,
        "totalInformationBitsIncludingObservationAndSchedule": int(
            row["totalInformationBitsIncludingObservation"]
        )
        + schedule_bits,
        "budgetValidationJson": json.dumps(
            checks, sort_keys=True, separators=(",", ":")
        ),
        "s14BudgetSuccess": all(checks.values()),
    }


def run_minimal_once(
    specification: Mapping[str, Any],
) -> tuple[dict[str, Any], Mapping[str, Any] | None, Mapping[str, Any]]:
    catalog = specification.get("catalog") or load_minimal_control_catalog()
    validate_minimal_control_catalog(catalog)
    policy_id = str(specification["controlPolicyId"])
    policy = policy_catalog(catalog)[policy_id]
    target_id = str(specification["targetId"])
    challenge_id = str(specification["challengeId"])
    identity = scenario_identity(
        str(specification["split"]),
        target_id,
        challenge_id,
        int(specification["replicate"]),
        policy_id,
    )
    base_arm = str(policy["baseArm"])
    row, trace, mask = run_hybrid_once(
        {
            "phase": str(specification["phase"]),
            "split": str(specification["split"]),
            "targetId": target_id,
            "challengeId": challenge_id,
            "armId": base_arm,
            "replicate": int(specification["replicate"]),
            "eventBudget": int(specification["eventBudget"]),
            "retainTrace": bool(specification.get("retainTrace", False)),
            "identityOverride": identity,
            "directEpochs": list(map(int, policy["directEpochs"])),
            "directCandidateLimit": int(policy["candidateLimit"]),
            "directActionRelayEnabled": policy["actionRelay"] == "enabled",
        }
    )
    row = {
        **row,
        "schemaVersion": RUN_VERSION,
        "controlPolicyId": policy_id,
        "baseArmId": base_arm,
        "scheduledDirectEpochsJson": json.dumps(
            list(map(int, policy["directEpochs"])), separators=(",", ":")
        ),
        "scheduledDirectQueryCount": len(policy["directEpochs"]),
        "directCandidateLimit": int(policy["candidateLimit"]),
        "directActionRelayEnabled": policy["actionRelay"] == "enabled",
        "topologyCompletionCalibrated": True,
        "optimizationEligibleEndpoint": challenge_id == "exact_maintenance",
        **_s14_budget_fields(row, policy),
    }
    row["s14MetricSummarySha256"] = _sha256(
        "E06/S14/minimal-control-metric/v1",
        {
            key: row[key]
            for key in sorted(row)
            if key.startswith(("terminalS01", "terminalS02", "minimumS01"))
            or key
            in {
                "terminalConjunctiveCompletion",
                "controlPolicyId",
                "scheduledDirectQueryCount",
                "directCandidateLimit",
                "totalInformationBitsIncludingObservationAndSchedule",
                "totalGraphDisplacement",
                "totalSourceWorkUnits",
                "totalComputationUnits",
                "totalOpportunityCostUnits",
            }
        },
    )
    return row, trace, mask


def run_minimal_condition_task(task: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    masks: dict[str, Mapping[str, Any]] = {}
    trace_ids = set(map(str, task.get("traceRunIds", ())))
    for replicate in task["replicates"]:
        identity = scenario_identity(
            str(task["split"]),
            str(task["targetId"]),
            str(task["challengeId"]),
            int(replicate),
            str(task["controlPolicyId"]),
        )
        try:
            row, trace, mask = run_minimal_once(
                {
                    **task,
                    "replicate": int(replicate),
                    "retainTrace": identity["runId"] in trace_ids,
                }
            )
            rows.append(row)
            masks[str(mask["pairingBlockId"])] = mask
            if trace is not None:
                traces.append({"runId": row["runId"], "episode": trace})
        except Exception as error:
            rows.append(
                {
                    "schemaVersion": RUN_VERSION,
                    **identity,
                    "phase": str(task["phase"]),
                    "split": str(task["split"]),
                    "targetId": str(task["targetId"]),
                    "challengeId": str(task["challengeId"]),
                    "controlPolicyId": str(task["controlPolicyId"]),
                    "replicate": int(replicate),
                    "failed": True,
                    "runStatus": "failed",
                    "censored": False,
                    "errorType": type(error).__name__,
                    "errorMessage": str(error),
                }
            )
            traces.append(
                {
                    "runId": identity["runId"],
                    "reason": "execution_failure",
                    "errorType": type(error).__name__,
                    "errorMessage": str(error),
                }
            )
    return {"rows": rows, "traces": traces, "masks": list(masks.values())}


def _layers_rows(size: int) -> list[str]:
    if size <= 0 or size % 3:
        raise ValueError("layer transfer size must be positive and divisible by three")
    tokens = "ABC"
    width = size // 3
    return [tokens[row // width] * size for row in range(size)]


def _irregular_expected(
    site_ids: Sequence[str],
    edges: Sequence[tuple[str, str]],
    boundary_ids: set[str],
) -> dict[str, Any]:
    degrees = Counter()
    for site_id in site_ids:
        degrees[sum(site_id in edge for edge in edges)] += 1
    return {
        "siteCount": len(site_ids),
        "occupiableSiteCount": len(site_ids),
        "edgeCount": len(edges),
        "componentCount": 1,
        "degreeHistogram": {str(k): v for k, v in sorted(degrees.items())},
        "vacancyCount": 0,
        "obstacleCount": 0,
        "fixedBoundaryCount": 0,
        "naturalBoundarySiteCount": len(boundary_ids),
        "signaledSiteCount": len(boundary_ids),
    }


@lru_cache(maxsize=2)
def build_transfer_fixture(
    fixture_id: str,
) -> tuple[Environment, Mapping[str, str]]:
    if fixture_id == "larger_square_layers_15x15":
        rows = _layers_rows(15)
        raw = {
            "environmentId": "s14_larger_square_layers_15x15",
            "title": "S14 uncalibrated 15x15 layer transfer diagnostic",
            "geometry": "square",
            "boundaryMode": "bounded",
            "occupancyMode": "fully_occupied",
            "vacancyLabel": ".",
            "obstacleToken": "#",
            "generator": {"kind": "rectangle", "rows": 15, "columns": 15},
            "obstacles": [],
            "fixedBoundary": {"mode": "none", "sites": [], "token": None},
            "boundarySignal": {
                "mode": "domain_edge_flags",
                "observable": True,
                "includeObstacleContact": False,
                "includeFixedRole": False,
            },
            "initialState": {"defaultToken": ".", "overrides": {}, "rows": rows},
            "targetBinding": None,
            "requireConnected": True,
            "notes": "S14 transfer diagnostic; no S01/S02 completion calibration.",
            "expected": _rectangle_expected(15, 15, 0),
        }
        environment = parse_environment_spec(raw)
        reference = dict(environment.initial_state)
        return environment, reference
    if fixture_id != "irregular_pruned_layers_9x9":
        raise ValueError("unknown S14 transfer fixture")
    rows = _layers_rows(9)
    site_ids = [f"n_r{row}_c{col}" for row in range(9) for col in range(9)]
    sites = []
    boundary_ids: set[str] = set()
    reference: dict[str, str] = {}
    for row in range(9):
        for col in range(9):
            site_id = f"n_r{row}_c{col}"
            tags = []
            if row == 0:
                tags.append("north")
            if row == 8:
                tags.append("south")
            if col == 0:
                tags.append("west")
            if col == 8:
                tags.append("east")
            if tags:
                boundary_ids.add(site_id)
            sites.append(
                {"siteId": site_id, "coordinate": [row, col], "boundaryTags": tags}
            )
            reference[site_id] = rows[row][col]
    removed = {
        tuple(sorted((f"n_r2_c{col}", f"n_r3_c{col}"))) for col in (1, 3, 5, 7)
    } | {tuple(sorted((f"n_r5_c{col}", f"n_r6_c{col}"))) for col in (0, 2, 6, 8)}
    edges = []
    for row in range(9):
        for col in range(9):
            for other in ((row + 1, col), (row, col + 1)):
                if other[0] >= 9 or other[1] >= 9:
                    continue
                edge = tuple(sorted((f"n_r{row}_c{col}", f"n_r{other[0]}_c{other[1]}")))
                if edge not in removed:
                    edges.append(edge)
    raw = {
        "environmentId": "s14_irregular_pruned_layers_9x9",
        "title": "S14 uncalibrated irregular pruned layer graph",
        "geometry": "irregular",
        "boundaryMode": "bounded",
        "occupancyMode": "fully_occupied",
        "vacancyLabel": ".",
        "obstacleToken": "#",
        "generator": {"kind": "explicit_graph", "sites": sites, "edges": edges},
        "obstacles": [],
        "fixedBoundary": {"mode": "none", "sites": [], "token": None},
        "boundarySignal": {
            "mode": "explicit_site_tags",
            "observable": True,
            "includeObstacleContact": False,
            "includeFixedRole": False,
        },
        "initialState": {"defaultToken": "A", "overrides": reference, "rows": None},
        "targetBinding": None,
        "requireConnected": True,
        "notes": "S14 transfer diagnostic; pruned graph has no completion calibration.",
        "expected": _irregular_expected(site_ids, edges, boundary_ids),
    }
    return parse_environment_spec(raw), dict(sorted(reference.items()))


def _transfer_initial_state(
    environment: Environment,
    reference: Mapping[str, str],
    challenge_id: str,
) -> tuple[MovementState, int]:
    state = initial_movement_state(environment)
    if {key: value.token for key, value in state.occupant_map.items()} != dict(
        reference
    ):
        raise ValueError("transfer reference and initial state differ")
    if challenge_id == "exact_reference_retention":
        return state, 0
    if challenge_id != "one_swap_displacement_diagnostic":
        raise ValueError("unknown S14 transfer challenge")
    edge = next(
        item
        for item in environment.edges
        if state.occupant_map[item[0]].token != state.occupant_map[item[1]].token
    )
    proposal = make_proposal(state, "adjacent_swap", edge)
    batch = resolve_batch(
        environment,
        state,
        (proposal,),
        batch_nonce=f"{environment.environment_id}:damage",
    )
    return parse_movement_state(batch["postState"]), int(
        batch["costLedger"]["totalGraphDisplacement"]
    )


def _token_components(
    environment: Environment, assignment: Mapping[str, str]
) -> Mapping[str, int]:
    neighbors = {site.site_id: set() for site in environment.occupiable_sites}
    for first, second in environment.edges:
        neighbors[first].add(second)
        neighbors[second].add(first)
    output: dict[str, int] = {}
    for token in sorted(set(assignment.values())):
        unseen = {site_id for site_id, value in assignment.items() if value == token}
        components = 0
        while unseen:
            components += 1
            queue = deque([unseen.pop()])
            while queue:
                current = queue.popleft()
                found = unseen & neighbors[current]
                unseen -= found
                queue.extend(found)
        output[token] = components
    return output


def graph_diagnostics(
    environment: Environment,
    assignment: Mapping[str, str],
    reference: Mapping[str, str],
) -> dict[str, Any]:
    if set(assignment) != set(reference):
        raise ValueError("transfer diagnostic site sets differ")
    mismatch = sum(assignment[key] != reference[key] for key in reference)
    observed_components = _token_components(environment, assignment)
    reference_components = _token_components(environment, reference)
    component_error = sum(
        abs(observed_components.get(token, 0) - reference_components.get(token, 0))
        for token in set(observed_components) | set(reference_components)
    )
    heterotypic = sum(
        assignment[first] != assignment[second] for first, second in environment.edges
    )
    reference_heterotypic = sum(
        reference[first] != reference[second] for first, second in environment.edges
    )
    edge_count = len(environment.edges)
    counts = Counter(assignment.values())
    n = len(assignment)
    expected_homotypic = sum(value * (value - 1) for value in counts.values()) / (
        n * (n - 1)
    )
    observed_homotypic = 1 - heterotypic / edge_count
    return {
        "referenceMismatchCount": mismatch,
        "referenceMismatchFraction": mismatch / n,
        "graphComponentError": component_error,
        "heterotypicEdgeCount": heterotypic,
        "heterotypicEdgeFraction": heterotypic / edge_count,
        "referenceHeterotypicEdgeFraction": reference_heterotypic / edge_count,
        "normalizedBoundaryError": abs(heterotypic - reference_heterotypic)
        / edge_count,
        "compositionExpectedHomotypicEdgeFraction": expected_homotypic,
        "observedHomotypicEdgeFraction": observed_homotypic,
        "compositionCorrectedHomotypicEdgeExcess": observed_homotypic
        - expected_homotypic,
    }


class _TransferTracker:
    def __init__(self, environment: Environment, reference: Mapping[str, str]) -> None:
        self.environment = environment
        self.reference = reference
        self.initial: Mapping[str, Any] | None = None
        self.minimum_mismatch = len(reference)
        self.last_state: MovementState | None = None

    def observe(
        self, _index: int, state: MovementState, _summary: Mapping[str, Any]
    ) -> None:
        assignment = {site_id: occupant.token for site_id, occupant in state.occupancy}
        diagnostics = graph_diagnostics(self.environment, assignment, self.reference)
        if self.initial is None:
            self.initial = diagnostics
        self.minimum_mismatch = min(
            self.minimum_mismatch, int(diagnostics["referenceMismatchCount"])
        )
        self.last_state = state


def _policy_source_reads(observation: Mapping[str, int]) -> int:
    return sum(
        int(observation[field])
        for field in (
            "actorTokenReads",
            "localTokenReads",
            "candidateDescriptorReads",
            "boundarySignalReads",
            "gradientSignalReads",
            "laggedConflictReads",
            "memoryReads",
            "counterRandomDraws",
            "affordanceTopologyReads",
            "affordanceSiteRoleReads",
        )
    )


def run_transfer_once(
    specification: Mapping[str, Any],
) -> tuple[dict[str, Any], Mapping[str, Any] | None]:
    catalog = specification.get("catalog") or load_minimal_control_catalog()
    validate_minimal_control_catalog(catalog)
    policy_id = str(specification["controlPolicyId"])
    policy = policy_catalog(catalog)[policy_id]
    fixture_id = str(specification["fixtureId"])
    challenge_id = str(specification["challengeId"])
    identity = scenario_identity(
        str(specification["split"]),
        fixture_id,
        challenge_id,
        int(specification["replicate"]),
        policy_id,
    )
    context, _targets, grammars, _environments = load_baseline_assets()
    environment, reference = build_transfer_fixture(fixture_id)
    context = replace(
        context,
        environments={**context.environments, environment.environment_id: environment},
    )
    initial_state, external_displacement = _transfer_initial_state(
        environment, reference, challenge_id
    )
    grammar = grammars["layers_ordered_contacts_v1"]
    base_arm = str(policy["baseArm"])
    channel_mode = "direct_intervention" if base_arm in DIRECT_BASE_ARMS else "none"
    definition = EpisodeDefinition(
        scenario_id=identity["scenarioId"],
        environment_id=environment.environment_id,
        policy_id="greedy_neighbor_satisfaction_v1",
        relation_grammar_id=grammar.grammar_id,
        channel_mode=channel_mode,
        transitions=int(specification["eventBudget"]),
        actor_batch_size=4,
        parameters={
            "summaryRecipients": 1,
            "directCandidateLimit": int(policy["candidateLimit"]),
        }
        if channel_mode == "direct_intervention"
        else {},
    )
    tracker = _TransferTracker(environment, reference)
    direct_epochs = frozenset(map(int, policy["directEpochs"]))
    result = run_cpu_episode(
        context,
        definition,
        include_selected_traces=bool(specification.get("retainTrace", False)),
        initial_state_override=initial_state,
        state_audit=tracker.observe,
        native_batch_gate=(lambda _index: False)
        if base_arm == "central_only"
        else None,
        direct_epoch_gate=(lambda epoch: epoch in direct_epochs),
        direct_action_gate=(
            None if policy["actionRelay"] == "enabled" else lambda _epoch: False
        ),
    )
    if tracker.last_state is None or tracker.initial is None:
        raise ValueError("S14 transfer tracker did not observe state")
    final_assignment = {
        site_id: occupant.token for site_id, occupant in tracker.last_state.occupancy
    }
    final = graph_diagnostics(environment, final_assignment, reference)
    movement = result["movementLedger"]
    observation = result["observationLedger"]
    channel = result["channelLedger"]
    allocated = 4 * int(specification["eventBudget"])
    used = sum(
        int(item["scheduledActorCount"])
        for item in result["transitionSummaries"]
        if not bool(item["directIntervention"])
    )
    query_slots = sum(
        int(item["scheduledActorCount"])
        for item in result["transitionSummaries"]
        if bool(item["directIntervention"])
    )
    policy_reads = _policy_source_reads(observation)
    source_work = (
        policy_reads
        + int(channel["sourceScalarReads"])
        + int(channel["controllerStateReads"])
    )
    computation = (
        int(observation["utilityEvaluations"])
        + int(observation["comparisonOperations"])
        + int(channel["controllerComputeUnits"])
    )
    opportunity = (
        allocated
        - used
        + int(channel["suppressedNativeActions"])
        + int(channel["opportunityCostUnits"])
    )
    initial_ids = {item.occupant_id for _, item in initial_state.occupancy}
    final_ids = {item.occupant_id for _, item in tracker.last_state.occupancy}
    row: dict[str, Any] = {
        "schemaVersion": TRANSFER_RUN_VERSION,
        "phase": str(specification["phase"]),
        "split": str(specification["split"]),
        **identity,
        "fixtureId": fixture_id,
        "challengeId": challenge_id,
        "controlPolicyId": policy_id,
        "baseArmId": base_arm,
        "replicate": int(specification["replicate"]),
        "environmentId": environment.environment_id,
        "geometry": environment.geometry,
        "siteCount": len(reference),
        "edgeCount": len(environment.edges),
        "eventBudgetTransitions": int(specification["eventBudget"]),
        "backend": "canonical_s07_cpu_oracle_fallback",
        "runStatus": "completed",
        "failed": False,
        "topologyCompletionCalibrated": False,
        "terminalConjunctiveCompletion": None,
        "terminalS01GlobalSuccess": None,
        "terminalS02GrammarAccepted": None,
        "initialReferenceMismatchFraction": float(
            tracker.initial["referenceMismatchFraction"]
        ),
        "minimumReferenceMismatchFraction": tracker.minimum_mismatch / len(reference),
        **{f"terminal{key[0].upper()}{key[1:]}": value for key, value in final.items()},
        "terminalReferenceMatchDiagnostic": final["referenceMismatchCount"] == 0,
        "initialStateSha256": movement_state_sha256(initial_state),
        "finalStateSha256": result["finalState"]["stateSha256"],
        "episodeSha256": result["episodeSha256"],
        "episodeCanonicalBytesSha256": hashlib.sha256(
            canonical_episode_result_bytes(result)
        ).hexdigest(),
        "acceptedMovements": int(movement["acceptedMovements"]),
        "submittedProposals": int(movement["submittedProposals"]),
        "conflictLosses": int(movement["conflictLosses"]),
        "invalidProposals": int(movement["invalidProposals"]),
        "totalGraphDisplacement": int(movement["totalGraphDisplacement"]),
        "observationCommunicatedBitsUpperBound": int(
            observation["communicatedBitsUpperBound"]
        ),
        "policyLogicalSourceReads": policy_reads,
        "policyUtilityEvaluations": int(observation["utilityEvaluations"]),
        "policyComparisonOperations": int(observation["comparisonOperations"]),
        "configurationBits": int(channel["configurationBits"]),
        "controllerInputBits": int(channel["controllerInputBits"]),
        "policyDeliveryBits": int(channel["policyDeliveryBits"]),
        "addressBits": int(channel["addressBits"]),
        "channelTotalInformationBits": int(channel["totalInformationBits"]),
        "totalInformationBitsIncludingObservation": int(
            observation["communicatedBitsUpperBound"]
        )
        + int(channel["totalInformationBits"]),
        "sourceScalarReads": int(channel["sourceScalarReads"]),
        "controllerStateReads": int(channel["controllerStateReads"]),
        "controllerComputeUnits": int(channel["controllerComputeUnits"]),
        "actuationAttempts": int(channel["actuationAttempts"]),
        "actuationSuccesses": int(channel["actuationSuccesses"]),
        "overrideActionUnits": int(channel["overrideActionUnits"]),
        "directMovementGraphDisplacement": int(channel["movementGraphDisplacement"]),
        "suppressedNativeActions": int(channel["suppressedNativeActions"]),
        "channelOpportunityCostUnits": int(channel["opportunityCostUnits"]),
        "allocatedNativeActorSlots": allocated,
        "usedNativeActorSlots": used,
        "directRecipientQuerySlots": query_slots,
        "foregoneNativeActorSlots": allocated - used,
        "externalDisplacedEntities": 2 if external_displacement else 0,
        "externalGraphDisplacement": external_displacement,
        "externalInterventionGraphDisplacement": external_displacement,
        "shamSuppressedControllerRecommendations": sum(
            int(
                event.get("actionRelayEnabled") is False
                and event.get("controllerRecommendation", {}).get("action")
                == "select_candidate"
            )
            for event in result["channelEvents"]
        ),
        "totalSourceWorkUnits": source_work,
        "totalComputationUnits": computation,
        "totalOpportunityCostUnits": opportunity,
        "budgetEnvelopeSuccess": bool(
            int(channel["configurationBits"]) <= 649
            and int(channel["controllerInputBits"]) <= query_slots * 352
            and int(channel["addressBits"]) <= query_slots * 7
            and int(channel["actuationAttempts"]) <= query_slots
            and int(channel["overrideActionUnits"]) <= query_slots
            and int(channel["movementGraphDisplacement"]) <= query_slots * 6
        ),
        "invariantSuccess": bool(
            initial_ids == final_ids
            and Counter(item.token for _, item in initial_state.occupancy)
            == Counter(item.token for _, item in tracker.last_state.occupancy)
            and movement_state_sha256(tracker.last_state)
            == result["finalState"]["stateSha256"]
        ),
        "permissionAuditSuccess": all(
            value is False for value in result["permissionAudit"].values()
        ),
        "traceSelected": bool(specification.get("retainTrace", False)),
    }
    row.update(_s14_budget_fields(row, policy))
    row["transferMetricSummarySha256"] = _sha256(
        "E06/S14/transfer-metric/v1",
        {
            key: row[key]
            for key in sorted(row)
            if key.startswith(
                (
                    "initialReference",
                    "minimumReference",
                    "terminalReference",
                    "terminalGraph",
                    "terminalNormalized",
                    "terminalComposition",
                )
            )
        },
    )
    trace = result if bool(specification.get("retainTrace", False)) else None
    return row, trace


def run_transfer_condition_task(task: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    trace_ids = set(map(str, task.get("traceRunIds", ())))
    for replicate in task["replicates"]:
        identity = scenario_identity(
            str(task["split"]),
            str(task["fixtureId"]),
            str(task["challengeId"]),
            int(replicate),
            str(task["controlPolicyId"]),
        )
        try:
            row, trace = run_transfer_once(
                {
                    **task,
                    "replicate": int(replicate),
                    "retainTrace": identity["runId"] in trace_ids,
                }
            )
            rows.append(row)
            if trace is not None:
                traces.append({"runId": row["runId"], "episode": trace})
        except Exception as error:
            rows.append(
                {
                    "schemaVersion": TRANSFER_RUN_VERSION,
                    **identity,
                    "phase": str(task["phase"]),
                    "split": str(task["split"]),
                    "fixtureId": str(task["fixtureId"]),
                    "challengeId": str(task["challengeId"]),
                    "controlPolicyId": str(task["controlPolicyId"]),
                    "replicate": int(replicate),
                    "failed": True,
                    "runStatus": "failed",
                    "errorType": type(error).__name__,
                    "errorMessage": str(error),
                }
            )
    return {"rows": rows, "traces": traces}
