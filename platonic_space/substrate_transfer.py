from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from morphospace2d import (
    CellIdentity,
    NeighborPreferenceRule,
    RecoveryPolicySpec,
    Substrate,
    TargetMorphology,
    audit_recovery_policy_payload,
    build_boundary_target,
    build_gradient_target,
    build_ring_target,
    build_sorted_row_target,
    build_stripes_target,
    run_recovery_benchmark,
)

from .world_schema import compact_json


SUBSTRATE_TRANSFER_SCHEMA_VERSION = "e07_s13_substrate_transfer.v1"
SUBSTRATE_TRANSFER_MODEL_VERSION = "e07_s13_direct_substrate_replay.v1"
SUBSTRATE_TRANSFER_CLAIM_BOUNDARY = (
    "Empirical computational transfer tests over local-swap simulator substrates only. "
    "S12 DSL policies are transferred through explicitly documented morphospace analogues; "
    "results are bounded simulation proxies, not causal, biological, clinical, or wet-lab evidence."
)

SOURCE_POLICY_ENTITY = "E01::cell_view_bubble"
NEAREST_BUBBLE_ENTITY = "E05::morphology::bubble"
SOURCE_TARGET_ID = "sorted_row_8"
SOURCE_WORLD_ENTITY = "E05_S03_target_sorted_row_8"
SOURCE_GOAL_ENTITY = "G_TARGET_MORPHOLOGY_SORTED_ROW_8"
DEFAULT_TRANSFER_SEEDS = (713001, 713002, 713003)
TARGET_MAX_STEPS = {
    "sorted_row_8": 800,
    "gradient_x_5x4": 1400,
    "vertical_stripes_6x4": 1600,
    "perimeter_boundary_6x5": 1800,
    "ring_7x7": 2200,
    "branch_graph_12": 1400,
}
TARGET_S08_ENTITIES = {
    "sorted_row_8": {
        "world": "E05_S03_target_sorted_row_8",
        "goal": "G_TARGET_MORPHOLOGY_SORTED_ROW_8",
    },
    "gradient_x_5x4": {
        "world": "E05_S03_target_gradient_x_5x4",
        "goal": "G_TARGET_MORPHOLOGY_GRADIENT_X_5X4",
    },
    "vertical_stripes_6x4": {
        "world": "E05_S03_target_vertical_stripes_6x4",
        "goal": "G_TARGET_MORPHOLOGY_VERTICAL_STRIPES_6X4",
    },
    "perimeter_boundary_6x5": {
        "world": "E05_S03_target_perimeter_boundary_6x5",
        "goal": "G_TARGET_MORPHOLOGY_PERIMETER_BOUNDARY_6X5",
    },
    "ring_7x7": {
        "world": "E05_S03_target_ring_7x7",
        "goal": "G_TARGET_MORPHOLOGY_RING_7X7",
    },
}
PROFILE_ANALOGUES = {
    "balanced_sorting_low_cost": {
        "analoguePolicyFamily": "rank_gradient_transfer",
        "s08TargetPolicyEntityId": "E05::morphology::classic_scalar_rank_swap",
        "rankWeight": 1.0,
        "axisWeight": 0.15,
        "affinityWeight": 0.0,
        "explorationEpsilon": 0.0,
        "memoryWeight": 0.0,
        "signalWeight": 0.0,
        "mappingCaveat": "Bubble-like local inversion is mapped to scalar-rank and axis-local swaps; several S12 designs exactly match Bubble-like references on the held-out E03 panel.",
    },
    "robust_frozen_sorting": {
        "analoguePolicyFamily": "memory_signal_rank_affinity_transfer",
        "s08TargetPolicyEntityId": "E05::morphology::e04_memory_signal_rank_affinity",
        "rankWeight": 0.90,
        "axisWeight": 0.10,
        "affinityWeight": 0.05,
        "explorationEpsilon": 0.005,
        "memoryWeight": 0.08,
        "signalWeight": 0.03,
        "mappingCaveat": "Stuck-frozen robustness has no frozen-cell action in the S13 recovery substrate, so it is represented by bounded local memory/signal penalties.",
    },
    "aggregation_chimera": {
        "analoguePolicyFamily": "rank_affinity_transfer",
        "s08TargetPolicyEntityId": "E05::morphology::e03_frontier_rank_affinity",
        "rankWeight": 0.75,
        "axisWeight": 0.05,
        "affinityWeight": 0.35,
        "explorationEpsilon": 0.01,
        "memoryWeight": 0.0,
        "signalWeight": 0.0,
        "mappingCaveat": "Candidate/null chimera aggregation is represented only by local identity-affinity sorting; exact mixed-population dominance or proliferation semantics are unsupported here.",
    },
}


@dataclass(frozen=True)
class TransferPolicyPanel:
    policies: tuple[RecoveryPolicySpec, ...]
    mapping_catalog: pd.DataFrame


@dataclass(frozen=True)
class TransferTargetPanel:
    targets: tuple[TargetMorphology, ...]
    target_catalog: pd.DataFrame


def _finite(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, set):
        return sorted(_json_ready(item) for item in value)
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _sha12(value: Any) -> str:
    payload = compact_json(_json_ready(value))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def dataframe_json_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in out.columns:
        if out[column].map(lambda value: isinstance(value, (dict, list, tuple, set))).any():
            out[column] = out[column].map(
                lambda value: compact_json(sorted(value) if isinstance(value, set) else value)
                if isinstance(value, (dict, list, tuple, set))
                else value
            )
        if out[column].dtype == "object":
            non_null = out[column].dropna()
            observed_types = {type(value) for value in non_null}
            if len(observed_types) > 1:
                out[column] = out[column].map(lambda value: None if value is None or pd.isna(value) else str(value))
    return out


def transfer_mapping_contract() -> dict[str, Any]:
    return {
        "schemaVersion": SUBSTRATE_TRANSFER_SCHEMA_VERSION,
        "sourceSubstrate": {
            "class": "E03/E01 one-dimensional adjacent-swap DSL row",
            "observation": "actor scalar value and immediate left/right scalar values where present",
            "action": "swap_left, swap_right, or wait",
            "goal": "increase row sortedness or profile-specific simulator competence",
        },
        "targetSubstrates": {
            "row_1d": "identity-preserving scrambled recovery on an E05 sorted-row target",
            "square_grid_2d": "identity-preserving scrambled recovery on E05 grid target morphologies",
            "irregular_graph": "identity-preserving scrambled recovery on a branch graph morphology",
        },
        "observationMapping": {
            "actor_value": "actor visible scalar_value",
            "left_right_neighbors": "all occupied adjacent graph neighbors, with no global ordering except public substrate geometry",
            "local_order_signal": "rank_position_error and axis_position_error from visible scalar/ap components and public local position",
            "local_affinity_signal": "actor-internal target-neighborhood preference satisfaction among adjacent visible identities",
        },
        "actionMapping": {
            "swap_left_or_right": "swap with the adjacent neighbor selected by local policy score",
            "wait": "no swap when no local improvement is available",
            "unsupported": "birth, death, detach, proliferation, long-range moves, and exact candidate/null dominance are not executed in S13",
        },
        "goalMapping": {
            "row_sorting": "reduce composite target error and earth-mover/rank placement error",
            "frozen_robustness": "bounded memory/signal local penalties as a proxy, without frozen-cell physics",
            "chimera_aggregation": "local identity affinity and neighborhood-role satisfaction as a proxy, without mixed-population growth",
        },
        "metricMapping": {
            "sourceCompetence": "mean relative_error_reduction on sorted_row_8",
            "transferCompetence": "mean relative_error_reduction on held-out non-row targets",
            "retention": "transferCompetence divided by sourceCompetence when sourceCompetence is positive",
            "failureModes": "no improvement, below random control, low retention, stalled waiting, no exact recovery, or retained partial competence",
        },
        "claimBoundary": SUBSTRATE_TRANSFER_CLAIM_BOUNDARY,
    }


def _identity(
    *,
    cell_id: int,
    scalar_value: float,
    ap_coordinate: float,
    organ_type: str,
    polarity: Sequence[float],
    adhesion_type: str,
    preferences: Sequence[Mapping[str, Any]] | None = None,
) -> CellIdentity:
    return CellIdentity(
        cell_id=cell_id,
        components={
            "scalar_value": scalar_value,
            "ap_coordinate": ap_coordinate,
            "organ_type": organ_type,
            "polarity": [float(item) for item in polarity],
            "adhesion_type": adhesion_type,
            "target_neighbor_preferences": list(preferences or []),
            "internal_state": {},
        },
    )


def _attach_satisfied_neighbor_preferences(
    substrate: Substrate,
    identities_by_position: Mapping[tuple[int, ...], CellIdentity],
) -> dict[tuple[int, ...], CellIdentity]:
    updated: dict[tuple[int, ...], CellIdentity] = {}
    for position, identity in identities_by_position.items():
        neighbor_identities = [
            identities_by_position[neighbor]
            for neighbor in substrate.neighbors(position)
            if neighbor in identities_by_position
        ]
        organ_counts = Counter(neighbor.components["organ_type"] for neighbor in neighbor_identities)
        adhesion_counts = Counter(neighbor.components["adhesion_type"] for neighbor in neighbor_identities)
        rules: list[dict[str, Any]] = []
        for organ_type, count in sorted(organ_counts.items()):
            rules.append(NeighborPreferenceRule("organ_type", organ_type, min_count=count, max_count=count, weight=0.5).to_record())
        own_adhesion_count = adhesion_counts.get(identity.components["adhesion_type"], 0)
        if own_adhesion_count:
            rules.append(
                NeighborPreferenceRule(
                    "adhesion_type",
                    identity.components["adhesion_type"],
                    min_count=own_adhesion_count,
                    max_count=own_adhesion_count,
                    weight=0.25,
                ).to_record()
            )
        components = dict(identity.components)
        components["target_neighbor_preferences"] = rules
        updated[position] = CellIdentity(cell_id=identity.cell_id, components=components)
    return updated


def build_branch_graph_target() -> TargetMorphology:
    nodes = (
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 0),
        (5, 0),
        (2, 1),
        (2, 2),
        (3, -1),
        (3, -2),
        (4, 1),
        (5, 1),
    )
    edges = (
        ((0, 0), (1, 0)),
        ((1, 0), (2, 0)),
        ((2, 0), (3, 0)),
        ((3, 0), (4, 0)),
        ((4, 0), (5, 0)),
        ((2, 0), (2, 1)),
        ((2, 1), (2, 2)),
        ((3, 0), (3, -1)),
        ((3, -1), (3, -2)),
        ((4, 0), (4, 1)),
        ((4, 1), (5, 1)),
    )
    substrate = Substrate.irregular_graph(edges, nodes=nodes)
    x_min = min(node[0] for node in nodes)
    x_max = max(node[0] for node in nodes)
    identities: dict[tuple[int, ...], CellIdentity] = {}
    for cell_id, position in enumerate(sorted(nodes)):
        x, y = position
        degree = len(substrate.neighbors(position))
        if degree == 1:
            organ_type = "boundary"
            adhesion = "adhesion_boundary"
            polarity = [1.0, 0.0] if y >= 0 else [-1.0, 0.0]
        elif y == 0:
            organ_type = "axis"
            adhesion = "adhesion_a"
            polarity = [1.0, 0.0]
        else:
            organ_type = "appendage"
            adhesion = "adhesion_b"
            polarity = [0.0, 1.0 if y > 0 else -1.0]
        identities[position] = _identity(
            cell_id=cell_id,
            scalar_value=cell_id + 1,
            ap_coordinate=(x - x_min) / max(1.0, float(x_max - x_min)),
            organ_type=organ_type,
            polarity=polarity,
            adhesion_type=adhesion,
        )
    identities = _attach_satisfied_neighbor_preferences(substrate, identities)
    target = TargetMorphology(
        target_id="branch_graph_12",
        title="Branch graph morphology",
        motif="branch_graph",
        substrate=substrate,
        identities_by_position=identities,
        description="Irregular graph target used as a substrate-transfer stress test for row-discovered policies.",
    )
    errors = target.validate()
    if errors:
        raise ValueError(f"invalid branch graph target: {'; '.join(errors)}")
    return target


def build_transfer_target_panel() -> TransferTargetPanel:
    targets = (
        build_sorted_row_target(8),
        build_gradient_target(5, 4),
        build_stripes_target(6, 4),
        build_boundary_target(6, 5),
        build_ring_target(7),
        build_branch_graph_target(),
    )
    rows: list[dict[str, Any]] = []
    for target in targets:
        s08 = TARGET_S08_ENTITIES.get(target.target_id, {})
        target_role = "source_retention" if target.target_id == SOURCE_TARGET_ID else "transfer_holdout"
        if target.substrate.substrate_type == "row_1d":
            substrate_class = "row_1d"
        elif target.substrate.substrate_type == "irregular_graph":
            substrate_class = "irregular_graph"
        else:
            substrate_class = "square_grid_2d"
        rows.append(
            {
                "schemaVersion": SUBSTRATE_TRANSFER_SCHEMA_VERSION,
                "targetId": target.target_id,
                "title": target.title,
                "motif": target.motif,
                "transferTargetRole": target_role,
                "substrateType": target.substrate.substrate_type,
                "substrateClass": substrate_class,
                "nodeCount": len(target.substrate.nodes),
                "edgeCount": len(target.substrate.edges()),
                "sourceS08WorldEntityId": SOURCE_WORLD_ENTITY,
                "targetS08WorldEntityId": s08.get("world"),
                "sourceS08GoalEntityId": SOURCE_GOAL_ENTITY,
                "targetS08GoalEntityId": s08.get("goal"),
                "maxSteps": int(TARGET_MAX_STEPS[target.target_id]),
                "snapshotInterval": max(100, int(TARGET_MAX_STEPS[target.target_id]) // 4),
                "targetValidationErrorsJson": target.validate(),
                "mappingCaveat": (
                    "S08 distance unavailable for this irregular graph target; transfer is still directly simulated."
                    if target.target_id not in TARGET_S08_ENTITIES
                    else "S08 target entity available; distance remains an empirical computational proxy."
                ),
            }
        )
    return TransferTargetPanel(targets=targets, target_catalog=pd.DataFrame(rows))


def _nearest_lookup(nearest_existing: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if nearest_existing.empty or "policyId" not in nearest_existing.columns:
        return {}
    return {str(row["policyId"]): dict(row) for _, row in nearest_existing.iterrows()}


def _policy_spec_from_row(row: Mapping[str, Any], nearest: Mapping[str, Any]) -> tuple[RecoveryPolicySpec, dict[str, Any]]:
    profile_id = str(row["designProfileId"])
    analogue = PROFILE_ANALOGUES[profile_id]
    source_policy_id = str(row["policyId"])
    transfer_policy_id = f"{source_policy_id}__s13_transfer"
    nearest_policy = str(nearest.get("nearestExistingPolicyId", "")) or None
    exact_match = bool(nearest.get("exactBehaviorMatchOnPanel", False))
    distance = _finite(nearest.get("behaviorDistanceOnHeldoutPanel"))
    caveats = [
        str(analogue["mappingCaveat"]),
        str(row.get("designCaveat", "")),
        "S08 policy distance uses E01 Bubble and E05 analogue proxies because S12 designs were created after S08 embeddings.",
    ]
    if nearest_policy:
        caveats.append(f"Nearest existing S12 held-out behavior reference: {nearest_policy}.")
    if exact_match:
        caveats.append("Exact behavior match to a Bubble-like S12 nearest-existing reference on the S12 held-out panel.")
    policy = RecoveryPolicySpec(
        policy_id=transfer_policy_id,
        family=f"s13_s12_{analogue['analoguePolicyFamily']}",
        description=f"S13 transfer analogue of S12 policy {source_policy_id} ({profile_id}).",
        rank_weight=float(analogue["rankWeight"]),
        axis_weight=float(analogue["axisWeight"]),
        affinity_weight=float(analogue["affinityWeight"]),
        exploration_epsilon=float(analogue["explorationEpsilon"]),
        memory_weight=float(analogue["memoryWeight"]),
        signal_weight=float(analogue["signalWeight"]),
        source_artifact="/artifacts/research_steps/S12/designed_policies.parquet",
        source_policy_id=source_policy_id,
        source_policy_label=str(row.get("profileName", profile_id)),
        parameters={
            "s12DesignProfileId": profile_id,
            "s12DesignRankWithinProfile": int(row.get("designRankWithinProfile", 0)),
            "s12StructureHash": str(row.get("structureHash", "")),
            "nearestExistingPolicyId": nearest_policy,
            "s12BehaviorDistanceOnHeldoutPanel": None if math.isnan(distance) else distance,
            "s12ExactBehaviorMatchOnPanel": exact_match,
        },
    )
    mapping = {
        "schemaVersion": SUBSTRATE_TRANSFER_SCHEMA_VERSION,
        "mappingId": f"map_{transfer_policy_id}",
        "mappingKind": "s12_designed_policy_transfer",
        "executed": True,
        "sourcePolicyId": source_policy_id,
        "transferPolicyId": transfer_policy_id,
        "sourceDesignProfileId": profile_id,
        "sourceProfileName": str(row.get("profileName", profile_id)),
        "sourceDesignRankWithinProfile": int(row.get("designRankWithinProfile", 0)),
        "sourceSubstrateClass": "row_1d_dsl",
        "targetSubstrateClasses": ["row_1d", "square_grid_2d", "irregular_graph"],
        "analoguePolicyFamily": str(analogue["analoguePolicyFamily"]),
        "rankWeight": float(analogue["rankWeight"]),
        "axisWeight": float(analogue["axisWeight"]),
        "affinityWeight": float(analogue["affinityWeight"]),
        "explorationEpsilon": float(analogue["explorationEpsilon"]),
        "memoryWeight": float(analogue["memoryWeight"]),
        "signalWeight": float(analogue["signalWeight"]),
        "s08SourcePolicyEntityId": SOURCE_POLICY_ENTITY,
        "nearestExistingS08ProxyEntityId": NEAREST_BUBBLE_ENTITY,
        "s08TargetPolicyEntityId": str(analogue["s08TargetPolicyEntityId"]),
        "nearestExistingPolicyId": nearest_policy,
        "s12BehaviorDistanceOnHeldoutPanel": None if math.isnan(distance) else distance,
        "s12ExactBehaviorMatchOnPanel": exact_match,
        "observationMappingJson": transfer_mapping_contract()["observationMapping"],
        "actionMappingJson": transfer_mapping_contract()["actionMapping"],
        "goalMappingJson": transfer_mapping_contract()["goalMapping"],
        "metricMappingJson": transfer_mapping_contract()["metricMapping"],
        "mappingCaveat": " ".join(item for item in caveats if item),
    }
    return policy, mapping


def _baseline_policy_specs() -> list[tuple[RecoveryPolicySpec, dict[str, Any]]]:
    specs = [
        RecoveryPolicySpec(
            policy_id="classic_scalar_rank_swap",
            family="s13_baseline_e05_rank",
            description="E05 scalar-rank local swap baseline used as the Bubble-like transfer baseline.",
            rank_weight=1.0,
            axis_weight=0.15,
            affinity_weight=0.0,
        ),
        RecoveryPolicySpec(
            policy_id="classic_neighbor_affinity_swap",
            family="s13_baseline_e05_affinity",
            description="E05 local neighbor-affinity baseline.",
            rank_weight=0.0,
            axis_weight=0.0,
            affinity_weight=1.0,
        ),
        RecoveryPolicySpec(
            policy_id="e03_frontier_rank_affinity",
            family="s13_baseline_e03_frontier_analogue",
            description="E03-inspired E05 rank-plus-affinity analogue baseline.",
            rank_weight=0.8,
            axis_weight=0.1,
            affinity_weight=0.25,
            exploration_epsilon=0.01,
        ),
        RecoveryPolicySpec(
            policy_id="e04_memory_signal_rank_affinity",
            family="s13_baseline_e04_memory_signal_analogue",
            description="E04-inspired E05 memory/signal rank-affinity analogue baseline.",
            rank_weight=0.75,
            axis_weight=0.1,
            affinity_weight=0.35,
            exploration_epsilon=0.02,
            memory_weight=0.08,
            signal_weight=0.05,
        ),
        RecoveryPolicySpec(
            policy_id="random_local_swap_null",
            family="s13_random_local_swap_null",
            description="Uniform random adjacent-swap null control.",
            rank_weight=0.0,
            axis_weight=0.0,
            affinity_weight=0.0,
            exploration_epsilon=1.0,
        ),
        RecoveryPolicySpec(
            policy_id="s13_wait_no_transfer_control",
            family="s13_no_transfer_control",
            description="No-transfer wait-only control with no local scoring and no exploration.",
            rank_weight=0.0,
            axis_weight=0.0,
            affinity_weight=0.0,
            exploration_epsilon=0.0,
        ),
    ]
    entity_lookup = {
        "classic_scalar_rank_swap": "E05::morphology::classic_scalar_rank_swap",
        "classic_neighbor_affinity_swap": "E05::morphology::classic_neighbor_affinity_swap",
        "e03_frontier_rank_affinity": "E05::morphology::e03_frontier_rank_affinity",
        "e04_memory_signal_rank_affinity": "E05::morphology::e04_memory_signal_rank_affinity",
        "random_local_swap_null": "E05::morphology::random_local_swap_null",
    }
    rows = []
    for policy in specs:
        mapping_kind = "random_transfer_control" if policy.policy_id == "random_local_swap_null" else "baseline_transfer_control"
        if policy.policy_id == "s13_wait_no_transfer_control":
            mapping_kind = "no_transfer_control"
        rows.append(
            (
                policy,
                {
                    "schemaVersion": SUBSTRATE_TRANSFER_SCHEMA_VERSION,
                    "mappingId": f"map_{policy.policy_id}",
                    "mappingKind": mapping_kind,
                    "executed": True,
                    "sourcePolicyId": None,
                    "transferPolicyId": policy.policy_id,
                    "sourceDesignProfileId": None,
                    "sourceProfileName": None,
                    "sourceDesignRankWithinProfile": None,
                    "sourceSubstrateClass": "control",
                    "targetSubstrateClasses": ["row_1d", "square_grid_2d", "irregular_graph"],
                    "analoguePolicyFamily": policy.family,
                    "rankWeight": float(policy.rank_weight),
                    "axisWeight": float(policy.axis_weight),
                    "affinityWeight": float(policy.affinity_weight),
                    "explorationEpsilon": float(policy.exploration_epsilon),
                    "memoryWeight": float(policy.memory_weight),
                    "signalWeight": float(policy.signal_weight),
                    "s08SourcePolicyEntityId": SOURCE_POLICY_ENTITY if policy.policy_id != "s13_wait_no_transfer_control" else None,
                    "nearestExistingS08ProxyEntityId": NEAREST_BUBBLE_ENTITY if policy.policy_id != "s13_wait_no_transfer_control" else None,
                    "s08TargetPolicyEntityId": entity_lookup.get(policy.policy_id),
                    "nearestExistingPolicyId": None,
                    "s12BehaviorDistanceOnHeldoutPanel": None,
                    "s12ExactBehaviorMatchOnPanel": None,
                    "observationMappingJson": transfer_mapping_contract()["observationMapping"],
                    "actionMappingJson": transfer_mapping_contract()["actionMapping"],
                    "goalMappingJson": transfer_mapping_contract()["goalMapping"],
                    "metricMappingJson": transfer_mapping_contract()["metricMapping"],
                    "mappingCaveat": (
                        "Control policy for baseline/random comparison; not a transferred S12 designed policy."
                        if policy.policy_id != "classic_scalar_rank_swap"
                        else "Bubble-like scalar-rank baseline used to contextualize S12 designs that are near or identical to Bubble-like held-out references."
                    ),
                },
            )
        )
    return rows


def _randomized_transfer_control(source_row: Mapping[str, Any]) -> tuple[RecoveryPolicySpec, dict[str, Any]]:
    source_policy_id = str(source_row["policyId"])
    seed = int(hashlib.sha256(source_policy_id.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    weights = {
        "rankWeight": float(rng.uniform(0.0, 1.0)),
        "axisWeight": float(rng.uniform(0.0, 0.4)),
        "affinityWeight": float(rng.uniform(0.0, 0.8)),
        "explorationEpsilon": float(rng.uniform(0.0, 0.08)),
        "memoryWeight": float(rng.uniform(0.0, 0.12)),
        "signalWeight": float(rng.uniform(0.0, 0.08)),
    }
    policy_id = f"{source_policy_id}__s13_randomized_transfer"
    policy = RecoveryPolicySpec(
        policy_id=policy_id,
        family="s13_randomized_transfer_control",
        description=f"Randomized transfer-control analogue for S12 policy {source_policy_id}.",
        rank_weight=weights["rankWeight"],
        axis_weight=weights["axisWeight"],
        affinity_weight=weights["affinityWeight"],
        exploration_epsilon=weights["explorationEpsilon"],
        memory_weight=weights["memoryWeight"],
        signal_weight=weights["signalWeight"],
        source_artifact="/artifacts/research_steps/S12/designed_policies.parquet",
        source_policy_id=source_policy_id,
        source_policy_label=str(source_row.get("profileName", source_row.get("designProfileId", ""))),
        parameters={"controlSeed": seed, "controlForSourcePolicyId": source_policy_id},
    )
    mapping = {
        "schemaVersion": SUBSTRATE_TRANSFER_SCHEMA_VERSION,
        "mappingId": f"map_{policy_id}",
        "mappingKind": "randomized_transfer_control",
        "executed": True,
        "sourcePolicyId": source_policy_id,
        "transferPolicyId": policy_id,
        "sourceDesignProfileId": str(source_row.get("designProfileId", "")),
        "sourceProfileName": str(source_row.get("profileName", "")),
        "sourceDesignRankWithinProfile": int(source_row.get("designRankWithinProfile", 0)),
        "sourceSubstrateClass": "row_1d_dsl_randomized_mapping_control",
        "targetSubstrateClasses": ["row_1d", "square_grid_2d", "irregular_graph"],
        "analoguePolicyFamily": policy.family,
        **weights,
        "s08SourcePolicyEntityId": None,
        "nearestExistingS08ProxyEntityId": None,
        "s08TargetPolicyEntityId": None,
        "nearestExistingPolicyId": None,
        "s12BehaviorDistanceOnHeldoutPanel": None,
        "s12ExactBehaviorMatchOnPanel": None,
        "observationMappingJson": transfer_mapping_contract()["observationMapping"],
        "actionMappingJson": transfer_mapping_contract()["actionMapping"],
        "goalMappingJson": transfer_mapping_contract()["goalMapping"],
        "metricMappingJson": transfer_mapping_contract()["metricMapping"],
        "mappingCaveat": "Randomized control preserving a link to the S12 source policy but not its designed mapping logic; no native S08 policy distance is assigned.",
    }
    return policy, mapping


def _unsupported_mapping_rows() -> list[dict[str, Any]]:
    base = {
        "schemaVersion": SUBSTRATE_TRANSFER_SCHEMA_VERSION,
        "executed": False,
        "sourcePolicyId": None,
        "transferPolicyId": None,
        "sourceDesignProfileId": None,
        "sourceProfileName": None,
        "sourceDesignRankWithinProfile": None,
        "sourceSubstrateClass": "row_1d_dsl",
        "targetSubstrateClasses": [],
        "rankWeight": None,
        "axisWeight": None,
        "affinityWeight": None,
        "explorationEpsilon": None,
        "memoryWeight": None,
        "signalWeight": None,
        "s08SourcePolicyEntityId": None,
        "nearestExistingS08ProxyEntityId": None,
        "s08TargetPolicyEntityId": None,
        "nearestExistingPolicyId": None,
        "s12BehaviorDistanceOnHeldoutPanel": None,
        "s12ExactBehaviorMatchOnPanel": None,
        "observationMappingJson": transfer_mapping_contract()["observationMapping"],
        "actionMappingJson": transfer_mapping_contract()["actionMapping"],
        "goalMappingJson": transfer_mapping_contract()["goalMapping"],
        "metricMappingJson": transfer_mapping_contract()["metricMapping"],
    }
    return [
        {
            **base,
            "mappingId": "unsupported_homeostatic_birth_death_transfer",
            "mappingKind": "unsupported_documented_mapping",
            "analoguePolicyFamily": "unsupported_homeostatic_transfer",
            "mappingCaveat": "S12 local adjacent-swap DSL has no birth, death, apoptosis, or homeostatic setpoint action, so exact transfer into E04 homeostatic tasks is not executed in S13.",
        },
        {
            **base,
            "mappingId": "unsupported_exact_chimera_growth_transfer",
            "mappingKind": "unsupported_documented_mapping",
            "analoguePolicyFamily": "unsupported_chimera_growth_transfer",
            "mappingCaveat": "S12 candidate/null chimera aggregation is transferred only as local affinity sorting; exact mixed-lineage growth, dominance, or graft semantics are outside the conservative recovery substrate.",
        },
    ]


def build_transfer_policy_panel(
    designed_policies: pd.DataFrame,
    nearest_existing: pd.DataFrame,
    *,
    include_randomized_controls: bool = True,
) -> TransferPolicyPanel:
    nearest_by_policy = _nearest_lookup(nearest_existing)
    policies: list[RecoveryPolicySpec] = []
    rows: list[dict[str, Any]] = []
    for _, design_row in designed_policies.sort_values(["designProfileId", "designRankWithinProfile", "policyId"], kind="mergesort").iterrows():
        policy, mapping = _policy_spec_from_row(design_row, nearest_by_policy.get(str(design_row["policyId"]), {}))
        policies.append(policy)
        rows.append(mapping)
        if include_randomized_controls:
            control_policy, control_mapping = _randomized_transfer_control(design_row)
            policies.append(control_policy)
            rows.append(control_mapping)
    for policy, mapping in _baseline_policy_specs():
        policies.append(policy)
        rows.append(mapping)
    rows.extend(_unsupported_mapping_rows())
    mapping_catalog = pd.DataFrame(rows)
    policy_ids = [policy.policy_id for policy in policies]
    if len(policy_ids) != len(set(policy_ids)):
        duplicates = sorted(policy_id for policy_id, count in Counter(policy_ids).items() if count > 1)
        raise ValueError(f"duplicate transfer policy IDs: {duplicates}")
    return TransferPolicyPanel(policies=tuple(policies), mapping_catalog=mapping_catalog)


def run_transfer_panel(
    policies: Sequence[RecoveryPolicySpec],
    targets: Sequence[TargetMorphology],
    *,
    seeds: Sequence[int] = DEFAULT_TRANSFER_SEEDS,
    max_steps_by_target: Mapping[str, int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    max_steps_by_target = dict(TARGET_MAX_STEPS if max_steps_by_target is None else max_steps_by_target)
    run_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    severity_rows: list[dict[str, Any]] = []
    for target in targets:
        max_steps = int(max_steps_by_target[target.target_id])
        snapshot_interval = max(100, max_steps // 4)
        for policy in policies:
            for seed in seeds:
                row, trace, severity = run_recovery_benchmark(
                    target,
                    policy,
                    seed=int(seed),
                    max_steps=max_steps,
                    snapshot_interval=snapshot_interval,
                )
                run_id = f"S13::{target.target_id}::{policy.policy_id}::seed{int(seed)}"
                row = dict(row)
                row.update(
                    {
                        "schema_version": SUBSTRATE_TRANSFER_SCHEMA_VERSION,
                        "upstream_recovery_schema_version": row.get("schema_version"),
                        "research_step_id": "S13",
                        "run_id": run_id,
                        "target_role": "source_retention" if target.target_id == SOURCE_TARGET_ID else "transfer_holdout",
                        "transfer_seed_index": list(seeds).index(seed),
                    }
                )
                run_rows.append(row)
                for trace_row in trace:
                    updated_trace = dict(trace_row)
                    updated_trace.update(
                        {
                            "schema_version": SUBSTRATE_TRANSFER_SCHEMA_VERSION,
                            "upstream_recovery_schema_version": trace_row.get("schema_version"),
                            "research_step_id": "S13",
                            "run_id": run_id,
                            "target_role": row["target_role"],
                        }
                    )
                    trace_rows.append(updated_trace)
                severity_row = dict(severity)
                severity_row.update(
                    {
                        "schemaVersion": SUBSTRATE_TRANSFER_SCHEMA_VERSION,
                        "researchStepId": "S13",
                        "runId": run_id,
                        "policyId": policy.policy_id,
                    }
                )
                severity_rows.append(severity_row)
    return pd.DataFrame(run_rows), pd.DataFrame(trace_rows), pd.DataFrame(severity_rows)


def summarize_transfer_runs(
    run_rows: pd.DataFrame,
    mapping_catalog: pd.DataFrame,
    target_catalog: pd.DataFrame,
) -> pd.DataFrame:
    executed_mappings = mapping_catalog[mapping_catalog["executed"].astype(bool)].copy()
    runs = run_rows.merge(
        executed_mappings[
            [
                "transferPolicyId",
                "mappingKind",
                "sourcePolicyId",
                "sourceDesignProfileId",
                "sourceProfileName",
                "analoguePolicyFamily",
                "s08SourcePolicyEntityId",
                "nearestExistingS08ProxyEntityId",
                "s08TargetPolicyEntityId",
                "nearestExistingPolicyId",
                "s12BehaviorDistanceOnHeldoutPanel",
                "s12ExactBehaviorMatchOnPanel",
                "mappingCaveat",
            ]
        ],
        left_on="policy_id",
        right_on="transferPolicyId",
        how="left",
    ).merge(
        target_catalog[
            [
                "targetId",
                "transferTargetRole",
                "substrateClass",
                "sourceS08WorldEntityId",
                "targetS08WorldEntityId",
                "sourceS08GoalEntityId",
                "targetS08GoalEntityId",
            ]
        ],
        left_on="target_id",
        right_on="targetId",
        how="left",
    )
    rows: list[dict[str, Any]] = []
    group_columns = [
        "policy_id",
        "policy_family",
        "mappingKind",
        "sourcePolicyId",
        "sourceDesignProfileId",
        "sourceProfileName",
        "analoguePolicyFamily",
        "target_id",
        "motif",
        "substrate_type",
        "substrateClass",
        "transferTargetRole",
    ]
    for keys, group in runs.groupby(group_columns, dropna=False, sort=True):
        record = dict(zip(group_columns, keys, strict=True))
        max_steps = pd.to_numeric(group["max_steps"], errors="coerce")
        wait = pd.to_numeric(group["wait_count"], errors="coerce")
        rejected = pd.to_numeric(group["rejected_count"], errors="coerce")
        record.update(
            {
                "schemaVersion": SUBSTRATE_TRANSFER_SCHEMA_VERSION,
                "runCount": int(len(group)),
                "seedCount": int(group["seed"].nunique()),
                "sourceMeanRelativeErrorReduction": math.nan,
                "transferRelativeErrorReductionMean": float(pd.to_numeric(group["relative_error_reduction"], errors="coerce").mean()),
                "transferRelativeErrorReductionStd": float(pd.to_numeric(group["relative_error_reduction"], errors="coerce").std(ddof=0)),
                "transferRelativeErrorReductionMin": float(pd.to_numeric(group["relative_error_reduction"], errors="coerce").min()),
                "transferRelativeErrorReductionMax": float(pd.to_numeric(group["relative_error_reduction"], errors="coerce").max()),
                "finalCompositeErrorMean": float(pd.to_numeric(group["final_composite_error"], errors="coerce").mean()),
                "initialCompositeErrorMean": float(pd.to_numeric(group["initial_composite_error"], errors="coerce").mean()),
                "bestCompositeErrorMean": float(pd.to_numeric(group["best_composite_error"], errors="coerce").mean()),
                "exactRecoveryFraction": float(group["exact_recovery"].astype(bool).mean()),
                "improvedFraction": float(group["improved"].astype(bool).mean()),
                "acceptedSwapCountMean": float(pd.to_numeric(group["accepted_swap_count"], errors="coerce").mean()),
                "explorationSwapCountMean": float(pd.to_numeric(group["exploration_swap_count"], errors="coerce").mean()),
                "waitFractionMean": float((wait / max_steps.replace(0, np.nan)).mean()),
                "rejectedFractionMean": float((rejected / max_steps.replace(0, np.nan)).mean()),
                "trajectoryCurvatureMean": float(pd.to_numeric(group["trajectory_curvature"], errors="coerce").mean()),
                "occupancyPreservedFraction": float(group["occupancy_preserved"].astype(bool).mean()),
                "cellIdsPreservedFraction": float(group["cell_ids_preserved"].astype(bool).mean()),
                "usesWholeTargetLeakageFraction": float(group["uses_whole_target_leakage"].astype(bool).mean()),
                "sourceS08PolicyEntityId": group["s08SourcePolicyEntityId"].dropna().iloc[0] if group["s08SourcePolicyEntityId"].notna().any() else None,
                "nearestExistingS08ProxyEntityId": group["nearestExistingS08ProxyEntityId"].dropna().iloc[0] if group["nearestExistingS08ProxyEntityId"].notna().any() else None,
                "targetS08PolicyEntityId": group["s08TargetPolicyEntityId"].dropna().iloc[0] if group["s08TargetPolicyEntityId"].notna().any() else None,
                "sourceS08WorldEntityId": group["sourceS08WorldEntityId"].dropna().iloc[0] if group["sourceS08WorldEntityId"].notna().any() else None,
                "targetS08WorldEntityId": group["targetS08WorldEntityId"].dropna().iloc[0] if group["targetS08WorldEntityId"].notna().any() else None,
                "sourceS08GoalEntityId": group["sourceS08GoalEntityId"].dropna().iloc[0] if group["sourceS08GoalEntityId"].notna().any() else None,
                "targetS08GoalEntityId": group["targetS08GoalEntityId"].dropna().iloc[0] if group["targetS08GoalEntityId"].notna().any() else None,
                "nearestExistingPolicyId": group["nearestExistingPolicyId"].dropna().iloc[0] if group["nearestExistingPolicyId"].notna().any() else None,
                "s12BehaviorDistanceOnHeldoutPanel": _finite(group["s12BehaviorDistanceOnHeldoutPanel"].dropna().iloc[0]) if group["s12BehaviorDistanceOnHeldoutPanel"].notna().any() else math.nan,
                "s12ExactBehaviorMatchOnPanel": bool(group["s12ExactBehaviorMatchOnPanel"].dropna().iloc[0]) if group["s12ExactBehaviorMatchOnPanel"].notna().any() else None,
                "mappingCaveat": group["mappingCaveat"].dropna().iloc[0] if group["mappingCaveat"].notna().any() else None,
            }
        )
        rows.append(record)
    summary = pd.DataFrame(rows)
    source = (
        summary[summary["target_id"].eq(SOURCE_TARGET_ID)][["policy_id", "transferRelativeErrorReductionMean"]]
        .rename(columns={"transferRelativeErrorReductionMean": "sourceMeanRelativeErrorReduction"})
        .copy()
    )
    summary = summary.drop(columns=["sourceMeanRelativeErrorReduction"]).merge(source, on="policy_id", how="left")
    source_mean = pd.to_numeric(summary["sourceMeanRelativeErrorReduction"], errors="coerce")
    transfer_mean = pd.to_numeric(summary["transferRelativeErrorReductionMean"], errors="coerce")
    summary["competenceRetentionRatio"] = np.where(source_mean > 1e-12, transfer_mean / source_mean, np.nan)
    summary["transferGap"] = source_mean - transfer_mean
    control_mean = (
        summary[summary["mappingKind"].isin(["randomized_transfer_control", "random_transfer_control", "no_transfer_control"])]
        .groupby("target_id")["transferRelativeErrorReductionMean"]
        .mean()
        .rename("targetRandomOrNoTransferControlMean")
    )
    bubble_mean = (
        summary[summary["policy_id"].eq("classic_scalar_rank_swap")]
        .groupby("target_id")["transferRelativeErrorReductionMean"]
        .mean()
        .rename("targetBubbleLikeBaselineMean")
    )
    summary = summary.merge(control_mean, on="target_id", how="left").merge(bubble_mean, on="target_id", how="left")
    summary["beatsRandomOrNoTransferControl"] = summary["transferRelativeErrorReductionMean"] > summary["targetRandomOrNoTransferControlMean"]
    summary["beatsBubbleLikeBaseline"] = summary["transferRelativeErrorReductionMean"] > summary["targetBubbleLikeBaselineMean"]
    summary["failureMode"] = [classify_transfer_failure(row) for row in summary.to_dict(orient="records")]
    return summary.sort_values(["target_id", "mappingKind", "policy_id"], kind="mergesort").reset_index(drop=True)


def classify_transfer_failure(row: Mapping[str, Any]) -> str:
    transfer = _finite(row.get("transferRelativeErrorReductionMean"))
    retention = _finite(row.get("competenceRetentionRatio"))
    random_mean = _finite(row.get("targetRandomOrNoTransferControlMean"))
    exact_fraction = _finite(row.get("exactRecoveryFraction"), 0.0)
    improved_fraction = _finite(row.get("improvedFraction"), 0.0)
    wait_fraction = _finite(row.get("waitFractionMean"), 0.0)
    if wait_fraction >= 0.85:
        return "stalled_waiting"
    if improved_fraction < 0.34:
        return "mostly_not_improved"
    if math.isfinite(random_mean) and transfer <= random_mean + 0.02:
        return "near_or_below_random_control"
    if math.isfinite(retention) and row.get("target_id") != SOURCE_TARGET_ID and retention < 0.5:
        return "low_competence_retention"
    if exact_fraction < 0.34 and transfer < 0.25:
        return "no_exact_recovery_low_reduction"
    return "retained_partial_competence"


class S08DistanceLookup:
    def __init__(self, distances: pd.DataFrame) -> None:
        self._distances: dict[tuple[str, str, str], float] = {}
        for row in distances[["entityType", "entityIdA", "entityIdB", "platonicDistance"]].itertuples(index=False):
            entity_type = str(row.entityType)
            a = str(row.entityIdA)
            b = str(row.entityIdB)
            value = _finite(row.platonicDistance)
            self._distances[(entity_type, a, b)] = value
            self._distances[(entity_type, b, a)] = value

    def get(self, entity_type: str, left: Any, right: Any) -> float:
        if left is None or right is None:
            return math.nan
        if pd.isna(left) or pd.isna(right):
            return math.nan
        left_id = str(left)
        right_id = str(right)
        if left_id == right_id:
            return 0.0
        return self._distances.get((str(entity_type), left_id, right_id), math.nan)


def annotate_s08_transfer_distances(summary: pd.DataFrame, distances: pd.DataFrame) -> pd.DataFrame:
    lookup = S08DistanceLookup(distances)
    rows: list[dict[str, Any]] = []
    for record in summary.to_dict(orient="records"):
        policy_distance = lookup.get("policy", record.get("sourceS08PolicyEntityId"), record.get("targetS08PolicyEntityId"))
        world_distance = lookup.get("world", record.get("sourceS08WorldEntityId"), record.get("targetS08WorldEntityId"))
        goal_distance = lookup.get("goal", record.get("sourceS08GoalEntityId"), record.get("targetS08GoalEntityId"))
        values = [value for value in (policy_distance, world_distance, goal_distance) if math.isfinite(value)]
        missing = [name for name, value in (("policy", policy_distance), ("world", world_distance), ("goal", goal_distance)) if not math.isfinite(value)]
        record.update(
            {
                "s08PolicyDistance": policy_distance,
                "s08WorldDistance": world_distance,
                "s08GoalDistance": goal_distance,
                "s08CompositeTransferDistance": float(np.mean(values)) if values else math.nan,
                "s08DistanceMissingComponents": ",".join(missing),
                "s08DistanceCaveat": (
                    "No complete S08 distance tuple for this row."
                    if missing
                    else "Composite is the mean of empirical S08 policy, world, and goal distances."
                ),
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


def distance_prediction_correlations(distance_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    frame = distance_summary[
        distance_summary["transferTargetRole"].eq("transfer_holdout")
        & distance_summary["mappingKind"].isin(["s12_designed_policy_transfer", "baseline_transfer_control"])
    ].copy()
    predictors = ["s08PolicyDistance", "s08WorldDistance", "s08GoalDistance", "s08CompositeTransferDistance"]
    outcomes = ["transferRelativeErrorReductionMean", "competenceRetentionRatio", "transferGap"]
    for predictor in predictors:
        for outcome in outcomes:
            subset = frame[[predictor, outcome]].apply(pd.to_numeric, errors="coerce").dropna()
            subset = subset[np.isfinite(subset[predictor]) & np.isfinite(subset[outcome])]
            unique_x = subset[predictor].nunique()
            unique_y = subset[outcome].nunique()
            if len(subset) >= 3 and unique_x > 1 and unique_y > 1:
                spearman = stats.spearmanr(subset[predictor], subset[outcome])
                pearson = stats.pearsonr(subset[predictor], subset[outcome])
                spearman_r = float(spearman.statistic)
                spearman_p = float(spearman.pvalue)
                pearson_r = float(pearson.statistic)
                pearson_p = float(pearson.pvalue)
                status = "computed"
            else:
                spearman_r = math.nan
                spearman_p = math.nan
                pearson_r = math.nan
                pearson_p = math.nan
                status = "insufficient_nonconstant_pairs"
            rows.append(
                {
                    "schemaVersion": SUBSTRATE_TRANSFER_SCHEMA_VERSION,
                    "predictor": predictor,
                    "outcome": outcome,
                    "n": int(len(subset)),
                    "uniquePredictorValues": int(unique_x),
                    "uniqueOutcomeValues": int(unique_y),
                    "spearmanR": spearman_r,
                    "spearmanP": spearman_p,
                    "pearsonR": pearson_r,
                    "pearsonP": pearson_p,
                    "status": status,
                    "interpretation": correlation_interpretation(predictor, outcome, spearman_r),
                }
            )
    return pd.DataFrame(rows)


def correlation_interpretation(predictor: str, outcome: str, spearman_r: float) -> str:
    if not math.isfinite(spearman_r):
        return "not interpretable because available pairs are sparse or constant"
    if outcome in {"transferRelativeErrorReductionMean", "competenceRetentionRatio"}:
        if spearman_r <= -0.35:
            return f"larger {predictor} is associated with lower transfer competence"
        if spearman_r >= 0.35:
            return f"larger {predictor} is unexpectedly associated with higher transfer competence"
    if outcome == "transferGap":
        if spearman_r >= 0.35:
            return f"larger {predictor} is associated with larger transfer gaps"
        if spearman_r <= -0.35:
            return f"larger {predictor} is unexpectedly associated with smaller transfer gaps"
    return "weak or directionally ambiguous association"


def validation_checks(
    *,
    designed_policies: pd.DataFrame,
    policy_panel: TransferPolicyPanel,
    target_panel: TransferTargetPanel,
    run_rows: pd.DataFrame,
    transfer_summary: pd.DataFrame,
    distance_correlations: pd.DataFrame,
    seeds: Sequence[int] = DEFAULT_TRANSFER_SEEDS,
    upstream_statuses: Sequence[Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    mapping = policy_panel.mapping_catalog
    target_catalog = target_panel.target_catalog
    expected_runs = len(policy_panel.policies) * len(target_panel.targets) * len(seeds)
    s12_sources = set(designed_policies["policyId"].astype(str))
    mapped_sources = set(mapping[mapping["mappingKind"].eq("s12_designed_policy_transfer")]["sourcePolicyId"].dropna().astype(str))
    hard_statuses_ok = True
    if upstream_statuses is not None:
        hard_statuses_ok = all(bool(row.get("success")) for row in upstream_statuses)
    rows = [
        {
            "checkId": "upstream_statuses_successful",
            "severity": "error",
            "success": bool(hard_statuses_ok),
            "observed": f"{sum(bool(row.get('success')) for row in upstream_statuses or [])}/{len(upstream_statuses or [])}",
            "expected": "all required upstream S01-S12 statuses successful",
        },
        {
            "checkId": "all_s12_designed_policies_mapped",
            "severity": "error",
            "success": s12_sources == mapped_sources,
            "observed": ",".join(sorted(mapped_sources)),
            "expected": ",".join(sorted(s12_sources)),
        },
        {
            "checkId": "transfer_mapping_documented",
            "severity": "error",
            "success": bool((mapping["executed"].astype(bool)).any() and mapping["observationMappingJson"].notna().all()),
            "observed": f"{int(mapping['executed'].astype(bool).sum())} executed mappings, {len(mapping)} total rows",
            "expected": "executed mappings and mapping contract fields present",
        },
        {
            "checkId": "baseline_and_random_controls_present",
            "severity": "error",
            "success": {
                "baseline_transfer_control",
                "random_transfer_control",
                "randomized_transfer_control",
                "no_transfer_control",
            }.issubset(set(mapping["mappingKind"].astype(str))),
            "observed": ",".join(sorted(set(mapping["mappingKind"].astype(str)))),
            "expected": "baseline, random, randomized, and no-transfer controls",
        },
        {
            "checkId": "substrate_classes_covered",
            "severity": "error",
            "success": {"row_1d", "square_grid_2d", "irregular_graph"}.issubset(set(target_catalog["substrateClass"].astype(str))),
            "observed": ",".join(sorted(set(target_catalog["substrateClass"].astype(str)))),
            "expected": "row_1d,square_grid_2d,irregular_graph",
        },
        {
            "checkId": "direct_run_count_expected",
            "severity": "error",
            "success": int(len(run_rows)) == expected_runs,
            "observed": str(len(run_rows)),
            "expected": str(expected_runs),
        },
        {
            "checkId": "conservative_state_invariants",
            "severity": "error",
            "success": bool(
                run_rows["occupancy_preserved"].astype(bool).all()
                and run_rows["cell_ids_preserved"].astype(bool).all()
                and not run_rows["uses_whole_target_leakage"].astype(bool).any()
            ),
            "observed": (
                f"occupancy={run_rows['occupancy_preserved'].astype(bool).mean():.3f}; "
                f"cellIds={run_rows['cell_ids_preserved'].astype(bool).mean():.3f}; "
                f"leakage={run_rows['uses_whole_target_leakage'].astype(bool).mean():.3f}"
            ),
            "expected": "occupancy/cell-id preservation 1.0 and whole-target leakage 0.0",
        },
        {
            "checkId": "policy_payload_audits_pass",
            "severity": "error",
            "success": all(audit_recovery_policy_payload(policy)["success"] for policy in policy_panel.policies),
            "observed": "audited transfer policy payloads",
            "expected": "no prohibited whole-target leakage keys",
        },
        {
            "checkId": "unsupported_mappings_documented",
            "severity": "warning",
            "success": int(mapping["mappingKind"].eq("unsupported_documented_mapping").sum()) >= 2,
            "observed": str(int(mapping["mappingKind"].eq("unsupported_documented_mapping").sum())),
            "expected": "at least two documented unsupported exact mappings",
        },
        {
            "checkId": "s08_distance_prediction_attempted",
            "severity": "warning",
            "success": bool(not distance_correlations.empty and (distance_correlations["status"].eq("computed")).any()),
            "observed": ",".join(sorted(set(distance_correlations.get("status", pd.Series(dtype=str)).astype(str)))),
            "expected": "at least one computable distance-outcome correlation",
        },
        {
            "checkId": "transfer_summary_has_failure_modes",
            "severity": "error",
            "success": bool(not transfer_summary.empty and transfer_summary["failureMode"].notna().all()),
            "observed": ",".join(sorted(set(transfer_summary.get("failureMode", pd.Series(dtype=str)).astype(str)))),
            "expected": "failure mode assigned to every policy-target summary",
        },
    ]
    return pd.DataFrame(rows)


def transfer_outcome_classification(
    transfer_summary: pd.DataFrame,
    distance_correlations: pd.DataFrame,
) -> str:
    candidate = transfer_summary[
        transfer_summary["mappingKind"].eq("s12_designed_policy_transfer")
        & transfer_summary["transferTargetRole"].eq("transfer_holdout")
    ].copy()
    if candidate.empty:
        return "null"
    candidate_retention = pd.to_numeric(candidate["competenceRetentionRatio"], errors="coerce")
    candidate_beats_random = candidate["beatsRandomOrNoTransferControl"].astype(bool)
    median_retention = float(candidate_retention.dropna().median()) if candidate_retention.notna().any() else math.nan
    composite_rows = distance_correlations[
        distance_correlations["predictor"].eq("s08CompositeTransferDistance")
        & distance_correlations["outcome"].eq("competenceRetentionRatio")
        & distance_correlations["status"].eq("computed")
    ]
    composite_r = _finite(composite_rows["spearmanR"].iloc[0]) if not composite_rows.empty else math.nan
    if candidate_beats_random.mean() >= 0.65 and math.isfinite(median_retention) and median_retention >= 0.5 and math.isfinite(composite_r) and composite_r <= -0.35:
        return "supportive"
    if candidate_beats_random.mean() < 0.5 or (math.isfinite(composite_r) and composite_r > 0.0):
        return "constraining/contradictory"
    return "null"
