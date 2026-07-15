"""Frozen S13 counterexample-search primitives.

The module deliberately owns no access to S06 or S11 outcome files.  It builds
new S13-only scenarios, evaluates the five already-frozen S08 signatures, and
provides deterministic genome mutation and confirmation perturbations.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np

from causal_simulator.screening import run_compact_screening
from reference_simulator.model import (
    Architecture,
    Cell,
    Direction,
    FaultMode,
    Policy,
    Scenario,
    canonical_json_bytes,
    sha256_json,
)
from reference_simulator.rng import bounded


SEARCH_SCHEMA_VERSION = "e02.s13.counterexample-search.v1"
PRESPECIFICATION_SHA256 = (
    "71ba19f859b40213f414c76bc91281d311228d7a3b309b89bff47e85b896e627"
)
TRAINING_MASTER_SEED = int("e0213000000000000000000000000001", 16)
CONFIRMATION_MASTER_SEED = int("e0213c00000000000000000000000001", 16)
TRAINING_DOMAIN = "E02/S13/training/v1"
CONFIRMATION_SCHEDULER_DOMAIN = "E02/S13/confirmation/scheduler/v1"
CONFIRMATION_FAULT_DOMAIN = "E02/S13/confirmation/fault-map/v1"
CONFIRMATION_BOOTSTRAP_DOMAIN = "E02/S13/confirmation/bootstrap/v1"
OBJECTIVE_IDS = (
    "E04_reference_advantage",
    "E06_active_advantage",
    "E08_weak_advantage",
    "E08_reference_advantage",
)


@dataclass(frozen=True, slots=True)
class Candidate:
    n: int
    value_profile: str
    fault_count: int
    occupancy: tuple[int, ...]
    fault_identities: tuple[int, ...]

    @property
    def stratum_id(self) -> str:
        return f"n{self.n}_{self.value_profile}_f{self.fault_count}"

    @property
    def candidate_id(self) -> str:
        return "s13c1:" + sha256_json(self.canonical_content())

    @property
    def fault_map_id(self) -> str:
        return "s13fp1:" + sha256_json(
            {
                "schemaVersion": SEARCH_SCHEMA_VERSION,
                "candidateId": self.candidate_id,
                "faultIdentities": list(self.fault_identities),
            }
        )

    def canonical_content(self) -> dict[str, Any]:
        return {
            "schemaVersion": SEARCH_SCHEMA_VERSION,
            "n": self.n,
            "valueProfile": self.value_profile,
            "direction": "ascending",
            "policyProfile": "Selection",
            "faultCount": self.fault_count,
            "occupancy": list(self.occupancy),
            "faultIdentities": list(self.fault_identities),
        }

    def to_record(self) -> dict[str, Any]:
        return {
            **self.canonical_content(),
            "stratumId": self.stratum_id,
            "candidateId": self.candidate_id,
            "faultMapId": self.fault_map_id,
            "initialValues": [value_for_identity(self.n, self.value_profile, item) for item in self.occupancy],
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "Candidate":
        return cls(
            n=int(record["n"]),
            value_profile=str(record["valueProfile"]),
            fault_count=int(record["faultCount"]),
            occupancy=tuple(map(int, record["occupancy"])),
            fault_identities=tuple(sorted(map(int, record["faultIdentities"]))),
        )

    def validate(self) -> None:
        if self.n not in {20, 50}:
            raise ValueError("S13 candidate size outside frozen region")
        if self.value_profile not in {"unique", "balanced_duplicate"}:
            raise ValueError("S13 candidate value profile outside frozen region")
        if self.fault_count not in {2, 4}:
            raise ValueError("S13 exact fault count outside frozen region")
        if tuple(sorted(self.occupancy)) != tuple(range(self.n)):
            raise ValueError("candidate occupancy is not a full identity permutation")
        if len(self.fault_identities) != self.fault_count:
            raise ValueError("candidate exact fault count was not preserved")
        if tuple(sorted(set(self.fault_identities))) != self.fault_identities:
            raise ValueError("candidate fault identities must be unique and sorted")
        if not set(self.fault_identities).issubset(range(self.n)):
            raise ValueError("candidate fault identity outside scenario")


def _address(*parts: object) -> str:
    return "/".join(map(str, parts))


def _shuffle(items: Sequence[int], *, master_seed: int, address: str, stream: str) -> tuple[int, ...]:
    result = list(items)
    cursor = 0
    for index in range(len(result) - 1, 0, -1):
        selected, consumed = bounded(master_seed, address, stream, cursor, index + 1)
        cursor += consumed
        result[index], result[selected] = result[selected], result[index]
    return tuple(result)


def _draw_distinct(
    upper: int,
    count: int,
    *,
    master_seed: int,
    address: str,
    stream: str,
) -> tuple[int, ...]:
    if not 0 <= count <= upper:
        raise ValueError("invalid distinct draw request")
    permutation = _shuffle(
        tuple(range(upper)), master_seed=master_seed, address=address, stream=stream
    )
    return tuple(permutation[:count])


def derive_seed(master_seed: int, *parts: object) -> int:
    digest = hashlib.sha256(
        canonical_json_bytes({"masterSeed": str(master_seed), "parts": list(parts)})
    ).digest()
    return int.from_bytes(digest[:16], "big")


def value_for_identity(n: int, value_profile: str, identity: int) -> int:
    if value_profile == "unique":
        return identity
    if value_profile == "balanced_duplicate":
        return identity // (n // 10)
    raise ValueError(value_profile)


def initial_candidate(n: int, value_profile: str, fault_count: int, slot: int) -> Candidate:
    address = _address(TRAINING_DOMAIN, "initial", n, value_profile, fault_count, slot)
    occupancy = _shuffle(
        tuple(range(n)),
        master_seed=TRAINING_MASTER_SEED,
        address=address,
        stream="initial_occupancy_s13_v1",
    )
    fault_order = _shuffle(
        tuple(range(n)),
        master_seed=TRAINING_MASTER_SEED,
        address=address,
        stream="initial_fault_identities_s13_v1",
    )
    candidate = Candidate(n, value_profile, fault_count, occupancy, tuple(sorted(fault_order[:fault_count])))
    candidate.validate()
    return candidate


def mutate_candidate(parent: Candidate, generation: int, slot: int, attempt: int) -> Candidate:
    """Apply one of the four frozen exact-count mutation operators."""

    parent.validate()
    address = _address(
        TRAINING_DOMAIN,
        "mutation",
        parent.stratum_id,
        generation,
        slot,
        attempt,
        parent.candidate_id,
    )
    operator = slot % 4
    occupancy = list(parent.occupancy)
    faults = set(parent.fault_identities)
    if operator in {0, 3}:
        left, right = _draw_distinct(
            parent.n,
            2,
            master_seed=TRAINING_MASTER_SEED,
            address=address,
            stream="occupancy_swap_s13_v1",
        )
        occupancy[left], occupancy[right] = occupancy[right], occupancy[left]
    elif operator == 1:
        first, second, third = _draw_distinct(
            parent.n,
            3,
            master_seed=TRAINING_MASTER_SEED,
            address=address,
            stream="occupancy_rotate_s13_v1",
        )
        occupancy[first], occupancy[second], occupancy[third] = (
            occupancy[third], occupancy[first], occupancy[second]
        )
    if operator in {2, 3}:
        remove_order = _shuffle(
            tuple(sorted(faults)),
            master_seed=TRAINING_MASTER_SEED,
            address=address,
            stream="fault_remove_s13_v1",
        )
        add_order = _shuffle(
            tuple(sorted(set(range(parent.n)) - faults)),
            master_seed=TRAINING_MASTER_SEED,
            address=address,
            stream="fault_add_s13_v1",
        )
        faults.remove(remove_order[0])
        faults.add(add_order[0])
    candidate = Candidate(
        parent.n,
        parent.value_profile,
        parent.fault_count,
        tuple(occupancy),
        tuple(sorted(faults)),
    )
    candidate.validate()
    return candidate


def confirmation_neighbor(candidate: Candidate, replicate: int) -> Candidate:
    """Create the frozen two-swap/one-fault-replacement local perturbation."""

    address = _address(
        CONFIRMATION_FAULT_DOMAIN, candidate.candidate_id, replicate
    )
    positions = _draw_distinct(
        candidate.n,
        4,
        master_seed=CONFIRMATION_MASTER_SEED,
        address=address,
        stream="two_occupancy_swaps_s13_v1",
    )
    occupancy = list(candidate.occupancy)
    occupancy[positions[0]], occupancy[positions[1]] = occupancy[positions[1]], occupancy[positions[0]]
    occupancy[positions[2]], occupancy[positions[3]] = occupancy[positions[3]], occupancy[positions[2]]
    faults = set(candidate.fault_identities)
    removed = _shuffle(
        tuple(sorted(faults)),
        master_seed=CONFIRMATION_MASTER_SEED,
        address=address,
        stream="fault_remove_s13_confirm_v1",
    )[0]
    added = _shuffle(
        tuple(sorted(set(range(candidate.n)) - faults)),
        master_seed=CONFIRMATION_MASTER_SEED,
        address=address,
        stream="fault_add_s13_confirm_v1",
    )[0]
    faults.remove(removed)
    faults.add(added)
    neighbor = Candidate(
        candidate.n,
        candidate.value_profile,
        candidate.fault_count,
        tuple(occupancy),
        tuple(sorted(faults)),
    )
    neighbor.validate()
    return neighbor


def candidate_distance(left: Candidate, right: Candidate) -> float:
    if (left.n, left.value_profile, left.fault_count) != (
        right.n, right.value_profile, right.fault_count
    ):
        return 1.0
    hamming = sum(a != b for a, b in zip(left.occupancy, right.occupancy)) / left.n
    left_faults, right_faults = set(left.fault_identities), set(right.fault_identities)
    jaccard_distance = 1.0 - len(left_faults & right_faults) / len(left_faults | right_faults)
    return 0.75 * hamming + 0.25 * jaccard_distance


BASE_SETTINGS = {
    "legalPrimitives": "NoOp|Swap|MemoryUpdate",
    "informationPermission": "policy_native_local",
    "proposalCandidatesPerOpportunity": "1",
    "sensing": "exact",
    "actionFailure": "none",
    "retry": "no_retry",
}

RUN_SIGNATURES: dict[str, dict[str, str]] = {
    "local_passive_skip_none": {
        **BASE_SETTINGS,
        "architecture": "distributed_local",
        "coordinatorProfile": "none",
        "mobility": "passive",
        "continuation": "skip_and_continue",
    },
    "local_stuck_skip_none": {
        **BASE_SETTINGS,
        "architecture": "distributed_local",
        "coordinatorProfile": "none",
        "mobility": "stuck",
        "continuation": "skip_and_continue",
    },
    "local_stuck_stop_none": {
        **BASE_SETTINGS,
        "architecture": "distributed_local",
        "coordinatorProfile": "none",
        "mobility": "stuck",
        "continuation": "stop_on_first_blocking_failure",
    },
    "weak_owner_passive_skip_none": {
        **BASE_SETTINGS,
        "architecture": "distributed_weak_coordinator",
        "coordinatorProfile": "none",
        "mobility": "passive",
        "continuation": "skip_and_continue",
    },
    "weak_owner_passive_skip_active": {
        **BASE_SETTINGS,
        "architecture": "distributed_weak_coordinator",
        "coordinatorProfile": "weak_frozen_budget",
        "mobility": "passive",
        "continuation": "skip_and_continue",
    },
}

CONTRAST_ARMS = {
    "E04_reference_advantage": (
        "local_stuck_skip_none", "local_stuck_stop_none", 1.0, "reference"
    ),
    "E06_active_advantage": (
        "local_stuck_skip_none", "local_passive_skip_none", -1.0, "active"
    ),
    "E08_weak_advantage": (
        "weak_owner_passive_skip_active", "weak_owner_passive_skip_none", -1.0, "active"
    ),
    "E08_reference_advantage": (
        "weak_owner_passive_skip_active", "weak_owner_passive_skip_none", 1.0, "reference"
    ),
}


def _scenario(
    candidate: Candidate,
    mobility: str,
    seed: int,
    generation_key: str,
) -> Scenario:
    fault_mode = FaultMode(mobility)
    fault_ids = set(candidate.fault_identities)
    cells = tuple(
        Cell(
            f"cell-{identity:04d}",
            value_for_identity(candidate.n, candidate.value_profile, identity),
            Policy.SELECTION,
            Direction.ASCENDING,
            fault_mode if identity in fault_ids else FaultMode.NORMAL,
        )
        for identity in range(candidate.n)
    )
    return Scenario.create(
        cells,
        initial_occupancy=tuple(f"cell-{identity:04d}" for identity in candidate.occupancy),
        seed=seed,
        max_activations=100 * candidate.n * candidate.n,
        architecture=Architecture.CELL_VIEW,
        generation_key=generation_key,
        fault_placement="explicit",
        requested_fault_count=candidate.fault_count,
    )


def evaluate_candidate(
    candidate_record: Mapping[str, Any],
    *,
    stage: str,
    scheduler: str,
    seed: int,
    instance_id: str,
) -> dict[str, Any]:
    """Evaluate all five frozen signatures for one candidate instance."""

    candidate = Candidate.from_record(candidate_record)
    candidate.validate()
    if stage not in {"training", "confirmation_exact", "confirmation_neighborhood"}:
        raise ValueError(stage)
    arm_results: dict[str, dict[str, Any]] = {}
    run_rows: list[dict[str, Any]] = []
    for signature, base_settings in RUN_SIGNATURES.items():
        settings = {**base_settings, "scheduler": scheduler}
        # Architecture ownership is held in settings; executable scenario IDs
        # are shared exactly where S08 declares common runtime roots.
        mobility = settings["mobility"]
        generation_key = f"{stage}/{instance_id}/{mobility}"
        scenario = _scenario(candidate, mobility, seed, generation_key)
        result = run_compact_screening(scenario, settings)
        arm_results[signature] = result
        run_rows.append(
            {
                "stage": stage,
                "instanceId": instance_id,
                "candidateId": candidate.candidate_id,
                "faultMapId": candidate.fault_map_id,
                "signature": signature,
                "scheduler": scheduler,
                "scenarioId": result["scenarioId"],
                "seed": str(seed),
                "normalizedResidualError": result["normalizedResidualError"],
                "strictAdjacentResidualCount": result["strictAdjacentResidualCount"],
                "successByBudget": result["successByBudget"],
                "stopReason": result["stopReason"],
                "completionOpportunity": result["completionOpportunity"],
                "s01UnitWeightFullCost": result["projection_s01UnitWeightFullCost"],
                "controllerExpandedSensitivity": result["projection_controllerExpandedSensitivity"],
                "deterministicResultSha256": result["deterministicResultSha256"],
                "compactReplayDigest": result["compactReplayDigest"],
                "contractValidationPass": result["contractValidationPass"],
                "contractValidationJson": result["contractValidationJson"],
                "nativeLedger": dict(result["nativeLedger"]),
                "streamCounters": dict(result["streamCounters"]),
            }
        )
    contrasts = contrast_metrics(arm_results)
    return {
        "candidate": candidate.to_record(),
        "stage": stage,
        "instanceId": instance_id,
        "scheduler": scheduler,
        "seed": str(seed),
        "runRows": run_rows,
        "contrasts": contrasts,
    }


def contrast_metrics(arm_results: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for objective_id, (active_name, reference_name, sign, target_role) in CONTRAST_ARMS.items():
        active = arm_results[active_name]
        reference = arm_results[reference_name]
        residual_delta = float(active["normalizedResidualError"]) - float(reference["normalizedResidualError"])
        active_success = int(active["successByBudget"])
        reference_success = int(reference["successByBudget"])
        target_only = (
            active_success == 1 and reference_success == 0
            if target_role == "active"
            else reference_success == 1 and active_success == 0
        )
        opposite_only = (
            reference_success == 1 and active_success == 0
            if target_role == "active"
            else active_success == 1 and reference_success == 0
        )
        target = active if target_role == "active" else reference
        metrics[objective_id] = {
            "objectiveId": objective_id,
            "activeSignature": active_name,
            "referenceSignature": reference_name,
            "activeMinusReferenceResidual": residual_delta,
            "directionalResidualScore": sign * residual_delta,
            "activeSuccess": active_success,
            "referenceSuccess": reference_success,
            "targetOnlySuccess": int(target_only),
            "oppositeOnlySuccess": int(opposite_only),
            "targetLogS01Cost": math.log(float(target["projection_s01UnitWeightFullCost"]) + 0.5),
            "discoveryThresholdMet": bool(sign * residual_delta >= 0.02 or target_only),
        }
    return metrics


def objective_sort_key(evaluation: Mapping[str, Any], objective_id: str) -> tuple[Any, ...]:
    metric = evaluation["contrasts"][objective_id]
    return (
        -int(metric["targetOnlySuccess"]),
        -float(metric["directionalResidualScore"]),
        float(metric["targetLogS01Cost"]),
        str(evaluation["candidate"]["candidateId"]),
    )


def flatten_evaluation(evaluation: Mapping[str, Any], generation: int) -> dict[str, Any]:
    candidate = dict(evaluation["candidate"])
    row: dict[str, Any] = {
        **candidate,
        "generation": generation,
        "stage": evaluation["stage"],
        "instanceId": evaluation["instanceId"],
        "scheduler": evaluation["scheduler"],
        "seed": evaluation["seed"],
    }
    for objective_id, metric in evaluation["contrasts"].items():
        prefix = objective_id + "__"
        for key, value in metric.items():
            if key != "objectiveId":
                row[prefix + key] = value
    return row


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return (math.nan, math.nan)
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = proportion + z * z / (2.0 * total)
    spread = z * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
    return ((centre - spread) / denominator, (centre + spread) / denominator)


def paired_bootstrap_interval(
    values: Sequence[float],
    *,
    objective_id: str,
    panel: str,
    replicates: int = 5000,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return (math.nan, math.nan)
    seed = derive_seed(
        CONFIRMATION_MASTER_SEED,
        CONFIRMATION_BOOTSTRAP_DOMAIN,
        objective_id,
        panel,
        len(array),
    )
    generator = np.random.Generator(np.random.PCG64(seed))
    # Chunking caps temporary memory while preserving the frozen replicate order.
    means: list[np.ndarray] = []
    remaining = replicates
    while remaining:
        count = min(500, remaining)
        indices = generator.integers(0, len(array), size=(count, len(array)))
        means.append(array[indices].mean(axis=1))
        remaining -= count
    distribution = np.concatenate(means)
    return tuple(map(float, np.quantile(distribution, [0.025, 0.975])))


def spearman_rank(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 3 or len(right) != len(left):
        return None
    left_array, right_array = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if np.ptp(left_array) == 0 or np.ptp(right_array) == 0:
        return None
    left_rank = np.argsort(np.argsort(left_array, kind="stable"), kind="stable")
    right_rank = np.argsort(np.argsort(right_array, kind="stable"), kind="stable")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Mapping[str, Any] | Sequence[Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()
