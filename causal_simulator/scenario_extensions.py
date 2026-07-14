"""Deterministic scale and input-structure scenarios for E02 S07.

This module constructs immutable *input* scenarios only.  Architecture,
scheduler, policy, fault-map, and exogenous-stream assignments remain owned by
later design steps.  In particular, no outcome-selected S06 placement can be
attached through this API.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
import hashlib
import itertools
import math
from typing import Any, Iterable, Mapping, Sequence

from reference_simulator.model import canonical_json_bytes, sha256_json
from reference_simulator.rng import bounded, u64


SCENARIO_EXTENSION_VERSION = "E02-scale-input-suite-v1"
PRESPECIFICATION_SHA256 = (
    "8ee409f3376d016d2bf75a83d9c23632e2a59fafd8dbe7f3d0a4c9bea39d56cc"
)
MASTER_SEED = int("e0207000000000000000000000000001", 16)
SIZES = (20, 50, 100, 200, 500)
MAX_GENERATION_ATTEMPTS = 4096


class ValueProfile(str, Enum):
    UNIQUE = "unique"
    BALANCED_DUPLICATE = "balanced_duplicate"
    UNEVEN_DUPLICATE = "uneven_duplicate"


class OrderStructure(str, Enum):
    NEARLY_SORTED = "nearly_sorted"
    REVERSE = "reverse"
    BLOCK_SCRAMBLED = "block_scrambled"


class ScenarioSplit(str, Enum):
    SCREENING_POOL = "screening_pool"
    CONFIRMATORY_HOLDOUT = "confirmatory_holdout"
    RUNTIME_VALIDATION = "runtime_validation"


SPLIT_REPLICATES = {
    ScenarioSplit.SCREENING_POOL: 250,
    ScenarioSplit.CONFIRMATORY_HOLDOUT: 1000,
    ScenarioSplit.RUNTIME_VALIDATION: 4,
}
TRACE_RATES = {20: 0.05, 50: 0.04, 100: 0.02, 200: 0.01, 500: 0.005}
STRUCTURAL_PLACEMENT_CLASSES = (
    "uniform_exact",
    "clustered_exact",
    "boundary_exact",
    "median_rank_exact",
)
FORBIDDEN_CONFIRMATORY_PLACEMENT_CLASS = "search_derived_exploratory"


def _address(
    split: ScenarioSplit,
    n: int,
    value_profile: ValueProfile,
    order_structure: OrderStructure,
    replicate_ordinal: int,
    generation_attempt: int,
) -> str:
    return "S07/" + sha256_json(
        {
            "schemaVersion": SCENARIO_EXTENSION_VERSION,
            "split": split.value,
            "n": n,
            "valueProfile": value_profile.value,
            "orderStructure": order_structure.value,
            "replicateOrdinal": replicate_ordinal,
            "generationAttempt": generation_attempt,
        }
    )


def value_counts(n: int, profile: ValueProfile) -> tuple[int, ...]:
    """Return the frozen exact multiplicity vector in increasing value rank."""

    if n not in SIZES:
        raise ValueError(f"n must be one of {SIZES}")
    if profile == ValueProfile.UNIQUE:
        return (1,) * n
    if n % 10:
        raise ValueError("duplicate profiles require n divisible by ten")
    if profile == ValueProfile.BALANCED_DUPLICATE:
        return (n // 10,) * 10

    remaining = n - 10
    exact = tuple(Fraction(remaining * weight, 55) for weight in range(1, 11))
    floors = [item.numerator // item.denominator for item in exact]
    leftover = remaining - sum(floors)
    priority = sorted(
        range(10),
        key=lambda index: (-(exact[index] - floors[index]), index),
    )
    for index in priority[:leftover]:
        floors[index] += 1
    counts = tuple(item + 1 for item in floors)
    if sum(counts) != n or min(counts) < 1 or len(set(counts)) == 1:
        raise AssertionError("uneven exact allocation failed")
    return counts


def _level_values(address: str, level_count: int) -> tuple[int, ...]:
    values: list[int] = []
    current = 0
    for rank in range(level_count):
        gap, _ = bounded(MASTER_SEED, address, "value_gap_s07_v1", rank, 16)
        current += gap + 1
        values.append(current)
    return tuple(values)


def _permutation(items: Sequence[int], address: str, stream: str) -> tuple[int, ...]:
    result = list(items)
    for index in range(len(result) - 1, 0, -1):
        selected, _ = bounded(MASTER_SEED, address, stream, index, index + 1)
        result[index], result[selected] = result[selected], result[index]
    return tuple(result)


def strict_unequal_inversions(values: Sequence[int | float]) -> int:
    """Count strict unequal-value inversions in O(n log n)."""

    ranks = {value: rank + 1 for rank, value in enumerate(sorted(set(values)))}
    tree = [0] * (len(ranks) + 1)

    def query(index: int) -> int:
        total = 0
        while index:
            total += tree[index]
            index -= index & -index
        return total

    def add(index: int) -> None:
        while index < len(tree):
            tree[index] += 1
            index += index & -index

    inversions = 0
    for seen, value in enumerate(values):
        rank = ranks[value]
        inversions += seen - query(rank)
        add(rank)
    return inversions


def maximum_strict_unequal_inversions(values: Sequence[int | float]) -> int:
    counts = Counter(values)
    return len(values) * (len(values) - 1) // 2 - sum(
        count * (count - 1) // 2 for count in counts.values()
    )


def minimum_duplicate_aware_footrule(
    values: Sequence[int | float], *, descending: bool = False
) -> int:
    """Minimum L1 position distance over all assignments among equal values."""

    positions: dict[int | float, list[int]] = defaultdict(list)
    for position, value in enumerate(values):
        positions[value].append(position)
    target_values = sorted(values, reverse=descending)
    target_positions: dict[int | float, list[int]] = defaultdict(list)
    for position, value in enumerate(target_values):
        target_positions[value].append(position)
    return sum(
        abs(current - target)
        for value in positions
        for current, target in zip(positions[value], target_positions[value])
    )


def brute_force_duplicate_aware_footrule(
    values: Sequence[int | float], *, descending: bool = False
) -> int:
    """Small-fixture oracle for the duplicate assignment rule."""

    if len(values) > 9:
        raise ValueError("brute-force oracle is limited to n<=9")
    target = sorted(values, reverse=descending)
    current_by_value: dict[int | float, tuple[int, ...]] = {}
    target_by_value: dict[int | float, tuple[int, ...]] = {}
    for value in set(values):
        current_by_value[value] = tuple(i for i, item in enumerate(values) if item == value)
        target_by_value[value] = tuple(i for i, item in enumerate(target) if item == value)
    candidates = []
    for value in sorted(current_by_value):
        current = current_by_value[value]
        targets = target_by_value[value]
        candidates.append(
            min(
                sum(abs(left - right) for left, right in zip(current, permutation))
                for permutation in itertools.permutations(targets)
            )
        )
    return sum(candidates)


def distance_descriptors(values: Sequence[int | float]) -> dict[str, int | float]:
    maximum = maximum_strict_unequal_inversions(values)
    inversions = strict_unequal_inversions(values)
    return {
        "strictUnequalInversionsAscending": inversions,
        "maximumStrictUnequalInversions": maximum,
        "normalizedStrictUnequalInversionsAscending": inversions / maximum,
        "strictAdjacentViolationsAscending": sum(
            left > right for left, right in zip(values, values[1:])
        ),
        "strictAdjacentViolationsDescending": sum(
            left < right for left, right in zip(values, values[1:])
        ),
        "minimumDuplicateAwareFootruleAscending": minimum_duplicate_aware_footrule(values),
        "minimumDuplicateAwareFootruleDescending": minimum_duplicate_aware_footrule(
            values, descending=True
        ),
    }


def _ordered_occupancy(
    order_structure: OrderStructure,
    values_by_identity: Sequence[int],
    address: str,
) -> tuple[int, ...]:
    n = len(values_by_identity)
    ascending = tuple(range(n))
    if order_structure == OrderStructure.REVERSE:
        return tuple(reversed(ascending))
    if order_structure == OrderStructure.NEARLY_SORTED:
        width = math.ceil(0.05 * n)
        keyed = []
        for index in ascending:
            jitter, _ = bounded(
                MASTER_SEED,
                address,
                "nearly_sorted_jitter_s07_v1",
                index,
                2 * width + 1,
            )
            tie = u64(
                MASTER_SEED,
                address,
                "nearly_sorted_tie_s07_v1",
                index,
            )
            keyed.append((index + jitter - width, tie, index))
        return tuple(item[2] for item in sorted(keyed))

    block_size = n // 10
    blocks = tuple(
        tuple(range(block * block_size, (block + 1) * block_size))
        for block in range(10)
    )
    block_order = _permutation(
        tuple(range(10)), address, "block_permutation_s07_v1"
    )
    return tuple(index for block in block_order for index in blocks[block])


@dataclass(frozen=True, slots=True)
class InputScenario:
    split: ScenarioSplit
    protected: bool
    n: int
    value_profile: ValueProfile
    order_structure: OrderStructure
    replicate_ordinal: int
    generation_attempt: int
    generation_address: str
    level_values: tuple[int, ...]
    level_counts: tuple[int, ...]
    values_by_identity: tuple[int, ...]
    initial_occupancy_indices: tuple[int, ...]
    initial_values: tuple[int, ...]
    descriptors: Mapping[str, int | float]
    scenario_seed: int
    input_scenario_id: str

    @classmethod
    def generate(
        cls,
        split: ScenarioSplit,
        n: int,
        value_profile: ValueProfile,
        order_structure: OrderStructure,
        replicate_ordinal: int,
        *,
        generation_attempt: int = 0,
    ) -> "InputScenario":
        if n not in SIZES:
            raise ValueError(f"n must be one of {SIZES}")
        if not 0 <= replicate_ordinal < SPLIT_REPLICATES[split]:
            raise ValueError("replicate ordinal is outside its semantic split")
        if not 0 <= generation_attempt < MAX_GENERATION_ATTEMPTS:
            raise ValueError("generation attempt outside frozen bound")
        address = _address(
            split,
            n,
            value_profile,
            order_structure,
            replicate_ordinal,
            generation_attempt,
        )
        counts = value_counts(n, value_profile)
        levels = _level_values(address, len(counts))
        values_by_identity = tuple(
            value for value, count in zip(levels, counts) for _ in range(count)
        )
        occupancy = _ordered_occupancy(order_structure, values_by_identity, address)
        initial_values = tuple(values_by_identity[index] for index in occupancy)
        descriptors = distance_descriptors(initial_values)
        normalized = float(descriptors["normalizedStrictUnequalInversionsAscending"])
        if order_structure == OrderStructure.NEARLY_SORTED and not 0 < normalized <= 0.15:
            raise ValueError("generation_reject_nearly_sorted_distance")
        if order_structure == OrderStructure.REVERSE and normalized != 1.0:
            raise AssertionError("reverse input must attain maximum strict distance")
        if order_structure == OrderStructure.BLOCK_SCRAMBLED and not 0.15 <= normalized <= 0.85:
            raise ValueError("generation_reject_block_distance")
        scenario_seed = u64(
            MASTER_SEED, address, "future_runtime_seed_s07_v1", 0
        )
        protected = split == ScenarioSplit.CONFIRMATORY_HOLDOUT
        content = {
            "schemaVersion": SCENARIO_EXTENSION_VERSION,
            "split": split.value,
            "protected": protected,
            "n": n,
            "valueProfile": value_profile.value,
            "orderStructure": order_structure.value,
            "replicateOrdinal": replicate_ordinal,
            "generationAttempt": generation_attempt,
            "generationAddress": address,
            "levelValues": list(levels),
            "levelCounts": list(counts),
            "initialOccupancyIndices": list(occupancy),
            "scenarioSeed": str(scenario_seed),
            "eventBudgetOpportunities": 100 * n * n,
        }
        return cls(
            split,
            protected,
            n,
            value_profile,
            order_structure,
            replicate_ordinal,
            generation_attempt,
            address,
            levels,
            counts,
            values_by_identity,
            occupancy,
            initial_values,
            descriptors,
            scenario_seed,
            "i1:" + sha256_json(content),
        )

    @property
    def identity_ids(self) -> tuple[str, ...]:
        return tuple(f"cell-{index:04d}" for index in range(self.n))

    @property
    def initial_occupancy(self) -> tuple[str, ...]:
        return tuple(self.identity_ids[index] for index in self.initial_occupancy_indices)

    @property
    def initial_values_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(list(self.initial_values))).hexdigest()

    @property
    def initial_occupancy_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(list(self.initial_occupancy))).hexdigest()

    @property
    def ascending_target_sha256(self) -> str:
        return hashlib.sha256(
            canonical_json_bytes(sorted(self.initial_values))
        ).hexdigest()

    @property
    def descending_target_sha256(self) -> str:
        return hashlib.sha256(
            canonical_json_bytes(sorted(self.initial_values, reverse=True))
        ).hexdigest()

    def target_feasibility(self) -> dict[str, bool]:
        ascending = tuple(sorted(self.initial_values))
        descending = tuple(reversed(ascending))
        return {
            "ascendingMultisetPreserved": Counter(ascending) == Counter(self.initial_values),
            "ascendingNonstrictOrdered": all(
                left <= right for left, right in zip(ascending, ascending[1:])
            ),
            "descendingMultisetPreserved": Counter(descending) == Counter(self.initial_values),
            "descendingNonstrictOrdered": all(
                left >= right for left, right in zip(descending, descending[1:])
            ),
        }

    def core_record(self) -> dict[str, Any]:
        return {
            "schemaVersion": "e02.s07.input_scenario.v1",
            "scenarioExtensionVersion": SCENARIO_EXTENSION_VERSION,
            "inputScenarioId": self.input_scenario_id,
            "split": self.split.value,
            "protected": self.protected,
            "allowedUse": {
                ScenarioSplit.SCREENING_POOL: "screening_and_diagnostics_only",
                ScenarioSplit.CONFIRMATORY_HOLDOUT: "locked_until_preregistered",
                ScenarioSplit.RUNTIME_VALIDATION: "runtime_validation_only",
            }[self.split],
            "n": self.n,
            "valueProfile": self.value_profile.value,
            "orderStructure": self.order_structure.value,
            "replicateOrdinal": self.replicate_ordinal,
            "generationAttempt": self.generation_attempt,
            "generationAddress": self.generation_address,
            "levelValues": list(self.level_values),
            "levelCounts": list(self.level_counts),
            "initialOccupancyIndices": list(self.initial_occupancy_indices),
            "initialValues": list(self.initial_values),
            "initialValuesSha256": self.initial_values_sha256,
            "initialOccupancySha256": self.initial_occupancy_sha256,
            "ascendingTargetValuesSha256": self.ascending_target_sha256,
            "descendingTargetValuesSha256": self.descending_target_sha256,
            "distinctValueCount": len(self.level_values),
            "duplicateIdentityFraction": 1.0 - len(self.level_values) / self.n,
            **self.descriptors,
            **self.target_feasibility(),
            "scenarioSeed": str(self.scenario_seed),
            "eventBudgetProfile": "profile_scaled_frozen_s07_v1",
            "eventBudgetOpportunities": 100 * self.n * self.n,
            "policyAssignmentStatus": "unassigned_S08",
            "architectureAssignmentStatus": "unassigned_S08",
            "schedulerAssignmentStatus": "unassigned_S08",
            "faultMapAssignmentStatus": "unassigned_S08_outcome_blind_only",
            "confirmatoryEligiblePlacementClasses": list(STRUCTURAL_PLACEMENT_CLASSES),
            "forbiddenConfirmatoryPlacementClass": FORBIDDEN_CONFIRMATORY_PLACEMENT_CLASS,
            "outcomeAccess": "none",
            "prespecificationSha256": PRESPECIFICATION_SHA256,
        }


def generate_unique_input_scenario(
    split: ScenarioSplit,
    n: int,
    value_profile: ValueProfile,
    order_structure: OrderStructure,
    replicate_ordinal: int,
    *,
    seen_initial_value_hashes: set[str],
) -> InputScenario:
    """Generate one deterministic valid row and reject within-cell collisions."""

    last_error: Exception | None = None
    for attempt in range(MAX_GENERATION_ATTEMPTS):
        try:
            scenario = InputScenario.generate(
                split,
                n,
                value_profile,
                order_structure,
                replicate_ordinal,
                generation_attempt=attempt,
            )
        except ValueError as error:
            last_error = error
            continue
        if scenario.initial_values_sha256 in seen_initial_value_hashes:
            continue
        seen_initial_value_hashes.add(scenario.initial_values_sha256)
        return scenario
    raise RuntimeError("failed frozen generation bound") from last_error


def trace_selection_count(split: ScenarioSplit, n: int) -> int:
    if split == ScenarioSplit.RUNTIME_VALIDATION:
        return 1
    return max(1, math.ceil(TRACE_RATES[n] * SPLIT_REPLICATES[split]))


def trace_selection_hash(scenario: InputScenario) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "domain": "E02/S07/trace-selection/v1",
                "inputScenarioId": scenario.input_scenario_id,
            }
        )
    ).hexdigest()


def assign_trace_selection(
    scenarios: Sequence[InputScenario],
) -> dict[str, tuple[bool, str]]:
    """Return exact, cell-balanced trace flags without looking at outcomes."""

    groups: dict[tuple[Any, ...], list[InputScenario]] = defaultdict(list)
    for scenario in scenarios:
        groups[
            (
                scenario.split,
                scenario.n,
                scenario.value_profile,
                scenario.order_structure,
            )
        ].append(scenario)
    result: dict[str, tuple[bool, str]] = {}
    for (split, n, _, _), rows in groups.items():
        selected = {
            item.input_scenario_id
            for item in sorted(rows, key=trace_selection_hash)[
                : trace_selection_count(split, n)
            ]
        }
        for item in rows:
            flag = item.input_scenario_id in selected
            reason = (
                "prespecified_hash_sample"
                if flag
                else "summary_only_unless_failure_or_anomaly"
            )
            result[item.input_scenario_id] = (flag, reason)
    return result


def scenario_from_record(record: Mapping[str, Any]) -> InputScenario:
    """Reconstruct and verify a stored extension row from its generation address."""

    scenario = InputScenario.generate(
        ScenarioSplit(record["split"]),
        int(record["n"]),
        ValueProfile(record["valueProfile"]),
        OrderStructure(record["orderStructure"]),
        int(record["replicateOrdinal"]),
        generation_attempt=int(record["generationAttempt"]),
    )
    expected = scenario.core_record()
    for key, value in expected.items():
        if key in record and record[key] != value:
            raise ValueError(f"stored scenario mismatch at {key}")
    return scenario
