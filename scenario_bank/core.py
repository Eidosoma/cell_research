"""Frozen S08 scenario construction, pairing, and seed semantics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import random
from typing import Any, Iterable, Mapping

from reference_simulator.model import (
    Architecture,
    Cell,
    Direction,
    FaultMode,
    Policy,
    Scenario,
    canonical_json_bytes,
)
from reference_simulator.rng import bounded, permutation


BANK_SCHEMA_VERSION = "e01.s08.paired_scenario_bank.v1"
BASE_DRAW_SCHEMA_VERSION = "e01.s08.base_draw_bank.v1"
CONDITION_SCHEMA_VERSION = "e01.s08.condition.v1"
SEED_DERIVATION_VERSION = "E01/S08/SHA256_COUNTER/v1"
MASTER_SEED = 0xE0108000000000000000000000000001
FROZEN_PUBLIC_COMMIT = "1fd2bd5921c1f6b423a71f691d5189106a8a1020"


@dataclass(frozen=True, slots=True)
class SplitSpec:
    name: str
    count: int
    protected: bool
    allowed_use: str
    historical_sample_size_basis: str


SPLITS = (
    SplitSpec(
        "paper_scale", 100, False,
        "figure-level replication; all 100 rows are analyzed together",
        "Matches reported N=100 for Figures 3-8; is a declared clean-room N for Figures 9-10 where N is unreported.",
    ),
    SplitSpec(
        "exploratory", 250, False,
        "method development, diagnostics, and contrast screening only",
        "FULL_PLAN exploratory starting scale; not historical evidence.",
    ),
    SplitSpec(
        "confirmatory_holdout", 1000, True,
        "locked until estimands, exclusions, and analysis are preregistered",
        "FULL_PLAN promoted paired-confirmation reserve; not a commitment to execute every row.",
    ),
    SplitSpec(
        "policy_search_holdout", 250, True,
        "E07 final testing only; prohibited for search, tuning, or surrogate training",
        "Cross-experiment protected test archive; not historical evidence.",
    ),
)


@dataclass(frozen=True, slots=True)
class ConditionSpec:
    condition_id: str
    family: str
    input_profile: str
    architecture: str
    policies: tuple[str, ...]
    assignment_profile: str
    direction_profile: str
    direction_map: tuple[tuple[str, str], ...]
    fault_mode: str
    requested_fault_count: int
    placement_profile: str
    analysis_label_profile: str
    profile_role: str
    max_activations: int
    historical_eligibility: str
    paper_condition_note: str

    def to_dict(self) -> dict[str, Any]:
        body = asdict(self)
        body["policies"] = list(self.policies)
        body["direction_map"] = {key: value for key, value in self.direction_map}
        body["schemaVersion"] = CONDITION_SCHEMA_VERSION
        body["conditionId"] = body.pop("condition_id")
        for old, new in (
            ("input_profile", "inputProfile"),
            ("assignment_profile", "assignmentProfile"),
            ("direction_profile", "directionProfile"),
            ("direction_map", "directionMap"),
            ("fault_mode", "faultMode"),
            ("requested_fault_count", "requestedFaultCount"),
            ("placement_profile", "placementProfile"),
            ("analysis_label_profile", "analysisLabelProfile"),
            ("profile_role", "profileRole"),
            ("max_activations", "maxActivations"),
            ("historical_eligibility", "historicalEligibility"),
            ("paper_condition_note", "paperConditionNote"),
        ):
            body[new] = body.pop(old)
        return body


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def derive_seed(stream: str, *address: Any) -> int:
    """Derive one unsigned 128-bit seed from an immutable semantic address."""
    if not stream or not stream.isascii():
        raise ValueError("stream must be nonempty ASCII")
    material = {
        "version": SEED_DERIVATION_VERSION,
        "stream": stream,
        "address": list(address),
    }
    digest = hashlib.sha256(
        b"E01/S08/seed/v1\x00"
        + MASTER_SEED.to_bytes(16, "big")
        + b"\x00"
        + canonical_json_bytes(material)
    ).digest()
    return int.from_bytes(digest[:16], "big")


def _pure_condition(architecture: str, policy: str) -> ConditionSpec:
    code = {"Bubble": "BUB", "Insertion": "INS", "Selection": "SEL"}[policy]
    arch = "CV" if architecture == "cell_view" else "TRAD"
    historical = (
        "C_adapter_endpoint_supported_no_fault"
        if architecture == "cell_view"
        else "reference_only_traditional_generator_absent"
    )
    return ConditionSpec(
        f"C-UNQ-PURE-{arch}-{code}-ASC", "unique_pure", "unique_1_100",
        architecture, (policy,), "pure", "consensus_ascending",
        ((policy, "ascending"),), "none", 0, "not_applicable",
        "none", "reference_primary", 1_000_000, historical,
        "Paper n=100/N=100 unique-value pure-policy condition.",
    )


def _fault_condition(
    architecture: str, policy: str, mode: str, count: int, placement: str
) -> ConditionSpec:
    pc = {"Bubble": "BUB", "Insertion": "INS", "Selection": "SEL"}[policy]
    ac = "CV" if architecture == "cell_view" else "TRAD"
    mc = "PAS" if mode == "passive" else "STK"
    plc = "CORR" if placement == "reference_without_replacement" else "LEGACY"
    if architecture == "cell_view" and mode == "passive" and placement == "legacy_with_replacement":
        historical = "C_adapter_endpoint_supported_with_bank_seed_override"
    elif mode == "stuck":
        historical = "reference_only_stuck_implementation_absent"
    elif architecture == "traditional":
        historical = "reference_only_traditional_generator_absent"
    else:
        historical = "reference_only_corrected_placement"
    return ConditionSpec(
        f"C-UNQ-FAULT-{ac}-{pc}-{mc}-F{count}-{plc}", "unique_fault",
        "unique_1_100", architecture, (policy,), "pure",
        "consensus_ascending", ((policy, "ascending"),), mode, count,
        placement, "none",
        "reference_primary" if placement == "reference_without_replacement" else "historical_rule_sensitivity",
        1_000_000, historical,
        "Paper placement is unreported; corrected and exact with-replacement rule profiles are separate.",
    )


def _chimera_condition(
    input_profile: str,
    policies: tuple[str, ...],
    assignment: str,
    direction_map: Mapping[str, str],
) -> ConditionSpec:
    prefix = "UNQ" if input_profile == "unique_1_100" else "REP"
    codes = {"Bubble": "BUB", "Insertion": "INS", "Selection": "SEL"}
    policy_code = "-".join(codes[item] for item in policies)
    assign_code = "EXACT" if assignment == "balanced_exact" else "RANDOM"
    opposed = len(set(direction_map.values())) > 1
    direction_code = "OPP" if opposed else "ASC"
    family = (
        "opposite_unique" if opposed and prefix == "UNQ"
        else "opposite_repeated" if opposed
        else "same_direction_unique" if prefix == "UNQ"
        else "same_direction_repeated"
    )
    role = "reference_primary" if assignment == "balanced_exact" else "assignment_sensitivity"
    return ConditionSpec(
        f"C-{prefix}-CHIM-{policy_code}-{assign_code}-{direction_code}", family,
        input_profile, "cell_view", policies, assignment,
        "opposed_by_algotype" if opposed else "consensus_ascending",
        tuple((policy, direction_map[policy]) for policy in policies),
        "none", 0, "not_applicable", "none", role, 1_000_000,
        "frozen_source_mixed_driver_not_validated_by_S04_adapter",
        "Exact composition is primary; independent random assignment is a separately named sensitivity profile.",
    )


def _control_condition(labels: tuple[str, str]) -> ConditionSpec:
    codes = {"Bubble": "BUB", "Insertion": "INS", "Selection": "SEL"}
    return ConditionSpec(
        f"C-UNQ-CONTROL-GHOST-{codes[labels[0]]}-{codes[labels[1]]}",
        "identical_policy_label_control", "unique_1_100", "cell_view",
        ("Bubble",), "pure", "consensus_ascending", (("Bubble", "ascending"),),
        "none", 0, "not_applicable", "ghost_exact_50_50:" + "+".join(labels),
        "negative_control", 1_000_000, "reference_only_control_recipe_absent",
        "Distinct immutable analysis labels; every executable policy is Bubble.",
    )


def _worked_condition(policy: str) -> ConditionSpec:
    code = {"Bubble": "BUB", "Insertion": "INS", "Selection": "SEL"}[policy]
    return ConditionSpec(
        f"C-WORK-N6-{code}-STUCK-V4", "worked_example_ambiguity_envelope",
        "worked_1_6_exhaustive", "cell_view", (policy,), "pure",
        "consensus_ascending", ((policy, "ascending"),), "stuck", 1,
        "explicit_stuck_value_4", "none", "ambiguity_envelope", 10_000,
        "reference_only_paper_algorithm_and_order_unreported",
        "Exhaustive envelope over all 720 input orders; does not claim the exact Figure 6 trace.",
    )


def build_condition_catalog() -> tuple[ConditionSpec, ...]:
    conditions: list[ConditionSpec] = []
    policies = ("Bubble", "Insertion", "Selection")
    for architecture in ("cell_view", "traditional"):
        conditions.extend(_pure_condition(architecture, policy) for policy in policies)
    for architecture in ("cell_view", "traditional"):
        for policy in policies:
            for mode in ("passive", "stuck"):
                for count in (1, 2, 3):
                    for placement in ("reference_without_replacement", "legacy_with_replacement"):
                        conditions.append(_fault_condition(architecture, policy, mode, count, placement))

    pairings = (("Bubble", "Insertion"), ("Bubble", "Selection"), ("Insertion", "Selection"))
    ascending = {policy: "ascending" for policy in policies}
    for pair in pairings:
        for assignment in ("balanced_exact", "independent_random"):
            conditions.append(_chimera_condition("unique_1_100", pair, assignment, ascending))
    for assignment in ("balanced_exact", "independent_random"):
        conditions.append(_chimera_condition("unique_1_100", policies, assignment, ascending))
    conditions.extend(_control_condition(pair) for pair in pairings)
    for pair in pairings:
        for assignment in ("balanced_exact", "independent_random"):
            conditions.append(_chimera_condition("repeated_1_10_x10", pair, assignment, ascending))

    opposing = (
        (("Bubble", "Selection"), {"Bubble": "descending", "Selection": "ascending"}),
        (("Bubble", "Insertion"), {"Bubble": "ascending", "Insertion": "descending"}),
        (("Insertion", "Selection"), {"Insertion": "ascending", "Selection": "descending"}),
    )
    for input_profile in ("unique_1_100", "repeated_1_10_x10"):
        for pair, direction_map in opposing:
            for assignment in ("balanced_exact", "independent_random"):
                conditions.append(_chimera_condition(input_profile, pair, assignment, direction_map))
    conditions.extend(_worked_condition(policy) for policy in policies)
    result = tuple(sorted(conditions, key=lambda item: item.condition_id))
    if len(result) != 110 or len({item.condition_id for item in result}) != len(result):
        raise AssertionError("condition catalog must contain 110 unique profiles")
    return result


def split_by_name(name: str) -> SplitSpec:
    return next(item for item in SPLITS if item.name == name)


def _base_values(input_profile: str) -> tuple[int, ...]:
    if input_profile == "unique_1_100":
        return tuple(range(1, 101))
    if input_profile == "repeated_1_10_x10":
        return tuple(value for value in range(1, 11) for _ in range(10))
    if input_profile == "worked_1_6_exhaustive":
        return tuple(range(1, 7))
    raise ValueError(f"unknown input profile {input_profile}")


def make_base_draw(input_profile: str, split: str, replicate: int) -> dict[str, Any]:
    values = _base_values(input_profile)
    n = len(values)
    if input_profile == "worked_1_6_exhaustive":
        if split != "ambiguity_envelope" or not 0 <= replicate < 720:
            raise ValueError("worked envelope requires the 720 lexicographic permutations")
        occupancy = tuple(itertools.islice(itertools.permutations(range(n)), replicate, replicate + 1))[0]
        seed_value: int | None = None
        protected = False
    else:
        spec = split_by_name(split)
        if not 0 <= replicate < spec.count:
            raise ValueError("replicate outside split")
        seed_value = derive_seed("initial_occupancy", input_profile, split, replicate)
        ids = tuple(str(index) for index in range(n))
        occupancy = tuple(int(index) for index in permutation(
            ids, seed_value, f"E01/S08/base/{input_profile}/{split}/{replicate}"
        ))
        protected = spec.protected
    initial_values = [values[index] for index in occupancy]
    material = {
        "schemaVersion": BASE_DRAW_SCHEMA_VERSION,
        "inputProfile": input_profile,
        "split": split,
        "replicateOrdinal": replicate,
        "valuesById": list(values),
        "initialOccupancyIndices": list(occupancy),
    }
    base_id = "b1:" + _canonical_hash(material)
    return {
        **material,
        "baseDrawId": base_id,
        "pairingBlockId": "p1:" + _canonical_hash({
            "inputProfile": input_profile, "split": split, "replicateOrdinal": replicate
        }),
        "n": n,
        "protected": protected,
        "initialOccupancySeed": str(seed_value) if seed_value is not None else None,
        "valuesByIdSha256": _canonical_hash(list(values)),
        "initialOccupancySha256": _canonical_hash(list(occupancy)),
        "initialValueSequenceSha256": _canonical_hash(initial_values),
    }


def _assignment(
    condition: ConditionSpec, base: Mapping[str, Any]
) -> tuple[list[str], list[str | None], int, dict[str, int]]:
    n = int(base["n"])
    split = str(base["split"])
    replicate = int(base["replicateOrdinal"])
    assignment_key = {
        "inputProfile": condition.input_profile,
        "policies": sorted(condition.policies),
        "profile": condition.assignment_profile,
        "analysisLabelProfile": condition.analysis_label_profile,
        "split": split,
        "replicate": replicate,
    }
    seed = derive_seed("algotype_assignment", assignment_key)
    generation_key = "E01/S08/assignment/" + _canonical_hash(assignment_key)
    ids = tuple(f"cell-{index:04d}" for index in range(n))
    assigned: dict[str, str] = {}
    policies = list(condition.policies)
    if condition.assignment_profile == "pure":
        assigned = {cell_id: policies[0] for cell_id in ids}
    elif condition.assignment_profile == "balanced_exact":
        ordered_ids = permutation(ids, seed, generation_key)
        base_count, remainder = divmod(n, len(policies))
        extra_offset = (replicate + [item.name for item in SPLITS].index(split)) % len(policies)
        counts = [base_count] * len(policies)
        for offset in range(remainder):
            counts[(extra_offset + offset) % len(policies)] += 1
        cursor = 0
        for policy, count in zip(policies, counts):
            for cell_id in ordered_ids[cursor : cursor + count]:
                assigned[cell_id] = policy
            cursor += count
    elif condition.assignment_profile == "independent_random":
        for index, cell_id in enumerate(ids):
            selected, _ = bounded(seed, generation_key, "independent_algotype", index, len(policies))
            assigned[cell_id] = policies[selected]
    else:
        raise ValueError(f"unsupported assignment profile {condition.assignment_profile}")
    policy_by_id = [assigned[cell_id] for cell_id in ids]
    labels: list[str | None] = [None] * n
    if condition.analysis_label_profile.startswith("ghost_exact_50_50:"):
        label_names = condition.analysis_label_profile.split(":", 1)[1].split("+")
        ordered_ids = permutation(ids, seed, generation_key + "/labels")
        label_by_id = {
            cell_id: f"ghost:{label_names[0] if ordinal < n // 2 else label_names[1]}"
            for ordinal, cell_id in enumerate(ordered_ids)
        }
        labels = [label_by_id[cell_id] for cell_id in ids]
    counts = {policy: policy_by_id.count(policy) for policy in sorted(set(policy_by_id))}
    return policy_by_id, labels, seed, counts


def _faults(
    condition: ConditionSpec, base: Mapping[str, Any]
) -> tuple[list[int], list[int], int | None]:
    requested = condition.requested_fault_count
    if requested == 0:
        return [], [], None
    if condition.placement_profile == "explicit_stuck_value_4":
        return [3], [3], None
    address = (
        condition.input_profile, base["split"], int(base["replicateOrdinal"]),
        condition.placement_profile, requested,
    )
    seed = derive_seed("fault_placement", *address)
    n = int(base["n"])
    if condition.placement_profile == "reference_without_replacement":
        ids = tuple(str(index) for index in range(n))
        draws = [int(index) for index in permutation(
            ids, seed, "E01/S08/fault/" + _canonical_hash(address)
        )[:requested]]
    elif condition.placement_profile == "legacy_with_replacement":
        generator = random.Random(seed)
        draws = [generator.randint(0, n - 1) for _ in range(requested)]
    else:
        raise ValueError(f"unsupported fault placement {condition.placement_profile}")
    return draws, sorted(set(draws)), seed


def materialize_scenario(
    condition: ConditionSpec, base: Mapping[str, Any]
) -> tuple[Scenario, dict[str, Any]]:
    n = int(base["n"])
    values = [int(value) for value in base["valuesById"]]
    occupancy_indices = [int(value) for value in base["initialOccupancyIndices"]]
    policies, labels, assignment_seed, composition = _assignment(condition, base)
    direction_map = dict(condition.direction_map)
    directions = [direction_map[policy] for policy in policies]
    fault_draws, fault_indices, fault_seed = _faults(condition, base)
    fault_modes = ["normal"] * n
    for index in fault_indices:
        fault_modes[index] = condition.fault_mode
    cells = tuple(
        Cell(
            f"cell-{index:04d}", values[index], Policy(policies[index]),
            Direction(directions[index]), FaultMode(fault_modes[index]), labels[index],
        )
        for index in range(n)
    )
    generation_key = f"E01/S08/{condition.condition_id}/{base['baseDrawId']}"
    runtime_seed = derive_seed(
        "reference_runtime", condition.input_profile, base["split"], int(base["replicateOrdinal"])
    )
    if condition.placement_profile == "not_applicable":
        scenario_placement = "explicit"
    elif condition.placement_profile == "explicit_stuck_value_4":
        scenario_placement = "explicit"
    else:
        scenario_placement = condition.placement_profile
    scenario = Scenario.create(
        cells,
        initial_occupancy=tuple(f"cell-{index:04d}" for index in occupancy_indices),
        seed=runtime_seed,
        max_activations=condition.max_activations,
        architecture=Architecture(condition.architecture),
        traditional_policy=(Policy(condition.policies[0]) if condition.architecture == "traditional" else None),
        generation_key=generation_key,
        fault_placement=scenario_placement,
        requested_fault_count=condition.requested_fault_count,
    )
    metadata = {
        "runtimeSeed": str(runtime_seed),
        "assignmentSeed": str(assignment_seed),
        "faultSeed": str(fault_seed) if fault_seed is not None else None,
        "faultDrawIndices": fault_draws,
        "faultDistinctIndices": fault_indices,
        "compositionCounts": composition,
        "directionCounts": {
            direction: directions.count(direction) for direction in sorted(set(directions))
        },
        "policyAssignmentSha256": _canonical_hash(policies),
        "directionAssignmentSha256": _canonical_hash(directions),
        "analysisLabelAssignmentSha256": _canonical_hash(labels),
        "faultAssignmentSha256": _canonical_hash(fault_modes),
        "scenarioJsonSha256": hashlib.sha256(scenario.to_json_bytes()).hexdigest(),
    }
    return scenario, metadata


def iter_base_draws() -> Iterable[dict[str, Any]]:
    for input_profile in ("unique_1_100", "repeated_1_10_x10"):
        for split in SPLITS:
            for replicate in range(split.count):
                yield make_base_draw(input_profile, split.name, replicate)
    for replicate in range(720):
        yield make_base_draw("worked_1_6_exhaustive", "ambiguity_envelope", replicate)


def seed_specification() -> dict[str, Any]:
    return {
        "schemaVersion": "e01.s08.seed_specification.v1",
        "masterSeedHex": f"0x{MASTER_SEED:032x}",
        "masterSeedDecimal": str(MASTER_SEED),
        "derivationVersion": SEED_DERIVATION_VERSION,
        "derivation": "uint128_be(first16(SHA256(domain || 0x00 || uint128_be(master) || 0x00 || canonical_json(address))))",
        "domain": "E01/S08/seed/v1",
        "canonicalJson": "UTF-8; sorted keys; compact separators",
        "streams": {
            "initial_occupancy": "input profile, split, replicate",
            "algotype_assignment": "input profile, policy set, assignment profile, analysis-label profile, split, replicate",
            "fault_placement": "input profile, split, replicate, placement profile, requested count; deliberately shared across policy, architecture, and fault mode",
            "reference_runtime": "input profile, split, replicate; same numeric seed across paired conditions, while S05 runtime addresses remain scenario-ID separated",
        },
        "legacyPlacementRule": {
            "profile": "legacy_with_replacement",
            "algorithm": "fresh CPython random.Random(bank-derived seed); requested calls to randint(0,n-1); distinct set becomes faulty identities",
            "sourceBasis": f"frozen public commit {FROZEN_PUBLIC_COMMIT} repeated random.randint rule and S04 adapter",
            "historicalStreamStatus": "unavailable_not_invented",
            "seedInterpretation": "bank-controlled validation override, not recovered publication randomness",
        },
        "correctedPlacementRule": {
            "profile": "reference_without_replacement",
            "algorithm": "counter-addressed Fisher-Yates permutation; take first f distinct identity indices",
            "guarantee": "realized distinct fault count equals requested count",
        },
        "workerOrderInfluence": "none",
    }


CONDITION_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "urn:e01:s08:condition:v1",
    "title": "E01 S08 immutable condition",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schemaVersion", "conditionId", "family", "inputProfile", "architecture",
        "policies", "assignmentProfile", "directionProfile", "directionMap",
        "faultMode", "requestedFaultCount", "placementProfile", "analysisLabelProfile",
        "profileRole", "maxActivations", "historicalEligibility", "paperConditionNote",
    ],
    "properties": {
        "schemaVersion": {"const": CONDITION_SCHEMA_VERSION},
        "conditionId": {"type": "string", "pattern": "^C-"},
        "family": {"type": "string", "minLength": 1},
        "inputProfile": {"enum": ["unique_1_100", "repeated_1_10_x10", "worked_1_6_exhaustive"]},
        "architecture": {"enum": ["cell_view", "traditional"]},
        "policies": {"type": "array", "minItems": 1, "uniqueItems": True, "items": {"enum": ["Bubble", "Insertion", "Selection"]}},
        "assignmentProfile": {"enum": ["pure", "balanced_exact", "independent_random"]},
        "directionProfile": {"enum": ["consensus_ascending", "opposed_by_algotype"]},
        "directionMap": {"type": "object", "additionalProperties": {"enum": ["ascending", "descending"]}, "minProperties": 1},
        "faultMode": {"enum": ["none", "passive", "stuck"]},
        "requestedFaultCount": {"type": "integer", "minimum": 0, "maximum": 3},
        "placementProfile": {"enum": ["not_applicable", "reference_without_replacement", "legacy_with_replacement", "explicit_stuck_value_4"]},
        "analysisLabelProfile": {"type": "string"},
        "profileRole": {"enum": ["reference_primary", "historical_rule_sensitivity", "assignment_sensitivity", "negative_control", "ambiguity_envelope"]},
        "maxActivations": {"type": "integer", "minimum": 1},
        "historicalEligibility": {"type": "string", "minLength": 1},
        "paperConditionNote": {"type": "string", "minLength": 1},
    },
}


BASE_DRAW_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "urn:e01:s08:base-draw:v1",
    "title": "E01 S08 immutable base draw",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schemaVersion", "baseDrawId", "pairingBlockId", "inputProfile", "split",
        "protected", "replicateOrdinal", "n", "initialOccupancySeed", "valuesById",
        "initialOccupancyIndices", "valuesByIdSha256", "initialOccupancySha256",
        "initialValueSequenceSha256",
    ],
    "properties": {
        "schemaVersion": {"const": BASE_DRAW_SCHEMA_VERSION},
        "baseDrawId": {"type": "string", "pattern": "^b1:[0-9a-f]{64}$"},
        "pairingBlockId": {"type": "string", "pattern": "^p1:[0-9a-f]{64}$"},
        "inputProfile": {"enum": ["unique_1_100", "repeated_1_10_x10", "worked_1_6_exhaustive"]},
        "split": {"enum": ["paper_scale", "exploratory", "confirmatory_holdout", "policy_search_holdout", "ambiguity_envelope"]},
        "protected": {"type": "boolean"},
        "replicateOrdinal": {"type": "integer", "minimum": 0},
        "n": {"enum": [6, 100]},
        "initialOccupancySeed": {
            "oneOf": [
                {"type": "null"},
                {"type": "string", "pattern": "^[0-9]+$"},
            ]
        },
        "valuesById": {"type": "array", "minItems": 6, "maxItems": 100, "items": {"type": "integer"}},
        "initialOccupancyIndices": {"type": "array", "minItems": 6, "maxItems": 100, "items": {"type": "integer", "minimum": 0, "maximum": 99}},
        "valuesByIdSha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "initialOccupancySha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "initialValueSequenceSha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
    },
}


SCENARIO_ROW_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "urn:e01:s08:scenario-row:v1",
    "title": "E01 S08 normalized immutable scenario row",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schemaVersion", "scenarioId", "conditionId", "conditionFamily", "profileRole",
        "conditionConfigSha256", "baseDrawId", "pairingBlockId", "inputProfile", "n",
        "split", "protected", "replicateOrdinal", "architecture", "policySet",
        "assignmentProfile", "directionProfile", "faultMode", "requestedFaultCount",
        "realizedFaultCount", "placementProfile", "faultDrawIndices", "faultDistinctIndices",
        "faultMapId", "runtimeSeed", "assignmentSeed", "faultSeed", "generationKey",
        "maxActivations", "scenarioContentSha256", "scenarioJsonSha256",
        "initialValuesSha256", "initialOccupancySha256", "policyAssignmentSha256",
        "directionAssignmentSha256", "analysisLabelAssignmentSha256", "faultAssignmentSha256",
        "compositionCountsJson", "directionCountsJson", "backendEligibility",
        "historicalRandomStreamStatus", "runtimeCouplingProfile", "referenceRngProfile",
        "claimIds", "configSchemaVersion",
    ],
    "properties": {
        "schemaVersion": {"const": BANK_SCHEMA_VERSION},
        "scenarioId": {"type": "string", "pattern": "^r1:[0-9a-f]{64}$"},
        "conditionId": {"type": "string", "pattern": "^C-"},
        "conditionFamily": {"type": "string", "minLength": 1},
        "profileRole": {"enum": ["reference_primary", "historical_rule_sensitivity", "assignment_sensitivity", "negative_control", "ambiguity_envelope"]},
        "conditionConfigSha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "baseDrawId": {"type": "string", "pattern": "^b1:[0-9a-f]{64}$"},
        "pairingBlockId": {"type": "string", "pattern": "^p1:[0-9a-f]{64}$"},
        "inputProfile": {"enum": ["unique_1_100", "repeated_1_10_x10", "worked_1_6_exhaustive"]},
        "n": {"enum": [6, 100]},
        "split": {"enum": ["paper_scale", "exploratory", "confirmatory_holdout", "policy_search_holdout", "ambiguity_envelope"]},
        "protected": {"type": "boolean"},
        "replicateOrdinal": {"type": "integer", "minimum": 0},
        "architecture": {"enum": ["cell_view", "traditional"]},
        "policySet": {"type": "array", "minItems": 1, "uniqueItems": True, "items": {"enum": ["Bubble", "Insertion", "Selection"]}},
        "assignmentProfile": {"enum": ["pure", "balanced_exact", "independent_random"]},
        "directionProfile": {"enum": ["consensus_ascending", "opposed_by_algotype"]},
        "faultMode": {"enum": ["none", "passive", "stuck"]},
        "requestedFaultCount": {"type": "integer", "minimum": 0, "maximum": 3},
        "realizedFaultCount": {"type": "integer", "minimum": 0, "maximum": 3},
        "placementProfile": {"enum": ["not_applicable", "reference_without_replacement", "legacy_with_replacement", "explicit_stuck_value_4"]},
        "faultDrawIndices": {"type": "array", "maxItems": 3, "items": {"type": "integer", "minimum": 0, "maximum": 99}},
        "faultDistinctIndices": {"type": "array", "maxItems": 3, "uniqueItems": True, "items": {"type": "integer", "minimum": 0, "maximum": 99}},
        "faultMapId": {"type": "string", "pattern": "^fm1:[0-9a-f]{64}$"},
        "runtimeSeed": {"type": "string", "pattern": "^[0-9]+$"},
        "assignmentSeed": {"type": "string", "pattern": "^[0-9]+$"},
        "faultSeed": {"oneOf": [{"type": "null"}, {"type": "string", "pattern": "^[0-9]+$"}]},
        "generationKey": {"type": "string", "minLength": 1},
        "maxActivations": {"type": "integer", "minimum": 1},
        "scenarioContentSha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "scenarioJsonSha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "initialValuesSha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "initialOccupancySha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "policyAssignmentSha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "directionAssignmentSha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "analysisLabelAssignmentSha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "faultAssignmentSha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "compositionCountsJson": {"type": "string", "pattern": "^\\{"},
        "directionCountsJson": {"type": "string", "pattern": "^\\{"},
        "backendEligibility": {"type": "string", "minLength": 1},
        "historicalRandomStreamStatus": {"const": "unavailable_not_invented"},
        "runtimeCouplingProfile": {"const": "shared_numeric_seed_by_base; runtime_draws_scenario_id_separated"},
        "referenceRngProfile": {"const": "sha256_counter_E01_v1"},
        "claimIds": {"type": "array", "uniqueItems": True, "items": {"type": "string", "pattern": "^F[0-9]{2}-"}},
        "configSchemaVersion": {"const": CONDITION_SCHEMA_VERSION},
    },
}
