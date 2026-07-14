"""S08 outcome-blind pairing and semantic random-stream contracts.

This module creates design identities and treatment-arm metadata only.  It does
not execute scientific outcomes, inspect protected results, or admit S06
search-derived placements.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from reference_simulator.model import canonical_json_bytes, sha256_json
from reference_simulator.rng import u64


PAIRING_INTERFACE_VERSION = "E02-paired-comparisons-v1"
PAIRING_SCHEMA_VERSION = "e02.s08.pairing_manifest.v1"
PAIRING_PRESPECIFICATION_SHA256 = (
    "be8afbeb1f44a01c724cb4b9873f1cf8967f2ce1a967138a5d15fcf4c5d98345"
)
STREAM_SPECIFICATION_SHA256 = (
    "21af6d99aae7468fc0a849b7fcdff3d118c035853d1721a43e5644dd85f67f34"
)
MASTER_SEED = int("e0208000000000000000000000000001", 16)
POLICIES = ("Bubble", "Insertion", "Selection")
DIRECTIONS = ("ascending", "descending")
FAULT_COUNTS = (2, 4, 6)
STRUCTURAL_CLASSES = (
    "uniform_exact",
    "clustered_exact",
    "boundary_exact",
    "median_rank_exact",
)
FORBIDDEN_PLACEMENTS = ("search_derived_exploratory", "adversarial_held_out")


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_id(prefix: str, value: Mapping[str, Any]) -> str:
    return f"{prefix}:{sha256_json(value)}"


@dataclass(frozen=True, slots=True)
class FaultProfile:
    placement_class: str
    fault_count: int

    def __post_init__(self) -> None:
        if self.placement_class not in STRUCTURAL_CLASSES:
            raise ValueError("S08 accepts only frozen outcome-blind S06 classes")
        if self.fault_count not in FAULT_COUNTS:
            raise ValueError("S08 fault count is outside the frozen 2/4/6 set")

    @property
    def profile_id(self) -> str:
        return f"{self.placement_class}:f{self.fault_count}"


FAULT_PROFILES = tuple(
    FaultProfile(placement_class, count)
    for placement_class in STRUCTURAL_CLASSES
    for count in FAULT_COUNTS
)


def _stratum(record: Mapping[str, Any]) -> tuple[str, int, str, str]:
    return (
        str(record["split"]),
        int(record["n"]),
        str(record["valueProfile"]),
        str(record["orderStructure"]),
    )


def assign_fault_profiles(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, FaultProfile]:
    """Allocate all 12 structural class/count profiles without outcome access."""

    grouped: dict[tuple[str, int, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[_stratum(record)].append(record)
    assigned: dict[str, FaultProfile] = {}
    for stratum, items in sorted(grouped.items()):
        address = "S08/fault-profile-stratum/" + sha256_json(
            {
                "split": stratum[0],
                "n": stratum[1],
                "valueProfile": stratum[2],
                "orderStructure": stratum[3],
            }
        )
        rotation = u64(
            MASTER_SEED, address, "fault_profile_assignment_s08_v1", 1
        ) % len(FAULT_PROFILES)
        ranked = sorted(
            items,
            key=lambda record: (
                u64(
                    MASTER_SEED,
                    str(record["inputScenarioId"]),
                    "fault_profile_assignment_s08_v1",
                    0,
                ),
                str(record["inputScenarioId"]),
            ),
        )
        for rank, record in enumerate(ranked):
            scenario_id = str(record["inputScenarioId"])
            if scenario_id in assigned:
                raise ValueError("input scenario appears in multiple allocation strata")
            assigned[scenario_id] = FAULT_PROFILES[(rank + rotation) % len(FAULT_PROFILES)]
    if len(assigned) != len(records):
        raise AssertionError("fault profile allocation lost or duplicated input rows")
    return assigned


def algotype_assignment_content(n: int, policy: str) -> dict[str, Any]:
    if n < 1 or policy not in POLICIES:
        raise ValueError("invalid homogeneous Algotype assignment")
    return {
        "schemaVersion": "e02.s08.algotype_assignment.v1",
        "n": n,
        "assignmentProfile": "homogeneous_exact",
        "policy": policy,
        "identityConvention": "cell-%04d",
        "counts": {policy: n},
    }


def algotype_assignment_id(n: int, policy: str) -> str:
    return content_id("aa1", algotype_assignment_content(n, policy))


def pairing_block_content(
    record: Mapping[str, Any],
    *,
    policy: str,
    direction: str,
    algotype_id: str,
    fault_map_id: str,
) -> dict[str, Any]:
    if policy not in POLICIES or direction not in DIRECTIONS:
        raise ValueError("pairing block has an undeclared policy or direction")
    return {
        "schemaVersion": PAIRING_SCHEMA_VERSION,
        "pairingInterfaceVersion": PAIRING_INTERFACE_VERSION,
        "inputScenarioId": str(record["inputScenarioId"]),
        "split": str(record["split"]),
        "n": int(record["n"]),
        "valueProfile": str(record["valueProfile"]),
        "orderStructure": str(record["orderStructure"]),
        "replicateOrdinal": int(record["replicateOrdinal"]),
        "algotypeAssignmentId": algotype_id,
        "policyProfile": policy,
        "direction": direction,
        "initialValuesSha256": str(record["initialValuesSha256"]),
        "initialOccupancySha256": str(record["initialOccupancySha256"]),
        "faultMapId": fault_map_id,
        "runtimeSeed": str(record["scenarioSeed"]),
        "eventBudgetProfile": str(record["eventBudgetProfile"]),
        "eventBudgetOpportunities": int(record["eventBudgetOpportunities"]),
        "pairingPrespecificationSha256": PAIRING_PRESPECIFICATION_SHA256,
        "streamSpecificationSha256": STREAM_SPECIFICATION_SHA256,
    }


def pairing_block_id(content: Mapping[str, Any]) -> str:
    return content_id("pb1", content)


def scenario_variant_id(
    pairing_id: str, mobility: str, fault_map_id: str
) -> str:
    if mobility not in {"normal", "passive", "stuck"}:
        raise ValueError("unknown S05 mobility profile")
    return content_id(
        "sv1",
        {
            "schemaVersion": "e02.s08.scenario_variant.v1",
            "pairingBlockId": pairing_id,
            "mobility": mobility,
            "faultMapId": None if mobility == "normal" else fault_map_id,
        },
    )


def _base_settings(*, mobility: str) -> dict[str, str]:
    return {
        "architecture": "distributed_local",
        "coordinatorProfile": "none",
        "scheduler": "uniform_random_activation",
        "mobility": mobility,
        "continuation": "skip_and_continue",
        "retry": "no_retry",
        "actionFailure": "none",
        "sensing": "exact",
        "informationPermission": "policy_native_local",
        "legalPrimitives": "NoOp|Swap|MemoryUpdate",
        "proposalCandidatesPerOpportunity": "1",
    }


def _arm(
    estimand: str,
    label: str,
    role: str,
    settings: Mapping[str, str],
    *,
    executable: bool = True,
) -> dict[str, Any]:
    return {
        "armId": content_id(
            "arm1",
            {
                "estimandId": estimand,
                "armLabel": label,
                "settings": dict(settings),
                "executable": executable,
            },
        ),
        "armLabel": label,
        "contrastRole": role,
        "executable": executable,
        "settings": dict(settings),
    }


def _two_arm(
    estimand: str,
    *,
    title: str,
    treatment: str,
    reference_value: str,
    active_value: str,
    base: Mapping[str, str],
    pairing_status: str,
    caveat: str | None = None,
) -> dict[str, Any]:
    reference = dict(base)
    active = dict(base)
    reference[treatment] = reference_value
    active[treatment] = active_value
    if treatment == "architecture":
        reference["coordinatorProfile"] = (
            "common_validator_only"
            if reference_value == "central_local_proposal_k1"
            else "none"
        )
        active["coordinatorProfile"] = (
            "common_validator_only"
            if active_value == "central_local_proposal_k1"
            else "none"
        )
    return {
        "estimandId": estimand,
        "title": title,
        "status": "paired_executable",
        "rngPairingStatus": pairing_status,
        "caveat": caveat,
        "arms": [
            _arm(estimand, f"{treatment}={reference_value}", "reference", reference),
            _arm(estimand, f"{treatment}={active_value}", "active", active),
        ],
    }


def _factorial(
    estimand: str,
    *,
    title: str,
    factor_a: str,
    reference_a: str,
    active_a: str,
    factor_b: str,
    reference_b: str,
    active_b: str,
    base: Mapping[str, str],
    pairing_status: str,
) -> dict[str, Any]:
    arms = []
    for value_a, role_a in ((reference_a, "reference"), (active_a, "active")):
        for value_b, role_b in ((reference_b, "reference"), (active_b, "active")):
            settings = dict(base)
            settings[factor_a] = value_a
            settings[factor_b] = value_b
            if factor_a == "architecture":
                settings["coordinatorProfile"] = (
                    "common_validator_only"
                    if value_a == "central_local_proposal_k1"
                    else "none"
                )
            arms.append(
                _arm(
                    estimand,
                    f"{factor_a}={value_a}|{factor_b}={value_b}",
                    f"{role_a}_{role_b}",
                    settings,
                )
            )
    return {
        "estimandId": estimand,
        "title": title,
        "status": "paired_executable",
        "rngPairingStatus": pairing_status,
        "caveat": None,
        "arms": arms,
    }


def contrast_catalog() -> dict[str, Any]:
    contrasts: list[dict[str, Any]] = []

    base = _base_settings(mobility="passive")
    base["architecture"] = "central_local_proposal_k1"
    base["coordinatorProfile"] = "common_validator_only"
    contrasts.append(
        _two_arm(
            "E02-S01-E01",
            title="Matched control-topology effect under faults",
            treatment="architecture",
            reference_value="central_local_proposal_k1",
            active_value="distributed_local",
            base=base,
            pairing_status="exact_shared_same_scheduler_and_scenario",
        )
    )

    base = _base_settings(mobility="normal")
    base["architecture"] = "central_local_proposal_k1"
    base["coordinatorProfile"] = "common_validator_only"
    contrasts.append(
        _two_arm(
            "E02-S01-E02",
            title="No-fault architecture parity",
            treatment="architecture",
            reference_value="central_local_proposal_k1",
            active_value="distributed_local",
            base=base,
            pairing_status="exact_shared_hard_parity",
        )
    )

    contrasts.append(
        _two_arm(
            "E02-S01-E03",
            title="Scheduler effect within distributed control",
            treatment="scheduler",
            reference_value="uniform_random_activation",
            active_value="random_permutation_sweep",
            base=_base_settings(mobility="passive"),
            pairing_status="scenario_paired_rng_unpaired_different_stream_consumption",
        )
    )

    contrasts.append(
        _two_arm(
            "E02-S01-E04",
            title="Continuation effect after blocking failure",
            treatment="continuation",
            reference_value="stop_on_first_blocking_failure",
            active_value="skip_and_continue",
            base=_base_settings(mobility="stuck"),
            pairing_status="shared_prefix_until_continuation_stop",
        )
    )

    contrasts.append(
        _two_arm(
            "E02-S01-E05",
            title="Bounded retry effect",
            treatment="retry",
            reference_value="no_retry",
            active_value="retry_later_bounded",
            base=_base_settings(mobility="stuck"),
            pairing_status="exact_shared_but_retry_exposure_inactive",
            caveat=(
                "Frozen actionFailure=none means S05 never queues a retry; this "
                "contrast is paired but mechanistically degenerate."
            ),
        )
    )

    contrasts.append(
        _two_arm(
            "E02-S01-E06",
            title="Mobility-fault semantics effect",
            treatment="mobility",
            reference_value="passive",
            active_value="stuck",
            base=_base_settings(mobility="passive"),
            pairing_status="scenario_paired_rng_unpaired_different_scenario_roots",
        )
    )

    e07_reference = _base_settings(mobility="passive")
    e07_reference.update(
        {
            "architecture": "central_local_proposal_k1",
            "coordinatorProfile": "common_validator_only",
            "decisionRule": "masked_global_rank_rule_v1",
            "informationPermission": "policy_native_local",
        }
    )
    e07_active = dict(e07_reference)
    e07_active["informationPermission"] = "full_global_state"
    contrasts.append(
        {
            "estimandId": "E02-S01-E07",
            "title": "Information-permission effect",
            "status": "unpaired_non_executable_preserved_information_boundary",
            "rngPairingStatus": "not_executable",
            "caveat": (
                "S02 rejects full_global_state from the matched interface; S03's "
                "legacy-global controller changes information, scheduling, and primitives."
            ),
            "arms": [
                _arm(
                    "E02-S01-E07",
                    "informationPermission=policy_native_local",
                    "reference",
                    e07_reference,
                    executable=False,
                ),
                _arm(
                    "E02-S01-E07",
                    "informationPermission=full_global_state",
                    "active",
                    e07_active,
                    executable=False,
                ),
            ],
        }
    )

    base = _base_settings(mobility="passive")
    base["architecture"] = "distributed_weak_coordinator"
    contrasts.append(
        _two_arm(
            "E02-S01-E08",
            title="Weak-coordinator effect",
            treatment="coordinatorProfile",
            reference_value="none",
            active_value="weak_frozen_budget",
            base=base,
            pairing_status="shared_prefix_same_scheduler_root",
        )
    )

    contrasts.append(
        _factorial(
            "E02-S01-E09",
            title="Architecture-by-scheduler interaction",
            factor_a="architecture",
            reference_a="central_local_proposal_k1",
            active_a="distributed_local",
            factor_b="scheduler",
            reference_b="uniform_random_activation",
            active_b="random_permutation_sweep",
            base=_base_settings(mobility="passive"),
            pairing_status="hybrid_exact_within_scheduler_rng_unpaired_across_schedulers",
        )
    )
    contrasts.append(
        _factorial(
            "E02-S01-E10",
            title="Architecture-by-mobility-fault interaction",
            factor_a="architecture",
            reference_a="central_local_proposal_k1",
            active_a="distributed_local",
            factor_b="mobility",
            reference_b="passive",
            active_b="stuck",
            base=_base_settings(mobility="passive"),
            pairing_status="hybrid_exact_within_mobility_rng_unpaired_across_mobility",
        )
    )
    contrasts.append(
        _factorial(
            "E02-S01-E11",
            title="Architecture-by-continuation interaction",
            factor_a="architecture",
            reference_a="central_local_proposal_k1",
            active_a="distributed_local",
            factor_b="continuation",
            reference_b="stop_on_first_blocking_failure",
            active_b="skip_and_continue",
            base=_base_settings(mobility="stuck"),
            pairing_status="hybrid_exact_within_continuation_shared_prefix_across_continuation",
        )
    )
    contrasts.append(
        _factorial(
            "E02-S01-E12",
            title="Scheduler-by-fault interaction",
            factor_a="scheduler",
            reference_a="uniform_random_activation",
            active_a="random_permutation_sweep",
            factor_b="mobility",
            reference_b="passive",
            active_b="stuck",
            base=_base_settings(mobility="passive"),
            pairing_status="scenario_paired_rng_unpaired_across_scheduler_or_mobility",
        )
    )

    executable_arms = [
        arm
        for contrast in contrasts
        for arm in contrast["arms"]
        if arm["executable"]
    ]
    return {
        "schemaVersion": "e02.s08.contrast_arm_catalog.v1",
        "researchStepId": "S08",
        "pairingInterfaceVersion": PAIRING_INTERFACE_VERSION,
        "primaryEstimandCount": len(contrasts),
        "executableEstimandCount": sum(
            item["status"] == "paired_executable" for item in contrasts
        ),
        "declaredUnpairedNonExecutableCount": sum(
            item["status"].startswith("unpaired_non_executable")
            for item in contrasts
        ),
        "executableArmCount": len(executable_arms),
        "allCatalogArmCount": sum(len(item["arms"]) for item in contrasts),
        "contrasts": contrasts,
    }


def executable_arms(catalog: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {"estimandId": contrast["estimandId"], **arm}
        for contrast in catalog["contrasts"]
        for arm in contrast["arms"]
        if arm["executable"]
    ]


def coupling_matrix_rows(catalog: Mapping[str, Any]) -> list[dict[str, str]]:
    """Long-form coupling claims used by validation and later analysis."""

    rows: list[dict[str, str]] = []
    for contrast in catalog["contrasts"]:
        estimand = contrast["estimandId"]
        overall = contrast["rngPairingStatus"]
        if overall == "not_executable":
            runtime = "not_executable"
        elif "rng_unpaired" in overall:
            runtime = "scenario_paired_rng_unpaired"
        elif "shared_prefix" in overall:
            runtime = "shared_prefix"
        else:
            runtime = "exact_shared"
        for component in (
            "initial_permutation",
            "fault_map",
            "algotype_assignment",
            "direction_assignment",
        ):
            rows.append(
                {
                    "estimandId": estimand,
                    "componentOrStream": component,
                    "coupling": "not_executable" if runtime == "not_executable" else "exact_shared",
                    "reason": (
                        contrast["caveat"]
                        if runtime == "not_executable"
                        else "immutable pairing-block component"
                    ),
                }
            )
        for stream in (
            "actor_selection",
            "bubble_side",
            "conflict_priority",
            "action_failure_exogenous",
            "sensing_error",
        ):
            rows.append(
                {
                    "estimandId": estimand,
                    "componentOrStream": stream,
                    "coupling": runtime,
                    "reason": overall,
                }
            )
    return rows


def canonical_catalog_bytes() -> bytes:
    return canonical_json_bytes(contrast_catalog()) + b"\n"
