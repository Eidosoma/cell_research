"""Outcome-independent logical identity plane for the frozen S08 design.

The four identities in this module are deliberately distinct:

* a reservation slot is an immutable S08P budget address;
* a runtime configuration is the immutable definition actually dispatched;
* a physical execution is complete-equivalence work that may be shared; and
* a logical result is one reserved attribution of that physical work.

Every commitment is generated before an outcome exists.  The physical-to-
logical expansion commitment authenticates the complete one-to-many membership
of a physical group, while each logical identity remains unique by slot.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import json
import math
from typing import Any, Callable, Mapping, Sequence

from src.environment_suite.contracts import canonical_sha256
from src.portfolio_preregistration.core import canonical_hash


IDENTITY_PLANE_VERSION = "e07.s08l.logical-identity-plane.v1"
RESERVATION_SCHEMA_VERSION = "e07.s08l.reservation-slot.v1"
RUNTIME_CONFIGURATION_SCHEMA_VERSION = "e07.s08l.runtime-configuration.v1"
PHYSICAL_EXECUTION_SCHEMA_VERSION = "e07.s08l.physical-execution.v1"
LOGICAL_RESULT_SCHEMA_VERSION = "e07.s08l.logical-result.v1"
EXPANSION_SCHEMA_VERSION = "e07.s08l.physical-to-logical-expansion.v1"

PhysicalIdentityResolver = Callable[[Mapping[str, Any]], str]


class IdentityPlaneError(ValueError):
    """Identity metadata is missing, ambiguous, inconsistent, or forged."""


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return str(value)


def _reservation_slot(work: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "stage",
        "generation",
        "taskId",
        "split",
        "scenarioFamilyOrdinal",
        "logicalSlotOrdinal",
        "logicalSlotId",
        "reservedConfigurationSlotId",
        "configurationRole",
    )
    missing = [field for field in required if field not in work]
    if missing:
        raise IdentityPlaneError(f"reservation metadata missing: {missing}")
    return {
        "schemaVersion": RESERVATION_SCHEMA_VERSION,
        "stage": str(work["stage"]),
        "generation": int(work["generation"]),
        "taskId": str(work["taskId"]),
        "split": str(work["split"]),
        "scenarioFamilyOrdinal": int(work["scenarioFamilyOrdinal"]),
        "logicalSlotOrdinal": int(work["logicalSlotOrdinal"]),
        "logicalSlotId": str(work["logicalSlotId"]),
        "reservedConfigurationSlotId": str(work["reservedConfigurationSlotId"]),
        "configurationRole": str(work["configurationRole"]),
        "pairedSlotId": _optional_string(work.get("pairedSlotId")),
        "smoke": bool(work.get("smoke", False)),
    }


def _runtime_configuration(work: Mapping[str, Any]) -> dict[str, Any]:
    if "configuration" not in work:
        raise IdentityPlaneError("runtime configuration is missing")
    configuration = work["configuration"]
    if str(configuration["taskId"]) != str(work["taskId"]):
        raise IdentityPlaneError("runtime configuration task differs from reservation")
    return {
        "schemaVersion": RUNTIME_CONFIGURATION_SCHEMA_VERSION,
        "taskId": str(configuration["taskId"]),
        "configurationId": str(configuration["configurationId"]),
        "configurationDefinitionSha256": canonical_hash(
            "E07/S08C/configuration-definition/v1", configuration
        ),
        "mode": str(configuration["mode"]),
        "memberSetId": str(configuration["memberSetId"]),
        "memberPolicySha256": [
            str(member["policySha256"]) for member in configuration["members"]
        ],
    }


def _physical_execution(
    work: Mapping[str, Any], resolver: PhysicalIdentityResolver
) -> dict[str, Any]:
    dedup_sha = str(resolver(work))
    if len(dedup_sha) != 64:
        raise IdentityPlaneError("physical equivalence resolver returned a non-SHA256")
    return {
        "schemaVersion": PHYSICAL_EXECUTION_SCHEMA_VERSION,
        "physicalDedupEquivalenceVersion": "e07.s08f.physical-work.v1",
        "physicalDedupEquivalenceSha256": dedup_sha,
    }


def _local_identity(
    work: Mapping[str, Any], resolver: PhysicalIdentityResolver
) -> dict[str, Any]:
    reservation = _reservation_slot(work)
    runtime = _runtime_configuration(work)
    physical = _physical_execution(work, resolver)
    reservation_sha = canonical_sha256("E07/S08L/reservation-slot/v1", reservation)
    runtime_sha = canonical_sha256("E07/S08L/runtime-configuration/v1", runtime)
    physical_sha = canonical_sha256("E07/S08L/physical-execution/v1", physical)
    logical = {
        "schemaVersion": LOGICAL_RESULT_SCHEMA_VERSION,
        "logicalSlotId": reservation["logicalSlotId"],
        "reservationSlotIdentitySha256": reservation_sha,
        "runtimeConfigurationIdentitySha256": runtime_sha,
        "physicalExecutionIdentitySha256": physical_sha,
    }
    logical_sha = canonical_sha256("E07/S08L/logical-result-identity/v1", logical)
    return {
        "reservationSlot": reservation,
        "reservationSlotIdentitySha256": reservation_sha,
        "runtimeConfiguration": runtime,
        "runtimeConfigurationIdentitySha256": runtime_sha,
        "physicalExecution": physical,
        "physicalExecutionIdentitySha256": physical_sha,
        "logicalResult": logical,
        "logicalResultIdentitySha256": logical_sha,
    }


def bind_identity_plane(
    work: Sequence[Mapping[str, Any]],
    *,
    physical_identity_resolver: PhysicalIdentityResolver,
) -> list[dict[str, Any]]:
    """Return an exact pre-outcome roster with authenticated identities."""

    rows = [deepcopy(dict(row)) for row in work]
    if any("identityPlane" in row for row in rows):
        raise IdentityPlaneError("identity plane must be bound exactly once")
    logical_slots = [str(row.get("logicalSlotId")) for row in rows]
    if len(set(logical_slots)) != len(logical_slots):
        raise IdentityPlaneError("logicalSlotId values must be unique")

    local = [_local_identity(row, physical_identity_resolver) for row in rows]
    logical_identities = [item["logicalResultIdentitySha256"] for item in local]
    if len(set(logical_identities)) != len(logical_identities):
        raise IdentityPlaneError("logical result identities must be unique")

    group_indices: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(local):
        group_indices[item["physicalExecutionIdentitySha256"]].append(index)

    expansions: dict[str, tuple[dict[str, Any], str]] = {}
    for physical_sha, indices in group_indices.items():
        members = sorted(
            (
                {
                    "logicalSlotId": local[index]["reservationSlot"]["logicalSlotId"],
                    "logicalResultIdentitySha256": local[index][
                        "logicalResultIdentitySha256"
                    ],
                    "reservationSlotIdentitySha256": local[index][
                        "reservationSlotIdentitySha256"
                    ],
                    "runtimeConfigurationIdentitySha256": local[index][
                        "runtimeConfigurationIdentitySha256"
                    ],
                }
                for index in indices
            ),
            key=lambda item: (
                item["logicalResultIdentitySha256"],
                item["logicalSlotId"],
            ),
        )
        expansion = {
            "schemaVersion": EXPANSION_SCHEMA_VERSION,
            "physicalExecutionIdentitySha256": physical_sha,
            "logicalResultCount": len(members),
            "members": members,
        }
        expansion_sha = canonical_sha256(
            "E07/S08L/physical-to-logical-expansion/v1", expansion
        )
        expansions[physical_sha] = (expansion, expansion_sha)

    for row, item in zip(rows, local, strict=True):
        expansion, expansion_sha = expansions[item["physicalExecutionIdentitySha256"]]
        binding = {
            "schemaVersion": IDENTITY_PLANE_VERSION,
            **item,
            "physicalToLogicalExpansion": expansion,
            "physicalToLogicalExpansionCommitmentSha256": expansion_sha,
        }
        binding["logicalIdentityBindingSha256"] = canonical_sha256(
            "E07/S08L/logical-identity-binding/v1",
            {
                "reservationSlotIdentitySha256": binding[
                    "reservationSlotIdentitySha256"
                ],
                "runtimeConfigurationIdentitySha256": binding[
                    "runtimeConfigurationIdentitySha256"
                ],
                "physicalExecutionIdentitySha256": binding[
                    "physicalExecutionIdentitySha256"
                ],
                "logicalResultIdentitySha256": binding["logicalResultIdentitySha256"],
                "physicalToLogicalExpansionCommitmentSha256": expansion_sha,
            },
        )
        row["identityPlane"] = binding
    return rows


def _validate_embedded_binding(
    row: Mapping[str, Any],
    *,
    physical_identity_resolver: PhysicalIdentityResolver,
) -> list[str]:
    errors: list[str] = []
    plane = row.get("identityPlane")
    if not isinstance(plane, Mapping):
        return ["missing identityPlane"]
    expected_local = _local_identity(row, physical_identity_resolver)
    for field, expected in expected_local.items():
        if plane.get(field) != expected:
            errors.append(f"{field} commitment mismatch")
    expansion = plane.get("physicalToLogicalExpansion")
    if not isinstance(expansion, Mapping):
        errors.append("missing physicalToLogicalExpansion")
        return errors
    expected_expansion_sha = canonical_sha256(
        "E07/S08L/physical-to-logical-expansion/v1", expansion
    )
    if plane.get("physicalToLogicalExpansionCommitmentSha256") != (
        expected_expansion_sha
    ):
        errors.append("physical expansion commitment mismatch")
    if expansion.get("physicalExecutionIdentitySha256") != plane.get(
        "physicalExecutionIdentitySha256"
    ):
        errors.append("physical expansion attached to wrong execution")
    members = expansion.get("members")
    if not isinstance(members, list):
        errors.append("physical expansion members missing")
        members = []
    if expansion.get("logicalResultCount") != len(members):
        errors.append("physical expansion cardinality mismatch")
    member = {
        "logicalSlotId": expected_local["reservationSlot"]["logicalSlotId"],
        "logicalResultIdentitySha256": expected_local["logicalResultIdentitySha256"],
        "reservationSlotIdentitySha256": expected_local[
            "reservationSlotIdentitySha256"
        ],
        "runtimeConfigurationIdentitySha256": expected_local[
            "runtimeConfigurationIdentitySha256"
        ],
    }
    if members.count(member) != 1:
        errors.append("logical member missing, copied, or ambiguous")
    if len({item.get("logicalSlotId") for item in members}) != len(members):
        errors.append("physical expansion repeats a logical slot")
    if len({item.get("logicalResultIdentitySha256") for item in members}) != len(
        members
    ):
        errors.append("physical expansion repeats a logical identity")
    expected_binding_sha = canonical_sha256(
        "E07/S08L/logical-identity-binding/v1",
        {
            "reservationSlotIdentitySha256": expected_local[
                "reservationSlotIdentitySha256"
            ],
            "runtimeConfigurationIdentitySha256": expected_local[
                "runtimeConfigurationIdentitySha256"
            ],
            "physicalExecutionIdentitySha256": expected_local[
                "physicalExecutionIdentitySha256"
            ],
            "logicalResultIdentitySha256": expected_local[
                "logicalResultIdentitySha256"
            ],
            "physicalToLogicalExpansionCommitmentSha256": expected_expansion_sha,
        },
    )
    if plane.get("logicalIdentityBindingSha256") != expected_binding_sha:
        errors.append("logical identity binding commitment mismatch")
    allowed = {
        "schemaVersion",
        *expected_local.keys(),
        "physicalToLogicalExpansion",
        "physicalToLogicalExpansionCommitmentSha256",
        "logicalIdentityBindingSha256",
    }
    if set(plane) != allowed:
        errors.append("identityPlane schema mismatch")
    if plane.get("schemaVersion") != IDENTITY_PLANE_VERSION:
        errors.append("identityPlane version mismatch")
    return errors


def validate_bound_identity_roster(
    work: Sequence[Mapping[str, Any]],
    *,
    physical_identity_resolver: PhysicalIdentityResolver,
) -> dict[str, Any]:
    """Validate exact local and roster-wide commitments, failing closed."""

    rows = [dict(row) for row in work]
    errors: list[dict[str, Any]] = []
    slots = [str(row.get("logicalSlotId")) for row in rows]
    if len(set(slots)) != len(slots):
        errors.append({"position": None, "error": "duplicate logicalSlotId"})
    for position, row in enumerate(rows):
        for error in _validate_embedded_binding(
            row, physical_identity_resolver=physical_identity_resolver
        ):
            errors.append({"position": position, "error": error})

    expected_by_slot: dict[str, Mapping[str, Any]] = {}
    try:
        stripped = [
            {key: value for key, value in row.items() if key != "identityPlane"}
            for row in rows
        ]
        expected = bind_identity_plane(
            stripped, physical_identity_resolver=physical_identity_resolver
        )
        expected_by_slot = {
            str(row["logicalSlotId"]): row["identityPlane"] for row in expected
        }
    except (IdentityPlaneError, KeyError, TypeError, ValueError) as exc:
        errors.append(
            {"position": None, "error": f"roster reconstruction failed: {exc}"}
        )
    for position, row in enumerate(rows):
        slot = str(row.get("logicalSlotId"))
        if expected_by_slot.get(slot) != row.get("identityPlane"):
            errors.append(
                {
                    "position": position,
                    "error": "embedded identity differs from exact roster commitment",
                }
            )

    planes = [
        row.get("identityPlane")
        for row in rows
        if isinstance(row.get("identityPlane"), Mapping)
    ]
    logical_ids = [str(plane.get("logicalResultIdentitySha256")) for plane in planes]
    reservation_ids = [
        str(plane.get("reservationSlotIdentitySha256")) for plane in planes
    ]
    physical_ids = [
        str(plane.get("physicalExecutionIdentitySha256")) for plane in planes
    ]
    return {
        "schemaVersion": "e07.s08l.identity-roster-validation.v1",
        "success": not errors and len(planes) == len(rows),
        "logicalRows": len(rows),
        "uniqueLogicalSlots": len(set(slots)),
        "uniqueReservationSlotIdentities": len(set(reservation_ids)),
        "uniqueRuntimeConfigurationIdentities": len(
            {str(plane.get("runtimeConfigurationIdentitySha256")) for plane in planes}
        ),
        "uniquePhysicalExecutionIdentities": len(set(physical_ids)),
        "uniqueLogicalResultIdentities": len(set(logical_ids)),
        "physicalDedupSavedLogicalRows": len(rows) - len(set(physical_ids)),
        "errors": errors,
    }


def validate_bound_identity_record(
    row: Mapping[str, Any],
    *,
    physical_identity_resolver: PhysicalIdentityResolver,
) -> dict[str, Any]:
    """Validate one binding and its embedded group commitment locally."""

    errors = _validate_embedded_binding(
        row, physical_identity_resolver=physical_identity_resolver
    )
    return {
        "schemaVersion": "e07.s08l.identity-record-validation.v1",
        "success": not errors,
        "logicalSlotId": str(row.get("logicalSlotId")),
        "errors": errors,
    }


def ensure_bound_identity_roster(
    work: Sequence[Mapping[str, Any]],
    *,
    physical_identity_resolver: PhysicalIdentityResolver,
) -> list[dict[str, Any]]:
    """Bind an unbound roster or validate an already bound roster exactly."""

    rows = [deepcopy(dict(row)) for row in work]
    present = ["identityPlane" in row for row in rows]
    if any(present) and not all(present):
        raise IdentityPlaneError("partially bound identity roster")
    if not any(present):
        return bind_identity_plane(
            rows, physical_identity_resolver=physical_identity_resolver
        )
    audit = validate_bound_identity_roster(
        rows, physical_identity_resolver=physical_identity_resolver
    )
    if not audit["success"]:
        raise IdentityPlaneError(
            f"identity roster failed closed with {len(audit['errors'])} error(s)"
        )
    return rows


def validate_persisted_logical_identity_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate result-side identity projection without reading outcomes."""

    errors: list[dict[str, Any]] = []
    group_members: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    logical_ids: list[str] = []
    stable_hashes: list[str] = []
    slots: list[str] = []
    for position, row in enumerate(rows):
        plane = row.get("identityPlane")
        if not isinstance(plane, Mapping):
            errors.append({"position": position, "error": "missing identityPlane"})
            continue
        reservation = plane.get("reservationSlot", {})
        runtime = plane.get("runtimeConfiguration", {})
        physical = plane.get("physicalExecution", {})
        expansion = plane.get("physicalToLogicalExpansion", {})
        checks = {
            "logicalSlotId": (
                str(row.get("logicalSlotId")) == str(reservation.get("logicalSlotId"))
            ),
            "reservedConfigurationSlotId": (
                str(row.get("reservedConfigurationSlotId"))
                == str(reservation.get("reservedConfigurationSlotId"))
            ),
            "configurationRole": (
                str(row.get("configurationRole"))
                == str(reservation.get("configurationRole"))
            ),
            "configurationId": (
                str(row.get("configurationId")) == str(runtime.get("configurationId"))
            ),
            "configurationDefinitionSha256": (
                str(row.get("configurationDefinitionSha256"))
                == str(runtime.get("configurationDefinitionSha256"))
            ),
            "physicalWorkSha256": (
                str(row.get("physicalWorkSha256"))
                == str(physical.get("physicalDedupEquivalenceSha256"))
            ),
            "reservationSlotIdentitySha256": (
                row.get("reservationSlotIdentitySha256")
                == plane.get("reservationSlotIdentitySha256")
            ),
            "runtimeConfigurationIdentitySha256": (
                row.get("runtimeConfigurationIdentitySha256")
                == plane.get("runtimeConfigurationIdentitySha256")
            ),
            "physicalExecutionIdentitySha256": (
                row.get("physicalExecutionIdentitySha256")
                == plane.get("physicalExecutionIdentitySha256")
            ),
            "logicalResultIdentitySha256": (
                row.get("logicalResultIdentitySha256")
                == plane.get("logicalResultIdentitySha256")
            ),
            "physicalToLogicalExpansionCommitmentSha256": (
                row.get("physicalToLogicalExpansionCommitmentSha256")
                == plane.get("physicalToLogicalExpansionCommitmentSha256")
            ),
            "logicalIdentityBindingSha256": (
                row.get("logicalIdentityBindingSha256")
                == plane.get("logicalIdentityBindingSha256")
            ),
        }
        for check, passed in checks.items():
            if not passed:
                errors.append({"position": position, "error": f"{check} mismatch"})
        expansion_sha = canonical_sha256(
            "E07/S08L/physical-to-logical-expansion/v1", expansion
        )
        if expansion_sha != plane.get("physicalToLogicalExpansionCommitmentSha256"):
            errors.append(
                {"position": position, "error": "forged expansion commitment"}
            )
        expected_result_sha = canonical_sha256(
            "E07/S08L/logical-result-record/v1",
            {
                "logicalResultIdentitySha256": plane.get("logicalResultIdentitySha256"),
                "logicalIdentityBindingSha256": plane.get(
                    "logicalIdentityBindingSha256"
                ),
                "physicalStableEvaluationSha256": row.get(
                    "physicalStableEvaluationSha256"
                ),
            },
        )
        if row.get("stableEvaluationSha256") != expected_result_sha:
            errors.append({"position": position, "error": "logical hash mismatch"})
        group_members[
            (
                str(plane.get("physicalExecutionIdentitySha256")),
                str(plane.get("physicalToLogicalExpansionCommitmentSha256")),
            )
        ].append(
            {
                "logicalSlotId": str(row.get("logicalSlotId")),
                "logicalResultIdentitySha256": plane.get("logicalResultIdentitySha256"),
                "reservationSlotIdentitySha256": plane.get(
                    "reservationSlotIdentitySha256"
                ),
                "runtimeConfigurationIdentitySha256": plane.get(
                    "runtimeConfigurationIdentitySha256"
                ),
            }
        )
        logical_ids.append(str(plane.get("logicalResultIdentitySha256")))
        stable_hashes.append(str(row.get("stableEvaluationSha256")))
        slots.append(str(row.get("logicalSlotId")))

    for position, row in enumerate(rows):
        plane = row.get("identityPlane")
        if not isinstance(plane, Mapping):
            continue
        key = (
            str(plane.get("physicalExecutionIdentitySha256")),
            str(plane.get("physicalToLogicalExpansionCommitmentSha256")),
        )
        expected_members = sorted(
            group_members[key],
            key=lambda item: (
                str(item["logicalResultIdentitySha256"]),
                item["logicalSlotId"],
            ),
        )
        declared = plane["physicalToLogicalExpansion"].get("members")
        if declared != expected_members:
            errors.append(
                {
                    "position": position,
                    "error": "ambiguous or incomplete physical expansion membership",
                }
            )

    return {
        "schemaVersion": "e07.s08l.persisted-identity-validation.v1",
        "success": not errors and len(slots) == len(rows),
        "logicalRows": len(rows),
        "uniqueLogicalSlots": len(set(slots)),
        "uniqueLogicalResultIdentities": len(set(logical_ids)),
        "uniqueStableLogicalHashes": len(set(stable_hashes)),
        "physicalExpansionGroups": len(group_members),
        "errors": errors,
    }


def identity_commitment_projection_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Project every pre-outcome identity plane into a compact durable ledger.

    The canonical JSON retains the complete expansion membership rather than
    only its digest.  This lets a fresh reader authenticate one-to-many
    membership without loading an outcome or reconstructing runtime work.
    """

    projected: list[dict[str, Any]] = []
    for position, row in enumerate(rows):
        plane = row.get("identityPlane")
        if not isinstance(plane, Mapping):
            raise IdentityPlaneError(f"identityPlane missing at position {position}")
        reservation = plane.get("reservationSlot")
        runtime = plane.get("runtimeConfiguration")
        physical = plane.get("physicalExecution")
        logical = plane.get("logicalResult")
        expansion = plane.get("physicalToLogicalExpansion")
        if not all(
            isinstance(item, Mapping)
            for item in (reservation, runtime, physical, logical, expansion)
        ):
            raise IdentityPlaneError(
                f"identityPlane component missing at position {position}"
            )
        projected.append(
            {
                "identityPlaneVersion": str(plane.get("schemaVersion")),
                "stage": str(reservation["stage"]),
                "generation": int(reservation["generation"]),
                "taskId": str(reservation["taskId"]),
                "split": str(reservation["split"]),
                "scenarioFamilyOrdinal": int(reservation["scenarioFamilyOrdinal"]),
                "logicalSlotOrdinal": int(reservation["logicalSlotOrdinal"]),
                "logicalSlotId": str(reservation["logicalSlotId"]),
                "reservedConfigurationSlotId": str(
                    reservation["reservedConfigurationSlotId"]
                ),
                "configurationRole": str(reservation["configurationRole"]),
                "runtimeConfigurationId": str(runtime["configurationId"]),
                "reservationSlotIdentitySha256": str(
                    plane["reservationSlotIdentitySha256"]
                ),
                "runtimeConfigurationIdentitySha256": str(
                    plane["runtimeConfigurationIdentitySha256"]
                ),
                "physicalExecutionIdentitySha256": str(
                    plane["physicalExecutionIdentitySha256"]
                ),
                "logicalResultIdentitySha256": str(
                    plane["logicalResultIdentitySha256"]
                ),
                "physicalToLogicalExpansionCommitmentSha256": str(
                    plane["physicalToLogicalExpansionCommitmentSha256"]
                ),
                "logicalIdentityBindingSha256": str(
                    plane["logicalIdentityBindingSha256"]
                ),
                "physicalExpansionLogicalResultCount": int(
                    expansion["logicalResultCount"]
                ),
                "identityPlaneJson": json.dumps(
                    dict(plane),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                ),
            }
        )
    return projected


def validate_identity_commitment_projection_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Authenticate a durable outcome-independent identity projection."""

    errors: list[dict[str, Any]] = []
    parsed: list[dict[str, Any]] = []
    for position, row in enumerate(rows):
        try:
            plane = json.loads(str(row["identityPlaneJson"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(
                {"position": position, "error": f"identity JSON invalid: {exc}"}
            )
            continue
        if not isinstance(plane, dict):
            errors.append(
                {"position": position, "error": "identity JSON is not object"}
            )
            continue
        reservation = plane.get("reservationSlot", {})
        runtime = plane.get("runtimeConfiguration", {})
        physical = plane.get("physicalExecution", {})
        logical = plane.get("logicalResult", {})
        expansion = plane.get("physicalToLogicalExpansion", {})
        component_checks = {
            "version": plane.get("schemaVersion") == IDENTITY_PLANE_VERSION,
            "reservation": (
                plane.get("reservationSlotIdentitySha256")
                == canonical_sha256("E07/S08L/reservation-slot/v1", reservation)
            ),
            "runtime": (
                plane.get("runtimeConfigurationIdentitySha256")
                == canonical_sha256("E07/S08L/runtime-configuration/v1", runtime)
            ),
            "physical": (
                plane.get("physicalExecutionIdentitySha256")
                == canonical_sha256("E07/S08L/physical-execution/v1", physical)
            ),
            "logical": (
                plane.get("logicalResultIdentitySha256")
                == canonical_sha256("E07/S08L/logical-result-identity/v1", logical)
            ),
            "expansion": (
                plane.get("physicalToLogicalExpansionCommitmentSha256")
                == canonical_sha256(
                    "E07/S08L/physical-to-logical-expansion/v1", expansion
                )
            ),
        }
        expected_binding = canonical_sha256(
            "E07/S08L/logical-identity-binding/v1",
            {
                "reservationSlotIdentitySha256": plane.get(
                    "reservationSlotIdentitySha256"
                ),
                "runtimeConfigurationIdentitySha256": plane.get(
                    "runtimeConfigurationIdentitySha256"
                ),
                "physicalExecutionIdentitySha256": plane.get(
                    "physicalExecutionIdentitySha256"
                ),
                "logicalResultIdentitySha256": plane.get("logicalResultIdentitySha256"),
                "physicalToLogicalExpansionCommitmentSha256": plane.get(
                    "physicalToLogicalExpansionCommitmentSha256"
                ),
            },
        )
        component_checks["binding"] = (
            plane.get("logicalIdentityBindingSha256") == expected_binding
        )
        denormalized = {
            "identityPlaneVersion": plane.get("schemaVersion"),
            "stage": reservation.get("stage"),
            "generation": reservation.get("generation"),
            "taskId": reservation.get("taskId"),
            "split": reservation.get("split"),
            "scenarioFamilyOrdinal": reservation.get("scenarioFamilyOrdinal"),
            "logicalSlotOrdinal": reservation.get("logicalSlotOrdinal"),
            "logicalSlotId": reservation.get("logicalSlotId"),
            "reservedConfigurationSlotId": reservation.get(
                "reservedConfigurationSlotId"
            ),
            "configurationRole": reservation.get("configurationRole"),
            "runtimeConfigurationId": runtime.get("configurationId"),
            "reservationSlotIdentitySha256": plane.get("reservationSlotIdentitySha256"),
            "runtimeConfigurationIdentitySha256": plane.get(
                "runtimeConfigurationIdentitySha256"
            ),
            "physicalExecutionIdentitySha256": plane.get(
                "physicalExecutionIdentitySha256"
            ),
            "logicalResultIdentitySha256": plane.get("logicalResultIdentitySha256"),
            "physicalToLogicalExpansionCommitmentSha256": plane.get(
                "physicalToLogicalExpansionCommitmentSha256"
            ),
            "logicalIdentityBindingSha256": plane.get("logicalIdentityBindingSha256"),
            "physicalExpansionLogicalResultCount": expansion.get("logicalResultCount"),
        }
        component_checks["denormalized"] = all(
            str(row.get(field)) == str(value) for field, value in denormalized.items()
        )
        for check, passed in component_checks.items():
            if not passed:
                errors.append(
                    {"position": position, "error": f"{check} commitment mismatch"}
                )
        parsed.append(plane)

    by_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for plane in parsed:
        reservation = plane["reservationSlot"]
        by_group[
            (
                str(plane["physicalExecutionIdentitySha256"]),
                str(plane["physicalToLogicalExpansionCommitmentSha256"]),
            )
        ].append(
            {
                "logicalSlotId": str(reservation["logicalSlotId"]),
                "logicalResultIdentitySha256": str(
                    plane["logicalResultIdentitySha256"]
                ),
                "reservationSlotIdentitySha256": str(
                    plane["reservationSlotIdentitySha256"]
                ),
                "runtimeConfigurationIdentitySha256": str(
                    plane["runtimeConfigurationIdentitySha256"]
                ),
            }
        )
    for position, plane in enumerate(parsed):
        key = (
            str(plane["physicalExecutionIdentitySha256"]),
            str(plane["physicalToLogicalExpansionCommitmentSha256"]),
        )
        expected_members = sorted(
            by_group[key],
            key=lambda item: (
                item["logicalResultIdentitySha256"],
                item["logicalSlotId"],
            ),
        )
        if plane["physicalToLogicalExpansion"].get("members") != expected_members:
            errors.append(
                {
                    "position": position,
                    "error": "physical expansion membership is incomplete or ambiguous",
                }
            )

    slots = [str(plane["reservationSlot"]["logicalSlotId"]) for plane in parsed]
    reservation_ids = [str(plane["reservationSlotIdentitySha256"]) for plane in parsed]
    logical_ids = [str(plane["logicalResultIdentitySha256"]) for plane in parsed]
    if len(set(slots)) != len(slots):
        errors.append({"position": None, "error": "logical slots are not unique"})
    if len(set(reservation_ids)) != len(reservation_ids):
        errors.append(
            {"position": None, "error": "reservation identities are not unique"}
        )
    if len(set(logical_ids)) != len(logical_ids):
        errors.append(
            {"position": None, "error": "logical result identities are not unique"}
        )
    return {
        "schemaVersion": "e07.s08m.identity-commitment-projection-validation.v1",
        "success": not errors and len(parsed) == len(rows),
        "logicalRows": len(rows),
        "parsedIdentityRows": len(parsed),
        "uniqueLogicalSlots": len(set(slots)),
        "uniqueReservationSlotIdentities": len(set(reservation_ids)),
        "uniqueLogicalResultIdentities": len(set(logical_ids)),
        "physicalExpansionGroups": len(by_group),
        "physicalExpansionMemberCount": sum(len(group) for group in by_group.values()),
        "errors": errors,
    }
