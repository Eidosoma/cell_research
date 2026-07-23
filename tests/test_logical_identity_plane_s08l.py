from __future__ import annotations

from copy import deepcopy
import json

import pytest

from src.environment_suite.contracts import canonical_sha256
from src.portfolio_search.identity import (
    IDENTITY_PLANE_VERSION,
    IdentityPlaneError,
    bind_identity_plane,
    ensure_bound_identity_roster,
    validate_bound_identity_record,
    validate_bound_identity_roster,
    validate_persisted_logical_identity_rows,
)


def _configuration(configuration_id: str, *, alias: str = "base") -> dict:
    return {
        "taskId": "qualification_task",
        "configurationId": configuration_id,
        "mode": "fixed_balanced_identity",
        "memberSetId": canonical_sha256("member-set", alias),
        "members": [
            {"policySha256": canonical_sha256("policy", "a")},
            {"policySha256": canonical_sha256("policy", "b")},
        ],
    }


def _work(
    slot: int,
    configuration: dict,
    *,
    role: str = "target_portfolio",
    reservation: str | None = None,
    family: int = 300,
) -> dict:
    return {
        "stage": "adaptive",
        "generation": 1,
        "taskId": "qualification_task",
        "split": "train",
        "scenarioFamilyOrdinal": family,
        "logicalSlotOrdinal": slot,
        "logicalSlotId": canonical_sha256("logical-slot", slot),
        "reservedConfigurationSlotId": reservation or f"reservation-{slot}",
        "configurationRole": role,
        "pairedSlotId": f"pair-{slot // 2}",
        "smoke": False,
        "configuration": deepcopy(configuration),
        "physicalKey": canonical_sha256("physical", [family, "shared"]),
    }


def _physical(row: dict) -> str:
    return row["physicalKey"]


def _persisted(bound: dict, physical_result: str = "f" * 64) -> dict:
    plane = deepcopy(bound["identityPlane"])
    runtime = plane["runtimeConfiguration"]
    reservation = plane["reservationSlot"]
    stable = canonical_sha256(
        "E07/S08L/logical-result-record/v1",
        {
            "logicalResultIdentitySha256": plane["logicalResultIdentitySha256"],
            "logicalIdentityBindingSha256": plane["logicalIdentityBindingSha256"],
            "physicalStableEvaluationSha256": physical_result,
        },
    )
    return {
        "logicalSlotId": reservation["logicalSlotId"],
        "reservedConfigurationSlotId": reservation["reservedConfigurationSlotId"],
        "configurationRole": reservation["configurationRole"],
        "configurationId": runtime["configurationId"],
        "configurationDefinitionSha256": runtime["configurationDefinitionSha256"],
        "physicalWorkSha256": plane["physicalExecution"][
            "physicalDedupEquivalenceSha256"
        ],
        "physicalStableEvaluationSha256": physical_result,
        "stableEvaluationSha256": stable,
        "identityPlane": plane,
        "reservationSlotIdentitySha256": plane["reservationSlotIdentitySha256"],
        "runtimeConfigurationIdentitySha256": plane[
            "runtimeConfigurationIdentitySha256"
        ],
        "physicalExecutionIdentitySha256": plane["physicalExecutionIdentitySha256"],
        "logicalResultIdentitySha256": plane["logicalResultIdentitySha256"],
        "physicalToLogicalExpansionCommitmentSha256": plane[
            "physicalToLogicalExpansionCommitmentSha256"
        ],
        "logicalIdentityBindingSha256": plane["logicalIdentityBindingSha256"],
    }


def test_repeated_same_configuration_and_scenario_get_distinct_logical_ids() -> None:
    config = _configuration("a" * 64)
    bound = bind_identity_plane(
        [_work(0, config), _work(1, config)], physical_identity_resolver=_physical
    )
    assert {row["identityPlane"]["schemaVersion"] for row in bound} == {
        IDENTITY_PLANE_VERSION
    }
    assert (
        len({row["identityPlane"]["physicalExecutionIdentitySha256"] for row in bound})
        == 1
    )
    assert (
        len({row["identityPlane"]["logicalResultIdentitySha256"] for row in bound}) == 2
    )
    audit = validate_bound_identity_roster(bound, physical_identity_resolver=_physical)
    assert audit["success"]
    assert audit["physicalDedupSavedLogicalRows"] == 1


def test_adaptive_reservation_reference_is_distinct_from_runtime_configuration() -> (
    None
):
    config = _configuration("b" * 64)
    bound = bind_identity_plane(
        [
            _work(
                0,
                config,
                reservation="frozen-target-reservation-token",
                role="target_portfolio",
            ),
            _work(
                1,
                config,
                reservation="frozen-comparator-reservation-token",
                role="matched_random_dynamic",
            ),
        ],
        physical_identity_resolver=_physical,
    )
    assert all(
        row["reservedConfigurationSlotId"] != row["configuration"]["configurationId"]
        for row in bound
    )
    assert (
        len({row["identityPlane"]["reservationSlotIdentitySha256"] for row in bound})
        == 2
    )
    assert validate_bound_identity_roster(bound, physical_identity_resolver=_physical)[
        "success"
    ]


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "copied",
        "reservation",
        "runtime",
        "physical",
        "expansion",
        "binding",
        "field_loss",
    ],
)
def test_missing_copied_and_forged_metadata_fail_closed(mutation: str) -> None:
    config = _configuration("c" * 64)
    bound = bind_identity_plane(
        [_work(0, config), _work(1, config)], physical_identity_resolver=_physical
    )
    attacked = deepcopy(bound)
    if mutation == "missing":
        attacked[0].pop("identityPlane")
    elif mutation == "copied":
        attacked[1]["identityPlane"] = deepcopy(attacked[0]["identityPlane"])
    elif mutation == "reservation":
        attacked[0]["identityPlane"]["reservationSlotIdentitySha256"] = "0" * 64
    elif mutation == "runtime":
        attacked[0]["identityPlane"]["runtimeConfigurationIdentitySha256"] = "0" * 64
    elif mutation == "physical":
        attacked[0]["identityPlane"]["physicalExecutionIdentitySha256"] = "0" * 64
    elif mutation == "expansion":
        attacked[0]["identityPlane"]["physicalToLogicalExpansion"]["members"].pop()
    elif mutation == "binding":
        attacked[0]["identityPlane"]["logicalIdentityBindingSha256"] = "0" * 64
    elif mutation == "field_loss":
        attacked[0]["identityPlane"]["runtimeConfiguration"].pop("mode")
    audit = validate_bound_identity_roster(
        attacked, physical_identity_resolver=_physical
    )
    assert not audit["success"]
    with pytest.raises(IdentityPlaneError):
        ensure_bound_identity_roster(attacked, physical_identity_resolver=_physical)


def test_duplicate_logical_slot_and_ambiguous_membership_fail_closed() -> None:
    config = _configuration("d" * 64)
    duplicate = [_work(0, config), _work(0, config)]
    with pytest.raises(IdentityPlaneError, match="logicalSlotId"):
        bind_identity_plane(duplicate, physical_identity_resolver=_physical)

    bound = bind_identity_plane(
        [_work(0, config), _work(1, config)], physical_identity_resolver=_physical
    )
    member = deepcopy(
        bound[0]["identityPlane"]["physicalToLogicalExpansion"]["members"][0]
    )
    bound[0]["identityPlane"]["physicalToLogicalExpansion"]["members"].append(member)
    local = validate_bound_identity_record(
        bound[0], physical_identity_resolver=_physical
    )
    assert not local["success"]


def test_serialization_replay_and_worker_order_are_exact() -> None:
    config = _configuration("e" * 64)
    rows = [_work(index, config, family=300 + index % 2) for index in range(8)]
    forward = bind_identity_plane(rows, physical_identity_resolver=_physical)
    reverse = bind_identity_plane(
        list(reversed(rows)), physical_identity_resolver=_physical
    )
    by_slot_forward = {row["logicalSlotId"]: row["identityPlane"] for row in forward}
    by_slot_reverse = {row["logicalSlotId"]: row["identityPlane"] for row in reverse}
    assert by_slot_forward == by_slot_reverse
    round_trip = json.loads(json.dumps(forward, sort_keys=True))
    assert round_trip == forward
    assert validate_bound_identity_roster(
        round_trip, physical_identity_resolver=_physical
    )["success"]


def test_persisted_rows_restore_reservations_and_keep_distinct_stable_hashes() -> None:
    config = _configuration("f" * 64)
    bound = bind_identity_plane(
        [_work(0, config), _work(1, config)], physical_identity_resolver=_physical
    )
    rows = [_persisted(row) for row in bound]
    audit = validate_persisted_logical_identity_rows(rows)
    assert audit["success"]
    assert audit["uniqueLogicalResultIdentities"] == 2
    assert audit["uniqueStableLogicalHashes"] == 2

    forged = deepcopy(rows)
    forged[1]["reservedConfigurationSlotId"] = "copied"
    assert not validate_persisted_logical_identity_rows(forged)["success"]
