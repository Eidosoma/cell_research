"""S12AA integration of the outcome-independent S12W estimator contract.

This module does not define a transfer estimand.  It projects the byte-frozen
S12R populations and contrasts through the S12W-qualified endpoint-domain,
exact-pairing, all-reserved, restricted-time, finite-evidentiary, and fixed
Holm rules.  Exact fixed slots are derived from the frozen structural roster
before outcomes and are supplied again at analysis time as an authenticated
commitment.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import pandas as pd

from src.morph2d.channels import CHANNEL_LEDGER_FIELDS
from src.morph2d.movements import LEDGER_FIELDS as MOVEMENT_LEDGER_FIELDS
from src.morph2d.policies import OBSERVATION_LEDGER_FIELDS
from src.phenotype_discovery.publication import canonical_sha256
from src.spatial_transfer.estimator_feasibility import (
    ENDPOINT_SPECS,
    FIXED_FAMILIES,
    EndpointSpec,
    EstimatorFeasibilityError,
    FixedSlot,
    apply_fixed_holm,
    build_fixed_slot_record,
    exact_one_to_one_pair,
    non_evidentiary_record,
    validate_estimator_payload,
)

NATIVE_SCHEDULER_ID = "E06_state_blind_identity_hash_batch4"
ALTERNATE_SCHEDULER_ID = "identity_round_robin_batch4_v1"
FAULT_FAMILY_ID = "identity_locus_target_breaking_spurious_swap_v1"

FAULT_COST_FIELDS = (
    "faultEventsScheduled",
    "faultEventsTriggered",
    "faultEligibilityEvaluations",
    "faultCounterDraws",
    "faultCommittedSpuriousSwaps",
    "faultDisplacedEntities",
    "faultDisplacedCells",
    "faultDisplacedVacancies",
    "faultGraphDisplacement",
    "faultPolicyOpportunityUnits",
    "faultNativeProposalUnits",
)
SCHEDULER_COST_FIELDS = (
    "schedulerActorPopulationReads",
    "schedulerStartOffsetCounterDraws",
    "schedulerCursorArithmeticOperations",
    "schedulerSelectedActorSlots",
    "schedulerStateReads",
    "schedulerOutcomeReads",
    "schedulerReplacementWithinSweep",
)
NATIVE_COST_FIELDS: Mapping[str, tuple[str, ...]] = {
    "movementLedgerJson": tuple(MOVEMENT_LEDGER_FIELDS),
    "observationLedgerJson": tuple(OBSERVATION_LEDGER_FIELDS),
    "channelLedgerJson": tuple(CHANNEL_LEDGER_FIELDS),
}


def _cost_spec(output_endpoint: str) -> EndpointSpec:
    return replace(
        ENDPOINT_SPECS["separate_cost_component_nonnegative_integer"],
        output_endpoint=output_endpoint,
    )


def _endpoint_kinds(condition_id: str) -> tuple[str, ...]:
    """Return the already-registered task-local endpoints for one condition."""

    common = (
        "terminal_conjunctive_completion_binary",
        "minimum_mismatch_fraction_continuous",
        "restricted_native_transition_time_at_32",
    )
    if "spurious_swap_fault" in condition_id:
        return (*common, "repair_by_transition_32_binary")
    return (*common, "departure_after_initial_completion_binary")


def _slot_record(slot: FixedSlot) -> dict[str, Any]:
    record = {
        **slot.identity_payload(),
        "testId": slot.test_id,
        "endpointKind": slot.endpoint.endpoint_kind,
        "endpointSourceColumn": slot.endpoint.source_column,
        "endpointOutputName": slot.endpoint.output_endpoint,
        "benefitDirection": slot.endpoint.benefit_direction,
        "binary": slot.endpoint.binary,
    }
    record["slotCommitmentSha256"] = canonical_sha256(
        "E07/S12AA/fixed-estimator-slot/v1",
        record,
    )
    return record


def _slot_from_record(record: Mapping[str, Any]) -> FixedSlot:
    spec = EndpointSpec(
        endpoint_kind=str(record["endpointKind"]),
        output_endpoint=str(record["endpointOutputName"]),
        source_column=str(record["endpointSourceColumn"]),
        benefit_direction=int(record["benefitDirection"]),
        binary=bool(record["binary"]),
    )
    slot = FixedSlot(
        family=str(record["family"]),
        contrast_id=str(record["contrastId"]),
        task_id=str(record["taskId"]),
        panel_id=str(record["panelId"]),
        condition_id=str(record["conditionId"]),
        lineage_id=str(record["lineageId"]),
        endpoint=spec,
        left_label=str(record["left"]),
        right_label=str(record["right"]),
    )
    if slot.test_id != record["testId"]:
        raise EstimatorFeasibilityError("fixed slot identity commitment mismatch")
    if (
        canonical_sha256(
            "E07/S12AA/fixed-estimator-slot/v1",
            {
                key: value
                for key, value in record.items()
                if key != "slotCommitmentSha256"
            },
        )
        != record["slotCommitmentSha256"]
    ):
        raise EstimatorFeasibilityError("fixed slot commitment hash mismatch")
    return slot


def _structural_frame(
    roster: Sequence[Mapping[str, Any]] | pd.DataFrame,
) -> pd.DataFrame:
    frame = roster.copy() if isinstance(roster, pd.DataFrame) else pd.DataFrame(roster)
    required = {
        "partition",
        "taskId",
        "panelId",
        "conditionId",
        "conditionRole",
        "lineageId",
        "baseConfigurationId",
        "schedulerFamilyId",
        "faultFamilyId",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise EstimatorFeasibilityError(
            f"structural roster lacks required columns: {missing}"
        )
    return frame[frame["partition"] == "post_lock_transfer"].copy()


def build_fixed_slot_registry(
    roster: Sequence[Mapping[str, Any]] | pd.DataFrame,
    lineages: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Freeze exact S12R/S12W multiplicity slots from structural metadata."""

    post = _structural_frame(roster)
    slots: dict[str, dict[str, Any]] = {}

    def register(slot: FixedSlot) -> None:
        record = _slot_record(slot)
        previous = slots.get(slot.test_id)
        if previous is not None and previous != record:
            raise EstimatorFeasibilityError("fixed test identity collision")
        slots[slot.test_id] = record

    def register_endpoints(
        groups: pd.DataFrame,
        *,
        family: str,
        contrast_id: str,
        lineage_id: str,
        left_label: str,
        right_label: str,
    ) -> None:
        combinations = (
            groups[["taskId", "panelId", "conditionId"]]
            .drop_duplicates()
            .sort_values(["taskId", "panelId", "conditionId"])
        )
        for row in combinations.itertuples(index=False):
            for endpoint_kind in _endpoint_kinds(str(row.conditionId)):
                register(
                    FixedSlot(
                        family=family,
                        contrast_id=contrast_id,
                        task_id=str(row.taskId),
                        panel_id=str(row.panelId),
                        condition_id=str(row.conditionId),
                        lineage_id=lineage_id,
                        endpoint=ENDPOINT_SPECS[endpoint_kind],
                        left_label=left_label,
                        right_label=right_label,
                    )
                )

    for lineage in lineages:
        lineage_id = str(lineage["lineageId"])
        parent_id = str(lineage["parentConfigurationId"])
        compressed_id = str(lineage["compressedConfigurationId"])
        zero = post[
            (post["lineageId"] == lineage_id) & (post["conditionRole"] == "zero_shot")
        ]
        register_endpoints(
            zero[zero["baseConfigurationId"] == compressed_id],
            family="parent_compressed_calibrated_endpoint_family",
            contrast_id="parent_vs_compressed_within_task_cell",
            lineage_id=lineage_id,
            left_label=compressed_id,
            right_label=parent_id,
        )
        for candidate_id in (parent_id, compressed_id):
            register_endpoints(
                zero[zero["baseConfigurationId"] == candidate_id],
                family="matched_random_calibrated_endpoint_family",
                contrast_id="candidate_vs_matched_random",
                lineage_id=lineage_id,
                left_label=candidate_id,
                right_label=str(lineage["matchedRandomConfigurationId"]),
            )

    adapted = post[post["conditionRole"] == "adapted_winner_slot"]
    for base_id, rows in adapted.groupby("baseConfigurationId", sort=True):
        lineage_ids = sorted(map(str, rows["lineageId"].unique()))
        if len(lineage_ids) != 1:
            raise EstimatorFeasibilityError("base maps to ambiguous lineage")
        register_endpoints(
            rows,
            family="adaptation_calibrated_endpoint_family",
            contrast_id="zero_shot_vs_bounded_adaptation",
            lineage_id=lineage_ids[0],
            left_label=f"{base_id}:adapted",
            right_label=f"{base_id}:zero_shot",
        )

    scope = post[
        post["conditionRole"].isin(
            ["zero_shot", "adapted_winner_slot", "matched_random"]
        )
    ]
    alternate = scope[scope["schedulerFamilyId"] == ALTERNATE_SCHEDULER_ID]
    for lineage_id, rows in alternate.groupby("lineageId", sort=True):
        register_endpoints(
            rows,
            family="scheduler_calibrated_endpoint_family",
            contrast_id="alternate_vs_native_scheduler",
            lineage_id=str(lineage_id),
            left_label="alternate_scheduler",
            right_label="native_scheduler",
        )

    faulted = scope[scope["faultFamilyId"] == FAULT_FAMILY_ID]
    for lineage_id, rows in faulted.groupby("lineageId", sort=True):
        register_endpoints(
            rows,
            family="fault_repair_endpoint_family",
            contrast_id="fault_vs_no_fault",
            lineage_id=str(lineage_id),
            left_label="target_breaking_fault",
            right_label="no_fault",
        )

    tasks = sorted(map(str, post["taskId"].unique()))
    for task_id in tasks:
        register(
            FixedSlot(
                family="failure_harm_family",
                contrast_id="reserved_failure_rate_against_zero",
                task_id=task_id,
                panel_id="all_reserved_panels",
                condition_id="all_frozen_conditions",
                lineage_id="all",
                endpoint=ENDPOINT_SPECS["reserved_failure_binary"],
                left_label="S12R_reserved_rows",
                right_label="zero_failures",
            )
        )
        for ledger_column, fields in sorted(NATIVE_COST_FIELDS.items()):
            for component in fields:
                register(
                    FixedSlot(
                        family="native_cost_harm_family_by_component",
                        contrast_id="candidate_vs_native_baseline",
                        task_id=task_id,
                        panel_id="all_reserved_panels",
                        condition_id="all_frozen_conditions",
                        lineage_id="all",
                        endpoint=_cost_spec(f"{ledger_column}:{component}"),
                        left_label="candidate",
                        right_label="native_baseline",
                    )
                )
        for contrast_id, left_label, right_label, fields in (
            (
                "alternate_scheduler_vs_native_scheduler",
                "alternate_scheduler",
                "native_scheduler",
                SCHEDULER_COST_FIELDS,
            ),
            (
                "target_breaking_fault_vs_no_fault",
                "target_breaking_fault",
                "no_fault",
                FAULT_COST_FIELDS,
            ),
        ):
            for component in fields:
                register(
                    FixedSlot(
                        family="extension_cost_harm_family_by_component",
                        contrast_id=contrast_id,
                        task_id=task_id,
                        panel_id="applicable_calibrated_panels",
                        condition_id="frozen_axis_pair",
                        lineage_id="all",
                        endpoint=_cost_spec(component),
                        left_label=left_label,
                        right_label=right_label,
                    )
                )

    records = [slots[key] for key in sorted(slots)]
    family_counts = {
        family: sum(row["family"] == family for row in records)
        for family in FIXED_FAMILIES
    }
    registry = {
        "schemaVersion": "e07.s12aa.fixed-estimator-slot-registry.v1",
        "researchStepId": "S12AA",
        "generatedBeforeAnyEpisode": True,
        "sourcePopulation": "byte_frozen_S12R_structural_roster",
        "outcomeFieldsUsed": 0,
        "fixedFamilies": list(FIXED_FAMILIES),
        "slotCount": len(records),
        "familySlotCounts": family_counts,
        "slots": records,
    }
    registry["semanticSha256"] = canonical_sha256(
        "E07/S12AA/fixed-estimator-slot-registry/v1",
        registry,
    )
    return registry


def _registry_slots(
    registry: Mapping[str, Any],
) -> tuple[dict[tuple[str, ...], FixedSlot], list[str]]:
    body = dict(registry)
    claimed = body.pop("semanticSha256", None)
    if claimed != canonical_sha256(
        "E07/S12AA/fixed-estimator-slot-registry/v1",
        body,
    ):
        raise EstimatorFeasibilityError("fixed slot registry hash mismatch")
    slots = [_slot_from_record(row) for row in registry["slots"]]
    by_identity: dict[tuple[str, ...], FixedSlot] = {}
    for slot in slots:
        key = (
            slot.family,
            slot.contrast_id,
            slot.task_id,
            slot.panel_id,
            slot.condition_id,
            slot.lineage_id,
            slot.endpoint.output_endpoint,
            slot.left_label,
            slot.right_label,
        )
        if key in by_identity:
            raise EstimatorFeasibilityError("duplicate fixed slot identity")
        by_identity[key] = slot
    return by_identity, [slot.test_id for slot in slots]


def _pair(
    left: pd.DataFrame,
    right: pd.DataFrame,
    key_fields: Sequence[str],
) -> pd.DataFrame:
    left = left.copy()
    right = right.copy()
    pair_column = "_s12aaPairIdentity"
    left[pair_column] = left[list(key_fields)].astype(str).agg("\x1f".join, axis=1)
    right[pair_column] = right[list(key_fields)].astype(str).agg("\x1f".join, axis=1)
    return exact_one_to_one_pair(left, right, [pair_column])


def _split_merged(
    merged: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_column = "_s12aaPairIdentity"
    left_rows: list[dict[str, Any]] = []
    right_rows: list[dict[str, Any]] = []
    for row in merged.to_dict("records"):
        left_row = {pair_column: row[pair_column]}
        right_row = {pair_column: row[pair_column]}
        for name, value in row.items():
            if name.endswith("_left"):
                left_row[name[:-5]] = value
            elif name.endswith("_right"):
                right_row[name[:-6]] = value
        left_rows.append(left_row)
        right_rows.append(right_row)
    return pd.DataFrame(left_rows), pd.DataFrame(right_rows)


def _slot_key(
    *,
    family: str,
    contrast_id: str,
    task_id: str,
    panel_id: str,
    condition_id: str,
    lineage_id: str,
    endpoint: str,
    left_label: str,
    right_label: str,
) -> tuple[str, ...]:
    return (
        family,
        contrast_id,
        task_id,
        panel_id,
        condition_id,
        lineage_id,
        endpoint,
        left_label,
        right_label,
    )


def build_paired_estimands(
    frame: pd.DataFrame,
    lineages: Sequence[Mapping[str, Any]],
    fixed_registry: Mapping[str, Any],
) -> dict[str, Any]:
    """Project one complete S12R result frame through the S12W controls."""

    regenerated = build_fixed_slot_registry(frame, lineages)
    if regenerated["semanticSha256"] != fixed_registry["semanticSha256"]:
        raise EstimatorFeasibilityError(
            "post-execution structural slots differ from prospective registry"
        )
    slots, expected_ids = _registry_slots(fixed_registry)
    post = frame[frame["partition"] == "post_lock_transfer"].copy()
    records: list[dict[str, Any]] = []

    def slot_for(**kwargs: str) -> FixedSlot:
        key = _slot_key(**kwargs)
        try:
            return slots[key]
        except KeyError as exc:
            raise EstimatorFeasibilityError(f"unregistered fixed slot: {key}") from exc

    def add_endpoint_records(
        paired: pd.DataFrame,
        *,
        family: str,
        contrast_id: str,
        lineage_id: str,
        left_label: str,
        right_label: str,
    ) -> None:
        if paired.empty:
            raise EstimatorFeasibilityError(
                "declared paired population unexpectedly empty"
            )
        group_columns = ["taskId_left", "panelId_left", "conditionId_left"]
        for (task_id, panel_id, condition_id), group in paired.groupby(
            group_columns,
            sort=True,
        ):
            left, right = _split_merged(group)
            for endpoint_kind in _endpoint_kinds(str(condition_id)):
                spec = ENDPOINT_SPECS[endpoint_kind]
                slot = slot_for(
                    family=family,
                    contrast_id=contrast_id,
                    task_id=str(task_id),
                    panel_id=str(panel_id),
                    condition_id=str(condition_id),
                    lineage_id=lineage_id,
                    endpoint=spec.output_endpoint,
                    left_label=left_label,
                    right_label=right_label,
                )
                records.append(
                    build_fixed_slot_record(
                        slot,
                        left=left,
                        right=right,
                        pair_keys=["_s12aaPairIdentity"],
                    )
                )

    for lineage in lineages:
        lineage_id = str(lineage["lineageId"])
        parent_id = str(lineage["parentConfigurationId"])
        compressed_id = str(lineage["compressedConfigurationId"])
        base = post[
            (post["lineageId"] == lineage_id) & (post["conditionRole"] == "zero_shot")
        ]
        add_endpoint_records(
            _pair(
                base[base["baseConfigurationId"] == compressed_id],
                base[base["baseConfigurationId"] == parent_id],
                ["scenarioFamilyId"],
            ),
            family="parent_compressed_calibrated_endpoint_family",
            contrast_id="parent_vs_compressed_within_task_cell",
            lineage_id=lineage_id,
            left_label=compressed_id,
            right_label=parent_id,
        )
        comparator = post[
            (post["lineageId"] == lineage_id)
            & (post["conditionRole"] == "matched_random")
        ]
        for candidate_id in (parent_id, compressed_id):
            add_endpoint_records(
                _pair(
                    base[base["baseConfigurationId"] == candidate_id],
                    comparator,
                    ["scenarioFamilyId"],
                ),
                family="matched_random_calibrated_endpoint_family",
                contrast_id="candidate_vs_matched_random",
                lineage_id=lineage_id,
                left_label=candidate_id,
                right_label=str(lineage["matchedRandomConfigurationId"]),
            )

    adapted = post[post["conditionRole"] == "adapted_winner_slot"]
    for base_id, left in adapted.groupby("baseConfigurationId", sort=True):
        right = post[
            (post["conditionRole"] == "zero_shot")
            & (post["baseConfigurationId"] == base_id)
        ]
        lineage_ids = sorted(map(str, left["lineageId"].unique()))
        if len(lineage_ids) != 1:
            raise EstimatorFeasibilityError("adapted base maps to ambiguous lineage")
        add_endpoint_records(
            _pair(left, right, ["scenarioFamilyId"]),
            family="adaptation_calibrated_endpoint_family",
            contrast_id="zero_shot_vs_bounded_adaptation",
            lineage_id=lineage_ids[0],
            left_label=f"{base_id}:adapted",
            right_label=f"{base_id}:zero_shot",
        )

    scope = post[
        post["conditionRole"].isin(
            ["zero_shot", "adapted_winner_slot", "matched_random"]
        )
    ]
    scheduler_keys = [
        "taskId",
        "panelId",
        "scenarioOrdinal",
        "conditionRole",
        "lineageId",
        "baseConfigurationId",
        "faultFamilyId",
    ]
    paired_scheduler = _pair(
        scope[scope["schedulerFamilyId"] == ALTERNATE_SCHEDULER_ID],
        scope[scope["schedulerFamilyId"] == NATIVE_SCHEDULER_ID],
        scheduler_keys,
    )
    for lineage_id, group in paired_scheduler.groupby(
        "lineageId_left",
        sort=True,
    ):
        add_endpoint_records(
            group,
            family="scheduler_calibrated_endpoint_family",
            contrast_id="alternate_vs_native_scheduler",
            lineage_id=str(lineage_id),
            left_label="alternate_scheduler",
            right_label="native_scheduler",
        )

    fault_keys = [
        "taskId",
        "panelId",
        "scenarioOrdinal",
        "conditionRole",
        "lineageId",
        "baseConfigurationId",
        "schedulerFamilyId",
    ]
    paired_fault = _pair(
        scope[scope["faultFamilyId"] == FAULT_FAMILY_ID],
        scope[scope["faultFamilyId"] == "none"],
        fault_keys,
    )
    for lineage_id, group in paired_fault.groupby("lineageId_left", sort=True):
        add_endpoint_records(
            group,
            family="fault_repair_endpoint_family",
            contrast_id="fault_vs_no_fault",
            lineage_id=str(lineage_id),
            left_label="target_breaking_fault",
            right_label="no_fault",
        )

    for task_id, task in post.groupby("taskId", sort=True):
        pair_column = "_s12aaPairIdentity"
        left = task.copy()
        left[pair_column] = left["logicalReservationId"].astype(str)
        right = left.copy()
        right["status"] = "calibrated_endpoint_retained"
        right["endpointAvailable"] = True
        right["failed"] = False
        right["censored"] = False
        right["diagnostic"] = False
        failure_slot = slot_for(
            family="failure_harm_family",
            contrast_id="reserved_failure_rate_against_zero",
            task_id=str(task_id),
            panel_id="all_reserved_panels",
            condition_id="all_frozen_conditions",
            lineage_id="all",
            endpoint="failed",
            left_label="S12R_reserved_rows",
            right_label="zero_failures",
        )
        records.append(
            build_fixed_slot_record(
                failure_slot,
                left=left,
                right=right,
                pair_keys=[pair_column],
            )
        )

        candidates = task[
            task["conditionRole"].isin(["zero_shot", "adapted_winner_slot"])
        ]
        native_rows = task[task["conditionRole"] == "native_baseline"]
        for ledger_column, fields in sorted(NATIVE_COST_FIELDS.items()):
            for component in fields:
                native_slot = slot_for(
                    family="native_cost_harm_family_by_component",
                    contrast_id="candidate_vs_native_baseline",
                    task_id=str(task_id),
                    panel_id="all_reserved_panels",
                    condition_id="all_frozen_conditions",
                    lineage_id="all",
                    endpoint=f"{ledger_column}:{component}",
                    left_label="candidate",
                    right_label="native_baseline",
                )
                record = non_evidentiary_record(
                    native_slot,
                    reason=(
                        "NO_ONE_TO_ONE_NATIVE_BASELINE_PAIR_AT_CANDIDATE_CARDINALITY"
                    ),
                )
                record["leftReservedCount"] = len(candidates)
                record["rightReservedCount"] = len(native_rows)
                records.append(record)

    for (
        paired,
        ledger_column,
        contrast_id,
        left_label,
        right_label,
        fields,
    ) in (
        (
            paired_scheduler,
            "schedulerLedgerJson",
            "alternate_scheduler_vs_native_scheduler",
            "alternate_scheduler",
            "native_scheduler",
            SCHEDULER_COST_FIELDS,
        ),
        (
            paired_fault,
            "faultLedgerJson",
            "target_breaking_fault_vs_no_fault",
            "target_breaking_fault",
            "no_fault",
            FAULT_COST_FIELDS,
        ),
    ):
        for task_id, group in paired.groupby("taskId_left", sort=True):
            left, right = _split_merged(group)
            for component in fields:
                left["separateCostComponent"] = left[ledger_column].map(
                    lambda value, name=component: json.loads(value)[name]
                )
                right["separateCostComponent"] = right[ledger_column].map(
                    lambda value, name=component: json.loads(value)[name]
                )
                cost_slot = slot_for(
                    family="extension_cost_harm_family_by_component",
                    contrast_id=contrast_id,
                    task_id=str(task_id),
                    panel_id="applicable_calibrated_panels",
                    condition_id="frozen_axis_pair",
                    lineage_id="all",
                    endpoint=component,
                    left_label=left_label,
                    right_label=right_label,
                )
                records.append(
                    build_fixed_slot_record(
                        cost_slot,
                        left=left,
                        right=right,
                        pair_keys=["_s12aaPairIdentity"],
                    )
                )

    adjusted = apply_fixed_holm(records, expected_test_ids=expected_ids)
    supportive = any(
        row["holmReject"]
        and row["benefitEffect"] is not None
        and row["benefitEffect"] > 0
        and row["family"]
        in {
            "parent_compressed_calibrated_endpoint_family",
            "adaptation_calibrated_endpoint_family",
            "matched_random_calibrated_endpoint_family",
            "scheduler_calibrated_endpoint_family",
            "fault_repair_endpoint_family",
        }
        for row in adjusted
    )
    harm = any(
        row["holmReject"]
        and row["benefitEffect"] is not None
        and row["benefitEffect"] < 0
        for row in adjusted
    )
    classification = (
        "constraining/contradictory" if harm else "supportive" if supportive else "null"
    )
    payload = {
        "schemaVersion": "e07.s12w.paired-estimands.v1",
        "researchStepId": "S12AA",
        "estimatorQualificationStepId": "S12W",
        "taskLocalOnly": True,
        "universalScore": None,
        "pairedBootstrapReplicates": 2_000,
        "confidenceLevel": 0.95,
        "multiplicityMethod": "Holm",
        "fixedFamilies": list(FIXED_FAMILIES),
        "fixedSlotRegistrySha256": fixed_registry["semanticSha256"],
        "recordCount": len(adjusted),
        "holmRejectedCount": sum(row["holmReject"] for row in adjusted),
        "evidentiaryRecordCount": sum(row["evidentiary"] for row in adjusted),
        "nonEvidentiaryRecordCount": sum(not row["evidentiary"] for row in adjusted),
        "supportiveEfficacySignalPresent": supportive,
        "adjustedHarmPresent": harm,
        "outcomeClassification": classification,
        "records": adjusted,
    }
    validate_estimator_payload(payload, expected_test_ids=expected_ids)
    return payload
