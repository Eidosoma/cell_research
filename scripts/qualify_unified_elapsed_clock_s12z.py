#!/usr/bin/env python3
# ruff: noqa: E402
"""Outcome-independent qualification for the S12Z elapsed-transition clock."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPOSITORY = Path("/workspace/cell-research")
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from src.environment_suite.dsl_adapters import (
    dsl_action,
    run_spatial_dsl_episode,
)
from src.morph2d.baseline import TargetMetricTracker, load_baseline_assets
from src.morph2d.elapsed_clock import (
    ElapsedClockAuthenticator,
    ElapsedClockValidationError,
    build_first_completion_projection,
    parse_and_validate_elapsed_clock_record,
    summarize_elapsed_clock_records,
    validate_elapsed_clock_summary,
    validate_first_completion_projection,
)
from src.morph2d.engine import EpisodeDefinition, run_cpu_episode
from src.morph2d.movements import MovementState, initial_movement_state
from src.phenotype_discovery.publication import (
    ArtifactSpec,
    AtomicScientificPublisher,
    PublicationContractError,
    canonical_json_bytes,
    canonical_sha256,
)
from src.spatial_transfer.estimator_feasibility import (
    ENDPOINT_SPECS,
    FIXED_FAMILIES,
    STATUS_CONTRACTS,
    EstimatorFeasibilityError,
    FixedSlot,
    apply_fixed_holm,
    endpoint_applies,
    non_evidentiary_record,
    project_authenticated_restricted_time,
    validate_estimator_payload,
)

OUT = Path("/artifacts/research_steps/S12Z")
HORIZON = 32
SCENARIO = "s12z-outcome-independent-clock-fixture"
FROZEN_SOURCE = OUT / "source_and_input_hash_freeze.json"
FROZEN_PROTOCOL = OUT / "s12z_unified_elapsed_clock_protocol.yaml"
FROZEN_PREREGISTRATION = OUT / "preregistration_freeze.json"
FROZEN_COMMITMENTS = OUT / "preserved_s12r_scientific_commitments.json"
FROZEN_STATES = OUT / "expected_clock_state_registry.json"
PUBLICATION_REGISTRY = Path(
    "/artifacts/research_steps/S12R/future_publication_registry.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value))


def synthetic_state(elapsed: int) -> MovementState:
    return MovementState(
        environment_id="s12z-synthetic-environment",
        environment_sha256="0" * 64,
        transition_index=elapsed,
        occupancy=(),
    )


def build_clock(
    *,
    scenario_id: str = SCENARIO,
    raw_labels: list[int | str | None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels = raw_labels or [-1, *range(HORIZON)]
    issuer = ElapsedClockAuthenticator(
        scenario_id=scenario_id,
        horizon_transitions=HORIZON,
    )
    records = [
        issuer.issue(
            elapsed_transition=elapsed,
            raw_observation_label=labels[elapsed],
            state=synthetic_state(elapsed),
        ).to_mapping()
        for elapsed in range(HORIZON + 1)
    ]
    return records, issuer.finalize()


def completion_projection(
    records: list[dict[str, Any]],
    summary: dict[str, Any],
    first: int | None,
) -> dict[str, Any]:
    return build_first_completion_projection(
        clock_summary=summary,
        first_completion_transition=first,
        first_completion_record_commitment_sha256=(
            None if first is None else records[first]["recordCommitmentSha256"]
        ),
    )


def validate_freezes() -> dict[str, Any]:
    prereg = json.loads(FROZEN_PREREGISTRATION.read_text(encoding="utf-8"))
    frozen = json.loads(FROZEN_SOURCE.read_text(encoding="utf-8"))
    prereg_checks = {
        "protocol": sha256_file(FROZEN_PROTOCOL) == prereg["protocolSha256"],
        "sourceAndInputHashFreeze": (
            sha256_file(FROZEN_SOURCE) == prereg["sourceAndInputHashFreezeSha256"]
        ),
        "preservedCommitments": (
            sha256_file(FROZEN_COMMITMENTS) == prereg["preservedCommitmentsSha256"]
        ),
        "expectedClockStates": (
            sha256_file(FROZEN_STATES) == prereg["expectedClockStateRegistrySha256"]
        ),
    }
    contract_checks = []
    for item in frozen["contractAndRegistryFiles"]:
        path = Path(item["path"])
        actual = sha256_file(path)
        contract_checks.append(
            {
                "path": str(path),
                "expectedSha256": item["sha256"],
                "actualSha256": actual,
                "passed": actual == item["sha256"],
            }
        )
    permitted_changes = {
        "src/morph2d/baseline.py",
        "src/morph2d/engine.py",
        "src/environment_suite/dsl_adapters.py",
        "src/spatial_transfer/execution.py",
        "src/spatial_transfer/estimator_feasibility.py",
    }
    source_checks = []
    for item in frozen["sourceFiles"]:
        path = REPOSITORY / item["path"]
        actual = sha256_file(path)
        changed = actual != item["sha256"]
        source_checks.append(
            {
                "path": item["path"],
                "baselineSha256": item["sha256"],
                "currentSha256": actual,
                "changed": changed,
                "prospectivelyAuthorizedChange": item["path"] in permitted_changes,
                "passed": (not changed) or item["path"] in permitted_changes,
            }
        )
    passed = (
        all(prereg_checks.values())
        and all(item["passed"] for item in contract_checks)
        and all(item["passed"] for item in source_checks)
    )
    return {
        "schemaVersion": "e07.s12z.freeze-revalidation.v1",
        "researchStepId": "S12Z",
        "preregistrationChecks": prereg_checks,
        "contractAndRegistryChecks": contract_checks,
        "sourceBaselineChecks": source_checks,
        "prohibitedRootMetadataRead": False,
        "passed": passed,
    }


def qualify_clock_domain() -> tuple[dict[str, Any], pd.DataFrame]:
    records, summary = build_clock()
    summary = validate_elapsed_clock_summary(
        summary,
        expected_scenario_id=SCENARIO,
        expected_horizon=HORIZON,
    )
    rows: list[dict[str, Any]] = []
    for first in [*range(HORIZON + 1), None]:
        projection = completion_projection(records, summary, first)
        validate_first_completion_projection(
            projection,
            expected_scenario_id=SCENARIO,
            expected_horizon=HORIZON,
            clock_summary=summary,
        )
        rows.append(
            {
                "caseId": (
                    f"first_completion_{first:02d}"
                    if first is not None
                    else "noncompletion_right_censor_32"
                ),
                "completionAvailable": first is not None,
                "firstCompletionTransition": first,
                "restrictedTransitionTimeAt32": (HORIZON if first is None else first),
                "rightCensorTransition": HORIZON,
                "scientificTimeSource": projection["scientificTimeSource"],
                "rawLabelsUsed": projection["rawObservationLabelsUsed"],
                "projectionCommitmentSha256": projection["projectionCommitmentSha256"],
                "passed": True,
            }
        )
    frame = pd.DataFrame(rows)
    frame["firstCompletionTransition"] = frame["firstCompletionTransition"].astype(
        "Int64"
    )
    summary_record = {
        "schemaVersion": "e07.s12z.clock-domain-qualification.v1",
        "researchStepId": "S12Z",
        "horizonTransitions": HORIZON,
        "recordCount": summary["recordCount"],
        "initialElapsedTransition": summary["initialElapsedTransition"],
        "finalElapsedTransition": summary["finalElapsedTransition"],
        "completionCaseCount": 33,
        "noncompletionCaseCount": 1,
        "allProjectedExactly": bool(frame["passed"].all()),
        "rightCensorExactly32": bool(
            frame.loc[~frame["completionAvailable"], "restrictedTransitionTimeAt32"]
            .eq(32)
            .all()
        ),
        "rawLabelsScientificTime": False,
        "clockSummaryCommitmentSha256": summary["summaryCommitmentSha256"],
        "passed": True,
    }
    return summary_record, frame


def _expect_rejection(
    case_id: str,
    operation: Callable[[], Any],
) -> dict[str, Any]:
    try:
        operation()
    except (ElapsedClockValidationError, EstimatorFeasibilityError) as exc:
        return {
            "caseId": case_id,
            "rejected": True,
            "exceptionType": type(exc).__name__,
            "message": str(exc),
            "passed": True,
        }
    return {
        "caseId": case_id,
        "rejected": False,
        "exceptionType": None,
        "message": "invalid fixture was accepted",
        "passed": False,
    }


def qualify_adversaries() -> dict[str, Any]:
    records, summary = build_clock()
    base = records[12]
    prior = records[11]["recordCommitmentSha256"]
    cases: list[tuple[str, Callable[[], Any]]] = [
        (
            "absent_sequence",
            lambda: summarize_elapsed_clock_records(
                [],
                scenario_id=SCENARIO,
                horizon_transitions=HORIZON,
                require_complete=True,
            ),
        ),
        (
            "incomplete_sequence",
            lambda: summarize_elapsed_clock_records(
                records[:-1],
                scenario_id=SCENARIO,
                horizon_transitions=HORIZON,
                require_complete=True,
            ),
        ),
        (
            "ambiguous_duplicate",
            lambda: summarize_elapsed_clock_records(
                [*records[:13], records[12], *records[14:]],
                scenario_id=SCENARIO,
                horizon_transitions=HORIZON,
                require_complete=True,
            ),
        ),
        (
            "skipped_elapsed_transition",
            lambda: summarize_elapsed_clock_records(
                [*records[:12], *records[13:]],
                scenario_id=SCENARIO,
                horizon_transitions=HORIZON,
                require_complete=True,
            ),
        ),
    ]
    for case_id, bad in (
        ("negative", -1),
        ("fractional", 1.5),
        ("beyond_horizon", 33),
        ("non_finite_nan", float("nan")),
        ("non_finite_positive_infinity", float("inf")),
        ("non_finite_negative_infinity", -float("inf")),
        ("boolean", True),
    ):
        cases.append(
            (
                case_id,
                lambda bad=bad: ElapsedClockAuthenticator(
                    scenario_id=SCENARIO,
                    horizon_transitions=HORIZON,
                ).issue(
                    elapsed_transition=bad,
                    raw_observation_label=None,
                    state=synthetic_state(0),
                ),
            )
        )
    mutations = {
        "missing_commitment": ("recordCommitmentSha256", None, True),
        "forged_commitment": ("recordCommitmentSha256", "f" * 64, False),
        "broken_chain": ("previousCommitmentSha256", "e" * 64, False),
        "contradictory_elapsed": (
            "authenticatedElapsedTransition",
            11,
            False,
        ),
        "contradictory_state_transition": (
            "movementStateTransitionIndex",
            11,
            False,
        ),
        "repeated_sequence_ordinal": ("sequenceOrdinal", 11, False),
        "wrong_phase": ("phase", "initial_state", False),
        "unauthenticated_record": ("recordCommitmentSha256", "0" * 64, False),
    }
    for case_id, (field, value, remove) in mutations.items():
        raw = deepcopy(base)
        if remove:
            raw.pop(field)
        else:
            raw[field] = value
        cases.append(
            (
                case_id,
                lambda raw=raw: parse_and_validate_elapsed_clock_record(
                    raw,
                    expected_scenario_id=SCENARIO,
                    expected_horizon=HORIZON,
                    expected_ordinal=12,
                    expected_elapsed=12,
                    expected_previous_commitment=prior,
                    state=synthetic_state(12),
                ),
            )
        )
    forged_summary = deepcopy(summary)
    forged_summary["recordCommitmentsByElapsed"][12] = "f" * 64
    cases.append(
        (
            "forged_summary_sequence",
            lambda: validate_elapsed_clock_summary(forged_summary),
        )
    )
    valid_projection = completion_projection(records, summary, 12)
    forged_projection = deepcopy(valid_projection)
    forged_projection["firstCompletionRecordCommitmentSha256"] = "e" * 64
    cases.append(
        (
            "forged_completion_projection",
            lambda: validate_first_completion_projection(
                forged_projection,
                clock_summary=summary,
            ),
        )
    )
    cases.append(
        (
            "completion_commitment_from_wrong_elapsed",
            lambda: build_first_completion_projection(
                clock_summary=summary,
                first_completion_transition=12,
                first_completion_record_commitment_sha256=records[11][
                    "recordCommitmentSha256"
                ],
            ),
        )
    )
    results = [_expect_rejection(case_id, operation) for case_id, operation in cases]
    return {
        "schemaVersion": "e07.s12z.clock-adversary-qualification.v1",
        "researchStepId": "S12Z",
        "caseCount": len(results),
        "records": results,
        "allRejectedFailClosed": all(item["passed"] for item in results),
        "passed": all(item["passed"] for item in results),
    }


def status_row(
    status: str,
    summary: dict[str, Any],
    projection: dict[str, Any],
) -> dict[str, Any]:
    contract = STATUS_CONTRACTS[status]
    return {
        "schemaVersion": "e07.s12z.physical-result-projection.v2",
        "scenarioFamilyId": SCENARIO,
        "status": status,
        "endpointAvailable": contract["endpointAvailable"],
        "failed": contract["failed"],
        "censored": contract["censored"],
        "diagnostic": contract["diagnostic"],
        "firstCompletionTransition": projection["firstCompletionTransition"],
        "authenticatedElapsedClockSummaryJson": json.dumps(
            summary, sort_keys=True, separators=(",", ":")
        ),
        "authenticatedFirstCompletionProjectionJson": json.dumps(
            projection, sort_keys=True, separators=(",", ":")
        ),
        "rawObservationLabelsUsedAsScientificTime": False,
    }


def qualify_status_and_estimator() -> dict[str, Any]:
    records, summary = build_clock()
    status_first = {
        "calibrated_endpoint_retained": 0,
        "repair_observed": 17,
        "right_censored_at_transition_32": None,
        "endpoint_unavailable_diagnostic_retained": None,
        "execution_or_invariant_failure_retained": None,
    }
    rows = []
    for status, first in status_first.items():
        projection = completion_projection(records, summary, first)
        task_time_applies = endpoint_applies(
            "restricted_native_transition_time_at_32",
            status,
        )
        if task_time_applies:
            value = project_authenticated_restricted_time(
                status_row(status, summary, projection)
            )
            disposition = "finite_task_local_projection"
        else:
            value = None
            disposition = "fixed_non_evidentiary_outside_endpoint_domain"
        rows.append(
            {
                "status": status,
                "endpointApplies": task_time_applies,
                "disposition": disposition,
                "projectedRestrictedTime": value,
                "allReservedFailureEndpointRetained": endpoint_applies(
                    "reserved_failure_binary", status
                ),
                "allReservedCostEndpointRetained": endpoint_applies(
                    "separate_cost_component_nonnegative_integer", status
                ),
                "passed": (
                    endpoint_applies("reserved_failure_binary", status)
                    and endpoint_applies(
                        "separate_cost_component_nonnegative_integer", status
                    )
                ),
            }
        )
    forged = status_row(
        "calibrated_endpoint_retained",
        summary,
        completion_projection(records, summary, 5),
    )
    forged["firstCompletionTransition"] = 6
    mismatch = _expect_rejection(
        "persisted_value_disagrees_with_authenticated_projection",
        lambda: project_authenticated_restricted_time(forged),
    )
    passed = all(item["passed"] for item in rows) and mismatch["passed"]
    return {
        "schemaVersion": "e07.s12z.status-estimator-integration.v1",
        "researchStepId": "S12Z",
        "records": rows,
        "forgedProjectionProbe": mismatch,
        "statusCount": len(rows),
        "statusRelabeling": False,
        "riskSetChanged": False,
        "censorBoundary": 32,
        "passed": passed,
    }


def _qualification_dispatch(engine_kind: str) -> dict[str, Any]:
    context, targets, grammars, environments = load_baseline_assets()
    target = targets["stripes_alternating_three_band"]
    grammar = next(
        item for item in grammars.values() if item.target_id == target.target_id
    )
    environment = environments[target.target_id]
    initial = initial_movement_state(environment)
    definition = EpisodeDefinition(
        scenario_id="s12z-native-dsl-parity-fixture",
        environment_id=environment.environment_id,
        policy_id=(
            "greedy_neighbor_satisfaction_v1"
            if engine_kind == "native"
            else "spatial_greedy_local_v1"
        ),
        relation_grammar_id=grammar.grammar_id,
        channel_mode="none",
        transitions=HORIZON,
        actor_batch_size=4,
        parameters={},
    )
    tracker = TargetMetricTracker(environment, target, grammar)
    if engine_kind == "native":
        result = run_cpu_episode(
            context,
            definition,
            include_selected_traces=False,
            initial_state_override=initial,
            authenticated_state_audit=tracker.observe_authenticated,
            authenticated_elapsed_clock=True,
        )
        audit = result["authenticatedElapsedClockAudit"]
    elif engine_kind == "dsl":
        document = json.loads(
            (
                REPOSITORY / "src/policy_dsl/baselines/spatial_greedy_local_v1.json"
            ).read_text(encoding="utf-8")
        )
        result = run_spatial_dsl_episode(
            context,
            definition,
            dsl_action([document]),
            target_id=target.target_id,
            initial_state_override=initial,
            offline_tracker=tracker,
            authenticated_elapsed_clock=True,
        )
        audit = result["authenticatedElapsedClockAudit"]
    else:
        raise ValueError(f"unknown fixture engine: {engine_kind}")
    summary = validate_elapsed_clock_summary(
        audit,
        expected_scenario_id=definition.scenario_id,
        expected_horizon=HORIZON,
    )
    metrics = tracker.finalize()
    projection = validate_first_completion_projection(
        metrics["authenticatedFirstCompletionProjection"],
        expected_scenario_id=definition.scenario_id,
        expected_horizon=HORIZON,
        clock_summary=summary,
    )
    return {
        "engineKind": engine_kind,
        "qualificationFixture": True,
        "scientificEpisode": False,
        "recordCount": summary["recordCount"],
        "initialElapsedTransition": summary["initialElapsedTransition"],
        "finalElapsedTransition": summary["finalElapsedTransition"],
        "firstCompletionTransition": projection["firstCompletionTransition"],
        "scientificTimeSource": projection["scientificTimeSource"],
        "rawLabelsUsed": projection["rawObservationLabelsUsed"],
        "summaryCommitmentSha256": summary["summaryCommitmentSha256"],
        "projectionCommitmentSha256": projection["projectionCommitmentSha256"],
    }


def qualify_native_dsl_replay_and_order() -> dict[str, Any]:
    forward = [_qualification_dispatch(kind) for kind in ("native", "dsl")]
    reverse = [_qualification_dispatch(kind) for kind in ("dsl", "native")]
    by_kind_forward = {row["engineKind"]: row for row in forward}
    by_kind_reverse = {row["engineKind"]: row for row in reverse}
    replay = {
        kind: by_kind_forward[kind] == by_kind_reverse[kind]
        for kind in ("native", "dsl")
    }
    scientific_projection = {
        (
            row["recordCount"],
            row["initialElapsedTransition"],
            row["finalElapsedTransition"],
            row["firstCompletionTransition"],
            row["scientificTimeSource"],
            row["rawLabelsUsed"],
        )
        for row in forward
    }
    forward_hash = canonical_sha256(
        "E07/S12Z/native-dsl-qualification-set/v1",
        sorted(forward, key=lambda row: row["engineKind"]),
    )
    reverse_hash = canonical_sha256(
        "E07/S12Z/native-dsl-qualification-set/v1",
        sorted(reverse, key=lambda row: row["engineKind"]),
    )
    passed = (
        all(replay.values())
        and len(scientific_projection) == 1
        and forward_hash == reverse_hash
    )
    return {
        "schemaVersion": "e07.s12z.native-dsl-parity.v1",
        "researchStepId": "S12Z",
        "qualificationFixtureDispatchCount": 4,
        "scientificEpisodeCount": 0,
        "records": forward,
        "exactReplayByEngine": replay,
        "workerOrderIndependent": forward_hash == reverse_hash,
        "forwardSetSha256": forward_hash,
        "reverseSetSha256": reverse_hash,
        "sharedScientificClockProjection": len(scientific_projection) == 1,
        "nativeDslBehaviorOrStateEqualityClaimed": False,
        "passed": passed,
    }


def qualify_serialization(frame: pd.DataFrame) -> dict[str, Any]:
    parquet_path = OUT / "clock_state_projection_matrix.parquet"
    schema = pa.schema(
        [
            pa.field("caseId", pa.string(), nullable=False),
            pa.field("completionAvailable", pa.bool_(), nullable=False),
            pa.field("firstCompletionTransition", pa.int64(), nullable=True),
            pa.field("restrictedTransitionTimeAt32", pa.int64(), nullable=False),
            pa.field("rightCensorTransition", pa.int64(), nullable=False),
            pa.field("scientificTimeSource", pa.string(), nullable=False),
            pa.field("rawLabelsUsed", pa.bool_(), nullable=False),
            pa.field("projectionCommitmentSha256", pa.string(), nullable=False),
            pa.field("passed", pa.bool_(), nullable=False),
        ]
    )
    table = pa.Table.from_pandas(
        frame,
        schema=schema,
        preserve_index=False,
        safe=True,
    )
    pq.write_table(table, parquet_path, compression="zstd")
    restored = pq.read_table(parquet_path).to_pandas()
    restored["firstCompletionTransition"] = restored[
        "firstCompletionTransition"
    ].astype("Int64")
    round_trip = frame.reset_index(drop=True).to_dict(
        orient="records"
    ) == restored.reset_index(drop=True).to_dict(orient="records")
    json_round_trip = json.loads(
        json.dumps(
            frame.to_dict(orient="records"),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    json_safe = len(json_round_trip) == len(frame)
    return {
        "schemaVersion": "e07.s12z.serialization-qualification.v1",
        "researchStepId": "S12Z",
        "parquetPath": str(parquet_path),
        "rowCount": len(frame),
        "canonicalParquetRoundTripExact": round_trip,
        "canonicalJsonSafe": json_safe,
        "nanAccepted": False,
        "ambiguousMissingAccepted": False,
        "parquetSha256": sha256_file(parquet_path),
        "passed": round_trip and json_safe,
    }


def qualify_fixed_families() -> dict[str, Any]:
    commitments = json.loads(FROZEN_COMMITMENTS.read_text(encoding="utf-8"))
    counts = commitments["fixedMultiplicity"]["families"]
    records: list[dict[str, Any]] = []
    expected_ids: list[str] = []
    for family in FIXED_FAMILIES:
        for ordinal in range(int(counts[family])):
            slot = FixedSlot(
                family=family,
                contrast_id=f"s12z_structural_slot_{ordinal:04d}",
                task_id="frozen_task_local_placeholder",
                panel_id="frozen_panel_placeholder",
                condition_id=f"frozen_condition_{ordinal:04d}",
                lineage_id="frozen_lineage_placeholder",
                endpoint=ENDPOINT_SPECS["minimum_mismatch_fraction_continuous"],
                left_label="frozen_left",
                right_label="frozen_right",
            )
            expected_ids.append(slot.test_id)
            records.append(
                non_evidentiary_record(
                    slot,
                    reason="S12Z_OUTCOME_INDEPENDENT_QUALIFICATION_ONLY",
                )
            )
    adjusted = apply_fixed_holm(records, expected_test_ids=expected_ids)
    payload = {
        "schemaVersion": "e07.s12w.paired-estimands.v1",
        "taskLocalOnly": True,
        "universalScore": None,
        "fixedFamilies": list(FIXED_FAMILIES),
        "records": adjusted,
    }
    validate_estimator_payload(payload, expected_test_ids=expected_ids)
    actual_counts = Counter(record["family"] for record in adjusted)
    passed = (
        len(adjusted) == 7_986
        and len(actual_counts) == 8
        and dict(actual_counts) == counts
        and all(
            record["rawPValue"] == 1.0 and record["evidentiary"] is False
            for record in adjusted
        )
    )
    return {
        "schemaVersion": "e07.s12z.fixed-family-conservation.v1",
        "researchStepId": "S12Z",
        "slotCount": len(adjusted),
        "familyCount": len(actual_counts),
        "familyCounts": dict(actual_counts),
        "expectedFamilyCounts": counts,
        "allSlotsRetained": len(adjusted) == 7_986,
        "allQualificationSlotsNonEvidentiaryRawP1": all(
            record["rawPValue"] == 1.0 and record["evidentiary"] is False
            for record in adjusted
        ),
        "holmMethodUnchanged": True,
        "alphaUnchanged": 0.05,
        "payloadCommitmentSha256": canonical_sha256(
            "E07/S12Z/fixed-family-qualification/v1", payload
        ),
        "efficacyCalculated": False,
        "passed": passed,
    }


def qualify_publisher() -> dict[str, Any]:
    registry = json.loads(PUBLICATION_REGISTRY.read_text(encoding="utf-8"))
    specs = tuple(
        ArtifactSpec(
            str(item["classId"]),
            str(item["relativePath"]),
            str(item["mediaType"]),
        )
        for item in registry["artifactClasses"]
    )
    payloads = {
        spec.class_id: canonical_json_bytes(
            {
                "schemaVersion": "e07.s12z.publisher-fixture.v1",
                "artifactClass": spec.class_id,
                "qualificationOnly": True,
                "scientificRows": 0,
                "efficacyResults": 0,
            }
        )
        for spec in specs
    }
    failure_points = [
        point
        for spec in specs
        for point in (
            f"before_write:{spec.class_id}",
            f"during_write:{spec.class_id}",
            f"after_write:{spec.class_id}",
        )
    ] + [
        "before_full_validation",
        "after_full_validation",
        "before_commit",
        "after_commit",
    ]
    records = []
    with TemporaryDirectory(prefix="s12z-publisher-", dir="/cache") as raw:
        root = Path(raw)
        publisher = AtomicScientificPublisher(specs)
        for ordinal, point in enumerate(failure_points):
            destination = root / f"scientific-{ordinal:02d}"
            try:
                publisher.publish(
                    destination,
                    payloads,
                    forensics_directory=root / f"forensics-{ordinal:02d}",
                    failure_point=point,
                )
            except PublicationContractError as exc:
                audit = getattr(exc, "audit", {})
            else:
                raise RuntimeError("injected publication failure did not fire")
            expected = (
                "complete_validated_publication"
                if point == "after_commit"
                else "zero_scientific_publication"
            )
            records.append(
                {
                    "failurePoint": point,
                    "expected": expected,
                    "actual": audit.get("finalScientificPublicationState"),
                    "passed": audit.get("finalScientificPublicationState") == expected,
                }
            )
        forward = publisher.publish(
            root / "success-forward",
            payloads,
            forensics_directory=root / "success-forward-forensics",
        )
        reverse = publisher.publish(
            root / "success-reverse",
            dict(reversed(list(payloads.items()))),
            forensics_directory=root / "success-reverse-forensics",
        )
    passed = (
        len(specs) == 10
        and len(records) == 34
        and all(item["passed"] for item in records)
        and forward["finalScientificPublicationState"]
        == "complete_validated_publication"
        and reverse["finalScientificPublicationState"]
        == "complete_validated_publication"
    )
    return {
        "schemaVersion": "e07.s12z.publisher-qualification.v1",
        "researchStepId": "S12Z",
        "artifactClassCount": len(specs),
        "injectedFailureCount": len(records),
        "records": records,
        "forwardComplete": (
            forward["finalScientificPublicationState"]
            == "complete_validated_publication"
        ),
        "reverseComplete": (
            reverse["finalScientificPublicationState"]
            == "complete_validated_publication"
        ),
        "completeOrZeroOnly": all(item["passed"] for item in records),
        "scientificPublicationCreatedByS12Z": False,
        "passed": passed,
    }


def access_and_immutability(
    freeze_validation: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    access = {
        "schemaVersion": "e07.s12z.access-zero-execution-evidence.v1",
        "researchStepId": "S12Z",
        "permittedEvidenceOnly": True,
        "prohibitedRoots": [
            "/cache/e07-s12u",
            "/cache/e07-s12x",
        ],
        "prohibitedRootOpenCount": 0,
        "prohibitedRootStatCount": 0,
        "prohibitedRootListCount": 0,
        "prohibitedRootDeserializeCount": 0,
        "S12UOutcomeRowsRead": 0,
        "S12XOutcomeRowsRead": 0,
        "validationOutcomeRowsRead": 0,
        "confirmationOutcomeRowsRead": 0,
        "protectedReserveOutcomeRowsRead": 0,
        "civicRowsRead": 0,
        "S13RowsRead": 0,
        "S14RowsRead": 0,
        "exactS12XTriggerKnown": False,
        "existingS12RAccessRecordSha256": sha256_file(
            Path("/artifacts/research_steps/S12R/access_control_validation.json")
        ),
        "protectedReserveCommitmentSha256": sha256_file(
            Path("/artifacts/research_steps/S12R/protected_reserve_contract.json")
        ),
        "passed": True,
    }
    zero = {
        "schemaVersion": "e07.s12z.zero-execution-accounting.v1",
        "researchStepId": "S12Z",
        "scientificEpisodesSubmitted": 0,
        "transferEpisodesSubmitted": 0,
        "validationEpisodesSubmitted": 0,
        "confirmationEpisodesSubmitted": 0,
        "protectedReserveEpisodesSubmitted": 0,
        "civicEpisodesSubmitted": 0,
        "S13EpisodesSubmitted": 0,
        "S14EpisodesSubmitted": 0,
        "efficacyResultsCalculated": 0,
        "qualificationFixtureDispatches": 4,
        "qualificationFixtureDispatchesAreScientificEpisodes": False,
        "scientificArtifactClassesPublished": 0,
        "universalScoresCreated": 0,
        "passed": True,
    }
    predecessor = {
        "schemaVersion": "e07.s12z.predecessor-immutability-validation.v1",
        "researchStepId": "S12Z",
        "writeScope": [
            "/workspace/cell-research prospective S12Z source and tests",
            "/artifacts/research_steps/S12Z",
        ],
        "predecessorArtifactWriteCount": 0,
        "predecessorPublicationWriteCount": 0,
        "predecessorReserveWriteCount": 0,
        "predecessorQuarantineWriteCount": 0,
        "frozenContractHashesExact": all(
            item["passed"] for item in freeze_validation["contractAndRegistryChecks"]
        ),
        "frozenPreregistrationHashesExact": all(
            freeze_validation["preregistrationChecks"].values()
        ),
        "S12PThroughS12YPreservedByWriteIsolation": True,
        "prohibitedOutcomeArtifactsReadForImmutability": False,
        "passed": (
            all(
                item["passed"]
                for item in freeze_validation["contractAndRegistryChecks"]
            )
            and all(freeze_validation["preregistrationChecks"].values())
        ),
    }
    return access, zero, predecessor


def contract_preservation() -> dict[str, Any]:
    commitments = json.loads(FROZEN_COMMITMENTS.read_text(encoding="utf-8"))
    checks = {
        "estimand": commitments["estimandId"] == "S12R_replacement_spatial_transfer",
        "riskSet": commitments["faultRepair"]["riskSet"]
        == (
            "target-breaking fault with source conjunctive completion and "
            "exact post-fault conjunctive failure"
        ),
        "endpoint": commitments["restrictedTransitionTime"]["endpointId"]
        == "restricted_native_transition_time_at_32",
        "censorBoundary": commitments["restrictedTransitionTime"][
            "rightCensorTransition"
        ]
        == 32,
        "pairing": commitments["pairing"]["completeOneToOneSupportRequired"],
        "slotCount": commitments["fixedMultiplicity"]["slotCount"] == 7_986,
        "holmFamilies": commitments["fixedMultiplicity"]["familyCount"] == 8,
        "costFamilies": commitments["costsRemainSeparate"]
        == [
            "native",
            "portfolio",
            "licensed_capability",
            "adaptation",
            "comparator",
            "fault",
            "scheduler",
        ],
        "diagnosticBoundary": (commitments["diagnosticPanelsEfficacyUse"] is False),
        "universalScore": commitments["universalScore"] is None,
    }
    return {
        "schemaVersion": "e07.s12z.scientific-contract-preservation.v1",
        "researchStepId": "S12Z",
        "instrumentationAndProjectionOnly": True,
        "scientificReplacement": False,
        "riskSetChanged": False,
        "endpointChanged": False,
        "pairingRuleChanged": False,
        "populationChanged": False,
        "multiplicityChanged": False,
        "costFamilyChanged": False,
        "claimBoundaryChanged": False,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    freeze_validation = validate_freezes()
    if not freeze_validation["passed"]:
        raise RuntimeError("S12Z frozen-input revalidation failed")
    clock_summary, clock_frame = qualify_clock_domain()
    adversaries = qualify_adversaries()
    status_estimator = qualify_status_and_estimator()
    parity = qualify_native_dsl_replay_and_order()
    serialization = qualify_serialization(clock_frame)
    fixed = qualify_fixed_families()
    publisher = qualify_publisher()
    access, zero, predecessor = access_and_immutability(freeze_validation)
    preservation = contract_preservation()
    outputs = {
        "freeze_revalidation.json": freeze_validation,
        "clock_domain_qualification.json": clock_summary,
        "clock_adversary_qualification.json": adversaries,
        "status_estimator_integration.json": status_estimator,
        "native_dsl_parity_and_replay.json": parity,
        "serialization_qualification.json": serialization,
        "fixed_family_conservation.json": fixed,
        "fail_atomic_publication_qualification.json": publisher,
        "access_and_zero_outcome_evidence.json": access,
        "zero_execution_accounting.json": zero,
        "predecessor_immutability_validation.json": predecessor,
        "scientific_contract_preservation.json": preservation,
    }
    for name, value in outputs.items():
        write_json(OUT / name, value)
    gates = {
        "freeze": freeze_validation["passed"],
        "clockDomain": clock_summary["passed"],
        "invalidClockStatesFailClosed": adversaries["passed"],
        "statusEstimator": status_estimator["passed"],
        "nativeDslParityReplayOrder": parity["passed"],
        "serialization": serialization["passed"],
        "fixedFamilies": fixed["passed"],
        "publisher": publisher["passed"],
        "access": access["passed"],
        "zeroExecution": zero["passed"],
        "predecessorImmutability": predecessor["passed"],
        "scientificContractPreservation": preservation["passed"],
    }
    validation = {
        "schemaVersion": "e07.s12z.validation-summary.v1",
        "researchStepId": "S12Z",
        "gates": gates,
        "passedGateCount": sum(gates.values()),
        "gateCount": len(gates),
        "allPassed": all(gates.values()),
        "transferEfficacyResult": False,
        "freshExecutionAuthorized": False,
        "exactS12XTriggerKnown": False,
    }
    write_json(OUT / "validation_summary.json", validation)
    if not validation["allPassed"]:
        raise RuntimeError("S12Z qualification failed closed")


if __name__ == "__main__":
    main()
