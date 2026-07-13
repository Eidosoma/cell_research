"""Loss-explicit adapters for the S05 reference and S04 historical records."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping

from reference_simulator.engine import evaluate_terminal, initial_state
from reference_simulator.model import LEDGER_FIELDS, RunResult, Scenario, state_hash

from .model import SourceArtifact, TraceBundle, source_artifact
from .schema import (
    ADAPTER_VERSION,
    EVENT_SCHEMA_URN,
    EVENT_SCHEMA_VERSION,
    TRACKED_FIELDS,
    assign_event_id,
    availability_record,
    observable_state_hash,
    sha256_json,
)


REFERENCE_PROFILE = "R-clean-room-reference-E01-v1"
HISTORICAL_PROFILE = "C-frozen-public-commit"
HISTORICAL_COMMIT = "1fd2bd5921c1f6b423a71f691d5189106a8a1020"


def _empty_availability(source: str, reason: str) -> dict[str, dict[str, Any]]:
    return {
        path: availability_record(
            "unavailable", source=source, reason_code=reason,
            note="The source record does not expose this field."
        )
        for path in TRACKED_FIELDS
    }


def _mark(
    availability: dict[str, dict[str, Any]],
    path: str,
    status: str,
    source: str,
    *,
    method: str | None = None,
    reason: str | None = None,
    note: str | None = None,
) -> None:
    availability[path] = availability_record(
        status, source=source, method=method, reason_code=reason, note=note
    )


def _source_artifacts(path: Path | None, classification: str) -> tuple[SourceArtifact, ...]:
    return (source_artifact(path.resolve(), classification),) if path else ()


def _base_event(
    *,
    backend_id: str,
    profile: str,
    evidence_layer: str,
    source_commit: str | None,
    run_id: str,
    scenario_id: str,
    event_index: int,
    sequence_basis: str,
    batch_id: str,
    batch_width: int,
    batch_ordinal: int,
) -> dict[str, Any]:
    return {
        "schemaVersion": EVENT_SCHEMA_VERSION,
        "schemaUri": EVENT_SCHEMA_URN,
        "recordType": "event",
        "backend": {
            "backendId": backend_id,
            "profile": profile,
            "evidenceLayer": evidence_layer,
            "sourceCommit": source_commit,
            "publicationSnapshotClaimed": False,
            "adapterVersion": ADAPTER_VERSION,
        },
        "run": {
            "runId": run_id,
            "scenarioId": scenario_id,
            "eventIndex": event_index,
            "sequenceBasis": sequence_basis,
            "batch": {
                "batchId": batch_id,
                "width": batch_width,
                "ordinal": batch_ordinal,
            },
        },
    }


def _reference_dict(result: RunResult | Mapping[str, Any]) -> dict[str, Any]:
    return result.to_dict() if isinstance(result, RunResult) else deepcopy(dict(result))


def adapt_reference_result(
    result: RunResult | Mapping[str, Any], *, source_path: Path | None = None
) -> TraceBundle:
    """Adapt a retained S05 full trace and independently replay every state hash."""
    raw = _reference_dict(result)
    if raw.get("schema") != "E01.reference-result.v1":
        raise ValueError("reference adapter requires E01.reference-result.v1")
    scenario = Scenario.from_dict(raw["scenario"])
    events = list(raw["events"])
    if raw["summary"].get("traceMode") != "full":
        raise ValueError("reference adapter requires trace_mode='full'")
    if len(events) != raw["summary"].get("retainedEventCount"):
        raise ValueError("reference retained event count mismatch")

    run_id = "reference:" + sha256_json(raw)
    state = initial_state(scenario)
    state.terminal = evaluate_terminal(scenario, state)
    computed_initial_hash = state_hash(scenario.scenario_id, state)
    if computed_initial_hash != raw["initialStateHash"]:
        raise ValueError("reference initial state hash mismatch")
    if events and state.terminal is not None:
        raise ValueError("reference trace contains events after an initially terminal state")
    if not events:
        if state.to_dict() != raw["finalState"]:
            raise ValueError("zero-event reference final state mismatch")
        if computed_initial_hash != raw["finalStateHash"]:
            raise ValueError("zero-event reference final hash mismatch")
        return TraceBundle(
            events=(), run_id=run_id, scenario_id=scenario.scenario_id,
            backend="reference", sequence_basis="activation",
            source_artifacts=_source_artifacts(source_path, "S05_pre_S06_reference_result"),
            source_run_summary={
                "schema": raw["schema"], "summary": raw["summary"],
                "initialStateHash": raw["initialStateHash"],
                "finalStateHash": raw["finalStateHash"],
            },
            stop_reason=raw["summary"].get("stopReason"),
        )

    adapted: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(events):
        first = events[cursor]
        width = int(first["batchWidth"])
        ordinal = int(first["batchOrdinal"])
        if ordinal != 0:
            raise ValueError(f"reference batch starts at nonzero ordinal: {ordinal}")
        group = events[cursor : cursor + width]
        if len(group) != width or [item["batchOrdinal"] for item in group] != list(range(width)):
            raise ValueError("reference batch width/ordinal structure is invalid")
        expected_indices = list(range(int(first["eventIndex"]), int(first["eventIndex"]) + width))
        if [item["eventIndex"] for item in group] != expected_indices:
            raise ValueError("reference event indices are not contiguous within batch")

        pre_state = state.clone()
        native_pre = state_hash(scenario.scenario_id, state)
        if any(item["preStateHash"] != native_pre for item in group):
            raise ValueError(f"reference pre-state hash mismatch at event {first['eventIndex']}")
        pre_occupancy = list(state.occupancy)
        pre_values = [scenario.cell_map[cell_id].value for cell_id in pre_occupancy]

        # Reconstruct the S05 atomic commit from the preserved proposal/decision records.
        for item in group:
            proposal = item["proposal"]
            if item["decision"] == "accepted" and proposal["kind"] == "Swap":
                actor_pos = int(proposal["actorPos"])
                target_pos = int(proposal["targetPos"])
                state.occupancy[actor_pos] = pre_occupancy[target_pos]
                state.occupancy[target_pos] = pre_occupancy[actor_pos]
            elif item["decision"] == "accepted" and proposal["kind"] == "MemoryUpdate":
                state.selection_cursors[proposal["actorId"]] = int(proposal["newCursor"])
            for key in LEDGER_FIELDS:
                state.ledger[key] += int(item["ledgerDelta"][key])
            for draw in item["randomAddressesAndDraws"]:
                stream = draw["stream"]
                state.stream_counters[stream] = state.stream_counters.get(stream, 0) + 1
        state.activation_count += width
        stop_values = {item["stopReasonIfAny"] for item in group}
        if len(stop_values) != 1:
            raise ValueError("reference batch has inconsistent stop metadata")
        state.terminal = next(iter(stop_values))
        native_post = state_hash(scenario.scenario_id, state)
        if any(item["postStateHash"] != native_post for item in group):
            raise ValueError(f"reference post-state hash mismatch at event {first['eventIndex']}")
        post_occupancy = list(state.occupancy)
        post_values = [scenario.cell_map[cell_id].value for cell_id in post_occupancy]

        for item in group:
            proposal = item["proposal"]
            event_index = int(item["eventIndex"])
            actor_id = item["actorId"]
            actor_pos_raw = int(proposal["actorPos"])
            actor_is_cell = actor_id in scenario.cell_map and actor_pos_raw >= 0
            target_pos = proposal["targetPos"]
            target_id = pre_occupancy[target_pos] if target_pos is not None else None
            accepted = item["decision"] == "accepted"
            is_swap = proposal["kind"] == "Swap"
            actor_post = int(target_pos) if accepted and is_swap else (actor_pos_raw if actor_is_cell else None)
            target_post = actor_pos_raw if accepted and is_swap else target_pos
            draws = [
                {
                    "stream": draw["stream"],
                    "eventIndex": int(draw["eventIndex"]),
                    "drawIndex": int(draw["drawIndex"]),
                    "uint64": str(draw["value"]),
                }
                for draw in item["randomAddressesAndDraws"]
            ]
            availability = _empty_availability("S05 pre-S06 event", "not_serialized_pre_S06")
            for path in ("run.scenarioId", "run.eventIndex", "actor.id", "actor.algotype",
                         "actor.direction", "observation.readCount",
                         "observation.valueComparisonCount", "proposal.kind", "proposal.reason",
                         "proposal.ordinal", "decision.status"):
                _mark(availability, path, "observed", "S05 pre-S06 event")
            _mark(availability, "decision.accepted", "derived_exact", "S05 pre-S06 event",
                  method="decision == accepted")
            _mark(availability, "actor.prePosition", "observed" if actor_is_cell else "not_applicable",
                  "S05 proposal", reason=None if actor_is_cell else "controller_has_no_cell_position")
            _mark(availability, "actor.postPosition", "derived_exact" if actor_is_cell else "not_applicable",
                  "S05 proposal plus atomic decision", method="accepted swap endpoint or unchanged position" if actor_is_cell else None,
                  reason=None if actor_is_cell else "controller_has_no_cell_position")
            for path in ("target.id", "target.prePosition", "target.postPosition", "proposal.targetPosition"):
                _mark(availability, path, "derived_exact" if target_pos is not None else "not_applicable",
                      "S05 proposal plus pre-batch occupancy", method="target position lookup" if target_pos is not None else None,
                      reason=None if target_pos is not None else "proposal_has_no_target")
            if proposal["newCursor"] is None:
                _mark(availability, "proposal.newCursor", "not_applicable", "S05 proposal",
                      reason="proposal_has_no_cursor_update")
            else:
                _mark(availability, "proposal.newCursor", "observed", "S05 proposal")
            if width > 1:
                _mark(availability, "proposal.priorityUint64", "observed", "S05 proposal")
            else:
                _mark(availability, "proposal.priorityUint64", "not_applicable", "S05 proposal",
                      reason="serial_scheduler_has_no_conflict_priority")
            if draws:
                for path in ("randomness.streamIdentifiers", "randomness.draws"):
                    _mark(availability, path, "observed", "S05 pre-S06 event")
                _mark(availability, "randomness.addressingProfile", "derived_exact", "S05 scenario",
                      method="scenario rngProfile")
            else:
                for path in ("randomness.streamIdentifiers", "randomness.addressingProfile", "randomness.draws"):
                    _mark(availability, path, "not_applicable", "S05 traditional controller",
                          reason="controller_is_deterministic_without_random_draws")
            _mark(availability, "randomness.seedOverride", "not_applicable", "S05 scenario",
                  reason="scenario seed is an input, not an adapter override")
            _mark(availability, "observation.details", "unavailable", "S05 pre-S06 event",
                  reason="logical_reads_not_serialized", note="Counts are exact; individual reads were not emitted.")
            _mark(availability, "observation.logicalProjection", "unavailable", "S05 pre-S06 event",
                  reason="logical_projection_not_serialized")

            actor_fault = scenario.cell_map[actor_id].fault.value if actor_is_cell else None
            target_fault = scenario.cell_map[target_id].fault.value if target_id else None
            _mark(availability, "faultInteraction.actorMode", "derived_exact" if actor_is_cell else "not_applicable",
                  "S05 scenario", method="cell identity lookup" if actor_is_cell else None,
                  reason=None if actor_is_cell else "controller_has_no_fault_mode")
            _mark(availability, "faultInteraction.targetMode", "derived_exact" if target_id else "not_applicable",
                  "S05 scenario", method="target identity lookup" if target_id else None,
                  reason=None if target_id else "proposal_has_no_target")
            _mark(availability, "faultInteraction.kind", "derived_exact", "S05 scenario/proposal/decision",
                  method="fault-mode and decision classification")
            _mark(availability, "displacement.count", "derived_exact", "S05 decision",
                  method="accepted swap displaces two identities; other events displace zero")
            _mark(availability, "displacement.positions", "derived_exact" if accepted and is_swap else "not_applicable",
                  "S05 proposal/decision", method="swap endpoints" if accepted and is_swap else None,
                  reason=None if accepted and is_swap else "event_did_not_displace_cells")
            for key in LEDGER_FIELDS:
                _mark(availability, f"costDelta.{key}", "observed", "S05 ledgerDelta")
            for path in ("state.nativePreHash", "state.nativePostHash"):
                _mark(availability, path, "observed", "S05 pre-S06 event")
            _mark(availability, "state.nativeHashScope", "derived_exact", "S05 model.state_hash",
                  method="scenario ID plus full RunState")
            for path in ("state.observablePreHash", "state.observablePostHash"):
                _mark(availability, path, "derived_exact", "S05 scenario and replayed occupancy",
                      method="canonical ordered_values_v1")
            if item["stopReasonIfAny"] is None:
                _mark(availability, "stop.reason", "not_applicable", "S05 event",
                      reason="event_is_nonterminal")
            else:
                _mark(availability, "stop.reason", "observed", "S05 event")
            for path in ("stop.terminal", "stop.censored"):
                _mark(availability, path, "derived_exact", "S05 stop metadata",
                      method="terminal presence and event_budget classification")

            if not actor_is_cell:
                actor_pre = actor_post = None
            else:
                actor_pre = actor_pos_raw
            fault_kind = "none"
            if actor_fault in {"passive", "stuck"}:
                fault_kind = f"actor_{actor_fault}"
            elif target_fault in {"passive", "stuck"}:
                fault_kind = f"target_{target_fault}_" + ("displaced" if accepted else "encountered")
            event = _base_event(
                backend_id="reference", profile=REFERENCE_PROFILE,
                evidence_layer="clean_room_reference", source_commit=None,
                run_id=run_id, scenario_id=scenario.scenario_id,
                event_index=event_index, sequence_basis="activation",
                batch_id=f"activation:{int(first['eventIndex'])}", batch_width=width,
                batch_ordinal=int(item["batchOrdinal"]),
            )
            event.update({
                "randomness": {
                    "streamIdentifiers": sorted({draw["stream"] for draw in draws}) if draws else None,
                    "addressingProfile": scenario.rng_profile if draws else None,
                    "draws": draws if draws else None,
                    "seedOverride": None,
                },
                "actor": {
                    "id": actor_id, "algotype": item["actorAlgotype"],
                    "direction": item["actorDirection"], "prePosition": actor_pre,
                    "postPosition": actor_post,
                },
                "target": {"id": target_id, "prePosition": target_pos, "postPosition": target_post},
                "observation": {
                    "readCount": item["observation"]["reads"],
                    "valueComparisonCount": item["observation"]["valueComparisons"],
                    "details": None, "logicalProjection": None,
                },
                "proposal": {
                    "kind": proposal["kind"], "reason": proposal["reason"],
                    "targetPosition": target_pos, "newCursor": proposal["newCursor"],
                    "priorityUint64": str(proposal["priority"]) if width > 1 else None,
                    "ordinal": int(proposal["ordinal"]),
                },
                "decision": {"status": item["decision"], "accepted": accepted},
                "faultInteraction": {
                    "actorMode": actor_fault, "targetMode": target_fault, "kind": fault_kind,
                },
                "displacement": {
                    "count": 2 if accepted and is_swap else 0,
                    "positions": ({
                        "changed": sorted([actor_pos_raw, int(target_pos)]),
                        "actorFrom": actor_pos_raw, "actorTo": int(target_pos),
                        "targetFrom": int(target_pos), "targetTo": actor_pos_raw,
                    } if accepted and is_swap else None),
                },
                "costDelta": dict(item["ledgerDelta"]),
                "state": {
                    "nativePreHash": "sha256:" + native_pre,
                    "nativePostHash": "sha256:" + native_post,
                    "nativeHashScope": "S05 scenario_id + full RunState v1",
                    "observablePreHash": observable_state_hash(pre_values),
                    "observablePostHash": observable_state_hash(post_values),
                    "observableHashScope": "ordered_values_v1",
                },
                "stop": {
                    "terminal": item["stopReasonIfAny"] is not None,
                    "reason": item["stopReasonIfAny"],
                    "censored": item["stopReasonIfAny"] == "event_budget",
                },
                "fieldAvailability": availability,
                "backendPayload": {"sourceSchemaVersion": item["schemaVersion"], "sourceEvent": item},
            })
            adapted.append(assign_event_id(event))
        cursor += width

    if state.to_dict() != raw["finalState"]:
        raise ValueError("reference reconstructed final state mismatch")
    if state_hash(scenario.scenario_id, state) != raw["finalStateHash"]:
        raise ValueError("reference reconstructed final hash mismatch")
    return TraceBundle(
        events=tuple(adapted), run_id=run_id, scenario_id=scenario.scenario_id,
        backend="reference", sequence_basis="activation",
        source_artifacts=_source_artifacts(source_path, "S05_pre_S06_reference_result"),
        source_run_summary={
            "schema": raw["schema"], "summary": raw["summary"],
            "initialStateHash": raw["initialStateHash"],
            "finalStateHash": raw["finalStateHash"],
            "eventDigest": raw["eventDigest"],
            "adapterReplayValidated": True,
        },
        stop_reason=raw["summary"].get("stopReason"),
    )


def adapt_historical_run(
    run: Mapping[str, Any], *, source_path: Path | None = None
) -> TraceBundle:
    """Adapt S04 StatusProbe output without manufacturing activation-level parity."""
    raw = deepcopy(dict(run))
    if raw.get("backendProfile") != HISTORICAL_PROFILE:
        raise ValueError("historical adapter requires C-frozen-public-commit")
    if raw.get("publicationSnapshotClaimed") is not False:
        raise ValueError("historical input must explicitly deny publication-snapshot identity")
    if raw.get("publicCommit") != HISTORICAL_COMMIT:
        raise ValueError("unexpected historical public commit")
    steps = raw.get("sortingSteps", [])
    cell_steps = raw.get("cellTypes", [])
    if len(cell_steps) not in (0, len(steps)):
        raise ValueError("historical sortingSteps/cellTypes length mismatch")
    if raw.get("metrics", {}).get("recordedSortingSteps") != len(steps):
        raise ValueError("historical recordedSortingSteps metric mismatch")
    if raw.get("metrics", {}).get("recordedCellTypeSteps") != len(cell_steps):
        raise ValueError("historical recordedCellTypeSteps metric mismatch")
    if raw.get("metrics", {}).get("swapCount") != len(steps):
        raise ValueError("historical swapCount does not match successful-swap records")
    expected_trace_hash = sha256_json(steps)
    if raw.get("traceSha256") != expected_trace_hash:
        raise ValueError("historical traceSha256 mismatch")
    previous = list(raw["inputValues"])
    if Counter(previous) != Counter(raw["finalValues"]):
        raise ValueError("historical input/final value multiset mismatch")

    scenario_material = {
        "backend": HISTORICAL_PROFILE, "publicCommit": raw["publicCommit"],
        "policy": raw["policy"], "inputValues": raw["inputValues"],
        "frozenIndices": raw.get("frozenIndicesAfterWithReplacementDraws", []),
        "stopMode": raw["stopMode"],
    }
    scenario_id = "h1:" + sha256_json(scenario_material)
    run_id = "historical:" + sha256_json(raw)
    adapted: list[dict[str, Any]] = []
    for ordinal, step in enumerate(steps):
        post = list(step)
        if Counter(previous) != Counter(post):
            raise ValueError(f"historical step {ordinal} changes the value multiset")
        changed = [index for index, (a, b) in enumerate(zip(previous, post)) if a != b]
        exact_positions = (
            len(changed) == 2
            and previous[changed[0]] == post[changed[1]]
            and previous[changed[1]] == post[changed[0]]
        )
        observationally_invisible = len(changed) == 0
        if not exact_positions and not observationally_invisible:
            raise ValueError(f"historical step {ordinal} is not a single observable swap")
        terminal = ordinal == len(steps) - 1
        availability = _empty_availability("S04 StatusProbe", "not_recorded_by_historical_probe")
        for path in ("run.scenarioId", "run.eventIndex"):
            _mark(availability, path, "derived_exact", "S04 run plus recorded swap ordinal",
                  method="canonical historical scenario/run material")
        _mark(availability, "randomness.seedOverride",
              "observed" if raw.get("seedOverride") is not None else "not_applicable",
              "S04 adapter run", reason=None if raw.get("seedOverride") is not None else "no_seed_override")
        for path in ("randomness.streamIdentifiers", "randomness.addressingProfile", "randomness.draws"):
            _mark(availability, path, "unavailable", "S04 StatusProbe",
                  reason="per_event_rng_not_recorded")
        for path in ("actor.id", "actor.prePosition", "actor.postPosition", "target.id",
                     "target.prePosition", "target.postPosition", "proposal.targetPosition"):
            _mark(availability, path, "unavailable", "S04 StatusProbe",
                  reason="cell_identity_and_swap_orientation_not_recorded")
        _mark(availability, "actor.algotype", "derived_exact", "S04 homogeneous policy run",
              method="run-level policy applies to all historical cell threads")
        _mark(availability, "actor.direction", "derived_exact", "S04 driver contract",
              method="ascending historical drivers used by validated adapter")
        for path in ("observation.readCount", "observation.valueComparisonCount",
                     "observation.details", "observation.logicalProjection"):
            _mark(availability, path, "unavailable", "S04 StatusProbe",
                  reason="activation_observations_not_recorded")
        _mark(availability, "proposal.kind", "derived_exact", "frozen StatusProbe implementation",
              method="probe appends sortingSteps only after successful swaps")
        _mark(availability, "proposal.reason", "unavailable", "S04 StatusProbe",
              reason="proposal_reason_not_recorded")
        for path in ("proposal.newCursor", "proposal.priorityUint64", "proposal.ordinal"):
            _mark(availability, path, "unavailable", "S04 StatusProbe",
                  reason="proposal_metadata_not_recorded")
        for path in ("decision.status", "decision.accepted"):
            _mark(availability, path, "derived_exact", "frozen StatusProbe implementation",
                  method="each record is emitted only after an accepted swap")
        for path in ("faultInteraction.actorMode", "faultInteraction.targetMode", "faultInteraction.kind"):
            _mark(availability, path, "unavailable", "S04 StatusProbe",
                  reason="swap_participant_identity_not_recorded")
        _mark(availability, "displacement.count", "derived_exact", "frozen successful swap record",
              method="two cell identities are exchanged per recorded swap")
        if exact_positions:
            _mark(availability, "displacement.positions", "derived_exact", "adjacent probe snapshots",
                  method="two unequal values exchange positions")
        else:
            _mark(availability, "displacement.positions", "derived_ambiguous", "adjacent probe snapshots",
                  reason="equal_value_swap_is_observationally_invisible",
                  note="A successful identity swap occurred, but its endpoints cannot be recovered from values.")
        for key in LEDGER_FIELDS:
            path = f"costDelta.{key}"
            if key in {"acceptedSwaps", "displacedCells"}:
                _mark(availability, path, "derived_exact", "frozen successful swap record",
                      method="one swap and two displaced identities per StatusProbe record")
            else:
                _mark(availability, path, "unavailable", "S04 StatusProbe",
                      reason="activation_level_cost_not_recorded")
        for path in ("state.nativePreHash", "state.nativePostHash", "state.nativeHashScope"):
            _mark(availability, path, "unavailable", "S04 StatusProbe",
                  reason="full_historical_runtime_state_not_recorded")
        for path in ("state.observablePreHash", "state.observablePostHash"):
            _mark(availability, path, "derived_exact", "S04 value snapshots",
                  method="canonical ordered_values_v1")
        _mark(availability, "stop.reason", "derived_exact" if terminal else "not_applicable",
              "S04 run-level stop metadata", method="attach run stop to final recorded swap" if terminal else None,
              reason=None if terminal else "record_is_not_final_recorded_swap")
        for path in ("stop.terminal", "stop.censored"):
            _mark(availability, path, "derived_exact", "S04 trace order and run stop metadata",
                  method="final-record position plus timeout classification")

        event = _base_event(
            backend_id="historical_frozen_public_commit", profile=HISTORICAL_PROFILE,
            evidence_layer="frozen_public_commit", source_commit=raw["publicCommit"],
            run_id=run_id, scenario_id=scenario_id, event_index=ordinal,
            sequence_basis="recorded_swap", batch_id=f"recorded_swap:{ordinal}",
            batch_width=1, batch_ordinal=0,
        )
        event.update({
            "randomness": {
                "streamIdentifiers": None, "addressingProfile": None, "draws": None,
                "seedOverride": str(raw["seedOverride"]) if raw.get("seedOverride") is not None else None,
            },
            "actor": {
                "id": None, "algotype": raw["policy"].capitalize(), "direction": "ascending",
                "prePosition": None, "postPosition": None,
            },
            "target": {"id": None, "prePosition": None, "postPosition": None},
            "observation": {
                "readCount": None, "valueComparisonCount": None,
                "details": None, "logicalProjection": None,
            },
            "proposal": {
                "kind": "Swap", "reason": None, "targetPosition": None,
                "newCursor": None, "priorityUint64": None, "ordinal": None,
            },
            "decision": {"status": "accepted", "accepted": True},
            "faultInteraction": {"actorMode": None, "targetMode": None, "kind": None},
            "displacement": {
                "count": 2,
                "positions": ({
                    "changed": changed, "actorFrom": None, "actorTo": None,
                    "targetFrom": None, "targetTo": None,
                } if exact_positions else None),
            },
            "costDelta": {
                "activations": None, "observationReads": None, "valueComparisons": None,
                "proposals": None, "noOps": None, "rejections": None,
                "memoryUpdates": None, "acceptedSwaps": 1, "displacedCells": 2,
                "conflictLosses": None,
            },
            "state": {
                "nativePreHash": None, "nativePostHash": None, "nativeHashScope": None,
                "observablePreHash": observable_state_hash(previous),
                "observablePostHash": observable_state_hash(post),
                "observableHashScope": "ordered_values_v1",
            },
            "stop": {
                "terminal": terminal,
                "reason": raw["stopReason"] if terminal else None,
                "censored": bool(raw.get("timedOut")) if terminal else False,
            },
            "fieldAvailability": availability,
            "backendPayload": {
                "sourceRecordKind": "StatusProbe successful-swap snapshot",
                "sourceSwapOrdinal": ordinal,
                "sourceSortingStep": post,
                "sourceCellTypeStep": cell_steps[ordinal] if cell_steps else None,
                "sourceTraceSha256": raw["traceSha256"],
            },
        })
        adapted.append(assign_event_id(event))
        previous = post
    if previous != list(raw["finalValues"]):
        raise ValueError("historical final snapshot does not equal finalValues")
    return TraceBundle(
        events=tuple(adapted), run_id=run_id, scenario_id=scenario_id,
        backend="historical_frozen_public_commit", sequence_basis="recorded_swap",
        source_artifacts=_source_artifacts(source_path, "S04_validated_historical_run_json"),
        source_run_summary={
            "backendProfile": raw["backendProfile"], "policy": raw["policy"],
            "publicCommit": raw["publicCommit"], "publicTree": raw.get("publicTree"),
            "publicationSnapshotClaimed": False, "inputValues": raw["inputValues"],
            "finalValues": raw["finalValues"], "metrics": raw["metrics"],
            "stopMode": raw["stopMode"], "stopReason": raw["stopReason"],
            "timedOut": raw.get("timedOut"), "threadShutdownClean": raw.get("threadShutdownClean"),
            "observabilityBoundary": "successful-swap value snapshots; not activation events",
        },
        stop_reason=raw["stopReason"],
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
