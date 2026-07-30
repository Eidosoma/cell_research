"""Authenticated elapsed-transition clock for prospective spatial episodes.

The E06 movement state already owns the scientific clock: the initial state is
elapsed transition zero and every committed native batch increments that value
by one.  This module makes that authority explicit and tamper-evident without
changing policy observations, movement legality, scheduling, stopping, costs,
or endpoint definitions.

Raw loop/observation labels remain provenance only.  They are deliberately
bound into each audit record but never converted into scientific time.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .movements import MovementState, movement_state_sha256

ELAPSED_CLOCK_SCHEMA_VERSION = "e07.s12z.authenticated-elapsed-clock-record.v1"
ELAPSED_CLOCK_SUMMARY_VERSION = "e07.s12z.authenticated-elapsed-clock-summary.v1"
ELAPSED_CLOCK_PROJECTION_VERSION = (
    "e07.s12z.authenticated-first-completion-projection.v1"
)
_RECORD_DOMAIN = "E07/S12Z/authenticated-elapsed-clock-record/v1"
_GENESIS_DOMAIN = "E07/S12Z/authenticated-elapsed-clock-genesis/v1"
_SUMMARY_DOMAIN = "E07/S12Z/authenticated-elapsed-clock-summary/v1"
_PROJECTION_DOMAIN = "E07/S12Z/authenticated-first-completion/v1"


class ElapsedClockValidationError(ValueError):
    """Elapsed-clock metadata is absent, contradictory, or unauthenticated."""


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ElapsedClockValidationError(
            "elapsed-clock metadata is not canonical JSON"
        ) from exc


def _domain_sha256(domain: str, payload: Any) -> str:
    return hashlib.sha256(
        domain.encode("ascii") + b"\0" + _canonical_json_bytes(payload)
    ).hexdigest()


def _strict_int(
    value: Any,
    *,
    field: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:
        if isinstance(value, float) and not math.isfinite(value):
            raise ElapsedClockValidationError(f"{field}: non-finite value")
        raise ElapsedClockValidationError(f"{field}: exact integer required")
    if minimum is not None and value < minimum:
        raise ElapsedClockValidationError(f"{field}: below declared domain")
    if maximum is not None and value > maximum:
        raise ElapsedClockValidationError(f"{field}: above declared domain")
    return value


def _raw_label(value: Any) -> int | str | None:
    if value is None or isinstance(value, str):
        return value
    if type(value) is int:
        return value
    raise ElapsedClockValidationError(
        "rawObservationLabel: only integer, string, or null provenance is allowed"
    )


def genesis_commitment(*, scenario_id: str, horizon_transitions: int) -> str:
    if not scenario_id:
        raise ElapsedClockValidationError("scenarioId: nonempty string required")
    horizon = _strict_int(
        horizon_transitions,
        field="horizonTransitions",
        minimum=0,
    )
    return _domain_sha256(
        _GENESIS_DOMAIN,
        {
            "schemaVersion": ELAPSED_CLOCK_SCHEMA_VERSION,
            "scenarioId": scenario_id,
            "horizonTransitions": horizon,
        },
    )


@dataclass(frozen=True)
class AuthenticatedElapsedClockRecord:
    """One engine-issued clock observation."""

    scenario_id: str
    horizon_transitions: int
    sequence_ordinal: int
    elapsed_transition: int
    phase: str
    raw_observation_label: int | str | None
    movement_state_transition_index: int
    movement_state_sha256: str
    previous_commitment_sha256: str
    record_commitment_sha256: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schemaVersion": ELAPSED_CLOCK_SCHEMA_VERSION,
            "scenarioId": self.scenario_id,
            "horizonTransitions": self.horizon_transitions,
            "sequenceOrdinal": self.sequence_ordinal,
            "authenticatedElapsedTransition": self.elapsed_transition,
            "phase": self.phase,
            "rawObservationLabel": self.raw_observation_label,
            "movementStateTransitionIndex": self.movement_state_transition_index,
            "movementStateSha256": self.movement_state_sha256,
            "previousCommitmentSha256": self.previous_commitment_sha256,
            "recordCommitmentSha256": self.record_commitment_sha256,
        }


_RECORD_KEYS = frozenset(
    {
        "schemaVersion",
        "scenarioId",
        "horizonTransitions",
        "sequenceOrdinal",
        "authenticatedElapsedTransition",
        "phase",
        "rawObservationLabel",
        "movementStateTransitionIndex",
        "movementStateSha256",
        "previousCommitmentSha256",
        "recordCommitmentSha256",
    }
)


def _record_body(
    *,
    scenario_id: str,
    horizon_transitions: int,
    sequence_ordinal: int,
    elapsed_transition: int,
    phase: str,
    raw_observation_label: int | str | None,
    movement_state_transition_index: int,
    movement_state_sha256_value: str,
    previous_commitment_sha256: str,
) -> dict[str, Any]:
    return {
        "schemaVersion": ELAPSED_CLOCK_SCHEMA_VERSION,
        "scenarioId": scenario_id,
        "horizonTransitions": horizon_transitions,
        "sequenceOrdinal": sequence_ordinal,
        "authenticatedElapsedTransition": elapsed_transition,
        "phase": phase,
        "rawObservationLabel": raw_observation_label,
        "movementStateTransitionIndex": movement_state_transition_index,
        "movementStateSha256": movement_state_sha256_value,
        "previousCommitmentSha256": previous_commitment_sha256,
    }


def parse_and_validate_elapsed_clock_record(
    raw: Mapping[str, Any],
    *,
    expected_scenario_id: str | None = None,
    expected_horizon: int | None = None,
    expected_ordinal: int | None = None,
    expected_elapsed: int | None = None,
    expected_previous_commitment: str | None = None,
    state: MovementState | None = None,
) -> AuthenticatedElapsedClockRecord:
    """Parse one exact record and verify every binding and commitment."""

    if not isinstance(raw, Mapping):
        raise ElapsedClockValidationError("elapsed-clock record must be a mapping")
    if set(raw) != _RECORD_KEYS:
        missing = sorted(_RECORD_KEYS - set(raw))
        extra = sorted(set(raw) - _RECORD_KEYS)
        raise ElapsedClockValidationError(
            f"elapsed-clock record schema mismatch; missing={missing}, extra={extra}"
        )
    if raw["schemaVersion"] != ELAPSED_CLOCK_SCHEMA_VERSION:
        raise ElapsedClockValidationError("unknown elapsed-clock schema version")
    scenario_id = raw["scenarioId"]
    if not isinstance(scenario_id, str) or not scenario_id:
        raise ElapsedClockValidationError("scenarioId: nonempty string required")
    horizon = _strict_int(
        raw["horizonTransitions"],
        field="horizonTransitions",
        minimum=0,
    )
    ordinal = _strict_int(
        raw["sequenceOrdinal"],
        field="sequenceOrdinal",
        minimum=0,
        maximum=horizon,
    )
    elapsed = _strict_int(
        raw["authenticatedElapsedTransition"],
        field="authenticatedElapsedTransition",
        minimum=0,
        maximum=horizon,
    )
    state_index = _strict_int(
        raw["movementStateTransitionIndex"],
        field="movementStateTransitionIndex",
        minimum=0,
        maximum=horizon,
    )
    if ordinal != elapsed:
        raise ElapsedClockValidationError(
            "sequence ordinal and authenticated elapsed transition disagree"
        )
    if state_index != elapsed:
        raise ElapsedClockValidationError(
            "movement state and authenticated elapsed transition disagree"
        )
    phase = raw["phase"]
    expected_phase = "initial_state" if elapsed == 0 else "post_transition"
    if phase != expected_phase:
        raise ElapsedClockValidationError(
            f"phase disagrees with elapsed transition {elapsed}"
        )
    label = _raw_label(raw["rawObservationLabel"])
    state_sha256 = raw["movementStateSha256"]
    previous = raw["previousCommitmentSha256"]
    commitment = raw["recordCommitmentSha256"]
    for field, value in (
        ("movementStateSha256", state_sha256),
        ("previousCommitmentSha256", previous),
        ("recordCommitmentSha256", commitment),
    ):
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ElapsedClockValidationError(f"{field}: lowercase SHA-256 required")
    if expected_scenario_id is not None and scenario_id != expected_scenario_id:
        raise ElapsedClockValidationError("scenario identity mismatch")
    if expected_horizon is not None and horizon != expected_horizon:
        raise ElapsedClockValidationError("horizon mismatch")
    if expected_ordinal is not None and ordinal != expected_ordinal:
        raise ElapsedClockValidationError("clock sequence is skipped or repeated")
    if expected_elapsed is not None and elapsed != expected_elapsed:
        raise ElapsedClockValidationError("unexpected authenticated elapsed transition")
    if (
        expected_previous_commitment is not None
        and previous != expected_previous_commitment
    ):
        raise ElapsedClockValidationError("elapsed-clock chain is broken")
    if state is not None:
        if state.transition_index != elapsed:
            raise ElapsedClockValidationError(
                "runtime movement state disagrees with authenticated elapsed transition"
            )
        if movement_state_sha256(state) != state_sha256:
            raise ElapsedClockValidationError(
                "runtime movement-state hash disagrees with clock record"
            )
    body = _record_body(
        scenario_id=scenario_id,
        horizon_transitions=horizon,
        sequence_ordinal=ordinal,
        elapsed_transition=elapsed,
        phase=phase,
        raw_observation_label=label,
        movement_state_transition_index=state_index,
        movement_state_sha256_value=state_sha256,
        previous_commitment_sha256=previous,
    )
    if _domain_sha256(_RECORD_DOMAIN, body) != commitment:
        raise ElapsedClockValidationError("forged elapsed-clock record commitment")
    return AuthenticatedElapsedClockRecord(
        scenario_id=scenario_id,
        horizon_transitions=horizon,
        sequence_ordinal=ordinal,
        elapsed_transition=elapsed,
        phase=phase,
        raw_observation_label=label,
        movement_state_transition_index=state_index,
        movement_state_sha256=state_sha256,
        previous_commitment_sha256=previous,
        record_commitment_sha256=commitment,
    )


class ElapsedClockAuthenticator:
    """Engine-owned issuer for one contiguous elapsed-clock sequence."""

    def __init__(self, *, scenario_id: str, horizon_transitions: int) -> None:
        self.scenario_id = scenario_id
        self.horizon_transitions = _strict_int(
            horizon_transitions,
            field="horizonTransitions",
            minimum=0,
        )
        self._previous = genesis_commitment(
            scenario_id=scenario_id,
            horizon_transitions=self.horizon_transitions,
        )
        self._records: list[AuthenticatedElapsedClockRecord] = []

    @property
    def records(self) -> tuple[AuthenticatedElapsedClockRecord, ...]:
        return tuple(self._records)

    def issue(
        self,
        *,
        elapsed_transition: int,
        raw_observation_label: int | str | None,
        state: MovementState,
    ) -> AuthenticatedElapsedClockRecord:
        expected = len(self._records)
        elapsed = _strict_int(
            elapsed_transition,
            field="authenticatedElapsedTransition",
            minimum=0,
            maximum=self.horizon_transitions,
        )
        if elapsed != expected:
            raise ElapsedClockValidationError(
                "clock issuer requires contiguous elapsed transitions"
            )
        phase = "initial_state" if elapsed == 0 else "post_transition"
        label = _raw_label(raw_observation_label)
        state_sha256 = movement_state_sha256(state)
        body = _record_body(
            scenario_id=self.scenario_id,
            horizon_transitions=self.horizon_transitions,
            sequence_ordinal=expected,
            elapsed_transition=elapsed,
            phase=phase,
            raw_observation_label=label,
            movement_state_transition_index=state.transition_index,
            movement_state_sha256_value=state_sha256,
            previous_commitment_sha256=self._previous,
        )
        commitment = _domain_sha256(_RECORD_DOMAIN, body)
        record = parse_and_validate_elapsed_clock_record(
            {**body, "recordCommitmentSha256": commitment},
            expected_scenario_id=self.scenario_id,
            expected_horizon=self.horizon_transitions,
            expected_ordinal=expected,
            expected_elapsed=elapsed,
            expected_previous_commitment=self._previous,
            state=state,
        )
        self._records.append(record)
        self._previous = record.record_commitment_sha256
        return record

    def finalize(self) -> dict[str, Any]:
        return summarize_elapsed_clock_records(
            [record.to_mapping() for record in self._records],
            scenario_id=self.scenario_id,
            horizon_transitions=self.horizon_transitions,
            require_complete=True,
        )


def summarize_elapsed_clock_records(
    records: Sequence[Mapping[str, Any]],
    *,
    scenario_id: str,
    horizon_transitions: int,
    require_complete: bool,
) -> dict[str, Any]:
    """Validate a chain and return its compact authenticated summary."""

    horizon = _strict_int(
        horizon_transitions,
        field="horizonTransitions",
        minimum=0,
    )
    if not records:
        raise ElapsedClockValidationError("elapsed-clock sequence is absent")
    expected_previous = genesis_commitment(
        scenario_id=scenario_id,
        horizon_transitions=horizon,
    )
    parsed: list[AuthenticatedElapsedClockRecord] = []
    for ordinal, raw in enumerate(records):
        item = parse_and_validate_elapsed_clock_record(
            raw,
            expected_scenario_id=scenario_id,
            expected_horizon=horizon,
            expected_ordinal=ordinal,
            expected_elapsed=ordinal,
            expected_previous_commitment=expected_previous,
        )
        parsed.append(item)
        expected_previous = item.record_commitment_sha256
    if require_complete and len(parsed) != horizon + 1:
        raise ElapsedClockValidationError(
            "elapsed-clock sequence is incomplete at episode finalization"
        )
    body = {
        "schemaVersion": ELAPSED_CLOCK_SUMMARY_VERSION,
        "scenarioId": scenario_id,
        "horizonTransitions": horizon,
        "recordCount": len(parsed),
        "initialElapsedTransition": parsed[0].elapsed_transition,
        "finalElapsedTransition": parsed[-1].elapsed_transition,
        "genesisCommitmentSha256": genesis_commitment(
            scenario_id=scenario_id,
            horizon_transitions=horizon,
        ),
        "initialRecordCommitmentSha256": parsed[0].record_commitment_sha256,
        "finalRecordCommitmentSha256": parsed[-1].record_commitment_sha256,
        "recordCommitmentsByElapsed": [
            item.record_commitment_sha256 for item in parsed
        ],
        "recordCommitmentsSha256": _domain_sha256(
            _SUMMARY_DOMAIN,
            [item.record_commitment_sha256 for item in parsed],
        ),
        "complete": bool(len(parsed) == horizon + 1),
        "rawObservationLabelsScientificTime": False,
    }
    return {
        **body,
        "summaryCommitmentSha256": _domain_sha256(_SUMMARY_DOMAIN, body),
    }


_SUMMARY_KEYS = frozenset(
    {
        "schemaVersion",
        "scenarioId",
        "horizonTransitions",
        "recordCount",
        "initialElapsedTransition",
        "finalElapsedTransition",
        "genesisCommitmentSha256",
        "initialRecordCommitmentSha256",
        "finalRecordCommitmentSha256",
        "recordCommitmentsByElapsed",
        "recordCommitmentsSha256",
        "complete",
        "rawObservationLabelsScientificTime",
        "summaryCommitmentSha256",
    }
)


def validate_elapsed_clock_summary(
    raw: Mapping[str, Any],
    *,
    expected_scenario_id: str | None = None,
    expected_horizon: int = 32,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != _SUMMARY_KEYS:
        raise ElapsedClockValidationError(
            "authenticated elapsed-clock summary is absent or ambiguous"
        )
    if raw["schemaVersion"] != ELAPSED_CLOCK_SUMMARY_VERSION:
        raise ElapsedClockValidationError("unknown elapsed-clock summary schema")
    scenario_id = raw["scenarioId"]
    if not isinstance(scenario_id, str) or not scenario_id:
        raise ElapsedClockValidationError("summary scenarioId is invalid")
    if expected_scenario_id is not None and scenario_id != expected_scenario_id:
        raise ElapsedClockValidationError("summary scenario identity mismatch")
    horizon = _strict_int(
        raw["horizonTransitions"],
        field="horizonTransitions",
        minimum=0,
    )
    if horizon != expected_horizon:
        raise ElapsedClockValidationError("summary horizon mismatch")
    count = _strict_int(raw["recordCount"], field="recordCount", minimum=1)
    initial = _strict_int(
        raw["initialElapsedTransition"],
        field="initialElapsedTransition",
        minimum=0,
        maximum=horizon,
    )
    final = _strict_int(
        raw["finalElapsedTransition"],
        field="finalElapsedTransition",
        minimum=0,
        maximum=horizon,
    )
    if initial != 0 or final != horizon or count != horizon + 1:
        raise ElapsedClockValidationError("elapsed-clock summary is incomplete")
    if raw["complete"] is not True:
        raise ElapsedClockValidationError("elapsed-clock summary is not complete")
    if raw["rawObservationLabelsScientificTime"] is not False:
        raise ElapsedClockValidationError(
            "raw observation labels cannot control scientific time"
        )
    for field in (
        "genesisCommitmentSha256",
        "initialRecordCommitmentSha256",
        "finalRecordCommitmentSha256",
        "recordCommitmentsSha256",
        "summaryCommitmentSha256",
    ):
        value = raw[field]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ElapsedClockValidationError(f"{field}: lowercase SHA-256 required")
    commitments = raw["recordCommitmentsByElapsed"]
    if not isinstance(commitments, list) or len(commitments) != count:
        raise ElapsedClockValidationError(
            "per-elapsed-transition commitments are absent or incomplete"
        )
    for commitment in commitments:
        if (
            not isinstance(commitment, str)
            or len(commitment) != 64
            or any(character not in "0123456789abcdef" for character in commitment)
        ):
            raise ElapsedClockValidationError(
                "per-elapsed-transition commitment is invalid"
            )
    if (
        commitments[0] != raw["initialRecordCommitmentSha256"]
        or commitments[-1] != raw["finalRecordCommitmentSha256"]
    ):
        raise ElapsedClockValidationError(
            "clock boundary commitments contradict the per-transition sequence"
        )
    if _domain_sha256(_SUMMARY_DOMAIN, commitments) != raw["recordCommitmentsSha256"]:
        raise ElapsedClockValidationError(
            "per-elapsed-transition commitments are forged"
        )
    if raw["genesisCommitmentSha256"] != genesis_commitment(
        scenario_id=scenario_id,
        horizon_transitions=horizon,
    ):
        raise ElapsedClockValidationError("elapsed-clock genesis is forged")
    body = {
        key: raw[key] for key in sorted(_SUMMARY_KEYS - {"summaryCommitmentSha256"})
    }
    if _domain_sha256(_SUMMARY_DOMAIN, body) != raw["summaryCommitmentSha256"]:
        raise ElapsedClockValidationError("elapsed-clock summary is forged")
    return dict(raw)


def build_first_completion_projection(
    *,
    clock_summary: Mapping[str, Any],
    first_completion_transition: int | None,
    first_completion_record_commitment_sha256: str | None,
) -> dict[str, Any]:
    """Bind first completion to a complete authenticated clock sequence."""

    summary = validate_elapsed_clock_summary(clock_summary)
    if first_completion_transition is None:
        if first_completion_record_commitment_sha256 is not None:
            raise ElapsedClockValidationError(
                "noncompletion cannot name a completion record"
            )
        first: int | None = None
    else:
        first = _strict_int(
            first_completion_transition,
            field="firstCompletionTransition",
            minimum=0,
            maximum=int(summary["horizonTransitions"]),
        )
        if (
            not isinstance(first_completion_record_commitment_sha256, str)
            or len(first_completion_record_commitment_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in first_completion_record_commitment_sha256
            )
        ):
            raise ElapsedClockValidationError(
                "completion record commitment is absent or invalid"
            )
        if (
            summary["recordCommitmentsByElapsed"][first]
            != first_completion_record_commitment_sha256
        ):
            raise ElapsedClockValidationError(
                "completion is not authenticated by its elapsed-clock record"
            )
    body = {
        "schemaVersion": ELAPSED_CLOCK_PROJECTION_VERSION,
        "scenarioId": summary["scenarioId"],
        "horizonTransitions": summary["horizonTransitions"],
        "clockSummaryCommitmentSha256": summary["summaryCommitmentSha256"],
        "clockFinalRecordCommitmentSha256": summary["finalRecordCommitmentSha256"],
        "firstCompletionTransition": first,
        "firstCompletionRecordCommitmentSha256": (
            first_completion_record_commitment_sha256
        ),
        "noncompletionRightCensorTransition": summary["horizonTransitions"],
        "scientificTimeSource": "authenticatedElapsedTransition",
        "rawObservationLabelsUsed": False,
    }
    return {
        **body,
        "projectionCommitmentSha256": _domain_sha256(_PROJECTION_DOMAIN, body),
    }


_PROJECTION_KEYS = frozenset(
    {
        "schemaVersion",
        "scenarioId",
        "horizonTransitions",
        "clockSummaryCommitmentSha256",
        "clockFinalRecordCommitmentSha256",
        "firstCompletionTransition",
        "firstCompletionRecordCommitmentSha256",
        "noncompletionRightCensorTransition",
        "scientificTimeSource",
        "rawObservationLabelsUsed",
        "projectionCommitmentSha256",
    }
)


def validate_first_completion_projection(
    raw: Mapping[str, Any],
    *,
    expected_scenario_id: str | None = None,
    expected_horizon: int = 32,
    clock_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the compact persisted first-completion clock projection."""

    if not isinstance(raw, Mapping) or set(raw) != _PROJECTION_KEYS:
        raise ElapsedClockValidationError(
            "authenticated first-completion projection is absent or ambiguous"
        )
    if raw["schemaVersion"] != ELAPSED_CLOCK_PROJECTION_VERSION:
        raise ElapsedClockValidationError("unknown completion projection schema")
    scenario_id = raw["scenarioId"]
    if not isinstance(scenario_id, str) or not scenario_id:
        raise ElapsedClockValidationError("projection scenarioId is invalid")
    if expected_scenario_id is not None and scenario_id != expected_scenario_id:
        raise ElapsedClockValidationError("projection scenario identity mismatch")
    horizon = _strict_int(
        raw["horizonTransitions"],
        field="horizonTransitions",
        minimum=0,
    )
    if horizon != expected_horizon:
        raise ElapsedClockValidationError("projection horizon mismatch")
    if raw["noncompletionRightCensorTransition"] != horizon:
        raise ElapsedClockValidationError("projection censor boundary moved")
    if raw["scientificTimeSource"] != "authenticatedElapsedTransition":
        raise ElapsedClockValidationError("projection uses an unauthorized clock")
    if raw["rawObservationLabelsUsed"] is not False:
        raise ElapsedClockValidationError(
            "projection used raw observation labels as scientific time"
        )
    first = raw["firstCompletionTransition"]
    record_commitment = raw["firstCompletionRecordCommitmentSha256"]
    if first is None:
        if record_commitment is not None:
            raise ElapsedClockValidationError(
                "noncompletion projection names a completion record"
            )
    else:
        _strict_int(
            first,
            field="firstCompletionTransition",
            minimum=0,
            maximum=horizon,
        )
        if (
            not isinstance(record_commitment, str)
            or len(record_commitment) != 64
            or any(
                character not in "0123456789abcdef" for character in record_commitment
            )
        ):
            raise ElapsedClockValidationError(
                "completion record commitment is absent or invalid"
            )
    for field in (
        "clockSummaryCommitmentSha256",
        "clockFinalRecordCommitmentSha256",
        "projectionCommitmentSha256",
    ):
        value = raw[field]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ElapsedClockValidationError(f"{field}: lowercase SHA-256 required")
    body = {
        key: raw[key]
        for key in sorted(_PROJECTION_KEYS - {"projectionCommitmentSha256"})
    }
    if _domain_sha256(_PROJECTION_DOMAIN, body) != raw["projectionCommitmentSha256"]:
        raise ElapsedClockValidationError("completion projection is forged")
    if clock_summary is not None:
        summary = validate_elapsed_clock_summary(
            clock_summary,
            expected_scenario_id=scenario_id,
            expected_horizon=horizon,
        )
        if (
            raw["clockSummaryCommitmentSha256"] != summary["summaryCommitmentSha256"]
            or raw["clockFinalRecordCommitmentSha256"]
            != summary["finalRecordCommitmentSha256"]
        ):
            raise ElapsedClockValidationError(
                "completion projection is detached from elapsed-clock summary"
            )
        if (
            first is not None
            and record_commitment != summary["recordCommitmentsByElapsed"][first]
        ):
            raise ElapsedClockValidationError(
                "completion record is not in the authenticated clock sequence"
            )
    return dict(raw)
