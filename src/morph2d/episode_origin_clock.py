"""Authenticated episode-relative clock retaining raw lifetime provenance.

This prospective S12AD clock binds the exact movement state at episode start.
Scientific elapsed time is the difference between each raw lifetime movement
index and that authenticated origin.  It does not mutate movement state,
policy observations, scheduling, costs, endpoints, or stopping semantics.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .movements import MovementState, movement_state_sha256

EPISODE_ORIGIN_CLOCK_RECORD_VERSION = (
    "e07.s12ad.authenticated-episode-origin-clock-record.v1"
)
EPISODE_ORIGIN_CLOCK_SUMMARY_VERSION = (
    "e07.s12ad.authenticated-episode-origin-clock-summary.v1"
)
EPISODE_ORIGIN_CLOCK_PROJECTION_VERSION = (
    "e07.s12ad.authenticated-episode-origin-completion-projection.v1"
)
_GENESIS_DOMAIN = "E07/S12AD/authenticated-episode-origin-genesis/v1"
_RECORD_DOMAIN = "E07/S12AD/authenticated-episode-origin-record/v1"
_SUMMARY_DOMAIN = "E07/S12AD/authenticated-episode-origin-summary/v1"
_PROJECTION_DOMAIN = "E07/S12AD/authenticated-episode-origin-completion/v1"


class EpisodeOriginClockValidationError(ValueError):
    """Episode-origin clock metadata is absent, contradictory, or forged."""


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
        raise EpisodeOriginClockValidationError(
            "episode-origin clock metadata is not canonical JSON"
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
            raise EpisodeOriginClockValidationError(f"{field}: non-finite value")
        raise EpisodeOriginClockValidationError(f"{field}: exact integer required")
    if minimum is not None and value < minimum:
        raise EpisodeOriginClockValidationError(f"{field}: below declared domain")
    if maximum is not None and value > maximum:
        raise EpisodeOriginClockValidationError(f"{field}: above declared domain")
    return value


def _sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise EpisodeOriginClockValidationError(
            f"{field}: lowercase SHA-256 required"
        )
    return value


def _raw_label(value: Any) -> int | str | None:
    if value is None or isinstance(value, str) or type(value) is int:
        return value
    raise EpisodeOriginClockValidationError(
        "rawObservationLabel: only integer, string, or null provenance is allowed"
    )


def episode_origin_genesis_commitment(
    *,
    scenario_id: str,
    horizon_transitions: int,
    episode_origin_transition_index: int,
    episode_origin_movement_state_sha256: str,
) -> str:
    if not isinstance(scenario_id, str) or not scenario_id:
        raise EpisodeOriginClockValidationError(
            "scenarioId: nonempty string required"
        )
    horizon = _strict_int(
        horizon_transitions,
        field="horizonTransitions",
        minimum=0,
    )
    origin = _strict_int(
        episode_origin_transition_index,
        field="episodeOriginTransitionIndex",
        minimum=0,
    )
    origin_hash = _sha256(
        episode_origin_movement_state_sha256,
        field="episodeOriginMovementStateSha256",
    )
    return _domain_sha256(
        _GENESIS_DOMAIN,
        {
            "schemaVersion": EPISODE_ORIGIN_CLOCK_RECORD_VERSION,
            "scenarioId": scenario_id,
            "horizonTransitions": horizon,
            "episodeOriginTransitionIndex": origin,
            "episodeOriginMovementStateSha256": origin_hash,
        },
    )


@dataclass(frozen=True)
class AuthenticatedEpisodeOriginClockRecord:
    scenario_id: str
    horizon_transitions: int
    sequence_ordinal: int
    elapsed_transition: int
    phase: str
    raw_observation_label: int | str | None
    episode_origin_transition_index: int
    episode_origin_movement_state_sha256: str
    movement_state_transition_index: int
    movement_state_sha256: str
    previous_commitment_sha256: str
    record_commitment_sha256: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schemaVersion": EPISODE_ORIGIN_CLOCK_RECORD_VERSION,
            "scenarioId": self.scenario_id,
            "horizonTransitions": self.horizon_transitions,
            "sequenceOrdinal": self.sequence_ordinal,
            "authenticatedElapsedTransition": self.elapsed_transition,
            "phase": self.phase,
            "rawObservationLabel": self.raw_observation_label,
            "episodeOriginTransitionIndex": self.episode_origin_transition_index,
            "episodeOriginMovementStateSha256": (
                self.episode_origin_movement_state_sha256
            ),
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
        "episodeOriginTransitionIndex",
        "episodeOriginMovementStateSha256",
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
    episode_origin_transition_index: int,
    episode_origin_movement_state_sha256: str,
    movement_state_transition_index: int,
    movement_state_sha256_value: str,
    previous_commitment_sha256: str,
) -> dict[str, Any]:
    return {
        "schemaVersion": EPISODE_ORIGIN_CLOCK_RECORD_VERSION,
        "scenarioId": scenario_id,
        "horizonTransitions": horizon_transitions,
        "sequenceOrdinal": sequence_ordinal,
        "authenticatedElapsedTransition": elapsed_transition,
        "phase": phase,
        "rawObservationLabel": raw_observation_label,
        "episodeOriginTransitionIndex": episode_origin_transition_index,
        "episodeOriginMovementStateSha256": episode_origin_movement_state_sha256,
        "movementStateTransitionIndex": movement_state_transition_index,
        "movementStateSha256": movement_state_sha256_value,
        "previousCommitmentSha256": previous_commitment_sha256,
    }


def parse_and_validate_episode_origin_clock_record(
    raw: Mapping[str, Any],
    *,
    expected_scenario_id: str | None = None,
    expected_horizon: int | None = None,
    expected_ordinal: int | None = None,
    expected_elapsed: int | None = None,
    expected_origin_transition_index: int | None = None,
    expected_origin_movement_state_sha256: str | None = None,
    expected_previous_commitment: str | None = None,
    state: MovementState | None = None,
) -> AuthenticatedEpisodeOriginClockRecord:
    if not isinstance(raw, Mapping):
        raise EpisodeOriginClockValidationError(
            "episode-origin clock record must be a mapping"
        )
    if set(raw) != _RECORD_KEYS:
        missing = sorted(_RECORD_KEYS - set(raw))
        extra = sorted(set(raw) - _RECORD_KEYS)
        raise EpisodeOriginClockValidationError(
            f"episode-origin clock schema mismatch; missing={missing}, extra={extra}"
        )
    if raw["schemaVersion"] != EPISODE_ORIGIN_CLOCK_RECORD_VERSION:
        raise EpisodeOriginClockValidationError(
            "unknown episode-origin clock record schema"
        )
    scenario_id = raw["scenarioId"]
    if not isinstance(scenario_id, str) or not scenario_id:
        raise EpisodeOriginClockValidationError(
            "scenarioId: nonempty string required"
        )
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
    origin = _strict_int(
        raw["episodeOriginTransitionIndex"],
        field="episodeOriginTransitionIndex",
        minimum=0,
    )
    lifetime_index = _strict_int(
        raw["movementStateTransitionIndex"],
        field="movementStateTransitionIndex",
        minimum=0,
    )
    if ordinal != elapsed:
        raise EpisodeOriginClockValidationError(
            "sequence ordinal and authenticated elapsed transition disagree"
        )
    if lifetime_index - origin != elapsed:
        raise EpisodeOriginClockValidationError(
            "lifetime movement state and episode-origin elapsed transition disagree"
        )
    phase = raw["phase"]
    expected_phase = "initial_state" if elapsed == 0 else "post_transition"
    if phase != expected_phase:
        raise EpisodeOriginClockValidationError(
            f"phase disagrees with episode elapsed transition {elapsed}"
        )
    label = _raw_label(raw["rawObservationLabel"])
    origin_hash = _sha256(
        raw["episodeOriginMovementStateSha256"],
        field="episodeOriginMovementStateSha256",
    )
    state_hash = _sha256(
        raw["movementStateSha256"], field="movementStateSha256"
    )
    previous = _sha256(
        raw["previousCommitmentSha256"], field="previousCommitmentSha256"
    )
    commitment = _sha256(
        raw["recordCommitmentSha256"], field="recordCommitmentSha256"
    )
    if elapsed == 0 and state_hash != origin_hash:
        raise EpisodeOriginClockValidationError(
            "initial movement-state hash differs from authenticated episode origin"
        )
    if expected_scenario_id is not None and scenario_id != expected_scenario_id:
        raise EpisodeOriginClockValidationError("scenario identity mismatch")
    if expected_horizon is not None and horizon != expected_horizon:
        raise EpisodeOriginClockValidationError("horizon mismatch")
    if expected_ordinal is not None and ordinal != expected_ordinal:
        raise EpisodeOriginClockValidationError(
            "episode-origin clock sequence is skipped or repeated"
        )
    if expected_elapsed is not None and elapsed != expected_elapsed:
        raise EpisodeOriginClockValidationError(
            "unexpected authenticated episode elapsed transition"
        )
    if (
        expected_origin_transition_index is not None
        and origin != expected_origin_transition_index
    ):
        raise EpisodeOriginClockValidationError("episode-origin index mismatch")
    if (
        expected_origin_movement_state_sha256 is not None
        and origin_hash != expected_origin_movement_state_sha256
    ):
        raise EpisodeOriginClockValidationError("episode-origin state hash mismatch")
    if (
        expected_previous_commitment is not None
        and previous != expected_previous_commitment
    ):
        raise EpisodeOriginClockValidationError(
            "episode-origin clock chain is broken"
        )
    if state is not None:
        if state.transition_index != lifetime_index:
            raise EpisodeOriginClockValidationError(
                "runtime lifetime movement index disagrees with clock record"
            )
        if movement_state_sha256(state) != state_hash:
            raise EpisodeOriginClockValidationError(
                "runtime movement-state hash disagrees with clock record"
            )
    body = _record_body(
        scenario_id=scenario_id,
        horizon_transitions=horizon,
        sequence_ordinal=ordinal,
        elapsed_transition=elapsed,
        phase=phase,
        raw_observation_label=label,
        episode_origin_transition_index=origin,
        episode_origin_movement_state_sha256=origin_hash,
        movement_state_transition_index=lifetime_index,
        movement_state_sha256_value=state_hash,
        previous_commitment_sha256=previous,
    )
    if _domain_sha256(_RECORD_DOMAIN, body) != commitment:
        raise EpisodeOriginClockValidationError(
            "forged episode-origin clock record commitment"
        )
    return AuthenticatedEpisodeOriginClockRecord(
        scenario_id=scenario_id,
        horizon_transitions=horizon,
        sequence_ordinal=ordinal,
        elapsed_transition=elapsed,
        phase=phase,
        raw_observation_label=label,
        episode_origin_transition_index=origin,
        episode_origin_movement_state_sha256=origin_hash,
        movement_state_transition_index=lifetime_index,
        movement_state_sha256=state_hash,
        previous_commitment_sha256=previous,
        record_commitment_sha256=commitment,
    )


class EpisodeOriginClockAuthenticator:
    """Engine-owned issuer relative to one authenticated lifetime origin."""

    def __init__(
        self,
        *,
        scenario_id: str,
        horizon_transitions: int,
        initial_state: MovementState,
    ) -> None:
        self.scenario_id = scenario_id
        self.horizon_transitions = _strict_int(
            horizon_transitions,
            field="horizonTransitions",
            minimum=0,
        )
        self.episode_origin_transition_index = _strict_int(
            initial_state.transition_index,
            field="episodeOriginTransitionIndex",
            minimum=0,
        )
        self.episode_origin_movement_state_sha256 = movement_state_sha256(
            initial_state
        )
        self._previous = episode_origin_genesis_commitment(
            scenario_id=scenario_id,
            horizon_transitions=self.horizon_transitions,
            episode_origin_transition_index=self.episode_origin_transition_index,
            episode_origin_movement_state_sha256=(
                self.episode_origin_movement_state_sha256
            ),
        )
        self._records: list[AuthenticatedEpisodeOriginClockRecord] = []

    @property
    def records(self) -> tuple[AuthenticatedEpisodeOriginClockRecord, ...]:
        return tuple(self._records)

    def issue(
        self,
        *,
        elapsed_transition: int,
        raw_observation_label: int | str | None,
        state: MovementState,
    ) -> AuthenticatedEpisodeOriginClockRecord:
        expected = len(self._records)
        elapsed = _strict_int(
            elapsed_transition,
            field="authenticatedElapsedTransition",
            minimum=0,
            maximum=self.horizon_transitions,
        )
        if elapsed != expected:
            raise EpisodeOriginClockValidationError(
                "clock issuer requires contiguous episode elapsed transitions"
            )
        phase = "initial_state" if elapsed == 0 else "post_transition"
        state_hash = movement_state_sha256(state)
        body = _record_body(
            scenario_id=self.scenario_id,
            horizon_transitions=self.horizon_transitions,
            sequence_ordinal=expected,
            elapsed_transition=elapsed,
            phase=phase,
            raw_observation_label=_raw_label(raw_observation_label),
            episode_origin_transition_index=self.episode_origin_transition_index,
            episode_origin_movement_state_sha256=(
                self.episode_origin_movement_state_sha256
            ),
            movement_state_transition_index=state.transition_index,
            movement_state_sha256_value=state_hash,
            previous_commitment_sha256=self._previous,
        )
        commitment = _domain_sha256(_RECORD_DOMAIN, body)
        record = parse_and_validate_episode_origin_clock_record(
            {**body, "recordCommitmentSha256": commitment},
            expected_scenario_id=self.scenario_id,
            expected_horizon=self.horizon_transitions,
            expected_ordinal=expected,
            expected_elapsed=elapsed,
            expected_origin_transition_index=self.episode_origin_transition_index,
            expected_origin_movement_state_sha256=(
                self.episode_origin_movement_state_sha256
            ),
            expected_previous_commitment=self._previous,
            state=state,
        )
        self._records.append(record)
        self._previous = record.record_commitment_sha256
        return record

    def finalize(self) -> dict[str, Any]:
        return summarize_episode_origin_clock_records(
            [record.to_mapping() for record in self._records],
            scenario_id=self.scenario_id,
            horizon_transitions=self.horizon_transitions,
            episode_origin_transition_index=self.episode_origin_transition_index,
            episode_origin_movement_state_sha256=(
                self.episode_origin_movement_state_sha256
            ),
            require_complete=True,
        )


def summarize_episode_origin_clock_records(
    records: Sequence[Mapping[str, Any]],
    *,
    scenario_id: str,
    horizon_transitions: int,
    episode_origin_transition_index: int,
    episode_origin_movement_state_sha256: str,
    require_complete: bool,
) -> dict[str, Any]:
    horizon = _strict_int(
        horizon_transitions,
        field="horizonTransitions",
        minimum=0,
    )
    origin = _strict_int(
        episode_origin_transition_index,
        field="episodeOriginTransitionIndex",
        minimum=0,
    )
    origin_hash = _sha256(
        episode_origin_movement_state_sha256,
        field="episodeOriginMovementStateSha256",
    )
    if not records:
        raise EpisodeOriginClockValidationError(
            "episode-origin clock sequence is absent"
        )
    expected_previous = episode_origin_genesis_commitment(
        scenario_id=scenario_id,
        horizon_transitions=horizon,
        episode_origin_transition_index=origin,
        episode_origin_movement_state_sha256=origin_hash,
    )
    parsed: list[AuthenticatedEpisodeOriginClockRecord] = []
    for ordinal, raw in enumerate(records):
        item = parse_and_validate_episode_origin_clock_record(
            raw,
            expected_scenario_id=scenario_id,
            expected_horizon=horizon,
            expected_ordinal=ordinal,
            expected_elapsed=ordinal,
            expected_origin_transition_index=origin,
            expected_origin_movement_state_sha256=origin_hash,
            expected_previous_commitment=expected_previous,
        )
        parsed.append(item)
        expected_previous = item.record_commitment_sha256
    if require_complete and len(parsed) != horizon + 1:
        raise EpisodeOriginClockValidationError(
            "episode-origin clock sequence is incomplete at finalization"
        )
    commitments = [item.record_commitment_sha256 for item in parsed]
    body = {
        "schemaVersion": EPISODE_ORIGIN_CLOCK_SUMMARY_VERSION,
        "scenarioId": scenario_id,
        "horizonTransitions": horizon,
        "episodeOriginTransitionIndex": origin,
        "episodeOriginMovementStateSha256": origin_hash,
        "recordCount": len(parsed),
        "initialElapsedTransition": parsed[0].elapsed_transition,
        "finalElapsedTransition": parsed[-1].elapsed_transition,
        "initialLifetimeTransitionIndex": parsed[0].movement_state_transition_index,
        "finalLifetimeTransitionIndex": parsed[-1].movement_state_transition_index,
        "genesisCommitmentSha256": episode_origin_genesis_commitment(
            scenario_id=scenario_id,
            horizon_transitions=horizon,
            episode_origin_transition_index=origin,
            episode_origin_movement_state_sha256=origin_hash,
        ),
        "initialRecordCommitmentSha256": parsed[0].record_commitment_sha256,
        "finalRecordCommitmentSha256": parsed[-1].record_commitment_sha256,
        "recordCommitmentsByElapsed": commitments,
        "recordCommitmentsSha256": _domain_sha256(_SUMMARY_DOMAIN, commitments),
        "complete": len(parsed) == horizon + 1,
        "rawObservationLabelsScientificTime": False,
        "rawLifetimeProvenanceRetained": True,
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
        "episodeOriginTransitionIndex",
        "episodeOriginMovementStateSha256",
        "recordCount",
        "initialElapsedTransition",
        "finalElapsedTransition",
        "initialLifetimeTransitionIndex",
        "finalLifetimeTransitionIndex",
        "genesisCommitmentSha256",
        "initialRecordCommitmentSha256",
        "finalRecordCommitmentSha256",
        "recordCommitmentsByElapsed",
        "recordCommitmentsSha256",
        "complete",
        "rawObservationLabelsScientificTime",
        "rawLifetimeProvenanceRetained",
        "summaryCommitmentSha256",
    }
)


def validate_episode_origin_clock_summary(
    raw: Mapping[str, Any],
    *,
    expected_scenario_id: str | None = None,
    expected_horizon: int = 32,
    expected_origin_transition_index: int | None = None,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != _SUMMARY_KEYS:
        raise EpisodeOriginClockValidationError(
            "authenticated episode-origin summary is absent or ambiguous"
        )
    if raw["schemaVersion"] != EPISODE_ORIGIN_CLOCK_SUMMARY_VERSION:
        raise EpisodeOriginClockValidationError(
            "unknown episode-origin clock summary schema"
        )
    scenario_id = raw["scenarioId"]
    if not isinstance(scenario_id, str) or not scenario_id:
        raise EpisodeOriginClockValidationError("summary scenarioId is invalid")
    if expected_scenario_id is not None and scenario_id != expected_scenario_id:
        raise EpisodeOriginClockValidationError("summary scenario identity mismatch")
    horizon = _strict_int(
        raw["horizonTransitions"],
        field="horizonTransitions",
        minimum=0,
    )
    if horizon != expected_horizon:
        raise EpisodeOriginClockValidationError("summary horizon mismatch")
    origin = _strict_int(
        raw["episodeOriginTransitionIndex"],
        field="episodeOriginTransitionIndex",
        minimum=0,
    )
    if (
        expected_origin_transition_index is not None
        and origin != expected_origin_transition_index
    ):
        raise EpisodeOriginClockValidationError("summary episode origin mismatch")
    origin_hash = _sha256(
        raw["episodeOriginMovementStateSha256"],
        field="episodeOriginMovementStateSha256",
    )
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
    initial_lifetime = _strict_int(
        raw["initialLifetimeTransitionIndex"],
        field="initialLifetimeTransitionIndex",
        minimum=0,
    )
    final_lifetime = _strict_int(
        raw["finalLifetimeTransitionIndex"],
        field="finalLifetimeTransitionIndex",
        minimum=0,
    )
    if (
        initial != 0
        or final != horizon
        or count != horizon + 1
        or initial_lifetime != origin
        or final_lifetime != origin + horizon
    ):
        raise EpisodeOriginClockValidationError(
            "episode-origin clock summary is incomplete or contradictory"
        )
    if raw["complete"] is not True:
        raise EpisodeOriginClockValidationError(
            "episode-origin clock summary is not complete"
        )
    if raw["rawObservationLabelsScientificTime"] is not False:
        raise EpisodeOriginClockValidationError(
            "raw observation labels cannot control scientific time"
        )
    if raw["rawLifetimeProvenanceRetained"] is not True:
        raise EpisodeOriginClockValidationError(
            "raw lifetime movement provenance is absent"
        )
    for field in (
        "genesisCommitmentSha256",
        "initialRecordCommitmentSha256",
        "finalRecordCommitmentSha256",
        "recordCommitmentsSha256",
        "summaryCommitmentSha256",
    ):
        _sha256(raw[field], field=field)
    commitments = raw["recordCommitmentsByElapsed"]
    if not isinstance(commitments, list) or len(commitments) != count:
        raise EpisodeOriginClockValidationError(
            "per-elapsed-transition commitments are absent or incomplete"
        )
    for commitment in commitments:
        _sha256(commitment, field="recordCommitmentsByElapsed")
    if (
        commitments[0] != raw["initialRecordCommitmentSha256"]
        or commitments[-1] != raw["finalRecordCommitmentSha256"]
    ):
        raise EpisodeOriginClockValidationError(
            "summary boundary commitments contradict the record sequence"
        )
    if _domain_sha256(_SUMMARY_DOMAIN, commitments) != raw[
        "recordCommitmentsSha256"
    ]:
        raise EpisodeOriginClockValidationError(
            "per-elapsed-transition commitments are forged"
        )
    if raw["genesisCommitmentSha256"] != episode_origin_genesis_commitment(
        scenario_id=scenario_id,
        horizon_transitions=horizon,
        episode_origin_transition_index=origin,
        episode_origin_movement_state_sha256=origin_hash,
    ):
        raise EpisodeOriginClockValidationError("episode-origin genesis is forged")
    body = {
        key: raw[key]
        for key in sorted(_SUMMARY_KEYS - {"summaryCommitmentSha256"})
    }
    if _domain_sha256(_SUMMARY_DOMAIN, body) != raw["summaryCommitmentSha256"]:
        raise EpisodeOriginClockValidationError(
            "episode-origin clock summary is forged"
        )
    return dict(raw)


def build_episode_origin_first_completion_projection(
    *,
    clock_summary: Mapping[str, Any],
    first_completion_transition: int | None,
    first_completion_record_commitment_sha256: str | None,
) -> dict[str, Any]:
    summary = validate_episode_origin_clock_summary(clock_summary)
    if first_completion_transition is None:
        if first_completion_record_commitment_sha256 is not None:
            raise EpisodeOriginClockValidationError(
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
        completion_commitment = _sha256(
            first_completion_record_commitment_sha256,
            field="firstCompletionRecordCommitmentSha256",
        )
        if summary["recordCommitmentsByElapsed"][first] != completion_commitment:
            raise EpisodeOriginClockValidationError(
                "completion is not authenticated by its episode elapsed record"
            )
    body = {
        "schemaVersion": EPISODE_ORIGIN_CLOCK_PROJECTION_VERSION,
        "scenarioId": summary["scenarioId"],
        "horizonTransitions": summary["horizonTransitions"],
        "episodeOriginTransitionIndex": summary["episodeOriginTransitionIndex"],
        "episodeOriginMovementStateSha256": summary[
            "episodeOriginMovementStateSha256"
        ],
        "clockSummaryCommitmentSha256": summary["summaryCommitmentSha256"],
        "clockFinalRecordCommitmentSha256": summary[
            "finalRecordCommitmentSha256"
        ],
        "firstCompletionTransition": first,
        "firstCompletionRecordCommitmentSha256": (
            first_completion_record_commitment_sha256
        ),
        "noncompletionRightCensorTransition": summary["horizonTransitions"],
        "rawObservationLabelsScientificTime": False,
        "rawLifetimeProvenanceRetained": True,
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
        "episodeOriginTransitionIndex",
        "episodeOriginMovementStateSha256",
        "clockSummaryCommitmentSha256",
        "clockFinalRecordCommitmentSha256",
        "firstCompletionTransition",
        "firstCompletionRecordCommitmentSha256",
        "noncompletionRightCensorTransition",
        "rawObservationLabelsScientificTime",
        "rawLifetimeProvenanceRetained",
        "projectionCommitmentSha256",
    }
)


def validate_episode_origin_first_completion_projection(
    raw: Mapping[str, Any],
    *,
    expected_scenario_id: str | None = None,
    expected_horizon: int = 32,
    clock_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != _PROJECTION_KEYS:
        raise EpisodeOriginClockValidationError(
            "episode-origin completion projection is absent or ambiguous"
        )
    if raw["schemaVersion"] != EPISODE_ORIGIN_CLOCK_PROJECTION_VERSION:
        raise EpisodeOriginClockValidationError(
            "unknown episode-origin completion projection schema"
        )
    if expected_scenario_id is not None and raw["scenarioId"] != expected_scenario_id:
        raise EpisodeOriginClockValidationError("projection scenario mismatch")
    horizon = _strict_int(
        raw["horizonTransitions"],
        field="horizonTransitions",
        minimum=0,
    )
    if horizon != expected_horizon:
        raise EpisodeOriginClockValidationError("projection horizon mismatch")
    _strict_int(
        raw["episodeOriginTransitionIndex"],
        field="episodeOriginTransitionIndex",
        minimum=0,
    )
    _sha256(
        raw["episodeOriginMovementStateSha256"],
        field="episodeOriginMovementStateSha256",
    )
    if raw["noncompletionRightCensorTransition"] != horizon:
        raise EpisodeOriginClockValidationError(
            "projection censor transition changed"
        )
    if raw["rawObservationLabelsScientificTime"] is not False:
        raise EpisodeOriginClockValidationError(
            "raw observation labels cannot control projection time"
        )
    if raw["rawLifetimeProvenanceRetained"] is not True:
        raise EpisodeOriginClockValidationError(
            "projection dropped lifetime movement provenance"
        )
    for field in (
        "clockSummaryCommitmentSha256",
        "clockFinalRecordCommitmentSha256",
        "projectionCommitmentSha256",
    ):
        _sha256(raw[field], field=field)
    first = raw["firstCompletionTransition"]
    completion_commitment = raw["firstCompletionRecordCommitmentSha256"]
    if first is None:
        if completion_commitment is not None:
            raise EpisodeOriginClockValidationError(
                "noncompletion projection named a completion record"
            )
    else:
        first = _strict_int(
            first,
            field="firstCompletionTransition",
            minimum=0,
            maximum=horizon,
        )
        _sha256(
            completion_commitment,
            field="firstCompletionRecordCommitmentSha256",
        )
    if clock_summary is not None:
        summary = validate_episode_origin_clock_summary(
            clock_summary,
            expected_scenario_id=str(raw["scenarioId"]),
            expected_horizon=horizon,
            expected_origin_transition_index=int(
                raw["episodeOriginTransitionIndex"]
            ),
        )
        if (
            raw["episodeOriginMovementStateSha256"]
            != summary["episodeOriginMovementStateSha256"]
            or raw["clockSummaryCommitmentSha256"]
            != summary["summaryCommitmentSha256"]
            or raw["clockFinalRecordCommitmentSha256"]
            != summary["finalRecordCommitmentSha256"]
        ):
            raise EpisodeOriginClockValidationError(
                "completion projection is detached from its episode-origin clock"
            )
        if first is not None and completion_commitment != summary[
            "recordCommitmentsByElapsed"
        ][first]:
            raise EpisodeOriginClockValidationError(
                "completion projection names the wrong elapsed record"
            )
    body = {
        key: raw[key]
        for key in sorted(_PROJECTION_KEYS - {"projectionCommitmentSha256"})
    }
    if _domain_sha256(_PROJECTION_DOMAIN, body) != raw[
        "projectionCommitmentSha256"
    ]:
        raise EpisodeOriginClockValidationError(
            "episode-origin completion projection is forged"
        )
    return dict(raw)
