"""Outcome and scenario access gates for the E07 S02 suite."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .contracts import (
    AccessDeniedError,
    AccessGrant,
    AccessPhase,
    ScenarioRecord,
    Split,
    SuiteValidationError,
)


@dataclass(slots=True)
class AccessAudit:
    scenario_requests: int = 0
    scenario_denials: int = 0
    outcome_requests: int = 0
    outcome_denials: int = 0
    materializer_invocations: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "scenarioRequests": self.scenario_requests,
            "scenarioDenials": self.scenario_denials,
            "outcomeRequests": self.outcome_requests,
            "outcomeDenials": self.outcome_denials,
            "materializerInvocations": self.materializer_invocations,
        }


class AccessBroker:
    """Authorize a scenario before a materializer or result can be touched."""

    __slots__ = (
        "records",
        "confirmation_unsealed",
        "expected_candidate_lock_sha256",
        "audit",
    )

    def __init__(
        self,
        records: Mapping[str, ScenarioRecord],
        *,
        confirmation_unsealed: bool = False,
        expected_candidate_lock_sha256: str | None = None,
    ) -> None:
        if confirmation_unsealed and not expected_candidate_lock_sha256:
            raise SuiteValidationError(
                "confirmation unseal requires an expected candidate lock"
            )
        self.records = records
        self.confirmation_unsealed = confirmation_unsealed
        self.expected_candidate_lock_sha256 = expected_candidate_lock_sha256
        self.audit = AccessAudit()

    def _record(self, scenario_id: str) -> ScenarioRecord:
        try:
            return self.records[scenario_id]
        except KeyError as exc:
            raise SuiteValidationError("unknown scenario ID") from exc

    def authorize_scenario(
        self, scenario_id: str, grant: AccessGrant
    ) -> ScenarioRecord:
        self.audit.scenario_requests += 1
        record = self._record(scenario_id)
        permitted = (
            record.split == Split.TRAIN
            or (
                record.split == Split.VALIDATION
                and grant.phase in {AccessPhase.VALIDATION, AccessPhase.CONFIRMATION}
            )
            or (
                record.split == Split.CONFIRMATION
                and grant.phase == AccessPhase.CONFIRMATION
                and self.confirmation_unsealed
                and grant.candidate_lock_sha256 == self.expected_candidate_lock_sha256
            )
        )
        if not permitted:
            self.audit.scenario_denials += 1
            raise AccessDeniedError(
                f"scenario split {record.split.value!r} is unavailable in "
                f"phase {grant.phase.value!r}"
            )
        if record.protected or record.materializer_id is None:
            self.audit.scenario_denials += 1
            raise AccessDeniedError(
                "protected confirmation scenario materialization is not present in S02"
            )
        self.audit.materializer_invocations += 1
        return record

    def authorize_outcome(self, scenario_id: str, grant: AccessGrant) -> ScenarioRecord:
        self.audit.outcome_requests += 1
        try:
            record = self.authorize_scenario(scenario_id, grant)
        except AccessDeniedError:
            self.audit.outcome_denials += 1
            raise
        if record.outcome_access == "sealed":
            self.audit.outcome_denials += 1
            raise AccessDeniedError("scenario outcome remains sealed")
        return record
