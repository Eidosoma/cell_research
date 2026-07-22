"""Unified reset/observation/step/cost/event/outcome evaluation interface."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .access import AccessBroker
from .contracts import (
    AccessDeniedError,
    AccessGrant,
    EvaluationAction,
    NativeEpisodeResult,
    ObservationEnvelope,
    OutcomeEnvelope,
    ScenarioRecord,
    StepRecord,
    SuiteValidationError,
    TaskContract,
    load_split_manifest,
    load_task_registry,
)
from .runners import RUNNERS


class UnifiedEnvironment:
    """One stateful evaluation handle with task-specific native authority.

    A suite step is one complete native evaluation unit, not a claim that all
    predecessors share the same primitive clock.  The task's horizon contract
    explicitly identifies whether that unit contains activation opportunities,
    phase-local opportunities, or fixed 2D transitions.
    """

    __slots__ = (
        "task",
        "record",
        "grant",
        "broker",
        "_result",
        "_step_record",
    )

    def __init__(
        self,
        task: TaskContract,
        record: ScenarioRecord,
        grant: AccessGrant,
        broker: AccessBroker,
    ) -> None:
        self.task = task
        self.record = record
        self.grant = grant
        self.broker = broker
        self._result: NativeEpisodeResult | None = None
        self._step_record: StepRecord | None = None

    def reset(self) -> ObservationEnvelope:
        if self._result is not None:
            raise SuiteValidationError("an evaluation handle cannot be reset twice")
        return self.observation()

    def observation(self) -> ObservationEnvelope:
        return ObservationEnvelope(
            task_id=self.task.task_id,
            scenario_id=self.record.scenario_id,
            split=self.record.split,
            environment_profile=self.task.environment_profile,
            action_contract=self.task.action_contract,
            observation_contract=self.task.observation_contract,
            horizon=self.task.horizon.to_dict(),
            terminal=self._result is not None,
        )

    def step(self, action: EvaluationAction) -> StepRecord:
        if self._result is not None:
            raise SuiteValidationError("terminal evaluation cannot be stepped again")
        try:
            runner = RUNNERS[self.task.runner_id]
        except KeyError as exc:
            raise SuiteValidationError("task runner is unavailable") from exc
        result = runner(self.record, action)
        if not result.validation or not all(result.validation.values()):
            result = NativeEpisodeResult(
                stop_reason=result.stop_reason,
                censored=result.censored,
                failed=True,
                native_costs=result.native_costs,
                native_event=result.native_event,
                native_outcome=result.native_outcome,
                replay_pass=result.replay_pass,
                validation=result.validation,
                provenance=result.provenance,
            )
        cost = {
            "schemaVersion": "e07.s02.cost-envelope.v1",
            "nativeLedgerFamilies": {
                family: dict(ledger) for family, ledger in result.native_costs.items()
            },
            "communicationLedger": None,
            "scalarCrossFamilyTotal": None,
            "normalizationApplied": False,
        }
        event = {
            "schemaVersion": "e07.s02.event-envelope.v1",
            "native": dict(result.native_event),
            "nativeStopReason": result.stop_reason,
            "replayPass": result.replay_pass,
            "validation": dict(result.validation),
            "provenance": dict(result.provenance),
            "outcomeMetricsIncluded": False,
        }
        self._result = result
        self._step_record = StepRecord(
            task_id=self.task.task_id,
            scenario_id=self.record.scenario_id,
            split=self.record.split,
            native_unit=self.task.horizon.native_unit,
            stop_reason=result.stop_reason,
            terminal=True,
            censored=result.censored,
            failed=result.failed,
            cost=MappingProxyType(cost),
            event=MappingProxyType(event),
        )
        return self._step_record

    def cost(self) -> Mapping[str, Any]:
        if self._step_record is None:
            raise SuiteValidationError("cost is unavailable before step")
        return self._step_record.cost

    def event(self) -> Mapping[str, Any]:
        if self._step_record is None:
            raise SuiteValidationError("event is unavailable before step")
        return self._step_record.event

    def outcome(self) -> OutcomeEnvelope:
        if self._result is None:
            raise SuiteValidationError("outcome is unavailable before terminal step")
        self.broker.authorize_outcome(self.record.scenario_id, self.grant)
        return OutcomeEnvelope(
            task_id=self.task.task_id,
            scenario_id=self.record.scenario_id,
            split=self.record.split,
            stop_reason=self._result.stop_reason,
            censored=self._result.censored,
            failed=self._result.failed,
            outcome=self._result.native_outcome,
            claim_boundary=self.task.claim_boundary,
        )


class EnvironmentSuite:
    """Registry, split, and access-control owner for unified environments."""

    __slots__ = ("metadata", "tasks", "split_metadata", "records", "broker")

    def __init__(
        self,
        task_registry_path: str | Path,
        split_manifest_path: str | Path,
    ) -> None:
        metadata, tasks = load_task_registry(task_registry_path)
        split_metadata, records = load_split_manifest(split_manifest_path, tasks)
        if set(task.runner_id for task in tasks.values()) - set(RUNNERS):
            raise SuiteValidationError("registry references unavailable runners")
        self.metadata = metadata
        self.tasks = tasks
        self.split_metadata = split_metadata
        self.records = records
        # S02 deliberately ships with no confirmation unseal path.
        self.broker = AccessBroker(records, confirmation_unsealed=False)

    def open(
        self, task_id: str, scenario_id: str, grant: AccessGrant
    ) -> UnifiedEnvironment:
        try:
            task = self.tasks[task_id]
        except KeyError as exc:
            raise SuiteValidationError("unknown task ID") from exc
        record = self.broker.authorize_scenario(scenario_id, grant)
        if record.task_id != task_id:
            raise SuiteValidationError("scenario/task mismatch")
        return UnifiedEnvironment(task, record, grant, self.broker)

    def open_for_action(
        self,
        task_id: str,
        scenario_id: str,
        grant: AccessGrant,
        action: EvaluationAction,
    ) -> UnifiedEnvironment:
        """Authorize an action/split pair before scenario materialization.

        S04A DSL adapters are deliberately train-only.  Callers that possess a
        policy action use this entry point so a protected or validation record
        is rejected before the broker increments its materializer counter.
        ``UnifiedEnvironment.step`` retains a second train-only check as
        defense in depth.
        """

        try:
            public_record = self.records[scenario_id]
        except KeyError as exc:
            raise SuiteValidationError("unknown scenario ID") from exc
        if action.mode == "dsl_episode" and public_record.split.value != "train":
            self.broker.audit.scenario_requests += 1
            self.broker.audit.scenario_denials += 1
            raise AccessDeniedError(
                "S04A DSL adapter scenarios are restricted to the frozen training split"
            )
        return self.open(task_id, scenario_id, grant)

    def public_registry(self) -> dict[str, Any]:
        return {
            "metadata": dict(self.metadata),
            "tasks": [self.tasks[key].to_dict() for key in sorted(self.tasks)],
            "splitMetadata": dict(self.split_metadata),
            "scenarios": [
                self.records[key].public_dict() for key in sorted(self.records)
            ],
        }
