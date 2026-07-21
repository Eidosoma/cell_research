"""Typed, leakage-resistant contracts for the E07 unified environment suite.

The suite standardizes the *evaluation control plane*.  Native simulators keep
authority over legality, policy observations, actions, events, stopping,
censoring, and costs.  In particular, this module deliberately defines no
cross-task scalar reward, cost total, or standardized time coordinate.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml


SUITE_VERSION = "e07.s02.environment-suite.v1"
TASK_REGISTRY_VERSION = "e07.s02.task-registry.v1"
SPLIT_MANIFEST_VERSION = "e07.s02.split-manifest.v1"


class SuiteValidationError(ValueError):
    """Raised when a task, scenario, action, or result violates the suite."""


class AccessDeniedError(PermissionError):
    """Raised before any protected scenario materializer can be invoked."""


class Split(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    CONFIRMATION = "confirmation"


class AccessPhase(str, Enum):
    DEVELOPMENT = "development"
    VALIDATION = "validation"
    CONFIRMATION = "confirmation"


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_sha256(domain: str, value: Any) -> str:
    return hashlib.sha256(
        domain.encode("ascii") + b"\x00" + canonical_json_bytes(value)
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class AccessGrant:
    """A non-secret capability describing the maximum authorized phase.

    Confirmation access is deliberately impossible in the S02 release: the
    broker additionally requires a later candidate lock and an explicit
    confirmation-unseal switch, neither of which exists in S02 artifacts.
    """

    phase: AccessPhase
    candidate_lock_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.candidate_lock_sha256 is not None and (
            len(self.candidate_lock_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in self.candidate_lock_sha256)
        ):
            raise SuiteValidationError("candidate lock must be lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class HorizonContract:
    native_unit: str
    adapter_step_granularity: str
    budgets: Mapping[str, int | str]
    terminal_precedence: tuple[str, ...]
    censoring_rule: str
    comparability_group: str
    comparable_only_within_group: bool
    cross_task_normalization: str

    def __post_init__(self) -> None:
        if not self.native_unit or not self.adapter_step_granularity:
            raise SuiteValidationError("horizon units must be explicit")
        if not self.budgets:
            raise SuiteValidationError("horizon requires at least one native budget")
        if not self.terminal_precedence:
            raise SuiteValidationError("terminal precedence must be nonempty")
        if self.cross_task_normalization != "forbidden":
            raise SuiteValidationError("cross-task horizon normalization is forbidden")

    def to_dict(self) -> dict[str, Any]:
        return {
            "nativeUnit": self.native_unit,
            "adapterStepGranularity": self.adapter_step_granularity,
            "budgets": dict(self.budgets),
            "terminalPrecedence": list(self.terminal_precedence),
            "censoringRule": self.censoring_rule,
            "comparabilityGroup": self.comparability_group,
            "comparableOnlyWithinGroup": self.comparable_only_within_group,
            "crossTaskNormalization": self.cross_task_normalization,
        }


@dataclass(frozen=True, slots=True)
class TaskContract:
    task_id: str
    title: str
    family: str
    predecessor: str
    runner_id: str
    environment_profile: str
    legality_authority: str
    action_contract: Mapping[str, Any]
    observation_contract: Mapping[str, Any]
    cost_contract: Mapping[str, Any]
    event_contract: Mapping[str, Any]
    stopping_contract: Mapping[str, Any]
    outcome_contract: Mapping[str, Any]
    claim_boundary: str
    horizon: HorizonContract
    optional_exclusions: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.task_id.startswith("e07_s02_"):
            raise SuiteValidationError("task IDs must use the e07_s02_ prefix")
        if not self.claim_boundary:
            raise SuiteValidationError("every task requires a claim boundary")
        if self.outcome_contract.get("policyVisibleOnline") is not False:
            raise SuiteValidationError(
                "outcome metrics must not be online policy inputs"
            )
        if self.observation_contract.get("forbiddenFields") is None:
            raise SuiteValidationError("observation contract requires forbidden fields")
        if self.cost_contract.get("scalarTotalPermitted") is not False:
            raise SuiteValidationError("scalar cross-family costs are forbidden")

    def to_dict(self) -> dict[str, Any]:
        return {
            "taskId": self.task_id,
            "title": self.title,
            "family": self.family,
            "predecessor": self.predecessor,
            "runnerId": self.runner_id,
            "environmentProfile": self.environment_profile,
            "legalityAuthority": self.legality_authority,
            "actionContract": dict(self.action_contract),
            "observationContract": dict(self.observation_contract),
            "costContract": dict(self.cost_contract),
            "eventContract": dict(self.event_contract),
            "stoppingContract": dict(self.stopping_contract),
            "outcomeContract": dict(self.outcome_contract),
            "claimBoundary": self.claim_boundary,
            "horizon": self.horizon.to_dict(),
            "optionalExclusions": list(self.optional_exclusions),
        }


@dataclass(frozen=True, slots=True)
class ScenarioRecord:
    scenario_id: str
    task_id: str
    split: Split
    materializer_id: str | None
    public_parameters: Mapping[str, Any]
    protected: bool
    outcome_access: str
    predecessor_partition: str

    def __post_init__(self) -> None:
        if self.protected != (self.split == Split.CONFIRMATION):
            raise SuiteValidationError("only confirmation records may be protected")
        if self.protected and self.materializer_id is not None:
            raise SuiteValidationError(
                "protected confirmation records must not name a materializer"
            )
        if self.protected and self.outcome_access != "sealed":
            raise SuiteValidationError("protected outcome access must be sealed")
        if not self.protected and not self.materializer_id:
            raise SuiteValidationError("development records require a materializer")

    def public_dict(self) -> dict[str, Any]:
        result = {
            "scenarioId": self.scenario_id,
            "taskId": self.task_id,
            "split": self.split.value,
            "publicParameters": dict(self.public_parameters),
            "protected": self.protected,
            "outcomeAccess": self.outcome_access,
            "predecessorPartition": self.predecessor_partition,
        }
        if not self.protected:
            result["materializerId"] = self.materializer_id
        return result


@dataclass(frozen=True, slots=True)
class ObservationEnvelope:
    task_id: str
    scenario_id: str
    split: Split
    environment_profile: str
    action_contract: Mapping[str, Any]
    observation_contract: Mapping[str, Any]
    horizon: Mapping[str, Any]
    terminal: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": "e07.s02.observation-envelope.v1",
            "taskId": self.task_id,
            "scenarioId": self.scenario_id,
            "split": self.split.value,
            "environmentProfile": self.environment_profile,
            "actionContract": dict(self.action_contract),
            "observationContract": dict(self.observation_contract),
            "horizon": dict(self.horizon),
            "terminal": self.terminal,
        }


@dataclass(frozen=True, slots=True)
class EvaluationAction:
    policy_id: str
    policy_sha256: str
    mode: str = "native_baseline"

    def __post_init__(self) -> None:
        if not self.policy_id:
            raise SuiteValidationError("policy ID must be nonempty")
        if len(self.policy_sha256) != 64 or any(
            ch not in "0123456789abcdef" for ch in self.policy_sha256
        ):
            raise SuiteValidationError("policy hash must be lowercase SHA-256")
        if self.mode != "native_baseline":
            raise SuiteValidationError("S02 validates native_baseline actions only")


@dataclass(frozen=True, slots=True)
class NativeEpisodeResult:
    stop_reason: str
    censored: bool
    failed: bool
    native_costs: Mapping[str, Mapping[str, int | float]]
    native_event: Mapping[str, Any]
    native_outcome: Mapping[str, Any]
    replay_pass: bool
    validation: Mapping[str, bool]
    provenance: Mapping[str, str]

    def __post_init__(self) -> None:
        if not self.stop_reason:
            raise SuiteValidationError("native stop reason must be explicit")
        for family, ledger in self.native_costs.items():
            if not family or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or value < 0
                for value in ledger.values()
            ):
                raise SuiteValidationError("native cost ledgers must be nonnegative")


@dataclass(frozen=True, slots=True)
class StepRecord:
    task_id: str
    scenario_id: str
    split: Split
    native_unit: str
    stop_reason: str
    terminal: bool
    censored: bool
    failed: bool
    cost: Mapping[str, Any]
    event: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": "e07.s02.step-record.v1",
            "taskId": self.task_id,
            "scenarioId": self.scenario_id,
            "split": self.split.value,
            "nativeUnit": self.native_unit,
            "stopReason": self.stop_reason,
            "terminal": self.terminal,
            "censored": self.censored,
            "failed": self.failed,
            "cost": dict(self.cost),
            "event": dict(self.event),
        }


@dataclass(frozen=True, slots=True)
class OutcomeEnvelope:
    task_id: str
    scenario_id: str
    split: Split
    stop_reason: str
    censored: bool
    failed: bool
    outcome: Mapping[str, Any]
    claim_boundary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": "e07.s02.outcome-envelope.v1",
            "taskId": self.task_id,
            "scenarioId": self.scenario_id,
            "split": self.split.value,
            "stopReason": self.stop_reason,
            "censored": self.censored,
            "failed": self.failed,
            "outcome": dict(self.outcome),
            "claimBoundary": self.claim_boundary,
        }


def _expect_exact_keys(
    raw: Mapping[str, Any], expected: set[str], context: str
) -> None:
    if set(raw) != expected:
        raise SuiteValidationError(
            f"{context} keys mismatch: missing={sorted(expected - set(raw))}, "
            f"extra={sorted(set(raw) - expected)}"
        )


def _parse_horizon(raw: Mapping[str, Any]) -> HorizonContract:
    _expect_exact_keys(
        raw,
        {
            "nativeUnit",
            "adapterStepGranularity",
            "budgets",
            "terminalPrecedence",
            "censoringRule",
            "comparabilityGroup",
            "comparableOnlyWithinGroup",
            "crossTaskNormalization",
        },
        "horizon",
    )
    return HorizonContract(
        native_unit=str(raw["nativeUnit"]),
        adapter_step_granularity=str(raw["adapterStepGranularity"]),
        budgets=MappingProxyType(dict(raw["budgets"])),
        terminal_precedence=tuple(map(str, raw["terminalPrecedence"])),
        censoring_rule=str(raw["censoringRule"]),
        comparability_group=str(raw["comparabilityGroup"]),
        comparable_only_within_group=bool(raw["comparableOnlyWithinGroup"]),
        cross_task_normalization=str(raw["crossTaskNormalization"]),
    )


def load_task_registry(
    path: str | Path,
) -> tuple[Mapping[str, Any], Mapping[str, TaskContract]]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if raw.get("schemaVersion") != TASK_REGISTRY_VERSION:
        raise SuiteValidationError("task registry schema version mismatch")
    if raw.get("suiteVersion") != SUITE_VERSION:
        raise SuiteValidationError("suite version mismatch")
    tasks: dict[str, TaskContract] = {}
    for item in raw.get("tasks", []):
        contract = TaskContract(
            task_id=str(item["taskId"]),
            title=str(item["title"]),
            family=str(item["family"]),
            predecessor=str(item["predecessor"]),
            runner_id=str(item["runnerId"]),
            environment_profile=str(item["environmentProfile"]),
            legality_authority=str(item["legalityAuthority"]),
            action_contract=MappingProxyType(dict(item["actionContract"])),
            observation_contract=MappingProxyType(dict(item["observationContract"])),
            cost_contract=MappingProxyType(dict(item["costContract"])),
            event_contract=MappingProxyType(dict(item["eventContract"])),
            stopping_contract=MappingProxyType(dict(item["stoppingContract"])),
            outcome_contract=MappingProxyType(dict(item["outcomeContract"])),
            claim_boundary=str(item["claimBoundary"]),
            horizon=_parse_horizon(item["horizon"]),
            optional_exclusions=tuple(map(str, item.get("optionalExclusions", []))),
        )
        if contract.task_id in tasks:
            raise SuiteValidationError("duplicate task ID")
        tasks[contract.task_id] = contract
    if len(tasks) < 7:
        raise SuiteValidationError(
            "registry must cover all seven required task families"
        )
    metadata = {key: value for key, value in raw.items() if key != "tasks"}
    return MappingProxyType(metadata), MappingProxyType(tasks)


def load_split_manifest(
    path: str | Path, tasks: Mapping[str, TaskContract]
) -> tuple[Mapping[str, Any], Mapping[str, ScenarioRecord]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schemaVersion") != SPLIT_MANIFEST_VERSION:
        raise SuiteValidationError("split manifest schema version mismatch")
    if raw.get("assignmentUsesOutcomes") is not False:
        raise SuiteValidationError("split assignment may not use outcomes")
    if raw.get("confirmationOutcomeAccess") != "sealed":
        raise SuiteValidationError("confirmation outcomes must remain sealed")
    records: dict[str, ScenarioRecord] = {}
    seen_by_task: dict[str, set[Split]] = {task_id: set() for task_id in tasks}
    for item in raw.get("scenarios", []):
        record = ScenarioRecord(
            scenario_id=str(item["scenarioId"]),
            task_id=str(item["taskId"]),
            split=Split(item["split"]),
            materializer_id=item.get("materializerId"),
            public_parameters=MappingProxyType(dict(item["publicParameters"])),
            protected=bool(item["protected"]),
            outcome_access=str(item["outcomeAccess"]),
            predecessor_partition=str(item["predecessorPartition"]),
        )
        if record.task_id not in tasks:
            raise SuiteValidationError("split record refers to unknown task")
        if record.scenario_id in records:
            raise SuiteValidationError("duplicate scenario ID")
        records[record.scenario_id] = record
        seen_by_task[record.task_id].add(record.split)
    required = {Split.TRAIN, Split.VALIDATION, Split.CONFIRMATION}
    if any(splits != required for splits in seen_by_task.values()):
        raise SuiteValidationError(
            "every task must freeze train/validation/confirmation"
        )
    metadata = {key: value for key, value in raw.items() if key != "scenarios"}
    return MappingProxyType(metadata), MappingProxyType(records)
