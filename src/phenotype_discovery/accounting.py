"""Fail-atomic, all-row accounting for S10 native-event execution.

The accounting plane is deliberately independent of scientific outcomes.  It
precommits one logical identity and two physical replay identities per
reservation, records a terminal disposition for every precommitted position,
and permits result publication only when every logical pair succeeds and
replays exactly.
"""

from __future__ import annotations

from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from copy import deepcopy
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence


PhysicalEvaluator = Callable[[Mapping[str, Any]], Mapping[str, Any]]
LogicalFinalizer = Callable[
    [Mapping[str, Any], Sequence[Mapping[str, Any]]], Mapping[str, Any]
]


class FailAtomicPhaseError(RuntimeError):
    """Raised only after the complete forensic disposition ledger is durable."""

    def __init__(self, message: str, *, accounting: Mapping[str, Any]):
        super().__init__(message)
        self.accounting = dict(accounting)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _commit(domain: str, value: Any) -> str:
    digest = hashlib.sha256()
    digest.update(domain.encode("ascii"))
    digest.update(b"\0")
    digest.update(_canonical_json(value))
    return digest.hexdigest()


def _atomic_json(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(
            value,
            handle,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, target)


def build_physical_replay_plan(
    reservations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build two distinct, pre-outcome physical commitments per logical row."""

    logical_ids = [str(row["logicalReservationId"]) for row in reservations]
    if len(set(logical_ids)) != len(logical_ids):
        raise ValueError("duplicate logical reservation identity")
    plan: list[dict[str, Any]] = []
    for logical_position, reservation in enumerate(reservations):
        reservation_body = deepcopy(dict(reservation))
        reservation_commitment = _commit(
            "E07/S10A/logical-reservation-commitment/v1", reservation_body
        )
        for replay_ordinal in range(2):
            identity_body = {
                "logicalPosition": logical_position,
                "logicalReservationId": str(reservation["logicalReservationId"]),
                "reservationCommitmentSha256": reservation_commitment,
                "replayOrdinal": replay_ordinal,
            }
            plan.append(
                {
                    "physicalPosition": 2 * logical_position + replay_ordinal,
                    **identity_body,
                    "physicalExecutionId": _commit(
                        "E07/S10A/physical-replay-identity/v1", identity_body
                    ),
                    "reservation": reservation_body,
                }
            )
    if len({row["physicalExecutionId"] for row in plan}) != 2 * len(reservations):
        raise RuntimeError("physical replay identities are not unique")
    return plan


def _precommit_document(
    reservations: Sequence[Mapping[str, Any]],
    physical_plan: Sequence[Mapping[str, Any]],
    *,
    phase: str,
) -> dict[str, Any]:
    logical = [
        {
            "logicalPosition": index,
            "logicalReservationId": str(row["logicalReservationId"]),
            "reservationCommitmentSha256": physical_plan[2 * index][
                "reservationCommitmentSha256"
            ],
            "state": "planned",
        }
        for index, row in enumerate(reservations)
    ]
    physical = [
        {
            key: row[key]
            for key in (
                "physicalPosition",
                "logicalPosition",
                "logicalReservationId",
                "reservationCommitmentSha256",
                "replayOrdinal",
                "physicalExecutionId",
            )
        }
        | {"state": "planned"}
        for row in physical_plan
    ]
    body: dict[str, Any] = {
        "schemaVersion": "e07.s10a.phase-disposition-ledger.v1",
        "researchStepId": "S10A",
        "phase": phase,
        "publicationState": "not_evaluated",
        "failAtomic": True,
        "logicalReservationCount": len(reservations),
        "physicalReplayCount": len(physical_plan),
        "replaysPerLogicalReservation": 2,
        "logicalDispositions": logical,
        "physicalDispositions": physical,
    }
    body["precommitSha256"] = _commit("E07/S10A/phase-disposition-precommit/v1", body)
    return body


def _validate_success_payload(payload: Mapping[str, Any]) -> None:
    required = {
        "resultBody",
        "deterministicResultSha256",
        "nativeReplaySha256",
        "availabilitySummary",
    }
    if not required.issubset(payload):
        raise ValueError(
            f"physical evaluator omitted {sorted(required - set(payload))}"
        )
    if not isinstance(payload["resultBody"], Mapping):
        raise TypeError("physical resultBody must be a mapping")
    if not isinstance(payload["availabilitySummary"], Mapping):
        raise TypeError("physical availabilitySummary must be a mapping")
    for key in ("deterministicResultSha256", "nativeReplaySha256"):
        value = str(payload[key])
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(f"{key} must be a lowercase SHA-256")


def execute_fail_atomic_replays(
    reservations: Sequence[Mapping[str, Any]],
    evaluator: PhysicalEvaluator,
    *,
    disposition_path: str | Path,
    phase: str,
    workers: int,
    finalizer: LogicalFinalizer | None = None,
    executor_kind: str = "process",
    initializer: Callable[[], None] | None = None,
    submission_order: Sequence[int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate two replays per row with complete durable dispositions.

    The disposition precommit is written before any evaluator is submitted.
    Every future is converted to a terminal success or exception record.
    Result bodies are returned only if all pairs succeed and both deterministic
    commitments match exactly.
    """

    if not 1 <= workers <= 8:
        raise ValueError("workers must be 1..8")
    plan = build_physical_replay_plan(reservations)
    precommit = _precommit_document(reservations, plan, phase=phase)
    _atomic_json(disposition_path, precommit)
    order = (
        list(range(len(plan))) if submission_order is None else list(submission_order)
    )
    if sorted(order) != list(range(len(plan))):
        raise ValueError("submission_order must be an exact physical-plan permutation")
    if executor_kind == "process":
        executor_class = ProcessPoolExecutor
        executor_kwargs: dict[str, Any] = {
            "initializer": initializer,
            "mp_context": multiprocessing.get_context("spawn"),
        }
    elif executor_kind == "thread":
        executor_class = ThreadPoolExecutor
        executor_kwargs = {}
        if initializer is not None:
            initializer()
    else:
        raise ValueError("executor_kind must be 'process' or 'thread'")

    started = time.perf_counter()
    payload_by_position: dict[int, Mapping[str, Any]] = {}
    disposition_by_position: dict[int, dict[str, Any]] = {}
    executor_error: BaseException | None = None
    try:
        with executor_class(max_workers=workers, **executor_kwargs) as executor:
            futures = {
                executor.submit(evaluator, plan[position]): position
                for position in order
            }
            for future in as_completed(futures):
                position = futures[future]
                physical = plan[position]
                base = {
                    key: physical[key]
                    for key in (
                        "physicalPosition",
                        "logicalPosition",
                        "logicalReservationId",
                        "reservationCommitmentSha256",
                        "replayOrdinal",
                        "physicalExecutionId",
                    )
                }
                try:
                    payload = future.result()
                    if not isinstance(payload, Mapping):
                        raise TypeError("physical evaluator result must be a mapping")
                    _validate_success_payload(payload)
                    payload_by_position[position] = payload
                    disposition_by_position[position] = {
                        **base,
                        "state": "success",
                        "deterministicResultSha256": str(
                            payload["deterministicResultSha256"]
                        ),
                        "nativeReplaySha256": str(payload["nativeReplaySha256"]),
                        "availabilitySummary": deepcopy(
                            dict(payload["availabilitySummary"])
                        ),
                    }
                except BaseException as exc:
                    disposition_by_position[position] = {
                        **base,
                        "state": "evaluator_exception",
                        "errorType": (
                            f"{type(exc).__module__}.{type(exc).__qualname__}"
                        ),
                        "errorMessage": str(exc),
                        "availabilitySummary": {
                            "state": "unavailable_due_to_evaluator_exception",
                            "observedFeatureCount": 0,
                            "unavailableFeatureCount": None,
                            "reasonCodeCounts": {
                                "evaluator_exception_before_complete_feature_record": 1
                            },
                        },
                    }
    except BaseException as exc:
        executor_error = exc

    # A broken executor should still leave one terminal record per precommit.
    for position, physical in enumerate(plan):
        if position not in disposition_by_position:
            disposition_by_position[position] = {
                **{
                    key: physical[key]
                    for key in (
                        "physicalPosition",
                        "logicalPosition",
                        "logicalReservationId",
                        "reservationCommitmentSha256",
                        "replayOrdinal",
                        "physicalExecutionId",
                    )
                },
                "state": "executor_exception",
                "errorType": (
                    f"{type(executor_error).__module__}."
                    f"{type(executor_error).__qualname__}"
                    if executor_error is not None
                    else "e07.s10a.MissingFutureDisposition"
                ),
                "errorMessage": (
                    str(executor_error)
                    if executor_error is not None
                    else "executor returned no terminal disposition"
                ),
                "availabilitySummary": {
                    "state": "unavailable_due_to_executor_exception",
                    "observedFeatureCount": 0,
                    "unavailableFeatureCount": None,
                    "reasonCodeCounts": {"executor_exception_before_feature_record": 1},
                },
            }

    physical_dispositions = [
        disposition_by_position[position] for position in range(len(plan))
    ]
    logical_dispositions: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for logical_position, reservation in enumerate(reservations):
        pair = physical_dispositions[2 * logical_position : 2 * logical_position + 2]
        success = [row["state"] == "success" for row in pair]
        if not all(success):
            state = "physical_failure"
            reason = "one_or_more_physical_replays_failed"
        else:
            commitments = {
                (
                    row["deterministicResultSha256"],
                    row["nativeReplaySha256"],
                )
                for row in pair
            }
            if len(commitments) != 1:
                state = "replay_mismatch"
                reason = "physical_replay_commitments_differ"
            else:
                state = "success"
                reason = None
        logical_disposition: dict[str, Any] = {
            "logicalPosition": logical_position,
            "logicalReservationId": str(reservation["logicalReservationId"]),
            "reservationCommitmentSha256": pair[0]["reservationCommitmentSha256"],
            "physicalExecutionIds": [str(row["physicalExecutionId"]) for row in pair],
            "state": state,
            "reasonCode": reason,
        }
        if state == "success":
            payloads = [
                payload_by_position[2 * logical_position + replay]
                for replay in range(2)
            ]
            try:
                if finalizer is None:
                    result = deepcopy(dict(payloads[0]["resultBody"]))
                else:
                    result = deepcopy(dict(finalizer(reservation, payloads)))
                results.append(result)
            except BaseException as exc:
                logical_disposition.update(
                    {
                        "state": "result_finalization_failure",
                        "reasonCode": "logical_result_finalizer_raised",
                        "errorType": (
                            f"{type(exc).__module__}.{type(exc).__qualname__}"
                        ),
                        "errorMessage": str(exc),
                    }
                )
        logical_dispositions.append(logical_disposition)

    all_success = all(row["state"] == "success" for row in logical_dispositions)
    published_rows = len(results) if all_success else 0
    final = {
        **{
            key: value
            for key, value in precommit.items()
            if key not in {"logicalDispositions", "physicalDispositions"}
        },
        "publicationState": "all_rows_releasable" if all_success else "withheld",
        "resultRowsPublishedByExecutor": 0,
        "resultRowsEligibleForCallerPublication": published_rows,
        "logicalDispositions": logical_dispositions,
        "physicalDispositions": physical_dispositions,
        "terminalLogicalDispositionCount": len(logical_dispositions),
        "terminalPhysicalDispositionCount": len(physical_dispositions),
        "successfulLogicalCount": sum(
            row["state"] == "success" for row in logical_dispositions
        ),
        "failedLogicalCount": sum(
            row["state"] != "success" for row in logical_dispositions
        ),
        "successfulPhysicalCount": sum(
            row["state"] == "success" for row in physical_dispositions
        ),
        "failedPhysicalCount": sum(
            row["state"] != "success" for row in physical_dispositions
        ),
        "accountingConserved": (
            len(logical_dispositions) == len(reservations)
            and len(physical_dispositions) == 2 * len(reservations)
        ),
        "workers": workers,
        "numericThreadsPerWorker": 1,
        "wallSeconds": time.perf_counter() - started,
    }
    final["dispositionSha256"] = _commit(
        "E07/S10A/phase-disposition-final/v1",
        {key: value for key, value in final.items() if key != "wallSeconds"},
    )
    _atomic_json(disposition_path, final)

    accounting = {
        "logicalRows": len(reservations),
        "physicalExecutions": 2 * len(reservations),
        "publishedRows": published_rows,
        "failedWorkerRows": final["failedPhysicalCount"],
        "workers": workers,
        "numericThreadsPerWorker": 1,
        "failAtomic": True,
        "completePerRowDispositionAccounting": final["accountingConserved"],
        "dispositionPath": str(Path(disposition_path)),
        "dispositionSha256": final["dispositionSha256"],
        "wallSeconds": final["wallSeconds"],
    }
    if not all_success:
        failures = [
            {
                "logicalPosition": row["logicalPosition"],
                "logicalReservationId": row["logicalReservationId"],
                "state": row["state"],
                "reasonCode": row["reasonCode"],
            }
            for row in logical_dispositions
            if row["state"] != "success"
        ]
        raise FailAtomicPhaseError(
            "S10 fail-atomic phase execution withheld all result rows; "
            f"{len(failures)} logical dispositions failed; "
            f"ledger={Path(disposition_path)}",
            accounting=accounting,
        )
    return results, accounting
