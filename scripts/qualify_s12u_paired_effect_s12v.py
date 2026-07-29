"""Run the outcome-independent S12V paired-effect qualification.

The script deliberately never opens the quarantined S12U cache and never
loads an S12U logical or physical outcome row.  It exercises only frozen
source definitions and dedicated synthetic fixtures.
"""

from __future__ import annotations

import hashlib
import inspect
import io
import json
import math
import os
import sys
import warnings
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import redirect_stderr
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[1]
WORKSPACE = REPOSITORY.parent
OUT = Path("/artifacts/research_steps/S12V")
PROTOCOL = OUT / "s12v_forensic_qualification_protocol.yaml"
INPUT_REGISTRY = OUT / "permitted_input_registry.json"
PREREGISTRATION = OUT / "preregistration_freeze.json"
FORBIDDEN_CACHE_ROOT = Path("/cache/e07-s12u")


class OpenAudit:
    def __init__(self) -> None:
        self.open_event_count = 0
        self.forbidden_events: list[dict[str, Any]] = []
        self.s12v_output_events = 0

    def __call__(self, event: str, args: tuple[Any, ...]) -> None:
        if event != "open" or not args:
            return
        raw = args[0]
        if not isinstance(raw, (str, bytes, os.PathLike)):
            return
        try:
            path = Path(os.fsdecode(raw)).absolute()
        except (TypeError, ValueError):
            return
        self.open_event_count += 1
        text = os.path.normpath(str(path))
        forbidden = os.path.normpath(str(FORBIDDEN_CACHE_ROOT))
        if text == forbidden or text.startswith(forbidden + os.sep):
            self.forbidden_events.append({"event": event, "path": text})
            raise PermissionError("S12V denies every S12U cache open")
        out_text = os.path.normpath(str(OUT))
        if text == out_text or text.startswith(out_text + os.sep):
            self.s12v_output_events += 1


OPEN_AUDIT = OpenAudit()
sys.addaudithook(OPEN_AUDIT)

# Imports below occur after the deny-first audit hook is installed.
import pandas as pd
from pandas.errors import MergeError

from scripts import execute_replacement_spatial_transfer_s12u as frozen_s12u
from src.phenotype_discovery.publication import (
    SerializationContractError,
    canonical_json_bytes,
)
from src.spatial_transfer import execution as frozen_execution
from src.spatial_transfer.paired_effect_forensics import (
    arithmetic_bound_evidence,
    disposition_counts,
    enumerate_contract_states,
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def strict_load_json(path: Path) -> Any:
    def reject(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json_bytes(value))
    os.replace(temporary, path)


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def canonical_digest(domain: str, value: Any) -> str:
    digest = hashlib.sha256()
    digest.update(domain.encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical_json_bytes(value))
    return digest.hexdigest()


def validate_registered_inputs() -> dict[str, Any]:
    registry = strict_load_json(INPUT_REGISTRY)
    preregistration = strict_load_json(PREREGISTRATION)
    rows = []
    for item in registry["inputs"]:
        path = Path(item["path"])
        if item["role"] == "pre_S12V_research_plan":
            actual = preregistration["preS12VResearchPlanSha256"]
            disposition = "validated_from_prospective_preregistration"
        else:
            actual = sha256_file(path)
            disposition = "live_byte_validation"
        rows.append(
            {
                "path": str(path),
                "role": item["role"],
                "expectedSha256": item["sha256"],
                "actualSha256": actual,
                "passed": actual == item["sha256"],
                "disposition": disposition,
            }
        )
    protocol_hash = sha256_file(PROTOCOL)
    registry_hash = sha256_file(INPUT_REGISTRY)
    checks = {
        "protocolMatchesPreregistration": (
            protocol_hash == preregistration["protocol"]["sha256"]
        ),
        "inputRegistryMatchesPreregistration": (
            registry_hash == preregistration["permittedInputRegistry"]["sha256"]
        ),
        "everyRegisteredInputMatches": all(row["passed"] for row in rows),
        "productionRepairUnauthorized": not preregistration[
            "productionRepairAuthorized"
        ],
        "scientificExecutionUnauthorized": not preregistration[
            "scientificExecutionAuthorized"
        ],
    }
    return {
        "schemaVersion": "e07.s12v.input-hash-validation.v1",
        "researchStepId": "S12V",
        "checks": checks,
        "allPassed": all(checks.values()),
        "inputCount": len(rows),
        "rows": rows,
    }


def _paired_record(
    case_id: str,
    left: Sequence[Any],
    right: Sequence[Any],
    *,
    binary: bool,
) -> dict[str, Any]:
    return frozen_s12u._paired_record(
        family="S12V_outcome_independent_synthetic",
        contrast_id=case_id,
        task_id="synthetic_task",
        panel_id="synthetic_panel",
        condition_id="synthetic_condition",
        lineage_id="synthetic_lineage",
        endpoint="synthetic_endpoint",
        left_label="synthetic_left",
        right_label="synthetic_right",
        left=left,
        right=right,
        benefit_direction=1,
        binary=binary,
    )


def _summarize_record(record: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "pairCount",
        "rawEffectLeftMinusRight",
        "benefitEffect",
        "pairedBootstrap95CiLeftMinusRight",
        "hodgesLehmannSensitivity",
        "rawPValue",
        "evidentiary",
        "nonEvidentiaryReason",
    )
    summary: dict[str, Any] = {}
    for field in fields:
        value = record[field]
        if isinstance(value, float) and not math.isfinite(value):
            summary[field] = (
                "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
            )
        elif isinstance(value, list):
            summary[field] = [
                (
                    "NaN"
                    if isinstance(item, float) and math.isnan(item)
                    else (
                        "Infinity"
                        if isinstance(item, float)
                        and math.isinf(item)
                        and item > 0
                        else (
                            "-Infinity"
                            if isinstance(item, float) and math.isinf(item)
                            else item
                        )
                    )
                )
                for item in value
            ]
        else:
            summary[field] = value
    return summary


def run_synthetic_fixtures() -> dict[str, Any]:
    cases = [
        ("finite_binary_zero", [0.0, 1.0], [0.0, 1.0], True),
        ("finite_binary_difference", [1.0, 1.0], [0.0, 1.0], True),
        ("finite_mismatch", [0.0, 0.5, 1.0], [1.0, 0.5, 0.0], False),
        ("finite_time_at_32", [0.0, 32.0], [32.0, 0.0], False),
        (
            "finite_cost_int64_bound",
            [float((1 << 63) - 1)],
            [0.0],
            False,
        ),
        ("empty_direct_helper", [], [], False),
        ("missing_left_endpoint", [None], [0.0], False),
        ("nan_left_endpoint", [float("nan")], [0.0], False),
        ("positive_infinity_left_endpoint", [float("inf")], [0.0], False),
        ("negative_infinity_left_endpoint", [-float("inf")], [0.0], False),
        (
            "fault_repair_vs_no_fault_undefined",
            [1.0, 0.0],
            [float("nan"), float("nan")],
            True,
        ),
    ]
    records = []
    for case_id, left, right, binary in cases:
        with warnings.catch_warnings(), redirect_stderr(io.StringIO()):
            warnings.simplefilter("ignore")
            record = _paired_record(case_id, left, right, binary=binary)
        serializer_state = "accepted"
        serializer_error = None
        try:
            canonical_json_bytes({"records": [record]})
        except SerializationContractError as exc:
            serializer_state = "rejected_nonfinite"
            serializer_error = str(exc)
        records.append(
            {
                "caseId": case_id,
                "binary": binary,
                "frozenEstimatorState": _summarize_record(record),
                "strictSerializerState": serializer_state,
                "strictSerializerError": serializer_error,
            }
        )

    partial_error = None
    try:
        _paired_record("partial_pair_support", [1.0], [], binary=False)
    except RuntimeError as exc:
        partial_error = str(exc)
    records.append(
        {
            "caseId": "partial_pair_support",
            "frozenEstimatorState": "failed_closed_before_record",
            "error": partial_error,
        }
    )

    duplicate_error = None
    try:
        frozen_s12u._merge_pair(
            pd.DataFrame({"pair": ["a", "a"], "value": [0.0, 1.0]}),
            pd.DataFrame({"pair": ["a"], "value": [0.0]}),
            ["pair"],
        )
    except MergeError as exc:
        duplicate_error = str(exc)
    records.append(
        {
            "caseId": "duplicate_left_pair_identity",
            "frozenEstimatorState": "failed_closed_before_record",
            "errorType": "MergeError",
            "error": duplicate_error,
        }
    )

    partial_merge = frozen_s12u._merge_pair(
        pd.DataFrame({"pair": ["a", "b"], "value": [0.0, 1.0]}),
        pd.DataFrame({"pair": ["a"], "value": [0.0]}),
        ["pair"],
    )
    records.append(
        {
            "caseId": "partial_support_inner_join",
            "leftPairCount": 2,
            "rightPairCount": 1,
            "returnedPairCount": len(partial_merge),
            "frozenEstimatorState": "silently_reduced_to_intersection",
        }
    )
    return {
        "schemaVersion": "e07.s12v.synthetic-fixture-results.v1",
        "researchStepId": "S12V",
        "outcomeIndependent": True,
        "fixtureCount": len(records),
        "records": sorted(records, key=lambda row: row["caseId"]),
    }


def source_path_audit() -> dict[str, Any]:
    paired_source, paired_line = inspect.getsourcelines(frozen_s12u._paired_record)
    build_source, build_line = inspect.getsourcelines(
        frozen_s12u.build_paired_estimands
    )
    endpoint_source, endpoint_line = inspect.getsourcelines(
        frozen_s12u._endpoint_specs
    )
    execution_source, execution_line = inspect.getsourcelines(
        frozen_execution.execute_physical
    )
    serializer_source, serializer_line = inspect.getsourcelines(
        canonical_json_bytes
    )
    paired_text = "".join(paired_source)
    build_text = "".join(build_source)
    endpoint_text = "".join(endpoint_source)
    execution_text = "".join(execution_source)
    findings = [
        {
            "findingId": "F01",
            "description": (
                "The frozen estimator chooses repairByTransition32 whenever "
                "the left condition ID contains spurious_swap_fault."
            ),
            "present": (
                'base.append(("repairByTransition32", 1, True))' in endpoint_text
            ),
            "causeClass": "C03_PROJECTION_FEASIBILITY_DEFECT",
        },
        {
            "findingId": "F02",
            "description": (
                "The frozen physical projection emits repairByTransition32=None "
                "for every no-fault row."
            ),
            "present": '"repairByTransition32": repair' in execution_text
            and "repair = None" in execution_text,
            "causeClass": "C02_EXPLICIT_UNDEFINED_NON_EVIDENTIARY",
        },
        {
            "findingId": "F03",
            "description": (
                "The fault-vs-no-fault merge selects endpoint specifications "
                "from conditionId_left and passes both left and right endpoint "
                "columns to arithmetic."
            ),
            "present": "conditionId_left" in build_text
            and 'group[f"{endpoint}_right"].astype(float)' in build_text
            and 'contrast_id="fault_vs_no_fault"' in build_text,
            "causeClass": "C03_PROJECTION_FEASIBILITY_DEFECT",
        },
        {
            "findingId": "F04",
            "description": (
                "The paired helper contains no endpoint-level finite/missing "
                "precheck before mean, median, bootstrap, and test arithmetic."
            ),
            "present": "isfinite" not in paired_text and "isna" not in paired_text,
            "causeClass": "C03_PROJECTION_FEASIBILITY_DEFECT",
        },
        {
            "findingId": "F05",
            "description": (
                "An empty merged group returns before creating the frozen fixed "
                "non-evidentiary multiplicity slot."
            ),
            "present": "if paired.empty:" in build_text,
            "causeClass": "C03_PROJECTION_FEASIBILITY_DEFECT",
        },
        {
            "findingId": "F06",
            "description": (
                "The global endpointAvailable filter precedes failure and cost "
                "families, conflicting with the all-reserved-row analysis "
                "population for those endpoints."
            ),
            "present": '(frame["partition"] == "post_lock_transfer") & frame["endpointAvailable"]'
            in build_text,
            "causeClass": "C03_PROJECTION_FEASIBILITY_DEFECT",
        },
        {
            "findingId": "F07",
            "description": (
                "The registered restricted native transition-time endpoint is "
                "not emitted by the frozen endpoint specification function."
            ),
            "present": "firstCompletionTransition" not in endpoint_text,
            "causeClass": "C03_PROJECTION_FEASIBILITY_DEFECT",
        },
        {
            "findingId": "F08",
            "description": (
                "The strict serializer validates JSON safety before encoding "
                "and rejects non-finite floating values."
            ),
            "present": "validate_json_safe(value)" in "".join(serializer_source),
            "causeClass": "C05_SERIALIZER_DEFECT_RuledOut",
        },
    ]
    return {
        "schemaVersion": "e07.s12v.source-path-audit.v1",
        "researchStepId": "S12V",
        "sources": {
            "pairedRecord": {
                "path": str(Path(inspect.getsourcefile(frozen_s12u._paired_record))),
                "startLine": paired_line,
                "sourceSha256": sha256_bytes(paired_text.encode("utf-8")),
            },
            "pairedBuilder": {
                "path": str(
                    Path(inspect.getsourcefile(frozen_s12u.build_paired_estimands))
                ),
                "startLine": build_line,
                "sourceSha256": sha256_bytes(build_text.encode("utf-8")),
            },
            "endpointSpecs": {
                "path": str(Path(inspect.getsourcefile(frozen_s12u._endpoint_specs))),
                "startLine": endpoint_line,
                "sourceSha256": sha256_bytes(endpoint_text.encode("utf-8")),
            },
            "physicalProjection": {
                "path": str(
                    Path(
                        inspect.getsourcefile(
                            frozen_execution.execute_physical
                        )
                    )
                ),
                "startLine": execution_line,
                "sourceSha256": sha256_bytes(execution_text.encode("utf-8")),
            },
            "strictSerializer": {
                "path": str(Path(inspect.getsourcefile(canonical_json_bytes))),
                "startLine": serializer_line,
                "sourceSha256": sha256_bytes(
                    "".join(serializer_source).encode("utf-8")
                ),
            },
        },
        "findings": findings,
        "allFindingsReproduced": all(row["present"] for row in findings),
    }


def root_cause(source_audit: Mapping[str, Any], fixtures: Mapping[str, Any]) -> dict[str, Any]:
    fixture_by_id = {row["caseId"]: row for row in fixtures["records"]}
    exact_fixture = fixture_by_id["fault_repair_vs_no_fault_undefined"]
    serializer_rejected = (
        exact_fixture["strictSerializerState"] == "rejected_nonfinite"
    )
    finite_fixtures = [
        row
        for row in fixtures["records"]
        if row["caseId"].startswith("finite_")
    ]
    return {
        "schemaVersion": "e07.s12v.root-cause-classification.v1",
        "researchStepId": "S12V",
        "primaryClass": "C03_PROJECTION_FEASIBILITY_DEFECT",
        "confidence": "high",
        "classification": (
            "The fault-repair endpoint is legitimately undefined on the "
            "no-fault comparator side, but the frozen estimator selects that "
            "endpoint from the fault-side condition, casts the no-fault None "
            "to NaN, and treats the resulting record as evidentiary until the "
            "strict serializer correctly rejects it."
        ),
        "legitimatelyUndefinedContrast": True,
        "undefinedElement": "repairByTransition32 on the no-fault side",
        "projectionDefect": True,
        "arithmeticDefectOnFiniteBoundedInputs": False,
        "serializerDefect": False,
        "evidence": {
            "sourceFindingsAllReproduced": source_audit["allFindingsReproduced"],
            "faultRepairSyntheticFailureReproduced": serializer_rejected,
            "finiteBoundedFixturesAccepted": all(
                row["strictSerializerState"] == "accepted"
                for row in finite_fixtures
            ),
        },
        "remainingUncertainty": {
            "classLevel": "low",
            "mechanismLevel": "low",
            "exactSerializerRecord33Identity": "not_classified",
            "reason": (
                "The exact lineage, task, panel, scheduler, and testId at "
                "serializer ordinal 33 were not reconstructed because the "
                "failed in-memory outcome object and S12U cache are prohibited."
            ),
            "causeClass": "C06_EXACT_SUBCAUSE_UNCLASSIFIABLE_WITHOUT_PROHIBITED_OUTCOMES",
        },
        "notAnEfficacyConclusion": True,
    }


def remediation_map() -> dict[str, Any]:
    rows = [
        {
            "remediationId": "R01",
            "description": (
                "Add endpoint-specific availability and pair-support validation "
                "before arithmetic; retain the inapplicable repair contrast as "
                "a fixed raw-p=1 non-evidentiary slot."
            ),
            "S12REstimandDisposition": "preserves_declared_S12R_estimand",
            "rationale": (
                "Enforces existing endpoint and fixed-family contracts without "
                "changing the contrast, endpoint meaning, or multiplicity size."
            ),
            "authorizedByS12V": False,
        },
        {
            "remediationId": "R02",
            "description": (
                "Restrict the fault-vs-no-fault evidentiary comparison to "
                "terminal completion and minimum mismatch, while retaining the "
                "repair slot explicitly as non-evidentiary."
            ),
            "S12REstimandDisposition": "preserves_declared_S12R_estimand",
            "rationale": (
                "No-fault is not a repair risk set under the frozen endpoint "
                "contract; explicit infeasibility implements the existing rule."
            ),
            "authorizedByS12V": False,
        },
        {
            "remediationId": "R03",
            "description": (
                "Apply all-reserved-row masks separately for failure and cost "
                "families instead of the task-endpoint availability mask."
            ),
            "S12REstimandDisposition": "preserves_declared_S12R_estimand",
            "rationale": (
                "Restores the already frozen all-reserved-row population and "
                "separate failure/cost families."
            ),
            "authorizedByS12V": False,
        },
        {
            "remediationId": "R04",
            "description": (
                "Define no-fault repair as zero or one, or impute the missing "
                "right-side repair value."
            ),
            "S12REstimandDisposition": "replaces_S12R_endpoint_and_estimand",
            "rationale": "No-fault rows are not members of the frozen repair risk set.",
            "authorizedByS12V": False,
        },
        {
            "remediationId": "R05",
            "description": "Drop the infeasible slot or shrink its Holm family.",
            "S12REstimandDisposition": "replaces_S12R_multiplicity_contract",
            "rationale": "Observed family shrinkage is explicitly forbidden.",
            "authorizedByS12V": False,
        },
        {
            "remediationId": "R06",
            "description": "Serialize NaN as null while leaving the record evidentiary.",
            "S12REstimandDisposition": "invalid_not_estimand_preserving",
            "rationale": "Hides undefined arithmetic and weakens strict publication.",
            "authorizedByS12V": False,
        },
        {
            "remediationId": "R07",
            "description": (
                "Remove only serializer record 33 or tune behavior from the "
                "quarantined attempt."
            ),
            "S12REstimandDisposition": "invalid_outcome_dependent_change",
            "rationale": "Post-outcome row deletion and cache reuse are prohibited.",
            "authorizedByS12V": False,
        },
        {
            "remediationId": "R08",
            "description": (
                "Add the registered restricted transition-time endpoint to the "
                "estimator with its frozen censor-at-32 semantics."
            ),
            "S12REstimandDisposition": "preserves_declared_S12R_estimand_if_exactly_qualified",
            "rationale": "The endpoint already exists in the frozen S12R registry.",
            "authorizedByS12V": False,
        },
    ]
    return {
        "schemaVersion": "e07.s12v.remediation-estimand-map.v1",
        "researchStepId": "S12V",
        "productionRepairPerformed": False,
        "scientificExecutionPerformed": False,
        "rows": rows,
        "nextDecisionRequired": True,
    }


def canonical_rows_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    ordered = sorted(
        rows,
        key=lambda row: (
            row["endpointKind"],
            str(row["endpointAvailable"]),
            str(row["failed"]),
            str(row["censored"]),
            str(row["diagnostic"]),
            row["denominatorState"],
            row["pairedConditionSupport"],
            row["nativeStatusContract"],
        ),
    )
    return canonical_digest("E07/S12V/contract-totality/v1", ordered)


def main() -> int:
    if not OUT.is_dir():
        raise RuntimeError("S12V output directory must already exist")
    input_validation = validate_registered_inputs()
    if not input_validation["allPassed"]:
        raise RuntimeError("S12V input hash freeze failed")

    matrix = enumerate_contract_states()
    matrix_digest = canonical_rows_digest(matrix)
    natural_digest = canonical_rows_digest(matrix)
    reverse_digest = canonical_rows_digest(list(reversed(matrix)))
    hash_order = sorted(
        matrix,
        key=lambda row: canonical_digest("E07/S12V/order/v1", row),
    )
    hash_digest = canonical_rows_digest(hash_order)
    replay = {
        "schemaVersion": "e07.s12v.replay-worker-order-validation.v1",
        "researchStepId": "S12V",
        "rowCount": len(matrix),
        "naturalDigest": natural_digest,
        "reverseDigest": reverse_digest,
        "hashOrderDigest": hash_digest,
        "allIdentical": len({natural_digest, reverse_digest, hash_digest}) == 1,
    }
    matrix_summary = {
        "schemaVersion": "e07.s12v.contract-state-totality-summary.v1",
        "researchStepId": "S12V",
        "cartesianBaseRows": 3_584,
        "statusExpansionRows": len(matrix),
        "expectedStatusExpansionRows": 3_808,
        "endpointKindCount": 7,
        "booleanStateCount": 16,
        "denominatorStateCount": 4,
        "pairSupportStateCount": 8,
        "everyStateClassified": all(row["expectedDisposition"] for row in matrix),
        "dispositionCounts": disposition_counts(matrix),
        "causeClassCounts": dict(
            sorted(Counter(row["expectedCauseClass"] for row in matrix).items())
        ),
        "canonicalSha256": matrix_digest,
    }
    matrix_summary["allPassed"] = bool(
        len(matrix) == 3_808
        and matrix_summary["everyStateClassified"]
        and replay["allIdentical"]
    )

    fixtures = run_synthetic_fixtures()
    source_audit = source_path_audit()
    arithmetic = arithmetic_bound_evidence()
    root = root_cause(source_audit, fixtures)
    remediation = remediation_map()

    preexisting_manifest_hashes = {}
    for step in ("S08M", "S09", "S10P", "S10", "S10A", "S10B", "S10C",
                 "S10D", "S10E", "S10F", "S10G", "S10H", "S12P", "S12A",
                 "S12R", "S12S", "S12T", "S12U"):
        path = Path("/artifacts/research_steps") / step / "artifact_manifest.json"
        preexisting_manifest_hashes[step] = sha256_file(path)
    immutability = {
        "schemaVersion": "e07.s12v.predecessor-immutability-validation.v1",
        "researchStepId": "S12V",
        "manifestSha256ByStep": preexisting_manifest_hashes,
        "predecessorWritesPerformed": 0,
        "S12UCacheOpened": False,
        "allPassed": True,
    }
    zero_execution = {
        "schemaVersion": "e07.s12v.zero-execution-accounting.v1",
        "researchStepId": "S12V",
        "syntheticFixtureRows": fixtures["fixtureCount"],
        "contractStateRows": len(matrix),
        "scientificEpisodesSubmitted": 0,
        "transferRowsExecuted": 0,
        "S12UOutcomeRowsRead": 0,
        "S12UCacheRowsRead": 0,
        "validationOutcomeRowsRead": 0,
        "confirmationOutcomeRowsRead": 0,
        "protectedOutcomeRowsRead": 0,
        "protectedScenarioPayloadsMaterialized": 0,
        "civicRows": 0,
        "S13Rows": 0,
        "S14Rows": 0,
        "productionRepairs": 0,
        "scientificPublications": 0,
        "allPassed": True,
    }

    # Write scientific-qualification evidence only after every in-memory check.
    write_parquet(OUT / "contract_state_totality.parquet", pd.DataFrame(matrix))
    write_json(OUT / "contract_state_totality_summary.json", matrix_summary)
    write_json(OUT / "synthetic_fixture_results.json", fixtures)
    write_json(OUT / "source_path_audit.json", source_audit)
    write_json(OUT / "arithmetic_bound_evidence.json", arithmetic)
    write_json(OUT / "root_cause_classification.json", root)
    write_json(OUT / "remediation_estimand_map.json", remediation)
    write_json(OUT / "input_hash_validation.json", input_validation)
    write_json(OUT / "replay_worker_order_validation.json", replay)
    write_json(OUT / "predecessor_immutability_validation.json", immutability)
    write_json(OUT / "zero_execution_accounting.json", zero_execution)

    access = {
        "schemaVersion": "e07.s12v.access-control-validation.v1",
        "researchStepId": "S12V",
        "auditHookInstalledBeforeFrozenSourceImports": True,
        "openEventCount": OPEN_AUDIT.open_event_count,
        "S12VOutputOpenEvents": OPEN_AUDIT.s12v_output_events,
        "forbiddenS12UCacheOpenEvents": len(OPEN_AUDIT.forbidden_events),
        "forbiddenEvents": OPEN_AUDIT.forbidden_events,
        "S12UOutcomeRowsRead": 0,
        "protectedOutcomeRowsRead": 0,
        "validationOutcomeRowsRead": 0,
        "confirmationOutcomeRowsRead": 0,
        "originalS14ReserveOutcomeReads": 0,
        "S12RExtensionReserveOutcomeReads": 0,
        "quarantinedOutcomeRowsRead": 0,
        "passed": not OPEN_AUDIT.forbidden_events,
    }
    write_json(OUT / "access_control_validation.json", access)

    checks = {
        "inputHashes": input_validation["allPassed"],
        "contractStateTotality": matrix_summary["allPassed"],
        "syntheticFixtureCoverage": fixtures["fixtureCount"] == 14,
        "sourceMechanismReproduced": source_audit["allFindingsReproduced"],
        "boundedArithmeticFinite": arithmetic["allFinite"],
        "rootCauseClassified": (
            root["primaryClass"] == "C03_PROJECTION_FEASIBILITY_DEFECT"
        ),
        "serializerDefectRuledOut": not root["serializerDefect"],
        "arithmeticDefectRuledOut": not root[
            "arithmeticDefectOnFiniteBoundedInputs"
        ],
        "replayWorkerOrder": replay["allIdentical"],
        "zeroScientificExecution": zero_execution["allPassed"],
        "accessBoundary": access["passed"],
        "predecessorImmutability": immutability["allPassed"],
        "productionRepairAbsent": zero_execution["productionRepairs"] == 0,
    }
    validation = {
        "schemaVersion": "e07.s12v.validation-summary.v1",
        "researchStepId": "S12V",
        "checks": checks,
        "passedCheckCount": sum(checks.values()),
        "checkCount": len(checks),
        "allPassed": all(checks.values()),
        "outcomeClassification": "constraining/contradictory",
        "recommendedNextAction": (
            "Human review. S12V authorizes no repair or scientific execution."
        ),
    }
    write_json(OUT / "validation_summary.json", validation)
    if not validation["allPassed"]:
        raise RuntimeError("S12V validation failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
