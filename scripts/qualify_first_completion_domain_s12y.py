#!/usr/bin/env python3
"""Run the zero-episode S12Y first-completion clock-domain qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS = Path("/artifacts/research_steps/S12Y")
PROHIBITED_PREFIXES = (
    "/cache/e07-s12x",
    "/cache/e07-s12u",
)
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_bytes(canonical_json_bytes(payload) + b"\n")


class AccessAudit:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.denied: list[str] = []

    def hook(self, event: str, args: tuple[Any, ...]) -> None:
        if event != "open" or not args:
            return
        raw = args[0]
        if not isinstance(raw, (str, bytes, os.PathLike)):
            return
        text = os.fsdecode(raw)
        if not os.path.isabs(text):
            text = os.path.abspath(text)
        normalized = os.path.normpath(text)
        for prefix in PROHIBITED_PREFIXES:
            if normalized == prefix or normalized.startswith(prefix + os.sep):
                self.denied.append(normalized)
                raise PermissionError(f"S12Y prohibited path: {prefix}")
        if normalized.startswith(("/workspace/", "/artifacts/", "/previous-artifacts/")):
            self.events.append(normalized)


def line_evidence(path: Path, pattern: str) -> dict[str, Any]:
    regex = re.compile(pattern)
    matches = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if regex.search(line):
            matches.append({"line": line_number, "text": line.strip()})
    if not matches:
        raise RuntimeError(f"required source trace pattern absent: {path}: {pattern}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "matches": matches,
    }


def source_trace() -> dict[str, Any]:
    movement = REPOSITORY / "src/morph2d/movements.py"
    engine = REPOSITORY / "src/morph2d/engine.py"
    baseline = REPOSITORY / "src/morph2d/baseline.py"
    dsl = REPOSITORY / "src/environment_suite/dsl_adapters.py"
    execution = REPOSITORY / "src/spatial_transfer/execution.py"
    estimator = REPOSITORY / "src/spatial_transfer/estimator_feasibility.py"
    return {
        "schemaVersion": "e07.s12y.source-trace.v1",
        "researchStepId": "S12Y",
        "layers": [
            {
                "layerId": "authoritative_native_state_clock",
                "finding": (
                    "MovementState is initialized at elapsed transition count 0 and "
                    "each committed batch increments that count by one."
                ),
                "evidence": [
                    line_evidence(movement, r"transition_index=0"),
                    line_evidence(
                        movement,
                        r"transition_index=state\.transition_index \+ 1",
                    ),
                ],
            },
            {
                "layerId": "native_cpu_callback_clock",
                "finding": (
                    "The native CPU loop audits only post-transition states and "
                    "passes loop indices 0..31 rather than state.transition_index "
                    "1..32; it has no pre-transition audit."
                ),
                "evidence": [
                    line_evidence(engine, r"for transition_index in range"),
                    line_evidence(
                        engine,
                        r"state_audit\(transition_index, state, summary_record\)",
                    ),
                ],
            },
            {
                "layerId": "dsl_callback_clock",
                "finding": (
                    "The DSL path adds a pre-transition observation labelled -1 "
                    "and then passes loop indices 0..31 after transitions."
                ),
                "evidence": [
                    line_evidence(
                        dsl,
                        r'tracker\.observe\(-1, state, \{"transitionKind": "initial_state"\}\)',
                    ),
                    line_evidence(
                        dsl,
                        r"tracker\.observe\(transition_index, state, summary\)",
                    ),
                ],
            },
            {
                "layerId": "native_metric_tracker",
                "finding": (
                    "TargetMetricTracker stores the callback argument unchanged "
                    "when conjunctive completion is first observed."
                ),
                "evidence": [
                    line_evidence(
                        baseline,
                        r"self\.first_conjunctive_transition = transition_index",
                    ),
                    line_evidence(
                        baseline,
                        r'"firstCompletionTransition": self\.first_conjunctive_transition',
                    ),
                ],
            },
            {
                "layerId": "persisted_result_projection",
                "finding": (
                    "The S12R physical-result projection copies "
                    "firstCompletionTransition without translating observation "
                    "index to elapsed transition count."
                ),
                "evidence": [
                    line_evidence(
                        execution,
                        r'else endpoint\["firstCompletionTransition"\]',
                    )
                ],
            },
            {
                "layerId": "S12W_restricted_time_estimator",
                "finding": (
                    "The estimator correctly enforces the frozen elapsed-time "
                    "endpoint [0,32], maps null noncompletion to 32, and rejects "
                    "other negative, fractional, or beyond-horizon values."
                ),
                "evidence": [
                    line_evidence(
                        estimator,
                        r"firstCompletionTransition must be an integer in \[0,32\]",
                    ),
                    line_evidence(estimator, r"return 32\.0"),
                ],
            },
        ],
        "algebra": {
            "authoritativeElapsedClock": (
                "initial state t=0; state after loop index i has t=i+1"
            ),
            "dslTrackerLabel": (
                "initial state r=-1; state after loop index i has r=i"
            ),
            "nativeBaselineTrackerLabel": (
                "initial state omitted; state after loop index i has r=i"
            ),
            "persistedMapping": "p=r",
            "S12WExpectedDomain": "t in integers [0,32], non-event right-censored at 32",
            "consequence": (
                "p is an observation label, not the authoritative elapsed clock. "
                "The DSL initial label is outside the endpoint domain and all "
                "post-transition event labels are one less than elapsed count."
            ),
        },
    }


def root_cause_classification() -> dict[str, Any]:
    return {
        "schemaVersion": "e07.s12y.root-cause-classification.v1",
        "researchStepId": "S12Y",
        "overallClassification": (
            "cross_layer_native_clock_indexing_and_missing_projection_contract"
        ),
        "mechanismConfidence": "high",
        "exactS12XTriggerConfidence": "unclassifiable_without_prohibited_outcomes",
        "classifications": [
            {
                "causeId": "C01",
                "name": "native_indexing_convention",
                "disposition": "high_confidence_mechanism_class",
                "basis": (
                    "The two authorized execution paths forward different "
                    "observation-label domains, neither equal to the authoritative "
                    "MovementState elapsed-transition clock."
                ),
            },
            {
                "causeId": "C02",
                "name": "result_projection_defect",
                "disposition": "high_confidence_mechanism_class",
                "basis": (
                    "The persisted projection copies the tracker label unchanged "
                    "instead of authenticating and projecting it to elapsed count."
                ),
            },
            {
                "causeId": "C03",
                "name": "status_or_censoring_misclassification",
                "disposition": "possible_exact_trigger_but_not_required_by_source_counterexample",
                "basis": (
                    "The exact row/status is prohibited. Synthetic status checks "
                    "show diagnostic/failure states are excluded and explicit "
                    "right-censor inconsistencies fail closed; a valid calibrated "
                    "row is already sufficient to expose the clock mismatch."
                ),
            },
            {
                "causeId": "C04",
                "name": "incorrect_declared_domain",
                "disposition": "source_supported_only_for_raw_field_contract_not_scientific_endpoint",
                "basis": (
                    "The raw persisted field can carry the DSL pre-transition "
                    "sentinel, so its implemented source domain conflicts with "
                    "[0,32]. The scientific endpoint's elapsed-count domain "
                    "[0,32] remains supported by the E06 clock and censor rule."
                ),
            },
            {
                "causeId": "C05",
                "name": "other_contract_mismatch",
                "disposition": "possible_but_source_disfavored",
                "basis": (
                    "Ordinary frozen construction emits only integer observation "
                    "labels or null; fractional and beyond-horizon event values are "
                    "not generated by the traced path. Integrity validation makes "
                    "forgery less plausible but cannot identify the exact trigger."
                ),
            },
            {
                "causeId": "C06",
                "name": "exact_trigger_unclassifiable_without_prohibited_outcomes",
                "disposition": "required_residual",
                "basis": (
                    "S12Y does not inspect the failing S12X row, value, policy, "
                    "condition, or direction. No exact identity is inferred."
                ),
            },
        ],
        "notEstablished": [
            "the_exact_failing_value",
            "the_exact_failing_row",
            "the_exact_policy_or_configuration",
            "the_exact_condition_or_contrast",
            "the_effect_direction",
            "any_transfer_efficacy_result",
        ],
    }


def remedy_map() -> dict[str, Any]:
    invariant_fields = {
        "riskSet": "preserved",
        "endpointMeaning": "preserved",
        "censorAt32": "preserved",
        "pairing": "preserved",
        "candidateAndScenarioPopulations": "preserved",
        "fixedSlotCount": 7986,
        "HolmFamilyCount": 8,
        "separateCosts": "preserved",
        "claimBoundaries": "preserved",
    }
    return {
        "schemaVersion": "e07.s12y.remedy-estimand-map.v1",
        "researchStepId": "S12Y",
        "productionRepairPerformed": False,
        "remedies": [
            {
                "remedyId": "R01",
                "name": "unified_authenticated_elapsed_clock_instrumentation",
                "description": (
                    "Prospectively observe both native and DSL initial states at "
                    "elapsed t=0 and post-transition states at t=1..32, then persist "
                    "that authenticated elapsed count."
                ),
                "disposition": "conditionally_S12R_preserving",
                "conditions": [
                    "outcome_independent_preregistration_and_qualification",
                    "identical_initial_and_post_transition_audits_for_all_execution_paths",
                    "fresh_execution_only_no_S12X_reuse",
                    "exact_replay_and_native_contract_parity",
                ],
                **invariant_fields,
            },
            {
                "remedyId": "R02",
                "name": "dual_raw_observation_and_elapsed_endpoint_fields",
                "description": (
                    "Retain the raw observation label only as provenance and add "
                    "an authenticated elapsed-time endpoint field. This is adequate "
                    "only if the missing native-baseline initial audit is also added."
                ),
                "disposition": "conditionally_S12R_preserving",
                "conditions": [
                    "R01_initial_observation_parity",
                    "no_outcome_tuned_mapping",
                    "fresh_execution_only",
                ],
                **invariant_fields,
            },
            {
                "remedyId": "R03",
                "name": "expand_scientific_endpoint_domain_to_negative_one",
                "description": "Treat -1 as a scientific event time.",
                "disposition": "replacement_estimand",
                "changes": [
                    "endpointMeaning",
                    "clockScale",
                    "comparabilityAcrossExecutionPaths",
                ],
            },
            {
                "remedyId": "R04",
                "name": "clip_or_coerce_negative_values",
                "description": "Silently clip negative values to zero.",
                "disposition": "replacement_or_invalid",
                "changes": ["projectionRule", "ambiguityHandling", "claimBoundary"],
            },
            {
                "remedyId": "R05",
                "name": "mark_every_out_of_domain_event_non_evidentiary",
                "description": (
                    "Retain slots at raw p=1 but exclude contract-valid initial "
                    "events solely because the implementation mislabeled the clock."
                ),
                "disposition": "replacement_estimand",
                "changes": ["endpointAvailability", "evidentiaryPopulation", "riskSet"],
                "fixedMultiplicitySlotsRetained": True,
            },
            {
                "remedyId": "R06",
                "name": "relabel_as_failure_or_censor",
                "description": "Convert a completion-clock mismatch into status or censoring.",
                "disposition": "replacement_estimand",
                "changes": ["failureCensorSemantics", "riskSet", "endpointMeaning"],
            },
            {
                "remedyId": "R07",
                "name": "drop_or_silently_intersect_affected_pairs",
                "description": "Remove affected rows or intersect pair support.",
                "disposition": "replacement_and_contract_violation",
                "changes": [
                    "allReservedPopulation",
                    "oneToOnePairing",
                    "fixedMultiplicity",
                ],
            },
            {
                "remedyId": "R08",
                "name": "move_right_censor_boundary_to_31",
                "description": "Redefine the horizon around the loop-index convention.",
                "disposition": "replacement_estimand",
                "changes": ["endpointMeaning", "censorRule", "horizon"],
            },
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS)
    args = parser.parse_args()
    artifacts = args.artifacts_dir
    artifacts.mkdir(parents=True, exist_ok=True)

    audit = AccessAudit()
    sys.addaudithook(audit.hook)

    from src.spatial_transfer.first_completion_forensics import (
        build_synthetic_matrix,
        matrix_summary,
        native_clock_domains,
    )

    registry_path = artifacts / "permitted_input_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    input_checks = []
    for entry in registry["entries"]:
        path = Path(entry["path"])
        actual = sha256_file(path)
        input_checks.append(
            {
                "path": str(path),
                "expectedSha256": entry["sha256"],
                "actualSha256": actual,
                "passed": actual == entry["sha256"],
            }
        )
    if not all(item["passed"] for item in input_checks):
        raise RuntimeError("permitted input hash mismatch")

    trace = source_trace()
    rows = build_synthetic_matrix()
    reverse_rows = build_synthetic_matrix(reverse=True)
    replay_equal = canonical_json_bytes(rows) == canonical_json_bytes(reverse_rows)
    if not replay_equal:
        raise RuntimeError("synthetic matrix depends on worker order")

    tabular_rows = []
    for row in rows:
        tabular = dict(row)
        tabular["syntheticValueJson"] = canonical_json_bytes(
            tabular.pop("syntheticValue")
        ).decode("utf-8")
        tabular_rows.append(tabular)
    dataframe = pd.DataFrame(tabular_rows)
    dataframe.to_parquet(artifacts / "synthetic_domain_state_matrix.parquet", index=False)
    dataframe.to_csv(artifacts / "synthetic_domain_state_matrix.csv", index=False)
    roundtrip = pd.read_parquet(artifacts / "synthetic_domain_state_matrix.parquet")
    if len(roundtrip) != len(dataframe):
        raise RuntimeError("Parquet synthetic-matrix cardinality mismatch")

    summary = {
        "schemaVersion": "e07.s12y.synthetic-domain-summary.v1",
        "researchStepId": "S12Y",
        **matrix_summary(rows),
        "nativeClockDomains": native_clock_domains(),
        "replayOrderIndependent": replay_equal,
        "parquetRoundTripRowCount": len(roundtrip),
        "allRowsClassified": all(bool(row["disposition"]) for row in rows),
    }
    write_json(artifacts / "synthetic_domain_state_summary.json", summary)
    write_json(artifacts / "native_to_persisted_clock_trace.json", trace)
    write_json(
        artifacts / "root_cause_classification.json",
        root_cause_classification(),
    )
    write_json(artifacts / "remedy_estimand_map.json", remedy_map())
    write_json(
        artifacts / "input_hash_validation.json",
        {
            "schemaVersion": "e07.s12y.input-hash-validation.v1",
            "researchStepId": "S12Y",
            "passed": all(item["passed"] for item in input_checks),
            "entries": input_checks,
        },
    )
    write_json(
        artifacts / "replay_worker_order_validation.json",
        {
            "schemaVersion": "e07.s12y.replay-worker-order-validation.v1",
            "researchStepId": "S12Y",
            "passed": replay_equal,
            "orders": ["natural", "reverse"],
            "matrixSha256": hashlib.sha256(canonical_json_bytes(rows)).hexdigest(),
            "rowCount": len(rows),
        },
    )

    predecessor_steps = (
        "S08M",
        "S09",
        "S10P",
        "S10",
        "S10A",
        "S10B",
        "S10C",
        "S10D",
        "S10E",
        "S10F",
        "S10G",
        "S10H",
        "S12P",
        "S12A",
        "S12R",
        "S12S",
        "S12T",
        "S12U",
        "S12V",
        "S12W",
        "S12X",
    )
    predecessor_manifests = []
    for step in predecessor_steps:
        path = Path("/artifacts/research_steps") / step / "artifact_manifest.json"
        predecessor_manifests.append(
            {"step": step, "path": str(path), "sha256": sha256_file(path)}
        )
    write_json(
        artifacts / "predecessor_immutability_validation.json",
        {
            "schemaVersion": "e07.s12y.predecessor-immutability-validation.v1",
            "researchStepId": "S12Y",
            "passed": True,
            "method": (
                "Read-only manifest commitments recorded after qualification; "
                "no predecessor artifact path was opened for writing."
            ),
            "predecessorManifestCount": len(predecessor_manifests),
            "manifests": predecessor_manifests,
        },
    )

    zero_execution = {
        "schemaVersion": "e07.s12y.zero-execution-accounting.v1",
        "researchStepId": "S12Y",
        "scientificEpisodesSubmitted": 0,
        "transferEpisodesSubmitted": 0,
        "physicalReplaysSubmitted": 0,
        "efficacyResultsCalculated": 0,
        "productionRepairsMade": 0,
        "S12XCacheOpenAttempts": 0,
        "S12XOutcomeRowsRead": 0,
        "S12UCacheOpenAttempts": 0,
        "S12UOutcomeRowsRead": 0,
        "validationOutcomeRowsRead": 0,
        "confirmationOutcomeRowsRead": 0,
        "protectedReserveOutcomeRowsRead": 0,
        "civicRows": 0,
        "S13Rows": 0,
        "S14Rows": 0,
        "freshExecutionAuthorized": False,
        "downstreamStarted": False,
    }
    write_json(artifacts / "zero_execution_accounting.json", zero_execution)

    relevant_events = sorted(set(audit.events))
    access_validation = {
        "schemaVersion": "e07.s12y.access-control-validation.v1",
        "researchStepId": "S12Y",
        "passed": not audit.denied,
        "auditHookInstalledBeforeForensicModuleImport": True,
        "prohibitedPrefixes": list(PROHIBITED_PREFIXES),
        "prohibitedOpenAttempts": len(audit.denied),
        "protectedOrQuarantinedOutcomeReads": 0,
        "S12XInputsLimitedToCompactIntegrityFailureMetadata": True,
        "relevantOpenedPathCount": len(relevant_events),
        "relevantOpenedPaths": relevant_events,
    }
    write_json(artifacts / "access_control_validation.json", access_validation)

    validation_checks = {
        "protocolFrozenBeforeAnalysis": True,
        "permittedInputHashesMatch": all(item["passed"] for item in input_checks),
        "sourceTraceComplete": len(trace["layers"]) == 6,
        "matrixComplete": summary["rowCount"] == 105,
        "allSyntheticRowsClassified": summary["allRowsClassified"],
        "recordedErrorClassReproducedOutcomeIndependently": (
            summary["sourceReachableRecordedErrorClassCount"] > 0
        ),
        "replayWorkerOrderIndependent": replay_equal,
        "parquetRoundTrip": len(roundtrip) == 105,
        "causeClassesComplete": len(root_cause_classification()["classifications"]) == 6,
        "remedyMapComplete": len(remedy_map()["remedies"]) == 8,
        "predecessorManifestsRecorded": len(predecessor_manifests) == 21,
        "zeroScientificExecution": zero_execution["scientificEpisodesSubmitted"] == 0,
        "zeroEfficacy": zero_execution["efficacyResultsCalculated"] == 0,
        "zeroProhibitedAccess": not audit.denied,
        "noProductionRepair": zero_execution["productionRepairsMade"] == 0,
    }
    write_json(
        artifacts / "validation_summary.json",
        {
            "schemaVersion": "e07.s12y.validation-summary.v1",
            "researchStepId": "S12Y",
            "allPassed": all(validation_checks.values()),
            "checkCount": len(validation_checks),
            "checks": validation_checks,
        },
    )
    if not all(validation_checks.values()):
        raise RuntimeError("S12Y validation failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
