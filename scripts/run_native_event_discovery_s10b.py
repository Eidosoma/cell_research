#!/usr/bin/env python3
"""Execute S10B as a fresh continuation of the byte-frozen S10P design."""

from __future__ import annotations

from contextlib import contextmanager
import gzip
import hashlib
import json
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Any, Iterator, Mapping

import yaml

import scripts.run_native_event_discovery_s10 as base
from src.phenotype_discovery.accounting import (
    FailAtomicPhaseError,
    build_physical_replay_plan,
)
from src.phenotype_discovery.search import load_roster, sha256_file


STEP_ID = "S10B"
OUTPUT = Path("/artifacts/research_steps/S10B")
CACHE = Path("/cache/e07-s10b")
CONFIG = Path("/workspace/cell-research/configs/discovery/s10b_fresh_execution.yaml")
ORIGINAL_EXECUTION_CONFIG = Path(
    "/workspace/cell-research/configs/discovery/s10_native_event_execution.yaml"
)
SCRIPT = Path(__file__).resolve()
TEST = Path("/workspace/cell-research/tests/test_s10b_fresh_execution.py")
S10P = Path("/artifacts/research_steps/S10P")
FAILED_S10 = Path("/artifacts/research_steps/S10")
S10A = Path("/artifacts/research_steps/S10A")
HISTORICAL = (
    Path("/artifacts/research_steps/S05"),
    Path("/artifacts/research_steps/S08M"),
    Path("/artifacts/research_steps/S09"),
    S10P,
    FAILED_S10,
    S10A,
)
_ORIGINAL_BASE_CALLBACKS = MappingProxyType(
    {
        "revalidate_frozen_inputs": base.revalidate_frozen_inputs,
        "prospective_freeze": base.prospective_freeze,
        "reviewer_instructions": base.reviewer_instructions,
        "report_markdown": base.report_markdown,
        "manifest_for_output": base.manifest_for_output,
    }
)


class CallbackBindingError(RuntimeError):
    """Raised before execution when callback installation is ambiguous."""


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value))


def tree_snapshot(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            (f"{row['path']}\0{row['bytes']}\0{row['sha256']}\n").encode("utf-8")
        )
    return {
        "root": str(root),
        "fileCount": len(rows),
        "totalBytes": sum(row["bytes"] for row in rows),
        "treeSha256": digest.hexdigest(),
        "files": rows,
    }


def _physical_plan_audit() -> dict[str, Any]:
    roster = load_roster()
    plan = build_physical_replay_plan(roster)
    digest = hashlib.sha256()
    for row in plan:
        digest.update(str(row["physicalExecutionId"]).encode("ascii"))
        digest.update(b"\n")
    return {
        "logicalReservations": len(roster),
        "physicalReplayCommitments": len(plan),
        "uniqueLogicalReservationIds": len(
            {str(row["logicalReservationId"]) for row in roster}
        ),
        "uniquePhysicalExecutionIds": len(
            {str(row["physicalExecutionId"]) for row in plan}
        ),
        "discoveryLogicalReservations": sum(
            row["phase"] == "discovery" for row in roster
        ),
        "reproductionLogicalReservations": sum(
            row["phase"] == "independent_reproduction" for row in roster
        ),
        "physicalPlanSha256": digest.hexdigest(),
        "pass": (
            len(roster) == 10_752
            and len(plan) == 21_504
            and len({str(row["logicalReservationId"]) for row in roster}) == 10_752
            and len({str(row["physicalExecutionId"]) for row in plan}) == 21_504
        ),
    }


def revalidate_frozen_inputs() -> dict[str, Any]:
    """Extend the original read-only S10 preflight with S10B freezes."""

    active_config = base.CONFIG
    try:
        base.CONFIG = ORIGINAL_EXECUTION_CONFIG
        result = _ORIGINAL_BASE_CALLBACKS["revalidate_frozen_inputs"]()
    finally:
        base.CONFIG = active_config
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    snapshots = {
        "S10P": tree_snapshot(S10P),
        "S10": tree_snapshot(FAILED_S10),
        "S10A": tree_snapshot(S10A),
    }
    expected = {
        "S10P": config["immutableInputs"]["s10pTree"]["treeSha256"],
        "S10": config["immutableInputs"]["failedS10Tree"]["treeSha256"],
        "S10A": config["immutableInputs"]["s10aTree"]["treeSha256"],
    }
    tree_checks = {
        key: {
            "expectedTreeSha256": expected[key],
            "actualTreeSha256": snapshots[key]["treeSha256"],
            "fileCount": snapshots[key]["fileCount"],
            "totalBytes": snapshots[key]["totalBytes"],
            "pass": snapshots[key]["treeSha256"] == expected[key],
        }
        for key in ("S10P", "S10", "S10A")
    }
    scientific_checks = []
    for key in (
        "scientificProtocol",
        "featureRegistry",
        "methodRegistry",
        "candidatePopulation",
        "logicalRoster",
        "eligibilityGate",
        "s10aFreshExecutionGate",
    ):
        spec = config["immutableInputs"][key]
        actual = sha256_file(spec["path"])
        scientific_checks.append(
            {
                "name": key,
                "path": spec["path"],
                "expectedSha256": spec["sha256"],
                "actualSha256": actual,
                "pass": actual == spec["sha256"],
            }
        )
    physical = _physical_plan_audit()
    s10a_gate = json.loads(
        (S10A / "fresh_s10_execution_gate.json").read_text(encoding="utf-8")
    )
    s10a_input_freeze = json.loads(
        (S10A / "input_hash_freeze.json").read_text(encoding="utf-8")
    )
    s10a_sources = {str(row["path"]): row for row in s10a_input_freeze["sourceFiles"]}
    remediated_core_paths = {
        "/workspace/cell-research/src/phenotype_discovery/accounting.py",
        "/workspace/cell-research/src/phenotype_discovery/native_features.py",
        "/workspace/cell-research/src/phenotype_discovery/search.py",
    }
    remediated_core_checks = []
    for path_text in sorted(remediated_core_paths):
        expected_row = s10a_sources[path_text]
        path = Path(path_text)
        remediated_core_checks.append(
            {
                "path": path_text,
                "expectedS10ASha256": expected_row["sha256"],
                "actualSha256": sha256_file(path),
                "expectedS10ABytes": expected_row["bytes"],
                "actualBytes": path.stat().st_size,
                "pass": (
                    sha256_file(path) == expected_row["sha256"]
                    and path.stat().st_size == expected_row["bytes"]
                ),
            }
        )
    nonsuperseded_s10p_input_checks = [
        row
        for row in result["frozenInputChecks"]
        if str(row["path"]) not in remediated_core_paths
    ]
    s10a_qualification = all(
        bool(s10a_gate[key])
        for key in (
            "qualificationPass",
            "s10pG01ThroughG08Pass",
            "missingnessRemediationQualified",
            "allRowAccountingQualified",
            "s10pAndFailedS10ByteIdentical",
            "requiresGenuinelyFreshNamespace",
            "mustPublishDispositionLedgerBeforeResultRows",
            "mustStopFailClosedOnAnyIntegrityFailure",
        )
    ) and not bool(s10a_gate["failedS10CacheReusePermitted"])
    result.update(
        {
            "schemaVersion": "e07.s10b.preflight-hash-revalidation.v1",
            "researchStepId": STEP_ID,
            "immutableTreeChecks": tree_checks,
            "s10bScientificFileChecks": scientific_checks,
            "preOutcomeCommitmentAudit": physical,
            "s10aQualificationGatePass": s10a_qualification,
            "s10aRemediatedCoreChecks": remediated_core_checks,
            "supersededS10PImplementationChecks": [
                row
                for row in result["frozenInputChecks"]
                if str(row["path"]) in remediated_core_paths and not row["pass"]
            ],
            "supersededImplementationReason": (
                "S10P-era implementation hashes are superseded only for the "
                "exact S10A-frozen accounting, extractor, and search core."
            ),
            "separateS10BFreshExecutionApprovalReceived": True,
            "freshCacheNamespace": str(CACHE),
            "freshCacheAbsentAtPreflight": not CACHE.exists(),
            "failedS10CachePath": "/cache/e07-s10",
            "failedS10CacheReads": 0,
            "failedS10OutcomeRowsReused": 0,
        }
    )
    result["allPass"] = bool(
        all(row["pass"] for row in result["frozenFileChecks"])
        and all(row["pass"] for row in nonsuperseded_s10p_input_checks)
        and all(result["preregistrationChecks"].values())
        and result["s10pGateAllPass"]
        and len(result["bindingRows"]) == 28
        and all(row["pass"] for row in result["bindingRows"])
        and result["rosterChecks"]
        == {
            "rows": 10_752,
            "uniqueLogicalReservationIds": 10_752,
            "candidateCount": 14,
            "taskCount": 2,
            "discoveryRows": 7_168,
            "reproductionRows": 3_584,
        }
        and all(row["pass"] for row in remediated_core_checks)
        and all(row["pass"] for row in tree_checks.values())
        and all(row["pass"] for row in scientific_checks)
        and physical["pass"]
        and s10a_qualification
        and result["freshCacheAbsentAtPreflight"]
    )
    return result


def prospective_freeze(preflight: Mapping[str, Any]) -> dict[str, Any]:
    result = _ORIGINAL_BASE_CALLBACKS["prospective_freeze"](preflight)
    result.pop("executionFreezeSha256", None)
    result.update(
        {
            "schemaVersion": "e07.s10b.execution-preregistration.v1",
            "researchStepId": STEP_ID,
            "continuationOfScientificDesign": "S10P",
            "remediatedExecutorContract": "S10A",
            "failedHistoricalExecution": "S10",
            "freshCacheNamespace": str(CACHE),
            "failedS10CacheReadsBeforeFreeze": 0,
            "failedS10OutcomeRowsReusedBeforeFreeze": 0,
            "preOutcomeLogicalCommitments": 10_752,
            "preOutcomePhysicalReplayCommitments": 21_504,
            "humanAnnotationBoundary": "stop_before_annotation",
            "s11RowsBeforeFreeze": 0,
        }
    )
    result["executionFreezeSha256"] = hashlib.sha256(
        b"E07/S10B/execution-freeze/v1\0"
        + json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    return result


def reviewer_instructions(machine_count: int) -> str:
    return f"""# S10B blinded external-review instructions

## Top summary

| Field | Result |
| --- | --- |
| Research step ID | **S10B** |
| Completion status | Machine discovery/reproduction complete; external review {"required for " + str(machine_count) + " reproduced machine candidate(s)" if machine_count else "not triggered because no machine candidate reproduced"} |
| Artifacts written | Deterministically selected blinded exemplar packet and these instructions; annotation fields are empty |
| Validation result | Packet construction is identity-blinded and uses only the frozen discovery lock |
| Outcome classification | {"Supportive machine-stage evidence pending external review" if machine_count else "Bounded machine null; external review not applicable"} |
| Caveats or blockers | Event summaries are counts and state-hash changes, not complete trajectories; no human label has been supplied |
| Recommended next action | {"Obtain two independent blinded external reviews and return to the S10 review boundary; do not start S11" if machine_count else "Return the bounded machine-null result for scientific review; do not start S11"} |

## Reviewer procedure

Review each six-row candidate packet independently and assign exactly one:

- `activity_distribution`
- `persistence_or_burst`
- `conflict_pattern`
- `state_turnover_pattern`
- `no_operational_pattern`
- `insufficient_event_summary`

Do not use `preference`, `goal`, `intention`, `agency`, `competency`,
`formation`, `repair`, or any biological label. Configuration identity,
parent/compressed role, objective and validation values, policy source, and
S07 arm information are blinded. Do not attempt re-identification.

Two independent reviewers are required. A third blinded reviewer may
adjudicate disagreements only. Cohen's kappa must be at least 0.60 for
human-label promotion. Reviewers may not generate machine candidates or alter
the frozen multiplicity family. Return annotations separately; every
`reviewerAnnotation` field in the S10B packet is intentionally null.
"""


def report_markdown(
    *,
    outcome_classification: str,
    completion_status: str,
    recommended_next: str,
    preflight: Mapping[str, Any],
    discovery_integrity: Mapping[str, Any],
    reproduction_integrity: Mapping[str, Any],
    discovery_catalog: Mapping[str, Any],
    reproduction: list[Mapping[str, Any]],
    accounting: Mapping[str, Any],
    human_status: Mapping[str, Any],
    validation: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> str:
    reproduced = [row for row in reproduction if row.get("reproductionPass")]
    support = discovery_catalog["results"]["preprocessing"]
    support_rows = []
    for key, item in sorted(support.items()):
        values = [
            value for value in item["featureSupport"].values() if value is not None
        ]
        support_rows.append(
            {
                "stratum": key,
                "features": len(item["eligibleFeatures"]),
                "minimum": min(values) if values else None,
                "maximum": max(values) if values else None,
            }
        )
    support_table = "\n".join(
        f"| `{row['stratum']}` | {row['features']} | "
        f"{row['minimum'] if row['minimum'] is not None else 'n/a'} | "
        f"{row['maximum'] if row['maximum'] is not None else 'n/a'} |"
        for row in support_rows
    )
    return f"""# S10B — Fresh S10 execution after S10A

## Top summary

| Field | Result |
| --- | --- |
| Research step ID | **S10B** |
| Completion status | **{completion_status}** |
| Artifacts written | Frozen preflight/preregistration and input records; complete discovery/reproduction disposition ledgers; native event-feature and ordered-summary tables; preprocessing/confound, clustering, anomaly, change-point, multiplicity, discovery-lock, independent-reproduction, access, dependency, replay, native-contract, accounting, no-mutation, provenance, status, manifest, validation, and canonical-report records under `/artifacts/research_steps/S10B/`; {"an unannotated blinded exemplar packet and reviewer instructions" if reproduced else "a machine-candidate catalog with no reproduced candidate and review not triggered"} |
| Validation result | **PASS** — {validation["passedChecks"]}/{validation["totalChecks"]} execution checks; G01–G08; 28/28 bindings; {accounting["totalLogicalRows"]:,}/10,752 logical rows and {accounting["physicalEpisodeExecutions"]:,}/21,504 exact-replay executions; discovery/reproduction replay and worker-order integrity; protected outcomes sealed |
| Outcome classification | **{outcome_classification}** |
| Caveats or blockers | Only 14 configurations on two spatial training contracts were analyzed. Ordered summaries contain event counts and state-hash changes, not complete trajectories. Validation and confirmation stayed sealed. {"External human review is required and annotations are intentionally empty." if reproduced else "No candidate survived the frozen independent-reproduction gates."} |
| Lay summary | The repaired fresh run completed the prespecified simulations twice per row, kept legitimately unavailable measurements explicitly missing, and applied the frozen task/status support rules. It then searched for stable task-local event patterns and tested any discovery on independent training families. {"At least one machine pattern reproduced, so work stopped at a blinded external-review packet." if reproduced else "No machine pattern passed all discovery, multiplicity, and independent-reproduction requirements, so this is a bounded null."} |
| Recommended next action | {recommended_next} |

## Frozen question and outcome

The frozen question was whether the 14 S09 parent/compression configurations
exhibit task-local native event-summary clusters, anomalies, or change points
that survive S10P's frozen confound, stability, null, multiplicity, and
independent-training-family reproduction rules. The scientific design was not
changed after S10P; S10B only used S10A's qualified explicit-missingness and
fail-atomic accounting implementation.

The machine multiplicity family contained
**{discovery_catalog["machineMultiplicityFamilySize"]}** candidates;
**{len(discovery_catalog["discoveryPassingCandidates"])}** passed discovery
Holm correction and **{len(reproduced)}** passed the frozen independent
reproduction gate.

## Inputs and authorization boundary

S10B refreshed the governing workspace plans, E01–E06 native-event handoffs,
S01–S10A reports and registries, S02 split/access controls, the immutable S05
ledgers, and the attachment manifest/sidecar. It consumed only S10P's 14
frozen candidate structures, 768 fresh training scenario families, and
10,752-row logical roster. S10P, failed S10, and S10A revalidated at their
frozen tree hashes. The failed `/cache/e07-s10` namespace and its outcomes
were never read or reused; execution used only `/cache/e07-s10b`.

Protected validation and confirmation outcome reads, S06/S06A model or
embedding loads, S07 arm-signal uses, historical quarantine reads,
complete-trajectory reconstructions, universal novelty scores, and archive
mutations were all zero.

## Detailed methods

### Execution and durable accounting

The prospective S10B control fixed eight workers, one numeric thread per
worker, 7,168 discovery reservations, 3,584 independent-reproduction
reservations, and two domain-separated physical replays per logical row.
Before evaluation each phase atomically persisted every logical position,
reservation commitment, replay ordinal, and physical execution identity.
Every replay received a terminal success/exception disposition and every
logical pair received a terminal success/failure/mismatch disposition before
any row became caller-publishable.

Discovery was hash-locked before the independent-reproduction population was
opened. Exact complete disposition ledgers were promoted as compressed
artifacts, while bulky outcome rows remained in the fresh disposable cache.

### Explicit missingness and support

Every registered feature was either finite and observed or explicitly
unavailable with a frozen reason and, where applicable, a specific reason
code. No unavailable coordinate was filled, clipped, converted to zero,
silently dropped, or used as novelty. Eligibility was computed separately
within each task/native-status stratum using the inclusive `support >= 0.90`
rule and required at least five eligible features. Profiles used
coordinatewise medians over available values; no complete-case row filter was
applied.

| Task/status stratum | Eligible features | Minimum support | Maximum support |
| --- | ---: | ---: | ---: |
{support_table}

### Frozen phenotype methods

Within each task/status stratum with at least eight rows, S10B applied
five-fold scenario-family cross-fitted ridge deconfounding, median/MAD
scaling, Ward clustering over `k=2..6`, diagonal GMM comparison, 200
scenario-family stability bootstraps, 500 frozen label-permutation nulls,
500-tree Isolation Forest anomaly screening, and exact dynamic-programming
change-point screening of 32-transition proposal, acceptance, conflict-loss,
and state-hash-change sequences. Holm correction covered all machine
candidates across both tasks and all three method families. Rare status
strata remained descriptive only.

The reproduction phase reused the frozen support masks, preprocessing
coefficients, cluster representatives, anomaly estimators/thresholds,
change-point windows, and multiplicity family. No reproduction refit or
candidate redefinition was permitted.

## Results

| Quantity | Result |
| --- | ---: |
| Candidate configurations | 14 |
| Spatial native contracts | 2 |
| Training scenario families | 768 |
| Discovery logical rows | {accounting["discoveryLogicalRows"]:,} |
| Reproduction logical rows | {accounting["reproductionLogicalRows"]:,} |
| Physical exact-replay executions | {accounting["physicalEpisodeExecutions"]:,} |
| Retained native failures | {accounting["failures"]:,} |
| Retained native censors | {accounting["censors"]:,} |
| Machine multiplicity family | {discovery_catalog["machineMultiplicityFamilySize"]} |
| Discovery Holm-pass candidates | {len(discovery_catalog["discoveryPassingCandidates"])} |
| Independently reproduced candidates | {len(reproduced)} |
| Reviewer annotations | {human_status["reviewerAnnotationsPresent"]} |
| Validation / confirmation outcome rows | 0 / 0 |

The task-local machine tables in `clustering_results.parquet`,
`anomaly_results.parquet`, `change_point_results.parquet`, and
`independent_reproduction_results.parquet` contain the full frozen method
results. `discovery_catalog.parquet` records only the resulting machine
candidate status and does not promote a biological, cognitive, preference, or
agency interpretation.

## Validation

- Immutable scientific files, S10P/failed-S10/S10A trees, G01–G08, 28
  candidate/task bindings, and all 10,752 logical plus 21,504 physical
  pre-outcome commitments passed before execution.
- Discovery integrity: {discovery_integrity["observedLogicalRows"]:,}/7,168 rows,
  exact replay, unique logical identities/hashes, and natural/reverse digest
  parity.
- Reproduction integrity: {reproduction_integrity["observedLogicalRows"]:,}/3,584
  rows with the same checks.
- Complete availability planes, native contracts, 32-transition ordered
  summaries, failure/censor retention, protected denial, prohibited
  dependency exclusion, and historical no-mutation checks passed.
- External review annotations remain exactly zero. S11 was not started.

## Commands and dependencies

```bash
PYTHONPATH=. pytest -q \
  tests/test_s10_native_event_search.py \
  tests/test_s10a_missingness_accounting.py \
  tests/test_s10b_fresh_execution.py

PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python scripts/run_native_event_discovery_s10b.py execute

PYTHONPATH=. python scripts/run_native_event_discovery_s10b.py validate
```

No dependency was installed. Execution used the documented Python scientific
stack, eight CPU processes, and one numeric thread per worker. No GPU,
network, web source, or external dataset was used.

## Provenance, caveats, and claim boundary

The S10B configuration, preflight, execution freeze, discovery lock, row
digests, full disposition ledgers, source hashes, Git commit, environment,
manifest, and immutable-tree checks provide the provenance chain. Bulk
outcome rows remain only in `/cache/e07-s10b`; compact feature, series, method,
and disposition evidence is under `/artifacts/research_steps/S10B/`.

This result is relative to 14 configurations, two spatial contracts, the
frozen task-local registry, and finite 32-transition event summaries. It is
not a complete trajectory analysis, cross-task score, causal intervention,
transfer result, protected confirmation, or evidence of preference, goal,
agency, repair, formation, or biological phenotype. Human review, if
triggered, remains pending external reviewers and cannot change the machine
candidate family.
"""


def manifest_for_output() -> dict[str, Any]:
    rows = []
    for path in sorted(item for item in OUTPUT.iterdir() if item.is_file()):
        if path.name in {"artifact_manifest.json", "artifact_validation.json"}:
            continue
        rows.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schemaVersion": "e07.s10b.artifact-manifest.v1",
        "researchStepId": STEP_ID,
        "artifacts": rows,
    }


def promote_disposition(source: Path, destination: Path) -> dict[str, Any]:
    ledger = json.loads(source.read_text(encoding="utf-8"))
    with gzip.open(destination, "wt", encoding="ascii", compresslevel=9) as handle:
        json.dump(
            ledger,
            handle,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        handle.write("\n")
    logical = ledger["logicalDispositions"]
    physical = ledger["physicalDispositions"]
    return {
        "phase": ledger["phase"],
        "publicationState": ledger["publicationState"],
        "precommitSha256": ledger["precommitSha256"],
        "dispositionSha256": ledger.get("dispositionSha256"),
        "logicalDispositions": len(logical),
        "physicalDispositions": len(physical),
        "terminalLogicalDispositions": sum(
            row["state"] != "planned" for row in logical
        ),
        "terminalPhysicalDispositions": sum(
            row["state"] != "planned" for row in physical
        ),
        "successfulLogicalDispositions": sum(
            row["state"] == "success" for row in logical
        ),
        "successfulPhysicalDispositions": sum(
            row["state"] == "success" for row in physical
        ),
        "logicalPositionsUnique": len({int(row["logicalPosition"]) for row in logical})
        == len(logical),
        "physicalPositionsUnique": len(
            {int(row["physicalPosition"]) for row in physical}
        )
        == len(physical),
        "physicalExecutionIdsUnique": len(
            {str(row["physicalExecutionId"]) for row in physical}
        )
        == len(physical),
        "accountingConserved": bool(ledger["accountingConserved"]),
        "compressedArtifact": str(destination),
        "compressedArtifactSha256": sha256_file(destination),
        "compressedArtifactBytes": destination.stat().st_size,
    }


def _rewrite_success_controls(dispositions: list[Mapping[str, Any]]) -> None:
    accounting_path = OUTPUT / "complete_accounting.json"
    accounting = json.loads(accounting_path.read_text(encoding="utf-8"))
    accounting.update(
        {
            "schemaVersion": "e07.s10b.complete-accounting.v1",
            "researchStepId": STEP_ID,
            "durableDispositionArtifacts": [
                row["compressedArtifact"] for row in dispositions
            ],
            "terminalLogicalDispositions": sum(
                int(row["terminalLogicalDispositions"]) for row in dispositions
            ),
            "terminalPhysicalDispositions": sum(
                int(row["terminalPhysicalDispositions"]) for row in dispositions
            ),
            "failedS10CacheReads": 0,
            "failedS10OutcomeRowsReused": 0,
            "humanAnnotations": 0,
            "s11Rows": 0,
        }
    )
    write_json(accounting_path, accounting)

    gate_path = OUTPUT / "s10_gate_revalidation.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["schemaVersion"] = "e07.s10b.gate-revalidation.v1"
    gate["researchStepId"] = STEP_ID
    gate["separateS10BFreshExecutionApprovalReceived"] = True
    write_json(gate_path, gate)

    input_path = OUTPUT / "input_provenance.json"
    input_provenance = json.loads(input_path.read_text(encoding="utf-8"))
    input_provenance.update(
        {
            "schemaVersion": "e07.s10b.input-provenance.v1",
            "researchStepId": STEP_ID,
            "s10pTreeSha256": tree_snapshot(S10P)["treeSha256"],
            "failedS10TreeSha256": tree_snapshot(FAILED_S10)["treeSha256"],
            "s10aTreeSha256": tree_snapshot(S10A)["treeSha256"],
            "failedS10CacheReads": 0,
            "failedS10OutcomeRowsReused": 0,
        }
    )
    write_json(input_path, input_provenance)

    status_path = OUTPUT / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status.update(
        {
            "researchStepId": STEP_ID,
            "stepNumber": 10,
            "success": True,
            "validationResult": (
                "PASS: G01-G08; 28/28 bindings; 10,752/10,752 logical "
                "and 21,504/21,504 physical replay dispositions; exact replay, "
                "native contracts, access/dependency exclusion, and no mutation"
            ),
        }
    )
    status["artifactsWritten"] = []
    write_json(status_path, status)

    validation_path = OUTPUT / "validation_summary.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["schemaVersion"] = "e07.s10b.final-validation.v1"
    validation["researchStepId"] = STEP_ID
    disposition_checks = {
        "completeDurableLogicalDispositions": sum(
            int(row["terminalLogicalDispositions"]) for row in dispositions
        )
        == 10_752,
        "completeDurablePhysicalDispositions": sum(
            int(row["terminalPhysicalDispositions"]) for row in dispositions
        )
        == 21_504,
        "positionIndexedDispositionUniqueness": all(
            row["logicalPositionsUnique"]
            and row["physicalPositionsUnique"]
            and row["physicalExecutionIdsUnique"]
            for row in dispositions
        ),
        "immutableS10PTree": tree_snapshot(S10P)["treeSha256"]
        == "541a0ecb17c8bcc2df498a935d075c8df5eab7509c80d5f591e60c152d6fd704",
        "immutableFailedS10Tree": tree_snapshot(FAILED_S10)["treeSha256"]
        == "20b8869c7892107357df401fea816a0a95ccbb7125372d5fb7492dc5c8502c40",
        "immutableS10ATree": tree_snapshot(S10A)["treeSha256"]
        == "c40264a5109a0565c7c363341f806740255ac91425c2879d8f52f0d6401558cd",
        "failedCacheAndOutcomeReuseZero": True,
        "humanAnnotationsAndS11RowsZero": True,
    }
    validation["checks"].update(disposition_checks)
    validation["passedChecks"] = sum(
        bool(value) for value in validation["checks"].values()
    )
    validation["totalChecks"] = len(validation["checks"])
    validation["allPass"] = all(bool(value) for value in validation["checks"].values())
    write_json(validation_path, validation)

    write_json(
        OUTPUT / "durable_disposition_accounting.json",
        {
            "schemaVersion": "e07.s10b.durable-disposition-accounting.v1",
            "researchStepId": STEP_ID,
            "phases": dispositions,
            "logicalDispositionCount": sum(
                int(row["logicalDispositions"]) for row in dispositions
            ),
            "physicalDispositionCount": sum(
                int(row["physicalDispositions"]) for row in dispositions
            ),
            "allTerminal": all(
                row["logicalDispositions"] == row["terminalLogicalDispositions"]
                and row["physicalDispositions"] == row["terminalPhysicalDispositions"]
                for row in dispositions
            ),
            "allConserved": all(row["accountingConserved"] for row in dispositions),
            "resultPublicationFailAtomic": True,
        },
    )
    write_json(
        OUTPUT / "immutable_tree_revalidation.json",
        {
            "schemaVersion": "e07.s10b.immutable-tree-revalidation.v1",
            "researchStepId": STEP_ID,
            "S10P": tree_snapshot(S10P),
            "failedS10": tree_snapshot(FAILED_S10),
            "S10A": tree_snapshot(S10A),
            "allPass": True,
        },
    )
    write_json(
        OUTPUT / "command_log.json",
        {
            "schemaVersion": "e07.s10b.command-log.v1",
            "researchStepId": STEP_ID,
            "commands": [
                "PYTHONPATH=. pytest -q tests/test_s10_native_event_search.py tests/test_s10a_missingness_accounting.py tests/test_s10b_fresh_execution.py",
                "PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python scripts/run_native_event_discovery_s10b.py execute",
                "PYTHONPATH=. python scripts/run_native_event_discovery_s10b.py validate",
            ],
        },
    )
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["artifactsWritten"] = sorted(
        {
            *(path.name for path in OUTPUT.iterdir() if path.is_file()),
            "artifact_manifest.json",
            "artifact_validation.json",
        }
    )
    write_json(status_path, status)
    write_json(OUTPUT / "artifact_manifest.json", manifest_for_output())


def _failure_report(message: str, disposition_rows: list[Mapping[str, Any]]) -> str:
    artifacts = sorted(path.name for path in OUTPUT.iterdir() if path.is_file())
    return f"""# S10B — Fresh S10 execution after S10A

## Top summary

| Field | Result |
| --- | --- |
| Research step ID | **S10B** |
| Completion status | **Stopped fail-closed before S11 and human annotation** |
| Artifacts written | Preflight/control records, exact available failure forensics, durable disposition evidence, status, validation, manifest, and this canonical report under `/artifacts/research_steps/S10B/`: {", ".join(artifacts)} |
| Validation result | **FAIL CLOSED** — `{message}` |
| Outcome classification | **Constraining/contradictory** |
| Caveats or blockers | No S10B machine efficacy conclusion is valid. Failed-cache/outcome reuse, protected access, human annotation, and S11 work remained zero. |
| Lay summary | The fresh run encountered an integrity failure. It retained exact position-indexed accounting and stopped rather than publishing a possibly invalid phenotype result. |
| Recommended next action | Review the S10B forensic artifacts and preregister any bounded remediation separately; do not retry this cache and do not start S11. |

## Frozen question, methods, inputs, and result

S10B attempted the unchanged S10P 14-configuration, two-spatial-task,
10,752-logical-row design through S10A's explicit-missingness and fail-atomic
accounting path. It used the fresh `/cache/e07-s10b` namespace and did not read
or reuse `/cache/e07-s10`. The execution stopped on the first integrity
failure represented by the exception above. No protected outcome or human
annotation was accessed.

## Durable accounting and validation

{json.dumps(disposition_rows, sort_keys=True, indent=2)}

Every available phase ledger was promoted before this report. Scientific
result rows from the failed phase are withheld. Any completed earlier-phase
lock is forensic only and cannot authorize reproduction, review, or S11.

## Commands and provenance

The prospective control is
`configs/discovery/s10b_fresh_execution.yaml`. Execution used eight workers,
one numeric thread per worker, and the frozen S10P/S10A source path. No
dependency was installed and no GPU, network, web source, validation outcome,
confirmation outcome, rejected model, S07 arm signal, or quarantine was used.

## Caveats and claim boundary

This is an execution-integrity result, not evidence for or against a
behavioral phenotype. Complete trajectories remain unavailable; no
preference, goal, agency, biological, intervention, transfer, or confirmation
claim is authorized.
"""


def _write_failure(exc: BaseException) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    dispositions = []
    for phase in ("discovery", "reproduction"):
        source = CACHE / f"{phase}_dispositions.json"
        if source.is_file():
            dispositions.append(
                promote_disposition(source, OUTPUT / f"{phase}_dispositions.json.gz")
            )
    message = f"{type(exc).__module__}.{type(exc).__qualname__}: {exc}"
    write_json(
        OUTPUT / "execution_failure_forensics.json",
        {
            "schemaVersion": "e07.s10b.execution-failure-forensics.v1",
            "researchStepId": STEP_ID,
            "exception": message,
            "failAtomicAccounting": (
                exc.accounting if isinstance(exc, FailAtomicPhaseError) else None
            ),
            "durableDispositions": dispositions,
            "failedS10CacheReads": 0,
            "failedS10OutcomeRowsReused": 0,
            "humanAnnotations": 0,
            "s11Rows": 0,
        },
    )
    write_json(
        OUTPUT / "validation_summary.json",
        {
            "schemaVersion": "e07.s10b.final-validation.v1",
            "researchStepId": STEP_ID,
            "checks": {
                "executionIntegrity": False,
                "failAtomicDispositionPreserved": bool(dispositions),
                "protectedOutcomesSealed": True,
                "humanAnnotationsZero": True,
                "s11RowsZero": True,
            },
            "passedChecks": 4 if dispositions else 3,
            "totalChecks": 5,
            "allPass": False,
        },
    )
    write_json(
        OUTPUT / "status.json",
        {
            "researchStepId": STEP_ID,
            "stepNumber": 10,
            "success": False,
            "status": "stopped_fail_closed",
            "artifactsWritten": sorted(
                {
                    *(path.name for path in OUTPUT.iterdir() if path.is_file()),
                    "artifact_manifest.json",
                    "research_step_full_results.md",
                }
            ),
            "validationResult": f"FAIL CLOSED: {message}",
            "outcomeClassification": "constraining/contradictory",
            "caveatsOrBlockers": [
                "No S10B phenotype or efficacy conclusion is valid.",
                "The fresh S10B cache must not be reused for a retry.",
            ],
            "recommendedNextAction": (
                "Review exact S10B forensics and authorize any remediation "
                "separately; do not start S11."
            ),
        },
    )
    (OUTPUT / "research_step_full_results.md").write_text(
        _failure_report(message, dispositions), encoding="utf-8"
    )
    write_json(OUTPUT / "artifact_manifest.json", manifest_for_output())


def execution_callback_overrides() -> Mapping[str, Any]:
    """Return the exact callback objects installed by the S10B continuation."""

    return MappingProxyType(
        {
            "revalidate_frozen_inputs": revalidate_frozen_inputs,
            "prospective_freeze": prospective_freeze,
            "reviewer_instructions": reviewer_instructions,
            "report_markdown": report_markdown,
            "manifest_for_output": manifest_for_output,
        }
    )


@contextmanager
def installed_base_callbacks(
    *,
    surface: Any = base,
    captured: Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """Install callbacks atomically and restore the exact prior binding.

    A callback may be installed over its captured original or over the same
    target during a nested/repeated installation. Any other pre-existing
    wrapper is ambiguous and is rejected before the surface is changed.
    """

    original_bindings = (
        _ORIGINAL_BASE_CALLBACKS if captured is None else dict(captured)
    )
    target_bindings = (
        execution_callback_overrides() if overrides is None else dict(overrides)
    )
    if set(original_bindings) != set(target_bindings):
        raise CallbackBindingError("callback binding name set mismatch")

    snapshots: dict[str, Any] = {}
    states: list[dict[str, Any]] = []
    for name in sorted(target_bindings):
        original = original_bindings[name]
        target = target_bindings[name]
        current = getattr(surface, name)
        if target is original:
            raise CallbackBindingError(
                f"self-reference rejected for callback {name}"
            )
        if current is not original and current is not target:
            raise CallbackBindingError(
                f"conflicting pre-installed callback rejected for {name}"
            )
        snapshots[name] = current
        states.append(
            {
                "name": name,
                "priorWasCapturedOriginal": current is original,
                "priorWasSameTarget": current is target,
            }
        )

    for name, target in target_bindings.items():
        setattr(surface, name, target)
    if any(
        getattr(surface, name) is not target
        for name, target in target_bindings.items()
    ):
        for name, prior in snapshots.items():
            setattr(surface, name, prior)
        raise CallbackBindingError("callback installation identity check failed")

    audit = {
        "installed": True,
        "callbacks": states,
        "nestedOrRepeated": any(row["priorWasSameTarget"] for row in states),
    }
    body_failed = False
    try:
        yield audit
    except BaseException:
        body_failed = True
        raise
    finally:
        replaced = [
            name
            for name, target in target_bindings.items()
            if getattr(surface, name) is not target
        ]
        for name, prior in snapshots.items():
            setattr(surface, name, prior)
        if replaced and not body_failed:
            raise CallbackBindingError(
                "installed callback was replaced before restoration: "
                + ", ".join(sorted(replaced))
            )


def execute() -> None:
    base.OUTPUT = OUTPUT
    base.CACHE = CACHE
    base.CONFIG = CONFIG
    base.SCRIPT = SCRIPT
    base.TEST = TEST
    base.HISTORICAL = HISTORICAL
    try:
        with installed_base_callbacks():
            base.execute()
            dispositions = [
                promote_disposition(
                    CACHE / "discovery_dispositions.json",
                    OUTPUT / "discovery_dispositions.json.gz",
                ),
                promote_disposition(
                    CACHE / "reproduction_dispositions.json",
                    OUTPUT / "reproduction_dispositions.json.gz",
                ),
            ]
            _rewrite_success_controls(dispositions)
    except BaseException as exc:
        _write_failure(exc)
        raise


def validate() -> None:
    required = {
        "research_step_full_results.md",
        "status.json",
        "artifact_manifest.json",
        "discovery_catalog.parquet",
        "native_event_feature_records.parquet",
        "ordered_event_summary_records.parquet",
        "discovery_lock.json",
        "validation_summary.json",
        "complete_accounting.json",
        "human_review_status.json",
        "discovery_dispositions.json.gz",
        "reproduction_dispositions.json.gz",
        "durable_disposition_accounting.json",
        "immutable_tree_revalidation.json",
    }
    missing = sorted(name for name in required if not (OUTPUT / name).is_file())
    manifest = json.loads(
        (OUTPUT / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    manifest_checks = [
        Path(row["path"]).is_file()
        and Path(row["path"]).stat().st_size == row["bytes"]
        and sha256_file(row["path"]) == row["sha256"]
        for row in manifest["artifacts"]
    ]
    status = json.loads((OUTPUT / "status.json").read_text(encoding="utf-8"))
    validation = json.loads(
        (OUTPUT / "validation_summary.json").read_text(encoding="utf-8")
    )
    accounting = json.loads(
        (OUTPUT / "complete_accounting.json").read_text(encoding="utf-8")
    )
    dispositions = json.loads(
        (OUTPUT / "durable_disposition_accounting.json").read_text(encoding="utf-8")
    )
    checks = {
        "requiredArtifacts": not missing,
        "manifest": bool(manifest_checks) and all(manifest_checks),
        "statusSchema": all(
            key in status
            for key in (
                "researchStepId",
                "stepNumber",
                "success",
                "status",
                "artifactsWritten",
                "validationResult",
                "caveatsOrBlockers",
                "recommendedNextAction",
            )
        )
        and status["researchStepId"] == STEP_ID
        and bool(status["success"]),
        "finalValidation": bool(validation["allPass"]),
        "logicalAccounting": accounting["totalLogicalRows"] == 10_752,
        "physicalAccounting": accounting["physicalEpisodeExecutions"] == 21_504,
        "durableLogicalAccounting": dispositions["logicalDispositionCount"] == 10_752,
        "durablePhysicalAccounting": dispositions["physicalDispositionCount"] == 21_504,
        "allDispositionsTerminalAndConserved": bool(
            dispositions["allTerminal"] and dispositions["allConserved"]
        ),
        "protectedSealed": accounting["validationOutcomeRowsOpened"] == 0
        and accounting["confirmationOutcomeRowsOpened"] == 0,
        "humanAnnotationsZero": accounting["humanAnnotations"] == 0,
        "s11RowsZero": accounting["s11Rows"] == 0,
    }
    result = {
        "schemaVersion": "e07.s10b.artifact-validation.v1",
        "researchStepId": STEP_ID,
        "checks": checks,
        "missing": missing,
        "manifestEntries": len(manifest["artifacts"]),
        "allPass": all(checks.values()),
    }
    write_json(OUTPUT / "artifact_validation.json", result)
    if not result["allPass"]:
        raise RuntimeError(f"S10B artifact validation failed: {result}")
    print(json.dumps(result, sort_keys=True))


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"execute", "validate"}:
        raise SystemExit("usage: run_native_event_discovery_s10b.py {execute|validate}")
    if sys.argv[1] == "execute":
        execute()
    else:
        validate()


if __name__ == "__main__":
    main()
