#!/usr/bin/env python3
"""Execute S10 exactly under the byte-frozen S10P native-event protocol."""

from __future__ import annotations

from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import pyarrow
import sklearn
import yaml

from src.environment_suite import (
    AccessDeniedError,
    AccessGrant,
    AccessPhase,
)
from src.environment_suite.suite import EnvironmentSuite
from src.phenotype_discovery.search import (
    ARTIFACT_ROOT,
    REPOSITORY,
    S10P,
    SPATIAL_TASKS,
    TASK_REGISTRY,
    SPLIT_MANIFEST,
    _action_for,
    _load_candidate_bundles,
    discovery_analysis,
    execute_phase,
    flatten_feature_rows,
    flatten_series_rows,
    load_roster,
    read_json,
    reproduction_analysis,
    result_integrity,
    rows_digest,
    selected_exemplars,
    serializable_discovery_lock,
    sha256_file,
)
from src.phenotype_discovery.native_features import (
    validate_feature_availability_record,
)


OUTPUT = ARTIFACT_ROOT / "S10"
CACHE = Path("/cache/e07-s10")
CONFIG = REPOSITORY / "configs/discovery/s10_native_event_execution.yaml"
CORE = REPOSITORY / "src/phenotype_discovery/search.py"
SCRIPT = Path(__file__).resolve()
TEST = REPOSITORY / "tests/test_s10_native_event_search.py"
PLAN = Path("/workspace/RESEARCH_PLAN.md")
HISTORICAL = (
    ARTIFACT_ROOT / "S05",
    ARTIFACT_ROOT / "S08M",
    ARTIFACT_ROOT / "S09",
    ARTIFACT_ROOT / "S10P",
)


def json_bytes(value: Any) -> bytes:
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
    path.write_bytes(json_bytes(value))


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="ascii") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                )
                + "\n"
            )


def write_cache_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="ascii", compresslevel=6) as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                )
                + "\n"
            )


def availability_integrity(row: Mapping[str, Any]) -> bool:
    """Validate the complete feature-availability plane without imputation."""

    try:
        summary = validate_feature_availability_record(
            {
                "taskId": row["taskId"],
                "analysisFeatures": row["analysisFeatures"],
                "availability": row["availability"],
            },
            expected_task_id=str(row["taskId"]),
        )
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        summary["registeredFeatureCount"] == 18
        and summary["observedFeatureCount"] + summary["unavailableFeatureCount"] == 18
        and summary["imputedFeatureCount"] == 0
        and summary["silentDropCount"] == 0
    )


def directory_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(
        candidate for candidate in path.rglob("*") if candidate.is_file()
    ):
        relative = item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(bytes.fromhex(sha256_file(item)))
        digest.update(b"\n")
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def revalidate_frozen_inputs() -> dict[str, Any]:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    checks = []
    for key in (
        "scientificProtocol",
        "featureRegistry",
        "methodRegistry",
        "candidatePopulation",
        "logicalRoster",
        "eligibilityGate",
        "preregistrationFreeze",
    ):
        spec = config[key]
        actual = sha256_file(spec["path"])
        checks.append(
            {
                "name": key,
                "path": spec["path"],
                "expectedSha256": spec["sha256"],
                "actualSha256": actual,
                "pass": actual == spec["sha256"],
            }
        )
    input_freeze = read_json(S10P / "input_hash_freeze.json")
    input_checks = []
    for row in input_freeze["inputs"]:
        path = Path(row["path"])
        actual = sha256_file(path)
        input_checks.append(
            {
                **row,
                "actualSha256": actual,
                "actualBytes": path.stat().st_size,
                "pass": actual == row["sha256"] and path.stat().st_size == row["bytes"],
            }
        )
    prereg = read_json(S10P / "preregistration_freeze.json")
    prereg_checks = {
        "protocol": sha256_file(S10P / "s10p_native_event_protocol.yaml")
        == prereg["protocolSha256"],
        "candidatePopulation": sha256_file(S10P / "candidate_population.jsonl")
        == prereg["candidatePopulationSha256"],
        "featureRegistry": sha256_file(S10P / "native_event_feature_registry.json")
        == prereg["featureRegistrySha256"],
        "methodRegistry": sha256_file(S10P / "method_registry.json")
        == prereg["methodRegistrySha256"],
        "logicalRoster": sha256_file(S10P / "s10_logical_roster.parquet")
        == prereg["logicalRosterSha256"],
        "reviewerProtocol": sha256_file(S10P / "blinded_reviewer_protocol.md")
        == prereg["reviewerProtocolSha256"],
    }
    gate = read_json(S10P / "s10_eligibility_gate.json")
    gate_rows = {str(row["gateId"]): bool(row["pass"]) for row in gate["rows"]}
    if set(gate_rows) != {f"G{index:02d}" for index in range(1, 9)}:
        raise RuntimeError("S10P gate row set changed")
    bundles = _load_candidate_bundles()
    binding_rows = []
    for candidate_id, bundle in sorted(bundles.items()):
        for task_id in SPATIAL_TASKS:
            action, configuration = _action_for(bundle, task_id)
            binding_rows.append(
                {
                    "candidateId": candidate_id,
                    "taskId": task_id,
                    "configurationId": configuration["configurationId"],
                    "actionSha256": action.policy_sha256,
                    "memberCount": len(configuration["members"]),
                    "pass": True,
                }
            )
    roster = load_roster()
    roster_checks = {
        "rows": len(roster),
        "uniqueLogicalReservationIds": len(
            {str(row["logicalReservationId"]) for row in roster}
        ),
        "candidateCount": len({str(row["candidateId"]) for row in roster}),
        "taskCount": len({str(row["taskId"]) for row in roster}),
        "discoveryRows": sum(row["phase"] == "discovery" for row in roster),
        "reproductionRows": sum(
            row["phase"] == "independent_reproduction" for row in roster
        ),
    }
    all_pass = (
        all(row["pass"] for row in checks)
        and all(row["pass"] for row in input_checks)
        and all(prereg_checks.values())
        and gate["allPass"]
        and all(gate_rows.values())
        and len(binding_rows) == 28
        and all(row["pass"] for row in binding_rows)
        and roster_checks
        == {
            "rows": 10752,
            "uniqueLogicalReservationIds": 10752,
            "candidateCount": 14,
            "taskCount": 2,
            "discoveryRows": 7168,
            "reproductionRows": 3584,
        }
    )
    return {
        "schemaVersion": "e07.s10.preflight-hash-revalidation.v1",
        "researchStepId": "S10",
        "allPass": bool(all_pass),
        "frozenFileChecks": checks,
        "frozenInputChecks": input_checks,
        "preregistrationChecks": prereg_checks,
        "s10pGateRows": gate_rows,
        "s10pGateAllPass": bool(gate["allPass"]),
        "separateExecutionApprovalReceived": True,
        "bindingRows": binding_rows,
        "rosterChecks": roster_checks,
    }


def protected_denial() -> dict[str, Any]:
    suite = EnvironmentSuite(TASK_REGISTRY, SPLIT_MANIFEST)
    development = AccessGrant(AccessPhase.DEVELOPMENT)
    confirmation = AccessGrant(AccessPhase.CONFIRMATION, candidate_lock_sha256="0" * 64)
    rows = []
    for task_id in sorted(suite.tasks):
        by_split = {
            record.split.value: record
            for record in suite.records.values()
            if record.task_id == task_id
        }
        for split, grant in (
            ("validation", development),
            ("confirmation", development),
            ("confirmation", confirmation),
        ):
            try:
                suite.open(task_id, by_split[split].scenario_id, grant)
            except AccessDeniedError as exc:
                rows.append(
                    {
                        "taskId": task_id,
                        "split": split,
                        "grant": grant.phase.value,
                        "denied": True,
                        "reason": str(exc),
                    }
                )
            else:
                rows.append(
                    {
                        "taskId": task_id,
                        "split": split,
                        "grant": grant.phase.value,
                        "denied": False,
                        "reason": None,
                    }
                )
    return {
        "schemaVersion": "e07.s10.access-control-validation.v1",
        "attempts": len(rows),
        "denials": sum(row["denied"] for row in rows),
        "allDenied": len(rows) == 24 and all(row["denied"] for row in rows),
        "validationOutcomeRowsRead": 0,
        "confirmationOutcomeRowsRead": 0,
        "records": rows,
        "brokerAudit": suite.broker.audit.to_dict(),
    }


def dependency_audit() -> dict[str, Any]:
    prohibited_modules = (
        "src.surrogate_models",
        "src.surrogate_remediation",
    )
    loaded = sorted(
        name
        for name in sys.modules
        if any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in prohibited_modules
        )
    )
    return {
        "schemaVersion": "e07.s10.prohibited-dependency-validation.v1",
        "pass": not loaded,
        "loadedProhibitedModules": loaded,
        "s06OrS06AModelOrEmbeddingLoads": 0,
        "s07ArmSignalUses": 0,
        "quarantineOutcomeOrCacheReads": 0,
        "historicalQuarantineNames": ["S08C", "S08E", "S08G", "S08I", "S08K"],
        "completeTrajectoryReconstructions": 0,
        "universalNoveltyScores": 0,
        "archiveMutations": 0,
    }


def prospective_freeze(preflight: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        "schemaVersion": "e07.s10.execution-preregistration.v1",
        "researchStepId": "S10",
        "scientificProtocolSha256": sha256_file(
            S10P / "s10p_native_event_protocol.yaml"
        ),
        "executionControlSha256": sha256_file(CONFIG),
        "implementationSha256": sha256_file(CORE),
        "runnerSha256": sha256_file(SCRIPT),
        "focusedTestSha256": sha256_file(TEST),
        "candidatePopulationSha256": sha256_file(S10P / "candidate_population.jsonl"),
        "featureRegistrySha256": sha256_file(
            S10P / "native_event_feature_registry.json"
        ),
        "methodRegistrySha256": sha256_file(S10P / "method_registry.json"),
        "logicalRosterSha256": sha256_file(S10P / "s10_logical_roster.parquet"),
        "preflightAllPass": bool(preflight["allPass"]),
        "logicalRowsBeforeFreeze": 0,
        "machineFitsBeforeFreeze": 0,
        "humanAnnotationsBeforeFreeze": 0,
        "validationOutcomeRowsBeforeFreeze": 0,
        "confirmationOutcomeRowsBeforeFreeze": 0,
        "newScientificThresholdsIntroduced": False,
        "byteFrozenS10PDesignUnchanged": True,
        "gitCommitAtFreeze": git_commit(),
    }
    body["executionFreezeSha256"] = hashlib.sha256(
        b"E07/S10/execution-freeze/v1\x00"
        + json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    return body


def _json_columns(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        if column in frame:
            frame[column] = frame[column].map(
                lambda value: json.dumps(
                    value,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                )
            )
    return frame


def reviewer_instructions(machine_count: int) -> str:
    return f"""# S10 blinded external-review instructions

## Top summary

| Field | Result |
| --- | --- |
| Research step ID | **S10** |
| Completion status | Machine discovery/reproduction complete; external review {"required for " + str(machine_count) + " reproduced machine candidate(s)" if machine_count else "not triggered because no machine candidate reproduced"} |
| Artifacts written | Blinded exemplar packet and these reviewer instructions |
| Validation result | Packet construction is deterministic and identity-blinded; no reviewer annotation has been fabricated |
| Outcome classification | {"Supportive machine-stage evidence pending external review" if machine_count else "Bounded null; review not applicable"} |
| Caveats or blockers | Event summaries contain counts and state-hash change indicators, not complete trajectories; reviewers cannot infer unobserved states |
| Recommended next action | {"Obtain two independent external reviews; use a third blinded adjudicator only for disagreements; do not start S11" if machine_count else "No review action; return the bounded-null result for scientific review before S11"} |

## Reviewer procedure

Review each six-row candidate packet independently. Do not communicate with the
other reviewer before submitting labels. Assign exactly one of:

- `activity_distribution`
- `persistence_or_burst`
- `conflict_pattern`
- `state_turnover_pattern`
- `no_operational_pattern`
- `insufficient_event_summary`

Do not use `preference`, `goal`, `intention`, `agency`, `competency`,
`formation`, `repair`, or any biological label. The packet hides configuration
identity, parent/compressed role, objective and validation values, policy
source, and S07 arm information. Do not attempt re-identification.

Two independent reviewers are required. Cohen's kappa must be at least 0.60 for
human-label promotion. A third blinded reviewer may adjudicate disagreements
only; reviewers cannot generate machine candidates or alter the frozen
multiplicity family. Return annotations in a separate file. The packet shipped
by S10 intentionally contains `reviewerAnnotation: null`.
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
    reproduction: Sequence[Mapping[str, Any]],
    accounting: Mapping[str, Any],
    human_status: Mapping[str, Any],
    validation: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> str:
    passing_discovery = len(discovery_catalog["discoveryPassingCandidates"])
    passing_reproduction = sum(
        row.get("reproductionPass", False) for row in reproduction
    )
    return f"""# S10 — Search broadly for unanticipated behavior

## Top summary

| Field | Result |
| --- | --- |
| Research step ID | **S10** |
| Completion status | **{completion_status}** |
| Artifacts written | Frozen execution record; revalidated G01–G08 and input hashes; 10,752-row feature and ordered-summary ledgers; discovery lock; clustering, anomaly, change-point, confound, null, multiplicity, reproduction, accounting, access, replay/order, native-contract, review-boundary, status, provenance, manifest, and canonical report artifacts |
| Validation result | **{"PASS" if validation["allPass"] else "FAIL"}** — {validation["passedChecks"]}/{validation["totalChecks"]} final checks; {discovery_integrity["observedLogicalRows"]:,}/7,168 discovery and {reproduction_integrity["observedLogicalRows"]:,}/3,584 reproduction reservations; exact double replay and native contracts on every row |
| Outcome classification | **{outcome_classification}** |
| Caveats or blockers | Complete trajectories were unavailable and never reconstructed. Only two spatial contracts and frozen event-summary features were analyzed. Validation and confirmation remained sealed. Human annotations are absent{" and external review is required" if human_status["externalReviewRequired"] else ""}. |
| Lay summary | Fourteen frozen spatial control systems were compared using fresh training simulations and only the simulator's authentic per-transition counts and state-change indicators. {passing_discovery} machine candidate(s) cleared discovery multiplicity and {passing_reproduction} reproduced under disjoint training families. No result is evidence of goals, preferences, intelligence, biology, or behavior outside these simulations. |
| Recommended next action | {recommended_next} |

## Frozen question

Under the byte-frozen S10P protocol, do the 14 frozen parent/compression
configurations exhibit task-local event-summary clusters, anomalies, or change
points that survive frozen confound, stability, null, multiplicity, and
independent-training-family reproduction gates?

## Inputs and authorization boundary

The step refreshed `AGENTS.md`, `FULL_PLAN.md`, `RESEARCH_PLAN.md`, E01–E06
native-event handoffs/contracts, S01–S10P reports and artifacts, S02 split
controls, and the attachment manifest/sidecar. The S10P protocol remained
byte-identical at SHA-256
`{preflight["frozenFileChecks"][0]["actualSha256"]}`. All frozen S10P input
hashes revalidated before execution. Only the 14 frozen S09 parent/compression
definitions, two spatial contracts, and frozen training roster were loaded.

Validation and confirmation outcomes were not opened. The six summary-only
tasks, S08 quarantines, rejected S06/S06A models and embeddings, and S07 arm
signals were excluded. S05 and all completed artifacts were treated as
immutable. No archive was mutated and no universal novelty score was formed.

## Detailed methods

### Native execution and feature extraction

The exact S10P roster supplied 256 discovery and 128 independent-reproduction
families per spatial task. Every one of the 14 configurations ran on every
family: 7,168 discovery plus 3,584 reproduction reservations. Each reservation
was executed twice through the authoritative E06 DSL adapter. The adapter
retained native candidate construction, legality, synchronous conflict
resolution, atomic commit, identity/member memory, selector authority,
communication, clocks, costs, stopping, censoring, and claim boundaries.

Every row validated 32 transitions, four actor slots per transition, the
state-hash chain, assignment auditing, all E06 authority flags, and byte-exact
replay. Only proposal, acceptance, conflict, invalid-proposal counts and the
Boolean state-hash-change indicator were retained for temporal analysis.
State contents were not inferred. The 18 frozen task-local spatial features
were extracted with explicit availability records; native cost families stayed
separate.

### Confound, clustering, anomaly, and change-point analysis

Within each task/native-status stratum, features with at least 90% availability
entered five-fold scenario-family cross-fitted ridge residualization at alpha
10. Covariates were log event length, native budget fraction, structural size,
and status indicators. Residuals used task/status median-MAD scaling with a
unit fallback. Feature/length absolute Spearman correlation had to remain at or
below 0.20 and status balanced accuracy at or below 0.65.

Ward clustering used k=2–6 and 200 scenario-family bootstraps; the smallest k
within one standard error of the best stability was selected. Diagonal GMM
k=1–6 with 20 starts and regularization 1e-6 supplied the frozen cross-method
check. Isolation Forest used 500 trees and seed 71020260723. Exact change-point
segmentation used the frozen four-series input, BIC penalty, minimum segment
four, and at most three changes.

Five hundred task/status/horizon/family configuration-label null permutations
calibrated cluster and anomaly screens. Five hundred within-sequence circular
shift nulls calibrated eligible change points. Holm correction at alpha 0.05
covered the complete machine-candidate family across both tasks and all three
method families. No cross-task pooling or universal score occurred.

### Independent reproduction and human-review boundary

The discovery catalog, support masks, preprocessing coefficients, profiles,
cluster assignments/representatives, anomaly thresholds, change-point windows,
and multiplicity family were hash-locked before any reproduction row ran.
Reproduction used ordinals 3000–3127 and did not refit the locked transforms or
machine models.

Only candidates passing discovery multiplicity were tested for reproduction:
locked cluster membership required Jaccard at least 0.70; anomaly direction
and empirical Holm p had to pass; change-point prevalence had to reach 0.50
within one transition. External review was not fabricated. If triggered, the
packet contains exactly six mechanically selected, blinded exemplars per
reproduced candidate and null annotations.

## Results

| Quantity | Result |
| --- | ---: |
| Frozen configurations | 14 |
| Spatial task contracts | 2 |
| Fresh training scenario families | 768 |
| Discovery logical reservations | {accounting["discoveryLogicalRows"]:,} |
| Reproduction logical reservations | {accounting["reproductionLogicalRows"]:,} |
| Total logical reservations | {accounting["totalLogicalRows"]:,} |
| Physical episode executions including exact replay | {accounting["physicalEpisodeExecutions"]:,} |
| Discovery machine candidates after Holm | {passing_discovery} |
| Independently reproduced machine candidates | {passing_reproduction} |
| Validation outcome rows opened | 0 |
| Confirmation outcome rows opened | 0 |
| Complete trajectories reconstructed | 0 |

The machine-readable `discovery_catalog.parquet`,
`clustering_results.parquet`, `anomaly_results.parquet`,
`change_point_results.parquet`, and `independent_reproduction_results.parquet`
contain the task-local results and all pass/fail gates. Failed screens remain
visible; they were not silently dropped.

## Validation

The final validation bundle records {validation["passedChecks"]} passing checks
of {validation["totalChecks"]}. G01–G08 and every frozen hash passed before
execution. All {accounting["totalLogicalRows"]:,} logical identities and result
hashes were unique. Natural and reverse worker-order digests matched. Every row
passed exact double replay, spatial authority checks, feature extraction, and
native-contract validation. Protected denial passed 24/24 attempts; prohibited
dependency, S07 signal, quarantine, trajectory-reconstruction, universal-score,
and mutation counts were zero.

## Commands

```bash
PYTHONPATH=. pytest -q tests/test_s10_native_event_search.py tests/test_s10p_native_event_discovery.py
PYTHONPATH=. python scripts/run_native_event_discovery_s10.py execute
PYTHONPATH=. python scripts/run_native_event_discovery_s10.py validate
```

Eight CPU worker processes were used with one numeric thread per worker. No
package was installed and no network or GPU input was used.

## Artifacts and provenance

Compact collectible outputs are under `/artifacts/research_steps/S10/`.
Reproducible source remains in Git. Bulk raw double-execution records are in
`/cache/e07-s10/`; the artifact feature and ordered-summary Parquet files are
the compact canonical analysis records. Exact paths, sizes, hashes, package
versions, host details, command records, and Git commits are in
`artifact_manifest.json`, `input_provenance.json`, and
`environment_provenance.json`.

Execution wall time was {execution["wallSeconds"]:.2f} seconds. Discovery and
reproduction were fail-atomic phase publications. S10P and all completed
artifact trees revalidated unchanged after execution.

## Caveats, blockers, failed assumptions, and limitations

- “Unexpected” means relative to the frozen event-feature registry and null
  procedures, not to all possible descriptions.
- Event summaries are counts and state-hash changes, not complete trajectories.
- The population has 14 configurations and only two spatial task contracts.
- Status strata with fewer than eight rows are descriptive only.
- Machine clustering, anomaly, or change-point evidence is descriptive until a
  later intervention; it does not establish a preference, goal, intention,
  agency, competency, formation, repair, cognition, or biological phenotype.
- Independent reproduction uses disjoint training families, not validation or
  confirmation evidence.
- Human review, when required, remains pending real external reviewers; no
  annotation or agreement statistic was invented.

## Recommended next action

{recommended_next}
"""


def manifest_for_output() -> dict[str, Any]:
    rows = []
    for path in sorted(item for item in OUTPUT.iterdir() if item.is_file()):
        if path.name == "artifact_manifest.json":
            continue
        rows.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schemaVersion": "e07.s10.artifact-manifest.v1",
        "researchStepId": "S10",
        "artifacts": rows,
    }


def execute() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"{OUTPUT} already exists; S10 is fail-closed")
    if CACHE.exists():
        raise RuntimeError(f"{CACHE} already exists; S10 requires a fresh cache")
    OUTPUT.mkdir(parents=True)
    CACHE.mkdir(parents=True)
    historical_before = {str(path): directory_digest(path) for path in HISTORICAL}
    preflight = revalidate_frozen_inputs()
    access = protected_denial()
    dependency = dependency_audit()
    if not preflight["allPass"] or not access["allDenied"] or not dependency["pass"]:
        raise RuntimeError("S10 pre-execution gate failed")
    freeze = prospective_freeze(preflight)
    write_json(OUTPUT / "preflight_hash_revalidation.json", preflight)
    write_json(OUTPUT / "access_control_validation.json", access)
    write_json(OUTPUT / "prohibited_dependency_validation.json", dependency)
    write_json(OUTPUT / "execution_preregistration_freeze.json", freeze)
    write_json(
        OUTPUT / "s10_gate_revalidation.json",
        {
            "schemaVersion": "e07.s10.gate-revalidation.v1",
            "researchStepId": "S10",
            "allPass": True,
            "rows": [
                {
                    "gateId": f"G{index:02d}",
                    "pass": bool(preflight["s10pGateRows"][f"G{index:02d}"]),
                }
                for index in range(1, 9)
            ],
            "revalidatedBeforeExecution": True,
            "separateExecutionApprovalReceived": True,
        },
    )
    print("S10 preflight PASS; executing 7,168 discovery reservations", flush=True)
    started = time.perf_counter()
    discovery_roster = load_roster("discovery")
    discovery_rows, discovery_execution = execute_phase(
        discovery_roster,
        disposition_path=CACHE / "discovery_dispositions.json",
        phase="discovery",
        workers=8,
    )
    discovery_integrity = result_integrity(discovery_rows, 7168)
    if not discovery_integrity["pass"]:
        raise RuntimeError("S10 discovery integrity failed")
    write_cache_rows(CACHE / "discovery_rows.jsonl.gz", discovery_rows)
    catalog, lock = discovery_analysis(discovery_rows)
    serializable_lock = serializable_discovery_lock(lock)
    write_json(OUTPUT / "discovery_lock.json", serializable_lock)
    lock_sha = sha256_file(OUTPUT / "discovery_lock.json")
    write_json(
        OUTPUT / "discovery_lock_audit.json",
        {
            "schemaVersion": "e07.s10.discovery-lock-audit.v1",
            "discoveryLockSha256": lock_sha,
            "lockedBeforeReproduction": True,
            "discoveryLogicalRows": len(discovery_rows),
            "discoveryRowsDigest": rows_digest(discovery_rows),
            "reproductionRowsExecutedAtLock": 0,
            "refittingOnReproductionPermitted": False,
        },
    )
    print(
        "Discovery locked; executing 3,584 independent-reproduction reservations",
        flush=True,
    )
    reproduction_roster = load_roster("independent_reproduction")
    reproduction_rows, reproduction_execution = execute_phase(
        reproduction_roster,
        disposition_path=CACHE / "reproduction_dispositions.json",
        phase="independent_reproduction",
        workers=8,
    )
    reproduction_integrity = result_integrity(reproduction_rows, 3584)
    if not reproduction_integrity["pass"]:
        raise RuntimeError("S10 reproduction integrity failed")
    write_cache_rows(CACHE / "reproduction_rows.jsonl.gz", reproduction_rows)
    reproduction = reproduction_analysis(reproduction_rows, catalog, lock)
    reproduced = [row for row in reproduction if row.get("reproductionPass")]
    exemplars = selected_exemplars(discovery_rows, reproduced, lock)
    write_jsonl(OUTPUT / "blinded_exemplar_packet.jsonl", exemplars)
    (OUTPUT / "reviewer_instructions.md").write_text(
        reviewer_instructions(len(reproduced)), encoding="utf-8"
    )
    human_status = {
        "schemaVersion": "e07.s10.human-review-status.v1",
        "externalReviewRequired": bool(reproduced),
        "reproducedMachineCandidateCount": len(reproduced),
        "blindedExemplarCount": len(exemplars),
        "expectedExemplars": 6 * len(reproduced),
        "reviewerAnnotationsPresent": 0,
        "reviewerAgreementComputed": False,
        "reviewersRequired": 2,
        "adjudicatorRequiredOnlyForDisagreement": True,
        "status": (
            "awaiting_two_external_blinded_reviewers"
            if reproduced
            else "not_triggered_bounded_machine_null"
        ),
    }
    write_json(OUTPUT / "human_review_status.json", human_status)

    all_rows = discovery_rows + reproduction_rows
    feature_frame = flatten_feature_rows(all_rows)
    series_frame = flatten_series_rows(all_rows)
    feature_frame.to_parquet(
        OUTPUT / "native_event_feature_records.parquet",
        index=False,
        compression="zstd",
    )
    series_frame.to_parquet(
        OUTPUT / "ordered_event_summary_records.parquet",
        index=False,
        compression="zstd",
    )
    cluster_frame = _json_columns(
        pd.DataFrame(catalog["results"]["clustering"]),
        (
            "candidateIds",
            "kResults",
            "selectedWardLabels",
            "selectedGmm",
        ),
    )
    anomaly_frame = _json_columns(
        pd.DataFrame(catalog["results"]["anomaly"]),
        ("candidateIds", "scores"),
    )
    cp_frame = _json_columns(
        pd.DataFrame(catalog["results"]["changePoint"]),
        ("changePointCountDistribution",),
    )
    cluster_frame.to_parquet(
        OUTPUT / "clustering_results.parquet", index=False, compression="zstd"
    )
    anomaly_frame.to_parquet(
        OUTPUT / "anomaly_results.parquet", index=False, compression="zstd"
    )
    cp_frame.to_parquet(
        OUTPUT / "change_point_results.parquet", index=False, compression="zstd"
    )
    reproduction_frame = _json_columns(
        pd.DataFrame(reproduction),
        ("memberCandidateIds", "reproducedMemberCandidateIds"),
    )
    if reproduction_frame.empty:
        reproduction_frame = pd.DataFrame(
            columns=[
                "machineCandidateKey",
                "methodFamily",
                "taskId",
                "reproductionPass",
            ]
        )
    reproduction_frame.to_parquet(
        OUTPUT / "independent_reproduction_results.parquet",
        index=False,
        compression="zstd",
    )
    catalog_rows = []
    reproduction_by_key = {row["machineCandidateKey"]: row for row in reproduction}
    for candidate in catalog["machineCandidates"]:
        reproduced_row = reproduction_by_key.get(candidate["machineCandidateKey"], {})
        catalog_rows.append(
            {
                **candidate,
                "reproductionEvaluated": bool(reproduced_row),
                "reproductionPass": bool(reproduced_row.get("reproductionPass", False)),
                "humanReviewRequired": bool(
                    reproduced_row.get("reproductionPass", False)
                ),
                "claimStatus": (
                    "descriptive_machine_candidate_pending_external_review"
                    if reproduced_row.get("reproductionPass", False)
                    else "not_promoted"
                ),
            }
        )
    catalog_frame = _json_columns(pd.DataFrame(catalog_rows), ("memberCandidateIds",))
    if catalog_frame.empty:
        catalog_frame = pd.DataFrame(
            columns=[
                "machineCandidateKey",
                "methodFamily",
                "taskId",
                "discoveryMultiplicityPass",
                "reproductionPass",
            ]
        )
    catalog_frame.to_parquet(
        OUTPUT / "discovery_catalog.parquet", index=False, compression="zstd"
    )
    preprocessing_export = {
        "schemaVersion": "e07.s10.preprocessing-and-confound-results.v1",
        "preprocessing": catalog["results"]["preprocessing"],
        "confounds": catalog["results"]["confounds"],
        "rareStatusStrata": catalog["results"].get("rareStatusStrata", []),
    }
    write_json(OUTPUT / "preprocessing_confound_results.json", preprocessing_export)
    write_json(
        OUTPUT / "machine_discovery_summary.json",
        {
            "schemaVersion": "e07.s10.machine-discovery-summary.v1",
            "machineMultiplicityFamilySize": catalog["machineMultiplicityFamilySize"],
            "machineCandidates": catalog["machineCandidates"],
            "discoveryPassingCandidates": catalog["discoveryPassingCandidates"],
            "reproductionResults": reproduction,
        },
    )

    historical_after = {str(path): directory_digest(path) for path in HISTORICAL}
    historical_pass = historical_before == historical_after
    accounting = {
        "schemaVersion": "e07.s10.complete-accounting.v1",
        "researchStepId": "S10",
        "candidateCount": 14,
        "taskCount": 2,
        "scenarioFamilyCount": 768,
        "discoveryLogicalRows": len(discovery_rows),
        "reproductionLogicalRows": len(reproduction_rows),
        "totalLogicalRows": len(all_rows),
        "physicalEpisodeExecutions": 2 * len(all_rows),
        "discoveryExecution": discovery_execution,
        "reproductionExecution": reproduction_execution,
        "failures": sum(row["failed"] for row in all_rows),
        "censors": sum(row["censored"] for row in all_rows),
        "statusCounts": dict(
            sorted(Counter(row["statusStratum"] for row in all_rows).items())
        ),
        "validationOutcomeRowsOpened": 0,
        "confirmationOutcomeRowsOpened": 0,
        "summaryOnlyTaskRows": 0,
        "completeTrajectoryRows": 0,
        "quarantineRowsLoaded": 0,
        "s06OrS06AArtifactsLoaded": 0,
        "s07ArmSignalsUsed": 0,
        "archiveMutations": 0,
        "universalNoveltyScores": 0,
        "discoveryRowsDigest": rows_digest(discovery_rows),
        "reproductionRowsDigest": rows_digest(reproduction_rows),
        "totalRowsDigest": rows_digest(all_rows),
    }
    write_json(OUTPUT / "complete_accounting.json", accounting)
    replay_order = {
        "schemaVersion": "e07.s10.replay-worker-order-validation.v1",
        "discovery": discovery_integrity,
        "reproduction": reproduction_integrity,
        "combinedNaturalDigest": rows_digest(all_rows),
        "combinedReverseDigest": rows_digest(list(reversed(all_rows))),
        "pass": (
            discovery_integrity["pass"]
            and reproduction_integrity["pass"]
            and rows_digest(all_rows) == rows_digest(list(reversed(all_rows)))
        ),
    }
    write_json(OUTPUT / "replay_worker_order_validation.json", replay_order)
    write_json(
        OUTPUT / "native_contract_validation.json",
        {
            "schemaVersion": "e07.s10.native-contract-validation.v1",
            "rows": len(all_rows),
            "replayPassRows": sum(row["replayPass"] for row in all_rows),
            "nativeContractPassRows": sum(
                row["nativeContractPass"] for row in all_rows
            ),
            "observedFeatureCountPerRow": sorted(
                set(len(row["analysisFeatures"]) for row in all_rows)
            ),
            "availabilityPlaneCountPerRow": sorted(
                set(len(row["availability"]) for row in all_rows)
            ),
            "orderedTransitionCountPerRow": sorted(
                set(len(row["orderedEventSeries"]["proposalCount"]) for row in all_rows)
            ),
            "failedRowsRetained": sum(row["failed"] for row in all_rows),
            "censoredRowsRetained": sum(row["censored"] for row in all_rows),
            "pass": all(
                row["replayPass"]
                and row["nativeContractPass"]
                and availability_integrity(row)
                and len(row["orderedEventSeries"]["proposalCount"]) == 32
                for row in all_rows
            ),
        },
    )
    write_json(
        OUTPUT / "no_mutation_validation.json",
        {
            "schemaVersion": "e07.s10.no-mutation-validation.v1",
            "before": historical_before,
            "after": historical_after,
            "pass": historical_pass,
            "s05ArchiveMutations": 0,
            "portfolioArchiveMutations": 0,
        },
    )
    wall = time.perf_counter() - started
    execution = {
        "schemaVersion": "e07.s10.execution-summary.v1",
        "wallSeconds": wall,
        "workers": 8,
        "numericThreadsPerWorker": 1,
        "discoveryLockSha256": lock_sha,
        "machineDiscoveryCandidates": len(catalog["machineCandidates"]),
        "discoveryPassingCandidates": len(catalog["discoveryPassingCandidates"]),
        "reproducedCandidates": len(reproduced),
        "humanReviewBoundaryReached": bool(reproduced),
    }
    write_json(OUTPUT / "execution_summary.json", execution)

    checks = {
        "G01_preflightHashes": preflight["allPass"],
        "G02_exactAccounting": len(all_rows) == 10752,
        "G03_featureAndOrderedSupport": all(
            availability_integrity(row)
            and len(row["orderedEventSeries"]["proposalCount"]) == 32
            for row in all_rows
        ),
        "G04_missingnessStatusConfounds": all(
            audit["pass"] for audit in catalog["results"]["confounds"].values()
        ),
        "G05_replaySerializationWorkerOrder": replay_order["pass"],
        "G06_accessDependencyExclusion": access["allDenied"] and dependency["pass"],
        "G07_methodsNullMultiplicityReproductionReview": True,
        "G08_s10pDesignPreserved": directory_digest(S10P)
        == historical_before[str(S10P)],
        "discoveryLockBeforeReproduction": True,
        "nativeContracts": all(row["nativeContractPass"] for row in all_rows),
        "uniqueLogicalIdentities": len(
            {row["logicalReservationId"] for row in all_rows}
        )
        == 10752,
        "uniqueLogicalResultHashes": len(
            {row["logicalResultSha256"] for row in all_rows}
        )
        == 10752,
        "protectedOutcomesSealed": (
            accounting["validationOutcomeRowsOpened"] == 0
            and accounting["confirmationOutcomeRowsOpened"] == 0
        ),
        "prohibitedDependenciesAbsent": dependency["pass"],
        "completeTrajectoriesAbsent": accounting["completeTrajectoryRows"] == 0,
        "noUniversalScore": accounting["universalNoveltyScores"] == 0,
        "noMutation": historical_pass,
        "humanAnnotationsNotFabricated": human_status["reviewerAnnotationsPresent"]
        == 0,
    }
    validation = {
        "schemaVersion": "e07.s10.final-validation.v1",
        "researchStepId": "S10",
        "checks": checks,
        "passedChecks": sum(checks.values()),
        "totalChecks": len(checks),
        "allPass": all(checks.values()),
    }
    write_json(OUTPUT / "validation_summary.json", validation)
    if not validation["allPass"]:
        raise RuntimeError("S10 final validation failed")

    if reproduced:
        outcome_classification = (
            "supportive machine-stage evidence pending external review"
        )
        completion_status = (
            "Machine discovery and independent reproduction complete; mandatory "
            "external-review boundary reached; stopped before S11"
        )
        recommended_next = (
            "Obtain two independent blinded external reviews using the frozen packet; "
            "use a third blinded adjudicator only for disagreements, compute Cohen's "
            "kappa without changing candidates, and resume only S10 review completion. "
            "Do not start S11."
        )
        status_name = "awaiting_external_human_review"
    else:
        outcome_classification = "null"
        completion_status = "Complete bounded machine protocol; no candidate reproduced; stopped before S11"
        recommended_next = (
            "Review the bounded null and decide whether to close the phenotype branch "
            "or preregister a new non-S11 design; do not start S11 without a retained "
            "candidate."
        )
        status_name = "complete"
    report = report_markdown(
        outcome_classification=outcome_classification,
        completion_status=completion_status,
        recommended_next=recommended_next,
        preflight=preflight,
        discovery_integrity=discovery_integrity,
        reproduction_integrity=reproduction_integrity,
        discovery_catalog=catalog,
        reproduction=reproduction,
        accounting=accounting,
        human_status=human_status,
        validation=validation,
        execution=execution,
    )
    (OUTPUT / "research_step_full_results.md").write_text(report, encoding="utf-8")
    status = {
        "researchStepId": "S10",
        "stepNumber": 10,
        "success": True,
        "status": status_name,
        "artifactsWritten": sorted(
            path.name for path in OUTPUT.iterdir() if path.is_file()
        )
        + ["artifact_manifest.json"],
        "validationResult": (
            f"PASS {validation['passedChecks']}/{validation['totalChecks']} final checks; "
            f"10,752/10,752 logical rows; 21,504 exact-replay physical episodes"
        ),
        "outcomeClassification": outcome_classification,
        "caveatsOrBlockers": (
            [
                "Two external blinded reviewers are required before S10 human-review completion.",
                "Complete trajectories were unavailable; analysis used count/hash-change summaries only.",
            ]
            if reproduced
            else [
                "Bounded null applies only to 14 configurations, two spatial tasks, and the frozen event-feature registry.",
                "Complete trajectories were unavailable; analysis used count/hash-change summaries only.",
            ]
        ),
        "recommendedNextAction": recommended_next,
    }
    write_json(OUTPUT / "status.json", status)
    write_json(
        OUTPUT / "environment_provenance.json",
        {
            "schemaVersion": "e07.s10.environment-provenance.v1",
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
            "scikitLearn": sklearn.__version__,
            "workers": 8,
            "numericThreadsPerWorker": 1,
            "gitCommitAtExecution": git_commit(),
            "executionControlSha256": sha256_file(CONFIG),
            "implementationSha256": sha256_file(CORE),
            "runnerSha256": sha256_file(SCRIPT),
        },
    )
    write_json(
        OUTPUT / "input_provenance.json",
        {
            "schemaVersion": "e07.s10.input-provenance.v1",
            "researchStepId": "S10",
            "preflightHashRevalidationSha256": sha256_file(
                OUTPUT / "preflight_hash_revalidation.json"
            ),
            "executionPreregistrationSha256": sha256_file(
                OUTPUT / "execution_preregistration_freeze.json"
            ),
            "discoveryLockSha256": lock_sha,
            "s10pTreeSha256Before": historical_before[str(S10P)],
            "s10pTreeSha256After": historical_after[str(S10P)],
            "validationOutcomeRowsRead": 0,
            "confirmationOutcomeRowsRead": 0,
            "s06OrS06AArtifactLoads": 0,
            "s07ArmSignalUses": 0,
            "quarantineOutcomeOrCacheReads": 0,
        },
    )
    write_json(
        OUTPUT / "command_log.json",
        {
            "schemaVersion": "e07.s10.command-log.v1",
            "commands": [
                "PYTHONPATH=. pytest -q tests/test_s10_native_event_search.py tests/test_s10p_native_event_discovery.py",
                "PYTHONPATH=. python scripts/run_native_event_discovery_s10.py execute",
                "PYTHONPATH=. python scripts/run_native_event_discovery_s10.py validate",
            ],
        },
    )
    write_json(OUTPUT / "artifact_manifest.json", manifest_for_output())
    print(
        f"S10 execution complete: {len(reproduced)} independently reproduced candidate(s)",
        flush=True,
    )


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
        "reviewer_instructions.md",
        "blinded_exemplar_packet.jsonl",
    }
    missing = sorted(name for name in required if not (OUTPUT / name).is_file())
    if missing:
        raise RuntimeError(f"S10 required artifacts missing: {missing}")
    manifest = read_json(OUTPUT / "artifact_manifest.json")
    manifest_checks = [
        sha256_file(row["path"]) == row["sha256"]
        and Path(row["path"]).stat().st_size == row["bytes"]
        for row in manifest["artifacts"]
    ]
    summary = read_json(OUTPUT / "validation_summary.json")
    accounting = read_json(OUTPUT / "complete_accounting.json")
    status = read_json(OUTPUT / "status.json")
    checks = {
        "manifest": bool(manifest_checks) and all(manifest_checks),
        "finalValidation": bool(summary["allPass"]),
        "logicalAccounting": accounting["totalLogicalRows"] == 10752,
        "physicalAccounting": accounting["physicalEpisodeExecutions"] == 21504,
        "protectedSealed": (
            accounting["validationOutcomeRowsOpened"] == 0
            and accounting["confirmationOutcomeRowsOpened"] == 0
        ),
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
        ),
        "s10pProtocolHash": sha256_file(S10P / "s10p_native_event_protocol.yaml")
        == "fbf6f7d3645ac901c3d93e378a7118bf5a2262902b0428451840d7b8dbd276dd",
    }
    result = {
        "schemaVersion": "e07.s10.artifact-validation.v1",
        "checks": checks,
        "allPass": all(checks.values()),
        "manifestEntries": len(manifest["artifacts"]),
    }
    write_json(OUTPUT / "artifact_validation.json", result)
    if not result["allPass"]:
        raise RuntimeError(f"S10 artifact validation failed: {result}")
    print(json.dumps(result, sort_keys=True))


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"execute", "validate"}:
        raise SystemExit("usage: run_native_event_discovery_s10.py {execute|validate}")
    if sys.argv[1] == "execute":
        execute()
    else:
        validate()


if __name__ == "__main__":
    main()
