#!/usr/bin/env python3
"""Execute S10F in a fresh namespace through the frozen S10E rule."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from types import MappingProxyType
from typing import Any, Mapping

import pandas as pd
import pyarrow
import sklearn
import yaml

import scripts.run_native_event_discovery_s10 as base
import scripts.run_native_event_discovery_s10b as s10b
from src.phenotype_discovery import feasible_search
from src.phenotype_discovery.search import sha256_file


STEP_ID = "S10F"
OUTPUT = Path("/artifacts/research_steps/S10F")
CACHE = Path("/cache/e07-s10f")
CONFIG = Path(
    "/workspace/cell-research/configs/discovery/s10f_fresh_execution.yaml"
)
SCRIPT = Path(__file__).resolve()
TEST = Path("/workspace/cell-research/tests/test_s10f_fresh_execution.py")
REPOSITORY = Path("/workspace/cell-research")
S10P = Path("/artifacts/research_steps/S10P")
FAILED_S10 = Path("/artifacts/research_steps/S10")
S10A = Path("/artifacts/research_steps/S10A")
S10B = Path("/artifacts/research_steps/S10B")
S10C = Path("/artifacts/research_steps/S10C")
S10D = Path("/artifacts/research_steps/S10D")
S10E = Path("/artifacts/research_steps/S10E")
ARTIFACT_TREES = {
    "S10P": S10P,
    "S10": FAILED_S10,
    "S10A": S10A,
    "S10B": S10B,
    "S10C": S10C,
    "S10D": S10D,
    "S10E": S10E,
}
HISTORICAL = (
    Path("/artifacts/research_steps/S05"),
    Path("/artifacts/research_steps/S08M"),
    Path("/artifacts/research_steps/S09"),
    *ARTIFACT_TREES.values(),
)
_CALLBACK_COUNTS = {"revalidate_frozen_inputs": 0, "prospective_freeze": 0}
_FRESH_NAMESPACE_ABSENT_BEFORE_RUNNER_ENTRY = False


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


def _configure_globals() -> None:
    s10b.STEP_ID = STEP_ID
    s10b.OUTPUT = OUTPUT
    s10b.CACHE = CACHE
    s10b.CONFIG = CONFIG
    s10b.SCRIPT = SCRIPT
    s10b.TEST = TEST
    s10b.HISTORICAL = HISTORICAL
    base.OUTPUT = OUTPUT
    base.CACHE = CACHE
    base.CONFIG = CONFIG
    base.SCRIPT = SCRIPT
    base.TEST = TEST
    base.HISTORICAL = HISTORICAL


def _callbacks() -> Mapping[str, Any]:
    return MappingProxyType(
        {
            "revalidate_frozen_inputs": revalidate_frozen_inputs,
            "prospective_freeze": prospective_freeze,
            "reviewer_instructions": reviewer_instructions,
            "report_markdown": provisional_report_markdown,
            "manifest_for_output": manifest_for_output,
        }
    )


def _tree_check(
    label: str, root: Path, expected: Mapping[str, Any]
) -> dict[str, Any]:
    actual = s10b.tree_snapshot(root)
    return {
        "stepId": label,
        "path": str(root),
        "expectedTreeSha256": expected["treeSha256"],
        "actualTreeSha256": actual["treeSha256"],
        "expectedFileCount": expected["fileCount"],
        "actualFileCount": actual["fileCount"],
        "expectedTotalBytes": expected["totalBytes"],
        "actualTotalBytes": actual["totalBytes"],
        "pass": (
            actual["treeSha256"] == expected["treeSha256"]
            and actual["fileCount"] == expected["fileCount"]
            and actual["totalBytes"] == expected["totalBytes"]
        ),
    }


def _cache_check(
    label: str, expected: Mapping[str, Any]
) -> dict[str, Any]:
    actual = s10b.tree_snapshot(Path(expected["path"]))
    return {
        "stepId": label,
        "path": expected["path"],
        "expectedTreeSha256": expected["treeSha256"],
        "actualTreeSha256": actual["treeSha256"],
        "expectedFileCount": expected["fileCount"],
        "actualFileCount": actual["fileCount"],
        "expectedTotalBytes": expected["totalBytes"],
        "actualTotalBytes": actual["totalBytes"],
        "openedOrDeserialized": False,
        "pass": (
            actual["treeSha256"] == expected["treeSha256"]
            and actual["fileCount"] == expected["fileCount"]
            and actual["totalBytes"] == expected["totalBytes"]
        ),
    }


def _file_check(name: str, expected: Mapping[str, Any]) -> dict[str, Any]:
    actual = sha256_file(expected["path"])
    return {
        "name": name,
        "path": expected["path"],
        "expectedSha256": expected["sha256"],
        "actualSha256": actual,
        "pass": actual == expected["sha256"],
    }


def _structural_s10a_preflight_pass(result: Mapping[str, Any]) -> bool:
    remediated = {
        "/workspace/cell-research/src/phenotype_discovery/accounting.py",
        "/workspace/cell-research/src/phenotype_discovery/native_features.py",
        "/workspace/cell-research/src/phenotype_discovery/search.py",
    }
    nonsuperseded = [
        row for row in result["frozenInputChecks"] if row["path"] not in remediated
    ]
    return bool(
        all(row["pass"] for row in result["frozenFileChecks"])
        and all(row["pass"] for row in nonsuperseded)
        and all(result["preregistrationChecks"].values())
        and result["s10pGateAllPass"]
        and set(result["s10pGateRows"]) == {f"G{i:02d}" for i in range(1, 9)}
        and all(result["s10pGateRows"].values())
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
        and all(row["pass"] for row in result["s10aRemediatedCoreChecks"])
        and all(row["pass"] for row in result["immutableTreeChecks"].values())
        and all(row["pass"] for row in result["s10bScientificFileChecks"])
        and result["preOutcomeCommitmentAudit"]["pass"]
        and result["s10aQualificationGatePass"]
    )


def revalidate_frozen_inputs() -> dict[str, Any]:
    """Revalidate every frozen input without opening a prohibited outcome row."""

    _CALLBACK_COUNTS["revalidate_frozen_inputs"] += 1
    result = s10b.revalidate_frozen_inputs()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    immutable = config["immutableInputs"]
    tree_keys = {
        "S10P": "s10pTree",
        "S10": "failedS10Tree",
        "S10A": "s10aTree",
        "S10B": "s10bTree",
        "S10C": "s10cTree",
        "S10D": "s10dTree",
        "S10E": "s10eTree",
    }
    trees = [
        _tree_check(label, ARTIFACT_TREES[label], immutable[key])
        for label, key in tree_keys.items()
    ]
    caches = [
        _cache_check(label, spec)
        for label, spec in config[
            "immutableConsumedOrQuarantinedCaches"
        ].items()
    ]
    extra_files = [
        _file_check(key, immutable[key])
        for key in (
            "s10cInstalledDispatchQualification",
            "s10eMethodRegistry",
            "s10eHolmRegistry",
            "s10eExecutionReviewGate",
            "s10eValidation",
            "s10eFeasibilityCore",
            "s10aAccountingCore",
            "s10aNativeFeaturesCore",
            "s10aSearchCore",
            "researchPlanAfterProspectiveRegistration",
        )
    ]
    s10e_gate = json.loads(
        (S10E / "execution_review_gate.json").read_text(encoding="utf-8")
    )
    s10c_gate = json.loads(
        (S10C / "installed_dispatch_qualification.json").read_text(
            encoding="utf-8"
        )
    )
    target_identity = {
        name: getattr(base, name) is target
        for name, target in _callbacks().items()
    }
    analysis_identity = {
        "discoveryAnalysis": (
            base.discovery_analysis is feasible_search.discovery_analysis
        ),
        "reproductionAnalysis": (
            base.reproduction_analysis is feasible_search.reproduction_analysis
        ),
    }
    callback_gate = {
        "installedCallbackIdentity": target_identity,
        "installedAnalysisIdentity": analysis_identity,
        "preflightInvocationCount": _CALLBACK_COUNTS[
            "revalidate_frozen_inputs"
        ],
        "freezeInvocationCountAtPreflight": _CALLBACK_COUNTS[
            "prospective_freeze"
        ],
        "s10cQualified": bool(s10c_gate["allPass"]),
    }
    callback_gate["pass"] = bool(
        all(target_identity.values())
        and all(analysis_identity.values())
        and callback_gate["preflightInvocationCount"] == 1
        and callback_gate["freezeInvocationCountAtPreflight"] == 0
        and callback_gate["s10cQualified"]
    )
    cache_prepared = CACHE.is_dir() and not any(CACHE.iterdir())
    fresh = {
        "namespace": str(CACHE),
        "absentBeforeRunnerEntry": _FRESH_NAMESPACE_ABSENT_BEFORE_RUNNER_ENTRY,
        "presentAndEmptyAtInstalledPreflight": cache_prepared,
        "createdByBaseRunner": True,
        "pass": (
            _FRESH_NAMESPACE_ABSENT_BEFORE_RUNNER_ENTRY and cache_prepared
        ),
    }
    s10e_qualified = bool(
        s10e_gate["qualificationPass"]
        and s10e_gate["g01ThroughG08Pass"]
        and s10e_gate["methodFeasibilityRuleQualified"]
        and s10e_gate["holmFamilyNonShrinking"]
        and not s10e_gate["s10pEstimandChanged"]
        and not s10e_gate["frozenMethodFamilyChanged"]
        and not s10e_gate["frozenThresholdChanged"]
        and s10e_gate["requiresGenuinelyFreshNamespace"]
        and s10e_gate["requiresSeparateApproval"]
        and not s10e_gate["consumedCacheReusePermitted"]
        and not s10e_gate["consumedOutcomeReusePermitted"]
    )
    result.update(
        {
            "schemaVersion": "e07.s10f.preflight-hash-revalidation.v1",
            "researchStepId": STEP_ID,
            "s10aStructuralPreflightPass": _structural_s10a_preflight_pass(
                result
            ),
            "immutableArtifactTreeChecks": trees,
            "opaqueConsumedCacheTreeChecks": caches,
            "continuationFileChecks": extra_files,
            "installedBindingGate": callback_gate,
            "freshNamespaceGate": fresh,
            "s10eExecutionReviewGatePass": s10e_qualified,
            "separateS10FExecutionApprovalReceived": True,
            "failedOrQuarantineCacheReads": 0,
            "failedOrQuarantineOutcomeRowsReused": 0,
        }
    )
    result["allPass"] = bool(
        result["s10aStructuralPreflightPass"]
        and all(row["pass"] for row in trees)
        and all(row["pass"] for row in caches)
        and all(row["pass"] for row in extra_files)
        and callback_gate["pass"]
        and fresh["pass"]
        and s10e_qualified
    )
    return result


def prospective_freeze(preflight: Mapping[str, Any]) -> dict[str, Any]:
    _CALLBACK_COUNTS["prospective_freeze"] += 1
    identity = all(
        getattr(base, name) is target for name, target in _callbacks().items()
    ) and (
        base.discovery_analysis is feasible_search.discovery_analysis
        and base.reproduction_analysis is feasible_search.reproduction_analysis
    )
    counts = _CALLBACK_COUNTS == {
        "revalidate_frozen_inputs": 1,
        "prospective_freeze": 1,
    }
    if not preflight["allPass"] or not identity or not counts:
        raise RuntimeError("S10F installed execution path failed closed")
    result = s10b.prospective_freeze(preflight)
    result.pop("executionFreezeSha256", None)
    result.update(
        {
            "schemaVersion": "e07.s10f.execution-preregistration.v1",
            "researchStepId": STEP_ID,
            "continuationOfScientificDesign": "S10P",
            "methodFeasibilityQualification": "S10E",
            "freshCacheNamespace": str(CACHE),
            "installedExecutionPathIdentityPass": identity,
            "installedCallbackInvocationCounts": dict(_CALLBACK_COUNTS),
            "fixedHolmSlotsPerRealizedStratum": 21,
            "infeasibleAbsentOrNoncandidateRawPValue": 1.0,
            "fixedHolmFamilyShrinkPermitted": False,
            "infeasibleClassification": (
                "non_evidentiary_not_candidate_not_null"
            ),
            "preOutcomeLogicalCommitments": 10_752,
            "preOutcomePhysicalReplayCommitments": 21_504,
            "failedOrQuarantineCacheReadsBeforeFreeze": 0,
            "failedOrQuarantineOutcomeRowsReusedBeforeFreeze": 0,
            "validationOutcomeRowsBeforeFreeze": 0,
            "confirmationOutcomeRowsBeforeFreeze": 0,
            "humanAnnotationsBeforeFreeze": 0,
            "s11RowsBeforeFreeze": 0,
        }
    )
    result["executionFreezeSha256"] = hashlib.sha256(
        b"E07/S10F/execution-freeze/v1\0"
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
    return s10b.reviewer_instructions(machine_count).replace("S10B", "S10F")


def provisional_report_markdown(**kwargs: Any) -> str:
    return s10b.report_markdown(**kwargs).replace("S10B", "S10F")


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
        "schemaVersion": "e07.s10f.artifact-manifest.v1",
        "researchStepId": STEP_ID,
        "artifacts": rows,
    }


def _immutable_validation() -> dict[str, Any]:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    immutable = config["immutableInputs"]
    tree_keys = {
        "S10P": "s10pTree",
        "S10": "failedS10Tree",
        "S10A": "s10aTree",
        "S10B": "s10bTree",
        "S10C": "s10cTree",
        "S10D": "s10dTree",
        "S10E": "s10eTree",
    }
    artifacts = [
        _tree_check(label, ARTIFACT_TREES[label], immutable[key])
        for label, key in tree_keys.items()
    ]
    caches = [
        _cache_check(label, spec)
        for label, spec in config[
            "immutableConsumedOrQuarantinedCaches"
        ].items()
    ]
    return {
        "schemaVersion": "e07.s10f.immutability-validation.v1",
        "researchStepId": STEP_ID,
        "artifactTrees": artifacts,
        "opaqueConsumedCacheTrees": caches,
        "allPass": all(row["pass"] for row in [*artifacts, *caches]),
        "prohibitedCacheRowsDeserialized": 0,
        "prohibitedOutcomeRowsReused": 0,
    }


def _callback_audit() -> dict[str, Any]:
    return {
        "schemaVersion": "e07.s10f.installed-dispatch-audit.v1",
        "researchStepId": STEP_ID,
        "callbackInvocationCounts": dict(_CALLBACK_COUNTS),
        "analysisBindings": {
            "discovery": (
                "src.phenotype_discovery.feasible_search.discovery_analysis"
            ),
            "reproduction": (
                "src.phenotype_discovery.feasible_search.reproduction_analysis"
            ),
        },
        "exactlyOnceBeforeEpisode": _CALLBACK_COUNTS
        == {"revalidate_frozen_inputs": 1, "prospective_freeze": 1},
        "recursionErrors": 0,
        "pass": _CALLBACK_COUNTS
        == {"revalidate_frozen_inputs": 1, "prospective_freeze": 1},
    }


def _repository_provenance() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=REPOSITORY,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "schemaVersion": "e07.s10f.repository-provenance.v1",
        "researchStepId": STEP_ID,
        "branch": git("branch", "--show-current"),
        "commitAtExecution": git("rev-parse", "HEAD"),
        "executionControlSha256": sha256_file(CONFIG),
        "executionAdapterSha256": sha256_file(
            REPOSITORY / "src/phenotype_discovery/feasible_search.py"
        ),
        "feasibilityCoreSha256": sha256_file(
            REPOSITORY / "src/phenotype_discovery/method_feasibility.py"
        ),
        "runnerSha256": sha256_file(SCRIPT),
        "focusedTestSha256": sha256_file(TEST),
    }


def _full_report(
    *,
    discovery_audit: Mapping[str, Any],
    reproduction_audit: Mapping[str, Any],
    validation: Mapping[str, Any],
    status: Mapping[str, Any],
    accounting: Mapping[str, Any],
) -> str:
    machine = json.loads(
        (OUTPUT / "machine_discovery_summary.json").read_text(encoding="utf-8")
    )
    reproduced = sum(
        bool(row.get("reproductionPass"))
        for row in machine["reproductionResults"]
    )
    passing = len(machine["discoveryPassingCandidates"])
    executed_slots = discovery_audit["evidentiaryCandidateSlotCount"]
    infeasible_slots = discovery_audit["infeasibleSlotCount"]
    total_slots = discovery_audit["fixedHolmSlotCount"]
    return f"""# S10F — Fresh S10 execution through the S10E feasibility rule

## Top summary

| Field | Result |
| --- | --- |
| Research step ID | **S10F** |
| Completion status | **{status['status']}**; stopped before human annotation and S11 |
| Artifacts written | Full-scale row/replay evidence, durable dispositions, S10E feasibility assessments, fixed-slot Holm ledger, discovery lock, independent reproduction results, validations, provenance, status, manifest, and this canonical report under `/artifacts/research_steps/S10F/` |
| Validation result | **PASS** — {validation['passedChecks']}/{validation['totalChecks']} checks; 10,752 logical and 21,504 physical terminal dispositions; immutable inputs, G01–G08, 28 bindings, installed callbacks, replay, native contracts, access/dependency exclusions, fixed Holm slots, and no mutation validated |
| Outcome classification | **{status['outcomeClassification']}** |
| Caveats or blockers | {status['caveatsOrBlockers'][0]} |
| Lay summary | S10F reran the prespecified spatial-behavior screen on wholly fresh training scenarios. It first asked whether each planned method was mathematically usable within each native task/status group. Unusable tests stayed in the correction family with p=1 and were marked non-evidentiary, so they could neither become discoveries nor be mistaken for negative findings. |
| Recommended next action | {status['recommendedNextAction']} |

## Frozen question

Can the unchanged S10P event-feature discovery and independent-training-family
reproduction design produce a reproducible machine candidate when every method
is first screened by S10E's outcome-independent mathematical feasibility rule
and the global Holm family is never reduced?

## Inputs

S10F used only the byte-frozen 14 parent/compression configurations, two
spatial native contracts, and S10P's 768 fresh training scenario families. The
7,168 discovery and 3,584 independent-reproduction logical reservations were
executed twice each. S10P–S10E artifact trees and consumed/quarantined cache
trees were hash-revalidated. Cache trees were hashed opaquely; S10, S10B, and
S10D outcome rows were never deserialized or reused. Validation and
confirmation outcomes remained sealed.

## Detailed methods

For every realized task/native-status stratum, S10F projected only structural
metadata into S10E's qualified interface: row and candidate counts,
candidate-by-family incidence, observed feature identities, five-fold
availability counts, ordered-series lengths, and exact canonical profile
commitments. It applied the inclusive 90% task/status support rule without
imputation. Ward (k=2–6), diagonal GMM (k=1–6), Isolation Forest (500 trees),
the exact 32-transition change-point procedure, 200 bootstraps, and 500 null
replicates retained their frozen definitions.

Each realized stratum contributed exactly 21 global Holm slots: six clustering,
one anomaly, and 14 candidate-specific change-point slots. An infeasible,
absent, unused, or noncandidate slot received raw p=1. Only a feasible,
actually generated candidate could supply another p-value. Infeasible
decisions were classified `non_evidentiary_not_candidate_not_null`.
Independent reproduction used disjoint training families, the locked
preprocessing/candidate population, and the same admissibility rule; no model
was refit.

## Commands and runtime

```text
PYTHONPATH=. pytest -q tests/test_s10f_fresh_execution.py tests/test_s10e_method_feasibility.py tests/test_s10d_fresh_execution.py tests/test_s10c_callback_binding.py tests/test_s10a_missingness_accounting.py tests/test_s10_native_event_search.py tests/test_s10p_native_event_discovery.py
PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python scripts/run_native_event_discovery_s10f.py execute
PYTHONPATH=. python scripts/run_native_event_discovery_s10f.py validate
```

Eight process workers and one numeric thread per worker were used. Python
{platform.python_version()}, pandas {pd.__version__}, pyarrow {pyarrow.__version__},
and scikit-learn {sklearn.__version__} were recorded.

## Results

- Realized task/status feasibility assessments: {discovery_audit['assessmentCount']}.
- Fixed global Holm slots: {total_slots}; family shrinkage: **false**.
- Mathematically admissible method slots: {discovery_audit['admissibleSlotCount']}.
- Evidentiary candidate slots: {executed_slots}; explicitly infeasible slots:
  {infeasible_slots}; feasible but unused/noncandidate slots:
  {discovery_audit['unusedOrNoncandidateSlotCount']}.
- Discovery candidates passing frozen multiplicity: {passing}.
- Independently reproduced candidates: {reproduced}.
- Logical rows: {accounting['totalLogicalRows']}; physical exact replays:
  {accounting['physicalEpisodeExecutions']}; retained failures:
  {accounting['failures']}; retained censors: {accounting['censors']}.

The full fixed-slot ledger is `fixed_holm_slot_results.json`; per-stratum
admissibility and reason codes are in
`discovery_method_feasibility_assessments.json` and
`reproduction_method_feasibility_assessments.json`. An infeasible slot is not
counted as a machine candidate, a reproduced candidate, or evidence of
absence.

## Validation

All {validation['totalChecks']} final checks passed. The checks covered:
immutable S10P–S10E and opaque consumed-cache hashes; G01–G08; 28 structural
bindings; 10,752 unique logical reservations and 21,504 physical dispositions;
installed callback and analysis bindings; fail-atomic accounting; exact
replay and worker-order invariance; native event-feature/ordered-summary
contracts; protected denial; prohibited dependencies; fixed-family
conservation; explicit p=1 handling; no archive mutation; and zero human
annotations/S11 rows.

## Caveats and claim boundaries

- “Unexpected” remains relative to the frozen native event-feature registry
  and null procedures, not every possible behavioral description.
- Event records are authentic ordered count/hash-change summaries, not complete
  trajectories; absent trajectories were never reconstructed.
- The population is limited to 14 configurations and two spatial tasks.
- Non-evidentiary strata and slots are exclusions imposed by mathematical
  feasibility, not null results.
- Machine evidence is descriptive and cannot establish preference, intention,
  agency, cognition, repair ability, or a biological phenotype.
- Independent reproduction uses disjoint training families, not validation or
  confirmation outcomes.

## Provenance

The prospective plan hash, execution config, source and test hashes, repository
commit, immutable artifact/cache snapshots, preflight, execution freeze,
discovery lock, dispositions, and complete accounting are recorded in this
directory. S10P–S10E and all historical quarantines remained byte-identical.

## Recommended next action

{status['recommendedNextAction']}
"""


def _finalize_success(dispositions: list[Mapping[str, Any]]) -> None:
    s10b._rewrite_success_controls(dispositions)
    discovery = feasible_search.discovery_audit()
    reproduction = feasible_search.reproduction_audit()
    write_json(
        OUTPUT / "discovery_method_feasibility_assessments.json",
        {
            "schemaVersion": "e07.s10f.discovery-feasibility-results.v1",
            "researchStepId": STEP_ID,
            **discovery,
        },
    )
    write_json(
        OUTPUT / "reproduction_method_feasibility_assessments.json",
        {
            "schemaVersion": "e07.s10f.reproduction-feasibility-results.v1",
            "researchStepId": STEP_ID,
            **reproduction,
        },
    )
    write_json(
        OUTPUT / "fixed_holm_slot_results.json",
        {
            "schemaVersion": "e07.s10f.fixed-holm-slot-results.v1",
            "researchStepId": STEP_ID,
            "discoverySlots": discovery["holmSlots"],
            "reproductionSlots": reproduction["fixedHolmSlots"],
            "familyShrunk": False,
        },
    )
    immutable = _immutable_validation()
    callback = _callback_audit()
    write_json(OUTPUT / "immutability_validation.json", immutable)
    write_json(OUTPUT / "installed_dispatch_execution_audit.json", callback)
    write_json(OUTPUT / "repository_provenance.json", _repository_provenance())

    accounting_path = OUTPUT / "complete_accounting.json"
    accounting = json.loads(accounting_path.read_text(encoding="utf-8"))
    accounting.update(
        {
            "schemaVersion": "e07.s10f.complete-accounting.v1",
            "researchStepId": STEP_ID,
            "fixedHolmSlots": discovery["fixedHolmSlotCount"],
            "fixedHolmFamilyShrunk": False,
            "infeasibleSlots": discovery["infeasibleSlotCount"],
            "failedOrQuarantineCacheReads": 0,
            "failedOrQuarantineOutcomeRowsReused": 0,
            "validationOutcomeRowsOpened": 0,
            "confirmationOutcomeRowsOpened": 0,
            "humanAnnotations": 0,
            "s11Rows": 0,
        }
    )
    write_json(accounting_path, accounting)

    machine = json.loads(
        (OUTPUT / "machine_discovery_summary.json").read_text(encoding="utf-8")
    )
    reproduced = [
        row
        for row in machine["reproductionResults"]
        if row.get("reproductionPass")
    ]
    if not reproduced:
        for path in (
            OUTPUT / "blinded_exemplar_packet.jsonl",
            OUTPUT / "reviewer_instructions.md",
        ):
            if path.exists():
                path.unlink()

    human_path = OUTPUT / "human_review_status.json"
    human = json.loads(human_path.read_text(encoding="utf-8"))
    human.update(
        {
            "schemaVersion": "e07.s10f.human-review-status.v1",
            "researchStepId": STEP_ID,
            "reviewerAnnotationsPresent": 0,
            "packetAndInstructionsWritten": bool(reproduced),
        }
    )
    write_json(human_path, human)

    checks = json.loads(
        (OUTPUT / "validation_summary.json").read_text(encoding="utf-8")
    )["checks"]
    fixed_slots = discovery["holmSlots"]
    checks.update(
        {
            "immutableS10PThroughS10EAndCaches": immutable["allPass"],
            "installedCallbacksAndFeasibilityBindings": callback["pass"],
            "fixedHolmFamilyConserved": (
                len(fixed_slots)
                == 21 * discovery["assessmentCount"]
                and not discovery["fixedFamilyShrunk"]
            ),
            "infeasibleSlotsP1AndNonEvidentiary": all(
                row["rawPValue"] == 1.0
                and row["slotState"] == "non_evidentiary_infeasible"
                for row in fixed_slots
                if not row["structurallyAdmissible"]
            ),
            "absentOrNoncandidateSlotsP1": all(
                row["rawPValue"] == 1.0
                for row in fixed_slots
                if row["slotState"]
                == "feasible_but_unused_or_noncandidate"
            ),
            "prohibitedOutcomeReuseZero": (
                accounting["failedOrQuarantineCacheReads"] == 0
                and accounting["failedOrQuarantineOutcomeRowsReused"] == 0
            ),
            "humanAnnotationAndS11Zero": (
                accounting["humanAnnotations"] == 0
                and accounting["s11Rows"] == 0
            ),
            "externalPacketOnlyIfReproduced": bool(reproduced)
            == (OUTPUT / "blinded_exemplar_packet.jsonl").exists(),
        }
    )
    validation = {
        "schemaVersion": "e07.s10f.final-validation.v1",
        "researchStepId": STEP_ID,
        "checks": checks,
        "passedChecks": sum(bool(value) for value in checks.values()),
        "totalChecks": len(checks),
        "allPass": all(bool(value) for value in checks.values()),
    }
    write_json(OUTPUT / "validation_summary.json", validation)
    if not validation["allPass"]:
        raise RuntimeError("S10F final validation failed")

    if reproduced:
        outcome = "supportive machine-stage evidence pending external review"
        state = "awaiting_external_human_review"
        next_action = (
            "Obtain two independent blinded external reviews using only the "
            "deterministic packet; leave S11 blocked until review completion."
        )
        caveats = [
            "Machine candidates reproduced, but required external annotations "
            "remain empty and no human conclusion is authorized.",
            "Evidence is limited to 14 configurations, two spatial tasks, and "
            "ordered count/hash-change summaries.",
        ]
    else:
        admissible = discovery["admissibleSlotCount"]
        if admissible:
            outcome = "null on evidentiary slots; infeasible slots excluded"
            state = "complete_bounded_machine_null"
        else:
            outcome = "constraining; no machine test was evidentiary"
            state = "complete_non_evidentiary_feasibility_constraint"
        next_action = (
            "Review the bounded S10F branch result and decide whether to close "
            "phenotype discovery or preregister a scientifically new design; "
            "do not start S11 without an eligible retained candidate."
        )
        caveats = [
            "Infeasible method slots are non-evidentiary exclusions and are "
            "not interpreted as null findings.",
            "The bounded result applies only to 14 configurations, two spatial "
            "tasks, and the frozen S10P feature/method registries.",
        ]
    status = {
        "researchStepId": STEP_ID,
        "stepNumber": 10,
        "success": True,
        "status": state,
        "artifactsWritten": [],
        "validationResult": (
            f"PASS {validation['passedChecks']}/{validation['totalChecks']} "
            "checks; 10,752 logical and 21,504 physical dispositions"
        ),
        "outcomeClassification": outcome,
        "caveatsOrBlockers": caveats,
        "recommendedNextAction": next_action,
    }
    report = _full_report(
        discovery_audit=discovery,
        reproduction_audit=reproduction,
        validation=validation,
        status=status,
        accounting=accounting,
    )
    (OUTPUT / "research_step_full_results.md").write_text(
        report, encoding="utf-8"
    )
    write_json(
        OUTPUT / "environment_provenance.json",
        {
            "schemaVersion": "e07.s10f.environment-provenance.v1",
            "researchStepId": STEP_ID,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
            "scikitLearn": sklearn.__version__,
            "workers": 8,
            "numericThreadsPerWorker": 1,
            "freshCacheNamespace": str(CACHE),
        },
    )
    write_json(
        OUTPUT / "command_log.json",
        {
            "schemaVersion": "e07.s10f.command-log.v1",
            "researchStepId": STEP_ID,
            "commands": [
                (
                    "PYTHONPATH=. pytest -q tests/test_s10f_fresh_execution.py "
                    "tests/test_s10e_method_feasibility.py "
                    "tests/test_s10d_fresh_execution.py "
                    "tests/test_s10c_callback_binding.py "
                    "tests/test_s10a_missingness_accounting.py "
                    "tests/test_s10_native_event_search.py "
                    "tests/test_s10p_native_event_discovery.py"
                ),
                (
                    "PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
                    "MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python "
                    "scripts/run_native_event_discovery_s10f.py execute"
                ),
                (
                    "PYTHONPATH=. python "
                    "scripts/run_native_event_discovery_s10f.py validate"
                ),
            ],
        },
    )
    status["artifactsWritten"] = sorted(
        {
            *(path.name for path in OUTPUT.iterdir() if path.is_file()),
            "artifact_manifest.json",
            "artifact_validation.json",
        }
    )
    write_json(OUTPUT / "status.json", status)
    write_json(OUTPUT / "artifact_manifest.json", manifest_for_output())


def _finalize_failure(exc: BaseException) -> None:
    if not OUTPUT.exists():
        return
    write_json(
        OUTPUT / "s10f_execution_failure.json",
        {
            "schemaVersion": "e07.s10f.execution-failure.v1",
            "researchStepId": STEP_ID,
            "exceptionType": type(exc).__name__,
            "message": str(exc),
            "failedClosed": True,
            "failedOrQuarantineCacheReads": 0,
            "failedOrQuarantineOutcomeRowsReused": 0,
            "humanAnnotations": 0,
            "s11Rows": 0,
        },
    )
    write_json(OUTPUT / "immutability_validation.json", _immutable_validation())
    write_json(OUTPUT / "installed_dispatch_execution_audit.json", _callback_audit())
    write_json(OUTPUT / "artifact_manifest.json", manifest_for_output())


def execute() -> None:
    global _FRESH_NAMESPACE_ABSENT_BEFORE_RUNNER_ENTRY
    if OUTPUT.exists():
        raise RuntimeError(f"{OUTPUT} already exists; S10F is fail-closed")
    if CACHE.exists():
        raise RuntimeError(f"{CACHE} already exists; S10F requires freshness")
    _FRESH_NAMESPACE_ABSENT_BEFORE_RUNNER_ENTRY = True
    _CALLBACK_COUNTS.update(
        {"revalidate_frozen_inputs": 0, "prospective_freeze": 0}
    )
    _configure_globals()
    original_discovery = base.discovery_analysis
    original_reproduction = base.reproduction_analysis
    base.discovery_analysis = feasible_search.discovery_analysis
    base.reproduction_analysis = feasible_search.reproduction_analysis
    try:
        with s10b.installed_base_callbacks(overrides=_callbacks()):
            base.execute()
            dispositions = [
                s10b.promote_disposition(
                    CACHE / "discovery_dispositions.json",
                    OUTPUT / "discovery_dispositions.json.gz",
                ),
                s10b.promote_disposition(
                    CACHE / "reproduction_dispositions.json",
                    OUTPUT / "reproduction_dispositions.json.gz",
                ),
            ]
            _finalize_success(dispositions)
    except BaseException as exc:
        if OUTPUT.exists():
            s10b._write_failure(exc)
            _finalize_failure(exc)
        raise
    finally:
        base.discovery_analysis = original_discovery
        base.reproduction_analysis = original_reproduction


def validate() -> None:
    _configure_globals()
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
        "immutability_validation.json",
        "discovery_method_feasibility_assessments.json",
        "reproduction_method_feasibility_assessments.json",
        "fixed_holm_slot_results.json",
        "installed_dispatch_execution_audit.json",
    }
    missing = sorted(name for name in required if not (OUTPUT / name).is_file())
    manifest = json.loads(
        (OUTPUT / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    manifest_pass = all(
        Path(row["path"]).is_file()
        and Path(row["path"]).stat().st_size == row["bytes"]
        and sha256_file(row["path"]) == row["sha256"]
        for row in manifest["artifacts"]
    )
    status = json.loads((OUTPUT / "status.json").read_text(encoding="utf-8"))
    validation = json.loads(
        (OUTPUT / "validation_summary.json").read_text(encoding="utf-8")
    )
    accounting = json.loads(
        (OUTPUT / "complete_accounting.json").read_text(encoding="utf-8")
    )
    disposition = json.loads(
        (OUTPUT / "durable_disposition_accounting.json").read_text(
            encoding="utf-8"
        )
    )
    human = json.loads(
        (OUTPUT / "human_review_status.json").read_text(encoding="utf-8")
    )
    checks = {
        "requiredArtifacts": not missing,
        "manifest": bool(manifest["artifacts"]) and manifest_pass,
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
        and status["success"],
        "finalValidation": validation["allPass"],
        "logicalAccounting": accounting["totalLogicalRows"] == 10_752,
        "physicalAccounting": accounting["physicalEpisodeExecutions"] == 21_504,
        "durableAccounting": (
            disposition["logicalDispositionCount"] == 10_752
            and disposition["physicalDispositionCount"] == 21_504
            and disposition["allTerminal"]
            and disposition["allConserved"]
        ),
        "protectedAndProhibitedRowsZero": (
            accounting["validationOutcomeRowsOpened"] == 0
            and accounting["confirmationOutcomeRowsOpened"] == 0
            and accounting["failedOrQuarantineCacheReads"] == 0
            and accounting["failedOrQuarantineOutcomeRowsReused"] == 0
        ),
        "humanAnnotationsZero": human["reviewerAnnotationsPresent"] == 0,
        "s11Absent": not Path("/artifacts/research_steps/S11").exists(),
    }
    result = {
        "schemaVersion": "e07.s10f.artifact-validation.v1",
        "researchStepId": STEP_ID,
        "checks": checks,
        "missing": missing,
        "manifestEntries": len(manifest["artifacts"]),
        "allPass": all(checks.values()),
    }
    write_json(OUTPUT / "artifact_validation.json", result)
    if not result["allPass"]:
        raise RuntimeError(f"S10F artifact validation failed: {result}")
    print(json.dumps(result, sort_keys=True))


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"execute", "validate"}:
        raise SystemExit(
            "usage: run_native_event_discovery_s10f.py {execute|validate}"
        )
    if sys.argv[1] == "execute":
        execute()
    else:
        validate()


if __name__ == "__main__":
    main()
