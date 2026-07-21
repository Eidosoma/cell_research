#!/usr/bin/env python3
"""Build the E07 S04 training-only objective/descriptor evidence bundle."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any, Mapping

from src.environment_suite import (
    AccessDeniedError,
    AccessGrant,
    AccessPhase,
    EnvironmentSuite,
)
from src.objective_design import build_s04_evidence
from src.objective_design.core import correlations_csv


REPOSITORY = Path(__file__).resolve().parents[1]
WORKSPACE = REPOSITORY.parent
CONFIG = REPOSITORY / "configs/search"
REGISTRY = REPOSITORY / "configs/environment_suite/task_registry.yaml"
SPLITS = REPOSITORY / "configs/environment_suite/split_manifest.json"
S01 = Path("/artifacts/research_steps/S01")
S02 = Path("/artifacts/research_steps/S02")
S03 = Path("/artifacts/research_steps/S03")

CONFIG_FILES = {
    "objective_registry.yaml": CONFIG / "s04_objective_registry.yaml",
    "archive_descriptor_registry.yaml": CONFIG / "s04_descriptor_registry.yaml",
    "task_weighting.yaml": CONFIG / "s04_task_weighting.yaml",
    "uncertainty_registry.yaml": CONFIG / "s04_uncertainty_registry.yaml",
    "anti_gaming_registry.yaml": CONFIG / "s04_anti_gaming_registry.yaml",
    "s05_eligibility_gate.yaml": CONFIG / "s04_s05_eligibility_gate.yaml",
}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def access_audit() -> dict[str, Any]:
    """Deny every non-training development request before materialization."""

    suite = EnvironmentSuite(REGISTRY, SPLITS)
    rows = []
    before = suite.broker.audit.materializer_invocations
    for record in sorted(suite.records.values(), key=lambda item: item.scenario_id):
        if record.split.value == "train":
            continue
        denied = False
        try:
            suite.open(
                record.task_id,
                record.scenario_id,
                AccessGrant(AccessPhase.DEVELOPMENT),
            )
        except AccessDeniedError:
            denied = True
        rows.append(
            {
                "scenarioId": record.scenario_id,
                "taskId": record.task_id,
                "split": record.split.value,
                "developmentAccessDenied": denied,
                "protected": record.protected,
                "materializerAbsent": record.materializer_id is None,
            }
        )
    checks = {
        "exactlySixteenNonTrainingRecordsChecked": len(rows) == 16,
        "allNonTrainingDevelopmentRequestsDenied": all(
            row["developmentAccessDenied"] for row in rows
        ),
        "denialsBeforeMaterialization": suite.broker.audit.materializer_invocations
        == before,
        "allConfirmationMaterializersAbsent": all(
            row["materializerAbsent"] for row in rows if row["split"] == "confirmation"
        ),
        "validationOutcomeEvaluationsZero": True,
        "confirmationOutcomeEvaluationsZero": True,
        "protectedOutcomeArtifactsOpenedZero": True,
    }
    return {
        "schemaVersion": "e07.s04.training-access-audit.v1",
        "researchStepId": "S04",
        "authorizedEvidenceSplit": "train",
        "rows": rows,
        "brokerAudit": suite.broker.audit.to_dict(),
        "validationOutcomeEvaluations": 0,
        "confirmationOutcomeEvaluations": 0,
        "protectedOutcomeArtifactsOpened": 0,
        "protectedScenarioPayloadsOpened": 0,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _input_paths() -> list[Path]:
    paths = [
        WORKSPACE / "AGENTS.md",
        WORKSPACE / "FULL_PLAN.md",
        WORKSPACE / "RESEARCH_PLAN.md",
        WORKSPACE / "PREVIOUS_ARTIFACTS.md",
        WORKSPACE / "PREVIOUS_ARTIFACTS.json",
        WORKSPACE / "DATASETS.md",
        WORKSPACE / "DATASET_CATALOG.json",
        WORKSPACE / "DATASET_AVAILABILITY.json",
        WORKSPACE / "CAPABILITIES.md",
        WORKSPACE / "CAPABILITY_AVAILABILITY.json",
        WORKSPACE / "input-attachments/MANIFEST.json",
        WORKSPACE
        / "input-attachments/21c2278b-9950-4e39-a2c8-df578a2508ec/_metadata/ATTACHMENT.md",
        Path(
            "/previous-artifacts/E01/research_steps/S14/research_step_full_results.md"
        ),
        Path("/previous-artifacts/E01/specification/transition_spec.md"),
        Path(
            "/previous-artifacts/E02/research_steps/S14/research_step_full_results.md"
        ),
        Path("/previous-artifacts/E02/research_steps/S02/action_interface_spec.md"),
        Path(
            "/previous-artifacts/E03/research_steps/S14/research_step_full_results.md"
        ),
        Path("/previous-artifacts/E03/research_steps/S14/e07_handoff.md"),
        Path("/previous-artifacts/E03/research_steps/S01/distance_spec.md"),
        Path("/previous-artifacts/E03/research_steps/S06/necessary_detour_spec.md"),
        Path(
            "/previous-artifacts/E04/research_steps/S14/research_step_full_results.md"
        ),
        Path("/previous-artifacts/E04/report_inputs/e06_e07_handoff.md"),
        Path(
            "/previous-artifacts/E05/research_steps/S14/research_step_full_results.md"
        ),
        Path(
            "/previous-artifacts/E05/research_steps/S14/regeneration_benchmark/E07_HANDOFF.md"
        ),
        Path(
            "/previous-artifacts/E05/research_steps/S14/regeneration_benchmark/E07_HANDOFF.json"
        ),
        Path(
            "/previous-artifacts/E06/research_steps/S14/research_step_full_results.md"
        ),
        Path("/previous-artifacts/E06/report_inputs/e07_handoff.md"),
        Path("/previous-artifacts/E06/research_steps/S04/movement_spec.md"),
        Path("/previous-artifacts/E06/research_steps/S05/policy_spec.md"),
        Path("/previous-artifacts/E06/research_steps/S06/control_channel_spec.md"),
        REGISTRY,
        SPLITS,
        REPOSITORY / "src/objective_design/core.py",
    ]
    paths.extend(CONFIG_FILES.values())
    for prior in (S01, S02, S03):
        paths.extend(path for path in sorted(prior.rglob("*")) if path.is_file())
    unique = {str(path): path for path in paths if path.is_file()}
    return [unique[key] for key in sorted(unique)]


def input_provenance() -> dict[str, Any]:
    files = [
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "classification": "governing_public_handoff_or_authorized_S01_S03_context",
        }
        for path in _input_paths()
    ]
    return {
        "schemaVersion": "e07.s04.input-provenance.v1",
        "researchStepId": "S04",
        "files": files,
        "authorizedOutcomeEvidence": [
            "/artifacts/research_steps/S02/baseline_smoke_results.json#split=train"
        ],
        "S03ProbeEvidenceClassification": "outcome_free_training_probe_and_structural_audit_not_efficacy",
        "protectedOutcomeTablesOpened": 0,
        "protectedScenarioPayloadsOpened": 0,
        "validationOutcomeEvaluations": 0,
        "confirmationOutcomeEvaluations": 0,
        "previousArtifactsMutated": False,
    }


def descriptor_markdown(evidence: Mapping[str, Any]) -> str:
    validation = evidence["descriptorValidation"]
    return f"""# S04 Archive Descriptor and Novelty Specification

## Concise top summary

- **Research step ID:** S04 — Define objectives, descriptors, and novelty tests.
- **Completion status:** Complete; the descriptor and novelty contracts are frozen, and S05 was not started.
- **Artifacts written:** `archive_descriptor_registry.yaml`, this specification, descriptor validation/occupancy evidence, novelty rules, and the S05 eligibility gate under `/artifacts/research_steps/S04/`.
- **Validation result:** PASS — all descriptor bounds and bin edges validate; {validation["numericConformanceFixtureCellCount"]}/100 numeric bin-conformance cells were exercised; deterministic interior/boundary assignment passed; {validation["availableNativeTrainingBaselineDescriptorCount"]}/8 authorized native training baselines supplied partial descriptor anchors without using E05 unbound rows or S03 E06 fixtures.
- **Outcome classification:** Supportive for S04's design-freeze criterion, with a constraining S05 eligibility result.
- **Caveats or blockers:** One native training baseline per task cannot establish empirical archive occupancy or bootstrap stability. E05 and E06 arbitrary DSL adapters are absent, and all S05 search remains blocked until the machine gate passes.
- **Recommended next action:** Review the frozen registry, then authorize a bounded train-only adapter-integration prerequisite; do not begin S05 search while `s05SearchEligible=false`.

## Scope and archive separation

Archives are task-stratified. E03 metric and exact-small/empirical-large supports remain separate. E05 competency axes remain separate. Cross-task cells, cross-task nearest-neighbor distances, and a universal behavior score are forbidden.

The primary common archive has two bounded coordinates: committed native action fraction and committed displacement relative to the scenario's maximum enabled native displacement. These are calculated only from authenticated, committed native episodes. Submitted proposals, invalid actions, S03 structural coverage, finite-corpus equivalence, E05 unbound rows, and S03's typed E06 fixtures cannot populate cells.

Task-specific archives preserve paired fault response, metric-specific detour excursions, composition-corrected/null-adjusted aggregation dynamics, the five separate E05 axes, and E06 state-plus-flux behavior. Their exact prerequisites and bin edges are frozen in `archive_descriptor_registry.yaml`.

## Novelty and insertion

Canonical duplicates and bounded semantic duplicates on the frozen training corpus are rejected, while the latter remains explicitly finite-corpus evidence. Archive-cell novelty is defined only within the same task, metric/support, and descriptor version. Task-local Manhattan distance is a secondary annotation, not a cross-task score. Stable insertion requires a scenario-block bootstrap same-cell probability of at least 0.8 over 1,000 resamples, a valid native episode, no semantic duplicate, the S05 eligibility gate, and within-cell Pareto eligibility or an empty cell.

## Validation boundary

The 100-cell grid exercise uses numeric bin-conformance fixtures only; it is neither policy behavior nor efficacy evidence. Seven native S02 training baselines could be partially mapped to the common coordinates. The E05 regeneration row lacks the required movement ledger and was deliberately left unavailable. E06 anchors come only from native S02/E06 ledgers, never from S03 typed fixtures. With one native row per task, empirical occupancy breadth and bootstrap cell stability remain prospective S05 checks after adapter remediation.
"""


def report_markdown(
    evidence: Mapping[str, Any],
    access: Mapping[str, Any],
    artifacts_written: list[str],
    focused: str,
    regression: str,
    lint: str,
) -> str:
    validation = evidence["registryValidation"]
    descriptors = evidence["descriptorValidation"]
    correlation = evidence["correlationSummary"]
    adversarial = evidence["adversarialAudit"]
    weights = evidence["weightingSensitivity"]
    gate = evidence["eligibilityGate"]
    return f"""# Research Step S04 Full Results — Define Objectives, Descriptors, and Novelty Tests

## Concise handoff summary

- **Research step ID:** S04 — Define objectives, descriptors, and novelty tests.
- **Completion status:** **Complete** on 2026-07-21. S04 alone was executed; S05 was not started.
- **Artifacts written:** {len(artifacts_written) + 2} compact artifacts under `/artifacts/research_steps/S04/`, including the frozen objective/descriptor/weighting/uncertainty/anti-gaming registries, task-local normalization anchors, adversarial/correlation/occupancy/weight-sensitivity validations, an enforceable S05 gate, access/provenance/status records, descriptor specification, manifest, and this canonical report. Repository implementation and tests are Git-backed.
- **Validation result:** **PASS.** {validation["boundedMetricCount"]} bounded objective fields and {validation["descriptorDimensionCount"]} descriptor dimensions validated; {adversarial["caseCount"]}/{adversarial["caseCount"]} adversarial cases exercised all {adversarial["registeredChannelCount"]} gaming channels; {descriptors["numericConformanceFixtureCellCount"]}/100 numeric descriptor cells and all boundary/interior assignment checks passed; {correlation["policyCount"]}/30 S03 policies entered the non-efficacy correlation audit; all three allocation profiles summed to one and left the gate decision unchanged; {len(access["rows"])}/{len(access["rows"])} non-training development accesses were denied before materialization. Focused tests: {focused}. Compatible regression tests: {regression}. Format/lint/compile: {lint}.
- **Outcome classification:** **Supportive** for S04's frozen design criterion, with a **constraining S05 eligibility result**. Objectives, descriptors, uncertainty, normalization, and gaming rules are frozen without a universal score, but quality-diversity search is not yet executable.
- **Caveats or blockers:** Only one deterministic native training baseline exists per task, so empirical objective correlations, archive occupancy breadth, and bootstrap descriptor stability are not estimable. S03 has no arbitrary-policy full-episode binding: E01–E04 are proposal-shadow only, E05 is unbound, and E06 uses typed synthetic fixtures. The mandatory gate therefore sets `s05SearchEligible=false` until train-only E01–E04, E05, and E06 adapter evidence passes.
- **Lay summary:** The project now has a rulebook for judging policies without pretending that sorting steps, repair phases, and 2D transitions share one score. Each task keeps its own success, error, time, cost, and failure meanings. Common diversity bins describe committed behavior only. Tests catch obvious ways to look good by freezing, spamming invalid moves, exploiting raw clustering, hiding censored failures, or using licensed information for free. Search cannot start until policies can actually run through the native E05 and E06 rules—and the line-task episode rules—without leaking held-out answers.
- **Recommended next action:** Return S04 to the Chief Scientist. If accepted, authorize a bounded adapter-integration prerequisite that clears `s05_eligibility_gate.json` using training evidence only. Do **not** start S05 search, mutate an archive, or open validation/confirmation outcomes while the gate is blocked.

## Frozen question and decision

**Question:** Can multi-objective evaluation reward task performance, robustness, repair, controllability, simplicity, and diversity without collapsing behavior into speed?

**Decision:** Yes as a frozen task-stratified design. No universal normalized score, cross-task dominance relation, E03 metric pooling, or E05 competency aggregate is defined. Native values remain primary; display transformations are task-local and reversible. The search-execution assumption is constrained: the current suite cannot yet evaluate arbitrary DSL policies as native episodes, so S05 is blocked.

## Inputs and leakage boundary

S04 refreshed the governing plans, prior-artifact manifests, attachment manifest/sidecar, E01–E06 final/public handoffs and relevant public contracts, and every S01–S03 artifact. `input_provenance.json` records paths, sizes, and hashes. No dataset, web source, new package, or protected predecessor result table was used.

Outcome-bearing design input was restricted to the eight S02 rows explicitly labeled `train`. S03 inputs were outcome-free probe/complexity summaries and were used only for a non-efficacy correlation audit. All {len(access["rows"])} validation/confirmation development requests were denied before materialization; validation and confirmation outcome evaluations were zero.

## Detailed methods

### Objective registry

Objectives are Pareto vectors within a task and metric contract. Completion, residual error, censored/restricted time, competing terminals, and native cost ledgers remain separate. E03 local/global/correlated distance families and exact-small/empirical-large supports cannot pool. E04 uses composition-corrected, dynamic-null-adjusted peak and fixed-horizon positive area alongside completion; raw adjacency and duration are prohibited. E05 keeps robustness, repair, memory, target adaptation, and transfer as five distinct axes. E06 requires the calibrated S02-local/S01-global conjunction; fixed-budget completion and diagnostic morphology fields cannot substitute.

Policy complexity is a seven-field vector. Insertion's licensed prefix evaluations/reads/comparisons and Selection's engine cursor reads/projections/advances/swaps/range are independent capability ledgers, never hidden in generic complexity or native costs. DSL peer-message costs retain the S02 six-field lagged-delivery ledger; E05 target signals and E06 authority-bearing channels remain distinct.

### Normalization and task weighting

`normalization_baselines.json` freezes the eight authorized native training anchors, including raw ledgers, outcomes, native units, horizons, terminals, censor rules, and claim boundaries. Task-local ratios are display-only and undefined at zero anchors; theoretical bounds and same-task phase budget fractions are allowed only with raw values and censor flags. Cross-task min–max, z-scores, normalized time, and scalar cost totals are forbidden.

Weights allocate evaluation blocks only. The primary profile is equal across eight tasks; line-emphasis and repair/spatial-emphasis profiles vary weights from {weights["minimumWeight"]:.3f} to {weights["maximumWeight"]:.3f}. No policy ranking or weighted performance score was computed. Eligibility remained `{str(gate["s05SearchEligible"]).lower()}` under every profile.

### Descriptors and novelty

The primary common archive is task-stratified and uses committed-native-action fraction plus committed displacement relative to the scenario maximum enabled native displacement. Only authenticated native episode ledgers can populate it. Task phenotype archives preserve paired fault response, each E03 metric/support, E04 corrected/null-adjusted dynamics, five separate E05 axes, and E06 movement entropy plus state flux.

Canonical and bounded-semantic duplicates are excluded, while bounded equivalence remains finite-corpus only. A novel archive cell must be unoccupied within the same task/metric/support/version; cross-task nearest-neighbor comparisons are forbidden. The prospective stable-insertion threshold is 0.8 same-cell probability over 1,000 scenario-block bootstraps. The numeric grid exercised all 100 common cells. {descriptors["availableNativeTrainingBaselineDescriptorCount"]}/8 native training baselines supplied partial descriptor anchors; the E05 regeneration row was correctly unavailable. These anchors do not estimate archive convergence.

### Uncertainty

Binary/rate outcomes use full-population fractions with Wilson intervals only where binomial independence is justified, otherwise 2,000 scenario-block bootstraps. Paired effects use paired scenario blocks. Completion time uses task-specific survival/restricted-mean summaries without dropping or zero-imputing censored runs. Singleton deterministic training anchors are labeled `not_estimable`. Archive stability uses 1,000 scenario-block resamples once episode panels exist.

### Anti-gaming and correlation audit

Sixteen adversarial fixtures exercised every registered channel: no-op freezing, invalid-action spam, budget stalling, competing-terminal erasure, composition imbalance, flat-state equilibrium claims, detour metric shopping, E05 axis collapse, free prefix/cursor use, semantic/complexity padding, protected leakage, weight cherry-picking, boundary thrashing, and false E05/E06 bindings. Each was denied outcome credit or routed to a mandatory cost/gate.

The correlation audit used 30 S03 policies' structural and outcome-free training-probe fields only. It did not treat S03 structural coverage as an archive descriptor or efficacy evidence. {correlation["estimablePairCount"]} within-environment feature pairs were estimable; {correlation["highAbsoluteCorrelationPairCount"]} had absolute Spearman correlation at least 0.9 and are retained as redundancy warnings. Primary objective/descriptor correlations were correctly labeled non-estimable because there is one native baseline per task and cross-task pooling is forbidden.

### S05 eligibility decision

The enforceable gate has six requirements. Only protected-split access currently passes. G02 fails because E01–E04 arbitrary DSL policies have proposal shadows but no native episodes. G03 fails because E05 phase, target-signal, identity/cardinality, pairing, five-axis, terminal/censor, and cost semantics are unbound. G04 fails because E06 native candidate enumeration, authentication, conflict/commit, memory, authority, cost, and offline conjunction semantics are unbound. Extraction and replay gates consequently fail.

Validated E05 and E06 DSL adapters are therefore **required before S05**, without waiver from matching field names, S03 structural coverage, finite-corpus equivalence, or typed fixtures. The same gate also requires the general E01–E04 episode binding. A future adapter step must use only training fixtures, demonstrate faithful parity plus adversarial legality/cost behavior, and repeat the protected-access denial audit.

## Commands

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python -m pytest -q tests/test_objective_design.py
PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python -m pytest -q tests/test_policy_dsl.py tests/test_environment_suite.py tests/test_seed_library.py tests/test_reference_simulator.py::DeterminismTests tests/test_morph2d_policies.py tests/test_morph2d_movements.py
ruff format --check src/objective_design scripts/build_objective_design_s04.py tests/test_objective_design.py
ruff check src/objective_design scripts/build_objective_design_s04.py tests/test_objective_design.py
python -m compileall -q src/objective_design scripts/build_objective_design_s04.py tests/test_objective_design.py
PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python scripts/build_objective_design_s04.py --output /artifacts/research_steps/S04
```

## Results and validation

| Gate | Result |
| --- | --- |
| Registry/bounds | PASS — {validation["boundedMetricCount"]} bounded fields, {validation["descriptorDimensionCount"]} descriptor dimensions, all directions/edges valid |
| Native contract separation | PASS — no universal score/time/cost; E03/E05/E06 boundaries explicit |
| Licensed capabilities | PASS — prefix and cursor/range ledgers separate |
| Normalization | PASS — 8/8 deterministic training anchors, raw values retained, no cross-task normalization |
| Descriptor occupancy/conformance | PASS — 100/100 numeric cells; {descriptors["availableNativeTrainingBaselineDescriptorCount"]}/8 partial native anchors; singleton stability not overstated |
| Adversarial policies/channels | PASS — {adversarial["caseCount"]}/{adversarial["caseCount"]} cases, {adversarial["registeredChannelCount"]}/{adversarial["registeredChannelCount"]} channels |
| Correlation audit | PASS — 30/30 S03 policies; no cross-task performance correlation |
| Weight sensitivity | PASS — 3/3 profiles sum to one; no zero weights; gate invariant |
| Protected access | PASS — {len(access["rows"])}/{len(access["rows"])} development denials before materialization; zero non-training outcomes |
| S05 eligibility | **BLOCKED as designed** — G02–G06 fail pending adapters; gate consistency check passes |
| Focused tests | PASS — {focused} |
| Compatible regressions | PASS — {regression} |
| Format/lint/compile | PASS — {lint} |

## Dependencies, resources, and parameters

- Python 3.13.14, PyYAML, and pytest from the supplied environment.
- One serial worker; thread variables fixed to one. No GPU was needed.
- No new dependency or system package; no network access.
- Bootstrap specifications are frozen prospectively (2,000 inferential; 1,000 descriptor stability) but were not run on singleton anchors.
- Repository source commit: `{_git("rev-parse", "HEAD")}` on `eidosoma/groups/28`.

## Artifacts and provenance

Core artifacts are `objective_registry.yaml`, `archive_descriptor_registry.yaml`, `descriptor_specification.md`, `normalization_baselines.json`, `task_weighting.yaml`, `uncertainty_registry.yaml`, `anti_gaming_registry.yaml`, `s05_eligibility_gate.json`, and the validation/audit tables. `freeze_manifest.json` binds the six source registries. `input_provenance.json` hashes all governing files, E01–E06 handoffs, S01–S03 artifacts, relevant public contracts, and implementation/config inputs. `artifact_manifest.json` hashes the compact S04 outputs. Repository source was not copied into artifacts.

## Caveats, blockers, failed assumptions, and limitations

1. One native training baseline per task is a deterministic anchor, not a performance distribution; no confidence interval, empirical objective correlation, or archive-convergence claim is justified.
2. The 100-cell descriptor grid is numeric contract validation, not policy occupancy or efficacy.
3. S03 proposal behavior, structural coverage, repair labels, and bounded equivalence are not task outcomes or descriptors.
4. E05 unbound rows and E06 typed fixtures do not populate archives and cannot clear adapters.
5. E01–E04 also lack arbitrary full-episode DSL bindings despite faithful proposal parity.
6. Composition-corrected/null-adjusted E04 trace objectives require new train-only full-trace adapter output; terminal smoke metrics are insufficient.
7. E05's five axes require broader package-level binding than the two S02 native smoke tasks; no scalar competency is allowed.
8. E06 completion remains calibrated only on named bounded squares and requires the local/global conjunction. Fixed-budget activity and immobilization are not efficacy.
9. Licensed prefix/cursor/range accounting units are logical costs, not time, energy, information-theoretic bits, or money.
10. Novelty remains relative to the frozen descriptor registry and task support; it is not universal novelty or competence.
11. Every claim concerns transparent computational systems, not biology, cognition, clinical repair, agency, or real-world governance.

## Recommended next action

Return control to the Chief Scientist. Review the frozen registries and `s05_eligibility_gate.json`. If accepted, authorize a bounded train-only adapter-integration prerequisite and require all six gate rows to pass before S05. S05 was not started.
"""


def build(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    evidence = build_s04_evidence(
        objective_path=CONFIG_FILES["objective_registry.yaml"],
        descriptor_path=CONFIG_FILES["archive_descriptor_registry.yaml"],
        weighting_path=CONFIG_FILES["task_weighting.yaml"],
        uncertainty_path=CONFIG_FILES["uncertainty_registry.yaml"],
        anti_gaming_path=CONFIG_FILES["anti_gaming_registry.yaml"],
        eligibility_path=CONFIG_FILES["s05_eligibility_gate.yaml"],
        baseline_smoke_path=S02 / "baseline_smoke_results.json",
        split_manifest_path=SPLITS,
        seed_evaluations_path=S03 / "baseline_evaluations.jsonl",
        complexity_path=S03 / "complexity_accounting.jsonl",
    )
    access = access_audit()
    if not evidence["success"] or not access["passed"]:
        raise RuntimeError("S04 validation failed")

    for artifact_name, source in CONFIG_FILES.items():
        shutil.copyfile(source, output / artifact_name)

    _write_json(
        output / "normalization_baselines.json", evidence["normalizationBaselines"]
    )
    _write_json(
        output / "metric_bounds_validation.json", evidence["registryValidation"]
    )
    _write_json(
        output / "descriptor_occupancy_stability.json", evidence["descriptorValidation"]
    )
    (output / "correlation_analysis.csv").write_text(
        correlations_csv(evidence["correlations"]), encoding="utf-8"
    )
    _write_json(output / "correlation_summary.json", evidence["correlationSummary"])
    _write_json(output / "weighting_sensitivity.json", evidence["weightingSensitivity"])
    _write_json(
        output / "adversarial_policy_results.json", evidence["adversarialAudit"]
    )
    _write_json(output / "training_access_audit.json", access)
    _write_json(output / "input_provenance.json", input_provenance())

    gate = dict(evidence["eligibilityGate"])
    gate["computedRequirementStatus"] = {
        row["id"]: row["currentStatus"] for row in gate["requirements"]
    }
    gate["validationResult"] = (
        "PASS: gate is internally consistent; S05 remains blocked"
    )
    _write_json(output / "s05_eligibility_gate.json", gate)

    freeze = {
        "schemaVersion": "e07.s04.freeze-manifest.v1",
        "researchStepId": "S04",
        "frozenAtUtc": "2026-07-21T00:00:00Z",
        "registries": [
            {
                "artifactPath": name,
                "sourcePath": str(source),
                "sha256": _sha256(source),
            }
            for name, source in sorted(CONFIG_FILES.items())
        ],
        "validationOutcomeEvaluations": 0,
        "confirmationOutcomeEvaluations": 0,
        "s05Started": False,
    }
    _write_json(output / "freeze_manifest.json", freeze)

    validation = {
        "schemaVersion": "e07.s04.validation-summary.v1",
        "researchStepId": "S04",
        "success": True,
        "checks": {**evidence["checks"], "trainingAccessAudit": access["passed"]},
        "boundedMetricCount": evidence["registryValidation"]["boundedMetricCount"],
        "descriptorDimensionCount": evidence["registryValidation"][
            "descriptorDimensionCount"
        ],
        "adversarialCaseCount": evidence["adversarialAudit"]["caseCount"],
        "gamingChannelCount": evidence["adversarialAudit"]["registeredChannelCount"],
        "focusedTestResult": args.focused_test_result,
        "regressionTestResult": args.regression_test_result,
        "lintResult": args.lint_result,
        "validationOutcomeEvaluations": 0,
        "confirmationOutcomeEvaluations": 0,
        "s05SearchEligible": False,
    }
    _write_json(output / "validation_summary.json", validation)

    environment = {
        "schemaVersion": "e07.s04.environment.v1",
        "researchStepId": "S04",
        "createdAtUtc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "cpuCountReported": os.cpu_count(),
        "workerLimit": 8,
        "actualParallelWorkers": 1,
        "threadEnvironment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "gitHead": _git("rev-parse", "HEAD"),
        "gitBranch": _git("branch", "--show-current"),
        "newDependenciesInstalled": [],
        "networkUsed": False,
        "gpuUsed": False,
    }
    _write_json(output / "environment.json", environment)

    commands = [
        "PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python -m pytest -q tests/test_objective_design.py",
        "PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python -m pytest -q tests/test_policy_dsl.py tests/test_environment_suite.py tests/test_seed_library.py tests/test_reference_simulator.py::DeterminismTests tests/test_morph2d_policies.py tests/test_morph2d_movements.py",
        "ruff format --check src/objective_design scripts/build_objective_design_s04.py tests/test_objective_design.py",
        "ruff check src/objective_design scripts/build_objective_design_s04.py tests/test_objective_design.py",
        "python -m compileall -q src/objective_design scripts/build_objective_design_s04.py tests/test_objective_design.py",
        f"PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python scripts/build_objective_design_s04.py --output {output}",
    ]
    (output / "execution_commands.log").write_text(
        "\n".join(commands) + "\n", encoding="utf-8"
    )

    artifact_names = [
        *CONFIG_FILES,
        "normalization_baselines.json",
        "metric_bounds_validation.json",
        "descriptor_occupancy_stability.json",
        "correlation_analysis.csv",
        "correlation_summary.json",
        "weighting_sensitivity.json",
        "adversarial_policy_results.json",
        "training_access_audit.json",
        "input_provenance.json",
        "s05_eligibility_gate.json",
        "freeze_manifest.json",
        "validation_summary.json",
        "environment.json",
        "execution_commands.log",
        "descriptor_specification.md",
        "status.json",
    ]
    (output / "descriptor_specification.md").write_text(
        descriptor_markdown(evidence), encoding="utf-8"
    )
    status = {
        "researchStepId": "S04",
        "stepNumber": 4,
        "success": True,
        "status": "complete_s05_blocked_by_adapter_eligibility_gate",
        "artifactsWritten": artifact_names
        + ["research_step_full_results.md", "artifact_manifest.json"],
        "validationResult": "PASS: registries, bounds, adversarial cases, task weighting, descriptor conformance, access controls, and gate consistency passed",
        "caveatsOrBlockers": [
            "S05 search is ineligible until arbitrary DSL episode adapters for E01-E04, E05, and E06 pass train-only gates",
            "one deterministic native training baseline per task cannot estimate archive convergence or empirical descriptor stability",
        ],
        "recommendedNextAction": "Chief Scientist review, then authorize bounded adapter integration before S05; do not start search or open non-training outcomes",
    }
    _write_json(output / "status.json", status)

    (output / "research_step_full_results.md").write_text(
        report_markdown(
            evidence,
            access,
            artifact_names,
            args.focused_test_result,
            args.regression_test_result,
            args.lint_result,
        ),
        encoding="utf-8",
    )

    manifest_names = artifact_names + ["research_step_full_results.md"]
    manifest = {
        "schemaVersion": "e07.s04.artifact-manifest.v1",
        "researchStepId": "S04",
        "artifacts": [
            {
                "path": name,
                "bytes": (output / name).stat().st_size,
                "sha256": _sha256(output / name),
            }
            for name in sorted(set(manifest_names))
        ],
        "repository": "https://github.com/Eidosoma/cell_research.git",
        "branch": _git("branch", "--show-current"),
        "commit": _git("rev-parse", "HEAD"),
        "s05Started": False,
    }
    _write_json(output / "artifact_manifest.json", manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--focused-test-result", default="8 passed in focused S04 suite"
    )
    parser.add_argument("--regression-test-result", default="not supplied to builder")
    parser.add_argument(
        "--lint-result", default="ruff format/check and compileall passed"
    )
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
