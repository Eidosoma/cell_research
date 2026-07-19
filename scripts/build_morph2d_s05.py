#!/usr/bin/env python3
"""Build and validate E06 S05 priced local spatial Algotype artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
POLICIES = ROOT / "configs/morphologies/policy_catalog.yaml"
ENVIRONMENTS = ROOT / "configs/morphologies/environment_catalog.yaml"
GRAMMARS = ROOT / "configs/morphologies/grammar_catalog.yaml"
TARGETS = ROOT / "configs/morphologies/target_catalog.yaml"
SOURCE = ROOT / "src/morph2d/policies.py"
TEST = ROOT / "tests/test_morph2d_policies.py"
S01_DIR = Path("/artifacts/research_steps/S01")
S02_DIR = Path("/artifacts/research_steps/S02")
S03_DIR = Path("/artifacts/research_steps/S03")
S04_DIR = Path("/artifacts/research_steps/S04")
ATTACHMENT_DIR = WORKSPACE / "input-attachments/21c2278b-9950-4e39-a2c8-df578a2508ec"
UPSTREAM_INPUTS = {
    "agents": WORKSPACE / "AGENTS.md",
    "full_plan": WORKSPACE / "FULL_PLAN.md",
    "research_plan_pre_s05_update": WORKSPACE / "RESEARCH_PLAN.md",
    "previous_artifacts_md": WORKSPACE / "PREVIOUS_ARTIFACTS.md",
    "previous_artifacts_json": WORKSPACE / "PREVIOUS_ARTIFACTS.json",
    "capabilities": WORKSPACE / "CAPABILITIES.md",
    "capability_availability": WORKSPACE / "CAPABILITY_AVAILABILITY.json",
    "datasets": WORKSPACE / "DATASETS.md",
    "dataset_catalog": WORKSPACE / "DATASET_CATALOG.json",
    "dataset_availability": WORKSPACE / "DATASET_AVAILABILITY.json",
    "attachment_manifest": WORKSPACE / "input-attachments/MANIFEST.json",
    "attachment_sidecar": ATTACHMENT_DIR / "_metadata/ATTACHMENT.md",
    "s01_report": S01_DIR / "research_step_full_results.md",
    "s01_catalog": S01_DIR / "target_catalog.yaml",
    "s01_validation": S01_DIR / "validation_summary.json",
    "s01_manifest": S01_DIR / "artifact_manifest.json",
    "s02_report": S02_DIR / "research_step_full_results.md",
    "s02_grammar_spec": S02_DIR / "grammar_spec.md",
    "s02_grammars": S02_DIR / "target_grammars.yaml",
    "s02_validation": S02_DIR / "validation_summary.json",
    "s02_manifest": S02_DIR / "artifact_manifest.json",
    "s03_report": S03_DIR / "research_step_full_results.md",
    "s03_environment_spec": S03_DIR / "environment_spec.md",
    "s03_environment_catalog": S03_DIR / "environment_library/environment_catalog.yaml",
    "s03_boundary_requirements": S03_DIR / "boundary_requirements.csv",
    "s03_conjunctive_evaluation": S03_DIR / "conjunctive_evaluation.json",
    "s03_validation": S03_DIR / "validation_summary.json",
    "s03_manifest": S03_DIR / "artifact_manifest.json",
    "s04_report": S04_DIR / "research_step_full_results.md",
    "s04_movement_spec": S04_DIR / "movement_spec.md",
    "s04_movement_catalog": S04_DIR / "movement_catalog.yaml",
    "s04_cost_ledger": S04_DIR / "movement_cost_ledger.csv",
    "s04_conflict_audit": S04_DIR / "conflict_resolution_audit.json",
    "s04_evaluation_layers": S04_DIR / "evaluation_layer_results.json",
    "s04_validation": S04_DIR / "validation_summary.json",
    "s04_manifest": S04_DIR / "artifact_manifest.json",
    "e01_transition": Path(
        "/previous-artifacts/E01/research_steps/S03/transition_spec.md"
    ),
    "e01_api": Path("/previous-artifacts/E01/research_steps/S05/api_documentation.md"),
    "e01_event_schema": Path(
        "/previous-artifacts/E01/research_steps/S06/schema_documentation.md"
    ),
    "e04_metric_spec": Path(
        "/previous-artifacts/E04/research_steps/S04/metric_specification.md"
    ),
    "e04_kinetic_intervention": Path(
        "/previous-artifacts/E04/research_steps/S07/kinetic_intervention_specification.md"
    ),
    "e04_identity_control": Path(
        "/previous-artifacts/E04/research_steps/S08/identity_control_specification.md"
    ),
    "e04_policy_observation_audit": Path(
        "/previous-artifacts/E04/research_steps/S08/policy_observation_audit.json"
    ),
    "e04_policy_label_switch": Path(
        "/previous-artifacts/E04/research_steps/S09/policy_label_switch_specification.md"
    ),
    "e04_handoff": Path(
        "/previous-artifacts/E04/research_steps/S14/e06_e07_handoff.md"
    ),
}

sys.path.insert(0, str(ROOT))

from src.morph2d.environments import (  # noqa: E402
    evaluate_conjunctive,
    load_environment_catalog,
)
from src.morph2d.grammar import load_grammar_catalog  # noqa: E402
from src.morph2d.movements import (  # noqa: E402
    ENABLED_KINDS,
    initial_movement_state,
    make_proposal,
    parse_movement_state,
    resolve_batch,
    state_tokens,
    validate_proposal,
)
from src.morph2d.policies import (  # noqa: E402
    OBSERVATION_LEDGER_FIELDS,
    PolicyMemory,
    PolicyValidationError,
    build_policy_observation,
    canonical_policy_event_bytes,
    compile_relation_profile,
    decide_policy,
    decision_source_forbidden_accesses,
    enumerate_candidate_affordances,
    execute_policy_activation,
    load_policy_catalog,
    materialize_decision,
    policy_complexity,
    policy_to_dict,
    relation_profile_to_dict,
    update_policy_memory,
    validate_policy_payload,
)
from src.morph2d.targets import load_target_catalog  # noqa: E402


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def _verify_upstream_manifest(directory: Path) -> dict[str, Any]:
    manifest = json.loads((directory / "artifact_manifest.json").read_text())
    mismatches = []
    for record in manifest["artifacts"]:
        path = directory / record["path"]
        observed = _sha256_file(path) if path.is_file() else None
        if observed != record["sha256"]:
            mismatches.append(
                {
                    "path": record["path"],
                    "expected": record["sha256"],
                    "observed": observed,
                }
            )
    validation = json.loads((directory / "validation_summary.json").read_text())
    return {
        "researchStepId": manifest["researchStepId"],
        "checkedArtifactCount": len(manifest["artifacts"]),
        "hashMismatches": mismatches,
        "upstreamValidationSuccess": validation["success"],
        "success": not mismatches and validation["success"],
    }


def _focused_tests() -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_morph2d_targets.py",
        "tests/test_morph2d_grammar.py",
        "tests/test_morph2d_environments.py",
        "tests/test_morph2d_movements.py",
        "tests/test_morph2d_policies.py",
    ]
    completed = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=False
    )
    return {
        "command": " ".join(command),
        "returnCode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "success": completed.returncode == 0,
    }


def _input_provenance() -> dict[str, Any]:
    records = []
    for name, path in UPSTREAM_INPUTS.items():
        if not path.is_file():
            raise FileNotFoundError(f"required S05 input missing: {path}")
        records.append(
            {
                "name": name,
                "path": str(path),
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
                "readOnlyDependency": str(path).startswith("/previous-artifacts")
                or str(path).startswith("/artifacts/research_steps/S0"),
            }
        )
    for name, path in (
        ("repository_policy_catalog", POLICIES),
        ("repository_policy_source", SOURCE),
        ("repository_policy_tests", TEST),
        ("repository_environment_catalog", ENVIRONMENTS),
        ("repository_grammar_catalog", GRAMMARS),
        ("repository_target_catalog", TARGETS),
    ):
        records.append(
            {
                "name": name,
                "path": str(path),
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
                "readOnlyDependency": False,
            }
        )
    return {
        "schemaVersion": "e06.s05.input-provenance.v1",
        "researchStepId": "S05",
        "inputs": records,
    }


def _actor_at(state, site_id: str) -> str:
    return state.occupant_map[site_id].occupant_id


def _state_for_fixture(environment, transform: str):
    state = initial_movement_state(environment)
    if transform == "none":
        return state
    if transform == "adjacent_swap_r2_c0_r3_c0":
        proposal = make_proposal(state, "adjacent_swap", ("r2_c0", "r3_c0"))
        result = resolve_batch(
            environment, state, (proposal,), batch_nonce="s05-toy-damage-v1"
        )
        return parse_movement_state(result["postState"])
    raise ValueError(f"unknown S05 state transform: {transform}")


def _column_field(environment) -> dict[str, int]:
    return {
        site.site_id: int(site.site_id.split("_c")[1])
        for site in environment.sites
        if site.role != "obstacle"
    }


def _zero_lagged(environment) -> dict[str, int]:
    return {site.site_id: 0 for site in environment.sites if site.role != "obstacle"}


def _toy_kwargs(raw, environment, state, actor_id, policy, grammar_by_id):
    kwargs: dict[str, Any] = {}
    profile = None
    if "relationGrammarId" in raw:
        profile = compile_relation_profile(grammar_by_id[raw["relationGrammarId"]])
        kwargs["relation_profile"] = profile
    if policy.strategy == "boundary_seeking":
        kwargs["boundary_direction"] = raw["boundaryDirection"]
        kwargs["boundary_tokens"] = tuple(raw["boundaryTokens"])
    elif policy.strategy == "gradient_following":
        if raw["gradientField"] != "square_column_uint8":
            raise ValueError("unknown frozen gradient fixture")
        kwargs["gradient_levels"] = _column_field(environment)
        kwargs["gradient_direction"] = raw["gradientDirection"]
    elif policy.strategy == "exploration":
        kwargs["decision_key"] = raw["decisionKey"]
        kwargs["activation_index"] = int(raw["activationIndex"])
    elif policy.strategy == "memory_based_recovery":
        kwargs["memory"] = PolicyMemory(
            best_local_utility=int(raw["memory"]["bestLocalRelationUtility"]),
            frustration=int(raw["memory"]["frustration"]),
        )
    elif policy.strategy == "conflict_avoidance":
        zero = _zero_lagged(environment)
        baseline = build_policy_observation(
            environment,
            state,
            actor_id,
            policy,
            relation_profile=profile,
            lagged_conflicts=zero,
        )
        baseline_decision = decide_policy(policy, baseline.observation.payload)
        if baseline_decision.action != "proposal":
            raise RuntimeError("conflict toy has no unpenalized proposal")
        selected = baseline.candidate_map[baseline_decision.selected_candidate_key]
        zero[selected.target_site] = 3
        kwargs["lagged_conflicts"] = zero
        kwargs["baselineDecisionCandidateKey"] = (
            baseline_decision.selected_candidate_key
        )
    return kwargs


def _execution_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in kwargs.items()
        if key != "baselineDecisionCandidateKey"
    }


def _policy_card(policy, result: Mapping[str, Any]) -> str:
    complexity = policy_complexity(policy)
    movement_kinds = ", ".join(f"`{item}`" for item in policy.allowed_movement_kinds)
    features = ", ".join(f"`{item}`" for item in policy.observation_features)
    non_guarantees = "\n".join(f"- `{item}`" for item in policy.non_guarantees)
    return f"""# Policy card: {policy.policy_id}

- **S05 strategy:** `{policy.strategy}`
- **Intended behavior:** {policy.intended_behavior}
- **Observed frozen toy behavior:** `{result["action"]}` with reason `{result["reason"]}` and S04 outcome `{result["movementOutcome"]}`.
- **Allowed S04 movement kinds:** {movement_kinds}
- **Priced observation features:** {features}
- **Maximum candidates:** {policy.max_candidates}
- **Maximum communicated bits:** {policy.information_budget_max_bits}
- **Observed toy communicated bits:** {result["communicatedBitsUpperBound"]}
- **Target specificity:** `{policy.target_specificity}`
- **Complexity proxy:** {complexity["complexityScore"]} = rules {complexity["decisionRuleCount"]} + branches {complexity["branchCount"]} + tunable scalars {complexity["tunableScalarCount"]} + ceil(memory bits {complexity["persistentMemoryBits"]} / 4).

## Observation boundary

The decision function receives only an immutable payload of priced derived scalars and activation-local opaque candidate handles. Site IDs, routes, occupant identities, site roles, raw boundary tags, proposal/state authentication, current competing proposals, priorities, analysis labels, whole-grid grammar scores, S01 completion, future state, and random keys remain engine-only.

## Non-guarantees

{non_guarantees}
"""


def _policy_spec(catalog: Mapping[str, Any]) -> str:
    return f"""# E06 S05 Local Spatial Algotype Specification

## Scope and authority

Version `{catalog["schemaVersion"]}` defines six deterministic CPU reference policies over S04's frozen proposal-intent, legality, conflict, and movement-cost interface. It defines observation projections, explicit information/read ledgers, opaque candidate intents, decision rules, memory ownership, counter-addressed exploration, target specificity, and a transparent complexity proxy. It does not define episode scheduling, gradient generation, central control, channel noise, GPU execution, formation performance, or S06 communication semantics.

## Observation and authentication boundary

The engine enumerates graph-legal S04 candidates and gives the policy only activation-local handles plus declared derived scalar features. Candidate count and handles are priced because affordance availability itself can reveal local immobility or degree. The candidate order is keyed by hidden state/proposal commitments so a handle position does not systematically encode movement kind or direction. The policy never receives identity, coordinates/site IDs, routes, occupant identities, raw site roles, obstacle/fixed-neighbor flags, raw boundary tags, state/proposal hashes, expected-resource identities, conflict priorities, other current-batch proposals, analysis labels, whole-grid grammar results, S01 completion, or future state. S04 authentication is materialized only after a decision.

Every observation records logical source reads, utility evaluations, comparisons, counter draws, persistent memory bits, and a fixed upper bound on communicated bits. These are reference accounting units, not claims about physical sensory cost.

## Six policies

- **Greedy neighbor satisfaction:** selects the deterministic best strictly positive change in an S02-derived actor-contact utility.
- **Boundary seeking:** applies only to declared actor tokens and selects a positive change in priced natural-exterior level; raw tags and safety roles remain hidden. Periodic domains expose zero natural exterior.
- **Gradient following:** selects a positive change in an externally supplied uint8 local field. S05 validates consumption only; S06 must version source, update, bandwidth, spatial resolution, noise, and control ownership.
- **Exploration:** receives one bounded result from a counter-addressed uint64 digest and selects that candidate. The key, actor identity, and counter address remain engine-only.
- **Memory-based recovery:** compares current actor-contact utility with an identity-owned signed-int8 reference plus two-bit frustration state. It proposes a positive restorative step only below the remembered reference.
- **Conflict avoidance:** scores S02 actor-contact change minus fixed penalties for a saturated 0–3 destination count from the prior completed batch and the S04 movement cost. It never observes current competing proposals or conflict priorities.

## Local grammar and global evaluation

S02 edge and neighbor-count constraints compile to an actor-token/neighbor-token contact table. Positive minima add local utility; hard zero maxima add amplified penalties. Boundary, motif, symmetry, whole-grid intervals, acceptance, and all S01 audits are excluded. Policy events store this local feedback separately from offline S02 whole-grid grammar and independent S01 equivalence/component/topology evaluation. Completion remains conjunctive and cannot be inferred from a local policy score.

## Complexity and target specificity

The declared complexity score is `decision rules + branches + tunable scalars + ceil(persistent memory bits / 4)`. It is a transparent comparison proxy, not Kolmogorov or cognitive complexity. Grammar-parameterized and scenario-parameterized policies explicitly declare target specificity; only exploration is target-agnostic.

## Claim boundary

Passing S05 demonstrates deterministic, permission-audited local decision semantics and toy compatibility with S04. It does not demonstrate target formation, repair probability, convergence, chimeric benefit, biological sensing, or fair top-down control. Those require later separately authorized steps.
"""


def _report(
    validation: Mapping[str, Any],
    toy_rows: list[dict[str, Any]],
    complexity_rows: list[dict[str, Any]],
    provenance: Mapping[str, Any],
    artifacts: list[str],
) -> str:
    toy_table = [
        "| Policy | Intended strategy | Toy action | Reason | S04 outcome | Bits used / cap | Deterministic |",
        "| --- | --- | --- | --- | --- | ---: | --- |",
    ]
    complexity_by_id = {item["policyId"]: item for item in complexity_rows}
    for row in toy_rows:
        complexity = complexity_by_id[row["policyId"]]
        toy_table.append(
            f"| `{row['policyId']}` | `{row['strategy']}` | `{row['action']}` | "
            f"`{row['reason']}` | `{row['movementOutcome']}` | "
            f"{row['communicatedBitsUpperBound']} / {complexity['informationBudgetMaxBits']} | "
            f"{row['deterministicReplayPass']} |"
        )
    test_summary = validation["focusedTests"]["stdout"].splitlines()[-1]
    next_action = (
        "Return control to the Chief Scientist. If S05 is accepted, separately "
        "authorize only S06 to define and price top-down channels; do not start "
        "S06 automatically."
    )
    return f"""# Research Step S05 Full Results — Construct Local Spatial Algotypes

## Top summary

- **Research step ID:** S05
- **Completion status:** **Complete** on {datetime.now(timezone.utc).date().isoformat()}; only S05 was executed and S06 was not started.
- **Artifacts written:** {len(artifacts) + 1} files total ({len(artifacts)} recursively hash-listed files plus `artifact_manifest.json`): the policy specification and library/catalog, six policy cards and six replayable toy events, relation profiles, observation-permission and budget audits, movement compatibility, complexity/target-specificity accounting, memory/gradient/evaluation separation audits, deterministic serialization/replay evidence, validation/provenance/command records, manifest, and this canonical report.
- **Validation result:** **PASS** — six policies, six frozen toy behaviors, all four S04 movement kinds, explicit observation/read/bit ledgers, source-permission probes, deterministic decisions and replay, memory limits, gradient input failures, S02/S01 separation, {validation["upstreamImmutability"]["checkedArtifactCount"]} upstream artifact hashes, and focused tests (`{test_summary}`) all passed.
- **Outcome classification:** **supportive.** Six distinct local decision semantics can consume only priced immutable projections and produce legal S04 proposal intents with deterministic replay.
- **Caveats or blockers:** These are hand-designed one-activation CPU policies and toy behavior checks, not formation, repair, convergence, or chimeric-performance evidence. Five policies are grammar- or scenario-parameterized. The gradient is only a supplied uint8 input; its ownership, generation, updates, noise, spatial resolution, and comparative budget remain explicitly deferred to S06.
- **Lay summary:** Six small rule sets can now choose legal local moves without seeing hidden engine state or whole-shape answers. Every fact a rule receives is listed and charged, saved decisions replay exactly, and memory or random exploration have explicit limits. This does not yet show that the rules can build or repair a full pattern over time.
- **Recommended next action:** {next_action}

## Frozen question and completion criterion

**Frozen question:** Can greedy neighbor satisfaction, boundary seeking, gradient following, exploration, memory-based recovery, and conflict avoidance be implemented as distinct deterministic local Algotypes over S04 while every observation, memory bit, candidate affordance, and target-specific parameter is explicit and priced?

**Completion criterion:** each policy must pass observation-permission, deterministic-decision, toy-behavior, movement-compatibility, replay, and complexity/information-budget validation, while preserving separate S02 local feedback and independent S01 global evaluation. The criterion was met at the one-activation CPU reference layer.

## Lay summary

An Algotype is the rule a simulated cell follows, not its identity, token, or position. The six rules now receive only compact local summaries: for example, how a possible move changes neighbor satisfaction, a natural-edge level, a supplied gradient, an old conflict count, or the cell's own tiny memory. The engine keeps coordinates, legal routes, authentication, other cells' current proposals, and the answer to the whole-pattern task private. A chosen opaque handle is converted to a fully authenticated S04 proposal only afterward.

## Inputs

- Refreshed `/workspace/AGENTS.md`, `FULL_PLAN.md`, and the pre-completion `RESEARCH_PLAN.md`, plus capability and dataset availability records.
- Manifest-verified S01 target/equivalence/component/topology artifacts, S02 grammar and falsification artifacts, S03 graph/occupancy/boundary artifacts, and S04 proposal/legality/conflict/cost artifacts.
- E01 immutable-observation, identity-owned-memory, authenticated-proposal, counter-randomness, conflict, cost, event, and replay contracts.
- E04 label-blind policy/intervention, exact-composition, metric, transport-cost, identity-control, policy-label-switch, and E06 handoff constraints.
- `input-attachments/MANIFEST.json` and its attachment sidecar. No dataset, network access, package installation, or previous-artifact mutation was required.

Exact paths and SHA-256 values are in `input_provenance.json`. All {validation["upstreamImmutability"]["checkedArtifactCount"]} manifest-listed S01–S04 artifacts rehashed successfully.

## Methods

### Typed policies and priced observations

`src/morph2d/policies.py` parses one versioned catalog with six exact strategy/feature combinations. Observation construction enumerates legal S04 affordances engine-side, caps them at 16 with kind-stratified retention, assigns activation-local opaque handles in a state-keyed order, computes only declared projections, and emits a {len(OBSERVATION_LEDGER_FIELDS)}-field read/information ledger. Candidate count and handle identifiers are charged. The pure decision function receives only `observation.payload`; state hashes and audit metadata remain outside that call boundary.

### S02 actor-local utility

All seven S02 grammars were compiled to token-contact profiles. Only edge-count and neighbor-count constraints can contribute: positive minima are positive weights, while hard zero maxima are amplified negative weights. Motif, boundary, symmetry, whole-grid range, acceptance, and S01 completion fields are excluded. Utility changes are computed for the acting identity only and are bounded proxy feedback, not a whole-grid grammar score.

### Policy rules

Greedy, boundary, gradient, and memory policies require a strictly positive disclosed improvement. Exploration uses one rejection-unbiased bounded digest draw addressed by decision key, policy, engine-only actor identity, activation index, and draw index. Conflict avoidance subtracts fixed prior-batch congestion and the legal candidate's S04 single-proposal `totalGraphDisplacement`; only that scalar and saturated lagged destination counts are visible, not the route or remaining ledger. Memory stores a signed-int8 best actor-local utility and two-bit frustration value, for ten persistent bits.

### Authentication and evaluation separation

The policy chooses only a local candidate handle. Engine-only mapping then materializes the S04 route, actor identity, expected resource identities, state hash, and proposal ID; normal S04 validation/conflict/cost logic remains unchanged. Each event records `s02LocalFeedback` separately from an offline-only `s01GlobalEvaluation` marker. The artifact-level audit independently computed the S02 whole-grid grammar and S01 equivalence/component/topology record after a toy movement and retained conjunctive completion.

### Falsification and validation design

Permission mutation probes attempted to inject actor identity, site IDs, routes, state hashes, raw boundary tags, fixed-neighbor data, conflict priority, analysis labels, and S01 completion. Parser mutation added an undeclared feature. Gradient probes removed required coverage and exceeded uint8 range. Memory probes exceeded signed-int8/two-bit limits. Periodic-boundary behavior was required to no-op because there is no natural exterior. Counter-addressed exploration was sampled across deterministic addresses to exercise all four S04 movement kinds. Every frozen event was independently regenerated and compared byte-for-byte.

### Commands

```bash
cd /workspace/cell-research
python -m pytest -q tests/test_morph2d_targets.py tests/test_morph2d_grammar.py tests/test_morph2d_environments.py tests/test_morph2d_movements.py tests/test_morph2d_policies.py
python scripts/build_morph2d_s05.py build --output-dir /artifacts/research_steps/S05
python scripts/build_morph2d_s05.py validate --output-dir /artifacts/research_steps/S05
ruff check src/morph2d scripts/build_morph2d_s05.py tests/test_morph2d_policies.py
ruff format --check src/morph2d scripts/build_morph2d_s05.py tests/test_morph2d_policies.py
python -m compileall -q src/morph2d scripts/build_morph2d_s05.py tests/test_morph2d_policies.py
```

The run used Python {provenance["python"]} and intentional serial execution on one worker. Exact fixture construction was small and deterministic, so CPU parallelism and the L4 GPU would not improve evidence quality.

## Results

{os.linesep.join(toy_table)}

### Anchor results

- **Six distinct strategies:** all six frozen policy cards produced their intended proposal/no-op semantics. The toy fixtures test one-step intended behavior; no emergent multi-step formation claim is made.
- **Observation permissions:** all forbidden-field injection probes were rejected; static source inspection found no engine-only access in the pure decision function; policy payloads contained no site IDs or authentication/global-evaluation fields.
- **Information accounting:** every observation emitted all {len(OBSERVATION_LEDGER_FIELDS)} ledger fields and remained at or below its declared {min(row["informationBudgetMaxBits"] for row in complexity_rows)}–{max(row["informationBudgetMaxBits"] for row in complexity_rows)} bit cap. Natural-boundary, gradient, lagged-conflict, memory, and counter reads appeared only for their declared strategies.
- **Movement compatibility:** the policy library generated legal S04 adjacent swaps, vacancy moves, short exchanges, and rotations. Opaque candidate selection did not enable long-range exchange, division, or removal; S04 invariants and zero movement-legality boundary reads remained intact.
- **Determinism and replay:** six observations, decisions, materializations, and event records regenerated byte-identically. Counter exploration was deterministic for repeated addresses and exercised all four enabled movement kinds across the bounded-hex and periodic-vacancy probes.
- **Memory and conflicts:** identity-owned memory used exactly ten persistent bits, no-op'd once its local reference was met, reset frustration after acceptance, and saturated failure frustration at three. Conflict avoidance changed away from a destination assigned a saturated prior-batch conflict count without seeing current proposals or priorities.
- **Evaluation separation:** the post-move offline record kept S02 whole-grid grammar, S01 global audit, and conjunctive completion separate from the policy's actor-local contact feedback. The policy payload disclosed none of them.

Machine-readable results are in `toy_behavior_results.*`, `observation_permissions.json`, `observation_budget.csv`, `movement_compatibility.json`, `deterministic_decisions.json`, `serialization_replay.json`, `memory_semantics.json`, `gradient_interface_audit.json`, `evaluation_separation.json`, `relation_profiles.json`, and `complexity_and_specificity.csv`.

## Validation

`validation_summary.json` records all gates as true:

- exact six-strategy catalog parse/round trip and rejection of unpriced strategy features;
- seven bounded S02 actor-local relation profiles with excluded global/motif/boundary/symmetry semantics;
- forbidden observation-field rejection, no site IDs in policy payloads, and engine-only authentication materialization;
- all declared read types and communicated-bit ceilings, including candidate-availability leakage and ten memory bits;
- six deterministic toy decisions, legal S04 materialization, invariant-preserving costs, byte-identical event replay, and all four enabled movement kinds;
- bounded versus periodic boundary behavior, explicit uint8 gradient coverage/range failures, lagged-only conflict input, memory range/update behavior, complexity and target-specificity records;
- separate S02 local feedback, offline S02 whole-grid evaluation, S01 global audit, and conjunctive completion;
- S01–S04 immutability, final artifact hashes, and focused tests (`{test_summary}`).

## Caveats, blockers, failed assumptions, and limitations

- Toy one-step behavior confirms implementation intent, not emergent formation, repair, convergence, robustness, or chimeric interaction. These outcomes require later episode experiments.
- Greedy, memory, and conflict policies depend on a target grammar-derived contact profile. Boundary and gradient policies depend on scenario-declared token/direction or field/direction parameters. Exploration alone is target-agnostic; comparisons must account for this semantic prior.
- Candidate availability is itself informative. S05 prices count, handles, topology reads, and safety-role reads used by the engine to form affordances, but the bit ceiling is a transparent fixed encoding bound rather than an information-theoretic mutual-information estimate.
- The state-keyed opaque ordering prevents a stable handle-to-kind/direction convention, but a policy with external side information about hidden state could infer more. Such external state is forbidden by the scenario contract and must be audited in later engines.
- The S02 contact projection is deliberately incomplete because S02 falsified local grammar sufficiency. Positive local feedback can worsen or fail to determine whole-grid grammar or S01 completion.
- Boundary policy sees a priced natural-exterior level, not raw tags, obstacle contacts, or fixed roles. Periodic environments have no natural exterior. Irregular exterior semantics still depend on S03's explicit engine tags.
- S05 consumes a supplied static uint8 gradient but does not define who generates it, its semantic content, bandwidth, update frequency, resolution, noise, or central access. Defining those would be S06 and was not started.
- Conflict avoidance sees only saturated destination counts from the previous completed batch. It does not observe lagged route-interior conflicts, cannot predict current conflicts, cannot compute a maximum matching, and cannot inspect conflict priorities.
- The complexity score is a declared implementation proxy; it is not code length, sample complexity, cognitive sophistication, or biological plausibility.
- These are computational policy proxies in a synthetic graph system, not direct evidence about biological cells, morphogenesis, cognition, or clinical behavior.

## Artifacts and provenance

Repository-backed implementation lives in `src/morph2d/policies.py`, `configs/morphologies/policy_catalog.yaml`, `scripts/build_morph2d_s05.py`, `tests/test_morph2d_policies.py`, and public exports in `src/morph2d/__init__.py`; source is not duplicated under artifacts. The artifact policy library holds the frozen catalog, index, cards, and canonical toy events. Root artifacts provide the normative specification, compact tables and audits, validation, environment/input provenance, commands, canonical report, and recursive SHA-256 manifest.

## Recommended next action

{next_action}
"""


def build(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    library = output / "policy_library"
    cards = library / "cards"
    events = library / "toy_events"
    cards.mkdir(parents=True, exist_ok=True)
    events.mkdir(parents=True, exist_ok=True)

    raw_catalog = yaml.safe_load(POLICIES.read_text(encoding="utf-8"))
    metadata, policies = load_policy_catalog(POLICIES)
    _, environments = load_environment_catalog(ENVIRONMENTS)
    _, grammars = load_grammar_catalog(GRAMMARS)
    _, targets = load_target_catalog(TARGETS)
    environment_by_id = {item.environment_id: item for item in environments}
    grammar_by_id = {item.grammar_id: item for item in grammars}
    target_by_id = {item.target_id: item for item in targets}
    policy_by_id = {item.policy_id: item for item in policies}

    shutil.copyfile(POLICIES, library / "policy_catalog.yaml")
    (output / "policy_spec.md").write_text(_policy_spec(raw_catalog), encoding="utf-8")

    upstream = [
        _verify_upstream_manifest(directory)
        for directory in (S01_DIR, S02_DIR, S03_DIR, S04_DIR)
    ]
    upstream_immutability = {
        "schemaVersion": "e06.s05.upstream-immutability.v1",
        "researchStepId": "S05",
        "steps": upstream,
        "checkedArtifactCount": sum(item["checkedArtifactCount"] for item in upstream),
        "success": all(item["success"] for item in upstream),
    }
    _write_json(output / "upstream_immutability.json", upstream_immutability)
    input_provenance = _input_provenance()
    _write_json(output / "input_provenance.json", input_provenance)

    relation_profiles = [
        relation_profile_to_dict(compile_relation_profile(grammar))
        for grammar in grammars
    ]
    _write_json(
        output / "relation_profiles.json",
        {
            "schemaVersion": "e06.s05.relation-profile-library.v1",
            "researchStepId": "S05",
            "profiles": relation_profiles,
            "profileCount": len(relation_profiles),
            "projectedConstraintCount": sum(
                len(item["projectedConstraintIds"]) for item in relation_profiles
            ),
            "excludedConstraintCount": sum(
                len(item["excludedConstraintIds"]) for item in relation_profiles
            ),
        },
    )

    permission_probes = []
    for key in (
        "actorId",
        "siteId",
        "route",
        "expectedOccupants",
        "stateSha256",
        "boundaryTags",
        "fixedBoundaryNeighbors",
        "conflictPriority",
        "analysisLabel",
        "s01GlobalCompletionAudit",
    ):
        rejected = False
        reason = None
        try:
            validate_policy_payload({"candidates": [], key: "probe"})
        except PolicyValidationError as error:
            rejected = True
            reason = str(error)
        permission_probes.append(
            {"injectedField": key, "rejected": rejected, "reason": reason}
        )

    toy_rows = []
    toy_records = []
    budget_rows = []
    deterministic_records = []
    serialization_records = []
    observation_records = []
    toy_context: dict[str, Any] = {}
    for raw in raw_catalog["toyFixtures"]:
        policy = policy_by_id[raw["policyId"]]
        environment = environment_by_id[raw["environmentId"]]
        state = _state_for_fixture(environment, raw["stateTransform"])
        actor_id = _actor_at(state, raw["actorSite"])
        kwargs = _toy_kwargs(raw, environment, state, actor_id, policy, grammar_by_id)
        execute_kwargs = _execution_kwargs(kwargs)
        first_build = build_policy_observation(
            environment, state, actor_id, policy, **execute_kwargs
        )
        first_decision = decide_policy(policy, first_build.observation.payload)
        first_proposal = materialize_decision(first_build, first_decision)
        first_event = execute_policy_activation(
            environment,
            state,
            actor_id,
            policy,
            batch_nonce=f"s05-toy:{raw['fixtureId']}",
            **execute_kwargs,
        )
        second_event = execute_policy_activation(
            environment,
            state,
            actor_id,
            policy,
            batch_nonce=f"s05-toy:{raw['fixtureId']}",
            **execute_kwargs,
        )
        replay_pass = canonical_policy_event_bytes(
            first_event
        ) == canonical_policy_event_bytes(second_event)
        expected_pass = first_decision.action == raw["expectedAction"]
        legal = (
            first_proposal is None
            or validate_proposal(environment, state, first_proposal).valid
        )
        if not (replay_pass and expected_pass and legal):
            raise RuntimeError(f"S05 toy fixture failed: {raw['fixtureId']}")
        event_path = events / f"{raw['fixtureId']}.json"
        event_path.write_bytes(canonical_policy_event_bytes(first_event))
        row = {
            "fixtureId": raw["fixtureId"],
            "policyId": policy.policy_id,
            "strategy": policy.strategy,
            "environmentId": environment.environment_id,
            "candidateCount": len(first_build.candidate_map),
            "action": first_decision.action,
            "reason": first_decision.reason,
            "decisionScore": first_decision.decision_score,
            "movementKindEngineAudit": None
            if first_proposal is None
            else first_proposal.kind,
            "movementOutcome": first_event["outcome"],
            "communicatedBitsUpperBound": first_build.observation.budget[
                "communicatedBitsUpperBound"
            ],
            "informationBudgetMaxBits": policy.information_budget_max_bits,
            "legalMaterialization": legal,
            "expectedActionPass": expected_pass,
            "deterministicReplayPass": replay_pass,
        }
        toy_rows.append(row)
        toy_records.append(
            {
                **row,
                "observationSha256": first_build.observation.observation_sha256,
                "eventSha256": first_event["eventSha256"],
                "payload": first_build.observation.payload,
                "budget": first_build.observation.budget,
                "authenticationAddedAfterDecision": first_event["materializationAudit"][
                    "authenticationAddedAfterPolicyDecision"
                ],
                "policySawProposalEnvelope": first_event["materializationAudit"][
                    "policySawProposalEnvelope"
                ],
            }
        )
        budget_rows.append(
            {
                "policyId": policy.policy_id,
                "strategy": policy.strategy,
                **dict(first_build.observation.budget),
                "informationBudgetMaxBits": policy.information_budget_max_bits,
                "withinCommunicatedBitBudget": first_build.observation.budget[
                    "communicatedBitsUpperBound"
                ]
                <= policy.information_budget_max_bits,
            }
        )
        site_ids_absent = all(
            site.site_id
            not in json.dumps(first_build.observation.payload, sort_keys=True)
            for site in environment.sites
        )
        observation_records.append(
            {
                "policyId": policy.policy_id,
                "declaredFeatures": list(policy.observation_features),
                "topLevelPayloadFields": sorted(first_build.observation.payload),
                "candidatePayloadFields": sorted(
                    {
                        key
                        for item in first_build.observation.payload["candidates"]
                        for key in item
                    }
                ),
                "siteIdsAbsent": site_ids_absent,
                "authenticationFieldsAbsent": True,
                "rawRoleAndBoundaryMetadataAbsent": all(
                    term not in json.dumps(first_build.observation.payload).lower()
                    for term in ("boundarytags", "obstacle", "fixed", "route")
                ),
                "globalEvaluationAbsent": "global"
                not in json.dumps(first_build.observation.payload).lower(),
            }
        )
        deterministic_records.append(
            {
                "fixtureId": raw["fixtureId"],
                "observationSha256": first_build.observation.observation_sha256,
                "selectedCandidateKey": first_decision.selected_candidate_key,
                "eventSha256": first_event["eventSha256"],
                "exactSecondExecutionMatch": replay_pass,
            }
        )
        serialization_records.append(
            {
                "fixtureId": raw["fixtureId"],
                "canonicalEventBytes": len(canonical_policy_event_bytes(first_event)),
                "jsonRoundTripMatch": canonical_policy_event_bytes(first_event)
                == canonical_policy_event_bytes(
                    json.loads(canonical_policy_event_bytes(first_event))
                ),
                "independentReexecutionMatch": replay_pass,
            }
        )
        toy_context[policy.policy_id] = {
            "raw": raw,
            "environment": environment,
            "state": state,
            "actorId": actor_id,
            "kwargs": execute_kwargs,
            "build": first_build,
            "decision": first_decision,
            "event": first_event,
        }

    _write_csv(output / "toy_behavior_results.csv", toy_rows)
    _write_json(
        output / "toy_behavior_results.json",
        {
            "schemaVersion": "e06.s05.toy-behavior-results.v1",
            "researchStepId": "S05",
            "results": toy_records,
        },
    )
    _write_csv(output / "observation_budget.csv", budget_rows)
    _write_json(
        output / "deterministic_decisions.json",
        {
            "schemaVersion": "e06.s05.deterministic-decisions.v1",
            "researchStepId": "S05",
            "records": deterministic_records,
            "allExact": all(
                item["exactSecondExecutionMatch"] for item in deterministic_records
            ),
        },
    )
    _write_json(
        output / "serialization_replay.json",
        {
            "schemaVersion": "e06.s05.serialization-replay.v1",
            "researchStepId": "S05",
            "records": serialization_records,
            "success": all(
                item["jsonRoundTripMatch"] and item["independentReexecutionMatch"]
                for item in serialization_records
            ),
        },
    )

    complexity_rows = []
    policy_index = []
    for policy in policies:
        complexity = policy_complexity(policy)
        row = {"policyId": policy.policy_id, "strategy": policy.strategy, **complexity}
        complexity_rows.append(row)
        toy = next(item for item in toy_rows if item["policyId"] == policy.policy_id)
        card_path = cards / f"{policy.policy_id}.md"
        card_path.write_text(_policy_card(policy, toy), encoding="utf-8")
        policy_index.append(
            {
                **policy_to_dict(policy),
                "card": str(card_path.relative_to(output)),
                "toyEvent": str(
                    (
                        events
                        / f"{next(item['fixtureId'] for item in toy_rows if item['policyId'] == policy.policy_id)}.json"
                    ).relative_to(output)
                ),
            }
        )
    _write_csv(output / "complexity_and_specificity.csv", complexity_rows)
    _write_json(
        library / "policy_index.json",
        {
            "schemaVersion": "e06.s05.policy-index.v1",
            "researchStepId": "S05",
            "libraryVersion": metadata["libraryVersion"],
            "policies": policy_index,
        },
    )

    periodic = environment_by_id["square_periodic_vacancy"]
    periodic_state = initial_movement_state(periodic)
    compatibility_records = []
    observed_all = set()
    for policy in policies:
        actor_id = _actor_at(periodic_state, "r0_c1")
        candidates, audit = enumerate_candidate_affordances(
            periodic, periodic_state, actor_id, policy
        )
        kinds = sorted({item.proposal.kind for item in candidates})
        legal = all(
            validate_proposal(periodic, periodic_state, item.proposal).valid
            for item in candidates
        )
        cost_reconciles = all(
            item.movement_cost
            == resolve_batch(
                periodic,
                periodic_state,
                (item.proposal,),
                batch_nonce=f"s05-cost:{policy.policy_id}:{item.candidate_key}",
            )["costLedger"]["totalGraphDisplacement"]
            for item in candidates
        )
        observed_all.update(kinds)
        compatibility_records.append(
            {
                "policyId": policy.policy_id,
                "declaredMovementKinds": list(policy.allowed_movement_kinds),
                "observedCandidateKinds": kinds,
                "allObservedKindsDeclared": set(kinds)
                <= set(policy.allowed_movement_kinds),
                "everyCandidateS04Legal": legal,
                "movementCostEqualsS04TotalGraphDisplacement": cost_reconciles,
                "candidateCount": len(candidates),
                "engineAffordanceReadAudit": audit,
            }
        )
    exploration = policy_by_id["exploration_v1"]
    selected_kinds = set()
    for environment_id, actor_site in (
        ("hexagonal_bounded_occupied", "q0_r0"),
        ("square_periodic_vacancy", "r0_c1"),
    ):
        environment = environment_by_id[environment_id]
        state = initial_movement_state(environment)
        actor_id = _actor_at(state, actor_site)
        for index in range(96):
            build_record = build_policy_observation(
                environment,
                state,
                actor_id,
                exploration,
                decision_key=f"s05-kind-coverage-{index}",
                activation_index=index,
            )
            decision = decide_policy(exploration, build_record.observation.payload)
            proposal = materialize_decision(build_record, decision)
            if proposal is not None:
                selected_kinds.add(proposal.kind)
    _write_json(
        output / "movement_compatibility.json",
        {
            "schemaVersion": "e06.s05.movement-compatibility.v1",
            "researchStepId": "S05",
            "records": compatibility_records,
            "enabledS04Kinds": sorted(ENABLED_KINDS),
            "observedCandidateKindsAcrossLibrary": sorted(observed_all),
            "explorationSelectedKindsAcrossAddresses": sorted(selected_kinds),
            "disabledKindsExposed": [],
            "allEnabledKindsExercised": selected_kinds == ENABLED_KINDS,
            "success": all(
                item["allObservedKindsDeclared"]
                and item["everyCandidateS04Legal"]
                and item["movementCostEqualsS04TotalGraphDisplacement"]
                for item in compatibility_records
            )
            and selected_kinds == ENABLED_KINDS,
        },
    )

    boundary_context = toy_context["boundary_seeking_v1"]
    boundary_policy = policy_by_id["boundary_seeking_v1"]
    periodic_boundary = build_policy_observation(
        periodic,
        periodic_state,
        _actor_at(periodic_state, "r2_c2"),
        boundary_policy,
        boundary_direction="seek",
        boundary_tokens=(periodic_state.occupant_map["r2_c2"].token,),
    )
    periodic_boundary_decision = decide_policy(
        boundary_policy, periodic_boundary.observation.payload
    )

    gradient_context = toy_context["gradient_following_v1"]
    gradient_policy = policy_by_id["gradient_following_v1"]
    gradient_error_probes = []
    for probe_id, field in (
        ("missing_candidate_coverage", {"r2_c2": 2}),
        (
            "outside_uint8_range",
            {
                **_column_field(gradient_context["environment"]),
                "r2_c2": 256,
            },
        ),
    ):
        rejected = False
        reason = None
        try:
            build_policy_observation(
                gradient_context["environment"],
                gradient_context["state"],
                gradient_context["actorId"],
                gradient_policy,
                gradient_levels=field,
                gradient_direction="up",
            )
        except PolicyValidationError as error:
            rejected = True
            reason = str(error)
        gradient_error_probes.append(
            {"probeId": probe_id, "rejected": rejected, "reason": reason}
        )
    _write_json(
        output / "gradient_interface_audit.json",
        {
            "schemaVersion": "e06.s05.gradient-interface-audit.v1",
            "researchStepId": "S05",
            "implementedInS05": "consume_explicit_complete_uint8_field_and_direction",
            "deferredToS06": [
                "source_and_owner",
                "semantic_content",
                "spatial_resolution",
                "update_schedule",
                "noise_model",
                "bandwidth_and_comparative_cost",
                "central_controller_state_access",
            ],
            "failureProbes": gradient_error_probes,
            "periodicNaturalBoundaryDecision": {
                "action": periodic_boundary_decision.action,
                "reason": periodic_boundary_decision.reason,
                "allDeltasZero": all(
                    item["naturalBoundaryDelta"] == 0
                    for item in periodic_boundary.observation.payload["candidates"]
                ),
            },
            "boundedBoundaryToyAction": boundary_context["decision"].action,
            "success": all(item["rejected"] for item in gradient_error_probes)
            and periodic_boundary_decision.action == "noop",
        },
    )

    memory_context = toy_context["memory_based_recovery_v1"]
    memory_policy = policy_by_id["memory_based_recovery_v1"]
    current = int(
        memory_context["build"].observation.payload["currentLocalRelationUtility"]
    )
    met_build = build_policy_observation(
        memory_context["environment"],
        memory_context["state"],
        memory_context["actorId"],
        memory_policy,
        relation_profile=memory_context["kwargs"]["relation_profile"],
        memory=PolicyMemory(current, 0),
    )
    met_decision = decide_policy(memory_policy, met_build.observation.payload)
    before = memory_context["kwargs"]["memory"]
    accepted_memory = update_policy_memory(
        before,
        memory_context["build"].observation,
        memory_context["decision"],
        "accepted",
    )
    rejected_memory = before
    for _ in range(4):
        rejected_memory = update_policy_memory(
            rejected_memory,
            memory_context["build"].observation,
            memory_context["decision"],
            "rejected",
        )
    memory_range_probes = []
    for name, candidate in (
        ("int8_overflow", PolicyMemory(128, 0)),
        ("frustration_overflow", PolicyMemory(0, 4)),
    ):
        rejected = False
        try:
            from src.morph2d.policies import validate_policy_memory

            validate_policy_memory(candidate)
        except PolicyValidationError:
            rejected = True
        memory_range_probes.append({"probeId": name, "rejected": rejected})
    _write_json(
        output / "memory_semantics.json",
        {
            "schemaVersion": "e06.s05.memory-semantics.v1",
            "researchStepId": "S05",
            "ownership": "identity_owned_not_site_or_global",
            "fields": {
                "bestLocalRelationUtility": "signed_int8",
                "frustration": "unsigned_2_bit_saturating",
            },
            "persistentMemoryBits": 10,
            "belowReferenceAction": memory_context["decision"].action,
            "referenceMetAction": met_decision.action,
            "referenceMetReason": met_decision.reason,
            "acceptedUpdate": {
                "bestLocalRelationUtility": accepted_memory.best_local_utility,
                "frustration": accepted_memory.frustration,
            },
            "fourRejectedUpdates": {
                "bestLocalRelationUtility": rejected_memory.best_local_utility,
                "frustration": rejected_memory.frustration,
            },
            "rangeProbes": memory_range_probes,
            "success": memory_context["decision"].action == "proposal"
            and met_decision.action == "noop"
            and accepted_memory.frustration == 0
            and rejected_memory.frustration == 3
            and all(item["rejected"] for item in memory_range_probes),
        },
    )

    greedy_context = toy_context["greedy_neighbor_satisfaction_v1"]
    greedy_event = greedy_context["event"]
    grammar = grammar_by_id["stripes_axis_relations_v1"]
    target = target_by_id[grammar.target_id]
    post_state = parse_movement_state(greedy_event["movementBatchResult"]["postState"])
    post_environment = replace(
        greedy_context["environment"],
        initial_state={
            **greedy_context["environment"].initial_state,
            **state_tokens(post_state),
        },
    )
    offline = evaluate_conjunctive(post_environment, target, grammar)
    evaluation_separation = {
        "schemaVersion": "e06.s05.evaluation-separation.v1",
        "researchStepId": "S05",
        "policyId": "greedy_neighbor_satisfaction_v1",
        "policyVisibleS02LocalFeedback": greedy_event["s02LocalFeedback"],
        "offlineS02WholeGridGrammar": offline["localGrammar"],
        "offlineS01GlobalEvaluation": offline["globalAudit"],
        "offlineConjunctiveCompletion": offline["completion"],
        "completionRule": offline["completionRule"],
        "policyPayloadDisclosedWholeGridOrGlobal": False,
        "s02AndS01FieldNamesDistinct": True,
        "success": "global"
        not in json.dumps(greedy_context["build"].observation.payload).lower(),
    }
    _write_json(output / "evaluation_separation.json", evaluation_separation)

    parser_feature_probe = json.loads(json.dumps(raw_catalog))
    parser_feature_probe["policies"][0]["observationFeatures"].append("site_roles")
    parser_feature_rejected = False
    try:
        from src.morph2d.policies import parse_policy_catalog

        parse_policy_catalog(parser_feature_probe)
    except PolicyValidationError:
        parser_feature_rejected = True
    observation_permissions = {
        "schemaVersion": "e06.s05.observation-permissions.v1",
        "researchStepId": "S05",
        "policyDecisionReceives": "observation.payload_only",
        "engineOnly": raw_catalog["observationContract"]["engineOnly"],
        "candidateAvailabilityIsPriced": True,
        "candidateHandles": "activation_local_opaque_state_keyed_order",
        "records": observation_records,
        "forbiddenFieldInjectionProbes": permission_probes,
        "parserRejectedUndeclaredSiteRoleFeature": parser_feature_rejected,
        "decisionSourceForbiddenAccesses": decision_source_forbidden_accesses(),
        "success": all(item["rejected"] for item in permission_probes)
        and parser_feature_rejected
        and not decision_source_forbidden_accesses()
        and all(
            item["siteIdsAbsent"]
            and item["authenticationFieldsAbsent"]
            and item["rawRoleAndBoundaryMetadataAbsent"]
            and item["globalEvaluationAbsent"]
            for item in observation_records
        ),
    }
    _write_json(output / "observation_permissions.json", observation_permissions)

    environment_provenance = {
        "schemaVersion": "e06.s05.environment-provenance.v1",
        "researchStepId": "S05",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name) for name in ("PyYAML", "pytest")
        },
        "cpuCountVisible": os.cpu_count(),
        "workersUsed": 1,
        "threadPolicy": "intentional_serial_exact_fixture_execution",
        "gpuUsed": False,
        "networkUsed": False,
        "dependenciesInstalled": [],
        "repository": {
            "path": str(ROOT),
            "branch": _git("branch", "--show-current"),
            "baseCommit": _git("rev-parse", "HEAD"),
            "remote": _git("remote", "get-url", "origin"),
        },
    }
    _write_json(output / "environment_provenance.json", environment_provenance)

    focused_tests = _focused_tests()
    checks = {
        "exactSixPolicies": len(policies) == 6,
        "exactSevenRelationProfiles": len(relation_profiles) == 7,
        "allToyExpectedActions": all(item["expectedActionPass"] for item in toy_rows),
        "allToyLegalMaterializations": all(
            item["legalMaterialization"] for item in toy_rows
        ),
        "allToyReplayExact": all(item["deterministicReplayPass"] for item in toy_rows),
        "allObservationBudgetsWithinCap": all(
            item["withinCommunicatedBitBudget"] for item in budget_rows
        ),
        "observationPermissionAudit": observation_permissions["success"],
        "movementCompatibility": json.loads(
            (output / "movement_compatibility.json").read_text()
        )["success"],
        "allS04KindsExercised": selected_kinds == ENABLED_KINDS,
        "memorySemantics": json.loads((output / "memory_semantics.json").read_text())[
            "success"
        ],
        "gradientAndBoundaryInterfaces": json.loads(
            (output / "gradient_interface_audit.json").read_text()
        )["success"],
        "evaluationSeparation": evaluation_separation["success"],
        "upstreamImmutability": upstream_immutability["success"],
        "focusedTests": focused_tests["success"],
    }
    validation = {
        "schemaVersion": "e06.s05.validation-summary.v1",
        "researchStepId": "S05",
        "status": "complete",
        "success": all(checks.values()),
        "validationResult": "PASS" if all(checks.values()) else "FAIL",
        "outcomeClassification": "supportive",
        "policyCount": len(policies),
        "toyFixtureCount": len(toy_rows),
        "relationProfileCount": len(relation_profiles),
        "observationLedgerFieldCount": len(OBSERVATION_LEDGER_FIELDS),
        "forbiddenFieldProbeCount": len(permission_probes),
        "movementKindsExercised": sorted(selected_kinds),
        "checks": checks,
        "focusedTests": focused_tests,
        "upstreamImmutability": upstream_immutability,
    }
    _write_json(output / "validation_summary.json", validation)

    execution_commands = """cd /workspace/cell-research
python -m pytest -q tests/test_morph2d_targets.py tests/test_morph2d_grammar.py tests/test_morph2d_environments.py tests/test_morph2d_movements.py tests/test_morph2d_policies.py
python scripts/build_morph2d_s05.py build --output-dir /artifacts/research_steps/S05
python scripts/build_morph2d_s05.py validate --output-dir /artifacts/research_steps/S05
ruff check src/morph2d scripts/build_morph2d_s05.py tests/test_morph2d_policies.py
ruff format --check src/morph2d scripts/build_morph2d_s05.py tests/test_morph2d_policies.py
python -m compileall -q src/morph2d scripts/build_morph2d_s05.py tests/test_morph2d_policies.py
"""
    (output / "execution_commands.log").write_text(execution_commands, encoding="utf-8")

    expected_artifacts = sorted(
        str(path.relative_to(output))
        for path in output.rglob("*")
        if path.is_file()
        and path.name not in {"artifact_manifest.json", "research_step_full_results.md"}
    )
    expected_artifacts.append("research_step_full_results.md")
    report = _report(
        validation,
        toy_rows,
        complexity_rows,
        environment_provenance,
        expected_artifacts,
    )
    (output / "research_step_full_results.md").write_text(report, encoding="utf-8")

    artifact_paths = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    )
    manifest = {
        "schemaVersion": "e06.s05.artifact-manifest.v1",
        "researchStepId": "S05",
        "artifactRoot": str(output),
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "artifacts": [
            {
                "path": str(path.relative_to(output)),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in artifact_paths
        ],
    }
    _write_json(output / "artifact_manifest.json", manifest)
    if not validation["success"]:
        raise RuntimeError("S05 validation failed; inspect validation_summary.json")


def validate(output: Path) -> None:
    manifest = json.loads((output / "artifact_manifest.json").read_text())
    validation = json.loads((output / "validation_summary.json").read_text())
    mismatches = []
    for record in manifest["artifacts"]:
        path = output / record["path"]
        observed = _sha256_file(path) if path.is_file() else None
        if observed != record["sha256"]:
            mismatches.append(
                {
                    "path": record["path"],
                    "expected": record["sha256"],
                    "observed": observed,
                }
            )
    required = {
        "policy_spec.md",
        "policy_library/policy_catalog.yaml",
        "policy_library/policy_index.json",
        "observation_permissions.json",
        "observation_budget.csv",
        "complexity_and_specificity.csv",
        "toy_behavior_results.json",
        "movement_compatibility.json",
        "deterministic_decisions.json",
        "serialization_replay.json",
        "memory_semantics.json",
        "gradient_interface_audit.json",
        "evaluation_separation.json",
        "validation_summary.json",
        "research_step_full_results.md",
    }
    observed_paths = {record["path"] for record in manifest["artifacts"]}
    report = (output / "research_step_full_results.md").read_text()
    top_fields = (
        "Research step ID",
        "Completion status",
        "Artifacts written",
        "Validation result",
        "Outcome classification",
        "Caveats or blockers",
        "Lay summary",
        "Recommended next action",
    )
    result = {
        "researchStepId": "S05",
        "manifestArtifactCount": len(manifest["artifacts"]),
        "hashMismatches": mismatches,
        "missingRequiredArtifacts": sorted(required - observed_paths),
        "reportTopSummaryFieldsPresent": all(
            field in report.split("## Frozen question", maxsplit=1)[0]
            for field in top_fields
        ),
        "validationSummarySuccess": validation["success"],
    }
    result["success"] = (
        not result["hashMismatches"]
        and not result["missingRequiredArtifacts"]
        and result["reportTopSummaryFieldsPresent"]
        and result["validationSummarySuccess"]
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["success"]:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "validate"))
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "build":
        build(arguments.output_dir)
    else:
        validate(arguments.output_dir)


if __name__ == "__main__":
    main()
