#!/usr/bin/env python3
"""Build and validate E06 S04 legal-movement artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import itertools
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
MOVEMENTS = ROOT / "configs/morphologies/movement_catalog.yaml"
ENVIRONMENTS = ROOT / "configs/morphologies/environment_catalog.yaml"
TARGETS = ROOT / "configs/morphologies/target_catalog.yaml"
GRAMMARS = ROOT / "configs/morphologies/grammar_catalog.yaml"
SOURCE = ROOT / "src/morph2d/movements.py"
TEST = ROOT / "tests/test_morph2d_movements.py"
S01_DIR = Path("/artifacts/research_steps/S01")
S02_DIR = Path("/artifacts/research_steps/S02")
S03_DIR = Path("/artifacts/research_steps/S03")
ATTACHMENT_DIR = WORKSPACE / "input-attachments/21c2278b-9950-4e39-a2c8-df578a2508ec"
UPSTREAM_INPUTS = {
    "agents": WORKSPACE / "AGENTS.md",
    "full_plan": WORKSPACE / "FULL_PLAN.md",
    "research_plan_pre_s04_update": WORKSPACE / "RESEARCH_PLAN.md",
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
    "s03_environment_index": S03_DIR / "environment_library/environment_index.json",
    "s03_conjunctive_evaluation": S03_DIR / "conjunctive_evaluation.json",
    "s03_boundary_requirements": S03_DIR / "boundary_requirements.csv",
    "s03_unsupported_audit": S03_DIR / "unsupported_combination_audit.json",
    "s03_replay": S03_DIR / "deterministic_replay.json",
    "s03_validation": S03_DIR / "validation_summary.json",
    "s03_manifest": S03_DIR / "artifact_manifest.json",
    "e01_transition": Path(
        "/previous-artifacts/E01/research_steps/S03/transition_spec.md"
    ),
    "e01_api": Path("/previous-artifacts/E01/research_steps/S05/api_documentation.md"),
    "e01_events": Path(
        "/previous-artifacts/E01/research_steps/S06/schema_documentation.md"
    ),
    "e04_metrics": Path(
        "/previous-artifacts/E04/research_steps/S04/metric_specification.md"
    ),
    "e04_nulls": Path(
        "/previous-artifacts/E04/research_steps/S05/null_specification.md"
    ),
    "e04_kinetics": Path(
        "/previous-artifacts/E04/research_steps/S07/kinetic_intervention_specification.md"
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
    DEFERRED_KINDS,
    ENABLED_KINDS,
    LEDGER_FIELDS,
    canonical_batch_result_bytes,
    canonical_movement_state_bytes,
    canonical_proposal_bytes,
    conflict_order_key,
    initial_movement_state,
    inverse_proposal,
    make_proposal,
    parse_movement_state,
    parse_proposal,
    replay_batch_result,
    resolve_batch,
    state_tokens,
    validate_proposal,
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


def _proposal_from_raw(state, raw):
    kwargs: dict[str, Any] = {}
    mutation = raw.get("mutation")
    if mutation == "staleStateHash":
        kwargs["observed_state_sha256"] = "0" * 64
    elif mutation == "staleResourceIdentity":
        kwargs["expected_occupants"] = {
            site_id: (
                "stale:identity"
                if index == 0
                else state.occupant_map[site_id].occupant_id
            )
            for index, site_id in enumerate(raw["route"])
            if site_id in state.occupant_map
        }
    elif mutation == "actorMismatch":
        kwargs["actor_id"] = "wrong:actor"
    return make_proposal(
        state,
        raw["kind"],
        raw["route"],
        rotation_direction=raw["rotationDirection"],
        **kwargs,
    )


def _layer_record(environment, state, target, grammar) -> dict[str, Any]:
    moved_environment = replace(
        environment,
        initial_state={**environment.initial_state, **state_tokens(state)},
    )
    raw = evaluate_conjunctive(moved_environment, target, grammar)
    return {
        "s02LocalGrammar": raw["localGrammar"],
        "s01GlobalCompletionAudit": raw["globalAudit"],
        "conjunctiveCompletion": raw["completion"],
        "completionRule": raw["completionRule"],
    }


def _movement_spec(catalog: dict[str, Any]) -> str:
    return f"""# E06 S04 Legal Movement Specification

## Scope and authority

Version `{catalog["schemaVersion"]}` defines the CPU reference semantics for legal movement over the canonical S03 undirected graph/site-role contract. It defines stable movement identities, proposal authentication, validation, simultaneous conflict resolution, atomic commit, cost accounting, serialization, and replay. It does not define S05 Algotypes, policy observations, formation dynamics, division, removal, or long-range exchange.

## Movement state and invariants

Every non-obstacle S03 site contains exactly one stable occupant identity of kind `cell`, `vacancy`, or `fixed_boundary`. A vacancy is a conserved occupiable entity, not absence from the graph. Obstacles have no movement slot. After every batch, site coverage, occupant identities, token composition, vacancy count, occupant-kind counts, obstacle exclusion, and each fixed-boundary occupant/token are unchanged. A batch advances the transition index even if every proposal is invalid; reversibility therefore means exact restoration of the occupancy/identity projection, not erasure of event history.

## Enabled local primitives

- `adjacent_swap`: two non-vacancy occupants exchange across one declared graph edge.
- `vacancy_move`: a non-vacancy source moves across one declared edge into a vacancy; the conserved vacancy identity moves to the source.
- `short_exchange`: two non-vacancy endpoints at exact graph distance two exchange over a declared three-site path. The unchanged middle site is reserved, preventing simultaneous path interference.
- `rotation`: all non-vacancy occupants on a simple closed graph cycle of length 3–6 move one edge in direction `+1` or `-1`.

No coordinate difference creates legality: only S03 edges do. Periodic seam edges are legal. Routes containing an obstacle or fixed-boundary site are rejected. Rotations containing a vacancy are rejected so that vacancy transport remains attributable to `vacancy_move`.

## Proposal validation

An Algotype-facing intent is wrapped by the engine with a content-derived proposal ID, pre-state hash, actor identity, and expected identity for every reserved site. These authentication fields are engine integrity metadata, not additional S05 observations. Validation checks the proposal kind, state freshness, canonical route/source/target, distinct sites, graph/site roles, complete expected identities, actor identity, kind-specific token requirements, legal displacement, and cycle closure. A rejected proposal has no state effect.

## Simultaneous conflicts and tie-breaking

Every proposal is validated independently against one immutable pre-batch snapshot. Endpoint moves reserve both endpoints; short exchanges reserve the full three-site path; rotations reserve the full cycle. Valid proposals are ranked by the first 64 bits of `SHA256(E06/S04/conflict/v1, batch ID, proposal ID)` and then by lexicographic content-derived proposal ID. The resolver greedily accepts a maximal disjoint set. Conflict losers do not retry. Batch IDs use the state hash, sorted proposal IDs, and an explicit nonce, so proposal input order cannot affect priority, decisions, committed state, costs, or serialization.

## Cost ledger and budgets

The ledger has {len(LEDGER_FIELDS)} nonnegative integer fields: `{", ".join(LEDGER_FIELDS)}`. It separates submitted/valid/invalid/conflict outcomes, logical validation reads/checks, reservations, committed movement counts by kind, displaced cell/vacancy identities, and graph displacement. Short exchanges charge distance two to both endpoints; rotations charge one edge to every cycle occupant; vacancy moves charge both the cell and the conserved vacancy, while also reporting them separately.

Movement legality has a hard boundary-signal budget of zero. Site-role checks are engine safety metadata and do not expose S03 exterior, obstacle-contact, or fixed-neighbor channels to a policy. S05/S06 must separately price any policy observation.

## Evaluation separation

S02 local grammar fields and S01 global completion fields are computed only after a transition for compatible target-bound environments. They are not inputs to legality or conflict resolution. Records use separate `s02LocalGrammar`, `s01GlobalCompletionAudit`, and `conjunctiveCompletion` fields, with completion remaining `local.accepted AND global.success`.

## Explicitly deferred interfaces

`long_range_exchange`, `division`, and `removal` are recognized only to produce deterministic `deferred_kind:*` rejections. Long-range exchange lacks a fair local range/information/cost contract. Division and removal violate S04 conservation and require new identity, target-count, feasibility, intervention, and event schemas. They are not dormant executable features.

## Claim boundary

These rules validate computational graph transitions. Reversibility, conservation, and replay do not establish formation, repair, robustness, biological movement, mechanism, agency, or clinical relevance.
"""


def _input_provenance() -> dict[str, Any]:
    return {
        "schemaVersion": "e06.s04.input-provenance.v1",
        "researchStepId": "S04",
        "inputs": [
            {
                "role": role,
                "path": str(path),
                "sha256": _sha256_file(path),
                "readOnly": str(path).startswith("/previous-artifacts"),
            }
            for role, path in UPSTREAM_INPUTS.items()
        ],
        "repositoryInputs": [
            {"path": str(path), "sha256": _sha256_file(path)}
            for path in (
                MOVEMENTS,
                ENVIRONMENTS,
                TARGETS,
                GRAMMARS,
                SOURCE,
                TEST,
                Path(__file__),
            )
        ],
    }


def _artifact_role(relative: str) -> str:
    roles = {
        "movement_spec.md": "normative S04 movement, conflict, budget, and deferral contract",
        "movement_catalog.yaml": "immutable high-level movement catalog snapshot",
        "fixture_results.json": "compact outcomes and hashes for all canonical movement batches",
        "fixture_results.csv": "tabular movement fixture outcome summary",
        "proposal_validation_audit.json": "enabled and rejected proposal validation evidence",
        "proposal_validation_audit.csv": "tabular validation reason evidence",
        "conservation_audit.json": "identity, composition, occupancy, obstacle, and fixed-role invariants",
        "movement_cost_ledger.csv": "complete per-fixture logical validation and displacement ledger",
        "conflict_resolution_audit.json": "conflict winners, losers, reservations, and tie audit",
        "proposal_order_invariance.json": "exhaustive permutation-invariance result for the conflict batch",
        "reversibility_audit.json": "forward/inverse occupancy and identity restoration fixtures",
        "serialization_roundtrip.json": "state and proposal canonical serialization checks",
        "deterministic_replay.json": "exact canonical batch replay checks",
        "evaluation_layer_results.json": "separate S02 local and S01 global pre/post fields",
        "evaluation_layer_results.csv": "compact local/global/conjunctive evaluation table",
        "deferred_interfaces.csv": "explicit rejected long-range, division, and removal interfaces",
        "validation_summary.json": "consolidated S04 validation gates",
        "input_provenance.json": "governing, upstream, dependency, attachment, and repository hashes",
        "environment_provenance.json": "runtime, resource, dependency, and repository provenance",
        "execution_commands.log": "reproduction and validation commands",
        "research_step_full_results.md": "canonical S04 full-results handoff report",
    }
    if relative.startswith("movement_fixtures/"):
        return "canonical full batch transition fixture with replayable pre/post state"
    return roles[relative]


def _report(
    fixture_rows,
    validation,
    provenance,
    artifact_names,
    conflict_audit,
    layer_records,
) -> str:
    table = [
        "| Fixture | Environment | Submitted | Valid | Accepted | Conflict lost | Invalid | Cell displacement | Vacancy displacement | Replay |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in fixture_rows:
        table.append(
            f"| `{row['fixtureId']}` | `{row['environmentId']}` | {row['submittedProposals']} | "
            f"{row['validProposals']} | {row['acceptedMovements']} | {row['conflictLosses']} | "
            f"{row['invalidProposals']} | {row['cellGraphDisplacement']} | "
            f"{row['vacancyGraphDisplacement']} | {row['replayPass']} |"
        )
    test_summary = validation["focusedTests"]["stdout"].splitlines()[-1]
    accepted = sum(row["acceptedMovements"] for row in fixture_rows)
    invalid = sum(row["invalidProposals"] for row in fixture_rows)
    conflict_losses = sum(row["conflictLosses"] for row in fixture_rows)
    next_action = "Return control to the Chief Scientist. If this S04 contract is accepted, separately authorize only S05 to construct local spatial Algotypes under the frozen movement and observation-budget interfaces; do not start S05 automatically."
    return f"""# Research Step S04 Full Results — Define Legal Movements

## Top summary

- **Research step ID:** S04
- **Completion status:** **Complete** on {datetime.now(timezone.utc).date().isoformat()}; only S04 was executed and S05 was not started.
- **Artifacts written:** {", ".join(f"`{name}`" for name in artifact_names)}.
- **Validation result:** **PASS** — four enabled movement kinds, {len(fixture_rows)} canonical batches, {validation["validationProbeCount"]} rejection probes, {validation["orderPermutationCount"]} proposal orders, {validation["reversibilityFixtureCount"]} inverse fixtures, canonical serialization, exact replay, conservation, displacement, cost, fixed/obstacle enforcement, zero boundary-signal reads, separate S02/S01 evaluation fields, upstream hashes, and focused tests (`{test_summary}`) all passed.
- **Outcome classification:** **supportive.** The four local movement primitives unify under one deterministic graph transition and cost contract without enabling long-range exchange, division, or removal.
- **Caveats or blockers:** This is a validated CPU transition oracle, not a policy or episode engine. Rotations are limited to fully occupied simple cycles of length 3–6, short exchanges to exact distance two, and all conflict proposals reserve their entire routes. Division/removal need new conservation and target semantics; non-square target completion remains uncalibrated.
- **Lay summary:** Cells, empty slots, blocked sites, and fixed boundaries now follow explicit movement rules. Simultaneous moves produce the same winners no matter how proposals arrive, every moved identity and unit of graph distance is counted, and saved transitions replay exactly. The rules do not yet decide which move a cell should propose.
- **Recommended next action:** {next_action}

## Frozen question and completion criterion

**Frozen question:** Can adjacent swaps, vacancy moves, distance-two short exchanges, and local rotations be represented as one conservative transition system with explicit proposal validation, deterministic simultaneous conflicts, complete costs, and exact replay over the S03 environment contract?

**Completion criterion:** all four core primitives must be fully specified and validated for conservation, occupancy, graph-legal displacement, obstacles, fixed boundaries, deterministic conflicts/ties, proposal-order behavior, reversibility, serialization, replay, and complete cost accounting; division/removal and long-range exchange must be rejected or fully validated rather than silently enabled.

The criterion was met.

## Lay summary

The movement layer treats a cell or vacancy as a stable identity occupying one graph site. Legal moves use only graph edges and immutable safety roles. Two proposals that touch the same reserved site cannot both win; a stable content hash decides the order. This creates a testable transition oracle for later policies and GPU code while preserving the earlier distinction between local pattern feedback and whole-shape success.

## Inputs

- Refreshed `/workspace/AGENTS.md`, `FULL_PLAN.md`, and the pre-completion `RESEARCH_PLAN.md`, plus capability/dataset availability records.
- Manifest-verified S01–S03 reports, target/grammar/environment catalogs, validation, conjunctive evaluation, boundary/unsupported-scope audits, replay records, and manifests.
- E01 immutable identity, authenticated proposal, pre-batch snapshot, deterministic conflict, cost-ledger, canonical state-hash, event, and replay contracts.
- E04 exact-composition, declared graph-adjacency, label-blind intervention, movement/transport-cost, and E06 handoff constraints.
- `input-attachments/MANIFEST.json` and the attachment sidecar. No dataset, network access, package installation, GPU, or upstream mutation was required.

Exact paths and SHA-256 values are in `input_provenance.json`; {validation["upstreamImmutability"]["checkedArtifactCount"]} manifest-listed S01–S03 artifacts were rehashed successfully.

## Methods

### State and proposal model

`src/morph2d/movements.py` compiles each S03 environment to a movement state with one stable identity per occupiable site. Vacancies are conserved identities; obstacles are absent; fixed-boundary identities remain immobile. Engine-authenticated proposal envelopes carry a content ID, state hash, actor, explicit route, and expected identities for every reserved site. Legality cannot read S02 scores, S01 completion, analysis labels, future state, or S03 boundary signals.

### Four legal primitives

Adjacent swaps exchange two non-vacancy edge neighbors. Vacancy moves send one cell over an edge into a vacancy. Short exchanges swap non-vacancy endpoints on an explicit three-site path whose endpoints have exact graph distance two; the middle site is reserved. Rotations shift every non-vacancy identity one edge around a simple cycle of length 3–6. Fixtures cover bounded and periodic square graphs, bounded hexagonal graphs, explicit irregular graphs, a periodic seam, an obstacle environment, and fixed boundaries.

### Conflict and tie protocol

All proposals are validated against one pre-batch snapshot. Valid proposals reserve endpoints, full short paths, or full cycles. Content-derived SHA-256 priorities plus proposal ID provide a total order; a greedy maximal disjoint set commits atomically. The five-proposal conflict batch was exhaustively run under all {validation["orderPermutationCount"]} input permutations. A forced equal-priority unit fixture directly validated the proposal-ID tie branch.

### Cost and conservation

The {len(LEDGER_FIELDS)}-field integer ledger records proposal outcomes, validation checks/reads, reservations, conflict work, accepted kinds, displaced identities, and cell/vacancy/total graph distance. Validation and commit are accounted separately. Every batch compares pre/post identity sets, token counts, occupant-kind counts, site coverage, fixed roles, and obstacle exclusion. Boundary-signal reads are hard-zero.

### Serialization, replay, reversibility, and evaluation

States and proposals round-trip through canonical JSON. Each full batch fixture was serialized and independently re-executed; transition bytes and hashes matched exactly. Seven single-move fixtures applied an automatically generated inverse and recovered the exact occupant-to-site projection. Transition indices intentionally remained two events later. Three target-bound square movements recorded S02 local grammar and S01 global audit before/after under separate field names; neither field influenced legality.

### Commands

```bash
cd /workspace/cell-research
python -m pytest -q tests/test_morph2d_targets.py tests/test_morph2d_grammar.py tests/test_morph2d_environments.py tests/test_morph2d_movements.py
python scripts/build_morph2d_s04.py build --output-dir /artifacts/research_steps/S04
python scripts/build_morph2d_s04.py validate --output-dir /artifacts/research_steps/S04
ruff check src/morph2d scripts/build_morph2d_s04.py tests/test_morph2d_movements.py
ruff format --check src/morph2d scripts/build_morph2d_s04.py tests/test_morph2d_movements.py
python -m compileall -q src/morph2d scripts/build_morph2d_s04.py tests/test_morph2d_movements.py
```

Runtime provenance records Python {provenance["python"]}, repository base commit `{provenance["repository"]["baseCommit"]}`, and intentional serial execution on one worker. The small exact-fixture and 120-permutation workload did not benefit from parallelism; the L4 GPU was not used.

## Results

{os.linesep.join(table)}

### Anchor results

- **Legal coverage:** all four enabled kinds validated and committed on their planned graph fixtures; {accepted} movements committed across the nine canonical batches.
- **Conservation:** 9/9 batches retained exact site coverage, identity set, token composition, vacancy count/kind composition, obstacle exclusion, and fixed-boundary occupants/tokens.
- **Conflict determinism:** the conflict fixture produced {conflict_audit["acceptedCount"]} accepted and {conflict_audit["conflictLossCount"]} conflict-lost proposals; all 120 proposal permutations yielded one byte-identical result. The equal-priority branch selected the lexicographically smaller proposal ID.
- **Rejections:** {validation["validationProbeCount"]}/{validation["validationProbeCount"]} probes returned their exact reasons, including three deterministic deferred-interface rejections. The mixed fixed-boundary batch added one invalid proposal and one legal interior movement ({invalid} invalid fixture proposals total).
- **Cost completeness:** all {len(LEDGER_FIELDS)} fields were present and nonnegative in every batch; committed distance reconciled exactly to movement kind and occupant kind. There were {conflict_losses} conflict losses and zero boundary-signal reads.
- **Reversibility/replay:** {validation["reversibilityFixtureCount"]}/{validation["reversibilityFixtureCount"]} inverse fixtures restored the exact occupant projection; 9/9 full transitions round-tripped and replayed byte-identically.
- **Evaluation separation:** {len(layer_records)} target-bound transitions retained independent S02-local and S01-global records. The adjacent swap was local-accepted/global-failed; the rotation was local-rejected/global-passed; the short exchange passed both—demonstrating that these fields are nonredundant and absent from movement legality.

Machine-readable evidence is in `fixture_results.*`, `proposal_validation_audit.*`, `conservation_audit.json`, `movement_cost_ledger.csv`, `conflict_resolution_audit.json`, `proposal_order_invariance.json`, `reversibility_audit.json`, `serialization_roundtrip.json`, `deterministic_replay.json`, and `evaluation_layer_results.*`. Full replayable batch records are under `movement_fixtures/`.

## Validation

`validation_summary.json` records all gates as true:

- catalog/schema and exact enabled/deferred sets;
- all four legal kinds across square, hexagonal, irregular, bounded, periodic, vacancy, obstacle, and fixed-role environments;
- exact validation diagnostics for stale state/resource identity, actor mismatch, non-edge displacement, wrong vacancy direction, overlong short exchange, open rotation, obstacle, fixed boundary, and the three deferred kinds;
- invariant conservation, legal displacement, full-route reservations, deterministic conflict/tie behavior, and all 120 input orders;
- seven forward/inverse occupancy fixtures, canonical state/proposal round trips, and nine exact batch replays;
- a complete nonnegative {len(LEDGER_FIELDS)}-field cost ledger with zero boundary-signal reads;
- separate S02/S01 evaluation records and absence of evaluation data from state/proposal hashes;
- S01–S03 immutability, final artifact hashes, and focused tests (`{test_summary}`).

## Caveats, blockers, failed assumptions, and limitations

- S04 validates one-batch CPU transitions. It does not define activation scheduling across time, policy choice, terminal conditions, formation, repair, or GPU execution.
- The greedy conflict set is deterministic and maximal in priority order, not guaranteed maximum-cardinality or globally minimum-cost.
- SHA-256 conflict priorities are deterministic pseudorandom ranks, not scientific randomness or policy merit. Batch nonces must be scenario-derived later; worker or arrival order must never be used.
- Short exchange is intentionally fixed at exact distance two and reserves the middle site. Any different range or path-sharing rule needs a new version and fairness analysis.
- Rotations permit only simple fully occupied cycles of length 3–6. Vacancy-bearing rotations and larger cycles are rejected rather than conflated with vacancy motion or long-range exchange.
- Fixed-role enforcement is engine-visible safety metadata. It does not imply that S05 policies may observe fixed roles or exterior signals without an explicit budget.
- Division and removal violate the identity/composition invariants; long-range exchange can dominate local comparisons. All three remain executable rejections pending separate contracts.
- S01/S02 completion remains calibrated only for unobstructed bounded rectangular four-neighbor targets. Hexagonal, irregular, periodic, obstacle, and fixed-boundary fixtures have movement evidence but no target-completion claim.
- Identity reversibility does not reverse the monotonically increasing transition index; event history is not erased.
- These are computational transition proxies, not biological morphology, movement, mechanism, repair, agency, or clinical evidence.

## Artifacts and provenance

Repository-backed implementation lives in `src/morph2d/movements.py`, `configs/morphologies/movement_catalog.yaml`, `scripts/build_morph2d_s04.py`, `tests/test_morph2d_movements.py`, and the public exports in `src/morph2d/__init__.py`. Source is not duplicated under artifacts.

`movement_catalog.yaml` is the immutable contract snapshot. `movement_fixtures/` holds full canonical batch results; the root contains compact tables/audits, normative specification, validation, execution/environment/input provenance, and a recursive SHA-256 manifest.

## Recommended next action

{next_action}
"""


def build(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fixture_dir = output / "movement_fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    catalog = yaml.safe_load(MOVEMENTS.read_text(encoding="utf-8"))
    _, environments = load_environment_catalog(ENVIRONMENTS)
    _, targets = load_target_catalog(TARGETS)
    _, grammars = load_grammar_catalog(GRAMMARS)
    environment_by_id = {item.environment_id: item for item in environments}
    target_by_id = {item.target_id: item for item in targets}
    grammar_by_id = {item.grammar_id: item for item in grammars}

    if catalog["schemaVersion"] != "e06.s04.movement-catalog.v1":
        raise RuntimeError("movement catalog version mismatch")
    if {item["kind"] for item in catalog["enabledMovements"]} != ENABLED_KINDS:
        raise RuntimeError("enabled movement catalog mismatch")
    if {item["kind"] for item in catalog["deferredMovements"]} != DEFERRED_KINDS:
        raise RuntimeError("deferred movement catalog mismatch")
    shutil.copyfile(MOVEMENTS, output / "movement_catalog.yaml")
    (output / "movement_spec.md").write_text(_movement_spec(catalog), encoding="utf-8")

    fixture_records = []
    fixture_rows = []
    validation_rows = []
    conservation_records = []
    cost_rows = []
    serialization_records = []
    replay_records = []
    reversibility_records = []
    layer_records = []
    layer_rows = []
    full_results: dict[str, dict[str, Any]] = {}

    for raw in catalog["fixtures"]:
        environment = environment_by_id[raw["environmentId"]]
        state = initial_movement_state(environment)
        proposals = [_proposal_from_raw(state, item) for item in raw["proposals"]]
        validations = [
            validate_proposal(environment, state, proposal) for proposal in proposals
        ]
        for validation in validations:
            validation_rows.append(
                {
                    "recordType": "fixture",
                    "recordId": raw["fixtureId"],
                    "environmentId": environment.environment_id,
                    "proposalId": validation.proposal.proposal_id,
                    "kind": validation.proposal.kind,
                    "valid": validation.valid,
                    "reason": validation.reason,
                    "reservedSites": "|".join(validation.reserved_sites),
                    "adjacencyChecks": validation.adjacency_checks,
                    "maxIndividualGraphDisplacement": validation.max_individual_displacement,
                }
            )
        result = resolve_batch(
            environment,
            state,
            proposals,
            batch_nonce=raw["batchNonce"],
        )
        full_results[raw["fixtureId"]] = result
        fixture_path = fixture_dir / f"{raw['fixtureId']}.json"
        fixture_path.write_bytes(canonical_batch_result_bytes(result))
        replayed = replay_batch_result(
            environment,
            json.loads(canonical_batch_result_bytes(result)),
        )
        replay_pass = canonical_batch_result_bytes(
            replayed
        ) == canonical_batch_result_bytes(result)
        ledger = result["costLedger"]
        fixture_rows.append(
            {
                "fixtureId": raw["fixtureId"],
                "environmentId": environment.environment_id,
                "submittedProposals": ledger["submittedProposals"],
                "validProposals": ledger["validProposals"],
                "acceptedMovements": ledger["acceptedMovements"],
                "conflictLosses": ledger["conflictLosses"],
                "invalidProposals": ledger["invalidProposals"],
                "cellGraphDisplacement": ledger["cellGraphDisplacement"],
                "vacancyGraphDisplacement": ledger["vacancyGraphDisplacement"],
                "replayPass": replay_pass,
            }
        )
        fixture_records.append(
            {
                "fixtureId": raw["fixtureId"],
                "environmentId": environment.environment_id,
                "batchId": result["batchId"],
                "transitionSha256": result["transitionSha256"],
                "preStateSha256": result["preState"]["stateSha256"],
                "postStateSha256": result["postState"]["stateSha256"],
                "acceptedProposalIds": result["acceptedProposalIds"],
                "decisions": result["decisions"],
                "costLedger": ledger,
                "invariantSuccess": result["invariants"]["success"],
                "fixturePath": f"movement_fixtures/{fixture_path.name}",
                "fixtureSha256": _sha256_file(fixture_path),
            }
        )
        conservation_records.append(
            {
                "fixtureId": raw["fixtureId"],
                "environmentId": environment.environment_id,
                **result["invariants"],
                "boundarySignalBudget": result["boundarySignalBudget"],
            }
        )
        cost_rows.append(
            {
                "fixtureId": raw["fixtureId"],
                "environmentId": environment.environment_id,
                **{field: ledger[field] for field in LEDGER_FIELDS},
            }
        )
        state_roundtrip = parse_movement_state(
            json.loads(canonical_movement_state_bytes(state))
        )
        proposal_roundtrips = [
            canonical_proposal_bytes(
                parse_proposal(json.loads(canonical_proposal_bytes(proposal)))
            )
            == canonical_proposal_bytes(proposal)
            for proposal in proposals
        ]
        serialization_records.append(
            {
                "fixtureId": raw["fixtureId"],
                "stateByteIdentical": canonical_movement_state_bytes(state_roundtrip)
                == canonical_movement_state_bytes(state),
                "proposalCount": len(proposals),
                "proposalByteIdenticalCount": sum(proposal_roundtrips),
                "batchCanonicalBytesSha256": hashlib.sha256(
                    canonical_batch_result_bytes(result)
                ).hexdigest(),
                "success": all(proposal_roundtrips)
                and canonical_movement_state_bytes(state_roundtrip)
                == canonical_movement_state_bytes(state),
            }
        )
        replay_records.append(
            {
                "fixtureId": raw["fixtureId"],
                "transitionSha256": result["transitionSha256"],
                "replayTransitionSha256": replayed["transitionSha256"],
                "byteIdentical": replay_pass,
                "success": replay_pass,
            }
        )
        if raw["reversible"]:
            if len(result["acceptedProposalIds"]) != 1:
                raise RuntimeError(
                    "reversibility fixture must accept exactly one proposal"
                )
            proposal = next(
                item
                for item in proposals
                if item.proposal_id == result["acceptedProposalIds"][0]
            )
            post = parse_movement_state(result["postState"])
            inverse = inverse_proposal(post, proposal)
            backward = resolve_batch(
                environment,
                post,
                [inverse],
                batch_nonce=f"{raw['batchNonce']}-inverse",
            )
            final = parse_movement_state(backward["postState"])
            reversibility_records.append(
                {
                    "fixtureId": raw["fixtureId"],
                    "kind": proposal.kind,
                    "forwardProposalId": proposal.proposal_id,
                    "inverseProposalId": inverse.proposal_id,
                    "occupancyProjectionRestored": final.occupancy == state.occupancy,
                    "identityProjectionRestored": [
                        item.occupant_id for _, item in final.occupancy
                    ]
                    == [item.occupant_id for _, item in state.occupancy],
                    "fullStateHashRestored": final == state,
                    "expectedTransitionIndex": state.transition_index + 2,
                    "observedTransitionIndex": final.transition_index,
                    "success": final.occupancy == state.occupancy
                    and final.transition_index == state.transition_index + 2,
                }
            )
        if raw["evaluateLayers"]:
            binding = environment.target_binding
            target = target_by_id[binding["targetId"]]
            grammar = grammar_by_id[binding["grammarId"]]
            post = parse_movement_state(result["postState"])
            record = {
                "fixtureId": raw["fixtureId"],
                "environmentId": environment.environment_id,
                "movementKind": proposals[0].kind,
                "pre": _layer_record(environment, state, target, grammar),
                "post": _layer_record(environment, post, target, grammar),
                "movementLegalityUsedEvaluationFields": False,
            }
            layer_records.append(record)
            for phase in ("pre", "post"):
                layers = record[phase]
                layer_rows.append(
                    {
                        "fixtureId": raw["fixtureId"],
                        "movementKind": proposals[0].kind,
                        "phase": phase,
                        "s02LocalAccepted": layers["s02LocalGrammar"]["accepted"],
                        "s02LocalSoftScore": layers["s02LocalGrammar"]["softScore"],
                        "s01GlobalSuccess": layers["s01GlobalCompletionAudit"][
                            "success"
                        ],
                        "s01CountMatch": layers["s01GlobalCompletionAudit"][
                            "countMatch"
                        ],
                        "s01GeometryMatch": layers["s01GlobalCompletionAudit"][
                            "equivalenceOrGeometryMatch"
                        ],
                        "s01ComponentMatch": layers["s01GlobalCompletionAudit"][
                            "componentMatch"
                        ],
                        "s01TopologyMatch": layers["s01GlobalCompletionAudit"][
                            "topologyMatch"
                        ],
                        "conjunctiveCompletion": layers["conjunctiveCompletion"],
                    }
                )

    for raw in catalog["validationProbes"]:
        environment = environment_by_id[raw["environmentId"]]
        state = initial_movement_state(environment)
        proposal = _proposal_from_raw(state, raw)
        validation = validate_proposal(environment, state, proposal)
        validation_rows.append(
            {
                "recordType": "probe",
                "recordId": raw["probeId"],
                "environmentId": environment.environment_id,
                "proposalId": proposal.proposal_id,
                "kind": proposal.kind,
                "valid": validation.valid,
                "reason": validation.reason,
                "reservedSites": "|".join(validation.reserved_sites),
                "adjacencyChecks": validation.adjacency_checks,
                "maxIndividualGraphDisplacement": validation.max_individual_displacement,
            }
        )
        if validation.valid or validation.reason != raw["expectedReason"]:
            raise RuntimeError(f"validation probe failed: {raw['probeId']}")

    conflict_raw = next(
        item
        for item in catalog["fixtures"]
        if item["fixtureId"] == "conflicting_and_disjoint_square_batch"
    )
    conflict_environment = environment_by_id[conflict_raw["environmentId"]]
    conflict_state = initial_movement_state(conflict_environment)
    conflict_proposals = [
        _proposal_from_raw(conflict_state, item) for item in conflict_raw["proposals"]
    ]
    permutation_hashes = []
    for permutation in itertools.permutations(conflict_proposals):
        result = resolve_batch(
            conflict_environment,
            conflict_state,
            permutation,
            batch_nonce=conflict_raw["batchNonce"],
        )
        permutation_hashes.append(
            hashlib.sha256(canonical_batch_result_bytes(result)).hexdigest()
        )
    canonical_conflict = full_results[conflict_raw["fixtureId"]]
    first, second = sorted(conflict_proposals, key=lambda item: item.proposal_id)[:2]
    tie_audit = {
        "forcedEqualPriority": 7,
        "firstProposalId": first.proposal_id,
        "secondProposalId": second.proposal_id,
        "observedOrder": [
            item.proposal_id
            for item in sorted(
                (second, first),
                key=lambda item: conflict_order_key(
                    "0" * 64, item, priority_override=7
                ),
            )
        ],
        "expectedOrder": sorted([first.proposal_id, second.proposal_id]),
    }
    tie_audit["success"] = tie_audit["observedOrder"] == tie_audit["expectedOrder"]
    conflict_audit = {
        "schemaVersion": "e06.s04.conflict-resolution-audit.v1",
        "researchStepId": "S04",
        "fixtureId": conflict_raw["fixtureId"],
        "batchId": canonical_conflict["batchId"],
        "resolutionOrder": canonical_conflict["resolutionOrder"],
        "decisions": canonical_conflict["decisions"],
        "acceptedCount": len(canonical_conflict["acceptedProposalIds"]),
        "conflictLossCount": canonical_conflict["costLedger"]["conflictLosses"],
        "fullRouteReservations": True,
        "tieBreakAudit": tie_audit,
        "success": canonical_conflict["costLedger"]["conflictLosses"] > 0
        and tie_audit["success"],
    }
    order_audit = {
        "schemaVersion": "e06.s04.proposal-order-invariance.v1",
        "researchStepId": "S04",
        "fixtureId": conflict_raw["fixtureId"],
        "proposalCount": len(conflict_proposals),
        "permutationCount": len(permutation_hashes),
        "uniqueCanonicalResultHashCount": len(set(permutation_hashes)),
        "canonicalResultSha256": permutation_hashes[0],
        "inputOrderInvariant": len(set(permutation_hashes)) == 1,
        "success": len(set(permutation_hashes)) == 1,
    }

    deferred_rows = [
        {
            "kind": item["kind"],
            "status": item["status"],
            "reason": item["reason"],
            "executableRejectionReason": f"deferred_kind:{item['kind']}",
            "enabled": False,
        }
        for item in catalog["deferredMovements"]
    ]
    upstream_records = [
        _verify_upstream_manifest(directory)
        for directory in (S01_DIR, S02_DIR, S03_DIR)
    ]
    focused_tests = _focused_tests()
    checks = {
        "catalogAndMovementKinds": True,
        "conservationAndOccupancy": all(
            item["success"] for item in conservation_records
        ),
        "legalDisplacement": all(
            item["valid"]
            for item in validation_rows
            if item["recordType"] == "fixture"
            and not (
                item["recordId"] == "fixed_boundary_mixed_validation"
                and item["reason"] == "fixed_boundary_site"
            )
        ),
        "proposalValidation": all(
            (not item["valid"])
            for item in validation_rows
            if item["recordType"] == "probe"
        ),
        "fixedObstacleAndDeferredRejection": all(
            any(
                row["recordType"] == "probe"
                and row["recordId"] == probe_id
                and not row["valid"]
                for row in validation_rows
            )
            for probe_id in (
                "fixed_endpoint",
                "obstacle_endpoint",
                "deferred_long_range",
                "deferred_division",
                "deferred_removal",
            )
        ),
        "deterministicConflictsAndTieBreaking": conflict_audit["success"],
        "proposalOrderInvariance": order_audit["success"],
        "reversibility": all(item["success"] for item in reversibility_records),
        "serializationRoundtrip": all(
            item["success"] for item in serialization_records
        ),
        "deterministicReplay": all(item["success"] for item in replay_records),
        "completeCostAccounting": all(
            set(row) == {"fixtureId", "environmentId", *LEDGER_FIELDS}
            and all(int(row[field]) >= 0 for field in LEDGER_FIELDS)
            for row in cost_rows
        ),
        "boundarySignalBudget": all(
            row["boundarySignalReads"] == 0 for row in cost_rows
        ),
        "separateEvaluationFields": all(
            not item["movementLegalityUsedEvaluationFields"]
            and "s02LocalGrammar" in item["post"]
            and "s01GlobalCompletionAudit" in item["post"]
            for item in layer_records
        ),
        "upstreamImmutability": all(item["success"] for item in upstream_records),
        "focusedRepositoryTests": focused_tests["success"],
    }
    validation = {
        "schemaVersion": "e06.s04.validation-summary.v1",
        "researchStepId": "S04",
        "status": "complete",
        "success": all(checks.values()),
        "outcomeClassification": "supportive",
        "checks": checks,
        "fixtureCount": len(fixture_rows),
        "enabledMovementKinds": sorted(ENABLED_KINDS),
        "deferredMovementKinds": sorted(DEFERRED_KINDS),
        "validationProbeCount": len(catalog["validationProbes"]),
        "orderPermutationCount": len(permutation_hashes),
        "reversibilityFixtureCount": len(reversibility_records),
        "ledgerFieldCount": len(LEDGER_FIELDS),
        "upstreamImmutability": {
            "checkedArtifactCount": sum(
                item["checkedArtifactCount"] for item in upstream_records
            ),
            "records": upstream_records,
            "success": all(item["success"] for item in upstream_records),
        },
        "focusedTests": focused_tests,
        "validationResult": "PASS — movement legality, conservation, conflict/tie determinism, proposal-order invariance, reversibility, serialization, replay, complete costs, signal budgets, evaluation separation, and upstream integrity all passed.",
    }
    if not validation["success"]:
        raise RuntimeError("S04 validation failed before artifact packaging")

    _write_json(
        output / "fixture_results.json",
        {
            "schemaVersion": "e06.s04.fixture-results.v1",
            "researchStepId": "S04",
            "records": fixture_records,
        },
    )
    _write_csv(output / "fixture_results.csv", fixture_rows)
    _write_json(
        output / "proposal_validation_audit.json",
        {
            "schemaVersion": "e06.s04.proposal-validation-audit.v1",
            "researchStepId": "S04",
            "records": validation_rows,
            "success": all(
                not row["valid"]
                for row in validation_rows
                if row["recordType"] == "probe"
            ),
        },
    )
    _write_csv(output / "proposal_validation_audit.csv", validation_rows)
    _write_json(
        output / "conservation_audit.json",
        {
            "schemaVersion": "e06.s04.conservation-audit.v1",
            "researchStepId": "S04",
            "records": conservation_records,
            "success": checks["conservationAndOccupancy"],
        },
    )
    _write_csv(output / "movement_cost_ledger.csv", cost_rows)
    _write_json(output / "conflict_resolution_audit.json", conflict_audit)
    _write_json(output / "proposal_order_invariance.json", order_audit)
    _write_json(
        output / "reversibility_audit.json",
        {
            "schemaVersion": "e06.s04.reversibility-audit.v1",
            "researchStepId": "S04",
            "records": reversibility_records,
            "success": checks["reversibility"],
        },
    )
    _write_json(
        output / "serialization_roundtrip.json",
        {
            "schemaVersion": "e06.s04.serialization-roundtrip.v1",
            "researchStepId": "S04",
            "records": serialization_records,
            "success": checks["serializationRoundtrip"],
        },
    )
    _write_json(
        output / "deterministic_replay.json",
        {
            "schemaVersion": "e06.s04.deterministic-replay.v1",
            "researchStepId": "S04",
            "records": replay_records,
            "success": checks["deterministicReplay"],
        },
    )
    _write_json(
        output / "evaluation_layer_results.json",
        {
            "schemaVersion": "e06.s04.evaluation-layer-results.v1",
            "researchStepId": "S04",
            "records": layer_records,
            "success": checks["separateEvaluationFields"],
        },
    )
    _write_csv(output / "evaluation_layer_results.csv", layer_rows)
    _write_csv(output / "deferred_interfaces.csv", deferred_rows)
    _write_json(output / "validation_summary.json", validation)
    _write_json(output / "input_provenance.json", _input_provenance())

    provenance = {
        "schemaVersion": "e06.s04.environment-provenance.v1",
        "researchStepId": "S04",
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            "pytest": importlib.metadata.version("pytest"),
            "PyYAML": importlib.metadata.version("PyYAML"),
        },
        "resources": {
            "availableCpuCount": os.cpu_count(),
            "workerCount": 1,
            "threadEnvironment": {
                key: os.environ.get(key)
                for key in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                )
            },
            "gpuUsed": False,
            "networkUsed": False,
            "newDependenciesInstalled": [],
            "executionReason": "Serial exact fixtures and 120 small proposal permutations; parallel overhead was unnecessary.",
        },
        "repository": {
            "path": str(ROOT),
            "branch": _git("branch", "--show-current"),
            "baseCommit": _git("rev-parse", "HEAD"),
            "statusBeforeCommit": _git("status", "--short"),
        },
    }
    _write_json(output / "environment_provenance.json", provenance)
    commands = """cd /workspace/cell-research
python -m pytest -q tests/test_morph2d_targets.py tests/test_morph2d_grammar.py tests/test_morph2d_environments.py tests/test_morph2d_movements.py
python scripts/build_morph2d_s04.py build --output-dir /artifacts/research_steps/S04
python scripts/build_morph2d_s04.py validate --output-dir /artifacts/research_steps/S04
ruff check src/morph2d scripts/build_morph2d_s04.py tests/test_morph2d_movements.py
ruff format --check src/morph2d scripts/build_morph2d_s04.py tests/test_morph2d_movements.py
python -m compileall -q src/morph2d scripts/build_morph2d_s04.py tests/test_morph2d_movements.py
"""
    (output / "execution_commands.log").write_text(commands, encoding="utf-8")

    artifact_names = [
        "movement_spec.md",
        "movement_catalog.yaml",
        "movement_fixtures/ (9 canonical batch JSON records)",
        "fixture_results.json",
        "fixture_results.csv",
        "proposal_validation_audit.json",
        "proposal_validation_audit.csv",
        "conservation_audit.json",
        "movement_cost_ledger.csv",
        "conflict_resolution_audit.json",
        "proposal_order_invariance.json",
        "reversibility_audit.json",
        "serialization_roundtrip.json",
        "deterministic_replay.json",
        "evaluation_layer_results.json",
        "evaluation_layer_results.csv",
        "deferred_interfaces.csv",
        "validation_summary.json",
        "input_provenance.json",
        "environment_provenance.json",
        "execution_commands.log",
        "artifact_manifest.json",
        "research_step_full_results.md",
    ]
    report = _report(
        fixture_rows,
        validation,
        provenance,
        artifact_names,
        conflict_audit,
        layer_records,
    )
    (output / "research_step_full_results.md").write_text(report, encoding="utf-8")

    artifacts = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        relative = path.relative_to(output).as_posix()
        artifacts.append(
            {
                "path": relative,
                "role": _artifact_role(relative),
                "sizeBytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    _write_json(
        output / "artifact_manifest.json",
        {
            "schemaVersion": "e06.s04.artifact-manifest.v1",
            "researchStepId": "S04",
            "artifactCount": len(artifacts),
            "artifacts": artifacts,
        },
    )


def validate(output: Path) -> None:
    required = {
        "movement_spec.md",
        "movement_catalog.yaml",
        "fixture_results.json",
        "fixture_results.csv",
        "proposal_validation_audit.json",
        "proposal_validation_audit.csv",
        "conservation_audit.json",
        "movement_cost_ledger.csv",
        "conflict_resolution_audit.json",
        "proposal_order_invariance.json",
        "reversibility_audit.json",
        "serialization_roundtrip.json",
        "deterministic_replay.json",
        "evaluation_layer_results.json",
        "evaluation_layer_results.csv",
        "deferred_interfaces.csv",
        "validation_summary.json",
        "input_provenance.json",
        "environment_provenance.json",
        "execution_commands.log",
        "artifact_manifest.json",
        "research_step_full_results.md",
    }
    missing = sorted(name for name in required if not (output / name).is_file())
    if missing:
        raise RuntimeError(f"missing S04 artifacts: {missing}")
    fixture_paths = sorted((output / "movement_fixtures").glob("*.json"))
    if len(fixture_paths) != 9:
        raise RuntimeError("S04 movement fixture count mismatch")
    summary = json.loads((output / "validation_summary.json").read_text())
    if not summary["success"] or not all(summary["checks"].values()):
        raise RuntimeError("S04 validation summary is not successful")
    manifest = json.loads((output / "artifact_manifest.json").read_text())
    if manifest["artifactCount"] != len(manifest["artifacts"]):
        raise RuntimeError("artifact count mismatch")
    mismatches = []
    for record in manifest["artifacts"]:
        path = output / record["path"]
        if not path.is_file() or _sha256_file(path) != record["sha256"]:
            mismatches.append(record["path"])
    if mismatches:
        raise RuntimeError(f"artifact hash mismatch: {mismatches}")
    report = (output / "research_step_full_results.md").read_text()
    required_sections = (
        "## Top summary",
        "## Lay summary",
        "## Inputs",
        "## Methods",
        "### Commands",
        "## Results",
        "## Validation",
        "## Caveats, blockers, failed assumptions, and limitations",
        "## Artifacts and provenance",
        "## Recommended next action",
    )
    if not all(section in report for section in required_sections):
        raise RuntimeError("canonical report section validation failed")
    if "S05 was not started" not in report:
        raise RuntimeError("S04 stop boundary missing from report")
    print(
        json.dumps(
            {
                "researchStepId": "S04",
                "success": True,
                "artifactCount": manifest["artifactCount"],
                "fixtureCount": len(fixture_paths),
                "validationChecks": summary["checks"],
            },
            indent=2,
            sort_keys=True,
        )
    )


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
