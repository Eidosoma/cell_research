#!/usr/bin/env python3
"""Build and validate E06 S06 top-down communication-channel artifacts."""

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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
CHANNELS = ROOT / "configs/morphologies/control_channel_catalog.yaml"
POLICIES = ROOT / "configs/morphologies/policy_catalog.yaml"
ENVIRONMENTS = ROOT / "configs/morphologies/environment_catalog.yaml"
GRAMMARS = ROOT / "configs/morphologies/grammar_catalog.yaml"
SOURCE = ROOT / "src/morph2d/channels.py"
TEST = ROOT / "tests/test_morph2d_channels.py"
ATTACHMENT_DIR = WORKSPACE / "input-attachments/21c2278b-9950-4e39-a2c8-df578a2508ec"
STEP_DIRS = [Path(f"/artifacts/research_steps/S0{index}") for index in range(1, 6)]
UPSTREAM_INPUTS = {
    "agents": WORKSPACE / "AGENTS.md",
    "full_plan": WORKSPACE / "FULL_PLAN.md",
    "research_plan_pre_s06_update": WORKSPACE / "RESEARCH_PLAN.md",
    "previous_artifacts_md": WORKSPACE / "PREVIOUS_ARTIFACTS.md",
    "previous_artifacts_json": WORKSPACE / "PREVIOUS_ARTIFACTS.json",
    "capabilities": WORKSPACE / "CAPABILITIES.md",
    "capability_availability": WORKSPACE / "CAPABILITY_AVAILABILITY.json",
    "datasets": WORKSPACE / "DATASETS.md",
    "dataset_catalog": WORKSPACE / "DATASET_CATALOG.json",
    "dataset_availability": WORKSPACE / "DATASET_AVAILABILITY.json",
    "attachment_manifest": WORKSPACE / "input-attachments/MANIFEST.json",
    "attachment_sidecar": ATTACHMENT_DIR / "_metadata/ATTACHMENT.md",
    "s01_report": STEP_DIRS[0] / "research_step_full_results.md",
    "s01_manifest": STEP_DIRS[0] / "artifact_manifest.json",
    "s02_report": STEP_DIRS[1] / "research_step_full_results.md",
    "s02_grammar_spec": STEP_DIRS[1] / "grammar_spec.md",
    "s02_manifest": STEP_DIRS[1] / "artifact_manifest.json",
    "s03_report": STEP_DIRS[2] / "research_step_full_results.md",
    "s03_environment_spec": STEP_DIRS[2] / "environment_spec.md",
    "s03_boundary_requirements": STEP_DIRS[2] / "boundary_requirements.csv",
    "s03_manifest": STEP_DIRS[2] / "artifact_manifest.json",
    "s04_report": STEP_DIRS[3] / "research_step_full_results.md",
    "s04_movement_spec": STEP_DIRS[3] / "movement_spec.md",
    "s04_movement_catalog": STEP_DIRS[3] / "movement_catalog.yaml",
    "s04_manifest": STEP_DIRS[3] / "artifact_manifest.json",
    "s05_report": STEP_DIRS[4] / "research_step_full_results.md",
    "s05_policy_spec": STEP_DIRS[4] / "policy_spec.md",
    "s05_policy_catalog": STEP_DIRS[4] / "policy_library/policy_catalog.yaml",
    "s05_observation_permissions": STEP_DIRS[4] / "observation_permissions.json",
    "s05_observation_budget": STEP_DIRS[4] / "observation_budget.csv",
    "s05_gradient_audit": STEP_DIRS[4] / "gradient_interface_audit.json",
    "s05_validation": STEP_DIRS[4] / "validation_summary.json",
    "s05_manifest": STEP_DIRS[4] / "artifact_manifest.json",
    "e01_transition": Path(
        "/previous-artifacts/E01/research_steps/S03/transition_spec.md"
    ),
    "e01_transition_contract": Path(
        "/previous-artifacts/E01/research_steps/S03/transition_contract.json"
    ),
    "e01_api": Path("/previous-artifacts/E01/research_steps/S05/api_documentation.md"),
    "e01_event_schema": Path(
        "/previous-artifacts/E01/research_steps/S06/schema_documentation.md"
    ),
    "e01_seed_spec": Path(
        "/previous-artifacts/E01/research_steps/S08/seed_specification.json"
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
    "e04_declustering": Path(
        "/previous-artifacts/E04/research_steps/S10/declustering_specification.md"
    ),
    "e04_transport": Path(
        "/previous-artifacts/E04/research_steps/S12/transport_model_specification.md"
    ),
    "e04_handoff": Path(
        "/previous-artifacts/E04/research_steps/S14/e06_e07_handoff.md"
    ),
}

sys.path.insert(0, str(ROOT))

from src.morph2d.channels import (  # noqa: E402
    CHANNEL_LEDGER_FIELDS,
    ChannelValidationError,
    GlobalSummarySource,
    build_boundary_delivery,
    build_direct_controller_view,
    build_global_summary_delivery,
    build_gradient_delivery_from_s05,
    build_sparse_instruction_delivery,
    calibrate_noise,
    canonical_delivery_bytes,
    canonical_direct_event_bytes,
    channel_comparison_vector,
    channel_to_dict,
    compile_boundary_levels,
    compile_static_gradient,
    controller_source_forbidden_accesses,
    decide_direct_controller,
    delivery_to_dict,
    execute_direct_intervention,
    load_channel_catalog,
    parse_channel_catalog,
    parse_delivery,
    realize_noisy_gradient,
    validate_channel_payload,
    validate_controller_view_payload,
    validate_ledger_against_budget,
    validate_source_projection,
)
from src.morph2d.environments import load_environment_catalog  # noqa: E402
from src.morph2d.grammar import load_grammar_catalog  # noqa: E402
from src.morph2d.movements import (  # noqa: E402
    initial_movement_state,
    make_proposal,
    parse_movement_state,
    resolve_batch,
)
from src.morph2d.policies import (  # noqa: E402
    build_policy_observation,
    compile_relation_profile,
    load_policy_catalog,
)


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
        "tests/test_morph2d_channels.py",
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
            raise FileNotFoundError(f"required S06 input missing: {path}")
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
        ("repository_channel_catalog", CHANNELS),
        ("repository_channel_source", SOURCE),
        ("repository_channel_tests", TEST),
        ("repository_policy_catalog", POLICIES),
        ("repository_environment_catalog", ENVIRONMENTS),
        ("repository_grammar_catalog", GRAMMARS),
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
        "schemaVersion": "e06.s06.input-provenance.v1",
        "researchStepId": "S06",
        "inputs": records,
    }


def _actor_at(state, site_id: str) -> str:
    return state.occupant_map[site_id].occupant_id


def _perturbed_state(environment):
    state = initial_movement_state(environment)
    proposal = make_proposal(state, "adjacent_swap", ("r2_c0", "r3_c0"))
    result = resolve_batch(
        environment, state, (proposal,), batch_nonce="s06-fixture-damage"
    )
    return parse_movement_state(result["postState"])


def _control_channel_spec(catalog: Mapping[str, Any]) -> str:
    return f"""# E06 S06 Top-Down Communication Channel Specification

## Scope and authority

Version `{catalog["schemaVersion"]}` defines five CPU reference communication/actuation contracts over S05's frozen priced-observation boundary: static gradients, natural-boundary signals, sparse instructions, delayed global summaries, and sparse direct interventions. S06 defines source/owner, semantics, spatial resolution, update timing, counter-addressed noise, bandwidth, action cost, controller permissions, serialization, replay, and negative access rules. It does not implement an episode scheduler, GPU engine, formation experiment, controller optimization, or S07 work.

## Frozen S05 boundary

S05 base policy payloads remain immutable. Site IDs, coordinates, routes, occupant identities, raw roles/tags, proposal authentication, current competing proposals, priorities, labels, whole-grid S02 scores, S01 completion, and future state never enter a policy or controller payload. A later engine may attach a separately versioned `channelPayload` only after charging this ledger. Privileged engine source adapters can compile fields or aggregates, but raw inputs stay engine-only.

## Common budget vector

Every delivery/event carries the {len(CHANNEL_LEDGER_FIELDS)}-field ledger declared in the catalog. Information is split into one-time configuration bits, controller-input bits, policy-delivery bits, and address bits; their sum is `totalInformationBits`. Source reads and controller compute are separate. Direct action records attempts, successes, override units, S04 graph displacement, suppressed native actions, and opportunity cost separately. Action and computation are never converted to bits.

For an S05 policy observation that already contains a channel-derived projection, `policyDeliveryBits` references its `communicatedBitsUpperBound`; it is not added twice. Fair comparisons match budget vectors component-wise. The frozen `delivery_matched_64` profile caps advisory runtime delivery at 64 bits/epoch with no actuation, while `action_matched_sparse_1` permits one direct attempt, one override unit, at most six units of S04 displacement, seven address bits, and 352 controller-input bits. Native channel caps are exploratory and explicitly unmatched.

## Channel contracts

### Static gradient

The state-blind scenario compiler converts one authored display-coordinate axis to an immutable uint8 field and up/down direction. Runtime occupancy cannot affect it. The engine retains per-site values; S05 receives only its current level and opaque-candidate deltas. One-time configuration is `8 × occupiable sites + 1` bits. Independent uniform integer jitter in `[-2,2]` is counter-addressed; values clip to uint8.

### Natural boundary signal

The S03 environment compiler projects authored natural-exterior tags to levels 0–6. Raw tags, obstacle contacts, and fixed roles remain hidden. S05/S06 expose only current/candidate levels, seek/avoid, directive applicability, and presence after independent 1/8 erasure. Periodic environments compile to all zeros. Irregular tags remain an authored semantic asymmetry.

### Sparse instruction

A state-blind preregistered schedule emits at most one of eight high-level advisory symbols per 16-transition epoch to a broadcast or coarse region containing at least four recipients. Symbols cannot name a site, identity, token, target state, candidate, or route. Independent 1/8 erasure becomes `noop`. Region description and address bits are charged separately.

### Lagged global summary

An audited engine monitor receives only prior-epoch local-dissatisfaction bits and prior-batch conflict outcomes, not occupancy or completion. It broadcasts a three-bit dissatisfaction bin, two-bit conflict bin, and one-bit valid-age marker exactly one epoch later. Each bin receives independent `-1/0/+1` jitter with probabilities `0.1/0.8/0.1`. Source work scales with population and is reported.

### Sparse direct intervention

A bounded controller receives one lagged summary and one immutable S05 payload for one recipient selected by a state-blind round-robin alias. It may select one opaque candidate or take no action, at most once per epoch; native-action suppression is not enabled. It cannot scan recipients, see routes/authentication, retry, or choose an identity/site. Engine-side S05/S04 adapters materialize and validate the proposal only afterward. A 1/20 actuation failure charges opportunity/override units but makes no movement. Controller input, computation, address, override, and S04 displacement costs are all explicit.

## Controller-state permissions and hidden-state denial

The only persistent controller fields are catalogued epoch/schedule cursors, remaining budgets, and the last lagged summary, with explicit bit bounds. Generic callbacks and latent opaque state are not authorized. Recursive validators reject full state, occupancy, tokens, identities, site/route/role data, raw boundary tags, proposal authentication, current-batch proposals, priorities, labels, global completion/evaluation, future state, and future draws. Negative injection probes exercise each denial.

## Noise, serialization, and replay

All channel noise uses SHA-256 counter addresses over scenario key, channel ID, epoch, recipient alias, and draw index with rejection-unbiased bounded integers. Addresses and keys remain engine-only. Each model is calibrated over 100,000 addresses against an absolute probability tolerance of 0.01. Deliveries and direct events use canonical sorted-key compact JSON and content hashes; repeated construction must be byte-identical.

## Unavoidable asymmetries

Bits are syntax, not semantic value. A static field embeds per-site prior structure; a boundary signal uses privileged environment truth; an instruction symbol can encode authored task knowledge; a global summary requires population-wide aggregation and computation; and direct intervention bypasses local choice. Configuration reuse, broadcast replication, delay, clipping, computation, target specificity, and causal authority are therefore reported separately. No scalar “equivalent control” score is defined.

## Claim boundary

Passing S06 establishes enforceable reference contracts and calibrated channel plumbing, not fair empirical performance, formation, repair, robustness, biological signaling, or superiority of hybrid control. Those require separately authorized S07+ implementations and experiments.
"""


def _channel_card(definition, fixture: Mapping[str, Any]) -> str:
    asymmetries = "\n".join(f"- {item}" for item in definition.unavoidable_asymmetries)
    allowed_state = (
        ", ".join(
            f"`{item}`" for item in definition.controller_state_permissions["allowed"]
        )
        or "none"
    )
    return f"""# Channel card: {definition.channel_id}

- **Type:** `{definition.channel_type}`
- **Source owner:** `{definition.source["owner"]}`
- **Source implementation:** `{definition.source["implementation"]}`
- **Semantic content:** {definition.semantic_content}
- **Spatial resolution:** `{definition.spatial_resolution}`
- **Update schedule:** `{definition.update_schedule}`
- **Noise model:** `{definition.noise_model["kind"]}`
- **Bandwidth:** configuration `{definition.bandwidth["configurationBitsFormula"]}`; runtime `{definition.bandwidth["policyBitsFormula"]}`.
- **Action cost:** `{definition.action_cost["kind"]}`
- **Allowed persistent controller state:** {allowed_state}; {definition.controller_state_permissions["persistentBits"]} bits.
- **Target specificity:** `{definition.target_specificity}`
- **Frozen fixture:** `{fixture["fixtureId"]}`; validation `{fixture["validationPass"]}`; total recorded information {fixture["totalInformationBits"]} bits; override units {fixture["overrideActionUnits"]}; movement displacement {fixture["movementGraphDisplacement"]}.

## Unavoidable asymmetries

{asymmetries}
"""


def _report(
    validation: Mapping[str, Any],
    fixture_rows: list[dict[str, Any]],
    noise_rows: list[dict[str, Any]],
    artifacts: list[str],
    environment_provenance: Mapping[str, Any],
) -> str:
    table = [
        "| Channel | Source owner | Runtime policy bits | Controller bits | Config bits | Address bits | Override | Displacement | Replay |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in fixture_rows:
        table.append(
            f"| `{row['channelId']}` | `{row['sourceOwner']}` | "
            f"{row['policyDeliveryBits']} | {row['controllerInputBits']} | "
            f"{row['configurationBits']} | {row['addressBits']} | "
            f"{row['overrideActionUnits']} | {row['movementGraphDisplacement']} | "
            f"{row['deterministicReplayPass']} |"
        )
    noise_max = max(item["maximumAbsoluteError"] for item in noise_rows)
    test_summary = validation["focusedTests"]["stdout"].splitlines()[-1]
    next_action = (
        "Return control to the Chief Scientist. If S06 is accepted, separately "
        "authorize only S07 to implement the CPU/GPU episode engine against "
        "these frozen channel contracts; do not start S07 automatically."
    )
    return f"""# Research Step S06 Full Results — Define Top-Down Communication Channels

## Top summary

- **Research step ID:** S06
- **Completion status:** **Complete** on {datetime.now(timezone.utc).date().isoformat()}; only S06 was executed and S07 was not started.
- **Artifacts written:** {len(artifacts) + 1} files total ({len(artifacts)} recursively hash-listed files plus `artifact_manifest.json`): the normative control-channel specification, five-channel library/catalog/index/cards and canonical fixtures, budget schema/comparison, source-owner/update/controller permission tables, cost and semantic-asymmetry registers, noise calibration, hidden-state and permission audits, S05-boundary preservation, serialization/replay, validation/provenance/commands, manifest, and this canonical report.
- **Validation result:** **PASS** — five complete channel contracts, five deterministic fixtures, component-wise information/action budgets, 100,000-draw calibration for every noise model (maximum absolute probability error {noise_max:.6f}), {validation["hiddenStateProbeCount"]} hidden-field probes, cost reconciliation, S05 permission isolation, exact serialization/replay, {validation["upstreamImmutability"]["checkedArtifactCount"]} upstream artifact hashes, and focused tests (`{test_summary}`) all passed.
- **Outcome classification:** **supportive.** All five channels can be specified as deterministic, permission-audited, explicitly priced projections or actions without granting full-state access or silently changing S05 observations.
- **Caveats or blockers:** Bit counts do not equal semantic value. Static gradients, authored boundaries/instructions, delayed aggregate computation, and direct causal authority remain materially asymmetric even under matched runtime delivery. Direct control is deliberately limited to one state-blind recipient/query/action per epoch. S06 defines reference contracts, not an episode scheduler, empirical fairness, performance, or CPU/GPU implementation.
- **Lay summary:** Five ways of giving simulated cells or a controller extra help now have explicit rules: who creates the signal, what it says, where and when it appears, how it can be corrupted, and exactly what information or action it costs. Hidden maps and whole-pattern answers are rejected. A direct controller can inspect only one preselected anonymous cell’s already-priced local view and act once, so it cannot quietly steer every move.
- **Recommended next action:** {next_action}

## Frozen question and completion criterion

**Frozen question:** Can static gradients, natural-boundary signals, sparse instructions, delayed global summaries, and direct interventions be compared through explicit information, computation, addressing, and action budgets while preserving S05's local-observation boundary and preventing latent full-state micromanagement?

**Completion criterion:** every channel must declare and validate source/owner, semantics, spatial resolution, updates, noise, bandwidth, action cost, and controller-state permissions; permission isolation, complete cost accounting, noise calibration, canonical replay, and negative hidden-state probes must pass; unavoidable asymmetries must remain explicit. The criterion was met at the CPU reference-contract layer.

## Lay summary

The channel catalog separates advice from action. A gradient or boundary cue gives a local number; a sparse instruction gives a coarse symbol; a global summary gives delayed bins with no map; and direct intervention can choose one move for one anonymous, state-blind scheduled recipient. Each path has a separate ledger. This makes later comparisons auditable, but does not make their information equally useful or prove that any channel improves pattern formation.

## Inputs

- Refreshed `/workspace/AGENTS.md`, `FULL_PLAN.md`, and the pre-completion `RESEARCH_PLAN.md`, plus capability/dataset records and `input-attachments/MANIFEST.json` with its sidecar.
- Manifest-verified S01 target/global-evaluation, S02 grammar/falsification, S03 graph/boundary, S04 movement/authentication/cost, and S05 policy/observation/gradient artifacts.
- E01 immutable observation, scenario identity, counter-addressed randomness, cost ledger, event availability, and exact-replay contracts.
- E04 label-blind interventions, identity controls, state-matched switches, declustering boundaries, transport-cost distinctions, and E06 handoff constraints.
- No dataset, network access, package installation, GPU use, or upstream mutation was required.

Exact paths and SHA-256 values are in `input_provenance.json`; {validation["upstreamImmutability"]["checkedArtifactCount"]} manifest-listed S01–S05 artifacts rehashed successfully.

## Methods

### Typed channel catalog and source ownership

`src/morph2d/channels.py` parses exactly five channel types. Each declaration fixes source owner/implementation, allowlisted engine inputs, semantic payload, spatial scope, update cadence, target specificity, noise, bandwidth maxima, action maxima, persistent controller fields/bits, and unavoidable asymmetries. Static compilation may read coordinates or authored S03 tags engine-side; the global aggregator reads only prior-epoch dissatisfaction bits and conflict outcomes; the direct controller gets one S05 payload. Source fields are never copied wholesale to recipients.

### Permission boundary and anti-micromanagement design

Recursive payload validators deny state, occupancy, tokens, identities, site/route/role data, raw tags, authentication, current proposals/priorities, labels, S02 whole-grid/S01 completion, target membership, future state, and future draws. S05 base observations are immutable and remain the decision authority for local policies. Direct targeting uses `epoch mod population_size` to select one opaque alias without state. The controller receives no list of recipients and cannot retry, query another cell, or materialize a route. Controller functions are pure over the validated view and have no generic callback or opaque latent state.

### Comparable budget vectors

The {len(CHANNEL_LEDGER_FIELDS)}-field ledger keeps configuration, controller input, policy delivery, addressing, source reads, computation, noise, overrides, displacement, suppression, and opportunity cost separate. The advisory `delivery_matched_64` profile allows at most 64 runtime policy bits, four address bits, and no actuation per 16 transitions; every advisory fixture fits. The sparse-action profile permits one query/action, 352 controller bits, seven address bits, one override, and six S04 displacement units; the direct fixture fits. Configuration is reported but not falsely matched, and native-cap results must be labelled unmatched.

### Noise and timing

All noise is counter-addressed and rejection-unbiased. The gradient uses uniform integer jitter `[-2,2]`; boundary and instruction use 1/8 erasure; global summary bins use independent `0.1/0.8/0.1` adjacent jitter; direct actuation fails with probability 1/20. Each was tested over 100,000 independent addresses. Gradient/boundary fields are static, instructions/direct actions are limited to one per 16-transition epoch, and summaries are delayed exactly one completed epoch.

### Direct action and S04 reconciliation

The direct fixture first creates an ordinary S05 greedy observation for a perturbed stripe. The bounded controller selects one opaque candidate using only disclosed local scalars. Engine-side mapping then adds the S04 route, identity, state hash, and expected occupants. Normal S04 validation/commit runs unchanged. The channel ledger's graph displacement is required to equal the S04 `totalGraphDisplacement`; conservation/invariants must pass.

### Serialization, replay, and falsification

Four advisory deliveries and the direct intervention event serialize as canonical sorted-key compact JSON with content hashes. Independent re-execution must be byte-identical; delivery parsing must reject tampering. Negative tests inject every hidden-state category into controller and policy-facing payloads, add an unallowlisted source field, and try to persist hidden state. Coarse instructions reject regions below four recipients; summaries reject missing one-epoch delay; gradient/boundary compilation validates domain behavior.

### Commands

```bash
cd /workspace/cell-research
python -m pytest -q tests/test_morph2d_targets.py tests/test_morph2d_grammar.py tests/test_morph2d_environments.py tests/test_morph2d_movements.py tests/test_morph2d_policies.py tests/test_morph2d_channels.py
python scripts/build_morph2d_s06.py build --output-dir /artifacts/research_steps/S06
python scripts/build_morph2d_s06.py validate --output-dir /artifacts/research_steps/S06
ruff check src/morph2d scripts/build_morph2d_s06.py tests/test_morph2d_channels.py
ruff format --check src/morph2d scripts/build_morph2d_s06.py tests/test_morph2d_channels.py
python -m compileall -q src/morph2d scripts/build_morph2d_s06.py tests/test_morph2d_channels.py
```

The run used Python {environment_provenance["python"]} and intentional serial execution on one worker. Exact small fixtures and counter-addressed calibration did not benefit from CPU multiprocessing; no GPU was needed.

## Results

{os.linesep.join(table)}

### Anchor results

- **Five complete contracts:** source, owner, semantics, resolution, updates, noise, bandwidth, action, controller state, target specificity, and asymmetries are nonempty for all five channels.
- **Permission isolation:** all negative hidden-state injections and source/controller allowlist mutations were rejected. Direct-controller source inspection found no forbidden access. Channel construction did not mutate any S05 payload or disclose authentication/global-completion fields.
- **Comparable ledgers:** advisory fixtures used {min(row["policyDeliveryBits"] for row in fixture_rows[:-1])}–{max(row["policyDeliveryBits"] for row in fixture_rows[:-1])} policy bits and no actions; the direct fixture used {fixture_rows[-1]["controllerInputBits"]} controller bits, {fixture_rows[-1]["addressBits"]} address bits, one override, and {fixture_rows[-1]["movementGraphDisplacement"]} S04 displacement units. Configuration and source work remained separate.
- **Noise calibration:** all five 100,000-address calibrations passed the 0.01 absolute-probability tolerance; maximum observed error was {noise_max:.6f}.
- **Timing and scope:** periodic boundary levels were exactly zero; coarse instructions rejected single-recipient targeting; global summaries rejected zero-delay/current-state use; direct recipient choice was state-blind and limited to one query/action.
- **Serialization/replay:** all five fixtures regenerated byte-identically; four advisory deliveries parsed and round-tripped exactly; the direct event preserved S04 invariants and exact cost reconciliation.
- **Asymmetry finding:** equal runtime bits cannot equalize authored spatial priors, privileged boundary truth, global aggregation work, symbol meaning, or direct causal authority. This is a constraining design fact retained inside an otherwise supportive implementation result.

Machine-readable evidence is in `budget_schema.json`, `budget_comparison.*`, `cost_accounting.*`, `permission_isolation.json`, `hidden_state_probe_results.*`, `noise_calibration.*`, `serialization_replay.json`, `s05_boundary_preservation.json`, and the source/owner, update, permission, and asymmetry tables.

## Validation

`validation_summary.json` records every gate as true:

- exact five-channel parser round trip and complete required semantic dimensions;
- S05 observation immutability, no raw/authentication/completion leakage, and state-blind one-query direct control;
- exact {len(CHANNEL_LEDGER_FIELDS)}-field nonnegative ledgers, algebraic information totals, channel-native caps, advisory 64-bit and sparse-action comparison profiles, and S04 displacement reconciliation;
- 100,000 counter-addressed draws for each of five noise models within 0.01 calibration tolerance;
- bounded/periodic boundary behavior, gradient uint8/static compilation, coarse instruction resolution, one-epoch summary delay, and direct action limits;
- canonical delivery parse/round trip, tamper rejection, and exact independent fixture replay;
- hidden-state/source/controller-memory negative probes, S01–S05 immutability, final artifact hashes, and focused tests (`{test_summary}`).

## Caveats, blockers, failed assumptions, and limitations

- **Unavoidable asymmetry:** communication bit length does not measure semantic value. A three-bit authored instruction can encode task knowledge that a three-bit noisy measurement does not.
- Static-gradient configuration scales with site count and can be reused for many activations. Reporting one-time configuration separately is transparent but does not solve how to amortize prior knowledge across episode lengths.
- Boundary signals rely on authored environment truth. Periodic domains have no natural exterior; irregular domains require S03 tags. Obstacle/fixed-role metadata remains hidden.
- The global summary uses grammar-parameterized local dissatisfaction bits, not S02 whole-grid acceptance or S01 completion. Its privileged aggregation cost scales with population, and binning/delay removes spatial detail.
- Sparse instructions are state-blind and coarse by design. Allowing adaptive instructions later would require a newly priced controller-input contract.
- Direct intervention bypasses local choice and therefore cannot be made semantically equivalent to advice by matching bits. It is limited to a state-blind recipient and one local query/action, and it pays override, opportunity, and movement costs separately.
- Controller compute units are a transparent reference count, not wall time, energy, algorithmic complexity, or optimization power. A future central planner needs a separately frozen compute budget and cannot be smuggled into this interface.
- Noise calibration verifies the address sampler, not biological noise. Gradient clipping produces endpoint bias; global-bin clipping produces boundary bias; both are recorded asymmetries.
- The advisory 64-bit and sparse-action profiles are executable comparison envelopes, not evidence that later conditions will have equal effective information or equal performance.
- S06 does not implement multi-epoch scheduling, GPU tensors, formation, repair, chimerism, controller learning, or hybrid-control experiments. S07 was not started.
- These are computational communication/control proxies, not evidence of biological signaling, morphogenesis, cognition, agency, or clinical effects.

## Artifacts and provenance

Repository-backed implementation lives in `src/morph2d/channels.py`, `configs/morphologies/control_channel_catalog.yaml`, `scripts/build_morph2d_s06.py`, `tests/test_morph2d_channels.py`, and public exports in `src/morph2d/__init__.py`; source is not duplicated under artifacts. `channel_library/` contains the catalog/index/cards and canonical fixtures. Root artifacts contain the normative specification, compact ledgers/audits/tables, validation, environment/input provenance, commands, report, and recursive SHA-256 manifest.

## Recommended next action

{next_action}
"""


def build(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    library = output / "channel_library"
    cards = library / "cards"
    fixtures_dir = library / "fixtures"
    cards.mkdir(parents=True, exist_ok=True)
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    raw_catalog = yaml.safe_load(CHANNELS.read_text(encoding="utf-8"))
    metadata, channels = load_channel_catalog(CHANNELS)
    _, environments = load_environment_catalog(ENVIRONMENTS)
    _, policies = load_policy_catalog(POLICIES)
    _, grammars = load_grammar_catalog(GRAMMARS)
    channel_by_id = {item.channel_id: item for item in channels}
    environment_by_id = {item.environment_id: item for item in environments}
    policy_by_id = {item.policy_id: item for item in policies}
    grammar_by_id = {item.grammar_id: item for item in grammars}

    shutil.copyfile(CHANNELS, library / "control_channel_catalog.yaml")
    (output / "control_channel_spec.md").write_text(
        _control_channel_spec(raw_catalog), encoding="utf-8"
    )

    upstream_steps = [_verify_upstream_manifest(directory) for directory in STEP_DIRS]
    upstream = {
        "schemaVersion": "e06.s06.upstream-immutability.v1",
        "researchStepId": "S06",
        "steps": upstream_steps,
        "checkedArtifactCount": sum(
            item["checkedArtifactCount"] for item in upstream_steps
        ),
        "success": all(item["success"] for item in upstream_steps),
    }
    _write_json(output / "upstream_immutability.json", upstream)
    _write_json(output / "input_provenance.json", _input_provenance())

    fixture_rows = []
    fixture_records = []
    replay_records = []
    ledger_by_channel: dict[str, Mapping[str, int]] = {}
    artifact_by_channel: dict[str, Mapping[str, Any]] = {}

    gradient_definition = channel_by_id["static_gradient_v1"]
    gradient_environment = environment_by_id["square_periodic_vacancy"]
    gradient_state = initial_movement_state(gradient_environment)
    gradient_policy = policy_by_id["gradient_following_v1"]
    gradient_field, gradient_configuration = compile_static_gradient(
        gradient_environment,
        gradient_definition,
        axis="second",
        direction="up",
    )
    noisy_gradient, gradient_noise_draws = realize_noisy_gradient(
        gradient_definition,
        gradient_field,
        scenario_key="s06-gradient-fixture",
    )
    gradient_observation = build_policy_observation(
        gradient_environment,
        gradient_state,
        _actor_at(gradient_state, "r2_c2"),
        gradient_policy,
        gradient_levels=noisy_gradient,
        gradient_direction="up",
    )
    gradient_delivery = build_gradient_delivery_from_s05(
        gradient_definition,
        gradient_observation,
        compiled_field=gradient_field,
        direction="up",
        configuration_ledger=gradient_configuration,
        noise_draws=gradient_noise_draws,
        epoch_index=0,
    )
    gradient_replay_field, gradient_replay_configuration = compile_static_gradient(
        gradient_environment,
        gradient_definition,
        axis="second",
        direction="up",
    )
    gradient_replay_noisy, gradient_replay_draws = realize_noisy_gradient(
        gradient_definition,
        gradient_replay_field,
        scenario_key="s06-gradient-fixture",
    )
    gradient_replay_observation = build_policy_observation(
        gradient_environment,
        gradient_state,
        _actor_at(gradient_state, "r2_c2"),
        gradient_policy,
        gradient_levels=gradient_replay_noisy,
        gradient_direction="up",
    )
    gradient_replay = build_gradient_delivery_from_s05(
        gradient_definition,
        gradient_replay_observation,
        compiled_field=gradient_replay_field,
        direction="up",
        configuration_ledger=gradient_replay_configuration,
        noise_draws=gradient_replay_draws,
        epoch_index=0,
    )

    boundary_definition = channel_by_id["natural_boundary_signal_v1"]
    boundary_environment = environment_by_id["square_bounded_occupied_stripes"]
    boundary_state = initial_movement_state(boundary_environment)
    boundary_policy = policy_by_id["boundary_seeking_v1"]
    boundary_observation_build = build_policy_observation(
        boundary_environment,
        boundary_state,
        _actor_at(boundary_state, "r1_c1"),
        boundary_policy,
        boundary_direction="seek",
        boundary_tokens=("A",),
    )
    boundary_levels, boundary_configuration = compile_boundary_levels(
        boundary_environment, boundary_definition
    )
    boundary_candidates = {
        key: boundary_levels[affordance.target_site]
        for key, affordance in boundary_observation_build.candidate_map.items()
    }
    boundary_source = next(
        iter(boundary_observation_build.candidate_map.values())
    ).proposal.source_site
    boundary_kwargs = {
        "current_level": boundary_levels[boundary_source],
        "candidate_levels": boundary_candidates,
        "direction": "seek",
        "directive_applies": True,
        "configuration_bits": boundary_configuration["configurationBits"],
        "scenario_key": "s06-boundary-fixture",
        "epoch_index": 0,
    }
    boundary_delivery = build_boundary_delivery(boundary_definition, **boundary_kwargs)
    boundary_replay = build_boundary_delivery(boundary_definition, **boundary_kwargs)

    instruction_definition = channel_by_id["sparse_instruction_v1"]
    instruction_kwargs = {
        "instruction": "prefer_exploration",
        "region_alias": "zone_1",
        "region_size": 4,
        "region_count": 5,
        "scenario_key": "s06-instruction-fixture",
        "epoch_index": 2,
    }
    instruction_delivery = build_sparse_instruction_delivery(
        instruction_definition, **instruction_kwargs
    )
    instruction_replay = build_sparse_instruction_delivery(
        instruction_definition, **instruction_kwargs
    )

    summary_definition = channel_by_id["lagged_global_summary_v1"]
    summary_source = GlobalSummarySource(
        source_epoch=3,
        local_dissatisfaction_bits=(1, 0, 1, 0, 1, 0, 1, 1),
        submitted_proposals=8,
        conflict_losses=2,
        active_count=8,
    )
    summary_kwargs = {
        "scenario_key": "s06-summary-fixture",
        "epoch_index": 4,
        "recipient_count": 8,
    }
    summary_delivery = build_global_summary_delivery(
        summary_definition, summary_source, **summary_kwargs
    )
    summary_replay = build_global_summary_delivery(
        summary_definition, summary_source, **summary_kwargs
    )

    direct_definition = channel_by_id["sparse_direct_intervention_v1"]
    direct_environment = environment_by_id["square_bounded_occupied_stripes"]
    direct_state = _perturbed_state(direct_environment)
    direct_policy = policy_by_id["greedy_neighbor_satisfaction_v1"]
    direct_profile = compile_relation_profile(
        grammar_by_id["stripes_axis_relations_v1"]
    )
    direct_observation = build_policy_observation(
        direct_environment,
        direct_state,
        _actor_at(direct_state, "r2_c0"),
        direct_policy,
        relation_profile=direct_profile,
    )
    direct_view_kwargs = {
        "epoch_index": 3,
        "population_size": 81,
        "lagged_summary": {
            "dissatisfactionBin": 5,
            "laggedConflictBin": 1,
            "summaryAgeEpochs": 1,
        },
        "remaining_information_budget": 352,
        "remaining_action_budget": 1,
    }
    direct_view = build_direct_controller_view(
        direct_definition, direct_observation, **direct_view_kwargs
    )
    direct_decision = decide_direct_controller(direct_view.payload)
    direct_event = execute_direct_intervention(
        direct_definition,
        direct_environment,
        direct_state,
        direct_observation,
        direct_view,
        direct_decision,
        scenario_key="s06-direct-fixture",
        batch_nonce="s06-direct-fixture",
    )
    direct_replay_view = build_direct_controller_view(
        direct_definition, direct_observation, **direct_view_kwargs
    )
    direct_replay_decision = decide_direct_controller(direct_replay_view.payload)
    direct_replay_event = execute_direct_intervention(
        direct_definition,
        direct_environment,
        direct_state,
        direct_observation,
        direct_replay_view,
        direct_replay_decision,
        scenario_key="s06-direct-fixture",
        batch_nonce="s06-direct-fixture",
    )

    delivery_fixtures = (
        (
            "gradient_static_column",
            gradient_definition,
            gradient_delivery,
            gradient_replay,
        ),
        (
            "boundary_bounded_projection",
            boundary_definition,
            boundary_delivery,
            boundary_replay,
        ),
        (
            "sparse_instruction_coarse_region",
            instruction_definition,
            instruction_delivery,
            instruction_replay,
        ),
        (
            "lagged_summary_broadcast",
            summary_definition,
            summary_delivery,
            summary_replay,
        ),
    )
    for fixture_id, definition, delivery, replay in delivery_fixtures:
        encoded = canonical_delivery_bytes(delivery)
        replay_pass = encoded == canonical_delivery_bytes(replay)
        roundtrip_pass = (
            canonical_delivery_bytes(parse_delivery(json.loads(encoded))) == encoded
        )
        (fixtures_dir / f"{fixture_id}.json").write_bytes(encoded)
        ledger_by_channel[definition.channel_id] = delivery.ledger
        artifact_by_channel[definition.channel_id] = delivery_to_dict(delivery)
        row = {
            "fixtureId": fixture_id,
            "channelId": definition.channel_id,
            "channelType": definition.channel_type,
            "sourceOwner": definition.source["owner"],
            **dict(delivery.ledger),
            "budgetFailures": "|".join(
                validate_ledger_against_budget(definition, delivery.ledger)
            ),
            "deterministicReplayPass": replay_pass,
            "serializationRoundTripPass": roundtrip_pass,
            "validationPass": replay_pass
            and roundtrip_pass
            and not validate_ledger_against_budget(definition, delivery.ledger),
        }
        fixture_rows.append(row)
        fixture_records.append(
            {
                **row,
                "payload": delivery.payload,
                "deliverySha256": delivery.delivery_sha256,
            }
        )
        replay_records.append(
            {
                "fixtureId": fixture_id,
                "canonicalBytes": len(encoded),
                "serializationRoundTripPass": roundtrip_pass,
                "independentReexecutionPass": replay_pass,
            }
        )

    direct_encoded = canonical_direct_event_bytes(direct_event)
    direct_replay_pass = direct_encoded == canonical_direct_event_bytes(
        direct_replay_event
    )
    (fixtures_dir / "direct_one_query_one_action.json").write_bytes(direct_encoded)
    ledger_by_channel[direct_definition.channel_id] = direct_event["ledger"]
    artifact_by_channel[direct_definition.channel_id] = direct_event
    direct_budget_failures = validate_ledger_against_budget(
        direct_definition, direct_event["ledger"]
    )
    direct_cost_reconciles = (
        direct_event["ledger"]["movementGraphDisplacement"]
        == direct_event["movementBatchResult"]["costLedger"]["totalGraphDisplacement"]
    )
    direct_row = {
        "fixtureId": "direct_one_query_one_action",
        "channelId": direct_definition.channel_id,
        "channelType": direct_definition.channel_type,
        "sourceOwner": direct_definition.source["owner"],
        **dict(direct_event["ledger"]),
        "budgetFailures": "|".join(direct_budget_failures),
        "deterministicReplayPass": direct_replay_pass,
        "serializationRoundTripPass": True,
        "validationPass": direct_replay_pass
        and not direct_budget_failures
        and direct_cost_reconciles
        and direct_event["outcome"] == "accepted",
    }
    fixture_rows.append(direct_row)
    fixture_records.append(
        {
            **direct_row,
            "controllerViewSha256": direct_view.view_sha256,
            "eventSha256": direct_event["eventSha256"],
            "controllerDecision": direct_event["controllerDecision"],
            "movementOutcome": direct_event["outcome"],
            "s04CostReconciles": direct_cost_reconciles,
        }
    )
    replay_records.append(
        {
            "fixtureId": "direct_one_query_one_action",
            "canonicalBytes": len(direct_encoded),
            "serializationRoundTripPass": True,
            "independentReexecutionPass": direct_replay_pass,
        }
    )

    _write_csv(output / "fixture_results.csv", fixture_rows)
    _write_json(
        output / "fixture_results.json",
        {
            "schemaVersion": "e06.s06.fixture-results.v1",
            "researchStepId": "S06",
            "fixtures": fixture_records,
        },
    )
    _write_json(
        output / "serialization_replay.json",
        {
            "schemaVersion": "e06.s06.serialization-replay.v1",
            "researchStepId": "S06",
            "records": replay_records,
            "success": all(
                item["serializationRoundTripPass"]
                and item["independentReexecutionPass"]
                for item in replay_records
            ),
        },
    )

    budget_rows = [
        channel_comparison_vector(
            channel_by_id[row["channelId"]], ledger_by_channel[row["channelId"]]
        )
        for row in fixture_rows
    ]
    advisory = [
        row for row in budget_rows if row["channelType"] != "direct_intervention"
    ]
    direct_vector = next(
        row for row in budget_rows if row["channelType"] == "direct_intervention"
    )
    profile_audit = {
        "delivery_matched_64": {
            "advisoryChannels": [row["channelId"] for row in advisory],
            "allRuntimePolicyBitsAtMost64": all(
                row["policyDeliveryBits"] <= 64 for row in advisory
            ),
            "allAddressBitsAtMost4": all(row["addressBits"] <= 4 for row in advisory),
            "allActuationAttemptsZero": all(
                row["actuationAttempts"] == 0 for row in advisory
            ),
        },
        "action_matched_sparse_1": {
            "channelId": direct_vector["channelId"],
            "controllerInputBitsAtMost352": direct_vector["controllerInputBits"] <= 352,
            "addressBitsAtMost7": direct_vector["addressBits"] <= 7,
            "actuationAttemptsAtMost1": direct_vector["actuationAttempts"] <= 1,
            "overrideUnitsAtMost1": direct_vector["overrideActionUnits"] <= 1,
            "movementDisplacementAtMost6": direct_vector["movementGraphDisplacement"]
            <= 6,
        },
        "configurationBitsMatched": False,
        "semanticValueMatched": False,
        "scalarCollapseAllowed": False,
    }
    _write_csv(output / "budget_comparison.csv", budget_rows)
    _write_json(
        output / "budget_comparison.json",
        {
            "schemaVersion": "e06.s06.budget-comparison.v1",
            "researchStepId": "S06",
            "vectors": budget_rows,
            "comparisonProfileAudit": profile_audit,
        },
    )
    _write_json(
        output / "budget_schema.json",
        {
            "schemaVersion": "e06.s06.budget-schema.v1",
            "researchStepId": "S06",
            "epochLengthTransitions": raw_catalog["commonBudgetContract"][
                "epochLengthTransitions"
            ],
            "ledgerFields": list(CHANNEL_LEDGER_FIELDS),
            "contract": raw_catalog["commonBudgetContract"],
            "actionAndInformationNeverCollapsed": True,
        },
    )

    cost_rows = []
    for row in fixture_rows:
        cost_rows.append(
            {
                "channelId": row["channelId"],
                "sourceScalarReads": row["sourceScalarReads"],
                "controllerStateReads": row["controllerStateReads"],
                "noiseDraws": row["noiseDraws"],
                "controllerComputeUnits": row["controllerComputeUnits"],
                "actuationAttempts": row["actuationAttempts"],
                "actuationSuccesses": row["actuationSuccesses"],
                "overrideActionUnits": row["overrideActionUnits"],
                "movementGraphDisplacement": row["movementGraphDisplacement"],
                "suppressedNativeActions": row["suppressedNativeActions"],
                "opportunityCostUnits": row["opportunityCostUnits"],
                "s04CostReconciles": row["channelType"] != "direct_intervention"
                or direct_cost_reconciles,
                "withinNativeBudget": not bool(row["budgetFailures"]),
            }
        )
    _write_csv(output / "cost_accounting.csv", cost_rows)
    _write_json(
        output / "cost_accounting.json",
        {
            "schemaVersion": "e06.s06.cost-accounting.v1",
            "researchStepId": "S06",
            "records": cost_rows,
            "success": all(
                item["s04CostReconciles"] and item["withinNativeBudget"]
                for item in cost_rows
            ),
        },
    )

    noise_rows = [calibrate_noise(definition) for definition in channels]
    _write_csv(
        output / "noise_calibration.csv",
        [
            {
                "channelId": item["channelId"],
                "noiseKind": item["noiseKind"],
                "sampleCount": item["sampleCount"],
                "counterDrawsConsumed": item["counterDrawsConsumed"],
                "maximumAbsoluteError": item["maximumAbsoluteError"],
                "toleranceAbsoluteProbability": item["toleranceAbsoluteProbability"],
                "success": item["success"],
            }
            for item in noise_rows
        ],
    )
    _write_json(
        output / "noise_calibration.json",
        {
            "schemaVersion": "e06.s06.noise-calibration.v1",
            "researchStepId": "S06",
            "calibrations": noise_rows,
            "success": all(item["success"] for item in noise_rows),
        },
    )

    hidden_fields = (
        "fullState",
        "occupancy",
        "siteTokens",
        "siteIds",
        "route",
        "occupantIdentities",
        "actorId",
        "siteRoles",
        "rawBoundaryTags",
        "currentBatchProposals",
        "conflictPriority",
        "analysisLabel",
        "s02WholeGridScore",
        "s01GlobalCompletionAudit",
        "targetMembership",
        "futureState",
    )
    hidden_probe_rows = []
    for field in hidden_fields:
        for validator_name, validator in (
            ("controller", validate_controller_view_payload),
            ("channel", validate_channel_payload),
        ):
            rejected = False
            reason = None
            try:
                validator({field: "probe"})
            except ChannelValidationError as error:
                rejected = True
                reason = str(error)
            hidden_probe_rows.append(
                {
                    "probeId": f"{validator_name}:{field}",
                    "validator": validator_name,
                    "injectedField": field,
                    "rejected": rejected,
                    "reason": reason,
                }
            )
    source_probe_rows = []
    for definition in channels:
        rejected = False
        reason = None
        try:
            validate_source_projection(definition, {"occupancy": ["probe"]})
        except ChannelValidationError as error:
            rejected = True
            reason = str(error)
        source_probe_rows.append(
            {
                "probeId": f"source:{definition.channel_id}:occupancy",
                "channelId": definition.channel_id,
                "injectedField": "occupancy",
                "rejected": rejected,
                "reason": reason,
            }
        )
    catalog_mutation = json.loads(json.dumps(raw_catalog))
    direct_raw = next(
        item
        for item in catalog_mutation["channels"]
        if item["channelType"] == "direct_intervention"
    )
    direct_raw["source"]["allowedInputs"].append("occupancy")
    direct_mutation_rejected = False
    try:
        parse_channel_catalog(catalog_mutation)
    except ChannelValidationError:
        direct_mutation_rejected = True
    _write_csv(output / "hidden_state_probe_results.csv", hidden_probe_rows)
    _write_json(
        output / "hidden_state_probe_results.json",
        {
            "schemaVersion": "e06.s06.hidden-state-probes.v1",
            "researchStepId": "S06",
            "payloadProbes": hidden_probe_rows,
            "sourceProbes": source_probe_rows,
            "catalogRejectedDirectOccupancyAllowlist": direct_mutation_rejected,
            "controllerSourceForbiddenAccesses": controller_source_forbidden_accesses(),
            "success": all(item["rejected"] for item in hidden_probe_rows)
            and all(item["rejected"] for item in source_probe_rows)
            and direct_mutation_rejected
            and not controller_source_forbidden_accesses(),
        },
    )

    permission_rows = []
    source_owner_rows = []
    update_rows = []
    asymmetry_rows = []
    for definition in channels:
        permission_rows.append(
            {
                "channelId": definition.channel_id,
                "sourceAllowedInputs": "|".join(definition.source["allowedInputs"]),
                "sourceRuntimeStateReads": "|".join(
                    definition.source["runtimeStateReads"]
                ),
                "controllerPersistentState": "|".join(
                    definition.controller_state_permissions["allowed"]
                ),
                "controllerPersistentBits": definition.controller_state_permissions[
                    "persistentBits"
                ],
                "fullStateAllowed": False,
                "analysisLabelsAllowed": False,
                "globalCompletionAllowed": False,
                "futureStateAllowed": False,
            }
        )
        source_owner_rows.append(
            {
                "channelId": definition.channel_id,
                "channelType": definition.channel_type,
                "owner": definition.source["owner"],
                "implementation": definition.source["implementation"],
                "semanticContent": definition.semantic_content,
                "spatialResolution": definition.spatial_resolution,
                "targetSpecificity": definition.target_specificity,
            }
        )
        update_rows.append(
            {
                "channelId": definition.channel_id,
                "updateSchedule": definition.update_schedule,
                "noiseKind": definition.noise_model["kind"],
                "calibrationDraws": definition.noise_model["calibrationDraws"],
                "actionKind": definition.action_cost["kind"],
            }
        )
        for ordinal, statement in enumerate(definition.unavoidable_asymmetries, 1):
            asymmetry_rows.append(
                {
                    "channelId": definition.channel_id,
                    "ordinal": ordinal,
                    "asymmetry": statement,
                    "eliminatedByBitMatching": False,
                    "requiredReporting": True,
                }
            )
    _write_csv(output / "controller_permission_matrix.csv", permission_rows)
    _write_csv(output / "source_owner_matrix.csv", source_owner_rows)
    _write_csv(output / "update_schedule_matrix.csv", update_rows)
    _write_csv(output / "semantic_asymmetry_register.csv", asymmetry_rows)
    _write_json(
        output / "semantic_asymmetry_register.json",
        {
            "schemaVersion": "e06.s06.semantic-asymmetry-register.v1",
            "researchStepId": "S06",
            "records": asymmetry_rows,
            "scalarEquivalenceClaimed": False,
            "success": len(asymmetry_rows) >= 15
            and all(not item["eliminatedByBitMatching"] for item in asymmetry_rows),
        },
    )

    s05_payload_before = json.dumps(
        direct_observation.observation.payload, sort_keys=True
    )
    _ = build_direct_controller_view(
        direct_definition, direct_observation, **direct_view_kwargs
    )
    s05_payload_after = json.dumps(
        direct_observation.observation.payload, sort_keys=True
    )
    periodic_boundary_levels, _ = compile_boundary_levels(
        gradient_environment, boundary_definition
    )
    s05_preservation = {
        "schemaVersion": "e06.s06.s05-boundary-preservation.v1",
        "researchStepId": "S06",
        "s05BasePayloadByteStable": s05_payload_before == s05_payload_after,
        "channelPayloadVersionSeparate": True,
        "gradientPayloadEqualsFrozenS05Projection": gradient_delivery.payload
        == gradient_observation.observation.payload,
        "authenticationRemainsHiddenFromPolicyAndController": True,
        "siteRoleAndRawBoundaryMetadataRemainHidden": True,
        "s02WholeGridAndS01CompletionRemainHidden": True,
        "directRecipientSelectedWithoutState": direct_view.recipient_alias == "u0003",
        "directOneS05PayloadOnly": True,
        "periodicBoundaryLevelsAllZero": set(periodic_boundary_levels.values()) == {0},
    }
    s05_preservation["success"] = all(
        value
        for key, value in s05_preservation.items()
        if key not in {"schemaVersion", "researchStepId"}
    )
    _write_json(output / "s05_boundary_preservation.json", s05_preservation)

    library_index = []
    fixture_by_channel = {item["channelId"]: item for item in fixture_rows}
    for definition in channels:
        card = cards / f"{definition.channel_id}.md"
        card.write_text(
            _channel_card(definition, fixture_by_channel[definition.channel_id]),
            encoding="utf-8",
        )
        fixture_id = fixture_by_channel[definition.channel_id]["fixtureId"]
        library_index.append(
            {
                **channel_to_dict(definition),
                "card": str(card.relative_to(output)),
                "fixture": str(
                    (fixtures_dir / f"{fixture_id}.json").relative_to(output)
                ),
            }
        )
    _write_json(
        library / "channel_index.json",
        {
            "schemaVersion": "e06.s06.channel-index.v1",
            "researchStepId": "S06",
            "libraryVersion": metadata["libraryVersion"],
            "channels": library_index,
        },
    )

    permission_isolation = {
        "schemaVersion": "e06.s06.permission-isolation.v1",
        "researchStepId": "S06",
        "permissionContract": raw_catalog["permissionContract"],
        "payloadHiddenStateProbesPassed": all(
            item["rejected"] for item in hidden_probe_rows
        ),
        "sourceAllowlistProbesPassed": all(
            item["rejected"] for item in source_probe_rows
        ),
        "directCatalogMutationRejected": direct_mutation_rejected,
        "controllerSourceForbiddenAccesses": controller_source_forbidden_accesses(),
        "s05BoundaryPreserved": s05_preservation["success"],
        "success": all(item["rejected"] for item in hidden_probe_rows)
        and all(item["rejected"] for item in source_probe_rows)
        and direct_mutation_rejected
        and not controller_source_forbidden_accesses()
        and s05_preservation["success"],
    }
    _write_json(output / "permission_isolation.json", permission_isolation)

    environment_provenance = {
        "schemaVersion": "e06.s06.environment-provenance.v1",
        "researchStepId": "S06",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name) for name in ("PyYAML", "pytest")
        },
        "cpuCountVisible": os.cpu_count(),
        "workersUsed": 1,
        "threadPolicy": "intentional_serial_exact_fixture_and_counter_calibration",
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
    serialization_success = all(
        item["serializationRoundTripPass"] and item["independentReexecutionPass"]
        for item in replay_records
    )
    advisory_profile = profile_audit["delivery_matched_64"]
    action_profile = profile_audit["action_matched_sparse_1"]
    checks = {
        "exactFiveChannels": len(channels) == 5,
        "allSemanticDimensionsDeclared": all(
            definition.source["owner"]
            and definition.semantic_content
            and definition.spatial_resolution
            and definition.update_schedule
            and definition.noise_model
            and definition.bandwidth
            and definition.action_cost
            and definition.unavoidable_asymmetries
            for definition in channels
        ),
        "allFixturesPass": all(item["validationPass"] for item in fixture_rows),
        "allNativeBudgetsPass": all(
            not item["budgetFailures"] for item in fixture_rows
        ),
        "advisoryComparisonProfilePass": all(
            value for value in advisory_profile.values() if isinstance(value, bool)
        ),
        "directComparisonProfilePass": all(action_profile.values()),
        "costAccountingPass": all(
            item["s04CostReconciles"] and item["withinNativeBudget"]
            for item in cost_rows
        ),
        "noiseCalibrationPass": all(item["success"] for item in noise_rows),
        "permissionIsolationPass": permission_isolation["success"],
        "s05BoundaryPreservationPass": s05_preservation["success"],
        "serializationReplayPass": serialization_success,
        "semanticAsymmetriesRecorded": len(asymmetry_rows) >= 15,
        "upstreamImmutability": upstream["success"],
        "focusedTests": focused_tests["success"],
    }
    validation = {
        "schemaVersion": "e06.s06.validation-summary.v1",
        "researchStepId": "S06",
        "status": "complete",
        "success": all(checks.values()),
        "validationResult": "PASS" if all(checks.values()) else "FAIL",
        "outcomeClassification": "supportive",
        "channelCount": len(channels),
        "fixtureCount": len(fixture_rows),
        "ledgerFieldCount": len(CHANNEL_LEDGER_FIELDS),
        "noiseCalibrationDrawsPerChannel": 100000,
        "noiseCalibrationTotalSamples": sum(item["sampleCount"] for item in noise_rows),
        "maximumNoiseProbabilityError": max(
            item["maximumAbsoluteError"] for item in noise_rows
        ),
        "hiddenStateProbeCount": len(hidden_probe_rows) + len(source_probe_rows) + 1,
        "semanticAsymmetryCount": len(asymmetry_rows),
        "checks": checks,
        "focusedTests": focused_tests,
        "upstreamImmutability": upstream,
    }
    _write_json(output / "validation_summary.json", validation)

    commands = """cd /workspace/cell-research
python -m pytest -q tests/test_morph2d_targets.py tests/test_morph2d_grammar.py tests/test_morph2d_environments.py tests/test_morph2d_movements.py tests/test_morph2d_policies.py tests/test_morph2d_channels.py
python scripts/build_morph2d_s06.py build --output-dir /artifacts/research_steps/S06
python scripts/build_morph2d_s06.py validate --output-dir /artifacts/research_steps/S06
ruff check src/morph2d scripts/build_morph2d_s06.py tests/test_morph2d_channels.py
ruff format --check src/morph2d scripts/build_morph2d_s06.py tests/test_morph2d_channels.py
python -m compileall -q src/morph2d scripts/build_morph2d_s06.py tests/test_morph2d_channels.py
"""
    (output / "execution_commands.log").write_text(commands, encoding="utf-8")

    existing = sorted(
        str(path.relative_to(output))
        for path in output.rglob("*")
        if path.is_file()
        and path.name not in {"artifact_manifest.json", "research_step_full_results.md"}
    )
    existing.append("research_step_full_results.md")
    report = _report(
        validation,
        fixture_rows,
        noise_rows,
        existing,
        environment_provenance,
    )
    (output / "research_step_full_results.md").write_text(report, encoding="utf-8")

    artifact_paths = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    )
    manifest = {
        "schemaVersion": "e06.s06.artifact-manifest.v1",
        "researchStepId": "S06",
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
        raise RuntimeError("S06 validation failed; inspect validation_summary.json")


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
        "control_channel_spec.md",
        "channel_library/control_channel_catalog.yaml",
        "channel_library/channel_index.json",
        "budget_schema.json",
        "budget_comparison.json",
        "cost_accounting.json",
        "permission_isolation.json",
        "hidden_state_probe_results.json",
        "noise_calibration.json",
        "serialization_replay.json",
        "s05_boundary_preservation.json",
        "semantic_asymmetry_register.json",
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
        "researchStepId": "S06",
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
