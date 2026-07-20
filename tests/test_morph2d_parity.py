from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import torch
import yaml

from src.morph2d.engine import (
    canonical_episode_result_bytes,
    load_engine_context,
    run_cpu_episode,
)
from src.morph2d.environments import parse_environment_spec
from src.morph2d.movements import resolve_batch
from src.morph2d.parity import (
    TransitionCase,
    compare_transition_cases,
    counter_permuted_state,
    enumerate_legal_proposals,
    exhaustive_identity_states,
    run_differential_episode,
)


ROOT = Path(__file__).resolve().parents[1]


def _device() -> str:
    return "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"


def _context():
    return load_engine_context(
        ROOT / "configs/morphologies/engine_catalog.yaml",
        environment_catalog=ROOT / "configs/morphologies/environment_catalog.yaml",
        policy_catalog=ROOT / "configs/morphologies/policy_catalog.yaml",
        grammar_catalog=ROOT / "configs/morphologies/grammar_catalog.yaml",
        channel_catalog=ROOT / "configs/morphologies/control_channel_catalog.yaml",
    )


def _small_environments():
    raw = yaml.safe_load(
        (ROOT / "configs/morphologies/parity_catalog.yaml").read_text()
    )
    return [parse_environment_spec(item) for item in raw["smallEnvironments"]]


def test_parity_catalog_requires_zero_tolerance_and_no_float_escape() -> None:
    raw = yaml.safe_load(
        (ROOT / "configs/morphologies/parity_catalog.yaml").read_text()
    )
    requirements = raw["parityRequirements"]
    assert requirements["integerTransitionAbsoluteTolerance"] == 0
    assert requirements["deterministicEpisodeCanonicalByteTolerance"] == 0
    assert requirements["pairedDistributionOutcomeTolerance"] == 0
    assert requirements["floatingGradient"] == {
        "applicable": False,
        "absoluteTolerance": None,
        "relativeTolerance": None,
        "reason": "Current S06 gradients, jitter, observations, decisions, and S07 transition tensors are integer-valued; no floating gradient enters the engine.",
    }


def test_small_environment_state_and_intent_spaces_are_exhaustive() -> None:
    square, vacancy, triangle = _small_environments()
    assert len(exhaustive_identity_states(square)) == 24
    assert len(exhaustive_identity_states(vacancy)) == 24
    assert len(exhaustive_identity_states(triangle)) == 6
    kinds = {
        item.kind
        for environment in (square, vacancy, triangle)
        for state in exhaustive_identity_states(environment)[:1]
        for item in enumerate_legal_proposals(environment, state)
    }
    assert kinds == {"adjacent_swap", "vacancy_move", "short_exchange", "rotation"}


def test_counter_permuted_state_replays_and_preserves_fixed_roles() -> None:
    context = _context()
    environment = context.environments["square_bounded_fixed_boundary"]
    first = counter_permuted_state(environment, "fixture-7", transition_index=7)
    second = counter_permuted_state(environment, "fixture-7", transition_index=7)
    assert first == second
    assert {
        site_id: occupant.occupant_id
        for site_id, occupant in first.occupancy
        if occupant.kind == "fixed_boundary"
    } == {
        site_id: occupant.occupant_id
        for site_id, occupant in second.occupancy
        if occupant.kind == "fixed_boundary"
    }


def test_exact_gpu_transition_and_mismatch_preservation_harness() -> None:
    environment = _small_environments()[2]
    state = exhaustive_identity_states(environment)[3]
    proposals = enumerate_legal_proposals(environment, state)[:2]
    expected = resolve_batch(
        environment, state, proposals, batch_nonce="s08-test-transition"
    )
    case = TransitionCase(
        case_id="s08-test-exact",
        phase="unit",
        state=state,
        proposals=proposals,
        batch_nonce="s08-test-transition",
        metadata={"test": True},
        expected_batch=expected,
    )
    rows, mismatches = compare_transition_cases(environment, [case], device=_device())
    assert rows[0]["success"]
    assert not mismatches

    corrupted = json.loads(json.dumps(expected))
    corrupted["transitionSha256"] = "0" * 64
    rows, mismatches = compare_transition_cases(
        environment,
        [replace(case, case_id="s08-test-injected", expected_batch=corrupted)],
        device=_device(),
    )
    assert not rows[0]["success"]
    assert mismatches[0]["rootCauseClassification"] == (
        "cpu_oracle_reexecution_divergence"
    )
    assert mismatches[0]["severity"] == "release_critical"


def test_complete_episode_shadow_does_not_change_cpu_oracle() -> None:
    context = _context()
    definition = replace(context.episodes[0], transitions=2)
    cpu = run_cpu_episode(context, definition, include_selected_traces=False)
    shadow, recorder = run_differential_episode(
        context,
        definition,
        device=_device(),
        phase="unit_episode",
        include_selected_traces=False,
    )
    assert canonical_episode_result_bytes(cpu) == canonical_episode_result_bytes(shadow)
    assert recorder.policy_rows
    assert recorder.transition_rows
    assert all(item["success"] for item in recorder.policy_rows)
    assert all(item["success"] for item in recorder.transition_rows)
    assert not recorder.mismatches
