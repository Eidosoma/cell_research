from __future__ import annotations

import pytest

from causal_simulator import evaluate_no_fault_parity
from reference_simulator.api import create_scenario


@pytest.mark.parametrize("policy", ["Bubble", "Insertion", "Selection"])
@pytest.mark.parametrize("direction", ["ascending", "descending"])
def test_e02_s01_e02_no_fault_exact_parity(policy: str, direction: str) -> None:
    scenario = create_scenario(
        [4, 1, 3, 2],
        policy=policy,
        direction=direction,
        seed=20260714,
        max_activations=500,
        generation_key=f"E02-S01-E02/{policy}/{direction}",
        permute=False,
    )
    parity = evaluate_no_fault_parity(scenario, trace_mode="full")
    assert parity.success, parity.comparisons
    assert parity.distributed.result.to_json_bytes() == parity.central_local.result.to_json_bytes()
