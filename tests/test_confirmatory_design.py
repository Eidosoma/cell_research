from __future__ import annotations

from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.analyze_confirmatory_look import bh_adjust, classify
from scripts.freeze_confirmatory_design import (
    FAULT_PROFILES,
    POLICY_DIRECTIONS,
    _balanced_permutation,
)
from scripts.validate_confirmatory_replay import load_arrow_records


def test_confirmatory_balanced_permutation_is_deterministic_and_bounded() -> None:
    first = _balanced_permutation(FAULT_PROFILES, 200, address="S11/test/fault")
    second = _balanced_permutation(FAULT_PROFILES, 200, address="S11/test/fault")
    assert first == second
    counts = Counter(first)
    assert set(counts) == set(FAULT_PROFILES)
    assert max(counts.values()) - min(counts.values()) == 1


def test_confirmatory_policy_direction_allocation_preserves_all_levels() -> None:
    values = _balanced_permutation(POLICY_DIRECTIONS, 200, address="S11/test/pd")
    counts = Counter(values)
    assert set(counts) == set(POLICY_DIRECTIONS)
    assert max(counts.values()) - min(counts.values()) == 1


def test_terminal_classification_respects_direction_equivalence_and_precision() -> None:
    assert classify(0.03, 0.021, 0.039, "positive", True) == (
        "directionally_reproduced_material", False
    )
    assert classify(0.005, 0.001, 0.009, "positive", True) == (
        "directionally_reproduced_small", True
    )
    assert classify(0.0, -0.01, 0.01, "negative", True) == (
        "practically_equivalent_to_zero", True
    )
    assert classify(-0.03, -0.04, -0.02, "positive", True) == (
        "contradictory", False
    )
    assert classify(0.03, 0.01, 0.05, "positive", False)[0] == "inconclusive_due_to_precision"


def test_bh_adjustment_is_monotone_in_sorted_p_values() -> None:
    raw = [0.01, 0.04, 0.03, 0.20]
    adjusted = bh_adjust(raw)
    ordered = sorted(zip(raw, adjusted))
    assert all(left[1] <= right[1] for left, right in zip(ordered, ordered[1:]))
    assert all(q >= p for p, q in zip(raw, adjusted))
    assert all(0 <= q <= 1 for q in adjusted)


def test_replay_loader_preserves_nested_lists(tmp_path) -> None:
    pairing_path = tmp_path / "pairing.parquet"
    scenario_path = tmp_path / "scenarios.parquet"
    pq.write_table(
        pa.Table.from_pylist([{"pairingBlockId": "pair-1", "faultRanks": [1, 3]}]),
        pairing_path,
    )
    pq.write_table(
        pa.Table.from_pylist([
            {
                "inputScenarioId": "input-1",
                "split": "confirmatory_holdout",
                "values": [3, 1, 2],
            },
            {"inputScenarioId": "input-2", "split": "screening", "values": [1]},
        ]),
        scenario_path,
    )

    pairing, scenarios = load_arrow_records(pairing_path, scenario_path, ["input-1"])

    assert pairing["pair-1"]["faultRanks"] == [1, 3]
    assert scenarios["input-1"]["values"] == [3, 1, 2]
    assert set(scenarios) == {"input-1"}
