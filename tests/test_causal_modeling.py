from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from causal_simulator.causal_modeling import (
    aalen_johansen,
    bh_adjust,
    build_bootstrap_weights,
    hash_fold,
    holm_adjust,
    interval_specific_completion_rates,
)
from scripts.run_causal_effects import heterogeneity_support_outputs


def test_aalen_johansen_matches_hand_calculated_competing_fixture() -> None:
    result = aalen_johansen([0.25, 0.5, 1.0], [1, 2, 0])
    assert np.isclose(result.cif_at_horizon, 1 / 3)
    assert np.isclose(result.restricted_mean_completion_free, 0.75)
    assert np.allclose(result.cif_completion, [0.0, 1 / 3, 1 / 3])


def test_multiplicity_adjustments_are_monotone_and_not_below_raw() -> None:
    raw = np.array([0.01, 0.04, 0.03, 0.20])
    for adjusted in (holm_adjust(raw), bh_adjust(raw)):
        assert np.all(adjusted >= raw)
        ordered = adjusted[np.argsort(raw)]
        assert np.all(np.diff(ordered) >= -1e-15)
        assert np.all((0 <= adjusted) & (adjusted <= 1))


def test_bootstrap_weights_are_deterministic_and_scale_stratified() -> None:
    blocks = pd.DataFrame({
        "pairingBlockId": [f"pb-{n}-{index}" for n in (20, 50, 100, 200, 500) for index in range(400)],
        "n": [n for n in (20, 50, 100, 200, 500) for _ in range(400)],
    })
    first = build_bootstrap_weights(blocks, "freeze-test", 3)
    second = build_bootstrap_weights(blocks, "freeze-test", 3)
    assert np.array_equal(first, second)
    assert first.shape == (3, 2000)
    assert np.all(first.sum(axis=1) == 2000)
    for offset in range(0, 2000, 400):
        assert np.all(first[:, offset:offset + 400].sum(axis=1) == 400)


def test_piecewise_completion_exposure_stops_at_terminal_time() -> None:
    arm = pd.DataFrame({
        "normalizedCompletionTime": [0.1, 0.5, 1.0],
        "completionObserved": [1, 0, 0],
    })
    rates = interval_specific_completion_rates(arm)
    assert rates.completionEvents.tolist() == [1, 0, 0]
    assert np.allclose(rates.normalizedExposure, [0.6, 0.75, 0.25])


def test_fold_address_is_stable_and_bounded() -> None:
    values = [hash_fold("freeze", f"block-{index}") for index in range(100)]
    assert values == [hash_fold("freeze", f"block-{index}") for index in range(100)]
    assert set(values).issubset(set(range(5)))
    assert len(set(values)) == 5


def test_s12_prespecification_freezes_required_boundaries() -> None:
    path = Path(__file__).resolve().parents[1] / "design/s12/modeling_prespecification.json"
    spec = json.loads(path.read_text())
    assert spec["frozenBeforeOutcomeModeling"]
    assert spec["outcomeModelFitsBeforeFreeze"] == 0
    assert spec["contractPreservation"]["selectedEstimandsInOrder"] == [
        "E02-S01-E04", "E02-S01-E06", "E02-S01-E03", "E02-S01-E08"
    ]
    assert spec["bootstrap"]["replicates"] == 2000
    assert spec["diagnostics"]["survival"]["alternativeIntervalsFrozenBeforeModeling"]
    assert not spec["contractPreservation"][
        "legalActionInformationSchedulerFaultAndOpportunityContractsMayChange"
    ]


def test_heterogeneity_range_ranking_and_support_rule() -> None:
    modifiers = {
        "n": [20, 20, 50, 50],
        "valueProfile": ["a", "a", "b", "b"],
        "orderStructure": ["a", "a", "b", "b"],
        "policyProfile": ["a", "a", "b", "b"],
        "direction": ["a", "a", "b", "b"],
        "placementClass": ["a", "a", "b", "b"],
        "faultCount": [1, 1, 2, 2],
    }
    paired = pd.DataFrame({
        "estimandId": ["E02-S01-E04"] * 4,
        **modifiers,
    })
    heterogeneity = pd.DataFrame([
        {
            "estimandId": "E02-S01-E04", "endpoint": "outcome",
            "modifier": "n", "level": "20", "pairCount": 120,
            "estimate": -0.2,
        },
        {
            "estimandId": "E02-S01-E04", "endpoint": "outcome",
            "modifier": "n", "level": "50", "pairCount": 120,
            "estimate": 0.1,
        },
    ])
    rankings, unsupported = heterogeneity_support_outputs(paired, heterogeneity)
    assert np.isclose(rankings.effectRange.iloc[0], 0.3)
    assert rankings.associationalRangeRank.iloc[0] == 1
    assert not unsupported.empty
    assert set(unsupported.action) == {
        "not_estimated_no_pooling_or_extrapolation"
    }
