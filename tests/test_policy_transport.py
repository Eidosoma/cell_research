from __future__ import annotations

from collections import Counter
import json
import numpy as np
import pytest

from analysis.policy_transport import (
    CALIBRATION_REPLICATES,
    CONTRACT,
    FIT_REPLICATES,
    S07_CACHE,
    STATIC_NAMES,
    TRANSPORT_NAMES,
    _read_jsonl,
    _residence,
    classify_split,
    extract_run,
    predict_model,
)


def _base_row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "regime": "native_control",
        "correlation_profile": "absent",
        "composition_class": "pairwise",
        "composition_profile": "p50_50",
        "first_policy_count": 50,
        "replicate_ordinal": 15,
    }
    row.update(updates)
    return row


def test_contract_carries_s11r_boundary_and_excludes_s11_shapley() -> None:
    contract = json.loads(CONTRACT.read_text())
    assert contract["researchStepId"] == "S12"
    assert "S11R" in contract["mechanisticConstraint"]["s11r"]
    assert "excluded" in contract["mechanisticConstraint"]["s11"].lower()
    assert not any("shapley" in name.lower() for name in STATIC_NAMES + TRANSPORT_NAMES)


def test_split_rules_are_disjoint() -> None:
    assert not set(FIT_REPLICATES) & set(CALIBRATION_REPLICATES)
    assert classify_split(_base_row()) == "development_fit"
    assert classify_split(_base_row(replicate_ordinal=5)) == "development_calibration"
    assert (
        classify_split(_base_row(first_policy_count=10, composition_profile="p10_90"))
        == "native_composition_holdout"
    )
    assert (
        classify_split(_base_row(correlation_profile="positive"))
        == "native_association_holdout"
    )
    assert (
        classify_split(_base_row(regime="activation_equal"))
        == "intervention_activation_equal"
    )
    assert (
        classify_split(_base_row(regime="joint_equal"))
        == "intervention_infeasible_sensitivity"
    )


def test_residence_known_patterns() -> None:
    identity = np.arange(100, dtype=np.uint8)
    occupancy = np.repeat(identity[None, :], 101, axis=0)
    labels = np.repeat(np.arange(2, dtype=np.uint8), 50)
    residence = _residence(occupancy, labels)
    assert residence["Bubble"] == 100.0
    assert residence["Insertion"] == 100.0
    assert residence["Selection"] == 0.0
    for index in range(1, 101):
        occupancy[index, :2] = occupancy[index, :2][::-1]
    residence = _residence(occupancy, labels)
    assert 100.0 / 101.0 <= residence["Bubble"] < 100.0


def test_mean_shape_prediction_preserves_initial_value() -> None:
    model = {"family": "mean_shape", "mean_residual": np.linspace(0.0, 0.2, 101)}
    x = np.zeros((2, 3))
    y0 = np.array([-0.1, 0.3])
    prediction = predict_model(model, x, y0)
    assert prediction.shape == (2, 101)
    np.testing.assert_array_equal(prediction[:, 0], y0)
    np.testing.assert_allclose(prediction[:, -1], y0 + 0.2)


@pytest.mark.skipif(
    not (S07_CACHE / "native_control.jsonl").exists(),
    reason="S07 raw fixture unavailable",
)
def test_s07_split_accounting_and_first_run_transport_validation() -> None:
    counts: Counter[str] = Counter()
    first = None
    for row in _read_jsonl(S07_CACHE / "native_control.jsonl"):
        counts[classify_split(row)] += 1
        if first is None:
            first = row
    assert counts == Counter(
        {
            "development_fit": 880,
            "development_calibration": 220,
            "native_composition_holdout": 450,
            "native_association_holdout": 3100,
        }
    )
    assert first is not None
    feature, ledger, validation = extract_run(
        first,
        {"achieved_spearman_rho": 0.0, "achieved_eta_squared": 0.0},
    )
    assert len(ledger) == 3
    assert validation["occupancy_permutation_passed"]
    assert validation["label_count_passed"]
    assert validation["curve_max_error"] <= 1e-12
    assert validation["structural_identity_max_error"] <= 1e-12
    assert validation["activation_accounting_max_error"] <= 1e-12
    assert validation["flux_conservation_error"] <= 1e-12
    assert all(name in feature for name in STATIC_NAMES + TRANSPORT_NAMES)
