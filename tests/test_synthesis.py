from __future__ import annotations

import json

import numpy as np
import pandas as pd

from causal_simulator.synthesis import (
    ENDPOINT_SPECS,
    ESTIMAND_LABELS,
    build_mechanism_tables,
    pareto_analysis,
    pareto_mask,
    summarize_s13,
    validate_complete_panel,
    variance_decomposition,
)


def test_pareto_mask_uses_strict_minimization_dominance() -> None:
    objectives = np.asarray(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0],
        ]
    )
    assert pareto_mask(objectives).tolist() == [True, True, True, False]


def _panel(blocks: int, settings: int, *, protected: bool) -> pd.DataFrame:
    rows = []
    for block in range(blocks):
        for setting in range(settings):
            rows.append(
                {
                    "pairingBlockId": f"b{block}",
                    "treatmentSignature": f"t{setting}",
                    "n": 20,
                    "normalizedResidualError": (block + setting) / 100,
                    "successByBudget": int((block + setting) % 2 == 0),
                    "projection_s01UnitWeightFullCost": block + setting + 1,
                    "contractValidationPass": True,
                    "protected": protected,
                }
            )
    return pd.DataFrame(rows)


def test_complete_panel_validation_detects_exact_crossing() -> None:
    panel = _panel(4, 3, protected=True)
    result = validate_complete_panel(panel, expected_blocks=4, expected_settings=3, protected=True)
    assert result["complete"]
    assert result["rowCount"] == 12


def test_mechanism_tables_orient_benefit_and_retain_all_cells() -> None:
    marginal_rows = []
    for estimand in ESTIMAND_LABELS:
        for endpoint in ENDPOINT_SPECS:
            marginal_rows.append(
                {
                    "estimandId": estimand,
                    "endpoint": endpoint,
                    "estimate": 0.1,
                    "bootstrapLow95": 0.05,
                    "bootstrapHigh95": 0.15,
                }
            )
    heterogeneity_rows = []
    for index in range(368):
        base = marginal_rows[index % len(marginal_rows)]
        heterogeneity_rows.append(
            {
                **base,
                "modifier": "n",
                "level": str(index),
                "pairCount": 10,
                "supported": True,
            }
        )
    marginal, heterogeneity = build_mechanism_tables(
        pd.DataFrame(marginal_rows), pd.DataFrame(heterogeneity_rows)
    )
    assert len(marginal) == 16
    assert len(heterogeneity) == 368
    residual = marginal[marginal.endpoint == "normalizedResidualError"]
    success = marginal[marginal.endpoint == "successByBudget"]
    assert (residual.benefit_estimate == -0.1).all()
    assert (success.benefit_estimate == 0.1).all()


def _pareto_fixture() -> pd.DataFrame:
    rows = []
    settings = [
        ("distributed_local", "none", "uniform_random_activation", "passive", "skip_and_continue"),
        ("distributed_local", "none", "random_permutation_sweep", "passive", "skip_and_continue"),
        ("distributed_local", "none", "uniform_random_activation", "stuck", "skip_and_continue"),
        ("distributed_local", "none", "uniform_random_activation", "stuck", "stop_on_first_blocking_failure"),
        ("distributed_weak_coordinator", "none", "uniform_random_activation", "passive", "skip_and_continue"),
        ("distributed_weak_coordinator", "weak_frozen_budget", "uniform_random_activation", "passive", "skip_and_continue"),
    ]
    for scale in (20, 50, 100, 200, 500):
        for block in range(2):
            for index, setting in enumerate(settings):
                architecture, coordinator, scheduler, mobility, continuation = setting
                rows.append(
                    {
                        "pairingBlockId": f"b{scale}-{block}",
                        "n": scale,
                        "treatmentSignature": f"t{index}",
                        "architecture": architecture,
                        "coordinatorProfile": coordinator,
                        "scheduler": scheduler,
                        "mobility": mobility,
                        "continuation": continuation,
                        "retry": "no_retry",
                        "normalizedResidualError": 0.01 * index + 0.001 * block,
                        "successByBudget": int(index < 3),
                        "projection_s01UnitWeightFullCost": 10 + index,
                        "projection_controllerExpandedSensitivity": 11 + index,
                        "projection_mechanismExpandedSensitivity": 12 + index,
                    }
                )
    return pd.DataFrame(rows)


def test_pareto_bootstrap_replays_and_preserves_cost_profiles() -> None:
    fixture = _pareto_fixture()
    first = pareto_analysis(fixture, replicates=30, seed=7)
    second = pareto_analysis(fixture, replicates=30, seed=7)
    assert all(left.equals(right) for left, right in zip(first, second, strict=True))
    summary, winners, dominance = first
    assert len(summary) == 18
    assert set(summary.costProfile) == {"s01Cost", "controllerExpandedCost", "mechanismExpandedCost"}
    assert len(winners) == 18
    assert len(dominance) == 90


def _variance_fixture() -> pd.DataFrame:
    settings = [
        ("distributed_local", "none", "uniform_random_activation", "passive", "skip_and_continue", "no_retry"),
        ("distributed_local", "none", "uniform_random_activation", "normal", "skip_and_continue", "no_retry"),
        ("distributed_local", "none", "uniform_random_activation", "stuck", "skip_and_continue", "no_retry"),
        ("distributed_local", "none", "uniform_random_activation", "stuck", "stop_on_first_blocking_failure", "no_retry"),
        ("distributed_local", "none", "uniform_random_activation", "stuck", "skip_and_continue", "retry_later_bounded"),
        ("distributed_local", "none", "random_permutation_sweep", "passive", "skip_and_continue", "no_retry"),
        ("distributed_local", "none", "random_permutation_sweep", "stuck", "skip_and_continue", "no_retry"),
        ("distributed_weak_coordinator", "none", "uniform_random_activation", "passive", "skip_and_continue", "no_retry"),
        ("distributed_weak_coordinator", "weak_frozen_budget", "uniform_random_activation", "passive", "skip_and_continue", "no_retry"),
        ("central_local_proposal_k1", "common_validator_only", "uniform_random_activation", "passive", "skip_and_continue", "no_retry"),
        ("central_local_proposal_k1", "common_validator_only", "uniform_random_activation", "normal", "skip_and_continue", "no_retry"),
        ("central_local_proposal_k1", "common_validator_only", "uniform_random_activation", "stuck", "skip_and_continue", "no_retry"),
        ("central_local_proposal_k1", "common_validator_only", "uniform_random_activation", "stuck", "stop_on_first_blocking_failure", "no_retry"),
        ("central_local_proposal_k1", "common_validator_only", "random_permutation_sweep", "passive", "skip_and_continue", "no_retry"),
    ]
    rows = []
    for scale in (20, 50, 100, 200, 500):
        for block in range(3):
            for index, setting in enumerate(settings):
                architecture, coordinator, scheduler, mobility, continuation, retry = setting
                residual = 0.01 + 0.005 * index + 0.001 * block + 0.00001 * scale
                rows.append(
                    {
                        "pairingBlockId": f"b{scale}-{block}",
                        "n": scale,
                        "treatmentSignature": f"t{index:02d}",
                        "architecture": architecture,
                        "coordinatorProfile": coordinator,
                        "scheduler": scheduler,
                        "mobility": mobility,
                        "continuation": continuation,
                        "retry": retry,
                        "normalizedResidualError": residual,
                        "successByBudget": float(residual < 0.05),
                        "projection_s01UnitWeightFullCost": 20 + 10 * index + scale + block,
                    }
                )
    return pd.DataFrame(rows)


def test_variance_partition_is_exact_and_deterministic() -> None:
    fixture = _variance_fixture()
    first, diagnostics = variance_decomposition(fixture, replicates=20, seed=9)
    second, diagnostics_replay = variance_decomposition(fixture, replicates=20, seed=9)
    assert first.equals(second)
    assert diagnostics == diagnostics_replay
    assert diagnostics["maxPartitionIdentityError"] < 1e-12
    sums = first.groupby("outcome").share.sum()
    assert np.allclose(sums, 1.0)


def test_s13_summary_keeps_brittle_instances_bounded() -> None:
    panel = {
        "directionalResidualMean": 0.1,
        "directionalResidualBootstrapLow95": 0.05,
        "directionalResidualBootstrapHigh95": 0.15,
        "directionalConcordanceRate": 0.4,
    }
    rows = []
    for index in range(8):
        classification = "stream_confirmed_brittle_instance" if index < 2 else "not_confirmed"
        rows.append(
            {
                "selectionOrder": index + 1,
                "candidateId": f"c{index}",
                "objectiveId": "E06_active_advantage" if index < 2 else "E04_reference_advantage",
                "fullyConfirmed": False,
                "classification": classification,
                "exactPanelJson": json.dumps(panel),
                "neighborhoodPanelJson": json.dumps(panel),
            }
        )
    table, summary = summarize_s13(pd.DataFrame(rows))
    assert len(table) == 8
    assert summary["fullyConfirmedCount"] == 0
    assert summary["brittleExactInstanceCount"] == 2
    assert not summary["globalAbsenceClaimPermitted"]
