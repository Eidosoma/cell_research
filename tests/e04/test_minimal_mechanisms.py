"""E04 S15 minimal mechanism synthesis tests."""

from __future__ import annotations

import unittest

import pandas as pd

from src.e04.minimal_mechanisms import (
    CENTRALIZED_MECHANISM_ID,
    MINIMAL_MECHANISM_ID,
    build_minimal_mechanism_table,
    validate_claim_boundaries,
)


def _local_df(rows: int = 2) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "uses_global_oracle": [False] * rows,
            "centralized_baseline": [False] * rows,
            "eligible_for_local_only_claims": [True] * rows,
        }
    )


class MinimalMechanismTests(unittest.TestCase):
    def _anchors(self) -> dict[str, float | int | str]:
        return {
            "s09_neighbor_mean_delta": 0.10,
            "s09_neighbor_positive_fraction": 0.95,
            "s09_neighbor_final_sortedness_delta": 10.0,
            "s09_non_neighbor_max_delta": 0.001,
            "s09_non_neighbor_max_positive_fraction": 0.10,
            "s10_best_mode": "diffusive_signaling",
            "s10_best_mean_delta": 0.004,
            "s10_best_positive_fraction": 0.25,
            "s10_max_positive_fraction": 0.30,
            "s11_neighbor_delta_vs_open_loop": -0.05,
            "s11_neighbor_positive_fraction_vs_open_loop": 0.25,
            "s11_best_signal_delta_vs_neighbor": 0.02,
            "s11_best_signal_positive_fraction_vs_neighbor": 0.40,
            "s12_combined_mean_local_order_auc_delta": 0.01,
            "s12_combined_max_local_order_auc_delta": 0.02,
            "s12_allowed_mean_local_order_auc_delta": -0.01,
            "s13_neighbor_mean_train_fitness": 0.96,
            "s13_neighbor_mean_heldout_fitness": 0.87,
            "s13_neighbor_transfer_retention": 0.90,
            "s13_neighbor_delta_vs_no_memory": 0.12,
            "s13_neighbor_positive_axis_fraction": 1.0,
            "s13_no_memory_mean_heldout_fitness": 0.75,
            "s14_local_rows": 1,
            "s14_central_rows": 1,
            "s14_local_mean_comparison_repair_score": 0.94,
            "s14_central_mean_comparison_repair_score": 0.99,
            "s14_mean_repair_score_retention": 0.94,
            "s14_mean_repair_score_gap": 0.05,
            "s14_mean_repair_quality_gap": 0.09,
            "s14_mean_energy_gap": -10.0,
        }

    def test_validate_claim_boundaries_passes_expected_synthesis_gates(self) -> None:
        evidence = {
            "s09": _local_df(),
            "s10": _local_df(),
            "s11": _local_df(),
            "s13": _local_df(),
            "s14": pd.DataFrame(
                {
                    "uses_global_oracle": [False, True],
                    "centralized_baseline": [False, True],
                    "eligible_for_local_only_claims": [True, False],
                }
            ),
        }
        checks = validate_claim_boundaries(evidence, self._anchors())

        self.assertTrue(checks["all_passed"])
        self.assertTrue(checks["s14_centralized_rows_excluded_from_local_claims"]["passed"])

    def test_validate_claim_boundaries_fails_if_centralized_row_is_claim_eligible(self) -> None:
        evidence = {
            "s09": _local_df(),
            "s10": _local_df(),
            "s11": _local_df(),
            "s13": _local_df(),
            "s14": pd.DataFrame(
                {
                    "uses_global_oracle": [False, True],
                    "centralized_baseline": [False, True],
                    "eligible_for_local_only_claims": [True, True],
                }
            ),
        }
        checks = validate_claim_boundaries(evidence, self._anchors())

        self.assertFalse(checks["all_passed"])
        self.assertFalse(checks["s14_centralized_rows_excluded_from_local_claims"]["passed"])

    def test_minimal_mechanism_table_marks_only_neighbor_memory_as_minimal_local(self) -> None:
        table = build_minimal_mechanism_table(self._anchors())
        minimal = table.loc[table["mechanism_id"] == MINIMAL_MECHANISM_ID].iloc[0]
        central = table.loc[table["mechanism_id"] == CENTRALIZED_MECHANISM_ID].iloc[0]

        self.assertTrue(bool(minimal["included_in_minimal_package"]))
        self.assertTrue(bool(minimal["local_only_claim_allowed"]))
        self.assertEqual(minimal["evidence_status"], "supportive_with_constraints")
        self.assertFalse(bool(central["included_in_minimal_package"]))
        self.assertFalse(bool(central["local_only_claim_allowed"]))


if __name__ == "__main__":
    unittest.main()
