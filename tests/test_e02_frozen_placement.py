from __future__ import annotations

import json
import unittest

from e02_deterministic_simulator import DeterministicEventSimulator
from scripts.e02_s10_input_distributions import input_values_for_profile, stable_seed
from scripts.e02_s11_frozen_placement import (
    CONTROL_PLACEMENT_CATEGORY,
    FROZEN_PLACEMENT_CATEGORIES,
    add_identity_logging,
    frozen_positions_for_category,
    placement_conditions,
    placement_confound_note,
    validate_placement_category,
    value_position_confound,
)


class TestE02FrozenPlacement(unittest.TestCase):
    def test_seeded_placement_categories_are_replayable_and_valid(self) -> None:
        n = 30
        values = input_values_for_profile("random_unique", n, stable_seed("s11-test-input"))
        for category in FROZEN_PLACEMENT_CATEGORIES:
            seed = stable_seed("s11-test", category)
            with self.subTest(category=category):
                first = frozen_positions_for_category(category, values, seed, 2)
                second = frozen_positions_for_category(category, values, seed, 2)
                self.assertEqual(first, second)
                self.assertTrue(validate_placement_category(category, first, values, 2))
                self.assertEqual(len(first), 2)
                self.assertEqual(len(set(first)), 2)

    def test_specific_placement_definitions(self) -> None:
        values = [10, 1, 30, 2, 20, 5, 8, 3, 7, 9]
        self.assertEqual(frozen_positions_for_category("array_ends", values, 1, 2), [0, 9])
        self.assertEqual(frozen_positions_for_category("center", values, 1, 2), [4, 5])
        self.assertEqual(frozen_positions_for_category("high_value", values, 1, 2), [2, 4])
        self.assertEqual(frozen_positions_for_category("low_value", values, 1, 2), [1, 3])
        clustered = frozen_positions_for_category("clustered", values, 123, 2)
        self.assertEqual(max(clustered) - min(clustered), 1)
        evenly = frozen_positions_for_category("evenly_spaced", values, 1, 2)
        self.assertGreaterEqual(evenly[1] - evenly[0], 2)
        self.assertEqual(frozen_positions_for_category(CONTROL_PLACEMENT_CATEGORY, values, 1, 0), [])

    def test_condition_matrix_shape(self) -> None:
        conditions = placement_conditions()
        self.assertEqual(len(conditions), 45)
        frozen = [condition for condition in conditions if condition.frozen_count > 0]
        controls = [condition for condition in conditions if condition.frozen_count == 0]
        self.assertEqual(len(frozen), 42)
        self.assertEqual(len(controls), 3)
        self.assertEqual(len(conditions) * 3, 135)

    def test_high_low_confound_documentation(self) -> None:
        for category in ("high_value", "low_value"):
            with self.subTest(category=category):
                self.assertTrue(value_position_confound(category))
                self.assertIn("confound", placement_confound_note(category).lower())
        self.assertFalse(value_position_confound("random"))
        self.assertFalse(value_position_confound(CONTROL_PLACEMENT_CATEGORY))

    def test_identity_logging_records_frozen_cell_ids_in_trace(self) -> None:
        sim = DeterministicEventSimulator(
            [2, 1, 3, 4],
            "bubble",
            frozen_positions=[1],
            frozen_variant="passive",
            scheduler_seed=1,
            tie_breaker_seed=2,
            condition_id="unit_identity_logging",
            research_step_id="S11",
        )
        frozen_ids = sim.frozen_cell_ids()
        add_identity_logging(sim, frozen_ids)
        result = sim.run(max_activations=50, max_swaps=20, max_comparisons=100)
        self.assertEqual(frozen_ids, [1])
        self.assertTrue(result.trace_rows)
        for row in result.trace_rows:
            self.assertEqual(json.loads(row["frozen_cell_ids_json"]), [1])
            self.assertIn("1", json.loads(row["frozen_cell_id_positions_json"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
