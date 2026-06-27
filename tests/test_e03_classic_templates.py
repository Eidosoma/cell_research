from __future__ import annotations

import unittest

from morphospace import (
    DSLPolicy,
    PolicyEventSimulator,
    bubble_template,
    classic_template_records,
    classic_template_table,
    insertion_template,
    parse_rule_program,
    policy_from_spec,
    selection_template,
)


def _forced_step(values, policy, *, forced_cell_id: int, forced_direction: int | None = None):
    sim = PolicyEventSimulator(values, policy, scheduler_seed=0, tie_breaker_seed=2)
    kwargs = {"forced_cell_id": forced_cell_id}
    if forced_direction is not None:
        kwargs["forced_direction"] = forced_direction
    outcome = sim.step(**kwargs)
    return outcome, sim.current_values()


class TestE03ClassicTemplates(unittest.TestCase):
    def test_records_round_trip_and_compile(self) -> None:
        records = classic_template_records()
        self.assertEqual([record.algorithm for record in records], ["bubble", "insertion", "selection"])
        for record in records:
            with self.subTest(algorithm=record.algorithm):
                self.assertFalse(record.exact_behavior)
                self.assertTrue(record.minimal_dsl_extension)
                parsed = parse_rule_program(record.dsl_program.to_json())
                self.assertEqual(parsed.to_dict(), record.dsl_program.to_dict())
                restored = policy_from_spec(DSLPolicy(parsed).to_spec())
                self.assertEqual(restored.to_spec().to_dict(), DSLPolicy(parsed).to_spec().to_dict())

    def test_flattened_table_has_searchable_parameters(self) -> None:
        rows = classic_template_table()
        self.assertEqual({row["algorithm"] for row in rows}, {"bubble", "insertion", "selection"})
        for row in rows:
            with self.subTest(algorithm=row["algorithm"]):
                self.assertIn("parameterDefaults", row)
                self.assertIn("parameterSpace", row)
                self.assertIn("minimalDslExtension", row)

    def test_templates_execute_on_small_unique_array(self) -> None:
        for algorithm, program in [
            ("bubble", bubble_template()),
            ("insertion", insertion_template()),
            ("selection", selection_template()),
        ]:
            with self.subTest(algorithm=algorithm):
                result = PolicyEventSimulator(
                    [5, 1, 4, 2, 3],
                    DSLPolicy(program),
                    scheduler_seed=10,
                    tie_breaker_seed=20,
                ).run(max_activations=100000)
                self.assertTrue(result.completed)
                self.assertEqual(result.final_values, [1, 2, 3, 4, 5])

    def test_bubble_template_matches_forced_s01_adjacent_swaps(self) -> None:
        program = DSLPolicy(bubble_template())
        for forced_cell_id, forced_direction in [(0, 1), (1, -1)]:
            with self.subTest(forced_cell_id=forced_cell_id, forced_direction=forced_direction):
                classic_outcome, classic_values = _forced_step(
                    [2, 1],
                    "bubble",
                    forced_cell_id=forced_cell_id,
                    forced_direction=forced_direction,
                )
                template_outcome, template_values = _forced_step(
                    [2, 1],
                    program,
                    forced_cell_id=forced_cell_id,
                    forced_direction=forced_direction,
                )
                self.assertEqual(template_values, classic_values)
                self.assertEqual(template_outcome.swapped, classic_outcome.swapped)
                self.assertEqual(template_outcome.target_position, classic_outcome.target_position)
                self.assertEqual(template_outcome.comparison_delta, classic_outcome.comparison_delta)

    def test_bubble_template_documents_random_direction_gap(self) -> None:
        classic_outcome, classic_values = _forced_step([3, 2, 1], "bubble", forced_cell_id=1)
        template_outcome, template_values = _forced_step(
            [3, 2, 1],
            DSLPolicy(bubble_template()),
            forced_cell_id=1,
        )
        self.assertNotEqual(template_values, classic_values)
        self.assertNotEqual(template_outcome.target_position, classic_outcome.target_position)

    def test_insertion_template_matches_enabled_s01_swap_and_exposes_prefix_gap(self) -> None:
        classic_outcome, classic_values = _forced_step([2, 1], "insertion", forced_cell_id=1)
        template_outcome, template_values = _forced_step(
            [2, 1],
            DSLPolicy(insertion_template()),
            forced_cell_id=1,
        )
        self.assertEqual(template_values, classic_values)
        self.assertEqual(template_outcome.swapped, classic_outcome.swapped)
        self.assertEqual(template_outcome.target_position, classic_outcome.target_position)

        classic_gap, classic_gap_values = _forced_step([3, 2, 1], "insertion", forced_cell_id=2)
        template_gap, template_gap_values = _forced_step(
            [3, 2, 1],
            DSLPolicy(insertion_template()),
            forced_cell_id=2,
        )
        self.assertFalse(classic_gap.swapped)
        self.assertTrue(template_gap.swapped)
        self.assertNotEqual(template_gap_values, classic_gap_values)

    def test_selection_template_partial_alignment_and_exact_gap(self) -> None:
        classic_outcome, classic_values = _forced_step([3, 1, 2], "selection", forced_cell_id=1)
        template_outcome, template_values = _forced_step(
            [3, 1, 2],
            DSLPolicy(selection_template()),
            forced_cell_id=1,
        )
        self.assertEqual(template_values, classic_values)
        self.assertEqual(template_outcome.swapped, classic_outcome.swapped)
        self.assertEqual(template_outcome.target_position, classic_outcome.target_position)

        classic_gap, classic_gap_values = _forced_step([3, 1, 2], "selection", forced_cell_id=0)
        template_gap, template_gap_values = _forced_step(
            [3, 1, 2],
            DSLPolicy(selection_template()),
            forced_cell_id=0,
        )
        self.assertFalse(classic_gap.swapped)
        self.assertTrue(template_gap.swapped)
        self.assertNotEqual(template_gap_values, classic_gap_values)


if __name__ == "__main__":
    unittest.main(verbosity=2)

