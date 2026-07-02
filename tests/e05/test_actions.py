"""E05 S04 action-set tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.e05.actions import (
    ADHESION_BONDS_KEY,
    REQUESTED_S04_ACTIONS,
    SIGNAL_INBOX_KEY,
    ActionExecutor,
    MorphogenesisActionRequest,
    action_state_signature,
    action_spec_rows,
    default_action_set,
)
from src.e05.cell_identity import attach_identity, identity_from_substrate_cell
from src.e05.substrates import SubstrateCell, array_substrate, square_grid_substrate
from src.e05.targets import morph_identity


def identity_cell(
    cell_id: str,
    ap: float,
    organ: str = "neural",
    polarity: tuple[float, float] = (1.0, 0.0),
    adhesion: str = "medium",
) -> SubstrateCell:
    return attach_identity(
        SubstrateCell(cell_id, value=ap, label=organ),
        morph_identity(f"{cell_id}_identity", ap, organ, polarity, adhesion),
    )


class ActionSetTests(unittest.TestCase):
    def test_default_action_catalog_includes_requested_s04_actions_and_costs(self) -> None:
        rows = action_spec_rows(default_action_set())
        action_types = {row["action_type"] for row in rows}
        self.assertTrue(set(REQUESTED_S04_ACTIONS).issubset(action_types))
        self.assertTrue(all(row["energy_cost"] >= 0.0 for row in rows))
        self.assertEqual({row["action_type"]: row["population_delta"] for row in rows}["divide"], 1)
        self.assertEqual({row["action_type"]: row["population_delta"] for row in rows}["die"], -1)

    def test_swap_and_crawl_are_local_and_population_conserving(self) -> None:
        executor = ActionExecutor()

        swap_state = array_substrate(2)
        swap_state.fill_sites((SubstrateCell("a", 2), SubstrateCell("b", 1)))
        swap_result = executor.execute(swap_state, MorphogenesisActionRequest("swap", 0, 1))
        self.assertTrue(swap_result.allowed)
        self.assertEqual(swap_result.energy_cost_charged, 1.0)
        self.assertEqual(swap_result.conservation_delta, 0)
        self.assertEqual(swap_state.values_in_site_order(), (1, 2))

        crawl_state = array_substrate(3)
        crawl_state.place_cell(0, SubstrateCell("crawler", 3))
        crawl_result = executor.execute(crawl_state, MorphogenesisActionRequest("crawl", 0, 1))
        self.assertTrue(crawl_result.allowed)
        self.assertEqual(crawl_result.energy_cost_charged, 1.2)
        self.assertEqual(crawl_result.population_before, crawl_result.population_after)
        self.assertEqual(crawl_state.site_of_cell("crawler"), 1)

    def test_illegal_moves_are_rejected_without_cost_or_state_change(self) -> None:
        state = square_grid_substrate(3, 3)
        state.place_cell(0, SubstrateCell("a", 1))
        state.place_cell(8, SubstrateCell("b", 2))
        before = state.signature()

        result = ActionExecutor().execute(state, MorphogenesisActionRequest("swap", 0, 8))

        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, "target_not_adjacent")
        self.assertEqual(result.energy_cost_charged, 0.0)
        self.assertFalse(result.state_changed)
        self.assertEqual(state.signature(), before)

    def test_rotate_polarity_updates_identity_locally(self) -> None:
        state = array_substrate(1)
        state.place_cell(0, identity_cell("cell", 0.5, polarity=(1.0, 0.0)))

        result = ActionExecutor().execute(
            state,
            MorphogenesisActionRequest("rotate_polarity", 0, parameters={"polarity": (0.0, 2.0)}),
        )

        self.assertTrue(result.allowed)
        self.assertTrue(result.state_changed)
        self.assertEqual(result.conservation_delta, 0)
        identity = identity_from_substrate_cell(state.cell_at(0))
        self.assertEqual(tuple(identity.components["polarity"]), (0.0, 1.0))

    def test_divide_and_die_track_birth_death_accounting(self) -> None:
        state = array_substrate(2)
        state.place_cell(0, identity_cell("parent", 0.1))
        executor = ActionExecutor()

        birth = executor.execute(state, MorphogenesisActionRequest("divide", 0, 1))
        self.assertTrue(birth.allowed)
        self.assertEqual(birth.birth_count, 1)
        self.assertEqual(birth.death_count, 0)
        self.assertEqual(birth.conservation_delta, 1)
        self.assertEqual(birth.population_after, 2)

        death = executor.execute(state, MorphogenesisActionRequest("die", 1))
        self.assertTrue(death.allowed)
        self.assertEqual(death.birth_count, 0)
        self.assertEqual(death.death_count, 1)
        self.assertEqual(death.conservation_delta, -1)
        self.assertEqual(death.population_after, 1)

    def test_adhere_and_detach_modify_reciprocal_local_bonds(self) -> None:
        state = array_substrate(2)
        state.fill_sites((identity_cell("left", 0.1), identity_cell("right", 0.9)))
        executor = ActionExecutor()

        adhered = executor.execute(state, MorphogenesisActionRequest("adhere", 0, 1))
        self.assertTrue(adhered.allowed)
        self.assertEqual(state.cell_at(0).metadata[ADHESION_BONDS_KEY], ["right"])
        self.assertEqual(state.cell_at(1).metadata[ADHESION_BONDS_KEY], ["left"])

        detached = executor.execute(state, MorphogenesisActionRequest("detach", 0, 1))
        self.assertTrue(detached.allowed)
        self.assertEqual(state.cell_at(0).metadata[ADHESION_BONDS_KEY], [])
        self.assertEqual(state.cell_at(1).metadata[ADHESION_BONDS_KEY], [])

    def test_exchange_signal_is_local_and_bounded(self) -> None:
        state = array_substrate(3)
        state.place_cell(0, identity_cell("source", 0.1))
        state.place_cell(1, identity_cell("target", 0.2))
        state.place_cell(2, identity_cell("far", 0.3))
        executor = ActionExecutor()

        sent = executor.execute(
            state,
            MorphogenesisActionRequest("exchange_signal", 0, 1, {"channel": "blocked", "value": 0.75}),
        )
        self.assertTrue(sent.allowed)
        inbox = state.cell_at(1).metadata[SIGNAL_INBOX_KEY]
        self.assertEqual(inbox[-1]["channel"], "blocked")
        self.assertEqual(inbox[-1]["access_scope"], "adjacent_neighbor")

        nonlocal_result = executor.execute(
            state,
            MorphogenesisActionRequest("exchange_signal", 0, 2, {"channel": "blocked", "value": 0.25}),
        )
        self.assertFalse(nonlocal_result.allowed)
        self.assertEqual(nonlocal_result.reason, "target_not_adjacent")

        out_of_range = executor.execute(
            state,
            MorphogenesisActionRequest("exchange_signal", 0, 1, {"channel": "target_seeking", "value": 2.0}),
        )
        self.assertFalse(out_of_range.allowed)
        self.assertEqual(out_of_range.reason, "signal_value_out_of_range")

    def test_legal_action_results_are_dry_runs(self) -> None:
        state = array_substrate(2)
        state.fill_sites((identity_cell("left", 0.1), identity_cell("right", 0.9)))
        before = action_state_signature(state)

        results = ActionExecutor().legal_action_results(state, 0)

        self.assertEqual(action_state_signature(state), before)
        self.assertIn("swap", {result.action_type for result in results})
        self.assertTrue(any(result.action_type == "swap" and result.allowed for result in results))


if __name__ == "__main__":
    unittest.main()
