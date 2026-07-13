import csv
import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_transition_spec_assets.py"
SPEC = importlib.util.spec_from_file_location("build_transition_spec_assets", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class TransitionSpecificationAssetsTest(unittest.TestCase):
    def test_all_hand_fixtures_pass_independent_oracle(self):
        self.assertEqual(module.validate_fixtures(module.fixture_data()), [])

    def test_duplicate_and_mixed_direction_terminal_semantics(self):
        fixtures = {row["fixtureId"]: row for row in module.fixture_data()}
        duplicate = fixtures["T15"]
        self.assertEqual(module.terminal(duplicate["pre"], 0, 10), "complete")
        mixed = fixtures["T16"]
        self.assertEqual(module.terminal(mixed["pre"], 5, 5), "event_budget")

    def test_counter_addressed_rng_known_vector_and_separation(self):
        value = module.counter_u64(7, "scenario-A", "actor_activation", 3, 0)
        self.assertEqual(value, 2584083456120664033)
        self.assertNotEqual(value, module.counter_u64(7, "scenario-A", "bubble_side", 3, 0))
        self.assertNotEqual(value, module.counter_u64(7, "scenario-A", "actor_activation", 4, 0))

    def test_claim_mapping_preserves_every_condition(self):
        rows = [
            {
                "claim_id": "X",
                "figure": "3",
                "panel": "A",
                "claim_text": "cell-view Bubble",
                "evidence_status": "specified",
                "architecture": "cell_view",
                "algorithm": "Bubble",
                "ambiguity_codes": "",
                **{column: "not reported" for column in module.CONDITION_COLUMNS if column not in {"architecture", "algorithm"}},
            }
        ]
        mapped = module.map_claims(rows)
        self.assertEqual(len(mapped), 1)
        self.assertIn("POL-BUBBLE", mapped[0]["specSectionIds"])
        self.assertEqual(set(module.json.loads(mapped[0]["conditionJson"])), set(module.CONDITION_COLUMNS))

    def test_full_registry_has_118_unique_claims_and_known_ambiguities(self):
        path = Path("/artifacts/research_steps/S01/claim_registry.csv")
        if not path.exists():
            self.skipTest("S01 artifact not mounted")
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        mapped = module.map_claims(rows)
        checks, errors = module.validate_contract(module.contract(), rows, mapped, module.fixture_data())
        self.assertEqual(errors, [], checks)


if __name__ == "__main__":
    unittest.main()
