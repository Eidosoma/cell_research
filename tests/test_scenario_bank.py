"""Focused contract tests for the immutable S08 paired scenario bank."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import unittest

import jsonschema
import pyarrow.parquet as pq

from reference_simulator.model import Scenario
from scenario_bank.builder import _claim_condition_mapping, _scenario_row
from scenario_bank.core import (
    BASE_DRAW_JSON_SCHEMA,
    CONDITION_JSON_SCHEMA,
    MASTER_SEED,
    SCENARIO_ROW_JSON_SCHEMA,
    build_condition_catalog,
    derive_seed,
    make_base_draw,
    materialize_scenario,
)


CLAIM_REGISTRY = Path("/artifacts/research_steps/S01/claim_registry.parquet")


class ScenarioBankCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.conditions = build_condition_catalog()

    def condition(self, condition_id: str):
        return next(item for item in self.conditions if item.condition_id == condition_id)

    def test_catalog_has_frozen_family_counts(self):
        self.assertEqual(len(self.conditions), 110)
        self.assertEqual(
            Counter(item.family for item in self.conditions),
            Counter(
                {
                    "unique_pure": 6,
                    "unique_fault": 72,
                    "same_direction_unique": 8,
                    "identical_policy_label_control": 3,
                    "same_direction_repeated": 6,
                    "opposite_unique": 6,
                    "opposite_repeated": 6,
                    "worked_example_ambiguity_envelope": 3,
                }
            ),
        )

    def test_seed_derivation_known_vectors(self):
        self.assertEqual(MASTER_SEED, 0xE0108000000000000000000000000001)
        self.assertEqual(
            derive_seed("initial_occupancy", "unique_1_100", "paper_scale", 0),
            336754745922564179643046861265416524746,
        )
        self.assertEqual(
            derive_seed("reference_runtime", "repeated_1_10_x10", "confirmatory_holdout", 999),
            73665498058050288982797826992834039279,
        )

    def test_base_draws_are_deterministic_and_preserve_distributions(self):
        unique = make_base_draw("unique_1_100", "paper_scale", 0)
        repeated = make_base_draw("repeated_1_10_x10", "exploratory", 12)
        self.assertEqual(unique, make_base_draw("unique_1_100", "paper_scale", 0))
        self.assertEqual(sorted(unique["valuesById"]), list(range(1, 101)))
        self.assertEqual(Counter(repeated["valuesById"]), Counter({value: 10 for value in range(1, 11)}))
        self.assertEqual(sorted(unique["initialOccupancyIndices"]), list(range(100)))
        self.assertEqual(sorted(repeated["initialOccupancyIndices"]), list(range(100)))

    def test_worked_envelope_has_lexicographic_endpoints(self):
        first = make_base_draw("worked_1_6_exhaustive", "ambiguity_envelope", 0)
        last = make_base_draw("worked_1_6_exhaustive", "ambiguity_envelope", 719)
        self.assertEqual(first["initialOccupancyIndices"], [0, 1, 2, 3, 4, 5])
        self.assertEqual(last["initialOccupancyIndices"], [5, 4, 3, 2, 1, 0])

    def test_all_three_configuration_schemas_are_strict(self):
        condition = self.condition("C-UNQ-PURE-CV-BUB-ASC")
        base = make_base_draw("unique_1_100", "paper_scale", 0)
        row = _scenario_row(condition, base, ["F03-A-BUBBLE"], "0" * 64)
        jsonschema.Draft202012Validator(CONDITION_JSON_SCHEMA).validate(condition.to_dict())
        jsonschema.Draft202012Validator(BASE_DRAW_JSON_SCHEMA).validate(base)
        jsonschema.Draft202012Validator(SCENARIO_ROW_JSON_SCHEMA).validate(row)
        invalid = dict(row)
        invalid["undeclaredField"] = "not allowed"
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.Draft202012Validator(SCENARIO_ROW_JSON_SCHEMA).validate(invalid)

    def test_exact_compositions_and_opposed_directions(self):
        base = make_base_draw("unique_1_100", "paper_scale", 7)
        pair, pair_metadata = materialize_scenario(
            self.condition("C-UNQ-CHIM-BUB-INS-EXACT-ASC"), base
        )
        three, three_metadata = materialize_scenario(
            self.condition("C-UNQ-CHIM-BUB-INS-SEL-EXACT-ASC"), base
        )
        opposed, opposed_metadata = materialize_scenario(
            self.condition("C-UNQ-CHIM-BUB-SEL-EXACT-OPP"), base
        )
        self.assertEqual(pair_metadata["compositionCounts"], {"Bubble": 50, "Insertion": 50})
        self.assertEqual(sorted(three_metadata["compositionCounts"].values()), [33, 33, 34])
        self.assertEqual(opposed_metadata["directionCounts"], {"ascending": 50, "descending": 50})
        self.assertEqual(Scenario.from_dict(pair.to_dict()), pair)
        self.assertEqual(Scenario.from_dict(three.to_dict()), three)
        self.assertEqual(Scenario.from_dict(opposed.to_dict()), opposed)

    def test_fault_profiles_are_separate_and_loss_is_explicit(self):
        base = make_base_draw("unique_1_100", "paper_scale", 30)
        corrected, corrected_metadata = materialize_scenario(
            self.condition("C-UNQ-FAULT-CV-BUB-PAS-F3-CORR"), base
        )
        legacy, legacy_metadata = materialize_scenario(
            self.condition("C-UNQ-FAULT-CV-BUB-PAS-F3-LEGACY"), base
        )
        self.assertEqual(corrected.realized_fault_count, 3)
        self.assertEqual(len(set(corrected_metadata["faultDrawIndices"])), 3)
        self.assertLessEqual(legacy.realized_fault_count, 3)
        self.assertEqual(legacy_metadata["faultDrawIndices"], [72, 68, 4])
        self.assertEqual(
            legacy.realized_fault_count,
            len(set(legacy_metadata["faultDrawIndices"])),
        )
        self.assertNotEqual(corrected.fault_placement, legacy.fault_placement)

    def test_ghost_controls_change_labels_not_executable_policy(self):
        base = make_base_draw("unique_1_100", "paper_scale", 0)
        scenario, metadata = materialize_scenario(
            self.condition("C-UNQ-CONTROL-GHOST-BUB-INS"), base
        )
        self.assertEqual({cell.policy.value for cell in scenario.cells}, {"Bubble"})
        self.assertEqual(Counter(cell.analysis_label for cell in scenario.cells), Counter({"ghost:Bubble": 50, "ghost:Insertion": 50}))
        self.assertEqual(len(metadata["analysisLabelAssignmentSha256"]), 64)

    def test_claim_registry_maps_all_claims_and_conditions(self):
        if not CLAIM_REGISTRY.exists():
            self.skipTest("S01 claim registry is not mounted")
        claims = pq.read_table(CLAIM_REGISTRY).to_pylist()
        rows = _claim_condition_mapping(claims, self.conditions)
        self.assertEqual(len({row["claimId"] for row in rows}), 118)
        mapped = {row["conditionId"] for row in rows if row["conditionId"]}
        self.assertEqual(mapped, {item.condition_id for item in self.conditions})
        no_scenario = {
            row["claimId"]
            for row in rows
            if row["coverageDisposition"] == "no_scenario_required"
        }
        self.assertEqual(
            no_scenario,
            {
                "F06-A-CONCEPTUAL-CONTEXT",
                "F06-D-DG-DEFINITION",
                "F08-C-AGGREGATION-DEFINITION",
            },
        )


if __name__ == "__main__":
    unittest.main()
