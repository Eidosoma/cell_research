from __future__ import annotations

import json
import unittest

from morphospace2d import (
    CellIdentity,
    NeighborPreferenceRule,
    actor_components,
    build_example_identity_catalog,
    default_identity_schema,
    evaluate_neighbor_preferences,
    identities_from_scalar_values,
    identity_to_cell_state,
    observable_components,
    scalar_values_from_identities,
    validate_identity_catalog,
)


class TestE05CellIdentity(unittest.TestCase):
    def test_default_schema_is_valid_and_marks_visibility(self) -> None:
        schema = default_identity_schema()
        self.assertEqual(schema.validate(), [])
        fields = schema.field_map()
        self.assertTrue(fields["scalar_value"].observable_to_neighbors)
        self.assertFalse(fields["target_neighbor_preferences"].observable_to_neighbors)
        self.assertTrue(fields["internal_state"].hidden)

    def test_scalar_values_round_trip_for_1d_baseline(self) -> None:
        values = [5, 1, 4, 2, 3]
        identities = identities_from_scalar_values(values)
        self.assertEqual(scalar_values_from_identities(identities), values)
        self.assertEqual([identity.scalar_value for identity in identities], values)

    def test_identity_serialization_round_trip(self) -> None:
        schema = default_identity_schema()
        identity = build_example_identity_catalog()[0]
        record = identity.to_record(schema=schema, include_hidden=True)
        json.dumps(record, sort_keys=True)
        restored = CellIdentity.from_record(record)
        self.assertEqual(restored.to_record(schema=schema, include_hidden=True), record)

    def test_hidden_and_actor_internal_fields_are_not_neighbor_observable(self) -> None:
        schema = default_identity_schema()
        identity = build_example_identity_catalog()[0]
        neighbor_view = observable_components(identity, schema)
        actor_view = actor_components(identity, schema)
        self.assertIn("scalar_value", neighbor_view)
        self.assertNotIn("target_neighbor_preferences", neighbor_view)
        self.assertNotIn("internal_state", neighbor_view)
        self.assertIn("target_neighbor_preferences", actor_view)
        self.assertNotIn("internal_state", actor_view)

    def test_identity_catalog_validates_and_bridges_to_cell_state(self) -> None:
        schema = default_identity_schema()
        identities = build_example_identity_catalog()
        self.assertEqual(validate_identity_catalog(identities, schema), [])
        cell_state = identity_to_cell_state(identities[0], schema)
        self.assertEqual(cell_state.value, identities[0].scalar_value)
        self.assertIn("organ_type", cell_state.identity)
        self.assertNotIn("internal_state", cell_state.identity)

    def test_neighborhood_preference_satisfied_and_penalized_cases(self) -> None:
        actor = CellIdentity(
            100,
            {
                "scalar_value": 1,
                "ap_coordinate": 0.0,
                "organ_type": "core",
                "polarity": [1.0, 0.0],
                "adhesion_type": "adhesion_a",
                "target_neighbor_preferences": [
                    NeighborPreferenceRule("organ_type", "boundary", min_count=1, weight=2.0).to_record()
                ],
                "internal_state": {},
            },
        )
        boundary_neighbor = CellIdentity(
            101,
            {
                "scalar_value": 2,
                "ap_coordinate": 1.0,
                "organ_type": "boundary",
                "polarity": [1.0, 0.0],
                "adhesion_type": "adhesion_boundary",
                "target_neighbor_preferences": [],
                "internal_state": {},
            },
        )
        core_neighbor = CellIdentity(
            102,
            {
                "scalar_value": 3,
                "ap_coordinate": 1.0,
                "organ_type": "core",
                "polarity": [1.0, 0.0],
                "adhesion_type": "adhesion_a",
                "target_neighbor_preferences": [],
                "internal_state": {},
            },
        )
        satisfied = evaluate_neighbor_preferences(actor, [boundary_neighbor, core_neighbor])
        penalized = evaluate_neighbor_preferences(actor, [core_neighbor])
        self.assertEqual(satisfied["penalty"], 0.0)
        self.assertEqual(satisfied["satisfiedRuleCount"], 1)
        self.assertGreater(penalized["penalty"], 0.0)
        self.assertLess(penalized["score"], satisfied["score"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
