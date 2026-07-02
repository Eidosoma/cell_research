"""E05 S02 identity-vector tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.e05.cell_identity import (
    CellIdentity,
    TargetNeighborPreference,
    attach_identity,
    default_morphogenesis_identity_schema,
    identity_from_substrate_cell,
    identity_ordered_values,
    local_identity_observation,
    scalar_identity,
    scalar_value_schema,
    should_swap_for_identity_order,
)
from src.e05.substrates import SubstrateCell, array_substrate, square_grid_substrate


def morph_identity(
    identity_id: str,
    ap: float,
    organ: str,
    polarity: tuple[float, float],
    adhesion: str,
    preferences: tuple[TargetNeighborPreference, ...] = (),
) -> CellIdentity:
    return CellIdentity(
        identity_id=identity_id,
        components={
            "ap_coordinate": ap,
            "organ_type": organ,
            "polarity": polarity,
            "adhesion_type": adhesion,
        },
        target_preferences=preferences,
    )


class CellIdentityTests(unittest.TestCase):
    def test_scalar_identity_order_decisions_match_value_comparisons(self) -> None:
        schema = scalar_value_schema(1, 4)
        left = scalar_identity(4, "left")
        right = scalar_identity(1, "right")
        self.assertTrue(should_swap_for_identity_order(left, right, schema, direction="increasing"))
        self.assertFalse(should_swap_for_identity_order(left, right, schema, direction="decreasing"))
        self.assertEqual(schema.compare_order(scalar_identity(2), scalar_identity(3)), -1)

    def test_scalar_value_sorting_recovered_as_special_case(self) -> None:
        result = identity_ordered_values((4, 1, 3, 2), direction="increasing")
        self.assertEqual(result["final_values"], (1, 2, 3, 4))
        self.assertEqual(result["swap_count"], 4)

        reverse = identity_ordered_values((1, 4, 2, 3), direction="decreasing")
        self.assertEqual(reverse["final_values"], (4, 3, 2, 1))
        self.assertEqual(reverse["swap_count"], 4)

        duplicates = identity_ordered_values((2, 1, 2), direction="increasing")
        self.assertEqual(duplicates["final_values"], (1, 2, 2))

        with self.assertRaises(ValueError):
            identity_ordered_values(())

    def test_identity_metadata_roundtrip_on_substrate_cell(self) -> None:
        identity = scalar_identity(7, "seven")
        cell = attach_identity(SubstrateCell("cell_a", value=7, label="scalar"), identity)
        recovered = identity_from_substrate_cell(cell)
        self.assertEqual(recovered.identity_id, "seven")
        self.assertEqual(recovered.components["value"], 7.0)
        self.assertEqual(cell.metadata["e05_identity"]["components"]["value"], 7.0)

    def test_local_identity_observation_exposes_neighbor_relations(self) -> None:
        schema = default_morphogenesis_identity_schema()
        substrate = array_substrate(3)
        actor = morph_identity(
            "actor",
            0.5,
            "neural",
            (1.0, 0.0),
            "high",
            (TargetNeighborPreference("right", "organ_type", "epidermis"),),
        )
        left = morph_identity("left", 0.25, "neural", (1.0, 0.0), "high")
        right = morph_identity("right", 0.75, "epidermis", (0.0, 1.0), "low")
        substrate.fill_sites(
            (
                attach_identity(SubstrateCell("left", value=0.25), left),
                attach_identity(SubstrateCell("actor", value=0.5), actor),
                attach_identity(SubstrateCell("right", value=0.75), right),
            )
        )

        observation = local_identity_observation(substrate, 1, schema)

        self.assertEqual(observation.actor_cell_id, "actor")
        self.assertEqual([relation.direction for relation in observation.neighbor_relations], ["left", "right"])
        self.assertEqual([relation.order_relation for relation in observation.neighbor_relations], [1, -1])
        self.assertEqual(observation.neighbor_relations[0].compatibility_score, 1.0)
        self.assertEqual(observation.neighbor_relations[1].target_preference_score, 1.0)

    def test_compatibility_and_polarity_distance_are_bounded(self) -> None:
        schema = default_morphogenesis_identity_schema()
        same = morph_identity("same", 0.2, "neural", (1.0, 0.0), "high")
        same_family = morph_identity("same_family", 0.8, "neural", (1.0, 0.0), "high")
        opposite = morph_identity("opposite", 0.2, "epidermis", (-1.0, 0.0), "low")

        self.assertEqual(schema.compatibility_score(same, same_family), 1.0)
        self.assertEqual(schema.component_distance("polarity", same, opposite), 1.0)
        self.assertLess(schema.compatibility_score(same, opposite), schema.compatibility_score(same, same_family))

    def test_target_neighbor_preference_scores_mismatch(self) -> None:
        schema = default_morphogenesis_identity_schema()
        substrate = square_grid_substrate(2, 1)
        actor = morph_identity(
            "actor",
            0.1,
            "boundary",
            (1.0, 0.0),
            "medium",
            (TargetNeighborPreference("east", "organ_type", "neural"),),
        )
        mismatch = morph_identity("mismatch", 0.2, "epidermis", (1.0, 0.0), "medium")
        substrate.fill_sites(
            (
                attach_identity(SubstrateCell("actor", value=0.1), actor),
                attach_identity(SubstrateCell("mismatch", value=0.2), mismatch),
            )
        )

        relation = local_identity_observation(substrate, 0, schema).neighbor_relations[0]
        self.assertEqual(relation.direction, "east")
        self.assertEqual(relation.target_preference_score, 0.0)

    def test_schema_validation_rejects_missing_and_invalid_components(self) -> None:
        schema = default_morphogenesis_identity_schema()
        with self.assertRaises(ValueError):
            schema.validate_identity(CellIdentity("missing", {"ap_coordinate": 0.1}))
        with self.assertRaises(ValueError):
            schema.validate_identity(
                morph_identity("bad_category", 0.1, "heart", (1.0, 0.0), "high")
            )
        with self.assertRaises(ValueError):
            TargetNeighborPreference("left", "organ_type", "neural", relation="contains")


if __name__ == "__main__":
    unittest.main()
