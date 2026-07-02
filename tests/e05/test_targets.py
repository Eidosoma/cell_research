"""E05 S03 target morphology tests."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.e05.cell_identity import attach_identity, scalar_identity
from src.e05.substrates import SubstrateCell
from src.e05.targets import (
    boundary_target,
    constructed_target_error,
    default_target_gallery,
    gradient_target,
    organ_like_target,
    render_target_gallery,
    ring_target,
    sorted_row_target,
    stripes_target,
    symmetry_target,
    target_render_mode,
    target_summary_rows,
)


class TargetMorphologyTests(unittest.TestCase):
    def test_sorted_row_target_recovers_scalar_sorted_values_and_zero_error(self) -> None:
        target = sorted_row_target((3, 1, 2))
        self.assertEqual(
            [target.identities_by_site[site_id].components["value"] for site_id in target.substrate.site_ids],
            [1.0, 2.0, 3.0],
        )
        self.assertEqual(constructed_target_error(target), 0.0)

    def test_gradient_target_has_expected_ap_coordinates_and_zero_error(self) -> None:
        target = gradient_target(4, 2)
        by_coordinate = {
            target.substrate.coordinate(site_id): identity.components["ap_coordinate"]
            for site_id, identity in target.identities_by_site.items()
        }
        self.assertEqual(by_coordinate[(0, 0)], 0.0)
        self.assertEqual(by_coordinate[(3, 1)], 1.0)
        self.assertEqual(constructed_target_error(target), 0.0)

    def test_stripes_boundary_ring_and_organ_targets_have_zero_error(self) -> None:
        for target in (
            stripes_target(4, 3),
            boundary_target(4, 3),
            ring_target(5, 5),
            organ_like_target(6, 4),
        ):
            with self.subTest(target=target.target_id):
                self.assertEqual(constructed_target_error(target), 0.0)
                organs = {
                    identity.components["organ_type"]
                    for identity in target.identities_by_site.values()
                }
                self.assertGreaterEqual(len(organs), 2)

    def test_boundary_target_marks_perimeter(self) -> None:
        target = boundary_target(4, 3)
        perimeter = {
            site_id
            for site_id in target.substrate.site_ids
            if target.identities_by_site[site_id].components["organ_type"] == "boundary"
        }
        self.assertEqual(perimeter, {0, 1, 2, 3, 4, 7, 8, 9, 10, 11})

    def test_symmetry_target_mirrors_organ_labels(self) -> None:
        target = symmetry_target(7, 5)
        width = target.substrate.dimensions["width"]
        height = target.substrate.dimensions["height"]
        for y in range(height):
            for x in range(width):
                left_site = y * width + x
                right_site = y * width + (width - 1 - x)
                self.assertEqual(
                    target.identities_by_site[left_site].components["organ_type"],
                    target.identities_by_site[right_site].components["organ_type"],
                )
        self.assertEqual(constructed_target_error(target), 0.0)

    def test_target_error_is_positive_for_wrong_constructed_state(self) -> None:
        target = sorted_row_target((1, 2, 3))
        state = target.substrate.copy_empty()
        wrong_cells = (
            attach_identity(SubstrateCell("wrong_0", value=3), scalar_identity(3, "wrong_0")),
            attach_identity(SubstrateCell("wrong_1", value=2), scalar_identity(2, "wrong_1")),
            attach_identity(SubstrateCell("wrong_2", value=1), scalar_identity(1, "wrong_2")),
        )
        state.fill_sites(wrong_cells)
        self.assertGreater(target.target_error(state), 0.0)

    def test_target_summary_rows_cover_gallery(self) -> None:
        targets = default_target_gallery()
        rows = target_summary_rows(targets)
        self.assertEqual(len(rows), len(targets))
        self.assertTrue(all(row["constructed_target_error"] == 0.0 for row in rows))
        self.assertTrue(all(row["target_hash"] for row in rows))
        modes_by_kind = {row["target_kind"]: row["render_mode"] for row in rows}
        self.assertEqual(modes_by_kind["sorted_row"], "value")
        self.assertEqual(modes_by_kind["gradient"], "ap_coordinate")
        self.assertEqual(modes_by_kind["stripes"], "organ_type")

    def test_target_render_mode_uses_gradient_coordinate(self) -> None:
        self.assertEqual(target_render_mode(sorted_row_target((3, 1, 2))), "value")
        self.assertEqual(target_render_mode(gradient_target(4, 2)), "ap_coordinate")
        self.assertEqual(target_render_mode(stripes_target(4, 3)), "organ_type")

    def test_render_target_gallery_writes_nonempty_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "gallery.png"
            render_target_gallery(default_target_gallery(), output)
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 1_000)
            self.assertEqual(output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
