from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from morphospace2d import (
    TARGET_SCHEMA_VERSION,
    audit_local_target_payload,
    build_standard_target_library,
    evaluate_target_energy,
    load_target_library_json,
    render_target_panel,
    scrambled_state,
    write_target_library_json,
)


class TestE05TargetMorphologies(unittest.TestCase):
    def test_standard_library_covers_required_motifs(self) -> None:
        targets = build_standard_target_library()
        motifs = {target.motif for target in targets}
        self.assertTrue({"gradient", "stripes", "ring", "sorted_row", "boundary", "organ_like"}.issubset(motifs))
        self.assertTrue(all(target.schema_version == TARGET_SCHEMA_VERSION for target in targets))
        self.assertTrue(all(target.validate() == [] for target in targets))

    def test_target_energy_returns_zero_on_exact_target_states(self) -> None:
        for target in build_standard_target_library():
            with self.subTest(target=target.target_id):
                energy = evaluate_target_energy(target, target.target_state())
                self.assertEqual(energy["totalEnergy"], 0.0)
                self.assertTrue(energy["withinTolerance"])

    def test_scrambled_states_have_positive_energy(self) -> None:
        for target in build_standard_target_library():
            with self.subTest(target=target.target_id):
                energy = evaluate_target_energy(target, scrambled_state(target))
                self.assertGreater(energy["totalEnergy"], 0.0)

    def test_target_configs_round_trip_through_json(self) -> None:
        targets = build_standard_target_library()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "targets.json"
            write_target_library_json(path, targets)
            loaded = load_target_library_json(path)
        self.assertEqual([target.target_id for target in loaded], [target.target_id for target in targets])
        for target in loaded:
            self.assertEqual(evaluate_target_energy(target, target.target_state())["totalEnergy"], 0.0)

    def test_local_policy_payloads_do_not_expose_global_maps_or_hidden_fields(self) -> None:
        for target in build_standard_target_library():
            for position in target.substrate.nodes[:3]:
                with self.subTest(target=target.target_id, position=position):
                    payload = target.local_policy_payload(position)
                    self.assertEqual(audit_local_target_payload(payload), [])
                    self.assertNotIn("all_target_positions", payload)
                    self.assertNotIn("target_map", payload)
                    self.assertNotIn("internal_state", payload["actorVisibleIdentity"])

    def test_render_target_panel_writes_nonempty_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "target_panel.png"
            render_target_panel(build_standard_target_library(), path)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
