import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "historical_backend" / "legacy_adapter.py"
SPEC = importlib.util.spec_from_file_location("legacy_adapter_under_test", MODULE_PATH)
legacy_adapter = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(legacy_adapter)


class HistoricalBackendAdapterTests(unittest.TestCase):
    def test_checksum_validation_accepts_exact_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            payload = b"immutable historical bytes\n"
            (source / "sample.py").write_bytes(payload)
            checksum = hashlib.sha256(payload).hexdigest()
            manifest = root / "checksums.sha256"
            manifest.write_text(f"{checksum}  sample.py\n", encoding="utf-8")
            result = legacy_adapter.verify_source_tree(source, manifest)
            self.assertTrue(result["contentValid"])
            self.assertEqual(result["filesChecked"], 1)

    def test_checksum_validation_rejects_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "sample.py").write_text("mutated\n", encoding="utf-8")
            manifest = root / "checksums.sha256"
            manifest.write_text(f"{'0' * 64}  sample.py\n", encoding="utf-8")
            result = legacy_adapter.verify_source_tree(source, manifest)
            self.assertFalse(result["contentValid"])
            self.assertEqual(len(result["mismatches"]), 1)

    def test_checksum_manifest_rejects_parent_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "checksums.sha256"
            manifest.write_text(f"{'0' * 64}  ../outside\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                legacy_adapter.read_checksum_manifest(manifest)

    def test_non_strict_sortedness_matches_frozen_driver(self):
        class Cell:
            def __init__(self, value):
                self.value = value

        self.assertTrue(legacy_adapter._is_sorted([Cell(1), Cell(1), Cell(2)]))
        self.assertFalse(legacy_adapter._is_sorted([Cell(2), Cell(1)]))


if __name__ == "__main__":
    unittest.main()
