from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "freeze_historical_artifact.py"


def load_module():
    spec = importlib.util.spec_from_file_location("freeze_historical_artifact", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FreezeHistoricalArtifactTests(unittest.TestCase):
    def test_git_blob_identity_matches_known_vector(self) -> None:
        module = load_module()
        self.assertEqual(
            module.git_blob_sha1(b"test content\n"),
            "d670460b4b4aece5915caf5c68d12f560a9fe3e4",
        )

    def test_source_analysis_is_static_and_finds_raw_reference(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "analysis.py"
            source.write_text(
                "import numpy as np\n"
                "values = np.load('/Users/example/missing.npy', allow_pickle=True)\n"
                "if __name__ == '__main__':\n"
                "    np.save('generated.npy', values)\n",
                encoding="utf-8",
            )
            tree = [{"path": "analysis.py", "suffix": ".py"}]
            imports, raw, scripts, summary = module.source_analysis(root, tree)

            self.assertEqual(len(imports), 1)
            self.assertEqual(len(raw), 2)
            self.assertEqual(summary["activeAllowPickleTrueLoads"], 1)
            self.assertEqual(summary["absoluteAuthorNpyPathReferenceLines"], 1)
            self.assertEqual(summary["absoluteAuthorPathReferenceLinesAll"], 1)
            self.assertEqual(scripts[0]["absoluteAuthorPathLines"], 1)
            self.assertEqual(scripts[0]["astParseResult"], "pass")
            self.assertTrue(scripts[0]["hasMainGuard"])

    def test_archive_path_screen_rejects_parent_traversal(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "test.zip"
            with __import__("zipfile").ZipFile(archive, "w") as handle:
                handle.writestr("root/good.txt", "ok")
                handle.writestr("../escape.txt", "bad")
            contents, unsafe = module.archive_bytes(archive, "zip")
            self.assertIn("good.txt", contents)
            self.assertEqual(unsafe, ["../escape.txt"])


if __name__ == "__main__":
    unittest.main()
