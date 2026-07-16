#!/usr/bin/env python3
"""Run the mounted E01 semantic fixtures for the E03 S14 release audit."""

from __future__ import annotations

from pathlib import Path

import pytest


FIXTURE = Path("/previous-artifacts/E01/research_steps/S03/toy_fixtures.json")
JUNIT = "/artifacts/research_steps/S14/mounted_reference_tests.junit.xml"


class MountedFixturePlugin:
    def pytest_collection_modifyitems(self, session, config, items) -> None:
        for item in items:
            module = getattr(item, "module", None)
            if module is not None and hasattr(module, "S03_FIXTURES"):
                module.S03_FIXTURES = FIXTURE


def main() -> int:
    return pytest.main(
        ["-q", f"--junitxml={JUNIT}", "tests/test_reference_simulator.py"],
        plugins=[MountedFixturePlugin()],
    )


if __name__ == "__main__":
    raise SystemExit(main())
