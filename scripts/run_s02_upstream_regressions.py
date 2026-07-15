#!/usr/bin/env python3
"""Run relevant E01 regression tests against their read-only E03 mount paths."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import analysis.chimeric_replication as chimeric
import analysis.delayed_gratification as delayed
import analysis.frozen_cell_results as frozen
import analysis.no_fault_sorting as no_fault


REPOSITORY = Path(__file__).resolve().parents[1]
E01_S08 = Path("/previous-artifacts/E01/research_steps/S08")
FROZEN_DG_SHA256 = "7ebf961eb7cced1ab196f2314718c007e2b9d164b8dfa161d407dae6086bc721"


def main() -> int:
    frozen_source = REPOSITORY / "analysis/delay_gratification_analysis_for_not_move.py"
    actual = hashlib.sha256(frozen_source.read_bytes()).hexdigest()
    if actual != FROZEN_DG_SHA256:
        raise RuntimeError("current frozen DG source differs from the E01 checksum manifest")
    no_fault.S08_DIR = E01_S08
    frozen.S08_DIR = E01_S08
    chimeric.S08_DIR = E01_S08
    delayed.FROZEN_DG_SOURCE = frozen_source
    return pytest.main(
        [
            "-q",
            "tests/test_detour_replay.py",
            "tests/test_distances.py",
            "tests/test_no_fault_sorting.py",
            "tests/test_frozen_cell_results.py",
            "tests/test_delayed_gratification.py",
            "tests/test_chimeric_replication.py",
            "--junitxml=/artifacts/research_steps/S02/regression_tests_mounted.junit.xml",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
