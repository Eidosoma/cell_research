#!/usr/bin/env python3
"""Execute S08M in its prospectively frozen, genuinely fresh namespace."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))
os.environ["E07_PORTFOLIO_STEP_ID"] = "S08M"
os.environ["E07_PORTFOLIO_CONTROL_PATH"] = str(
    REPOSITORY / "configs/portfolio/s08m_execution_continuation.yaml"
)
os.environ["E07_PORTFOLIO_CACHE_ROOT"] = "/cache/e07-s08m"

ENGINE_PATH = REPOSITORY / "scripts/execute_portfolio_search_s08g.py"
SPEC = importlib.util.spec_from_file_location("e07_s08_execution_engine", ENGINE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load frozen execution engine: {ENGINE_PATH}")
ENGINE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ENGINE)
main = ENGINE.main


if __name__ == "__main__":
    raise SystemExit(main())
