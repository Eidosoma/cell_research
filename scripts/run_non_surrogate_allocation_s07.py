#!/usr/bin/env python3
"""Run the approved S07 non-surrogate allocation in fail-closed stages."""

from __future__ import annotations

import argparse
import json

from src.non_surrogate_allocation.analysis import analyze
from src.non_surrogate_allocation.core import freeze_execution_plan, run_smoke, run_substantive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "smoke", "execute", "analyze"))
    args = parser.parse_args()
    result = {
        "freeze": freeze_execution_plan,
        "smoke": run_smoke,
        "execute": run_substantive,
        "analyze": analyze,
    }[args.stage]()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
