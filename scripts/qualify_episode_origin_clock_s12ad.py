"""Repository entry point for the frozen S12AD qualification runner."""

from __future__ import annotations

import runpy
from pathlib import Path

RUNNER = Path(
    "/artifacts/research_steps/S12AD/code/"
    "run_formal_episode_origin_remedy_qualification_s12ad.py"
)


def main() -> None:
    if not RUNNER.is_file():
        raise RuntimeError("the frozen S12AD formal runner is unavailable")
    runpy.run_path(str(RUNNER), run_name="__main__")


if __name__ == "__main__":
    main()
