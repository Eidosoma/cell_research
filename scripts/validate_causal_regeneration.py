#!/usr/bin/env python3
"""Regenerate deterministic S12 design-based outputs and compare them exactly."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from causal_simulator.causal_modeling import canonical_json_bytes  # noqa: E402


TABLES = (
    "model_arm_data.parquet",
    "paired_analysis_data.parquet",
    "marginal_effects.parquet",
    "heterogeneity_effects.parquet",
    "two_way_heterogeneity.parquet",
    "heterogeneity_omnibus.parquet",
    "heterogeneity_rankings.parquet",
    "unsupported_heterogeneity_cells.parquet",
    "rng_unpaired_sensitivity.parquet",
    "survival_summaries.parquet",
    "survival_curves.parquet",
    "leave_one_scale_out.parquet",
    "e04_cost_components.parquet",
    "cost_transform_sensitivity.parquet",
)
JSONS = (
    "bootstrap_stability.json",
    "e04_resource_tradeoff.json",
    "effect_execution_summary.json",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, default=Path("/artifacts/research_steps/S12"))
    parser.add_argument("--regenerated", type=Path, default=Path("/cache/s12_regenerated"))
    args = parser.parse_args()
    if args.regenerated.exists():
        shutil.rmtree(args.regenerated)
    args.regenerated.mkdir(parents=True)
    shutil.copy2(
        args.reference / "model_freeze_manifest.json",
        args.regenerated / "model_freeze_manifest.json",
    )
    environment = os.environ.copy()
    environment.update({
        "PYTHONPATH": str(REPOSITORY),
        "OPENBLAS_NUM_THREADS": "8",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONWARNINGS": "ignore",
    })
    command = [
        sys.executable, str(REPOSITORY / "scripts/run_causal_effects.py"),
        "--output", str(args.regenerated), "--bootstrap-replicates", "2000",
    ]
    completed = subprocess.run(
        command, cwd=REPOSITORY, env=environment, text=True,
        capture_output=True, check=True,
    )
    comparisons = {}
    for name in TABLES:
        reference = pd.read_parquet(args.reference / name)
        regenerated = pd.read_parquet(args.regenerated / name)
        try:
            pd.testing.assert_frame_equal(
                reference, regenerated, check_exact=True, check_dtype=True,
                check_categorical=True,
            )
            comparisons[name] = True
        except AssertionError:
            comparisons[name] = False
    for name in JSONS:
        comparisons[name] = json.loads((args.reference / name).read_text()) == json.loads(
            (args.regenerated / name).read_text()
        )
    record = {
        "schemaVersion": "e02.s12.deterministic_regeneration_validation.v1",
        "researchStepId": "S12",
        "command": " ".join(command),
        "threadEnvironment": {
            key: environment[key]
            for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
        "comparisons": comparisons,
        "tablesCompared": len(TABLES),
        "jsonDocumentsCompared": len(JSONS),
        "subprocessStdout": completed.stdout.strip(),
        "success": all(comparisons.values()),
    }
    path = args.reference / "deterministic_regeneration_validation.json"
    path.write_bytes(canonical_json_bytes(record) + b"\n")
    print(json.dumps({"success": record["success"], "comparisons": len(comparisons)}, sort_keys=True))
    if not record["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
