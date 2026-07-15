#!/usr/bin/env python3
"""Apply the S09 reachability-safe exact-effect classification to built tables."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path("/artifacts/research_steps/S09")
REPOSITORY = Path(__file__).resolve().parents[1]


def write(frame: pd.DataFrame, name: str, schema: str) -> None:
    table = pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(
        {b"schemaVersion": schema.encode(), b"researchStepId": b"S09"}
    )
    pq.write_table(
        table, ROOT / name, compression="zstd", compression_level=9,
        use_dictionary=True, write_statistics=True,
    )


def main() -> None:
    optimum = pq.read_table(ROOT / "structural_optimum_changes.parquet").to_pandas()
    optimum["arm_goal_reachable"] = optimum.arm_classification.isin(
        ["complete_start", "reachable_no_detour", "necessary_detour"]
    )
    optimum["delta_minimum_excursion"] = np.where(
        optimum.arm_goal_reachable,
        optimum.arm_minimum_excursion - optimum.baseline_minimum_excursion,
        np.nan,
    )
    optimum["necessity_resolved_reachable_no_detour"] = (
        (optimum.baseline_classification == "necessary_detour")
        & (optimum.arm_classification == "reachable_no_detour")
    )
    optimum["necessary_detour_persisted"] = (
        (optimum.baseline_classification == "necessary_detour")
        & (optimum.arm_classification == "necessary_detour")
    )
    optimum["reachability_destroyed"] = (
        (optimum.baseline_classification == "necessary_detour")
        & optimum.arm_classification.isin(["unreachable_active", "quiescent"])
    )
    if "necessity_removed" in optimum:
        optimum = optimum.drop(columns=["necessity_removed"])
    write(
        optimum,
        "structural_optimum_changes.parquet",
        "e03.s09.structural_optimum_changes.v2",
    )

    pairs = pq.read_table(ROOT / "paired_effects.parquet").to_pandas()
    metrics = pq.read_table(ROOT / "paired_metric_effects.parquet").to_pandas()
    module_path = REPOSITORY / "scripts/build_s09_interventions.py"
    spec = importlib.util.spec_from_file_location("s09_builder", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    summary = module.effect_summaries(pairs, metrics, optimum)
    write(summary, "effect_summary.parquet", "e03.s09.effect_summary.v2")
    print(
        {
            "optimumRows": len(optimum),
            "effectSummaryRows": len(summary),
            "reachabilitySafe": True,
        }
    )


if __name__ == "__main__":
    main()
