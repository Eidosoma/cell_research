#!/usr/bin/env python3
"""Analyze one frozen S11 look and apply only its precision stopping rule."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from reference_simulator.model import canonical_json_bytes


ENDPOINTS = {
    "normalizedResidualError": ("normalizedResidualError", "additive"),
    "successByBudget": ("successByBudget", "additive"),
    "logCompletionOpportunity": ("completionOpportunity", "log"),
    "logS01UnitWeightFullCost": ("projection_s01UnitWeightFullCost", "log"),
}
PRIMARY = (
    ("E02-S01-E04", "successByBudget", "positive"),
    ("E02-S01-E06", "normalizedResidualError", "positive"),
    ("E02-S01-E03", "normalizedResidualError", "negative"),
    ("E02-S01-E08", "normalizedResidualError", "positive"),
)


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    pq.write_table(
        pa.Table.from_pandas(frame, preserve_index=False), path,
        compression="zstd", compression_level=9, use_dictionary=True,
        write_statistics=True, version="2.6",
    )


def read_runs(cache: Path, look: int) -> pd.DataFrame:
    paths = [cache / f"stage_{stage:02d}_confirmatory_runs.parquet" for stage in range(1, look + 1)]
    if any(not path.exists() for path in paths):
        raise FileNotFoundError("a required nested confirmatory stage is missing")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def paired_effects(runs: pd.DataFrame, contrast_path: Path, look: int) -> pd.DataFrame:
    mapping = pd.read_parquet(contrast_path)
    mapping = mapping[mapping.confirmatoryLook <= look]
    merge_columns = [
        "runDesignId", "pairingBlockId", "confirmatoryLook", "n", "valueProfile",
        "orderStructure", "policyProfile", "direction", "faultProfileId",
        "placementClass", "faultCount", *{item[0] for item in ENDPOINTS.values()},
    ]
    merged = mapping.merge(
        runs[merge_columns], on=["runDesignId", "pairingBlockId", "confirmatoryLook"],
        validate="many_to_one",
    )
    rows: list[dict[str, Any]] = []
    metadata = [
        "n", "valueProfile", "orderStructure", "policyProfile", "direction",
        "faultProfileId", "placementClass", "faultCount", "confirmatoryLook",
    ]
    for (estimand, block), group in merged.groupby(["estimandId", "pairingBlockId"], sort=True):
        roles = {row.contrastRole: row for row in group.itertuples(index=False)}
        if set(roles) != {"reference", "active"}:
            raise AssertionError(f"incomplete two-arm contrast for {estimand}/{block}")
        first = roles["reference"]
        row: dict[str, Any] = {
            "schemaVersion": "e02.s11.paired_effect.v1",
            "estimandId": estimand,
            "pairingBlockId": block,
            **{name: getattr(first, name) for name in metadata},
        }
        for endpoint, (column, transform) in ENDPOINTS.items():
            def value(role: str) -> float:
                raw = float(getattr(roles[role], column))
                return math.log(raw + 0.5) if transform == "log" else raw
            row[endpoint] = value("active") - value("reference")
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["estimandId", "pairingBlockId"]).reset_index(drop=True)


def _counter_index(design_hash: str, look: int, replicate: int, scale: int, draw: int, size: int) -> int:
    payload = f"{design_hash}/{look}/{replicate}/{scale}/{draw}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % size


def bootstrap_indices(effects: pd.DataFrame, design_hash: str, look: int, replicates: int) -> dict[int, np.ndarray]:
    indices: dict[int, np.ndarray] = {}
    for scale in sorted(effects.n.unique()):
        size = int((effects[(effects.estimandId == PRIMARY[0][0]) & (effects.n == scale)]).shape[0])
        array = np.empty((replicates, size), dtype=np.int32)
        for replicate in range(replicates):
            array[replicate] = [
                _counter_index(design_hash, look, replicate, int(scale), draw, size)
                for draw in range(size)
            ]
        indices[int(scale)] = array
    return indices


def bootstrap_draws(group: pd.DataFrame, endpoint: str, indices: dict[int, np.ndarray]) -> np.ndarray:
    total = len(group)
    result = np.zeros(next(iter(indices.values())).shape[0], dtype=float)
    for scale, part in group.groupby("n", sort=True):
        values = part.sort_values("pairingBlockId")[endpoint].to_numpy(float)
        result += values[indices[int(scale)]].sum(axis=1) / total
    return result


def bh_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.ones(len(p_values), dtype=float)
    running = 1.0
    for reverse_rank, index in enumerate(order[::-1], start=1):
        rank = len(p_values) - reverse_rank + 1
        running = min(running, p_values[index] * len(p_values) / rank)
        adjusted[index] = running
    return adjusted.tolist()


def classify(estimate: float, low: float, high: float, direction: str, precision: bool) -> tuple[str, bool]:
    practical = low > -0.02 and high < 0.02
    if not precision:
        return "inconclusive_due_to_precision", practical
    correct = low > 0 if direction == "positive" else high < 0
    opposite = high < 0 if direction == "positive" else low > 0
    if opposite:
        return "contradictory", practical
    if correct:
        return (
            "directionally_reproduced_material" if abs(estimate) >= 0.02
            else "directionally_reproduced_small",
            practical,
        )
    if practical:
        return "practically_equivalent_to_zero", practical
    return "inconclusive", practical


def analyze(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    look_dir = args.output / f"look_{args.look:02d}"
    look_dir.mkdir(parents=True, exist_ok=True)
    freeze = json.loads((args.output / "confirmatory_freeze_manifest.json").read_text())
    specification = json.loads((args.output / "confirmatory_prespecification.json").read_text())
    runs = read_runs(args.cache, args.look)
    expected_pairs = args.look * 1000
    if len(runs) != expected_pairs * 6 or runs.pairingBlockId.nunique() != expected_pairs:
        raise AssertionError("confirmatory look run accounting mismatch")
    effects = paired_effects(runs, args.contrast_map, args.look)
    if len(effects) != expected_pairs * 4:
        raise AssertionError("confirmatory paired-effect accounting mismatch")
    write_parquet(look_dir / "paired_effects.parquet", effects)

    replicates = int(specification["multiplicityAndIntervals"]["bootstrapReplicatesPerLook"])
    indices = bootstrap_indices(effects, freeze["designFreezeSha256"], args.look, replicates)
    all_draws: dict[tuple[str, str], np.ndarray] = {}
    secondary_rows: list[dict[str, Any]] = []
    for estimand in freeze["selectedEstimands"]:
        group = effects[effects.estimandId == estimand]
        for endpoint in ENDPOINTS:
            observed = float(group[endpoint].mean())
            draws = bootstrap_draws(group, endpoint, indices)
            all_draws[(estimand, endpoint)] = draws
            low, high = np.quantile(draws, [0.025, 0.975])
            p_value = min(1.0, 2 * min(
                (np.count_nonzero(draws <= 0) + 1) / (replicates + 1),
                (np.count_nonzero(draws >= 0) + 1) / (replicates + 1),
            ))
            secondary_rows.append({
                "schemaVersion": "e02.s11.secondary_effect.v1",
                "look": args.look, "pairCount": len(group), "estimandId": estimand,
                "endpoint": endpoint, "estimate": observed,
                "bootstrapLow95": float(low), "bootstrapHigh95": float(high),
                "bootstrapHalfWidth95": float((high - low) / 2),
                "bootstrapReplicates": replicates, "unadjustedBootstrapP": p_value,
            })
    q_values = bh_adjust([row["unadjustedBootstrapP"] for row in secondary_rows])
    for row, q_value in zip(secondary_rows, q_values, strict=True):
        row["terminalBhQ"] = q_value
    secondary = pd.DataFrame(secondary_rows)
    write_parquet(look_dir / "secondary_effects.parquet", secondary)

    primary_records = []
    observed_se = []
    centered_t = []
    for estimand, endpoint, direction in PRIMARY:
        group = effects[effects.estimandId == estimand]
        estimate = float(group[endpoint].mean())
        draws = all_draws[(estimand, endpoint)]
        se = float(np.std(draws, ddof=1))
        if se == 0:
            centered = np.zeros_like(draws)
        else:
            centered = (draws - estimate) / se
        observed_se.append((estimate, se))
        centered_t.append(centered)
        primary_records.append({
            "schemaVersion": "e02.s11.primary_effect.v1",
            "look": args.look, "pairCount": len(group), "estimandId": estimand,
            "endpoint": endpoint, "frozenDirection": direction,
            "estimate": estimate, "bootstrapStandardError": se,
        })
    max_abs_t = np.max(np.abs(np.column_stack(centered_t)), axis=1)
    critical = float(np.quantile(max_abs_t, 0.99))
    for record, (estimate, se) in zip(primary_records, observed_se, strict=True):
        low, high = estimate - critical * se, estimate + critical * se
        half_width = critical * se
        z = math.inf if se == 0 and estimate != 0 else (abs(estimate / se) if se else 0.0)
        adjusted_p = (np.count_nonzero(max_abs_t >= z) + 1) / (replicates + 1)
        precision = half_width <= 0.015
        classification, practical = classify(
            estimate, low, high, record["frozenDirection"], precision
        )
        record.update({
            "simultaneousLow99": low, "simultaneousHigh99": high,
            "simultaneousHalfWidth99": half_width,
            "familyCriticalMaxAbsT": critical,
            "sequentialMultiplicityAdjustedP": float(adjusted_p),
            "precisionTarget": 0.015, "precisionPass": precision,
            "equivalenceMargin": 0.02, "practicalEquivalent": practical,
            "classificationAtLook": classification,
        })
    primary = pd.DataFrame(primary_records)
    write_parquet(look_dir / "primary_effects.parquet", primary)

    precision_pass = bool(primary.precisionPass.all())
    continue_sampling = not precision_pass and args.look < 5
    action = (
        "continue_for_frozen_primary_precision" if continue_sampling
        else ("stop_for_frozen_primary_precision" if precision_pass else "stop_at_frozen_maximum_inconclusive_precision")
    )
    decision = {
        "schemaVersion": "e02.s11.adaptation_decision.v1",
        "researchStepId": "S11", "look": args.look,
        "cumulativePairingBlocksPerContrast": expected_pairs,
        "cumulativeRunRows": len(runs), "cumulativePairedEffectRows": len(effects),
        "primaryPrecisionPass": precision_pass,
        "primaryPrecisionPassByEstimand": {
            row.estimandId: bool(row.precisionPass) for row in primary.itertuples(index=False)
        },
        "continue": continue_sampling, "action": action,
        "decisionInputs": "only simultaneous primary half-widths versus 0.015 and the frozen maximum look",
        "observedDirectionUsedForAdaptation": False,
        "significanceUsedForAdaptation": False,
        "secondaryEndpointUsedForAdaptation": False,
        "runtimeUsedForAdaptation": False,
        "contrastOrArmDropped": False,
        "designFreezeSha256": freeze["designFreezeSha256"],
        "ruleCompliant": True,
    }
    write_json(look_dir / "adaptation_decision.json", decision)
    write_json(look_dir / "bootstrap_validation.json", {
        "schemaVersion": "e02.s11.bootstrap_validation.v1",
        "researchStepId": "S11", "look": args.look,
        "bootstrapReplicates": replicates, "scaleStratified": True,
        "scales": sorted(map(int, effects.n.unique())),
        "pairsPerScalePerContrast": {
            str(n): int(len(effects[(effects.estimandId == PRIMARY[0][0]) & (effects.n == n)]))
            for n in sorted(effects.n.unique())
        },
        "familyCriticalMaxAbsT": critical,
        "deterministicAddressRule": "SHA256(design-freeze/look/replicate/scale/draw)",
        "success": True,
    })
    print(json.dumps({
        "look": args.look, "pairs": expected_pairs, "runs": len(runs),
        "action": action, "primaryPrecisionPass": precision_pass,
        "primary": primary[["estimandId", "endpoint", "estimate", "simultaneousHalfWidth99", "classificationAtLook"]].to_dict("records"),
    }, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--look", type=int, required=True, choices=range(1, 6))
    parser.add_argument("--cache", type=Path, default=Path("/cache/s11"))
    parser.add_argument("--output", type=Path, default=Path("/artifacts/research_steps/S11"))
    parser.add_argument("--contrast-map", type=Path, default=Path("/artifacts/research_steps/S11/confirmatory_contrast_map.parquet"))
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
