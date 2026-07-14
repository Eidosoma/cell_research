#!/usr/bin/env python3
"""Promote the terminal S10 look and freeze the prespecified S11 selections."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from reference_simulator.model import canonical_json_bytes, sha256_json


ENDPOINTS = (
    "normalizedResidualError", "successByBudget",
    "logCompletionOpportunity", "logS01UnitWeightFullCost",
)
PRECISION_THRESHOLDS = {
    "normalizedResidualError": 0.02,
    "successByBudget": 0.02,
    "logCompletionOpportunity": math.log(1.10),
    "logS01UnitWeightFullCost": math.log(1.10),
}


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    pq.write_table(
        pa.Table.from_pandas(frame, preserve_index=False), path,
        compression="zstd", compression_level=9, use_dictionary=True,
        write_statistics=True, version="2.6",
    )


def direction(value: float) -> str:
    if value > 0:
        return "positive_active_minus_reference"
    if value < 0:
        return "negative_active_minus_reference"
    return "zero"


def finalize(args: argparse.Namespace) -> None:
    look_dir = args.output / f"look_{args.look:02d}"
    decision = json.loads((look_dir / "adaptation_decision.json").read_text())
    if decision["continue"]:
        raise AssertionError("cannot finalize a look whose frozen rule says continue")
    if decision["action"] not in {"stop_for_frozen_precision_and_stability", "stop_at_frozen_maximum"}:
        raise AssertionError(f"cannot freeze scientific selections after {decision['action']}")

    stage_paths = [args.cache / f"stage_{stage:02d}_screening_runs.parquet" for stage in range(1, args.look + 1)]
    runs = pd.concat([pd.read_parquet(path) for path in stage_paths], ignore_index=True).sort_values("runDesignId").reset_index(drop=True)
    expected_blocks = args.look * 50
    if len(runs) != expected_blocks * 14 or runs.pairingBlockId.nunique() != expected_blocks:
        raise AssertionError("terminal run accounting mismatch")
    write_parquet(args.output / "screening_runs.parquet", runs)

    for name in (
        "paired_effects.parquet", "bootstrap_intervals.parquet",
        "sparse_model_coefficients.parquet", "interaction_ranking.parquet",
        "sparse_model_validation.json", "parity_and_negative_control.json",
    ):
        shutil.copy2(look_dir / name, args.output / name)
    shutil.copy2(look_dir / "bootstrap_intervals.parquet", args.output / "screening_effect_estimates.parquet")

    cost_columns = [
        "runDesignId", "pairingBlockId", "n", "screeningStage",
        "treatmentSignature", "architecture", "coordinatorProfile", "scheduler",
        "mobility", "continuation", "retry", "nativeLedgerJson",
        "costSchemaVersion", "costSchemaSha256", "contractValidationPass",
        *sorted(column for column in runs if column.startswith(("cost_", "projection_", "normalization_"))),
    ]
    write_parquet(args.output / "cost_ledger.parquet", runs[cost_columns])

    effects = pd.read_parquet(look_dir / "paired_effects.parquet")
    intervals = pd.read_parquet(look_dir / "bootstrap_intervals.parquet")
    ranking = pd.read_parquet(look_dir / "interaction_ranking.parquet")
    coefficients = pd.read_parquet(look_dir / "sparse_model_coefficients.parquet")
    sensitivity_rows = []
    for estimand, group in effects.groupby("estimandId", sort=True):
        for endpoint in ENDPOINTS:
            full = float(group[endpoint].mean())
            sensitivity_rows.append({
                "estimandId": estimand, "endpoint": endpoint,
                "analysis": "full_unshrunk", "excludedScale": None,
                "pairCount": len(group), "estimate": full,
                "direction": direction(full), "deltaFromFull": 0.0,
            })
            for excluded in sorted(group.n.unique()):
                subset = group[group.n != excluded]
                estimate = float(subset[endpoint].mean())
                sensitivity_rows.append({
                    "estimandId": estimand, "endpoint": endpoint,
                    "analysis": "leave_one_scale_out", "excludedScale": int(excluded),
                    "pairCount": len(subset), "estimate": estimate,
                    "direction": direction(estimate), "deltaFromFull": estimate - full,
                })
    sensitivity = pd.DataFrame(sensitivity_rows)
    write_parquet(args.output / "leave_one_scale_out_sensitivity.parquet", sensitivity)

    candidates = set(ranking.estimandId)
    candidate_sensitivity = sensitivity[sensitivity.estimandId.isin(candidates)]
    full_lookup = candidate_sensitivity[candidate_sensitivity.analysis == "full_unshrunk"].set_index(["estimandId", "endpoint"]).estimate
    loso = candidate_sensitivity[candidate_sensitivity.analysis == "leave_one_scale_out"].copy()
    loso["sameDirectionAsFull"] = loso.apply(
        lambda row: direction(float(row.estimate)) == direction(float(full_lookup.loc[(row.estimandId, row.endpoint)])), axis=1,
    )
    main = coefficients[coefficients.term == "main"].copy()
    penalty_direction = []
    for (estimand, endpoint), group in main.groupby(["estimandId", "endpoint"], sort=True):
        unshrunk = float(full_lookup.loc[(estimand, endpoint)])
        signs = {direction(unshrunk), *(direction(float(value)) for value in group.coefficient)}
        penalty_direction.append({
            "estimandId": estimand, "endpoint": endpoint,
            "models": int(group.model.nunique()), "unshrunkEstimate": unshrunk,
            "allPenaltyDirectionsAgreeWithUnshrunk": len(signs) == 1,
            "directionsJson": canonical_json_bytes({row.model: direction(float(row.coefficient)) for row in group.itertuples(index=False)}).decode("utf-8"),
        })
    penalty_frame = pd.DataFrame(penalty_direction)
    write_parquet(args.output / "penalty_direction_sensitivity.parquet", penalty_frame)
    e05 = effects[effects.estimandId == "E02-S01-E05"]
    e05_zero = bool((e05[list(ENDPOINTS)].abs() <= 1e-15).all(axis=None))
    selected_ids = set(decision["selectedEstimands"])
    selected_loso = loso[loso.estimandId.isin(selected_ids)]
    selected_penalty = penalty_frame[penalty_frame.estimandId.isin(selected_ids)]
    sensitivity_summary = {
        "schemaVersion": "e02.s10.sensitivity_validation.v1",
        "researchStepId": "S10",
        "look": args.look,
        "leaveOneScaleOutRows": len(loso),
        "leaveOneScaleOutDirectionAgreement": int(loso.sameDirectionAsFull.sum()),
        "leaveOneScaleOutDirectionTotal": len(loso),
        "maximumAbsoluteLeaveOneScaleDeltaByEndpoint": {
            endpoint: float(part.deltaFromFull.abs().max())
            for endpoint, part in loso.groupby("endpoint", sort=True)
        },
        "penaltyDirectionAgreementRows": int(penalty_frame.allPenaltyDirectionsAgreeWithUnshrunk.sum()),
        "penaltyDirectionTotal": len(penalty_frame),
        "selectedLeaveOneScaleOutDirectionAgreement": int(selected_loso.sameDirectionAsFull.sum()),
        "selectedLeaveOneScaleOutDirectionTotal": len(selected_loso),
        "selectedPenaltyDirectionAgreementRows": int(selected_penalty.allPenaltyDirectionsAgreeWithUnshrunk.sum()),
        "selectedPenaltyDirectionTotal": len(selected_penalty),
        "e05InactiveRows": len(e05),
        "e05AllEndpointContrastsExactlyZero": e05_zero,
        "e05ExclusionSensitivity": "exactly invariant for estimand-only means and block-diagonal estimand feature columns; E05 has no candidate feature and all observed contrasts are zero",
        "success": e05_zero and len(loso) > 0 and len(penalty_frame) > 0,
    }
    write_json(args.output / "sensitivity_validation.json", sensitivity_summary)

    selected = list(decision["selectedEstimands"])
    rank_lookup = ranking.set_index("estimandId")
    interval_lookup = intervals.set_index(["estimandId", "endpoint"])
    records = []
    for priority, estimand in enumerate(selected, start=1):
        item = rank_lookup.loc[estimand]
        endpoint_records = {}
        for endpoint in ENDPOINTS:
            interval = interval_lookup.loc[(estimand, endpoint)]
            endpoint_records[endpoint] = {
                "estimate": float(interval.estimate),
                "direction": direction(float(interval.estimate)),
                "bootstrapLow95": float(interval.bootstrapLow95),
                "bootstrapHigh95": float(interval.bootstrapHigh95),
                "bootstrapHalfWidth": float(interval.bootstrapHalfWidth),
                "precisionThreshold": PRECISION_THRESHOLDS[endpoint],
                "precisionPass": bool(float(interval.bootstrapHalfWidth) <= PRECISION_THRESHOLDS[endpoint]),
            }
        records.append({
            "priority": priority,
            "estimandId": estimand,
            "utilityRank": int(item.utilityRank),
            "utilityScore": float(item.utilityScore),
            "inconclusiveUnderFrozenRule": bool(item.inconclusive),
            "primaryDirectionConcordant": bool(item.primaryDirectionConcordant),
            "endpointEffects": endpoint_records,
        })
    rule_bytes = args.selection_rule.read_bytes()
    selection_path = args.output / "s11_frozen_selections.json"
    existing_frozen_at = None
    if selection_path.exists():
        existing_frozen_at = json.loads(selection_path.read_text()).get("frozenAtUtc")
    selection = {
        "schemaVersion": "e02.s10.s11_frozen_selections.v1",
        "researchStepId": "S10",
        "appliesToNextStep": "S11",
        "frozenAtUtc": existing_frozen_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "terminalLook": args.look,
        "cumulativePairingBlocks": expected_blocks,
        "stopAction": decision["action"],
        "precisionPass": decision["precisionPass"],
        "selectionStable": decision["selectionStable"],
        "rankSpearman": decision["rankSpearman"],
        "exactParityGatePass": decision["exactParityPass"],
        "selectedEstimands": selected,
        "selectionRecords": records,
        "ineligible": {
            "E02-S01-E05": "retry_inactive_under_frozen_actionFailure_none",
            "E02-S01-E07": "non_executable_under_preserved_information_boundary",
        },
        "selectionRulePath": str(args.selection_rule),
        "selectionRuleSha256": hashlib.sha256(rule_bytes).hexdigest(),
        "protectedOutcomeReads": 0,
        "confirmatoryOutcomesOpened": False,
        "frozen": True,
    }
    selection["selectionSha256"] = sha256_json(selection)
    write_json(selection_path, selection)
    print(json.dumps({
        "look": args.look, "runs": len(runs), "selected": selected,
        "stopAction": decision["action"], "precisionPass": decision["precisionPass"],
        "sensitivity": sensitivity_summary,
    }, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--look", type=int, required=True, choices=range(1, 6))
    parser.add_argument("--cache", type=Path, default=Path("/cache/s10"))
    parser.add_argument("--output", type=Path, default=Path("/artifacts/research_steps/S10"))
    parser.add_argument("--selection-rule", type=Path, default=Path("design/s10/s11_selection_rule.json"))
    finalize(parser.parse_args())


if __name__ == "__main__":
    main()
