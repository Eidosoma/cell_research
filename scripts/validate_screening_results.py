#!/usr/bin/env python3
"""Validate terminal S10 accounting, contracts, adaptation, and separation."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from causal_simulator.costing import COST_SCHEMA_SHA256, COST_SCHEMA_VERSION, S01_ADDITIVE_FIELDS
from reference_simulator.model import canonical_json_bytes


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def validate(args: argparse.Namespace) -> None:
    runs = pd.read_parquet(args.output / "screening_runs.parquet")
    effects = pd.read_parquet(args.output / "paired_effects.parquet")
    pairing = pd.read_parquet(args.output / "screening_pairing_blocks.parquet")
    design = pd.read_parquet(args.output / "screening_design.parquet")
    mapping = pd.read_parquet(args.output / "screening_contrast_map.parquet")
    ranking = pd.read_parquet(args.output / "interaction_ranking.parquet")
    coefficients = pd.read_parquet(args.output / "sparse_model_coefficients.parquet")
    freeze = json.loads((args.output / "design_freeze_manifest.json").read_text())
    adaptation = json.loads((args.output / "adaptation_log.json").read_text())
    selection = json.loads((args.output / "s11_frozen_selections.json").read_text())
    parity = json.loads((args.output / "parity_and_negative_control.json").read_text())
    model = json.loads((args.output / "sparse_model_validation.json").read_text())
    replay = json.loads((args.output / "replay_validation.json").read_text())
    sensitivity = json.loads((args.output / "sensitivity_validation.json").read_text())

    look = int(selection["terminalLook"])
    expected_blocks = look * 50
    expected_runs = expected_blocks * 14
    expected_effects = expected_blocks * 11
    run_ids = set(runs.runDesignId)
    terminal_design = design[design.screeningStage <= look]
    terminal_pairing = pairing[pairing.screeningStage <= look]
    terminal_mapping = mapping[mapping.screeningStage <= look]

    ledger_identity_pass = 0
    for row in runs.itertuples(index=False):
        ledger = json.loads(row.nativeLedgerJson)
        checks = (
            row.cost_activations == ledger["activations"],
            row.cost_proposals == ledger["proposals"],
            row.cost_observationRecordReads == ledger["observationReads"],
            row.cost_valueComparisons == ledger["valueComparisons"],
            row.cost_proposals == row.cost_noOps + row.cost_rejections + row.cost_memoryUpdates + row.cost_acceptedSwaps + row.cost_conflictLosses,
            row.cost_displacedCells == 2 * row.cost_acceptedSwaps,
            row.cost_valueReads == row.cost_statusReads == row.cost_observationRecordReads,
            row.cost_targetCalculations == row.cost_policyCandidateConstructions == row.cost_proposals,
            row.cost_failedProposals == row.cost_rejections + row.cost_conflictLosses,
            row.cost_coordinatorMessages == 2 * row.cost_coordinatorEligibleDecisions,
            row.cost_coordinatorCandidateEvaluations == row.cost_coordinatorEligibleDecisions,
            row.cost_schedulerCandidateInspections == 0,
            row.projection_s01UnitWeightFullCost == sum(getattr(row, f"cost_{field}") for field in S01_ADDITIVE_FIELDS),
            row.projection_zeroCostControl == 0,
            row.completionOpportunity == row.activationCount == row.cost_activations,
        )
        ledger_identity_pass += all(checks)

    entries = adaptation["entries"]
    adaptation_compliance = (
        len(entries) == look
        and [entry["look"] for entry in entries] == list(range(1, look + 1))
        and [entry["cumulativePairingBlocks"] for entry in entries] == [50 * index for index in range(1, look + 1)]
        and all(entry["continue"] for entry in entries[:-1])
        and not entries[-1]["continue"]
        and all(entry["protectedOutcomeReads"] == 0 for entry in entries)
        and adaptation["frozenDesignSha256"] == freeze["designFreezeSha256"]
        and adaptation["allEntriesComply"]
    )
    treatment_counts = runs.groupby("treatmentSignature").size()
    stage_counts = runs.groupby("screeningStage").size()
    scale_counts = runs.groupby("n").size()
    estimand_counts = effects.groupby("estimandId").size()
    selected = set(selection["selectedEstimands"])

    checks = {
        "runRows": len(runs) == expected_runs,
        "uniqueRunIds": runs.runDesignId.nunique() == expected_runs,
        "runIdsMatchFrozenTerminalDesign": run_ids == set(terminal_design.runDesignId),
        "pairingBlocks": runs.pairingBlockId.nunique() == expected_blocks,
        "stageAccounting": set(stage_counts.index) == set(range(1, look + 1)) and set(stage_counts.values) == {700},
        "scaleAccounting": set(scale_counts.index) == {20, 50, 100, 200, 500} and set(scale_counts.values) == {expected_runs // 5},
        "treatmentMarginals": len(treatment_counts) == 14 and set(treatment_counts.values) == {expected_blocks},
        "pairingRows": len(terminal_pairing) == expected_blocks,
        "pairingScreeningOnly": set(terminal_pairing.split) == {"screening_pool"} and not terminal_pairing.protected.any(),
        "pairingOutcomeBlind": set(terminal_pairing.outcomeAccess) == {"none"} and not terminal_pairing.protectedOutcomeOpened.any(),
        "noSearchDerivedPlacements": not terminal_pairing.searchDerivedPlacementAssigned.any() and set(terminal_pairing.faultMapOutcomeAccess) == {"none"},
        "contrastMapAccounting": len(terminal_mapping) == expected_blocks * 30 and set(terminal_mapping.runDesignId).issubset(run_ids),
        "e07HasNoRunAssignments": "E02-S01-E07" not in set(terminal_mapping.estimandId),
        "pairedEffects": len(effects) == expected_effects and len(estimand_counts) == 11 and set(estimand_counts.values) == {expected_blocks},
        "e02ExactParity": parity["E02-S01-E02"]["success"] and parity["E02-S01-E02"]["pairingBlocks"] == expected_blocks,
        "e05Inactive": parity["E02-S01-E05"]["success"] and parity["E02-S01-E05"]["pairingBlocks"] == expected_blocks,
        "contractValidation": bool(runs.contractValidationPass.all()),
        "s09CostSchema": set(runs.costSchemaVersion) == {COST_SCHEMA_VERSION} and set(runs.costSchemaSha256) == {COST_SCHEMA_SHA256},
        "ledgerIdentities": ledger_identity_pass == expected_runs,
        "modelRowsAndRank": model["success"] and model["rows"] == expected_blocks * 9 and model["matrixRank"] == model["expectedRank"] == 81,
        "modelConvergence": model["allConverged"] and len(model["fits"]) == 12,
        "rankingCompleteness": len(ranking) == 9 and set(ranking.utilityRank) == set(range(1, 10)),
        "coefficientCompleteness": len(coefficients) == 4 * 3 * 81,
        "sensitivity": sensitivity["success"],
        "replayMinimum": replay["success"] and replay["samples"] >= 20 and replay["successfulSamples"] == replay["samples"],
        "adaptationLogCompliance": adaptation_compliance,
        "selectionEligible": not selected.intersection({"E02-S01-E05", "E02-S01-E07"}) and len(selected) <= 6,
        "selectionParityGate": selection["exactParityGatePass"] and selection["frozen"],
        "protectedOutcomeReadsZero": selection["protectedOutcomeReads"] == 0 and not selection["confirmatoryOutcomesOpened"] and replay["protectedOutcomeReads"] == 0,
    }
    success = all(checks.values())
    by_stage = [
        {
            "screeningStage": int(stage),
            "runRows": int(count),
            "pairingBlocks": int(runs[runs.screeningStage == stage].pairingBlockId.nunique()),
            "contractPass": int(runs[runs.screeningStage == stage].contractValidationPass.sum()),
        }
        for stage, count in stage_counts.items()
    ]
    summary = {
        "schemaVersion": "e02.s10.validation_summary.v1",
        "researchStepId": "S10",
        "validationResult": "PASS" if success else "FAIL",
        "success": success,
        "terminalLook": look,
        "runRows": len(runs),
        "pairingBlocks": expected_blocks,
        "pairedEffectRows": len(effects),
        "treatmentSignatures": len(treatment_counts),
        "ledgerIdentityRowsPassed": ledger_identity_pass,
        "exactReplaySamplesPassed": replay["successfulSamples"],
        "stopReasonCounts": dict(sorted(Counter(runs.stopReason).items())),
        "checks": checks,
        "failedChecks": [name for name, passed in checks.items() if not passed],
        "byStage": by_stage,
        "protectedOutcomeReads": 0,
        "searchDerivedAssignments": 0,
    }
    write_json(args.output / "validation_summary.json", summary)
    write_json(args.output / "run_accounting_validation.json", {
        "schemaVersion": "e02.s10.run_accounting_validation.v1",
        "researchStepId": "S10",
        "expectedRuns": expected_runs,
        "observedRuns": len(runs),
        "byStage": by_stage,
        "scaleCounts": {str(key): int(value) for key, value in scale_counts.items()},
        "treatmentCounts": {str(key): int(value) for key, value in treatment_counts.items()},
        "estimandPairCounts": {str(key): int(value) for key, value in estimand_counts.items()},
        "success": all(checks[name] for name in ("runRows", "uniqueRunIds", "runIdsMatchFrozenTerminalDesign", "stageAccounting", "scaleAccounting", "treatmentMarginals", "pairedEffects")),
    })
    print(json.dumps(summary, sort_keys=True))
    if not success:
        raise AssertionError(f"S10 validation failed: {summary['failedChecks']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/artifacts/research_steps/S10"))
    validate(parser.parse_args())


if __name__ == "__main__":
    main()
