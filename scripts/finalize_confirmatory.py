#!/usr/bin/env python3
"""Promote the terminal frozen S11 look and write compact validation evidence."""

from __future__ import annotations

import argparse
from collections import Counter
import json
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

from reference_simulator.model import canonical_json_bytes


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    pq.write_table(
        pa.Table.from_pandas(frame, preserve_index=False), path,
        compression="zstd", compression_level=9, use_dictionary=True,
        write_statistics=True, version="2.6",
    )


def finalize(args: argparse.Namespace) -> None:
    terminal_dir = args.output / f"look_{args.look:02d}"
    terminal = json.loads((terminal_dir / "adaptation_decision.json").read_text())
    if terminal["continue"]:
        raise AssertionError("cannot finalize a look whose frozen rule says continue")
    if args.look < 5 and terminal["action"] != "stop_for_frozen_primary_precision":
        raise AssertionError("an early terminal look must pass the frozen precision rule")

    paths = [args.cache / f"stage_{stage:02d}_confirmatory_runs.parquet" for stage in range(1, args.look + 1)]
    runs = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    runs = runs.sort_values("runDesignId").reset_index(drop=True)
    expected_pairs = args.look * 1000
    expected_runs = expected_pairs * 6
    if len(runs) != expected_runs or runs.runDesignId.nunique() != expected_runs:
        raise AssertionError("terminal confirmatory run accounting mismatch")
    write_parquet(args.output / "confirmatory_results.parquet", runs)
    for name in ("paired_effects.parquet", "primary_effects.parquet", "secondary_effects.parquet"):
        shutil.copy2(terminal_dir / name, args.output / name)

    primary = pd.read_parquet(terminal_dir / "primary_effects.parquet")
    secondary = pd.read_parquet(terminal_dir / "secondary_effects.parquet")
    primary.to_csv(args.output / "primary_effects.csv", index=False)
    secondary.to_csv(args.output / "secondary_effects.csv", index=False)

    decisions = [
        json.loads((args.output / f"look_{look:02d}" / "adaptation_decision.json").read_text())
        for look in range(1, args.look + 1)
    ]
    sequential = {
        "schemaVersion": "e02.s11.sequential_sampling_log.v1",
        "researchStepId": "S11", "terminalLook": args.look,
        "terminalPairingBlocksPerContrast": expected_pairs,
        "decisions": decisions,
        "zeroAdaptationOutsideFrozenRule": all(item["ruleCompliant"] for item in decisions),
        "terminalAction": terminal["action"],
    }
    write_json(args.output / "sequential_sampling_log.json", sequential)

    cost_columns = [
        "runDesignId", "pairingBlockId", "n", "confirmatoryLook",
        "treatmentSignature", "architecture", "coordinatorProfile", "scheduler",
        "mobility", "continuation", "retry", "nativeLedgerJson",
        "costSchemaVersion", "costSchemaSha256", "contractValidationPass",
        *sorted(column for column in runs if column.startswith(("cost_", "projection_", "normalization_"))),
    ]
    write_parquet(args.output / "cost_ledger.parquet", runs[cost_columns])

    selected_pairing = pd.read_parquet(args.pairing)
    selected_pairing = selected_pairing[selected_pairing.confirmatoryLook <= args.look]
    design = pd.read_parquet(args.design)
    design = design[design.confirmatoryLook <= args.look]
    contrast = pd.read_parquet(args.contrast_map)
    contrast = contrast[contrast.confirmatoryLook <= args.look]

    expected_ids = set(design.runDesignId)
    integrity_checks = {
        "protectedHoldoutOnly": bool((selected_pairing.split == "confirmatory_holdout").all() and selected_pairing.protected.all()),
        "priorProtectedOutcomeFlagsFalse": bool((~selected_pairing.protectedOutcomeOpened).all()),
        "confirmatoryEligibleMapsOnly": bool(selected_pairing.faultMapConfirmatoryEligible.all()),
        "noSearchDerivedMaps": bool((~selected_pairing.searchDerivedPlacementAssigned).all()),
        "actualRunIdsExactlyFrozen": set(runs.runDesignId) == expected_ids,
        "selectedBlockIdsExactlyFrozenPrefix": set(runs.pairingBlockId) == set(selected_pairing.pairingBlockId),
        "protectedResultRows": bool(runs.protected.all() and (runs.split == "confirmatory_holdout").all()),
    }
    write_json(args.output / "holdout_integrity_validation.json", {
        "schemaVersion": "e02.s11.holdout_integrity_validation.v1",
        "researchStepId": "S11", "checks": integrity_checks,
        "pairingBlocks": len(selected_pairing), "runRows": len(runs),
        "success": all(integrity_checks.values()),
    })

    run_counts = Counter(runs.treatmentSignature)
    arm_counts = Counter(zip(contrast.estimandId, contrast.contrastRole))
    completeness_checks = {
        "runRows": len(runs) == expected_runs,
        "uniqueRunIds": runs.runDesignId.nunique() == expected_runs,
        "pairingBlocks": runs.pairingBlockId.nunique() == expected_pairs,
        "sixSignaturesPerBlock": len(run_counts) == 6 and set(run_counts.values()) == {expected_pairs},
        "fourContrastsEightArms": len(arm_counts) == 8 and set(arm_counts.values()) == {expected_pairs},
        "pairedEffectRows": len(pd.read_parquet(terminal_dir / "paired_effects.parquet")) == expected_pairs * 4,
        "noMissingRunStatus": not runs[[
            "runDesignId", "pairingBlockId", "stopReason", "contractValidationPass"
        ]].isna().any(axis=1).any(),
    }
    write_json(args.output / "pair_run_completeness_validation.json", {
        "schemaVersion": "e02.s11.pair_run_completeness_validation.v1",
        "researchStepId": "S11", "checks": completeness_checks,
        "expectedPairsPerContrast": expected_pairs, "actualPairsPerContrast": {
            estimand: int(count) for (estimand, role), count in sorted(arm_counts.items()) if role == "active"
        },
        "expectedRuns": expected_runs, "actualRuns": len(runs),
        "success": all(completeness_checks.values()),
    })

    marginal_rows = []
    for (estimand, role), part in contrast.groupby(["estimandId", "contrastRole"], sort=True):
        linked = part.merge(
            runs[["runDesignId", "n", "valueProfile", "orderStructure", "policyProfile", "direction", "faultProfileId"]],
            on="runDesignId", validate="many_to_one",
        )
        marginal_rows.append({
            "estimandId": estimand, "contrastRole": role, "pairCount": len(linked),
            "nCountsJson": canonical_json_bytes(linked.n.value_counts().sort_index().to_dict()).decode(),
            "valueProfileCountsJson": canonical_json_bytes(linked.valueProfile.value_counts().sort_index().to_dict()).decode(),
            "orderStructureCountsJson": canonical_json_bytes(linked.orderStructure.value_counts().sort_index().to_dict()).decode(),
            "policyCountsJson": canonical_json_bytes(linked.policyProfile.value_counts().sort_index().to_dict()).decode(),
            "directionCountsJson": canonical_json_bytes(linked.direction.value_counts().sort_index().to_dict()).decode(),
            "faultProfileCountsJson": canonical_json_bytes(linked.faultProfileId.value_counts().sort_index().to_dict()).decode(),
        })
    marginals = pd.DataFrame(marginal_rows)
    write_parquet(args.output / "treatment_marginals.parquet", marginals)
    marginal_checks = {
        "eightArmRows": len(marginals) == 8,
        "equalPairCounts": set(marginals.pairCount) == {expected_pairs},
        "withinEstimandMarginalsExact": all(
            len({getattr(row, column) for row in group.itertuples(index=False)}) == 1
            for _, group in marginals.groupby("estimandId")
            for column in [
                "nCountsJson", "valueProfileCountsJson", "orderStructureCountsJson",
                "policyCountsJson", "directionCountsJson", "faultProfileCountsJson",
            ]
        ),
    }
    write_json(args.output / "treatment_marginal_validation.json", {
        "schemaVersion": "e02.s11.treatment_marginal_validation.v1",
        "researchStepId": "S11", "checks": marginal_checks,
        "success": all(marginal_checks.values()),
    })

    contract_json = [json.loads(item) for item in runs.contractValidationJson]
    ledger_checks = {
        "allContractValidationsPass": bool(runs.contractValidationPass.all()),
        "allFieldLevelLedgerChecksPass": all(all(item.values()) for item in contract_json),
        "oneProposalPerActivation": bool((runs.cost_activations == runs.cost_proposals).all()),
        "proposalPartition": bool((
            runs.cost_proposals == runs.cost_noOps + runs.cost_rejections + runs.cost_memoryUpdates
            + runs.cost_acceptedSwaps + runs.cost_conflictLosses
        ).all()),
        "displacementIdentity": bool((runs.cost_displacedCells == 2 * runs.cost_acceptedSwaps).all()),
        "s01ProjectionIdentity": bool((
            runs.projection_s01UnitWeightFullCost
            == runs.cost_activations + runs.cost_observationRecordReads
            + runs.cost_valueComparisons + runs.cost_targetCalculations + runs.cost_proposals
            + runs.cost_noOps + runs.cost_rejections + runs.cost_memoryUpdates
            + runs.cost_acceptedSwaps + runs.cost_displacedCells + runs.cost_conflictLosses
            + runs.cost_coordinatorMessages
        ).all()),
        "costSchemaSingleVersion": runs.costSchemaVersion.nunique() == 1 and runs.costSchemaSha256.nunique() == 1,
    }
    write_json(args.output / "ledger_identity_validation.json", {
        "schemaVersion": "e02.s11.ledger_identity_validation.v1",
        "researchStepId": "S11", "checks": ledger_checks,
        "runRows": len(runs), "success": all(ledger_checks.values()),
    })

    stopping_checks = {
        "minimumSampleMet": expected_pairs >= 1000,
        "maximumNotExceeded": expected_pairs <= 5000,
        "nestedLooksOnly": [item["cumulativePairingBlocksPerContrast"] for item in decisions] == [1000 * i for i in range(1, args.look + 1)],
        "priorLooksRequiredContinuation": all(item["continue"] for item in decisions[:-1]),
        "terminalStopped": not decisions[-1]["continue"],
        "terminalActionMatchesPrecision": (
            decisions[-1]["primaryPrecisionPass"] and decisions[-1]["action"] == "stop_for_frozen_primary_precision"
        ) or (
            args.look == 5 and not decisions[-1]["primaryPrecisionPass"]
            and decisions[-1]["action"] == "stop_at_frozen_maximum_inconclusive_precision"
        ),
        "noOutcomeDirectionAdaptation": all(not item["observedDirectionUsedForAdaptation"] for item in decisions),
        "noSignificanceAdaptation": all(not item["significanceUsedForAdaptation"] for item in decisions),
        "noSecondaryAdaptation": all(not item["secondaryEndpointUsedForAdaptation"] for item in decisions),
        "noRuntimeAdaptation": all(not item["runtimeUsedForAdaptation"] for item in decisions),
        "noArmOrContrastDropping": all(not item["contrastOrArmDropped"] for item in decisions),
    }
    write_json(args.output / "stopping_rule_validation.json", {
        "schemaVersion": "e02.s11.stopping_rule_validation.v1",
        "researchStepId": "S11", "checks": stopping_checks,
        "terminalLook": args.look, "terminalPairsPerContrast": expected_pairs,
        "success": all(stopping_checks.values()),
    })
    print(json.dumps({
        "terminalLook": args.look, "pairsPerContrast": expected_pairs,
        "runs": len(runs), "classifications": dict(zip(primary.estimandId, primary.classificationAtLook)),
    }, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--look", type=int, required=True, choices=range(1, 6))
    parser.add_argument("--cache", type=Path, default=Path("/cache/s11"))
    parser.add_argument("--output", type=Path, default=Path("/artifacts/research_steps/S11"))
    parser.add_argument("--design", type=Path, default=Path("/artifacts/research_steps/S11/confirmatory_design.parquet"))
    parser.add_argument("--pairing", type=Path, default=Path("/artifacts/research_steps/S11/confirmatory_pairing_blocks.parquet"))
    parser.add_argument("--contrast-map", type=Path, default=Path("/artifacts/research_steps/S11/confirmatory_contrast_map.parquet"))
    finalize(parser.parse_args())


if __name__ == "__main__":
    main()
