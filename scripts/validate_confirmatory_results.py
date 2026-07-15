#!/usr/bin/env python3
"""Consolidate S11's frozen-design, execution, inference, and replay gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from reference_simulator.model import canonical_json_bytes


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(args: argparse.Namespace) -> None:
    freeze = json.loads((args.output / "confirmatory_freeze_manifest.json").read_text())
    design_validation = json.loads((args.output / "design_validation.json").read_text())
    named = {
        "holdoutIntegrity": "holdout_integrity_validation.json",
        "pairRunCompleteness": "pair_run_completeness_validation.json",
        "treatmentMarginals": "treatment_marginal_validation.json",
        "stoppingRule": "stopping_rule_validation.json",
        "ledgerIdentities": "ledger_identity_validation.json",
        "deterministicReplay": "deterministic_replay_validation.json",
    }
    records = {key: json.loads((args.output / path).read_text()) for key, path in named.items()}
    runs = pd.read_parquet(args.output / "confirmatory_results.parquet")
    primary = pd.read_parquet(args.output / "primary_effects.parquet")
    secondary = pd.read_parquet(args.output / "secondary_effects.parquet")
    sequential = json.loads((args.output / "sequential_sampling_log.json").read_text())
    spec_hash = sha256_file(args.output / "confirmatory_prespecification.json")
    checks = {
        "designFreezeBeforeProtectedAccess": bool(
            freeze["frozenBeforeProtectedOutcomeAccess"]
            and freeze["protectedOutcomeReadsBeforeFreeze"] == 0
            and freeze["protectedOutcomeAccessAuthorizedOnlyAfterThisFreeze"]
        ),
        "designFreezeHashBound": freeze["designFreezeSha256"] == sequential["decisions"][0]["designFreezeSha256"],
        "prespecificationHashUnchanged": spec_hash == freeze["hashes"]["confirmatoryPrespecificationSha256"],
        "designValidation": design_validation["success"],
        **{key: record["success"] for key, record in records.items()},
        "minimumOneThousandPairs": records["pairRunCompleteness"]["expectedPairsPerContrast"] >= 1000,
        "fourFrozenPrimaryRows": len(primary) == 4 and set(primary.estimandId) == {
            "E02-S01-E04", "E02-S01-E06", "E02-S01-E03", "E02-S01-E08"
        },
        "sixteenSecondaryRows": len(secondary) == 16,
        "allRunContractsPass": bool(runs.contractValidationPass.all()),
        "zeroAdaptationOutsideRule": sequential["zeroAdaptationOutsideFrozenRule"],
        "terminalPrecisionOrBoundedInconclusive": bool(
            primary.precisionPass.all()
            or sequential["terminalPairingBlocksPerContrast"] == 5000
        ),
        "e08Retained": "E02-S01-E08" in set(primary.estimandId),
    }
    summary: dict[str, Any] = {
        "schemaVersion": "e02.s11.validation_summary.v1",
        "researchStepId": "S11",
        "stepNumber": 11,
        "checks": checks,
        "terminalLook": sequential["terminalLook"],
        "pairsPerSelectedContrast": sequential["terminalPairingBlocksPerContrast"],
        "runRows": len(runs),
        "primaryClassifications": {
            row.estimandId: row.classificationAtLook for row in primary.itertuples(index=False)
        },
        "success": all(checks.values()),
    }
    args.path.write_bytes(canonical_json_bytes(summary) + b"\n")
    print(json.dumps({
        "success": summary["success"], "checks": len(checks),
        "pairsPerContrast": summary["pairsPerSelectedContrast"],
    }, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/artifacts/research_steps/S11"))
    parser.add_argument("--path", type=Path, default=Path("/artifacts/research_steps/S11/validation_summary.json"))
    validate(parser.parse_args())


if __name__ == "__main__":
    main()
