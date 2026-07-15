#!/usr/bin/env python3
"""Freeze and validate S12's analysis contract before any S12 model fit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import pyarrow.parquet as pq


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = REPOSITORY / "design/s12/modeling_prespecification.json"
DEFAULT_OUTPUT = Path("/artifacts/research_steps/S12")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def freeze(args: argparse.Namespace) -> None:
    spec = json.loads(args.spec.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    if spec["researchStepId"] != "S12" or not spec["frozenBeforeOutcomeModeling"]:
        raise AssertionError("invalid S12 pre-outcome-modeling declaration")
    if spec["outcomeModelFitsBeforeFreeze"] != 0:
        raise AssertionError("S12 model fits occurred before freeze")

    input_hashes: dict[str, str] = {}
    input_rows: dict[str, int] = {}
    checks: dict[str, bool] = {}
    for name, contract in spec["inputContracts"].items():
        path = Path(contract["path"])
        digest = sha256_file(path)
        input_hashes[name] = digest
        checks[f"inputHash:{name}"] = digest == contract["sha256"]
        if "requiredRows" in contract:
            rows = pq.read_metadata(path).num_rows
            input_rows[name] = rows
            checks[f"inputRows:{name}"] = rows == contract["requiredRows"]

    preserved = spec["contractPreservation"]
    checks.update({
        "fourFrozenEstimands": preserved["selectedEstimandsInOrder"] == [
            "E02-S01-E04", "E02-S01-E06", "E02-S01-E03", "E02-S01-E08"
        ],
        "terminalPrefixOnly": preserved["permittedConfirmatoryLooks"] == [1, 2]
        and preserved["terminalS11PairingBlocks"] == 2000,
        "rngUnpairedPreserved": preserved["couplingClassifications"]["E02-S01-E03"].startswith("scenario_paired_rng_unpaired")
        and preserved["couplingClassifications"]["E02-S01-E06"].startswith("scenario_paired_rng_unpaired"),
        "s09Preserved": preserved["costSchemaVersion"] == "E02.complete-cost-ledger.v1",
        "noContractChange": not preserved["legalActionInformationSchedulerFaultAndOpportunityContractsMayChange"],
        "noSearchAssignments": preserved["searchDerivedAssignmentsMaximum"] == 0,
        "strongHeredity": bool(spec["modelFormulas"]["strongHeredity"]),
        "noOutcomeSelection": any("post-outcome" in item for item in spec["forbiddenActions"]),
        "phAlternativeFrozen": spec["diagnostics"]["survival"]["alternativeIntervalsFrozenBeforeModeling"],
        "bootstrapReplicates": spec["bootstrap"]["replicates"] == 2000,
        "noExtrapolation": bool(spec["analysisPopulations"]["heterogeneity"]["unsupportedRule"]),
    })
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise AssertionError(f"S12 freeze validation failed: {failed}")

    target = args.output / "modeling_prespecification.json"
    shutil.copyfile(args.spec, target)
    spec_sha = sha256_file(target)
    freeze_payload = {
        "schemaVersion": "e02.s12.model_freeze_manifest.v1",
        "researchStepId": "S12",
        "frozenAtUtc": spec["frozenAtUtc"],
        "frozenBeforeOutcomeModeling": True,
        "outcomeModelFitsBeforeFreeze": 0,
        "protectedOutcomeModelingAuthorizedOnlyAfterThisFreeze": True,
        "modelingPrespecificationSha256": spec_sha,
        "inputHashes": input_hashes,
        "inputRows": input_rows,
        "checks": checks,
        "success": True,
    }
    freeze_sha = hashlib.sha256(canonical_bytes(freeze_payload)).hexdigest()
    freeze_payload["modelFreezeSha256"] = freeze_sha
    (args.output / "model_freeze_manifest.json").write_bytes(
        canonical_bytes(freeze_payload) + b"\n"
    )
    print(json.dumps({
        "modelFreezeSha256": freeze_sha,
        "modelingPrespecificationSha256": spec_sha,
        "checks": len(checks),
        "success": True,
    }, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    freeze(parser.parse_args())


if __name__ == "__main__":
    main()
