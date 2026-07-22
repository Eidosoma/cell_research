#!/usr/bin/env python3
"""Run the mandatory S08 pre-smoke gate; never cross a failed gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any

from src.portfolio_search import run_preflight
from src.portfolio_search.preflight import CONTROL_PATH, S08, sha256_file


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_outputs(result: dict[str, Any]) -> None:
    S08.mkdir(parents=True, exist_ok=True)
    write_json(S08 / "preflight_hash_revalidation.json", result["hashValidation"])
    write_json(
        S08 / "configuration_binding_validation.json", result["bindingValidation"]
    )
    write_json(S08 / "access_control_validation.json", result["accessValidation"])
    write_json(S08 / "dependency_exclusion_audit.json", result["dependencyValidation"])
    write_json(S08 / "budget_accounting.json", result["accounting"])
    write_json(S08 / "no_mutation_audit.json", result["noMutation"])
    write_json(
        S08 / "gate_revalidation.json",
        {
            "schemaVersion": "e07.s08.gate-revalidation.v1",
            "researchStepId": "S08",
            "success": result["success"],
            "status": result["status"],
            "blockedGateIds": result["blockedGateIds"],
            "rows": result["gateRows"],
            "historicalS08AGatePass": result["historicalS08AGatePass"],
            "smokeAuthorized": result["success"],
            "substantiveSearchAuthorized": False,
        },
    )
    write_json(
        S08 / "smoke_gate_status.json",
        {
            "schemaVersion": "e07.s08.smoke-gate-status.v1",
            "researchStepId": "S08",
            **result["frozenSmoke"],
            "reason": "G04 executable-binding revalidation failed before scenario materialization",
        },
    )
    write_json(
        S08 / "validation_summary.json",
        {
            "schemaVersion": "e07.s08.validation-summary.v1",
            "researchStepId": "S08",
            "success": result["success"],
            "outcomeClassification": "constraining/contradictory",
            "checks": {
                "frozenHashes": result["hashValidation"]["success"],
                "historicalS08AGates": result["historicalS08AGatePass"],
                "staticCompositionLegality": result["legalityValidation"][
                    "illegalCompositions"
                ]
                == 0,
                "executableBindings": result["bindingValidation"]["success"],
                "protectedDenial": result["accessValidation"]["success"],
                "dependencyExclusion": result["dependencyValidation"]["success"],
                "budgetAccounting": result["accounting"]["success"],
                "noMutation": result["noMutation"]["success"],
                "smokePassed": result["frozenSmoke"]["pass"],
            },
            "configurationRowsChecked": result["bindingValidation"][
                "configurationRowsChecked"
            ],
            "configurationRowsBlocked": result["bindingValidation"][
                "configurationRowsBlocked"
            ],
            "smokeConfigurationsBlocked": result["bindingValidation"][
                "frozenSmokeConfigurationsBlocked"
            ],
            "episodeEvaluations": 0,
        },
    )
    write_json(
        S08 / "status.json",
        {
            "researchStepId": "S08",
            "stepNumber": "08",
            "success": False,
            "status": "blocked_before_smoke",
            "artifactsWritten": [
                "access_control_validation.json",
                "budget_accounting.json",
                "configuration_binding_validation.json",
                "dependency_exclusion_audit.json",
                "environment.json",
                "execution_commands.log",
                "gate_revalidation.json",
                "input_provenance.json",
                "no_mutation_audit.json",
                "preflight_hash_revalidation.json",
                "provenance.json",
                "research_step_full_results.md",
                "smoke_gate_status.json",
                "status.json",
                "test_validation.json",
                "validation_summary.json",
                "artifact_manifest.json",
            ],
            "validationResult": "FAIL CLOSED: G04; 56/1,216 frozen configurations and 3/40 smoke configurations cannot bind",
            "caveatsOrBlockers": [
                "The S08A dispatcher requires at least two members per native carrier, but 56 frozen Chimera portfolios contain a legal one-member carrier stratum.",
                "The frozen smoke, search, validation, and all archive mutations were not started.",
            ],
            "recommendedNextAction": result["recommendedNextAction"],
        },
    )
    write_json(
        S08 / "provenance.json",
        {
            "schemaVersion": "e07.s08.provenance.v1",
            "researchStepId": "S08",
            "executionControlPath": str(CONTROL_PATH),
            "executionControlSha256": sha256_file(CONTROL_PATH),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "repositoryBranch": "eidosoma/groups/28",
            "preflightOnly": True,
            "smokeExecuted": False,
            "searchExecuted": False,
        },
    )
    write_json(
        S08 / "environment.json",
        {
            "schemaVersion": "e07.s08.environment.v1",
            "researchStepId": "S08",
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpuCountVisible": os.cpu_count(),
            "evaluationWorkersUsed": 0,
            "numericThreadsPerWorker": 1,
            "gpuUsed": False,
            "networkUsed": False,
            "newDependenciesInstalled": [],
        },
    )
    write_json(
        S08 / "input_provenance.json",
        {
            "schemaVersion": "e07.s08.input-provenance.v1",
            "researchStepId": "S08",
            "frozenFileChecks": result["hashValidation"]["fileChecks"],
            "frozenInputChecks": result["hashValidation"]["frozenInputChecks"],
            "trees": result["hashValidation"]["treesBefore"],
        },
    )


def write_manifest() -> None:
    files = []
    for path in sorted(item for item in S08.iterdir() if item.is_file()):
        if path.name == "artifact_manifest.json":
            continue
        files.append(
            {
                "path": path.name,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    write_json(
        S08 / "artifact_manifest.json",
        {
            "schemaVersion": "e07.s08.artifact-manifest.v1",
            "researchStepId": "S08",
            "artifactCount": len(files),
            "artifacts": files,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "manifest"))
    args = parser.parse_args()
    if args.command == "manifest":
        write_manifest()
        return 0
    result = run_preflight()
    write_outputs(result)
    print(
        json.dumps(
            {key: result[key] for key in ("success", "status", "blockedGateIds")},
            sort_keys=True,
        )
    )
    return 0 if result["success"] else 2


if __name__ == "__main__":
    sys.exit(main())
