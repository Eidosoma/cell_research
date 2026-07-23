#!/usr/bin/env python3
"""Independent compact validation of promoted S09 artifacts."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.testing import assert_frame_equal

from src.environment_suite import portfolio_action
from src.environment_suite.contracts import canonical_sha256
from src.policy_dsl import compile_policy
from src.portfolio_search.preflight import sha256_file


ROOT = Path("/artifacts/research_steps/S09")


def rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    bundles = rows(ROOT / "compressed_policies.jsonl")
    registry = rows(ROOT / "edit_registry.jsonl")
    certificates = rows(ROOT / "minimality_certificates.jsonl")
    eligible = sorted(
        json.loads((ROOT / "input_validation.json").read_text())[
            "eligibleConfigurationIds"
        ]
    )
    bundle_checks = []
    for bundle in bundles:
        config = bundle["configuration"]
        documents = bundle["documentsByPolicySha256"]
        member_hash_pass = True
        for member in config["members"]:
            compiled = compile_policy(documents[member["policySha256"]])
            member_hash_pass &= (
                compiled.policy_sha256 == member["policySha256"]
            )
        definition = deepcopy(config)
        observed_id = definition.pop("configurationId")
        expected_id = canonical_sha256(
            "E07/S09/runtime-configuration/v1", definition
        )
        action = portfolio_action(
            [
                documents[member["policySha256"]]
                for member in config["members"]
            ],
            config,
        )
        bundle_checks.append(
            {
                "parentConfigurationId": bundle[
                    "parentConfigurationId"
                ],
                "variantConfigurationId": bundle[
                    "variantConfigurationId"
                ],
                "memberHashPass": bool(member_hash_pass),
                "configurationHashPass": observed_id == expected_id,
                "portfolioActionSha256": action.policy_sha256,
                "portfolioSize": len(config["members"]),
            }
        )
    edit_checks = []
    for edit in registry:
        config = edit["configuration"]
        documents = edit["documentsByPolicySha256"]
        action = portfolio_action(
            [
                documents[member["policySha256"]]
                for member in config["members"]
            ],
            config,
        )
        edit_checks.append(
            action.policy_sha256 == edit["variantActionSha256"]
        )
    csv_frame = pd.read_csv(ROOT / "ablation_effects.csv")
    parquet_frame = pd.read_parquet(ROOT / "ablation_effects.parquet")
    csv_compare = csv_frame.astype(object).where(csv_frame.notna(), None)
    parquet_compare = parquet_frame.astype(object).where(
        parquet_frame.notna(), None
    )
    try:
        assert_frame_equal(
            csv_compare,
            parquet_compare,
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError:
        csv_parquet_pass = False
    else:
        csv_parquet_pass = True
    result = {
        "schemaVersion": "e07.s09.configuration-hash-validation.v1",
        "researchStepId": "S09",
        "success": (
            len(bundles) == 7
            and sorted(
                row["parentConfigurationId"] for row in bundles
            )
            == eligible
            and len(certificates) == 7
            and len(registry) == 125
            and all(
                row["memberHashPass"]
                and row["configurationHashPass"]
                for row in bundle_checks
            )
            and all(edit_checks)
            and csv_parquet_pass
        ),
        "eligibleParentCount": len(eligible),
        "compressedBundleCount": len(bundles),
        "certificateCount": len(certificates),
        "firstOrderEditCount": len(registry),
        "allFirstOrderActionHashesPass": all(edit_checks),
        "csvParquetParityPass": csv_parquet_pass,
        "ablationEffectRows": len(csv_frame),
        "bundleChecks": bundle_checks,
        "candidateLockSha256": json.loads(
            (ROOT / "input_validation.json").read_text()
        )["candidateLockSha256"],
        "protocolSha256": sha256_file(
            ROOT / "s09_ablation_protocol.yaml"
        ),
    }
    if not result["success"]:
        raise RuntimeError(json.dumps(result, sort_keys=True))
    (ROOT / "configuration_hash_validation.json").write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n",
        encoding="ascii",
    )
    print(json.dumps({"success": True, "checks": len(bundle_checks)}))


if __name__ == "__main__":
    main()
