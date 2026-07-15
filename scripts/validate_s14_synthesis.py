#!/usr/bin/env python3
"""Audit frozen S14 artifacts, release provenance, and claim boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import pandas as pd


SCHEMA = "e02.s14.validation.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/artifacts/research_steps/S14"))
    parser.add_argument("--report-inputs", type=Path, default=Path("/artifacts/report_inputs"))
    parser.add_argument("--release", type=Path, default=Path("/artifacts/release/causal_simulator_extension"))
    parser.add_argument("--require-report", action="store_true")
    args = parser.parse_args()

    required = [
        args.output / "synthesis_freeze_manifest.json",
        args.output / "mechanism_effects.parquet",
        args.output / "mechanism_heterogeneity.parquet",
        args.output / "pareto_summary.parquet",
        args.output / "pareto_dominance.parquet",
        args.output / "variance_decomposition.parquet",
        args.output / "variance_decomposition_diagnostics.json",
        args.output / "s13_sensitivity.parquet",
        args.output / "claim_to_evidence_matrix.csv",
        args.output / "caveat_register.csv",
        args.output / "fresh_smoke_validation.json",
        args.output / "validation_summary.json",
        args.output / "input_provenance.json",
        args.output / "environment_provenance.json",
        args.release / "release_manifest.json",
        Path("/artifacts/release/causal-simulator-extension.tar.zst.NOT_CREATED.json"),
        args.report_inputs / "evidence_index.csv",
        args.report_inputs / "methods_summary.md",
        args.report_inputs / "figure_index.csv",
        args.report_inputs / "table_index.csv",
        args.report_inputs / "claim_to_evidence_matrix.csv",
        args.report_inputs / "caveat_register.csv",
        args.report_inputs / "provenance_manifest.json",
        args.report_inputs / "lay_summary.md",
        args.report_inputs / "report_bundle_manifest.json",
    ]
    if args.require_report:
        required.extend([args.output / "research_step_full_results.md", args.output / "status.json"])
    missing = [str(path) for path in required if not path.exists()]

    mechanisms = pd.read_parquet(args.output / "mechanism_effects.parquet")
    heterogeneity = pd.read_parquet(args.output / "mechanism_heterogeneity.parquet")
    pareto = pd.read_parquet(args.output / "pareto_summary.parquet")
    variance = pd.read_parquet(args.output / "variance_decomposition.parquet")
    sensitivity = pd.read_parquet(args.output / "s13_sensitivity.parquet")
    claims = pd.read_csv(args.output / "claim_to_evidence_matrix.csv")
    release = json.loads((args.release / "release_manifest.json").read_text())
    smoke = json.loads((args.output / "fresh_smoke_validation.json").read_text())
    provenance = json.loads((args.output / "input_provenance.json").read_text())
    validation = json.loads((args.output / "validation_summary.json").read_text())

    current_head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    source_hashes = all(
        Path(item["path"]).exists() and sha256_file(Path(item["path"])) == item["sha256"]
        for item in release["sourceFiles"]
    )
    input_hashes = all(
        Path(item["path"]).exists() and sha256_file(Path(item["path"])) == item["sha256"]
        for item in provenance["inputs"]
    )
    causal_classes = claims[claims.claimClass == "supported_causal_within_simulated_support"]
    causal_scope_valid = set(causal_classes.claimId).issubset({"C02", "C03"})
    bounded_null = claims.loc[claims.claimId == "C08", "caveat"].str.contains(
        "no global-absence", case=False, regex=False
    ).all()
    brittle = claims.loc[claims.claimId == "C09", "claimClass"].eq("constraining_brittle").all()
    descriptive = claims.loc[claims.claimId.isin(["C06", "C07"]), "claimClass"].eq("descriptive_only").all()
    e04_tradeoff = claims.loc[claims.claimId == "C02", "claim"].str.contains("resource", case=False).all()
    rng_caveats = all(
        claims.loc[claims.claimId == claim_id, "caveat"].str.contains("RNG-unpaired", case=False).all()
        for claim_id in ("C03", "C04")
    )
    claim_checks = {
        "causalLanguageRestrictedToRandomizedContrasts": bool(causal_scope_valid),
        "S13BoundedNullExplicit": bool(bounded_null),
        "brittleE06NotPromoted": bool(brittle),
        "ParetoAndVarianceDescriptive": bool(descriptive),
        "E04ResourceTradeoffExplicit": bool(e04_tradeoff),
        "E03E06RngUnpairedExplicit": bool(rng_caveats),
        "noUniversalRankingClaim": not claims.claim.str.contains("always best|universally best", case=False, regex=True).any(),
    }
    write_json(
        args.output / "claim_boundary_review.json",
        {
            "schemaVersion": SCHEMA,
            "researchStepId": "S14",
            "status": "PASS" if all(claim_checks.values()) else "FAIL",
            "checks": claim_checks,
        },
    )

    provenance_checks = {
        "inputHashesMatch": input_hashes,
        "releaseSourceHashesMatch": source_hashes,
        "releaseCommitMatchesCurrentHead": release["commit"] == current_head,
        "freshSmokePass": bool(smoke["success"]),
        "freshSmokeReplayPass": bool(smoke["deterministicReplay"]),
        "analysisReplayPass": bool(validation["checks"]["deterministicAnalysisReplay"]),
        "prespecificationHashUnchanged": bool(validation["checks"]["prespecificationHashUnchanged"]),
    }
    write_json(
        args.output / "replay_provenance_validation.json",
        {
            "schemaVersion": SCHEMA,
            "researchStepId": "S14",
            "status": "PASS" if all(provenance_checks.values()) else "FAIL",
            "checks": provenance_checks,
        },
    )

    content_checks = {
        "requiredFilesPresent": not missing,
        "marginalMechanismRows": len(mechanisms) == 16,
        "heterogeneityRows": len(heterogeneity) == 368,
        "paretoRows": len(pareto) == 18,
        "paretoCostProfiles": pareto.costProfile.nunique() == 3,
        "varianceRows": len(variance) == 24,
        "varianceIdentity": bool((variance.groupby("outcome").share.sum().sub(1).abs() < 1e-10).all()),
        "S13AllCandidates": len(sensitivity) == 8 and sensitivity.candidateId.nunique() == 8,
        "S13NoFullConfirmations": int(sensitivity.fullyConfirmed.sum()) == 0,
        "S13TwoBrittle": int((sensitivity.synthesisLabel == "brittle_exact_instance_reversal").sum()) == 2,
        "claimCount": len(claims) == 10,
        "claimBoundaryReview": all(claim_checks.values()),
        "provenanceReview": all(provenance_checks.values()),
    }
    status = "PASS" if all(content_checks.values()) else "FAIL"
    files = sorted(path for path in required if path.exists())
    audit = {
        "schemaVersion": SCHEMA,
        "researchStepId": "S14",
        "status": status,
        "missing": missing,
        "checks": content_checks,
        "files": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in files
        ],
    }
    write_json(args.output / "artifact_audit.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
