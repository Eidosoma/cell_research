#!/usr/bin/env python3
"""Validate E03 S14 taxonomy rules, witnesses, provenance, and report bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import pandas as pd


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from src.detours.taxonomy import (  # noqa: E402
    CATEGORY_ORDER,
    EvidenceFlags,
    anthropomorphic_assertion_hits,
    strongest_supported_category,
    supported_categories,
)


SCHEMA = "e03.s14.validation.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=lambda item: item.item() if hasattr(item, "item") else str(item),
        )
        + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def digest_frame(path: Path) -> str:
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    payload = frame.to_json(orient="records", date_format="iso", double_precision=12)
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_replay(output: Path) -> dict[str, bool]:
    witnesses = pd.read_parquet(output / "witness_replay_validation.parquet")
    scenarios = pd.read_parquet(output / "benchmark_scenarios.parquet")
    source_hashes = all(
        Path("/artifacts", row.source_artifact).exists()
        and sha256_file(Path("/artifacts", row.source_artifact)) == row.source_sha256
        for row in scenarios.itertuples(index=False)
    )
    return {
        "sevenWitnesses": len(witnesses) == 7,
        "allWitnessesPassed": bool(witnesses.passed.all()),
        "allBoundaryIdsUnique": scenarios.boundary.nunique() == 7,
        "allBenchmarksConfirmationOnly": bool(scenarios.confirmation_only.all()),
        "noBenchmarkTrainingAllowed": not bool(scenarios.training_allowed.any()),
        "allBenchmarkSourceHashesMatch": source_hashes,
    }


def deterministic_rebuild(output: Path, report_inputs: Path, scratch: Path) -> dict[str, Any]:
    replay_output = scratch / "S14"
    replay_inputs = scratch / "report_inputs"
    if scratch.exists():
        shutil.rmtree(scratch)
    replay_output.mkdir(parents=True)
    replay_inputs.mkdir(parents=True)
    command = [
        sys.executable,
        str(REPOSITORY / "scripts" / "build_e03_s14_taxonomy.py"),
        "--output",
        str(replay_output),
        "--report-inputs",
        str(replay_inputs),
    ]
    completed = subprocess.run(
        command,
        cwd=REPOSITORY,
        check=False,
        capture_output=True,
        text=True,
    )
    keys = [
        "taxonomy_categories.parquet",
        "claim_to_evidence_matrix.parquet",
        "benchmark_scenarios.parquet",
        "benchmark_witness_traces.parquet",
        "witness_replay_validation.parquet",
        "classification_summary.csv",
        "evidence_boundary_matrix.csv",
    ]
    comparisons = {}
    if completed.returncode == 0:
        for name in keys:
            comparisons[name] = digest_frame(output / name) == digest_frame(replay_output / name)
    return {
        "command": " ".join(command),
        "returnCode": completed.returncode,
        "stdoutTail": completed.stdout[-2000:],
        "stderrTail": completed.stderr[-2000:],
        "comparisons": comparisons,
        "success": completed.returncode == 0 and len(comparisons) == len(keys) and all(comparisons.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/artifacts/research_steps/S14"))
    parser.add_argument("--report-inputs", type=Path, default=Path("/artifacts/report_inputs"))
    parser.add_argument("--require-report", action="store_true")
    parser.add_argument("--replay-only", action="store_true")
    parser.add_argument("--skip-rebuild", action="store_true")
    parser.add_argument("--scratch", type=Path, default=Path("/cache/e03_s14_deterministic_rebuild"))
    args = parser.parse_args()

    replay_checks = validate_replay(args.output)
    if args.replay_only:
        result = {
            "schemaVersion": SCHEMA,
            "researchStepId": "S14",
            "mode": "replay_only",
            "checks": replay_checks,
            "success": all(replay_checks.values()),
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        raise SystemExit(0 if result["success"] else 1)

    required = [
        args.output / "detour_taxonomy.md",
        args.output / "taxonomy_categories.csv",
        args.output / "taxonomy_categories.parquet",
        args.output / "claim_to_evidence_matrix.csv",
        args.output / "claim_to_evidence_matrix.parquet",
        args.output / "classification_summary.csv",
        args.output / "benchmark_scenarios.csv",
        args.output / "benchmark_scenarios.parquet",
        args.output / "benchmark_witness_traces.parquet",
        args.output / "witness_replay_validation.parquet",
        args.output / "adaptive_detour_gate_audit.json",
        args.output / "anthropomorphic_language_audit.json",
        args.output / "e07_holdout_manifest.json",
        args.output / "e07_handoff.md",
        args.output / "taxonomy_ladder.png",
        args.output / "taxonomy_ladder.svg",
        args.output / "evidence_boundary_map.png",
        args.output / "evidence_boundary_map.svg",
        args.output / "representative_witnesses.png",
        args.output / "representative_witnesses.svg",
        args.output / "evidence_boundary_matrix.csv",
        args.output / "evidence_index.csv",
        args.output / "caveat_register.csv",
        args.output / "input_immutability.json",
        args.output / "environment.json",
        args.output / "provenance_manifest.json",
        args.report_inputs / "evidence_index.csv",
        args.report_inputs / "evidence_index.json",
        args.report_inputs / "methods_summary.md",
        args.report_inputs / "methods_summary.json",
        args.report_inputs / "figure_index.csv",
        args.report_inputs / "table_index.csv",
        args.report_inputs / "claim_to_evidence_matrix.csv",
        args.report_inputs / "claim_to_evidence_matrix.parquet",
        args.report_inputs / "caveat_register.csv",
        args.report_inputs / "caveat_register.json",
        args.report_inputs / "provenance_manifest.json",
        args.report_inputs / "lay_summary.md",
        args.report_inputs / "lay_summary.json",
        args.report_inputs / "taxonomy_categories.csv",
        args.report_inputs / "classification_summary.csv",
        args.report_inputs / "benchmark_scenarios.csv",
        args.report_inputs / "evidence_boundary_matrix.csv",
        args.report_inputs / "report_bundle_manifest.json",
    ]
    if args.require_report:
        required.extend(
            [
                args.output / "research_step_full_results.md",
                args.output / "status.json",
                args.output / "result_summary.json",
                args.output / "artifact_manifest.json",
            ]
        )
    missing = [str(path) for path in required if not path.exists()]

    contract = read_json(REPOSITORY / "analysis" / "e03_s14_detour_taxonomy_contract.json")
    categories = pd.read_parquet(args.output / "taxonomy_categories.parquet")
    claims = pd.read_parquet(args.output / "claim_to_evidence_matrix.parquet")
    evidence = pd.read_csv(args.output / "evidence_index.csv")
    scenarios = pd.read_parquet(args.output / "benchmark_scenarios.parquet")
    witnesses = pd.read_parquet(args.output / "benchmark_witness_traces.parquet")
    adaptive = read_json(args.output / "adaptive_detour_gate_audit.json")
    language = read_json(args.output / "anthropomorphic_language_audit.json")
    immutability = read_json(args.output / "input_immutability.json")
    provenance = read_json(args.output / "provenance_manifest.json")
    report_manifest = read_json(args.report_inputs / "report_bundle_manifest.json")

    rule_errors: list[str] = []
    for row in claims.itertuples(index=False):
        flags = EvidenceFlags(**json.loads(row.flags_json))
        expected = strongest_supported_category(flags)
        if row.strongest_supported_category != expected:
            rule_errors.append(f"{row.claim_id}:{row.strongest_supported_category}!={expected}")
        if json.loads(row.supported_categories) != list(supported_categories(flags)):
            rule_errors.append(f"{row.claim_id}:supported-category-list")

    evidence_hashes = all(
        Path(row.path).exists() and sha256_file(Path(row.path)) == row.sha256
        for row in evidence.itertuples(index=False)
    )
    provenance_hashes = all(
        Path(row["path"]).exists() and sha256_file(Path(row["path"])) == row["sha256"]
        for row in provenance["inputs"]
    )
    report_hashes = all(
        Path(row["path"]).exists() and sha256_file(Path(row["path"])) == row["sha256"]
        for row in report_manifest["inputs"]
    )
    registry_ids = {row["claimId"] for row in contract["majorClaimRegistry"]}
    observed_ids = set(claims.claim_id)
    category_names = set(categories.category)
    taxonomy_text = (args.output / "detour_taxonomy.md").read_text(encoding="utf-8")
    e07_text = (args.output / "e07_handoff.md").read_text(encoding="utf-8")
    assertion_hits = anthropomorphic_assertion_hits(claims.claim.tolist())
    lay_text = (args.report_inputs / "lay_summary.md").read_text(encoding="utf-8")
    assertion_hits.extend(anthropomorphic_assertion_hits([lay_text]))
    valid_evidence_ids = set(evidence.evidence_id)
    claim_evidence_links = all(
        bool(row.evidence_ids)
        and set(str(row.evidence_ids).split(";")).issubset(valid_evidence_ids)
        and all(
            (Path("/artifacts") / relative).exists()
            if str(relative).startswith("research_steps/")
            else Path(relative).exists()
            for relative in str(row.evidence_paths).split(";")
        )
        for row in claims.itertuples(index=False)
    )
    markdown_summary_fields = [
        "Research step ID",
        "Completion status",
        "Artifacts written",
        "Validation result",
        "Outcome classification",
        "Caveats or blockers",
        "Recommended next action",
    ]

    checks = {
        "requiredFilesPresent": not missing,
        "contractHasFiveCategories": len(contract["categories"]) == 5,
        "categoryOrderExact": [row["category"] for row in contract["categories"]] == list(CATEGORY_ORDER),
        "taxonomyCategoryRows": category_names == set(CATEGORY_ORDER) | {"not_supported_in_E03"},
        "claimRegistryExact": registry_ids == observed_ids and len(claims) == 22,
        "oneStrongestCategoryPerClaim": claims.strongest_supported_category.notna().all(),
        "categoryRulesRecomputed": not rule_errors,
        "adaptiveCategoryEmpty": int((claims.strongest_supported_category == "adaptive_detour").sum()) == 0,
        "adaptiveGateFailsOnlyDeclaredNullGate": adaptive["failedGate"] == "matched_null_exceedance" and not adaptive["adaptiveDetourPass"],
        "nativeLargeClaimLocal": claims.set_index("claim_id").loc["C15", "strongest_supported_category"] == "local_observable_backtracking",
        "engineeredLargeClaimGlobal": claims.set_index("claim_id").loc["C17", "strongest_supported_category"] == "global_regression",
        "crossLayerDirectionNotPooled": claims.set_index("claim_id").loc["C18", "strongest_supported_category"] == "not_supported_in_E03",
        "footruleEmdNotIndependent": claims.set_index("claim_id").loc["C22", "strongest_supported_category"] == "not_supported_in_E03",
        "allWitnessBoundariesPresent": set(scenarios.boundary) == set(contract["benchmarkPolicy"]["requiredBoundaries"]),
        "witnessTraceCoverage": set(scenarios.benchmark_id) == set(witnesses.benchmark_id),
        "replayChecks": all(replay_checks.values()),
        "evidenceLinksExistAndHash": evidence_hashes,
        "everyClaimHasValidEvidenceLinks": claim_evidence_links,
        "provenanceLinksExistAndHash": provenance_hashes,
        "inputsImmutable": immutability["success"] and not immutability["changed"],
        "reportBundleHashes": report_hashes,
        "anthropomorphicAssertionAudit": language["success"] and not assertion_hits,
        "taxonomyTopSummaryComplete": all(field in taxonomy_text for field in markdown_summary_fields),
        "e07TopSummaryComplete": all(field in e07_text for field in markdown_summary_fields),
        "e07LeakageRuleExplicit": "must not use" in e07_text and "confirmation-only" in e07_text,
    }

    rebuild = {"success": True, "skipped": True}
    if not args.skip_rebuild:
        rebuild = deterministic_rebuild(args.output, args.report_inputs, args.scratch)
        checks["deterministicRebuild"] = bool(rebuild["success"])

    if args.require_report and not missing:
        report_text = (args.output / "research_step_full_results.md").read_text(encoding="utf-8")
        status = read_json(args.output / "status.json")
        required_status = {
            "researchStepId",
            "stepNumber",
            "success",
            "status",
            "artifactsWritten",
            "validationResult",
            "caveatsOrBlockers",
            "recommendedNextAction",
        }
        checks.update(
            {
                "canonicalReportTopSummaryComplete": all(field in report_text for field in markdown_summary_fields),
                "canonicalReportDetailedSections": all(
                    header in report_text
                    for header in [
                        "## Lay summary",
                        "## Frozen question",
                        "## Inputs",
                        "## Methods",
                        "## Commands",
                        "## Results",
                        "## Validation",
                        "## Caveats",
                        "## Provenance",
                    ]
                ),
                "statusSchemaComplete": required_status.issubset(status),
                "statusIsS14Complete": status.get("researchStepId") == "S14" and status.get("stepNumber") == 14 and status.get("success") is True,
            }
        )

    status_value = "PASS" if not missing and all(checks.values()) else "FAIL"
    result = {
        "schemaVersion": SCHEMA,
        "researchStepId": "S14",
        "status": status_value,
        "success": status_value == "PASS",
        "checks": checks,
        "missing": missing,
        "categoryRuleErrors": rule_errors,
        "replayChecks": replay_checks,
        "deterministicRebuild": rebuild,
        "counts": {
            "claims": len(claims),
            "benchmarks": len(scenarios),
            "witnessRows": len(witnesses),
            "evidenceLinks": len(evidence),
            "reportInputs": len(report_manifest["inputs"]),
        },
    }
    canonical_json(args.output / "validation_results.json", result)
    (args.output / "validation.log").write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
            default=lambda item: item.item() if hasattr(item, "item") else str(item),
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
            default=lambda item: item.item() if hasattr(item, "item") else str(item),
        )
    )
    raise SystemExit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
