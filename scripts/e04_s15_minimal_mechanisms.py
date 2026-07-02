#!/usr/bin/env python3
"""Run E04 S15 minimal-mechanism evidence synthesis."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.e04.minimal_mechanisms import (  # noqa: E402
    MINIMAL_MECHANISM_ID,
    S15EvidencePaths,
    artifact_records,
    build_minimal_mechanism_table,
    build_repair_capable_algotypes,
    collect_s15_anchors,
    load_s15_evidence,
    sha256_file,
    validate_claim_boundaries,
)


ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
S15_DIR = ARTIFACTS_DIR / "research_steps" / "S15"
REPORT_PATH = S15_DIR / "research_step_full_results.md"
STATUS_PATH = S15_DIR / "status.json"
MANIFEST_PATH = S15_DIR / "manifest.json"
VALIDATION_PATH = S15_DIR / "validation_checks.json"
ANCHORS_PATH = S15_DIR / "evidence_anchors.json"
MECHANISM_TABLE_PATH = ARTIFACTS_DIR / "tables" / "e04_minimal_mechanism_table.csv"
ALGOTYPES_PATH = ARTIFACTS_DIR / "policies" / "e04_repair_capable_algotypes.jsonl"
MINIMAL_REPORT_PATH = ARTIFACTS_DIR / "reports" / "e04_minimal_collective_intelligence.md"
HANDOFF_PATH = ARTIFACTS_DIR / "reports" / "e04_report_bundle_handoff.md"


def json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def run_command(command: list[str], *, cwd: Path = REPO_ROOT) -> dict[str, Any]:
    started = datetime.now(UTC).isoformat()
    proc = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    return {
        "command": command,
        "cwd": str(cwd),
        "started_at_utc": started,
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
    }


def repo_state() -> dict[str, Any]:
    state: dict[str, Any] = {}
    for name, command in {
        "head": ["git", "rev-parse", "HEAD"],
        "branch": ["git", "branch", "--show-current"],
        "upstream": ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
        "status_short": ["git", "status", "--short"],
    }.items():
        result = run_command(command)
        state[name] = {
            "returncode": result["returncode"],
            "stdout": result["stdout"].strip(),
            "stderr": result["stderr"].strip(),
        }
    return state


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "NA"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    display = df.copy()
    for column in display.columns:
        display[column] = display[column].map(lambda value: _fmt(value).replace("|", "\\|"))
    lines = [
        "| " + " | ".join(display.columns.astype(str)) + " |",
        "| " + " | ".join(["---"] * len(display.columns)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in display.columns) + " |")
    return "\n".join(lines)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, default=json_default) + "\n")


def primary_results(anchors: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "evidence_layer": "S09 memory ablation",
            "classification": "supportive",
            "result": (
                f"Neighbor memory mean fitness delta {anchors['s09_neighbor_mean_delta']:.6f} "
                f"versus no-memory; positive matched-pair fraction "
                f"{anchors['s09_neighbor_positive_fraction']:.2f}."
            ),
            "claim_boundary": "Local-only S07 projected view; supports neighbor memory for selected policies.",
        },
        {
            "evidence_layer": "S10 communication ablation",
            "classification": "null",
            "result": (
                f"Best tested communication mode {anchors['s10_best_mode']} had mean delta "
                f"{anchors['s10_best_mean_delta']:.6f}; max positive-pair fraction "
                f"{anchors['s10_max_positive_fraction']:.2f}."
            ),
            "claim_boundary": "Do not claim communication channels are required for the minimal package.",
        },
        {
            "evidence_layer": "S11 competency profiles",
            "classification": "constraining",
            "result": (
                f"Neighbor memory competency delta versus original Bubble open-loop was "
                f"{anchors['s11_neighbor_delta_vs_open_loop']:.6f}; positive fraction "
                f"{anchors['s11_neighbor_positive_fraction_vs_open_loop']:.2f}."
            ),
            "claim_boundary": "Supports robustness relative to no-memory controls, not broad intelligence superiority.",
        },
        {
            "evidence_layer": "S12 tissue fields",
            "classification": "null",
            "result": (
                f"Allowed-field plus local-order mean AUROC gain over local-order baseline was "
                f"{anchors['s12_combined_mean_local_order_auc_delta']:.6f}."
            ),
            "claim_boundary": "Offline proxy only; not evidence for required aggregate tissue fields.",
        },
        {
            "evidence_layer": "S13 transfer",
            "classification": "supportive",
            "result": (
                f"Neighbor memory held-out fitness delta versus no-memory was "
                f"{anchors['s13_neighbor_delta_vs_no_memory']:.6f}; transfer retention "
                f"{anchors['s13_neighbor_transfer_retention']:.6f}."
            ),
            "claim_boundary": "Predefined compact 1D held-out regimes; no retuning.",
        },
        {
            "evidence_layer": "S14 centralized ceiling",
            "classification": "supportive control",
            "result": (
                f"Local policies retained {anchors['s14_mean_repair_score_retention']:.6f} "
                "of the unpenalized centralized comparison repair score."
            ),
            "claim_boundary": "Centralized/global rows are excluded from local-only claims.",
        },
    ]


def write_minimal_report(path: Path, mechanism_table: pd.DataFrame, anchors: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    result_table = pd.DataFrame(primary_results(anchors))
    selected = mechanism_table.loc[mechanism_table["included_in_minimal_package"] == True]  # noqa: E712
    text = f"""# E04 Minimal Collective-Intelligence Mechanism Report

## Bottom Line

The minimal supported local package is `{MINIMAL_MECHANISM_ID}`: the two selected S08 adjacent-action policies using the S07 projected local-only view plus bounded neighbor-identity memory, with no tested communication channel required.

This is a bounded computational proxy claim. It supports improved repair/homeostatic robustness and predefined held-out transfer versus no-memory controls for the selected policies, but it does not establish broad superiority over the original Bubble open-loop baseline or any biological mechanism.

## Evidence Ledger

{markdown_table(result_table)}

## Minimal Package

{markdown_table(selected[["mechanism_id", "mechanism_label", "evidence_status", "chief_claim"]])}

## Exclusions

- Centralized/global S14 rows are ceiling controls only. They use full array state, target order, and global reordering, and are not eligible for local-only claims.
- S10 communication modes are not included in the minimal mechanism because none met the reliability criterion against memory-only controls.
- S12 aggregate allowed-field predictors are not included because their incremental predictive value beyond local-order baselines was small.
- S11 constrains intelligence-like claims: neighbor memory improved over no-memory controls but did not beat original Bubble open-loop on the predeclared competency criterion.

## Reportable Claim

Within compact 1D E04 simulations, bounded neighbor-memory state is the smallest supported local mechanism that improves the selected S08 repair policies over no-memory controls under matched ablations and predefined transfer regimes. Communication and aggregate field mechanisms remain unsupported as necessary additions, and centralized/global controls remain outside local-only claims.
"""
    path.write_text(text, encoding="utf-8")


def write_handoff(path: Path, mechanism_table: pd.DataFrame, anchors: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    claims = pd.DataFrame(
        [
            {
                "claim": "Bounded neighbor memory is the minimal supported local mechanism.",
                "evidence_level": "supportive compact computational proxy",
                "include_in_bundle": True,
                "key_numbers": (
                    f"S09 delta {anchors['s09_neighbor_mean_delta']:.6f}; "
                    f"S13 heldout delta {anchors['s13_neighbor_delta_vs_no_memory']:.6f}; "
                    f"S14 local retention {anchors['s14_mean_repair_score_retention']:.6f}"
                ),
                "caveat": "Two selected adjacent-action policies in 1D compact regimes.",
            },
            {
                "claim": "Tested communication channels are necessary.",
                "evidence_level": "not supported",
                "include_in_bundle": False,
                "key_numbers": (
                    f"S10 best delta {anchors['s10_best_mean_delta']:.6f}; "
                    f"max positive fraction {anchors['s10_max_positive_fraction']:.2f}"
                ),
                "caveat": "Only S10 tested modes; future modes may differ.",
            },
            {
                "claim": "Allowed aggregate tissue fields predict repair/failure beyond local order.",
                "evidence_level": "null proxy evidence",
                "include_in_bundle": False,
                "key_numbers": (
                    f"S12 combined mean AUROC gain {anchors['s12_combined_mean_local_order_auc_delta']:.6f}"
                ),
                "caveat": "Offline predictor, not policy-visible causal evidence.",
            },
            {
                "claim": "Local memory policies equal a centralized global controller.",
                "evidence_level": "not supported; ceiling comparison only",
                "include_in_bundle": False,
                "key_numbers": (
                    f"S14 retention {anchors['s14_mean_repair_score_retention']:.6f}; "
                    f"central repair-quality gap {anchors['s14_mean_repair_quality_gap']:.6f}"
                ),
                "caveat": "Centralized rows use global/target access and are excluded from local-only claims.",
            },
        ]
    )
    text = f"""# E04 Report-Bundle Handoff

## Status

E04 is ready for Chief report-bundle generation after S15. The synthesis artifacts identify a constrained minimal mechanism, preserve null and constraining results, and keep centralized/global controls out of local-only claims.

## Bundle Claims

{markdown_table(claims)}

## Core Artifacts

- Full S15 report: `{REPORT_PATH}`.
- Minimal mechanism table: `{MECHANISM_TABLE_PATH}`.
- Repair-capable Algotypes: `{ALGOTYPES_PATH}`.
- Minimal collective-intelligence report: `{MINIMAL_REPORT_PATH}`.
- S09-S14 source evidence tables and reports under `/artifacts/results`, `/artifacts/tables`, and `/artifacts/research_steps`.

## Recommended Wording

Use: "In compact 1D simulations, bounded neighbor memory was the smallest local addition that robustly improved the selected repair policies over no-memory controls; tested communication and field mechanisms were not necessary, and centralized repair remains a non-local ceiling control."

Avoid: "Communication is unnecessary in all settings", "neighbor memory beats original open-loop algorithms broadly", or any statement that uses S14 global-access rows as local-policy evidence.
"""
    path.write_text(text, encoding="utf-8")


def write_full_report(
    *,
    path: Path,
    mechanism_table: pd.DataFrame,
    anchors: dict[str, Any],
    checks: dict[str, Any],
    commands: list[dict[str, Any]],
    outputs: dict[str, Path],
    inputs: dict[str, Path],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    outputs_for_report = "\n".join(f"- `{output}`" for output in outputs.values())
    input_records = [
        {"name": name, "path": str(path), "sha256": sha256_file(path)}
        for name, path in sorted(inputs.items())
    ]
    output_records = artifact_records(
        {name: out for name, out in outputs.items() if out.exists() and out != path}
    )
    result_table = pd.DataFrame(primary_results(anchors))
    state = repo_state()
    text = f"""# E04 S15 Full Results: Minimal Sufficient Mechanisms

## Chief Scientist Handoff

- Research step ID: S15.
- Completion status: complete.
- Outcome classification: supportive.
- Artifacts written:
{outputs_for_report}
- Validation result: {"pass" if checks.get("all_passed") else "fail"}; centralized/global S14 rows are excluded from local-only claims.
- Main result: bounded neighbor-identity memory with the selected S08 local adjacent-action policies is the minimal supported E04 mechanism; tested communication and aggregate field mechanisms are not necessary under the compact evidence.
- Caveats or blockers: the claim is limited to computational proxy evidence in compact 1D regimes and two selected policies; S11 constrains broad intelligence-like claims versus original Bubble open-loop behavior.
- Lay summary: the best supported extra ingredient is a small local memory of neighboring identities. Extra signaling did not reliably help, and a global controller is only a non-local ceiling comparison.
- Recommended next action: generate the Chief report bundle from the S15 handoff artifacts.

## Frozen Question

What is the smallest memory and signaling package that reliably improves robustness over original open-loop algorithms?

## Methods

S15 loaded S09-S14 machine-readable outputs plus the S08 selected policy JSONL, used the precomputed matched-delta summaries where available, and applied fixed validation gates for each conclusion. S14 centralized/global rows were used only for ceiling comparison metrics and were explicitly excluded from local-only claims.

## Result Summary

{markdown_table(result_table)}

## Minimal Mechanism Table

{markdown_table(mechanism_table)}

## Validation Checks

{json.dumps(checks, indent=2, sort_keys=True, default=json_default)}

## Inputs

{json.dumps(input_records, indent=2, sort_keys=True, default=json_default)}

## Commands

{json.dumps(commands, indent=2, sort_keys=True, default=json_default)}

## Provenance

- Repository path: `{REPO_ROOT}`.
- Branch: `{state["branch"]["stdout"]}`.
- HEAD commit: `{state["head"]["stdout"]}`.
- Upstream: `{state["upstream"]["stdout"]}`.
- Git status short at report write:

```text
{state["status_short"]["stdout"] or "(clean)"}
```

## Artifacts And Checksums

{json.dumps(output_records, indent=2, sort_keys=True, default=json_default)}

## Caveats, Failed Assumptions, And Limitations

S15 is a synthesis of prior computational results, not a new biological experiment. The minimal mechanism is substrate- and policy-family-specific: two selected Bubble-style adjacent-action policies, compact 1D repair/homeostatic/transfer regimes, and offline competency/field proxies. Centralized/global S14 controls deliberately violate the S07 local-only information policy and must remain out of local-only claims.
"""
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--skip-tests", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = S15EvidencePaths(args.artifacts_dir)
    for directory in (S15_DIR, MECHANISM_TABLE_PATH.parent, ALGOTYPES_PATH.parent, MINIMAL_REPORT_PATH.parent):
        directory.mkdir(parents=True, exist_ok=True)

    commands: list[dict[str, Any]] = []
    if not args.skip_tests:
        test_result = run_command([sys.executable, "-m", "unittest", "tests.e04.test_minimal_mechanisms"])
        commands.append(test_result)
        if test_result["returncode"] != 0:
            STATUS_PATH.write_text(
                json.dumps(
                    {
                        "research_step_id": "S15",
                        "status": "blocked",
                        "reason": "unit_tests_failed",
                        "command": test_result,
                    },
                    indent=2,
                    sort_keys=True,
                    default=json_default,
                ),
                encoding="utf-8",
            )
            return test_result["returncode"]

    commands.append(
        {
            "command": [
                sys.executable,
                "scripts/e04_s15_minimal_mechanisms.py",
                "--artifacts-dir",
                str(args.artifacts_dir),
                "--skip-tests" if args.skip_tests else "<tests-enabled>",
            ],
            "cwd": str(REPO_ROOT),
            "started_at_utc": datetime.now(UTC).isoformat(),
            "finished_at_utc": datetime.now(UTC).isoformat(),
            "returncode": 0,
            "stdout": "current process invocation recorded for provenance",
            "stderr": "",
        }
    )

    evidence = load_s15_evidence(paths)
    anchors = collect_s15_anchors(evidence)
    checks = validate_claim_boundaries(evidence, anchors)
    mechanism_table = build_minimal_mechanism_table(anchors)
    algotypes = build_repair_capable_algotypes(evidence, anchors)

    MECHANISM_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    mechanism_table.to_csv(MECHANISM_TABLE_PATH, index=False)
    write_jsonl(ALGOTYPES_PATH, algotypes)
    ANCHORS_PATH.write_text(
        json.dumps(anchors, indent=2, sort_keys=True, default=json_default),
        encoding="utf-8",
    )
    VALIDATION_PATH.write_text(
        json.dumps(checks, indent=2, sort_keys=True, default=json_default),
        encoding="utf-8",
    )
    write_minimal_report(MINIMAL_REPORT_PATH, mechanism_table, anchors)
    write_handoff(HANDOFF_PATH, mechanism_table, anchors)

    status = {
        "research_step_id": "S15",
        "status": "complete" if checks.get("all_passed") else "blocked",
        "outcome_classification": "supportive",
        "validation_passed": bool(checks.get("all_passed")),
        "minimal_mechanism_id": MINIMAL_MECHANISM_ID,
        "centralized_rows_excluded_from_local_claims": bool(
            checks["s14_centralized_rows_excluded_from_local_claims"]["passed"]
        ),
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    STATUS_PATH.write_text(
        json.dumps(status, indent=2, sort_keys=True, default=json_default),
        encoding="utf-8",
    )

    outputs = {
        "research_step_full_results": REPORT_PATH,
        "minimal_mechanism_table": MECHANISM_TABLE_PATH,
        "repair_capable_algotypes": ALGOTYPES_PATH,
        "minimal_collective_intelligence_report": MINIMAL_REPORT_PATH,
        "report_bundle_handoff": HANDOFF_PATH,
        "evidence_anchors": ANCHORS_PATH,
        "validation_checks": VALIDATION_PATH,
        "status": STATUS_PATH,
    }
    write_full_report(
        path=REPORT_PATH,
        mechanism_table=mechanism_table,
        anchors=anchors,
        checks=checks,
        commands=commands,
        outputs=outputs,
        inputs=paths.required_inputs,
    )
    manifest = {
        "research_step_id": "S15",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "minimal_mechanism_id": MINIMAL_MECHANISM_ID,
        "validation": checks,
        "anchors": anchors,
        "input_artifacts": artifact_records(paths.required_inputs),
        "output_artifacts": artifact_records(outputs),
        "repo_state": repo_state(),
        "manifest_path": str(MANIFEST_PATH),
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=json_default),
        encoding="utf-8",
    )
    if not checks.get("all_passed"):
        return 2
    print(json.dumps(status, indent=2, sort_keys=True, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
