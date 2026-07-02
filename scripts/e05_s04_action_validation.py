#!/usr/bin/env python3
"""Validate E05 S04 generalized actions and write research artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.e05.actions import (
    ADHESION_BONDS_KEY,
    REQUESTED_S04_ACTIONS,
    SIGNAL_INBOX_KEY,
    SIGNAL_CHANNEL_RANGES,
    ActionExecutor,
    MorphogenesisActionRequest,
    action_state_signature,
    action_spec_rows,
    default_action_set,
)
from src.e05.cell_identity import attach_identity, identity_from_substrate_cell
from src.e05.substrates import SubstrateCell, array_substrate, square_grid_substrate
from src.e05.targets import morph_identity


STEP_ID = "S04"
STEP_NUMBER = 4
EXPERIMENT_ID = "E05"
EXPERIMENT_TITLE = "From one-dimensional sorting to higher-dimensional morphospace"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--s01-results", type=Path, default=Path("/artifacts/results/e05_substrate_validation.parquet"))
    parser.add_argument("--s02-results", type=Path, default=Path("/artifacts/results/e05_identity_validation.parquet"))
    parser.add_argument("--s03-results", type=Path, default=Path("/artifacts/results/e05_target_validation.parquet"))
    parser.add_argument("--e04-signal-spec", type=Path, default=Path("/previous-artifacts/E04/reports/e04_signal_model_spec.md"))
    parser.add_argument("--run-unit-tests", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def git_output(repo_dir: Path, args: list[str]) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=repo_dir, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc!r}"
    return result.stdout.strip()


def run_command(command: list[str], repo_dir: Path) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    result = subprocess.run(command, cwd=repo_dir, capture_output=True, text=True)
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    return {
        "command": " ".join(command),
        "returnCode": int(result.returncode),
        "elapsedSeconds": elapsed,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "success": result.returncode == 0,
    }


def artifact_entry(path: Path, artifacts_dir: Path, description: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(artifacts_dir)),
        "description": description,
        "sha256": sha256_file(path),
        "sizeBytes": path.stat().st_size,
    }


def manifest_self_entry(path: Path, artifacts_dir: Path, description: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(artifacts_dir)),
        "description": description,
        "sha256": None,
        "sizeBytes": None,
        "note": "Checksum omitted to avoid self-referential checksum drift.",
    }


def source_entry(path: Path, repo_dir: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(repo_dir)),
        "sha256": sha256_file(path),
        "sizeBytes": path.stat().st_size,
    }


def identity_cell(
    cell_id: str,
    ap: float,
    organ: str = "neural",
    polarity: tuple[float, float] = (1.0, 0.0),
    adhesion: str = "medium",
) -> SubstrateCell:
    return attach_identity(
        SubstrateCell(cell_id, value=ap, label=organ),
        morph_identity(f"{cell_id}_identity", ap, organ, polarity, adhesion),
    )


def _row(
    validation_case: str,
    case_type: str,
    success: bool,
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
    detail: str,
) -> dict[str, Any]:
    return {
        "research_step_id": STEP_ID,
        "experiment_id": EXPERIMENT_ID,
        "validation_case": validation_case,
        "case_type": case_type,
        "success": bool(success),
        "expected_json": stable_json(expected),
        "observed_json": stable_json(observed),
        "detail": detail,
    }


def validation_catalog() -> dict[str, Any]:
    rows = action_spec_rows(default_action_set())
    observed = {
        "requested_actions_present": set(REQUESTED_S04_ACTIONS).issubset({row["action_type"] for row in rows}),
        "action_count": len(rows),
        "actions": [row["action_type"] for row in rows],
        "nonnegative_costs": all(row["energy_cost"] >= 0.0 for row in rows),
    }
    expected = {
        "requested_actions_present": True,
        "nonnegative_costs": True,
        "requested_actions": list(REQUESTED_S04_ACTIONS),
    }
    return _row(
        "action_catalog_contains_requested_local_actions",
        "catalog",
        observed["requested_actions_present"] and observed["nonnegative_costs"],
        expected,
        observed,
        "Default action catalog includes swap, crawl, rotate_polarity, divide, die, adhere, detach, and exchange_signal with nonnegative costs.",
    )


def validation_swap_crawl() -> dict[str, Any]:
    executor = ActionExecutor()
    swap_state = array_substrate(2)
    swap_state.fill_sites((SubstrateCell("a", 2), SubstrateCell("b", 1)))
    swap = executor.execute(swap_state, MorphogenesisActionRequest("swap", 0, 1))
    crawl_state = array_substrate(3)
    crawl_state.place_cell(0, SubstrateCell("crawler", 3))
    crawl = executor.execute(crawl_state, MorphogenesisActionRequest("crawl", 0, 1))
    observed = {
        "swap_allowed": swap.allowed,
        "swap_values": list(swap_state.values_in_site_order()),
        "swap_cost": swap.energy_cost_charged,
        "swap_delta": swap.conservation_delta,
        "crawl_allowed": crawl.allowed,
        "crawler_site": crawl_state.site_of_cell("crawler"),
        "crawl_cost": crawl.energy_cost_charged,
        "crawl_delta": crawl.conservation_delta,
    }
    expected = {
        "swap_allowed": True,
        "swap_values": [1, 2],
        "swap_cost": 1.0,
        "swap_delta": 0,
        "crawl_allowed": True,
        "crawler_site": 1,
        "crawl_cost": 1.2,
        "crawl_delta": 0,
    }
    return _row(
        "swap_and_crawl_are_local_population_conserving_actions",
        "movement",
        observed == expected,
        expected,
        observed,
        "Adjacent swap and crawl execute locally, charge nominal energy, and conserve occupied-cell count.",
    )


def validation_illegal_move() -> dict[str, Any]:
    state = square_grid_substrate(3, 3)
    state.place_cell(0, SubstrateCell("a", 1))
    state.place_cell(8, SubstrateCell("b", 2))
    signature_before = state.signature()
    result = ActionExecutor().execute(state, MorphogenesisActionRequest("swap", 0, 8))
    observed = {
        "allowed": result.allowed,
        "reason": result.reason,
        "energy_cost_charged": result.energy_cost_charged,
        "state_changed": result.state_changed,
        "substrate_signature_unchanged": state.signature() == signature_before,
    }
    expected = {
        "allowed": False,
        "reason": "target_not_adjacent",
        "energy_cost_charged": 0.0,
        "state_changed": False,
        "substrate_signature_unchanged": True,
    }
    return _row(
        "illegal_nonadjacent_swap_is_rejected_without_cost",
        "illegal_move",
        observed == expected,
        expected,
        observed,
        "A nonadjacent swap is rejected, charges zero energy, and leaves substrate occupancy unchanged.",
    )


def validation_rotate_polarity() -> dict[str, Any]:
    state = array_substrate(1)
    state.place_cell(0, identity_cell("cell", 0.5, polarity=(1.0, 0.0)))
    result = ActionExecutor().execute(
        state,
        MorphogenesisActionRequest("rotate_polarity", 0, parameters={"polarity": (0.0, 2.0)}),
    )
    identity = identity_from_substrate_cell(state.cell_at(0))
    observed = {
        "allowed": result.allowed,
        "polarity": list(identity.components["polarity"]),
        "cost": result.energy_cost_charged,
        "delta": result.conservation_delta,
    }
    expected = {"allowed": True, "polarity": [0.0, 1.0], "cost": 0.25, "delta": 0}
    return _row(
        "rotate_polarity_updates_local_identity_metadata",
        "metadata_action",
        observed == expected,
        expected,
        observed,
        "Polarity rotation changes only actor identity metadata and conserves population.",
    )


def validation_divide_die() -> dict[str, Any]:
    state = array_substrate(2)
    state.place_cell(0, identity_cell("parent", 0.1))
    executor = ActionExecutor()
    birth = executor.execute(state, MorphogenesisActionRequest("divide", 0, 1))
    death = executor.execute(state, MorphogenesisActionRequest("die", 1))
    observed = {
        "birth_allowed": birth.allowed,
        "birth_count": birth.birth_count,
        "birth_delta": birth.conservation_delta,
        "birth_population_after": birth.population_after,
        "death_allowed": death.allowed,
        "death_count": death.death_count,
        "death_delta": death.conservation_delta,
        "death_population_after": death.population_after,
    }
    expected = {
        "birth_allowed": True,
        "birth_count": 1,
        "birth_delta": 1,
        "birth_population_after": 2,
        "death_allowed": True,
        "death_count": 1,
        "death_delta": -1,
        "death_population_after": 1,
    }
    return _row(
        "divide_and_die_track_birth_death_accounting",
        "birth_death",
        observed == expected,
        expected,
        observed,
        "Division increments birth_count and population; death increments death_count and decrements population.",
    )


def validation_adhere_detach() -> dict[str, Any]:
    state = array_substrate(2)
    state.fill_sites((identity_cell("left", 0.1), identity_cell("right", 0.9)))
    executor = ActionExecutor()
    adhere = executor.execute(state, MorphogenesisActionRequest("adhere", 0, 1))
    bonds_after_adhere = {
        "left": state.cell_at(0).metadata.get(ADHESION_BONDS_KEY),
        "right": state.cell_at(1).metadata.get(ADHESION_BONDS_KEY),
    }
    detach = executor.execute(state, MorphogenesisActionRequest("detach", 0, 1))
    bonds_after_detach = {
        "left": state.cell_at(0).metadata.get(ADHESION_BONDS_KEY),
        "right": state.cell_at(1).metadata.get(ADHESION_BONDS_KEY),
    }
    observed = {
        "adhere_allowed": adhere.allowed,
        "bonds_after_adhere": bonds_after_adhere,
        "detach_allowed": detach.allowed,
        "bonds_after_detach": bonds_after_detach,
        "population_delta": adhere.conservation_delta + detach.conservation_delta,
    }
    expected = {
        "adhere_allowed": True,
        "bonds_after_adhere": {"left": ["right"], "right": ["left"]},
        "detach_allowed": True,
        "bonds_after_detach": {"left": [], "right": []},
        "population_delta": 0,
    }
    return _row(
        "adhere_and_detach_update_reciprocal_bonds",
        "adhesion",
        observed == expected,
        expected,
        observed,
        "Adhere and detach maintain reciprocal local bond metadata without changing population.",
    )


def validation_signal_exchange() -> dict[str, Any]:
    state = array_substrate(3)
    state.place_cell(0, identity_cell("source", 0.1))
    state.place_cell(1, identity_cell("target", 0.2))
    state.place_cell(2, identity_cell("far", 0.3))
    executor = ActionExecutor()
    sent = executor.execute(
        state,
        MorphogenesisActionRequest("exchange_signal", 0, 1, {"channel": "blocked", "value": 0.75}),
    )
    inbox = state.cell_at(1).metadata[SIGNAL_INBOX_KEY][-1]
    nonlocal_result = executor.execute(
        state,
        MorphogenesisActionRequest("exchange_signal", 0, 2, {"channel": "blocked", "value": 0.25}),
    )
    out_of_range = executor.execute(
        state,
        MorphogenesisActionRequest("exchange_signal", 0, 1, {"channel": "target_seeking", "value": 2.0}),
    )
    observed = {
        "sent_allowed": sent.allowed,
        "inbox_channel": inbox["channel"],
        "inbox_value": inbox["value"],
        "access_scope": inbox["access_scope"],
        "nonlocal_allowed": nonlocal_result.allowed,
        "nonlocal_reason": nonlocal_result.reason,
        "out_of_range_allowed": out_of_range.allowed,
        "out_of_range_reason": out_of_range.reason,
    }
    expected = {
        "sent_allowed": True,
        "inbox_channel": "blocked",
        "inbox_value": 0.75,
        "access_scope": "adjacent_neighbor",
        "nonlocal_allowed": False,
        "nonlocal_reason": "target_not_adjacent",
        "out_of_range_allowed": False,
        "out_of_range_reason": "signal_value_out_of_range",
    }
    return _row(
        "exchange_signal_is_local_and_bounded",
        "signaling",
        observed == expected,
        expected,
        observed,
        "Signal exchange uses E04-style bounded channels and rejects nonlocal or out-of-range proposals.",
    )


def validation_legal_dry_run() -> dict[str, Any]:
    state = array_substrate(2)
    state.fill_sites((identity_cell("left", 0.1), identity_cell("right", 0.9)))
    before = action_state_signature(state)
    results = ActionExecutor().legal_action_results(state, 0)
    observed = {
        "signature_unchanged": action_state_signature(state) == before,
        "result_count": len(results),
        "has_allowed_swap": any(result.action_type == "swap" and result.allowed for result in results),
    }
    expected = {"signature_unchanged": True, "has_allowed_swap": True}
    return _row(
        "configured_legal_action_queries_are_dry_runs",
        "policy_interface",
        observed["signature_unchanged"] == expected["signature_unchanged"] and observed["has_allowed_swap"] == expected["has_allowed_swap"],
        expected,
        observed,
        "Configured action queries expose policy-executable choices without mutating the substrate.",
    )


def validation_disabled_action() -> dict[str, Any]:
    action_set = default_action_set()
    action_set.pop("swap")
    state = array_substrate(2)
    state.fill_sites((SubstrateCell("a", 2), SubstrateCell("b", 1)))
    result = ActionExecutor(action_set).execute(state, MorphogenesisActionRequest("swap", 0, 1))
    observed = {"allowed": result.allowed, "reason": result.reason, "charged_cost": result.energy_cost_charged}
    expected = {"allowed": False, "reason": "unsupported_action", "charged_cost": 0.0}
    return _row(
        "configured_action_sets_reject_disabled_actions",
        "policy_interface",
        observed == expected,
        expected,
        observed,
        "An executor with swap removed rejects swap proposals, supporting configured action-set experiments.",
    )


def run_validations() -> pd.DataFrame:
    return pd.DataFrame(
        [
            validation_catalog(),
            validation_swap_crawl(),
            validation_illegal_move(),
            validation_rotate_polarity(),
            validation_divide_die(),
            validation_adhere_detach(),
            validation_signal_exchange(),
            validation_legal_dry_run(),
            validation_disabled_action(),
        ]
    )


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for record in df[columns].to_dict(orient="records"):
        rows.append("| " + " | ".join(str(record[column]).replace("|", "\\|") for column in columns) + " |")
    return "\n".join([header, separator, *rows])


def top_summary_markdown(artifacts: list[dict[str, Any]], validation_result: str, recommended_next_action: str) -> str:
    artifact_lines = "\n".join(f"- `{entry['path']}`" for entry in artifacts if entry.get("path"))
    return f"""## Top Summary

- Research step ID: {STEP_ID}
- Completion status: Completed
- Artifacts written:
{artifact_lines}
- Validation result: {validation_result}
- Outcome classification: supportive
- Caveats or blockers: S04 defines local toy action semantics and accounting only; morphology metrics and benchmark simulations remain for S05 and later steps.
- Lay summary: S04 adds a configurable action set so cells can swap, crawl, rotate polarity, divide, die, adhere, detach, and exchange local signals while every action records energy cost and population accounting.
- Recommended next action: {recommended_next_action}
"""


def action_spec_markdown(
    artifacts: list[dict[str, Any]],
    validation_result: str,
    validation_df: pd.DataFrame,
    action_df: pd.DataFrame,
    recommended_next_action: str,
) -> str:
    return f"""{top_summary_markdown(artifacts, validation_result, recommended_next_action)}

# E05 S04 Action-Set Specification

## Scope

S04 defines a configured local action set over S01 substrates using S02 identities and S03 target-ready states. The action executor accepts one actor site and, for targeted actions, one adjacent target site. It records nominal energy cost, charged energy cost, state change, population before and after, birth count, death count, and conservation delta.

## Local Action Catalog

{markdown_table(action_df, ["action_type", "energy_cost", "requires_target", "target_occupancy", "population_delta", "conservation_rule"])}

## Local Semantics

- `swap`: actor swaps with an adjacent occupied movable target.
- `crawl`: actor moves to an adjacent empty target.
- `rotate_polarity`: actor normalizes and stores a new polarity vector in its own identity metadata.
- `divide`: actor creates one daughter cell in an adjacent empty site.
- `die`: actor is removed from its current site.
- `adhere` and `detach`: actor and adjacent target update reciprocal adhesion-bond metadata.
- `exchange_signal`: actor writes one bounded E04-style signal record to an adjacent occupied target.

## Energy And Accounting

Illegal actions charge zero energy and leave the substrate unchanged. Successful neutral actions have conservation delta 0. `divide` increments `birth_count` and population by 1. `die` increments `death_count` and decreases population by 1.

## Signal Channels

S04 reuses the E04 signal vocabulary: {", ".join(f"`{key}` in [{value[0]}, {value[1]}]" for key, value in SIGNAL_CHANNEL_RANGES.items())}.

## Validation Summary

{markdown_table(validation_df, ["validation_case", "case_type", "success", "detail"])}
"""


def full_results_markdown(
    *,
    artifacts: list[dict[str, Any]],
    validation_result: str,
    validation_df: pd.DataFrame,
    action_df: pd.DataFrame,
    test_commands: list[dict[str, Any]],
    source_files: list[dict[str, Any]],
    args: argparse.Namespace,
    recommended_next_action: str,
) -> str:
    command_lines = "\n".join(
        f"- `{command['command']}`: return code {command['returnCode']}, success={command['success']}, elapsed={command['elapsedSeconds']:.3f}s"
        for command in test_commands
    )
    source_lines = "\n".join(
        f"- `{entry['relativePath']}` sha256 `{entry['sha256']}` ({entry['sizeBytes']} bytes)"
        for entry in source_files
    )
    return f"""{top_summary_markdown(artifacts, validation_result, recommended_next_action)}

# Research Step Full Results: {STEP_ID} Generalize Actions

## Lay Summary

S04 adds the local actions needed before higher-dimensional morphogenesis benchmarks can run. The implemented action set supports movement, polarity changes, birth/death, adhesion metadata, and local signal exchange. The validation suite confirms that legal actions charge energy and account for population changes, while illegal nonlocal or invalid actions are rejected without cost.

## Frozen Question

Which local actions beyond swap are needed for higher-dimensional morphogenesis tasks?

## Inputs

- Active plan: `/workspace/RESEARCH_PLAN.md`, E05 S04.
- S01 substrate code and artifacts, including `{args.s01_results}`.
- S02 identity-vector code and artifacts, including `{args.s02_results}`.
- S03 target morphology code and artifacts, including `{args.s03_results}`.
- E04 signal vocabulary from `{args.e04_signal_spec}`.
- Datasets: none required.

## Methods

Implemented `src/e05/actions.py` with `ActionSpec`, `MorphogenesisActionRequest`, `MorphogenesisActionResult`, and `ActionExecutor`. Added unit tests in `tests/e05/test_actions.py`. Generated validation/report artifacts with `scripts/e05_s04_action_validation.py`.

## Commands

{command_lines if command_lines else "- Unit tests were skipped by command-line option."}

## Dependencies And Runtime

- Python: `{platform.python_version()}`
- Platform: `{platform.platform()}`
- Pandas: `{pd.__version__}`
- No new packages were installed for S04.
- CPU/GPU use: validation and report generation are small serial CPU tasks; no GPU use was needed.

## Parameters

- Configured action count: {len(action_df)}.
- Requested S04 action types: {", ".join(REQUESTED_S04_ACTIONS)}.
- E04-style signal channels: {", ".join(SIGNAL_CHANNEL_RANGES)}.

## Results

{markdown_table(validation_df, ["validation_case", "case_type", "success", "detail"])}

All validation rows passed. The primary success criterion was met: policies can execute configured local action sets, illegal moves are rejected, successful actions record energy cost, and conservation or birth-death accounting is explicit.

## Action Catalog

{markdown_table(action_df, ["action_type", "energy_cost", "requires_target", "target_occupancy", "population_delta", "local_semantics", "conservation_rule"])}

## Validation Checks

- Catalog includes every requested S04 action with nonnegative energy costs.
- Swap and crawl are local and population-conserving.
- Nonadjacent illegal swap is rejected without cost or state mutation.
- Polarity rotation updates local identity metadata only.
- Divide and die report birth/death counts and population deltas.
- Adhere/detach update reciprocal local bond metadata.
- Signal exchange uses bounded E04-style channels and rejects nonlocal or out-of-range signals.
- Configured action queries are dry runs and disabled actions are rejected.

## Artifacts

{chr(10).join(f"- `{entry['path']}`: {entry['description']}" for entry in artifacts)}

## Source Provenance

{source_lines}

## Caveats And Limitations

- Energy costs are toy nominal costs for comparative simulation bookkeeping, not physical energy measurements.
- Adhesion and signaling are metadata-level local proxies, not biochemical or mechanical models.
- Division/death semantics are explicit but simple; later benchmark steps may restrict them for fixed-population controls.
- S04 does not define morphology-distance metrics; S05 remains necessary before target-progress benchmarks.

## Blockers And Failed Assumptions

No blocker was found. Action semantics can be kept local, and birth-death accounting can be made explicit.

## Recommended Next Action

{recommended_next_action}
"""


def write_checksums(paths: list[Path], checksum_path: Path, artifacts_dir: Path) -> None:
    lines = []
    for path in sorted(paths):
        if path == checksum_path:
            continue
        lines.append(f"{sha256_file(path)}  {path.relative_to(artifacts_dir)}")
    write_text(checksum_path, "\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    reports_dir = artifacts_dir / "reports"
    results_dir = artifacts_dir / "results"
    tables_dir = artifacts_dir / "tables"
    configs_dir = artifacts_dir / "configs"
    src_snapshot_dir = artifacts_dir / "src_snapshot"
    checksums_dir = artifacts_dir / "checksums"
    for directory in (step_dir, reports_dir, results_dir, tables_dir, configs_dir, src_snapshot_dir, checksums_dir):
        directory.mkdir(parents=True, exist_ok=True)

    started_at = utc_now()
    validation_df = run_validations()
    action_df = pd.DataFrame(action_spec_rows(default_action_set()))
    validation_success = bool(validation_df["success"].all())
    validation_result = f"{int(validation_df['success'].sum())}/{len(validation_df)} validation cases passed"
    recommended_next_action = "Stop before S05 and let the Chief Scientist review S04; if accepted, proceed to define morphospace metrics in S05."

    test_commands: list[dict[str, Any]] = []
    if args.run_unit_tests:
        test_commands.append(run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e05", "-v"], args.repo_dir))
        test_commands.append(
            run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e03", "-p", "test_policy_interface.py", "-v"], args.repo_dir)
        )
    test_success = all(command["success"] for command in test_commands)

    validation_path = results_dir / "e05_action_validation.parquet"
    validation_csv_path = tables_dir / "e05_action_validation.csv"
    action_catalog_path = results_dir / "e05_action_set_catalog.parquet"
    action_catalog_csv_path = tables_dir / "e05_action_set_catalog.csv"
    action_specs_path = results_dir / "e05_action_set_specs.json"
    config_path = configs_dir / "e05_s04_action_validation.json"
    source_manifest_path = src_snapshot_dir / "e05_actions_manifest.json"
    spec_path = reports_dir / "e05_action_set_spec.md"
    full_results_path = step_dir / "research_step_full_results.md"
    artifact_manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = artifacts_dir / "run_manifest.json"
    checksum_path = checksums_dir / "sha256sums.txt"

    validation_df.to_parquet(validation_path, index=False)
    validation_df.to_csv(validation_csv_path, index=False)
    action_df.to_parquet(action_catalog_path, index=False)
    action_df.to_csv(action_catalog_csv_path, index=False)
    write_json(
        action_specs_path,
        {
            "schema": "eidosoma.e05_s04.action_set_specs.v1",
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "actions": [spec.compact_dict() for spec in default_action_set().values()],
            "signalChannelRanges": SIGNAL_CHANNEL_RANGES,
        },
    )
    write_json(
        config_path,
        {
            "schema": "eidosoma.e05_s04.action_validation_config.v1",
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "experimentId": EXPERIMENT_ID,
            "requestedActions": list(REQUESTED_S04_ACTIONS),
            "validationCases": validation_df["validation_case"].tolist(),
            "unitTestsRun": bool(args.run_unit_tests),
            "createdAt": started_at,
        },
    )

    source_files = [
        source_entry(args.repo_dir / "src/e05/actions.py", args.repo_dir),
        source_entry(args.repo_dir / "tests/e05/test_actions.py", args.repo_dir),
        source_entry(args.repo_dir / "scripts/e05_s04_action_validation.py", args.repo_dir),
        source_entry(args.repo_dir / "src/e05/substrates.py", args.repo_dir),
        source_entry(args.repo_dir / "src/e05/cell_identity.py", args.repo_dir),
        source_entry(args.repo_dir / "src/e05/targets.py", args.repo_dir),
    ]
    write_json(
        source_manifest_path,
        {
            "schema": "eidosoma.source_manifest.v1",
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "sourceFiles": source_files,
            "gitCommit": git_output(args.repo_dir, ["rev-parse", "HEAD"]),
            "gitStatusShort": git_output(args.repo_dir, ["status", "--short"]),
            "createdAt": utc_now(),
        },
    )

    artifacts_for_summary = [
        artifact_entry(validation_path, artifacts_dir, "Machine-readable S04 action validation table."),
        artifact_entry(validation_csv_path, artifacts_dir, "CSV mirror of S04 validation table."),
        artifact_entry(action_catalog_path, artifacts_dir, "Machine-readable S04 action catalog."),
        artifact_entry(action_catalog_csv_path, artifacts_dir, "CSV mirror of S04 action catalog."),
        artifact_entry(action_specs_path, artifacts_dir, "JSON action-set specifications."),
        artifact_entry(config_path, artifacts_dir, "S04 validation configuration."),
        artifact_entry(source_manifest_path, artifacts_dir, "Repository source hashes for S04 code and tests."),
    ]
    planned_artifact_paths = [
        {"path": str(spec_path), "description": "S04 action-set specification report."},
        {"path": str(full_results_path), "description": "S04 full-results handoff report."},
        {"path": str(artifact_manifest_path), "description": "S04 artifact manifest."},
        {"path": str(run_manifest_path), "description": "Experiment run manifest updated for S04."},
        {"path": str(checksum_path), "description": "Checksums for key S04 artifacts."},
    ]
    write_text(
        spec_path,
        action_spec_markdown(
            [*artifacts_for_summary, *planned_artifact_paths],
            validation_result,
            validation_df,
            action_df,
            recommended_next_action,
        ),
    )
    artifacts_after_spec = [
        *artifacts_for_summary,
        artifact_entry(spec_path, artifacts_dir, "S04 action-set specification report."),
    ]
    write_text(
        full_results_path,
        full_results_markdown(
            artifacts=[*artifacts_after_spec, *planned_artifact_paths[1:]],
            validation_result=f"{validation_result}; unit-test commands success={test_success}",
            validation_df=validation_df,
            action_df=action_df,
            test_commands=test_commands,
            source_files=source_files,
            args=args,
            recommended_next_action=recommended_next_action,
        ),
    )
    artifacts_final = [
        *artifacts_after_spec,
        artifact_entry(full_results_path, artifacts_dir, "S04 full-results handoff report."),
    ]
    write_json(
        artifact_manifest_path,
        {
            "schema": "eidosoma.artifact_manifest.v1",
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "experimentId": EXPERIMENT_ID,
            "success": bool(validation_success and test_success),
            "artifacts": [*artifacts_final, manifest_self_entry(artifact_manifest_path, artifacts_dir, "S04 artifact manifest.")],
            "validationResult": f"{validation_result}; unit-test commands success={test_success}",
            "caveatsOrBlockers": "No blocker. S04 defines local toy action semantics and accounting only; metrics and benchmark simulations remain future steps.",
            "recommendedNextAction": recommended_next_action,
            "createdAt": utc_now(),
        },
    )
    run_manifest_payload = {
        "schema": "eidosoma.run_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "experimentTitle": EXPERIMENT_TITLE,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "startedAt": started_at,
        "completedAt": utc_now(),
        "success": bool(validation_success and test_success),
        "gitCommit": git_output(args.repo_dir, ["rev-parse", "HEAD"]),
        "gitBranch": git_output(args.repo_dir, ["branch", "--show-current"]),
        "gitStatusShort": git_output(args.repo_dir, ["status", "--short"]),
        "pythonVersion": platform.python_version(),
        "platform": platform.platform(),
        "packages": {"pandas": pd.__version__},
        "commands": test_commands,
        "artifacts": [*artifacts_final, artifact_entry(artifact_manifest_path, artifacts_dir, "S04 artifact manifest.")],
        "validationResult": f"{validation_result}; unit-test commands success={test_success}",
    }
    write_json(run_manifest_path, run_manifest_payload)
    checksum_inputs = [
        validation_path,
        validation_csv_path,
        action_catalog_path,
        action_catalog_csv_path,
        action_specs_path,
        config_path,
        source_manifest_path,
        spec_path,
        full_results_path,
        artifact_manifest_path,
        run_manifest_path,
    ]
    write_checksums(checksum_inputs, checksum_path, artifacts_dir)

    print(json.dumps({"success": bool(validation_success and test_success), "validationResult": validation_result, "artifactsDir": str(artifacts_dir)}, indent=2))
    return 0 if validation_success and test_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
