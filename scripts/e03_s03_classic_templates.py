#!/usr/bin/env python3
"""Execute E03 S03 classic-policy template validation and artifact packaging.

S03 parameterizes Bubble, Insertion, and Selection as DSL-linked records or
templates, validates them against the S01 wrapped classics where the S02 DSL can
represent the behavior, records expected exactness gaps, writes artifacts, and
stops before S04.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True

from morphospace import (  # noqa: E402
    DSLPolicy,
    PolicyEventSimulator,
    bubble_template,
    classic_template_records,
    insertion_template,
    parse_rule_program,
    policy_from_spec,
    selection_template,
)


EXPERIMENT_ID = "E03"
STEP_ID = "S03"
STEP_NUMBER = 3
STATUS = "completed"
OUTCOME_CLASSIFICATION = "supportive"
DEFAULT_ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_command(args: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> dict[str, Any]:
    merged_env = os.environ.copy()
    merged_env["PYTHONDONTWRITEBYTECODE"] = "1"
    if env:
        merged_env.update(env)
    try:
        proc = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            env=merged_env,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return {
            "args": args,
            "returncode": proc.returncode,
            "ok": proc.returncode == 0,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except Exception as exc:  # pragma: no cover - defensive provenance path
        return {
            "args": args,
            "returncode": None,
            "ok": False,
            "stdout": "",
            "stderr": repr(exc),
        }


def get_git_metadata() -> dict[str, Any]:
    commit = run_command(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
    branch = run_command(["git", "branch", "--show-current"], cwd=REPO_ROOT)
    status = run_command(["git", "status", "--short"], cwd=REPO_ROOT)
    remote = run_command(["git", "remote", "-v"], cwd=REPO_ROOT)
    return {
        "commit": commit["stdout"].strip() if commit["ok"] else "unknown",
        "branch": branch["stdout"].strip() if branch["ok"] else "unknown",
        "dirtyStatus": status["stdout"].strip(),
        "remote": remote["stdout"].strip(),
    }


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return json_ready(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compact_json(value: Any) -> str:
    return json.dumps(json_ready(value), sort_keys=True, separators=(",", ":"))


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def clean(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            if not math.isfinite(value):
                return ""
            return f"{value:.4f}".rstrip("0").rstrip(".")
        return str(value).replace("\n", " ").replace("|", "\\|")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(clean(item) for item in row) + " |")
    return "\n".join(lines)


def flatten_template_records() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in classic_template_records():
        row = record.to_dict(include_program=False)
        row["ruleCount"] = len(record.dsl_program.rules)
        row["dslProgramJson"] = record.dsl_program.to_json()
        row["parameterDefaultsJson"] = compact_json(row.pop("parameterDefaults"))
        row["parameterSpaceJson"] = compact_json(row.pop("parameterSpace"))
        row["minimalDslExtensionJson"] = compact_json(row.pop("minimalDslExtension"))
        row["caveatsJson"] = compact_json(row.pop("caveats"))
        rows.append(row)
    return pd.DataFrame(rows)


def forced_step(
    values: list[int],
    policy: Any,
    *,
    forced_cell_id: int,
    forced_direction: int | None = None,
    tie_breaker_seed: int = 2,
) -> tuple[Any, list[int], list[dict[str, Any]]]:
    sim = PolicyEventSimulator(
        values,
        policy,
        scheduler_seed=0,
        tie_breaker_seed=tie_breaker_seed,
        condition_id=f"{STEP_ID}_forced_step",
        implementation="s03_template_validation",
        research_step_id=STEP_ID,
    )
    kwargs = {"forced_cell_id": int(forced_cell_id)}
    if forced_direction is not None:
        kwargs["forced_direction"] = int(forced_direction)
    outcome = sim.step(**kwargs)
    return outcome, sim.current_values(), sim.current_policy_states()


def outcome_summary(outcome: Any, values: list[int]) -> dict[str, Any]:
    return {
        "activated": outcome.activated,
        "actorCellId": outcome.actor_cell_id,
        "actorPositionBefore": outcome.actor_position_before,
        "actorPositionAfter": outcome.actor_position_after,
        "targetPosition": outcome.target_position,
        "swapped": outcome.swapped,
        "comparisonDelta": outcome.comparison_delta,
        "blockedMoveAttempt": outcome.blocked_move_attempt,
        "reason": outcome.reason,
        "values": values,
    }


def same_action_result(left: dict[str, Any], right: dict[str, Any], *, include_comparison: bool = True) -> bool:
    fields = ["actorCellId", "actorPositionBefore", "actorPositionAfter", "targetPosition", "swapped", "values"]
    if include_comparison:
        fields.append("comparisonDelta")
    return all(left[field] == right[field] for field in fields)


def append_s01_pair_row(
    rows: list[dict[str, Any]],
    *,
    condition_id: str,
    algorithm: str,
    classic_policy: str,
    template_policy: DSLPolicy,
    values: list[int],
    forced_cell_id: int,
    forced_direction: int | None = None,
    expected_relationship: str,
    include_comparison: bool = True,
    detail: str,
) -> None:
    classic_outcome, classic_values, classic_states = forced_step(
        values,
        classic_policy,
        forced_cell_id=forced_cell_id,
        forced_direction=forced_direction,
    )
    template_outcome, template_values, template_states = forced_step(
        values,
        template_policy,
        forced_cell_id=forced_cell_id,
        forced_direction=forced_direction,
    )
    classic_summary = outcome_summary(classic_outcome, classic_values)
    template_summary = outcome_summary(template_outcome, template_values)
    matched = same_action_result(classic_summary, template_summary, include_comparison=include_comparison)
    if expected_relationship == "match":
        success = matched
    elif expected_relationship == "partial_match":
        success = same_action_result(classic_summary, template_summary, include_comparison=False)
    elif expected_relationship == "expected_gap":
        success = not matched
    else:
        raise ValueError(f"unknown expected relationship: {expected_relationship}")
    rows.append(
        {
            "validationFamily": "s01_wrapped_classic_alignment",
            "conditionId": condition_id,
            "algorithm": algorithm,
            "policyId": template_policy.policy_id,
            "linkedS01Policy": classic_policy,
            "expectedRelationship": expected_relationship,
            "matchedS01": matched,
            "success": success,
            "validationDetail": detail,
            "classicOutcomeJson": compact_json(classic_summary),
            "templateOutcomeJson": compact_json(template_summary),
            "classicStateJson": compact_json(classic_states),
            "templateStateJson": compact_json(template_states),
        }
    )


def run_template_validations(s01_status_path: Path, s02_status_path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    records = classic_template_records()

    for record in records:
        parsed = parse_rule_program(record.dsl_program.to_json())
        rows.append(
            {
                "validationFamily": "template_serialization",
                "conditionId": f"{record.algorithm}_program_round_trip",
                "algorithm": record.algorithm,
                "policyId": record.policy_id,
                "linkedS01Policy": record.linked_s01_policy_id,
                "expectedRelationship": "round_trip",
                "matchedS01": None,
                "success": parsed.to_dict() == record.dsl_program.to_dict(),
                "validationDetail": "Canonical DSL JSON parse/serialize round-trip preserved the template program.",
            }
        )
        restored = policy_from_spec(DSLPolicy(record.dsl_program).to_spec())
        rows.append(
            {
                "validationFamily": "template_serialization",
                "conditionId": f"{record.algorithm}_policy_spec_restore",
                "algorithm": record.algorithm,
                "policyId": record.policy_id,
                "linkedS01Policy": record.linked_s01_policy_id,
                "expectedRelationship": "policy_spec_restore",
                "matchedS01": None,
                "success": restored.to_spec().to_dict() == DSLPolicy(record.dsl_program).to_spec().to_dict(),
                "validationDetail": "S01 PolicySpec factory restores the DSL template policy.",
            }
        )
        rows.append(
            {
                "validationFamily": "template_metadata",
                "conditionId": f"{record.algorithm}_extension_metadata_present",
                "algorithm": record.algorithm,
                "policyId": record.policy_id,
                "linkedS01Policy": record.linked_s01_policy_id,
                "expectedRelationship": "metadata",
                "matchedS01": None,
                "success": (not record.exact_behavior)
                and bool(record.minimal_dsl_extension)
                and bool(record.validation_scope)
                and bool(record.parameter_space),
                "validationDetail": "Template is searchable and carries explicit exactness-gap metadata.",
            }
        )

    for algorithm, program in [
        ("bubble", bubble_template()),
        ("insertion", insertion_template()),
        ("selection", selection_template()),
    ]:
        result = PolicyEventSimulator(
            [5, 1, 4, 2, 3],
            DSLPolicy(program),
            scheduler_seed=10,
            tie_breaker_seed=20,
            condition_id=f"{STEP_ID}_{algorithm}_toy_execution",
            implementation="s03_template_validation",
            research_step_id=STEP_ID,
        ).run(max_activations=100000)
        rows.append(
            {
                "validationFamily": "toy_execution",
                "conditionId": f"{algorithm}_template_sorts_unique_five_cell_array",
                "algorithm": algorithm,
                "policyId": program.policy_id,
                "linkedS01Policy": f"classic_{algorithm}",
                "expectedRelationship": "executable_proxy",
                "matchedS01": None,
                "success": result.completed and result.final_values == [1, 2, 3, 4, 5],
                "validationDetail": "Template executed through the S01 simulator boundary and sorted a five-cell unique array.",
                "completed": result.completed,
                "stopReason": result.stop_reason,
                "swapCount": result.swap_count,
                "comparisonCount": result.comparison_count,
                "finalValuesJson": compact_json(result.final_values),
            }
        )

    append_s01_pair_row(
        rows,
        condition_id="bubble_forced_right_single_neighbor_match",
        algorithm="bubble",
        classic_policy="bubble",
        template_policy=DSLPolicy(bubble_template()),
        values=[2, 1],
        forced_cell_id=0,
        forced_direction=1,
        expected_relationship="match",
        detail="Forced right Bubble inversion matches the DSL local-inversion template on action, target, comparison, and values.",
    )
    append_s01_pair_row(
        rows,
        condition_id="bubble_forced_left_single_neighbor_match",
        algorithm="bubble",
        classic_policy="bubble",
        template_policy=DSLPolicy(bubble_template()),
        values=[2, 1],
        forced_cell_id=1,
        forced_direction=-1,
        expected_relationship="match",
        detail="Forced left Bubble inversion matches the DSL local-inversion template on action, target, comparison, and values.",
    )
    append_s01_pair_row(
        rows,
        condition_id="bubble_two_sided_random_direction_expected_gap",
        algorithm="bubble",
        classic_policy="bubble",
        template_policy=DSLPolicy(bubble_template()),
        values=[3, 2, 1],
        forced_cell_id=1,
        expected_relationship="expected_gap",
        detail="With both neighbors inverted, S01 Bubble samples direction first while the S03 template uses left-first priority.",
    )
    append_s01_pair_row(
        rows,
        condition_id="insertion_enabled_left_swap_match",
        algorithm="insertion",
        classic_policy="insertion",
        template_policy=DSLPolicy(insertion_template()),
        values=[2, 1],
        forced_cell_id=1,
        expected_relationship="match",
        detail="When the sorted-prefix guard is trivially enabled, the adjacent-left Insertion template matches S01 action and values.",
    )
    append_s01_pair_row(
        rows,
        condition_id="insertion_unsorted_prefix_expected_gap",
        algorithm="insertion",
        classic_policy="insertion",
        template_policy=DSLPolicy(insertion_template()),
        values=[3, 2, 1],
        forced_cell_id=2,
        expected_relationship="expected_gap",
        detail="S01 waits because the left prefix is unsorted; S02 DSL lacks the prefix_sorted_left guard and swaps the immediate inversion.",
    )
    append_s01_pair_row(
        rows,
        condition_id="selection_min_to_initial_target_partial_match",
        algorithm="selection",
        classic_policy="selection",
        template_policy=DSLPolicy(selection_template()),
        values=[3, 1, 2],
        forced_cell_id=1,
        expected_relationship="partial_match",
        include_comparison=False,
        detail="The value-rank Selection proxy matches a simple min-to-target swap at action/result level but not comparison accounting.",
    )
    append_s01_pair_row(
        rows,
        condition_id="selection_actor_already_at_ideal_expected_gap",
        algorithm="selection",
        classic_policy="selection",
        template_policy=DSLPolicy(selection_template()),
        values=[3, 1, 2],
        forced_cell_id=0,
        expected_relationship="expected_gap",
        detail="S01 Selection waits when the actor is at its ideal target; the value-rank proxy retargets by actor value and swaps.",
    )

    for step_id, path in [("S01", s01_status_path), ("S02", s02_status_path)]:
        status = read_json(path) if path.exists() else {}
        rows.append(
            {
                "validationFamily": "upstream_boundary",
                "conditionId": f"{step_id.lower()}_completed_anchor",
                "algorithm": step_id,
                "policyId": step_id,
                "linkedS01Policy": None,
                "expectedRelationship": "upstream_status",
                "matchedS01": None,
                "success": bool(status.get("success")),
                "validationDetail": f"{step_id} status confirms required upstream boundary completed successfully.",
                "logPath": str(path),
            }
        )

    return pd.DataFrame(rows)


def run_repo_unit_tests(step_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    cmd = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"]
    result = run_command(cmd, cwd=REPO_ROOT)
    log_path = step_dir / "repo_unit_test_log.txt"
    log_path.write_text(
        "$ " + " ".join(cmd) + "\n\nSTDOUT\n" + result["stdout"] + "\n\nSTDERR\n" + result["stderr"],
        encoding="utf-8",
    )
    row = {
        "validationFamily": "repo_unit_tests",
        "conditionId": "repo_unit_tests",
        "algorithm": "all",
        "policyId": "all",
        "linkedS01Policy": None,
        "expectedRelationship": "repo_tests",
        "matchedS01": None,
        "success": result["ok"],
        "validationDetail": f"Repository unittest discovery return code {result['returncode']}.",
        "logPath": str(log_path),
    }
    payload = {
        "command": cmd,
        "returnCode": result["returncode"],
        "success": result["ok"],
        "logPath": str(log_path),
    }
    return row, payload


def validation_counts(validation_df: pd.DataFrame) -> pd.DataFrame:
    return (
        validation_df.groupby("validationFamily", dropna=False)["success"]
        .agg(total="count", passed="sum")
        .reset_index()
        .assign(failed=lambda df: df["total"] - df["passed"])
    )


def write_validation_plot(validation_df: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = validation_counts(validation_df)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(counts["validationFamily"], counts["passed"], color="#2b8cbe", label="passed")
    ax.barh(
        counts["validationFamily"],
        counts["failed"],
        left=counts["passed"],
        color="#d95f0e",
        label="failed",
    )
    ax.set_xlabel("Validation checks")
    ax.set_title("E03 S03 classic-template validations")
    ax.legend(loc="lower right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def artifact_preview(artifacts_written: list[str], limit: int = 20) -> str:
    preview = "\n".join(f"- `{path}`" for path in artifacts_written[:limit])
    if len(artifacts_written) > limit:
        preview += f"\n- ... {len(artifacts_written) - limit} additional artifact path(s) in status.json"
    return preview


def render_extension_report(
    template_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    artifacts_written: list[str],
    validation_result: str,
    caveats: list[str],
    recommended_next_action: str,
) -> str:
    rows = []
    for _, row in template_df.iterrows():
        rows.append(
            [
                row["algorithm"],
                row["policyId"],
                row["exactBehavior"],
                row["representationType"],
                row["minimalDslExtensionJson"],
            ]
        )
    s01_rows = validation_df[
        validation_df["validationFamily"].eq("s01_wrapped_classic_alignment")
    ][["conditionId", "algorithm", "expectedRelationship", "matchedS01", "success", "validationDetail"]].values.tolist()
    return f"""# S03 Classic Template Extension Report

- Research step ID: {STEP_ID}
- Completion status: {STATUS}
- Artifacts written:
{artifact_preview(artifacts_written)}
- Validation result: {validation_result}
- Caveats or blockers: {"; ".join(caveats)}
- Recommended next action: {recommended_next_action}

## Frozen Question

Bubble, Insertion, and Selection can be represented as points or small regions in the rule space rather than isolated classes.

## Template Summary

{markdown_table(["Algorithm", "Policy ID", "Exact behavior", "Representation", "Minimal DSL extension"], rows)}

## S01 Alignment Checks

{markdown_table(["Condition", "Algorithm", "Expected relationship", "Matched S01", "Success", "Detail"], s01_rows)}

## Exactness Boundary

The S03 records satisfy the searchable-region goal, but no record claims full trace-level equivalence to S01. Bubble needs direction sampling before predicate evaluation, Insertion needs a sorted-left-prefix predicate with Frozen Cell reset semantics, and Selection needs dynamic target-state predicates plus legal state-progress semantics. These are intentionally documented as S04-or-later DSL design inputs rather than added during S03.
"""


def render_validation_report(validation_df: pd.DataFrame, artifacts_written: list[str], caveats: list[str]) -> str:
    validation_passed = bool(validation_df["success"].all())
    validation_result = "passed" if validation_passed else "failed"
    failed = validation_df[~validation_df["success"].astype(bool)]
    failure_text = "None"
    if not failed.empty:
        failure_text = markdown_table(
            ["Family", "Condition", "Algorithm", "Policy", "Detail"],
            failed[["validationFamily", "conditionId", "algorithm", "policyId", "validationDetail"]].head(20).values.tolist(),
        )
    rows = validation_counts(validation_df)[["validationFamily", "passed", "total", "failed"]].values.tolist()
    return f"""# S03 Validation Report

- Research step ID: {STEP_ID}
- Completion status: {STATUS}
- Artifacts written:
{artifact_preview(artifacts_written)}
- Validation result: {validation_result}; {int(validation_df["success"].sum())} of {len(validation_df)} checks passed.
- Caveats or blockers: {"; ".join(caveats)}
- Recommended next action: stop before S04 for Chief Scientist review; if approved, define the competence-vector schema in S04.

## Validation Counts

{markdown_table(["Validation family", "Passed", "Total", "Failed"], rows)}

## Failed Checks

{failure_text}
"""


def render_summary(validation_df: pd.DataFrame, artifacts_written: list[str], caveats: list[str]) -> str:
    validation_passed = bool(validation_df["success"].all())
    validation_result = "passed" if validation_passed else "failed"
    outcome = OUTCOME_CLASSIFICATION if validation_passed else "constraining/contradictory"
    return f"""# S03 Summary

- Research step ID: {STEP_ID}
- Completion status: {STATUS}
- Artifacts written: `{artifacts_written[0]}` and {len(artifacts_written) - 1} additional files listed in `status.json` and `artifact_manifest.json`.
- Validation result: {validation_result}; {int(validation_df["success"].sum())} of {len(validation_df)} checks passed.
- Outcome classification: {outcome}
- Caveats or blockers: {"; ".join(caveats)}
- Lay summary: S03 makes Bubble, Insertion, and Selection searchable in the S02 rule space by writing DSL-linked template records. Bubble and Insertion match S01 on simple forced adjacent-swap cases, and Selection has a validated target-seeking proxy with a simple action-level alignment case. The validations also preserve expected failures where S02 lacks exact classic semantics.
- Recommended next action: stop before S04 for Chief Scientist review. If approved, S04 should define the competence-vector schema while carrying forward the exactness gaps as documented DSL-extension candidates.
"""


def collect_artifacts(paths: list[Path]) -> list[dict[str, Any]]:
    artifacts = []
    seen: set[Path] = set()
    for path in paths:
        if path.exists() and path.is_file() and path.resolve() not in seen:
            seen.add(path.resolve())
            artifacts.append({"path": str(path), "sizeBytes": path.stat().st_size, "sha256": sha256_path(path)})
    return sorted(artifacts, key=lambda item: item["path"])


def copy_code_artifacts(step_dir: Path) -> list[Path]:
    code_dir = step_dir / "code"
    copied: list[Path] = []
    package_dst = code_dir / "morphospace"
    if package_dst.exists():
        shutil.rmtree(package_dst)
    shutil.copytree(REPO_ROOT / "morphospace", package_dst, ignore=shutil.ignore_patterns("__pycache__"))
    copied.extend(sorted(path for path in package_dst.rglob("*.py")))

    script_dst = code_dir / "scripts" / "e03_s03_classic_templates.py"
    script_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO_ROOT / "scripts" / "e03_s03_classic_templates.py", script_dst)
    copied.append(script_dst)

    test_dst = code_dir / "tests" / "test_e03_classic_templates.py"
    test_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO_ROOT / "tests" / "test_e03_classic_templates.py", test_dst)
    copied.append(test_dst)
    return copied


def write_run_manifest(provenance_dir: Path, artifacts_written: list[str], validation_passed: bool) -> Path:
    provenance_dir.mkdir(parents=True, exist_ok=True)
    path = provenance_dir / "run_manifest.json"
    manifest = read_json(path) if path.exists() else {"schema": "eidosoma.run_manifest.v1", "experimentId": EXPERIMENT_ID}
    manifest.update(
        {
            "experimentId": EXPERIMENT_ID,
            "lastResearchStepId": STEP_ID,
            "lastStepNumber": STEP_NUMBER,
            "updatedAt": utc_now(),
            "git": get_git_metadata(),
            "platform": platform.platform(),
            "python": sys.version,
        }
    )
    manifest.setdefault("researchSteps", {})
    manifest["researchSteps"][STEP_ID] = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "status": STATUS,
        "success": validation_passed,
        "artifactsWritten": artifacts_written,
        "validationResult": "passed" if validation_passed else "failed",
        "completedAt": utc_now(),
        "outcomeClassification": OUTCOME_CLASSIFICATION if validation_passed else "constraining/contradictory",
    }
    write_json(path, manifest)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    args = parser.parse_args()

    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    provenance_dir = artifacts_dir / "provenance"
    s01_status_path = artifacts_dir / "research_steps" / "S01" / "status.json"
    s02_status_path = artifacts_dir / "research_steps" / "S02" / "status.json"
    step_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    started_at = utc_now()
    template_records = classic_template_records()
    template_df = flatten_template_records()
    validation_df = run_template_validations(s01_status_path, s02_status_path)
    repo_row, repo_command = run_repo_unit_tests(step_dir)
    validation_df = pd.concat([validation_df, pd.DataFrame([repo_row])], ignore_index=True)

    templates_json = step_dir / "classic_policy_templates.json"
    templates_pretty = step_dir / "classic_policy_templates_pretty.txt"
    templates_csv = step_dir / "classic_policy_templates.csv"
    templates_parquet = step_dir / "classic_policy_templates.parquet"
    templates_results_csv = results_dir / "e03_s03_classic_policy_templates.csv"
    templates_results_parquet = results_dir / "e03_s03_classic_policy_templates.parquet"
    validation_csv = step_dir / "classic_template_validation.csv"
    validation_parquet = step_dir / "classic_template_validation.parquet"
    validation_results_csv = results_dir / "e03_s03_classic_template_validation.csv"
    validation_results_parquet = results_dir / "e03_s03_classic_template_validation.parquet"
    validation_plot = step_dir / "classic_template_validation.png"
    validation_results_plot = results_dir / "e03_s03_classic_template_validation.png"

    write_json(
        templates_json,
        {
            "schema": "eidosoma.e03.s03.classic_policy_templates.v1",
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "templates": [record.to_dict(include_program=True) for record in template_records],
        },
    )
    templates_pretty.write_text(
        "\n\n".join(record.dsl_program.pretty() for record in template_records) + "\n",
        encoding="utf-8",
    )
    template_df.to_csv(templates_csv, index=False)
    template_df.to_parquet(templates_parquet, index=False)
    template_df.to_csv(templates_results_csv, index=False)
    template_df.to_parquet(templates_results_parquet, index=False)

    validation_df.to_csv(validation_csv, index=False)
    validation_df.to_parquet(validation_parquet, index=False)
    validation_df.to_csv(validation_results_csv, index=False)
    validation_df.to_parquet(validation_results_parquet, index=False)
    write_validation_plot(validation_df, validation_plot)
    shutil.copy2(validation_plot, validation_results_plot)

    copied_code = copy_code_artifacts(step_dir)
    extension_report_path = step_dir / "classic_template_extension_report.md"
    validation_report_path = step_dir / "validation_report.md"
    summary_path = step_dir / "summary.md"
    run_manifest_path = provenance_dir / "run_manifest.json"
    status_path = step_dir / "status.json"
    artifact_manifest_path = step_dir / "artifact_manifest.json"

    artifacts_written_paths = [
        templates_json,
        templates_pretty,
        templates_csv,
        templates_parquet,
        templates_results_csv,
        templates_results_parquet,
        validation_csv,
        validation_parquet,
        validation_results_csv,
        validation_results_parquet,
        validation_plot,
        validation_results_plot,
        step_dir / "repo_unit_test_log.txt",
        *copied_code,
        extension_report_path,
        validation_report_path,
        summary_path,
        run_manifest_path,
        status_path,
        artifact_manifest_path,
    ]
    artifacts_written = [str(path) for path in artifacts_written_paths]
    validation_passed = bool(validation_df["success"].all())
    validation_result = (
        "passed: template serialization, executable toy policies, S01 wrapper alignment cases, expected exactness-gap checks, upstream anchors, and repository tests passed"
        if validation_passed
        else "failed: one or more S03 classic-template validations failed"
    )
    caveats = [
        "No S03 template claims full trace-level equality to S01 because S02 lacks direction-choice, prefix-guard, and Selection scan-state primitives.",
        "Bubble and Insertion exact matches are limited to forced adjacent-swap cases where missing DSL primitives are not exercised.",
        "Selection is represented by an executable value-rank target proxy plus documented extensions for exact scan-and-advance behavior.",
        "No competence-vector schema, generated policy corpus, GPU simulator, or S04 artifact was started.",
    ]
    recommended_next_action = "Stop before S04 for Chief Scientist review; if approved, define the competence-vector schema in S04."

    extension_report_path.write_text(
        render_extension_report(
            template_df,
            validation_df,
            artifacts_written,
            validation_result,
            caveats,
            recommended_next_action,
        ),
        encoding="utf-8",
    )
    validation_report_path.write_text(render_validation_report(validation_df, artifacts_written, caveats), encoding="utf-8")
    summary_path.write_text(render_summary(validation_df, artifacts_written, caveats), encoding="utf-8")
    write_run_manifest(provenance_dir, artifacts_written, validation_passed)

    status = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": validation_passed,
        "status": STATUS,
        "artifactsWritten": artifacts_written,
        "validationResult": validation_result,
        "caveatsOrBlockers": caveats,
        "recommendedNextAction": recommended_next_action,
        "outcomeClassification": OUTCOME_CLASSIFICATION if validation_passed else "constraining/contradictory",
        "startedAt": started_at,
        "completedAt": utc_now(),
        "workerCount": 1,
        "threadEnvironment": {
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        },
        "repoUnitTestCommand": repo_command,
        "s01StatusPath": str(s01_status_path),
        "s02StatusPath": str(s02_status_path),
        "git": get_git_metadata(),
    }
    write_json(status_path, status)

    artifact_manifest = {
        "schema": "eidosoma.e03.s03.artifact_manifest.v1",
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "generatedAt": utc_now(),
        "artifacts": collect_artifacts([Path(path) for path in artifacts_written]),
    }
    write_json(artifact_manifest_path, artifact_manifest)
    artifact_manifest["artifacts"] = collect_artifacts([Path(path) for path in artifacts_written])
    write_json(artifact_manifest_path, artifact_manifest)

    print(json.dumps({"success": validation_passed, "statusPath": str(status_path)}, indent=2))
    return 0 if validation_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

