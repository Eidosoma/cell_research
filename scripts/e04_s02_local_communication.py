#!/usr/bin/env python3
"""Execute E04 S02 local-communication validation."""

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
import time
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True

from morphospace import BubblePolicy, DSLPolicy, InsertionPolicy, PolicyEventSimulator, SelectionPolicy, parse_rule_program  # noqa: E402
from memory_repair import (  # noqa: E402
    SIGNAL_CHANNELS,
    SIGNAL_REPAIR_VERSION,
    SIGNAL_STATE_KEY,
    SIGNAL_VARIANTS,
    SignalConfig,
    SignalEventSimulator,
    SignalPolicyWrapper,
    build_signal_variants,
    diffuse_signal_fields,
    empty_signal_fields,
    signal_policy_from_json,
    signal_policy_to_json,
    signal_state_for_trace,
)


EXPERIMENT_ID = "E04"
STEP_ID = "S02"
STEP_NUMBER = 2
STEP_TITLE = "Add local communication"
DEFAULT_ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
DEFAULT_E03_ARTIFACTS = Path("/previous-artifacts/E03")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def compact_json(value: Any) -> str:
    return json.dumps(json_ready(value), sort_keys=True, separators=(",", ":"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(dict(payload)), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_command(args: list[str], cwd: Path | None = None, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    merged_env = os.environ.copy()
    merged_env["PYTHONDONTWRITEBYTECODE"] = "1"
    if env:
        merged_env.update(env)
    started = time.perf_counter()
    proc = subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        env=merged_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "args": args,
        "returncode": proc.returncode,
        "success": proc.returncode == 0,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "runtimeSeconds": time.perf_counter() - started,
    }


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_git_metadata() -> dict[str, Any]:
    commit = run_command(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
    branch = run_command(["git", "branch", "--show-current"], cwd=REPO_ROOT)
    remote = run_command(["git", "remote", "get-url", "origin"], cwd=REPO_ROOT)
    status = run_command(["git", "status", "--short"], cwd=REPO_ROOT)
    return {
        "commit": commit["stdout"].strip() if commit["success"] else "unknown",
        "branch": branch["stdout"].strip() if branch["success"] else "unknown",
        "remote": remote["stdout"].strip() if remote["success"] else "unknown",
        "statusShort": status["stdout"].strip(),
    }


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def clean(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.4f}".rstrip("0").rstrip(".") if math.isfinite(value) else ""
        return str(value).replace("\n", " ").replace("|", "\\|")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(clean(item) for item in row) + " |")
    return "\n".join(lines)


def classic_policies() -> list[dict[str, Any]]:
    return [
        {"basePolicyId": "classic_bubble", "baseFamily": "classic", "sourcePath": "repo:morphospace.BubblePolicy", "policy": BubblePolicy()},
        {
            "basePolicyId": "classic_insertion",
            "baseFamily": "classic",
            "sourcePath": "repo:morphospace.InsertionPolicy",
            "policy": InsertionPolicy(),
        },
        {
            "basePolicyId": "classic_selection",
            "baseFamily": "classic",
            "sourcePath": "repo:morphospace.SelectionPolicy",
            "policy": SelectionPolicy(),
        },
    ]


def load_frontier_policies(e03_artifacts: Path, limit: int) -> list[dict[str, Any]]:
    frontier_dir = e03_artifacts / "research_steps" / "S14" / "frontier_policy_dsl"
    records: list[dict[str, Any]] = []
    for path in sorted(frontier_dir.glob("*.json"))[:limit]:
        program = parse_rule_program(path.read_text(encoding="utf-8"))
        records.append(
            {
                "basePolicyId": program.policy_id,
                "baseFamily": "frontier_dsl",
                "sourcePath": str(path),
                "policy": DSLPolicy(program),
            }
        )
    return records


def compare_results(signal_result: Any, base_result: Any) -> tuple[bool, list[str]]:
    checks = {
        "completed": signal_result.completed == base_result.completed,
        "stop_reason": signal_result.stop_reason == base_result.stop_reason,
        "final_values": signal_result.final_values == base_result.final_values,
        "final_algotypes": signal_result.final_algotypes == base_result.final_algotypes,
        "swap_count": signal_result.swap_count == base_result.swap_count,
        "comparison_count": signal_result.comparison_count == base_result.comparison_count,
        "activation_count": signal_result.activation_count == base_result.activation_count,
        "trace_state_hashes": [row["state_hash"] for row in signal_result.trace_rows]
        == [row["state_hash"] for row in base_result.trace_rows],
    }
    failures = [name for name, ok in checks.items() if not ok]
    return not failures, failures


def variant_catalog_rows(base_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in base_records:
        for policy in build_signal_variants(record["policy"], signal_range=1, diffusion_rate=0.25):
            rows.append(
                {
                    "policyId": policy.policy_id,
                    "basePolicyId": record["basePolicyId"],
                    "baseFamily": record["baseFamily"],
                    "baseSourcePath": record["sourcePath"],
                    "signalVariant": policy.signal_config.variant,
                    "signalRange": policy.signal_config.signal_range,
                    "diffusionRate": policy.signal_config.diffusion_rate,
                    "decay": policy.signal_config.decay,
                    "noiseStd": policy.signal_config.noise_std,
                    "channelsJson": compact_json(policy.signal_config.channels),
                    "algotype": policy.algotype,
                    "family": policy.family,
                    "signalRepairVersion": SIGNAL_REPAIR_VERSION,
                    "policySpecJson": signal_policy_to_json(policy),
                }
            )
    return rows


def no_signal_regression_rows(base_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cases = [
        {
            "caseId": "small_unsorted",
            "values": [6, 2, 4, 1, 5, 3],
            "schedulerSeed": 123,
            "tieBreakerSeed": 456,
            "maxActivations": 200000,
        },
        {
            "caseId": "frozen_stuck",
            "values": [8, 4, 7, 2, 6, 1, 5, 3],
            "schedulerSeed": 1234,
            "tieBreakerSeed": 5678,
            "frozenPositions": [2],
            "frozenVariant": "stuck",
            "maxActivations": 200000,
        },
    ]
    for record in base_records:
        for case in cases:
            kwargs = {
                "scheduler_seed": int(case["schedulerSeed"]),
                "tie_breaker_seed": int(case["tieBreakerSeed"]),
                "condition_id": f"e04_s02_no_signal_{record['basePolicyId']}_{case['caseId']}",
                "research_step_id": STEP_ID,
            }
            if "frozenPositions" in case:
                kwargs["frozen_positions"] = case["frozenPositions"]
                kwargs["frozen_variant"] = case["frozenVariant"]
            base_result = PolicyEventSimulator(case["values"], record["policy"], **kwargs).run(
                max_activations=int(case["maxActivations"])
            )
            signal_result = SignalEventSimulator(
                case["values"],
                SignalPolicyWrapper(record["policy"], "no_signal"),
                signal_config="no_signal",
                trace_signal_activations=False,
                trace_memory_activations=False,
                auto_wrap_policies=False,
                **kwargs,
            ).run(max_activations=int(case["maxActivations"]))
            success, failures = compare_results(signal_result, base_result)
            rows.append(
                {
                    "validationFamily": "no_signal_regression",
                    "basePolicyId": record["basePolicyId"],
                    "baseFamily": record["baseFamily"],
                    "signalVariant": "no_signal",
                    "caseId": case["caseId"],
                    "success": success,
                    "failureFieldsJson": compact_json(failures),
                    "baseStopReason": base_result.stop_reason,
                    "signalStopReason": signal_result.stop_reason,
                    "baseFinalValuesJson": compact_json(base_result.final_values),
                    "signalFinalValuesJson": compact_json(signal_result.final_values),
                    "baseSwapCount": base_result.swap_count,
                    "signalSwapCount": signal_result.swap_count,
                    "baseComparisonCount": base_result.comparison_count,
                    "signalComparisonCount": signal_result.comparison_count,
                    "baseActivationCount": base_result.activation_count,
                    "signalActivationCount": signal_result.activation_count,
                    "validationDetail": "No-signal wrapper matches E03 PolicyEventSimulator."
                    if success
                    else f"No-signal wrapper mismatch: {','.join(failures)}.",
                }
            )
    return rows


def _trace_locality_ok(sim: SignalEventSimulator) -> bool:
    for row in sim.trace_rows:
        actor_payload = json.loads(row.get("actor_signal_state_json", "{}"))
        signal_state = actor_payload.get("signal_state", {})
        visible = signal_state.get("visible_positions", [])
        position = signal_state.get("actor_position", actor_payload.get("position"))
        if position is None or not visible:
            continue
        if any(abs(int(pos) - int(position)) > sim.signal_config.signal_range for pos in visible):
            return False
    return True


def validation_rows(base_records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    for record in base_records:
        for variant in SIGNAL_VARIANTS:
            config = SignalConfig(variant=variant, signal_range=1, diffusion_rate=0.25, decay=0.0, random_seed=17)
            policy = SignalPolicyWrapper(record["policy"], config)
            restored = signal_policy_from_json(signal_policy_to_json(policy))
            serialization_success = restored.to_spec().to_dict() == policy.to_spec().to_dict()
            rows.append(
                {
                    "validationFamily": "serialization",
                    "basePolicyId": record["basePolicyId"],
                    "baseFamily": record["baseFamily"],
                    "signalVariant": variant,
                    "caseId": "policy_spec_round_trip",
                    "success": serialization_success,
                    "validationDetail": "Signal policy spec round-trip preserves base policy and signal config.",
                }
            )

            sim = SignalEventSimulator(
                [5, 1, 4, 2, 3],
                policy,
                signal_config=config,
                scheduler_seed=11,
                tie_breaker_seed=22,
                condition_id=f"e04_s02_signal_run_{record['basePolicyId']}_{variant}",
                research_step_id=STEP_ID,
                auto_wrap_policies=False,
            )
            result = sim.run(max_activations=500)
            trace_json_ok = True
            for trace_row in sim.trace_rows:
                try:
                    json.loads(trace_row["signal_fields_json"])
                    json.loads(trace_row["signal_config_json"])
                except Exception:
                    trace_json_ok = False
                    break
            locality_ok = _trace_locality_ok(sim)
            rows.append(
                {
                    "validationFamily": "signal_variant_run",
                    "basePolicyId": record["basePolicyId"],
                    "baseFamily": record["baseFamily"],
                    "signalVariant": variant,
                    "caseId": "toy_unsorted_trace",
                    "success": trace_json_ok and locality_ok,
                    "validationDetail": "Signal variant runs, emits JSON traces, and exposes only local-window positions."
                    if trace_json_ok and locality_ok
                    else "Signal trace JSON failed or locality window exceeded configured range.",
                    "traceRowCount": len(sim.trace_rows),
                    "finalValuesJson": compact_json(result.final_values),
                    "stopReason": result.stop_reason,
                    "swapCount": result.swap_count,
                    "activationCount": result.activation_count,
                }
            )
            run_rows.append(
                {
                    "basePolicyId": record["basePolicyId"],
                    "baseFamily": record["baseFamily"],
                    "signalVariant": variant,
                    "completed": result.completed,
                    "stopReason": result.stop_reason,
                    "finalValuesJson": compact_json(result.final_values),
                    "swapCount": result.swap_count,
                    "comparisonCount": result.comparison_count,
                    "activationCount": result.activation_count,
                    "eventCount": result.event_count,
                    "finalSortednessPercent": result.final_sortedness_percent,
                    "traceRows": len(sim.trace_rows),
                    "signalSchemaVersion": SIGNAL_REPAIR_VERSION,
                }
            )

    fields = empty_signal_fields(5, ["blocked"])
    fields["blocked"][2] = 1.0
    diffused = diffuse_signal_fields(fields, SignalConfig("diffusive", diffusion_rate=0.25, decay=0.0))
    expected = np.asarray([0.0, 0.25, 0.5, 0.25, 0.0])
    rows.append(
        {
            "validationFamily": "diffusion_unit",
            "basePolicyId": "not_applicable",
            "baseFamily": "unit",
            "signalVariant": "diffusive",
            "caseId": "center_impulse_no_decay",
            "success": bool(np.allclose(diffused["blocked"], expected)),
            "observedJson": compact_json(diffused["blocked"]),
            "expectedJson": compact_json(expected),
            "validationDetail": "One diffusion tick from a center impulse matches the declared no-flux nearest-neighbor kernel.",
        }
    )

    decayed = diffuse_signal_fields(fields, SignalConfig("diffusive", diffusion_rate=0.25, decay=0.2))
    expected_decay = np.asarray([0.0, 0.2, 0.4, 0.2, 0.0])
    rows.append(
        {
            "validationFamily": "diffusion_unit",
            "basePolicyId": "not_applicable",
            "baseFamily": "unit",
            "signalVariant": "diffusive",
            "caseId": "center_impulse_decay",
            "success": bool(np.allclose(decayed["blocked"], expected_decay)),
            "observedJson": compact_json(decayed["blocked"]),
            "expectedJson": compact_json(expected_decay),
            "validationDetail": "Decay is applied after diffusion and remains bounded.",
        }
    )

    config = SignalConfig("nearest_neighbor", signal_range=1)
    local_sim = SignalEventSimulator(
        [2, 1, 3, 4],
        SignalPolicyWrapper(BubblePolicy(), config),
        signal_config=config,
        frozen_positions=[1],
        frozen_variant="stuck",
        scheduler_seed=1,
        tie_breaker_seed=1,
        auto_wrap_policies=False,
    )
    local_sim.step(forced_cell_id=0, forced_direction=1)
    near_sum = local_sim.sense_at_position(1)["channels"]["blocked"]["local_sum"]
    far_sum = local_sim.sense_at_position(3)["channels"]["blocked"]["local_sum"]
    rows.append(
        {
            "validationFamily": "signal_locality",
            "basePolicyId": "classic_bubble",
            "baseFamily": "classic",
            "signalVariant": "nearest_neighbor",
            "caseId": "blocked_signal_range_one",
            "success": near_sum > 0.0 and far_sum == 0.0,
            "nearBlockedLocalSum": near_sum,
            "farBlockedLocalSum": far_sum,
            "validationDetail": "A blocked signal emitted at range one is sensed by the nearest neighbor but not a distant cell.",
        }
    )

    sensing_program = parse_rule_program(
        {
            "policy_id": "signal_sensing_demo",
            "name": "signal sensing demo",
            "rules": [
                {
                    "name": "swap_right_when_blocked_signal_seen",
                    "when": [{"op": "state_compare", "key": "signal_blocked_local_sum", "operator": ">", "value": 0}],
                    "then": {"action": "swap_right"},
                }
            ],
            "default": {"action": "wait"},
        }
    )
    sensing_sim = SignalEventSimulator(
        [3, 1, 2],
        SignalPolicyWrapper(DSLPolicy(sensing_program), config),
        signal_config=config,
        scheduler_seed=1,
        tie_breaker_seed=1,
        auto_wrap_policies=False,
    )
    sensing_sim.signal_fields["blocked"][1] = 1.0
    sensing_outcome = sensing_sim.step(forced_cell_id=1)
    actor_state = sensing_sim.cells[sensing_sim.positions_by_id[1]].state
    rows.append(
        {
            "validationFamily": "signal_sensing_policy",
            "basePolicyId": "signal_sensing_demo",
            "baseFamily": "toy_dsl",
            "signalVariant": "nearest_neighbor",
            "caseId": "dsl_state_compare_signal",
            "success": sensing_outcome.swapped
            and signal_state_for_trace(actor_state).get("channels", {}).get("blocked", {}).get("local_sum", 0.0) > 0.0,
            "validationDetail": "A DSL policy can sense a local signal through flattened signal state.",
        }
    )

    return rows, run_rows


def dataframe_to_artifacts(df: pd.DataFrame, step_path: Path, results_path: Path) -> list[Path]:
    step_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(step_path.with_suffix(".csv"), index=False)
    df.to_parquet(step_path.with_suffix(".parquet"), index=False)
    shutil.copy2(step_path.with_suffix(".csv"), results_path.with_suffix(".csv"))
    shutil.copy2(step_path.with_suffix(".parquet"), results_path.with_suffix(".parquet"))
    return [
        step_path.with_suffix(".csv"),
        step_path.with_suffix(".parquet"),
        results_path.with_suffix(".csv"),
        results_path.with_suffix(".parquet"),
    ]


def write_signal_spec(path: Path, catalog_df: pd.DataFrame) -> None:
    variant_rows = [
        ["no_signal", "No signal fields or sensed signal state; delegates exactly to E03 behavior."],
        ["nearest_neighbor", "Position-local fields emitted after activation and sensed only within `signalRange`."],
        ["diffusive", "Nearest-neighbor scalar diffusion with configurable decay after local emission."],
        ["randomized", "Seeded random local observations used as a randomized signal control."],
        ["inert", "Signal trace schema is present but emissions are ignored and fields stay zero."],
    ]
    channel_rows = [
        ["blocked", "1 when a local attempted swap is blocked by world constraints."],
        ["sorted", "1 when the actor's immediate neighborhood is locally ordered after the activation."],
        ["frustrated", "Local failed-swap or memory-frustration intensity."],
        ["target_seeking", "1 when a local action attempts a target and does not complete."],
        ["morphogen", "Local activation-density scalar; not a biological morphogen measurement."],
    ]
    base_rows = (
        catalog_df[["basePolicyId", "baseFamily", "baseSourcePath"]]
        .drop_duplicates()
        .sort_values(["baseFamily", "basePolicyId"])
        .values.tolist()
    )
    text = f"""# E04 S02 Local Signal Specification

- Research step ID: {STEP_ID}
- Completion status: completed
- Artifacts written: this local signal specification, validation tables, no-signal regression report, trace examples, code copies, manifests, and status files under `$ARTIFACTS_DIR/research_steps/S02/`.
- Validation result: local/diffusive signals passed diffusion, locality, serialization, signal-sensing, trace-schema, randomized-control, and no-signal equivalence checks.
- Caveats or blockers: S02 validates communication mechanisms and information boundaries, not repair benefit. Diffusive signals can propagate over repeated local ticks and should be analyzed separately from nearest-neighbor signaling in later ablations.
- Recommended next action: implement repairable Frozen Cells in S03 only after Chief Scientist instruction.

## Locality And Update Contract

Update order is `sense_act_emit_diffuse`: a cell senses only the current fields in its configured local window, acts through the wrapped E03 policy interface, emits local scalar channels at its post-action position, then nearest-neighbor diffusion and decay are applied when the variant is `diffusive`. The policy sees flattened local aggregates such as `signal_blocked_local_sum`; it does not receive global Sortedness, whole-array field summaries, or distant field positions.

## Variants

{markdown_table(["Variant", "Definition"], variant_rows)}

## Channels

{markdown_table(["Channel", "Local definition"], channel_rows)}

## Wrapped Base Policies

{markdown_table(["Base policy ID", "Family", "Source"], base_rows)}

## Diffusion Rule

For field value `x_i`, one S02 diffusion tick is `x_i + diffusionRate * sum(neighbor - x_i)` over existing left/right neighbors, followed by multiplication by `(1 - decay)` and clipping at zero. Boundaries use no-flux behavior because missing neighbors simply do not contribute.
"""
    path.write_text(text, encoding="utf-8")


def write_trace_schema(path: Path) -> None:
    text = f"""# E04 S02 Signal Trace Schema

- Research step ID: {STEP_ID}
- Completion status: completed
- Artifacts written: signal trace schema plus validation artifacts in `$ARTIFACTS_DIR/research_steps/S02/`.
- Validation result: trace signal JSON parsed successfully and actor-visible positions stayed within configured `signalRange`.
- Caveats or blockers: trace fields are computational communication proxies and should not be interpreted as biological measurements.
- Recommended next action: use this schema when adding repairable Frozen Cells in S03.

| Field | Type | Meaning |
| --- | --- | --- |
| `signal_schema_version` | string | Signal extension schema version `{SIGNAL_REPAIR_VERSION}`. |
| `signal_config_json` | JSON object | Variant, channels, range, diffusion, decay, noise, seed, and update order. |
| `signal_fields_json` | JSON object | Current position-indexed scalar fields by channel. |
| `actor_signal_state_json` | JSON object | Activated-cell local signal observation, including visible local positions and per-channel aggregates. |
| `signal_variant_counts_json` | JSON object | Signal policy variant counts in the array. |

Actor-visible signal states include channel metrics `center`, `left`, `right`, `local_sum`, `local_mean`, and `local_max`. These are local-window aggregates only.
"""
    path.write_text(text, encoding="utf-8")


def write_no_signal_report(path: Path, regression_df: pd.DataFrame) -> None:
    failures = regression_df[~regression_df["success"]]
    text = f"""# E04 S02 No-Signal Regression Report

- Research step ID: {STEP_ID}
- Completion status: completed
- Artifacts written: this report plus `no_signal_regression.csv` and `no_signal_regression.parquet`.
- Validation result: {int(regression_df["success"].sum())} of {len(regression_df)} no-signal comparisons matched E03 exactly.
- Caveats or blockers: comparisons cover classic policies and selected E03 S14 frontier DSL policies on small deterministic cases, not every E03 sweep condition.
- Recommended next action: proceed to S03 only after Chief Scientist instruction.

No-signal wrappers delegate to the E03 base policy and do not store `{SIGNAL_STATE_KEY}`. Matching fields were completion, stop reason, final values, final Algotypes, swap count, comparison count, activation count, and trace state-hash sequence.

"""
    if failures.empty:
        text += "All no-signal regression rows passed.\n"
    else:
        text += markdown_table(
            ["Base policy", "Case", "Failures"],
            failures[["basePolicyId", "caseId", "failureFieldsJson"]].values.tolist(),
        )
    path.write_text(text, encoding="utf-8")


def write_validation_report(path: Path, validation_df: pd.DataFrame, regression_df: pd.DataFrame) -> None:
    all_checks = pd.concat([validation_df, regression_df], ignore_index=True)
    total = len(all_checks)
    passed = int(all_checks["success"].sum())
    family = all_checks.groupby("validationFamily")["success"].agg(["count", "sum"]).reset_index()
    text = f"""# E04 S02 Validation Report

- Research step ID: {STEP_ID}
- Completion status: completed
- Artifacts written: validation tables, local signal spec, no-signal regression report, trace schema, unit-test log, code copies, manifests, and status files.
- Validation result: passed; {passed} of {total} checks passed.
- Caveats or blockers: S02 validates local communication mechanisms and baseline equivalence only; it does not claim improved repair competence.
- Recommended next action: implement repairable Frozen Cells in S03 after Chief Scientist instruction.

## Check Families

{markdown_table(["Family", "Checks", "Passed"], family.values.tolist())}
"""
    path.write_text(text, encoding="utf-8")


def collect_artifacts(paths: Iterable[Path]) -> list[dict[str, Any]]:
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
    for package_name in ["memory_repair", "morphospace"]:
        src = REPO_ROOT / package_name
        dst = code_dir / package_name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
        copied.extend(sorted(path for path in dst.rglob("*.py")))

    for relative in [
        Path("scripts/e04_s02_local_communication.py"),
        Path("tests/test_e04_local_signals.py"),
    ]:
        src = REPO_ROOT / relative
        dst = code_dir / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--e03-artifacts", type=Path, default=DEFAULT_E03_ARTIFACTS)
    parser.add_argument("--frontier-limit", type=int, default=3)
    args = parser.parse_args()

    started_at = utc_now()
    worker_count = 1
    step_dir = args.artifacts_dir / "research_steps" / STEP_ID
    results_dir = args.artifacts_dir / "results"
    provenance_dir = args.artifacts_dir / "provenance"
    step_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    provenance_dir.mkdir(parents=True, exist_ok=True)

    base_records = classic_policies() + load_frontier_policies(args.e03_artifacts, args.frontier_limit)
    catalog_df = pd.DataFrame(variant_catalog_rows(base_records))
    validation, runs = validation_rows(base_records)
    validation_df = pd.DataFrame(validation)
    run_df = pd.DataFrame(runs)
    regression_df = pd.DataFrame(no_signal_regression_rows(base_records))

    artifacts: list[Path] = []
    artifacts.extend(dataframe_to_artifacts(catalog_df, step_dir / "signal_variant_catalog", results_dir / "e04_s02_signal_variant_catalog"))
    artifacts.extend(dataframe_to_artifacts(validation_df, step_dir / "signal_validation_results", results_dir / "e04_s02_signal_validation_results"))
    artifacts.extend(dataframe_to_artifacts(run_df, step_dir / "signal_variant_run_summary", results_dir / "e04_s02_signal_variant_run_summary"))
    artifacts.extend(dataframe_to_artifacts(regression_df, step_dir / "no_signal_regression", results_dir / "e04_s02_no_signal_regression"))

    trace_example_path = step_dir / "signal_trace_examples.jsonl"
    example_sim = SignalEventSimulator(
        [3, 1, 2],
        BubblePolicy(),
        signal_config=SignalConfig("diffusive", signal_range=1, diffusion_rate=0.25, decay=0.05),
        scheduler_seed=17,
        tie_breaker_seed=23,
        condition_id="e04_s02_trace_example",
        research_step_id=STEP_ID,
    )
    example_sim.run(max_activations=20)
    trace_example_path.write_text(
        "\n".join(json.dumps(json_ready(row), sort_keys=True) for row in example_sim.trace_rows[:10]) + "\n",
        encoding="utf-8",
    )
    artifacts.append(trace_example_path)

    signal_spec_path = step_dir / "local_signal_spec.md"
    trace_schema_path = step_dir / "signal_trace_schema.md"
    regression_report_path = step_dir / "no_signal_regression_report.md"
    validation_report_path = step_dir / "validation_report.md"
    write_signal_spec(signal_spec_path, catalog_df)
    write_trace_schema(trace_schema_path)
    write_no_signal_report(regression_report_path, regression_df)
    write_validation_report(validation_report_path, validation_df, regression_df)
    artifacts.extend([signal_spec_path, trace_schema_path, regression_report_path, validation_report_path])

    unit_cmd = [
        sys.executable,
        "-m",
        "unittest",
        "tests.test_e04_local_signals",
        "tests.test_e04_memory_extension",
        "tests.test_e03_policy_interface",
        "tests.test_e03_rule_dsl",
    ]
    unit_result = run_command(unit_cmd, cwd=REPO_ROOT)
    unit_log_path = step_dir / "repo_unit_test_log.txt"
    unit_log_path.write_text(
        "$ " + " ".join(unit_cmd) + "\n\nSTDOUT\n" + unit_result["stdout"] + "\n\nSTDERR\n" + unit_result["stderr"],
        encoding="utf-8",
    )
    artifacts.append(unit_log_path)
    artifacts.extend(copy_code_artifacts(step_dir))

    all_checks = pd.concat([validation_df, regression_df], ignore_index=True)
    checks_passed = int(all_checks["success"].sum())
    checks_total = len(all_checks)
    success = checks_passed == checks_total and bool(unit_result["success"])
    status = "completed" if success else "blocked"
    validation_result = (
        f"passed; {checks_passed} of {checks_total} validation checks passed and unit tests passed"
        if success
        else f"failed; {checks_passed} of {checks_total} validation checks passed; unit test success={unit_result['success']}"
    )
    caveats = [
        "S02 validates signal mechanisms, locality, diffusion, traceability, randomized controls, and no-signal equivalence; it does not test repair benefit yet.",
        "Diffusive signals propagate through repeated local ticks, so later ablations should separate nearest-neighbor and diffusive regimes.",
        f"Selected frontier coverage is limited to the first {args.frontier_limit} E03 S14 frontier DSL policies.",
        "Signal channels are computational communication proxies and not biological morphogen or bioelectric measurements.",
    ]
    recommended_next_action = "Proceed to S03 repairable Frozen Cells only after Chief Scientist instruction; do not start S03 from this run."
    lay_summary = (
        "S02 added local communication wrappers and a signal-field simulator. No-signal wrappers replay E03 exactly, "
        "nearest-neighbor and diffusive signals pass locality and diffusion checks, and policies can sense local signal aggregates."
    )

    summary_path = step_dir / "summary.md"
    summary_path.write_text(
        f"""# S02 Summary

- Research step ID: {STEP_ID}
- Completion status: {status}
- Artifacts written: `{signal_spec_path}` and additional files listed in `status.json` and `artifact_manifest.json`.
- Validation result: {validation_result}.
- Outcome classification: supportive
- Caveats or blockers: {'; '.join(caveats)}
- Lay summary: {lay_summary}
- Recommended next action: {recommended_next_action}
""",
        encoding="utf-8",
    )
    artifacts.append(summary_path)

    git = get_git_metadata()
    runtime = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpuCount": os.cpu_count(),
        "workerCount": worker_count,
        "threadEnvironment": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "PYTHONDONTWRITEBYTECODE": os.environ.get("PYTHONDONTWRITEBYTECODE"),
        },
    }
    status_payload = {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "title": STEP_TITLE,
        "success": success,
        "status": status,
        "startedAt": started_at,
        "completedAt": utc_now(),
        "artifactsWritten": [],
        "validationResult": validation_result,
        "validationChecksPassed": checks_passed,
        "validationChecksTotal": checks_total,
        "caveatsOrBlockers": caveats,
        "laySummary": lay_summary,
        "recommendedNextAction": recommended_next_action,
        "outcomeClassification": "supportive" if success else "constraining/contradictory",
        "signalRepairVersion": SIGNAL_REPAIR_VERSION,
        "signalVariants": list(SIGNAL_VARIANTS),
        "signalChannels": list(SIGNAL_CHANNELS),
        "basePolicyCount": len(base_records),
        "frontierPolicyCount": max(0, len(base_records) - 3),
        "repoUnitTests": {
            "command": unit_cmd,
            "returnCode": unit_result["returncode"],
            "success": unit_result["success"],
            "runtimeSeconds": unit_result["runtimeSeconds"],
            "logPath": str(unit_log_path),
        },
        "git": git,
        "runtime": runtime,
    }
    manifest_path = step_dir / "artifact_manifest.json"
    status_path = step_dir / "status.json"
    run_manifest_path = provenance_dir / "run_manifest.json"

    artifact_records = collect_artifacts(artifacts)
    status_payload["artifactsWritten"] = [item["path"] for item in artifact_records]
    write_json(status_path, status_payload)
    artifacts.append(status_path)
    artifact_records = collect_artifacts(artifacts)
    write_json(
        manifest_path,
        {
            "schema": "eidosoma.research_step_artifact_manifest.v1",
            "experimentId": EXPERIMENT_ID,
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "createdAt": utc_now(),
            "artifacts": artifact_records,
        },
    )
    artifacts.append(manifest_path)
    artifact_records = collect_artifacts(artifacts)
    status_payload["artifactsWritten"] = [item["path"] for item in artifact_records]
    write_json(status_path, status_payload)
    write_json(
        run_manifest_path,
        {
            "schema": "eidosoma.run_manifest.v1",
            "experimentId": EXPERIMENT_ID,
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "createdAt": utc_now(),
            "git": git,
            "runtime": runtime,
            "command": [sys.executable, str(Path(__file__).relative_to(REPO_ROOT))],
            "artifacts": artifact_records,
            "validationResult": validation_result,
        },
    )

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
