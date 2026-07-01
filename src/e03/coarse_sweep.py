"""Coarse E03 S07 morphospace sweep utilities.

S07 intentionally remains a coarse screen: it evaluates DSL local policies
under a deterministic position scheduler, not the full E02 public simulator.
The S06 compatibility table decides whether a policy uses the JAX batched
local-step path or the CPU DSL interpreter fallback.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from src.e03.gpu_batch_simulator import (
    CompiledPolicy,
    _require_jax,
    _stack_compiled,
    _step_batch,
    compile_policy,
    compatibility_record_for_policy,
    load_generated_policy_records,
    status_to_code,
)
from src.e03.rule_dsl import DSLArrayState, DSLInterpreter, DSLPolicy, parse_policy


SWEEP_SCHEMA = "eidosoma.e03.coarse_policy_sweep.v1"
DEFAULT_SWEEP_SEED = 2026070107


@dataclass(frozen=True)
class SweepConfig:
    """One deterministic array/scheduler condition."""

    config_id: str
    phase: str
    split: str
    array_size: int
    seed: int
    event_cap: int
    input_profile: str = "random_permutation"
    scheduler: str = "cyclic_scan_seed_offset"


@dataclass(frozen=True)
class PolicyRecord:
    """One S05 policy plus S06 routing metadata."""

    policy: DSLPolicy
    policy_id: str
    policy_name: str
    source_kind: str
    semantic_hash: str
    dsl_sha256: str
    features: dict[str, Any]
    compiles_for_batch: bool
    requires_cpu_fallback: bool
    fallback_reasons: tuple[str, ...]
    route: str


def screen_configs() -> list[SweepConfig]:
    """Return train/held-out small-array screen configs for all policies."""

    configs: list[SweepConfig] = []
    for split, seeds in {
        "train": (1101, 1102, 1103),
        "heldout": (2101, 2102),
    }.items():
        for seed in seeds:
            configs.append(
                SweepConfig(
                    config_id=f"screen_{split}_n8_seed{seed}",
                    phase="screen",
                    split=split,
                    array_size=8,
                    seed=seed,
                    event_cap=32,
                )
            )
    for split, seeds in {
        "train": (1201, 1202),
        "heldout": (2201,),
    }.items():
        for seed in seeds:
            configs.append(
                SweepConfig(
                    config_id=f"screen_{split}_n16_seed{seed}",
                    phase="screen",
                    split=split,
                    array_size=16,
                    seed=seed,
                    event_cap=64,
                )
            )
    return configs


def scale_configs() -> list[SweepConfig]:
    """Return sparse transfer-probe configs for selected policies."""

    return [
        SweepConfig("scale_heldout_n100_seed3101", "scale", "heldout", 100, 3101, 160),
        SweepConfig("scale_heldout_n100_seed3102", "scale", "heldout", 100, 3102, 160),
        SweepConfig("scale_sparse_n1000_seed4101", "scale_sparse", "heldout", 1000, 4101, 32),
    ]


def parse_fallback_reasons(text: Any) -> tuple[str, ...]:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ()
    try:
        value = json.loads(str(text))
    except json.JSONDecodeError:
        return (str(text),)
    if isinstance(value, list):
        return tuple(str(item) for item in value)
    return (str(value),)


def load_policy_records(policy_library: Path, compatibility_path: Path | None = None) -> list[PolicyRecord]:
    """Load S05 policies and route them using the S06 compatibility flags."""

    records = load_generated_policy_records(policy_library)
    compatibility: dict[str, dict[str, Any]] = {}
    if compatibility_path is not None and compatibility_path.exists():
        comp_df = pd.read_parquet(compatibility_path)
        compatibility = {str(row["policy_id"]): dict(row) for row in comp_df.to_dict(orient="records")}

    loaded: list[PolicyRecord] = []
    for record in records:
        policy = parse_policy(record["dslSource"])
        policy_id = str(record["policyId"])
        comp = compatibility.get(policy_id)
        if comp is None:
            comp = compatibility_record_for_policy(policy)
        compiles_for_batch = bool(comp.get("compiles_for_batch", False))
        requires_cpu_fallback = bool(comp.get("requires_cpu_fallback", True))
        route = "jax_batch" if compiles_for_batch and not requires_cpu_fallback else "cpu_fallback"
        loaded.append(
            PolicyRecord(
                policy=policy,
                policy_id=policy_id,
                policy_name=str(record["policyName"]),
                source_kind=str(record["sourceKind"]),
                semantic_hash=str(record["semanticHash"]),
                dsl_sha256=str(record["dslSha256"]),
                features=dict(record.get("features", {})),
                compiles_for_batch=compiles_for_batch,
                requires_cpu_fallback=requires_cpu_fallback,
                fallback_reasons=parse_fallback_reasons(comp.get("fallback_reasons_json")),
                route=route,
            )
        )
    return loaded


def initial_values(array_size: int, seed: int, input_profile: str = "random_permutation") -> tuple[int, ...]:
    if input_profile != "random_permutation":
        raise ValueError(f"Unsupported input profile: {input_profile}")
    rng = np.random.default_rng(seed)
    values = np.arange(1, array_size + 1, dtype=np.int32)
    rng.shuffle(values)
    return tuple(int(value) for value in values)


def actor_schedule(array_size: int, event_cap: int, seed: int) -> tuple[int, ...]:
    """Return a deterministic cyclic position schedule with seed offset."""

    offset = seed % array_size
    return tuple(int((offset + step) % array_size) for step in range(event_cap))


def inversion_count(values: Sequence[int]) -> int:
    """Count pairwise inversions for unique integer arrays."""

    arr = [int(value) for value in values]
    if len(arr) < 2:
        return 0
    ranks = {value: idx + 1 for idx, value in enumerate(sorted(arr))}
    tree = [0] * (len(arr) + 2)

    def add(idx: int) -> None:
        while idx < len(tree):
            tree[idx] += 1
            idx += idx & -idx

    def prefix(idx: int) -> int:
        total = 0
        while idx > 0:
            total += tree[idx]
            idx -= idx & -idx
        return total

    inversions = 0
    for seen, value in enumerate(arr):
        rank = ranks[value]
        inversions += seen - prefix(rank)
        add(rank)
    return int(inversions)


def adjacent_error_count(values: Sequence[int]) -> int:
    return int(sum(int(values[idx] > values[idx + 1]) for idx in range(len(values) - 1)))


def sortedness_metrics(values: Sequence[int]) -> dict[str, Any]:
    n = len(values)
    inv = inversion_count(values)
    max_inv = n * (n - 1) // 2
    adjacent_errors = adjacent_error_count(values)
    return {
        "inversion_count": inv,
        "max_inversions": max_inv,
        "inversion_sortedness": 1.0 if max_inv == 0 else 1.0 - (inv / max_inv),
        "adjacent_error_count": adjacent_errors,
        "adjacent_sortedness": 1.0 if n <= 1 else 1.0 - (adjacent_errors / (n - 1)),
        "is_sorted": inv == 0,
    }


def _base_row(record: PolicyRecord, config: SweepConfig, values: Sequence[int]) -> dict[str, Any]:
    return {
        "schema": SWEEP_SCHEMA,
        "experiment_id": "E03",
        "research_step_id": "S07",
        "policy_id": record.policy_id,
        "policy_name": record.policy_name,
        "source_kind": record.source_kind,
        "semantic_hash": record.semantic_hash,
        "dsl_sha256": record.dsl_sha256,
        "route": record.route,
        "compiles_for_batch": bool(record.compiles_for_batch),
        "requires_cpu_fallback": bool(record.requires_cpu_fallback),
        "fallback_reasons_json": json.dumps(list(record.fallback_reasons), separators=(",", ":")),
        "config_id": config.config_id,
        "phase": config.phase,
        "split": config.split,
        "heldout": config.split == "heldout",
        "array_size": int(config.array_size),
        "seed": int(config.seed),
        "event_cap": int(config.event_cap),
        "input_profile": config.input_profile,
        "scheduler": config.scheduler,
        "initial_values_json": json.dumps(list(values), separators=(",", ":")),
    }


def _finalize_row(
    row: dict[str, Any],
    *,
    final_values: Sequence[int],
    compare_count: int,
    swap_count: int,
    update_count: int,
    wait_count: int,
    elapsed_seconds: float,
    execution_status: str = "ok",
    error_message: str = "",
    backend: str = "cpu",
    device: str = "cpu",
) -> dict[str, Any]:
    initial = json.loads(row["initial_values_json"])
    initial_metrics = sortedness_metrics(initial)
    final_metrics = sortedness_metrics(final_values)
    timed_out = execution_status == "ok" and not bool(final_metrics["is_sorted"])
    if execution_status != "ok":
        run_status = "invalid"
        convergence_status = "not_evaluated"
    elif timed_out:
        run_status = "timeout_event_cap"
        convergence_status = "timeout_event_cap"
    else:
        run_status = "ok_sorted"
        convergence_status = "sorted"
    row = dict(row)
    row.update(
        {
            "execution_status": execution_status,
            "run_status": run_status,
            "convergence_status": convergence_status,
            "timed_out": bool(timed_out),
            "invalid": execution_status != "ok",
            "error_message": error_message[:500],
            "backend": backend,
            "device": device,
            "events_executed": int(row["event_cap"]) if execution_status == "ok" else 0,
            "compare_count": int(compare_count),
            "swap_count": int(swap_count),
            "update_count": int(update_count),
            "wait_count": int(wait_count),
            "work_count": int(compare_count + swap_count + update_count),
            "initial_inversion_count": int(initial_metrics["inversion_count"]),
            "final_inversion_count": int(final_metrics["inversion_count"]),
            "initial_inversion_sortedness": float(initial_metrics["inversion_sortedness"]),
            "final_inversion_sortedness": float(final_metrics["inversion_sortedness"]),
            "inversion_sortedness_delta": float(final_metrics["inversion_sortedness"] - initial_metrics["inversion_sortedness"]),
            "initial_adjacent_error_count": int(initial_metrics["adjacent_error_count"]),
            "final_adjacent_error_count": int(final_metrics["adjacent_error_count"]),
            "final_adjacent_sortedness": float(final_metrics["adjacent_sortedness"]),
            "final_is_sorted": bool(final_metrics["is_sorted"]),
            "final_values_head_json": json.dumps(list(final_values)[:20], separators=(",", ":")),
            "final_values_tail_json": json.dumps(list(final_values)[-20:], separators=(",", ":")),
            "elapsed_seconds": float(elapsed_seconds),
        }
    )
    return row


def run_cpu_policy_config(record: PolicyRecord, config: SweepConfig) -> dict[str, Any]:
    values = initial_values(config.array_size, config.seed, config.input_profile)
    schedule = actor_schedule(config.array_size, config.event_cap, config.seed)
    row = _base_row(record, config, values)
    started = time.perf_counter()
    try:
        interpreter = DSLInterpreter(record.policy)
        rng = random.Random((config.seed * 1000003) ^ int(record.dsl_sha256[:12], 16))
        statuses: tuple[str, ...] = tuple("ACTIVE" for _ in values)
        ideal_position: int | None = None
        compare_count = 0
        swap_count = 0
        update_count = 0
        wait_count = 0
        current_values = values
        for actor_index in schedule:
            state = DSLArrayState(
                values=current_values,
                statuses=statuses,
                actor_index=int(actor_index),
                ideal_position=ideal_position,
            )
            result = interpreter.step_state(state, rng)
            current_values = result.state_after.values
            statuses = tuple(result.state_after.statuses or statuses)
            ideal_position = result.state_after.ideal_position
            compare_count += int(bool(result.action.compare_counted))
            swap_count += int(result.action.action_type == "swap")
            update_count += int(result.action.action_type == "update_state")
            wait_count += int(result.action.action_type == "wait")
        elapsed = time.perf_counter() - started
        return _finalize_row(
            row,
            final_values=current_values,
            compare_count=compare_count,
            swap_count=swap_count,
            update_count=update_count,
            wait_count=wait_count,
            elapsed_seconds=elapsed,
            backend="cpu",
            device="cpu",
        )
    except Exception as exc:  # pragma: no cover - validation records the row
        elapsed = time.perf_counter() - started
        return _finalize_row(
            row,
            final_values=values,
            compare_count=0,
            swap_count=0,
            update_count=0,
            wait_count=0,
            elapsed_seconds=elapsed,
            execution_status="invalid",
            error_message=repr(exc),
            backend="cpu",
            device="cpu",
        )


def run_jax_config_batch(records: Sequence[PolicyRecord], config: SweepConfig, *, seed: int = DEFAULT_SWEEP_SEED) -> list[dict[str, Any]]:
    """Run one config for many batch-ready policies with the S06 JAX kernel."""

    if not records:
        return []
    values = initial_values(config.array_size, config.seed, config.input_profile)
    schedule = actor_schedule(config.array_size, config.event_cap, config.seed)
    base_rows = [_base_row(record, config, values) for record in records]
    started = time.perf_counter()
    try:
        jax, jnp = _require_jax()
        devices = jax.devices()
        device = devices[0]
        compiled: list[CompiledPolicy] = [compile_policy(record.policy) for record in records]
        arrays = _stack_compiled(compiled, jnp)
        arrays = {key: jax.device_put(value, device) for key, value in arrays.items()}
        batch_size = len(records)
        value_array = jax.device_put(jnp.asarray([values] * batch_size, dtype=jnp.int32), device)
        status_array = jax.device_put(
            jnp.asarray([[status_to_code("ACTIVE")] * config.array_size for _ in range(batch_size)], dtype=jnp.int32),
            device,
        )
        actor_index = jax.device_put(jnp.asarray([schedule[0]] * batch_size, dtype=jnp.int32), device)
        ideal_position = jax.device_put(jnp.full(batch_size, -1, dtype=jnp.int32), device)
        left_boundary = jax.device_put(jnp.zeros(batch_size, dtype=jnp.int32), device)
        right_boundary = jax.device_put(jnp.full(batch_size, config.array_size - 1, dtype=jnp.int32), device)
        reverse_direction = jax.device_put(jnp.zeros(batch_size, dtype=bool), device)
        compare_count = jax.device_put(jnp.zeros(batch_size, dtype=jnp.int32), device)
        swap_count = jax.device_put(jnp.zeros(batch_size, dtype=jnp.int32), device)
        update_count = jax.device_put(jnp.zeros(batch_size, dtype=jnp.int32), device)
        wait_count = jax.device_put(jnp.zeros(batch_size, dtype=jnp.int32), device)
        rng = np.random.default_rng(seed ^ config.seed)
        condition_rolls = jax.device_put(
            jnp.asarray(
                rng.random((config.event_cap, batch_size, compiled[0].max_rules, compiled[0].max_conditions), dtype=np.float32),
                dtype=jnp.float32,
            ),
            device,
        )
        choice_rolls = jax.device_put(
            jnp.asarray(
                rng.random((config.event_cap, batch_size, compiled[0].max_rules, compiled[0].max_actions), dtype=np.float32),
                dtype=jnp.float32,
            ),
            device,
        )
        schedule_array = jax.device_put(jnp.asarray(schedule, dtype=jnp.int32), device)

        def scan_body(carry: tuple[Any, ...], inputs: tuple[Any, ...]) -> tuple[tuple[Any, ...], None]:
            (
                carried_values,
                carried_statuses,
                carried_ideal,
                carried_compare,
                carried_swap,
                carried_update,
                carried_wait,
            ) = carry
            scheduled_actor, condition_roll, choice_roll = inputs
            scheduled_actor_index = jnp.full(batch_size, scheduled_actor, dtype=jnp.int32)
            (
                next_values,
                next_statuses,
                _actor_after,
                next_ideal,
                next_compare,
                next_swap,
                next_update,
                next_wait,
            ) = _step_batch(
                arrays,
                carried_values,
                carried_statuses,
                scheduled_actor_index,
                carried_ideal,
                left_boundary,
                right_boundary,
                reverse_direction,
                condition_roll,
                choice_roll,
                carried_compare,
                carried_swap,
                carried_update,
                carried_wait,
                jnp,
            )
            return (
                next_values,
                next_statuses,
                next_ideal,
                next_compare,
                next_swap,
                next_update,
                next_wait,
            ), None

        @jax.jit
        def run_scan(initial_carry: tuple[Any, ...]) -> tuple[Any, ...]:
            final_carry, _ = jax.lax.scan(scan_body, initial_carry, (schedule_array, condition_rolls, choice_rolls))
            return final_carry

        (
            value_array,
            status_array,
            ideal_position,
            compare_count,
            swap_count,
            update_count,
            wait_count,
        ) = run_scan((value_array, status_array, ideal_position, compare_count, swap_count, update_count, wait_count))
        value_array.block_until_ready()
        elapsed = time.perf_counter() - started
        final_values = np.asarray(jax.device_get(value_array))
        compares = np.asarray(jax.device_get(compare_count))
        swaps = np.asarray(jax.device_get(swap_count))
        updates = np.asarray(jax.device_get(update_count))
        waits = np.asarray(jax.device_get(wait_count))
        backend = str(getattr(device, "platform", jax.default_backend()))
        device_text = str(device)
        return [
            _finalize_row(
                base_rows[idx],
                final_values=tuple(int(value) for value in final_values[idx].tolist()),
                compare_count=int(compares[idx]),
                swap_count=int(swaps[idx]),
                update_count=int(updates[idx]),
                wait_count=int(waits[idx]),
                elapsed_seconds=elapsed / max(batch_size, 1),
                backend=backend,
                device=device_text,
            )
            for idx in range(batch_size)
        ]
    except Exception as exc:  # pragma: no cover - validation records the rows
        elapsed = time.perf_counter() - started
        return [
            _finalize_row(
                row,
                final_values=values,
                compare_count=0,
                swap_count=0,
                update_count=0,
                wait_count=0,
                elapsed_seconds=elapsed / max(len(base_rows), 1),
                execution_status="invalid",
                error_message=repr(exc),
                backend="jax_error",
                device="jax_error",
            )
            for row in base_rows
        ]


def run_configs(records: Sequence[PolicyRecord], configs: Sequence[SweepConfig], *, seed: int = DEFAULT_SWEEP_SEED) -> pd.DataFrame:
    """Run all policies for all configs, routing via S06 compatibility."""

    jax_records = [record for record in records if record.route == "jax_batch"]
    cpu_records = [record for record in records if record.route != "jax_batch"]
    rows: list[dict[str, Any]] = []
    for config in configs:
        rows.extend(run_jax_config_batch(jax_records, config, seed=seed))
        for record in cpu_records:
            rows.append(run_cpu_policy_config(record, config))
    return pd.DataFrame(rows)


def policy_summary_frame(run_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate row-level sweep records into one competence row per policy."""

    rows: list[dict[str, Any]] = []
    grouped = run_df.groupby("policy_id", sort=False)
    for policy_id, group in grouped:
        screen = group[group["phase"] == "screen"]
        heldout = screen[screen["heldout"] == True]  # noqa: E712
        scale = group[group["phase"].isin(["scale", "scale_sparse"])]
        n100 = scale[scale["array_size"] == 100]
        n1000 = scale[scale["array_size"] == 1000]
        basis = heldout if not heldout.empty else screen
        route = str(group["route"].iloc[0])
        rows.append(
            {
                "schema": "eidosoma.e03.policy_competence_s07.v1",
                "experiment_id": "E03",
                "research_step_id": "S07",
                "policy_id": policy_id,
                "policy_name": str(group["policy_name"].iloc[0]),
                "source_kind": str(group["source_kind"].iloc[0]),
                "route": route,
                "requires_cpu_fallback": bool(group["requires_cpu_fallback"].iloc[0]),
                "screen_run_count": int(len(screen)),
                "heldout_run_count": int(len(heldout)),
                "scale_run_count": int(len(scale)),
                "invalid_run_count": int(group["invalid"].sum()),
                "timeout_run_count": int(group["timed_out"].sum()),
                "screen_train_final_sortedness_mean": float(screen[screen["heldout"] == False]["final_inversion_sortedness"].mean()),  # noqa: E712
                "screen_heldout_final_sortedness_mean": float(heldout["final_inversion_sortedness"].mean()) if not heldout.empty else np.nan,
                "screen_heldout_sorted_run_fraction": float(heldout["final_is_sorted"].mean()) if not heldout.empty else np.nan,
                "screen_heldout_improvement_mean": float(heldout["inversion_sortedness_delta"].mean()) if not heldout.empty else np.nan,
                "screen_heldout_work_mean": float(heldout["work_count"].mean()) if not heldout.empty else np.nan,
                "screen_score": float(
                    basis["final_inversion_sortedness"].mean()
                    + 0.25 * basis["inversion_sortedness_delta"].mean()
                    - 0.0005 * basis["work_count"].mean()
                )
                if not basis.empty
                else np.nan,
                "n100_final_sortedness_mean": float(n100["final_inversion_sortedness"].mean()) if not n100.empty else np.nan,
                "n1000_final_sortedness_mean": float(n1000["final_inversion_sortedness"].mean()) if not n1000.empty else np.nan,
                "best_final_sortedness": float(group["final_inversion_sortedness"].max()),
                "classic_dsl_seed": bool(str(group["source_kind"].iloc[0]) == "classic_dsl_seed"),
            }
        )
    return pd.DataFrame(rows)


def select_scale_policy_ids(summary: pd.DataFrame, *, target_count: int = 48, sparse_count: int = 12) -> tuple[list[str], list[str]]:
    """Select promising and landmark policies for n=100 and n=1000 probes."""

    selected: list[str] = []

    def add(ids: Iterable[str]) -> None:
        for policy_id in ids:
            if policy_id not in selected:
                selected.append(policy_id)

    classics = summary[summary["classic_dsl_seed"] == True]["policy_id"].tolist()  # noqa: E712
    add(classics)
    ranked = summary.sort_values(["screen_score", "screen_heldout_final_sortedness_mean"], ascending=False)
    add(ranked["policy_id"].head(max(target_count, 1)).tolist())
    for route in ("jax_batch", "cpu_fallback"):
        add(ranked[ranked["route"] == route]["policy_id"].head(8).tolist())
    for source_kind in sorted(summary["source_kind"].dropna().unique()):
        add(ranked[ranked["source_kind"] == source_kind]["policy_id"].head(3).tolist())
    scale_ids = selected[:target_count]
    sparse_ids = [policy_id for policy_id in scale_ids if policy_id in classics]
    sparse_ids.extend([policy_id for policy_id in scale_ids if policy_id not in sparse_ids])
    return scale_ids, sparse_ids[:sparse_count]


def validation_frame(
    run_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    policy_count: int,
    *,
    require_scale: bool = True,
) -> pd.DataFrame:
    cases: list[dict[str, Any]] = []

    def add(name: str, success: bool, expected: str, observed: Any, notes: str) -> None:
        cases.append(
            {
                "validation_case": name,
                "success": bool(success),
                "expected": expected,
                "observed": str(observed),
                "notes": notes,
            }
        )

    add("all_policies_screened", summary_df["policy_id"].nunique() == policy_count, str(policy_count), summary_df["policy_id"].nunique(), "Every S05 policy has an aggregate competence row.")
    add("heldout_rows_present", int(run_df["heldout"].sum()) > 0, "held-out rows > 0", int(run_df["heldout"].sum()), "S07 must include held-out seeds.")
    add("jax_route_present", int((run_df["route"] == "jax_batch").sum()) > 0, "JAX-routed rows > 0", int((run_df["route"] == "jax_batch").sum()), "S06-compatible deterministic policies route through JAX.")
    add("cpu_fallback_present", int((run_df["route"] == "cpu_fallback").sum()) > 0, "CPU fallback rows > 0", int((run_df["route"] == "cpu_fallback").sum()), "Exact stochastic/memory/signal policies route through CPU fallback.")
    add("fallback_flags_consistent", bool((run_df["requires_cpu_fallback"] == (run_df["route"] == "cpu_fallback")).all()), "fallback flag matches route", run_df[["requires_cpu_fallback", "route"]].drop_duplicates().to_dict(orient="records"), "Routing uses S06 compatibility flags.")
    add("invalid_runs_classified", "invalid" in run_df.columns and "execution_status" in run_df.columns, "invalid classification columns present", run_df.columns.tolist(), "Invalid runs are explicit, even when count is zero.")
    add("timeouts_classified", int(run_df["timed_out"].sum()) >= 0 and "timeout_event_cap" in set(run_df["convergence_status"]), "timeout rows classified", int(run_df["timed_out"].sum()), "Non-sorted runs at the event cap are labeled as timeouts.")
    scale_rows = int((run_df["phase"].isin(["scale", "scale_sparse"])).sum())
    add(
        "scale_probe_present",
        scale_rows > 0 if require_scale else True,
        "scale rows > 0" if require_scale else "not required for unit smoke",
        scale_rows,
        "Selected policies include n=100 and sparse n=1000 probes.",
    )
    add("no_unclassified_status", bool(run_df["run_status"].notna().all()), "all run_status non-null", int(run_df["run_status"].notna().sum()), "Every row has an explicit run status.")
    return pd.DataFrame(cases)


def result_digest(frame: pd.DataFrame) -> str:
    import hashlib

    cols = [
        "policy_id",
        "config_id",
        "route",
        "run_status",
        "final_inversion_count",
        "compare_count",
        "swap_count",
        "update_count",
        "wait_count",
    ]
    payload = frame[cols].sort_values(["policy_id", "config_id"]).to_dict(orient="records")
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
