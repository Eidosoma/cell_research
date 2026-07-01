"""Auditable E03 policy atlas utilities for S15.

The atlas is a static audit artifact: it joins measured behavior, class labels,
source records, frontier-candidate status, classic-position context, and small
representative trace previews without rerunning large sweeps.
"""

from __future__ import annotations

import hashlib
import html
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.e03.coarse_sweep import actor_schedule, sortedness_metrics
from src.e03.frontier_candidates import initial_values_for_profile
from src.e03.policy_generation import semantic_hash
from src.e03.rule_dsl import DSLArrayState, DSLInterpreter, parse_policy, stable_json


ATLAS_SCHEMA = "eidosoma.e03.policy_atlas.v1"
DEFAULT_ATLAS_SEED = 2026070115


@dataclass(frozen=True)
class AtlasTraceConfig:
    """One compact representative-trace preview configuration."""

    config_id: str = "s15_trace_n8_seed15001"
    array_size: int = 8
    seed: int = 15001
    event_cap: int = 32
    input_profile: str = "random_permutation"
    scheduler: str = "cyclic_scan_seed_offset"
    checkpoints: tuple[int, ...] = (0, 1, 2, 4, 8, 16, 32)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        if value is None or pd.isna(value):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _json_list(values: Sequence[Any]) -> str:
    return stable_json([value for value in values if str(value) and str(value) != "nan"])


def _short_source(source: Any, limit: int = 900) -> str:
    text = "" if source is None or (isinstance(source, float) and pd.isna(source)) else str(source).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n..."


def exemplar_role_frame(exemplars: pd.DataFrame) -> pd.DataFrame:
    """Collapse S11 exemplar inspection rows to one row per policy."""

    if exemplars.empty:
        return pd.DataFrame(columns=["policy_id", "exemplar_roles_json", "exemplar_status_json"])
    grouped = exemplars.groupby("policy_id", dropna=False).agg(
        exemplar_roles_json=("exemplar_role", lambda values: _json_list(sorted(set(str(value) for value in values)))),
        exemplar_status_json=("manual_inspection_status", lambda values: _json_list(sorted(set(str(value) for value in values)))),
    )
    return grouped.reset_index()


def phase_boundary_frame(boundaries: pd.DataFrame) -> pd.DataFrame:
    """Collapse S09 boundary candidates to one row per base policy."""

    if boundaries.empty:
        return pd.DataFrame(columns=["policy_id"])
    df = boundaries.copy()
    for column in ("transition_strength", "replication_available", "replication_direction_match"):
        if column in df.columns:
            if column.startswith("replication"):
                df[column] = df[column].map(_as_bool).astype(float)
            else:
                df[column] = pd.to_numeric(df[column], errors="coerce")
    grouped = df.groupby("base_policy_id", dropna=False).agg(
        s09_boundary_count=("base_policy_id", "size"),
        s09_boundary_axes_json=("axis_name", lambda values: _json_list(sorted(set(str(value) for value in values)))),
        s09_boundary_kinds_json=("boundary_kind", lambda values: _json_list(sorted(set(str(value) for value in values)))),
        s09_max_transition_strength=("transition_strength", "max"),
        s09_replicated_fraction=("replication_available", "mean"),
        s09_direction_match_fraction=("replication_direction_match", "mean"),
    )
    return grouped.reset_index().rename(columns={"base_policy_id": "policy_id"})


def ablation_claim_frame(claims: pd.DataFrame) -> pd.DataFrame:
    """Collapse S12 ablation claims to one row per original policy."""

    if claims.empty:
        return pd.DataFrame(columns=["policy_id"])
    df = claims.copy()
    df["heldout_delta_final_sortedness"] = pd.to_numeric(df.get("heldout_delta_final_sortedness", 0.0), errors="coerce")
    grouped = df.groupby("original_policy_id", dropna=False).agg(
        s12_claim_count=("claim_type", "size"),
        s12_claim_types_json=("claim_type", lambda values: _json_list(sorted(set(str(value) for value in values)))),
        s12_ablation_components_json=("ablation_component", lambda values: _json_list(sorted(set(str(value) for value in values)))),
        s12_min_heldout_delta=("heldout_delta_final_sortedness", "min"),
        s12_max_heldout_delta=("heldout_delta_final_sortedness", "max"),
    )
    return grouped.reset_index().rename(columns={"original_policy_id": "policy_id"})


def classic_classification_frame(classic_analysis: pd.DataFrame) -> pd.DataFrame:
    if classic_analysis.empty:
        return pd.DataFrame(columns=["classic_family", "classic_position_classification", "classification_confidence"])
    rows = []
    for record in classic_analysis.to_dict(orient="records"):
        rows.append(
            {
                "classic_family": str(record.get("algorithm_label", "")),
                "classic_position_classification": str(record.get("classic_position_classification", "")),
                "classification_confidence": str(record.get("classification_confidence", "")),
            }
        )
    return pd.DataFrame(rows).drop_duplicates("classic_family")


def _merge_fill(base: pd.DataFrame, extra: pd.DataFrame, *, on: str = "policy_id", suffix: str) -> pd.DataFrame:
    if extra.empty:
        return base
    out = base.merge(extra, on=on, how="left", suffixes=("", suffix), validate="one_to_one")
    for column in extra.columns:
        if column == on:
            continue
        suffixed = f"{column}{suffix}"
        if suffixed in out.columns:
            if column in out.columns:
                out[column] = out[column].where(out[column].notna(), out[suffixed])
            else:
                out[column] = out[suffixed]
            out = out.drop(columns=[suffixed])
    return out


def build_atlas_index(
    *,
    embeddings: pd.DataFrame,
    clusters: pd.DataFrame,
    source_table: pd.DataFrame,
    frontier_candidates: pd.DataFrame,
    exemplars: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    classic_analysis: pd.DataFrame,
    phase_boundaries: pd.DataFrame,
    ablation_claims: pd.DataFrame,
) -> pd.DataFrame:
    """Build one linked atlas row per S10 policy."""

    base_cols = [
        "policy_id",
        "policy_name",
        "source_kind",
        "route",
        "requires_cpu_fallback",
        "screen_score",
        "screen_heldout_final_sortedness_mean",
        "screen_heldout_improvement_mean",
        "screen_heldout_work_mean",
        "n100_final_sortedness_mean",
        "n1000_final_sortedness_mean",
        "best_final_sortedness",
        "timeout_run_count",
        "qd_candidate",
        "s08_archive_winner",
        "classic_landmark",
        "classic_family",
        "s09_diagnostic_available",
        "embedding_x",
        "embedding_y",
        "embedding_z",
        "embedding_feature_count",
        "embedding_missing_fraction",
    ]
    index = embeddings[[column for column in base_cols if column in embeddings.columns]].copy()

    cluster_cols = [
        "policy_id",
        "class_id",
        "cautious_label",
        "label_confidence",
        "cluster_distance",
        "usesIdeal",
        "usesPrefixSorted",
        "usesRandomCondition",
        "usesProbabilisticAction",
        "usesMemory",
        "usesSignal",
        "swapActionCount",
        "stateActionCount",
        "waitActionCount",
        "rule_count",
        "dsl_excerpt",
    ]
    index = _merge_fill(index, clusters[[column for column in cluster_cols if column in clusters.columns]], suffix="_cluster")

    source_cols = [
        "policy_id",
        "computed_dsl_sha256",
        "computed_semantic_hash",
        "computed_policy_name",
        "source_hash_valid",
        "source_parse_success",
        "code_source",
        "dsl_source",
    ]
    index = _merge_fill(index, source_table[[column for column in source_cols if column in source_table.columns]], suffix="_source")

    frontier_cols = [
        "policy_id",
        "curated_rank",
        "s14_candidate_status",
        "selection_score",
        "selection_reasons_json",
        "pareto_frontier_labels_json",
        "s14_mean_final_sortedness",
        "s14_perturbation_mean_final_sortedness",
        "s14_n32plus_mean_final_sortedness",
        "s14_retention_ratio_vs_s07",
        "s14_mean_work",
        "s14_timeout_fraction",
    ]
    index = _merge_fill(index, frontier_candidates[[column for column in frontier_cols if column in frontier_candidates.columns]], suffix="_frontier")
    index = _merge_fill(index, exemplar_role_frame(exemplars), suffix="_exemplar")
    index = _merge_fill(index, phase_boundary_frame(phase_boundaries), suffix="_phase")
    index = _merge_fill(index, ablation_claim_frame(ablation_claims), suffix="_ablation")

    classic_lookup = classic_classification_frame(classic_analysis)
    if not classic_lookup.empty:
        index = index.merge(classic_lookup, on="classic_family", how="left", validate="many_to_one")
    else:
        index["classic_position_classification"] = ""
        index["classification_confidence"] = ""

    class_context_cols = ["class_id", "policy_count", "mean_screen_score", "cautious_label", "label_basis"]
    class_context = cluster_summary[[column for column in class_context_cols if column in cluster_summary.columns]].copy()
    if not class_context.empty:
        class_context = class_context.rename(
            columns={
                "policy_count": "class_policy_count",
                "mean_screen_score": "class_mean_screen_score",
                "cautious_label": "class_summary_label",
                "label_basis": "class_label_basis",
            }
        )
        index = index.merge(class_context, on="class_id", how="left", validate="many_to_one")

    bool_columns = [
        "qd_candidate",
        "s08_archive_winner",
        "classic_landmark",
        "s09_diagnostic_available",
        "source_hash_valid",
        "source_parse_success",
    ]
    for column in bool_columns:
        if column not in index.columns:
            index[column] = False
        index[column] = index[column].map(_as_bool)
    for column in ("curated_rank", "s09_boundary_count", "s12_claim_count"):
        if column not in index.columns:
            index[column] = np.nan if column == "curated_rank" else 0
    index["is_frontier_candidate"] = pd.to_numeric(index["curated_rank"], errors="coerce").notna()
    index["is_s11_exemplar"] = index.get("exemplar_roles_json", pd.Series("", index=index.index)).fillna("").astype(str).ne("")
    index["has_full_dsl_source"] = index.get("dsl_source", pd.Series("", index=index.index)).fillna("").astype(str).str.len() > 0
    index["dsl_source_excerpt"] = index.get("dsl_source", pd.Series("", index=index.index)).map(_short_source)
    index["atlas_anchor"] = index["policy_id"].astype(str).map(lambda value: "policy-" + value.replace(":", "-").replace("/", "-"))
    index["atlas_policy_url"] = index["atlas_anchor"].map(lambda anchor: f"e03_policy_atlas.html#{anchor}")

    def role(row: pd.Series) -> str:
        if bool(row["is_frontier_candidate"]):
            return "frontier_candidate"
        if bool(row["classic_landmark"]):
            return "classic_landmark"
        if bool(row["is_s11_exemplar"]):
            return "s11_exemplar"
        if bool(row["s08_archive_winner"]):
            return "archive_winner"
        if bool(row["qd_candidate"]):
            return "qd_candidate"
        return "policy"

    index["atlas_role"] = index.apply(role, axis=1)
    index["trace_available"] = False
    index["schema"] = ATLAS_SCHEMA
    index["experiment_id"] = "E03"
    index["research_step_id"] = "S15"

    sort_cols = ["is_frontier_candidate", "classic_landmark", "is_s11_exemplar", "screen_score", "policy_id"]
    index = index.sort_values(sort_cols, ascending=[False, False, False, False, True], kind="mergesort").reset_index(drop=True)
    return index


def representative_policy_ids(index: pd.DataFrame, *, max_count: int = 80) -> list[str]:
    """Choose policies for compact trace previews and static detail cards."""

    selected: list[str] = []

    def add(ids: Sequence[Any]) -> None:
        for value in ids:
            policy_id = str(value)
            if policy_id and policy_id not in selected:
                selected.append(policy_id)

    add(index[index["is_frontier_candidate"]].sort_values("curated_rank")["policy_id"].tolist())
    add(index[index["classic_landmark"]]["policy_id"].tolist())
    add(index[index["is_s11_exemplar"]]["policy_id"].tolist())
    for _, group in index.sort_values("screen_score", ascending=False, kind="mergesort").groupby("class_id", sort=True):
        add(group.head(2)["policy_id"].tolist())
    source_ok = set(index[index["has_full_dsl_source"]]["policy_id"].astype(str))
    return [policy_id for policy_id in selected if policy_id in source_ok][:max_count]


def _trace_rng_seed(policy_id: str, config: AtlasTraceConfig, seed: int = DEFAULT_ATLAS_SEED) -> int:
    digest = hashlib.sha256(f"{seed}|{policy_id}|{config.config_id}|{config.seed}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def trace_policy(
    *,
    policy_id: str,
    policy_name: str,
    dsl_source: str,
    class_id: str,
    atlas_role: str,
    config: AtlasTraceConfig = AtlasTraceConfig(),
    seed: int = DEFAULT_ATLAS_SEED,
) -> dict[str, Any]:
    """Run one compact deterministic DSL trace preview."""

    policy = parse_policy(dsl_source)
    values = initial_values_for_profile(config.array_size, config.seed, config.input_profile)
    schedule = actor_schedule(config.array_size, config.event_cap, config.seed)
    rng = random.Random(_trace_rng_seed(policy_id, config, seed=seed))
    interpreter = DSLInterpreter(policy)
    statuses: tuple[str, ...] = tuple("ACTIVE" for _ in values)
    ideal_position: int | None = None
    current_values = values
    compare_count = 0
    swap_count = 0
    update_count = 0
    wait_count = 0
    snapshots: list[dict[str, Any]] = []

    def snapshot(event: int, actor: int | None, action: str) -> None:
        metrics = sortedness_metrics(current_values)
        snapshots.append(
            {
                "event": int(event),
                "actor": None if actor is None else int(actor),
                "action": action,
                "values": list(current_values),
                "inversionCount": int(metrics["inversion_count"]),
                "inversionSortedness": round(float(metrics["inversion_sortedness"]), 6),
            }
        )

    checkpoints = set(int(value) for value in config.checkpoints)
    snapshot(0, None, "initial")
    for event, actor_index in enumerate(schedule, start=1):
        state = DSLArrayState(values=current_values, statuses=statuses, actor_index=int(actor_index), ideal_position=ideal_position)
        result = interpreter.step_state(state, rng)
        current_values = result.state_after.values
        statuses = tuple(result.state_after.statuses or statuses)
        ideal_position = result.state_after.ideal_position
        compare_count += int(bool(result.action.compare_counted))
        swap_count += int(result.action.action_type == "swap")
        update_count += int(result.action.action_type == "update_state")
        wait_count += int(result.action.action_type == "wait")
        if event in checkpoints:
            snapshot(event, int(actor_index), str(result.action.action_type))
    if snapshots[-1]["event"] != config.event_cap:
        snapshot(config.event_cap, int(schedule[-1]), "final")

    initial_metrics = sortedness_metrics(values)
    final_metrics = sortedness_metrics(current_values)
    return {
        "schema": ATLAS_SCHEMA,
        "experiment_id": "E03",
        "research_step_id": "S15",
        "policy_id": str(policy_id),
        "policy_name": str(policy_name),
        "class_id": str(class_id),
        "atlas_role": str(atlas_role),
        "config_id": config.config_id,
        "array_size": int(config.array_size),
        "seed": int(config.seed),
        "event_cap": int(config.event_cap),
        "input_profile": config.input_profile,
        "initial_values_json": stable_json(list(values)),
        "final_values_json": stable_json(list(current_values)),
        "snapshots_json": stable_json(snapshots),
        "snapshot_count": int(len(snapshots)),
        "initial_inversion_count": int(initial_metrics["inversion_count"]),
        "final_inversion_count": int(final_metrics["inversion_count"]),
        "initial_inversion_sortedness": float(initial_metrics["inversion_sortedness"]),
        "final_inversion_sortedness": float(final_metrics["inversion_sortedness"]),
        "inversion_sortedness_delta": float(final_metrics["inversion_sortedness"] - initial_metrics["inversion_sortedness"]),
        "final_is_sorted": bool(final_metrics["is_sorted"]),
        "compare_count": int(compare_count),
        "swap_count": int(swap_count),
        "update_count": int(update_count),
        "wait_count": int(wait_count),
        "work_count": int(compare_count + swap_count + update_count),
        "trace_digest": hashlib.sha256(stable_json(snapshots).encode("utf-8")).hexdigest(),
    }


def representative_trace_frame(index: pd.DataFrame, policy_ids: Sequence[str], *, config: AtlasTraceConfig = AtlasTraceConfig()) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    source_map = {str(row["policy_id"]): row for row in index.to_dict(orient="records")}
    for policy_id in policy_ids:
        row = source_map[str(policy_id)]
        rows.append(
            trace_policy(
                policy_id=str(policy_id),
                policy_name=str(row.get("policy_name", "")),
                dsl_source=str(row.get("dsl_source", "")),
                class_id=str(row.get("class_id", "")),
                atlas_role=str(row.get("atlas_role", "")),
                config=config,
            )
        )
    return pd.DataFrame(rows)


def mark_trace_availability(index: pd.DataFrame, traces: pd.DataFrame) -> pd.DataFrame:
    out = index.copy()
    trace_ids = set(traces["policy_id"].astype(str)) if not traces.empty else set()
    out["trace_available"] = out["policy_id"].astype(str).isin(trace_ids)
    out["trace_url"] = np.where(out["trace_available"], "e03_policy_atlas.html#trace-" + out["policy_id"].astype(str).str.replace(":", "-", regex=False), "")
    return out


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if pd.isna(value):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return value


def _records_for_html(frame: pd.DataFrame, columns: Sequence[str], *, max_source_chars: int = 600) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in frame[[column for column in columns if column in frame.columns]].to_dict(orient="records"):
        out = {key: _jsonable(value) for key, value in record.items()}
        if "dsl_source_excerpt" in out and isinstance(out["dsl_source_excerpt"], str):
            out["dsl_source_excerpt"] = out["dsl_source_excerpt"][:max_source_chars]
        records.append(out)
    return records


def atlas_digest(index: pd.DataFrame, traces: pd.DataFrame) -> str:
    payload = {
        "policyIds": sorted(index["policy_id"].astype(str).tolist()),
        "frontierIds": sorted(index[index["is_frontier_candidate"]]["policy_id"].astype(str).tolist()),
        "traceDigests": sorted(traces.get("trace_digest", pd.Series(dtype=str)).astype(str).tolist()),
    }
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def render_atlas_html(
    *,
    index: pd.DataFrame,
    traces: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    classic_analysis: pd.DataFrame,
    artifact_links: Mapping[str, str],
    figure_links: Mapping[str, str],
    digest: str,
) -> str:
    """Render a self-contained static atlas page with embedded policy data."""

    map_columns = [
        "policy_id",
        "policy_name",
        "class_id",
        "cautious_label",
        "atlas_role",
        "atlas_anchor",
        "embedding_x",
        "embedding_y",
        "screen_score",
        "screen_heldout_final_sortedness_mean",
        "screen_heldout_work_mean",
        "curated_rank",
        "is_frontier_candidate",
        "classic_landmark",
        "s08_archive_winner",
        "qd_candidate",
        "trace_available",
        "dsl_source_excerpt",
        "selection_reasons_json",
        "pareto_frontier_labels_json",
        "classic_position_classification",
        "s09_boundary_count",
        "s12_claim_count",
    ]
    trace_records = _records_for_html(traces, traces.columns.tolist(), max_source_chars=0)
    policy_records = _records_for_html(index, map_columns)
    class_records = _records_for_html(cluster_summary, cluster_summary.columns.tolist())
    classic_records = _records_for_html(classic_analysis, classic_analysis.columns.tolist())
    payload = {
        "schema": ATLAS_SCHEMA,
        "digest": digest,
        "generatedAt": "2026-07-01",
        "policies": policy_records,
        "traces": trace_records,
        "classes": class_records,
        "classicAnalysis": classic_records,
        "artifactLinks": dict(artifact_links),
        "figureLinks": dict(figure_links),
    }
    data_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_jsonable).replace("</", "<\\/")

    frontier_count = int(index["is_frontier_candidate"].sum())
    classic_count = int(index["classic_landmark"].sum())
    class_count = int(index["class_id"].nunique())
    trace_count = int(len(traces))
    cards = [
        ("Policies", f"{len(index):,}"),
        ("S11 Classes", str(class_count)),
        ("S14 Candidates", str(frontier_count)),
        ("Classic DSL Landmarks", str(classic_count)),
        ("Representative Traces", str(trace_count)),
    ]
    card_html = "\n".join(f"<div class='metric'><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>" for label, value in cards)
    artifact_html = "\n".join(
        f"<li><a href='{html.escape(path)}'>{html.escape(label)}</a></li>" for label, path in artifact_links.items()
    )
    figure_html = "\n".join(
        f"<li><a href='{html.escape(path)}'>{html.escape(label)}</a></li>" for label, path in figure_links.items()
    )
    highlighted = index[index["atlas_role"].isin(["frontier_candidate", "classic_landmark", "s11_exemplar"])].head(80)
    detail_sections = []
    trace_map = {str(row["policy_id"]): row for row in traces.to_dict(orient="records")}
    for row in highlighted.to_dict(orient="records"):
        policy_id = str(row["policy_id"])
        trace = trace_map.get(policy_id)
        trace_text = "No compact trace preview for this row."
        if trace is not None:
            trace_text = f"Trace final sortedness {float(trace['final_inversion_sortedness']):.3f}; snapshots {int(trace['snapshot_count'])}; digest {trace['trace_digest'][:12]}."
        detail_sections.append(
            f"""
            <section class="policy-detail" id="{html.escape(str(row['atlas_anchor']))}">
              <h3>{html.escape(policy_id)} <small>{html.escape(str(row.get('policy_name', '')))}</small></h3>
              <p><strong>Role:</strong> {html.escape(str(row.get('atlas_role', '')))} | <strong>Class:</strong> {html.escape(str(row.get('class_id', '')))} - {html.escape(str(row.get('cautious_label', '')))}</p>
              <p><strong>Screen score:</strong> {html.escape(str(round(float(row.get('screen_score', 0.0) or 0.0), 6)))} | <strong>S14 rank:</strong> {html.escape(str(row.get('curated_rank', '')))}</p>
              <p id="trace-{html.escape(policy_id.replace(':', '-'))}"><strong>Representative trace:</strong> {html.escape(trace_text)}</p>
              <pre>{html.escape(str(row.get('dsl_source_excerpt', '')))}</pre>
            </section>
            """
        )
    detail_html = "\n".join(detail_sections)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>E03 Policy Morphospace Atlas</title>
  <style>
    :root {{ --ink:#172026; --muted:#5f6b73; --line:#d9e1e7; --bg:#f7f9fb; --panel:#ffffff; --blue:#356d9d; --green:#3b7f5d; --gold:#9a6b21; --red:#9c3b3b; }}
    body {{ margin:0; font-family: system-ui, -apple-system, Segoe UI, sans-serif; color:var(--ink); background:var(--bg); }}
    header {{ padding:24px 32px 16px; background:var(--panel); border-bottom:1px solid var(--line); }}
    h1 {{ margin:0 0 8px; font-size:28px; letter-spacing:0; }}
    h2 {{ margin-top:28px; font-size:20px; }}
    h3 {{ margin:0 0 8px; font-size:16px; }}
    p {{ line-height:1.45; }}
    main {{ padding:20px 32px 40px; }}
    .summary {{ max-width:1160px; color:var(--muted); }}
    .metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; margin:18px 0; max-width:1100px; }}
    .metric {{ border:1px solid var(--line); background:var(--panel); padding:12px; border-radius:6px; }}
    .metric span {{ display:block; color:var(--muted); font-size:12px; }}
    .metric strong {{ display:block; font-size:24px; margin-top:4px; }}
    .toolbar {{ display:flex; gap:10px; flex-wrap:wrap; align-items:center; margin:14px 0; }}
    input, select, button {{ font:inherit; padding:7px 9px; border:1px solid var(--line); border-radius:6px; background:var(--panel); }}
    button {{ cursor:pointer; }}
    .layout {{ display:grid; grid-template-columns:minmax(420px,1fr) 360px; gap:18px; align-items:start; }}
    #map {{ width:100%; height:620px; border:1px solid var(--line); background:var(--panel); border-radius:6px; }}
    #detail {{ border:1px solid var(--line); background:var(--panel); border-radius:6px; padding:14px; min-height:280px; position:sticky; top:12px; }}
    .legend {{ display:flex; gap:10px; flex-wrap:wrap; color:var(--muted); font-size:12px; }}
    .dot {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:4px; vertical-align:-1px; }}
    table {{ border-collapse:collapse; width:100%; background:var(--panel); border:1px solid var(--line); }}
    th, td {{ text-align:left; padding:7px 8px; border-bottom:1px solid var(--line); font-size:13px; vertical-align:top; }}
    th {{ background:#eef3f6; position:sticky; top:0; z-index:1; }}
    .table-wrap {{ max-height:520px; overflow:auto; border:1px solid var(--line); border-radius:6px; }}
    pre {{ white-space:pre-wrap; word-break:break-word; background:#f2f5f7; border:1px solid var(--line); padding:10px; border-radius:6px; font-size:12px; }}
    a {{ color:var(--blue); }}
    .policy-detail {{ border-top:1px solid var(--line); padding-top:14px; margin-top:14px; }}
    .two-col {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }}
    @media (max-width: 920px) {{ main, header {{ padding-left:16px; padding-right:16px; }} .layout, .two-col {{ grid-template-columns:1fr; }} #detail {{ position:static; }} #map {{ height:460px; }} }}
  </style>
</head>
<body>
<header>
  <h1>E03 Policy Morphospace Atlas</h1>
  <p class="summary">Static audit atlas for the E03 DSL policy morphospace. It links S10 embeddings, S11 classes, S12 ablation evidence, S13 classic positions, S14 frontier candidates, source excerpts, compact trace previews, and caveats. Digest: <code>{html.escape(digest)}</code>.</p>
  <div class="metrics">{card_html}</div>
</header>
<main>
  <section>
    <h2>Policy Map</h2>
    <div class="toolbar">
      <label>Class <select id="classFilter"><option value="">All classes</option></select></label>
      <label>Role <select id="roleFilter"><option value="">All roles</option></select></label>
      <input id="searchBox" placeholder="Search policy ID or name" aria-label="Search policy ID or name">
      <button id="resetBtn">Reset</button>
    </div>
    <div class="legend">
      <span><i class="dot" style="background:var(--blue)"></i>ordinary/QD</span>
      <span><i class="dot" style="background:var(--green)"></i>S14 candidate</span>
      <span><i class="dot" style="background:var(--gold)"></i>classic landmark</span>
      <span><i class="dot" style="background:var(--red)"></i>S11 exemplar/archive</span>
    </div>
    <div class="layout">
      <svg id="map" role="img" aria-label="E03 behavior embedding map"></svg>
      <aside id="detail"><h3>Select a policy</h3><p>Click a point or table row to inspect source excerpts, metrics, class labels, frontier status, and trace availability.</p></aside>
    </div>
  </section>

  <section>
    <h2>Policy Index</h2>
    <p id="rowCount"></p>
    <div class="table-wrap"><table id="policyTable"><thead><tr><th>Policy</th><th>Role</th><th>Class</th><th>Score</th><th>Held-out</th><th>Links</th></tr></thead><tbody></tbody></table></div>
  </section>

  <section class="two-col">
    <div>
      <h2>Artifacts</h2>
      <ul>{artifact_html}</ul>
    </div>
    <div>
      <h2>Figures</h2>
      <ul>{figure_html}</ul>
    </div>
  </section>

  <section>
    <h2>Caveats</h2>
    <ul>
      <li>Atlas coordinates are exploratory S10 PCA coordinates over proxy DSL behavior, not causal or biological equivalence classes.</li>
      <li>S14 candidates are validated in the DSL proxy simulator under disorder and size perturbations; direct public-simulator Frozen Cell robustness, full DG trajectories, and chimeric aggregation are downstream.</li>
      <li>Classic interface wrappers remain exact public-method baselines, while the embedded classic points are DSL landmarks or shadows.</li>
      <li>Representative traces are compact preview runs on n=8, not replacements for S07/S14 sweeps.</li>
    </ul>
  </section>

  <section>
    <h2>Highlighted Policy Details</h2>
    {detail_html}
  </section>
</main>
<script id="atlas-data" type="application/json">{data_json}</script>
<script>
const atlas = JSON.parse(document.getElementById('atlas-data').textContent);
const policies = atlas.policies;
const traces = Object.fromEntries(atlas.traces.map(t => [t.policy_id, t]));
const classFilter = document.getElementById('classFilter');
const roleFilter = document.getElementById('roleFilter');
const searchBox = document.getElementById('searchBox');
const detail = document.getElementById('detail');
const svg = document.getElementById('map');
const tbody = document.querySelector('#policyTable tbody');
const rowCount = document.getElementById('rowCount');
const classes = [...new Set(policies.map(p => p.class_id).filter(Boolean))].sort();
const roles = [...new Set(policies.map(p => p.atlas_role).filter(Boolean))].sort();
for (const cls of classes) {{ const opt = document.createElement('option'); opt.value = cls; opt.textContent = cls; classFilter.appendChild(opt); }}
for (const role of roles) {{ const opt = document.createElement('option'); opt.value = role; opt.textContent = role; roleFilter.appendChild(opt); }}
function colorFor(p) {{
  if (p.is_frontier_candidate) return '#3b7f5d';
  if (p.classic_landmark) return '#9a6b21';
  if (p.atlas_role === 's11_exemplar' || p.s08_archive_winner) return '#9c3b3b';
  return '#356d9d';
}}
function radiusFor(p) {{
  if (p.is_frontier_candidate) return 5.5;
  if (p.classic_landmark) return 5.2;
  if (p.atlas_role === 's11_exemplar') return 4.4;
  return 2.4;
}}
function filtered() {{
  const cls = classFilter.value;
  const role = roleFilter.value;
  const query = searchBox.value.trim().toLowerCase();
  return policies.filter(p => (!cls || p.class_id === cls) && (!role || p.atlas_role === role) && (!query || (p.policy_id + ' ' + (p.policy_name || '')).toLowerCase().includes(query)));
}}
function showPolicy(p) {{
  const tr = traces[p.policy_id];
  const traceHtml = tr ? `<p><strong>Trace:</strong> final sortedness ${{Number(tr.final_inversion_sortedness).toFixed(3)}}, snapshots ${{tr.snapshot_count}}, sorted ${{tr.final_is_sorted}}</p>` : '<p><strong>Trace:</strong> no compact trace preview.</p>';
  detail.innerHTML = `<h3>${{p.policy_id}} <small>${{p.policy_name || ''}}</small></h3>
    <p><strong>Role:</strong> ${{p.atlas_role}} | <strong>Class:</strong> ${{p.class_id || ''}} - ${{p.cautious_label || ''}}</p>
    <p><strong>Score:</strong> ${{Number(p.screen_score || 0).toFixed(3)}} | <strong>Held-out:</strong> ${{Number(p.screen_heldout_final_sortedness_mean || 0).toFixed(3)}} | <strong>Work:</strong> ${{Number(p.screen_heldout_work_mean || 0).toFixed(2)}}</p>
    <p><strong>S14 rank:</strong> ${{p.curated_rank || 'n/a'}} | <strong>Classic position:</strong> ${{p.classic_position_classification || 'n/a'}}</p>
    ${{traceHtml}}
    <p><a href="#${{p.atlas_anchor}}">Static detail anchor</a></p>
    <pre>${{(p.dsl_source_excerpt || '').replace(/[&<>]/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;'}}[c]))}}</pre>`;
}}
function draw() {{
  const rows = filtered();
  const xs = policies.map(p => Number(p.embedding_x)).filter(Number.isFinite);
  const ys = policies.map(p => Number(p.embedding_y)).filter(Number.isFinite);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const w = svg.clientWidth || 800, h = svg.clientHeight || 620, pad = 28;
  svg.setAttribute('viewBox', `0 0 ${{w}} ${{h}}`);
  svg.innerHTML = '';
  for (const p of rows) {{
    const x = pad + (Number(p.embedding_x) - minX) / (maxX - minX || 1) * (w - 2 * pad);
    const y = h - pad - (Number(p.embedding_y) - minY) / (maxY - minY || 1) * (h - 2 * pad);
    const c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    c.setAttribute('cx', x); c.setAttribute('cy', y); c.setAttribute('r', radiusFor(p));
    c.setAttribute('fill', colorFor(p)); c.setAttribute('opacity', p.atlas_role === 'policy' ? '0.44' : '0.88');
    c.setAttribute('stroke', '#172026'); c.setAttribute('stroke-width', p.is_frontier_candidate || p.classic_landmark ? '0.8' : '0.2');
    c.addEventListener('click', () => showPolicy(p));
    const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
    title.textContent = `${{p.policy_id}} ${{p.policy_name || ''}} ${{p.class_id || ''}}`;
    c.appendChild(title);
    svg.appendChild(c);
  }}
  renderTable(rows);
}}
function renderTable(rows) {{
  const limit = 300;
  rowCount.textContent = `${{rows.length}} policies match the current filters. Showing first ${{Math.min(limit, rows.length)}} rows.`;
  tbody.innerHTML = '';
  for (const p of rows.slice(0, limit)) {{
    const tr = document.createElement('tr');
    tr.innerHTML = `<td><a href="#${{p.atlas_anchor}}">${{p.policy_id}}</a><br>${{p.policy_name || ''}}</td><td>${{p.atlas_role}}</td><td>${{p.class_id || ''}}</td><td>${{Number(p.screen_score || 0).toFixed(3)}}</td><td>${{Number(p.screen_heldout_final_sortedness_mean || 0).toFixed(3)}}</td><td>${{p.trace_available ? 'trace' : ''}} ${{p.is_frontier_candidate ? 'frontier' : ''}}</td>`;
    tr.addEventListener('click', () => showPolicy(p));
    tbody.appendChild(tr);
  }}
}}
classFilter.addEventListener('change', draw);
roleFilter.addEventListener('change', draw);
searchBox.addEventListener('input', draw);
document.getElementById('resetBtn').addEventListener('click', () => {{ classFilter.value=''; roleFilter.value=''; searchBox.value=''; draw(); }});
draw();
</script>
</body>
</html>
"""


def validation_frame(
    *,
    index: pd.DataFrame,
    traces: pd.DataFrame,
    html_path: Path,
    summary_path: Path,
    handoff_path: Path,
    artifact_paths: Mapping[str, Path],
    figure_paths: Mapping[str, Path],
    expected_policy_count: int,
    expected_frontier_count: int,
    unit_success: bool,
) -> pd.DataFrame:
    cases: list[dict[str, Any]] = []

    def add(name: str, success: bool, expected: str, observed: Any, notes: str) -> None:
        cases.append({"validation_case": name, "success": bool(success), "expected": expected, "observed": str(observed), "notes": notes})

    add("all_policy_ids_indexed", len(index) == expected_policy_count, str(expected_policy_count), len(index), "Atlas index should cover every S10 policy.")
    add("unique_policy_ids", index["policy_id"].nunique() == len(index), "all policy IDs unique", index["policy_id"].nunique(), "One atlas row per stable policy ID.")
    add("all_classes_present", index["class_id"].nunique() == 8, "8 S11 classes", index["class_id"].nunique(), "All S11 classes are represented.")
    add("frontier_candidates_linked", int(index["is_frontier_candidate"].sum()) == expected_frontier_count, str(expected_frontier_count), int(index["is_frontier_candidate"].sum()), "S14 curated candidates are flagged in the atlas.")
    add("classic_landmarks_linked", int(index["classic_landmark"].sum()) >= 6, ">=6 classic DSL landmarks", int(index["classic_landmark"].sum()), "Embedded Bubble/Insertion/Selection DSL landmarks are present.")
    add("source_coverage", bool(index["has_full_dsl_source"].all()), "all policies have DSL source", int(index["has_full_dsl_source"].sum()), "Source table links rules for all atlas policies.")
    add("representative_traces_present", len(traces) >= 30, ">=30 trace previews", len(traces), "S14 candidates, classics, and exemplars should have compact traces where source exists.")
    add("trace_ids_in_index", set(traces["policy_id"].astype(str)).issubset(set(index["policy_id"].astype(str))), "all trace IDs in index", len(set(traces["policy_id"].astype(str)) - set(index["policy_id"].astype(str))), "Trace previews link back to atlas rows.")
    trace_json_ok = bool(traces["snapshots_json"].map(lambda text: isinstance(json.loads(str(text)), list)).all()) if not traces.empty else False
    add("trace_snapshots_load", trace_json_ok, "all snapshots JSON parse", trace_json_ok, "Representative trace snapshots are machine-readable.")
    add("atlas_html_written", html_path.exists() and html_path.stat().st_size > 1000, "HTML atlas exists", html_path, "Static atlas page was written.")
    html_text = html_path.read_text(encoding="utf-8") if html_path.exists() else ""
    frontier_ids = index[index["is_frontier_candidate"]]["policy_id"].astype(str).tolist()
    add("atlas_html_contains_frontier_ids", all(policy_id in html_text for policy_id in frontier_ids), "all S14 IDs in HTML", len([policy_id for policy_id in frontier_ids if policy_id in html_text]), "Frontier candidates are embedded in the atlas data/detail sections.")
    add("summary_report_written", summary_path.exists() and summary_path.stat().st_size > 500, "summary markdown exists", summary_path, "Atlas summary report was written.")
    add("handoff_report_written", handoff_path.exists() and handoff_path.stat().st_size > 500, "handoff markdown exists", handoff_path, "Report-bundle handoff was written.")
    missing_artifacts = [str(path) for path in artifact_paths.values() if not path.exists() or path.stat().st_size == 0]
    add("artifact_links_resolve", not missing_artifacts, "all linked artifacts exist", missing_artifacts, "Atlas artifact links resolve on disk.")
    missing_figures = [str(path) for path in figure_paths.values() if not path.exists() or path.stat().st_size == 0]
    add("figure_links_resolve", not missing_figures, "all linked figures exist", missing_figures, "Referenced figure assets resolve on disk.")
    add("unit_tests_passed", bool(unit_success), "unit tests pass", unit_success, "S15 runner executed the E03 unit suite.")
    return pd.DataFrame(cases)
