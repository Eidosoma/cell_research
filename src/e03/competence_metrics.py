"""Competence-vector metrics for E03 policy morphospace work."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


SCHEMA_VERSION = "eidosoma.e03.competence_vector.v1"
CLASSIC_ALGORITHMS = ("bubble", "insertion", "selection")


@dataclass(frozen=True)
class CompetenceSourcePaths:
    """Required artifact locations for building S04 classic vectors."""

    policy_library_path: Path
    e01_dir: Path
    e02_dir: Path


class CompetenceInputError(RuntimeError):
    """Raised when required upstream inputs are missing or malformed."""


def stable_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def stable_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise CompetenceInputError(f"Required CSV input is missing: {path}")
    return pd.read_csv(path)


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise CompetenceInputError(f"Required Parquet input is missing: {path}")
    return pd.read_parquet(path)


def _as_float(value: Any) -> float:
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _bounded_0_1(value: float) -> float:
    if math.isnan(value):
        return math.nan
    return max(0.0, min(1.0, value))


def normalize_lower_better(values: Iterable[float]) -> list[float]:
    """Min-max normalize finite values so lower raw values receive higher scores."""

    raw = [_as_float(value) for value in values]
    finite = [value for value in raw if math.isfinite(value)]
    if not finite:
        return [math.nan for _ in raw]
    low = min(finite)
    high = max(finite)
    if math.isclose(low, high):
        return [1.0 if math.isfinite(value) else math.nan for value in raw]
    return [((high - value) / (high - low)) if math.isfinite(value) else math.nan for value in raw]


def normalize_higher_better(values: Iterable[float]) -> list[float]:
    """Min-max normalize finite values so higher raw values receive higher scores."""

    raw = [_as_float(value) for value in values]
    finite = [value for value in raw if math.isfinite(value)]
    if not finite:
        return [math.nan for _ in raw]
    low = min(finite)
    high = max(finite)
    if math.isclose(low, high):
        return [1.0 if math.isfinite(value) else math.nan for value in raw]
    return [((value - low) / (high - low)) if math.isfinite(value) else math.nan for value in raw]


def load_policy_library(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise CompetenceInputError(f"S03 policy library is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    policies = payload.get("policies")
    if not isinstance(policies, list) or not policies:
        raise CompetenceInputError(f"S03 policy library has no policies list: {path}")
    required = {"policyId", "algorithm", "representationType", "exactness", "direction"}
    for entry in policies:
        missing = required.difference(entry)
        if missing:
            raise CompetenceInputError(f"S03 policy entry missing keys {sorted(missing)}")
    return policies


def _algorithm_frame() -> pd.DataFrame:
    return pd.DataFrame({"algorithm": list(CLASSIC_ALGORITHMS)})


def _metric_row(efficiency: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = efficiency[(efficiency["count_metric"] == metric) & efficiency["algorithm"].isin(CLASSIC_ALGORITHMS)]
    if set(rows["algorithm"]) != set(CLASSIC_ALGORITHMS):
        raise CompetenceInputError(f"E01 efficiency table lacks all classic algorithms for metric {metric!r}")
    return rows[["algorithm", "cell_view_mean"]].rename(columns={"cell_view_mean": metric})


def _build_efficiency_context(e01_dir: Path) -> pd.DataFrame:
    efficiency = _read_csv(e01_dir / "tables/e01_efficiency_numeric_table.csv")
    frame = _algorithm_frame()
    for metric in ("swap_only_steps", "comparison_steps_observed", "compare_plus_swap_steps"):
        frame = frame.merge(_metric_row(efficiency, metric), on="algorithm", how="left")

    counts = _read_parquet(e01_dir / "results/e01_efficiency_counts.parquet")
    counts = counts[(counts["mode"] == "cell_view") & counts["algorithm"].isin(CLASSIC_ALGORITHMS)]
    final = (
        counts.groupby("algorithm", as_index=False)
        .agg(
            final_sortedness_percent_mean=("final_sortedness_percent", "mean"),
            baseline_run_count=("repeat_index", "count"),
        )
    )
    frame = frame.merge(final, on="algorithm", how="left")
    frame["efficiency_score"] = normalize_lower_better(frame["compare_plus_swap_steps"])
    frame["movement_energy_score"] = normalize_lower_better(frame["swap_only_steps"])
    frame["final_sortedness_score"] = frame["final_sortedness_percent_mean"].map(lambda value: _bounded_0_1(_as_float(value) / 100.0))
    return frame


def _build_frozen_context(e01_dir: Path) -> pd.DataFrame:
    frozen = _read_csv(e01_dir / "tables/e01_frozen_cell_robustness_numeric_table.csv")
    frozen = frozen[(frozen["mode"] == "cell_view") & frozen["algorithm"].isin(CLASSIC_ALGORITHMS)]
    if frozen.empty:
        raise CompetenceInputError("E01 frozen robustness table has no cell_view classic rows")
    summary = (
        frozen.groupby("algorithm", as_index=False)
        .agg(
            frozen_condition_count=("condition_id", "count"),
            frozen_mean_final_sortedness_percent=("mean_final_sortedness_percent", "mean"),
            frozen_mean_monotonicity_error=("mean_final_monotonicity_error", "mean"),
            frozen_worst_monotonicity_error=("mean_final_monotonicity_error", "max"),
        )
    )
    summary["frozen_cell_robustness_score"] = summary["frozen_mean_final_sortedness_percent"].map(
        lambda value: _bounded_0_1(_as_float(value) / 100.0)
    )
    return summary


def _build_dg_context(e01_dir: Path) -> pd.DataFrame:
    dg = _read_csv(e01_dir / "tables/e01_dg_numeric_table.csv")
    dg = dg[
        (dg["mode"] == "cell_view")
        & (dg["algorithm"].isin(CLASSIC_ALGORITHMS))
        & (dg["frozen_semantics"] == "none")
        & (dg["frozen_count"] == 0)
    ]
    if set(dg["algorithm"]) != set(CLASSIC_ALGORITHMS):
        raise CompetenceInputError("E01 DG table lacks unperturbed cell_view rows for all classics")
    summary = dg[
        [
            "algorithm",
            "mean_dg_primary",
            "mean_dg_total_ratio",
            "mean_dg_event_count",
        ]
    ].rename(
        columns={
            "mean_dg_primary": "dg_primary_mean",
            "mean_dg_total_ratio": "dg_total_ratio_mean",
            "mean_dg_event_count": "dg_event_count_mean",
        }
    )
    summary["dg_tendency_score"] = normalize_higher_better(summary["dg_primary_mean"])
    return summary


def _build_aggregation_context(e01_dir: Path) -> pd.DataFrame:
    aggregation = _read_csv(e01_dir / "tables/e01_aggregation_peak_table.csv")
    aggregation = aggregation[aggregation["is_negative_control"] == False].copy()  # noqa: E712
    records: list[dict[str, Any]] = []
    for algorithm in CLASSIC_ALGORITHMS:
        rows = aggregation[aggregation["algotype_mix"].str.contains(algorithm, regex=False)]
        records.append(
            {
                "algorithm": algorithm,
                "aggregation_mix_count": int(len(rows)),
                "aggregation_peak_delta_percent_mean": _as_float(rows["peak_minus_expected_random_left_neighbor_percent"].mean()),
                "aggregation_peak_percent_mean": _as_float(rows["mean_curve_peak_aggregation_left_neighbor_percent"].mean()),
                "aggregation_tendency_score": _bounded_0_1(_as_float(rows["peak_minus_expected_random_left_neighbor_percent"].mean()) / 100.0),
            }
        )
    return pd.DataFrame.from_records(records)


def _parse_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return {}
    parsed = json.loads(str(value))
    return parsed if isinstance(parsed, dict) else {}


def _parse_json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    parsed = json.loads(str(value))
    return parsed if isinstance(parsed, list) else []


def _build_conflict_context(e01_dir: Path) -> pd.DataFrame:
    conflict = _read_csv(e01_dir / "tables/e01_conflict_equilibria_summary.csv")
    records: list[dict[str, Any]] = []
    for algorithm in CLASSIC_ALGORITHMS:
        fractions: list[float] = []
        unique_fractions: list[float] = []
        duplicate_fractions: list[float] = []
        for row in conflict.to_dict(orient="records"):
            components = _parse_json_list(row.get("component_algorithms_json"))
            if algorithm not in components:
                continue
            counts = _parse_json_object(row.get("dominant_label_counts_json"))
            repetitions = max(1.0, _as_float(row.get("repetitions_observed")))
            fraction = _as_float(counts.get(algorithm, 0.0)) / repetitions
            fractions.append(fraction)
            if "opposite_unique" in str(row.get("algotype_mix", "")):
                unique_fractions.append(fraction)
            if "opposite_duplicate" in str(row.get("algotype_mix", "")):
                duplicate_fractions.append(fraction)
        records.append(
            {
                "algorithm": algorithm,
                "conflict_rows_observed": len(fractions),
                "conflict_dominance_score": _as_float(pd.Series(fractions).mean()) if fractions else math.nan,
                "conflict_unique_dominance_score": _as_float(pd.Series(unique_fractions).mean()) if unique_fractions else math.nan,
                "conflict_duplicate_dominance_score": _as_float(pd.Series(duplicate_fractions).mean()) if duplicate_fractions else math.nan,
            }
        )
    return pd.DataFrame.from_records(records)


def _build_path_curvature_context(e02_dir: Path) -> pd.DataFrame:
    summary_path = e02_dir / "research_steps/S09/e02_alternative_metric_summary.csv"
    alternative = _read_csv(summary_path)
    rows = alternative[
        (alternative["run_family"] == "unperturbed_baseline")
        & (alternative["metric_name"] == "sortedness_adjacency_distance")
        & (alternative["algotype_mix"].isin(CLASSIC_ALGORITHMS))
    ].copy()
    records: list[dict[str, Any]] = []
    for algorithm in CLASSIC_ALGORITHMS:
        match = rows[rows["algotype_mix"] == algorithm]
        curvature = _as_float(match["mean_path_curvature_ratio"].iloc[0]) if len(match) else math.nan
        records.append(
            {
                "algorithm": algorithm,
                "path_curvature_ratio": curvature,
                "path_directness_score": _bounded_0_1(1.0 / curvature) if math.isfinite(curvature) and curvature > 0 else math.nan,
                "path_curvature_source_n": int(match["n_runs"].iloc[0]) if len(match) else 0,
                "path_curvature_source_metric": "sortedness_adjacency_distance" if len(match) else None,
            }
        )
    return pd.DataFrame.from_records(records)


def _build_transfer_context(e01_dir: Path) -> pd.DataFrame:
    scaled = _read_parquet(e01_dir / "results/e01_scaled_replication.parquet")
    scaled = (
        scaled[
            (scaled["mode"] == "cell_view")
            & (scaled["algorithm"].isin(CLASSIC_ALGORITHMS))
            & (scaled["frozen_semantics"] == "none")
            & (scaled["frozen_count"] == 0)
            & (scaled["task_family"] == "efficiency")
        ]
        .copy()
    )
    records: list[dict[str, Any]] = []
    for algorithm in CLASSIC_ALGORITHMS:
        rows = scaled[scaled["algorithm"] == algorithm]
        lengths = sorted(int(value) for value in rows["array_length"].dropna().unique())
        available = len(lengths) >= 2
        score = math.nan
        if available:
            by_length = rows.groupby("array_length")["final_sortedness_percent"].mean()
            score = _bounded_0_1(_as_float(by_length.min()) / 100.0)
        records.append(
            {
                "algorithm": algorithm,
                "transfer_available": bool(available),
                "transfer_array_size_count": len(lengths),
                "transfer_array_sizes_json": json.dumps(lengths, separators=(",", ":")),
                "transfer_across_array_sizes_score": score,
            }
        )
    return pd.DataFrame.from_records(records)


def build_algorithm_metric_context(e01_dir: Path, e02_dir: Path) -> pd.DataFrame:
    """Build one E01/E02 metric-context row per classic algorithm."""

    frame = _build_efficiency_context(e01_dir)
    for context in (
        _build_frozen_context(e01_dir),
        _build_dg_context(e01_dir),
        _build_aggregation_context(e01_dir),
        _build_conflict_context(e01_dir),
        _build_path_curvature_context(e02_dir),
        _build_transfer_context(e01_dir),
    ):
        frame = frame.merge(context, on="algorithm", how="left")
    frame["metric_context_schema"] = SCHEMA_VERSION
    frame["metric_context_source"] = "E01 canonical metrics plus E02 alternative-metric curvature context"
    return frame


def _projection_for_policy(policy: Mapping[str, Any]) -> tuple[bool, str, str, str]:
    representation = str(policy["representationType"])
    exactness = str(policy["exactness"])
    if representation == "interface_wrapper" and exactness == "exact_public_method":
        return (
            True,
            "canonical_exact_public_method",
            "canonical_classic_baseline",
            "Metrics are directly assigned from E01/E02 algorithm rows for the exact S01 public-method wrapper.",
        )
    if representation == "dsl" and exactness == "exact_active_unfrozen_subset":
        return (
            False,
            "algorithm_family_context_only",
            "subset_proxy_not_full_policy_evaluation",
            "Metrics are inherited from the corresponding classic algorithm context only as a neighborhood anchor; this DSL policy is exact only for active/no-frozen subset fixtures.",
        )
    return (
        False,
        "approximate_shadow_context_only",
        "approximate_proxy_not_full_policy_evaluation",
        "Metrics are inherited from the corresponding classic algorithm context only as a rough landmark; S03 documents this policy as an approximate shadow.",
    )


def build_policy_competence_vectors(paths: CompetenceSourcePaths) -> pd.DataFrame:
    """Join S03 policies to the S04 classic metric context."""

    policies = load_policy_library(paths.policy_library_path)
    context = build_algorithm_metric_context(paths.e01_dir, paths.e02_dir)
    context_by_algorithm = context.set_index("algorithm").to_dict(orient="index")
    rows: list[dict[str, Any]] = []
    for policy in policies:
        algorithm = str(policy["algorithm"])
        if algorithm not in context_by_algorithm:
            raise CompetenceInputError(f"No metric context exists for algorithm {algorithm!r}")
        canonical, projection_scope, trust_level, caveat = _projection_for_policy(policy)
        row = {
            "schema_version": SCHEMA_VERSION,
            "policy_id": policy["policyId"],
            "algorithm": algorithm,
            "representation_type": policy["representationType"],
            "exactness": policy["exactness"],
            "direction": policy["direction"],
            "canonical_classic_baseline": canonical,
            "metric_projection_scope": projection_scope,
            "metric_trust_level": trust_level,
            "metric_assignment_caveat": caveat,
        }
        row.update(context_by_algorithm[algorithm])
        hash_payload = {
            key: row[key]
            for key in sorted(row)
            if key
            not in {
                "competence_vector_hash",
                "metric_assignment_caveat",
                "metric_context_source",
            }
        }
        row["competence_vector_hash"] = stable_hash(_json_safe_mapping(hash_payload))[:24]
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _json_safe_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, float) and math.isnan(value):
            safe[key] = None
        elif hasattr(value, "item"):
            item = value.item()
            safe[key] = None if isinstance(item, float) and math.isnan(item) else item
        else:
            safe[key] = value
    return safe


def _ordered_algorithms(frame: pd.DataFrame, value_column: str, ascending: bool) -> str:
    rows = frame[["algorithm", value_column]].sort_values([value_column, "algorithm"], ascending=[ascending, True])
    return ",".join(rows["algorithm"].tolist())


def validate_competence_vectors(vectors: pd.DataFrame) -> pd.DataFrame:
    """Return deterministic validation cases for the S04 classic vectors."""

    canonical = vectors[vectors["canonical_classic_baseline"] == True].copy()  # noqa: E712
    cases: list[dict[str, Any]] = []

    def add(case: str, success: bool, expected: str, observed: Any, notes: str) -> None:
        cases.append(
            {
                "validation_case": case,
                "success": bool(success),
                "expected": expected,
                "observed": str(observed),
                "notes": notes,
            }
        )

    add(
        "all_s03_policies_present",
        len(vectors) == 9 and vectors["policy_id"].nunique() == 9,
        "9 unique policy IDs from S03",
        f"{len(vectors)} rows, {vectors['policy_id'].nunique()} unique IDs",
        "Confirms the vector table covers the full S03 classic policy library.",
    )
    add(
        "canonical_interface_policy_count",
        len(canonical) == 3 and set(canonical["algorithm"]) == set(CLASSIC_ALGORITHMS),
        "3 exact interface-wrapper canonical baselines",
        canonical[["policy_id", "algorithm"]].to_dict(orient="records"),
        "Only exact public-method wrappers are treated as canonical baselines.",
    )
    add(
        "canonical_final_sortedness_matches_e01",
        bool((canonical["final_sortedness_percent_mean"].round(8) == 100.0).all()),
        "all canonical classics have 100% final Sortedness in E01 efficiency runs",
        canonical[["algorithm", "final_sortedness_percent_mean"]].to_dict(orient="records"),
        "Uses E01 cell-view efficiency counts.",
    )
    compare_order = _ordered_algorithms(canonical, "compare_plus_swap_steps", ascending=True)
    add(
        "compare_plus_efficiency_order_matches_e01",
        compare_order == "insertion,bubble,selection",
        "insertion,bubble,selection",
        compare_order,
        "Lower compare-plus-swap steps are more efficient.",
    )
    dg_order = _ordered_algorithms(canonical, "dg_primary_mean", ascending=False)
    add(
        "dg_tendency_order_matches_e01",
        dg_order == "selection,insertion,bubble",
        "selection,insertion,bubble",
        dg_order,
        "Higher E01 DG primary means stronger delayed-gratification tendency.",
    )
    frozen_order = _ordered_algorithms(canonical, "frozen_mean_monotonicity_error", ascending=True)
    add(
        "frozen_error_order_matches_e01",
        frozen_order == "selection,bubble,insertion",
        "selection,bubble,insertion",
        frozen_order,
        "Lower E01 mean final monotonicity error across passive/stuck counts is more robust.",
    )
    movement = canonical.set_index("algorithm")["swap_only_steps"].to_dict()
    movement_success = (
        math.isclose(_as_float(movement.get("bubble")), _as_float(movement.get("insertion")), rel_tol=0.0, abs_tol=1e-9)
        and _as_float(movement.get("selection")) < _as_float(movement.get("bubble"))
    )
    add(
        "movement_energy_order_matches_e01",
        movement_success,
        "selection lower movement cost; bubble and insertion tied",
        movement,
        "Movement-energy proxy is E01 cell-view swap-only steps.",
    )
    conflict_order = _ordered_algorithms(canonical, "conflict_dominance_score", ascending=False)
    add(
        "conflict_dominance_proxy_order",
        conflict_order == "bubble,selection,insertion",
        "bubble,selection,insertion",
        conflict_order,
        "Conflict dominance is a sparse E01 opposite-goal proxy using dominant-label counts.",
    )
    add(
        "path_curvature_context_present",
        bool(canonical["path_curvature_ratio"].notna().all() and (canonical["path_curvature_source_n"] > 0).all()),
        "E02 sortedness-adjacency curvature exists for all canonical classics",
        canonical[["algorithm", "path_curvature_ratio", "path_curvature_source_n"]].to_dict(orient="records"),
        "This is an E02 ten-run alternative-metric context, not a replacement for E01 final Sortedness.",
    )
    transfer_available = canonical["transfer_available"].astype(bool).tolist()
    add(
        "transfer_marked_unavailable_for_single_size_context",
        transfer_available == [False, False, False],
        "transfer unavailable because upstream scaled context has only one array length",
        transfer_available,
        "The transfer field is part of the frozen schema but not estimated from single-size evidence.",
    )
    return pd.DataFrame.from_records(cases)


def vector_column_spec() -> list[dict[str, str]]:
    """Human-readable schema for S04 and downstream S05-S08 consumers."""

    return [
        {"column": "policy_id", "type": "string", "meaning": "Stable S03 policy identifier."},
        {"column": "algorithm", "type": "string", "meaning": "Classic algorithm family: bubble, insertion, or selection."},
        {"column": "representation_type", "type": "string", "meaning": "S03 representation, either interface_wrapper or dsl."},
        {"column": "exactness", "type": "string", "meaning": "S03 exactness classification."},
        {"column": "canonical_classic_baseline", "type": "bool", "meaning": "True only for exact public-method interface wrappers."},
        {"column": "metric_projection_scope", "type": "string", "meaning": "Whether metrics are canonical or inherited context for a DSL landmark."},
        {"column": "final_sortedness_score", "type": "float", "meaning": "E01 final Sortedness percent divided by 100."},
        {"column": "efficiency_score", "type": "float", "meaning": "Min-max lower-is-better score from E01 compare-plus-swap steps."},
        {"column": "frozen_cell_robustness_score", "type": "float", "meaning": "Mean E01 final Frozen Cell Sortedness percent divided by 100."},
        {"column": "dg_tendency_score", "type": "float", "meaning": "Min-max higher-is-stronger score from E01 DG primary."},
        {"column": "aggregation_tendency_score", "type": "float", "meaning": "Mean same-goal chimera aggregation peak delta above random divided by 100."},
        {"column": "conflict_dominance_score", "type": "float", "meaning": "Mean dominant-label fraction in E01 opposite-goal conflict rows containing the algorithm."},
        {"column": "movement_energy_score", "type": "float", "meaning": "Min-max lower-is-better score from E01 swap-only steps."},
        {"column": "path_directness_score", "type": "float", "meaning": "1/path_curvature_ratio from E02 sortedness-adjacency alternative metric."},
        {"column": "transfer_across_array_sizes_score", "type": "float|null", "meaning": "Minimum final Sortedness across array sizes, available only when at least two sizes exist."},
        {"column": "competence_vector_hash", "type": "string", "meaning": "Stable hash prefix over policy identity and metric values."},
    ]
