"""E06 S07 mosaic-formation classification over S06 dominance contests."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from e02_deterministic_simulator.metrics import aggregation

from .arrangements import interface_count, mean_positions, run_lengths
from .dominance import (
    DEFAULT_S05_BOUNDARY_CAVEATS_PATH,
    UNRESOLVED_WINNER,
    artifact_records,
    load_boundary_caveats,
    predefined_winner_criteria,
    score_dominance_results,
)
from .goals import augment_goal_results
from .mixtures import (
    DEFAULT_PANEL_PATH,
    compact_json,
    compact_result_csv,
    git_value,
    load_ready_panel,
    parse_json_maybe,
    run_conditions,
    write_json,
)
from .panel import CLAIM_BOUNDARY


STEP_ID = "S07"
STEP_NUMBER = 7
MOSAIC_CLASSIFIER_VERSION = "e06_s07_mosaic_classifier.v1"
DEFAULT_S06_MOSAICS_PATH = Path("/artifacts/results/e06_dominance_mosaics.parquet")
DEFAULT_S06_CONTESTS_PATH = Path("/artifacts/results/e06_dominance_contests.parquet")
DEFAULT_S06_PAIR_SUMMARY_PATH = Path("/artifacts/research_steps/S06/dominance_pair_summary.parquet")
DEFAULT_S06_CONTEXT_SUMMARY_PATH = Path("/artifacts/research_steps/S06/dominance_context_summary.parquet")
DEFAULT_S06_WINNER_CRITERIA_PATH = Path("/artifacts/research_steps/S06/dominance_winner_criteria.json")
REQUESTED_MOSAIC_CLASSES: tuple[str, ...] = (
    "homogeneous",
    "patchy",
    "layered",
    "polarized",
    "oscillatory",
    "frozen_conflict",
    "mixed",
)
STOP_SENSITIVITY_CAPS: tuple[int, int] = (3000, 9000)


HAND_LABELED_EXEMPLARS: tuple[dict[str, str], ...] = (
    {
        "conditionId": "s06_p000_o0_50_50_alternating_duplicate_1_10_x10_seed0",
        "expectedS07MosaicClass": "homogeneous",
        "handLabelRationale": "completed duplicate-value alternating row with high target quality and high interface density",
    },
    {
        "conditionId": "s06_p000_o1_50_50_random_duplicate_1_10_x10_seed0",
        "expectedS07MosaicClass": "homogeneous",
        "handLabelRationale": "completed duplicate-value random row with high target quality and mixed final labels",
    },
    {
        "conditionId": "s06_p000_o0_75_25_graft_like_reversed_unique_seed0",
        "expectedS07MosaicClass": "patchy",
        "handLabelRationale": "resource-capped graft-like row with high aggregation but multiple retained patches",
    },
    {
        "conditionId": "s06_p001_o0_75_25_graft_like_reversed_unique_seed0",
        "expectedS07MosaicClass": "patchy",
        "handLabelRationale": "resource-capped graft-like reversed row with high aggregation and non-layered interfaces",
    },
    {
        "conditionId": "s06_p004_o0_25_75_contiguous_patch_reversed_unique_seed0",
        "expectedS07MosaicClass": "layered",
        "handLabelRationale": "near-complete segregation with one low-density interface",
    },
    {
        "conditionId": "s06_p005_o1_25_75_contiguous_patch_reversed_unique_seed0",
        "expectedS07MosaicClass": "layered",
        "handLabelRationale": "near-complete contiguous two-layer state under reversed values",
    },
    {
        "conditionId": "s06_p004_o0_25_75_contiguous_patch_random_unique_seed0",
        "expectedS07MosaicClass": "polarized",
        "handLabelRationale": "contiguous random-value row with separated policy centers and more than one boundary",
    },
    {
        "conditionId": "s06_p004_o0_75_25_contiguous_patch_random_unique_seed0",
        "expectedS07MosaicClass": "polarized",
        "handLabelRationale": "highly separated policy centers without the single-interface layered pattern",
    },
    {
        "conditionId": "s06_p000_o0_50_50_contiguous_patch_random_unique_seed0",
        "expectedS07MosaicClass": "oscillatory",
        "handLabelRationale": "resource-capped high-interface row with low target quality and high motion-persistence proxy",
    },
    {
        "conditionId": "s06_p003_o1_75_25_graft_like_random_unique_seed0",
        "expectedS07MosaicClass": "oscillatory",
        "handLabelRationale": "resource-capped handoff/frontier row with low target quality and high persistence proxy",
    },
    {
        "conditionId": "s06_p004_o0_50_50_alternating_reversed_unique_seed0",
        "expectedS07MosaicClass": "frozen_conflict",
        "handLabelRationale": "no-move opposite-goal row with maximal interface density and strong dominance proxy",
    },
    {
        "conditionId": "s06_p004_o0_25_75_random_reversed_unique_seed0",
        "expectedS07MosaicClass": "frozen_conflict",
        "handLabelRationale": "no-move reversed-value row with unresolved target quality and strong dominance proxy",
    },
    {
        "conditionId": "s06_p000_o0_25_75_random_random_unique_seed0",
        "expectedS07MosaicClass": "mixed",
        "handLabelRationale": "completed but low target-quality row that should not be overcalled homogeneous",
    },
    {
        "conditionId": "s06_p000_o0_50_50_graft_like_duplicate_1_10_x10_seed0",
        "expectedS07MosaicClass": "mixed",
        "handLabelRationale": "completed graft-like duplicate row with high target quality but low interface density",
    },
)


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return float(number) if np.isfinite(number) else default


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def load_required_table(path: Path, required_columns: Sequence[str], label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    table = pd.read_parquet(path)
    missing = sorted(set(required_columns) - set(table.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")
    if table.empty:
        raise ValueError(f"{label} is empty: {path}")
    return table


def load_s06_mosaics(path: Path = DEFAULT_S06_MOSAICS_PATH) -> pd.DataFrame:
    required = [
        "conditionId",
        "s06PairKey",
        "pairCategory",
        "ratioLabel",
        "arrangementType",
        "valueProfile",
        "policyIdsJson",
        "initialPanelPolicyIdsJson",
        "finalPanelPolicyIdsJson",
        "runSucceeded",
        "completed",
        "stopReason",
        "finalAggregation",
        "finalInterfaceDensity",
        "finalTargetQualityScore",
        "dominanceProxyScore",
        "winnerPolicyId",
        "winnerConfidenceClass",
        "oscillationRiskProxy",
        "metricBoundary",
        "metricBoundaryCaveatFlagsJson",
    ]
    return load_required_table(path, required, "S06 dominance mosaics")


def mosaic_rule_definitions() -> pd.DataFrame:
    rows = [
        {
            "s07MosaicClass": "homogeneous",
            "ruleOrder": 1,
            "rule": "completed value-order state with high target quality and high final interface density",
            "thresholdsJson": compact_json({"completed": True, "finalTargetQualityScoreMin": 0.70, "finalInterfaceDensityMin": 0.35}),
            "metricBoundary": "Homogeneous means policy labels remain spatially interspersed after value sorting; it is not biological homogeneity.",
        },
        {
            "s07MosaicClass": "oscillatory",
            "ruleOrder": 2,
            "rule": "resource-capped row with high motion-persistence proxy, low target quality, and retained interfaces",
            "thresholdsJson": compact_json({"oscillationRiskProxyMin": 0.75, "finalTargetQualityScoreMaxExclusive": 0.55, "finalInterfaceDensityMin": 0.25}),
            "metricBoundary": "Oscillatory is a resource-cap and final-state proxy; no trajectory-cycle detector is available in S07.",
        },
        {
            "s07MosaicClass": "frozen_conflict",
            "ruleOrder": 3,
            "rule": "no-move local equilibrium with poor or unresolved target quality and nontrivial dominance pressure",
            "thresholdsJson": compact_json({"stopReason": "no_cell_can_move_after_two_checks", "finalTargetQualityScoreMaxExclusive": 0.70, "dominanceProxyScoreMin": 0.25}),
            "metricBoundary": "Frozen conflict is a simulator stop-state proxy under opposite goals.",
        },
        {
            "s07MosaicClass": "layered",
            "ruleOrder": 4,
            "rule": "very high final aggregation, at most two final policy interfaces, and a long contiguous run",
            "thresholdsJson": compact_json({"finalAggregationMin": 0.95, "finalInterfaceCountMax": 2, "longestRunFractionMin": 0.45}),
            "metricBoundary": "Layering is a one-dimensional policy-label morphology.",
        },
        {
            "s07MosaicClass": "polarized",
            "ruleOrder": 5,
            "rule": "high policy-center separation plus high aggregation without meeting the stricter layered rule",
            "thresholdsJson": compact_json({"policyPositionSeparationMin": 0.42, "finalAggregationMin": 0.75}),
            "metricBoundary": "Polarization is based on one-dimensional mean policy positions.",
        },
        {
            "s07MosaicClass": "patchy",
            "ruleOrder": 6,
            "rule": "high aggregation or a moderately long run with low final interface density",
            "thresholdsJson": compact_json({"finalAggregationMin": 0.80, "or": {"longestRunFractionMin": 0.30, "finalInterfaceDensityMax": 0.35}}),
            "metricBoundary": "Patchiness is computed from final labels, not from a continuous tissue boundary.",
        },
        {
            "s07MosaicClass": "mixed",
            "ruleOrder": 7,
            "rule": "usable final state that does not clear the more specific morphology rules",
            "thresholdsJson": compact_json({}),
            "metricBoundary": "Mixed is the residual class and may contain multiple unresolved subtypes.",
        },
    ]
    return pd.DataFrame(rows)


def _motion_persistence_proxy(row: Mapping[str, Any]) -> float:
    stop = str(row.get("stopReason", "")).lower()
    if _safe_bool(row.get("completed", False)):
        return 0.0
    if "max_activation" in stop:
        return 1.0
    if stop.startswith("max_"):
        return 0.75
    if "no_cell_can_move" in stop or "no_move" in stop:
        return 0.15
    return 0.5


def _quasi_stable_state_class(row: Mapping[str, Any], target_q: float, interface_density: float) -> str:
    stop = str(row.get("stopReason", "")).lower()
    if _safe_bool(row.get("completed", False)):
        return "completed_value_order"
    if "no_cell_can_move" in stop or "no_move" in stop:
        return "local_no_move_equilibrium"
    if "max_activation" in stop and target_q < 0.55 and interface_density >= 0.25:
        return "resource_capped_motion_proxy"
    if stop.startswith("max_"):
        return "resource_capped_partial"
    return str(row.get("finalStateClass", "partial") or "partial")


def _position_separation(final_ids: Sequence[str], policy_ids: Sequence[str]) -> float:
    if len(policy_ids) < 2 or not final_ids:
        return float("nan")
    means = mean_positions(final_ids, policy_ids)
    values = [safe_float(means.get(str(pid))) for pid in policy_ids[:2]]
    if not all(np.isfinite(value) for value in values):
        return float("nan")
    return float(abs(values[0] - values[1]))


def _feature_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    policy_ids = [str(item) for item in parse_json_maybe(row.get("policyIdsJson", ""), [])]
    initial_ids = [str(item) for item in parse_json_maybe(row.get("initialPanelPolicyIdsJson", ""), [])]
    final_ids = [str(item) for item in parse_json_maybe(row.get("finalPanelPolicyIdsJson", ""), [])]
    n = len(final_ids) or int(row.get("n", 0) or 0)
    lengths = run_lengths(final_ids)
    final_interfaces = interface_count(final_ids) if final_ids else int(row.get("finalInterfaceCount", 0) or 0)
    initial_interfaces = interface_count(initial_ids) if initial_ids else int(row.get("initialInterfaceCount", 0) or 0)
    final_density = float(final_interfaces / max(1, len(final_ids) - 1)) if final_ids else safe_float(row.get("finalInterfaceDensity"))
    initial_density = float(initial_interfaces / max(1, len(initial_ids) - 1)) if initial_ids else float("nan")
    final_agg = aggregation(final_ids) if final_ids else safe_float(row.get("finalAggregation"))
    target_q = safe_float(row.get("finalTargetQualityScore"))
    dominance = safe_float(row.get("dominanceProxyScore"))
    longest = max(lengths) if lengths else 0
    payload = {
        "n": int(n),
        "policyCount": int(len(policy_ids)),
        "initialInterfaceCountS07": int(initial_interfaces),
        "initialInterfaceDensityS07": initial_density,
        "finalInterfaceCountS07": int(final_interfaces),
        "finalInterfaceDensityS07": final_density,
        "interfaceCountDeltaS07": int(final_interfaces - initial_interfaces),
        "finalRunCount": int(len(lengths)),
        "finalLongestRun": int(longest),
        "longestRunFraction": float(longest / max(1, len(final_ids))),
        "finalMeanRunLength": float(np.mean(lengths)) if lengths else float("nan"),
        "policyPositionSeparation": _position_separation(final_ids, policy_ids),
        "finalAggregationS07": final_agg,
        "finalTargetQualityScoreS07": target_q,
        "dominanceProxyScoreS07": dominance,
        "motionPersistenceProxy": _motion_persistence_proxy(row),
        "quasiStableStateClass": _quasi_stable_state_class(row, target_q, final_density),
        "finalPolicyCountsJsonS07": compact_json(dict(Counter(final_ids))) if final_ids else compact_json({}),
    }
    return payload


def _classify_row(row: Mapping[str, Any]) -> dict[str, Any]:
    features = _feature_payload(row)
    completed = _safe_bool(row.get("completed", False))
    succeeded = _safe_bool(row.get("runSucceeded", False))
    stop = str(row.get("stopReason", "")).lower()
    target_q = safe_float(features["finalTargetQualityScoreS07"])
    dominance = safe_float(features["dominanceProxyScoreS07"])
    agg = safe_float(features["finalAggregationS07"])
    density = safe_float(features["finalInterfaceDensityS07"])
    interfaces = int(features["finalInterfaceCountS07"])
    longest_fraction = safe_float(features["longestRunFraction"])
    separation = safe_float(features["policyPositionSeparation"])
    winner_conf = str(row.get("winnerConfidenceClass", ""))
    caveats = [
        "classification_uses_final_state_policy_labels",
        "opposite_goal_computational_proxy",
        "metric_boundary_caveats_inherited_from_s06",
    ]

    if not succeeded:
        label = "run_failed"
        confidence = "high"
        rationale = "simulation did not succeed, so no morphology class was assigned"
    elif completed and target_q >= 0.70 and density >= 0.35:
        label = "homogeneous"
        confidence = "high" if target_q >= 0.85 and density >= 0.45 else "moderate"
        rationale = "completed high-quality value order while policy labels stayed highly interspersed"
    elif (
        features["motionPersistenceProxy"] >= 0.75
        and target_q < 0.55
        and density >= 0.25
        and ("max_" in stop or safe_float(row.get("oscillationRiskProxy")) >= 0.75)
    ):
        label = "oscillatory"
        confidence = "moderate"
        rationale = "resource-capped row retained interfaces with low target quality and high motion-persistence proxy"
        caveats.append("oscillation_proxy_not_trajectory_cycle_detection")
    elif ("no_cell_can_move" in stop or "no_move" in stop) and (target_q < 0.70 or winner_conf == "unresolved") and dominance >= 0.25:
        label = "frozen_conflict"
        confidence = "high" if dominance >= 0.75 else "moderate"
        rationale = "local no-move state retained conflict pressure without high target quality"
        caveats.append("frozen_conflict_stop_reason_proxy")
    elif agg >= 0.95 and interfaces <= 2 and longest_fraction >= 0.45:
        label = "layered"
        confidence = "high" if agg >= 0.98 and interfaces <= 1 else "moderate"
        rationale = "final labels formed a near-contiguous low-interface layer"
    elif separation >= 0.42 and agg >= 0.75:
        label = "polarized"
        confidence = "high" if separation >= 0.47 else "moderate"
        rationale = "policy mean positions separated strongly without satisfying the stricter layered rule"
    elif agg >= 0.80 or (longest_fraction >= 0.30 and density <= 0.35):
        label = "patchy"
        confidence = "moderate"
        rationale = "final labels retained aggregated patches without a clean two-layer state"
    else:
        label = "mixed"
        confidence = "moderate"
        rationale = "usable final state did not clear the specific homogeneous, patchy, layered, polarized, oscillatory, or frozen-conflict rules"

    if str(row.get("valueProfile", "")) == "duplicate_1_10_x10":
        caveats.append("duplicate_value_target_ties")
    if safe_float(row.get("oscillationRiskProxy")) >= 0.75:
        caveats.append("resource_cap_unresolved_dynamics")
    inherited = parse_json_maybe(row.get("metricBoundaryCaveatFlagsJson", ""), [])
    if isinstance(inherited, Sequence) and not isinstance(inherited, str):
        caveats.extend(str(item) for item in inherited)
    return {
        **features,
        "s07MosaicClass": label,
        "s07MosaicClassConfidence": confidence,
        "s07MosaicRationale": rationale,
        "s07MosaicCaveatsJson": compact_json(sorted(set(caveats))),
        "s07ClassifierVersion": MOSAIC_CLASSIFIER_VERSION,
    }


def classify_mosaic_formations(
    s06_mosaics: pd.DataFrame,
    pair_summary: pd.DataFrame | None = None,
    context_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    classified_rows: list[dict[str, Any]] = []
    pair_lookup: dict[str, dict[str, Any]] = {}
    if pair_summary is not None and not pair_summary.empty:
        pair_lookup = {str(row["s06PairKey"]): row for row in pair_summary.to_dict(orient="records")}
    for row in s06_mosaics.to_dict(orient="records"):
        out = {**row, **_classify_row(row)}
        pair = pair_lookup.get(str(row.get("s06PairKey", "")), {})
        out["s06ContextDependencyClass"] = str(pair.get("contextDependencyClass", ""))
        out["s06PairModalWinnerPolicyId"] = str(pair.get("modalWinnerPolicyId", ""))
        out["s06PairModalWinnerShare"] = safe_float(pair.get("modalWinnerShare"))
        out["s06PairUnresolvedCount"] = int(pair.get("unresolvedCount", 0) or 0)
        out["s06PairMeanDominanceProxyScore"] = safe_float(pair.get("meanDominanceProxyScore"))
        out["s06PairMeanOscillationRiskProxy"] = safe_float(pair.get("meanOscillationRiskProxy"))
        classified_rows.append(out)
    out_df = pd.DataFrame(classified_rows)
    if context_summary is not None and not context_summary.empty:
        context_cols = [
            "ratioLabel",
            "arrangementType",
            "valueProfile",
            "perturbationType",
            "meanDominanceProxyScore",
            "meanFinalTargetQualityScore",
            "unresolvedRate",
            "meanOscillationRiskProxy",
        ]
        available = [column for column in context_cols if column in context_summary.columns]
        renamed = context_summary[available].rename(
            columns={
                "meanDominanceProxyScore": "s06ContextMeanDominanceProxyScore",
                "meanFinalTargetQualityScore": "s06ContextMeanFinalTargetQualityScore",
                "unresolvedRate": "s06ContextUnresolvedRate",
                "meanOscillationRiskProxy": "s06ContextMeanOscillationRiskProxy",
            }
        )
        merge_keys = [key for key in ["ratioLabel", "arrangementType", "valueProfile", "perturbationType"] if key in renamed.columns and key in out_df.columns]
        if merge_keys:
            out_df = out_df.merge(renamed, on=merge_keys, how="left")
    return out_df.reset_index(drop=True)


def build_priority_candidates(classified: pd.DataFrame, *, max_rows: int = 120) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in classified.to_dict(orient="records"):
        tags: list[str] = []
        score = 0.0
        dependency = str(row.get("s06ContextDependencyClass", ""))
        if "context_dependent" in dependency:
            tags.append("context_dependent_pair")
            score += 3.0
        if str(row.get("winnerPolicyId", "")) == UNRESOLVED_WINNER or str(row.get("winnerConfidenceClass", "")) == "unresolved":
            tags.append("unresolved_winner")
            score += 2.5
        if safe_float(row.get("s06PairUnresolvedCount")) > 0:
            tags.append("pair_has_unresolved_contexts")
            score += min(1.5, safe_float(row.get("s06PairUnresolvedCount")) / 12.0)
        if safe_float(row.get("dominanceProxyScore")) >= 0.35:
            tags.append("dominance_sensitive")
            score += min(1.5, safe_float(row.get("dominanceProxyScore")))
        if str(row.get("s07MosaicClass", "")) in {"oscillatory", "frozen_conflict", "layered", "polarized"}:
            tags.append(f"class_{row.get('s07MosaicClass')}")
            score += 1.0
        if safe_float(row.get("oscillationRiskProxy")) >= 0.75:
            tags.append("resource_cap_unresolved_dynamics")
            score += 0.75
        if safe_float(row.get("finalTargetQualityScore")) < 0.55:
            tags.append("low_target_quality")
            score += 0.5
        rows.append(
            {
                "conditionId": str(row["conditionId"]),
                "s06PairKey": str(row.get("s06PairKey", "")),
                "pairCategory": str(row.get("pairCategory", "")),
                "ratioLabel": str(row.get("ratioLabel", "")),
                "arrangementType": str(row.get("arrangementType", "")),
                "valueProfile": str(row.get("valueProfile", "")),
                "s07MosaicClass": str(row.get("s07MosaicClass", "")),
                "winnerPolicyId": str(row.get("winnerPolicyId", "")),
                "winnerConfidenceClass": str(row.get("winnerConfidenceClass", "")),
                "s06ContextDependencyClass": dependency,
                "s07PriorityScore": float(score),
                "s07PriorityTagsJson": compact_json(sorted(set(tags))),
            }
        )
    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["s07PriorityScore", "s06PairKey", "conditionId"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    out.insert(0, "s07PriorityRank", np.arange(1, len(out) + 1, dtype=int))
    return out.head(int(max_rows)).reset_index(drop=True)


def build_hand_labeled_exemplars(classified: pd.DataFrame) -> pd.DataFrame:
    exemplar_df = pd.DataFrame(HAND_LABELED_EXEMPLARS)
    rows: list[dict[str, Any]] = []
    lookup = {str(row["conditionId"]): row for row in classified.to_dict(orient="records")}
    for exemplar in exemplar_df.to_dict(orient="records"):
        condition_id = str(exemplar["conditionId"])
        row = lookup.get(condition_id)
        observed = str(row.get("s07MosaicClass", "")) if row else ""
        rows.append(
            {
                **exemplar,
                "observedS07MosaicClass": observed,
                "classifierAgreement": bool(row is not None and observed == exemplar["expectedS07MosaicClass"]),
                "conditionFound": bool(row is not None),
                "s07MosaicRationale": str(row.get("s07MosaicRationale", "")) if row else "",
                "s07MosaicClassConfidence": str(row.get("s07MosaicClassConfidence", "")) if row else "",
            }
        )
    return pd.DataFrame(rows)


def assign_mosaic_clusters(classified: pd.DataFrame, *, random_state: int = 907) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = [
        "finalAggregationS07",
        "finalInterfaceDensityS07",
        "longestRunFraction",
        "policyPositionSeparation",
        "dominanceProxyScoreS07",
        "finalTargetQualityScoreS07",
        "motionPersistenceProxy",
    ]
    feature_df = classified[feature_cols].apply(pd.to_numeric, errors="coerce")
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    feature_df = feature_df.fillna(feature_df.median(numeric_only=True)).fillna(0.0)
    try:
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        x = scaler.fit_transform(feature_df.to_numpy(dtype=float))
        k = min(7, max(1, len(classified)))
        model = KMeans(n_clusters=k, random_state=int(random_state), n_init=20)
        labels = model.fit_predict(x)
        distances = np.min(model.transform(x), axis=1)
        pca = PCA(n_components=2, random_state=int(random_state))
        coords = pca.fit_transform(x)
    except Exception:
        k = min(7, max(1, len(classified)))
        raw = pd.to_numeric(classified["dominanceProxyScoreS07"], errors="coerce").fillna(0.0)
        labels = pd.qcut(raw.rank(method="first"), q=k, labels=False, duplicates="drop").fillna(0).astype(int).to_numpy()
        distances = np.zeros(len(classified), dtype=float)
        coords = np.column_stack([raw.to_numpy(dtype=float), pd.to_numeric(classified["finalInterfaceDensityS07"], errors="coerce").fillna(0.0).to_numpy(dtype=float)])
    assignments = classified[["conditionId", "s06PairKey", "pairCategory", "ratioLabel", "arrangementType", "valueProfile", "s07MosaicClass"]].copy()
    assignments["s07MosaicClusterId"] = [f"cluster_{int(label):02d}" for label in labels]
    assignments["s07MosaicClusterDistance"] = distances.astype(float)
    assignments["s07ClusterPca1"] = coords[:, 0].astype(float)
    assignments["s07ClusterPca2"] = coords[:, 1].astype(float)
    summary = (
        assignments.groupby("s07MosaicClusterId", dropna=False)
        .agg(
            conditionCount=("conditionId", "size"),
            meanClusterDistance=("s07MosaicClusterDistance", "mean"),
            classCountsJson=("s07MosaicClass", lambda values: compact_json(dict(Counter(map(str, values))))),
            pairCount=("s06PairKey", "nunique"),
            arrangementCount=("arrangementType", "nunique"),
            valueProfileCount=("valueProfile", "nunique"),
        )
        .reset_index()
        .sort_values("s07MosaicClusterId", kind="mergesort")
    )
    return assignments, summary


def summarize_mosaic_classes(classified: pd.DataFrame) -> pd.DataFrame:
    return (
        classified.groupby("s07MosaicClass", dropna=False)
        .agg(
            conditionCount=("conditionId", "size"),
            pairCount=("s06PairKey", "nunique"),
            meanFinalAggregation=("finalAggregationS07", "mean"),
            meanFinalInterfaceDensity=("finalInterfaceDensityS07", "mean"),
            meanLongestRunFraction=("longestRunFraction", "mean"),
            meanPolicyPositionSeparation=("policyPositionSeparation", "mean"),
            meanDominanceProxyScore=("dominanceProxyScoreS07", "mean"),
            meanFinalTargetQualityScore=("finalTargetQualityScoreS07", "mean"),
            resourceCappedRate=("stopReason", lambda values: float(np.mean(pd.Series(values).astype(str).str.startswith("max_")))),
            noMoveRate=("stopReason", lambda values: float(np.mean(pd.Series(values).astype(str).str.contains("no_cell_can_move")))),
        )
        .reset_index()
        .sort_values("conditionCount", ascending=False, kind="mergesort")
    )


def summarize_mosaic_contexts(classified: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["ratioLabel", "arrangementType", "valueProfile", "perturbationType"]
    for keys, group in classified.groupby(group_cols, dropna=False, sort=False):
        key_payload = dict(zip(group_cols, keys, strict=True))
        class_counts = Counter(group["s07MosaicClass"].astype(str))
        modal_class, modal_count = class_counts.most_common(1)[0]
        rows.append(
            {
                **key_payload,
                "conditionCount": int(len(group)),
                "modalS07MosaicClass": modal_class,
                "modalS07MosaicClassShare": float(modal_count / max(1, len(group))),
                "s07MosaicClassCountsJson": compact_json(dict(class_counts)),
                "meanFinalAggregation": float(group["finalAggregationS07"].mean()),
                "meanFinalInterfaceDensity": float(group["finalInterfaceDensityS07"].mean()),
                "meanPolicyPositionSeparation": float(group["policyPositionSeparation"].mean()),
                "meanDominanceProxyScore": float(group["dominanceProxyScoreS07"].mean()),
                "meanFinalTargetQualityScore": float(group["finalTargetQualityScoreS07"].mean()),
                "unresolvedWinnerRate": float((group["winnerPolicyId"].astype(str) == UNRESOLVED_WINNER).mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_mosaic_pairs(classified: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pair_key, group in classified.groupby("s06PairKey", dropna=False, sort=False):
        class_counts = Counter(group["s07MosaicClass"].astype(str))
        modal_class, modal_count = class_counts.most_common(1)[0]
        rows.append(
            {
                "s06PairKey": str(pair_key),
                "pairCategory": str(group["pairCategory"].iloc[0]),
                "conditionCount": int(len(group)),
                "s06ContextDependencyClass": str(group["s06ContextDependencyClass"].iloc[0]),
                "modalS07MosaicClass": modal_class,
                "modalS07MosaicClassShare": float(modal_count / max(1, len(group))),
                "s07MosaicClassCountsJson": compact_json(dict(class_counts)),
                "mosaicClassDiversity": int(len(class_counts)),
                "meanFinalAggregation": float(group["finalAggregationS07"].mean()),
                "meanFinalInterfaceDensity": float(group["finalInterfaceDensityS07"].mean()),
                "meanPolicyPositionSeparation": float(group["policyPositionSeparation"].mean()),
                "meanDominanceProxyScore": float(group["dominanceProxyScoreS07"].mean()),
                "meanFinalTargetQualityScore": float(group["finalTargetQualityScoreS07"].mean()),
                "unresolvedWinnerRate": float((group["winnerPolicyId"].astype(str) == UNRESOLVED_WINNER).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["mosaicClassDiversity", "meanDominanceProxyScore", "s06PairKey"],
        ascending=[False, False, True],
        kind="mergesort",
    )


def select_stop_sensitivity_conditions(classified: pd.DataFrame, priority: pd.DataFrame, *, max_conditions: int = 14) -> pd.DataFrame:
    selected_ids: list[str] = []
    exemplar_ids = [item["conditionId"] for item in HAND_LABELED_EXEMPLARS]
    available = set(classified["conditionId"].astype(str))
    for condition_id in exemplar_ids:
        if condition_id in available and condition_id not in selected_ids:
            selected_ids.append(condition_id)
    for row in priority.to_dict(orient="records"):
        if len(selected_ids) >= int(max_conditions):
            break
        condition_id = str(row["conditionId"])
        if condition_id not in selected_ids and condition_id in available:
            selected_ids.append(condition_id)
    return classified[classified["conditionId"].astype(str).isin(selected_ids)].copy().reset_index(drop=True)


def load_winner_criteria(path: Path, boundary_caveats_path: Path = DEFAULT_S05_BOUNDARY_CAVEATS_PATH) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    boundary_caveats = load_boundary_caveats(boundary_caveats_path)
    return predefined_winner_criteria(boundary_caveats)


def run_stop_condition_sensitivity(
    selected_conditions: pd.DataFrame,
    baseline_classified: pd.DataFrame,
    ready_panel: pd.DataFrame,
    criteria: Mapping[str, Any],
    *,
    workers: int = 1,
    caps: Sequence[int] = STOP_SENSITIVITY_CAPS,
    max_swaps: int = 4000,
    max_comparisons: int = 30000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if selected_conditions.empty:
        return pd.DataFrame(), pd.DataFrame()
    replay_frames: list[pd.DataFrame] = []
    for cap in caps:
        rows: list[dict[str, Any]] = []
        for row in selected_conditions.to_dict(orient="records"):
            condition = dict(row)
            source_id = str(condition["conditionId"])
            condition["s07SourceConditionId"] = source_id
            condition["conditionId"] = f"{source_id}_s07_cap{int(cap)}"
            condition["researchStepId"] = STEP_ID
            condition["implementationPrefix"] = "e06_s07_sensitivity"
            condition["s07ReplayMaxActivations"] = int(cap)
            condition["s07SensitivityVariant"] = f"max_activations_{int(cap)}"
            rows.append(condition)
        replay_conditions = pd.DataFrame(rows)
        raw = run_conditions(
            replay_conditions,
            ready_panel,
            workers=int(workers),
            max_activations=int(cap),
            max_swaps=int(max_swaps),
            max_comparisons=int(max_comparisons),
        )
        goal = augment_goal_results(raw)
        scored = score_dominance_results(goal, criteria).copy()
        scored["researchStepId"] = STEP_ID
        scored["sourceResearchStepId"] = "S07_stop_sensitivity"
        classified = classify_mosaic_formations(scored)
        classified["s07ReplayMaxActivations"] = int(cap)
        replay_frames.append(classified)
    replay = pd.concat(replay_frames, ignore_index=True) if replay_frames else pd.DataFrame()
    baseline_lookup = {
        str(row["conditionId"]): row
        for row in baseline_classified.to_dict(orient="records")
    }
    rows = []
    for row in replay.to_dict(orient="records"):
        source_id = str(row.get("s07SourceConditionId", ""))
        base = baseline_lookup.get(source_id, {})
        rows.append(
            {
                "s07SourceConditionId": source_id,
                "conditionId": str(row["conditionId"]),
                "s07ReplayMaxActivations": int(row["s07ReplayMaxActivations"]),
                "baselineS07MosaicClass": str(base.get("s07MosaicClass", "")),
                "replayS07MosaicClass": str(row.get("s07MosaicClass", "")),
                "classMatchesBaseline": bool(str(base.get("s07MosaicClass", "")) == str(row.get("s07MosaicClass", ""))),
                "baselineStopReason": str(base.get("stopReason", "")),
                "replayStopReason": str(row.get("stopReason", "")),
                "stopReasonMatchesBaseline": bool(str(base.get("stopReason", "")) == str(row.get("stopReason", ""))),
                "baselineFinalTargetQualityScore": safe_float(base.get("finalTargetQualityScore")),
                "replayFinalTargetQualityScore": safe_float(row.get("finalTargetQualityScore")),
                "baselineDominanceProxyScore": safe_float(base.get("dominanceProxyScore")),
                "replayDominanceProxyScore": safe_float(row.get("dominanceProxyScore")),
                "baselineFinalInterfaceDensity": safe_float(base.get("finalInterfaceDensityS07")),
                "replayFinalInterfaceDensity": safe_float(row.get("finalInterfaceDensityS07")),
                "runSucceeded": _safe_bool(row.get("runSucceeded", False)),
            }
        )
    return pd.DataFrame(rows), replay


def validation_checks(
    s06_mosaics: pd.DataFrame,
    classified: pd.DataFrame,
    priority: pd.DataFrame,
    exemplars: pd.DataFrame,
    stop_sensitivity: pd.DataFrame,
    cluster_assignments: pd.DataFrame,
) -> pd.DataFrame:
    observed_classes = set(classified["s07MosaicClass"].astype(str))
    requested_classes = set(REQUESTED_MOSAIC_CLASSES)
    exemplar_rate = float(exemplars["classifierAgreement"].mean()) if not exemplars.empty else 0.0
    stop_caps = set(pd.to_numeric(stop_sensitivity.get("s07ReplayMaxActivations", pd.Series(dtype=int)), errors="coerce").dropna().astype(int))
    checks = [
        {
            "checkId": "s06_inputs_present",
            "success": bool(not s06_mosaics.empty and s06_mosaics["conditionId"].is_unique),
            "detail": f"loaded {len(s06_mosaics)} S06 dominance mosaic rows",
        },
        {
            "checkId": "one_classification_per_s06_row",
            "success": bool(len(classified) == len(s06_mosaics) and classified["conditionId"].is_unique),
            "detail": f"{len(classified)} S07 classifications for {len(s06_mosaics)} S06 rows",
        },
        {
            "checkId": "requested_mosaic_classes_represented",
            "success": bool(requested_classes <= observed_classes),
            "detail": f"observed classes: {sorted(observed_classes)}",
        },
        {
            "checkId": "priority_candidates_cover_context_dependent_and_unresolved",
            "success": bool(
                not priority.empty
                and priority["s07PriorityTagsJson"].astype(str).str.contains("context_dependent_pair").any()
                and priority["s07PriorityTagsJson"].astype(str).str.contains("unresolved_winner").any()
            ),
            "detail": f"{len(priority)} prioritized S07 candidates written",
        },
        {
            "checkId": "hand_labeled_exemplar_agreement",
            "success": bool(not exemplars.empty and exemplars["conditionFound"].map(bool).all() and exemplar_rate >= 0.85),
            "detail": f"{int(exemplars['classifierAgreement'].sum())}/{len(exemplars)} hand-labeled exemplars matched",
        },
        {
            "checkId": "stop_condition_sensitivity_completed",
            "success": bool(
                not stop_sensitivity.empty
                and set(STOP_SENSITIVITY_CAPS) <= stop_caps
                and stop_sensitivity["runSucceeded"].map(bool).all()
                and stop_sensitivity["replayS07MosaicClass"].astype(str).str.len().gt(0).all()
            ),
            "detail": (
                f"{len(stop_sensitivity)} replay rows across caps {sorted(stop_caps)}; "
                f"class stability {float(stop_sensitivity['classMatchesBaseline'].mean()) if not stop_sensitivity.empty else 0.0:.3f}"
            ),
        },
        {
            "checkId": "cluster_assignments_complete",
            "success": bool(len(cluster_assignments) == len(classified) and cluster_assignments["s07MosaicClusterId"].astype(str).str.len().gt(0).all()),
            "detail": f"{len(cluster_assignments)} rows assigned to mosaic feature clusters",
        },
        {
            "checkId": "metric_boundary_caveats_retained",
            "success": bool(
                classified["s07MosaicCaveatsJson"].astype(str).str.contains("metric_boundary_caveats_inherited_from_s06").all()
                and classified["metricBoundary"].astype(str).str.len().gt(0).all()
            ),
            "detail": "S06 metric-boundary text and S07 caveat flags retained on all classified rows",
        },
    ]
    return pd.DataFrame(checks)


def write_mosaic_plots(
    classified: pd.DataFrame,
    class_summary: pd.DataFrame,
    cluster_assignments: pd.DataFrame,
    exemplars: pd.DataFrame,
    figure_dir: Path,
    step_dir: Path,
) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if not class_summary.empty:
        plot_df = class_summary.sort_values("conditionCount", ascending=True)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(plot_df["s07MosaicClass"], plot_df["conditionCount"])
        ax.set_xlabel("S06 contest rows")
        ax.set_title("E06 S07 mosaic class counts")
        fig.tight_layout()
        for path in [
            figure_dir / "e06_s07_mosaic_class_counts.png",
            figure_dir / "e06_s07_mosaic_class_counts.pdf",
            step_dir / "mosaic_class_counts.png",
            step_dir / "mosaic_class_counts.pdf",
        ]:
            fig.savefig(path, dpi=180 if path.suffix == ".png" else None)
            paths.append(path)
        plt.close(fig)
    if not classified.empty:
        pivot = pd.crosstab(classified["s07MosaicClass"], classified["stopReason"])
        fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(pivot.columns)), max(4, 0.45 * len(pivot.index) + 2)))
        image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(pivot.columns)), labels=pivot.columns, rotation=35, ha="right")
        ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
        ax.set_title("E06 S07 mosaic class by stop condition")
        ax.set_xlabel("Stop reason")
        ax.set_ylabel("Mosaic class")
        fig.colorbar(image, ax=ax, label="Row count")
        fig.tight_layout()
        for path in [
            figure_dir / "e06_s07_class_stop_heatmap.png",
            figure_dir / "e06_s07_class_stop_heatmap.pdf",
            step_dir / "class_stop_heatmap.png",
            step_dir / "class_stop_heatmap.pdf",
        ]:
            fig.savefig(path, dpi=180 if path.suffix == ".png" else None)
            paths.append(path)
        plt.close(fig)
    if not cluster_assignments.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        classes = sorted(cluster_assignments["s07MosaicClass"].astype(str).unique())
        for klass in classes:
            hits = cluster_assignments[cluster_assignments["s07MosaicClass"].astype(str) == klass]
            ax.scatter(hits["s07ClusterPca1"], hits["s07ClusterPca2"], s=28, label=klass, alpha=0.75)
        ax.set_xlabel("Mosaic feature PC1")
        ax.set_ylabel("Mosaic feature PC2")
        ax.set_title("E06 S07 mosaic feature clusters")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        for path in [
            figure_dir / "e06_s07_mosaic_feature_clusters.png",
            figure_dir / "e06_s07_mosaic_feature_clusters.pdf",
            step_dir / "mosaic_feature_clusters.png",
            step_dir / "mosaic_feature_clusters.pdf",
        ]:
            fig.savefig(path, dpi=180 if path.suffix == ".png" else None)
            paths.append(path)
        plt.close(fig)
    exemplar_ids = [str(value) for value in exemplars["conditionId"].head(14)] if not exemplars.empty else []
    exemplar_rows = classified[classified["conditionId"].astype(str).isin(exemplar_ids)].copy()
    if not exemplar_rows.empty:
        exemplar_rows["_order"] = exemplar_rows["conditionId"].map({condition_id: i for i, condition_id in enumerate(exemplar_ids)})
        exemplar_rows = exemplar_rows.sort_values("_order", kind="mergesort")
        fig, axes = plt.subplots(len(exemplar_rows), 1, figsize=(12, max(4, 0.45 * len(exemplar_rows))), squeeze=False)
        class_to_int = {klass: i for i, klass in enumerate(sorted(classified["s07MosaicClass"].astype(str).unique()))}
        for ax, row in zip(axes[:, 0], exemplar_rows.itertuples(index=False), strict=False):
            final_ids = [str(item) for item in parse_json_maybe(getattr(row, "finalPanelPolicyIdsJson"), [])]
            policy_ids = [str(item) for item in parse_json_maybe(getattr(row, "policyIdsJson"), [])]
            mapping = {pid: i for i, pid in enumerate(policy_ids)}
            values = np.asarray([[mapping.get(pid, 0) for pid in final_ids]], dtype=float)
            ax.imshow(values, aspect="auto", interpolation="nearest", cmap="tab20")
            label = f"{getattr(row, 'conditionId')} | {getattr(row, 's07MosaicClass')}"
            ax.set_ylabel(label[:72], rotation=0, ha="right", va="center", fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
            if getattr(row, "s07MosaicClass") in class_to_int:
                ax.set_title("")
        fig.suptitle("E06 S07 hand-labeled exemplar final policy-label states", fontsize=11)
        fig.tight_layout()
        for path in [
            figure_dir / "e06_s07_mosaic_exemplar_state_diagrams.png",
            figure_dir / "e06_s07_mosaic_exemplar_state_diagrams.pdf",
            step_dir / "mosaic_exemplar_state_diagrams.png",
            step_dir / "mosaic_exemplar_state_diagrams.pdf",
        ]:
            fig.savefig(path, dpi=180 if path.suffix == ".png" else None)
            paths.append(path)
        plt.close(fig)
    return paths


def _write_markdown_rules(rules: pd.DataFrame, path: Path) -> Path:
    lines = [
        "# E06 S07 Mosaic Classification Rules",
        "",
        f"- Research step ID: `{STEP_ID}`",
        f"- Step number: {STEP_NUMBER}",
        f"- Classifier version: `{MOSAIC_CLASSIFIER_VERSION}`",
        "",
        "Rules are applied in order. All labels are computational final-state proxies for one-dimensional policy arrays.",
        "",
    ]
    for row in rules.sort_values("ruleOrder", kind="mergesort").itertuples(index=False):
        lines.extend(
            [
                f"## {row.ruleOrder}. `{row.s07MosaicClass}`",
                "",
                f"- Rule: {row.rule}",
                f"- Thresholds: `{row.thresholdsJson}`",
                f"- Metric boundary: {row.metricBoundary}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_markdown_report(
    *,
    step_dir: Path,
    status: Mapping[str, Any],
    artifacts_written: Sequence[str],
    class_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
    exemplars: pd.DataFrame,
    stop_sensitivity: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> Path:
    class_lines = "\n".join(
        f"- `{row.s07MosaicClass}`: {int(row.conditionCount)} rows, mean dominance {row.meanDominanceProxyScore:.3f}, "
        f"mean target quality {row.meanFinalTargetQualityScore:.3f}"
        for row in class_summary.itertuples(index=False)
    ) or "- No class summaries available."
    pair_lines = "\n".join(
        f"- `{row.s06PairKey}`: modal class `{row.modalS07MosaicClass}` ({row.modalS07MosaicClassShare:.2f}), "
        f"{int(row.mosaicClassDiversity)} classes, {row.s06ContextDependencyClass}"
        for row in pair_summary.head(5).itertuples(index=False)
    ) or "- No pair summaries available."
    agreement = float(exemplars["classifierAgreement"].mean()) if not exemplars.empty else 0.0
    stability = float(stop_sensitivity["classMatchesBaseline"].mean()) if not stop_sensitivity.empty else 0.0
    text = f"""# Research Step S07: Study mosaic formation

## Completion status

Research step ID: `S07`. {status['status']} on {status['completedAt']}. Outcome classification: {status['outcomeClassification']}.

## Artifacts written

{chr(10).join(f"- `{path}`" for path in artifacts_written)}

## Validation result

{status['validationResult']}. Classified {status['classificationCount']} S06 final states, matched {int(exemplars['classifierAgreement'].sum()) if not exemplars.empty else 0}/{len(exemplars)} hand-labeled exemplars, and ran {len(stop_sensitivity)} stop-condition sensitivity replays. Validation checks passed: {int(validation_df['success'].sum())}/{len(validation_df)}.

## Caveats or blockers

S07 labels are computational final-state classes over one-dimensional policy arrays. The oscillatory class is a resource-cap and motion-persistence proxy, not direct cycle detection. Stop-condition sensitivity was intentionally bounded to hand-labeled and priority contexts rather than a full S06 rerun. Biological terms remain analogies for local-policy simulations only.

## Lay summary

S07 converted the S06 dominance-contest final states into a mosaic taxonomy. The classifier separated completed interspersed states, aggregated patch states, near-layered states, polarized states, resource-capped motion-proxy states, no-move conflict states, and residual mixed states.

## Class summary

{class_lines}

## Priority pair/context summary

{pair_lines}

## Classifier and stop-condition checks

- Hand-labeled exemplar agreement: {agreement:.3f}
- Stop-condition class stability versus S06 baseline: {stability:.3f}

## Recommended next action

{status['recommendedNextAction']}
"""
    path = step_dir / "summary.md"
    path.write_text(text, encoding="utf-8")
    return path


def run_s07_mosaic_formation(
    *,
    artifacts_dir: Path | None = None,
    repo_root: Path | None = None,
    panel_path: Path = DEFAULT_PANEL_PATH,
    s06_mosaics_path: Path = DEFAULT_S06_MOSAICS_PATH,
    s06_contests_path: Path = DEFAULT_S06_CONTESTS_PATH,
    s06_pair_summary_path: Path = DEFAULT_S06_PAIR_SUMMARY_PATH,
    s06_context_summary_path: Path = DEFAULT_S06_CONTEXT_SUMMARY_PATH,
    winner_criteria_path: Path = DEFAULT_S06_WINNER_CRITERIA_PATH,
    workers: int | None = None,
    sensitivity_max_conditions: int = 14,
) -> dict[str, Any]:
    artifacts_dir = artifacts_dir or Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
    repo_root = repo_root or Path(__file__).resolve().parents[1]
    workers = int(workers if workers is not None else min(8, os.cpu_count() or 1))
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    figures_dir = artifacts_dir / "figures" / "e06"
    provenance_dir = artifacts_dir / "provenance"
    for path in [step_dir, results_dir, figures_dir, provenance_dir]:
        path.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    ready_panel = load_ready_panel(panel_path)
    s06_mosaics = load_s06_mosaics(s06_mosaics_path)
    s06_contests = load_required_table(s06_contests_path, ["conditionId", "s06PairKey", "winnerPolicyId"], "S06 dominance contests")
    pair_summary = load_required_table(
        s06_pair_summary_path,
        ["s06PairKey", "contextDependencyClass", "modalWinnerPolicyId", "modalWinnerShare", "unresolvedCount", "meanDominanceProxyScore"],
        "S06 dominance pair summary",
    )
    context_summary = load_required_table(
        s06_context_summary_path,
        ["ratioLabel", "arrangementType", "valueProfile", "perturbationType", "meanDominanceProxyScore", "unresolvedRate"],
        "S06 dominance context summary",
    )
    criteria = load_winner_criteria(winner_criteria_path)

    rules = mosaic_rule_definitions()
    classified = classify_mosaic_formations(s06_mosaics, pair_summary, context_summary)
    priority = build_priority_candidates(classified)
    exemplars = build_hand_labeled_exemplars(classified)
    cluster_assignments, cluster_summary = assign_mosaic_clusters(classified)
    class_summary = summarize_mosaic_classes(classified)
    mosaic_context_summary = summarize_mosaic_contexts(classified)
    mosaic_pair_summary = summarize_mosaic_pairs(classified)
    sensitivity_conditions = select_stop_sensitivity_conditions(
        classified,
        priority,
        max_conditions=int(sensitivity_max_conditions),
    )
    stop_sensitivity, stop_replays = run_stop_condition_sensitivity(
        sensitivity_conditions,
        classified,
        ready_panel,
        criteria,
        workers=workers,
    )
    validation_df = validation_checks(
        s06_mosaics,
        classified,
        priority,
        exemplars,
        stop_sensitivity,
        cluster_assignments,
    )

    rules_json = step_dir / "mosaic_classification_rules.json"
    rules_md = step_dir / "mosaic_classification_rules.md"
    rules_csv = step_dir / "mosaic_classification_rules.csv"
    rules_parquet = step_dir / "mosaic_classification_rules.parquet"
    step_class_csv = step_dir / "mosaic_classifications.csv"
    step_class_parquet = step_dir / "mosaic_classifications.parquet"
    result_class_csv = results_dir / "e06_mosaic_classifications.csv"
    result_class_parquet = results_dir / "e06_mosaic_classifications.parquet"
    result_mosaic_csv = results_dir / "e06_dominance_mosaics.csv"
    result_mosaic_parquet = results_dir / "e06_dominance_mosaics.parquet"
    class_summary_csv = step_dir / "mosaic_class_summary.csv"
    class_summary_parquet = step_dir / "mosaic_class_summary.parquet"
    context_summary_csv = step_dir / "mosaic_context_summary.csv"
    context_summary_parquet = step_dir / "mosaic_context_summary.parquet"
    pair_summary_csv = step_dir / "mosaic_pair_summary.csv"
    pair_summary_parquet = step_dir / "mosaic_pair_summary.parquet"
    priority_csv = step_dir / "priority_mosaic_candidates.csv"
    priority_parquet = step_dir / "priority_mosaic_candidates.parquet"
    exemplar_csv = step_dir / "hand_labeled_exemplars.csv"
    exemplar_parquet = step_dir / "hand_labeled_exemplars.parquet"
    agreement_csv = step_dir / "classifier_agreement.csv"
    agreement_parquet = step_dir / "classifier_agreement.parquet"
    sensitivity_conditions_csv = step_dir / "stop_sensitivity_condition_selection.csv"
    sensitivity_conditions_parquet = step_dir / "stop_sensitivity_condition_selection.parquet"
    sensitivity_csv = step_dir / "stop_condition_sensitivity.csv"
    sensitivity_parquet = step_dir / "stop_condition_sensitivity.parquet"
    replay_csv = step_dir / "stop_condition_replay_rows.csv"
    replay_parquet = step_dir / "stop_condition_replay_rows.parquet"
    cluster_csv = step_dir / "mosaic_cluster_assignments.csv"
    cluster_parquet = step_dir / "mosaic_cluster_assignments.parquet"
    cluster_summary_csv = step_dir / "mosaic_cluster_summary.csv"
    cluster_summary_parquet = step_dir / "mosaic_cluster_summary.parquet"
    validation_csv = step_dir / "validation_checks.csv"
    validation_parquet = step_dir / "validation_checks.parquet"
    status_path = step_dir / "status.json"
    manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = provenance_dir / "run_manifest.json"

    rules_payload = {
        "schema": "eidosoma.e06_s07_mosaic_classifier.v1",
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "classifierVersion": MOSAIC_CLASSIFIER_VERSION,
        "requestedClasses": list(REQUESTED_MOSAIC_CLASSES),
        "ruleOrder": rules.to_dict(orient="records"),
        "claimBoundary": CLAIM_BOUNDARY,
    }
    write_json(rules_json, rules_payload)
    _write_markdown_rules(rules, rules_md)
    for path, df in [
        (rules_csv, rules),
        (step_class_csv, compact_result_csv(classified)),
        (result_class_csv, compact_result_csv(classified)),
        (result_mosaic_csv, compact_result_csv(classified)),
        (class_summary_csv, class_summary),
        (context_summary_csv, mosaic_context_summary),
        (pair_summary_csv, mosaic_pair_summary),
        (priority_csv, priority),
        (exemplar_csv, exemplars),
        (agreement_csv, exemplars),
        (sensitivity_conditions_csv, compact_result_csv(sensitivity_conditions)),
        (sensitivity_csv, stop_sensitivity),
        (replay_csv, compact_result_csv(stop_replays) if not stop_replays.empty else stop_replays),
        (cluster_csv, cluster_assignments),
        (cluster_summary_csv, cluster_summary),
        (validation_csv, validation_df),
    ]:
        df.to_csv(path, index=False)
    for path, df in [
        (rules_parquet, rules),
        (step_class_parquet, classified),
        (result_class_parquet, classified),
        (result_mosaic_parquet, classified),
        (class_summary_parquet, class_summary),
        (context_summary_parquet, mosaic_context_summary),
        (pair_summary_parquet, mosaic_pair_summary),
        (priority_parquet, priority),
        (exemplar_parquet, exemplars),
        (agreement_parquet, exemplars),
        (sensitivity_conditions_parquet, sensitivity_conditions),
        (sensitivity_parquet, stop_sensitivity),
        (replay_parquet, stop_replays),
        (cluster_parquet, cluster_assignments),
        (cluster_summary_parquet, cluster_summary),
        (validation_parquet, validation_df),
    ]:
        df.to_parquet(path, index=False)

    figure_paths = write_mosaic_plots(classified, class_summary, cluster_assignments, exemplars, figures_dir, step_dir)
    completed_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    validation_result = "passed" if validation_df["success"].map(bool).all() else "failed"
    observed_classes = sorted(set(classified["s07MosaicClass"].astype(str)))
    agreement_rate = float(exemplars["classifierAgreement"].mean()) if not exemplars.empty else float("nan")
    stop_stability_rate = float(stop_sensitivity["classMatchesBaseline"].mean()) if not stop_sensitivity.empty else float("nan")
    outcome = "supportive" if validation_result == "passed" and set(REQUESTED_MOSAIC_CLASSES) <= set(observed_classes) else "constraining/contradictory"
    artifacts = [
        rules_json,
        rules_md,
        rules_csv,
        rules_parquet,
        step_class_csv,
        step_class_parquet,
        result_class_csv,
        result_class_parquet,
        result_mosaic_csv,
        result_mosaic_parquet,
        class_summary_csv,
        class_summary_parquet,
        context_summary_csv,
        context_summary_parquet,
        pair_summary_csv,
        pair_summary_parquet,
        priority_csv,
        priority_parquet,
        exemplar_csv,
        exemplar_parquet,
        agreement_csv,
        agreement_parquet,
        sensitivity_conditions_csv,
        sensitivity_conditions_parquet,
        sensitivity_csv,
        sensitivity_parquet,
        replay_csv,
        replay_parquet,
        cluster_csv,
        cluster_parquet,
        cluster_summary_csv,
        cluster_summary_parquet,
        validation_csv,
        validation_parquet,
        *figure_paths,
        step_dir / "summary.md",
        status_path,
        manifest_path,
        run_manifest_path,
    ]
    status = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": validation_result == "passed",
        "status": "completed" if validation_result == "passed" else "completed_with_validation_failure",
        "artifactsWritten": [str(path) for path in artifacts],
        "validationResult": validation_result,
        "caveatsOrBlockers": [
            "Mosaic labels are computational final-state proxies over one-dimensional policy-label arrays.",
            "Oscillatory states are resource-cap and motion-persistence proxies, not direct trajectory-cycle detections.",
            "Stop-condition sensitivity replays were bounded to hand-labeled and priority contexts rather than all 576 S06 contests.",
            "S07 reused S06 opposite-goal dominance metadata and inherits its metric-boundary caveats.",
            CLAIM_BOUNDARY,
        ],
        "recommendedNextAction": (
            "Chief Scientist review, then proceed to S08 interface rules using S07 mosaic classes and priority contexts "
            "as baseline targets; do not start S08 until review is complete."
        ),
        "laySummary": (
            "S07 classified S06 final states into homogeneous, patchy, layered, polarized, oscillatory, frozen-conflict, "
            "and mixed mosaic classes, then checked hand-labeled exemplars and stop-condition sensitivity."
        ),
        "outcomeClassification": outcome,
        "classificationCount": int(len(classified)),
        "s06ContestCount": int(len(s06_contests)),
        "observedMosaicClasses": observed_classes,
        "handLabeledExemplarCount": int(len(exemplars)),
        "handLabeledAgreementRate": agreement_rate,
        "stopSensitivityReplayCount": int(len(stop_sensitivity)),
        "stopSensitivityClassStabilityRate": stop_stability_rate,
        "priorityCandidateCount": int(len(priority)),
        "clusterCount": int(cluster_assignments["s07MosaicClusterId"].nunique()) if not cluster_assignments.empty else 0,
        "workerCount": int(workers),
        "completedAt": completed_at,
        "wallTimeSeconds": float(time.perf_counter() - started),
        "newDependenciesInstalled": [],
        "sourceCodeLocation": str(repo_root),
        "sourceCodeArtifactPolicy": "repository-backed source was committed to git; source files were not copied into artifacts per workspace instructions",
    }
    summary_path = write_markdown_report(
        step_dir=step_dir,
        status=status,
        artifacts_written=[str(path) for path in artifacts],
        class_summary=class_summary,
        pair_summary=mosaic_pair_summary,
        exemplars=exemplars,
        stop_sensitivity=stop_sensitivity,
        validation_df=validation_df,
    )
    if summary_path not in artifacts:
        artifacts.append(summary_path)
    status["artifactsWritten"] = [str(path) for path in artifacts]
    write_json(status_path, status)

    manifest_payload = {
        "schema": "eidosoma.step_artifact_manifest.v1",
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "mosaicClassifierVersion": MOSAIC_CLASSIFIER_VERSION,
        "inputArtifacts": {
            "s06Mosaics": str(s06_mosaics_path),
            "s06Contests": str(s06_contests_path),
            "s06PairSummary": str(s06_pair_summary_path),
            "s06ContextSummary": str(s06_context_summary_path),
            "winnerCriteria": str(winner_criteria_path),
        },
        "artifacts": artifact_records(artifacts),
    }
    write_json(manifest_path, manifest_payload)

    existing_manifest: dict[str, Any] = {}
    if run_manifest_path.exists():
        try:
            existing_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing_manifest = {}
    research_steps = dict(existing_manifest.get("researchSteps", {}))
    research_steps[STEP_ID] = status
    run_manifest = {
        **existing_manifest,
        "schema": "eidosoma.run_manifest.v1",
        "experimentId": "E06",
        "lastResearchStepId": STEP_ID,
        "lastStepNumber": STEP_NUMBER,
        "updatedAt": completed_at,
        "git": {
            "branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"], repo_root),
            "commit": git_value(["rev-parse", "HEAD"], repo_root),
            "dirtyStatus": git_value(["status", "--short"], repo_root),
            "remote": git_value(["remote", "-v"], repo_root),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpuCount": os.cpu_count(),
            "workerCount": int(workers),
            "threadEnvironment": {
                key: os.environ.get(key)
                for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]
                if os.environ.get(key) is not None
            },
            "newDependenciesInstalled": [],
        },
        "researchSteps": research_steps,
        "artifacts": artifact_records(artifacts),
    }
    write_json(run_manifest_path, run_manifest)
    manifest_payload["artifacts"] = artifact_records(artifacts)
    write_json(manifest_path, manifest_payload)
    return {
        "status": status,
        "rules": rules,
        "classifications": classified,
        "classSummary": class_summary,
        "contextSummary": mosaic_context_summary,
        "pairSummary": mosaic_pair_summary,
        "priority": priority,
        "exemplars": exemplars,
        "stopSensitivity": stop_sensitivity,
        "stopReplays": stop_replays,
        "clusterAssignments": cluster_assignments,
        "clusterSummary": cluster_summary,
        "validation": validation_df,
        "artifactPaths": [str(path) for path in artifacts],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run E06 S07 mosaic-formation classification")
    parser.add_argument("--artifacts-dir", type=Path, default=None)
    parser.add_argument("--panel-path", type=Path, default=DEFAULT_PANEL_PATH)
    parser.add_argument("--s06-mosaics-path", type=Path, default=DEFAULT_S06_MOSAICS_PATH)
    parser.add_argument("--s06-contests-path", type=Path, default=DEFAULT_S06_CONTESTS_PATH)
    parser.add_argument("--s06-pair-summary-path", type=Path, default=DEFAULT_S06_PAIR_SUMMARY_PATH)
    parser.add_argument("--s06-context-summary-path", type=Path, default=DEFAULT_S06_CONTEXT_SUMMARY_PATH)
    parser.add_argument("--winner-criteria-path", type=Path, default=DEFAULT_S06_WINNER_CRITERIA_PATH)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--sensitivity-max-conditions", type=int, default=14)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_s07_mosaic_formation(
        artifacts_dir=args.artifacts_dir,
        panel_path=args.panel_path,
        s06_mosaics_path=args.s06_mosaics_path,
        s06_contests_path=args.s06_contests_path,
        s06_pair_summary_path=args.s06_pair_summary_path,
        s06_context_summary_path=args.s06_context_summary_path,
        winner_criteria_path=args.winner_criteria_path,
        workers=args.workers,
        sensitivity_max_conditions=args.sensitivity_max_conditions,
    )
    status = result["status"]
    print(
        f"{STEP_ID} {status['status']}: {status['classificationCount']} mosaic classifications, "
        f"{status['stopSensitivityReplayCount']} sensitivity replays, validation {status['validationResult']}"
    )
    return 0 if status["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
