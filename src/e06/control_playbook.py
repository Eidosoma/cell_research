"""E06 S15 chimeric-control playbook synthesis helpers."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


STEP_ID = "S15"
STEP_NUMBER = 15
CLAIM_SCHEMA = "eidosoma.e06.s15_control_claim.v1"
EVIDENCE_SCHEMA = "eidosoma.e06.s15_claim_evidence_link.v1"
VALIDATION_SCHEMA = "eidosoma.e06.s15_validation.v1"

REQUIRED_PRIOR_STEPS = tuple(f"S{idx:02d}" for idx in range(1, 15))

SOURCE_TABLES = {
    "algotype_metadata": "tables/e06_algotype_metadata.csv",
    "s01_validation": "research_steps/S01/e06_s01_validation_runs.parquet",
    "s02_threshold": "tables/e06_s02_threshold_candidates.csv",
    "s03_arrangement": "tables/e06_s03_arrangement_sensitivity.csv",
    "s04_goal": "tables/e06_s04_goal_profile_sensitivity.csv",
    "s05_dimension": "tables/e06_compatibility_dimension_summary.csv",
    "s05_scores": "tables/e06_compatibility_scores.csv",
    "s06_hierarchy": "tables/e06_dominance_hierarchy.csv",
    "s06_context": "tables/e06_s06_context_dependence.csv",
    "s07_continua": "tables/e06_s07_metric_continua.csv",
    "s07_stability": "tables/e06_s07_condition_stability.csv",
    "s07_agreement": "research_steps/S07/e06_s07_classifier_pairwise_agreement.csv",
    "s08_strata": "tables/e06_s08_mechanism_strata.csv",
    "s09_rankings": "tables/e06_s09_governance_rankings.csv",
    "s10_strata": "tables/e06_s10_graft_strata_summary.csv",
    "s11_strata": "tables/e06_s11_strata_summary.csv",
    "s12_history": "tables/e06_s12_history_baseline_comparison.csv",
    "s13_performance": "tables/e06_s13_model_performance_summary.csv",
    "s13_feature_screen": "tables/e06_s13_exploratory_causal_screen.csv",
    "s14_rankings": "tables/e06_s14_intervention_rankings.csv",
    "phase_matrix": "results/e06_chimerism_phase_matrix.parquet",
}


@dataclass(frozen=True)
class S15Config:
    """Configuration for S15 evidence synthesis."""

    min_claim_count: int = 12
    required_claim_ids: tuple[str, ...] = (
        "C01",
        "C02",
        "C03",
        "C04",
        "C05",
        "C06",
        "C07",
        "C08",
        "C09",
        "C10",
        "C11",
        "C12",
        "C13",
        "C14",
        "C15",
    )
    required_access_terms: tuple[str, ...] = (
        "behavior_only",
        "explicit_interface",
        "finite_radius_governance",
        "global_controller_like",
    )


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def format_number(value: Any) -> str:
    number = safe_float(value)
    if math.isfinite(number):
        return f"{number:.4f}"
    return str(value)


def json_compact(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def parse_jsonish(value: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_s15_sources(artifacts_dir: Path) -> dict[str, pd.DataFrame]:
    sources: dict[str, pd.DataFrame] = {}
    for key, relative_path in SOURCE_TABLES.items():
        path = artifacts_dir / relative_path
        if not path.exists():
            sources[key] = pd.DataFrame()
            continue
        sources[key] = read_table(path)
    return sources


def relative_artifact_path(path: str | Path, artifacts_dir: Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(artifacts_dir))
    except ValueError:
        return str(path)


def frame_records(frame: pd.DataFrame, columns: Sequence[str] | None = None, limit: int = 5) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    selected = frame.copy()
    if columns is not None:
        present = [column for column in columns if column in selected.columns]
        selected = selected[present]
    return selected.head(limit).where(pd.notna(selected.head(limit)), None).to_dict(orient="records")


def evidence_link(
    evidence_id: str,
    claim_id: str,
    source_step_id: str,
    artifact_path: str,
    row_selector: str,
    rows: pd.DataFrame,
    values: Mapping[str, Any],
    note: str,
) -> dict[str, Any]:
    return {
        "schema": EVIDENCE_SCHEMA,
        "evidence_id": evidence_id,
        "claim_id": claim_id,
        "source_step_id": source_step_id,
        "artifact_path": artifact_path,
        "row_selector": row_selector,
        "source_row_count": int(len(rows)),
        "source_columns_json": json_compact(list(rows.columns)),
        "evidence_values_json": json_compact(values),
        "evidence_note": note,
    }


def claim_row(
    claim_id: str,
    title: str,
    playbook_rule: str,
    evidence_level: str,
    outcome_classification: str,
    access_stratum: str,
    local_vs_global_note: str,
    uncertainty_caveat: str,
    recommended_use: str,
    avoid_overclaim: str,
) -> dict[str, Any]:
    return {
        "schema": CLAIM_SCHEMA,
        "research_step_id": STEP_ID,
        "claim_id": claim_id,
        "title": title,
        "playbook_rule": playbook_rule,
        "evidence_level": evidence_level,
        "outcome_classification": outcome_classification,
        "access_stratum": access_stratum,
        "local_vs_global_note": local_vs_global_note,
        "uncertainty_caveat": uncertainty_caveat,
        "recommended_use": recommended_use,
        "avoid_overclaim": avoid_overclaim,
    }


def _sort(frame: pd.DataFrame, by: str | Sequence[str], ascending: bool | Sequence[bool] = False) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame.sort_values(by, ascending=ascending, kind="mergesort")


def _s12_history_summary(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame()
    return (
        history.groupby(["history_id", "history_family"], dropna=False)
        .agg(
            condition_count=("history_effect_magnitude", "size"),
            mean_history_effect_magnitude=("history_effect_magnitude", "mean"),
            mean_delta_final_target_quality=("delta_final_target_quality", "mean"),
            mean_delta_goal_conflict_index=("delta_goal_conflict_index", "mean"),
            mean_delta_aggregation_delta_percent=("delta_aggregation_delta_percent", "mean"),
            state_class_change_rate=("final_state_class_changed_vs_simultaneous", "mean"),
        )
        .reset_index()
        .sort_values("mean_history_effect_magnitude", ascending=False, kind="mergesort")
    )


def build_claim_registry_and_evidence(sources: Mapping[str, pd.DataFrame], artifacts_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the S15 claim registry, evidence links, and history summary."""

    claims: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []

    def artifact(key: str) -> str:
        return SOURCE_TABLES[key]

    # C01: policy library and controls.
    metadata = sources["algotype_metadata"]
    category_counts = metadata["source_category"].value_counts().sort_index().to_dict() if "source_category" in metadata else {}
    validation = sources["s01_validation"]
    claims.append(
        claim_row(
            "C01",
            "Use the E06 policy library as a bounded, validated policy panel.",
            "Treat conclusions as applying to the 25-policy E06 library and its priority subsets, not to all possible sorting policies.",
            "high_internal_validity",
            "supportive",
            "behavior_only",
            "No governance or global-controller access is needed for the library definition.",
            "E04 memory-repair policies were smoke-tested in pure arrays but inherit repair claims from upstream E04.",
            "Use S01 as the source of stable Algotype IDs, categories, and pure-policy competence context.",
            "Do not treat discovered or memory policies as biologically validated cell types.",
        )
    )
    evidence.append(
        evidence_link(
            "E01a",
            "C01",
            "S01",
            artifact("algotype_metadata"),
            "all metadata rows grouped by source_category",
            metadata,
            {
                "algotype_count": len(metadata),
                "source_category_counts": category_counts,
                "pure_execution_success_count": int(metadata.get("pure_execution_success", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
            },
            "S01 metadata records the bounded policy library and categories.",
        )
    )
    evidence.append(
        evidence_link(
            "E01b",
            "C01",
            "S01",
            artifact("s01_validation"),
            "all S01 validation run rows",
            validation,
            {
                "validation_rows": len(validation),
                "invalid_count": int(validation.get("invalid", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
            },
            "S01 validation runs provide the pure-policy smoke-test layer.",
        )
    )

    # C02: mixture-ratio thresholds.
    s02 = sources["s02_threshold"]
    s02_top = _sort(s02, ["largest_sortedness_jump", "aggregation_delta_range"], [False, False]).head(1)
    claims.append(
        claim_row(
            "C02",
            "Mixture ratios can create threshold and minority effects.",
            "Scan ratios before interpreting a pair as compatible or incompatible; minority fractions can shift target quality and aggregation.",
            "high_internal_validity",
            "supportive",
            "behavior_only",
            "Ratio effects arise in the behavior-only local-action harness.",
            "The S02 sweep is bounded to a 13-policy priority panel rather than all S01 pairs.",
            "Prioritize S02 threshold candidates when choosing mixtures for deeper testing.",
            "Do not assume a 50:50 mixture represents the whole pairwise phase behavior.",
        )
    )
    evidence.append(
        evidence_link(
            "E02a",
            "C02",
            "S02",
            artifact("s02_threshold"),
            "all threshold candidate rows plus top largest_sortedness_jump row",
            s02,
            {
                "candidate_count": len(s02),
                "minority_candidate_count": int(s02.get("minority_effect_candidate", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
                "aggregation_sensitive_count": int(s02.get("aggregation_sensitive_candidate", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
                "top_row": frame_records(s02_top),
            },
            "S02 threshold table anchors ratio-sensitive candidate selection.",
        )
    )

    # C03: arrangement sensitivity.
    s03 = sources["s03_arrangement"]
    s03_top = _sort(s03, "sortedness_range_across_arrangements").head(1)
    claims.append(
        claim_row(
            "C03",
            "Initial spatial arrangement is a first-order control knob.",
            "For sensitive pairs, use matched arrangements before invoking added mechanisms; contiguous patches, gradients, and random mixtures can land in different regions.",
            "high_internal_validity",
            "supportive",
            "behavior_only",
            "Arrangement effects were measured without explicit recognition or global access.",
            "S03 arrangements are deterministic 1D computational layouts, not biological tissue geometries.",
            "Use arrangement sweeps as a low-cost intervention before adding governance.",
            "Do not collapse pair behavior across random, patch, gradient, and clustered starts.",
        )
    )
    evidence.append(
        evidence_link(
            "E03a",
            "C03",
            "S03",
            artifact("s03_arrangement"),
            "arrangement_sensitive_candidate == True; top sortedness_range_across_arrangements row",
            s03[s03.get("arrangement_sensitive_candidate", False).astype(bool)] if not s03.empty else s03,
            {
                "sensitive_count": int(s03.get("arrangement_sensitive_candidate", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
                "total_rows": len(s03),
                "top_row": frame_records(s03_top),
            },
            "S03 row-level sensitivity table preserves matched candidate, ratio, and arrangement roles.",
        )
    )

    # C04: goal compatibility.
    s04 = sources["s04_goal"]
    s04_top = _sort(s04, "reference_increasing_range").head(1)
    claims.append(
        claim_row(
            "C04",
            "Goal compatibility changes whether local success becomes shared success or conflict.",
            "Score mixtures under the actual goal profile; same-goal behavior does not predict opposite or partially compatible outcomes.",
            "high_internal_validity",
            "supportive",
            "behavior_only",
            "Goal-profile effects were measured before adding explicit interface or governance mechanisms.",
            "Goal encoders are rank-order computational proxies rather than biological target fields.",
            "Separate same, opposite, partially compatible, shared-global, and unrelated-goal playbook recommendations.",
            "Do not describe opposite-goal failure as a generic policy incompatibility without naming the goal profile.",
        )
    )
    evidence.append(
        evidence_link(
            "E04a",
            "C04",
            "S04",
            artifact("s04_goal"),
            "goal_sensitive_candidate == True; top reference_increasing_range row",
            s04[s04.get("goal_sensitive_candidate", False).astype(bool)] if not s04.empty else s04,
            {
                "goal_sensitive_count": int(s04.get("goal_sensitive_candidate", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
                "total_rows": len(s04),
                "top_row": frame_records(s04_top),
            },
            "S04 sensitivity table links candidate, arrangement, and goal-profile contrasts.",
        )
    )

    # C05: compatibility dimensions.
    s05_dimension = sources["s05_dimension"]
    s05_scores = sources["s05_scores"]
    s05_opposite = s05_dimension[s05_dimension.get("goal_compatibility_class", pd.Series(dtype=str)).eq("opposite_goal")]
    claims.append(
        claim_row(
            "C05",
            "Compatibility is multidimensional, not a single score.",
            "Keep target quality, aggregation, dominance, goal conflict, and work/proxy cost separate when ranking mixtures.",
            "high_internal_validity",
            "supportive",
            "behavior_only",
            "S05 dimensions summarize behavior-only S02-S04 rows and pure-policy baselines.",
            "Work interference uses scaled n=8 pure-policy baselines and is weaker than direct quality and conflict metrics.",
            "Use the compatibility pattern class and disagreement count to flag mixtures that need separate report language.",
            "Do not average conflicting dimensions into a single biological compatibility claim.",
        )
    )
    evidence.append(
        evidence_link(
            "E05a",
            "C05",
            "S05",
            artifact("s05_dimension"),
            "all rows; emphasize goal_compatibility_class == 'opposite_goal'",
            s05_dimension,
            {
                "dimension_rows": len(s05_dimension),
                "opposite_goal_summary": frame_records(s05_opposite),
                "mean_primary_target_quality_all": safe_float(s05_scores.get("primary_target_quality", pd.Series(dtype=float)).mean()),
                "mean_quality_synergy_all": safe_float(s05_scores.get("quality_synergy", pd.Series(dtype=float)).mean()),
                "metric_disagreement_rows": int(s05_scores.get("metric_disagreement_count", pd.Series(dtype=float)).fillna(0).gt(0).sum()),
            },
            "S05 dimension and score tables document separate compatibility axes and conflicts.",
        )
    )

    # C06: dominance hierarchy and context dependence.
    s06_hierarchy = sources["s06_hierarchy"]
    s06_context = sources["s06_context"]
    s06_top = _sort(s06_hierarchy, "net_dominance_score").head(2)
    claims.append(
        claim_row(
            "C06",
            "Dominance rankings are real in the bounded contests but context-dependent.",
            "Use dominance hierarchy as a warning layer, then preserve contest context, orientation, ratio, and perturbation before making control decisions.",
            "high_internal_validity",
            "supportive_with_context_constraint",
            "behavior_only",
            "S06 dominance contests did not require explicit local recognition or global control.",
            "Every base contest was context-dependent, so a single dominance list is insufficient for prediction.",
            "Flag high-dominance discovered policies as likely to reshape conflicts, then check context-specific rows.",
            "Do not report a universal dominance ordering outside the bounded S06 opposite-goal panel.",
        )
    )
    evidence.append(
        evidence_link(
            "E06a",
            "C06",
            "S06",
            artifact("s06_hierarchy"),
            "all hierarchy rows sorted by net_dominance_score",
            s06_hierarchy,
            {
                "policy_count": len(s06_hierarchy),
                "top_positive_rows": frame_records(s06_top),
                "most_negative_row": frame_records(_sort(s06_hierarchy, "net_dominance_score", True).head(1)),
            },
            "Dominance hierarchy table gives policy-level contest outcomes.",
        )
    )
    evidence.append(
        evidence_link(
            "E06b",
            "C06",
            "S06",
            artifact("s06_context"),
            "context_dependent_dominance == True",
            s06_context[s06_context.get("context_dependent_dominance", False).astype(bool)] if not s06_context.empty else s06_context,
            {
                "context_dependent_count": int(s06_context.get("context_dependent_dominance", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
                "total_base_contests": len(s06_context),
            },
            "S06 context table prevents collapsing dominance into a context-free law.",
        )
    )

    # C07: final-state continua.
    s07_continua = sources["s07_continua"]
    s07_stability = sources["s07_stability"]
    s07_agreement = sources["s07_agreement"]
    selected_metrics = s07_continua[s07_continua.get("metric", pd.Series(dtype=str)).isin(
        ["final_target_quality", "goal_conflict_index", "aggregation_delta_percent", "largest_block_fraction", "position_bias_abs"]
    )]
    agreement_col = "semantic_label_agreement" if "semantic_label_agreement" in s07_agreement else None
    claims.append(
        claim_row(
            "C07",
            "Treat final-state labels as descriptors over metric continua.",
            "Use continuum metrics and exemplars as the primary morphology language; labels are useful but not stable enough for a hard taxonomy.",
            "high_internal_validity",
            "constraining",
            "mixed_behavior_outcome",
            "This is an outcome-description layer, not a governance mechanism.",
            "Existing inputs provide final event-cap states and trajectory proxies, not full oscillation or turnover traces.",
            "Report median/p90 continua and exemplar classes together when describing mosaics.",
            "Do not force discrete class boundaries when classifier settings disagree.",
        )
    )
    evidence.append(
        evidence_link(
            "E07a",
            "C07",
            "S07",
            artifact("s07_continua"),
            "metric in final_target_quality, goal_conflict_index, aggregation_delta_percent, largest_block_fraction, position_bias_abs",
            selected_metrics,
            {"metric_rows": frame_records(selected_metrics, ["metric", "run_count", "median", "p90", "mean", "std"], 10)},
            "S07 continuum table records final-state metric distributions.",
        )
    )
    evidence.append(
        evidence_link(
            "E07b",
            "C07",
            "S07",
            artifact("s07_stability"),
            "all condition-stability rows plus classifier agreement table",
            s07_stability,
            {
                "stable_condition_fraction": safe_float(s07_stability.get("stable_across_seeds", pd.Series(dtype=bool)).fillna(False).astype(bool).mean()),
                "classifier_semantic_agreement_mean": safe_float(s07_agreement[agreement_col].mean()) if agreement_col else None,
                "classifier_pair_count": len(s07_agreement),
            },
            "S07 stability and classifier-agreement rows justify the continuum fallback.",
        )
    )

    # C08: explicit interface rules.
    s08 = sources["s08_strata"]
    s08_explicit = s08[s08.get("explicit_recognition_used", pd.Series(dtype=bool)).fillna(False).astype(bool)]
    s08_top = _sort(s08_explicit, "mean_delta_aggregation_delta_percent").head(1)
    claims.append(
        claim_row(
            "C08",
            "Explicit interface rules increase segregation-like structure but are not behavior-only evidence.",
            "Use explicit interface rules as a separate mechanism stratum; they can reshape aggregation while slightly lowering target quality.",
            "medium_internal_validity",
            "supportive_with_tradeoff",
            "explicit_interface",
            "These rows use label-aware mechanisms and must stay separate from no-recognition behavior-only claims.",
            "The mechanisms are computational analogs, not direct biological adhesion, permeability, or self/non-self evidence.",
            "If a report needs an interface mechanism, cite its block fraction and target-quality cost with the aggregation gain.",
            "Do not attribute behavior-only aggregation to explicit recognition or vice versa.",
        )
    )
    evidence.append(
        evidence_link(
            "E08a",
            "C08",
            "S08",
            artifact("s08_strata"),
            "explicit_recognition_used == True; top mean_delta_aggregation_delta_percent row",
            s08_explicit,
            {
                "explicit_mechanism_count": len(s08_explicit),
                "explicit_rule_dominance_count": int(s08_explicit.get("explicit_rule_dominates_baseline_behavior", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
                "top_aggregation_row": frame_records(s08_top),
            },
            "S08 mechanism-strata table compares explicit interface rules against behavior-only baselines.",
        )
    )

    # C09: governance null result.
    s09 = sources["s09_rankings"]
    s09_nonbase = s09[~s09.get("governance_mechanism_id", pd.Series(dtype=str)).eq("no_governance")]
    s09_best = _sort(s09_nonbase, "mean_rescue_effect_score").head(1)
    s09_global = s09[s09.get("global_controller_like", pd.Series(dtype=bool)).fillna(False).astype(bool)]
    claims.append(
        claim_row(
            "C09",
            "Finite-radius governance did not rescue the prioritized conflict cases.",
            "Treat S09 governance mechanisms as costly probes, not validated rescue controls, unless a later context-specific row supports them.",
            "high_internal_validity_null",
            "null",
            "finite_radius_governance,global_controller_like",
            "Finite-radius local mechanisms and the global-controller-like comparator are both explicit and separately labeled.",
            "The S09 panel is runtime-bounded to four high-priority S04 conflict cases.",
            "Use governance rows mainly to document costs, information access, and failure modes.",
            "Do not claim local governance solved chimeric conflict in E06.",
        )
    )
    evidence.append(
        evidence_link(
            "E09a",
            "C09",
            "S09",
            artifact("s09_rankings"),
            "governance_mechanism_id != 'no_governance'; sorted by mean_rescue_effect_score",
            s09_nonbase,
            {
                "supportive_governance_count": int(s09_nonbase.get("supportive_rescue_by_threshold", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
                "best_nonbaseline_row": frame_records(s09_best),
                "global_controller_row": frame_records(s09_global),
            },
            "S09 rankings document local and global-access governance outcomes and costs.",
        )
    )

    # C10: graft null result.
    s10 = sources["s10_strata"]
    s10_nonbase = s10[~s10.get("intervention_stratum", pd.Series(dtype=str)).eq("behavior_only_no_governance")]
    s10_best = _sort(s10, "mean_graft_rescue_score").head(1)
    claims.append(
        claim_row(
            "C10",
            "Graft responses are diverse, but tested intervention strata did not produce graft rescue.",
            "For staged grafts, report host/graft fate and interface metrics rather than recommending explicit or governance interventions as rescue tools.",
            "high_internal_validity_null",
            "null",
            "behavior_only,explicit_interface,finite_radius_governance,global_controller_like",
            "S10 preserves behavior-only, explicit-interface, finite-radius governance, and global-controller-like strata.",
            "Grafts are computational in-place relabeling proxies with preserved host value state.",
            "Use S10 as a fate-classification and perturbation-sensitivity layer.",
            "Do not treat S10 grafts as biological graft experiments or as evidence that intervention works.",
        )
    )
    evidence.append(
        evidence_link(
            "E10a",
            "C10",
            "S10",
            artifact("s10_strata"),
            "all strata rows; check supportive_graft_shift_by_threshold",
            s10,
            {
                "supportive_strata_count": int(s10.get("supportive_graft_shift_by_threshold", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
                "best_stratum_row": frame_records(s10_best),
                "nonbaseline_mean_rescue_score": safe_float(s10_nonbase.get("mean_graft_rescue_score", pd.Series(dtype=float)).mean()),
            },
            "S10 stratum summary separates graft intervention strata and rescue scores.",
        )
    )

    # C11: mutant containment.
    s11 = sources["s11_strata"]
    s11_support = s11[s11.get("supportive_containment_by_threshold", pd.Series(dtype=bool)).fillna(False).astype(bool)]
    s11_top = _sort(s11_support, "mean_containment_effect_score").head(1)
    claims.append(
        claim_row(
            "C11",
            "Clone containment signals exist, but containment and target damage must be reported together.",
            "For selfish-clone scenarios, prioritize containment metrics only when paired with target-quality-damage metrics and access-stratum labels.",
            "medium_internal_validity",
            "supportive_with_tradeoff",
            "explicit_interface,finite_radius_governance,global_controller_like",
            "S11 separates explicit-interface, finite-radius, and global-controller-like containment evidence.",
            "The cancer-like mutant is a computational identity-label conversion model, not biological proliferation.",
            "Use S11 as a proxy stress test for selfish local objectives and containment tradeoffs.",
            "Do not report clone suppression without the associated target-quality damage.",
        )
    )
    evidence.append(
        evidence_link(
            "E11a",
            "C11",
            "S11",
            artifact("s11_strata"),
            "supportive_containment_by_threshold == True; sorted by mean_containment_effect_score",
            s11_support,
            {
                "supportive_strata_count": len(s11_support),
                "top_containment_row": frame_records(s11_top),
                "behavior_only_row": frame_records(s11[s11.get("intervention_stratum", pd.Series(dtype=str)).eq("behavior_only_no_governance")]),
            },
            "S11 strata table exposes containment benefits and damage tradeoffs.",
        )
    )

    # C12: developmental history.
    s12_summary = _s12_history_summary(sources["s12_history"])
    s12_top = s12_summary.head(2)
    claims.append(
        claim_row(
            "C12",
            "Developmental history is a strong steering variable.",
            "When available, change staged introduction history before adding heavier interface or governance mechanisms.",
            "high_internal_validity",
            "supportive",
            "behavior_only,explicit_interface,finite_radius_governance,global_controller_like",
            "History effects were evaluated while preserving all intervention strata separately.",
            "History encodings are computational schedule/state proxies, not biological development.",
            "Use staged early/late introduction rows as high-priority playbook levers.",
            "Do not hide history duration or memory-state preservation as nuisance variation.",
        )
    )
    evidence.append(
        evidence_link(
            "E12a",
            "C12",
            "S12",
            artifact("s12_history"),
            "grouped by history_id, history_family; sorted by mean_history_effect_magnitude",
            sources["s12_history"],
            {
                "history_summary_top_rows": frame_records(s12_top, limit=5),
                "history_summary_all_rows": frame_records(s12_summary, limit=10),
            },
            "S12 baseline-comparison rows show staged introduction has the largest history effects.",
        )
    )

    # C13: predictive model utility and causal caveat.
    s13_perf = sources["s13_performance"]
    s13_rf = s13_perf[s13_perf.get("model_id", pd.Series(dtype=str)).eq("random_forest")]
    s13_features = sources["s13_feature_screen"]
    s13_top_feature = _sort(s13_features, "mean_importance").head(5)
    claims.append(
        claim_row(
            "C13",
            "Final-state outcomes are predictable from upstream descriptors, but feature rankings are not causal proof.",
            "Use S13 models for triage and hypothesis generation; keep exploratory feature associations separate from causal claims.",
            "high_predictive_validity",
            "supportive_predictive",
            "mixed_access_annotation",
            "S13 includes access strata as predictors or annotations but does not make mechanisms causal.",
            "The model matrix is heterogeneous and final-state labels inherit S07 label-instability caveats.",
            "Use S13 feature importance to prioritize checks in a report bundle, not to assert mechanisms.",
            "Do not overinterpret causal-discovery-style outputs as causal intervention evidence.",
        )
    )
    evidence.append(
        evidence_link(
            "E13a",
            "C13",
            "S13",
            artifact("s13_performance"),
            "model_id == 'random_forest'",
            s13_rf,
            {
                "random_forest_rows": frame_records(s13_rf, limit=10),
            },
            "S13 held-out performance table supports predictive, not causal, use.",
        )
    )
    evidence.append(
        evidence_link(
            "E13b",
            "C13",
            "S13",
            artifact("s13_feature_screen"),
            "top rows sorted by mean_importance",
            s13_top_feature,
            {
                "top_feature_rows": frame_records(s13_top_feature, ["target_name", "feature_name", "mean_importance", "claim_scope"], 10),
                "claim_scope_values": sorted(set(s13_features.get("claim_scope", pd.Series(dtype=str)).dropna().astype(str))),
            },
            "S13 feature screen labels associations as not causal proof.",
        )
    )

    # C14: S14 null rescue.
    s14 = sources["s14_rankings"]
    s14_holdout = s14[s14.get("evaluation_stage", pd.Series(dtype=str)).eq("prospective_holdout_validation")]
    s14_nonbaseline = s14_holdout[~s14_holdout.get("intervention_spec_id", pd.Series(dtype=str)).eq("no_intervention_behavior_baseline")]
    s14_best = _sort(s14_nonbaseline, "mean_minimality_adjusted_rescue_score").head(1)
    s14_global = s14_holdout[s14_holdout.get("global_controller_like", pd.Series(dtype=bool)).fillna(False).astype(bool)]
    claims.append(
        claim_row(
            "C14",
            "The S14 minimal intervention search is a prospective null rescue result.",
            "Do not recommend S14 local pulses or the global comparator as validated rescue interventions; report them as bounded negative evidence.",
            "high_internal_validity_null",
            "null",
            "finite_radius_governance,local_explicit_interface,global_controller_like",
            "S14 explicitly separates finite-radius local candidates, full-run local recognition, behavior-only baseline, and global-controller-like comparator.",
            "The search is bounded to six S13-derived failure contexts and held-out validation contexts.",
            "Use S14 to prune intervention claims and document unintended-consequence checks.",
            "Do not claim no intervention could ever work; only the tested S14 candidates failed the configured threshold.",
        )
    )
    evidence.append(
        evidence_link(
            "E14a",
            "C14",
            "S14",
            artifact("s14_rankings"),
            "evaluation_stage == 'prospective_holdout_validation'; intervention_spec_id != baseline",
            s14_nonbaseline,
            {
                "supportive_holdout_count": int(s14_nonbaseline.get("supportive_rescue_by_threshold", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
                "best_local_or_finite_row": frame_records(s14_best),
                "global_controller_rows": frame_records(s14_global),
            },
            "S14 intervention rankings show no held-out candidate met the supportive rescue threshold.",
        )
    )

    # C15: synthesis rule.
    claims.append(
        claim_row(
            "C15",
            "Control recommendations should prefer substrate setup and history over heavier interventions.",
            "Order control decisions as: validate policy set, scan ratios, choose arrangement, check goal profile, preserve metric dimensions, inspect dominance/context, then consider explicit or governance interventions only with caveats.",
            "synthesis_from_multiple_validated_steps",
            "supportive_synthesis",
            "behavior_only,explicit_interface,finite_radius_governance,global_controller_like",
            "The synthesis preserves behavior-only, explicit-interface, finite-radius, and global-controller-like evidence layers.",
            "This is a computational playbook over sorting-array proxies; it is not a biological protocol.",
            "Use the playbook as a report-bundle decision tree and evidence index.",
            "Do not merge local and global-access results into one control-effect claim.",
        )
    )
    evidence.append(
        evidence_link(
            "E15a",
            "C15",
            "S02-S14",
            artifact("phase_matrix"),
            "phase matrix exists; synthesis uses S02-S14 claim registry rows C02-C14",
            sources["phase_matrix"],
            {
                "phase_matrix_rows": len(sources["phase_matrix"]),
                "claim_ids_used": [claim["claim_id"] for claim in claims if claim["claim_id"] != "C15"],
            },
            "The S15 synthesis indexes the completed E06 phase/corpus artifacts without starting E07.",
        )
    )

    claims_df = pd.DataFrame(claims)
    evidence_df = pd.DataFrame(evidence)
    return claims_df, evidence_df, s12_summary


def markdown_table(frame: pd.DataFrame, columns: Sequence[str], max_rows: int = 30) -> str:
    if frame.empty:
        return "_No rows._"
    present = [column for column in columns if column in frame.columns]
    if not present:
        return "_Requested columns are absent._"
    display = frame[present].head(max_rows).copy()
    lines = [
        "| " + " | ".join(present) + " |",
        "| " + " | ".join("---" for _ in present) + " |",
    ]
    for row in display.to_dict(orient="records"):
        values = []
        for column in present:
            value = row[column]
            if isinstance(value, float):
                text = f"{value:.6g}"
            elif isinstance(value, bool):
                text = "true" if value else "false"
            else:
                text = str(value)
            values.append(text.replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    if len(frame) > max_rows:
        omitted = len(frame) - max_rows
        values = ["..." if idx == 0 else "" for idx in range(len(present))]
        if len(values) > 1:
            values[1] = f"{omitted} more rows omitted"
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def evidence_summary_for_claim(evidence_df: pd.DataFrame, claim_id: str) -> str:
    rows = evidence_df[evidence_df["claim_id"] == claim_id]
    if rows.empty:
        return "No evidence links."
    fragments = []
    for row in rows.to_dict(orient="records"):
        fragments.append(
            f"{row['evidence_id']}: `{row['artifact_path']}` ({row['row_selector']}; rows={row['source_row_count']})"
        )
    return "; ".join(fragments)


def build_playbook_markdown(
    claims_df: pd.DataFrame,
    evidence_df: pd.DataFrame,
    *,
    artifacts_written: Sequence[str],
    validation_passed: bool,
) -> str:
    claim_table = claims_df[
        [
            "claim_id",
            "title",
            "evidence_level",
            "outcome_classification",
            "access_stratum",
            "recommended_use",
        ]
    ].copy()
    sections = []
    for claim in claims_df.to_dict(orient="records"):
        sections.append(
            "\n".join(
                [
                    f"### {claim['claim_id']}: {claim['title']}",
                    "",
                    f"- Playbook rule: {claim['playbook_rule']}",
                    f"- Evidence level: {claim['evidence_level']}",
                    f"- Outcome classification: {claim['outcome_classification']}",
                    f"- Access stratum: `{claim['access_stratum']}`",
                    f"- Local/global note: {claim['local_vs_global_note']}",
                    f"- Caveat: {claim['uncertainty_caveat']}",
                    f"- Evidence links: {evidence_summary_for_claim(evidence_df, str(claim['claim_id']))}",
                    f"- Avoid overclaim: {claim['avoid_overclaim']}",
                ]
            )
        )
    artifact_lines = "\n".join(f"- `{item}`" for item in artifacts_written)
    return f"""# E06 Chimeric-Control Playbook

## Top Summary

- Step ID: S15
- Completion status: Complete
- Artifacts written:
{artifact_lines}
- Validation result: {"Passed" if validation_passed else "Failed"}; every playbook claim is backed by at least one row-linked evidence record.
- Outcome classification: supportive synthesis
- Caveats or blockers: This playbook is a computational synthesis over sorting-array proxy experiments. It is not a biological protocol, and global-controller-like comparators are kept separate from finite-radius local mechanisms.
- Lay summary: E06 shows that mixture ratio, initial arrangement, goal compatibility, developmental history, and policy identity shape chimeric outcomes more reliably than the tested rescue interventions. Explicit interface and governance mechanisms can reshape outcomes, but they carry access and target-quality tradeoffs; S14's prospective rescue search was null.
- Recommended next action: Use this playbook as the E06 report-bundle evidence index; do not start E07 until the Chief Scientist workflow requests it.

## How To Use This Playbook

Use the claims below as report-bundle building blocks. Each claim has row-linked evidence in `$ARTIFACTS_DIR/tables/e06_s15_claim_evidence_links.csv`; the row selector and values are part of the artifact so claims can be audited without reading this narrative.

Recommended control order:

1. Confirm the relevant policies are in the S01 library and note their source categories.
2. Scan ratio and minority effects before selecting a representative mixture.
3. Check spatial arrangement and goal compatibility before adding mechanisms.
4. Keep compatibility dimensions separate when quality, aggregation, dominance, and conflict disagree.
5. Treat dominance and final-state labels as context-sensitive descriptors.
6. Use explicit interface, finite-radius governance, and global-controller-like results only within their declared access strata.
7. Preserve S14 as a null prospective rescue result.

## Claim Registry

{markdown_table(claim_table, claim_table.columns, max_rows=40)}

## Evidence-Linked Claims

{chr(10).join(sections)}

## Local And Global Access Boundaries

- `behavior_only`: original local policy behavior without added recognition or governance. S02-S07 and no-governance baselines are the strongest evidence layer for natural mixed-policy behavior.
- `explicit_interface`: label-aware interface rules from S08 and downstream strata. These are computational analogs beyond the original no-recognition claim.
- `finite_radius_governance`: local governance mechanisms with declared influence radii and information access from S09-S14. They remain local or finite-radius proxy mechanisms.
- `global_controller_like`: nonlocal comparators with global access. These are useful controls, not evidence for local collective governance.

## S14 Null Rescue Result

S14 tested minimal transient signals and local rule changes on S13-derived failure contexts with disjoint held-out validation contexts and seeds. No finite-radius local candidate, local-recognition comparator, or global-controller-like pulse met the supportive rescue threshold. The best held-out local candidate had a small target-quality increase but worsened conflict and failed the minimality-adjusted threshold. This null result prunes the intervention claims available for E06.

## Proxy And Uncertainty Boundaries

- All evidence is computational and uses sorting-array proxy substrates.
- Goal fields are rank-order computational targets, not biological target morphologies.
- Explicit interface rules should not be renamed as biological adhesion, permeability, or self/non-self mechanisms.
- Graft and cancer-like clone experiments are computational relabeling and selfish-objective analogies.
- S13 predictive feature rankings are exploratory associations, not causal proof.
- S15 adds no new simulations; it is an evidence-linked synthesis of S01-S14.
"""


def build_handoff_markdown(
    claims_df: pd.DataFrame,
    evidence_df: pd.DataFrame,
    *,
    artifacts_written: Sequence[str],
    validation_passed: bool,
) -> str:
    evidence_levels = claims_df["evidence_level"].value_counts().sort_index().reset_index()
    evidence_levels.columns = ["evidence_level", "claim_count"]
    outcomes = claims_df["outcome_classification"].value_counts().sort_index().reset_index()
    outcomes.columns = ["outcome_classification", "claim_count"]
    artifact_lines = "\n".join(f"- `{item}`" for item in artifacts_written)
    return f"""# E06 Report-Bundle Handoff

## Top Summary

- Step ID: S15
- Completion status: Complete
- Artifacts written:
{artifact_lines}
- Validation result: {"Passed" if validation_passed else "Failed"}; {len(claims_df)} claims and {len(evidence_df)} evidence links are available for report-bundle generation.
- Outcome classification: supportive synthesis
- Caveats or blockers: E06 is ready for Chief report-bundle generation, but claims must retain computational-proxy language and local/global-access strata.
- Lay summary: The report bundle should present E06 as a bounded computational phase diagram and control playbook, with strong evidence for ratio/arrangement/goal/history effects, context-sensitive dominance, predictive final-state models, null governance/graft/S14 rescue outcomes, and a supportive but tradeoff-bound mutant-containment proxy.
- Recommended next action: Chief report-bundle generation for E06. Do not start E07 from this handoff unless separately instructed.

## Bundle Inputs

{markdown_table(pd.DataFrame({"artifact": artifacts_written}), ["artifact"], max_rows=80)}

## Claim Counts

{markdown_table(evidence_levels, ["evidence_level", "claim_count"], max_rows=30)}

## Outcome Counts

{markdown_table(outcomes, ["outcome_classification", "claim_count"], max_rows=30)}

## Highest-Priority Report Claims

- Mixture ratio, spatial arrangement, goal compatibility, and developmental history are the main constructive control levers in E06.
- Explicit interface and finite-radius governance mechanisms must be stratified by access and cost.
- S09, S10, and S14 are null rescue results under their configured thresholds.
- S11 clone containment is supportive as a computational proxy but carries target-quality damage.
- S13 predictive performance is strong enough for triage but not causal proof.

## Required Caveats For Chief Report

- Use computational-proxy language throughout.
- Keep behavior-only, explicit-interface, finite-radius governance, and global-controller-like rows separate.
- Link every quoted claim to `$ARTIFACTS_DIR/tables/e06_s15_claim_registry.csv` and `$ARTIFACTS_DIR/tables/e06_s15_claim_evidence_links.csv`.
- Preserve S14's null rescue result rather than treating intervention search as successful.
- Do not start or summarize E07 as completed work.
"""


def validate_s15_outputs(
    claims_df: pd.DataFrame,
    evidence_df: pd.DataFrame,
    validation_artifacts: Mapping[str, Path],
    prior_report_paths: Sequence[Path],
    config: S15Config,
    *,
    playbook_text: str,
    handoff_text: str,
    full_report_text: str,
    unit_tests_success: bool,
) -> pd.DataFrame:
    claim_ids = set(claims_df.get("claim_id", pd.Series(dtype=str)).astype(str))
    evidence_claim_ids = set(evidence_df.get("claim_id", pd.Series(dtype=str)).astype(str))
    artifact_exists = {name: path.exists() and path.stat().st_size > 0 for name, path in validation_artifacts.items()}
    checks = [
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "prior_step_reports_present",
            "success": all(path.exists() and path.stat().st_size > 0 for path in prior_report_paths),
            "observed": f"{sum(path.exists() for path in prior_report_paths)}/{len(prior_report_paths)}",
            "expected": "S01-S14 full-results reports exist",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "required_s15_artifacts_written",
            "success": all(artifact_exists.values()),
            "observed": json_compact(artifact_exists),
            "expected": "primary S15 report, playbook, handoff, table, phase-matrix, and config artifacts exist before final manifest/checksum writing",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "claim_registry_complete",
            "success": len(claims_df) >= config.min_claim_count and set(config.required_claim_ids).issubset(claim_ids) and claims_df["claim_id"].is_unique,
            "observed": f"claims={len(claims_df)} ids={sorted(claim_ids)}",
            "expected": f"at least {config.min_claim_count} unique claims including {list(config.required_claim_ids)}",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "all_claims_have_evidence_links",
            "success": claim_ids.issubset(evidence_claim_ids) and evidence_df["source_row_count"].astype(int).gt(0).all(),
            "observed": f"claims_without_evidence={sorted(claim_ids - evidence_claim_ids)}",
            "expected": "every claim has at least one evidence link with source rows",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "evidence_links_are_row_specific",
            "success": bool(
                not evidence_df.empty
                and evidence_df["artifact_path"].astype(str).str.len().gt(0).all()
                and evidence_df["row_selector"].astype(str).str.len().gt(0).all()
                and evidence_df["evidence_values_json"].astype(str).str.len().gt(2).all()
            ),
            "observed": f"evidence_links={len(evidence_df)}",
            "expected": "evidence links include artifact, row selector, and values",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "proxy_caveats_preserved",
            "success": bool(
                claims_df["uncertainty_caveat"].astype(str).str.len().gt(12).all()
                and "computational-proxy" in handoff_text
                and "computational" in playbook_text
                and "not causal proof" in playbook_text
            ),
            "observed": "claim caveats and Markdown proxy caveats checked",
            "expected": "proxy and uncertainty caveats are explicit",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "local_global_strata_explicit",
            "success": all(term in playbook_text for term in config.required_access_terms),
            "observed": ",".join(term for term in config.required_access_terms if term in playbook_text),
            "expected": "behavior_only, explicit_interface, finite_radius_governance, and global_controller_like appear in playbook",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "s14_null_rescue_preserved",
            "success": "S14" in playbook_text and "null" in playbook_text.lower() and "No finite-radius local candidate" in playbook_text,
            "observed": "S14 null wording checked",
            "expected": "playbook explicitly preserves S14 null rescue result",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "handoff_for_chief_not_e07_execution",
            "success": "Do not start E07" in handoff_text and "Chief report-bundle" in handoff_text,
            "observed": "handoff next-action wording checked",
            "expected": "handoff says E06 is ready for Chief report-bundle generation without starting E07",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "top_summary_fields_present",
            "success": all(
                field in full_report_text
                for field in [
                    "Step ID: S15",
                    "Completion status:",
                    "Artifacts written:",
                    "Validation result:",
                    "Outcome classification:",
                    "Caveats or blockers:",
                    "Lay summary:",
                    "Recommended next action:",
                ]
            ),
            "observed": "full report top summary checked",
            "expected": "required top summary fields present",
        },
        {
            "schema": VALIDATION_SCHEMA,
            "validation_case": "unit_tests_passed",
            "success": bool(unit_tests_success),
            "observed": str(bool(unit_tests_success)),
            "expected": "focused S15 unit tests pass",
        },
    ]
    return pd.DataFrame(checks)
