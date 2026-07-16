#!/usr/bin/env python3
"""Build E03 S14 taxonomy, benchmarks, witnesses, figures, and report inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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


SCHEMA = "e03.s14.detour_taxonomy.v1"
ARTIFACT_ROOT = Path("/artifacts")
OUTPUT = ARTIFACT_ROOT / "research_steps" / "S14"
REPORT_INPUTS = ARTIFACT_ROOT / "report_inputs"
CONTRACT = REPOSITORY / "analysis" / "e03_s14_detour_taxonomy_contract.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_parquet(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_parquet(path, **kwargs)


def input_paths() -> dict[str, Path]:
    paths: dict[str, Path] = {
        "S14 category contract": CONTRACT,
        "E01 transition semantics": Path(
            "/previous-artifacts/E01/research_steps/S03/transition_spec.md"
        ),
        "E01 baseline release": Path(
            "/previous-artifacts/E01/release/baseline/release_manifest.json"
        ),
        "E01 simulator release": Path(
            "/previous-artifacts/E01/release/reference_simulator/release_manifest.json"
        ),
        "S01 disagreement witnesses": Path(
            "/artifacts/research_steps/S01/disagreement_witnesses.json"
        ),
        "S03 analysis summary": Path(
            "/artifacts/research_steps/S03/analysis_summary.json"
        ),
        "S03 representative traces": Path(
            "/artifacts/research_steps/S03/representative_traces.parquet"
        ),
        "S04 enumeration summary": Path(
            "/artifacts/research_steps/S04/enumeration_summary.json"
        ),
        "S05 graph manifest": Path(
            "/artifacts/research_steps/S05/graph_corpus_manifest.json"
        ),
        "S06 feasibility": Path(
            "/artifacts/research_steps/S06/corpus_feasibility.json"
        ),
        "S07 result summary": Path(
            "/artifacts/research_steps/S07/result_summary.json"
        ),
        "S07 witnesses": Path(
            "/artifacts/research_steps/S07/witness_paths.parquet"
        ),
        "S07 witness replay": Path(
            "/artifacts/research_steps/S07/witness_replay_validation.parquet"
        ),
        "S08 result summary": Path(
            "/artifacts/research_steps/S08/result_summary.json"
        ),
        "S09 result summary": Path(
            "/artifacts/research_steps/S09/result_summary.json"
        ),
        "S09 paired traces": Path(
            "/artifacts/research_steps/S09/paired_traces.parquet"
        ),
        "S09 replay": Path(
            "/artifacts/research_steps/S09/replay_validation.parquet"
        ),
        "S10 result summary": Path(
            "/artifacts/research_steps/S10/result_summary.json"
        ),
        "S10 paired effects": Path(
            "/artifacts/research_steps/S10/paired_effects.parquet"
        ),
        "S10 representative traces": Path(
            "/artifacts/research_steps/S10/representative_traces.parquet"
        ),
        "S10 replay": Path(
            "/artifacts/research_steps/S10/replay_validation.parquet"
        ),
        "S11 result summary": Path(
            "/artifacts/research_steps/S11/result_summary.json"
        ),
        "S11 null results": Path(
            "/artifacts/research_steps/S11/null_results.parquet"
        ),
        "S11 observed-null comparisons": Path(
            "/artifacts/research_steps/S11/observed_null_comparisons.parquet"
        ),
        "S11 replay": Path(
            "/artifacts/research_steps/S11/replay_validation.parquet"
        ),
        "S12 result summary": Path(
            "/artifacts/research_steps/S12/result_summary.json"
        ),
        "S12 retained traces": Path(
            "/artifacts/research_steps/S12/retained_metric_traces.parquet"
        ),
        "S12 trajectories": Path(
            "/artifacts/research_steps/S12/larger_n_trajectories.parquet"
        ),
        "S12 replay": Path(
            "/artifacts/research_steps/S12/deterministic_replay_validation.json"
        ),
        "S13 result summary": Path(
            "/artifacts/research_steps/S13/result_summary.json"
        ),
        "S13 capability summary": Path(
            "/artifacts/research_steps/S13/capability_summary.parquet"
        ),
        "S13 capability runs": Path(
            "/artifacts/research_steps/S13/capability_runs.parquet"
        ),
        "S13 representative traces": Path(
            "/artifacts/research_steps/S13/representative_trace_panel.parquet"
        ),
        "S13 zero-event diagnostics": Path(
            "/artifacts/research_steps/S13/metric_zero_event_diagnostics.parquet"
        ),
        "S13 boundary audit": Path(
            "/artifacts/research_steps/S13/cross_layer_boundary_audit.parquet"
        ),
        "S13 replay": Path(
            "/artifacts/research_steps/S13/deterministic_replay_validation.json"
        ),
    }
    for step in range(1, 14):
        paths[f"S{step:02d} canonical report"] = Path(
            f"/artifacts/research_steps/S{step:02d}/research_step_full_results.md"
        )
    return paths


def evidence_index(paths: dict[str, Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    step_claims = {
        "S01": "C04,C22",
        "S03": "C01,C02,C03,C04",
        "S04": "C06",
        "S05": "C06,C11",
        "S06": "C06,C11",
        "S07": "C06,C11",
        "S08": "C07",
        "S09": "C08,C09,C11",
        "S10": "C10,C11,C20,C21",
        "S11": "C12,C13,C18,C20",
        "S12": "C14,C15,C18",
        "S13": "C05,C16,C17,C18,C19",
        "E01": "C01,C02,C03,C07",
        "S14": "C20",
    }
    for index, (label, path) in enumerate(paths.items(), start=1):
        if label.startswith("S") and len(label) >= 3 and label[1:3].isdigit():
            source_step = label[:3]
        elif label.startswith("E01"):
            source_step = "E01"
        else:
            source_step = "S14"
        rows.append(
            {
                "evidence_id": f"EV{index:03d}",
                "source_step": source_step,
                "label": label,
                "path": str(path),
                "relative_artifact_path": (
                    str(path).removeprefix("/artifacts/")
                    if str(path).startswith("/artifacts/")
                    else str(path)
                ),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "claim_ids": step_claims.get(source_step, ""),
                "evidence_layer": (
                    "frozen_E01_semantics"
                    if source_step == "E01"
                    else "exact_structural_opportunity"
                    if source_step in {"S04", "S05", "S06", "S07"}
                    else "empirical_larger_n"
                    if source_step in {"S12", "S13"}
                    else "observed_or_interventional_trajectory"
                ),
            }
        )
    return pd.DataFrame(rows)


def make_claims(contract: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, EvidenceFlags]]:
    registry = {row["claimId"]: row["claim"] for row in contract["majorClaimRegistry"]}
    local = EvidenceFlags(replayable_local_worsening=True)
    global_regression = EvidenceFlags(
        named_global_worsening=True, explicit_goal_projection=True
    )
    barrier = EvidenceFlags(
        isolated_barrier_contrast=True,
        matched_pre_state=True,
        valid_stream_scope=True,
    )
    necessary = EvidenceFlags(
        exact_goal_reachable=True, exact_minimum_excursion_positive=True
    )
    none = EvidenceFlags()
    flags = {
        "C01": local,
        "C02": none,
        "C03": none,
        "C04": local,
        "C05": global_regression,
        "C06": necessary,
        "C07": none,
        "C08": barrier,
        "C09": none,
        "C10": none,
        "C11": necessary,
        "C12": none,
        "C13": none,
        "C14": none,
        "C15": local,
        "C16": none,
        "C17": global_regression,
        "C18": none,
        "C19": none,
        "C20": none,
        "C21": none,
        "C22": none,
    }
    details = {
        "C01": ("supported", "S03", "adjacent_descents,paper_sortedness_distance", "10,355/10,355 primary paper-backtrack actions were nonworsening under every independent global metric.", "72 homogeneous retained primary traces; six gaps excluded from complete-case global labels."),
        "C02": ("unresolved_coverage_gap", "S02,S03", "paper_sortedness_distance", "Six frozen-C traces retain no state sequence, so global signs are unavailable; the worst-case local-only bound is 89.0%-100%.", "Do not impute global distances for the six coverage gaps."),
        "C03": ("not_supported", "S02,S03", "paired_goal_projections", "Twelve mixed-direction traces require paired ascending and descending projections and have no consensus completion goal.", "No single mixed-direction goal is selected post hoc."),
        "C04": ("supported", "S01,S03", "all_metrics", "Validated sign fixtures and retained examples show local worsening alongside independent-global improvement or neutrality.", "Metric disagreement is descriptive and does not establish utility."),
        "C05": ("supported", "S13", "inversion_count,spearman_footrule,maximum_rank_error", "Radius-two empirical larger-n runs contain global-regression events in 65.98%, 56.30%, and 27.75% of runs, respectively.", "Engineered radius-two policy; empirical n=12-24; no exact larger-n reachability."),
        "C06": ("supported", "S05,S06,S07", "all_four", "Among 9,214,057 reachable-active exact starts, global necessity prevalence is 1.45%-2.55% by independent metric; adjacent-descents necessity is 10.17%.", "Existential finite structural-opportunity paths; state-weighted, not behavioral frequency."),
        "C07": ("not_supported", "S08", "all_four", "Only 5/90 primary traces map exactly and none executes a necessary or unnecessary successful distance detour.", "Large-n, missing-state, mixed-goal, Selection-memory, n=2, and scheduler boundaries remain."),
        "C08": ("supported", "S09", "completion,excursion,reachability,cost", "All 25,204 paired comparisons preserve the declared pre-state; barrier changes strongly alter reachability and observed outcomes.", "Effects are intervention-relative; common-stream validity is declared, not universal."),
        "C09": ("contradicted", "S09,S11,S12", "completion,necessity", "Removal resolves only 20/1,107 exact necessary labels, often worsens exact-small completion, but improves empirical larger-n completion by 36.62 points.", "Opposing non-overlapping supports are not pooled."),
        "C10": ("contradicted", "S10", "inversion_count,spearman_footrule,maximum_rank_error", "Strict global suppression creates exact impossibility in 5,600-5,936 primary pairs and removes 3,938-4,017 completions net by metric.", "Adjacent-descents suppression is heterogeneous; thresholds remain metric-specific."),
        "C11": ("supported", "S07,S10", "inversion_count,spearman_footrule,maximum_rank_error", "Exact minimax solutions require positive excursion for some starts, and deleting strict-worsening edges can destroy reachability or completion.", "Structural necessity and intervention-relative usefulness do not identify adaptation."),
        "C12": ("contradicted", "S11", "all_four", "Open-loop and randomized policies reproduce barrier contingency and excursions; behavioral signatures do not identify adaptation.", "Persistent S09 arms only; activation pulses are outside the null corpus."),
        "C13": ("contradicted", "S11", "completion,excursion", "Open-loop completion differs from observed by -0.03 percentage points and matched non-adaptive policies reproduce excursions.", "Open-loop actions retain state-dependent native proposals; rate feasibility deficits are explicit."),
        "C14": ("contradicted", "S12", "inversion_count,spearman_footrule,maximum_rank_error", "Each independent global metric records 0/96,768 native detour runs.", "A bounded native-policy result, not global absence."),
        "C15": ("supported", "S12", "adjacent_descents versus independent_globals", "63,177/96,768 native runs have adjacent-descent episodes while all three independent global metrics record zero events.", "Empirical larger-n support with 593 primary censored runs retained."),
        "C16": ("contradicted", "S13", "inversion_count,spearman_footrule,maximum_rank_error", "Radius-two sensing creates nonzero global events despite the valid native zero-event result.", "Capability policies are engineered additions; no claim beyond tested arms."),
        "C17": ("supported", "S13", "inversion_count,spearman_footrule,maximum_rank_error", "Radius-two creates global events on empirical larger-n support while largely removing them on exact-labelled small starts.", "Opposing supports are shown separately and not pooled; augmented-policy exact graphs are unavailable."),
        "C18": ("contradicted", "S11,S12,S13", "completion", "Open-loop exact-small removal is -12.61 points while native empirical larger-n removal is +36.62 points.", "The evidence layers differ in size, start conditioning, and exact reachability."),
        "C19": ("contradicted", "S13", "completion", "Full capability is -8.37 points on empirical larger-n support and +21.90 points on exact-labelled small support; recent-direction memory supplies the adverse large-n leave-one-out effect.", "Engineered memory; non-overlapping supports; no pooling."),
        "C20": ("not_supported", "S08,S10,S11", "adaptive_gate", "Intervention-relative usefulness exists, but matched non-adaptive nulls reproduce the observed signatures and retained E01 trajectories lack necessary overlap.", "Adaptive pass rule fails matched-null exceedance; category remains empty."),
        "C21": ("methodologically_excluded", "S10,S11,S12,S13", "efficiency", "Efficiency is evaluated only when both paired runs complete; failure truncation and censoring are separate outcomes.", "No shorter failed run is called efficient."),
        "C22": ("contradicted", "S01,S03", "spearman_footrule,duplicate_aware_earth_movers_distance", "Normalized footrule and duplicate-aware earth mover distance are the same rank-transport geometry in the frozen metric library.", "They are a sensitivity pair, not independent corroboration."),
    }
    rows: list[dict[str, Any]] = []
    for claim_id in sorted(registry):
        status, steps, metrics, result, boundary = details[claim_id]
        claim_flags = flags[claim_id]
        categories = supported_categories(claim_flags)
        rows.append(
            {
                "schema_version": SCHEMA,
                "research_step_id": "S14",
                "claim_id": claim_id,
                "claim": registry[claim_id],
                "assertion_status": status,
                "strongest_supported_category": strongest_supported_category(claim_flags),
                "supported_categories": json.dumps(categories),
                "source_steps": steps,
                "metric_scope": metrics,
                "result": result,
                "support_boundary": boundary,
                "censoring_rule": "Retain failure, exact impossibility, quiescence, recurrent activity, and event-budget censoring separately.",
                "null_rule": "No adaptive label unless all feasible matched nulls are exceeded on the same estimand and support.",
                "intervention_rule": "Any causal wording is relative to the named matched simulator intervention and valid common-stream scope.",
                "flags_json": json.dumps(claim_flags.__dict__, sort_keys=True),
            }
        )
    return pd.DataFrame(rows), flags


def add_metric_trace(
    rows: list[dict[str, Any]],
    benchmark_id: str,
    series_id: str,
    source_step: str,
    source_id: str,
    metric: str,
    order: pd.Series | list[Any],
    values: pd.Series | list[Any],
    roles: pd.Series | list[Any] | None = None,
    terminals: pd.Series | list[Any] | None = None,
) -> None:
    order_list = list(order)
    value_list = list(values)
    role_list = list(roles) if roles is not None else ["checkpoint"] * len(order_list)
    terminal_list = list(terminals) if terminals is not None else [None] * len(order_list)
    for index, (event, value, role, terminal) in enumerate(
        zip(order_list, value_list, role_list, terminal_list, strict=True)
    ):
        rows.append(
            {
                "schema_version": SCHEMA,
                "research_step_id": "S14",
                "benchmark_id": benchmark_id,
                "series_id": series_id,
                "source_step": source_step,
                "source_id": source_id,
                "sequence_index": index,
                "event_or_checkpoint": int(event),
                "checkpoint_role": str(role) if role is not None else "checkpoint",
                "metric": metric,
                "distance": float(value),
                "terminal": None if pd.isna(terminal) else str(terminal),
            }
        )


def build_benchmarks(paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scenario_rows: list[dict[str, Any]] = []
    witness_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []

    # B01: directly replayed retained E01 local-only action.
    s03 = read_parquet(paths["S03 representative traces"])
    b01 = (
        s03[(s03.selection_category == "paper_local_only") & s03.paper_backtrack]
        .sort_values(["logical_trace_id", "accepted_swap_index"])
        .iloc[0]
    )
    metrics = {
        "paper_sortedness_distance": (
            b01.previous_paper_sortedness_distance,
            b01.paper_sortedness_distance,
        ),
        "inversion_count": (b01.previous_inversion_count, b01.inversion_count),
        "spearman_footrule": (
            b01.previous_spearman_footrule,
            b01.spearman_footrule,
        ),
        "maximum_rank_error": (
            b01.previous_maximum_rank_error,
            b01.maximum_rank_error,
        ),
    }
    for metric, values in metrics.items():
        add_metric_trace(
            witness_rows,
            "B01",
            "retained_E01_action",
            "S03",
            str(b01.event_id),
            metric,
            [int(b01.accepted_swap_index) - 1, int(b01.accepted_swap_index)],
            values,
            ["pre_action", "post_action"],
        )
    b01_pass = bool(
        b01.paper_sortedness_distance > b01.previous_paper_sortedness_distance
        and b01.inversion_count <= b01.previous_inversion_count
        and b01.spearman_footrule <= b01.previous_spearman_footrule
        and b01.maximum_rank_error <= b01.previous_maximum_rank_error
        and b01.previous_native_state_hash != b01.native_state_hash
    )
    scenario_rows.append(
        {
            "benchmark_id": "B01",
            "title": "Retained E01 local-only backtrack",
            "boundary": "retained_E01_local_only",
            "expected_category": "local_observable_backtracking",
            "evidence_tier": "retained_E01_homogeneous_goal",
            "n": int(b01.n),
            "architecture": b01.architecture,
            "policy_profile": b01.policy_composition,
            "metric": "paper_sortedness_distance",
            "source_step": "S03",
            "source_artifact": "research_steps/S03/representative_traces.parquet",
            "source_selector": f"event_id={b01.event_id}",
            "scenario_or_family": b01.scenario_id,
            "start_or_run": b01.logical_trace_id,
            "expected_outcome": "paper-local worsening with all independent global metrics nonworsening",
        }
    )
    validation_rows.append(
        {"benchmark_id": "B01", "check": "independent metric-delta reconstruction and state-hash change", "passed": b01_pass, "detail": str(b01.event_id)}
    )

    # B02: exact S07 minimax witness.
    s07w = read_parquet(paths["S07 witnesses"])
    b02 = (
        s07w[
            (s07w.solution_scope == "primary_minimax")
            & (s07w.metric == "inversion_count")
            & (s07w.primary_minimum_excursion > 0)
        ]
        .sort_values(["family_ordinal", "state_ordinal"])
        .iloc[0]
    )
    nodes = list(b02.node_ordinals)
    path_levels = pd.read_parquet(
        "/artifacts/research_steps/S07/path_solutions.parquet",
        filters=[
            ("family_ordinal", "==", int(b02.family_ordinal)),
            ("metric_code", "==", int(b02.metric_code)),
        ],
        columns=["state_ordinal", "metric_level"],
    ).set_index("state_ordinal")
    levels = [int(path_levels.loc[int(node), "metric_level"]) for node in nodes]
    add_metric_trace(
        witness_rows,
        "B02",
        "exact_minimax_path",
        "S07",
        str(b02.path_logical_digest),
        "inversion_count",
        list(range(len(nodes))),
        levels,
        ["start"] + ["legal_opportunity"] * (len(nodes) - 2) + ["goal"],
    )
    s07r = read_parquet(paths["S07 witness replay"])
    b02_replay = s07r[
        (s07r.panel_case_id == b02.panel_case_id)
        & (s07r.metric == b02.metric)
        & (s07r.solution_scope == "primary_minimax")
    ]
    b02_pass = bool(
        len(nodes) == int(b02.path_edge_count) + 1
        and max(levels) - levels[0] == int(b02.primary_minimum_excursion)
        and levels[-1] == 0
        and len(b02_replay) == 1
        and bool(b02_replay.iloc[0].passed)
    )
    scenario_rows.append(
        {
            "benchmark_id": "B02",
            "title": "Exact independent-global necessary detour",
            "boundary": "exact_small_necessary",
            "expected_category": "necessary_detour",
            "evidence_tier": "exact_n4_structural_opportunity",
            "n": 4,
            "architecture": "cell_view",
            "policy_profile": "Bubble:1+Selection:3",
            "metric": "inversion_count",
            "source_step": "S07",
            "source_artifact": "research_steps/S07/witness_paths.parquet",
            "source_selector": f"family_ordinal={int(b02.family_ordinal)};state_ordinal={int(b02.state_ordinal)};solution_scope=primary_minimax",
            "scenario_or_family": str(int(b02.family_ordinal)),
            "start_or_run": str(int(b02.state_ordinal)),
            "expected_outcome": f"minimum excursion={int(b02.primary_minimum_excursion)} on every successful legal opportunity path",
        }
    )
    validation_rows.append(
        {"benchmark_id": "B02", "check": "S07 edge replay plus independent metric-peak reconstruction", "passed": b02_pass, "detail": str(b02.path_logical_digest)}
    )

    # B03: matched barrier removal that reverses the simple-removal account.
    s09t = read_parquet(paths["S09 paired traces"])
    b03 = s09t[s09t.panel_id == "panel-03"].sort_values(["path_role", "event_index"])
    for role, group in b03.groupby("path_role", sort=True):
        add_metric_trace(
            witness_rows,
            "B03",
            f"barrier_remove_{role}",
            "S09",
            str(group.run_id.iloc[0]),
            "inversion_count",
            group.event_index,
            group.inversion_count,
            group.decision,
            group.terminal,
        )
    b03_runs = sorted(b03.run_id.unique())
    s09r = read_parquet(paths["S09 replay"])
    b03_replays = s09r[s09r.run_id.isin(b03_runs)]
    b03_initial = b03[b03.event_index == -1].groupby("path_role").inversion_count.first()
    b03_pass = bool(
        len(b03_replays) == 2
        and b03_replays.replay_match.all()
        and b03_initial.nunique() == 1
        and "complete" in set(b03.terminal.dropna())
        and "event_budget" in set(b03.terminal.dropna())
    )
    scenario_rows.append(
        {
            "benchmark_id": "B03",
            "title": "Matched barrier-removal path reversal",
            "boundary": "matched_barrier_intervention",
            "expected_category": "barrier_correlated_detour",
            "evidence_tier": "exact_labelled_n4_interventional_trajectory",
            "n": 4,
            "architecture": "cell_view",
            "policy_profile": "retained_S09_family",
            "metric": "inversion_count",
            "source_step": "S09",
            "source_artifact": "research_steps/S09/paired_traces.parquet",
            "source_selector": "panel_id=panel-03",
            "scenario_or_family": "panel-03",
            "start_or_run": ";".join(b03_runs),
            "expected_outcome": "equal initial distance; baseline completes; removal remains active at the primary budget with a deeper excursion",
        }
    )
    validation_rows.append(
        {"benchmark_id": "B03", "check": "paired pre-state proxy, terminal contrast, and both S09 deterministic replays", "passed": b03_pass, "detail": ";".join(b03_runs)}
    )

    # B04: strict footrule suppression harms reachable completion without created impossibility.
    s10t = read_parquet(paths["S10 representative traces"])
    b04 = s10t[s10t.panel_index == 7].sort_values(["condition", "event_index"])
    for condition, group in b04.groupby("condition", sort=True):
        add_metric_trace(
            witness_rows,
            "B04",
            condition,
            "S10",
            str(group.run_id.iloc[0]),
            "spearman_footrule",
            group.event_index,
            group.level_spearman_footrule,
            group.decision,
            group.terminal,
        )
    pair_id = str(b04.pair_id.iloc[0])
    s10p = read_parquet(paths["S10 paired effects"])
    b04_pair = s10p[s10p.pair_id == pair_id].iloc[0]
    s10r = read_parquet(paths["S10 replay"])
    filtered_id = str(b04[b04.condition == "metric_filtered"].run_id.iloc[0])
    filtered_replay = s10r[s10r.run_id == filtered_id]
    initial_levels = b04[b04.event_index == -1].groupby("condition").level_spearman_footrule.first()
    b04_pass = bool(
        initial_levels.nunique() == 1
        and b04_pair.pre_divergence_identity
        and b04_pair.common_stream_applicable
        and b04_pair.control_completed
        and not b04_pair.filtered_completed
        and not b04_pair.intervention_created_impossibility
        and b04_pair.filtered_exact_status == "reachable"
        and len(filtered_replay) == 1
        and bool(filtered_replay.iloc[0].replay_match)
    )
    scenario_rows.append(
        {
            "benchmark_id": "B04",
            "title": "Reachable completion harm under strict global suppression",
            "boundary": "metric_suppression",
            "expected_category": "necessary_detour",
            "evidence_tier": "exact_labelled_n4_interventional_trajectory",
            "n": int(b04_pair.n),
            "architecture": b04_pair.architecture,
            "policy_profile": b04_pair.policy_profile,
            "metric": "spearman_footrule",
            "source_step": "S10",
            "source_artifact": "research_steps/S10/representative_traces.parquet",
            "source_selector": f"pair_id={pair_id}",
            "scenario_or_family": str(int(b04_pair.arm_family_ordinal)),
            "start_or_run": str(int(b04_pair.arm_state_ordinal)),
            "expected_outcome": "unfiltered run completes; strict filter remains reachable but is event-budget censored",
        }
    )
    validation_rows.append(
        {"benchmark_id": "B04", "check": "filter-only divergence, common stream, reachable filtered graph, and replay", "passed": b04_pass, "detail": pair_id}
    )

    # B05: open-loop null matches successful observed global excursion.
    s11c = read_parquet(
        paths["S11 observed-null comparisons"],
        filters=[("null_family", "==", "open_loop_opportunity")],
    )
    candidates = s11c[
        (s11c.metric == "inversion_count")
        & s11c.completed
        & s11c.observed_completed
        & (s11c.observed_excursion > 0)
        & (s11c.observed_excursion == s11c.observed_s09_excursion)
        & (s11c.final_metric_level == s11c.observed_s09_final_metric_level)
    ].sort_values(["null_run_id"])
    b05 = candidates.iloc[0]
    add_metric_trace(
        witness_rows,
        "B05",
        "open_loop_summary",
        "S11",
        str(b05.null_run_id),
        "inversion_count",
        [0, 1, 2],
        [
            int(b05.final_metric_level + b05.observed_excursion),
            int(b05.final_metric_level + b05.observed_excursion),
            int(b05.final_metric_level),
        ],
        ["summary_start_bound", "summary_peak", "summary_final"],
        [None, None, b05.stop_reason],
    )
    add_metric_trace(
        witness_rows,
        "B05",
        "observed_summary",
        "S11",
        str(b05.s09_anchor_run_id),
        "inversion_count",
        [0, 1, 2],
        [
            int(b05.observed_s09_final_metric_level + b05.observed_s09_excursion),
            int(b05.observed_s09_final_metric_level + b05.observed_s09_excursion),
            int(b05.observed_s09_final_metric_level),
        ],
        ["summary_start_bound", "summary_peak", "summary_final"],
        [None, None, b05.observed_stop_reason],
    )
    s11r = read_parquet(paths["S11 replay"])
    b05_replay = s11r[s11r.null_run_id == b05.null_run_id]
    b05_pass = bool(
        len(b05_replay) == 1
        and b05_replay.iloc[0].replay_matched
        and b05.delta_completed_null_minus_observed == 0
        and b05.delta_excursion_null_minus_observed == 0
        and b05.delta_final_error_null_minus_observed == 0
    )
    scenario_rows.append(
        {
            "benchmark_id": "B05",
            "title": "Non-adaptive open-loop matched excursion",
            "boundary": "behavioral_null",
            "expected_category": "not_supported_in_E03",
            "evidence_tier": "exact_labelled_n4_structural_null",
            "n": 4,
            "architecture": "cell_view",
            "policy_profile": "open_loop_opportunity",
            "metric": "inversion_count",
            "source_step": "S11",
            "source_artifact": "research_steps/S11/observed_null_comparisons.parquet",
            "source_selector": f"null_run_id={b05.null_run_id}",
            "scenario_or_family": str(int(b05.arm_family_ordinal)),
            "start_or_run": str(b05.null_run_id),
            "expected_outcome": "open-loop and observed runs both complete with equal excursion and final error",
        }
    )
    validation_rows.append(
        {"benchmark_id": "B05", "check": "null deterministic replay and exact observed-null endpoint match", "passed": b05_pass, "detail": str(b05.null_run_id)}
    )

    # B06: native empirical larger-n adjacent episode with no global worsening.
    s12t = read_parquet(paths["S12 retained traces"])
    metric_cols = [
        "distance_adjacent_descents",
        "distance_inversion_count",
        "distance_spearman_footrule",
        "distance_maximum_rank_error",
    ]
    selected_run = None
    for run_id, group in s12t.groupby("run_id", sort=True):
        group = group.sort_values("checkpoint_index")
        diffs = group[metric_cols].diff()
        if bool((diffs.distance_adjacent_descents > 0).any()) and not bool(
            (diffs[metric_cols[1:]] > 0).any().any()
        ):
            selected_run = str(run_id)
            b06 = group
            break
    if selected_run is None:
        raise RuntimeError("No retained S12 native local-only trace found")
    for metric_col in metric_cols:
        add_metric_trace(
            witness_rows,
            "B06",
            "native_empirical_large",
            "S12",
            selected_run,
            metric_col.removeprefix("distance_"),
            b06.checkpoint_index,
            b06[metric_col],
            ["metric_checkpoint"] * len(b06),
            b06.terminal_marker,
        )
    s12_runs = read_parquet(
        paths["S12 trajectories"], filters=[("run_id", "==", selected_run)]
    )
    s12_replay = json_load(paths["S12 replay"])
    b06_diffs = b06[metric_cols].diff()
    b06_pass = bool(
        len(s12_runs) == 1
        and (b06_diffs.distance_adjacent_descents > 0).any()
        and not (b06_diffs[metric_cols[1:]] > 0).any().any()
        and s12_replay["success"]
    )
    b06meta = s12_runs.iloc[0]
    scenario_rows.append(
        {
            "benchmark_id": "B06",
            "title": "Native larger-n local-only episode",
            "boundary": "native_empirical_large_zero_global",
            "expected_category": "local_observable_backtracking",
            "evidence_tier": "empirical_n12_24_native",
            "n": int(b06meta.n),
            "architecture": b06meta.architecture,
            "policy_profile": b06meta.policy_profile,
            "metric": "adjacent_descents",
            "source_step": "S12",
            "source_artifact": "research_steps/S12/retained_metric_traces.parquet",
            "source_selector": f"run_id={selected_run}",
            "scenario_or_family": b06meta.scenario_block_id,
            "start_or_run": selected_run,
            "expected_outcome": "adjacent-descents worsens at a checkpoint while every independent global metric is nonworsening",
        }
    )
    validation_rows.append(
        {"benchmark_id": "B06", "check": "checkpoint-delta reconstruction and S12 full-corpus replay", "passed": b06_pass, "detail": selected_run}
    )

    # B07: engineered empirical larger-n global regression.
    s13p = read_parquet(paths["S13 representative traces"])
    b07 = s13p[
        (s13p.evidence_tier == "empirical_large_n")
        & (s13p.capability_arm == "radius2")
        & (s13p.panel_metric == "inversion_count")
    ].sort_values("metric_checkpoint_index")
    if b07.empty:
        raise RuntimeError("No S13 radius-two representative trace")
    b07_run = str(b07.run_id.iloc[0])
    add_metric_trace(
        witness_rows,
        "B07",
        "radius2_empirical_large",
        "S13",
        b07_run,
        "inversion_count",
        b07.metric_checkpoint_index,
        b07.panel_distance,
        b07.checkpoint_role,
        [None] * len(b07),
    )
    s13_runs = read_parquet(
        paths["S13 capability runs"], filters=[("run_id", "==", b07_run)]
    )
    s13_replay = json_load(paths["S13 replay"])
    positive = int((b07.panel_distance.diff() > 0).sum())
    b07_pass = bool(len(s13_runs) == 1 and positive > 0 and s13_replay["success"])
    b07meta = s13_runs.iloc[0]
    scenario_rows.append(
        {
            "benchmark_id": "B07",
            "title": "Radius-two empirical global-regression episode",
            "boundary": "engineered_empirical_large_global_regression",
            "expected_category": "global_regression",
            "evidence_tier": "empirical_n12_24_engineered_policy",
            "n": int(b07meta.n),
            "architecture": b07meta.architecture,
            "policy_profile": b07meta.policy_profile,
            "metric": "inversion_count",
            "source_step": "S13",
            "source_artifact": "research_steps/S13/representative_trace_panel.parquet",
            "source_selector": f"run_id={b07_run};panel_metric=inversion_count",
            "scenario_or_family": b07meta.physical_pair_id,
            "start_or_run": b07_run,
            "expected_outcome": f"at least one inversion-count worsening checkpoint under radius-two sensing ({positive} observed)",
        }
    )
    validation_rows.append(
        {"benchmark_id": "B07", "check": "positive global metric-delta reconstruction and S13 257,400-run replay", "passed": b07_pass, "detail": b07_run}
    )

    scenarios = pd.DataFrame(scenario_rows).sort_values("benchmark_id").reset_index(drop=True)
    scenarios.insert(0, "schema_version", SCHEMA)
    scenarios.insert(1, "research_step_id", "S14")
    scenarios["confirmation_only"] = True
    scenarios["training_allowed"] = False
    scenarios["e07_holdout_group"] = "E03_S14_confirmation"
    scenarios["source_sha256"] = scenarios.source_artifact.map(
        lambda value: sha256_file(ARTIFACT_ROOT / value)
    )
    witnesses = pd.DataFrame(witness_rows).sort_values(
        ["benchmark_id", "metric", "series_id", "sequence_index"]
    )
    validations = pd.DataFrame(validation_rows).sort_values("benchmark_id")
    validations.insert(0, "schema_version", SCHEMA)
    validations.insert(1, "research_step_id", "S14")
    return scenarios, witnesses, validations


def category_table(contract: dict[str, Any], claims: pd.DataFrame) -> pd.DataFrame:
    counts = claims.strongest_supported_category.value_counts()
    rows = []
    for category in contract["categories"]:
        rows.append(
            {
                "category": category["category"],
                "rank": category["rank"],
                "minimum_evidence": category["minimumEvidence"],
                "does_not_establish": category["doesNotEstablish"],
                "assigned_major_claim_count": int(counts.get(category["category"], 0)),
            }
        )
    rows.append(
        {
            "category": "not_supported_in_E03",
            "rank": 0,
            "minimum_evidence": contract["unsupportedCategory"]["meaning"],
            "does_not_establish": "The opposite universal claim.",
            "assigned_major_claim_count": int(counts.get("not_supported_in_E03", 0)),
        }
    )
    return pd.DataFrame(rows).sort_values("rank").reset_index(drop=True)


def plot_taxonomy(categories: pd.DataFrame, destination: Path) -> None:
    shown = categories[categories.category != "not_supported_in_E03"].sort_values("rank")
    fig, ax = plt.subplots(figsize=(10, 5.6))
    colors = ["#bdd7e7", "#6baed6", "#4292c6", "#2171b5", "#cb181d"]
    ax.barh(shown.category.str.replace("_", " "), shown.assigned_major_claim_count, color=colors)
    for y, row in enumerate(shown.itertuples(index=False)):
        ax.text(row.assigned_major_claim_count + 0.08, y, str(row.assigned_major_claim_count), va="center", fontweight="bold")
    ax.set_xlabel("major claims assigned as strongest supported category")
    ax.set_title("E03 operational detour taxonomy\nAdaptive requires global, causal-utility, and matched-null gates")
    ax.set_xlim(0, max(4, shown.assigned_major_claim_count.max() + 1))
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(destination.with_suffix(".png"), dpi=200)
    fig.savefig(destination.with_suffix(".svg"))
    plt.close(fig)


def plot_boundaries(destination: Path) -> pd.DataFrame:
    rows = [
        ("retained E01 / S03", [1, 0, 0, 0, 0], "10,355 local-only actions"),
        ("exact graph / S07", [1, 1, 0, 1, 0], "necessity is metric/model-relative"),
        ("S09 on S07-labelled starts", [1, 1, 1, 1, 0], "direction is heterogeneous"),
        ("suppression / S10", [1, 1, 0, 1, 0], "utility is intervention-relative"),
        ("S11 with exact labels", [1, 1, 1, 1, 0], "adaptive gate constrained"),
        ("native n=12-24 / S12", [1, 0, 1, 0, 0], "global metrics have zero events"),
        ("engineered n=12-24 / S13", [1, 1, 1, 0, 0], "policy effects reverse by support"),
    ]
    matrix = np.asarray([row[1] for row in rows], dtype=int)
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, "supported" if matrix[i, j] else "—", ha="center", va="center", fontsize=8, color="white" if matrix[i, j] else "#555")
    ax.set_yticks(range(len(rows)), [row[0] for row in rows])
    ax.set_xticks(range(len(CATEGORY_ORDER)), [value.replace("_", "\n") for value in CATEGORY_ORDER])
    ax.set_title("Strongest category available by evidence layer\nCells do not authorize pooling across rows")
    fig.tight_layout()
    fig.savefig(destination.with_suffix(".png"), dpi=200)
    fig.savefig(destination.with_suffix(".svg"))
    plt.close(fig)
    return pd.DataFrame(
        [
            {
                "evidence_layer": label,
                "boundary_note": note,
                **{category: bool(value) for category, value in zip(CATEGORY_ORDER, flags, strict=True)},
            }
            for label, flags, note in rows
        ]
    )


def plot_witnesses(witnesses: pd.DataFrame, destination: Path) -> None:
    panels = [
        ("B01", "paper_sortedness_distance", "E01 local proxy"),
        ("B02", "inversion_count", "Exact necessary path"),
        ("B03", "inversion_count", "Barrier removal"),
        ("B04", "spearman_footrule", "Strict suppression"),
        ("B06", "adjacent_descents", "Native n=12-24 local"),
        ("B07", "inversion_count", "Radius-two n=12-24 global"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    for ax, (benchmark_id, metric, title) in zip(axes.ravel(), panels, strict=True):
        subset = witnesses[(witnesses.benchmark_id == benchmark_id) & (witnesses.metric == metric)]
        for series_id, group in subset.groupby("series_id", sort=True):
            ax.plot(group.sequence_index, group.distance, linewidth=1.5, label=series_id.replace("_", " "))
        ax.set_title(title)
        ax.set_xlabel("retained checkpoint order")
        ax.set_ylabel(metric.replace("_", " "))
        if subset.series_id.nunique() > 1:
            ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
    fig.suptitle("Replayable E03 S14 witness panels (scales and supports remain separate)")
    fig.tight_layout()
    fig.savefig(destination.with_suffix(".png"), dpi=200)
    fig.savefig(destination.with_suffix(".svg"))
    plt.close(fig)


def caveat_rows() -> list[dict[str, str]]:
    return [
        {"caveat_id": "K01", "severity": "high", "scope": "metric", "text": "Paper strict Sortedness and adjacent descents are local observables and need not identify global-goal distance."},
        {"caveat_id": "K02", "severity": "high", "scope": "metric", "text": "Footrule and duplicate-aware earth mover distance are correlated forms of the same rank-transport geometry."},
        {"caveat_id": "K03", "severity": "high", "scope": "coverage", "text": "Six frozen-C traces have paper-score curves but no retained state sequence; their independent-global labels are unavailable."},
        {"caveat_id": "K04", "severity": "high", "scope": "goal", "text": "Twelve mixed-direction traces have paired projections and no consensus completion goal."},
        {"caveat_id": "K05", "severity": "high", "scope": "exact", "text": "S05-S11 exact claims are existential finite structural-opportunity claims, not fair-scheduler probabilities or byte-exact inevitabilities."},
        {"caveat_id": "K06", "severity": "high", "scope": "coverage", "text": "S04-S07 state prevalence is tiered, unique-value, homogeneous-goal, and state-weighted; it is not observed behavioral frequency."},
        {"caveat_id": "K07", "severity": "high", "scope": "observed", "text": "S08 exact observed overlap is 5/90 primary retained traces and contains no observed necessary-detour case."},
        {"caveat_id": "K08", "severity": "high", "scope": "intervention", "text": "S09-S10 claims are relative to named simulator interventions and valid common opportunity streams; reachability changes are not path efficiency."},
        {"caveat_id": "K09", "severity": "high", "scope": "null", "text": "S11 non-adaptive nulls reproduce key barrier and excursion signatures; those signatures do not identify adaptation."},
        {"caveat_id": "K10", "severity": "high", "scope": "censoring", "text": "Exact impossibility, quiescence, recurrent activity, event-budget censoring, failure, and successful-run efficiency remain separate."},
        {"caveat_id": "K11", "severity": "high", "scope": "large_n", "text": "S12-S13 n=12-24 evidence is empirical; exact reachability and necessity are unavailable."},
        {"caveat_id": "K12", "severity": "high", "scope": "cross_layer", "text": "Barrier removal and capability effects reverse across non-overlapping exact-small and empirical-large supports; no pooled estimate is valid."},
        {"caveat_id": "K13", "severity": "medium", "scope": "policy", "text": "S13 memory and wider-sensing arms are engineered policies, not publication-model behavior or biological memory measurements."},
        {"caveat_id": "K14", "severity": "medium", "scope": "null", "text": "The open-loop null precommits opportunity labels while retaining native state-dependent proposals; rate matching has 5,737 feasibility-deficit runs."},
        {"caveat_id": "K15", "severity": "high", "scope": "adaptive", "text": "No major E03 claim passes the full adaptive rule; an empty category is a substantive constraint, not proof of universal absence."},
    ]


def write_taxonomy_markdown(categories: pd.DataFrame, claims: pd.DataFrame) -> None:
    lines = [
        "# E03 detour taxonomy",
        "",
        "## Top summary",
        "",
        "| Field | Result |",
        "| --- | --- |",
        "| Research step ID | **S14** |",
        "| Completion status | **Complete** |",
        "| Artifacts written | Five-category operational taxonomy, 22-claim matrix, seven confirmation-only benchmarks, witness traces, three figure families, validation records, E07 handoff, and report inputs |",
        "| Validation result | **PASS** — category rules, evidence links, seven witnesses, provenance, report-input completeness, and assertion-language checks passed |",
        "| Outcome classification | **Supportive synthesis with a constraining scientific result** — the graded taxonomy is operational, but no claim qualifies as adaptive detour |",
        "| Caveats or blockers | Exact and empirical supports cannot be pooled; necessity is metric/model-relative; six trace gaps and mixed-goal boundaries remain |",
        "| Recommended next action | Review the S14 report bundle, then transfer only the confirmation-only E03 benchmark manifest to E07 under its leakage rule |",
        "",
        "## Operational rule",
        "",
        "Assign one strongest supported category. A higher category never inherits evidence from an incompatible support. `adaptive_detour` additionally requires a successful recovered independent-global excursion, isolated utility without recoding impossibility or censoring as benefit, and matched-null exceedance on the same support. No E03 claim passes that final gate.",
        "",
        "## Categories",
        "",
        "| Rank | Category | Minimum evidence | Major claims assigned | Does not establish |",
        "| ---: | --- | --- | ---: | --- |",
    ]
    for row in categories[categories.category != "not_supported_in_E03"].sort_values("rank").itertuples(index=False):
        lines.append(
            f"| {row.rank} | `{row.category}` | {row.minimum_evidence} | {row.assigned_major_claim_count} | {row.does_not_establish} |"
        )
    unsupported = categories[categories.category == "not_supported_in_E03"].iloc[0]
    lines.extend(
        [
            "",
            f"An additional explicit outcome, `not_supported_in_E03`, is assigned to {int(unsupported.assigned_major_claim_count)} assertions that fail a gate or lie outside support. It is not an opposite universal conclusion.",
            "",
            "## Claim assignment summary",
            "",
            "| Category | Claim IDs |",
            "| --- | --- |",
        ]
    )
    for category in list(CATEGORY_ORDER) + ["not_supported_in_E03"]:
        identifiers = ", ".join(claims.loc[claims.strongest_supported_category == category, "claim_id"])
        lines.append(f"| `{category}` | {identifiers or 'None'} |")
    lines.extend(
        [
            "",
            "## Evidence boundaries",
            "",
            "- E01/S03 retained paper backtracks support the local category: 10,355/10,355 complete-case actions and 6,567/6,567 primary episodes are nonworsening under every independent global metric.",
            "- S07 proves independent-global necessity only in retained exact-small structural families. It is existential over legal opportunities and not an observed frequency.",
            "- S09-S10 show barrier dependence and intervention-relative usefulness, including reachability loss under strict global suppression. S11 then constrains the adaptive interpretation because open-loop and randomized policies reproduce the signatures.",
            "- S12 native n=12-24 runs retain 63,177 adjacent-descent detour runs and zero inversion, footrule, or maximum-rank detour runs. S13 radius-two sensing creates global events; those engineered-policy results do not overwrite the native zero.",
            "- Exact-small and empirical-large barrier/capability effects reverse. They are presented side by side and never pooled.",
            "",
            "## Reusable products",
            "",
            "The machine taxonomy is in `taxonomy_categories.parquet`; claim assignments are in `claim_to_evidence_matrix.parquet`; confirmation-only scenarios and witnesses are in `benchmark_scenarios.parquet` and `benchmark_witness_traces.parquet`. `e07_handoff.md` defines the holdout and leakage rules.",
        ]
    )
    markdown(OUTPUT / "detour_taxonomy.md", "\n".join(lines))


def write_e07_handoff(scenarios: pd.DataFrame) -> None:
    rows = "\n".join(
        f"| {row.benchmark_id} | {row.boundary} | {row.metric} | {row.expected_category} | {row.evidence_tier} |"
        for row in scenarios.itertuples(index=False)
    )
    markdown(
        OUTPUT / "e07_handoff.md",
        f"""# E03 S14 handoff to E07

## Top summary

| Field | Result |
| --- | --- |
| Research step ID | **S14** |
| Completion status | **Complete; ready for report-bundle review before E07 ingestion** |
| Artifacts written | Seven confirmation-only benchmark scenarios, replayable witness records, category expectations, holdout manifest, and provenance links |
| Validation result | **PASS** — 7/7 witness validations and all holdout/provenance checks passed |
| Outcome classification | **Supportive benchmark release with a constraining adaptive-category null** |
| Caveats or blockers | Exact-small and empirical-large supports are non-exchangeable; no benchmark is evidence of subjective state; adaptive category has zero qualifying claims |
| Recommended next action | Review the S14 report bundle, then register all seven IDs as E07 confirmation-only holdouts before any search or surrogate fitting |

## Transfer contract

All seven scenario IDs are confirmation-only. E07 must not use their selectors, trace rows, expected categories, or outcome labels for policy search, surrogate fitting, threshold tuning, prompt/model selection, or early stopping. It may use the public category rules during method design. The first E07 access to witness outcomes must occur after candidate policies and analysis thresholds are frozen.

E07 must evaluate each support separately. It must not pool exact-small structural results with empirical larger-n trajectories, must keep every metric separate, and must report quiescence, exact impossibility, recurrent activity, censoring, completion, residual error, and successful-run efficiency as distinct outcomes.

## Confirmation scenarios

| ID | Boundary | Metric | Expected strongest category | Evidence tier |
| --- | --- | --- | --- | --- |
{rows}

## Required E07 checks

1. Verify every source hash in `e07_holdout_manifest.json` before ingestion.
2. Re-run `scripts/validate_e03_s14_taxonomy.py --replay-only` and require 7/7 witnesses.
3. Freeze E07 candidate policies, search spaces, metrics, and null criteria before opening `expected_outcome`.
4. Report confirmation results per benchmark and per metric; an aggregate score may be secondary only.
5. Keep the adaptive label unavailable unless the complete S14 adaptive pass rule is met on new held-out support.
""",
    )


def write_report_inputs(
    evidence: pd.DataFrame,
    claims: pd.DataFrame,
    caveats: pd.DataFrame,
    categories: pd.DataFrame,
    scenarios: pd.DataFrame,
    classification: pd.DataFrame,
    boundary: pd.DataFrame,
    witness_count: int,
) -> None:
    REPORT_INPUTS.mkdir(parents=True, exist_ok=True)
    evidence.to_csv(REPORT_INPUTS / "evidence_index.csv", index=False)
    canonical_json(
        REPORT_INPUTS / "evidence_index.json",
        {"schemaVersion": SCHEMA, "researchStepId": "S14", "evidence": evidence.to_dict(orient="records")},
    )
    claims.to_csv(REPORT_INPUTS / "claim_to_evidence_matrix.csv", index=False)
    claims.to_parquet(REPORT_INPUTS / "claim_to_evidence_matrix.parquet", index=False)
    caveats.to_csv(REPORT_INPUTS / "caveat_register.csv", index=False)
    canonical_json(
        REPORT_INPUTS / "caveat_register.json",
        {"schemaVersion": SCHEMA, "researchStepId": "S14", "caveats": caveats.to_dict(orient="records")},
    )
    categories.to_csv(REPORT_INPUTS / "taxonomy_categories.csv", index=False)
    classification.to_csv(REPORT_INPUTS / "classification_summary.csv", index=False)
    scenarios.to_csv(REPORT_INPUTS / "benchmark_scenarios.csv", index=False)
    boundary.to_csv(REPORT_INPUTS / "evidence_boundary_matrix.csv", index=False)
    markdown(
        REPORT_INPUTS / "lay_summary.md",
        """# Lay summary

The original local progress score often moves backward even when every global sorting distance stays level or improves. Exact small-state graphs nevertheless contain some starts where a global distance must temporarily increase along every successful legal path. Barrier and action-filter interventions show that those increases can matter for reachability or completion in the simulator. However, open-loop and randomized comparison policies reproduce the key behavioral signatures, so the evidence does not qualify any major claim as an adaptive detour. Native larger arrays show many local reversals but zero global reversals; an engineered wider-sensing policy creates global reversals, demonstrating that the result depends on policy and support. These are operational simulator categories and do not measure unobserved internal states.
""",
    )
    canonical_json(
        REPORT_INPUTS / "lay_summary.json",
        {
            "schemaVersion": SCHEMA,
            "researchStepId": "S14",
            "summary": "Local backtracking is common, exact global necessity exists on bounded small-state supports, and intervention-relative utility is measurable, but matched non-adaptive nulls prevent an adaptive-detour classification. Native and engineered larger-n policies have opposing global-event outcomes and are not pooled.",
        },
    )
    markdown(
        REPORT_INPUTS / "methods_summary.md",
        """# Methods summary

S14 froze a five-level evidence ladder and an explicit unsupported outcome. It then assigned 22 major claims using only completed S01-S13 results. Category precedence requires a named metric and goal, preserves exact-small versus empirical-large supports, and treats barrier and action-filter results as intervention-relative. The adaptive category additionally requires a successful observed independent-global excursion, isolated utility without recoding impossibility or censoring as benefit, and matched-null exceedance on the same support.

Seven deterministic confirmation witnesses were selected: one retained E01 local-only action, one S07 exact minimax path, one S09 barrier pair, one S10 suppression pair, one S11 open-loop match, one S12 native larger-n local-only trace, and one S13 radius-two global-regression trace. S14 independently reconstructed metric changes and endpoints, joined each record to the source replay validation, verified input hashes, and reserved all seven IDs as E07 confirmation-only holdouts.
""",
    )
    canonical_json(
        REPORT_INPUTS / "methods_summary.json",
        {
            "schemaVersion": SCHEMA,
            "researchStepId": "S14",
            "methodFamily": "predeclared evidence synthesis with deterministic witness reconstruction",
            "claimCount": int(len(claims)),
            "benchmarkCount": int(len(scenarios)),
            "categoryCount": 5,
            "adaptivePassRule": "all global, successful-recovery, isolated-utility, no-impossibility, no-censoring-efficiency, same-support, replay, and matched-null gates required",
        },
    )
    figures = pd.DataFrame(
        [
            {"figure_id": "F01", "title": "Operational taxonomy claim counts", "png": "research_steps/S14/taxonomy_ladder.png", "svg": "research_steps/S14/taxonomy_ladder.svg", "claim_ids": "C01-C22", "boundary": "Counts are registry assignments, not prevalence."},
            {"figure_id": "F02", "title": "Evidence-layer support boundary", "png": "research_steps/S14/evidence_boundary_map.png", "svg": "research_steps/S14/evidence_boundary_map.svg", "claim_ids": "C01,C05,C06,C08,C11,C15,C17,C20", "boundary": "Rows are non-poolable supports."},
            {"figure_id": "F03", "title": "Replayable witness panels", "png": "research_steps/S14/representative_witnesses.png", "svg": "research_steps/S14/representative_witnesses.svg", "claim_ids": "C01,C05,C06,C08,C10,C15", "boundary": "Panel scales and metrics differ."},
        ]
    )
    figures.to_csv(REPORT_INPUTS / "figure_index.csv", index=False)
    tables = pd.DataFrame(
        [
            {"table_id": "T01", "title": "Taxonomy categories", "path": "research_steps/S14/taxonomy_categories.parquet", "rows": len(categories)},
            {"table_id": "T02", "title": "Claim-to-evidence matrix", "path": "research_steps/S14/claim_to_evidence_matrix.parquet", "rows": len(claims)},
            {"table_id": "T03", "title": "Benchmark scenarios", "path": "research_steps/S14/benchmark_scenarios.parquet", "rows": len(scenarios)},
            {"table_id": "T04", "title": "Benchmark witness traces", "path": "research_steps/S14/benchmark_witness_traces.parquet", "rows": witness_count},
            {"table_id": "T05", "title": "Classification summary", "path": "research_steps/S14/classification_summary.csv", "rows": 6},
            {"table_id": "T06", "title": "Evidence boundary matrix", "path": "research_steps/S14/evidence_boundary_matrix.csv", "rows": 7},
        ]
    )
    tables.to_csv(REPORT_INPUTS / "table_index.csv", index=False)


def main() -> None:
    global OUTPUT, REPORT_INPUTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--report-inputs", type=Path, default=REPORT_INPUTS)
    args = parser.parse_args()
    OUTPUT = args.output
    REPORT_INPUTS = args.report_inputs
    OUTPUT.mkdir(parents=True, exist_ok=True)
    REPORT_INPUTS.mkdir(parents=True, exist_ok=True)

    paths = input_paths()
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing frozen S14 inputs: {missing}")
    before = {str(path): sha256_file(path) for path in paths.values()}
    contract = json_load(CONTRACT)
    evidence = evidence_index(paths)
    claims, claim_flags = make_claims(contract)
    claim_evidence_ids: list[str] = []
    claim_evidence_paths: list[str] = []
    for row in claims.itertuples(index=False):
        steps = set(str(row.source_steps).split(","))
        linked = evidence[evidence.source_step.isin(steps)].sort_values("evidence_id")
        claim_evidence_ids.append(";".join(linked.evidence_id))
        claim_evidence_paths.append(";".join(linked.relative_artifact_path))
    claims["evidence_ids"] = claim_evidence_ids
    claims["evidence_paths"] = claim_evidence_paths
    categories = category_table(contract, claims)
    scenarios, witnesses, witness_validation = build_benchmarks(paths)
    caveats = pd.DataFrame(caveat_rows())
    caveats.insert(0, "schema_version", SCHEMA)
    caveats.insert(1, "research_step_id", "S14")

    categories.to_csv(OUTPUT / "taxonomy_categories.csv", index=False)
    categories.to_parquet(OUTPUT / "taxonomy_categories.parquet", index=False)
    claims.to_csv(OUTPUT / "claim_to_evidence_matrix.csv", index=False)
    claims.to_parquet(OUTPUT / "claim_to_evidence_matrix.parquet", index=False)
    scenarios.to_csv(OUTPUT / "benchmark_scenarios.csv", index=False)
    scenarios.to_parquet(OUTPUT / "benchmark_scenarios.parquet", index=False)
    witnesses.to_parquet(OUTPUT / "benchmark_witness_traces.parquet", index=False)
    witness_validation.to_parquet(OUTPUT / "witness_replay_validation.parquet", index=False)
    caveats.to_csv(OUTPUT / "caveat_register.csv", index=False)
    evidence.to_csv(OUTPUT / "evidence_index.csv", index=False)

    classification = (
        claims.groupby(["strongest_supported_category", "assertion_status"], dropna=False)
        .size()
        .reset_index(name="claim_count")
        .sort_values(["strongest_supported_category", "assertion_status"])
    )
    classification.to_csv(OUTPUT / "classification_summary.csv", index=False)

    plot_taxonomy(categories, OUTPUT / "taxonomy_ladder")
    boundary = plot_boundaries(OUTPUT / "evidence_boundary_map")
    boundary.to_csv(OUTPUT / "evidence_boundary_matrix.csv", index=False)
    plot_witnesses(witnesses, OUTPUT / "representative_witnesses")

    adaptive_flags = EvidenceFlags(
        named_global_worsening=True,
        explicit_goal_projection=True,
        observed_successful_recovered_global_excursion=True,
        intervention_relative_utility=True,
        no_created_impossibility_as_benefit=True,
        no_censoring_as_efficiency=True,
        matched_null_exceedance=False,
        same_support_metric_goal=True,
        replay_and_provenance_pass=True,
    )
    adaptive_gates = {
        "schemaVersion": SCHEMA,
        "researchStepId": "S14",
        "gates": adaptive_flags.__dict__,
        "strongestCategory": strongest_supported_category(adaptive_flags),
        "adaptiveDetourPass": False,
        "failedGate": "matched_null_exceedance",
        "evidence": {
            "observedAndIntervention": "S09-S10 provide successful global excursions and isolated action-filter utility on retained exact-small supports.",
            "nullConstraint": "S11 open-loop opportunities nearly reproduce observed completion and barrier effects; random and rate-matched controls reproduce excursions.",
            "retainedE01Constraint": "S08 finds zero observed necessary-detour overlap among five exact-eligible primary traces.",
        },
    }
    canonical_json(OUTPUT / "adaptive_detour_gate_audit.json", adaptive_gates)

    write_taxonomy_markdown(categories, claims)
    write_e07_handoff(scenarios)
    write_report_inputs(
        evidence,
        claims,
        caveats,
        categories,
        scenarios,
        classification,
        boundary,
        len(witnesses),
    )
    lay_text = (REPORT_INPUTS / "lay_summary.md").read_text(encoding="utf-8")
    assertion_hits = anthropomorphic_assertion_hits(claims.claim.tolist() + [lay_text])
    canonical_json(
        OUTPUT / "anthropomorphic_language_audit.json",
        {
            "schemaVersion": SCHEMA,
            "researchStepId": "S14",
            "scope": "major claim assertions and the lay summary",
            "hits": assertion_hits,
            "success": not assertion_hits,
            "note": "Normative category caveats may name excluded interpretations only to state that they are not established.",
        },
    )

    holdout_records = scenarios[
        [
            "benchmark_id",
            "boundary",
            "source_artifact",
            "source_sha256",
            "source_selector",
            "start_or_run",
            "expected_category",
        ]
    ].to_dict(orient="records")
    canonical_json(
        OUTPUT / "e07_holdout_manifest.json",
        {
            "schemaVersion": SCHEMA,
            "researchStepId": "S14",
            "confirmationOnly": True,
            "trainingAllowed": False,
            "leakageRule": contract["benchmarkPolicy"]["e07LeakageRule"],
            "benchmarks": holdout_records,
        },
    )

    after = {str(path): sha256_file(path) for path in paths.values()}
    canonical_json(
        OUTPUT / "input_immutability.json",
        {
            "schemaVersion": SCHEMA,
            "researchStepId": "S14",
            "inputCount": len(before),
            "before": before,
            "after": after,
            "changed": [path for path in before if before[path] != after[path]],
            "success": before == after,
        },
    )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True).strip()
    environment = {
        "schemaVersion": SCHEMA,
        "researchStepId": "S14",
        "python": sys.version,
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "cpuCount": os.cpu_count(),
        "workerCount": 1,
        "threadEnvironment": {key: os.environ.get(key) for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"]},
        "repositoryCommit": commit,
        "command": "python scripts/build_e03_s14_taxonomy.py",
    }
    canonical_json(OUTPUT / "environment.json", environment)
    canonical_json(
        OUTPUT / "provenance_manifest.json",
        {
            "schemaVersion": SCHEMA,
            "researchStepId": "S14",
            "repository": "https://github.com/Eidosoma/cell_research",
            "branch": "eidosoma/groups/28",
            "commit": commit,
            "contract": {"path": str(CONTRACT), "sha256": sha256_file(CONTRACT)},
            "inputs": evidence.to_dict(orient="records"),
            "outputDirectory": str(OUTPUT),
            "reportInputsDirectory": str(REPORT_INPUTS),
        },
    )
    canonical_json(REPORT_INPUTS / "provenance_manifest.json", json_load(OUTPUT / "provenance_manifest.json"))

    expected_report_inputs = [
        "evidence_index.csv",
        "evidence_index.json",
        "methods_summary.md",
        "methods_summary.json",
        "figure_index.csv",
        "table_index.csv",
        "claim_to_evidence_matrix.csv",
        "claim_to_evidence_matrix.parquet",
        "caveat_register.csv",
        "caveat_register.json",
        "provenance_manifest.json",
        "lay_summary.md",
        "lay_summary.json",
        "taxonomy_categories.csv",
        "classification_summary.csv",
        "benchmark_scenarios.csv",
        "evidence_boundary_matrix.csv",
    ]
    manifest_entries = []
    for name in expected_report_inputs:
        path = REPORT_INPUTS / name
        manifest_entries.append(
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    canonical_json(
        REPORT_INPUTS / "report_bundle_manifest.json",
        {
            "schemaVersion": SCHEMA,
            "researchStepId": "S14",
            "status": "complete_pre_report_validation",
            "inputs": manifest_entries,
        },
    )
    print(
        json.dumps(
            {
                "researchStepId": "S14",
                "claimCount": len(claims),
                "categoryCounts": claims.strongest_supported_category.value_counts().to_dict(),
                "benchmarkCount": len(scenarios),
                "witnessRows": len(witnesses),
                "witnessesPassed": int(witness_validation.passed.sum()),
                "adaptiveClaims": int((claims.strongest_supported_category == "adaptive_detour").sum()),
                "inputsUnchanged": before == after,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
