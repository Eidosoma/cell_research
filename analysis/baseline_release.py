"""S14 claim reconciliation, release packaging, and reproducibility audits.

The module only synthesizes frozen S01--S13 evidence.  It does not rerun a
scientific population, infer missing publication bytes, or treat public HEAD as
the publication snapshot.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


REPOSITORY = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/artifacts")
WORKSPACE = Path("/workspace")
CONTRACT = REPOSITORY / "analysis" / "s14_release_contract.json"
REGISTRY = ARTIFACT_ROOT / "research_steps" / "S01" / "claim_registry.parquet"
CLASSIFICATIONS = {
    "reproduced",
    "approximately_reproduced",
    "implementation_sensitive",
    "not_reproduced",
    "not_testable",
}
S14_SCHEMA = "e01.s14.claim_evidence_matrix.v1"


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
    ).encode("utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value)!r}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, *, label: str | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "label": label or path.name,
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def report_input_records(report_inputs: Path) -> list[dict[str, Any]]:
    """Inventory report inputs without creating an impossible self-hash."""
    return [
        file_record(path)
        for path in sorted(report_inputs.iterdir())
        if path.is_file() and path.name != "report_bundle_manifest.json"
    ]


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(table, path, compression="zstd", version="2.6")


def git_output(*args: str, cwd: Path = REPOSITORY) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=cwd, text=True, stderr=subprocess.STDOUT
    ).strip()


def _row_dict(row: pd.Series | Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in dict(row).items():
        if pd.isna(value) if not isinstance(value, (list, dict, tuple)) else False:
            result[str(key)] = None
        elif isinstance(value, np.generic):
            result[str(key)] = value.item()
        else:
            result[str(key)] = value
    return result


def _raw_mapping(value: str) -> str:
    direct = {
        "supportive": "reproduced",
        "supported_clean_room": "reproduced",
        "reproduced_to_display_precision": "reproduced",
        "reproduced_within_frozen_margin": "reproduced",
        "hand_fixture_reproduced_metric_zero": "reproduced",
        "qualitatively_reproduced_in_envelope": "reproduced",
        "rank_reproduced": "reproduced",
        "approximately_reproduced": "approximately_reproduced",
        "approximately_reproduced_within_frozen_margin": "approximately_reproduced",
        "peak:approximately_reproduced;progress:reproduced_to_display_precision": "approximately_reproduced",
        "implemented_with_boundary_decision": "implementation_sensitive",
        "implementation_sensitive_formula_conflict": "implementation_sensitive",
        "constrained_clean_room_qualitative": "implementation_sensitive",
        "metric_dependent_clean_room_qualitative": "implementation_sensitive",
        "peak:not_reproduced;progress:approximately_reproduced": "implementation_sensitive",
        "contradicted_clean_room": "not_reproduced",
        "contradictory": "not_reproduced",
        "not_reproduced": "not_reproduced",
        "rank_not_reproduced": "not_reproduced",
        "endpoint:not_reproduced;shape:not_reproduced": "not_reproduced",
        "not_recoverable": "not_testable",
        "not_testable_backend_unavailable": "not_testable",
    }
    if value not in direct:
        raise KeyError(f"unmapped raw classification: {value}")
    return direct[value]


def _historical_status(figure: int, claim_id: str, claim: Mapping[str, Any]) -> str:
    if figure == 3:
        return "partial_frozen_HEAD_cell_view_endpoint_only; traditional_generator_absent"
    if figure == 4:
        return "not_testable_missing_traditional_generator_and_complete_cost_ledger"
    if figure == 5:
        if claim_id == "F05-A-PASSIVE-DEFINITION":
            return "static_frozen_HEAD_passive_like_FREEZE_behavior_available"
        if claim_id == "F05-A-STUCK-DEFINITION":
            return "not_testable_distinct_stuck_implementation_absent"
        if "PASSIVE" in claim_id and "CELL_VIEW" in claim_id:
            return "frozen_HEAD_passive_cell_view_endpoint_available_at_exact_legacy_placement"
        return "not_testable_required_traditional_or_stuck_backend_absent"
    if figure == 6 and claim_id == "F06-D-DG-DEFINITION":
        return "static_frozen_HEAD_metric_function_available; publication_formula_identity_unresolved"
    if figure in {6, 7}:
        return "not_testable_historical_trajectories_or_required_generators_unavailable"
    return "not_testable_historical_mixed_arrays_streams_or_validated_endpoint_unavailable"


def _evidence_path(figure: int, claim_id: str) -> Path:
    if figure == 3:
        return ARTIFACT_ROOT / "research_steps" / "S09" / "figure3_claim_classifications.csv"
    if figure == 4:
        return ARTIFACT_ROOT / "research_steps" / "S10" / "figure4_claim_classifications.csv"
    if figure == 5:
        if claim_id.startswith("F05-A-"):
            return ARTIFACT_ROOT / "research_steps" / "S03" / "transition_contract.json"
        if claim_id.endswith("-RANK"):
            return ARTIFACT_ROOT / "research_steps" / "S11" / "figure5_rank_classifications.csv"
        return ARTIFACT_ROOT / "research_steps" / "S11" / "reported_mean_comparison.csv"
    if figure in {6, 7}:
        return ARTIFACT_ROOT / "research_steps" / "S12" / "figure6_7_claim_classifications.csv"
    if figure == 8:
        return ARTIFACT_ROOT / "research_steps" / "S13" / "figure8_claim_classifications.csv"
    return ARTIFACT_ROOT / "research_steps" / "S13" / "figure9_10_claim_classifications.csv"


def _step_for_figure(figure: int) -> str:
    return {3: "S09", 4: "S10", 5: "S11", 6: "S12", 7: "S12", 8: "S13", 9: "S13", 10: "S13"}[figure]


def _s13_censoring() -> dict[str, dict[str, Any]]:
    source = pd.read_parquet(ARTIFACT_ROOT / "research_steps" / "S13" / "chimeric_results.parquet")
    source = source[
        source.condition_family.isin(["opposite_unique", "opposite_repeated"])
        & source.assignment_profile.eq("balanced_exact")
    ]
    by_condition: dict[str, dict[str, Any]] = {}
    for condition_id, group in source.groupby("condition_id"):
        by_condition[str(condition_id)] = {
            "runs": len(group),
            "eventBudget": int(group.stop_reason.eq("event_budget").sum()),
            "quiescent": int(group.stop_reason.eq("quiescent").sum()),
            "budget": int(group.event_budget.iloc[0]),
        }
    return by_condition


def build_claim_matrix() -> pd.DataFrame:
    contract = json.loads(CONTRACT.read_text())
    registry = pd.read_parquet(REGISTRY).sort_values("claim_id").reset_index(drop=True)
    if len(registry) != 118 or registry.claim_id.nunique() != 118:
        raise AssertionError("S01 registry is not the frozen 118-claim input")

    f3 = pd.read_csv(ARTIFACT_ROOT / "research_steps" / "S09" / "figure3_claim_classifications.csv").set_index("claim_id")
    f4 = pd.read_csv(ARTIFACT_ROOT / "research_steps" / "S10" / "figure4_claim_classifications.csv").set_index("claim_id")
    f5_means = pd.read_csv(ARTIFACT_ROOT / "research_steps" / "S11" / "reported_mean_comparison.csv")
    f5_ranks = pd.read_csv(ARTIFACT_ROOT / "research_steps" / "S11" / "figure5_rank_classifications.csv")
    f5_contrasts = pd.read_csv(ARTIFACT_ROOT / "research_steps" / "S11" / "primary_architecture_contrasts.csv")
    f67 = pd.read_csv(ARTIFACT_ROOT / "research_steps" / "S12" / "figure6_7_claim_classifications.csv").set_index("claim_id")
    f8 = pd.read_csv(ARTIFACT_ROOT / "research_steps" / "S13" / "figure8_claim_classifications.csv").set_index("claim_id")
    f910 = pd.read_csv(ARTIFACT_ROOT / "research_steps" / "S13" / "figure9_10_claim_classifications.csv").set_index("claim_id")
    censoring = _s13_censoring()

    rows: list[dict[str, Any]] = []
    for registry_row in registry.to_dict(orient="records"):
        claim = _row_dict(registry_row)
        claim_id = str(claim["claim_id"])
        figure = int(claim["figure"])
        raw = ""
        actual: Any = None
        rationale = ""
        primary_layer = "clean_room_reference"
        analysis_split = "not_applicable"
        estimate = ci_low = ci_high = None
        architecture_contrast = "not_applicable"
        legacy_result = "not_applicable"
        historical_result = _historical_status(figure, claim_id, claim)

        if figure == 3:
            item = _row_dict(f3.loc[claim_id])
            raw = str(item["qualitative_classification"])
            actual = item
            rationale = str(item["decision_rule"])
            estimate, ci_low, ci_high = item["estimate"], item["adjusted_ci_low"], item["adjusted_ci_high"]
            analysis_split = "confirmatory_holdout"
        elif figure == 4:
            item = _row_dict(f4.loc[claim_id])
            raw = str(item["replication_classification"])
            actual = item
            rationale = (
                f"R ratio={item['observed_ratio_of_means']}; adjusted interval="
                f"[{item['ratio_ci98_3333_low']},{item['ratio_ci98_3333_high']}]; "
                f"direction reproduced={item['direction_reproduced']}."
            )
            estimate, ci_low, ci_high = item["observed_ratio_of_means"], item["ratio_ci98_3333_low"], item["ratio_ci98_3333_high"]
            analysis_split = "confirmatory_holdout"
        elif figure == 5:
            if claim_id == "F05-A-PASSIVE-DEFINITION":
                raw = "supportive"
                actual = {"paper": "cannot initiate; may be displaced", "C": "FREEZE passive-like", "R": "cannot initiate; may be displaced"}
                rationale = "Paper, frozen-HEAD passive-like behavior, and R agree on the operational passive rule."
                primary_layer = "paper_plus_static_C_plus_clean_room_R"
            elif claim_id == "F05-A-STUCK-DEFINITION":
                raw = "supportive"
                actual = {"paper": "cannot initiate or move", "C": None, "R": "cannot initiate or move"}
                rationale = "Paper and R agree; frozen public HEAD has no distinct stuck implementation."
                primary_layer = "paper_plus_clean_room_R"
            elif claim_id.endswith("-RANK"):
                primary = f5_ranks[
                    f5_ranks.claim_id.eq(claim_id)
                    & f5_ranks.backend_profile.eq("R-clean-room-reference-E01-v1")
                    & f5_ranks.placement_profile.eq("reference_without_replacement")
                    & f5_ranks.split.eq("confirmatory_holdout")
                ]
                if len(primary) != 1:
                    raise AssertionError(f"missing primary Figure 5 rank row for {claim_id}")
                item = _row_dict(primary.iloc[0])
                raw = str(item["classification"])
                actual = item
                rationale = f"Corrected confirmatory R rank: {item['classification']} under {item['frozen_rule']}."
                analysis_split = "confirmatory_holdout"
                legacy_rows = f5_ranks[
                    f5_ranks.claim_id.eq(claim_id)
                    & f5_ranks.placement_profile.eq("legacy_with_replacement")
                ]
                legacy_result = canonical_json_bytes([_row_dict(row) for _, row in legacy_rows.iterrows()]).decode()
                if claim_id == "F05-B-PASSIVE-RANK" and any(legacy_rows.backend_profile.eq("C-frozen-public-commit")):
                    c_row = legacy_rows[legacy_rows.backend_profile.eq("C-frozen-public-commit")].iloc[0]
                    if str(c_row.classification) != raw:
                        raw = "implementation_sensitive_formula_conflict"
                        rationale += " Frozen-HEAD C gives the opposite rank, so the reconciled result is backend-sensitive."
            else:
                primary = f5_means[
                    f5_means.claim_id.eq(claim_id)
                    & f5_means.backend_profile.eq("R-clean-room-reference-E01-v1")
                    & f5_means.placement_profile.eq("reference_without_replacement")
                ]
                if len(primary) != 1:
                    raise AssertionError(f"missing primary Figure 5 mean row for {claim_id}")
                item = _row_dict(primary.iloc[0])
                raw = str(item["classification"])
                actual = item
                rationale = (
                    f"Corrected paper-scale R mean={item['simulated_mean']} versus paper={item['paper_reported_mean']} "
                    f"under the frozen ±{item['equivalence_margin_errors']}-error margin."
                )
                estimate, ci_low, ci_high = item["simulated_mean"], item["ci95_low"], item["ci95_high"]
                analysis_split = "paper_scale"
                legacy = f5_means[
                    f5_means.claim_id.eq(claim_id)
                    & f5_means.backend_profile.eq("R-clean-room-reference-E01-v1")
                    & f5_means.placement_profile.eq("legacy_with_replacement")
                ]
                legacy_result = canonical_json_bytes(_row_dict(legacy.iloc[0])).decode() if len(legacy) == 1 else "unavailable"
                c = f5_means[f5_means.claim_id.eq(claim_id) & f5_means.backend_profile.eq("C-frozen-public-commit")]
                if len(c) == 1:
                    historical_result += "; classification=" + str(c.iloc[0].classification)
                contrast = f5_contrasts[
                    f5_contrasts.policy.eq(item["policy"])
                    & f5_contrasts.fault_mode.eq(item["fault_mode"])
                    & f5_contrasts.requested_fault_count.eq(item["requested_fault_count"])
                ]
                if len(contrast) == 1:
                    architecture_contrast = str(contrast.iloc[0].claim_classification)
        elif figure in {6, 7}:
            if claim_id == "F06-A-CONCEPTUAL-CONTEXT":
                raw = "not_recoverable"
                actual = None
                rationale = "Registered non-empirical artwork/context; no simulator result exists to reproduce."
                primary_layer = "paper_non_empirical"
            else:
                item = _row_dict(f67.loc[claim_id])
                raw = str(item["endpoint_classification"])
                actual = item
                rationale = str(item["caveat"])
                estimate, ci_low, ci_high = item["estimate"], item["adjusted_ci_low"], item["adjusted_ci_high"]
                analysis_split = str(item["analysis_split"])
        else:
            source = f8 if figure == 8 else f910
            item = _row_dict(source.loc[claim_id])
            raw = str(item["classification"])
            actual = item.get("actual_json")
            rationale = str(item["rationale"])
            analysis_split = "paper_scale"

        final = _raw_mapping(raw)
        if claim_id in {"F09-ABC-AGGREGATION-RISE", "F09-ABC-WINNER-ORDER"}:
            final = "implementation_sensitive"
            rationale += " Composite stability/inference is constrained by 108/300 exact unique event-budget terminals."
        if claim_id == "F09-B-FINAL-SORTEDNESS":
            final = "implementation_sensitive"
            rationale += " Endpoint is approximate, but 10/100 exact runs are censored rather than stable."

        censor_detail: dict[str, Any] | None = None
        censor_map = {
            "F09-A-FINAL-SORTEDNESS": "C-UNQ-CHIM-BUB-SEL-EXACT-OPP",
            "F09-B-FINAL-SORTEDNESS": "C-UNQ-CHIM-BUB-INS-EXACT-OPP",
            "F09-C-FINAL-SORTEDNESS": "C-UNQ-CHIM-INS-SEL-EXACT-OPP",
            "F10-A-REPEATED-OPPOSITE": "C-REP-CHIM-BUB-SEL-EXACT-OPP",
            "F10-B-REPEATED-OPPOSITE": "C-REP-CHIM-BUB-INS-EXACT-OPP",
            "F10-C-REPEATED-OPPOSITE": "C-REP-CHIM-INS-SEL-EXACT-OPP",
        }
        if claim_id in censor_map:
            censor_detail = censoring[censor_map[claim_id]]
        elif claim_id in {"F09-ABC-AGGREGATION-RISE", "F09-ABC-WINNER-ORDER"}:
            selected = [value for key, value in censoring.items() if key.startswith("C-UNQ")]
            censor_detail = {
                "runs": sum(value["runs"] for value in selected),
                "eventBudget": sum(value["eventBudget"] for value in selected),
                "quiescent": sum(value["quiescent"] for value in selected),
                "budget": 1_000_000,
            }

        strict_sensitivity = "not_applicable"
        if figure == 10:
            strict_sensitivity = "strict_paper_and_non_strict_reference_metrics_both_retained; qualitative_result_metric_dependent"
        elif figure == 8 and "D-" in claim_id:
            strict_sensitivity = "duplicate_values_present; claim_metric_is_aggregation_not_sortedness"

        placement_sensitivity = "not_applicable"
        if figure == 5:
            placement_sensitivity = "reference_without_replacement_primary; legacy_with_replacement_retained_separately"
        elif figure == 7:
            placement_sensitivity = "legacy_with_replacement_historical_primary; corrected_without_replacement_sensitivity_retained"
        elif figure in {8, 9, 10}:
            placement_sensitivity = "exact_composition_primary; independently_randomized_composition_retained_separately"

        evidence = _evidence_path(figure, claim_id)
        outcome = (
            "supportive"
            if final in {"reproduced", "approximately_reproduced"}
            else "null"
            if final == "not_testable"
            else "constraining/contradictory"
        )
        contradiction = bool(
            claim["evidence_status"] == "contradictory"
            or final in {"implementation_sensitive", "not_reproduced"}
            or architecture_contrast == "contradictory"
        )
        rows.append(
            {
                "reconciliation_schema_version": S14_SCHEMA,
                "reconciliation_research_step_id": "S14",
                **claim,
                "evidence_step_id": _step_for_figure(figure),
                "primary_evidence_layer": primary_layer,
                "primary_evidence_path": str(evidence),
                "primary_evidence_sha256": sha256_file(evidence),
                "publication_snapshot_identified": False,
                "historical_backend_status": historical_result,
                "reference_classification_raw": raw,
                "legacy_or_sensitivity_result_json": legacy_result,
                "architecture_contrast_classification": architecture_contrast,
                "final_replication_classification": final,
                "claim_outcome_classification": outcome,
                "analysis_split": analysis_split,
                "estimate": estimate,
                "adjusted_ci_low": ci_low,
                "adjusted_ci_high": ci_high,
                "actual_result_json": canonical_json_bytes(actual).decode() if actual is not None else "null",
                "strict_duplicate_sensitivity": strict_sensitivity,
                "event_budget_censoring_present": bool(censor_detail and censor_detail["eventBudget"]),
                "event_budget_censoring_json": canonical_json_bytes(censor_detail).decode() if censor_detail else "null",
                "placement_or_composition_sensitivity": placement_sensitivity,
                "contradiction_preserved": contradiction,
                "reconciliation_rationale": rationale,
                "recommended_claim_use": (
                    "usable_with_named_evidence_layer_and_caveats"
                    if final in {"reproduced", "approximately_reproduced"}
                    else "retain_as_constraint_or_unresolved_evidence; do_not_generalize"
                ),
            }
        )

    result = pd.DataFrame(rows).sort_values(["figure", "claim_id"]).reset_index(drop=True)
    if len(result) != 118 or result.claim_id.nunique() != 118:
        raise AssertionError("S14 matrix does not cover all 118 claims")
    if set(result.final_replication_classification) - CLASSIFICATIONS:
        raise AssertionError("S14 matrix contains an invalid final classification")
    if set(result.claim_id) != set(registry.claim_id):
        raise AssertionError("S14 matrix claim IDs differ from S01")
    if set(contract["claimRegistry"]["nonEmpiricalClaimIds"]) != set(
        result[result.claim_kind.eq("non_empirical_context")].claim_id
    ):
        raise AssertionError("non-empirical claim accounting changed")
    return result


def validate_claim_matrix(frame: pd.DataFrame) -> dict[str, Any]:
    expected_by_figure = {3: 4, 4: 6, 5: 40, 6: 4, 7: 29, 8: 26, 9: 6, 10: 3}
    observed = {int(key): int(value) for key, value in frame.figure.value_counts().sort_index().items()}
    checks = {
        "rows118": len(frame) == 118,
        "uniqueClaimIds118": frame.claim_id.nunique() == 118,
        "exactRegistryIdSet": set(frame.claim_id) == set(pd.read_parquet(REGISTRY).claim_id),
        "figureCoverage": observed == expected_by_figure,
        "allFinalClassified": frame.final_replication_classification.notna().all(),
        "validVocabulary": not bool(set(frame.final_replication_classification) - CLASSIFICATIONS),
        "publicationSnapshotNeverClaimed": not bool(frame.publication_snapshot_identified.any()),
        "historicalStatusComplete": frame.historical_backend_status.astype(bool).all(),
        "censoringExplicit": int(frame.event_budget_censoring_present.sum()) == 6,
        "figure10StrictSensitivityExplicit": frame[frame.figure.eq(10)].strict_duplicate_sensitivity.str.contains("both_retained").all(),
        "nonEmpiricalExactlyOne": int(frame.claim_kind.eq("non_empirical_context").sum()) == 1,
        "contradictionsPreserved": int(frame.contradiction_preserved.sum()) > 0,
    }
    return {
        "schemaVersion": "e01.s14.claim_validation.v1",
        "researchStepId": "S14",
        "success": all(checks.values()),
        "checks": checks,
        "figureCounts": observed,
        "classificationCounts": {
            str(key): int(value)
            for key, value in frame.final_replication_classification.value_counts().sort_index().items()
        },
        "outcomeCounts": {
            str(key): int(value)
            for key, value in frame.claim_outcome_classification.value_counts().sort_index().items()
        },
    }


def run_reference_smoke() -> dict[str, Any]:
    from reference_simulator import audit_reference_result, create_scenario, exact_replay, run_scenario

    samples = []
    for index, policy in enumerate(("Bubble", "Insertion", "Selection")):
        scenario = create_scenario(
            [4, 1, 3, 2, 0, 5],
            policy=policy,
            seed=0xE011400 + index,
            max_activations=20_000,
            generation_key=f"E01/S14/fresh-smoke/{policy.lower()}",
            permute=False,
        )
        result = run_scenario(scenario, trace_mode="full")
        replay = exact_replay(result)
        audit = audit_reference_result(result)
        samples.append(
            {
                "policy": policy,
                "scenarioId": scenario.scenario_id,
                "stopReason": result.summary["stopReason"],
                "acceptedSwaps": result.summary["ledger"]["acceptedSwaps"],
                "finalStateHash": result.final_state_hash,
                "byteExactReplay": replay.to_json_bytes() == result.to_json_bytes(),
                "invariantAuditPassed": bool(audit["success"]),
            }
        )
    return {
        "schemaVersion": "e01.s14.reference_smoke.v1",
        "researchStepId": "S14",
        "success": all(
            item["stopReason"] == "complete"
            and item["byteExactReplay"]
            and item["invariantAuditPassed"]
            for item in samples
        ),
        "samples": samples,
    }


def _pixel_hash(path: Path) -> dict[str, Any]:
    from PIL import Image

    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        digest = hashlib.sha256()
        digest.update(f"{rgba.width}x{rgba.height}|RGBA|".encode())
        digest.update(rgba.tobytes())
        return {"width": rgba.width, "height": rgba.height, "pixelSha256": digest.hexdigest()}


def regenerate_figures(scratch: Path) -> dict[str, Any]:
    """Regenerate Figures 3--10 and the overlay from collectible artifacts only."""
    from analysis import chimeric_replication as s13
    from analysis import delayed_gratification as s12
    from analysis import efficiency_costs as s10
    from analysis import frozen_cell_results as s11
    from analysis import no_fault_sorting as s09

    scratch.mkdir(parents=True, exist_ok=True)

    trajectory_rows = pd.read_parquet(
        ARTIFACT_ROOT / "research_steps" / "S09" / "paper_and_selected_swap_trajectories.parquet"
    )
    trajectory_rows = trajectory_rows[trajectory_rows.split.eq("paper_scale")]
    lightweight_results: list[dict[str, Any]] = []
    group_columns = ["backend_profile", "scenario_id", "condition_id", "split", "architecture", "policy"]
    for keys, group in trajectory_rows.groupby(group_columns, sort=False):
        record = dict(zip(group_columns, keys))
        record["raw_curve"] = group.sort_values("successful_swap_index")[["successful_swap_index", "sortedness_percent"]].to_numpy().tolist()
        lightweight_results.append(record)
    envelopes = pd.read_parquet(
        ARTIFACT_ROOT / "research_steps" / "S09" / "trajectory_envelopes.parquet"
    ).to_dict(orient="records")
    s09._plot_figures(lightweight_results, envelopes, scratch)

    no_fault = pd.read_parquet(ARTIFACT_ROOT / "research_steps" / "S09" / "no_fault_runs.parquet")
    ledger = s10.build_cost_ledger(no_fault)
    s10.figure4(ledger, scratch / "figure4_reconstruction.png", scratch / "figure4_reconstruction.svg")

    frozen = pd.read_parquet(ARTIFACT_ROOT / "research_steps" / "S11" / "frozen_cell_results.parquet")
    s11.draw_figure5(frozen, scratch)

    original_dg = pd.read_parquet(ARTIFACT_ROOT / "research_steps" / "S12" / "original_dg_results.parquet")
    summary7 = s12.figure7_summary(original_dg)
    s12.draw_figure7(summary7, scratch)
    envelope6 = pd.read_parquet(ARTIFACT_ROOT / "research_steps" / "S12" / "figure6_ambiguity_envelope.parquet")
    _, _, fixture6, augmented6 = s12.figure6_analysis(envelope6)
    s12.draw_figure6(augmented6, fixture6, scratch)

    summary13 = pd.read_csv(ARTIFACT_ROOT / "research_steps" / "S13" / "condition_summary.csv")
    trajectories13 = pd.read_parquet(ARTIFACT_ROOT / "research_steps" / "S13" / "chimeric_trajectories.parquet")
    s13._plot_figure8(summary13, trajectories13, scratch)
    s13._plot_opposing(trajectories13, scratch, repeated=False)
    s13._plot_opposing(trajectories13, scratch, repeated=True)

    source_map = {
        "figure3_reconstruction": ARTIFACT_ROOT / "research_steps" / "S09",
        "historical_reference_overlay": ARTIFACT_ROOT / "research_steps" / "S09",
        "figure4_reconstruction": ARTIFACT_ROOT / "research_steps" / "S10",
        "figure5_reconstruction": ARTIFACT_ROOT / "research_steps" / "S11",
        "figure6_reconstruction": ARTIFACT_ROOT / "research_steps" / "S12",
        "figure7_reconstruction": ARTIFACT_ROOT / "research_steps" / "S12",
        "figure8_reconstruction": ARTIFACT_ROOT / "research_steps" / "S13",
        "figure9_reconstruction": ARTIFACT_ROOT / "research_steps" / "S13",
        "figure10_reconstruction": ARTIFACT_ROOT / "research_steps" / "S13",
    }
    rows = []
    for name, source_dir in source_map.items():
        original_png = source_dir / f"{name}.png"
        regenerated_png = scratch / f"{name}.png"
        original_svg = source_dir / f"{name}.svg"
        regenerated_svg = scratch / f"{name}.svg"
        original_pixels = _pixel_hash(original_png)
        regenerated_pixels = _pixel_hash(regenerated_png)
        rows.append(
            {
                "figure": name,
                "originalPng": file_record(original_png),
                "regeneratedPng": file_record(regenerated_png),
                "originalPixels": original_pixels,
                "regeneratedPixels": regenerated_pixels,
                "pixelExact": original_pixels == regenerated_pixels,
                "svgRegenerated": regenerated_svg.is_file() and regenerated_svg.stat().st_size > 0,
                "originalSvg": file_record(original_svg),
                "regeneratedSvg": file_record(regenerated_svg),
                "svgByteExactExpected": False,
                "svgByteExactReason": "Matplotlib embeds generation timestamps and process-local element IDs; PNG pixel identity is the frozen visual audit.",
            }
        )
    return {
        "schemaVersion": "e01.s14.figure_regeneration_audit.v1",
        "researchStepId": "S14",
        "success": all(row["pixelExact"] and row["svgRegenerated"] for row in rows),
        "figures": rows,
        "regeneratedDirectory": str(scratch),
        "collectibleIntermediatesWritten": False,
    }


RELEASE_SOURCE_PATHS = (
    "analysis/baseline_release.py",
    "analysis/s14_release_contract.json",
    "analysis/no_fault_sorting.py",
    "analysis/efficiency_costs.py",
    "analysis/frozen_cell_results.py",
    "analysis/delayed_gratification.py",
    "analysis/chimeric_replication.py",
    "analysis/s09_confirmatory_preregistration.json",
    "analysis/s10_efficiency_preregistration.json",
    "analysis/s11_frozen_cell_preregistration.json",
    "analysis/s12_delayed_gratification_preregistration.json",
    "analysis/s13_chimera_preregistration.json",
    "reference_simulator",
    "scenario_bank",
    "shared_events",
    "scripts/build_scenario_bank.py",
    "scripts/replicate_no_fault.py",
    "scripts/replicate_efficiency.py",
    "scripts/run_frozen_cell_results.py",
    "scripts/replicate_delayed_gratification.py",
    "scripts/replicate_chimeras.py",
    "scripts/reconcile_release_baseline.py",
    "scripts/validate_invariants.py",
    "scripts/validate_reference_backend.py",
    "scripts/validate_shared_events.py",
    "tests/test_baseline_release.py",
    "tests/test_reference_simulator.py",
    "tests/test_scenario_bank.py",
    "tests/test_shared_events.py",
    "tests/test_invariants.py",
    "tests/test_no_fault_sorting.py",
    "tests/test_efficiency_costs.py",
    "tests/test_frozen_cell_results.py",
    "tests/test_delayed_gratification.py",
    "tests/test_chimeric_replication.py"
)


def source_hash_manifest(commit: str) -> dict[str, Any]:
    expanded: set[str] = set()
    tree = git_output("ls-tree", "-r", "--name-only", commit).splitlines()
    for item in RELEASE_SOURCE_PATHS:
        if item in tree:
            expanded.add(item)
        else:
            prefix = item.rstrip("/") + "/"
            expanded.update(path for path in tree if path.startswith(prefix))
    rows = []
    for path in sorted(expanded):
        payload = subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=REPOSITORY)
        mode_type_blob = git_output("ls-tree", commit, "--", path).split()
        rows.append(
            {
                "path": path,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "gitBlob": mode_type_blob[2],
                "gitMode": mode_type_blob[0],
            }
        )
    return {
        "schemaVersion": "e01.s14.source_hash_manifest.v1",
        "researchStepId": "S14",
        "repository": "https://github.com/Eidosoma/cell_research.git",
        "branch": "eidosoma/groups/28",
        "commit": commit,
        "tree": git_output("rev-parse", f"{commit}^{{tree}}"),
        "sourceArchiveCreated": False,
        "sourceArchivePathFromPlan": "/artifacts/release/reference-simulator-source.tar.zst",
        "sourceArchiveReason": "AGENTS.md requires repository source to remain in Git; no repository license file or frozen-public-source redistribution grant was found.",
        "historicalSourceIncluded": False,
        "files": rows,
    }


def _link_or_replace(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _prior_artifact_inventory() -> list[dict[str, Any]]:
    rows = []
    for step in range(1, 14):
        directory = ARTIFACT_ROOT / "research_steps" / f"S{step:02d}"
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                rows.append({"researchStepId": f"S{step:02d}", **file_record(path)})
    return rows


def input_provenance() -> dict[str, Any]:
    context_paths = [
        WORKSPACE / "AGENTS.md",
        WORKSPACE / "FULL_PLAN.md",
        WORKSPACE / "RESEARCH_PLAN.md",
        WORKSPACE / "DATASETS.md",
        WORKSPACE / "DATASET_CATALOG.json",
        WORKSPACE / "DATASET_AVAILABILITY.json",
        WORKSPACE / "CAPABILITIES.md",
        WORKSPACE / "CAPABILITY_AVAILABILITY.json",
        WORKSPACE / "input-attachments" / "MANIFEST.json",
        WORKSPACE / "input-attachments" / "21c2278b-9950-4e39-a2c8-df578a2508ec" / "_metadata" / "ATTACHMENT.md",
        WORKSPACE / "input-attachments" / "21c2278b-9950-4e39-a2c8-df578a2508ec" / "pdf-markdown.md",
    ]
    attachment_figures = sorted(
        (WORKSPACE / "input-attachments" / "21c2278b-9950-4e39-a2c8-df578a2508ec" / "figures").glob("figure-*.png")
    )
    prior = _prior_artifact_inventory()
    return {
        "schemaVersion": "e01.s14.input_provenance.v1",
        "researchStepId": "S14",
        "context": [file_record(path) for path in [*context_paths, *attachment_figures]],
        "priorArtifacts": prior,
        "priorArtifactCount": len(prior),
        "priorArtifactBytes": sum(item["bytes"] for item in prior),
        "datasets": {"required": False, "catalogEntries": 0, "availabilityStatus": "not_required"},
        "previousArtifacts": {"contextFilesPresent": False, "mountPresent": Path("/previous-artifacts").exists()},
        "historicalQuarantine": {
            "path": "/cache/e01_s02/historical-worktree",
            "includedInRelease": False,
            "lawfulManifest": file_record(ARTIFACT_ROOT / "research_steps" / "S02" / "lawful_source_manifest.json"),
        },
    }


def _caveats(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {"id": "CAV-PUBLICATION-SNAPSHOT", "severity": "critical", "scope": "all_claims", "text": "No public ref identifies the exact publication snapshot; public HEAD is a frozen comparator only."},
        {"id": "CAV-LICENSE", "severity": "critical", "scope": "source_packaging", "text": "No license file or redistribution grant was found; no source archive is emitted."},
        {"id": "CAV-RAW-DATA", "severity": "critical", "scope": "historical_replication", "text": "No historical .npy bytes exist in reachable public history; regenerated outputs are never substituted."},
        {"id": "CAV-HISTORICAL-OBSERVABILITY", "severity": "high", "scope": "cross_backend", "text": "Historical C records successful swaps only and lacks activation identities, rejected actions, most costs, and native state hashes."},
        {"id": "CAV-SCHEDULER", "severity": "high", "scope": "historical_C", "text": "Fixed-input/fixed-seed C trajectories vary with OS/GIL scheduling; wall time is descriptive only."},
        {"id": "CAV-TRADITIONAL", "severity": "high", "scope": "Figures_3_7", "text": "Historical traditional generators are absent; R traditional policies are named clean-room controls."},
        {"id": "CAV-FAULT-PLACEMENT", "severity": "high", "scope": "Figures_5_7", "text": "Exact-legacy with-replacement placement can under-realize requested faults and remains separate from corrected placement."},
        {"id": "CAV-PASSIVE-REVERSAL", "severity": "high", "scope": "Figure_5", "text": "Eight passive architecture contrasts reverse the blanket claim under the named clean-room traditional semantics; Bubble f=1 ties."},
        {"id": "CAV-DG", "severity": "high", "scope": "Figures_6_7", "text": "Journal, arXiv, and frozen-code DG definitions differ; DG is a trajectory proxy, not foresight or global distance."},
        {"id": "CAV-COMPOSITION", "severity": "high", "scope": "Figure_8", "text": "A universal 0.5 aggregation baseline is incompatible with equal three-way composition; composition-conditioned nulls are required."},
        {"id": "CAV-DUPLICATES", "severity": "high", "scope": "Figure_10", "text": "Strict paper and non-strict reference Sortedness produce different repeated-value trajectories."},
        {"id": "CAV-CENSORING", "severity": "critical", "scope": "Figures_9_10", "text": "465/1,200 opposing R runs reached the one-million-activation budget and are censored, not stable."},
        {"id": "CAV-HISTORICAL-STREAMS", "severity": "high", "scope": "all_scenarios", "text": "All historical random streams remain unavailable_not_invented; S08 streams are clean-room declarations."},
        {"id": "CAV-PROXY", "severity": "high", "scope": "interpretation", "text": "Computational sorting, aggregation, and DG are bounded model measurements and do not establish biological mechanism, cognition, or foresight."},
        {"id": "CAV-FINITE-TESTING", "severity": "medium", "scope": "reference_backend", "text": "The invariant campaign is strong finite evidence rather than a formal proof."},
    ]


def _component_records() -> dict[str, list[dict[str, Any]]]:
    figures = []
    for figure, step in ((3, 9), (4, 10), (5, 11), (6, 12), (7, 12), (8, 13), (9, 13), (10, 13)):
        for suffix in ("png", "svg"):
            figures.append(file_record(ARTIFACT_ROOT / "research_steps" / f"S{step:02d}" / f"figure{figure}_reconstruction.{suffix}"))
    return {
        "schemas": [
            file_record(ARTIFACT_ROOT / "research_steps" / "S03" / "transition_contract.json"),
            file_record(ARTIFACT_ROOT / "research_steps" / "S06" / "event_schema.json"),
            file_record(ARTIFACT_ROOT / "research_steps" / "S06" / "trace_manifest_schema.json"),
            file_record(ARTIFACT_ROOT / "research_steps" / "S08" / "scenario_bank_schema.json"),
            file_record(ARTIFACT_ROOT / "research_steps" / "S08" / "condition_schema.json"),
        ],
        "scenarios": [
            file_record(ARTIFACT_ROOT / "scenarios" / "paired_scenario_bank.parquet"),
            file_record(ARTIFACT_ROOT / "research_steps" / "S08" / "base_draw_bank.parquet"),
            file_record(ARTIFACT_ROOT / "research_steps" / "S08" / "condition_catalog.parquet"),
            file_record(ARTIFACT_ROOT / "research_steps" / "S08" / "split_manifest.json"),
            file_record(ARTIFACT_ROOT / "research_steps" / "S08" / "seed_specification.json"),
        ],
        "configurations": [file_record(REPOSITORY / "analysis" / f"s{step:02d}_{name}") for step, name in (
            (9, "confirmatory_preregistration.json"),
            (10, "efficiency_preregistration.json"),
            (11, "frozen_cell_preregistration.json"),
            (12, "delayed_gratification_preregistration.json"),
            (13, "chimera_preregistration.json"),
            (14, "release_contract.json"),
        )],
        "selectedTraces": [
            file_record(ARTIFACT_ROOT / "research_steps" / "S06" / "examples" / "reference_trace.jsonl.zst"),
            file_record(ARTIFACT_ROOT / "research_steps" / "S09" / "selected_traces.jsonl.zst"),
            file_record(ARTIFACT_ROOT / "research_steps" / "S11" / "exact_replay_samples.json"),
            file_record(ARTIFACT_ROOT / "research_steps" / "S12" / "figure6_selected_trajectories.parquet"),
            file_record(ARTIFACT_ROOT / "research_steps" / "S13" / "selected_swap_traces.parquet"),
        ],
        "figures": figures,
    }


def build_release(output: Path, release_dir: Path, report_inputs: Path, commit: str | None = None) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    release_dir.mkdir(parents=True, exist_ok=True)
    report_inputs.mkdir(parents=True, exist_ok=True)
    commit = git_output("rev-parse", commit or "HEAD")
    frame = build_claim_matrix()
    validation = validate_claim_matrix(frame)
    if not validation["success"]:
        raise AssertionError(validation)

    matrix_parquet = output / "claim_to_evidence_matrix.parquet"
    matrix_csv = output / "claim_to_evidence_matrix.csv"
    write_parquet(matrix_parquet, frame)
    frame.to_csv(matrix_csv, index=False)
    write_json(output / "claim_validation.json", validation)

    summary = (
        frame.groupby(["figure", "final_replication_classification"], dropna=False)
        .size().rename("claim_count").reset_index()
    )
    summary.to_csv(output / "classification_summary.csv", index=False)
    figure_rows = []
    for figure, group in frame.groupby("figure", sort=True):
        counts = group.final_replication_classification.value_counts().to_dict()
        figure_rows.append(
            {
                "figure": int(figure),
                "registered_claims": len(group),
                **{f"claims_{name}": int(counts.get(name, 0)) for name in sorted(CLASSIFICATIONS)},
                "claims_with_historical_unavailability": int(group.historical_backend_status.str.startswith("not_testable").sum()),
                "claims_with_event_budget_censoring": int(group.event_budget_censoring_present.sum()),
                "claims_with_strict_duplicate_sensitivity": int(group.strict_duplicate_sensitivity.str.contains("both_retained").sum()),
                "claims_preserving_contradiction": int(group.contradiction_preserved.sum()),
            }
        )
    pd.DataFrame(figure_rows).to_csv(output / "figure_reconciliation.csv", index=False)

    unresolved = frame[frame.final_replication_classification.eq("not_testable")][
        ["claim_id", "figure", "claim_text", "historical_backend_status", "reconciliation_rationale"]
    ]
    unresolved.to_csv(output / "unresolved_claims.csv", index=False)
    unresolved_audit = {
        "schemaVersion": "e01.s14.unresolved_claim_audit.v1",
        "researchStepId": "S14",
        "success": frame.final_replication_classification.notna().all() and frame.claim_id.nunique() == 118,
        "registeredClaims": 118,
        "classifiedClaims": int(frame.final_replication_classification.notna().sum()),
        "unclassifiedClaims": int(frame.final_replication_classification.isna().sum()),
        "notTestableClaims": len(unresolved),
        "implementationSensitiveClaims": int(frame.final_replication_classification.eq("implementation_sensitive").sum()),
        "notReproducedClaims": int(frame.final_replication_classification.eq("not_reproduced").sum()),
        "publicationSnapshotIdentified": False,
        "criterion": "Pass means every row is explicitly classified; not-testable and implementation-sensitive rows remain visible rather than being forced to supportive.",
    }
    write_json(output / "unresolved_claim_audit.json", unresolved_audit)

    source_manifest = source_hash_manifest(commit)
    write_json(release_dir / "source_hash_manifest.json", source_manifest)
    git_pointer = {
        "schemaVersion": "e01.s14.git_pointer.v1",
        "researchStepId": "S14",
        "repository": "https://github.com/Eidosoma/cell_research.git",
        "branch": "eidosoma/groups/28",
        "commit": commit,
        "tree": source_manifest["tree"],
        "referencePackagePath": "reference_simulator/",
        "historicalBackendPointer": "/artifacts/release/historical_backend/manifest.json",
        "publicationSnapshotClaimed": False,
        "sourceArchiveCreated": False,
        "historicalSourceIncluded": False,
    }
    write_json(release_dir / "git_pointer.json", git_pointer)
    write_json(release_dir / "reference-simulator-source.NOT_CREATED.json", {
        "researchStepId": "S14",
        "plannedPath": "/artifacts/release/reference-simulator-source.tar.zst",
        "created": False,
        "replacement": [str(release_dir / "git_pointer.json"), str(release_dir / "source_hash_manifest.json")],
        "reason": source_manifest["sourceArchiveReason"],
    })

    components = _component_records()
    for name, records in components.items():
        write_json(release_dir / f"{name}.json", {
            "schemaVersion": f"e01.s14.{name}.v1",
            "researchStepId": "S14",
            "items": records,
        })
    caveats = _caveats(frame)
    write_json(release_dir / "caveat_register.json", {"schemaVersion": "e01.s14.caveat_register.v1", "researchStepId": "S14", "caveats": caveats})

    commands = {
        "schemaVersion": "e01.s14.figure_commands.v1",
        "researchStepId": "S14",
        "oneCommand": "python scripts/reconcile_release_baseline.py reproduce --artifact-root /artifacts --scratch /cache/e01_s14/reproduction",
        "figureRegeneration": "python scripts/reconcile_release_baseline.py figures --artifact-root /artifacts --scratch /cache/e01_s14/regenerated_figures",
        "sourceFunctions": {
            "Figure3": "analysis.no_fault_sorting._plot_figures",
            "Figure4": "analysis.efficiency_costs.figure4",
            "Figure5": "analysis.frozen_cell_results.draw_figure5",
            "Figure6": "analysis.delayed_gratification.draw_figure6",
            "Figure7": "analysis.delayed_gratification.draw_figure7",
            "Figure8": "analysis.chimeric_replication._plot_figure8",
            "Figure9": "analysis.chimeric_replication._plot_opposing(repeated=False)",
            "Figure10": "analysis.chimeric_replication._plot_opposing(repeated=True)",
        },
        "inputRule": "Collectible compact artifacts only; no /cache trajectory checkpoint is required.",
    }
    write_json(release_dir / "figure_commands.json", commands)
    write_json(release_dir / "one_command_reproduction.json", {
        "schemaVersion": "e01.s14.one_command.v1",
        "researchStepId": "S14",
        "command": commands["oneCommand"],
        "workingDirectory": "immutable Git worktree at the release commit",
        "writes": "/cache/e01_s14/reproduction only",
        "mutatesCollectibleArtifacts": False,
        "requiresHistoricalQuarantine": False,
    })

    _link_or_replace(matrix_parquet, report_inputs / "claim_to_evidence_matrix.parquet")
    _link_or_replace(matrix_csv, report_inputs / "claim_to_evidence_matrix.csv")
    summary.to_csv(report_inputs / "classification_summary.csv", index=False)
    pd.DataFrame(figure_rows).to_csv(report_inputs / "figure_reconciliation.csv", index=False)
    write_json(report_inputs / "lay_summary.json", {
        "researchStepId": "S14",
        "text": "The deterministic simulator and its audit trail are ready for downstream computational work, but this is a reconstructed clean-room baseline rather than an exact replay of the publication. Sorting endpoints are strong; several efficiency, passive-fault, delayed-gratification, aggregation-magnitude, and opposing-direction claims are constrained or contradicted.",
    })
    write_json(report_inputs / "methods_summary.json", {
        "researchStepId": "S14",
        "method": "Claim-ID-preserving synthesis of frozen S01 paper claims with separately labeled C and R evidence from S09--S13; no new scientific population was run.",
        "classificationVocabulary": sorted(CLASSIFICATIONS),
        "evidencePrecedence": json.loads(CONTRACT.read_text())["evidencePrecedence"],
    })
    write_json(report_inputs / "evidence_index.json", {
        "researchStepId": "S14",
        "claimMatrix": file_record(report_inputs / "claim_to_evidence_matrix.parquet"),
        "figures": components["figures"],
        "selectedTraces": components["selectedTraces"],
        "schemas": components["schemas"],
        "scenarios": components["scenarios"],
    })
    write_json(report_inputs / "table_index.json", {
        "researchStepId": "S14",
        "tables": [
            file_record(report_inputs / "claim_to_evidence_matrix.parquet"),
            file_record(report_inputs / "claim_to_evidence_matrix.csv"),
            file_record(report_inputs / "classification_summary.csv"),
            file_record(report_inputs / "figure_reconciliation.csv"),
        ],
    })
    write_json(report_inputs / "caveat_register.json", {"researchStepId": "S14", "caveats": caveats})

    provenance = input_provenance()
    write_json(output / "input_provenance.json", provenance)
    write_json(report_inputs / "provenance_manifest.json", {
        "researchStepId": "S14",
        "inputProvenance": file_record(output / "input_provenance.json"),
        "sourceHashManifest": file_record(release_dir / "source_hash_manifest.json"),
        "gitPointer": file_record(release_dir / "git_pointer.json"),
    })
    write_json(release_dir / "provenance_manifest.json", {
        "researchStepId": "S14",
        "inputProvenance": file_record(output / "input_provenance.json"),
        "sourceHashManifest": file_record(release_dir / "source_hash_manifest.json"),
        "claimMatrix": file_record(matrix_parquet),
    })

    report_bundle_manifest = {
        "schemaVersion": "e01.s14.report_bundle_manifest.v1",
        "researchStepId": "S14",
        "complete": True,
        "inputs": report_input_records(report_inputs),
        "releaseDirectory": str(release_dir),
    }
    write_json(report_inputs / "report_bundle_manifest.json", report_bundle_manifest)

    release_manifest = {
        "schemaVersion": "e01.s14.baseline_release.v1",
        "researchStepId": "S14",
        "releaseClassification": "validated_clean_room_baseline_with_constraining_historical_replication_evidence",
        "repositoryCommit": commit,
        "repositoryTree": source_manifest["tree"],
        "sourceArchiveCreated": False,
        "historicalSourceIncluded": False,
        "publicationSnapshotClaimed": False,
        "claims": validation,
        "components": [file_record(path) for path in sorted(release_dir.iterdir()) if path.is_file() and path.name != "release_manifest.json"],
        "reportInputs": str(report_inputs),
        "oneCommand": commands["oneCommand"],
    }
    write_json(release_dir / "release_manifest.json", release_manifest)
    return {
        "claimValidation": validation,
        "unresolvedAudit": unresolved_audit,
        "releaseManifest": file_record(release_dir / "release_manifest.json"),
        "reportBundleManifest": file_record(report_inputs / "report_bundle_manifest.json"),
        "repositoryCommit": commit,
    }


def verify_source_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    mismatches = []
    for item in manifest["files"]:
        payload = subprocess.check_output(
            ["git", "show", f"{manifest['commit']}:{item['path']}"], cwd=REPOSITORY
        )
        observed = hashlib.sha256(payload).hexdigest()
        if observed != item["sha256"]:
            mismatches.append({"path": item["path"], "expected": item["sha256"], "observed": observed})
    return {
        "schemaVersion": "e01.s14.source_checksum_audit.v1",
        "researchStepId": "S14",
        "success": not mismatches and git_output("cat-file", "-t", manifest["commit"]) == "commit",
        "commit": manifest["commit"],
        "tree": manifest["tree"],
        "filesChecked": len(manifest["files"]),
        "mismatches": mismatches,
        "sourceArchiveAbsent": not Path(manifest["sourceArchivePathFromPlan"]).exists(),
        "historicalSourceIncluded": manifest["historicalSourceIncluded"],
    }


def _walk_records(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        if (
            isinstance(value.get("path"), str)
            and Path(value["path"]).is_absolute()
            and isinstance(value.get("sha256"), str)
        ):
            yield value
        for child in value.values():
            yield from _walk_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_records(child)


def artifact_link_audit(release_dir: Path, report_inputs: Path) -> dict[str, Any]:
    checked: dict[str, dict[str, Any]] = {}
    issues = []
    roots = [release_dir, report_inputs]
    for root in roots:
        for manifest_path in sorted(root.glob("*.json")):
            try:
                value = json.loads(manifest_path.read_text())
            except json.JSONDecodeError as error:
                issues.append({"manifest": str(manifest_path), "error": f"invalid_json:{error}"})
                continue
            for record in _walk_records(value):
                path = Path(record["path"])
                key = str(path)
                if key in checked:
                    continue
                observed = None
                exists = path.is_file()
                if exists:
                    observed = sha256_file(path)
                valid = exists and observed == record["sha256"]
                checked[key] = {"exists": exists, "hashMatches": valid, "expectedSha256": record["sha256"], "observedSha256": observed}
                if not valid:
                    issues.append({"path": key, **checked[key]})
    required = [
        release_dir / "release_manifest.json",
        release_dir / "git_pointer.json",
        release_dir / "source_hash_manifest.json",
        release_dir / "one_command_reproduction.json",
        report_inputs / "claim_to_evidence_matrix.parquet",
        report_inputs / "evidence_index.json",
        report_inputs / "report_bundle_manifest.json",
    ]
    missing_required = [str(path) for path in required if not path.is_file()]
    return {
        "schemaVersion": "e01.s14.artifact_link_audit.v1",
        "researchStepId": "S14",
        "success": not issues and not missing_required,
        "manifestsScanned": sum(1 for root in roots for _ in root.glob("*.json")),
        "recordsChecked": len(checked),
        "missingRequired": missing_required,
        "issues": issues,
    }


def downstream_readiness(artifact_root: Path, audits: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    gates = {
        "S07InvariantGate": json.loads((artifact_root / "research_steps" / "S07" / "validation_summary.json").read_text())["success"],
        "S08ScenarioGate": json.loads((artifact_root / "research_steps" / "S08" / "validation_summary.json").read_text())["success"],
        "S06SchemaGate": json.loads((artifact_root / "research_steps" / "S06" / "validation_summary.json").read_text())["success"],
        "claimReconciliation": audits["claims"]["success"],
        "sourceChecksums": audits["checksums"]["success"],
        "artifactLinks": audits["links"]["success"],
        "figureRegeneration": audits["figures"]["success"],
        "referenceSmoke": audits["smoke"]["success"],
        "unclassifiedClaimsZero": audits["unresolved"]["unclassifiedClaims"] == 0,
    }
    carry_forward = [
        "Use the R simulator as a reconstructed clean-room baseline, not an exact publication replay.",
        "Keep C recorded_swap evidence separate from R activation evidence.",
        "Retain exact-legacy/corrected placement and strict/non-strict duplicate metric labels.",
        "Treat event-budget terminals as censored and preserve opposing-direction stop metadata.",
        "Do not use policy_search_holdout or other protected splits without the successor experiment's declared gate.",
    ]
    return {
        "schemaVersion": "e01.s14.downstream_readiness.v1",
        "researchStepId": "S14",
        "success": all(gates.values()),
        "readiness": "ready_with_mandatory_constraints" if all(gates.values()) else "not_ready",
        "gates": gates,
        "eligibleSuccessorExperiments": ["E02", "E03", "E04"] if all(gates.values()) else [],
        "startedSuccessorExperiment": False,
        "mandatoryCarryForward": carry_forward,
    }


def environment_record(worktree: Path | None = None, venv: Path | None = None) -> dict[str, Any]:
    return {
        "schemaVersion": "e01.s14.environment.v1",
        "researchStepId": "S14",
        "timestampUtc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pa.__version__,
        "cpuCountVisible": os.cpu_count(),
        "workerLimit": 8,
        "workersUsedForS14Synthesis": 1,
        "threadEnvironment": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "freshWorktree": str(worktree) if worktree else None,
        "freshVenv": str(venv) if venv else None,
        "newDependenciesInstalled": [],
    }


def reproduce(artifact_root: Path, scratch: Path, *, write_collectible_audits: Path | None = None) -> dict[str, Any]:
    global ARTIFACT_ROOT, REGISTRY
    ARTIFACT_ROOT = artifact_root
    REGISTRY = artifact_root / "research_steps" / "S01" / "claim_registry.parquet"
    scratch.mkdir(parents=True, exist_ok=True)
    release_dir = artifact_root / "release" / "baseline"
    report_inputs = artifact_root / "report_inputs"
    claims = validate_claim_matrix(pd.read_parquet(report_inputs / "claim_to_evidence_matrix.parquet"))
    checksums = verify_source_manifest(release_dir / "source_hash_manifest.json")
    smoke = run_reference_smoke()
    figures = regenerate_figures(scratch / "figures")
    links = artifact_link_audit(release_dir, report_inputs)
    unresolved = json.loads((artifact_root / "research_steps" / "S14" / "unresolved_claim_audit.json").read_text())
    readiness = downstream_readiness(artifact_root, {
        "claims": claims,
        "checksums": checksums,
        "smoke": smoke,
        "figures": figures,
        "links": links,
        "unresolved": unresolved,
    })
    result = {
        "schemaVersion": "e01.s14.one_command_reproduction_audit.v1",
        "researchStepId": "S14",
        "success": all(item["success"] for item in (claims, checksums, smoke, figures, links, readiness)),
        "claimAudit": claims,
        "checksumAudit": checksums,
        "referenceSmoke": smoke,
        "figureAudit": figures,
        "artifactLinkAudit": links,
        "unresolvedClaimAudit": unresolved,
        "downstreamReadiness": readiness,
        "scratch": str(scratch),
        "collectibleArtifactsMutated": False,
    }
    write_json(scratch / "one_command_reproduction_audit.json", result)
    if write_collectible_audits is not None:
        write_collectible_audits.mkdir(parents=True, exist_ok=True)
        write_json(write_collectible_audits / "checksum_audit.json", checksums)
        write_json(write_collectible_audits / "reference_smoke.json", smoke)
        write_json(write_collectible_audits / "figure_regeneration_audit.json", figures)
        write_json(write_collectible_audits / "artifact_link_audit.json", links)
        write_json(write_collectible_audits / "downstream_readiness_audit.json", readiness)
        write_json(write_collectible_audits / "one_command_reproduction_audit.json", result)
    return result


def finalize_artifact_manifest(output: Path, release_dir: Path, report_inputs: Path) -> dict[str, Any]:
    artifacts = []
    for scope, directory in (("S14", output), ("release", release_dir), ("report_inputs", report_inputs)):
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.name != "artifact_manifest.json":
                artifacts.append({"scope": scope, **file_record(path)})
    result = {
        "schemaVersion": "e01.s14.artifact_manifest.v1",
        "researchStepId": "S14",
        "artifactCount": len(artifacts),
        "artifacts": artifacts,
        "manifestSelfHashExcluded": True,
        "historicalSourceIncluded": False,
        "sourceArchiveCreated": False,
    }
    write_json(output / "artifact_manifest.json", result)
    return result
