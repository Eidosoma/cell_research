#!/usr/bin/env python3
"""Package the E01 baseline replication and regression suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd


EXPERIMENT_ID = "E01"
STEP_ID = "S15"
STEP_NUMBER = 15
STATUS = "completed"
OUTCOME_CLASSIFICATION = "supportive"
REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_MANIFEST_DEFAULT = Path("/artifacts/provenance/run_manifest.json")


FIGURE_REFERENCES = [
    {
        "paperFigure": "Figure 3",
        "title": "No-Frozen sortedness trajectories",
        "producerStep": "S04",
        "path": "/artifacts/figures/e01/figure03_sortedness_trajectories.png",
        "pdfPath": "/artifacts/figures/e01/figure03_sortedness_trajectories.pdf",
        "role": "replicated_figure",
        "summary": "Traditional and cell-view Bubble, Insertion, and Selection all reach final 100 percent Sortedness.",
    },
    {
        "paperFigure": "Figure 4",
        "title": "Efficiency comparisons",
        "producerStep": "S05",
        "path": "/artifacts/figures/e01/figure04_efficiency.png",
        "pdfPath": "/artifacts/figures/e01/figure04_efficiency.pdf",
        "role": "replicated_figure",
        "summary": "Swap-only and swap-plus-comparison efficiency comparisons for matched no-Frozen runs.",
    },
    {
        "paperFigure": "Figure 5",
        "title": "Frozen Cell robustness",
        "producerStep": "S07",
        "path": "/artifacts/figures/e01/figure05_frozen_robustness.png",
        "pdfPath": "/artifacts/figures/e01/figure05_frozen_robustness.pdf",
        "role": "replicated_figure",
        "summary": "Final monotonicity error under passive and stuck Frozen Cell perturbations.",
    },
    {
        "paperFigure": "Figure 7",
        "title": "Delayed Gratification",
        "producerStep": "S08",
        "path": "/artifacts/figures/e01/figure07_dg.png",
        "pdfPath": "/artifacts/figures/e01/figure07_dg.pdf",
        "role": "replicated_figure",
        "summary": "Primary stuck-Frozen Delayed Gratification trends.",
    },
    {
        "paperFigure": "Figure 7 sensitivity",
        "title": "Delayed Gratification variant sensitivity",
        "producerStep": "S08",
        "path": "/artifacts/figures/e01/figure07_dg_variant_sensitivity.png",
        "pdfPath": "/artifacts/figures/e01/figure07_dg_variant_sensitivity.pdf",
        "role": "sensitivity_figure",
        "summary": "Passive Frozen Cell sensitivity for the DG metric.",
    },
    {
        "paperFigure": "Figure 8",
        "title": "Same-goal chimera sortedness",
        "producerStep": "S09",
        "path": "/artifacts/figures/e01/figure08_same_goal_chimera_sortedness.png",
        "pdfPath": "/artifacts/figures/e01/figure08_same_goal_chimera_sortedness.pdf",
        "role": "replicated_figure",
        "summary": "Same-goal pure and mixed cell-view chimera completion.",
    },
    {
        "paperFigure": "Figure 8",
        "title": "Same-goal chimera efficiency",
        "producerStep": "S09",
        "path": "/artifacts/figures/e01/figure08_same_goal_chimera_efficiency.png",
        "pdfPath": "/artifacts/figures/e01/figure08_same_goal_chimera_efficiency.pdf",
        "role": "replicated_figure",
        "summary": "Mixed-efficiency interpolation against pure-policy controls.",
    },
    {
        "paperFigure": "Figure 8",
        "title": "Unique-value Aggregation",
        "producerStep": "S10",
        "path": "/artifacts/figures/e01/figure08_aggregation_unique.png",
        "pdfPath": "/artifacts/figures/e01/figure08_aggregation_unique.pdf",
        "role": "replicated_figure",
        "summary": "Same-goal mixed-Algotype Aggregation curves above fixed-count nulls.",
    },
    {
        "paperFigure": "Figure 8 controls",
        "title": "Aggregation controls",
        "producerStep": "S10",
        "path": "/artifacts/figures/e01/figure08_aggregation_controls.png",
        "pdfPath": "/artifacts/figures/e01/figure08_aggregation_controls.pdf",
        "role": "control_figure",
        "summary": "Available pure and fixed-count random-label controls for Aggregation.",
    },
    {
        "paperFigure": "Figure 8 repeated values",
        "title": "Duplicate-value chimeras",
        "producerStep": "S11",
        "path": "/artifacts/figures/e01/figure08_duplicate_chimeras.png",
        "pdfPath": "/artifacts/figures/e01/figure08_duplicate_chimeras.pdf",
        "role": "replicated_figure",
        "summary": "Same-goal duplicate-value chimeras with repeated values.",
    },
    {
        "paperFigure": "Figure 8 repeated values",
        "title": "Duplicate versus unique Aggregation",
        "producerStep": "S11",
        "path": "/artifacts/figures/e01/figure08_duplicate_vs_unique_aggregation.png",
        "pdfPath": "/artifacts/figures/e01/figure08_duplicate_vs_unique_aggregation.pdf",
        "role": "comparison_figure",
        "summary": "Duplicate-value Aggregation compared with unique-value S10 Aggregation.",
    },
    {
        "paperFigure": "Figure 9",
        "title": "Unique-value opposite-direction chimeras",
        "producerStep": "S12",
        "path": "/artifacts/figures/e01/figure09_unique_opposite.png",
        "pdfPath": "/artifacts/figures/e01/figure09_unique_opposite.pdf",
        "role": "replicated_figure",
        "summary": "Opposite-direction unique-value equilibria and dominance-like outcomes.",
    },
    {
        "paperFigure": "Figure 10",
        "title": "Repeated-value opposite-direction chimeras",
        "producerStep": "S12",
        "path": "/artifacts/figures/e01/figure10_repeated_opposite.png",
        "pdfPath": "/artifacts/figures/e01/figure10_repeated_opposite.pdf",
        "role": "replicated_figure",
        "summary": "Repeated-value opposite-direction equilibria; two paper claims remain not replicated.",
    },
    {
        "paperFigure": "S14 scale",
        "title": "Scaled CI width changes",
        "producerStep": "S14",
        "path": "/artifacts/figures/e01/figure_s14_ci_width_changes.png",
        "pdfPath": "/artifacts/figures/e01/figure_s14_ci_width_changes.pdf",
        "role": "scaled_validation_figure",
        "summary": "N=1000 CI widths compared with N=100 baseline widths.",
    },
    {
        "paperFigure": "S14 scale",
        "title": "Scaled same-goal Aggregation",
        "producerStep": "S14",
        "path": "/artifacts/figures/e01/figure_s14_scaled_aggregation.png",
        "pdfPath": "/artifacts/figures/e01/figure_s14_scaled_aggregation.pdf",
        "role": "scaled_validation_figure",
        "summary": "N=1000 same-goal mixed-Algotype Aggregation curves.",
    },
]


TABLE_REFERENCES = [
    {"title": "Code-to-paper mapping matrix", "producerStep": "S02", "path": "/artifacts/research_steps/S02/mapping_matrix.csv", "role": "mapping"},
    {"title": "Function index", "producerStep": "S02", "path": "/artifacts/research_steps/S02/function_index.csv", "role": "mapping"},
    {"title": "Baseline config", "producerStep": "S03", "path": "/artifacts/configs/e01_baseline_config.json", "role": "configuration"},
    {"title": "Condition matrix", "producerStep": "S03", "path": "/artifacts/research_steps/S03/condition_matrix.csv", "role": "configuration"},
    {"title": "Seed table", "producerStep": "S03", "path": "/artifacts/research_steps/S03/seed_table.csv", "role": "configuration"},
    {"title": "Figure 3 condition summary", "producerStep": "S04", "path": "/artifacts/results/e01_s04_condition_summary.csv", "role": "result_table"},
    {"title": "Figure 3 replicate summary", "producerStep": "S04", "path": "/artifacts/results/e01_s04_replicate_summary.parquet", "role": "result_table"},
    {"title": "Efficiency table", "producerStep": "S05", "path": "/artifacts/results/e01_efficiency.parquet", "role": "result_table"},
    {"title": "Efficiency pairwise table", "producerStep": "S05", "path": "/artifacts/results/e01_efficiency_pairwise.parquet", "role": "result_table"},
    {"title": "Efficiency statistics", "producerStep": "S06", "path": "/artifacts/results/e01_efficiency_statistics.parquet", "role": "statistics"},
    {"title": "All statistics", "producerStep": "S06", "path": "/artifacts/results/e01_statistics.parquet", "role": "statistics"},
    {"title": "Frozen Cell robustness summary", "producerStep": "S07", "path": "/artifacts/results/e01_frozen_cell_robustness_summary.parquet", "role": "result_table"},
    {"title": "Figure 5 backing table", "producerStep": "S07", "path": "/artifacts/results/e01_frozen_cell_figure05_backing.parquet", "role": "figure_backing"},
    {"title": "Delayed Gratification summary", "producerStep": "S08", "path": "/artifacts/results/e01_delayed_gratification_summary.parquet", "role": "result_table"},
    {"title": "Delayed Gratification trends", "producerStep": "S08", "path": "/artifacts/results/e01_delayed_gratification_trends.parquet", "role": "result_table"},
    {"title": "Same-goal chimera summary", "producerStep": "S09", "path": "/artifacts/results/e01_same_goal_chimera_summary.parquet", "role": "result_table"},
    {"title": "Same-goal chimera efficiency", "producerStep": "S09", "path": "/artifacts/results/e01_same_goal_chimera_efficiency.parquet", "role": "result_table"},
    {"title": "Aggregation peak summary", "producerStep": "S10", "path": "/artifacts/results/e01_aggregation_peak_summary.parquet", "role": "result_table"},
    {"title": "Aggregation statistics", "producerStep": "S10", "path": "/artifacts/results/e01_aggregation_statistics.parquet", "role": "statistics"},
    {"title": "Duplicate-value chimera summary", "producerStep": "S11", "path": "/artifacts/results/e01_duplicate_value_chimera_summary.parquet", "role": "result_table"},
    {"title": "Duplicate versus unique Aggregation", "producerStep": "S11", "path": "/artifacts/results/e01_duplicate_value_unique_comparison.parquet", "role": "comparison_table"},
    {"title": "Opposite-direction chimera summary", "producerStep": "S12", "path": "/artifacts/results/e01_opposite_direction_chimera_summary.parquet", "role": "result_table"},
    {"title": "Opposite-direction paper comparison", "producerStep": "S12", "path": "/artifacts/results/e01_opposite_direction_paper_comparison.parquet", "role": "comparison_table"},
    {"title": "Replication classification", "producerStep": "S13", "path": "/artifacts/results/e01_replication_classification.parquet", "role": "classification"},
    {"title": "Replication classification summary", "producerStep": "S13", "path": "/artifacts/results/e01_replication_classification_summary.parquet", "role": "classification"},
    {"title": "Scaled replication rows", "producerStep": "S14", "path": "/artifacts/results/e01_scaled_replication.parquet", "role": "scaled_result_table"},
    {"title": "Scaled replication summary", "producerStep": "S14", "path": "/artifacts/results/e01_scaled_replication_summary.parquet", "role": "scaled_result_table"},
    {"title": "Scaled CI comparison", "producerStep": "S14", "path": "/artifacts/results/e01_scaled_ci_comparison.parquet", "role": "scaled_result_table"},
    {"title": "Scaled efficiency pairwise", "producerStep": "S14", "path": "/artifacts/results/e01_scaled_efficiency_pairwise.parquet", "role": "scaled_result_table"},
    {"title": "Scaled Delayed Gratification trends", "producerStep": "S14", "path": "/artifacts/results/e01_scaled_delayed_gratification_trends.parquet", "role": "scaled_result_table"},
    {"title": "Scaled Aggregation peaks", "producerStep": "S14", "path": "/artifacts/results/e01_scaled_aggregation_peak_summary.parquet", "role": "scaled_result_table"},
]


REPORT_REFERENCES = [
    {"title": "Divergence log", "producerStep": "S13", "path": "/artifacts/reports/e01_divergence_log.md", "role": "claim_audit"},
    {"title": "Code-to-paper map", "producerStep": "S02", "path": "/artifacts/research_steps/S02/code_paper_map.md", "role": "methods"},
    {"title": "Baseline config notes", "producerStep": "S03", "path": "/artifacts/research_steps/S03/config_notes.md", "role": "methods"},
    {"title": "S14 scaled summary", "producerStep": "S14", "path": "/artifacts/research_steps/S14/summary.md", "role": "summary"},
]


REGRESSION_TEST_CODE = r'''#!/usr/bin/env python3
"""Regression tests for the packaged E01 baseline artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import unittest
from pathlib import Path

import pandas as pd


ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sortedness_percent(values: list[int], direction: str = "increasing") -> float:
    if len(values) < 2:
        return 100.0
    if direction == "increasing":
        ok = sum(left <= right for left, right in zip(values, values[1:]))
    elif direction == "decreasing":
        ok = sum(left >= right for left, right in zip(values, values[1:]))
    else:
        raise ValueError(direction)
    return 100.0 * ok / (len(values) - 1)


def monotonicity_error(values: list[int], direction: str = "increasing") -> int:
    return (len(values) - 1) - int(round(sortedness_percent(values, direction) * (len(values) - 1) / 100.0))


def aggregation(sequence: list[str]) -> float:
    if len(sequence) < 2:
        return 1.0
    same = sum(left == right for left, right in zip(sequence, sequence[1:]))
    return same / (len(sequence) - 1)


class TestE01BaselineRegression(unittest.TestCase):
    def test_source_statuses_succeeded(self) -> None:
        for step in range(1, 15):
            status_path = ARTIFACTS_DIR / "research_steps" / f"S{step:02d}" / "status.json"
            self.assertTrue(status_path.exists(), status_path)
            status = json.loads(status_path.read_text())
            self.assertEqual(status["researchStepId"], f"S{step:02d}")
            self.assertTrue(status["success"], status_path)
            self.assertEqual(status["status"], "completed")

    def test_metric_formulas_on_toy_arrays(self) -> None:
        self.assertEqual(sortedness_percent([1, 2, 3, 4]), 100.0)
        self.assertEqual(sortedness_percent([4, 3, 2, 1], "decreasing"), 100.0)
        self.assertAlmostEqual(sortedness_percent([1, 3, 2, 4]), 100.0 * 2 / 3)
        self.assertEqual(monotonicity_error([1, 3, 2, 4]), 1)
        self.assertAlmostEqual(aggregation(["A", "A", "B", "B"]), 2 / 3)
        self.assertAlmostEqual(aggregation(["A", "B", "A", "B"]), 0.0)

    def test_baseline_config_contract(self) -> None:
        config = json.loads((ARTIFACTS_DIR / "configs" / "e01_baseline_config.json").read_text())
        self.assertEqual(config["configVersion"], "e01_baseline_config.v1")
        self.assertEqual(config["paperBaseline"]["uniqueInputProfile"]["n"], 100)
        self.assertEqual(config["paperBaseline"]["repeatCount"], 100)
        condition_matrix = pd.read_csv(ARTIFACTS_DIR / "research_steps" / "S03" / "condition_matrix.csv")
        seed_table = pd.read_csv(ARTIFACTS_DIR / "research_steps" / "S03" / "seed_table.csv")
        self.assertEqual(len(condition_matrix), 64)
        self.assertEqual(len(seed_table), 6400)
        self.assertTrue((condition_matrix["n"] == 100).all())
        self.assertTrue((condition_matrix["repeatCount"] == 100).all())

    def test_no_frozen_completion(self) -> None:
        s04 = pd.read_parquet(ARTIFACTS_DIR / "results" / "e01_s04_replicate_summary.parquet")
        self.assertEqual(len(s04), 600)
        self.assertTrue(s04["completed"].all())
        self.assertTrue((s04["final_sortedness_percent"] == 100.0).all())
        scaled = pd.read_parquet(ARTIFACTS_DIR / "results" / "e01_scaled_replication.parquet")
        no_frozen = scaled[scaled["scaledFamily"] == "no_frozen_efficiency"]
        self.assertEqual(len(no_frozen), 6000)
        self.assertTrue((no_frozen["finalSortednessPercent"] == 100.0).all())

    def test_efficiency_anchor_directions(self) -> None:
        pairwise = pd.read_parquet(ARTIFACTS_DIR / "results" / "e01_scaled_efficiency_pairwise.parquet")
        rows = {(row.algorithm, row.metricId): row for row in pairwise.itertuples(index=False)}
        self.assertEqual(rows[("bubble", "swapCount")].cellViewMinusTraditionalMean, 0.0)
        self.assertEqual(rows[("insertion", "swapCount")].cellViewMinusTraditionalMean, 0.0)
        self.assertLess(rows[("bubble", "swapPlusComparisonSteps")].cellViewMinusTraditionalMean, 0.0)
        self.assertGreater(rows[("insertion", "swapPlusComparisonSteps")].cellViewMinusTraditionalMean, 0.0)
        self.assertGreater(rows[("selection", "swapCount")].cellViewMinusTraditionalMean, 0.0)
        self.assertGreater(rows[("selection", "swapPlusComparisonSteps")].cellViewMinusTraditionalMean, 0.0)

    def test_frozen_cell_robustness_anchor(self) -> None:
        summary = pd.read_parquet(ARTIFACTS_DIR / "results" / "e01_frozen_cell_robustness_summary.parquet")
        subset = summary[(summary["frozenCount"] > 0) & (summary["frozenVariant"] != "none")]
        checked = 0
        for (algorithm, variant, frozen_count), group in subset.groupby(["algorithm", "frozenVariant", "frozenCount"]):
            values = group.set_index("implementation")["meanFinalMonotonicityError"].to_dict()
            self.assertLessEqual(values["cell_view"], values["traditional"] + 1e-12, (algorithm, variant, frozen_count, values))
            checked += 1
        self.assertEqual(checked, 18)

    def test_delayed_gratification_primary_trends(self) -> None:
        trends = pd.read_parquet(ARTIFACTS_DIR / "results" / "e01_scaled_delayed_gratification_trends.parquet")
        stuck = trends[(trends["frozenVariant"] == "stuck") & (trends["algorithm"].isin(["bubble", "insertion"]))]
        self.assertEqual(len(stuck), 4)
        self.assertTrue(stuck["monotoneNonDecreasing"].all())
        self.assertTrue((stuck["linearSlopePerFrozenCell"] > 0).all())

    def test_aggregation_above_null(self) -> None:
        peaks = pd.read_parquet(ARTIFACTS_DIR / "results" / "e01_aggregation_peak_summary.parquet")
        mixed = peaks[peaks["isMixedCondition"]]
        self.assertEqual(len(mixed), 4)
        self.assertTrue((mixed["peakAggregationMinusNullBootstrapCi95Lower"] > 0).all())
        scaled = pd.read_parquet(ARTIFACTS_DIR / "results" / "e01_scaled_aggregation_peak_summary.parquet")
        self.assertEqual(len(scaled), 4)
        self.assertTrue((scaled["peakAggregationMinusNullMean"] > 0).all())

    def test_reproducibility_classification_counts(self) -> None:
        table = pd.read_parquet(ARTIFACTS_DIR / "results" / "e01_replication_classification.parquet")
        self.assertEqual(len(table), 32)
        counts = table["classification"].value_counts().to_dict()
        self.assertEqual(counts.get("exact"), 3)
        self.assertEqual(counts.get("statistically consistent"), 9)
        self.assertEqual(counts.get("directionally consistent"), 17)
        self.assertEqual(counts.get("not replicated"), 3)

    def test_report_bundle_references_exist_and_hash(self) -> None:
        manifest_path = ARTIFACTS_DIR / "report_bundle_inputs" / "bundle_manifest.json"
        self.assertTrue(manifest_path.exists(), manifest_path)
        manifest = json.loads(manifest_path.read_text())
        for section in ["figures", "tables", "reports"]:
            self.assertGreater(len(manifest[section]), 0, section)
            for item in manifest[section]:
                path = Path(item["path"])
                self.assertTrue(path.exists(), path)
                self.assertGreater(path.stat().st_size, 0, path)
                if item.get("sha256"):
                    self.assertEqual(sha256_path(path), item["sha256"], path)

    def test_scaled_counts(self) -> None:
        scaled = pd.read_parquet(ARTIFACTS_DIR / "results" / "e01_scaled_replication.parquet")
        counts = scaled.groupby("scaledFamily").size().to_dict()
        self.assertEqual(counts.get("no_frozen_efficiency"), 6000)
        self.assertEqual(counts.get("frozen_robustness"), 36000)
        self.assertEqual(counts.get("same_goal_aggregation"), 4000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
'''


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "sizeBytes": path.stat().st_size if path.exists() else 0,
        "sha256": sha256_path(path) if path.exists() and path.is_file() else "",
    }
    if extra:
        record.update(extra)
    return record


def run_command(args: list[str], cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(args, cwd=cwd or REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def git_info() -> dict[str, Any]:
    return {
        "repository": str(REPO_ROOT),
        "branch": run_command(["git", "branch", "--show-current"]),
        "commit": run_command(["git", "rev-parse", "HEAD"]),
        "statusShort": run_command(["git", "status", "--short"]),
        "remoteOriginUrl": run_command(["git", "remote", "get-url", "origin"]),
    }


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def clean(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            if not math.isfinite(value):
                return ""
            return f"{value:.4g}"
        return str(value).replace("\n", " ").replace("|", "\\|")

    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(clean(value) for value in row) + " |")
    return "\n".join(lines)


def write_table(df: pd.DataFrame, parquet_path: Path, csv_path: Path) -> list[Path]:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    return [parquet_path, csv_path]


def table_shape(path: Path) -> tuple[int | None, int | None]:
    if not path.exists():
        return None, None
    try:
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        elif path.suffix == ".csv":
            df = pd.read_csv(path)
        else:
            return None, None
        return int(len(df)), int(len(df.columns))
    except Exception:
        return None, None


def enrich_reference(item: dict[str, Any]) -> dict[str, Any]:
    path = Path(item["path"])
    rows, cols = table_shape(path) if item.get("role", "").endswith("table") or path.suffix in {".csv", ".parquet"} else (None, None)
    record = {
        **item,
        "exists": path.exists(),
        "sizeBytes": path.stat().st_size if path.exists() else 0,
        "sha256": sha256_path(path) if path.exists() and path.is_file() else "",
        "rowCount": rows,
        "columnCount": cols,
    }
    if item.get("pdfPath"):
        pdf_path = Path(str(item["pdfPath"]))
        record["pdfExists"] = pdf_path.exists()
        record["pdfSizeBytes"] = pdf_path.stat().st_size if pdf_path.exists() else 0
        record["pdfSha256"] = sha256_path(pdf_path) if pdf_path.exists() and pdf_path.is_file() else ""
    return record


def load_statuses(artifacts_dir: Path, through_step: int = 14) -> list[dict[str, Any]]:
    statuses = []
    for step in range(1, through_step + 1):
        path = artifacts_dir / "research_steps" / f"S{step:02d}" / "status.json"
        payload = load_json(path)
        statuses.append(
            {
                "researchStepId": payload.get("researchStepId", f"S{step:02d}"),
                "stepNumber": payload.get("stepNumber", step),
                "success": bool(payload.get("success")),
                "status": payload.get("status", ""),
                "validationResult": payload.get("validationResult", ""),
                "outcomeClassification": payload.get("outcomeClassification", ""),
                "artifactCount": len(payload.get("artifactsWritten", [])),
                "statusPath": str(path),
            }
        )
    return statuses


def build_artifact_index(artifacts_dir: Path, exclude_paths: set[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    allowed_roots = {"configs", "figures", "provenance", "reports", "research_steps", "results", "tests", "traces", "report_bundle_inputs"}
    for path in sorted(artifacts_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(artifacts_dir)
        if rel.parts[0] not in allowed_roots:
            continue
        if path.resolve() in exclude_paths:
            continue
        step_match = re.search(r"S\d{2}", str(rel))
        rows.append(
            {
                "experimentId": EXPERIMENT_ID,
                "researchStepId": step_match.group(0) if step_match else "",
                "artifactCategory": rel.parts[0],
                "relativePath": str(rel),
                "path": str(path),
                "fileName": path.name,
                "extension": path.suffix,
                "sizeBytes": path.stat().st_size,
                "sha256": sha256_path(path),
            }
        )
    return pd.DataFrame(rows)


def write_checksum_file(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for record in sorted(records, key=lambda item: item["path"]):
        if record.get("sha256"):
            lines.append(f"{record['sha256']}  {record['path']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def source_summaries(artifacts_dir: Path) -> dict[str, Any]:
    classification = pd.read_parquet(artifacts_dir / "results" / "e01_replication_classification.parquet")
    scaled = pd.read_parquet(artifacts_dir / "results" / "e01_scaled_replication.parquet")
    scaled_ci = pd.read_parquet(artifacts_dir / "results" / "e01_scaled_ci_comparison.parquet")
    s14_runtime = pd.read_parquet(artifacts_dir / "results" / "e01_scaled_runtime_summary.parquet")
    s04 = pd.read_parquet(artifacts_dir / "results" / "e01_s04_replicate_summary.parquet")
    s10_peaks = pd.read_parquet(artifacts_dir / "results" / "e01_aggregation_peak_summary.parquet")
    s14_peaks = pd.read_parquet(artifacts_dir / "results" / "e01_scaled_aggregation_peak_summary.parquet")
    dg_trends = pd.read_parquet(artifacts_dir / "results" / "e01_scaled_delayed_gratification_trends.parquet")
    ci_focus = scaled_ci[scaled_ci["ciWidthRatioS14OverBaseline"].notna()].copy()
    return {
        "classificationCounts": classification["classification"].value_counts().to_dict(),
        "notReplicatedClaims": classification.loc[classification["classification"] == "not replicated", "claimId"].tolist(),
        "classificationByFigure": pd.crosstab(classification["paperFigure"], classification["classification"]).reset_index().to_dict(orient="records"),
        "s04ReplicateCount": int(len(s04)),
        "s04Final100Count": int((s04["final_sortedness_percent"] == 100.0).sum()),
        "scaledFamilyCounts": scaled.groupby("scaledFamily").size().to_dict(),
        "scaledNoFrozenFinal100": int(((scaled["scaledFamily"] == "no_frozen_efficiency") & (scaled["finalSortednessPercent"] == 100.0)).sum()),
        "scaledCiRatioMeanByFamily": ci_focus.groupby("scaledFamily")["ciWidthRatioS14OverBaseline"].mean().to_dict(),
        "s14Runtime": s14_runtime.to_dict(orient="records"),
        "s10MixedPeakMinLowerCi": float(s10_peaks.loc[s10_peaks["isMixedCondition"], "peakAggregationMinusNullBootstrapCi95Lower"].min()),
        "s14PeakMinusNullMeanMin": float(s14_peaks["peakAggregationMinusNullMean"].min()),
        "s14DgStuckPositiveTrendCount": int(
            len(
                dg_trends[
                    (dg_trends["frozenVariant"] == "stuck")
                    & (dg_trends["algorithm"].isin(["bubble", "insertion"]))
                    & (dg_trends["monotoneNonDecreasing"])
                    & (dg_trends["linearSlopePerFrozenCell"] > 0)
                ]
            )
        ),
    }


def render_replication_report(
    report_path: Path,
    statuses: list[dict[str, Any]],
    summaries: dict[str, Any],
    figures: list[dict[str, Any]],
    tables: list[dict[str, Any]],
) -> None:
    classification_counts = summaries["classificationCounts"]
    status_rows = [
        [
            item["researchStepId"],
            item["status"],
            item["success"],
            item["outcomeClassification"],
            item["validationResult"],
        ]
        for item in statuses
    ]
    figure_rows = [
        [item["paperFigure"], item["producerStep"], item["title"], item["path"], "yes" if item["exists"] else "missing"]
        for item in figures
    ]
    table_rows = [
        [
            item["producerStep"],
            item["title"],
            item["path"],
            item.get("rowCount", ""),
            item.get("columnCount", ""),
        ]
        for item in tables
    ]
    count_rows = [[key, value] for key, value in sorted(classification_counts.items())]
    ci_rows = [[key, value] for key, value in sorted(summaries["scaledCiRatioMeanByFamily"].items())]
    runtime_rows = [
        [
            row["family"],
            row["conditionCount"],
            row["replicatesPerCondition"],
            row["taskCount"],
            row["wallSeconds"],
            row["workerCount"],
        ]
        for row in summaries["s14Runtime"]
    ]

    lines = [
        "# E01 Replication Report",
        "",
        "## Scope",
        "",
        "This report packages the E01 baseline replication of Zhang, Goldstein, and Levin's sorting-model paper. "
        "It summarizes the direct computational measurements produced in S01 through S15, links the report-ready figures and tables, "
        "records unresolved caveats, and defines a regression suite for later Experiments. Claims remain scoped to simulator behavior and computed proxy metrics.",
        "",
        "## Anchor Findings",
        "",
        f"- Sorting completion is exact for the no-Frozen baseline: {summaries['s04Final100Count']} of {summaries['s04ReplicateCount']} S04 runs reached final 100 percent Sortedness.",
        "- Efficiency claims are directionally or statistically replicated for five of six paper-style S06 conclusions; comparison-inclusive Insertion is constraining under the archived independent-test convention.",
        "- Frozen Cell robustness is supportive under deterministic wrapper semantics: cell-view final monotonicity error is no greater than traditional for every f=1..3 passive/stuck algorithm comparison.",
        f"- Same-goal Aggregation remains above fixed-count random-label nulls; the minimum S10 mixed-condition lower bootstrap bound is {summaries['s10MixedPeakMinLowerCi']:.4f}.",
        "- Duplicate-value and opposite-direction chimeras are constraining where the regenerated outputs narrow or contradict the paper's repeated-value claims.",
        f"- S14 scaled selected core families to N=1000 and kept the core conclusions stable; scaled family row counts are {json.dumps(summaries['scaledFamilyCounts'], sort_keys=True)}.",
        "",
        "## Reproducibility Classification",
        "",
        markdown_table(["Classification", "Claim rows"], count_rows),
        "",
        "Not replicated claims: " + ", ".join(summaries["notReplicatedClaims"]) + ".",
        "",
        "## Step Status",
        "",
        markdown_table(["Step", "Status", "Success", "Outcome", "Validation"], status_rows),
        "",
        "## Figure Index",
        "",
        markdown_table(["Paper figure", "Producer", "Title", "Path", "Exists"], figure_rows),
        "",
        "## Table Index",
        "",
        markdown_table(["Producer", "Title", "Path", "Rows", "Columns"], table_rows),
        "",
        "## Scaled Replication",
        "",
        markdown_table(["Family", "Mean S14/baseline CI width ratio"], ci_rows),
        "",
        markdown_table(["Family", "Conditions", "N per condition", "Tasks", "Wall seconds", "Workers"], runtime_rows),
        "",
        "S14 did not run N=10000 because no selected family was clearly cheap as a whole under the 8-worker cap. "
        "The scale step stores compact per-replicate summaries and 101-bin Aggregation curves rather than full raw per-swap traces.",
        "",
        "## Regression Suite",
        "",
        "The packaged regression suite is under `/artifacts/tests/regression_tests/`. It checks status success for S01-S14, baseline config and seed-matrix counts, toy metric formulas, no-Frozen completion, scaled efficiency directions, Frozen Cell robustness anchors, primary DG trends, Aggregation above null, reproducibility classification counts, scaled row counts, and report-bundle reference hashes.",
        "",
        "## Caveats",
        "",
        "- The original OS-thread scheduling and original random seeds are unavailable; this baseline uses S03 deterministic seeds and pseudo-scheduled wrappers around archived cell-view classes.",
        "- Traditional Bubble, Insertion, and Selection use explicit wrappers because S02 found no clean original paper-setting traditional runner.",
        "- Comparison counts, Delayed Gratification, and Aggregation are documented computational proxy metrics with step-specific interpretation notes.",
        "- The exact Figure 8 same-code negative control was not available, so E01 uses fixed-count random-label null baselines for Aggregation.",
        "- Opposite-direction repeated-value dominance and similarity claims are not replicated under this deterministic baseline.",
        "",
        "## Recommended Next Action",
        "",
        "Hand control back to Chief Scientist review. E01 is packaged; start E02 only after explicit instruction in a downstream workflow.",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def write_regression_tests(test_dir: Path) -> Path:
    test_dir.mkdir(parents=True, exist_ok=True)
    (test_dir / "README.md").write_text(
        "# E01 Regression Tests\n\n"
        "Run with:\n\n"
        "```bash\n"
        "ARTIFACTS_DIR=/artifacts python -m unittest discover -s /artifacts/tests/regression_tests -p 'test_*.py' -v\n"
        "```\n\n"
        "These tests validate the packaged E01 baseline artifacts for later Experiments.\n",
        encoding="utf-8",
    )
    test_path = test_dir / "test_e01_baseline_regression.py"
    test_path.write_text(REGRESSION_TEST_CODE, encoding="utf-8")
    return test_path


def run_regression_tests(test_dir: Path, artifacts_dir: Path) -> dict[str, Any]:
    command = [sys.executable, "-m", "unittest", "discover", "-s", str(test_dir), "-p", "test_*.py", "-v"]
    start = time.perf_counter()
    proc = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        env={**os.environ, "ARTIFACTS_DIR": str(artifacts_dir), "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.perf_counter() - start
    log_path = test_dir / "regression_test_log.txt"
    log_path.write_text(proc.stdout, encoding="utf-8")
    result = {
        "command": " ".join(command),
        "returnCode": proc.returncode,
        "success": proc.returncode == 0,
        "wallSeconds": elapsed,
        "logPath": str(log_path),
        "testDirectory": str(test_dir),
    }
    write_json(test_dir / "regression_test_results.json", result)
    return result


def write_bundle_inputs(
    bundle_dir: Path,
    report_path: Path,
    figures: list[dict[str, Any]],
    tables: list[dict[str, Any]],
    reports: list[dict[str, Any]],
    statuses: list[dict[str, Any]],
    artifact_index_path: Path,
    generated_at: str,
) -> list[Path]:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    figures_df = pd.DataFrame(figures)
    tables_df = pd.DataFrame(tables)
    reports_df = pd.DataFrame(reports)
    statuses_df = pd.DataFrame(statuses)
    written = []
    written += write_table(figures_df, bundle_dir / "figures_index.parquet", bundle_dir / "figures_index.csv")
    written += write_table(tables_df, bundle_dir / "tables_index.parquet", bundle_dir / "tables_index.csv")
    written += write_table(reports_df, bundle_dir / "reports_index.parquet", bundle_dir / "reports_index.csv")
    written += write_table(statuses_df, bundle_dir / "step_status_index.parquet", bundle_dir / "step_status_index.csv")
    report_copy = bundle_dir / "replication_report.md"
    shutil.copy2(report_path, report_copy)
    written.append(report_copy)
    readme = bundle_dir / "README.md"
    readme.write_text(
        "# E01 Report Bundle Inputs\n\n"
        "This directory contains indexes and the report draft needed by a later report-bundle workflow. "
        "It does not contain a generated final bundle, archive, or downstream E02 output.\n",
        encoding="utf-8",
    )
    written.append(readme)
    manifest = {
        "schema": "eidosoma.report_bundle_inputs.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "generatedAt": generated_at,
        "reportPath": str(report_path),
        "reportCopyPath": str(report_copy),
        "artifactIndexPath": str(artifact_index_path),
        "figures": figures,
        "tables": tables,
        "reports": reports,
        "stepStatuses": statuses,
        "bundleGenerationStarted": False,
        "downstreamExperimentStarted": False,
    }
    manifest_path = bundle_dir / "bundle_manifest.json"
    write_json(manifest_path, manifest)
    written.append(manifest_path)
    checksum_records = [file_record(Path(item["path"])) for item in figures + tables + reports]
    checksum_records.extend(file_record(path) for path in written if path.exists())
    checksum_path = bundle_dir / "checksums.sha256"
    write_checksum_file(checksum_path, checksum_records)
    written.append(checksum_path)
    return written


def render_validation_md(path: Path, output_paths: list[Path], checks: list[str], caveats: list[str], validation_result: str, recommended_next_action: str) -> None:
    lines = [
        "# S15 Validation",
        "",
        f"- Research step ID: {STEP_ID}",
        f"- Step number: {STEP_NUMBER}",
        f"- Completion status: {STATUS}",
        "- Artifacts written:",
    ]
    lines.extend(f"- `{item}`" for item in sorted(str(path) for path in output_paths))
    lines.extend(
        [
            f"- Validation result: {validation_result}",
            "- Caveats or blockers:",
        ]
    )
    lines.extend(f"- {item}" for item in caveats)
    lines.append(f"- Recommended next action: {recommended_next_action}")
    lines.extend(["", "## Checks", ""])
    lines.extend(f"- {item}" for item in checks)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_summary_md(
    path: Path,
    output_paths: list[Path],
    validation_result: str,
    caveats: list[str],
    recommended_next_action: str,
    summaries: dict[str, Any],
    regression_result: dict[str, Any],
) -> None:
    lines = [
        "# S15 Status Summary",
        "",
        f"- Research step ID: {STEP_ID}",
        f"- Step number: {STEP_NUMBER}",
        f"- Completion status: {STATUS}",
        f"- Outcome classification: {OUTCOME_CLASSIFICATION}",
        "- Artifacts written:",
    ]
    lines.extend(f"- `{item}`" for item in sorted(str(path) for path in output_paths))
    lines.extend(
        [
            f"- Validation result: {validation_result}",
            "- Caveats or blockers:",
        ]
    )
    lines.extend(f"- {item}" for item in caveats)
    lines.extend(
        [
            "- Lay summary: S15 packages the E01 baseline into a report draft, artifact index, report-bundle input indexes, and regression tests. The package validates that referenced figures and tables exist, checks curated checksums, and confirms the regression tests pass.",
            f"- Recommended next action: {recommended_next_action}",
            "",
            "## Classification Counts",
            "",
            markdown_table(["Classification", "Rows"], [[key, value] for key, value in sorted(summaries["classificationCounts"].items())]),
            "",
            "## Regression Tests",
            "",
            markdown_table(
                ["Command", "Return code", "Success", "Wall seconds", "Log"],
                [[regression_result["command"], regression_result["returnCode"], regression_result["success"], regression_result["wallSeconds"], regression_result["logPath"]]],
            ),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_outputs(
    output_paths: list[Path],
    figures: list[dict[str, Any]],
    tables: list[dict[str, Any]],
    reports: list[dict[str, Any]],
    statuses: list[dict[str, Any]],
    regression_result: dict[str, Any],
    artifacts_dir: Path,
) -> tuple[bool, list[str], list[str], str]:
    failures: list[str] = []
    checks: list[str] = []
    caveats = [
        "S15 packages existing S01-S14 evidence and does not rerun the scientific simulations.",
        "Report-bundle inputs are indexes and a report draft only; no final report bundle archive or downstream E02 work was generated.",
        "Regression tests validate stable artifact-level invariants and metric conventions; they are not a substitute for rerunning all expensive simulations.",
        "All E01 deterministic-wrapper, pseudo-scheduler, comparison-count, DG, Aggregation, and unavailable-original-seed caveats carry forward.",
    ]
    source_status_failures = [item for item in statuses if not item["success"] or item["status"] != "completed"]
    if source_status_failures:
        failures.append(f"Source status failures: {source_status_failures}")
    else:
        checks.append("All S01-S14 source status files report success and completed status.")
    missing_refs = [
        item["path"]
        for item in figures + tables + reports
        if not Path(item["path"]).exists() or Path(item["path"]).stat().st_size == 0
    ]
    if missing_refs:
        failures.append(f"Missing or empty report references: {missing_refs}")
    else:
        checks.append("All referenced report figures, tables, and reports exist and are non-empty.")
    if regression_result.get("success"):
        checks.append("Packaged regression tests passed.")
    else:
        failures.append(f"Packaged regression tests failed; see {regression_result.get('logPath')}.")
    missing_outputs = [str(path) for path in output_paths if not path.exists() or path.stat().st_size == 0]
    if missing_outputs:
        failures.append(f"Missing or empty S15 outputs: {missing_outputs}.")
    else:
        checks.append("All declared S15 outputs exist and are non-empty.")
    bundle_dir = artifacts_dir / "report_bundle_inputs"
    if (bundle_dir / "bundle_manifest.json").exists() and not (bundle_dir / "report_bundle.tar.gz").exists() and not (bundle_dir / "report_bundle.zip").exists():
        checks.append("Report-bundle inputs exist, and no final bundle archive was generated.")
    else:
        failures.append("Report-bundle input validation failed or a final bundle archive was generated.")
    e02_paths = [artifacts_dir / "research_steps" / "E02", artifacts_dir / "E02", artifacts_dir / "e02"]
    if any(path.exists() for path in e02_paths):
        failures.append("An E02 artifact path exists; S15 must stop before E02.")
    else:
        checks.append("No E02 artifact directory was created.")
    success = not failures
    validation_result = (
        "passed: S15 packaged the E01 baseline, regression tests passed, referenced figures/tables exist, and no E02 or final report bundle was generated."
        if success
        else "failed: " + " ".join(failures)
    )
    return success, checks + failures, caveats, validation_result


def collect_output_records(paths: list[Path]) -> list[dict[str, Any]]:
    records = []
    seen: set[Path] = set()
    for path in sorted(paths):
        resolved = path.resolve()
        if resolved in seen or not path.exists() or not path.is_file():
            continue
        seen.add(resolved)
        records.append(file_record(path))
    return records


def update_run_manifest(path: Path, status_payload: dict[str, Any], artifact_manifest_payload: dict[str, Any]) -> None:
    manifest = load_json(path)
    manifest.setdefault("experimentId", EXPERIMENT_ID)
    manifest.setdefault("researchSteps", {})
    manifest["updatedAt"] = status_payload["generatedAt"]
    manifest["latestResearchStepId"] = STEP_ID
    manifest["researchSteps"][STEP_ID] = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "status": status_payload["status"],
        "success": status_payload["success"],
        "validationResult": status_payload["validationResult"],
        "recommendedNextAction": status_payload["recommendedNextAction"],
        "artifactCount": artifact_manifest_payload.get("artifactCount"),
        "artifacts": artifact_manifest_payload.get("artifacts", []),
        "generatedAt": status_payload["generatedAt"],
        "git": status_payload.get("git", {}),
    }
    write_json(path, manifest)


def run_mode(args: argparse.Namespace) -> int:
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    code_dir = step_dir / "code"
    reports_dir = artifacts_dir / "reports"
    results_dir = artifacts_dir / "results"
    tests_dir = artifacts_dir / "tests" / "regression_tests"
    bundle_dir = artifacts_dir / "report_bundle_inputs"
    generated_at = utc_now()

    step_dir.mkdir(parents=True, exist_ok=True)
    code_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    statuses = load_statuses(artifacts_dir)
    summaries = source_summaries(artifacts_dir)
    figures = [enrich_reference(item) for item in FIGURE_REFERENCES]
    tables = [enrich_reference(item) for item in TABLE_REFERENCES]
    reports = [enrich_reference(item) for item in REPORT_REFERENCES]

    report_path = reports_dir / "replication_report.md"
    render_replication_report(report_path, statuses, summaries, figures, tables)
    reports.append(enrich_reference({"title": "Replication report", "producerStep": STEP_ID, "path": str(report_path), "role": "final_report"}))

    test_path = write_regression_tests(tests_dir)
    regression_manifest = {
        "schema": "eidosoma.e01_regression_tests.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "generatedAt": generated_at,
        "testPath": str(test_path),
        "expectedRunner": f"{sys.executable} -m unittest discover -s {tests_dir} -p 'test_*.py' -v",
    }
    write_json(tests_dir / "regression_manifest.json", regression_manifest)

    artifact_index_path = results_dir / "e01_artifact_index.parquet"
    bundle_written = write_bundle_inputs(bundle_dir, report_path, figures, tables, reports, statuses, artifact_index_path, generated_at)
    regression_result = run_regression_tests(tests_dir, artifacts_dir)

    excluded_from_index = {
        (results_dir / "e01_artifact_index.parquet").resolve(),
        (results_dir / "e01_artifact_index.csv").resolve(),
        (step_dir / "artifact_manifest.json").resolve(),
    }
    artifact_index = build_artifact_index(artifacts_dir, excluded_from_index)
    artifact_index_csv = results_dir / "e01_artifact_index.csv"
    write_table(artifact_index, artifact_index_path, artifact_index_csv)
    artifact_index_json = results_dir / "e01_artifact_index_summary.json"
    write_json(
        artifact_index_json,
        {
            "schema": "eidosoma.e01_artifact_index_summary.v1",
            "experimentId": EXPERIMENT_ID,
            "researchStepId": STEP_ID,
            "generatedAt": generated_at,
            "artifactCount": int(len(artifact_index)),
            "totalSizeBytes": int(artifact_index["sizeBytes"].sum()) if not artifact_index.empty else 0,
            "countsByCategory": artifact_index.groupby("artifactCategory").size().to_dict() if not artifact_index.empty else {},
            "artifactIndexParquet": str(artifact_index_path),
            "artifactIndexCsv": str(artifact_index_csv),
        },
    )

    code_copy = code_dir / Path(__file__).name
    shutil.copy2(Path(__file__), code_copy)

    output_paths = [
        report_path,
        artifact_index_path,
        artifact_index_csv,
        artifact_index_json,
        tests_dir / "README.md",
        test_path,
        tests_dir / "regression_manifest.json",
        tests_dir / "regression_test_results.json",
        tests_dir / "regression_test_log.txt",
        *bundle_written,
        code_copy,
    ]

    success, validation_checks, caveats, validation_result = validate_outputs(
        output_paths=output_paths,
        figures=figures,
        tables=tables,
        reports=reports,
        statuses=statuses,
        regression_result=regression_result,
        artifacts_dir=artifacts_dir,
    )
    recommended_next_action = "Hand control back to the Chief Scientist workflow for review; E01 is complete and E02 should start only after explicit downstream instruction."

    validation_json = step_dir / "validation.json"
    validation_md = step_dir / "validation.md"
    status_json = step_dir / "status.json"
    summary_md = step_dir / "summary.md"
    methods_md = step_dir / "methods.md"
    checksums_path = step_dir / "checksums.sha256"
    artifact_manifest_json = step_dir / "artifact_manifest.json"

    output_paths.extend([validation_json, validation_md, status_json, summary_md, methods_md, checksums_path, artifact_manifest_json])

    render_validation_md(validation_md, output_paths, validation_checks, caveats, validation_result, recommended_next_action)
    write_json(
        validation_json,
        {
            "experimentId": EXPERIMENT_ID,
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "success": bool(success),
            "status": STATUS if success else "failed_validation",
            "validationResult": validation_result,
            "checks": validation_checks,
            "caveatsOrBlockers": caveats,
            "recommendedNextAction": recommended_next_action,
        },
    )
    render_summary_md(summary_md, output_paths, validation_result, caveats, recommended_next_action, summaries, regression_result)
    methods_md.write_text(
        "# S15 Methods\n\n"
        "S15 packages existing S01-S14 outputs without rerunning simulations. It builds a curated replication report, "
        "a full artifact index with checksums, report-bundle input indexes, and a Python unittest regression suite. "
        "Validation checks source step statuses, referenced figures and tables, regression test success, absence of E02 outputs, "
        "and absence of final report-bundle archives.\n",
        encoding="utf-8",
    )
    artifacts_written = sorted(str(path) for path in output_paths)
    status_payload = {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": bool(success),
        "status": STATUS if success else "failed_validation",
        "generatedAt": generated_at,
        "artifactsWritten": artifacts_written,
        "validationResult": validation_result,
        "caveatsOrBlockers": caveats,
        "recommendedNextAction": recommended_next_action,
        "outcomeClassification": OUTCOME_CLASSIFICATION if success else "constraining/contradictory",
        "git": git_info(),
        "runtime": {
            "pythonVersion": sys.version,
            "pythonExecutable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "osCpuCount": os.cpu_count(),
            "workerCount": 1,
            "gpuUsed": False,
            "numpyVersion": np.__version__,
            "pandasVersion": pd.__version__,
        },
        "sourceStepStatusCount": len(statuses),
        "figureReferenceCount": len(figures),
        "tableReferenceCount": len(tables),
        "reportReferenceCount": len(reports),
        "artifactIndexRows": int(len(artifact_index)),
        "regressionTests": regression_result,
        "classificationCounts": summaries["classificationCounts"],
    }
    write_json(status_json, status_payload)

    checksum_records = collect_output_records([path for path in output_paths if path != checksums_path and path != artifact_manifest_json])
    write_checksum_file(checksums_path, checksum_records)

    manifest_payload = {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": bool(success),
        "status": STATUS if success else "failed_validation",
        "generatedAt": generated_at,
        "artifactsWritten": artifacts_written,
        "artifactCount": 0,
        "artifacts": [],
        "validationResult": validation_result,
        "caveatsOrBlockers": caveats,
        "recommendedNextAction": recommended_next_action,
    }
    output_records = collect_output_records([Path(path) for path in artifacts_written if Path(path) != artifact_manifest_json])
    manifest_payload["artifactCount"] = len(output_records)
    manifest_payload["artifacts"] = output_records
    write_json(artifact_manifest_json, manifest_payload)
    update_run_manifest(RUN_MANIFEST_DEFAULT, status_payload, manifest_payload)

    print(json.dumps({"success": success, "validationResult": validation_result, "artifactIndexRows": len(artifact_index)}, indent=2))
    return 0 if success else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run_mode(parse_args()))
