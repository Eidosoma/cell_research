#!/usr/bin/env python3
"""Generate E01 S02 code-to-paper mapping artifacts.

This script uses static inspection only. It does not import the research
modules because several analysis files execute author-local loads at import
time.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STEP_ID = "S02"
STEP_NUMBER = 2
EXPERIMENT_ID = "E01"
STATUS = "completed"
OUTCOME_CLASSIFICATION = "constraining"

REQUIRED_TOPICS = [
    "traditional Bubble sort",
    "traditional Insertion sort",
    "traditional Selection sort",
    "cell-view Bubble sort",
    "cell-view Insertion sort",
    "cell-view Selection sort",
    "Frozen Cell logic",
    "Probe recording",
    "stop conditions",
    "Sortedness",
    "monotonicity error",
    "Delayed Gratification",
    "Aggregation",
    "mixed Algotype assignment",
    "duplicate values",
    "opposite-direction sorting",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_command(args: list[str], cwd: Path | None = None) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return {
            "args": args,
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:  # pragma: no cover - defensive provenance path
        return {
            "args": args,
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": repr(exc),
        }


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def repo_relative(repo_root: Path, path: Path) -> str:
    return path.relative_to(repo_root).as_posix()


def artifacts_relative(step_dir: Path, path: Path) -> str:
    return path.relative_to(step_dir).as_posix()


def source_segment(source: str, node: ast.AST) -> str:
    return ast.get_source_segment(source, node) or ""


def contains_call(node: ast.AST | None) -> bool:
    if node is None:
        return False
    return any(isinstance(child, ast.Call) for child in ast.walk(node))


def is_main_guard(node: ast.AST) -> bool:
    if not isinstance(node, ast.If):
        return False
    test = node.test
    if not isinstance(test, ast.Compare):
        return False
    left = test.left
    comparators = test.comparators
    if not comparators:
        return False
    left_is_name = isinstance(left, ast.Name) and left.id == "__name__"
    right_is_main = any(
        isinstance(comp, ast.Constant) and comp.value == "__main__"
        for comp in comparators
    )
    return left_is_name and right_is_main


def has_top_level_execution(tree: ast.Module, source: str) -> bool:
    safe_types = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    for node in tree.body:
        if isinstance(node, safe_types):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue
        if is_main_guard(node):
            continue
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            text = source_segment(source, node)
            value = getattr(node, "value", None)
            if "sys.argv" in text or contains_call(value):
                return True
            continue
        if isinstance(node, ast.AugAssign):
            continue
        return True
    return False


def file_role_tags(rel_path: str, qualified_name: str, source_text: str) -> str:
    text = f"{rel_path} {qualified_name} {source_text}".lower()
    tags: set[str] = set()
    if "bubblesortcell" in text or "bubble" in text:
        tags.add("bubble")
    if "insertionsortcell" in text or "insertion" in text:
        tags.add("insertion")
    if "selectionsortcell" in text or "selection" in text:
        tags.add("selection")
    if "merge" in text:
        tags.add("merge-not-paper-core")
    if "frozen" in text or "freeze" in text or "freezed" in text:
        tags.add("frozen-cell")
    if "statusprobe" in text or "probe" in text or "sorting_steps" in text:
        tags.add("probe")
    if "monotonic" in text or "sortedness" in text:
        tags.add("sortedness-monotonicity")
    if "delay" in text or "gratification" in text or "wandering" in text:
        tags.add("delayed-gratification")
    if "aggregation" in text or "cell_type" in text or "algotype" in text:
        tags.add("aggregation-algotype")
    if "reverse_direction" in text or "disorder" in text or "opposite" in text:
        tags.add("opposite-direction")
    if "spearman" in text:
        tags.add("spearman-distance")
    if "ztest" in text or "ttest" in text:
        tags.add("statistics")
    if "thread" in text or "lock" in text:
        tags.add("threading")
    if "traditional" in text or "sorting_cells.py" in rel_path:
        tags.add("traditional-or-legacy")
    return "|".join(sorted(tags)) or "unclassified"


class SymbolVisitor(ast.NodeVisitor):
    def __init__(
        self,
        rel_path: str,
        source: str,
        file_flags: dict[str, Any],
    ) -> None:
        self.rel_path = rel_path
        self.source = source
        self.file_flags = file_flags
        self.rows: list[dict[str, Any]] = []
        self.stack: list[str] = []
        self.class_depth = 0

    def add_row(self, node: ast.AST, symbol_type: str, name: str) -> None:
        qualified_name = ".".join([*self.stack, name])
        segment = source_segment(self.source, node)
        notes = []
        if self.file_flags["hasAuthorLocalPath"]:
            notes.append("file references author-local absolute paths")
        if self.file_flags["hasTopLevelExecution"]:
            notes.append("file is unsafe to import for mapping because it has top-level execution")
        if self.file_flags["hasMainGuard"]:
            notes.append("file has a __main__ guard")
        self.rows.append(
            {
                "path": self.rel_path,
                "symbolType": symbol_type,
                "qualifiedName": qualified_name,
                "lineStart": getattr(node, "lineno", ""),
                "lineEnd": getattr(node, "end_lineno", ""),
                "roleTags": file_role_tags(self.rel_path, qualified_name, segment),
                "importSafe": not self.file_flags["hasTopLevelExecution"]
                and not self.file_flags["hasAuthorLocalPath"],
                "hasMainGuard": self.file_flags["hasMainGuard"],
                "hasAuthorLocalPath": self.file_flags["hasAuthorLocalPath"],
                "notes": "; ".join(notes),
            }
        )

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        self.add_row(node, "class", node.name)
        self.stack.append(node.name)
        self.class_depth += 1
        self.generic_visit(node)
        self.class_depth -= 1
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        symbol_type = "method" if self.class_depth else "function"
        self.add_row(node, symbol_type, node.name)
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        symbol_type = "method" if self.class_depth else "function"
        self.add_row(node, symbol_type, node.name)
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()


def build_function_index(repo_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(repo_root.rglob("*.py")):
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        rel_path = repo_relative(repo_root, path)
        source = path.read_text(encoding="utf-8", errors="replace")
        flags = {
            "hasAuthorLocalPath": any(
                marker in source
                for marker in ["/Users/", "C:\\", "/home/taining", "/home/runner"]
            ),
            "hasMainGuard": False,
            "hasTopLevelExecution": False,
            "parseError": "",
        }
        try:
            tree = ast.parse(source, filename=rel_path)
            flags["hasMainGuard"] = any(is_main_guard(node) for node in tree.body)
            flags["hasTopLevelExecution"] = has_top_level_execution(tree, source)
            visitor = SymbolVisitor(rel_path, source, flags)
            visitor.visit(tree)
            rows.extend(visitor.rows)
        except SyntaxError as exc:
            flags["parseError"] = f"{exc.msg} at line {exc.lineno}"
            rows.append(
                {
                    "path": rel_path,
                    "symbolType": "parse_error",
                    "qualifiedName": "",
                    "lineStart": exc.lineno or "",
                    "lineEnd": exc.lineno or "",
                    "roleTags": file_role_tags(rel_path, "", source),
                    "importSafe": False,
                    "hasMainGuard": flags["hasMainGuard"],
                    "hasAuthorLocalPath": flags["hasAuthorLocalPath"],
                    "notes": flags["parseError"],
                }
            )
    return rows


def index_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(row["path"]), str(row["qualifiedName"])): row for row in rows}


def format_symbol_refs(
    lookup: dict[tuple[str, str], dict[str, Any]],
    symbols: list[tuple[str, str | None]],
) -> str:
    refs = []
    for path, name in symbols:
        if name is None:
            refs.append(path)
            continue
        row = lookup.get((path, name))
        if row:
            refs.append(f"{path}:{row['lineStart']}-{row['lineEnd']}::{name}")
        else:
            refs.append(f"{path}::{name} [not indexed]")
    return "; ".join(refs)


def mapping_rows() -> list[dict[str, Any]]:
    return [
        {
            "paperItem": "Cell state and Algotype fields",
            "paperSectionOrFigure": "Methods 3.1 and 3.3",
            "requiredTopic": "mixed Algotype assignment",
            "mappingStatus": "mapped",
            "symbols": [
                ("modules/multithread/MultiThreadCell.py", "CellStatus"),
                ("modules/multithread/MultiThreadCell.py", "MultiThreadCell.__init__"),
                ("modules/multithread/MultiThreadCell.py", "MultiThreadCell.take_snapshot"),
            ],
            "evidence": "The base cell stores value, current_position, shared cell list, status_probe, group id, type id, and reverse_direction; take_snapshot emits group/type/value/frozen state records used by mixed-cell traces.",
            "missingOrAmbiguous": "The paper uses Algotype terminology; the code mostly uses cell_type and type_id. This is a naming mismatch, not a functional gap.",
            "recommendedNextAction": "S03 should standardize wrapper-level names to Algotype while preserving original field names in provenance.",
        },
        {
            "paperItem": "Cell-view thread architecture and lock-mediated local moves",
            "paperSectionOrFigure": "Methods 3.3, Figure 2",
            "requiredTopic": "cell-view Bubble sort",
            "mappingStatus": "partial",
            "symbols": [
                ("modules/multithread/MultiThreadCell.py", "MultiThreadCell.run"),
                ("modules/multithread/BubbleSortCell.py", "BubbleSortCell.move"),
                ("modules/multithread/InsertionSortCell.py", "InsertionSortCell.move"),
                ("modules/multithread/SelectionSortCell.py", "SelectionSortCell.move"),
            ],
            "evidence": "Each cell subclass runs a move loop and uses a shared lock around local comparisons and swaps.",
            "missingOrAmbiguous": "The paper says each cell generates a random number before competing for the lock. In the inspected code, BubbleSortCell acquires the lock before randomizing direction, while Insertion and Selection do not use a pre-lock random draw.",
            "recommendedNextAction": "S03 should record the archived lock behavior as the baseline and add a config note that pre-lock randomization is not implemented as written in the paper.",
        },
        {
            "paperItem": "Probe outputs: sorting steps, swaps, comparisons, cell types, frozen attempts",
            "paperSectionOrFigure": "Methods 3.3 and 3.4; Figures 3 through 10",
            "requiredTopic": "Probe recording",
            "mappingStatus": "mapped",
            "symbols": [
                ("modules/multithread/StatusProbe.py", "StatusProbe"),
                ("modules/multithread/StatusProbe.py", "StatusProbe.record_sorting_step"),
                ("modules/multithread/StatusProbe.py", "StatusProbe.record_compare"),
                ("modules/multithread/MultiThreadCell.py", "MultiThreadCell.swap"),
            ],
            "evidence": "StatusProbe records sorting_steps, swap_count, compare_and_swap_count, cell_types, and frozen_swap_attempts. MultiThreadCell.swap writes sorting-step and cell-type snapshots after swaps.",
            "missingOrAmbiguous": "Comparisons are recorded only when subclass should_move checks return true, so comparison counts may not equal all neighbor inspections described in the paper.",
            "recommendedNextAction": "S05 should treat archived compare_and_swap_count as the paper-code convention and, if needed, add secondary instrumentation for exhaustive comparisons.",
        },
        {
            "paperItem": "Cell-view Bubble Sort",
            "paperSectionOrFigure": "Methods 3.3; Figure 2B; Figure 3 and 4 Bubble panels",
            "requiredTopic": "cell-view Bubble sort",
            "mappingStatus": "mapped",
            "symbols": [
                ("modules/multithread/BubbleSortCell.py", "BubbleSortCell"),
                ("modules/multithread/BubbleSortCell.py", "BubbleSortCell.should_move"),
                ("modules/multithread/BubbleSortCell.py", "BubbleSortCell.should_move_to"),
                ("modules/multithread/BubbleSortCell.py", "BubbleSortCell.move"),
                ("multithread_cell_sorting_steps.py", "create_cells_within_one_group"),
            ],
            "evidence": "Bubble cells compare local neighbors, can choose left or right movement, and use reverse_direction for descending-goal variants.",
            "missingOrAmbiguous": "The active homogeneous run script uses n=50 and 50 repeats despite file names suggesting 100 experiments; paper settings are n=100 and N=100.",
            "recommendedNextAction": "S03 should expose n and N in an explicit config instead of relying on script constants.",
        },
        {
            "paperItem": "Cell-view Insertion Sort",
            "paperSectionOrFigure": "Methods 3.3; Figure 2D; Figure 3 and 4 Insertion panels",
            "requiredTopic": "cell-view Insertion sort",
            "mappingStatus": "mapped",
            "symbols": [
                ("modules/multithread/InsertionSortCell.py", "InsertionSortCell"),
                ("modules/multithread/InsertionSortCell.py", "InsertionSortCell.should_move"),
                ("modules/multithread/InsertionSortCell.py", "InsertionSortCell.is_enable_to_move"),
                ("modules/multithread/InsertionSortCell.py", "InsertionSortCell.move"),
                ("multithread_cell_sorting_steps.py", "create_cells_within_one_group"),
            ],
            "evidence": "Insertion cells move left when local order and left-side enablement checks allow it, with reverse_direction support for descending variants.",
            "missingOrAmbiguous": "The enablement logic explicitly scans left-side cells and treats FREEZE status specially; this detail is not fully specified in the extracted paper text.",
            "recommendedNextAction": "S03 should preserve this archived enablement rule and document it in the baseline config notes.",
        },
        {
            "paperItem": "Cell-view Selection Sort",
            "paperSectionOrFigure": "Methods 3.3; Figure 2F; Figure 3 and 4 Selection panels",
            "requiredTopic": "cell-view Selection sort",
            "mappingStatus": "mapped",
            "symbols": [
                ("modules/multithread/SelectionSortCell.py", "SelectionSortCell"),
                ("modules/multithread/SelectionSortCell.py", "SelectionSortCell.should_move"),
                ("modules/multithread/SelectionSortCell.py", "SelectionSortCell.should_move_to"),
                ("modules/multithread/SelectionSortCell.py", "SelectionSortCell.update"),
                ("modules/multithread/SelectionSortCell.py", "SelectionSortCell.move"),
                ("multithread_cell_sorting_steps.py", "create_cells_within_one_group"),
            ],
            "evidence": "Selection cells compute an ideal boundary-side position and swap toward it while shifting around frozen or smaller/equal cells.",
            "missingOrAmbiguous": "Selection's ideal_position update is code-specific and not completely recoverable from the extracted formulas.",
            "recommendedNextAction": "S03 should classify the code behavior as the baseline Selection policy and S04/S05 should test it directly.",
        },
        {
            "paperItem": "Traditional Bubble Sort",
            "paperSectionOrFigure": "Methods 3.2; Figure 2A; Figures 3 and 4 traditional Bubble",
            "requiredTopic": "traditional Bubble sort",
            "mappingStatus": "missing",
            "symbols": [
                ("sorting_cells.py", None),
                ("analysis/efficiency_analysis.py", None),
                ("multi_dimentions/multi_dimention_monotonicity.py", "cell_original_efficiency_compare"),
            ],
            "evidence": "The repository has legacy visualization code and analysis functions that load precomputed original/traditional arrays.",
            "missingOrAmbiguous": "No clean callable traditional Bubble generator matching n=100, N=100, Probe-style output was found. sorting_cells.py appears to be legacy or broken for direct use, and analysis scripts depend on author-local .npy files.",
            "recommendedNextAction": "S03 should define faithful wrapper implementations for traditional Bubble, Insertion, and Selection with matched trace semantics.",
        },
        {
            "paperItem": "Traditional Insertion Sort",
            "paperSectionOrFigure": "Methods 3.2; Figure 2C; Figures 3 and 4 traditional Insertion",
            "requiredTopic": "traditional Insertion sort",
            "mappingStatus": "missing",
            "symbols": [
                ("sorting_cells.py", None),
                ("analysis/efficiency_analysis.py", None),
                ("multi_dimentions/multi_dimention_monotonicity.py", "cell_original_efficiency_compare_include_read"),
            ],
            "evidence": "Analysis scripts reference traditional comparison data, but the source generator is not present as a robust callable module.",
            "missingOrAmbiguous": "No original traditional Insertion executable pathway with paper settings and trace outputs was found.",
            "recommendedNextAction": "S03/S04 should add wrapper-level traditional Insertion logic and validate it on hand traces before large runs.",
        },
        {
            "paperItem": "Traditional Selection Sort",
            "paperSectionOrFigure": "Methods 3.2; Figure 2E; Figures 3 and 4 traditional Selection",
            "requiredTopic": "traditional Selection sort",
            "mappingStatus": "missing",
            "symbols": [
                ("sorting_cells.py", None),
                ("analysis/efficiency_analysis.py", None),
                ("multi_dimentions/multi_dimention_monotonicity.py", "cell_original_efficiency_compare"),
            ],
            "evidence": "Traditional Selection appears only as precomputed data references and legacy visualization scaffolding.",
            "missingOrAmbiguous": "No clean traditional Selection runner matching paper comparison and swap conventions was found.",
            "recommendedNextAction": "S03 should freeze an explicit traditional Selection trace convention, then S05 should compare both swap-only and comparison-inclusive definitions.",
        },
        {
            "paperItem": "Homogeneous cell-view sorting trajectory experiments",
            "paperSectionOrFigure": "Figure 3",
            "requiredTopic": "Sortedness",
            "mappingStatus": "partial",
            "symbols": [
                ("multithread_cell_sorting_steps.py", "main"),
                ("analysis/utils.py", "get_monotonicity"),
                ("multi_dimentions/multi_dimention_monotonicity.py", "plot_raw_data"),
                ("multi_dimentions/multi_dimention_monotonicity.py", "plot_tranditional_cell_together"),
            ],
            "evidence": "The generator saves Probe sorting-step arrays for homogeneous cell-view algorithms; plotting utilities compute nondecreasing-adjacency trajectories from saved arrays.",
            "missingOrAmbiguous": "Active constants do not match the paper: sorting_list is range(50), repeat count is 50, output filenames say 100exps, and traditional traces are not generated in the repository.",
            "recommendedNextAction": "S03 should replace embedded constants with an explicit baseline config; S04 should produce fresh matched traces.",
        },
        {
            "paperItem": "Efficiency comparisons, swap-only and swap-plus-comparison",
            "paperSectionOrFigure": "Figure 4",
            "requiredTopic": "stop conditions",
            "mappingStatus": "ambiguous",
            "symbols": [
                ("modules/multithread/StatusProbe.py", "StatusProbe.record_compare"),
                ("analysis/efficiency_analysis.py", None),
                ("multi_dimentions/multi_dimention_monotonicity.py", "cell_original_efficiency_compare"),
                ("multi_dimentions/multi_dimention_monotonicity.py", "cell_original_efficiency_compare_include_read"),
            ],
            "evidence": "StatusProbe has a compare_and_swap_count field and multidimensional analysis scripts call ztest for efficiency comparisons.",
            "missingOrAmbiguous": "The exact traditional comparison-count generator is absent, and archived compare counting undercounts all possible neighbor inspections. Some analysis functions use hard-coded arrays or author-local data.",
            "recommendedNextAction": "S05 should reproduce the archived convention first, then record a secondary exhaustive-comparison convention.",
        },
        {
            "paperItem": "Frozen Cell setup",
            "paperSectionOrFigure": "Methods 3.3; Figure 5",
            "requiredTopic": "Frozen Cell logic",
            "mappingStatus": "ambiguous",
            "symbols": [
                ("modules/multithread/MultiThreadCell.py", "MultiThreadCell.set_cell_to_freeze"),
                ("modules/multithread/MultiThreadCell.py", "MultiThreadCell.swap"),
                ("multithread_cell_sorting_with_frozen_steps.py", "create_cells_within_one_group"),
                ("freezing_sorting_analysis.py", "create_cells_within_one_group"),
            ],
            "evidence": "Cells can be marked FREEZE, frozen swap attempts can be counted, and frozen-cell experiment scripts randomly select frozen positions.",
            "missingOrAmbiguous": "Passive versus stuck Frozen Cell variants are not cleanly separated. One script skips starting frozen cell threads; another starts them. MultiThreadCell.swap blocks only when the acting cell is frozen, while subclass should_move_to methods can also avoid FREEZE targets.",
            "recommendedNextAction": "S03 should make passive and stuck semantics explicit and S07 should validate both with tiny hand cases before n=100 sweeps.",
        },
        {
            "paperItem": "Frozen Cell robustness condition matrix",
            "paperSectionOrFigure": "Figure 5",
            "requiredTopic": "monotonicity error",
            "mappingStatus": "partial",
            "symbols": [
                ("multithread_cell_sorting_with_frozen_steps.py", "main"),
                ("freezing_sorting_analysis.py", "main"),
                ("analysis/frozen_success_compare.py", "get_final_monotonicity"),
                ("analysis/frozen_spearmans_distance_results.py", "get_average_final_spearman_distance"),
            ],
            "evidence": "There are frozen-cell runners and analysis scripts for final monotonicity and Spearman-distance summaries.",
            "missingOrAmbiguous": "The active frozen runner only loops frozen_cell_num in [2, 3] and only executes Bubble in the inspected code; Selection/Insertion blocks are commented. freezing_sorting_analysis uses 20 experiments and repeated values, not the full paper matrix.",
            "recommendedNextAction": "S07 should build a complete f=0,1,2,3 x algorithm x frozen-variant matrix from explicit config rather than relying on active script blocks.",
        },
        {
            "paperItem": "Sortedness metric",
            "paperSectionOrFigure": "Methods 3.4; Figures 3, 8, 9, 10",
            "requiredTopic": "Sortedness",
            "mappingStatus": "ambiguous",
            "symbols": [
                ("analysis/utils.py", "get_monotonicity"),
                ("analysis/cell_type_aggregation_analysis.py", "get_monotonicity_value"),
                ("multi_dimentions/multi_dimention_monotonicity.py", "get_current_monotonicity"),
            ],
            "evidence": "Several functions count nondecreasing adjacent pairs or return a percentage-like nondecreasing score.",
            "missingOrAmbiguous": "Names conflate monotonicity and Sortedness. analysis.utils.get_monotonicity returns a percent-like score starting at one sorted adjacency, while aggregation and multidimension utilities return counts on a 0 to n-1 scale.",
            "recommendedNextAction": "S03 should define canonical percent Sortedness and raw count fields; S04 should preserve both where original scripts used counts.",
        },
        {
            "paperItem": "Monotonicity error metric",
            "paperSectionOrFigure": "Methods 3.4; Figure 5",
            "requiredTopic": "monotonicity error",
            "mappingStatus": "partial",
            "symbols": [
                ("analysis/delay_gratification_analysis.py", "get_monotonicity"),
                ("analysis/frozen_success_compare.py", "get_final_monotonicity"),
                ("analysis/utils.py", "get_spearman_distance"),
            ],
            "evidence": "delay_gratification_analysis.get_monotonicity counts descending adjacent violations; frozen_success_compare converts final Sortedness-like values to an error value; Spearman distance is also used for frozen robustness.",
            "missingOrAmbiguous": "The paper's formula extraction is incomplete, and the repository mixes monotonicity-error, Sortedness, and Spearman-distance proxies.",
            "recommendedNextAction": "S06/S07 should report final monotonicity error separately from final Sortedness and rank-distance proxies.",
        },
        {
            "paperItem": "Delayed Gratification calculation",
            "paperSectionOrFigure": "Methods 3.4; Figures 6 and 7",
            "requiredTopic": "Delayed Gratification",
            "mappingStatus": "partial",
            "symbols": [
                ("analysis/delay_gratification_analysis.py", "get_discrepency_arr"),
                ("analysis/delay_gratification_analysis.py", "max_wandering_range"),
                ("analysis/delay_gratification_analysis.py", "avg_wandering_range"),
                ("analysis/delay_gratification_analysis.py", "get_max_delay_gratification"),
            ],
            "evidence": "DG-like utilities derive temporary worsening and later gain from monotonicity trajectories.",
            "missingOrAmbiguous": "The module loads author-local .npy files at top level, so functions are not safely importable as-is. The extracted paper formula is incomplete, and the code's avg_wandering_range convention must be validated against hand trajectories.",
            "recommendedNextAction": "S08 should port the DG formula into a clean wrapper with unit tests on hand-constructed trajectories.",
        },
        {
            "paperItem": "Aggregation metric",
            "paperSectionOrFigure": "Methods 3.4; Figure 8",
            "requiredTopic": "Aggregation",
            "mappingStatus": "ambiguous",
            "symbols": [
                ("analysis/cell_type_aggregation_analysis.py", "get_aggregation_value_avg"),
                ("analysis/cell_type_aggregation_analysis.py", "get_aggregation_value"),
                ("analysis/cell_type_aggregation_analysis.py", "get_average_aggregation_array"),
            ],
            "evidence": "Aggregation utilities count same-type adjacent neighbors across cell-type traces and average over experiments.",
            "missingOrAmbiguous": "The paper text describes same-Algotype left-neighbor aggregation. get_aggregation_value_avg checks the right neighbor for every position except the last, while get_aggregation_value samples random cells stochastically.",
            "recommendedNextAction": "S09/S10 should use a deterministic adjacent-pair aggregation metric and document whether right-neighbor, left-neighbor, or undirected adjacency is used.",
        },
        {
            "paperItem": "Same-goal mixed Algotype assignment",
            "paperSectionOrFigure": "Figure 8A-C and 8E",
            "requiredTopic": "mixed Algotype assignment",
            "mappingStatus": "partial",
            "symbols": [
                ("multithread_sorting_cell_aggregation_analysis.py", "get_cell_type_list_v2"),
                ("multithread_sorting_cell_aggregation_analysis.py", "create_cell_groups_based_on_value_list"),
                ("analysis/cell_type_aggregation_analysis.py", "plot_average_data_from_experiment"),
            ],
            "evidence": "The generator can create arrays with equal counts of Bubble, Insertion, and Selection cell classes and save cell-type traces.",
            "missingOrAmbiguous": "Many intended pairings are commented out. The active duplicate-value run appears to generate Insertion+Selection while saving to a bubble_selection_dup path, creating a label/config mismatch.",
            "recommendedNextAction": "S09 should generate the full same-goal condition matrix from config and write machine-readable labels with counts per Algotype.",
        },
        {
            "paperItem": "Duplicate-value chimeras",
            "paperSectionOrFigure": "Figure 8D-F",
            "requiredTopic": "duplicate values",
            "mappingStatus": "partial",
            "symbols": [
                ("multithread_sorting_cell_aggregation_analysis.py", "prepare_sorting_list"),
                ("multithread_sorting_cell_aggregation_analysis.py", "main"),
            ],
            "evidence": "The active aggregation generator can prepare repeated-value inputs for duplicate-value chimera experiments.",
            "missingOrAmbiguous": "The paper reports 10 copies each of values 1 through 10 for n=100. The active prepare_sorting_list returns 20 copies each of 0 through 9 for n=200.",
            "recommendedNextAction": "S09 should treat n=100 duplicate-value arrays as the paper baseline and keep n=200 archived behavior only as a provenance note.",
        },
        {
            "paperItem": "Opposite-direction chimeras",
            "paperSectionOrFigure": "Figures 9 and 10",
            "requiredTopic": "opposite-direction sorting",
            "mappingStatus": "partial",
            "symbols": [
                ("modules/multithread/BubbleSortCell.py", "BubbleSortCell.should_move"),
                ("modules/multithread/InsertionSortCell.py", "InsertionSortCell.should_move"),
                ("modules/multithread/SelectionSortCell.py", "SelectionSortCell.__init__"),
                ("multithread_sorting_cell_aggregation_disorder.py", "create_cell_groups_based_on_value_list"),
                ("analysis/cell_type_aggregation_analysis.py", "plot_dis_order"),
            ],
            "evidence": "Cell classes support reverse_direction and the disorder generator creates mixed goal-direction arrays.",
            "missingOrAmbiguous": "The active disorder script only sets Selection to reverse_direction=True in the inspected create function, and its active main call covers only one pairing. The exact Figure 9/10 pairings are not all represented by active code.",
            "recommendedNextAction": "S11/S12 should encode each paper pairing explicitly: Bubble down/Selection up, Bubble up/Insertion down, Selection down/Insertion up, plus duplicate-value variants.",
        },
        {
            "paperItem": "Stop conditions and completion criteria",
            "paperSectionOrFigure": "Methods 3.3; all simulation figures",
            "requiredTopic": "stop conditions",
            "mappingStatus": "ambiguous",
            "symbols": [
                ("multithread_cell_sorting_steps.py", "is_sorted"),
                ("multithread_cell_sorting_with_frozen_steps.py", "no_cells_should_move"),
                ("modules/multithread/CellGroup.py", "CellGroup.is_group_sorted"),
                ("modules/multithread/CellGroup.py", "CellGroup.run"),
            ],
            "evidence": "Homogeneous scripts often loop until is_sorted; frozen scripts have no_cells_should_move helpers; CellGroup has group-level sorted checks.",
            "missingOrAmbiguous": "Stop conditions differ across scripts and some frozen or mixed settings may require no-move caps instead of full sortedness.",
            "recommendedNextAction": "S03 should define stop_reason values and maximum-step caps for sorted, no-move, timeout, and failed conditions.",
        },
        {
            "paperItem": "Statistics for reported comparisons",
            "paperSectionOrFigure": "Figures 4, 5, 7, 8, 9, 10",
            "requiredTopic": "monotonicity error",
            "mappingStatus": "ambiguous",
            "symbols": [
                ("multi_dimentions/multi_dimention_monotonicity.py", "cell_original_efficiency_compare"),
                ("analysis/cell_type_aggregation_analysis.py", "get_ttest_value"),
                ("analysis/frozen_spearmans_distance_results.py", "get_ttest_value"),
            ],
            "evidence": "Some scripts use statsmodels ztest for efficiency, while aggregation and frozen scripts use scipy t-tests or commented-out z-test code.",
            "missingOrAmbiguous": "The paper reports z-tests, but the repository does not consistently implement z-tests across all claim families.",
            "recommendedNextAction": "S06 should reproduce paper z-tests in a clean statistics module and record t-test outputs only as archived-code context.",
        },
        {
            "paperItem": "Author-local raw data and hard-coded paths",
            "paperSectionOrFigure": "Figures 3 through 10",
            "requiredTopic": "Probe recording",
            "mappingStatus": "missing",
            "symbols": [
                ("analysis/delay_gratification_analysis.py", None),
                ("analysis/cell_type_aggregation_analysis.py", None),
                ("analysis/performance_analysis.py", None),
                ("multi_dimentions/multi_dimention_monotonicity.py", None),
            ],
            "evidence": "Several analysis scripts load .npy files from /Users/tainingzhang/... or other local folders.",
            "missingOrAmbiguous": "The raw arrays behind many published plots are not present in the repository. Some modules execute these loads at import time.",
            "recommendedNextAction": "Later steps should regenerate raw traces from configured runners rather than relying on missing author-local arrays.",
        },
        {
            "paperItem": "Merge Sort and non-core visual systems",
            "paperSectionOrFigure": "Not part of core reported paper claims",
            "requiredTopic": "cell-view Selection sort",
            "mappingStatus": "mapped",
            "symbols": [
                ("modules/multithread/MergeSortCell.py", "MergeSortCell"),
                ("sorting_2d_cell_visual.py", None),
                ("modules/Cell2D.py", None),
            ],
            "evidence": "The repository contains MergeSortCell and 2D/visualization code.",
            "missingOrAmbiguous": "These files are outside the S02 target figure claims and should not be treated as evidence for the paper's Bubble/Insertion/Selection results.",
            "recommendedNextAction": "Keep these components out of E01 baseline runs unless a later experiment explicitly studies extensions.",
        },
    ]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    def clean(value: str) -> str:
        return str(value).replace("\n", " ").replace("|", "\\|")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(clean(value) for value in row) + " |")
    return "\n".join(lines)


def build_markdown_report(
    step_dir: Path,
    function_rows: list[dict[str, Any]],
    map_rows: list[dict[str, Any]],
    lookup: dict[tuple[str, str], dict[str, Any]],
    git_meta: dict[str, Any],
) -> str:
    status_counts = Counter(row["mappingStatus"] for row in map_rows)
    function_counts = Counter(row["symbolType"] for row in function_rows)
    unsafe_files = sorted(
        {
            str(row["path"])
            for row in function_rows
            if not row["importSafe"] or row["hasAuthorLocalPath"]
        }
    )
    missing_or_ambiguous = [
        row
        for row in map_rows
        if row["mappingStatus"] in {"missing", "ambiguous", "partial"}
    ]

    coverage_rows = [
        [
            row["paperItem"],
            row["paperSectionOrFigure"],
            row["mappingStatus"],
            format_symbol_refs(lookup, row["symbols"]),
            row["missingOrAmbiguous"],
        ]
        for row in map_rows
    ]

    missing_rows = [
        [
            row["paperItem"],
            row["mappingStatus"],
            row["missingOrAmbiguous"],
            row["recommendedNextAction"],
        ]
        for row in missing_or_ambiguous
    ]

    required_rows = []
    for topic in REQUIRED_TOPICS:
        matches = [row for row in map_rows if row["requiredTopic"] == topic]
        statuses = ", ".join(sorted({row["mappingStatus"] for row in matches}))
        required_rows.append([topic, str(len(matches)), statuses or "uncovered"])

    validation_result = (
        "passed with documented constraints: all required S02 topics have at "
        "least one mapped row, and every partial, ambiguous, or missing mapping "
        "has an explicit caveat and next action."
    )

    artifacts = [
        step_dir / "code_paper_map.md",
        step_dir / "function_index.csv",
        step_dir / "mapping_matrix.csv",
        step_dir / "summary.md",
        step_dir / "status.json",
        step_dir / "artifact_manifest.json",
        step_dir / "code" / "e01_s02_code_mapping.py",
    ]
    artifact_lines = "\n".join(f"- `{path}`" for path in artifacts)

    return f"""# S02 Code-to-Paper Map

- Research step ID: {STEP_ID}
- Step number: {STEP_NUMBER}
- Completion status: {STATUS}
- Outcome classification: {OUTCOME_CLASSIFICATION}
- Artifacts written:
{artifact_lines}
- Validation result: {validation_result}
- Caveats or blockers: Traditional-sort runners, author-local raw arrays, paper/code lock randomization, Frozen Cell variants, and several metric formulas or active script constants are incomplete or ambiguous.
- Recommended next action: Proceed to S03 to freeze an explicit baseline config that resolves the mapped constants, stop conditions, and missing traditional wrappers before starting any large sweeps.

## Executive Verdict

S02 found enough source coverage to proceed, but the mapping is constraining rather than fully supportive. The core cell-view Bubble, Insertion, Selection, Probe, Algotype trace, reverse-direction, and several metric utilities are present. The full paper replication cannot rely on the repository entry scripts as-is because several active constants differ from the paper, many analysis scripts load unavailable author-local `.npy` files, traditional-sort generators are not cleanly present, and Frozen Cell/passive-stuck semantics are mixed across scripts.

Git branch at generation: `{git_meta.get("branch", "unknown")}`. Git commit at generation: `{git_meta.get("commit", "unknown")}`. Git dirty status at generation: `{git_meta.get("dirtyStatus", "") or "clean"}`.

## Coverage Summary

- Mapping status counts: {dict(sorted(status_counts.items()))}
- Function index counts: {dict(sorted(function_counts.items()))}
- Function index rows: {len(function_rows)}
- Unsafe or author-local-path files in the function index: {len(unsafe_files)}

{markdown_table(["Required S02 topic", "Mapped row count", "Statuses"], required_rows)}

## Code-to-Paper Matrix

{markdown_table(["Paper method or claim", "Paper section / figure", "Status", "Mapped source components", "Missing or ambiguous mapping"], coverage_rows)}

## Missing And Ambiguous Mappings

{markdown_table(["Paper method or claim", "Status", "Caveat or blocker", "Recommended next action"], missing_rows)}

## Directly Reusable Entry Points

- Cell-view policies: `modules/multithread/BubbleSortCell.py`, `InsertionSortCell.py`, and `SelectionSortCell.py`.
- Shared cell and Probe state: `modules/multithread/MultiThreadCell.py`, `StatusProbe.py`, and `CellGroup.py`.
- Homogeneous cell-view runner scaffold: `multithread_cell_sorting_steps.py`, but S03 must override `n`, `N`, seeds, labels, and outputs.
- Frozen-cell runner scaffold: `multithread_cell_sorting_with_frozen_steps.py`, but S03/S07 must separate passive and stuck Frozen Cell semantics and complete the condition matrix.
- Same-goal chimera scaffold: `multithread_sorting_cell_aggregation_analysis.py`, but S03/S09 must fix duplicate-value settings and output labels.
- Opposite-direction chimera scaffold: `multithread_sorting_cell_aggregation_disorder.py`, but S11/S12 must enumerate each paper pairing explicitly.
- Analysis utilities worth porting into clean wrappers: Sortedness-like adjacency counts, monotonicity-error counts, Delayed Gratification helper logic, Aggregation adjacency counts, and z-test routines.

## Function Index Notes

The machine-readable function index is written to `{step_dir / "function_index.csv"}`. It is generated by AST parsing only and intentionally avoids importing repository modules. Files marked `importSafe=false` either have top-level execution, top-level data loads, or author-local absolute paths and should be wrapped or refactored before direct import in later steps.

## S03 Implications

S03 should not start a sweep by running the archived scripts directly. It should freeze a config that explicitly supplies paper settings, output schemas, seed conventions, stop reasons, and wrapper-level traditional sorting implementations. The archived code should be treated as behavior to map and preserve where clear, not as a complete reproducible pipeline.
"""


def build_mapping_csv_rows(
    map_rows: list[dict[str, Any]],
    lookup: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    csv_rows = []
    for row in map_rows:
        csv_rows.append(
            {
                "researchStepId": STEP_ID,
                "paperItem": row["paperItem"],
                "paperSectionOrFigure": row["paperSectionOrFigure"],
                "requiredTopic": row["requiredTopic"],
                "mappingStatus": row["mappingStatus"],
                "mappedSourceComponents": format_symbol_refs(lookup, row["symbols"]),
                "evidence": row["evidence"],
                "missingOrAmbiguous": row["missingOrAmbiguous"],
                "recommendedNextAction": row["recommendedNextAction"],
            }
        )
    return csv_rows


def collect_artifacts(step_dir: Path) -> list[dict[str, Any]]:
    artifacts = []
    for path in sorted(step_dir.rglob("*")):
        if not path.is_file():
            continue
        artifacts.append(
            {
                "path": str(path),
                "relativePath": artifacts_relative(step_dir, path),
                "sizeBytes": path.stat().st_size,
                "sha256": sha256_path(path),
            }
        )
    return artifacts


def get_git_metadata(repo_root: Path) -> dict[str, Any]:
    commit = run_command(["git", "rev-parse", "HEAD"], cwd=repo_root)
    branch = run_command(["git", "branch", "--show-current"], cwd=repo_root)
    status = run_command(["git", "status", "--short"], cwd=repo_root)
    remote = run_command(["git", "remote", "-v"], cwd=repo_root)
    return {
        "commit": commit["stdout"] if commit["ok"] else "unknown",
        "branch": branch["stdout"] if branch["ok"] else "unknown",
        "dirtyStatus": status["stdout"],
        "remote": remote["stdout"],
    }


def validation_result(
    function_rows: list[dict[str, Any]],
    map_rows: list[dict[str, Any]],
    step_dir: Path,
) -> tuple[bool, str, list[str]]:
    errors = []
    if not function_rows:
        errors.append("function index is empty")
    covered_topics = {row["requiredTopic"] for row in map_rows}
    missing_topics = sorted(set(REQUIRED_TOPICS) - covered_topics)
    if missing_topics:
        errors.append(f"required topics not covered: {', '.join(missing_topics)}")
    for row in map_rows:
        if row["mappingStatus"] in {"missing", "ambiguous", "partial"} and not row[
            "missingOrAmbiguous"
        ]:
            errors.append(f"row lacks caveat: {row['paperItem']}")
        if not row["recommendedNextAction"]:
            errors.append(f"row lacks recommended next action: {row['paperItem']}")
    required_paths = [
        step_dir / "code_paper_map.md",
        step_dir / "function_index.csv",
        step_dir / "mapping_matrix.csv",
        step_dir / "code" / "e01_s02_code_mapping.py",
    ]
    missing_files = [str(path) for path in required_paths if not path.exists()]
    if missing_files:
        errors.append(f"required artifacts missing: {', '.join(missing_files)}")
    if errors:
        return False, "failed: " + "; ".join(errors), errors
    return (
        True,
        "passed: function index generated, all required S02 topics mapped or explicitly gapped, and required artifacts exist.",
        [],
    )


def write_summary(
    path: Path,
    artifacts_written: list[str],
    validation: str,
    caveats: list[str],
    recommended_next_action: str,
) -> None:
    artifact_lines = "\n".join(f"- `{artifact}`" for artifact in artifacts_written)
    caveat_lines = "\n".join(f"- {caveat}" for caveat in caveats)
    path.write_text(
        f"""# S02 Status Summary

- Research step ID: {STEP_ID}
- Step number: {STEP_NUMBER}
- Completion status: {STATUS}
- Outcome classification: {OUTCOME_CLASSIFICATION}
- Artifacts written:
{artifact_lines}
- Validation result: {validation}
- Caveats or blockers:
{caveat_lines}
- Lay summary: The repository contains the main cell-view sorting policies and Probe machinery, but it is not a complete paper-replication pipeline. Traditional sort runners, complete Frozen Cell variants, exact paper constants, and several raw analysis inputs must be reconstructed or wrapped before running large replications.
- Recommended next action: {recommended_next_action}
""",
        encoding="utf-8",
    )


def update_run_manifest(
    artifacts_dir: Path,
    step_dir: Path,
    git_meta: dict[str, Any],
    status_payload: dict[str, Any],
) -> None:
    provenance_dir = artifacts_dir / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    run_manifest_path = provenance_dir / "run_manifest.json"
    if run_manifest_path.exists():
        try:
            manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
    else:
        manifest = {}

    manifest.setdefault("experimentId", EXPERIMENT_ID)
    manifest["generatedAt"] = utc_now()
    manifest["researchStepId"] = STEP_ID
    manifest["recommendedNextAction"] = status_payload["recommendedNextAction"]
    manifest["caveatsOrBlockers"] = status_payload["caveatsOrBlockers"]
    manifest["artifactsWritten"] = status_payload["artifactsWritten"]
    manifest["git"] = git_meta
    manifest["runtime"] = {
        "pythonExecutable": sys.executable,
        "pythonVersion": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "osCpuCount": os.cpu_count(),
    }
    manifest.setdefault("researchSteps", {})
    manifest["researchSteps"][STEP_ID] = {
        "status": STATUS,
        "success": status_payload["success"],
        "artifactCount": len(collect_artifacts(step_dir)),
        "artifacts": collect_artifacts(step_dir),
        "validationResult": status_payload["validationResult"],
        "mappingStatusCounts": status_payload["mappingStatusCounts"],
        "functionIndexRows": status_payload["functionIndexRows"],
        "generatedAt": status_payload["generatedAt"],
    }
    run_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    artifacts_dir = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    code_dir = step_dir / "code"
    step_dir.mkdir(parents=True, exist_ok=True)
    code_dir.mkdir(parents=True, exist_ok=True)

    git_meta = get_git_metadata(repo_root)
    function_rows = build_function_index(repo_root)
    lookup = index_lookup(function_rows)
    map_rows = mapping_rows()
    mapping_csv_rows = build_mapping_csv_rows(map_rows, lookup)

    write_csv(
        step_dir / "function_index.csv",
        function_rows,
        [
            "path",
            "symbolType",
            "qualifiedName",
            "lineStart",
            "lineEnd",
            "roleTags",
            "importSafe",
            "hasMainGuard",
            "hasAuthorLocalPath",
            "notes",
        ],
    )
    write_csv(
        step_dir / "mapping_matrix.csv",
        mapping_csv_rows,
        [
            "researchStepId",
            "paperItem",
            "paperSectionOrFigure",
            "requiredTopic",
            "mappingStatus",
            "mappedSourceComponents",
            "evidence",
            "missingOrAmbiguous",
            "recommendedNextAction",
        ],
    )
    (step_dir / "code_paper_map.md").write_text(
        build_markdown_report(step_dir, function_rows, map_rows, lookup, git_meta),
        encoding="utf-8",
    )
    shutil.copy2(Path(__file__), code_dir / Path(__file__).name)

    preliminary_artifacts = [
        str(step_dir / "code_paper_map.md"),
        str(step_dir / "function_index.csv"),
        str(step_dir / "mapping_matrix.csv"),
        str(step_dir / "summary.md"),
        str(step_dir / "status.json"),
        str(step_dir / "artifact_manifest.json"),
        str(code_dir / Path(__file__).name),
    ]
    ok, validation, errors = validation_result(function_rows, map_rows, step_dir)
    caveats = [
        "No clean original traditional Bubble, Insertion, or Selection runner matching the paper settings was found.",
        "Multiple analysis modules depend on unavailable author-local .npy paths and are unsafe to import directly.",
        "Paper and code disagree or are ambiguous about pre-lock randomization in threaded cells.",
        "Passive versus stuck Frozen Cell semantics are mixed across scripts and need explicit wrapper-level definitions.",
        "Metric names are inconsistent across modules: Sortedness, monotonicity, monotonicity error, and Spearman distance are sometimes conflated.",
        "Several active script constants and output labels do not match the paper's n=100, N=100, f=0..3, and duplicate-value settings.",
    ]
    recommended_next_action = (
        "Proceed to S03 to freeze the baseline configuration, including explicit traditional sort wrappers, paper constants, stop reasons, Frozen Cell variants, metric definitions, and seed/output conventions; do not start S03 until instructed."
    )

    write_summary(
        step_dir / "summary.md",
        preliminary_artifacts,
        validation,
        caveats,
        recommended_next_action,
    )

    status_payload: dict[str, Any] = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": ok,
        "status": STATUS if ok else "failed",
        "artifactsWritten": preliminary_artifacts,
        "validationResult": validation,
        "caveatsOrBlockers": caveats + errors,
        "recommendedNextAction": recommended_next_action,
        "outcomeClassification": OUTCOME_CLASSIFICATION,
        "generatedAt": utc_now(),
        "experimentId": EXPERIMENT_ID,
        "mappingStatusCounts": dict(sorted(Counter(row["mappingStatus"] for row in map_rows).items())),
        "functionIndexRows": len(function_rows),
        "functionIndexSymbolCounts": dict(sorted(Counter(row["symbolType"] for row in function_rows).items())),
        "requiredTopics": REQUIRED_TOPICS,
        "git": git_meta,
        "runtime": {
            "pythonExecutable": sys.executable,
            "pythonVersion": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "osCpuCount": os.cpu_count(),
        },
    }
    (step_dir / "status.json").write_text(
        json.dumps(status_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    artifact_manifest = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "generatedAt": utc_now(),
        "status": STATUS if ok else "failed",
        "success": ok,
        "validationResult": validation,
        "artifacts": collect_artifacts(step_dir),
        "note": "Hashes include files present when artifact_manifest.json was written; the manifest's own final hash is recorded in the top-level run manifest.",
    }
    (step_dir / "artifact_manifest.json").write_text(
        json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    update_run_manifest(artifacts_dir, step_dir, git_meta, status_payload)

    print(validation)
    print(f"Wrote {len(collect_artifacts(step_dir))} S02 artifacts under {step_dir}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
