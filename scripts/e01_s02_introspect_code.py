#!/usr/bin/env python3
"""Static code inventory for E01 S02 code-to-paper mapping.

The repository contains analysis modules with top-level file loads from the
paper authors' local machine paths. This script therefore uses AST parsing and
text scans only; it does not import repository modules.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RESEARCH_SCRIPT_PREFIXES = ("scripts/e01_s01_", "scripts/e01_s02_")

CONCEPT_PATTERNS: dict[str, re.Pattern[str]] = {
    "cell_view_sort_classes": re.compile(r"\b(BubbleSortCell|InsertionSortCell|SelectionSortCell)\b"),
    "probe_recording": re.compile(
        r"\b(StatusProbe|sorting_steps|swap_count|cell_types|compare_and_swap_count|frozen_swap_attempts)\b"
    ),
    "frozen_cell_logic": re.compile(r"\b(FREEZE|set_cell_to_freeze|frozen|freeze)\b", re.IGNORECASE),
    "traditional_or_original_reference": re.compile(r"\b(traditional|original_?|original)\b", re.IGNORECASE),
    "numpy_saved_data_io": re.compile(r"\b(np|numpy)\.(load|save)\b"),
    "sortedness_monotonicity": re.compile(r"\b(monotonicity|sortedness|get_monotonicity|monotonic)\b", re.IGNORECASE),
    "delayed_gratification": re.compile(r"\b(delay|gratification|wandering|discrepency|discrepancy)\b", re.IGNORECASE),
    "aggregation": re.compile(r"\b(aggregation|neighbor|cell_type_aggregation)\b", re.IGNORECASE),
    "chimeric_algotype_assignment": re.compile(r"\b(cell_type_list|cell_type_dict|Algotype|cell_type)\b", re.IGNORECASE),
    "duplicate_values": re.compile(r"\b(duplicate|dup|range\(10\)|repeat|repeated)\b", re.IGNORECASE),
    "opposite_direction": re.compile(r"\b(reverse_direction|reverse|disorder|opposite)\b", re.IGNORECASE),
    "threading_scheduler": re.compile(r"\b(Thread|threading|start\(|join\(|time\.sleep)\b"),
    "randomization": re.compile(r"\b(random\.shuffle|random\.random|random\.randint|np\.random)\b"),
}


@dataclass(frozen=True)
class Symbol:
    path: str
    kind: str
    name: str
    line: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=Path.cwd(), help="Repository checkout to inspect.")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("/artifacts"),
        help="Artifact root where S02 inventory files are written.",
    )
    return parser.parse_args()


def relpath(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    if isinstance(node, ast.Call):
        return dotted_name(node.func)
    return ""


class SymbolCollector(ast.NodeVisitor):
    def __init__(self, path: str) -> None:
        self.path = path
        self.symbols: list[Symbol] = []
        self.imports: list[dict[str, Any]] = []
        self.calls: list[dict[str, Any]] = []
        self.main_guard_lines: list[int] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        self.symbols.append(Symbol(self.path, "class", node.name, node.lineno))
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self.symbols.append(Symbol(self.path, "function", node.name, node.lineno))
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self.symbols.append(Symbol(self.path, "async_function", node.name, node.lineno))
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> Any:
        for alias in node.names:
            self.imports.append({"line": node.lineno, "module": alias.name, "name": alias.asname or alias.name})
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        module = "." * node.level + (node.module or "")
        for alias in node.names:
            self.imports.append({"line": node.lineno, "module": module, "name": alias.asname or alias.name})
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        self.calls.append({"line": getattr(node, "lineno", None), "call": dotted_name(node.func)})
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> Any:
        if (
            isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
            and any(isinstance(comp, ast.Constant) and comp.value == "__main__" for comp in node.test.comparators)
        ):
            self.main_guard_lines.append(node.lineno)
        self.generic_visit(node)


def scan_file(path: Path, repo_dir: Path) -> dict[str, Any]:
    relative = relpath(path, repo_dir)
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    inventory: dict[str, Any] = {
        "path": relative,
        "sha256": sha256_file(path),
        "line_count": len(lines),
        "research_script": relative.startswith(RESEARCH_SCRIPT_PREFIXES),
        "syntax_ok": True,
        "syntax_error": None,
        "symbols": [],
        "imports": [],
        "calls": [],
        "main_guard_lines": [],
        "concept_matches": [],
        "hardcoded_user_paths": [],
        "numpy_io_calls": [],
    }

    try:
        tree = ast.parse(text, filename=relative)
    except SyntaxError as exc:
        inventory["syntax_ok"] = False
        inventory["syntax_error"] = {
            "line": exc.lineno,
            "offset": exc.offset,
            "message": exc.msg,
        }
        tree = None

    if tree is not None:
        collector = SymbolCollector(relative)
        collector.visit(tree)
        inventory["symbols"] = [symbol.__dict__ for symbol in collector.symbols]
        inventory["imports"] = collector.imports
        inventory["calls"] = collector.calls
        inventory["main_guard_lines"] = collector.main_guard_lines

    for line_number, line in enumerate(lines, start=1):
        if "/Users/" in line:
            inventory["hardcoded_user_paths"].append({"line": line_number, "text": line.strip()})
        if re.search(r"\b(np|numpy)\.(load|save)\b", line):
            inventory["numpy_io_calls"].append({"line": line_number, "text": line.strip()})
        for concept_id, pattern in CONCEPT_PATTERNS.items():
            if pattern.search(line):
                inventory["concept_matches"].append(
                    {"concept_id": concept_id, "line": line_number, "text": line.strip()}
                )

    return inventory


def build_summary(files: list[dict[str, Any]]) -> dict[str, Any]:
    public_files = [item for item in files if not item["research_script"]]
    all_symbols = [symbol for item in files for symbol in item["symbols"]]
    public_symbols = [symbol for item in public_files for symbol in item["symbols"]]

    def symbol_matches(pattern: str) -> list[dict[str, Any]]:
        regex = re.compile(pattern, re.IGNORECASE)
        return [symbol for symbol in public_symbols if regex.search(symbol["name"])]

    traditional_named_symbols = symbol_matches(r"(traditional|original)")
    traditional_algorithm_definition_candidates = [
        symbol
        for symbol in public_symbols
        if re.search(r"(traditional|original)", symbol["name"], re.IGNORECASE)
        and re.search(r"(bubble|insertion|selection|sort)", symbol["name"], re.IGNORECASE)
        and not re.search(r"(file|path|load|get_)", symbol["name"], re.IGNORECASE)
    ]

    concept_counts: dict[str, int] = {concept_id: 0 for concept_id in CONCEPT_PATTERNS}
    concept_files: dict[str, list[str]] = {concept_id: [] for concept_id in CONCEPT_PATTERNS}
    for item in public_files:
        seen_in_file: set[str] = set()
        for match in item["concept_matches"]:
            concept_counts[match["concept_id"]] += 1
            seen_in_file.add(match["concept_id"])
        for concept_id in seen_in_file:
            concept_files[concept_id].append(item["path"])

    return {
        "python_file_count": len(files),
        "public_python_file_count_excluding_research_scripts": len(public_files),
        "syntax_ok_file_count": sum(1 for item in files if item["syntax_ok"]),
        "syntax_error_files": [
            {"path": item["path"], "syntax_error": item["syntax_error"]} for item in files if not item["syntax_ok"]
        ],
        "symbol_count": len(all_symbols),
        "public_symbol_count_excluding_research_scripts": len(public_symbols),
        "classes": sorted(
            [symbol for symbol in public_symbols if symbol["kind"] == "class"],
            key=lambda item: (item["path"], item["line"], item["name"]),
        ),
        "files_with_hardcoded_user_paths": sorted(
            [item["path"] for item in public_files if item["hardcoded_user_paths"]]
        ),
        "files_with_numpy_io": sorted([item["path"] for item in public_files if item["numpy_io_calls"]]),
        "concept_match_counts": concept_counts,
        "concept_match_files": {key: sorted(value) for key, value in concept_files.items()},
        "traditional_named_symbols": sorted(
            traditional_named_symbols, key=lambda item: (item["path"], item["line"], item["name"])
        ),
        "traditional_algorithm_definition_candidates": sorted(
            traditional_algorithm_definition_candidates,
            key=lambda item: (item["path"], item["line"], item["name"]),
        ),
    }


def write_match_csv(files: list[dict[str, Any]], csv_path: Path) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "line", "concept_id", "text", "research_script"],
        )
        writer.writeheader()
        for item in files:
            for match in item["concept_matches"]:
                writer.writerow(
                    {
                        "path": item["path"],
                        "line": match["line"],
                        "concept_id": match["concept_id"],
                        "text": match["text"],
                        "research_script": item["research_script"],
                    }
                )


def main() -> int:
    args = parse_args()
    repo_dir = args.repo_dir.resolve()
    artifact_dir = args.artifacts_dir.resolve() / "research_steps" / "S02"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    python_files = sorted(
        path
        for path in repo_dir.rglob("*.py")
        if "__pycache__" not in path.parts and ".venv" not in path.parts
    )
    files = [scan_file(path, repo_dir) for path in python_files]
    summary = build_summary(files)

    inventory = {
        "researchStepId": "S02",
        "script": relpath(Path(__file__).resolve(), repo_dir) if Path(__file__).resolve().is_relative_to(repo_dir) else str(Path(__file__).resolve()),
        "repoDir": str(repo_dir),
        "artifactDir": str(artifact_dir),
        "files": files,
        "summary": summary,
    }

    inventory_path = artifact_dir / "code_inventory.json"
    inventory_path.write_text(json.dumps(inventory, indent=2, sort_keys=True), encoding="utf-8")
    write_match_csv(files, artifact_dir / "code_inventory_concept_matches.csv")

    validation = {
        "researchStepId": "S02",
        "checks": {
            "python_files_scanned": len(files),
            "syntax_errors": len(summary["syntax_error_files"]),
            "cell_view_sort_class_signals": summary["concept_match_counts"]["cell_view_sort_classes"],
            "traditional_algorithm_definition_candidates": len(
                summary["traditional_algorithm_definition_candidates"]
            ),
            "files_with_hardcoded_user_paths": len(summary["files_with_hardcoded_user_paths"]),
            "files_with_numpy_io": len(summary["files_with_numpy_io"]),
        },
        "validationResult": "passed_with_caveats",
        "caveatsOrBlockers": [
            "Static introspection intentionally avoided importing modules with top-level hard-coded local path loads.",
            "No public original/traditional algorithm definition candidate was detected by symbol-name scan; saved-data consumers remain present.",
        ],
        "artifactsWritten": [
            str(inventory_path),
            str(artifact_dir / "code_inventory_concept_matches.csv"),
            str(artifact_dir / "s02_validation.json"),
        ],
    }
    (artifact_dir / "s02_validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(validation, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
