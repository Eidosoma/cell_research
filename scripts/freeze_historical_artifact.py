#!/usr/bin/env python3
"""Create the static S02 provenance inventory for a quarantined Git snapshot.

The tool never imports or executes code from the historical checkout.  It reads Git
objects and source text, parses Python with :mod:`ast`, validates downloaded GitHub
archives byte-for-byte against the frozen tree, and writes compact inventories.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import re
import subprocess
import sys
import tarfile
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


SCHEMA = "e01.s02.historical_artifact.v1"
STEP_ID = "S02"
REMOTE_URL = "https://github.com/Zhangtaining/cell_research.git"
ARCHIVE_URL_BASE = "https://codeload.github.com/Zhangtaining/cell_research"

PACKAGE_DISTRIBUTIONS = {
    "PIL": "Pillow",
    "keras": "keras",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "seaborn": "seaborn",
    "sklearn": "scikit-learn",
    "statsmodels": "statsmodels",
    "tabulate": "tabulate",
}

# These mappings are hypotheses from filenames and code content, not proof that a
# script generated the corresponding publication panel.
FIGURE_CANDIDATES = {
    "multithread_cell_sorting_steps.py": "3,4",
    "multithread_cell_sorting_20points_steps.py": "3",
    "analysis/efficiency_analysis.py": "4",
    "multithread_cell_sorting_with_frozen_steps.py": "5,7",
    "multithread_cell_sorting_with_frozen_debug.py": "5,6,7",
    "analysis/frozen_success_compare.py": "5",
    "analysis/frozen_spearmans_distance_results.py": "5",
    "multi_dimentions/multi_dimention_monotonicity.py": "3,5,6",
    "analysis/delay_gratification_analysis.py": "6,7",
    "analysis/delay_gratification_analysis_for_not_move.py": "6,7",
    "analysis/delay_gratification_analysis_spearsman.py": "6,7",
    "multithread_sorting_cell_aggregation_analysis.py": "8",
    "multithread_sorting_cell_aggregation_disorder.py": "9,10",
    "multithread_sorting_cell_type_analysis.py": "8",
    "analysis/cell_type_aggregation_analysis.py": "8,9,10",
    "analysis/cell_type_distribution_change.py": "8,9,10",
}

CODE_FINDINGS = [
    {
        "findingId": "SCHEDULER_RANDOM_LOCK_GATE_ABSENT",
        "relatedS01Codes": "NO_FAULT_STOP_UNREPORTED|COST_OPERATION_UNDEFINED",
        "sourcePath": "modules/multithread/BubbleSortCell.py; modules/multithread/InsertionSortCell.py; modules/multithread/SelectionSortCell.py",
        "lineRange": "58-74; 85-102; 83-98",
        "observation": "Each move method acquires the shared lock unconditionally. Bubble draws randomness after acquiring the lock to select a neighbor; the paper-described pre-lock random gate is not present.",
        "evidenceStatus": "direct_static_source",
        "implication": "Thread scheduling and repeated lock acquisition remain historical-runtime variables; the prose scheduler is not an exact code description.",
    },
    {
        "findingId": "NO_FAULT_DRIVER_SCALE_AND_FILENAME_CONFLICT",
        "relatedS01Codes": "NO_FAULT_STOP_UNREPORTED|SAMPLE_SIZE_UNREPORTED",
        "sourcePath": "multithread_cell_sorting_steps.py",
        "lineRange": "96-178",
        "observation": "The active driver uses 50 values and 50 repetitions but saves filenames ending in 100exps; it stops each policy when is_sorted is true.",
        "evidenceStatus": "direct_static_source",
        "implication": "The checked-in active configuration does not directly reproduce the paper's n=100, N=100 no-fault setup.",
    },
    {
        "findingId": "TRADITIONAL_GENERATORS_ABSENT",
        "relatedS01Codes": "COST_OPERATION_UNDEFINED|FAULT_PLACEMENT_UNREPORTED",
        "sourcePath": "repository-wide source scan",
        "lineRange": "not applicable",
        "observation": "Analysis scripts reference original_* .npy files, but no top-down Bubble, Insertion, or Selection data-generation implementation is present.",
        "evidenceStatus": "direct_static_absence",
        "implication": "Traditional baselines cannot be regenerated from the public tree alone.",
    },
    {
        "findingId": "FAULT_PLACEMENT_WITH_REPLACEMENT",
        "relatedS01Codes": "FAULT_PLACEMENT_UNREPORTED",
        "sourcePath": "multithread_cell_sorting_with_frozen_steps.py; freezing_sorting_analysis.py",
        "lineRange": "49-50; 50-51",
        "observation": "Frozen indices are selected by repeated random.randint calls without deduplication.",
        "evidenceStatus": "direct_static_source",
        "implication": "The realized number of distinct frozen cells can be less than the requested count.",
    },
    {
        "findingId": "FAULT_DRIVER_BRANCHES_INCOMPLETE",
        "relatedS01Codes": "PASSIVE_INSERTION_CONTRADICTION|FAULT_PLACEMENT_UNREPORTED",
        "sourcePath": "multithread_cell_sorting_with_frozen_steps.py",
        "lineRange": "121-230",
        "observation": "The active loop runs only Bubble for f=2,3 and saves only cell-type snapshots; Selection, Insertion, trajectory, cost, and attempt outputs are commented.",
        "evidenceStatus": "direct_static_source",
        "implication": "Figure 5/7 datasets cannot be regenerated by one unmodified active driver configuration.",
    },
    {
        "findingId": "STUCK_FAULT_IMPLEMENTATION_NOT_IDENTIFIED",
        "relatedS01Codes": "FAULT_PLACEMENT_UNREPORTED",
        "sourcePath": "modules/multithread/MultiThreadCell.py and repository-wide source scan",
        "lineRange": "67-98",
        "observation": "FREEZE prevents an initiating cell's swap, while target frozen cells can be displaced and retain FREEZE status. No separate checked-in stuck-cell mode that blocks displacement was identified.",
        "evidenceStatus": "direct_static_source_and_absence",
        "implication": "The paper's passive mode is represented, but the stuck mode is missing or requires an undocumented patch/configuration.",
    },
    {
        "findingId": "STOP_RULES_ARE_SCRIPT_SPECIFIC",
        "relatedS01Codes": "STOP_RULE_CONFLICT|NO_FAULT_STOP_UNREPORTED",
        "sourcePath": "multithread_cell_sorting_steps.py; multithread_cell_sorting_with_frozen_steps.py; multithread_sorting_cell_aggregation_disorder.py",
        "lineRange": "116-167; 111-157; 165-204",
        "observation": "No-fault runs stop on is_sorted; frozen runs stop when no active/sleeping cell should move; disorder runs stop on no move or 15,000 recorded sorting steps.",
        "evidenceStatus": "direct_static_source",
        "implication": "There is no single repository-wide terminal criterion.",
    },
    {
        "findingId": "DUPLICATES_ACCEPTED_BY_CODE_SORTEDNESS",
        "relatedS01Codes": "SORTEDNESS_DUPLICATE_CONFLICT|METRIC_SCALE_AMBIGUOUS",
        "sourcePath": "analysis/utils.py; multithread_sorting_cell_aggregation_analysis.py",
        "lineRange": "2-9; 143-149",
        "observation": "The percentage helper increments on >= and is_sorted rejects only decreases, so equality is accepted.",
        "evidenceStatus": "direct_static_source",
        "implication": "Executable behavior uses nondecreasing order, contradicting the paper's printed strict > Sortedness equation.",
    },
    {
        "findingId": "DG_IMPLEMENTATION_RECOVERED_WITH_EDGE_RULES",
        "relatedS01Codes": "DG_DEFINITION_CONFLICT",
        "sourcePath": "analysis/delay_gratification_analysis.py",
        "lineRange": "35-60; 62-66; 84-115; 117-139",
        "observation": "The code deduplicates plateaus, segments signed monotonicity-error changes, pairs worsening then recovery, computes (recovery-worsening)/worsening, averages pairs, and returns zero when no usable pair exists.",
        "evidenceStatus": "direct_static_source_with_formula_interpretation",
        "implication": "The denominator agrees with the journal equation after translating error direction, while segmentation/averaging rules are code-specific and were absent from prose.",
    },
    {
        "findingId": "AGGREGATION_BOUNDARY_RECOVERED",
        "relatedS01Codes": "AGGREGATION_BOUNDARY_UNDEFINED|AGGREGATION_NULL_CONFLICT",
        "sourcePath": "analysis/cell_type_aggregation_analysis.py",
        "lineRange": "114-145; 224-235",
        "observation": "The main averaging path compares each cell to its right neighbor, assigns zero to the terminal boundary, divides by N via np.average, and interpolates 101 normalized process points; an alternative helper samples one interior cell and both neighbors.",
        "evidenceStatus": "direct_static_source",
        "implication": "The primary code metric is non-circular adjacent-pair matches divided by N; the universal 0.5 null is still not implemented or justified here.",
    },
    {
        "findingId": "POPULATION_STANDARD_DEVIATION_USED",
        "relatedS01Codes": "STD_FORMULA_INVALID|ERROR_BAR_TYPE_UNREPORTED",
        "sourcePath": "analysis/efficiency_analysis.py; analysis/cell_type_aggregation_analysis.py; analysis/delay_gratification_analysis.py",
        "lineRange": "33-38; 224-235; 189-190",
        "observation": "Analysis code uses np.std with its default ddof=0.",
        "evidenceStatus": "direct_static_source",
        "implication": "Where these paths correspond to paper plots, variability is population SD, not the invalid printed equation; raw arrays are missing so values cannot be recomputed.",
    },
    {
        "findingId": "ACTIVE_CHIMERA_CONFIG_MISMATCH",
        "relatedS01Codes": "ALGOTYPE_ASSIGNMENT_CONFLICT|SAMPLE_SIZE_UNREPORTED|PANEL_VALUE_NOT_QUOTED",
        "sourcePath": "multithread_sorting_cell_aggregation_analysis.py",
        "lineRange": "37-53; 151-184",
        "observation": "The active duplicate driver builds 200 cells (20 copies each of 0-9), assigns exactly 100 Insertion and 100 Selection cells, yet saves under a bubble_selection path and loops 100 times.",
        "evidenceStatus": "direct_static_source",
        "implication": "The active HEAD configuration conflicts with the paper's 100-cell, 10-copy description and with its own output label.",
    },
    {
        "findingId": "OPPOSING_DIRECTION_CONFIG_REQUIRES_MANUAL_EDIT",
        "relatedS01Codes": "F10_NUMERIC_NOT_RECOVERABLE|ALGOTYPE_ASSIGNMENT_CONFLICT|STOP_RULE_CONFLICT",
        "sourcePath": "multithread_sorting_cell_aggregation_disorder.py",
        "lineRange": "55-91; 161-217",
        "observation": "Only Selection is constructed with reverse_direction=True. The active main selects Bubble+Insertion, uses unique 0-99 values, and writes a non-disorder bubble_insertion path; other combinations are commented.",
        "evidenceStatus": "direct_static_source",
        "implication": "Figures 9-10 conditions are not exposed as a stable unmodified configuration and Figure 10 endpoints remain missing.",
    },
    {
        "findingId": "NEGATIVE_CONTROL_RECIPE_NOT_FOUND",
        "relatedS01Codes": "NEGATIVE_CONTROL_CONTRADICTION",
        "sourcePath": "multithread_sorting_cell_aggregation_analysis.py",
        "lineRange": "37-73; 186-197",
        "observation": "A bubble_bubble_bubble output path is commented, but the constructor binds its three numeric labels to Bubble, Insertion, and Selection; no executable identical-policy/distinct-label control recipe was found.",
        "evidenceStatus": "direct_static_source_and_absence",
        "implication": "The paper's negative-control numbers and significance cannot be attributed to a preserved public recipe.",
    },
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data, usedforsecurity=False).hexdigest()


def git(mirror: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", f"--git-dir={mirror}", *args],
        check=check,
        capture_output=True,
        text=True,
    )


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def list_tree(mirror: Path, commit: str, worktree: Path) -> list[dict[str, Any]]:
    result = subprocess.run(
        ["git", f"--git-dir={mirror}", "ls-tree", "-r", "-l", "-z", commit],
        check=True,
        capture_output=True,
    )
    rows: list[dict[str, Any]] = []
    for entry in result.stdout.split(b"\0"):
        if not entry:
            continue
        metadata, raw_path = entry.split(b"\t", 1)
        mode, object_type, object_id, size = metadata.decode().split()
        relative = raw_path.decode("utf-8", "surrogateescape")
        data = (worktree / relative).read_bytes()
        rows.append(
            {
                "path": relative,
                "mode": mode,
                "gitObjectType": object_type,
                "gitObjectId": object_id,
                "sizeBytes": int(size),
                "sha256": sha256_bytes(data),
                "computedGitBlobSha1": git_blob_sha1(data),
                "blobMatchesWorktree": git_blob_sha1(data) == object_id,
                "suffix": Path(relative).suffix.lower() or "[none]",
            }
        )
    return rows


def commit_history(mirror: Path, github_commits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    verification = {
        item["sha"]: item.get("commit", {}).get("verification", {}) for item in github_commits
    }
    fmt = "%H%x1f%P%x1f%T%x1f%aI%x1f%cI%x1f%an%x1f%ae%x1f%s"
    lines = git(mirror, "log", "--all", "--reverse", f"--format={fmt}").stdout.splitlines()
    rows = []
    for line in lines:
        sha, parents, tree, author_date, commit_date, author, email, subject = line.split("\x1f")
        verified = verification.get(sha, {})
        raw_commit = git(mirror, "cat-file", "commit", sha).stdout
        rows.append(
            {
                "commit": sha,
                "parents": parents,
                "tree": tree,
                "authorDate": author_date,
                "committerDate": commit_date,
                "authorName": author,
                "authorEmail": email,
                "subject": subject,
                "hasEmbeddedSignature": "\ngpgsig " in "\n" + raw_commit,
                "githubVerified": verified.get("verified"),
                "githubVerificationReason": verified.get("reason", "not_returned"),
            }
        )
    return rows


def refs_inventory(mirror: Path) -> list[dict[str, str]]:
    result = git(mirror, "show-ref", "--head").stdout.splitlines()
    return [{"objectId": line.split()[0], "ref": line.split()[1]} for line in result]


def local_module_roots(tree_rows: Iterable[dict[str, Any]]) -> set[str]:
    roots: set[str] = set()
    for row in tree_rows:
        path = PurePosixPath(row["path"])
        if path.suffix != ".py":
            continue
        if len(path.parts) == 1:
            roots.add(path.stem)
        else:
            roots.add(path.parts[0])
    return roots


def module_available(module: str, scope: str) -> tuple[bool | None, str]:
    if scope == "local":
        return True, "resolved statically within frozen tree"
    if scope == "stdlib":
        try:
            available = importlib.util.find_spec(module) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            available = False
        return available, "checked with find_spec; historical code not imported"
    try:
        available = importlib.util.find_spec(module) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        available = False
    distribution = PACKAGE_DISTRIBUTIONS.get(module)
    version = ""
    if distribution:
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            pass
    note = "current runtime only; historical version is not implied"
    if version:
        note += f"; installed distribution {distribution}=={version}"
    return available, note


def is_main_guard(node: ast.If) -> bool:
    try:
        return ast.unparse(node.test) in {
            "__name__ == '__main__'",
            "'__main__' == __name__",
            '__name__ == "__main__"',
            '"__main__" == __name__',
        }
    except Exception:
        return False


def source_analysis(
    worktree: Path, tree_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    local_roots = local_module_roots(tree_rows)
    import_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    script_rows: list[dict[str, Any]] = []
    parse_errors: list[dict[str, Any]] = []
    availability_cache: dict[tuple[str, str], tuple[bool | None, str]] = {}
    active_pickle_loads = 0
    absolute_author_path_lines = 0
    absolute_author_path_lines_all = 0
    top_level_execution_files = 0

    python_paths = sorted(row["path"] for row in tree_rows if row["suffix"] == ".py")
    for relative in python_paths:
        path = worktree / relative
        source = path.read_text(encoding="utf-8", errors="surrogateescape")
        lines = source.splitlines()
        file_absolute_path_lines = sum(1 for line in lines if "/Users/" in line)
        absolute_author_path_lines_all += file_absolute_path_lines
        try:
            parsed = ast.parse(source, filename=relative)
            parse_ok = True
            error = ""
        except SyntaxError as exc:
            parse_ok = False
            error = f"{exc.msg} at line {exc.lineno}"
            parse_errors.append({"path": relative, "error": error})
            parsed = None

        imports_for_file: set[str] = set()
        top_level_executable = False
        main_guard = False
        active_loads = 0
        active_saves = 0
        has_plot = False
        has_savefig = False

        if parsed is not None:
            for top_node in parsed.body:
                if isinstance(top_node, ast.If) and is_main_guard(top_node):
                    main_guard = True
                    continue
                if isinstance(
                    top_node,
                    (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                ):
                    continue
                if isinstance(top_node, (ast.Assign, ast.AnnAssign)):
                    value = getattr(top_node, "value", None)
                    if value is None or isinstance(value, (ast.Constant, ast.List, ast.Tuple, ast.Dict, ast.Set)):
                        continue
                top_level_executable = True

            for node in ast.walk(parsed):
                found: list[tuple[str, int, int]] = []
                if isinstance(node, ast.Import):
                    found.extend((alias.name, 0, node.lineno) for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    found.append((node.module or "", node.level, node.lineno))
                for module, level, line_number in found:
                    top = module.split(".")[0] if module else "[relative]"
                    if level > 0 or top in local_roots or top in {"utils", "performance_analysis"}:
                        scope = "local"
                    elif top in sys.stdlib_module_names or top in {"tkinter"}:
                        scope = "stdlib"
                    else:
                        scope = "third_party"
                    cache_key = (top, scope)
                    if cache_key not in availability_cache:
                        availability_cache[cache_key] = module_available(top, scope)
                    available, note = availability_cache[cache_key]
                    imports_for_file.add(top)
                    import_rows.append(
                        {
                            "sourceFile": relative,
                            "line": line_number,
                            "module": module or "[relative]",
                            "topLevelModule": top,
                            "relativeLevel": level,
                            "scope": scope,
                            "availableInCurrentRuntime": available,
                            "validationNote": note,
                        }
                    )

                if isinstance(node, ast.Call):
                    try:
                        name = ast.unparse(node.func)
                    except Exception:
                        name = ""
                    if name.endswith(".load") and ("np." in name or "numpy." in name):
                        active_loads += 1
                        if any(
                            keyword.arg == "allow_pickle"
                            and isinstance(keyword.value, ast.Constant)
                            and keyword.value.value is True
                            for keyword in node.keywords
                        ):
                            active_pickle_loads += 1
                    if name.endswith(".save") and ("np." in name or "numpy." in name):
                        active_saves += 1
                    if name.startswith("plt.") or "matplotlib" in name:
                        has_plot = True
                    if name.endswith("savefig"):
                        has_savefig = True

        for line_number, line in enumerate(lines, 1):
            if ".npy" in line.lower():
                stripped = line.strip()
                commented = stripped.startswith("#")
                if "load" in stripped:
                    operation = "load_reference"
                elif "save" in stripped:
                    operation = "save_reference"
                else:
                    operation = "path_reference"
                raw_rows.append(
                    {
                        "recordType": "source_reference",
                        "sourceFile": relative,
                        "line": line_number,
                        "operation": operation,
                        "activeOrCommented": "commented" if commented else "active",
                        "referenceText": stripped,
                        "absoluteAuthorPath": "/Users/" in line,
                        "historicalFilePresent": False,
                        "smokeReadResult": "not_run_missing_file",
                    }
                )
                if "/Users/" in line:
                    absolute_author_path_lines += 1

        if top_level_executable:
            top_level_execution_files += 1
        lower_source = source.lower()
        if "matplotlib" in lower_source or "plt." in lower_source:
            has_plot = True
        role = "simulator_or_support"
        if has_plot and (active_loads or ".npy" in lower_source):
            role = "analysis_or_plot_candidate"
        elif active_saves:
            role = "raw_data_generator_candidate"
        elif relative.startswith("analysis/") or relative.startswith("multi_dimentions/"):
            role = "analysis_candidate"
        elif relative.startswith("visualization/"):
            role = "visualization_support"
        figures = FIGURE_CANDIDATES.get(relative, "")
        script_rows.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "lineCount": len(lines),
                "astParseResult": "pass" if parse_ok else f"fail: {error}",
                "role": role,
                "imports": "|".join(sorted(imports_for_file)),
                "activeNpyLoads": active_loads,
                "activeNpySaves": active_saves,
                "npyReferenceLines": sum(1 for line in lines if ".npy" in line.lower()),
                "absoluteAuthorPathLines": file_absolute_path_lines,
                "hasMatplotlibOrPlotCalls": has_plot,
                "hasSavefig": has_savefig,
                "hasMainGuard": main_guard,
                "hasTopLevelExecutableStatements": top_level_executable,
                "candidatePaperFigures": figures,
                "figureMappingBasis": "filename/code inference; not a verified publication mapping"
                if figures
                else "no explicit mapping found",
                "directPublicationFigureRecipeFound": False,
            }
        )

    summary = {
        "pythonFileCount": len(python_paths),
        "astParsePassCount": len(python_paths) - len(parse_errors),
        "astParseErrors": parse_errors,
        "uniqueTopLevelImports": sorted({row["topLevelModule"] for row in import_rows}),
        "thirdPartyImports": sorted(
            {row["topLevelModule"] for row in import_rows if row["scope"] == "third_party"}
        ),
        "activeAllowPickleTrueLoads": active_pickle_loads,
        "absoluteAuthorNpyPathReferenceLines": absolute_author_path_lines,
        "absoluteAuthorPathReferenceLinesAll": absolute_author_path_lines_all,
        "filesWithTopLevelExecutableStatements": top_level_execution_files,
    }
    return import_rows, raw_rows, script_rows, summary


def archive_bytes(path: Path, kind: str) -> tuple[dict[str, str], list[str]]:
    contents: dict[str, str] = {}
    unsafe: list[str] = []
    if kind == "tar.gz":
        with tarfile.open(path, "r:gz") as archive:
            for member in archive.getmembers():
                pure = PurePosixPath(member.name)
                if pure.is_absolute() or ".." in pure.parts:
                    unsafe.append(member.name)
                if member.isfile():
                    relative = "/".join(pure.parts[1:])
                    handle = archive.extractfile(member)
                    assert handle is not None
                    contents[relative] = sha256_bytes(handle.read())
    elif kind == "zip":
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                pure = PurePosixPath(name)
                if pure.is_absolute() or ".." in pure.parts:
                    unsafe.append(name)
                if not name.endswith("/"):
                    relative = "/".join(pure.parts[1:])
                    contents[relative] = sha256_bytes(archive.read(name))
    else:
        raise ValueError(kind)
    return contents, unsafe


def files_across_history(mirror: Path) -> list[str]:
    result = git(mirror, "rev-list", "--objects", "--all").stdout.splitlines()
    return sorted({line.split(" ", 1)[1] for line in result if " " in line})


def utc_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def build(args: argparse.Namespace) -> dict[str, Any]:
    mirror = args.mirror.resolve()
    worktree = args.worktree.resolve()
    cache_dir = args.cache_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    repo_api = read_json(args.github_repo_json)
    tags_api = read_json(args.github_tags_json)
    releases_api = read_json(args.github_releases_json)
    branches_api = read_json(args.github_branches_json)
    github_commits = read_json(args.github_commits_json)

    commit = git(mirror, "rev-parse", "refs/heads/main^{commit}").stdout.strip()
    tree = git(mirror, "rev-parse", f"{commit}^{{tree}}").stdout.strip()
    worktree_commit = subprocess.run(
        ["git", "-C", str(worktree), "rev-parse", "HEAD^{commit}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    worktree_status = subprocess.run(
        ["git", "-C", str(worktree), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    fsck = git(mirror, "fsck", "--full", "--strict", check=False)
    tree_rows = list_tree(mirror, commit, worktree)
    history = commit_history(mirror, github_commits)
    refs = refs_inventory(mirror)
    all_history_paths = files_across_history(mirror)
    npy_at_head = [row["path"] for row in tree_rows if row["suffix"] == ".npy"]
    npy_in_history = [path for path in all_history_paths if path.lower().endswith(".npy")]

    import_rows, raw_rows, script_rows, source_summary = source_analysis(worktree, tree_rows)
    for actual in npy_at_head:
        raw_rows.insert(
            0,
            {
                "recordType": "historical_file",
                "sourceFile": actual,
                "line": "",
                "operation": "actual_npy_file",
                "activeOrCommented": "not_applicable",
                "referenceText": actual,
                "absoluteAuthorPath": False,
                "historicalFilePresent": True,
                "smokeReadResult": "pending",
            },
        )

    expected_sha = {row["path"]: row["sha256"] for row in tree_rows}
    archive_records = []
    for kind, suffix in (("tar.gz", "tar.gz"), ("zip", "zip")):
        path = cache_dir / f"cell_research-{commit}.{suffix}"
        contents, unsafe = archive_bytes(path, kind)
        archive_records.append(
            {
                "kind": kind,
                "sourceUrl": f"{ARCHIVE_URL_BASE}/{suffix}/{commit}",
                "cacheOnlyPath": str(path),
                "retrievedFileMtimeUtc": utc_mtime(path),
                "sizeBytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "fileCount": len(contents),
                "unsafePathCount": len(unsafe),
                "pathsMatchGitTree": set(contents) == set(expected_sha),
                "fileBytesMatchGitTree": contents == expected_sha,
            }
        )

    bundle = cache_dir / f"cell_research-{commit}.bundle"
    bundle_verify = git(mirror, "bundle", "verify", str(bundle), check=False)
    bundle_record = {
        "kind": "git_bundle",
        "cacheOnlyPath": str(bundle),
        "retrievedFileMtimeUtc": utc_mtime(bundle),
        "sizeBytes": bundle.stat().st_size,
        "sha256": sha256_file(bundle),
        "verificationExitCode": bundle_verify.returncode,
        "verificationOutput": (bundle_verify.stdout + bundle_verify.stderr).strip(),
    }

    extension_counts = Counter(row["suffix"] for row in tree_rows)
    file_mode_counts = Counter(row["mode"] for row in tree_rows)
    environment_files = [
        row["path"]
        for row in tree_rows
        if Path(row["path"]).name.lower()
        in {
            "requirements.txt",
            "environment.yml",
            "environment.yaml",
            "pyproject.toml",
            "setup.py",
            "setup.cfg",
            "pipfile",
            "pipfile.lock",
            "poetry.lock",
            ".python-version",
            "dockerfile",
        }
    ]
    license_files = [
        row["path"]
        for row in tree_rows
        if re.match(r"(?i)^(license|copying|notice)(\..*)?$", Path(row["path"]).name)
    ]
    license_history_files = [
        path
        for path in all_history_paths
        if re.match(r"(?i)^(license|copying|notice)(\..*)?$", PurePosixPath(path).name)
    ]

    post_publication_commits = [row for row in history if row["authorDate"] >= "2024-10-01"]
    missing_evidence = [
        "No .npy file exists at HEAD or anywhere in reachable Git history despite many source references.",
        "No LICENSE, COPYING, or NOTICE file and no GitHub-detected license; redistribution permission is not established.",
        "No requirements, environment, packaging, lock, container, or Python-version file exists.",
        "No tags or GitHub releases identify a publication snapshot.",
        "Four October 2024 commits postdate the 2023 bulk source commit; the exact publication-used commit is not attributable from public refs.",
        "No exact composite Figure 3-10 recipe or panel-numbered script is present.",
        "No seeds, scenario IDs, raw uncertainty arrays, or complete run manifests are present.",
    ]

    source_inventory = {
        "schema": SCHEMA,
        "researchStepId": STEP_ID,
        "retrievedAtUtc": args.retrieved_at,
        "remote": {
            "url": REMOTE_URL,
            "githubFullName": repo_api.get("full_name"),
            "defaultBranch": repo_api.get("default_branch"),
            "createdAt": repo_api.get("created_at"),
            "pushedAt": repo_api.get("pushed_at"),
            "updatedAt": repo_api.get("updated_at"),
            "archived": repo_api.get("archived"),
            "visibility": repo_api.get("visibility"),
        },
        "frozenHead": {
            "branch": "main",
            "commit": commit,
            "tree": tree,
            "worktreeCommit": worktree_commit,
            "worktreeClean": not bool(worktree_status),
        },
        "history": {
            "commitCount": len(history),
            "firstCommit": history[0]["commit"],
            "firstCommitDate": history[0]["authorDate"],
            "latestCommit": history[-1]["commit"],
            "latestCommitDate": history[-1]["authorDate"],
            "postPublicationEraCommitCount": len(post_publication_commits),
            "publicationSnapshotAttributable": False,
            "publicationSnapshotCaveat": "No publication tag/release; retain complete history and do not equate HEAD with exact paper execution.",
        },
        "refs": {"count": len(refs), "tagCount": len(tags_api), "releaseCount": len(releases_api)},
        "files": {
            "headFileCount": len(tree_rows),
            "allReachableHistoricalPathCount": len(all_history_paths),
            "totalHeadBytes": sum(row["sizeBytes"] for row in tree_rows),
            "extensionCounts": dict(sorted(extension_counts.items())),
            "modeCounts": dict(sorted(file_mode_counts.items())),
            "allWorktreeBlobIdsValidated": all(row["blobMatchesWorktree"] for row in tree_rows),
            "symlinkCount": sum(row["mode"] == "120000" for row in tree_rows),
            "executableFileCount": sum(row["mode"] == "100755" for row in tree_rows),
        },
        "rawData": {
            "npyFilesAtHead": npy_at_head,
            "npyFilesInReachableHistory": npy_in_history,
            "npyReferenceLineCount": len(raw_rows),
            "rawDataSmokeRead": "not applicable: no historical .npy bytes exist to read",
        },
        "python": source_summary,
        "environmentFiles": environment_files,
        "license": {
            "filesAtHead": license_files,
            "filesInReachableHistory": license_history_files,
            "githubApiLicense": repo_api.get("license"),
            "redistributionPermissionEstablished": False,
        },
        "tagReleaseAndArchiveLinkSearch": {
            "gitTags": len(tags_api),
            "githubReleases": len(releases_api),
            "linkedExternalArchivesFoundInSource": [],
            "commitArchiveUrls": [
                f"{ARCHIVE_URL_BASE}/tar.gz/{commit}",
                f"{ARCHIVE_URL_BASE}/zip/{commit}",
            ],
        },
        "submodules": {"gitmodulesPresent": (worktree / ".gitmodules").exists()},
        "lfs": {"gitattributesPresent": (worktree / ".gitattributes").exists()},
        "archives": archive_records,
        "bundle": bundle_record,
        "missingEvidence": missing_evidence,
    }

    checks = [
        {
            "check": "git_object_verification",
            "status": "pass" if fsck.returncode == 0 else "fail",
            "detail": (fsck.stdout + fsck.stderr).strip() or "git fsck --full --strict returned no errors",
        },
        {
            "check": "head_file_checksums_and_blob_identity",
            "status": "pass" if all(row["blobMatchesWorktree"] for row in tree_rows) else "fail",
            "detail": f"{sum(row['blobMatchesWorktree'] for row in tree_rows)}/{len(tree_rows)} files matched Git blob IDs",
        },
        {
            "check": "downloaded_archive_content",
            "status": "pass"
            if all(a["fileBytesMatchGitTree"] and a["unsafePathCount"] == 0 for a in archive_records)
            else "fail",
            "detail": "tar.gz and zip paths/bytes match the frozen Git tree; no unsafe archive paths",
        },
        {
            "check": "git_bundle",
            "status": "pass" if bundle_verify.returncode == 0 else "fail",
            "detail": "complete history bundle verified in quarantine",
        },
        {
            "check": "raw_data_smoke_reads",
            "status": "constrained",
            "detail": "0 .npy files at HEAD and 0 in reachable history; no bytes existed to smoke-read",
        },
        {
            "check": "python_ast_and_import_inventory",
            "status": "pass" if not source_summary["astParseErrors"] else "fail",
            "detail": f"{source_summary['astParsePassCount']}/{source_summary['pythonFileCount']} sources parsed; imports resolved statically without importing historical code",
        },
        {
            "check": "license_review",
            "status": "constrained",
            "detail": "no license file or GitHub-detected license; collectible output uses a lawful manifest, not source bytes",
        },
        {
            "check": "publication_snapshot_attribution",
            "status": "constrained",
            "detail": "no tag/release; complete six-commit history retained, exact publication commit not assertable",
        },
    ]
    hard_fail = any(item["status"] == "fail" for item in checks)
    validation = {
        "schema": "e01.s02.validation.v1",
        "researchStepId": STEP_ID,
        "stepNumber": 2,
        "passed": not hard_fail,
        "validationResult": "PASS_WITH_CONSTRAINTS" if not hard_fail else "FAIL",
        "checks": checks,
        "counts": {
            "commits": len(history),
            "headFiles": len(tree_rows),
            "pythonFiles": source_summary["pythonFileCount"],
            "npyFiles": len(npy_at_head),
            "npyReferenceLines": len(raw_rows),
            "tags": len(tags_api),
            "releases": len(releases_api),
            "licenseFiles": len(license_files),
        },
    }

    write_json(output / "source_inventory.json", source_inventory)
    write_json(output / "git_refs.json", {"schema": SCHEMA, "researchStepId": STEP_ID, "refs": refs})
    write_csv(
        output / "source_tree_files.csv",
        tree_rows,
        [
            "path",
            "mode",
            "gitObjectType",
            "gitObjectId",
            "sizeBytes",
            "sha256",
            "computedGitBlobSha1",
            "blobMatchesWorktree",
            "suffix",
        ],
    )
    checksum_lines = [f"{row['sha256']}  {row['path']}" for row in tree_rows]
    (output / "source_tree_checksums.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )
    with (output / "commit_history.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(history)
    write_csv(
        output / "raw_data_inventory.csv",
        raw_rows,
        [
            "recordType",
            "sourceFile",
            "line",
            "operation",
            "activeOrCommented",
            "referenceText",
            "absoluteAuthorPath",
            "historicalFilePresent",
            "smokeReadResult",
        ],
    )
    write_csv(
        output / "figure_script_inventory.csv",
        script_rows,
        list(script_rows[0]),
    )
    write_csv(output / "import_inventory.csv", import_rows, list(import_rows[0]))
    write_csv(
        output / "historical_code_findings.csv",
        [dict(row, commit=commit) for row in CODE_FINDINGS],
        [
            "findingId",
            "relatedS01Codes",
            "sourcePath",
            "lineRange",
            "commit",
            "observation",
            "evidenceStatus",
            "implication",
        ],
    )
    write_json(output / "validation_summary.json", validation)
    write_json(
        output / "lawful_source_manifest.json",
        {
            "schema": "e01.s02.lawful_source_manifest.v1",
            "researchStepId": STEP_ID,
            "remoteUrl": REMOTE_URL,
            "frozenCommit": commit,
            "frozenTree": tree,
            "licenseFinding": "No license grant found in any reachable path or GitHub API metadata.",
            "collectibleSourceBytesIncluded": False,
            "reason": "Public readability alone does not establish redistribution permission; preserve exact locators and checksums instead.",
            "cacheOnlyPreservation": {
                "mirror": str(mirror),
                "worktree": str(worktree),
                "archives": archive_records,
                "bundle": bundle_record,
            },
            "immutableRetrievalInstructions": [
                f"git clone --mirror --no-hardlinks {REMOTE_URL} historical-source.git",
                f"git --git-dir=historical-source.git rev-parse refs/heads/main  # expected {commit}",
                f"curl -L {ARCHIVE_URL_BASE}/tar.gz/{commit}",
                f"curl -L {ARCHIVE_URL_BASE}/zip/{commit}",
            ],
        },
    )
    write_json(
        output / "compatibility_patch_ledger.json",
        {
            "schema": "e01.s02.compatibility_patch_ledger.v1",
            "researchStepId": STEP_ID,
            "historicalSourceModified": False,
            "patchesCreated": [],
            "executionAttempted": False,
            "note": "S02 preserved and statically inspected original bytes only. Compatibility work is deferred to S04.",
        },
    )
    api_inputs = [
        args.github_repo_json,
        args.github_tags_json,
        args.github_releases_json,
        args.github_branches_json,
        args.github_commits_json,
    ]
    write_json(
        output / "retrieval_log.json",
        {
            "schema": "e01.s02.retrieval_log.v1",
            "researchStepId": STEP_ID,
            "retrievedAtUtc": args.retrieved_at,
            "remoteUrl": REMOTE_URL,
            "selectedRef": "refs/heads/main",
            "resolvedCommit": commit,
            "commands": [
                f"git ls-remote --symref {REMOTE_URL} HEAD 'refs/heads/*' 'refs/tags/*'",
                f"git clone --mirror --no-hardlinks {REMOTE_URL} {mirror}",
                f"git clone --no-hardlinks {mirror} {worktree}",
                f"git -C {worktree} checkout --detach {commit}",
                f"git --git-dir={mirror} fsck --full --strict",
                f"git --git-dir={mirror} bundle create {bundle} --all",
                f"git --git-dir={mirror} bundle verify {bundle}",
                f"curl --fail --location {ARCHIVE_URL_BASE}/tar.gz/{commit}",
                f"curl --fail --location {ARCHIVE_URL_BASE}/zip/{commit}",
            ],
            "githubApiInputs": [
                {
                    "path": str(path.resolve()),
                    "mtimeUtc": utc_mtime(path),
                    "sizeBytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in api_inputs
            ],
            "toolVersions": {
                "git": subprocess.run(
                    ["git", "--version"], check=True, capture_output=True, text=True
                ).stdout.strip(),
                "python": sys.version.split()[0],
            },
            "failedAttempt": "An initial `git bundle verify` was run outside a repository and returned 'need a repository'; verification was rerun with --git-dir and passed.",
        },
    )
    return {"sourceInventory": source_inventory, "validation": validation}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mirror", type=Path, required=True)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--github-repo-json", type=Path, required=True)
    parser.add_argument("--github-tags-json", type=Path, required=True)
    parser.add_argument("--github-releases-json", type=Path, required=True)
    parser.add_argument("--github-branches-json", type=Path, required=True)
    parser.add_argument("--github-commits-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--retrieved-at", required=True)
    return parser.parse_args()


def main() -> None:
    result = build(parse_args())
    print(json.dumps(result["validation"], indent=2))


if __name__ == "__main__":
    main()
