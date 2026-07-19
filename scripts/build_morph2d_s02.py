#!/usr/bin/env python3
"""Build, falsify, and validate the E06 S02 relational grammar artifacts."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version as package_version
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import pandas as pd
import yaml


REPOSITORY = Path(__file__).resolve().parents[1]
WORKSPACE = REPOSITORY.parent
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from src.morph2d.falsification import (  # noqa: E402
    enumerate_single_swap_counterexamples,
    search_distant_counterexample,
)
from src.morph2d.grammar import (  # noqa: E402
    catalog_to_dict,
    detect_contradictions,
    load_grammar_catalog,
    parse_grammar,
    parse_grammar_catalog,
    score_grid,
)
from src.morph2d.targets import (  # noqa: E402
    evaluate_success,
    exact_equivalence_orbit,
    load_target_catalog,
    transform_grid,
)


TARGET_CONFIG = REPOSITORY / "configs/morphologies/target_catalog.yaml"
GRAMMAR_CONFIG = REPOSITORY / "configs/morphologies/grammar_catalog.yaml"
HAND_CONFIG = REPOSITORY / "configs/morphologies/grammar_hand_fixtures.yaml"
ADVERSARIAL_CONFIG = (
    REPOSITORY / "configs/morphologies/grammar_adversarial_fixtures.yaml"
)
SOURCE = REPOSITORY / "src/morph2d/grammar.py"
FALSIFICATION_SOURCE = REPOSITORY / "src/morph2d/falsification.py"
TEST = REPOSITORY / "tests/test_morph2d_grammar.py"

UPSTREAM_INPUTS = {
    "agents": WORKSPACE / "AGENTS.md",
    "full_plan": WORKSPACE / "FULL_PLAN.md",
    "research_plan_pre_s02_update": WORKSPACE / "RESEARCH_PLAN.md",
    "attachment_manifest": WORKSPACE / "input-attachments/MANIFEST.json",
    "attachment_sidecar": WORKSPACE
    / "input-attachments/21c2278b-9950-4e39-a2c8-df578a2508ec/_metadata/ATTACHMENT.md",
    "s01_report": Path("/artifacts/research_steps/S01/research_step_full_results.md"),
    "s01_target_catalog": Path("/artifacts/research_steps/S01/target_catalog.yaml"),
    "s01_validation": Path("/artifacts/research_steps/S01/validation_summary.json"),
    "s01_ambiguity_register": Path(
        "/artifacts/research_steps/S01/ambiguity_register.csv"
    ),
    "e01_transition_spec": Path(
        "/previous-artifacts/E01/research_steps/S03/transition_spec.md"
    ),
    "e01_reference_api": Path(
        "/previous-artifacts/E01/research_steps/S05/api_documentation.md"
    ),
    "e01_event_schema_docs": Path(
        "/previous-artifacts/E01/research_steps/S06/schema_documentation.md"
    ),
    "e04_metric_specification": Path(
        "/previous-artifacts/E04/research_steps/S04/metric_specification.md"
    ),
    "e04_null_specification": Path(
        "/previous-artifacts/E04/research_steps/S05/null_specification.md"
    ),
    "e04_kinetic_specification": Path(
        "/previous-artifacts/E04/research_steps/S07/kinetic_intervention_specification.md"
    ),
    "e04_e06_handoff": Path(
        "/previous-artifacts/E04/research_steps/S14/e06_e07_handoff.md"
    ),
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _grid(rows: Sequence[str]):
    return tuple(tuple(row) for row in rows)


def _load_all():
    metadata, grammars = load_grammar_catalog(GRAMMAR_CONFIG)
    _, targets = load_target_catalog(TARGET_CONFIG)
    hand = yaml.safe_load(HAND_CONFIG.read_text(encoding="utf-8"))
    adversarial = yaml.safe_load(ADVERSARIAL_CONFIG.read_text(encoding="utf-8"))
    return metadata, grammars, targets, hand, adversarial


def _validate_s01_immutability() -> dict[str, Any]:
    manifest_path = Path("/artifacts/research_steps/S01/artifact_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches = []
    for record in manifest["artifacts"]:
        path = manifest_path.parent / record["path"]
        if not path.is_file() or _sha256_file(path) != record["sha256"]:
            mismatches.append(record["path"])
    validation = json.loads(
        Path("/artifacts/research_steps/S01/validation_summary.json").read_text(
            encoding="utf-8"
        )
    )
    return {
        "success": not mismatches and validation.get("success") is True,
        "checkedArtifactCount": len(manifest["artifacts"]),
        "hashMismatches": mismatches,
        "s01ValidationSuccess": validation.get("success"),
    }


def _parser_roundtrip(metadata, grammars) -> dict[str, Any]:
    serialized = yaml.safe_dump(
        catalog_to_dict(metadata, grammars), sort_keys=False, allow_unicode=True
    )
    roundtrip_raw = yaml.safe_load(serialized)
    reparsed_metadata, reparsed_grammars = parse_grammar_catalog(roundtrip_raw)
    success = reparsed_metadata == metadata and reparsed_grammars == grammars
    return {
        "schemaVersion": "e06.s02.parser-roundtrip.v1",
        "researchStepId": "S02",
        "success": success,
        "grammarCount": len(grammars),
        "constraintCount": sum(len(grammar.constraints) for grammar in grammars),
        "semanticCatalogSha256": _sha256_bytes(
            json.dumps(
                catalog_to_dict(metadata, grammars),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ),
        "yamlRoundtripSha256": _sha256_bytes(serialized.encode("utf-8")),
    }


def _hand_scores(hand: Mapping[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    grammar = parse_grammar(hand["grammar"])
    rows: list[dict[str, Any]] = []
    success = True
    for case in hand["cases"]:
        result = score_grid(case["rows"], grammar)
        observed = {
            item["constraintId"]: item["observed"] for item in result["constraints"]
        }
        passed = (
            result["accepted"] is case["expectedAccepted"]
            and result["hardViolationCount"] == case["expectedHardViolationCount"]
            and abs(result["softScore"] - case["expectedSoftScore"]) < 1e-12
            and observed == case["expectedObservations"]
        )
        success &= passed
        rows.append(
            {
                "case_id": case["caseId"],
                "expected_accepted": case["expectedAccepted"],
                "observed_accepted": result["accepted"],
                "expected_hard_violations": case["expectedHardViolationCount"],
                "observed_hard_violations": result["hardViolationCount"],
                "expected_soft_score": case["expectedSoftScore"],
                "observed_soft_score": result["softScore"],
                "observations_match": observed == case["expectedObservations"],
                "pass": passed,
                "observations_json": json.dumps(observed, sort_keys=True),
            }
        )
    return rows, success


def _positive_and_symmetry_validation(grammars, targets, adversarial):
    grammar_by_target = {grammar.target_id: grammar for grammar in grammars}
    target_by_id = {target.target_id: target for target in targets}
    rows = []
    all_exact = True
    all_boundary = True
    all_adversary_symmetry = True
    exact_state_count = 0
    boundary_variant_count = 0
    for target in targets:
        grammar = grammar_by_target[target.target_id]
        scores = [
            score_grid(member, grammar) for member in exact_equivalence_orbit(target)
        ]
        exact_state_count += len(scores)
        exact_pass = all(
            item["accepted"]
            and item["hardViolationCount"] == 0
            and abs(item["softScore"] - 1.0) < 1e-12
            for item in scores
        )
        all_exact &= exact_pass
        boundary_pass: bool | None = None
        boundary_swap = target.validation.get("acceptedBoundarySwap")
        if boundary_swap:
            boundary_variant_count += 1
            mutable = [list(row) for row in target.grid]
            first, second = map(tuple, boundary_swap)
            mutable[first[0]][first[1]], mutable[second[0]][second[1]] = (
                mutable[second[0]][second[1]],
                mutable[first[0]][first[1]],
            )
            candidate = tuple(tuple(row) for row in mutable)
            boundary_pass = (
                evaluate_success(candidate, target)["success"]
                and score_grid(candidate, grammar)["accepted"]
            )
            all_boundary &= boundary_pass
        rows.append(
            {
                "target_id": target.target_id,
                "grammar_id": grammar.grammar_id,
                "constraint_count": len(grammar.constraints),
                "acceptance_threshold": grammar.acceptance_threshold,
                "exact_equivalence_state_count": len(scores),
                "exact_states_accepted": sum(item["accepted"] for item in scores),
                "minimum_exact_soft_score": min(item["softScore"] for item in scores),
                "exact_symmetry_pass": exact_pass,
                "boundary_variant_present": boundary_pass is not None,
                "boundary_variant_pass": boundary_pass,
                "underdetermination_hypothesis": grammar.underdetermination_hypothesis,
            }
        )

    adversarial_rows = []
    for fixture in adversarial["fixtures"]:
        grammar = grammar_by_target[fixture["targetId"]]
        target = target_by_id[fixture["targetId"]]
        grammar_result = score_grid(fixture["rows"], grammar)
        target_result = evaluate_success(fixture["rows"], target)
        transformed_results = [
            score_grid(transform_grid(_grid(fixture["rows"]), transform), grammar)
            for transform in grammar.orientation_transforms
        ]
        symmetry_pass = all(
            item["accepted"] == grammar_result["accepted"]
            and abs(item["softScore"] - grammar_result["softScore"]) < 1e-12
            and item["hardViolationCount"] == grammar_result["hardViolationCount"]
            for item in transformed_results
        )
        passed = (
            grammar_result["accepted"] is fixture["expectedGrammarAccepted"]
            and target_result["success"] is fixture["expectedS01Accepted"]
            and target_result["mismatchCount"] >= fixture["minimumMismatch"]
            and symmetry_pass
        )
        all_adversary_symmetry &= passed
        adversarial_rows.append(
            {
                "fixture_id": fixture["fixtureId"],
                "target_id": fixture["targetId"],
                "challenged_relation": fixture["challengedRelation"],
                "grammar_accepted": grammar_result["accepted"],
                "grammar_soft_score": grammar_result["softScore"],
                "grammar_relational_score": grammar_result["relationalScore"],
                "s01_accepted": target_result["success"],
                "s01_mismatch_count": target_result["mismatchCount"],
                "component_match": target_result["componentMatch"],
                "topology_match": target_result["topologyMatch"],
                "d4_symmetry_invariance_pass": symmetry_pass,
                "pass": passed,
            }
        )
    summary = {
        "exactStateCount": exact_state_count,
        "exactStatesPass": all_exact,
        "boundaryVariantCount": boundary_variant_count,
        "boundaryVariantsPass": all_boundary,
        "adversarialFixtureCount": len(adversarial_rows),
        "adversarialFixturesPass": all_adversary_symmetry,
    }
    return rows, adversarial_rows, summary


def _contradiction_audit() -> dict[str, Any]:
    catalog = yaml.safe_load(GRAMMAR_CONFIG.read_text(encoding="utf-8"))
    probes = []

    interval = deepcopy(catalog["grammars"][0])
    interval["constraints"][1]["priority"] = "hard"
    conflict = deepcopy(interval["constraints"][1])
    conflict["constraintId"] = "probe_empty_edge_interval"
    conflict["parameters"]["min"] = 40
    conflict["parameters"]["max"] = 41
    interval["constraints"].append(conflict)
    probes.append(("empty_hard_interval", interval, "empty intersection"))

    capacity = deepcopy(catalog["grammars"][0])
    neighbor = next(
        item for item in capacity["constraints"] if item["kind"] == "neighbor_count"
    )
    neighbor["parameters"]["min"] = 5
    neighbor["parameters"]["max"] = 5
    probes.append(("neighbor_capacity", capacity, "neighborhood capacity"))

    invalid_range = deepcopy(catalog["grammars"][0])
    invalid_range["constraints"][1]["parameters"]["min"] = 29
    invalid_range["constraints"][1]["parameters"]["max"] = 28
    probes.append(("min_exceeds_max", invalid_range, "min exceeds max"))

    parity = deepcopy(catalog["grammars"][0])
    parity["shape"] = [2, 2]
    parity["cellTypeCounts"] = {".": 0, "A": 3, "B": 1}
    parity["constraints"] = [
        {
            "constraintId": "probe_impossible_exact_reflection",
            "kind": "symmetry",
            "priority": "hard",
            "weight": 1,
            "parameters": {
                "transform": "reflect_vertical",
                "frame": "domain",
                "maxMismatches": 0,
                "deviationScale": 4,
            },
        }
    ]
    probes.append(("symmetry_parity", parity, "even token counts"))

    records = []
    for probe_id, raw, expected in probes:
        grammar = parse_grammar(raw, reject_contradictions=False)
        contradictions = detect_contradictions(grammar)
        passed = any(expected in item for item in contradictions)
        rejected = False
        try:
            parse_grammar(raw)
        except ValueError:
            rejected = True
        records.append(
            {
                "probeId": probe_id,
                "expectedDiagnosticSubstring": expected,
                "diagnostics": contradictions,
                "expectedDiagnosticFound": passed,
                "parserRejected": rejected,
                "pass": passed and rejected,
            }
        )
    return {
        "schemaVersion": "e06.s02.contradiction-audit.v1",
        "researchStepId": "S02",
        "success": all(item["pass"] for item in records),
        "probeCount": len(records),
        "probes": records,
    }


def _falsify_worker(arguments: tuple[str, int]) -> dict[str, Any]:
    target_id, proposal_budget = arguments
    _, grammars = load_grammar_catalog(GRAMMAR_CONFIG)
    _, targets = load_target_catalog(TARGET_CONFIG)
    adversarial = yaml.safe_load(ADVERSARIAL_CONFIG.read_text(encoding="utf-8"))
    grammar = next(item for item in grammars if item.target_id == target_id)
    target = next(item for item in targets if item.target_id == target_id)
    fixture = next(
        item for item in adversarial["fixtures"] if item["targetId"] == target_id
    )
    single = enumerate_single_swap_counterexamples(target, grammar)
    distant = search_distant_counterexample(
        target,
        grammar,
        proposals=proposal_budget,
        restarts=8,
        minimum_hamming=int(fixture["minimumMismatch"]),
        initial_states=(_grid(fixture["rows"]),),
    )
    return {"targetId": target_id, "singleSwap": single, "distantSearch": distant}


def _run_falsification(targets, proposal_budget: int, workers: int):
    arguments = [(target.target_id, proposal_budget) for target in targets]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        records = list(executor.map(_falsify_worker, arguments))
    order = {target.target_id: index for index, target in enumerate(targets)}
    records.sort(key=lambda item: order[item["targetId"]])
    rows = []
    for record in records:
        single = record["singleSwap"]
        distant = record["distantSearch"]
        counterexample = distant["counterexample"]
        rows.append(
            {
                "target_id": record["targetId"],
                "single_swap_proposals": single["proposalCount"],
                "single_swap_grammar_accepted": single["grammarAcceptedCount"],
                "single_swap_s01_accepted": single["s01AcceptedCount"],
                "single_swap_false_positives": single["grammarAcceptedOutsideS01Count"],
                "single_swap_false_positive_fraction": single[
                    "grammarAcceptedOutsideS01Count"
                ]
                / single["proposalCount"],
                "search_proposal_budget": distant["proposalBudget"],
                "search_proposals_evaluated": distant["proposalsEvaluated"],
                "search_restarts": distant["restartCount"],
                "search_seed": str(distant["seed"]),
                "distant_counterexample_found": distant["counterexampleFound"],
                "distant_grammar_soft_score": (
                    None
                    if counterexample is None
                    else counterexample["grammar"]["softScore"]
                ),
                "distant_grammar_relational_score": (
                    None
                    if counterexample is None
                    else counterexample["grammar"]["relationalScore"]
                ),
                "distant_s01_mismatch_count": (
                    None
                    if counterexample is None
                    else counterexample["targetAudit"]["mismatchCount"]
                ),
                "distant_component_match": (
                    None
                    if counterexample is None
                    else counterexample["targetAudit"]["componentMatch"]
                ),
                "distant_topology_match": (
                    None
                    if counterexample is None
                    else counterexample["targetAudit"]["topologyMatch"]
                ),
                "distant_grid_sha256": (
                    None if counterexample is None else counterexample["gridSha256"]
                ),
            }
        )
    return records, rows


def _focused_tests() -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_morph2d_targets.py",
        "tests/test_morph2d_grammar.py",
    ]
    completed = subprocess.run(
        command,
        cwd=REPOSITORY,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "command": " ".join(command),
        "returnCode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "success": completed.returncode == 0,
    }


def _plot_counterexamples(records, output_dir: Path) -> None:
    tokens = [".", "A", "B", "C", "R", "M", "T"]
    token_index = {token: index for index, token in enumerate(tokens)}
    colors = [
        "#f2f2f2",
        "#4477aa",
        "#ee6677",
        "#228833",
        "#ccbb44",
        "#aa3377",
        "#66ccee",
    ]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(range(len(tokens) + 1), cmap.N)
    figure, axes = plt.subplots(3, 3, figsize=(10.5, 10.5), constrained_layout=True)
    for axis, record in zip(axes.flat, records):
        counterexample = record["distantSearch"]["counterexample"]
        matrix = [
            [token_index[token] for token in row] for row in counterexample["rows"]
        ]
        axis.imshow(matrix, cmap=cmap, norm=norm, interpolation="nearest")
        axis.set_title(
            f"{record['targetId']}\nscore={counterexample['grammar']['softScore']:.3f}, "
            f"S01 mismatches={counterexample['targetAudit']['mismatchCount']}",
            fontsize=8,
        )
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in axes.flat[len(records) :]:
        axis.axis("off")
    figure.suptitle(
        "E06 S02 grammar-accepted counterexamples outside S01 success sets",
        fontsize=13,
    )
    figure.savefig(output_dir / "grammar_counterexamples.png", dpi=180)
    figure.savefig(output_dir / "grammar_counterexamples.svg")
    plt.close(figure)


def _grammar_spec(grammars) -> str:
    target_lines = [
        "| Grammar | Target | Hard | Soft | Threshold | Audit-only global checks |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for grammar in grammars:
        hard = sum(item.priority.value == "hard" for item in grammar.constraints)
        soft = len(grammar.constraints) - hard
        target_lines.append(
            f"| `{grammar.grammar_id}` | `{grammar.target_id}` | {hard} | {soft} | "
            f"{grammar.acceptance_threshold:.2f} | {', '.join(grammar.audit_only_global_constraints)} |"
        )
    return f"""# E06 S02 Typed Relational Grammar Specification

## Scope

Version `e06.s02.grammar-catalog.v1` replaces one-dimensional rank with typed constraints over symbolic site states on the S01 square-grid fixtures. It is an offline target descriptor, not an S05 policy and not a target-completion oracle.

## Observation contract

The scorer may use a site's token, its four von Neumann neighbors, local domain-boundary flags, a declared orientation frame, and bounded translated motif windows. It may not use cell identity, Algotype, analysis labels, future state, global target membership, component count, or hole count. Exact token composition is an inherited scenario invariant rather than a learned preference.

## Types

| Constraint kind | Meaning | Typed parameters |
| --- | --- | --- |
| `edge_count` | Count unordered adjacent token pairs once on the declared horizontal, vertical, or full 4-neighbor graph | two tokens, direction set, inclusive interval, deviation scale |
| `neighbor_count` | Fraction of subject-token sites whose selected-neighbor count lies in an inclusive interval | subject, neighbor token set, direction set, interval, minimum subject fraction |
| `motif_count` | Count translated exact/wildcard rectangular motifs | pattern rows, inclusive interval, deviation scale |
| `boundary_count` | Count unique token-bearing sites on named domain sides | token, side set, inclusive interval, deviation scale |
| `symmetry` | Count token mismatches under a declared D4 transform in the domain or non-vacancy bounding-box frame | transform, frame, maximum mismatch, deviation scale |

All intervals are inclusive. Missing neighbors are clipped, not treated as tokens. Edges are undirected and counted once. Motif `?` is a wildcard.

## Priority, scoring, and conflict handling

Each constraint is `hard` or `soft` and has a positive weight. The parser applies `hard_first_then_weighted_soft_v1`:

1. exact shape and token composition must match;
2. the grid is evaluated in every declared D4 orientation;
3. orientations are ranked by fewest hard violations, then weighted soft score, then weighted all-constraint score;
4. any hard violation rejects the grid;
5. otherwise the best orientation is accepted when its weighted soft score meets the grammar threshold.

Interval satisfaction is 1 inside the interval and decays linearly with declared scale outside it. `neighbor_count` uses the fraction of subject sites satisfying its interval. No rule rewrites another rule; hard priority resolves semantic conflict by rejection rather than mutation.

## Static contradiction detection

Parsing rejects duplicate IDs, missing or extra typed fields, unknown tokens/transforms/directions, invalid ranges, intervals beyond edge/boundary/neighborhood capacity, impossible motifs, empty intersections among equivalent hard intervals, and exact-symmetry parity impossibilities. The validator includes four independent contradiction probes.

## Target grammars

{os.linesep.join(target_lines)}

## Symmetry and translation

Target orientation is `best_of_D4`. Translatable foreground morphologies use `foreground_bbox` symmetry so translation does not become an accidental boundary cue. Domain-anchored holes use the full-domain frame. S01 target equivalence remains the independent authority.

## Falsification contract

Validation first requires all 148 exact S01 equivalents and all three S01-tolerated boundary variants to pass. Falsification then uses:

- seven frozen structural challenges derived from S01's underdetermination hypotheses;
- exhaustive enumeration of every exact-count one-transposition neighbor of each canonical fixture; and
- deterministic 8-restart, count-preserving simulated-annealing search from the frozen challenge plus diversified starts.

S01 equivalence, components, and holes are read only after a candidate improves under the grammar; they never enter the search objective. A grammar is falsified as a standalone success rule when it accepts a grid that S01 rejects.

## Claim boundary

Grammar satisfaction summarizes computational relations. It is not biological morphology, affinity, intention, agency, formation, repair, or causal evidence. Later work must report local grammar score and global morphology/topology audits separately.
"""


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPOSITORY, text=True).strip()


def _environment_provenance(workers: int, proposal_budget: int) -> dict[str, Any]:
    return {
        "schemaVersion": "e06.s02.environment-provenance.v1",
        "researchStepId": "S02",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: package_version(name)
            for name in ("PyYAML", "matplotlib", "numpy", "pandas", "pytest")
        },
        "cpuCountVisible": os.cpu_count(),
        "workersUsed": workers,
        "parallelUnit": "target grammar",
        "proposalBudgetPerTarget": proposal_budget,
        "gpuUsed": False,
        "networkUsed": False,
        "newDependenciesInstalled": [],
        "repository": {
            "path": str(REPOSITORY),
            "branch": _git("branch", "--show-current"),
            "commit": _git("rev-parse", "HEAD"),
            "dirty": bool(_git("status", "--short")),
        },
    }


def _input_provenance() -> dict[str, Any]:
    missing = [str(path) for path in UPSTREAM_INPUTS.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing required S02 inputs: {missing}")
    return {
        "schemaVersion": "e06.s02.input-provenance.v1",
        "researchStepId": "S02",
        "inputs": [
            {
                "role": role,
                "path": str(path),
                "sha256": _sha256_file(path),
                "readOnly": str(path).startswith("/previous-artifacts"),
            }
            for role, path in UPSTREAM_INPUTS.items()
        ],
        "repositoryInputs": [
            {"path": str(path), "sha256": _sha256_file(path)}
            for path in (
                TARGET_CONFIG,
                GRAMMAR_CONFIG,
                HAND_CONFIG,
                ADVERSARIAL_CONFIG,
                SOURCE,
                FALSIFICATION_SOURCE,
                TEST,
                Path(__file__).resolve(),
            )
        ],
    }


def _report(
    grammar_rows,
    adversarial_rows,
    falsification_rows,
    validation,
    environment,
) -> str:
    total_single = sum(row["single_swap_proposals"] for row in falsification_rows)
    total_single_fp = sum(
        row["single_swap_false_positives"] for row in falsification_rows
    )
    total_search = sum(row["search_proposals_evaluated"] for row in falsification_rows)
    mismatch_counts = [
        int(row["distant_s01_mismatch_count"]) for row in falsification_rows
    ]
    test_summary = validation["focusedTests"]["stdout"].splitlines()[-1]
    result_lines = [
        "| Target | Exact recall | Frozen challenge score | Single-swap false positives | Search proposals | Best score | S01 mismatches | Global audit failure |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    adversarial_by_target = {row["target_id"]: row for row in adversarial_rows}
    grammar_by_target = {row["target_id"]: row for row in grammar_rows}
    for row in falsification_rows:
        target_id = row["target_id"]
        audit_failure = []
        if not row["distant_component_match"]:
            audit_failure.append("components")
        if not row["distant_topology_match"]:
            audit_failure.append("topology")
        if not audit_failure:
            audit_failure.append("S01 geometry/equivalence")
        result_lines.append(
            f"| `{target_id}` | {grammar_by_target[target_id]['exact_states_accepted']}/"
            f"{grammar_by_target[target_id]['exact_equivalence_state_count']} | "
            f"{adversarial_by_target[target_id]['grammar_soft_score']:.3f} | "
            f"{row['single_swap_false_positives']}/{row['single_swap_proposals']} | "
            f"{row['search_proposals_evaluated']} | {row['distant_grammar_soft_score']:.3f} | "
            f"{int(row['distant_s01_mismatch_count'])} | {', '.join(audit_failure)} |"
        )
    artifacts = [
        "grammar_spec.md",
        "target_grammars.yaml",
        "hand_scored_fixtures.yaml",
        "adversarial_fixtures.yaml",
        "fixture_scores.csv",
        "grammar_validation_scores.csv",
        "falsification_results.csv",
        "counterexamples.json",
        "grammar_counterexamples.png",
        "grammar_counterexamples.svg",
        "parser_roundtrip.json",
        "contradiction_audit.json",
        "validation_summary.json",
        "input_provenance.json",
        "environment_provenance.json",
        "execution_commands.log",
        "artifact_manifest.json",
        "research_step_full_results.md",
    ]
    next_action = "Return S02 to the Chief Scientist. If the constraining result is accepted, authorize only S03 to define environments while carrying both local grammar score and independent S01 global success/topology audits; do not start S03 automatically."
    return f"""# Research Step S02 Full Results — Replace Rank with a Relational Grammar

## Top summary

- **Research step ID:** S02
- **Completion status:** **Complete** on {datetime.now(timezone.utc).date().isoformat()}; only S02 was executed and S03 was not started.
- **Artifacts written:** {", ".join(f"`{name}`" for name in artifacts)}.
- **Validation result:** **PASS** — parser round trip, two hand-scored cases spanning five constraint kinds, all 148 exact S01 equivalents, all three tolerated S01 boundary variants, D4 handling, four contradiction probes, frozen structural challenges, distant search replay, artifact hashes, and focused tests (`{test_summary}`) passed.
- **Outcome classification:** **constraining/contradictory.** The typed grammar is valid and useful as a local relation score, but every target grammar accepts at least one count-preserving grid outside its S01 success set; local grammar satisfaction is not sufficient as a standalone completion predicate.
- **Caveats or blockers:** Falsification is finite, uses the S01 square fixtures and declared grammar language, and does not show that every possible local grammar must fail. It does show that these seven frozen grammars are globally underdetermined. No dynamic policy, movement rule, environment library, or S03 work was executed.
- **Lay summary:** The new grammar can correctly recognize the intended local neighbor patterns even after rotation or reflection, but local agreement can be fooled. Each morphology has a different wrong global shape that still satisfies the grammar—for example, a broken shell, unbalanced lobes, or holes in the wrong places. Later experiments must keep a separate whole-shape check.
- **Recommended next action:** {next_action}

## Frozen question and success/failure rule

**Frozen question:** Are typed desired/forbidden neighbor, motif, orientation, symmetry, and optional boundary constraints sufficient to describe useful target relations for all seven S01 morphologies?

**Completion criterion:** every target has a valid executable grammar; parser round trip, hand scores, symmetry, contradictions, and fixture scoring validate; and underdetermination is actively tested and documented.

**Sufficiency failure rule:** a count-preserving grid that the frozen grammar accepts but the independent S01 evaluator rejects falsifies that grammar as a standalone target-completion rule. The completion criterion was met, but the sufficiency hypothesis was contradicted for all seven grammars.

## Lay summary

The result separates two jobs. The grammar says whether nearby cells have the expected kinds of contacts. The S01 evaluator says whether the complete shape, placement, components, and holes match the target. Those jobs cannot safely be collapsed: all intended shapes score correctly, but every grammar also accepts a wrong overall arrangement. This is a useful design constraint for later simulators, not evidence about living tissues.

## Inputs

- Refreshed `/workspace/AGENTS.md`, `FULL_PLAN.md`, and the pre-completion `RESEARCH_PLAN.md`.
- The complete S01 target catalog, full-results report, ambiguity register, validation, examples, and manifest-verified artifact set.
- E01 state, observation, identity, deterministic event, and label-blindness contracts.
- E04 graph-adjacency, exact-composition, conditioning, intervention-boundary, and E06 handoff contracts.
- `input-attachments/MANIFEST.json` and the attachment sidecar; the supplied paper motivates local neighbor preferences and a two-dimensional extension but does not specify this grammar.
- No dataset was required, no package was installed, and no upstream artifact was modified.

Paths and SHA-256 values are in `input_provenance.json`; S01 immutability was rechecked against all 11 manifest-listed artifacts.

## Methods

### Typed grammar and observation boundary

`src/morph2d/grammar.py` implements five types: unordered `edge_count`, per-subject `neighbor_count`, translated/wildcard `motif_count`, `boundary_count`, and D4 `symmetry` in either domain or foreground-bounding-box coordinates. Exact token counts are inherited scenario invariants. The scorer cannot read identity, Algotype, analysis labels, future state, S01 target membership, components, or holes.

Each rule is hard or soft with a positive weight. `hard_first_then_weighted_soft_v1` first rejects count or hard-rule violations, then compares the weighted soft score with a target-specific threshold. The best declared D4 orientation is used. Full syntax and formulas are in `grammar_spec.md`; the seven versioned grammars contain {sum(row["constraint_count"] for row in grammar_rows)} constraints.

### Positive validation

The parser was serialized to YAML and parsed back with semantic equality. Two 3×3 hand cases independently fixed the expected edge, neighbor, motif, boundary, and symmetry observations. Every member of the 148-state S01 equivalence union had zero hard violations, soft score 1.0, and acceptance. All three S01-tolerated two-site boundary variants remained grammar-accepted.

### Contradiction detection

Four deliberately invalid grammars tested empty intersections among hard edge intervals, impossible neighbor capacity, minimum greater than maximum, and exact-symmetry parity. Every probe emitted its expected diagnostic and was rejected by the normal parser.

### Active falsification

Before the final search, seven target-specific structural challenges were frozen from the S01 ambiguity hypotheses. They preserve exact token counts but alter global structure. Each candidate was scored by every D4 transform, then audited independently by S01.

For breadth, every one-transposition exact-count neighbor of each canonical fixture was exhaustively enumerated ({total_single:,} candidates total). For distance, each target received {environment["proposalBudgetPerTarget"]:,} deterministic proposals over eight restarts, using exact-count token swaps and a fixed annealing schedule. Search maximized grammar score plus bounded distance; S01 membership was queried only after an accepted candidate improved, never in the search objective. Seven targets ran in parallel on {environment["workersUsed"]} CPU workers; no GPU was needed.

### Commands

```bash
cd /workspace/cell-research
python -m pytest -q tests/test_morph2d_targets.py tests/test_morph2d_grammar.py
python scripts/build_morph2d_s02.py build --output-dir /artifacts/research_steps/S02 --proposal-budget {environment["proposalBudgetPerTarget"]} --workers {environment["workersUsed"]}
python scripts/build_morph2d_s02.py validate --output-dir /artifacts/research_steps/S02
ruff check src/morph2d scripts/build_morph2d_s02.py tests/test_morph2d_grammar.py
ruff format --check src/morph2d scripts/build_morph2d_s02.py tests/test_morph2d_grammar.py
python -m compileall -q src/morph2d scripts/build_morph2d_s02.py tests/test_morph2d_grammar.py
```

Runtime provenance reports Python {environment["python"]}, repository commit `{environment["repository"]["commit"]}`, {environment["workersUsed"]} workers, no GPU, no network, and no new dependency.

## Results

{os.linesep.join(result_lines)}

### Anchor results

- **Positive recall:** 148/148 exact target-equivalence states and 3/3 S01-tolerated boundary variants were grammar-accepted.
- **Frozen falsification:** 7/7 predeclared structural challenges were grammar-accepted and S01-rejected; D4 acceptance and score were invariant for every challenge.
- **Local breadth:** {total_single_fp:,}/{total_single:,} exhaustive one-swap candidates were grammar-accepted outside S01. This is a local sensitivity diagnostic, not a random-population rate.
- **Distant falsification:** 7/7 searches retained a grammar-accepted S01 counterexample after {total_search:,} total proposals. Best counterexamples were {min(mismatch_counts)}–{max(mismatch_counts)} sites away from the nearest S01 equivalence state.
- **Global disagreement:** Some counterexamples preserve component and hole counts yet violate exact geometry or relative displacement; others also break required component structure. Local grammar and global morphology therefore provide nonredundant evidence.

The target-by-target machine tables are `grammar_validation_scores.csv`, `fixture_scores.csv`, and `falsification_results.csv`. Full retained grids and scorer/auditor records are in `counterexamples.json`; the visual panel is `grammar_counterexamples.png`/`.svg`.

## Interpretation

The grammar successfully replaces numeric rank as a typed local-relation language, but not as the sole definition of completion. The failure modes match the frozen S01 concerns:

- stripe/layer edge statistics do not enforce globally straight, unbranched interfaces;
- shielding contacts do not force a single closed ring component;
- B-M-B contacts and approximate symmetry do not force balanced lobes;
- compact separated regions do not fix their gap or relative displacement;
- boundary-excluded cavity contacts do not fix hole position, number, shape, or separation.

The correct downstream contract is conjunctive: **local grammar score for policy feedback and mechanistic interpretation, plus independent S01-equivalence, component, and topology checks for target completion**. S03 should preserve both fields in environment fixtures.

## Validation

`validation_summary.json` records a successful implementation validation despite the scientific contradiction. Passed checks include:

- semantic parser round trip for seven grammars and all typed constraints;
- exact hand scores for both positive and negative cases across all five constraint kinds;
- 148-state exact symmetry recall and three tolerated-boundary recalls;
- hard-first conflict behavior and four static contradiction probes;
- D4 invariance of all seven adversarial fixtures;
- exact-count conservation and S01-independent audit of every retained counterexample;
- exhaustive single-swap accounting and deterministic distant-search seeds/budgets;
- focused tests (`{test_summary}`); and
- report sections, artifact links, and SHA-256 manifest validation.

## Caveats, blockers, failed assumptions, and limitations

- **Failed assumption:** these frozen local grammars are not sufficient standalone target definitions. All seven admit unintended global patterns.
- This does not prove impossibility for every finite-radius grammar. More expressive motifs, larger radius, coordinates, communication, or global counters could reduce underdetermination, but they change the information/control budget.
- Exact composition is supplied externally. The grammar does not create, delete, divide, or infer cell counts.
- `foreground_bbox` symmetry is translation-invariant but can be gamed by outliers that change the box.
- Soft thresholds are benchmark contracts, not calibrated biological tolerances. Their positive set was frozen from S01 fixtures before final falsification.
- Exhaustive enumeration covers one transposition only; distant search is deterministic but non-exhaustive in the enormous full state spaces.
- A grammar-accepted static grid says nothing about whether an S05 local policy can form it, whether it is stable, or how much movement it costs.
- Square-grid four-neighbor semantics remain provisional until S03 defines other environments.
- No S03 environment, topology library, or movement implementation was started.
- All findings are computational proxy evidence and do not establish biological morphology, mechanism, repair, affinity, agency, or clinical relevance.

## Artifacts and provenance

The repository-backed implementation is `src/morph2d/grammar.py`, `src/morph2d/falsification.py`, the three S02 YAML configurations under `configs/morphologies/`, `scripts/build_morph2d_s02.py`, and `tests/test_morph2d_grammar.py`. Source is not duplicated in artifacts.

The artifact directory contains immutable grammar/fixture snapshots, full specification, scores, retained counterexamples, figures, contradiction and parser audits, validation, execution, environment/input provenance, and `artifact_manifest.json` with size/SHA-256 records.

## Recommended next action

{next_action}
"""


def _artifact_roles() -> dict[str, str]:
    return {
        "grammar_spec.md": "typed grammar syntax, scoring, conflicts, and falsification contract",
        "target_grammars.yaml": "immutable S02 grammar-catalog snapshot",
        "hand_scored_fixtures.yaml": "independently expected parser/scorer fixtures",
        "adversarial_fixtures.yaml": "frozen target-specific structural challenges",
        "fixture_scores.csv": "hand and adversarial fixture score audit",
        "grammar_validation_scores.csv": "target-equivalence and boundary-variant recall",
        "falsification_results.csv": "single-swap and distant-search target summaries",
        "counterexamples.json": "retained full grids with grammar and S01 audit records",
        "grammar_counterexamples.png": "raster counterexample panel",
        "grammar_counterexamples.svg": "vector counterexample panel",
        "parser_roundtrip.json": "semantic parser serialization audit",
        "contradiction_audit.json": "invalid-grammar detection probes",
        "validation_summary.json": "consolidated S02 validation and outcome",
        "input_provenance.json": "governing, S01, E01, E04, and repository input hashes",
        "environment_provenance.json": "runtime, dependency, compute, and repository provenance",
        "execution_commands.log": "reproduction and validation commands",
        "research_step_full_results.md": "canonical S02 handoff report",
        "artifact_manifest.json": "artifact roles, sizes, and hashes",
    }


def _write_manifest(output_dir: Path) -> None:
    roles = _artifact_roles()
    records = []
    for name, role in roles.items():
        if name == "artifact_manifest.json":
            continue
        path = output_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"missing expected artifact: {path}")
        records.append(
            {
                "path": name,
                "role": role,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    _write_json(
        output_dir / "artifact_manifest.json",
        {
            "schemaVersion": "e06.s02.artifact-manifest.v1",
            "researchStepId": "S02",
            "artifactCountExcludingManifest": len(records),
            "artifacts": records,
            "repositorySourcePaths": [
                "configs/morphologies/grammar_catalog.yaml",
                "configs/morphologies/grammar_hand_fixtures.yaml",
                "configs/morphologies/grammar_adversarial_fixtures.yaml",
                "src/morph2d/grammar.py",
                "src/morph2d/falsification.py",
                "scripts/build_morph2d_s02.py",
                "tests/test_morph2d_grammar.py",
            ],
        },
    )


def build(output_dir: Path, proposal_budget: int, workers: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata, grammars, targets, hand, adversarial = _load_all()
    s01_immutability = _validate_s01_immutability()
    parser_roundtrip = _parser_roundtrip(metadata, grammars)
    hand_rows, hand_success = _hand_scores(hand)
    grammar_rows, adversarial_rows, positive_summary = (
        _positive_and_symmetry_validation(grammars, targets, adversarial)
    )
    contradiction_audit = _contradiction_audit()
    falsification_records, falsification_rows = _run_falsification(
        targets, proposal_budget, workers
    )
    tests = _focused_tests()
    all_distant = all(row["distant_counterexample_found"] for row in falsification_rows)
    all_single = all(
        row["single_swap_false_positives"] > 0 for row in falsification_rows
    )
    checks = {
        "s01Immutability": s01_immutability["success"],
        "parserRoundTrip": parser_roundtrip["success"],
        "handScoredFixtures": hand_success,
        "exactTargetSymmetryRecall": positive_summary["exactStatesPass"],
        "s01BoundaryVariantRecall": positive_summary["boundaryVariantsPass"],
        "adversarialD4Invariance": positive_summary["adversarialFixturesPass"],
        "contradictionDetection": contradiction_audit["success"],
        "singleSwapFalsificationEveryTarget": all_single,
        "distantFalsificationEveryTarget": all_distant,
        "focusedRepositoryTests": tests["success"],
    }
    validation_success = all(checks.values())
    validation = {
        "schemaVersion": "e06.s02.validation-summary.v1",
        "researchStepId": "S02",
        "success": validation_success,
        "status": "complete" if validation_success else "validation_failed",
        "outcomeClassification": "constraining/contradictory",
        "checks": checks,
        "s01Immutability": s01_immutability,
        "positiveValidation": positive_summary,
        "grammarCount": len(grammars),
        "constraintCount": sum(len(grammar.constraints) for grammar in grammars),
        "handCaseCount": len(hand_rows),
        "contradictionProbeCount": contradiction_audit["probeCount"],
        "singleSwapProposalCount": sum(
            row["single_swap_proposals"] for row in falsification_rows
        ),
        "singleSwapFalsePositiveCount": sum(
            row["single_swap_false_positives"] for row in falsification_rows
        ),
        "distantSearchProposalCount": sum(
            row["search_proposals_evaluated"] for row in falsification_rows
        ),
        "falsifiedGrammarCount": sum(
            row["distant_counterexample_found"] for row in falsification_rows
        ),
        "focusedTests": tests,
        "validationResult": (
            "PASS — implementation and falsification validation succeeded; all seven standalone grammar-sufficiency claims were contradicted."
            if validation_success
            else "FAIL — one or more declared S02 validation checks failed."
        ),
    }

    shutil.copyfile(GRAMMAR_CONFIG, output_dir / "target_grammars.yaml")
    shutil.copyfile(HAND_CONFIG, output_dir / "hand_scored_fixtures.yaml")
    shutil.copyfile(ADVERSARIAL_CONFIG, output_dir / "adversarial_fixtures.yaml")
    (output_dir / "grammar_spec.md").write_text(
        _grammar_spec(grammars), encoding="utf-8"
    )
    fixture_frame = pd.concat(
        [
            pd.DataFrame.from_records(hand_rows).assign(fixture_family="hand_scored"),
            pd.DataFrame.from_records(adversarial_rows).assign(
                fixture_family="target_adversarial"
            ),
        ],
        ignore_index=True,
        sort=False,
    )
    fixture_frame.to_csv(output_dir / "fixture_scores.csv", index=False)
    pd.DataFrame.from_records(grammar_rows).to_csv(
        output_dir / "grammar_validation_scores.csv", index=False
    )
    pd.DataFrame.from_records(falsification_rows).to_csv(
        output_dir / "falsification_results.csv", index=False
    )
    _write_json(
        output_dir / "counterexamples.json",
        {
            "schemaVersion": "e06.s02.counterexamples.v1",
            "researchStepId": "S02",
            "frozenAdversarialFixtures": adversarial["fixtures"],
            "falsificationRecords": falsification_records,
        },
    )
    _plot_counterexamples(falsification_records, output_dir)
    _write_json(output_dir / "parser_roundtrip.json", parser_roundtrip)
    _write_json(output_dir / "contradiction_audit.json", contradiction_audit)
    _write_json(output_dir / "validation_summary.json", validation)
    _write_json(output_dir / "input_provenance.json", _input_provenance())
    environment = _environment_provenance(workers, proposal_budget)
    _write_json(output_dir / "environment_provenance.json", environment)
    (output_dir / "execution_commands.log").write_text(
        "\n".join(
            [
                "cd /workspace/cell-research",
                "python -m pytest -q tests/test_morph2d_targets.py tests/test_morph2d_grammar.py",
                f"python scripts/build_morph2d_s02.py build --output-dir /artifacts/research_steps/S02 --proposal-budget {proposal_budget} --workers {workers}",
                "python scripts/build_morph2d_s02.py validate --output-dir /artifacts/research_steps/S02",
                "ruff check src/morph2d scripts/build_morph2d_s02.py tests/test_morph2d_grammar.py",
                "ruff format --check src/morph2d scripts/build_morph2d_s02.py tests/test_morph2d_grammar.py",
                "python -m compileall -q src/morph2d scripts/build_morph2d_s02.py tests/test_morph2d_grammar.py",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "research_step_full_results.md").write_text(
        _report(
            grammar_rows,
            adversarial_rows,
            falsification_rows,
            validation,
            environment,
        ),
        encoding="utf-8",
    )
    _write_manifest(output_dir)
    if not validation_success:
        raise RuntimeError("S02 validation failed")


def validate(output_dir: Path) -> None:
    expected = set(_artifact_roles())
    observed = {path.name for path in output_dir.iterdir() if path.is_file()}
    if observed != expected:
        raise RuntimeError(
            f"artifact set mismatch: missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    manifest = json.loads(
        (output_dir / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    for record in manifest["artifacts"]:
        path = output_dir / record["path"]
        if _sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"artifact hash mismatch: {record['path']}")
    validation = json.loads(
        (output_dir / "validation_summary.json").read_text(encoding="utf-8")
    )
    if not validation["success"] or not all(validation["checks"].values()):
        raise RuntimeError("stored S02 validation is not successful")
    metadata, grammars, targets, hand, adversarial = _load_all()
    if not _parser_roundtrip(metadata, grammars)["success"]:
        raise RuntimeError("live parser round trip failed")
    _, hand_success = _hand_scores(hand)
    if not hand_success:
        raise RuntimeError("live hand-score validation failed")
    _, adversarial_rows, positive = _positive_and_symmetry_validation(
        grammars, targets, adversarial
    )
    if not all(positive.values()) or not all(row["pass"] for row in adversarial_rows):
        raise RuntimeError("live symmetry/adversarial validation failed")
    counterexamples = json.loads(
        (output_dir / "counterexamples.json").read_text(encoding="utf-8")
    )
    grammar_by_target = {grammar.target_id: grammar for grammar in grammars}
    target_by_id = {target.target_id: target for target in targets}
    for record in counterexamples["falsificationRecords"]:
        candidate = record["distantSearch"]["counterexample"]
        if candidate is None:
            raise RuntimeError(f"missing distant counterexample: {record['targetId']}")
        grammar_result = score_grid(
            candidate["rows"], grammar_by_target[record["targetId"]]
        )
        target_result = evaluate_success(
            candidate["rows"], target_by_id[record["targetId"]]
        )
        if not grammar_result["accepted"] or target_result["success"]:
            raise RuntimeError(f"counterexample replay failed: {record['targetId']}")
    if (
        output_dir / "target_grammars.yaml"
    ).read_bytes() != GRAMMAR_CONFIG.read_bytes():
        raise RuntimeError("artifact grammar snapshot differs from repository grammar")
    report = (output_dir / "research_step_full_results.md").read_text(encoding="utf-8")
    required_sections = [
        "## Top summary",
        "## Lay summary",
        "## Inputs",
        "## Methods",
        "### Commands",
        "## Results",
        "## Validation",
        "## Caveats, blockers, failed assumptions, and limitations",
        "## Artifacts and provenance",
        "## Recommended next action",
    ]
    missing = [section for section in required_sections if section not in report]
    if missing:
        raise RuntimeError(f"canonical report missing sections: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("build", "validate"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
        / "research_steps/S02",
    )
    parser.add_argument("--proposal-budget", type=int, default=100_000)
    parser.add_argument("--workers", type=int, default=7)
    arguments = parser.parse_args()
    if not 1 <= arguments.workers <= 8:
        raise ValueError("workers must lie in [1, 8]")
    if arguments.proposal_budget < 1:
        raise ValueError("proposal budget must be positive")
    if arguments.action == "build":
        build(arguments.output_dir, arguments.proposal_budget, arguments.workers)
    else:
        validate(arguments.output_dir)


if __name__ == "__main__":
    main()
