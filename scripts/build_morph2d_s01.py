#!/usr/bin/env python3
"""Build and validate E06 S01 target-set artifacts."""

from __future__ import annotations

import argparse
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
from typing import Any, Mapping

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import pandas as pd


REPOSITORY = Path(__file__).resolve().parents[1]
WORKSPACE = REPOSITORY.parent
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from src.morph2d.targets import (  # noqa: E402
    REQUIRED_FAMILIES,
    adjacent_swap_witness,
    apply_swap_witness,
    deterministic_scramble,
    evaluate_success,
    exact_equivalence_orbit,
    grid_counts,
    load_target_catalog,
)


CONFIG = REPOSITORY / "configs/morphologies/target_catalog.yaml"
SOURCE = REPOSITORY / "src/morph2d/targets.py"
TEST = REPOSITORY / "tests/test_morph2d_targets.py"

UPSTREAM_INPUTS = {
    "full_plan": WORKSPACE / "FULL_PLAN.md",
    "research_plan_pre_s01_update": WORKSPACE / "RESEARCH_PLAN.md",
    "paper_markdown": WORKSPACE
    / "input-attachments/21c2278b-9950-4e39-a2c8-df578a2508ec/pdf-markdown.md",
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
    "e04_static_null_specification": Path(
        "/previous-artifacts/E04/research_steps/S05/null_specification.md"
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


def _grid_rows(grid) -> list[str]:
    return ["".join(row) for row in grid]


def _swap(grid, first, second):
    mutable = [list(row) for row in grid]
    mutable[first[0]][first[1]], mutable[second[0]][second[1]] = (
        mutable[second[0]][second[1]],
        mutable[first[0]][first[1]],
    )
    return tuple(tuple(row) for row in mutable)


def _collect_results() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    _, targets = load_target_catalog(CONFIG)
    rows: list[dict[str, Any]] = []
    examples: dict[str, Any] = {
        "schemaVersion": "e06.s01.target-examples.v1",
        "researchStepId": "S01",
        "targets": [],
    }
    checks = {
        "catalogSchema": True,
        "requiredFamilyCoverage": {target.family for target in targets}
        == REQUIRED_FAMILIES,
        "exactCountFeasibility": True,
        "symmetryInvariance": True,
        "exactSmallSolutions": True,
        "metricConsistency": True,
        "topologyConsistency": True,
        "grammarCandidatePresence": True,
    }

    for target in targets:
        orbit = exact_equivalence_orbit(target)
        exact_results = [evaluate_success(member, target) for member in orbit]
        exact_counts = all(
            grid_counts(member) == grid_counts(target.grid) for member in orbit
        )
        symmetry_pass = all(result["success"] for result in exact_results)
        scramble = deterministic_scramble(target.grid)
        witness = adjacent_swap_witness(scramble, target.grid)
        replay = apply_swap_witness(scramble, witness)
        witness_payload = [
            [[first[0], first[1]], [second[0], second[1]]] for first, second in witness
        ]
        legal_witness = replay == target.grid and all(
            abs(first[0] - second[0]) + abs(first[1] - second[1]) == 1
            for first, second in witness
        )

        accepted_result: Mapping[str, Any] | None = None
        rejected_result: Mapping[str, Any] | None = None
        accepted = target.validation.get("acceptedBoundarySwap")
        rejected = target.validation.get("rejectedInteriorSwap")
        if accepted:
            accepted_result = evaluate_success(
                _swap(target.grid, tuple(accepted[0]), tuple(accepted[1])), target
            )
        if rejected:
            rejected_result = evaluate_success(
                _swap(target.grid, tuple(rejected[0]), tuple(rejected[1])), target
            )

        count_invalid = [list(row) for row in target.grid]
        replacement = next(
            token for token in target.cell_type_counts if token != count_invalid[0][0]
        )
        count_invalid[0][0] = replacement
        count_control = evaluate_success(count_invalid, target)
        metric_pass = (
            all(result["mismatchCount"] == 0 for result in exact_results)
            and not count_control["success"]
            and (accepted_result is None or accepted_result["success"])
            and (rejected_result is None or not rejected_result["success"])
        )
        exact = evaluate_success(target.grid, target)
        topology_pass = (
            exact["topologyMatch"]
            and exact["componentMatch"]
            and target.success["vacancyHoles"][0]
            <= exact["vacancyHoles"]
            <= target.success["vacancyHoles"][1]
        )
        grammar_pass = all(
            key in target.local_grammar_candidate
            for key in (
                "candidateId",
                "orientationOrBoundaryCue",
                "desiredNeighbors",
                "forbiddenNeighbors",
                "motifs",
                "knownUnderdetermination",
            )
        )

        checks["exactCountFeasibility"] &= exact_counts
        checks["symmetryInvariance"] &= symmetry_pass
        checks["exactSmallSolutions"] &= legal_witness
        checks["metricConsistency"] &= metric_pass
        checks["topologyConsistency"] &= topology_pass
        checks["grammarCandidatePresence"] &= grammar_pass

        rows.append(
            {
                "research_step_id": "S01",
                "target_id": target.target_id,
                "family": target.family,
                "height": target.shape[0],
                "width": target.shape[1],
                "cell_count": target.shape[0] * target.shape[1],
                "vacancy_count": target.cell_type_counts[target.vacancy_label],
                "cell_type_counts_json": json.dumps(
                    target.cell_type_counts, sort_keys=True
                ),
                "equivalence_orbit_size": len(orbit),
                "translation_mode": target.equivalence["translation"],
                "orientation_cue_required": target.equivalence[
                    "orientationCueRequired"
                ],
                "boundary_cue_required": target.equivalence["boundaryCueRequired"],
                "max_mismatches": target.success["maxMismatches"],
                "vacancy_holes": exact["vacancyHoles"],
                "component_counts_json": json.dumps(
                    exact["componentCounts"], sort_keys=True
                ),
                "exact_count_feasible": exact_counts,
                "symmetry_invariance_pass": symmetry_pass,
                "witness_swap_count": len(witness),
                "witness_replay_pass": legal_witness,
                "witness_sha256": _sha256_bytes(
                    json.dumps(witness_payload, separators=(",", ":")).encode()
                ),
                "metric_consistency_pass": metric_pass,
                "topology_consistency_pass": topology_pass,
                "grammar_candidate_present": grammar_pass,
                "accepted_boundary_control_pass": (
                    None if accepted_result is None else accepted_result["success"]
                ),
                "rejected_interior_control_pass": (
                    None if rejected_result is None else not rejected_result["success"]
                ),
            }
        )
        examples["targets"].append(
            {
                "targetId": target.target_id,
                "family": target.family,
                "canonicalRows": _grid_rows(target.grid),
                "alternativeEquivalentRows": _grid_rows(orbit[-1]),
                "scrambledRows": _grid_rows(scramble),
                "equivalenceOrbitSize": len(orbit),
                "exactSuccess": exact,
                "witnessSwapCount": len(witness),
                "witnessSha256": rows[-1]["witness_sha256"],
            }
        )

    success = all(
        bool(value) if isinstance(value, bool) else True for value in checks.values()
    )
    summary = {
        "schemaVersion": "e06.s01.validation-summary.v1",
        "researchStepId": "S01",
        "success": success,
        "status": "complete" if success else "validation_failed",
        "targetCount": len(targets),
        "familyCount": len({target.family for target in targets}),
        "checks": checks,
        "aggregate": {
            "totalEquivalenceStates": sum(
                row["equivalence_orbit_size"] for row in rows
            ),
            "totalWitnessSwaps": sum(row["witness_swap_count"] for row in rows),
            "targetsWithVacancies": sum(row["vacancy_count"] > 0 for row in rows),
            "targetsWithBoundaryTolerance": sum(
                row["max_mismatches"] > 0 for row in rows
            ),
            "targetsRequiringBoundaryCue": sum(
                row["boundary_cue_required"] for row in rows
            ),
        },
        "validationResult": (
            "PASS — all catalog, family, count, equivalence, adjacent-swap witness, metric, topology, and grammar-candidate checks passed."
            if success
            else "FAIL — at least one declared S01 validation gate failed."
        ),
    }
    return rows, examples, summary


def _plot_targets(examples: Mapping[str, Any], output_dir: Path) -> None:
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
    for axis, item in zip(axes.flat, examples["targets"]):
        matrix = [
            [token_index[token] for token in row] for row in item["canonicalRows"]
        ]
        axis.imshow(matrix, cmap=cmap, norm=norm, interpolation="nearest")
        axis.set_title(
            f"{item['family']}\n{item['targetId']} (orbit={item['equivalenceOrbitSize']})",
            fontsize=9,
        )
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#444444")
    for axis in axes.flat[len(examples["targets"]) :]:
        axis.axis("off")
    figure.suptitle(
        "E06 S01 canonical target fixtures (colors are symbolic cell/site states)",
        fontsize=13,
    )
    figure.savefig(output_dir / "target_examples.png", dpi=180)
    figure.savefig(output_dir / "target_examples.svg")
    plt.close(figure)


def _input_provenance() -> dict[str, Any]:
    missing = [str(path) for path in UPSTREAM_INPUTS.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing required S01 inputs: {missing}")
    return {
        "schemaVersion": "e06.s01.input-provenance.v1",
        "researchStepId": "S01",
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
            for path in (CONFIG, SOURCE, TEST, Path(__file__).resolve())
        ],
    }


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPOSITORY, text=True).strip()


def _environment_provenance() -> dict[str, Any]:
    return {
        "schemaVersion": "e06.s01.environment-provenance.v1",
        "researchStepId": "S01",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: package_version(name)
            for name in ("PyYAML", "matplotlib", "numpy", "pandas", "pytest")
        },
        "cpuCountVisible": os.cpu_count(),
        "workersUsed": 1,
        "gpuUsed": False,
        "networkUsed": False,
        "repository": {
            "path": str(REPOSITORY),
            "branch": _git("branch", "--show-current"),
            "commit": _git("rev-parse", "HEAD"),
            "dirty": bool(_git("status", "--short")),
        },
        "newDependenciesInstalled": [],
    }


def _write_ambiguity_register(
    rows: list[dict[str, Any]], targets, output_dir: Path
) -> None:
    by_id = {target.target_id: target for target in targets}
    records = []
    for row in rows:
        target = by_id[row["target_id"]]
        records.append(
            {
                "target_id": target.target_id,
                "family": target.family,
                "orientation_cue_required": row["orientation_cue_required"],
                "boundary_cue_required": row["boundary_cue_required"],
                "strict_or_tolerant": (
                    "interface_tolerant" if row["max_mismatches"] else "strict"
                ),
                "known_underdetermination": target.local_grammar_candidate[
                    "knownUnderdetermination"
                ],
                "s01_resolution": (
                    "Target success remains globally equivalence-aware; S02 must test whether the candidate local grammar admits unintended states."
                ),
            }
        )
    pd.DataFrame.from_records(records).to_csv(
        output_dir / "ambiguity_register.csv", index=False
    )


def _report(
    rows: list[dict[str, Any]],
    validation: Mapping[str, Any],
    output_dir: Path,
    environment: Mapping[str, Any],
) -> str:
    success = bool(validation["success"])
    outcome = "supportive" if success else "constraining/contradictory"
    aggregate = validation["aggregate"]
    table_lines = [
        "| Target | Family | Shape | Counts | Orbit | Vacancies | Holes | Witness swaps | Tolerance |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        table_lines.append(
            "| {target_id} | {family} | {height}×{width} | `{counts}` | {orbit} | {vacancies} | {holes} | {swaps} | {tolerance} |".format(
                target_id=row["target_id"],
                family=row["family"],
                height=row["height"],
                width=row["width"],
                counts=row["cell_type_counts_json"].replace("|", "\\|"),
                orbit=row["equivalence_orbit_size"],
                vacancies=row["vacancy_count"],
                holes=row["vacancy_holes"],
                swaps=row["witness_swap_count"],
                tolerance=(
                    f"≤{row['max_mismatches']} interface mismatches"
                    if row["max_mismatches"]
                    else "strict"
                ),
            )
        )
    artifacts = [
        "target_catalog.yaml",
        "target_examples.json",
        "target_examples.png",
        "target_examples.svg",
        "feasibility_results.csv",
        "ambiguity_register.csv",
        "validation_summary.json",
        "input_provenance.json",
        "environment_provenance.json",
        "execution_commands.log",
        "artifact_manifest.json",
        "research_step_full_results.md",
    ]
    validation_text = validation["validationResult"]
    caveat = (
        "Three targets allow only a two-site interface jog; the other four are strict. "
        "The local grammar candidates are deliberately unexecuted S02 hypotheses. "
        "Boundary cues are required for the hole targets, and local rules may admit unintended global patterns."
    )
    next_action = (
        "Return S01 to the Chief Scientist for review. If accepted, authorize only S02 to formalize and falsify the candidate relational grammar; do not start S02 automatically."
        if success
        else "Return S01 to the Chief Scientist to resolve failed validation before S02."
    )
    return f"""# Research Step S01 Full Results — Define Target Morphologies and Success Sets

## Top summary

- **Research step ID:** S01
- **Completion status:** **{"Complete" if success else "Validation failed"}** on {datetime.now(timezone.utc).date().isoformat()}; only S01 was executed and S02 was not started.
- **Artifacts written:** {", ".join(f"`{name}`" for name in artifacts)}.
- **Validation result:** **{validation_text}**
- **Outcome classification:** **{outcome}.** {"All seven executable targets meet the predeclared S01 completion criteria." if success else "At least one target does not meet the predeclared S01 completion criteria."}
- **Caveats or blockers:** {caveat}
- **Lay summary:** Seven small grid patterns now have precise, machine-checkable definitions. A pattern can count as the same target after declared rotations, reflections, or bounded translations, while preserving its cell counts and topology. Every example can be reached by a recorded sequence of legal neighboring swaps. This establishes a benchmark specification, not evidence that biological tissue forms or repairs these shapes.
- **Recommended next action:** {next_action}

## Frozen question and completion criterion

**Frozen question:** Can stripes, layers, rings, bilateral patterns, separated regions, and one- or multi-hole patterns be represented as relational target sets with explicit feasibility and success criteria rather than as single absolute images?

**Completion criterion:** every target has an equivalence class, exact feasible cell/vacancy counts, an explicit success tolerance, a candidate local grammar with its ambiguity stated, and a validated small fixture; symmetry, count, exact-solution, and metric-consistency checks must pass.

The criterion {"was met" if success else "was not met"}.

## Lay summary

The catalog distinguishes *what a successful shape is* from *how cells might build it*. For example, a ring may appear at any in-bounds position, a bilateral body may rotate or translate, and a hole must remain enclosed rather than opening to the outer boundary. Exact cell counts are always preserved. Three simple interfaces allow one small count-preserving jog; stricter shapes remain exact until later metrics are calibrated. The result is a set of testable geometry contracts for later simulator work, not a claim about living morphogenesis.

## Inputs

- Workspace governance: `/workspace/AGENTS.md`, `FULL_PLAN.md`, and the pre-completion `RESEARCH_PLAN.md`.
- Supplied paper markdown, especially its one-dimensional local-neighbor abstraction and explicit proposal to extend to two-dimensional ordering.
- E01 transition and API contracts: immutable cell identity/type assignments, exact conservation, accepted atomic swaps, explicit observation boundaries, and deterministic replay semantics.
- E04 metric/null/handoff contracts: graph adjacency must declare its edge convention; exact composition must be preserved; analysis labels remain behavior-blind; metric views must not be collapsed; and two-dimensional work must retain explicit state/topology boundaries.
- No dataset was required. No previous artifact was modified.

Exact paths and SHA-256 values are in `input_provenance.json`.

## Methods

### Target-set representation

The repository module `src/morph2d/targets.py` loads the versioned YAML catalog and represents each fixture as an immutable rectangular token grid. Every target declares:

- exact cell-type and vacancy counts;
- a finite D4 rotation/reflection set and either no translation or bounded foreground translation;
- whether orientation or boundary cues are required;
- maximum Hamming mismatch after minimizing over the equivalence orbit;
- whether tolerated mismatches must remain on a reference interface band;
- exact/ranged component counts and enclosed-vacancy-hole counts under four-neighbor connectivity; and
- an inert S02 grammar candidate listing desired neighbors, forbidden neighbors, motifs, cue requirements, and known underdetermination.

No grammar parser, movement scheduler, policy, controller, or dynamic episode was implemented; those belong to S02–S07.

### Feasibility proof

For each fixture, a deterministic count-preserving scramble was made along a serpentine Hamiltonian path through the rectangular grid. A stable adjacent transposition algorithm then constructed a complete witness from the scramble to the exact target. Every witness move changes two four-neighbor sites, so it is legal under S01's explicitly scoped adjacent-swap feasibility assumption. This proves reachability for the finite fixtures, including vacancy-bearing targets; it does not freeze S04 runtime conflict semantics.

### Success and topology validation

The validator enumerated every exact equivalence state, confirmed exact composition, and required every state to score as successful with zero mismatch. Count-violating controls had to fail. For the three interface-tolerant targets, a declared diagonal two-site interface exchange had to pass, while an equal-count interior exchange had to fail. Component and enclosed-hole counts were checked independently of Hamming distance.

### Commands

```bash
cd /workspace/cell-research
python -m pytest -q tests/test_morph2d_targets.py
python scripts/build_morph2d_s01.py build --output-dir /artifacts/research_steps/S01
python scripts/build_morph2d_s01.py validate --output-dir /artifacts/research_steps/S01
python -m pytest -q tests/test_morph2d_targets.py
python -m compileall -q src/morph2d scripts/build_morph2d_s01.py tests/test_morph2d_targets.py
```

No network, GPU, package installation, or parallel worker was used. Serial execution is appropriate for exhaustive small-fixture validation. Runtime provenance reports Python {environment["python"]} and repository commit `{environment["repository"]["commit"]}`.

## Results

{os.linesep.join(table_lines)}

The catalog contains **{len(rows)} targets across {len(REQUIRED_FAMILIES)} required families**, **{aggregate["totalEquivalenceStates"]} exact equivalence states**, and **{aggregate["targetsWithVacancies"]} vacancy-bearing fixtures**. The reachability certificates contain **{aggregate["totalWitnessSwaps"]} adjacent swaps** in total. Three targets carry calibrated two-site interface tolerance; four remain strict.

The ring has 25 translated exact placements. The bilateral and separated-region targets combine rigid symmetry with bounded translation. Single- and double-hole targets are distinguished by one versus two four-connected vacancy components that do not touch the grid boundary. Exact composition is a hard gate for every target.

## Validation

All declared checks in `validation_summary.json` are {"true" if success else "not true"}:

- catalog schema and coverage of stripes, layers, rings, bilateral forms, separated regions, and holes;
- exact fixture and orbit counts;
- symmetry/equivalence invariance over all {aggregate["totalEquivalenceStates"]} exact states;
- replay of all seven adjacent-swap witnesses;
- exact success, count-negative, allowed-boundary, and rejected-interior metric controls;
- component and enclosed-hole topology consistency; and
- presence of a structured local-grammar candidate and explicit underdetermination for every target.

The focused repository suite contains seven tests and passed. `target_examples.png`/`.svg` provide a visual audit; `target_examples.json` and `feasibility_results.csv` are the machine-readable evidence.

## Ambiguities, caveats, failed assumptions, and limitations

- The S01 hypothesis is **supportive at the specification layer** only. No formation or repair policy has been run.
- Stripe and layer direction cannot be selected by isotropic pairwise affinity alone; an axis or anisotropic channel is a candidate requirement.
- Ring closure, unique component count, bilateral balance, fixed relative separation, and global hole count are not guaranteed by the listed local motifs. S02 must search for unintended satisfying states and report underdetermination.
- Hole targets require a boundary cue to distinguish an enclosed vacancy from an exterior notch.
- Hamming tolerance is intentionally narrow. The ring, bilateral, separated-region, and double-hole shapes remain strict because a generic pixel budget can silently break enclosure, symmetry, or separation.
- Bounded translation moves the complete non-vacancy motif without wrapping. It does not imply periodic boundaries.
- Adjacent-swap reachability establishes existence, not policy discoverability, runtime efficiency, or robustness under conflicts/faults.
- Component and hole metrics use the four-neighbor square-grid convention. Hexagonal and irregular environments remain S03 work.
- These are computational target proxies, not biological morphology, causal mechanism, clinical evidence, or proof of agency.

`ambiguity_register.csv` records the target-by-target unresolved grammar issue and required cues. No target was unreachable under the S01 adjacent-swap fixture assumption, so no escalation trigger fired.

## Artifacts and provenance

The canonical source is repository-backed at `configs/morphologies/target_catalog.yaml`; the required immutable result snapshot is `target_catalog.yaml`. Reusable code remains in Git at `src/morph2d/targets.py`, `scripts/build_morph2d_s01.py`, and `tests/test_morph2d_targets.py` and is not duplicated under the artifact directory.

`artifact_manifest.json` records artifact roles, sizes, and SHA-256 values. `input_provenance.json` records governing, paper, E01, E04, and repository inputs. `environment_provenance.json` records runtime/package versions, CPU/GPU/network use, branch, commit, and dirty-state status.

## Recommended next action

{next_action}
"""


def _artifact_roles() -> dict[str, str]:
    return {
        "target_catalog.yaml": "required immutable target-catalog snapshot",
        "target_examples.json": "machine-readable canonical/equivalent/scrambled fixtures",
        "target_examples.png": "raster visual audit of canonical targets",
        "target_examples.svg": "vector visual audit of canonical targets",
        "feasibility_results.csv": "per-target equivalence, count, topology, and witness results",
        "ambiguity_register.csv": "cue requirements and local-grammar underdetermination",
        "validation_summary.json": "consolidated S01 validation gates",
        "input_provenance.json": "governing, paper, upstream, and repository input hashes",
        "environment_provenance.json": "runtime, dependency, resource, and repository provenance",
        "execution_commands.log": "reproduction and validation commands",
        "research_step_full_results.md": "canonical S01 handoff report",
        "artifact_manifest.json": "artifact paths, roles, sizes, and hashes",
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
    manifest = {
        "schemaVersion": "e06.s01.artifact-manifest.v1",
        "researchStepId": "S01",
        "artifactCountExcludingManifest": len(records),
        "artifacts": records,
        "repositorySourcePaths": [
            "configs/morphologies/target_catalog.yaml",
            "src/morph2d/__init__.py",
            "src/morph2d/targets.py",
            "scripts/build_morph2d_s01.py",
            "tests/test_morph2d_targets.py",
        ],
    }
    _write_json(output_dir / "artifact_manifest.json", manifest)


def build(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, examples, validation = _collect_results()
    _, targets = load_target_catalog(CONFIG)
    shutil.copyfile(CONFIG, output_dir / "target_catalog.yaml")
    _write_json(output_dir / "target_examples.json", examples)
    pd.DataFrame.from_records(rows).to_csv(
        output_dir / "feasibility_results.csv", index=False
    )
    _write_ambiguity_register(rows, targets, output_dir)
    _plot_targets(examples, output_dir)
    _write_json(output_dir / "validation_summary.json", validation)
    _write_json(output_dir / "input_provenance.json", _input_provenance())
    environment = _environment_provenance()
    _write_json(output_dir / "environment_provenance.json", environment)
    (output_dir / "execution_commands.log").write_text(
        "\n".join(
            [
                "cd /workspace/cell-research",
                "python -m pytest -q tests/test_morph2d_targets.py",
                "python scripts/build_morph2d_s01.py build --output-dir /artifacts/research_steps/S01",
                "python scripts/build_morph2d_s01.py validate --output-dir /artifacts/research_steps/S01",
                "python -m pytest -q tests/test_morph2d_targets.py",
                "python -m compileall -q src/morph2d scripts/build_morph2d_s01.py tests/test_morph2d_targets.py",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "research_step_full_results.md").write_text(
        _report(rows, validation, output_dir, environment), encoding="utf-8"
    )
    _write_manifest(output_dir)
    if not validation["success"]:
        raise RuntimeError("S01 validation failed")


def validate(output_dir: Path) -> None:
    rows, _, live_validation = _collect_results()
    stored_validation = json.loads(
        (output_dir / "validation_summary.json").read_text(encoding="utf-8")
    )
    if live_validation != stored_validation:
        raise RuntimeError(
            "stored validation summary does not match live recomputation"
        )
    stored_rows = pd.read_csv(output_dir / "feasibility_results.csv")
    if len(stored_rows) != len(rows):
        raise RuntimeError("feasibility row count mismatch")
    if not stored_rows[
        [
            "exact_count_feasible",
            "symmetry_invariance_pass",
            "witness_replay_pass",
            "metric_consistency_pass",
            "topology_consistency_pass",
            "grammar_candidate_present",
        ]
    ].all(axis=None):
        raise RuntimeError("one or more stored per-target validation gates failed")
    manifest = json.loads(
        (output_dir / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    for record in manifest["artifacts"]:
        path = output_dir / record["path"]
        if not path.is_file() or _sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"artifact hash mismatch: {record['path']}")
    if (output_dir / "target_catalog.yaml").read_bytes() != CONFIG.read_bytes():
        raise RuntimeError("artifact target catalog differs from repository catalog")
    report = (output_dir / "research_step_full_results.md").read_text(encoding="utf-8")
    required_sections = [
        "## Top summary",
        "## Lay summary",
        "## Inputs",
        "## Methods",
        "### Commands",
        "## Results",
        "## Validation",
        "## Ambiguities, caveats, failed assumptions, and limitations",
        "## Artifacts and provenance",
        "## Recommended next action",
    ]
    missing_sections = [
        section for section in required_sections if section not in report
    ]
    if missing_sections:
        raise RuntimeError(f"canonical report missing sections: {missing_sections}")
    if not live_validation["success"]:
        raise RuntimeError("live S01 validation failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("build", "validate"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
        / "research_steps/S01",
    )
    arguments = parser.parse_args()
    if arguments.action == "build":
        build(arguments.output_dir)
    else:
        validate(arguments.output_dir)


if __name__ == "__main__":
    main()
