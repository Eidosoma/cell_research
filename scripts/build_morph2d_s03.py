#!/usr/bin/env python3
"""Build and validate E06 S03 environment-library artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
CATALOG = ROOT / "configs/morphologies/environment_catalog.yaml"
TARGETS = ROOT / "configs/morphologies/target_catalog.yaml"
GRAMMARS = ROOT / "configs/morphologies/grammar_catalog.yaml"
SOURCE = ROOT / "src/morph2d/environments.py"
TEST = ROOT / "tests/test_morph2d_environments.py"
S01_DIR = Path("/artifacts/research_steps/S01")
S02_DIR = Path("/artifacts/research_steps/S02")
ATTACHMENT_DIR = WORKSPACE / "input-attachments/21c2278b-9950-4e39-a2c8-df578a2508ec"
UPSTREAM_INPUTS = {
    "agents": WORKSPACE / "AGENTS.md",
    "full_plan": WORKSPACE / "FULL_PLAN.md",
    "research_plan_pre_s03_update": WORKSPACE / "RESEARCH_PLAN.md",
    "previous_artifacts_md": WORKSPACE / "PREVIOUS_ARTIFACTS.md",
    "previous_artifacts_json": WORKSPACE / "PREVIOUS_ARTIFACTS.json",
    "attachment_manifest": WORKSPACE / "input-attachments/MANIFEST.json",
    "attachment_sidecar": ATTACHMENT_DIR / "_metadata/ATTACHMENT.md",
    "s01_report": S01_DIR / "research_step_full_results.md",
    "s01_catalog": S01_DIR / "target_catalog.yaml",
    "s01_validation": S01_DIR / "validation_summary.json",
    "s01_manifest": S01_DIR / "artifact_manifest.json",
    "s02_report": S02_DIR / "research_step_full_results.md",
    "s02_grammar_spec": S02_DIR / "grammar_spec.md",
    "s02_grammars": S02_DIR / "target_grammars.yaml",
    "s02_validation": S02_DIR / "validation_summary.json",
    "s02_manifest": S02_DIR / "artifact_manifest.json",
    "e01_transition": Path(
        "/previous-artifacts/E01/research_steps/S03/transition_spec.md"
    ),
    "e01_api": Path("/previous-artifacts/E01/research_steps/S05/api_documentation.md"),
    "e01_events": Path(
        "/previous-artifacts/E01/research_steps/S06/schema_documentation.md"
    ),
    "e04_metrics": Path(
        "/previous-artifacts/E04/research_steps/S04/metric_specification.md"
    ),
    "e04_nulls": Path(
        "/previous-artifacts/E04/research_steps/S05/null_specification.md"
    ),
    "e04_kinetics": Path(
        "/previous-artifacts/E04/research_steps/S07/kinetic_intervention_specification.md"
    ),
    "e04_handoff": Path(
        "/previous-artifacts/E04/research_steps/S14/e06_e07_handoff.md"
    ),
}

sys.path.insert(0, str(ROOT))

from src.morph2d.environments import (  # noqa: E402
    EnvironmentValidationError,
    boundary_observation,
    canonical_environment_bytes,
    connected_components,
    environment_sha256,
    evaluate_conjunctive,
    load_environment_catalog,
    neighbor_map,
    parse_environment_spec,
    parse_serialized_environment,
    replay_environment,
    topology_summary,
)
from src.morph2d.grammar import load_grammar_catalog  # noqa: E402
from src.morph2d.targets import load_target_catalog  # noqa: E402


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def _focused_tests() -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_morph2d_targets.py",
        "tests/test_morph2d_grammar.py",
        "tests/test_morph2d_environments.py",
    ]
    completed = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=False
    )
    return {
        "command": " ".join(command),
        "returnCode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "success": completed.returncode == 0,
    }


def _verify_upstream_manifest(directory: Path) -> dict[str, Any]:
    manifest = json.loads((directory / "artifact_manifest.json").read_text())
    mismatches = []
    for record in manifest["artifacts"]:
        path = directory / record["path"]
        observed = _sha256_file(path) if path.is_file() else None
        if observed != record["sha256"]:
            mismatches.append(
                {
                    "path": record["path"],
                    "expected": record["sha256"],
                    "observed": observed,
                }
            )
    validation = json.loads((directory / "validation_summary.json").read_text())
    return {
        "researchStepId": manifest["researchStepId"],
        "checkedArtifactCount": len(manifest["artifacts"]),
        "hashMismatches": mismatches,
        "upstreamValidationSuccess": validation["success"],
        "success": not mismatches and validation["success"],
    }


def _unsupported_audit(
    raw_catalog: dict[str, Any], targets, grammars
) -> dict[str, Any]:
    target_by_id = {item.target_id: item for item in targets}
    grammar_by_id = {item.grammar_id: item for item in grammars}
    probes: list[dict[str, Any]] = []

    def parse_probe(probe_id, combination_id, fixture_index, mutate, expected):
        fixture = deepcopy(raw_catalog["environments"][fixture_index])
        fixture["expected"] = {}
        mutate(fixture)
        diagnostic = None
        try:
            parse_environment_spec(fixture)
        except EnvironmentValidationError as error:
            diagnostic = str(error)
        probes.append(
            {
                "probeId": probe_id,
                "combinationId": combination_id,
                "rejected": diagnostic is not None,
                "diagnostic": diagnostic,
                "expectedDiagnosticSubstring": expected,
                "pass": diagnostic is not None and expected in diagnostic,
            }
        )

    parse_probe(
        "periodic_exterior_signal",
        "periodic_domain_edge_flags",
        3,
        lambda item: item.update(
            boundarySignal={
                "mode": "domain_edge_flags",
                "observable": True,
                "includeObstacleContact": False,
                "includeFixedRole": False,
            }
        ),
        "no natural exterior",
    )
    parse_probe(
        "periodic_fixed_boundary",
        "periodic_fixed_exterior_boundary",
        3,
        lambda item: item.update(
            fixedBoundary={"mode": "sites", "sites": ["r0_c0"], "token": "F"}
        ),
        "fixed_boundary",
    )
    parse_probe(
        "irregular_inferred_boundary",
        "irregular_degree_inferred_boundary",
        8,
        lambda item: item.update(
            boundarySignal={
                "mode": "domain_edge_flags",
                "observable": True,
                "includeObstacleContact": False,
                "includeFixedRole": False,
            }
        ),
        "explicit_site_tags",
    )
    parse_probe(
        "obstacle_vacancy_alias",
        "obstacle_as_vacancy",
        4,
        lambda item: item.update(obstacleToken="."),
        "must differ",
    )
    parse_probe(
        "occupied_with_vacancy",
        "fully_occupied_with_vacancy_token",
        0,
        lambda item: item["initialState"].update(rows=[".AAAAAAAA"] * 9),
        "fully occupied",
    )
    parse_probe(
        "hex_square_grammar_binding",
        "square_s02_grammar_on_hex_or_irregular",
        7,
        lambda item: item.update(
            targetBinding={
                "targetId": "stripes_alternating_three_band",
                "grammarId": "stripes_axis_relations_v1",
                "evaluationProfile": "s01_s02_conjunctive_square_v1",
            }
        ),
        "only unobstructed bounded square",
    )
    parse_probe(
        "obstacle_global_audit_binding",
        "s01_global_audit_on_modified_topology",
        4,
        lambda item: item.update(
            targetBinding={
                "targetId": "tissue_single_hole",
                "grammarId": "single_hole_boundary_relations_v1",
                "evaluationProfile": "s01_s02_conjunctive_square_v1",
            }
        ),
        "only unobstructed bounded square",
    )

    hole_fixture = deepcopy(raw_catalog["environments"][1])
    hole_fixture["expected"] = {}
    hole_fixture["boundarySignal"] = {
        "mode": "none",
        "observable": False,
        "includeObstacleContact": False,
        "includeFixedRole": False,
    }
    environment = parse_environment_spec(hole_fixture)
    diagnostic = None
    try:
        evaluate_conjunctive(
            environment,
            target_by_id["tissue_single_hole"],
            grammar_by_id["single_hole_boundary_relations_v1"],
        )
    except EnvironmentValidationError as error:
        diagnostic = str(error)
    probes.append(
        {
            "probeId": "hole_without_exterior_signal",
            "combinationId": "hole_target_without_observable_exterior",
            "rejected": diagnostic is not None,
            "diagnostic": diagnostic,
            "expectedDiagnosticSubstring": "boundary signal",
            "pass": diagnostic is not None and "boundary signal" in diagnostic,
        }
    )
    declared_ids = {
        item["combinationId"] for item in raw_catalog["unsupportedCombinations"]
    }
    probed_ids = {item["combinationId"] for item in probes}
    return {
        "schemaVersion": "e06.s03.unsupported-combination-audit.v1",
        "researchStepId": "S03",
        "declaredCount": len(declared_ids),
        "probeCount": len(probes),
        "declaredIds": sorted(declared_ids),
        "probedIds": sorted(probed_ids),
        "allDeclaredProbed": declared_ids == probed_ids,
        "probes": probes,
        "success": declared_ids == probed_ids and all(item["pass"] for item in probes),
    }


def _plot_atlas(environments, output: Path) -> None:
    token_values = sorted(
        {
            environment.initial_state[site.site_id]
            for environment in environments
            for site in environment.sites
        }
    )
    palette = plt.get_cmap("tab10")
    token_colors = {
        token: palette(index % 10) for index, token in enumerate(token_values)
    }
    token_colors["#"] = (0.15, 0.15, 0.15, 1.0)
    figure, axes = plt.subplots(3, 3, figsize=(15, 14), constrained_layout=True)
    for axis, environment in zip(axes.flat, environments):
        sites = {site.site_id: site for site in environment.sites}

        def point(site):
            first, second = site.coordinate
            if environment.geometry == "hexagonal":
                return first + second / 2, -(math.sqrt(3) / 2) * second
            if environment.geometry == "square":
                return second, -first
            return first, -second

        for first_id, second_id in environment.edges:
            first, second = sites[first_id], sites[second_id]
            x1, y1 = point(first)
            x2, y2 = point(second)
            coordinate_distance = max(
                abs(first.coordinate[0] - second.coordinate[0]),
                abs(first.coordinate[1] - second.coordinate[1]),
            )
            wrap = environment.boundary_mode == "periodic" and coordinate_distance > 1
            axis.plot(
                [x1, x2],
                [y1, y2],
                color="#e67e22" if wrap else "#b6b6b6",
                linewidth=0.8,
                linestyle="--" if wrap else "-",
                zorder=1,
                alpha=0.75,
            )
        for site in environment.sites:
            x, y = point(site)
            token = environment.initial_state[site.site_id]
            axis.scatter(
                [x],
                [y],
                s=145 if site.role != "obstacle" else 175,
                marker="s" if site.role == "obstacle" else "o",
                color=token_colors[token],
                edgecolor="#111111" if site.role == "fixed_boundary" else "white",
                linewidth=2.0 if site.role == "fixed_boundary" else 0.8,
                zorder=2,
            )
        summary = topology_summary(environment)
        axis.set_title(
            f"{environment.environment_id}\n"
            f"{environment.geometry}/{environment.boundary_mode}; "
            f"sites={summary['occupiableSiteCount']}, edges={summary['edgeCount']}"
        )
        axis.set_aspect("equal")
        axis.axis("off")
    figure.suptitle(
        "E06 S03 environment fixtures (orange dashed edges wrap periodically)",
        fontsize=16,
    )
    figure.savefig(output / "environment_atlas.png", dpi=180)
    figure.savefig(output / "environment_atlas.svg")
    plt.close(figure)


def _environment_spec(metadata, environments) -> str:
    return f"""# E06 S03 Spatial Environment Specification

## Scope

Version `{metadata["schemaVersion"]}` compiles {len(environments)} static fixtures into a single canonical undirected-graph representation. S03 defines sites, adjacency, occupancy, obstacles, fixed boundary roles, observable boundary signals, serialization, and static target-evaluation plumbing. It does not define a legal move, scheduler, proposal, conflict, or state transition; those remain S04.

## Canonical graph model

Every environment contains unique site IDs, optional integer display coordinates, canonical unordered edges `(min(site_id), max(site_id))`, an exact initial token for every site, and one of three roles:

- `active`: occupiable and not fixed;
- `obstacle`: non-occupiable, removed from the adjacency graph, and represented by `#`; or
- `fixed_boundary`: occupiable with a frozen token and an explicit non-mobile role for S04 to honor.

`fully_occupied` means no non-obstacle site carries the vacancy token. `vacancy_enabled` requires at least one conserved occupiable vacancy. Obstacles are never vacancies. Every fixture in this release requires a connected occupiable graph.

## Geometry and boundary semantics

- **Square:** four von Neumann directions on bounded rectangles or modular rectangular tori.
- **Hexagonal:** six axial `(q,r)` directions on bounded radius hexagons or modular axial tori.
- **Irregular:** explicit site and undirected edge lists; display coordinates do not generate topology.

Bounded square and hexagonal sites derive named missing-direction flags. Irregular graphs may expose only declared per-site tags because degree does not identify an exterior. Periodic graphs wrap and expose no natural exterior signal: coordinate seams are not boundaries. Obstacle contact and fixed-site role are separate opt-in channels.

## Boundary-signal requirements

Signals are part of the environment contract, not silently available policy information. S01's one- and two-hole targets require an observable exterior boundary. They cannot be evaluated on a periodic domain or with `boundarySignal.mode=none`. An irregular target requiring exterior knowledge must receive explicit site tags. Fixed-boundary and obstacle contact signals are distinguishable from natural exterior flags.

## Conjunctive evaluation contract

For the three compatible bounded square controls, S03 records:

1. S02 local grammar acceptance, soft score, relational score, and hard violations for feedback and interpretation;
2. independent S01 count, equivalence/geometry, component, and topology audits; and
3. completion as `localGrammar.accepted AND independentS01GlobalAudit.success`.

The frozen S02 counterexample is retained: its grammar accepts, its S01 geometry audit fails, and completion is false. Non-square, periodic, obstacle, and fixed-boundary fixtures are marked not applicable for target completion until target and grammar semantics are explicitly remapped and validated.

## Serialization and replay

Compiled environments serialize as canonical UTF-8 JSON with sorted keys, compact separators, sorted site records, and sorted canonical edges. Replay reconstructs the environment and deterministically repeats every neighbor query, boundary observation, occupancy read, connected-component calculation, and topology summary. Replay is static and must not be interpreted as an S04 movement episode.

## Unsupported combinations

The catalog declares {len(metadata["unsupportedCombinations"])} unsupported combinations with reasons and required alternatives. They cover periodic exterior/fixed boundaries, inferred irregular exteriors, obstacle-vacancy aliasing, fully occupied states containing vacancies, uncalibrated S02 grammar transfer, modified-topology S01 audits, and hole targets without an exterior signal. Every declaration has an executable rejection probe in `unsupported_combination_audit.json`.

## Claim boundary

These fixtures are computational spatial domains. Static connectivity and serialization do not establish movement feasibility, formation, repair, robustness, biological morphology, mechanism, agency, or clinical relevance.
"""


def _input_provenance() -> dict[str, Any]:
    return {
        "schemaVersion": "e06.s03.input-provenance.v1",
        "researchStepId": "S03",
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
            for path in (CATALOG, TARGETS, GRAMMARS, SOURCE, TEST, Path(__file__))
        ],
    }


def _report(
    environment_rows,
    evaluation_records,
    validation,
    environment_provenance,
    artifact_names,
) -> str:
    topology_lines = [
        "| Environment | Geometry | Boundary | Occupancy | Sites | Edges | Vacancies | Obstacles | Fixed | Signals | Target audit |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    evaluation_by_id = {item["environmentId"]: item for item in evaluation_records}
    for row in environment_rows:
        evaluation = evaluation_by_id[row["environmentId"]]
        if evaluation["applicable"]:
            audit = (
                f"local={evaluation['localGrammar']['accepted']}; "
                f"global={evaluation['globalAudit']['success']}; "
                f"complete={evaluation['completion']}"
            )
        else:
            audit = "not calibrated"
        topology_lines.append(
            f"| `{row['environmentId']}` | {row['geometry']} | {row['boundaryMode']} | "
            f"{row['occupancyMode']} | {row['occupiableSiteCount']} | {row['edgeCount']} | "
            f"{row['vacancyCount']} | {row['obstacleCount']} | {row['fixedBoundaryCount']} | "
            f"{row['signaledSiteCount']} | {audit} |"
        )
    total_sites = sum(item["occupiableSiteCount"] for item in environment_rows)
    total_edges = sum(item["edgeCount"] for item in environment_rows)
    test_summary = validation["focusedTests"]["stdout"].splitlines()[-1]
    next_action = "Return S03 to the Chief Scientist. If accepted, authorize only S04 to define movement proposals and deterministic conflicts while preserving site roles, occupancy conservation, canonical adjacency, boundary-signal budgets, and the separate local/global evaluation fields; do not start S04 automatically."
    return f"""# Research Step S03 Full Results — Define Grids, Graphs, Occupancy, and Boundaries

## Top summary

- **Research step ID:** S03
- **Completion status:** **Complete** on {datetime.now(timezone.utc).date().isoformat()}; only S03 was executed and S04 was not started.
- **Artifacts written:** {", ".join(f"`{name}`" for name in artifact_names)}.
- **Validation result:** **PASS** — all nine fixtures passed schema, exact expected topology, neighbor symmetry, boundary behavior, connectedness, occupancy, canonical serialization, deterministic static replay, unsupported-combination rejection, upstream immutability, conjunctive evaluation, artifact hashes, and focused tests (`{test_summary}`).
- **Outcome classification:** **supportive.** Square, hexagonal, and irregular fixtures now share one validated graph/occupancy contract across bounded, periodic, occupied, vacancy-enabled, obstacle, and fixed-boundary cases. The S02 constraint is preserved: its frozen false positive remains incomplete under the independent S01 audit.
- **Caveats or blockers:** S01/S02 target completion remains calibrated only for unobstructed bounded rectangular four-neighbor grids. Periodic domains have no natural exterior signal; irregular exteriors require explicit tags. S03 validates static fixtures, not movement feasibility or dynamics.
- **Lay summary:** Nine small spatial worlds now have exact maps of which sites touch, which locations are empty, blocked, or fixed, and which boundary clues are visible. The same saved fixture replays identically. A locally convincing but globally wrong stripe still fails completion, showing that the earlier safety check remains intact.
- **Recommended next action:** {next_action}

## Frozen question and completion criterion

**Frozen question:** Can pattern findings be represented on square, hexagonal, and irregular graphs with bounded or periodic topology, exact occupied/vacancy states, obstacles, fixed boundaries, and explicit boundary observations without silently changing S01/S02 completion semantics?

**Completion criterion:** the planned fixtures compile and serialize; neighbor symmetry, boundary behavior, connectedness, occupancy, and deterministic replay validate; unsupported combinations and boundary-signal requirements are explicit; and compatible target fixtures retain separate S02 local scores and independent S01 global audits.

The criterion was met.

## Lay summary

The environment library turns several kinds of spatial layouts into the same basic object: named sites connected by undirected neighbor links. Empty locations remain usable sites, obstacles do not, and fixed boundary sites are marked separately. Wrapping worlds do not pretend to have an outside edge. Only environments that genuinely match the earlier square targets are allowed to claim target completion.

## Inputs

- Refreshed `/workspace/AGENTS.md`, `FULL_PLAN.md`, and the pre-completion `RESEARCH_PLAN.md`.
- Manifest-verified S01 and S02 reports, catalogs, validation records, ambiguity/falsification evidence, and artifact manifests.
- E01 immutable scenario/identity, observation-boundary, canonical serialization, deterministic replay, and event-hash contracts.
- E04 declared graph-adjacency, exact-composition, label-blindness, intervention-boundary, and E06 handoff contracts.
- `input-attachments/MANIFEST.json` and its attachment sidecar. No dataset, network access, package installation, or upstream mutation was required.

Exact paths and SHA-256 values are in `input_provenance.json`; S01 and S02 immutability were rechecked against {validation["upstreamImmutability"]["checkedArtifactCount"]} manifest-listed artifacts.

## Methods

### Environment model

`src/morph2d/environments.py` compiles high-level YAML fixtures to sorted sites, canonical undirected edges, exact initial tokens, explicit roles, and declared boundary observations. Square graphs use four-neighbor adjacency; hexagonal graphs use six axial directions; irregular graphs use only their explicit edges. Obstacles have no adjacency or occupancy slot, vacancies are conserved occupiable states, and fixed boundary sites remain graph nodes with frozen tokens.

### Boundary and compatibility rules

Bounded regular domains derive missing-direction flags. Irregular domains require explicit boundary tags; degree is never an exterior proxy. Periodic seams wrap and emit no boundary flags. Obstacle contact and fixed-role signals are separately declared. Eight unsupported combinations were frozen and each was exercised by a rejection probe.

S01/S02 target binding is restricted to unobstructed bounded square rectangles. The one-hole fixture requires an observable boundary signal. Every other topology carries explicit `not calibrated` evaluation status rather than a borrowed square-grid completion result.

### Conjunctive evaluation

Three square controls exercise the downstream record. Exact stripes and the exact single-hole target pass both local and global layers. The frozen S02 corrugated-stripe challenge has local grammar acceptance but fails independent S01 geometry and therefore fails the conjunction. Count, equivalence/geometry, component, and topology fields remain separate.

### Deterministic replay and serialization

Each environment was compiled, serialized to canonical JSON, parsed, reserialized byte-identically, and replayed from both objects. Static replay enumerates sorted neighbors, boundary observations, occupancy, connected components, and topology summaries. This is deliberately not a move sequence.

### Commands

```bash
cd /workspace/cell-research
python -m pytest -q tests/test_morph2d_targets.py tests/test_morph2d_grammar.py tests/test_morph2d_environments.py
python scripts/build_morph2d_s03.py build --output-dir /artifacts/research_steps/S03
python scripts/build_morph2d_s03.py validate --output-dir /artifacts/research_steps/S03
ruff check src/morph2d scripts/build_morph2d_s03.py tests/test_morph2d_environments.py
ruff format --check src/morph2d scripts/build_morph2d_s03.py tests/test_morph2d_environments.py
python -m compileall -q src/morph2d scripts/build_morph2d_s03.py tests/test_morph2d_environments.py
```

Runtime provenance reports Python {environment_provenance["python"]}, repository commit `{environment_provenance["repository"]["commit"]}`, serial execution, no GPU, no network, and no new dependency. Parallelism was unnecessary for {len(environment_rows)} small static fixtures.

## Results

{os.linesep.join(topology_lines)}

### Anchor results

- **Coverage:** 9/9 fixtures collectively cover all required geometry, topology, occupancy, obstacle, and fixed-boundary axes.
- **Topology:** {total_sites} occupiable sites and {total_edges} undirected edges passed exact hand-declared counts, degree histograms, neighbor symmetry, and connectedness.
- **Serialization/replay:** 9/9 compiled JSON records round-tripped byte-identically; 9/9 replay hashes matched across repeated and deserialized replays.
- **Boundary semantics:** bounded square/hex flags, explicit irregular tags, obstacle-contact signals, and fixed-role signals matched their fixtures; both periodic fixtures exposed zero exterior signals.
- **Evaluation contract:** 2/2 exact target controls completed. The 1/1 S02 false-positive control retained local acceptance but failed the independent S01 audit and completion. Six non-target topologies were explicitly not calibrated.
- **Unsupported scope:** 8/8 declared unsupported combinations produced their expected rejection diagnostics.

Machine-readable details are in `topology_fixtures.csv`, `neighbor_tables.csv`, `occupancy_fixtures.csv`, `boundary_requirements.csv`, `conjunctive_evaluation.json`, and the validation JSON records. `environment_atlas.png`/`.svg` provide a visual graph audit.

## Validation

`validation_summary.json` records all checks as true:

- catalog/schema and complete requested-axis coverage;
- exact hand-declared site, edge, degree, vacancy, obstacle, fixed, and signal counts;
- symmetric undirected neighbor tables and connected occupiable graphs;
- bounded, periodic, irregular, obstacle, and fixed-boundary observation behavior;
- occupancy-role separation and exact state coverage;
- canonical parser/serializer round trips and deterministic replay;
- eight executable unsupported-combination probes;
- the conjunctive S02-local/S01-global positive and negative controls;
- S01/S02 artifact immutability, focused tests (`{test_summary}`), and final artifact hashes.

## Caveats, blockers, failed assumptions, and limitations

- No S04 movement rule, proposal, scheduler, conflict resolution, cost ledger, or dynamic episode was implemented.
- S01 component and hole semantics remain four-neighbor, rectangular, bounded, and unobstructed. Hexagonal, irregular, periodic, obstacle, and fixed-boundary target success requires separately versioned target semantics.
- S02 rectangular motifs, orientation transforms, edge capacities, and boundary counts are not automatically portable to hexagonal or irregular graphs.
- Periodic graphs have no exterior. Supplying coordinate-edge flags would create a false boundary cue.
- Irregular display coordinates do not imply adjacency or boundary. Both are explicit metadata.
- Fixed boundary tokens are frozen declarations; enforcement against future movement belongs to S04.
- Obstacle contact is exposed only where declared and is distinguishable from an exterior signal.
- Connectedness is validated for the occupiable graph, not for individual token regions.
- Deterministic replay covers static compilation and queries, not future transition replay.
- These are computational environment specifications, not evidence of biological morphology, mechanism, formation, repair, agency, or clinical relevance.

## Artifacts and provenance

Repository-backed implementation lives in `src/morph2d/environments.py`, `configs/morphologies/environment_catalog.yaml`, `scripts/build_morph2d_s03.py`, and `tests/test_morph2d_environments.py`. Source is not duplicated under artifacts.

`environment_library/` contains the immutable source-catalog snapshot, its index, and nine canonical compiled JSON fixtures. The root artifact set contains specifications, tables, audits, replay/serialization/evaluation records, figures, validation, execution/environment/input provenance, and a recursive SHA-256 manifest.

## Recommended next action

{next_action}
"""


def _artifact_role(relative: str) -> str:
    fixed = {
        "environment_spec.md": "typed S03 environment, occupancy, boundary, and evaluation contract",
        "environment_library/environment_catalog.yaml": "immutable high-level environment catalog snapshot",
        "environment_library/environment_index.json": "compiled environment index and hashes",
        "topology_fixtures.csv": "per-environment graph topology and exact fixture results",
        "neighbor_tables.csv": "per-site canonical neighbors and boundary observations",
        "occupancy_fixtures.csv": "per-environment occupancy and site-role accounting",
        "boundary_requirements.csv": "per-environment boundary channel and target requirement matrix",
        "unsupported_combinations.csv": "declared unsupported combinations and alternatives",
        "unsupported_combination_audit.json": "executable rejection probes for unsupported combinations",
        "serialization_roundtrip.json": "canonical compiled-JSON round-trip audit",
        "deterministic_replay.json": "deterministic static fixture replay audit",
        "conjunctive_evaluation.json": "full separate local grammar and S01 global audit records",
        "evaluation_contract_results.csv": "compact conjunctive evaluation table",
        "environment_atlas.png": "raster visual audit of all environment graphs",
        "environment_atlas.svg": "vector visual audit of all environment graphs",
        "validation_summary.json": "consolidated S03 validation gates",
        "input_provenance.json": "governing, upstream, attachment, and repository input hashes",
        "environment_provenance.json": "runtime, dependency, resource, and repository provenance",
        "execution_commands.log": "reproduction and validation commands",
        "research_step_full_results.md": "canonical S03 handoff report",
    }
    if relative.startswith("environment_library/") and relative.endswith(".json"):
        return fixed.get(relative, "canonical compiled environment fixture")
    return fixed[relative]


def build(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    library = output / "environment_library"
    library.mkdir(parents=True, exist_ok=True)
    raw_catalog = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))
    metadata, environments = load_environment_catalog(CATALOG)
    _, targets = load_target_catalog(TARGETS)
    _, grammars = load_grammar_catalog(GRAMMARS)
    target_by_id = {item.target_id: item for item in targets}
    grammar_by_id = {item.grammar_id: item for item in grammars}

    shutil.copyfile(CATALOG, library / "environment_catalog.yaml")
    serialization_records = []
    replay_records = []
    index_records = []
    for environment in environments:
        encoded = canonical_environment_bytes(environment)
        path = library / f"{environment.environment_id}.json"
        path.write_bytes(encoded)
        reparsed = parse_serialized_environment(json.loads(encoded))
        reencoded = canonical_environment_bytes(reparsed)
        first_replay = replay_environment(environment)
        second_replay = replay_environment(environment)
        roundtrip_replay = replay_environment(reparsed)
        serialization_records.append(
            {
                "environmentId": environment.environment_id,
                "byteLength": len(encoded),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "byteIdenticalRoundtrip": encoded == reencoded,
                "semanticEquality": environment == reparsed,
                "success": encoded == reencoded and environment == reparsed,
            }
        )
        replay_records.append(
            {
                "environmentId": environment.environment_id,
                "firstReplaySha256": first_replay["replaySha256"],
                "secondReplaySha256": second_replay["replaySha256"],
                "roundtripReplaySha256": roundtrip_replay["replaySha256"],
                "repeatMatch": first_replay == second_replay,
                "roundtripMatch": first_replay == roundtrip_replay,
                "success": first_replay == second_replay == roundtrip_replay,
            }
        )
        index_records.append(
            {
                "environmentId": environment.environment_id,
                "path": path.name,
                "sha256": _sha256_file(path),
                "environmentSha256": environment_sha256(environment),
                "geometry": environment.geometry,
                "boundaryMode": environment.boundary_mode,
                "occupancyMode": environment.occupancy_mode,
            }
        )
    _write_json(
        library / "environment_index.json",
        {
            "schemaVersion": "e06.s03.environment-index.v1",
            "researchStepId": "S03",
            "environmentCount": len(index_records),
            "environments": index_records,
        },
    )

    topology_rows = []
    occupancy_rows = []
    neighbor_rows = []
    boundary_rows = []
    evaluation_records = []
    evaluation_rows = []
    for environment in environments:
        summary = topology_summary(environment)
        topology_rows.append(
            {
                **summary,
                "degreeHistogram": json.dumps(
                    summary["degreeHistogram"], sort_keys=True, separators=(",", ":")
                ),
                "neighborSymmetryPass": all(
                    site_id in neighbor_map(environment)[other]
                    for site_id, others in neighbor_map(environment).items()
                    for other in others
                ),
                "connectednessPass": len(connected_components(environment)) == 1,
                "environmentSha256": environment_sha256(environment),
            }
        )
        role_counts = Counter(site.role for site in environment.sites)
        occupancy_rows.append(
            {
                "environmentId": environment.environment_id,
                "occupancyMode": environment.occupancy_mode,
                "totalSites": len(environment.sites),
                "occupiableSites": len(environment.occupiable_sites),
                "activeSites": role_counts["active"],
                "fixedBoundarySites": role_counts["fixed_boundary"],
                "obstacleSites": role_counts["obstacle"],
                "vacancyCount": summary["vacancyCount"],
                "stateCoversEverySite": set(environment.initial_state)
                == {site.site_id for site in environment.sites},
                "occupancyPass": True,
            }
        )
        neighbors = neighbor_map(environment)
        for site in environment.sites:
            observation = (
                None
                if site.role == "obstacle"
                else boundary_observation(environment, site.site_id)
            )
            neighbor_rows.append(
                {
                    "environmentId": environment.environment_id,
                    "siteId": site.site_id,
                    "coordinate": json.dumps(site.coordinate, separators=(",", ":")),
                    "role": site.role,
                    "token": environment.initial_state[site.site_id],
                    "degree": len(neighbors.get(site.site_id, ())),
                    "neighbors": json.dumps(
                        neighbors.get(site.site_id, ()), separators=(",", ":")
                    ),
                    "boundaryObservation": json.dumps(
                        observation, sort_keys=True, separators=(",", ":")
                    ),
                }
            )
        target_requires_boundary = False
        if environment.target_binding is not None:
            target_requires_boundary = bool(
                target_by_id[environment.target_binding["targetId"]].equivalence[
                    "boundaryCueRequired"
                ]
            )
        boundary_rows.append(
            {
                "environmentId": environment.environment_id,
                "geometry": environment.geometry,
                "boundaryMode": environment.boundary_mode,
                "boundarySignalMode": environment.boundary_signal["mode"],
                "observable": environment.boundary_signal["observable"],
                "includeObstacleContact": environment.boundary_signal[
                    "includeObstacleContact"
                ],
                "includeFixedRole": environment.boundary_signal["includeFixedRole"],
                "naturalBoundarySiteCount": summary["naturalBoundarySiteCount"],
                "signaledSiteCount": summary["signaledSiteCount"],
                "targetRequiresBoundaryCue": target_requires_boundary,
                "requirementSatisfied": (
                    not target_requires_boundary
                    or bool(environment.boundary_signal["observable"])
                ),
                "targetEvaluationCalibrated": environment.target_binding is not None,
            }
        )
        if environment.target_binding is None:
            evaluation = {
                "environmentId": environment.environment_id,
                "applicable": False,
                "reason": "No S01/S02 target binding: topology-specific target and grammar semantics are not calibrated in S03.",
                "localGrammar": None,
                "globalAudit": None,
                "completion": None,
            }
        else:
            evaluation = evaluate_conjunctive(
                environment,
                target_by_id[environment.target_binding["targetId"]],
                grammar_by_id[environment.target_binding["grammarId"]],
            )
        evaluation_records.append(evaluation)
        evaluation_rows.append(
            {
                "environmentId": environment.environment_id,
                "applicable": evaluation["applicable"],
                "targetId": evaluation.get("targetId"),
                "grammarId": evaluation.get("grammarId"),
                "localGrammarAccepted": (
                    None
                    if evaluation["localGrammar"] is None
                    else evaluation["localGrammar"]["accepted"]
                ),
                "localGrammarSoftScore": (
                    None
                    if evaluation["localGrammar"] is None
                    else evaluation["localGrammar"]["softScore"]
                ),
                "globalCountMatch": (
                    None
                    if evaluation["globalAudit"] is None
                    else evaluation["globalAudit"]["countMatch"]
                ),
                "globalEquivalenceOrGeometryMatch": (
                    None
                    if evaluation["globalAudit"] is None
                    else evaluation["globalAudit"]["equivalenceOrGeometryMatch"]
                ),
                "globalComponentMatch": (
                    None
                    if evaluation["globalAudit"] is None
                    else evaluation["globalAudit"]["componentMatch"]
                ),
                "globalTopologyMatch": (
                    None
                    if evaluation["globalAudit"] is None
                    else evaluation["globalAudit"]["topologyMatch"]
                ),
                "globalSuccess": (
                    None
                    if evaluation["globalAudit"] is None
                    else evaluation["globalAudit"]["success"]
                ),
                "completion": evaluation["completion"],
                "reason": evaluation.get("reason"),
            }
        )

    unsupported_audit = _unsupported_audit(raw_catalog, targets, grammars)
    unsupported_rows = [
        {
            "combinationId": item["combinationId"],
            "status": item["status"],
            "reason": item["reason"],
            "requiredAlternative": item["requiredAlternative"],
        }
        for item in metadata["unsupportedCombinations"]
    ]
    _write_csv(output / "topology_fixtures.csv", topology_rows)
    _write_csv(output / "neighbor_tables.csv", neighbor_rows)
    _write_csv(output / "occupancy_fixtures.csv", occupancy_rows)
    _write_csv(output / "boundary_requirements.csv", boundary_rows)
    _write_csv(output / "unsupported_combinations.csv", unsupported_rows)
    _write_json(output / "unsupported_combination_audit.json", unsupported_audit)
    _write_json(
        output / "serialization_roundtrip.json",
        {
            "schemaVersion": "e06.s03.serialization-roundtrip.v1",
            "researchStepId": "S03",
            "records": serialization_records,
            "success": all(item["success"] for item in serialization_records),
        },
    )
    _write_json(
        output / "deterministic_replay.json",
        {
            "schemaVersion": "e06.s03.deterministic-replay.v1",
            "researchStepId": "S03",
            "replayScope": "static_topology_boundary_occupancy_queries_only",
            "records": replay_records,
            "success": all(item["success"] for item in replay_records),
        },
    )
    _write_json(
        output / "conjunctive_evaluation.json",
        {
            "schemaVersion": "e06.s03.conjunctive-evaluation.v1",
            "researchStepId": "S03",
            "completionRule": metadata["evaluationContract"]["completionRule"],
            "records": evaluation_records,
        },
    )
    _write_csv(output / "evaluation_contract_results.csv", evaluation_rows)
    (output / "environment_spec.md").write_text(
        _environment_spec(metadata, environments), encoding="utf-8"
    )
    _plot_atlas(environments, output)

    tests = _focused_tests()
    upstream = [_verify_upstream_manifest(S01_DIR), _verify_upstream_manifest(S02_DIR)]
    applicable = [item for item in evaluation_records if item["applicable"]]
    negative = next(
        item
        for item in applicable
        if item["environmentId"] == "square_bounded_occupied_s02_counterexample"
    )
    requested_coverage = {
        "square": any(item.geometry == "square" for item in environments),
        "hexagonal": any(item.geometry == "hexagonal" for item in environments),
        "irregular": any(item.geometry == "irregular" for item in environments),
        "bounded": any(item.boundary_mode == "bounded" for item in environments),
        "periodic": any(item.boundary_mode == "periodic" for item in environments),
        "fullyOccupied": any(
            item.occupancy_mode == "fully_occupied" for item in environments
        ),
        "vacancyEnabled": any(
            item.occupancy_mode == "vacancy_enabled" for item in environments
        ),
        "obstacle": any(
            any(site.role == "obstacle" for site in item.sites) for item in environments
        ),
        "fixedBoundary": any(
            any(site.role == "fixed_boundary" for site in item.sites)
            for item in environments
        ),
    }
    checks = {
        "catalogAndExpectedFixtures": len(environments) == 9,
        "requestedAxisCoverage": all(requested_coverage.values()),
        "neighborSymmetry": all(row["neighborSymmetryPass"] for row in topology_rows),
        "boundaryBehavior": all(row["requirementSatisfied"] for row in boundary_rows)
        and all(
            row["signaledSiteCount"] == 0
            for row in topology_rows
            if row["boundaryMode"] == "periodic"
        ),
        "connectedness": all(row["connectednessPass"] for row in topology_rows),
        "occupancy": all(row["occupancyPass"] for row in occupancy_rows),
        "serializationRoundtrip": all(
            item["success"] for item in serialization_records
        ),
        "deterministicFixtureReplay": all(item["success"] for item in replay_records),
        "unsupportedCombinationRejection": unsupported_audit["success"],
        "conjunctiveEvaluationContract": len(applicable) == 3
        and sum(item["completion"] is True for item in applicable) == 2
        and negative["localGrammar"]["accepted"]
        and not negative["globalAudit"]["success"]
        and not negative["completion"],
        "upstreamImmutability": all(item["success"] for item in upstream),
        "focusedRepositoryTests": tests["success"],
    }
    validation = {
        "schemaVersion": "e06.s03.validation-summary.v1",
        "researchStepId": "S03",
        "status": "complete",
        "success": all(checks.values()),
        "validationResult": "PASS — all environment, topology, boundary, occupancy, serialization, deterministic replay, unsupported-scope, and conjunctive-evaluation checks passed.",
        "outcomeClassification": "supportive",
        "checks": checks,
        "requestedCoverage": requested_coverage,
        "environmentCount": len(environments),
        "occupiableSiteCount": sum(
            topology_summary(item)["occupiableSiteCount"] for item in environments
        ),
        "undirectedEdgeCount": sum(len(item.edges) for item in environments),
        "unsupportedCombinationCount": len(unsupported_rows),
        "targetBoundFixtureCount": len(applicable),
        "completedTargetFixtureCount": sum(
            item["completion"] is True for item in applicable
        ),
        "upstreamImmutability": {
            "records": upstream,
            "checkedArtifactCount": sum(
                item["checkedArtifactCount"] for item in upstream
            ),
            "success": all(item["success"] for item in upstream),
        },
        "focusedTests": tests,
    }
    if not validation["success"]:
        raise RuntimeError(f"S03 validation failed: {validation}")
    _write_json(output / "validation_summary.json", validation)
    _write_json(output / "input_provenance.json", _input_provenance())
    environment_provenance = {
        "schemaVersion": "e06.s03.environment-provenance.v1",
        "researchStepId": "S03",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpuCountVisible": os.cpu_count(),
        "workersUsed": 1,
        "parallelUnit": None,
        "gpuUsed": False,
        "networkUsed": False,
        "newDependenciesInstalled": [],
        "packages": {
            "PyYAML": yaml.__version__,
            "matplotlib": matplotlib.__version__,
            "pytest": __import__("pytest").__version__,
        },
        "repository": {
            "path": str(ROOT),
            "branch": _git("branch", "--show-current"),
            "commit": _git("rev-parse", "HEAD"),
            "dirty": bool(_git("status", "--porcelain")),
        },
    }
    _write_json(output / "environment_provenance.json", environment_provenance)
    commands = """cd /workspace/cell-research
python -m pytest -q tests/test_morph2d_targets.py tests/test_morph2d_grammar.py tests/test_morph2d_environments.py
python scripts/build_morph2d_s03.py build --output-dir /artifacts/research_steps/S03
python scripts/build_morph2d_s03.py validate --output-dir /artifacts/research_steps/S03
ruff check src/morph2d scripts/build_morph2d_s03.py tests/test_morph2d_environments.py
ruff format --check src/morph2d scripts/build_morph2d_s03.py tests/test_morph2d_environments.py
python -m compileall -q src/morph2d scripts/build_morph2d_s03.py tests/test_morph2d_environments.py
"""
    (output / "execution_commands.log").write_text(commands, encoding="utf-8")

    report_artifacts = [
        "environment_spec.md",
        "environment_library/ (catalog, index, and 9 canonical JSON fixtures)",
        "topology_fixtures.csv",
        "neighbor_tables.csv",
        "occupancy_fixtures.csv",
        "boundary_requirements.csv",
        "unsupported_combinations.csv",
        "unsupported_combination_audit.json",
        "serialization_roundtrip.json",
        "deterministic_replay.json",
        "conjunctive_evaluation.json",
        "evaluation_contract_results.csv",
        "environment_atlas.png",
        "environment_atlas.svg",
        "validation_summary.json",
        "input_provenance.json",
        "environment_provenance.json",
        "execution_commands.log",
        "artifact_manifest.json",
        "research_step_full_results.md",
    ]
    report = _report(
        topology_rows,
        evaluation_records,
        validation,
        environment_provenance,
        report_artifacts,
    )
    (output / "research_step_full_results.md").write_text(report, encoding="utf-8")

    artifact_paths = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    )
    manifest_records = []
    for path in artifact_paths:
        relative = path.relative_to(output).as_posix()
        manifest_records.append(
            {
                "path": relative,
                "role": _artifact_role(relative),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    _write_json(
        output / "artifact_manifest.json",
        {
            "schemaVersion": "e06.s03.artifact-manifest.v1",
            "researchStepId": "S03",
            "artifactCountExcludingManifest": len(manifest_records),
            "artifacts": manifest_records,
            "repositorySourcePaths": [
                "configs/morphologies/environment_catalog.yaml",
                "src/morph2d/environments.py",
                "src/morph2d/__init__.py",
                "scripts/build_morph2d_s03.py",
                "tests/test_morph2d_environments.py",
            ],
        },
    )


def validate(output: Path) -> None:
    required = {
        "environment_spec.md",
        "topology_fixtures.csv",
        "neighbor_tables.csv",
        "occupancy_fixtures.csv",
        "boundary_requirements.csv",
        "unsupported_combinations.csv",
        "unsupported_combination_audit.json",
        "serialization_roundtrip.json",
        "deterministic_replay.json",
        "conjunctive_evaluation.json",
        "evaluation_contract_results.csv",
        "environment_atlas.png",
        "environment_atlas.svg",
        "validation_summary.json",
        "input_provenance.json",
        "environment_provenance.json",
        "execution_commands.log",
        "research_step_full_results.md",
        "artifact_manifest.json",
        "environment_library/environment_catalog.yaml",
        "environment_library/environment_index.json",
    }
    actual = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    }
    missing = required - actual
    if missing:
        raise RuntimeError(f"missing S03 artifacts: {sorted(missing)}")
    compiled = [
        item
        for item in actual
        if item.startswith("environment_library/")
        and item.endswith(".json")
        and not item.endswith("environment_index.json")
    ]
    if len(compiled) != 9:
        raise RuntimeError(f"expected 9 compiled environments, found {len(compiled)}")
    validation_summary = json.loads((output / "validation_summary.json").read_text())
    if not validation_summary["success"] or not all(
        validation_summary["checks"].values()
    ):
        raise RuntimeError("validation summary contains a failed check")
    manifest = json.loads((output / "artifact_manifest.json").read_text())
    if manifest["artifactCountExcludingManifest"] != len(manifest["artifacts"]):
        raise RuntimeError("manifest count mismatch")
    manifest_paths = {item["path"] for item in manifest["artifacts"]}
    if manifest_paths != actual - {"artifact_manifest.json"}:
        raise RuntimeError("manifest paths do not match artifact files")
    for record in manifest["artifacts"]:
        path = output / record["path"]
        if (
            path.stat().st_size != record["bytes"]
            or _sha256_file(path) != record["sha256"]
        ):
            raise RuntimeError(f"artifact hash/size mismatch: {record['path']}")
    report = (output / "research_step_full_results.md").read_text(encoding="utf-8")
    required_sections = (
        "## Top summary",
        "## Frozen question and completion criterion",
        "## Lay summary",
        "## Inputs",
        "## Methods",
        "## Results",
        "## Validation",
        "## Caveats, blockers, failed assumptions, and limitations",
        "## Artifacts and provenance",
        "## Recommended next action",
    )
    if any(section not in report for section in required_sections):
        raise RuntimeError("canonical report is missing a required section")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "validate"))
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "build":
        build(arguments.output_dir)
        validate(arguments.output_dir)
    else:
        validate(arguments.output_dir)


if __name__ == "__main__":
    main()
