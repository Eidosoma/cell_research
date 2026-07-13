#!/usr/bin/env python3
"""Build and validate the S01 figure-by-figure claim registry.

The registry is a manual, provenance-labelled transcription of the supplied journal
paper's Results text/captions and a bounded recovery pass against arXiv:2401.05375v1.
It intentionally leaves undocumented fields null and attaches ambiguity codes rather
than inferring historical implementation details.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


SCHEMA_VERSION = "e01.s01.claim_registry.v1"
STEP_ID = "S01"
DEFAULT_SOURCE = Path(
    "/workspace/input-attachments/21c2278b-9950-4e39-a2c8-df578a2508ec/"
    "pdf-markdown.md"
)
DEFAULT_OUTPUT = Path("/artifacts/research_steps/S01")
DEFAULT_RECOVERY_PDF = Path("/cache/e01_s01/arxiv-2401.05375.pdf")


COLUMNS: dict[str, str] = {
    "schema_version": "string",
    "research_step_id": "string",
    "claim_id": "string",
    "figure": "Int64",
    "panel": "string",
    "claim_group": "string",
    "claim_kind": "string",
    "claim_text": "string",
    "expected_outcome": "string",
    "evidence_status": "string",
    "executable_spec_status": "string",
    "array_size": "Int64",
    "repetition_count": "Int64",
    "value_distribution": "string",
    "architecture": "string",
    "algorithm": "string",
    "algotype_composition": "string",
    "sorting_direction": "string",
    "fault_type": "string",
    "fault_count": "Int64",
    "placement_sampling": "string",
    "stopping_rule": "string",
    "metric_name": "string",
    "metric_formula": "string",
    "cost_definition": "string",
    "reported_mean": "Float64",
    "reported_variability": "Float64",
    "variability_type": "string",
    "reported_statistic": "Float64",
    "statistic_type": "string",
    "reported_p_value": "string",
    "reported_effect": "string",
    "reported_value_text": "string",
    "reported_unit": "string",
    "source_section": "string",
    "source_locator": "string",
    "source_basis": "string",
    "ambiguity_codes": "string",
    "ambiguity_note": "string",
    "requires_code_reconciliation": "boolean",
    "notes": "string",
}


AMBIGUITIES = {
    "AGGREGATION_BOUNDARY_UNDEFINED": (
        "Aggregation equation uses the previous cell but does not define i=0 handling or "
        "whether the array is circular."
    ),
    "AGGREGATION_NULL_CONFLICT": (
        "The paper uses 0.5 as a universal random baseline, but an equally mixed three-Algotype "
        "array has a different simple same-neighbor chance expectation."
    ),
    "ALGOTYPE_ASSIGNMENT_CONFLICT": (
        "The paper variously says Algotypes are randomly assigned and equally represented; exact "
        "composition enforcement is not stated."
    ),
    "COST_OPERATION_UNDEFINED": (
        "The swap/comparison ledger does not define failed attempts, repeated reads, lock attempts, "
        "or architecture-specific comparison counting."
    ),
    "DG_DEFINITION_CONFLICT": (
        "The displayed equation and journal Figure 6 caption use (increase-decrease)/decrease, "
        "whereas the arXiv caption describes increase/decrease."
    ),
    "DG_SIGN_CONFLICT": (
        "Selection DG prose reports z=+17.21 while the caption reports z=-17.21 for the same "
        "directional contrast."
    ),
    "ERROR_BAR_TYPE_UNREPORTED": (
        "A panel shows error bars but does not identify SD, SE, confidence interval, or another "
        "quantity."
    ),
    "FAULT_PLACEMENT_UNREPORTED": (
        "The number of frozen cells is given, but their placement sampling rule is not."
    ),
    "FIGURE_EXTRACTION_CORRUPT": (
        "The supplied extracted bitmap is visually corrupted; a readable official arXiv panel was "
        "used where available."
    ),
    "F10_NUMERIC_NOT_RECOVERABLE": (
        "Figure 10's plotted numerical endpoints cannot be read from the supplied bitmap and are "
        "not quoted in the journal text."
    ),
    "LINEARITY_NOT_TESTED": (
        "The paper calls chimeric efficiency linear but reports no fitted linear model, deviation, "
        "uncertainty, or equivalence margin."
    ),
    "METRIC_SCALE_AMBIGUOUS": (
        "The displayed Sortedness/Aggregation equations return fractions, while text and axes also "
        "use percentages; the conversion convention is not explicit."
    ),
    "NEGATIVE_CONTROL_CONTRADICTION": (
        "The control is described as two identical Bubble Algotypes, but prose assigns pairwise "
        "mixed-Algotype labels and the journal version calls the control significantly aggregated."
    ),
    "NO_FAULT_STOP_UNREPORTED": (
        "The no-fault run stopping rule is not stated at claim level."
    ),
    "P_VALUE_CONFLICT": (
        "The same comparison is reported as p=0 in a caption and p<<0.01 in Results prose."
    ),
    "PANEL_VALUE_NOT_QUOTED": (
        "The value is transcribed from a plotted label/curve rather than quoted in Results prose."
    ),
    "PASSIVE_INSERTION_CONTRADICTION": (
        "Figure 5 shows passive cell-view Insertion error exceeding traditional error at f=2 and "
        "f=3, contradicting the blanket prose claim."
    ),
    "SAMPLE_SIZE_UNREPORTED": (
        "The panel/result does not state its repetition count."
    ),
    "SORTEDNESS_DUPLICATE_CONFLICT": (
        "The printed Sortedness equation uses a strict comparison, so equal adjacent values do not "
        "count as sorted even though duplicate-valued experiments are described as sorted."
    ),
    "STD_FORMULA_INVALID": (
        "The printed standard-deviation equation lacks a squared deviation term and would collapse "
        "toward zero; exact intended variability calculation requires code reconciliation."
    ),
    "STOP_RULE_CONFLICT": (
        "Stopping is variously described as fully sorted, no legal move, unchanged Sortedness for "
        "several steps, or two unchanged array checks."
    ),
    "TEST_SPECIFICATION_INCOMPLETE": (
        "The z-test lacks a stated estimand, paired/unpaired form, variance assumptions, and "
        "multiplicity handling."
    ),
    "VARIABILITY_UNREPORTED": (
        "A reported mean/effect lacks a numerical variability estimate."
    ),
}


PANEL_REQUIREMENTS = {
    3: {"A", "B", "C"},
    4: {"A", "B"},
    5: {"A", "B", "C"},
    6: {"A", "B", "C", "D"},
    7: {"A", "B", "C"},
    8: {"A", "B", "C", "D", "E"},
    9: {"A", "B", "C"},
    10: {"A", "B", "C"},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _null_record() -> dict[str, Any]:
    record: dict[str, Any] = {key: None for key in COLUMNS}
    record.update(
        schema_version=SCHEMA_VERSION,
        research_step_id=STEP_ID,
        evidence_status="specified",
        executable_spec_status="needs_decision",
        source_basis="supplied_journal_text",
        requires_code_reconciliation=False,
    )
    return record


def build_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    def add(**kwargs: Any) -> None:
        unknown = set(kwargs) - set(COLUMNS)
        if unknown:
            raise KeyError(f"Unknown registry fields: {sorted(unknown)}")
        row = _null_record()
        row.update(kwargs)
        records.append(row)

    common_unique = dict(
        array_size=100,
        repetition_count=100,
        value_distribution="random permutation of integers 1..100 without duplication",
    )

    # Figure 3: each lettered row compares traditional and cell-view trajectories.
    for panel, algorithm, detail in (
        ("A", "Bubble", "both architectures reach 100% Sortedness; trajectory shapes differ"),
        ("B", "Insertion", "both architectures reach 100% Sortedness; trajectories are described as similar"),
        ("C", "Selection", "both architectures reach 100% Sortedness; cell-view uses more swaps"),
    ):
        add(
            claim_id=f"F03-{panel}-TRAJECTORY",
            figure=3,
            panel=panel,
            claim_group="no_fault_sorting",
            claim_kind="qualitative_and_endpoint",
            claim_text=f"Traditional and cell-view {algorithm} complete sorting; {detail}.",
            expected_outcome=detail,
            architecture="traditional_vs_cell_view",
            algorithm=algorithm,
            sorting_direction="increasing",
            fault_type="none",
            fault_count=0,
            metric_name="Sortedness trajectory",
            metric_formula="S = count(i=0 or V_i > V_(i-1)) / N (printed equation; percentage scaling implicit)",
            reported_mean=100.0,
            reported_value_text="final Sortedness = 100% for every displayed trajectory",
            reported_unit="percent Sortedness",
            source_section="Results; Figure 3 caption",
            source_locator="supplied markdown lines 158, 164-166",
            ambiguity_codes="METRIC_SCALE_AMBIGUOUS|NO_FAULT_STOP_UNREPORTED",
            ambiguity_note="The endpoint is explicit, but the stop rule and fraction-to-percent conversion are not.",
            requires_code_reconciliation=True,
            **common_unique,
        )

    add(
        claim_id="F03-C-SWAP-BURDEN",
        figure=3,
        panel="C",
        claim_group="no_fault_sorting",
        claim_kind="qualitative_contrast",
        claim_text="Cell-view Selection requires more swaps than traditional Selection.",
        expected_outcome="cell-view swap count > traditional swap count",
        architecture="traditional_vs_cell_view",
        algorithm="Selection",
        sorting_direction="increasing",
        fault_type="none",
        fault_count=0,
        metric_name="swap count",
        cost_definition="successful swaps only",
        source_section="Figure 3 caption",
        source_locator="supplied markdown line 164",
        ambiguity_codes="NO_FAULT_STOP_UNREPORTED|VARIABILITY_UNREPORTED",
        ambiguity_note="Figure 3 states direction only; Figure 4 supplies the numerical contrast.",
        requires_code_reconciliation=True,
        **common_unique,
    )

    # Figure 4: two cost definitions x three algorithms.
    efficiency = [
        ("A", "Bubble", "swaps only", 0.73, "0.47", "no significant difference", None),
        ("A", "Insertion", "swaps only", 1.26, "0.24", "no significant difference", None),
        ("A", "Selection", "swaps only", 120.43, "<<0.01 (caption also says 0)", "cell-view takes 11x more swaps", "P_VALUE_CONFLICT"),
        ("B", "Bubble", "successful swaps plus comparisons", -68.96, "<<0.01", "cell-view total is 1.5x lower", "COST_OPERATION_UNDEFINED"),
        ("B", "Insertion", "successful swaps plus comparisons", -71.19, "<<0.01", "cell-view total is 2.03x lower", "COST_OPERATION_UNDEFINED"),
        ("B", "Selection", "successful swaps plus comparisons", 106.55, "<<0.01", "cell-view total is 1.17x greater", "COST_OPERATION_UNDEFINED"),
    ]
    for panel, algorithm, cost, z_value, p_value, effect, extra_code in efficiency:
        codes = ["TEST_SPECIFICATION_INCOMPLETE", "VARIABILITY_UNREPORTED"]
        if extra_code:
            codes.append(extra_code)
        add(
            claim_id=f"F04-{panel}-{algorithm.upper()}",
            figure=4,
            panel=panel,
            claim_group="efficiency",
            claim_kind="numerical_contrast",
            claim_text=f"Traditional vs cell-view {algorithm} efficiency under {cost}.",
            expected_outcome=effect,
            architecture="traditional_vs_cell_view",
            algorithm=algorithm,
            sorting_direction="increasing",
            fault_type="none",
            fault_count=0,
            metric_name="total sorting steps",
            cost_definition=cost,
            reported_statistic=z_value,
            statistic_type="z",
            reported_p_value=p_value,
            reported_effect=effect,
            source_section="Results 4.1; Figure 4",
            source_locator="supplied markdown lines 162, 168, 172-174",
            ambiguity_codes="|".join(codes),
            ambiguity_note="Test construction and numerical variability are not reported.",
            requires_code_reconciliation=True,
            **common_unique,
        )

    # Figure 5A: fault semantics.
    for claim_id, fault_type, definition in (
        ("F05-A-PASSIVE-DEFINITION", "passive", "cannot initiate swaps but can be moved by others"),
        ("F05-A-STUCK-DEFINITION", "stuck", "cannot initiate or participate in swaps"),
    ):
        add(
            claim_id=claim_id,
            figure=5,
            panel="A",
            claim_group="fault_semantics",
            claim_kind="definition",
            claim_text=f"A {fault_type} Frozen Cell {definition}.",
            expected_outcome=definition,
            architecture="traditional_and_cell_view",
            fault_type=fault_type,
            metric_name="fault semantics",
            source_section="Figure 5 caption; definitions",
            source_locator="supplied markdown lines 55-70, 200",
            evidence_status="specified",
            executable_spec_status="needs_decision",
            ambiguity_codes="FAULT_PLACEMENT_UNREPORTED",
            ambiguity_note="Behavioral semantics are stated, but how frozen identities/positions are sampled is not.",
            requires_code_reconciliation=True,
        )

    passive = {
        1: {"Bubble": (2.85, 0.00), "Insertion": (2.93, 1.67), "Selection": (10.80, 2.24)},
        2: {"Bubble": (4.71, 0.80), "Insertion": (4.65, 5.28), "Selection": (16.33, 4.36)},
        3: {"Bubble": (6.44, 2.64), "Insertion": (6.38, 9.96), "Selection": (21.66, 13.24)},
    }
    stuck = {
        1: {"Bubble": (2.85, 1.91), "Insertion": (2.93, 1.83), "Selection": (10.80, 1.00)},
        2: {"Bubble": (4.71, 3.72), "Insertion": (4.65, 3.65), "Selection": (16.33, 1.96)},
        3: {"Bubble": (6.44, 5.37), "Insertion": (6.38, 5.33), "Selection": (21.66, 2.91)},
    }
    for panel, fault_type, values in (("B", "passive", passive), ("C", "stuck", stuck)):
        for fault_count, algorithms in values.items():
            for algorithm, pair in algorithms.items():
                for architecture, mean in zip(("traditional", "cell_view"), pair, strict=True):
                    codes = ["PANEL_VALUE_NOT_QUOTED", "ERROR_BAR_TYPE_UNREPORTED", "FAULT_PLACEMENT_UNREPORTED"]
                    status = "specified"
                    if fault_type == "passive" and algorithm == "Insertion" and fault_count in (2, 3):
                        codes.append("PASSIVE_INSERTION_CONTRADICTION")
                        status = "contradictory"
                    add(
                        claim_id=f"F05-{panel}-{fault_type.upper()}-{algorithm.upper()}-F{fault_count}-{architecture.upper()}",
                        figure=5,
                        panel=panel,
                        claim_group="frozen_cell_error",
                        claim_kind="panel_mean",
                        claim_text=(
                            f"Mean final monotonicity error for {architecture} {algorithm} with "
                            f"{fault_count} {fault_type} Frozen Cell(s)."
                        ),
                        expected_outcome="lower error indicates greater error tolerance",
                        evidence_status=status,
                        executable_spec_status="needs_decision",
                        architecture=architecture,
                        algorithm=algorithm,
                        sorting_direction="increasing",
                        fault_type=fault_type,
                        fault_count=fault_count,
                        placement_sampling="not reported",
                        metric_name="final monotonicity error",
                        metric_formula="E = count(i>0 and V_i < V_(i-1)); equality is not an error",
                        reported_mean=mean,
                        reported_unit="adjacent order violations",
                        source_section="Results 4.2; Figure 5",
                        source_locator="supplied markdown lines 178-180, 200; readable arXiv v1 Figure 5",
                        source_basis="supplied_journal_text+official_arxiv_panel_recovery",
                        ambiguity_codes="|".join(codes),
                        ambiguity_note=(
                            "Traditional means and several cell-view means are panel-label transcriptions. "
                            "Error-bar definition is absent."
                        ),
                        requires_code_reconciliation=True,
                        **common_unique,
                    )

    add(
        claim_id="F05-B-PASSIVE-RANK",
        figure=5,
        panel="B",
        claim_group="frozen_cell_error",
        claim_kind="qualitative_rank",
        claim_text="For passive faults, cell-view tolerance ranks Bubble > Insertion > Selection.",
        expected_outcome="Bubble has lowest and Selection highest final monotonicity error",
        evidence_status="contradictory",
        executable_spec_status="needs_decision",
        architecture="cell_view",
        algorithm="Bubble|Insertion|Selection",
        sorting_direction="increasing",
        fault_type="passive",
        fault_count=None,
        placement_sampling="not reported",
        metric_name="final monotonicity error",
        source_section="Results 4.2; Figure 5 caption",
        source_locator="supplied markdown lines 178-180, 200",
        ambiguity_codes="PASSIVE_INSERTION_CONTRADICTION|FAULT_PLACEMENT_UNREPORTED",
        ambiguity_note="At f=2 and f=3, the panel has passive cell-view Insertion error above traditional error.",
        requires_code_reconciliation=True,
        **common_unique,
    )
    add(
        claim_id="F05-C-STUCK-RANK",
        figure=5,
        panel="C",
        claim_group="frozen_cell_error",
        claim_kind="qualitative_rank",
        claim_text="For stuck faults, cell-view Selection has the greatest error tolerance.",
        expected_outcome="Selection has the lowest final monotonicity error",
        architecture="cell_view",
        algorithm="Bubble|Insertion|Selection",
        sorting_direction="increasing",
        fault_type="stuck",
        placement_sampling="not reported",
        metric_name="final monotonicity error",
        source_section="Results 4.2; Figure 5 caption",
        source_locator="supplied markdown lines 178-180, 200",
        ambiguity_codes="FAULT_PLACEMENT_UNREPORTED",
        ambiguity_note="Placement rule is absent.",
        requires_code_reconciliation=True,
        **common_unique,
    )

    # Figure 6: illustrative DG panels and printed definition.
    add(
        claim_id="F06-A-CONCEPTUAL-CONTEXT",
        figure=6,
        panel="A",
        claim_group="delayed_gratification",
        claim_kind="non_empirical_context",
        claim_text="Conceptual barrier-navigation illustration; no simulator result is reported.",
        expected_outcome="not an executable empirical claim",
        evidence_status="not_empirical",
        executable_spec_status="not_applicable",
        source_section="Figure 6 caption",
        source_locator="supplied markdown line 214",
    )
    add(
        claim_id="F06-B-FROZEN-CELL-EXAMPLE",
        figure=6,
        panel="B",
        claim_group="delayed_gratification",
        claim_kind="worked_example",
        claim_text="A six-cell frozen-cell trace temporarily decreases Sortedness before later recovery.",
        expected_outcome="Sortedness count follows 2,2,4,4,3,4 with a temporary 4-to-3 drop",
        evidence_status="partially_specified",
        executable_spec_status="needs_decision",
        array_size=6,
        value_distribution="worked array containing values 1..6",
        architecture="cell_view",
        sorting_direction="increasing",
        fault_type="stuck (shown frozen value 4)",
        fault_count=1,
        metric_name="Sortedness count",
        reported_value_text="panel sequence: 2,2,4,4,3,4; prose: value 3 detours to fourth position",
        source_section="Figure 6 caption and panel",
        source_locator="supplied markdown line 214; readable arXiv v1 Figure 6",
        source_basis="supplied_journal_text+official_arxiv_panel_recovery",
        ambiguity_codes="PANEL_VALUE_NOT_QUOTED|STOP_RULE_CONFLICT",
        ambiguity_note="Full actor/scheduler event sequence is not stated and must be reconciled later.",
        requires_code_reconciliation=True,
    )
    add(
        claim_id="F06-C-MULTIPLE-LOCAL-DROPS",
        figure=6,
        panel="C",
        claim_group="delayed_gratification",
        claim_kind="qualitative_trajectory",
        claim_text="Arrays with more Frozen Cells show multiple local reductions in Sortedness.",
        expected_outcome="one or more temporary negative Sortedness increments, increasing in context",
        evidence_status="partially_specified",
        executable_spec_status="needs_decision",
        architecture="cell_view",
        algorithm="not stated in caption",
        fault_type="stuck",
        metric_name="Sortedness trajectory",
        source_section="Figure 6 caption",
        source_locator="supplied markdown line 214",
        ambiguity_codes="SAMPLE_SIZE_UNREPORTED|FAULT_PLACEMENT_UNREPORTED",
        ambiguity_note="Algorithm, repetitions, exact fault count, and placement are absent from this panel description.",
        requires_code_reconciliation=True,
    )
    add(
        claim_id="F06-D-DG-DEFINITION",
        figure=6,
        panel="D",
        claim_group="delayed_gratification",
        claim_kind="metric_definition",
        claim_text="Delayed Gratification normalizes net recovery beyond a preceding drop by the drop size.",
        expected_outcome="D = (DeltaS_increasing - DeltaS_decreasing) / DeltaS_decreasing",
        evidence_status="contradictory",
        executable_spec_status="needs_decision",
        metric_name="Delayed Gratification",
        metric_formula="D = (DeltaS_increasing - DeltaS_decreasing) / DeltaS_decreasing",
        source_section="Methods 3.4; Figure 6 caption",
        source_locator="supplied markdown lines 136-140, 214; arXiv v1 pages 9 and 30",
        source_basis="supplied_journal_text+official_arxiv_equation_recovery",
        ambiguity_codes="DG_DEFINITION_CONFLICT|METRIC_SCALE_AMBIGUOUS",
        ambiguity_note="The arXiv caption instead describes increase/decrease; zero-drop and segment-merging cases are undefined.",
        requires_code_reconciliation=True,
    )

    # Figure 7: all visible bar means, then reported across-condition contrasts/ranks.
    dg_values = {
        "A": {
            "Bubble": {
                0: (0.08, 0.24),
                1: (0.12, 0.29),
                2: (0.17, 0.32),
                3: (0.21, 0.37),
            }
        },
        "B": {
            "Insertion": {
                0: (1.08, 1.10),
                1: (1.12, 1.13),
                2: (1.15, 1.15),
                3: (1.18, 1.19),
            }
        },
        "C": {
            "Selection": {
                0: (6.10, 3.33),
                1: (5.67, 3.39),
                2: (5.94, 3.30),
                3: (5.80, 3.07),
            }
        },
    }
    for panel, algorithms in dg_values.items():
        for algorithm, fault_values in algorithms.items():
            for fault_count, pair in fault_values.items():
                for architecture, mean in zip(("traditional", "cell_view"), pair, strict=True):
                    add(
                        claim_id=f"F07-{panel}-{algorithm.upper()}-F{fault_count}-{architecture.upper()}",
                        figure=7,
                        panel=panel,
                        claim_group="delayed_gratification",
                        claim_kind="panel_mean",
                        claim_text=(
                            f"Mean DG for {architecture} {algorithm} with {fault_count} stuck Frozen Cell(s)."
                        ),
                        expected_outcome="panel mean reproduced under the paper's DG implementation",
                        executable_spec_status="needs_decision",
                        architecture=architecture,
                        algorithm=algorithm,
                        sorting_direction="increasing",
                        fault_type="stuck",
                        fault_count=fault_count,
                        placement_sampling="not reported",
                        metric_name="Delayed Gratification",
                        metric_formula="D = (DeltaS_increasing - DeltaS_decreasing) / DeltaS_decreasing",
                        reported_mean=mean,
                        reported_unit="DG index",
                        source_section="Results 4.3; Figure 7",
                        source_locator="supplied markdown lines 184-188, 228; readable arXiv v1 Figure 7",
                        source_basis="supplied_journal_text+official_arxiv_panel_recovery",
                        ambiguity_codes=(
                            "PANEL_VALUE_NOT_QUOTED|ERROR_BAR_TYPE_UNREPORTED|FAULT_PLACEMENT_UNREPORTED|"
                            "DG_DEFINITION_CONFLICT"
                        ),
                        ambiguity_note="Bar label is readable; error-bar definition, placement, and edge-case metric logic are not.",
                        requires_code_reconciliation=True,
                        **common_unique,
                    )

    dg_contrasts = [
        ("A", "Bubble", 0.16, 34.04, "<<0.01", "cell-view > traditional", None),
        ("B", "Insertion", 0.03, 0.60, "0.55", "cell-view approximately equals traditional", None),
        ("C", "Selection", 2.77, -17.21, "<<0.01", "cell-view < traditional", "DG_SIGN_CONFLICT"),
    ]
    for panel, algorithm, difference, z_value, p_value, direction, extra in dg_contrasts:
        codes = ["DG_DEFINITION_CONFLICT", "TEST_SPECIFICATION_INCOMPLETE", "FAULT_PLACEMENT_UNREPORTED"]
        if extra:
            codes.append(extra)
        add(
            claim_id=f"F07-{panel}-{algorithm.upper()}-CONTRAST",
            figure=7,
            panel=panel,
            claim_group="delayed_gratification",
            claim_kind="numerical_contrast",
            claim_text=f"Average DG difference between cell-view and traditional {algorithm}.",
            expected_outcome=direction,
            evidence_status="contradictory" if extra else "specified",
            executable_spec_status="needs_decision",
            architecture="traditional_vs_cell_view",
            algorithm=algorithm,
            sorting_direction="increasing",
            fault_type="stuck",
            placement_sampling="not reported",
            metric_name="Delayed Gratification",
            metric_formula="D = (DeltaS_increasing - DeltaS_decreasing) / DeltaS_decreasing",
            reported_mean=difference,
            reported_statistic=z_value,
            statistic_type="z",
            reported_p_value=p_value,
            reported_effect=direction,
            reported_unit="DG index difference (magnitude; subtraction order not consistently stated)",
            source_section="Results 4.3; Figure 7 caption",
            source_locator="supplied markdown lines 184, 228",
            ambiguity_codes="|".join(codes),
            ambiguity_note="The exact pooling across f=0..3 and z-test construction are not stated.",
            requires_code_reconciliation=True,
            **common_unique,
        )

    for claim_id, panel, text_value, z_value in (
        ("F07-ABC-SELECTION-RANK", "A|B|C", "Selection DG > Insertion DG and Bubble DG", 40.81),
        ("F07-AB-INSERTION-BUBBLE-RANK", "A|B", "Insertion DG > Bubble DG in both architectures", 98.04),
    ):
        add(
            claim_id=claim_id,
            figure=7,
            panel=panel,
            claim_group="delayed_gratification",
            claim_kind="algorithm_rank",
            claim_text=text_value,
            expected_outcome=text_value,
            executable_spec_status="needs_decision",
            architecture="traditional_and_cell_view",
            algorithm="Bubble|Insertion|Selection",
            sorting_direction="increasing",
            fault_type="stuck",
            metric_name="Delayed Gratification",
            reported_statistic=z_value,
            statistic_type="z",
            reported_p_value="<<0.01",
            source_section="Figure 7 caption",
            source_locator="supplied markdown line 228",
            ambiguity_codes="DG_DEFINITION_CONFLICT|TEST_SPECIFICATION_INCOMPLETE",
            ambiguity_note="The exact contrast, pooling, and variance model behind the z statistic are unstated.",
            requires_code_reconciliation=True,
            **common_unique,
        )

    # Figure 8A: completion and aggregation in pairwise/three-way chimeras.
    unique_mixes = (
        ("Bubble+Selection", "50:50 intended or random; exact enforcement unstated"),
        ("Bubble+Insertion", "50:50 intended or random; exact enforcement unstated"),
        ("Selection+Insertion", "50:50 intended or random; exact enforcement unstated"),
        ("Bubble+Insertion+Selection", "equal thirds intended, impossible exactly at n=100 unless imbalanced"),
    )
    for composition, composition_note in unique_mixes:
        slug = composition.replace("+", "-").upper()
        add(
            claim_id=f"F08-A-{slug}-COMPLETION",
            figure=8,
            panel="A",
            claim_group="chimeric_completion",
            claim_kind="endpoint",
            claim_text=f"The {composition} same-direction chimera completely sorts the array.",
            expected_outcome="final Sortedness = 100%",
            architecture="cell_view",
            algorithm=composition,
            algotype_composition=composition_note,
            sorting_direction="increasing for all Algotypes",
            fault_type="none",
            fault_count=0,
            stopping_rule="fully sorted OR no cell finds a better position; main thread checks unchanged array twice",
            metric_name="final Sortedness",
            reported_mean=100.0,
            reported_unit="percent Sortedness",
            source_section="Results 4.4; Figure 8 caption",
            source_locator="supplied markdown lines 192-196, 248",
            ambiguity_codes="ALGOTYPE_ASSIGNMENT_CONFLICT|STOP_RULE_CONFLICT|METRIC_SCALE_AMBIGUOUS",
            ambiguity_note="Composition and stop branches must be represented explicitly in the future scenario schema.",
            requires_code_reconciliation=True,
            **common_unique,
        )

    efficiency_means = (
        ("PURE-BUBBLE", "Bubble", "100% Bubble", 2448.8),
        ("PURE-INSERTION", "Insertion", "100% Insertion", 2482.8),
        ("PURE-SELECTION", "Selection", "100% Selection", 1095.5),
        ("MIX-BUBBLE-INSERTION", "Bubble+Insertion", "two-type mix; exact counts unstated", 2476.02),
        ("MIX-BUBBLE-SELECTION", "Bubble+Selection", "two-type mix; exact counts unstated", 1740.9),
        ("MIX-INSERTION-SELECTION", "Insertion+Selection", "two-type mix; exact counts unstated", 1534.77),
    )
    for slug, algorithm, composition, mean in efficiency_means:
        add(
            claim_id=f"F08-B-{slug}-STEPS",
            figure=8,
            panel="B",
            claim_group="chimeric_efficiency",
            claim_kind="mean",
            claim_text=f"Mean successful swaps to complete {algorithm} cell-view sorting.",
            expected_outcome=f"mean swap count approximately {mean}",
            architecture="cell_view",
            algorithm=algorithm,
            algotype_composition=composition,
            sorting_direction="increasing",
            fault_type="none",
            fault_count=0,
            stopping_rule="fully sorted OR no cell finds a better position; main thread checks unchanged array twice",
            metric_name="successful swap count",
            cost_definition="successful swaps only",
            reported_mean=mean,
            reported_unit="swaps",
            source_section="Results 4.4; Figure 8B",
            source_locator="supplied markdown line 198",
            ambiguity_codes="VARIABILITY_UNREPORTED|ALGOTYPE_ASSIGNMENT_CONFLICT|STOP_RULE_CONFLICT",
            ambiguity_note="No numerical variability is reported for the mean and exact mix counts are unstated.",
            requires_code_reconciliation=True,
            **common_unique,
        )
    add(
        claim_id="F08-B-LINEAR-EFFICIENCY",
        figure=8,
        panel="B",
        claim_group="chimeric_efficiency",
        claim_kind="qualitative_inference",
        claim_text="Pairwise-chimera efficiency is roughly the average of its two pure Algotypes.",
        expected_outcome="mixed mean lies between pure means and approximately at their arithmetic mean",
        evidence_status="partially_specified",
        executable_spec_status="needs_decision",
        architecture="cell_view",
        algorithm="pairwise mixes",
        sorting_direction="increasing for all Algotypes",
        metric_name="successful swap count",
        cost_definition="successful swaps only",
        source_section="Results 4.4",
        source_locator="supplied markdown line 198",
        ambiguity_codes="LINEARITY_NOT_TESTED|VARIABILITY_UNREPORTED",
        ambiguity_note="Being between endpoints is weaker than linear interpolation; no formal linearity criterion is reported.",
        requires_code_reconciliation=True,
        **common_unique,
    )
    add(
        claim_id="F08-C-AGGREGATION-DEFINITION",
        figure=8,
        panel="C",
        claim_group="aggregation",
        claim_kind="metric_definition",
        claim_text="Aggregation is the fraction of cells whose directly adjacent left neighbor has the same Algotype.",
        expected_outcome="A = count(T_i = T_(i-1)) / N",
        evidence_status="partially_specified",
        executable_spec_status="needs_decision",
        metric_name="Aggregation Value",
        metric_formula="A = sum_i indicator(T_i = T_(i-1)) / N; i=0 behavior not printed",
        source_section="Methods 3.4; Figure 8C",
        source_locator="supplied markdown lines 142-150, 248",
        source_basis="supplied_journal_text+official_arxiv_equation_recovery",
        ambiguity_codes="AGGREGATION_BOUNDARY_UNDEFINED|METRIC_SCALE_AMBIGUOUS",
        ambiguity_note="The first-cell term and percent/fraction presentation require an explicit specification decision.",
        requires_code_reconciliation=True,
    )

    controls = (
        ("BUBBLE-INSERTION", "reported Bubble+Insertion label; prose also says identical Bubble control", 0.61, 0.04),
        ("BUBBLE-SELECTION", "reported Bubble+Selection label; prose also says identical Bubble control", 0.65, 0.05),
        ("INSERTION-SELECTION", "reported Insertion+Selection label; prose also says identical Bubble control", 0.57, 0.04),
    )
    for slug, composition, mean, std in controls:
        add(
            claim_id=f"F08-A-CONTROL-{slug}",
            figure=8,
            panel="A",
            claim_group="aggregation_control",
            claim_kind="peak_mean",
            claim_text=f"Reported negative-control peak aggregation for {composition}.",
            expected_outcome="no deviation from the appropriate random-assortment null",
            evidence_status="contradictory",
            executable_spec_status="needs_decision",
            architecture="cell_view",
            algorithm="control identity unresolved",
            algotype_composition=composition,
            sorting_direction="increasing",
            fault_type="none",
            fault_count=0,
            metric_name="peak mean Aggregation Value",
            reported_mean=mean,
            reported_variability=std,
            variability_type="standard deviation",
            reported_p_value="journal Results says <<0.01; captions say no significant deviation",
            reported_unit="fraction",
            source_section="Results 4.4; Figure 8 caption",
            source_locator="supplied markdown lines 194, 206, 248",
            ambiguity_codes="NEGATIVE_CONTROL_CONTRADICTION|AGGREGATION_BOUNDARY_UNDEFINED",
            ambiguity_note="Control identity and significance direction cannot be resolved from paper prose alone.",
            requires_code_reconciliation=True,
            **common_unique,
        )

    peaks = (
        ("BUBBLE-SELECTION", "Bubble+Selection", 0.72, 42),
        ("BUBBLE-INSERTION", "Bubble+Insertion", 0.65, 21),
        ("SELECTION-INSERTION", "Selection+Insertion", 0.69, 19),
        ("THREE-WAY", "Bubble+Insertion+Selection", 0.62, 22),
    )
    for slug, composition, mean, progress in peaks:
        codes = ["ALGOTYPE_ASSIGNMENT_CONFLICT", "AGGREGATION_BOUNDARY_UNDEFINED"]
        if slug == "THREE-WAY":
            codes.append("AGGREGATION_NULL_CONFLICT")
        add(
            claim_id=f"F08-A-CHIMERA-{slug}-PEAK",
            figure=8,
            panel="A",
            claim_group="aggregation",
            claim_kind="peak_mean",
            claim_text=f"{composition} reaches a transient mean aggregation peak.",
            expected_outcome=f"peak mean {mean} at {progress}% of normalized sorting progress",
            executable_spec_status="needs_decision",
            architecture="cell_view",
            algorithm=composition,
            algotype_composition="reported equal representation; assignment also described as random",
            sorting_direction="increasing for all Algotypes",
            fault_type="none",
            fault_count=0,
            stopping_rule="fully sorted OR no legal improvement; unchanged array checked twice",
            metric_name="peak mean Aggregation Value",
            reported_mean=mean,
            reported_p_value="<<0.01 vs stated negative control",
            reported_value_text=f"maximum at {progress}% of sorting process",
            reported_unit="fraction",
            source_section="Results 4.4; Figure 8A",
            source_locator="supplied markdown line 208",
            ambiguity_codes="|".join(codes),
            ambiguity_note="Null, composition enforcement, and aggregation boundary require explicit decisions.",
            requires_code_reconciliation=True,
            **common_unique,
        )
    add(
        claim_id="F08-A-UNIQUE-START-END-BASELINE",
        figure=8,
        panel="A",
        claim_group="aggregation",
        claim_kind="endpoint",
        claim_text="Unique-value chimeras begin and end near Aggregation Value 0.5.",
        expected_outcome="mean aggregation approximately 0.5 at 0% and 100% progress",
        executable_spec_status="needs_decision",
        architecture="cell_view",
        algorithm="all pairwise and three-way mixes",
        algotype_composition="random assignment; reported equal representation",
        sorting_direction="increasing for all Algotypes",
        metric_name="mean Aggregation Value",
        reported_mean=0.5,
        reported_value_text="approximately 0.5 at both start and end",
        reported_unit="fraction",
        source_section="Results 4.4; Figure 8 caption",
        source_locator="supplied markdown lines 208, 248",
        ambiguity_codes="AGGREGATION_NULL_CONFLICT|ALGOTYPE_ASSIGNMENT_CONFLICT|AGGREGATION_BOUNDARY_UNDEFINED",
        ambiguity_note="A universal 0.5 baseline is inconsistent with a simple equal three-type same-neighbor null.",
        requires_code_reconciliation=True,
        **common_unique,
    )

    duplicate_common = dict(
        array_size=100,
        repetition_count=100,
        value_distribution="integers 1..10 with exactly 10 copies of each value",
        architecture="cell_view",
        sorting_direction="increasing for all Algotypes",
        fault_type="none",
        fault_count=0,
        stopping_rule="fully sorted OR no legal improvement; unchanged array checked twice",
    )
    for slug, composition, final_mean in (
        ("BUBBLE-SELECTION", "Bubble+Selection", 0.65),
        ("INSERTION-SELECTION", "Insertion+Selection", 0.70),
    ):
        add(
            claim_id=f"F08-D-{slug}-FINAL",
            figure=8,
            panel="D",
            claim_group="duplicate_value_aggregation",
            claim_kind="final_mean",
            claim_text=f"Duplicate-value {composition} retains elevated final aggregation.",
            expected_outcome=f"final mean Aggregation Value {final_mean}",
            executable_spec_status="needs_decision",
            algorithm=composition,
            algotype_composition="two-type mix; exact counts unstated",
            metric_name="final mean Aggregation Value",
            reported_mean=final_mean,
            reported_unit="fraction",
            source_section="Results 4.4; Figure 8D",
            source_locator="supplied markdown lines 210-212",
            ambiguity_codes="AGGREGATION_BOUNDARY_UNDEFINED|ALGOTYPE_ASSIGNMENT_CONFLICT|SORTEDNESS_DUPLICATE_CONFLICT",
            ambiguity_note="Duplicate treatment conflicts with the strict printed Sortedness comparator.",
            requires_code_reconciliation=True,
            **duplicate_common,
        )
    for slug, composition, peak, progress in (
        ("BUBBLE-SELECTION", "Bubble+Selection", 0.69, 100),
        ("BUBBLE-INSERTION", "Bubble+Insertion", 0.63, 13),
        ("SELECTION-INSERTION", "Selection+Insertion", 0.71, 100),
    ):
        add(
            claim_id=f"F08-D-{slug}-PEAK",
            figure=8,
            panel="D",
            claim_group="duplicate_value_aggregation",
            claim_kind="peak_mean",
            claim_text=f"Duplicate-value {composition} reaches a reported aggregation maximum.",
            expected_outcome=f"peak mean {peak} at {progress}% normalized progress",
            executable_spec_status="needs_decision",
            algorithm=composition,
            algotype_composition="two-type mix; exact counts unstated",
            metric_name="peak mean Aggregation Value",
            reported_mean=peak,
            reported_value_text=f"maximum at {progress}% of sorting process",
            reported_unit="fraction",
            source_section="Results 4.4; Figure 8D",
            source_locator="supplied markdown line 218",
            ambiguity_codes="AGGREGATION_BOUNDARY_UNDEFINED|ALGOTYPE_ASSIGNMENT_CONFLICT|SORTEDNESS_DUPLICATE_CONFLICT",
            ambiguity_note="Aggregation boundary, exact composition, and equal-value Sortedness semantics need decisions.",
            requires_code_reconciliation=True,
            **duplicate_common,
        )
    add(
        claim_id="F08-E-DUPLICATE-EXAMPLES",
        figure=8,
        panel="E",
        claim_group="duplicate_value_aggregation",
        claim_kind="illustrative_endpoints",
        claim_text="Two duplicate-value final examples show respectively no visible within-value clustering and clear clustering.",
        expected_outcome="both endpoint patterns are possible under the reported stochastic process",
        evidence_status="partially_specified",
        executable_spec_status="needs_decision",
        algorithm="mixed Algotypes",
        algotype_composition="not stated for the two examples",
        metric_name="visual within-value clustering",
        source_section="Figure 8E caption",
        source_locator="supplied markdown line 248",
        ambiguity_codes="SAMPLE_SIZE_UNREPORTED|ALGOTYPE_ASSIGNMENT_CONFLICT",
        ambiguity_note="Example scenario IDs, policies, and seeds are not supplied.",
        requires_code_reconciliation=True,
        **duplicate_common,
    )

    # Figure 9: opposite directions, unique values.
    opposite = (
        ("A", "Bubble decreasing + Selection increasing", 42.50, "drops, rebounds to ~48, then ends below 44"),
        ("B", "Bubble increasing + Insertion decreasing", 73.73, "approximately monotonic increase; ends above 50"),
        ("C", "Insertion increasing + Selection decreasing", 38.31, "approximately monotonic decrease; ends below 50"),
    )
    for panel, composition, final_mean, trajectory in opposite:
        add(
            claim_id=f"F09-{panel}-FINAL-SORTEDNESS",
            figure=9,
            panel=panel,
            claim_group="opposite_direction_chimeras",
            claim_kind="final_mean_and_trajectory",
            claim_text=f"Opposite-direction {composition} stabilizes without reaching 100% Sortedness.",
            expected_outcome=trajectory,
            evidence_status="partially_specified",
            executable_spec_status="needs_decision",
            architecture="cell_view",
            algorithm=composition,
            algotype_composition="two-type mix; exact counts unstated",
            sorting_direction=composition,
            fault_type="none",
            fault_count=0,
            stopping_rule="stable/no further change; main thread checks unchanged array twice",
            metric_name="average final Sortedness",
            reported_mean=final_mean,
            reported_value_text=trajectory,
            reported_unit="percent Sortedness",
            source_section="Results 4.4; Figure 9",
            source_locator="supplied markdown lines 220, 262",
            ambiguity_codes="SAMPLE_SIZE_UNREPORTED|ALGOTYPE_ASSIGNMENT_CONFLICT|STOP_RULE_CONFLICT|METRIC_SCALE_AMBIGUOUS",
            ambiguity_note="Repetition count and exact mix are not stated for Figure 9.",
            requires_code_reconciliation=True,
            array_size=100,
            repetition_count=None,
            value_distribution="random permutation of integers 1..100 without duplication",
        )
    add(
        claim_id="F09-ABC-STARTING-SORTEDNESS",
        figure=9,
        panel="A|B|C",
        claim_group="opposite_direction_chimeras",
        claim_kind="initial_mean",
        claim_text="All three unique-value opposite-direction conditions start near 50% Sortedness.",
        expected_outcome="comparable starting Sortedness near 50%",
        executable_spec_status="needs_decision",
        array_size=100,
        value_distribution="random permutation of integers 1..100 without duplication",
        architecture="cell_view",
        algorithm="three pairwise mixes",
        algotype_composition="two-type mixes; exact counts unstated",
        sorting_direction="opposed within each array",
        metric_name="initial Sortedness",
        reported_mean=50.0,
        reported_value_text="approximately 50%",
        reported_unit="percent Sortedness",
        source_section="Results 4.4",
        source_locator="supplied markdown line 220",
        ambiguity_codes="SAMPLE_SIZE_UNREPORTED|ALGOTYPE_ASSIGNMENT_CONFLICT|METRIC_SCALE_AMBIGUOUS",
        ambiguity_note="Matching/tolerance for comparable initial Sortedness is not defined.",
        requires_code_reconciliation=True,
    )
    add(
        claim_id="F09-ABC-WINNER-ORDER",
        figure=9,
        panel="A|B|C",
        claim_group="opposite_direction_chimeras",
        claim_kind="qualitative_rank",
        claim_text="Relative effectiveness ranks Bubble > Selection > Insertion.",
        expected_outcome="Bubble > Selection > Insertion under the paper's directional interpretation",
        evidence_status="partially_specified",
        executable_spec_status="needs_decision",
        array_size=100,
        value_distribution="random permutation of integers 1..100 without duplication",
        architecture="cell_view",
        algorithm="Bubble|Selection|Insertion",
        sorting_direction="opposed pairwise",
        metric_name="direction-specific final Sortedness",
        source_section="Figure 9 caption",
        source_locator="supplied markdown line 262",
        ambiguity_codes="SAMPLE_SIZE_UNREPORTED|TEST_SPECIFICATION_INCOMPLETE",
        ambiguity_note="No formal winner estimand, uncertainty, or hypothesis test is provided.",
        requires_code_reconciliation=True,
    )
    add(
        claim_id="F09-ABC-AGGREGATION-RISE",
        figure=9,
        panel="A|B|C",
        claim_group="opposite_direction_chimeras",
        claim_kind="qualitative_endpoint",
        claim_text="Aggregation rises from approximately 0.5 and stabilizes above its starting average in all conditions.",
        expected_outcome="final mean Aggregation Value > initial mean approximately 0.5",
        evidence_status="partially_specified",
        executable_spec_status="needs_decision",
        array_size=100,
        value_distribution="random permutation of integers 1..100 without duplication",
        architecture="cell_view",
        algorithm="three pairwise mixes",
        sorting_direction="opposed pairwise",
        metric_name="mean Aggregation Value",
        reported_mean=0.5,
        reported_value_text="starting average ~0.5; exact final means not quoted",
        reported_unit="fraction",
        source_section="Figure 9 caption",
        source_locator="supplied markdown line 262",
        ambiguity_codes="PANEL_VALUE_NOT_QUOTED|AGGREGATION_BOUNDARY_UNDEFINED|SAMPLE_SIZE_UNREPORTED",
        ambiguity_note="Only the direction of change is textually specified.",
        requires_code_reconciliation=True,
    )

    # Figure 10: exact plotted numbers are explicitly unavailable from supplied extraction.
    for panel, composition in (
        ("A", "Bubble decreasing + Selection increasing"),
        ("B", "Bubble increasing + Insertion decreasing"),
        ("C", "Insertion increasing + Selection decreasing"),
    ):
        add(
            claim_id=f"F10-{panel}-REPEATED-OPPOSITE",
            figure=10,
            panel=panel,
            claim_group="opposite_direction_duplicate_chimeras",
            claim_kind="not_recoverable_numeric_panel",
            claim_text=f"Repeated-value {composition} reaches a stable Sortedness/Aggregation state similar to Figure 9.",
            expected_outcome="stable non-100% Sortedness; final aggregation above starting ~0.5",
            evidence_status="not_recoverable",
            executable_spec_status="unavailable",
            array_size=100,
            repetition_count=None,
            value_distribution="integers 1..10 with exactly 10 copies of each value",
            architecture="cell_view",
            algorithm=composition,
            algotype_composition="two-type mix; exact counts unstated",
            sorting_direction=composition,
            fault_type="none",
            fault_count=0,
            stopping_rule="stable/no further change; exact criterion inherited but not uniquely stated",
            metric_name="average final Sortedness and Aggregation Value",
            reported_mean=None,
            reported_value_text="exact plotted endpoint not recoverable; qualitative caption retained",
            source_section="Results 4.4; Figure 10 caption",
            source_locator="supplied markdown lines 220, 266; supplied figure-10.png",
            ambiguity_codes=(
                "F10_NUMERIC_NOT_RECOVERABLE|FIGURE_EXTRACTION_CORRUPT|SAMPLE_SIZE_UNREPORTED|"
                "SORTEDNESS_DUPLICATE_CONFLICT|AGGREGATION_BOUNDARY_UNDEFINED|STOP_RULE_CONFLICT"
            ),
            ambiguity_note="No numerical value was guessed from the corrupted bitmap.",
            requires_code_reconciliation=True,
        )

    return records


def to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    for record in records:
        metric = str(record.get("metric_name") or "").lower()
        if record.get("metric_formula") is None:
            if "aggregation" in metric:
                record["metric_formula"] = "A = sum_i indicator(T_i = T_(i-1)) / N; i=0 handling unresolved"
            elif "sortedness" in metric:
                record["metric_formula"] = "S = sum_i indicator(i=0 or V_i > V_(i-1)) / N; percentage scaling implicit"
            elif "monotonicity" in metric:
                record["metric_formula"] = "E = sum_i indicator(i>0 and V_i < V_(i-1))"
            elif "delayed gratification" in metric:
                record["metric_formula"] = "D = (DeltaS_increasing - DeltaS_decreasing) / DeltaS_decreasing"
            elif "step" in metric or "swap" in metric:
                record["metric_formula"] = "sum event costs under cost_definition"
            else:
                record["metric_formula"] = "not applicable or not numerically defined"
        if record.get("cost_definition") is None:
            record["cost_definition"] = "not applicable to this claim"
        if record.get("stopping_rule") is None:
            if record.get("claim_kind") in {"definition", "metric_definition", "non_empirical_context"}:
                record["stopping_rule"] = "not applicable to this claim"
            else:
                record["stopping_rule"] = "not reported at claim level"
        if record.get("placement_sampling") is None:
            fault_type = str(record.get("fault_type") or "").lower()
            fault_count = record.get("fault_count")
            if fault_type == "none" or fault_count == 0:
                record["placement_sampling"] = "not applicable (no fault)"
            elif fault_type:
                record["placement_sampling"] = "not reported"
            else:
                record["placement_sampling"] = "not applicable to this claim"
    frame = pd.DataFrame.from_records(records, columns=list(COLUMNS))
    for column, dtype in COLUMNS.items():
        frame[column] = frame[column].astype(dtype)
    return frame.sort_values(["figure", "panel", "claim_id"], kind="stable").reset_index(drop=True)


def validate_source(source_path: Path) -> list[str]:
    errors: list[str] = []
    if not source_path.is_file():
        return [f"Missing source markdown: {source_path}"]
    text = source_path.read_text(encoding="utf-8")
    for figure in range(3, 11):
        if f"Figure {figure}." not in text:
            errors.append(f"Source is missing Figure {figure} caption")
    for section in ("## 4.1.", "## 4.2.", "## 4.3.", "## 4.4."):
        if section not in text:
            errors.append(f"Source is missing Results section {section}")
    return errors


def validate_frame(frame: pd.DataFrame) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if frame.empty:
        errors.append("Registry is empty")
    duplicates = frame.loc[frame["claim_id"].duplicated(), "claim_id"].tolist()
    if duplicates:
        errors.append(f"Duplicate claim IDs: {duplicates}")
    required_non_null = [
        "schema_version",
        "research_step_id",
        "claim_id",
        "figure",
        "panel",
        "claim_group",
        "claim_kind",
        "claim_text",
        "expected_outcome",
        "evidence_status",
        "executable_spec_status",
        "source_section",
        "source_locator",
        "source_basis",
        "placement_sampling",
        "stopping_rule",
        "metric_formula",
        "cost_definition",
        "requires_code_reconciliation",
    ]
    for column in required_non_null:
        count = int(frame[column].isna().sum())
        if count:
            errors.append(f"{column} has {count} null values")
    if set(frame["figure"].dropna().astype(int)) != set(range(3, 11)):
        errors.append("Registry does not cover exactly Figures 3-10")
    for figure, required_panels in PANEL_REQUIREMENTS.items():
        observed: set[str] = set()
        for value in frame.loc[frame["figure"] == figure, "panel"].dropna():
            observed.update(str(value).split("|"))
        missing = required_panels - observed
        if missing:
            errors.append(f"Figure {figure} is missing panel coverage: {sorted(missing)}")
    invalid_ambiguities: set[str] = set()
    for value in frame["ambiguity_codes"].dropna():
        invalid_ambiguities.update(code for code in str(value).split("|") if code not in AMBIGUITIES)
    if invalid_ambiguities:
        errors.append(f"Unknown ambiguity codes: {sorted(invalid_ambiguities)}")
    unavailable = frame["evidence_status"] == "not_recoverable"
    if frame.loc[unavailable, "ambiguity_codes"].isna().any():
        errors.append("Every not_recoverable row must carry an ambiguity code")
    numeric_kinds = {"mean", "panel_mean", "peak_mean", "final_mean", "numerical_contrast", "initial_mean"}
    numeric = frame["claim_kind"].isin(numeric_kinds)
    missing_numeric = numeric & frame["reported_mean"].isna() & frame["reported_statistic"].isna()
    if missing_numeric.any():
        errors.append(f"{int(missing_numeric.sum())} numeric rows lack a mean or statistic")
    if not (frame["evidence_status"] == "contradictory").any():
        warnings.append("No contradictory rows were recorded")
    coverage = {
        str(figure): {
            "rowCount": int((frame["figure"] == figure).sum()),
            "panels": sorted(
                {
                    panel
                    for value in frame.loc[frame["figure"] == figure, "panel"].dropna()
                    for panel in str(value).split("|")
                }
            ),
        }
        for figure in range(3, 11)
    }
    statuses = {str(key): int(value) for key, value in frame["evidence_status"].value_counts().sort_index().items()}
    executable = {
        str(key): int(value)
        for key, value in frame["executable_spec_status"].value_counts().sort_index().items()
    }
    ambiguity_counter: Counter[str] = Counter()
    for value in frame["ambiguity_codes"].dropna():
        ambiguity_counter.update(str(value).split("|"))
    return {
        "schemaVersion": SCHEMA_VERSION,
        "researchStepId": STEP_ID,
        "passed": not errors,
        "rowCount": int(len(frame)),
        "uniqueClaimIdCount": int(frame["claim_id"].nunique()),
        "figureCoverage": coverage,
        "evidenceStatusCounts": statuses,
        "executableSpecStatusCounts": executable,
        "ambiguityCodeCounts": dict(sorted(ambiguity_counter.items())),
        "errors": errors,
        "warnings": warnings,
    }


def write_ambiguity_log(frame: pd.DataFrame, path: Path) -> None:
    affected: dict[str, list[str]] = defaultdict(list)
    for row in frame.itertuples(index=False):
        if pd.isna(row.ambiguity_codes):
            continue
        for code in str(row.ambiguity_codes).split("|"):
            affected[code].append(str(row.claim_id))
    lines = [
        "# S01 Ambiguity Log",
        "",
        "This log lists specification gaps and contradictions without resolving them from historical code. "
        "Claim IDs point into `claim_registry.csv`/`.parquet`. Code reconciliation is deferred to S02+.",
        "",
        "## Summary",
        "",
        f"- Ambiguity codes used: {len(affected)}",
        f"- Registry rows affected: {int(frame['ambiguity_codes'].notna().sum())} of {len(frame)}",
        f"- Explicit not-recoverable rows: {int((frame['evidence_status'] == 'not_recoverable').sum())}",
        "",
        "## Issues",
        "",
    ]
    for code in sorted(affected):
        ids = affected[code]
        lines.extend(
            [
                f"### {code}",
                "",
                AMBIGUITIES[code],
                "",
                f"Affected claims ({len(ids)}): {', '.join(f'`{claim_id}`' for claim_id in ids)}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_metric_definitions(path: Path) -> None:
    text = """# S01 Paper Metric Definitions

This document transcribes the paper's metric layer. Equations missing from the supplied Docling extraction were recovered visually from the official arXiv v1 manuscript (`arXiv:2401.05375v1`). They are paper definitions, not validated implementations.

## Total sorting steps, mean, and variability

- A step is counted under two analysis-specific ledgers: successful swaps only, or successful swaps plus comparisons.
- Mean: `C = (sum_i c_i) / N`.
- Printed variability: `sigma = sqrt((sum_i (c_i - C)) / N)`. The displayed equation lacks the conventional squared-deviation term and is therefore algebraically invalid as a standard deviation. Do not repair it silently; S02 must reconcile the analysis code.
- Undefined: failed swap attempts, lock attempts, repeated reads, rejected proposals, exact comparison instrumentation, and whether the inference treats matched inputs as paired.

## Monotonicity error

- Increasing-order printed rule: `E = sum_i 1[i > 0 and V_i < V_(i-1)]` (the paper writes a conditional special case for `i=0`).
- Equality is not an error under the displayed rule.
- Maximum error is `N-1` for a strictly reverse-ordered unique array.

## Sortedness

- Printed increasing-order rule: `S = (sum_i 1[i = 0 or V_i > V_(i-1)]) / N`.
- Text and axes express `S` as percent, but the displayed equation is a fraction unless multiplied by 100.
- The strict `>` makes equal adjacent values fail the Sortedness test, which conflicts with ordinary nondecreasing sorting and complicates Figures 8D/E and 10. This must not be silently changed to `>=`.

## Delayed Gratification (DG)

- Let `DeltaS_decreasing` be the magnitude of a local Sortedness drop from a preceding peak and `DeltaS_increasing` the subsequent consecutive recovery/gain.
- Displayed Methods equation and journal Figure 6 caption: `D = (DeltaS_increasing - DeltaS_decreasing) / DeltaS_decreasing`.
- The official arXiv Figure 6 caption instead describes `DeltaS_increasing / DeltaS_decreasing`.
- Undefined: segmentation of plateaus, multiple alternating drops/recoveries, zero denominators, terminal drops, aggregation across segments, and whether the per-run statistic is summed, averaged, or maximized.

## Aggregation Value

- Printed rule: `A = (sum_i 1[T_i = T_(i-1)]) / N`.
- The journal prose calls this the fraction/percentage of cells whose directly adjacent left neighbor shares their Algotype.
- Undefined: `i=0` handling (exclude, zero, or wrap to `T_(N-1)`), denominator after boundary handling, and percent-vs-fraction display.
- A universal 0.5 random baseline is appropriate only for a balanced two-type approximation under common conventions. It is not the simple same-neighbor expectation for an equal three-type mixture, yet the Results say 0.5 “in all cases.”

## Statistical tests

- The journal Methods section names z-tests; arXiv v1 also mentions t-tests.
- Reported figures provide z values and p-value text but not the exact estimator, paired/unpaired form, variance estimator, sidedness, or multiplicity adjustment.
- Treat these as historical targets to reproduce, not as sufficiently specified confirmatory analyses.
"""
    path.write_text(text, encoding="utf-8")


def write_manifest(
    output_dir: Path,
    source_path: Path,
    recovery_pdf: Path | None,
    code_path: Path,
) -> None:
    outputs = []
    for name in (
        "claim_registry.csv",
        "claim_registry.parquet",
        "ambiguity_log.md",
        "paper_metric_definitions.md",
        "validation_summary.json",
        "research_step_full_results.md",
    ):
        candidate = output_dir / name
        if candidate.is_file():
            outputs.append(
                {
                    "path": str(candidate),
                    "sizeBytes": candidate.stat().st_size,
                    "sha256": sha256_file(candidate),
                }
            )
    inputs = [
        {
            "path": str(source_path),
            "role": "supplied journal PDF extraction",
            "sizeBytes": source_path.stat().st_size,
            "sha256": sha256_file(source_path),
        }
    ]
    figures_dir = source_path.parent / "figures"
    for figure in range(3, 11):
        figure_path = figures_dir / f"figure-{figure:02d}.png"
        if figure_path.is_file():
            inputs.append(
                {
                    "path": str(figure_path),
                    "role": f"supplied extracted Figure {figure}",
                    "sizeBytes": figure_path.stat().st_size,
                    "sha256": sha256_file(figure_path),
                }
            )
    if recovery_pdf is not None and recovery_pdf.is_file():
        inputs.append(
            {
                "path": str(recovery_pdf),
                "role": "official arXiv v1 recovery copy (cache-only)",
                "sourceUrl": "https://arxiv.org/pdf/2401.05375",
                "sizeBytes": recovery_pdf.stat().st_size,
                "sha256": sha256_file(recovery_pdf),
            }
        )
    manifest = {
        "schema": "e01.s01.artifact_manifest.v1",
        "researchStepId": STEP_ID,
        "schemaVersion": SCHEMA_VERSION,
        "date": "2026-07-13",
        "inputs": inputs,
        "reproducibleCode": {
            "path": str(code_path),
            "sha256": sha256_file(code_path),
            "repositoryBacked": True,
        },
        "outputs": outputs,
    }
    (output_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-markdown", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--recovery-pdf", type=Path, default=DEFAULT_RECOVERY_PDF)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_errors = validate_source(args.source_markdown)
    frame = to_frame(build_records())
    validation = validate_frame(frame)
    validation["sourceValidationErrors"] = source_errors
    validation["passed"] = bool(validation["passed"] and not source_errors)
    if args.validate_only:
        print(json.dumps(validation, indent=2, sort_keys=True))
        return 0 if validation["passed"] else 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "claim_registry.csv", index=False, lineterminator="\n")
    frame.to_parquet(args.output_dir / "claim_registry.parquet", index=False, engine="pyarrow")
    write_ambiguity_log(frame, args.output_dir / "ambiguity_log.md")
    write_metric_definitions(args.output_dir / "paper_metric_definitions.md")
    (args.output_dir / "validation_summary.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_manifest(
        args.output_dir,
        args.source_markdown,
        args.recovery_pdf if args.recovery_pdf.is_file() else None,
        Path(__file__).resolve(),
    )
    print(json.dumps(validation, indent=2, sort_keys=True))
    return 0 if validation["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
