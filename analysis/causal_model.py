"""Freeze and validate the E02 S01 causal design.

This module deliberately does not implement any E02 simulator transition.  It
turns the S01 design decisions into deterministic, machine-readable artifacts
and checks them against the released E01 reference vocabulary.  Later E02
steps may implement the contracts, but must not silently rewrite them.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping, Sequence

import yaml

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from reference_simulator.model import (
    Architecture as E01Architecture,
    FaultMode as E01FaultMode,
    LEDGER_FIELDS,
    Policy as E01Policy,
    SEMANTICS_VERSION,
)


SCHEMA_VERSION = "e02.s01.estimand_registry.v1"
RESEARCH_STEP_ID = "S01"
FROZEN_DATE = "2026-07-14"
DEFAULT_OUTPUT = Path("/artifacts/research_steps/S01")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_record(path_text: str) -> dict[str, Any]:
    path = Path(path_text)
    record: dict[str, Any] = {"path": path_text, "exists": path.is_file()}
    if path.is_file():
        record.update({"bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return record


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _levels(*items: tuple[str, str, str, str | None]) -> list[dict[str, Any]]:
    """Build factor levels as (id, meaning, implementation status, gate)."""
    result: list[dict[str, Any]] = []
    for level_id, meaning, status, gate in items:
        row: dict[str, Any] = {
            "id": level_id,
            "meaning": meaning,
            "implementationStatus": status,
        }
        if gate:
            row["implementationGate"] = gate
        result.append(row)
    return result


def factors() -> list[dict[str, Any]]:
    return [
        {
            "id": "architecture",
            "dagNode": "A_ARCH",
            "role": "randomized_treatment",
            "interventionUnit": "scenario-run",
            "interventionContract": (
                "Set who owns proposal construction, selection, and commit while using the "
                "common S02 read/propose/validate/commit interface. One proposal evaluation is "
                "one opportunity. No architecture may gain an undeclared global read."
            ),
            "levels": _levels(
                (
                    "central_global_legacy",
                    "A global controller reads the state and chooses a conventional primary action; this is the E01 named traditional control and an intentionally bundled legacy comparator.",
                    "available_unmatched_in_E01",
                    None,
                ),
                (
                    "central_local_proposal_k1",
                    "Exactly one scheduled actor receives policy-legal information and emits one proposal; a central service validates and commits it without seeing additional state for selection.",
                    "specified_not_implemented",
                    "S02 common action interface and S03 architecture implementation",
                ),
                (
                    "distributed_local",
                    "Exactly one scheduled actor receives policy-legal information, emits its own proposal, and uses the same validator/committer; this generalizes E01 cell_view.",
                    "partially_available_in_E01",
                    "S02 must route E01 behavior through the common interface",
                ),
                (
                    "distributed_weak_coordinator",
                    "Distributed actors use the common interface and a coordinator restricted by explicit message, bandwidth, and intervention-frequency budgets.",
                    "specified_not_implemented",
                    "S03 architecture implementation",
                ),
            ),
            "forbiddenBundling": [
                "scheduler family",
                "information permission",
                "continuation policy",
                "retry policy",
                "fault consequence",
                "event budget",
            ],
        },
        {
            "id": "scheduler",
            "dagNode": "S_SCHED",
            "role": "randomized_treatment",
            "interventionUnit": "scenario-run",
            "interventionContract": (
                "Set the opportunity sequence independently of architecture. Count every offered "
                "actor opportunity, including faults, boundaries, no-ops, rejections, and conflict losses."
            ),
            "levels": _levels(
                ("deterministic_scan", "Repeat the canonical cell-ID scan.", "specified_not_implemented", "S04"),
                ("uniform_random_activation", "Select one immutable identity uniformly per opportunity with a counter-addressed stream.", "available_for_E01_cell_view", "S02/S04 must expose it to all matched architectures"),
                ("random_permutation_sweep", "Use an independently counter-addressed random permutation per sweep.", "specified_not_implemented", "S04"),
                ("synchronous_batch_deterministic_conflict", "Build a frozen batch and resolve overlapping resources with deterministic priority.", "partially_available_in_E01", "S04 validation across architectures"),
                ("fair_adversarial", "Choose opportunities adversarially subject to a frozen finite-window fairness bound and bounded lookahead.", "specified_not_implemented", "S04"),
            ),
        },
        {
            "id": "information_permission",
            "dagNode": "I_INFO",
            "role": "randomized_treatment",
            "interventionUnit": "scenario-run",
            "interventionContract": (
                "Apply a read mask before proposal construction and account for every allowed read. "
                "Policy-native local means Bubble selected-neighbor, Insertion strict-prefix, and Selection cursor-target information."
            ),
            "levels": _levels(
                ("policy_native_local", "E01 policy-specific legal observation only.", "available_for_E01_cell_view", "S02 enforcement tests"),
                ("proposal_envelope_only", "A selector sees proposal metadata but no raw array values or hidden identity fields.", "specified_not_implemented", "S02"),
                ("weak_global_summary", "Only prespecified aggregate summaries within a fixed bit budget are visible.", "specified_not_implemented", "S03"),
                ("full_global_state", "The complete current occupancy/value state is visible and every read is charged.", "available_for_E01_traditional_bundle", "S02 named permission profile"),
            ),
        },
        {
            "id": "continuation_policy",
            "dagNode": "C_CONTINUE",
            "role": "randomized_treatment",
            "interventionUnit": "scenario-run",
            "interventionContract": (
                "After a failed or blocked proposal, set whether the run stops or returns control "
                "to the scheduler. Completion, semantic quiescence, and budget precedence remain unchanged."
            ),
            "levels": _levels(
                ("stop_on_first_blocking_failure", "Terminate with named stop reason at the first eligible blocking failure.", "specified_not_implemented", "S05"),
                ("skip_and_continue", "Record the failure/no-op and offer the next scheduler opportunity.", "implicit_in_E01_reference", "S05 explicit shared primitive"),
            ),
        },
        {
            "id": "retry_policy",
            "dagNode": "R_RETRY",
            "role": "randomized_treatment",
            "interventionUnit": "scenario-run",
            "interventionContract": (
                "Set whether a failed proposal is discarded or re-enqueued. Each retry is a new, "
                "charged opportunity; it never reuses an unlogged random draw."
            ),
            "levels": _levels(
                ("no_retry", "Discard the failed proposal and continue according to the continuation policy.", "implicit_in_E01_reference", "S05 explicit shared primitive"),
                ("retry_later_bounded", "Re-enqueue after at least one other opportunity, up to a declared attempt cap.", "specified_not_implemented", "S05"),
            ),
        },
        {
            "id": "mobility_fault",
            "dagNode": "F_FAULT",
            "role": "randomized_treatment",
            "interventionUnit": "cell identity within scenario-run",
            "interventionContract": (
                "Assign immutable identity-level mobility status before execution: normal initiates "
                "and may be displaced; passive cannot initiate but may be displaced; stuck can do neither."
            ),
            "levels": _levels(
                ("normal", "Fully mobile identity.", "available_in_E01", None),
                ("passive", "Cannot initiate, may be displaced.", "available_in_E01", None),
                ("stuck", "Cannot initiate or be displaced.", "available_in_E01", None),
            ),
        },
        {
            "id": "action_failure",
            "dagNode": "F_FAULT",
            "role": "randomized_treatment",
            "interventionUnit": "proposal opportunity",
            "interventionContract": (
                "Use an exogenous named stream to fail an otherwise legal proposal after construction "
                "and before commit; preserve the same marginal stream under valid paired contrasts."
            ),
            "levels": _levels(
                ("none", "No exogenous action failure.", "available_in_E01", None),
                ("bernoulli_p", "Independent failure at frozen probability p.", "specified_not_implemented", "S05"),
                ("transient_markov", "Failure state follows a frozen two-state exogenous process.", "specified_not_implemented", "S05"),
            ),
        },
        {
            "id": "sensing_error",
            "dagNode": "F_FAULT",
            "role": "randomized_treatment",
            "interventionUnit": "logical observation read",
            "interventionContract": (
                "Transform policy-visible observations with an exogenous named stream while retaining "
                "ground-truth state for validation and charging both attempted reads and error handling."
            ),
            "levels": _levels(
                ("exact", "Observation equals the legal ground-truth projection.", "available_in_E01", None),
                ("noisy_value_or_status", "Value or status read is corrupted under a frozen error kernel.", "specified_not_implemented", "S05"),
            ),
        },
        {
            "id": "fault_placement",
            "dagNode": "F_FAULT",
            "role": "randomized_treatment_or_stratifier",
            "interventionUnit": "scenario",
            "interventionContract": (
                "Assign exactly the declared fault count unless the profile is explicitly named legacy_with_replacement. "
                "Placement is fixed before outcome execution and is part of scenario identity."
            ),
            "levels": _levels(
                ("uniform_exact", "Uniform identities without replacement.", "available_in_E01", None),
                ("clustered_exact", "Exact-count contiguous or distance-bounded cluster.", "specified_not_implemented", "S06"),
                ("boundary_exact", "Exact-count edge-biased placement.", "specified_not_implemented", "S06"),
                ("median_rank_exact", "Exact-count placement by prespecified value-rank region.", "specified_not_implemented", "S06"),
                ("adversarial_held_out", "Search-derived placement confirmed on independent streams.", "specified_not_implemented", "S06/S13"),
                ("legacy_with_replacement", "Repeated randint draws may under-realize distinct faults; sensitivity only.", "available_in_E01", None),
            ),
        },
        {
            "id": "coordinator_budget",
            "dagNode": "Q_COORD",
            "role": "randomized_treatment",
            "interventionUnit": "scenario-run",
            "interventionContract": (
                "Set coordinator message bits, candidate proposals inspected per opportunity, and "
                "minimum opportunities between interventions; charge every message and intervention."
            ),
            "levels": _levels(
                ("none", "Zero coordinator messages and interventions.", "available_in_E01_cell_view", "S03 named profile"),
                ("common_validator_only", "One proposal envelope may be validated/committed; no proposal choice or extra state read.", "specified_not_implemented", "S02/S03"),
                ("weak_frozen_budget", "A prespecified small message and intervention budget.", "specified_not_implemented", "S03"),
            ),
        },
        {
            "id": "event_budget",
            "dagNode": "B_BUDGET",
            "role": "design_treatment",
            "interventionUnit": "scenario-run",
            "interventionContract": (
                "Set a policy- and architecture-independent opportunity ceiling scaled by n and input profile. "
                "Budget exhaustion is retained as by-budget failure and right censoring for time-to-completion."
            ),
            "levels": _levels(
                ("profile_scaled_frozen", "Use the value frozen before opening a split.", "available_in_E01", "S07/S10 freeze E02 values"),
            ),
        },
    ]


def outcomes() -> list[dict[str, Any]]:
    return [
        {
            "id": "normalized_residual_error",
            "dagNode": "Y_TASK",
            "definition": "final strict adjacent inversion count divided by max(n-1,1), evaluated at complete, quiescent, or budget terminal state",
            "direction": "lower_is_better",
            "terminalHandling": "include complete, quiescent, and event_budget; invariant/infrastructure failures are missing with reason",
            "primaryScale": "paired mean difference",
            "equivalenceMargin": {"lower": -0.02, "upper": 0.02, "unit": "fraction of adjacent edges"},
        },
        {
            "id": "success_by_budget",
            "dagNode": "Y_TASK",
            "definition": "indicator that reference non-strict consensus completion occurs before the frozen opportunity budget",
            "direction": "higher_is_better",
            "terminalHandling": "complete=1; quiescent/event_budget=0; invariant/infrastructure failures are missing with reason",
            "primaryScale": "paired risk difference",
            "equivalenceMargin": {"lower": -0.02, "upper": 0.02, "unit": "probability"},
        },
        {
            "id": "completion_time",
            "dagNode": "Y_TIME",
            "definition": "number of charged opportunities until completion",
            "direction": "lower_is_better",
            "terminalHandling": "event_budget is right-censored; quiescence is a competing failure terminal; invariant/infrastructure failures are administrative failures",
            "primaryScale": "restricted mean completion time and cause-specific contrasts",
            "equivalenceMargin": {"ratioLower": 0.90, "ratioUpper": 1.10, "unit": "restricted mean opportunity ratio"},
        },
        {
            "id": "unit_weight_full_cost",
            "dagNode": "Y_COST",
            "definition": (
                "activations + observation_reads + value_comparisons + target_calculations + proposals + no_ops + "
                "rejections + memory_updates + accepted_swaps + displaced_cells + conflict_losses + coordinator_messages"
            ),
            "direction": "lower_is_better_at_equal_task_outcome",
            "terminalHandling": "report through every retained terminal state; never mix wall time into algorithmic cost",
            "primaryScale": "paired geometric mean ratio with paired difference sensitivity",
            "equivalenceMargin": {"ratioLower": 0.90, "ratioUpper": 1.10, "unit": "cost ratio"},
        },
        {
            "id": "cost_ledger_vector",
            "dagNode": "Y_COST",
            "definition": (
                "separate counts for reads, comparisons, target calculations, proposals, failed proposals, swaps, displacement, activations, "
                "coordinator messages, and wall time; component counts are never collapsed without a named projection"
            ),
            "direction": "component_specific",
            "terminalHandling": "report through every retained terminal state",
            "primaryScale": "paired difference and ratio where defined; multiplicity controlled by prespecified family",
            "equivalenceMargin": None,
        },
        {
            "id": "exact_replay_parity",
            "dagNode": "Y_TASK",
            "definition": "byte-exact final state, stop reason, event digest, and ledger identity under a declared parity contrast",
            "direction": "must_match",
            "terminalHandling": "all terminal classes must match",
            "primaryScale": "all-or-none engineering invariant",
            "equivalenceMargin": {"exact": True},
        },
    ]


def populations() -> list[dict[str, Any]]:
    common = [
        "scenario schema and invariants validate",
        "condition was assigned before outcome execution",
        "run belongs to the declared split",
        "all planned paired arms are launched or carry an explicit infrastructure-failure record",
    ]
    return [
        {
            "id": "P_PARITY",
            "purpose": "primitive and architecture-label parity gate before scientific simulation",
            "inclusion": common + ["small hand-checkable and exhaustive states", "normal/passive/stuck toy faults", "all three E01 policies"],
            "exclusion": ["no outcome-dependent exclusion"],
            "terminalHandling": "every terminal class retained; exact replay required",
        },
        {
            "id": "P_SCREENING_ITS",
            "purpose": "intention-to-simulate adaptive screening population",
            "inclusion": common + ["S10 screening split", "all frozen n/input/policy/placement strata selected by the design"],
            "exclusion": ["protected confirmatory and policy-search holdouts", "no exclusion for quiescence or budget exhaustion"],
            "terminalHandling": "complete/quiescent/event_budget retained; invariant/infrastructure failures separately accounted",
        },
        {
            "id": "P_CONFIRMATORY_ITS",
            "purpose": "primary held-out intention-to-simulate causal population",
            "inclusion": common + ["S11 confirmatory holdout", "all prespecified selected contrasts", "sizes 20, 50, 100, 200, and 500 if retained by the frozen S07 compute gate"],
            "exclusion": ["no outcome-dependent exclusion", "no policy-search holdout access"],
            "terminalHandling": "success-by-budget treats quiescent/event_budget as 0; residual state includes all three; completion time censors event_budget and treats quiescence as competing failure",
        },
        {
            "id": "P_SUPPORTED_HETEROGENEITY",
            "purpose": "descriptive and model-based effect heterogeneity within confirmatory support",
            "inclusion": common + ["P_CONFIRMATORY_ITS", "policy/size/input/placement cells with prespecified minimum support"],
            "exclusion": ["empty or unsupported factorial cells", "no extrapolation beyond observed support"],
            "terminalHandling": "same as P_CONFIRMATORY_ITS",
        },
        {
            "id": "P_E01_LEGACY_DESCRIPTIVE",
            "purpose": "bounded paper-scale legacy-bundle comparison only",
            "inclusion": common + ["n=100 paper-scale profiles", "unique 1..100 inputs", "explicit E01 evidence-layer label"],
            "exclusion": ["never pooled with E02 matched causal estimands"],
            "terminalHandling": "preserve E01 backend-specific stop and censoring metadata",
        },
    ]


def dag() -> dict[str, Any]:
    nodes = [
        ("X_SCENARIO", "Baseline scenario", "baseline"),
        ("U_STREAMS", "Exogenous named streams", "baseline"),
        ("V_BUILD", "Frozen implementation/runtime", "design_constant"),
        ("A_ARCH", "Architecture", "treatment"),
        ("S_SCHED", "Scheduler", "treatment"),
        ("I_INFO", "Information permission", "treatment"),
        ("F_FAULT", "Fault process and placement", "treatment"),
        ("C_CONTINUE", "Continuation rule", "treatment"),
        ("R_RETRY", "Retry rule", "treatment"),
        ("Q_COORD", "Coordinator budget", "treatment"),
        ("B_BUDGET", "Event budget", "treatment"),
        ("M_OBS", "Realized legal observations", "mediator"),
        ("M_OPPORTUNITY", "Opportunity/fairness process", "mediator"),
        ("M_FAULT_EXPOSURE", "Realized fault exposure", "mediator"),
        ("M_POSTFAULT", "Post-failure activity/retries", "mediator"),
        ("M_COORD", "Messages and coordination", "mediator"),
        ("M_PROPOSAL", "Proposal sequence", "mediator"),
        ("M_ACTIONS", "Accepted/rejected actions", "mediator"),
        ("M_TRAJECTORY", "State trajectory", "mediator"),
        ("M_TERMINAL", "Terminal/censoring class", "mediator"),
        ("Y_TASK", "Success and residual error", "outcome"),
        ("Y_TIME", "Completion time", "outcome"),
        ("Y_COST", "Cost ledger", "outcome"),
    ]
    edges = [
        ("X_SCENARIO", "M_OBS"), ("X_SCENARIO", "M_FAULT_EXPOSURE"),
        ("X_SCENARIO", "M_PROPOSAL"), ("U_STREAMS", "M_OPPORTUNITY"),
        ("U_STREAMS", "M_FAULT_EXPOSURE"), ("V_BUILD", "M_PROPOSAL"),
        ("V_BUILD", "M_ACTIONS"), ("V_BUILD", "Y_COST"), ("V_BUILD", "Y_TIME"),
        ("A_ARCH", "M_COORD"), ("A_ARCH", "M_PROPOSAL"),
        ("S_SCHED", "M_OPPORTUNITY"), ("I_INFO", "M_OBS"),
        ("F_FAULT", "M_FAULT_EXPOSURE"), ("C_CONTINUE", "M_POSTFAULT"),
        ("R_RETRY", "M_POSTFAULT"), ("Q_COORD", "M_COORD"),
        ("B_BUDGET", "M_TERMINAL"), ("M_OBS", "M_PROPOSAL"),
        ("M_OPPORTUNITY", "M_PROPOSAL"), ("M_OPPORTUNITY", "M_POSTFAULT"),
        ("M_FAULT_EXPOSURE", "M_PROPOSAL"), ("M_FAULT_EXPOSURE", "M_POSTFAULT"),
        ("M_FAULT_EXPOSURE", "M_ACTIONS"), ("M_POSTFAULT", "M_PROPOSAL"),
        ("M_COORD", "M_PROPOSAL"), ("M_COORD", "M_ACTIONS"),
        ("M_PROPOSAL", "M_ACTIONS"), ("M_PROPOSAL", "Y_COST"),
        ("M_ACTIONS", "M_TRAJECTORY"), ("M_ACTIONS", "Y_COST"),
        ("M_TRAJECTORY", "M_TERMINAL"), ("M_TRAJECTORY", "Y_TASK"),
        ("M_TERMINAL", "Y_TASK"), ("M_TERMINAL", "Y_TIME"),
    ]
    return {
        "nodes": [{"id": node_id, "label": label, "role": role} for node_id, label, role in nodes],
        "edges": [{"from": source, "to": target} for source, target in edges],
    }


def _estimand(
    estimand_id: str,
    title: str,
    treatment: str,
    contrast: Sequence[str],
    population: str,
    outcome: str,
    fixed: Mapping[str, str],
    hypothesis: str,
    *,
    effect: str = "total_effect",
    tier: str = "primary",
    analysis: str = "paired randomization/bootstrap with held-out confirmation",
    secondary: Sequence[str] = ("success_by_budget", "completion_time", "unit_weight_full_cost", "cost_ledger_vector"),
    decision_rule: str = "policy_native_rules_v1",
) -> dict[str, Any]:
    return {
        "id": estimand_id,
        "title": title,
        "tier": tier,
        "effectType": effect,
        "treatment": treatment,
        "contrast": {"active": contrast[0], "reference": contrast[1]},
        "fixedInterventions": {key: item for key, item in fixed.items() if key != treatment},
        "fixedDecisionRule": decision_rule,
        "standardizeOver": ["policy", "size", "input_structure", "fault_count", "placement_stratum"],
        "population": population,
        "primaryOutcome": outcome,
        "secondaryOutcomes": list(secondary),
        "effectScale": "outcome-specific paired contrast defined in outcomes",
        "adjustmentSet": ["X_SCENARIO:pairing_block"],
        "identification": "randomized factorial intervention with valid scenario pairing; condition-specific streams remain marginally valid even when exact common-random-number coupling is invalid",
        "analysisPlan": analysis,
        "hypothesis": hypothesis,
        "executableIntervention": True,
    }


def estimands() -> list[dict[str, Any]]:
    base = {
        "information_permission": "policy_native_local",
        "continuation_policy": "skip_and_continue",
        "retry_policy": "no_retry",
        "action_failure": "none",
        "sensing_error": "exact",
        "coordinator_budget": "common_validator_only",
        "event_budget": "profile_scaled_frozen",
    }
    rows = [
        _estimand(
            "E02-S01-E01", "Matched control-topology effect under faults", "architecture",
            ("distributed_local", "central_local_proposal_k1"), "P_CONFIRMATORY_ITS",
            "normalized_residual_error", {**base, "scheduler": "uniform_random_activation"},
            "After matching primitives, information, scheduling, continuation, retry, and fault semantics, the remaining topology effect is equivalent to zero on the primary task scale.",
        ),
        _estimand(
            "E02-S01-E02", "No-fault architecture parity", "architecture",
            ("distributed_local", "central_local_proposal_k1"), "P_PARITY",
            "exact_replay_parity", {**base, "scheduler": "uniform_random_activation", "mobility_fault": "normal", "fault_placement": "uniform_exact"},
            "With one proposal per opportunity and no faults, central-local and distributed labels are behaviorally identical.",
            analysis="exact event/state/ledger differential test before any scientific run",
            secondary=(),
        ),
        _estimand(
            "E02-S01-E03", "Scheduler effect within distributed control", "scheduler",
            ("random_permutation_sweep", "uniform_random_activation"), "P_CONFIRMATORY_ITS",
            "normalized_residual_error", {**base, "architecture": "distributed_local"},
            "Fairer sweep opportunities reduce residual error under faults relative to independent uniform activation.",
        ),
        _estimand(
            "E02-S01-E04", "Continuation effect after blocking failure", "continuation_policy",
            ("skip_and_continue", "stop_on_first_blocking_failure"), "P_CONFIRMATORY_ITS",
            "success_by_budget", {**base, "architecture": "distributed_local", "scheduler": "uniform_random_activation"},
            "Continuing after a local blocking failure increases success by budget.",
        ),
        _estimand(
            "E02-S01-E05", "Bounded retry effect", "retry_policy",
            ("retry_later_bounded", "no_retry"), "P_CONFIRMATORY_ITS",
            "success_by_budget", {**base, "architecture": "distributed_local", "scheduler": "uniform_random_activation", "continuation_policy": "skip_and_continue"},
            "Bounded deferred retry improves task success only for transient/action failures and incurs explicit cost.",
        ),
        _estimand(
            "E02-S01-E06", "Mobility-fault semantics effect", "mobility_fault",
            ("stuck", "passive"), "P_CONFIRMATORY_ITS",
            "normalized_residual_error", {**base, "architecture": "distributed_local", "scheduler": "uniform_random_activation", "fault_placement": "uniform_exact"},
            "Stuck faults cause more residual error than passive faults when placement and count are fixed.",
        ),
        _estimand(
            "E02-S01-E07", "Information-permission effect", "information_permission",
            ("full_global_state", "policy_native_local"), "P_CONFIRMATORY_ITS",
            "normalized_residual_error", {**base, "architecture": "central_local_proposal_k1", "scheduler": "uniform_random_activation"},
            "Additional global information changes both task outcomes and sensing cost; it is not part of a pure architecture effect.",
            decision_rule="masked_global_rank_rule_v1",
        ),
        _estimand(
            "E02-S01-E08", "Weak-coordinator effect", "coordinator_budget",
            ("weak_frozen_budget", "none"), "P_CONFIRMATORY_ITS",
            "normalized_residual_error", {**base, "architecture": "distributed_weak_coordinator", "scheduler": "uniform_random_activation"},
            "A weak coordinator can improve robustness only at a measurable message/intervention cost.",
        ),
    ]
    interactions = [
        (
            "E02-S01-E09", "Architecture-by-scheduler interaction",
            "The distributed-minus-central-local effect differs between uniform activation and random-permutation sweeps.",
            {
                "architecture": {"active": "distributed_local", "reference": "central_local_proposal_k1"},
                "scheduler": {"active": "random_permutation_sweep", "reference": "uniform_random_activation"},
            },
        ),
        (
            "E02-S01-E10", "Architecture-by-mobility-fault interaction",
            "The distributed-minus-central-local effect differs between stuck and passive faults.",
            {
                "architecture": {"active": "distributed_local", "reference": "central_local_proposal_k1"},
                "mobility_fault": {"active": "stuck", "reference": "passive"},
            },
        ),
        (
            "E02-S01-E11", "Architecture-by-continuation interaction",
            "Stopping versus continuing explains more of the bundled robustness contrast than topology alone.",
            {
                "architecture": {"active": "distributed_local", "reference": "central_local_proposal_k1"},
                "continuation_policy": {"active": "skip_and_continue", "reference": "stop_on_first_blocking_failure"},
            },
        ),
        (
            "E02-S01-E12", "Scheduler-by-fault interaction",
            "Scheduler fairness matters more under stuck than passive faults.",
            {
                "scheduler": {"active": "random_permutation_sweep", "reference": "uniform_random_activation"},
                "mobility_fault": {"active": "stuck", "reference": "passive"},
            },
        ),
    ]
    for estimand_id, title, hypothesis, factorial_contrast in interactions:
        treatment_factors = list(factorial_contrast)
        fixed = {
            key: item for key, item in {**base, "scheduler": "uniform_random_activation"}.items()
            if key not in treatment_factors
        }
        if "architecture" not in treatment_factors:
            fixed["architecture"] = "distributed_local"
        rows.append(
            {
                "id": estimand_id,
                "title": title,
                "tier": "primary_interaction",
                "effectType": "factorial_difference_in_differences",
                "treatment": "*".join(treatment_factors),
                "contrast": {"active": "joint active cell", "reference": "additive main-effect expectation"},
                "factorialContrast": factorial_contrast,
                "fixedInterventions": fixed,
                "fixedDecisionRule": "policy_native_rules_v1",
                "standardizeOver": ["policy", "size", "input_structure", "fault_count", "placement_stratum"],
                "population": "P_CONFIRMATORY_ITS",
                "primaryOutcome": "normalized_residual_error",
                "secondaryOutcomes": ["success_by_budget", "completion_time", "unit_weight_full_cost", "cost_ledger_vector"],
                "effectScale": "paired difference-in-differences on the outcome-specific scale",
                "adjustmentSet": ["X_SCENARIO:pairing_block"],
                "identification": "randomized crossed interventions with complete support in the selected confirmatory design",
                "analysisPlan": "paired marginal contrast plus prespecified hierarchical interaction model; no post-treatment covariates",
                "hypothesis": hypothesis,
                "executableIntervention": True,
            }
        )
    rows.append(
        {
            "id": "E02-S01-D01",
            "title": "E01 native traditional-versus-cell-view bundle",
            "tier": "descriptive_anchor",
            "effectType": "descriptive_not_pure_architecture",
            "treatment": "architecture",
            "contrast": {"active": "distributed_local with E01-native settings", "reference": "central_global_legacy with E01-native settings"},
            "fixedInterventions": {},
            "standardizeOver": ["E01 paper-scale scenario pairs where available"],
            "population": "P_E01_LEGACY_DESCRIPTIVE",
            "primaryOutcome": "normalized_residual_error",
            "secondaryOutcomes": ["success_by_budget", "cost_ledger_vector"],
            "effectScale": "descriptive paired or unpaired contrast as E01 observability permits",
            "adjustmentSet": [],
            "identification": "not identified as an architecture-only effect because action primitive, information, scheduler, continuation, and historical semantics are bundled",
            "analysisPlan": "report as bounded legacy anchor; never pool with matched causal estimands",
            "hypothesis": "The native bundle can differ substantially even if the matched topology effect is null.",
            "executableIntervention": True,
        }
    )
    return rows


def registry() -> dict[str, Any]:
    return {
        "schemaVersion": SCHEMA_VERSION,
        "researchStepId": RESEARCH_STEP_ID,
        "frozenDate": FROZEN_DATE,
        "experimentId": "E02",
        "title": "Factorial decomposition of architecture, scheduling, information, and faults",
        "evidenceLayer": "E01 clean-room reference R; not exact publication replay",
        "designStatus": "frozen_for_S02_implementation",
        "frozenQuestion": "Can pathways from architecture and scheduler to success and cost be represented with explicit interventions and measurable mediators?",
        "primaryDecisionRule": (
            "Classify the residual matched topology effect as equivalent only when the simultaneous held-out intervals "
            "for normalized residual-error difference and success risk difference lie within [-0.02,0.02], and the "
            "restricted-mean completion-time and unit-cost ratios lie within [0.90,1.10]. Exact parity E02-S01-E02 "
            "must pass before scientific runs."
        ),
        "evidenceBoundaries": [
            "Use E01 R as a reconstructed clean-room baseline, not exact publication replay.",
            "Keep frozen public C recorded swaps separate from R activations.",
            "Retain exact-legacy versus corrected placement and strict versus non-strict duplicate labels.",
            "Treat budget terminals as by-budget failure and right censoring for completion time; never as stable completion.",
            "Do not access E01 or E02 protected holdouts outside their declared gate.",
        ],
        "mediatorPolicy": {
            "primaryEffects": "total effects and randomized factorial controlled contrasts",
            "allowed": [
                "Estimate a factor effect by physically intervening on that factor while fixing other treatment factors.",
                "Use baseline scenario descriptors and pairing blocks for precision or prespecified heterogeneity.",
                "Report mediator distributions descriptively by assigned treatment.",
            ],
            "prohibited": [
                "Do not adjust primary total-effect models for observations, opportunities, proposals, failures, retries, accepted actions, trajectory, terminal class, or any cost counter.",
                "Do not report natural direct or natural indirect effects; cross-world mediator assumptions are not justified.",
                "Do not call regression attenuation after mediator adjustment a causal decomposition.",
                "Do not condition task effects on completion because completion is post-treatment.",
            ],
            "futureInterventionalMediation": (
                "S12 may estimate interventional analog effects only for manipulable factors already represented here "
                "and only after auditing exposure-induced mediator-outcome confounding; otherwise mechanism measures remain descriptive."
            ),
        },
        "randomizationAndPairing": {
            "assignment": "randomize factorial condition within immutable scenario/pairing block before execution",
            "shared": ["initial values/occupancy", "fault map where the map intervention is fixed", "Algotype/policy assignment", "direction assignment"],
            "streamRule": "couple named streams only when consumption semantics match; otherwise retain equal marginals with condition-addressed streams",
            "workerRule": "worker order never enters seeds",
        },
        "multiplicity": {
            "primaryFamily": [f"E02-S01-E{i:02d}" for i in range(1, 13)],
            "rule": "simultaneous 95% familywise intervals by max-t paired bootstrap or Holm-adjusted paired randomization tests; estimation and intervals remain primary",
            "secondary": "false-discovery-rate control within named outcome/component families; effect sizes and uncertainty always reported",
        },
        "dag": dag(),
        "factors": factors(),
        "outcomes": outcomes(),
        "analysisPopulations": populations(),
        "estimands": estimands(),
        "implementationEscalations": [
            {
                "id": "ESC-S02-01",
                "condition": "central_global_legacy requires global scans/nonlocal actions, especially traditional Selection",
                "decision": "keep it as a descriptive bundled comparator; do not use it as the reference arm for a pure architecture effect",
            },
            {
                "id": "ESC-S02-02",
                "condition": "central_local_proposal_k1 and distributed_local cannot produce byte-identical events through one common interface",
                "decision": "stop S02 and return for design review; E02-S01-E02 is a hard gate",
            },
            {
                "id": "ESC-S04-01",
                "condition": "an architecture cannot consume the same declared opportunity unit under a scheduler",
                "decision": "mark the factorial cell structurally unsupported; do not impute or relabel it",
            },
        ],
    }


def _topological_order(graph: Mapping[str, Any]) -> list[str]:
    nodes = [row["id"] for row in graph["nodes"]]
    incoming = {node: 0 for node in nodes}
    outgoing = {node: [] for node in nodes}
    for edge in graph["edges"]:
        incoming[edge["to"]] += 1
        outgoing[edge["from"]].append(edge["to"])
    ready = sorted(node for node, degree in incoming.items() if degree == 0)
    order: list[str] = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        for target in sorted(outgoing[node]):
            incoming[target] -= 1
            if incoming[target] == 0:
                ready.append(target)
                ready.sort()
    if len(order) != len(nodes):
        raise ValueError("causal graph contains a directed cycle")
    return order


def validate_design(value: Mapping[str, Any]) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}

    def add(name: str, passed: bool, detail: str) -> None:
        checks[name] = {"passed": bool(passed), "detail": detail}

    graph = value["dag"]
    node_ids = [row["id"] for row in graph["nodes"]]
    roles = {row["id"]: row["role"] for row in graph["nodes"]}
    edge_nodes = {item for edge in graph["edges"] for item in (edge["from"], edge["to"])}
    add("dagNodeIdsUnique", len(node_ids) == len(set(node_ids)), f"{len(node_ids)} nodes")
    add("dagEdgesDefined", edge_nodes <= set(node_ids), f"{len(graph['edges'])} edges")
    try:
        topo = _topological_order(graph)
        add("dagAcyclic", True, f"topological order contains {len(topo)} nodes")
    except ValueError as exc:
        topo = []
        add("dagAcyclic", False, str(exc))

    factor_map = {row["id"]: row for row in value["factors"]}
    factor_levels = {
        factor_id: {level["id"] for level in factor["levels"]}
        for factor_id, factor in factor_map.items()
    }
    outcome_map = {row["id"]: row for row in value["outcomes"]}
    population_map = {row["id"]: row for row in value["analysisPopulations"]}
    estimand_ids = [row["id"] for row in value["estimands"]]
    add("factorIdsUnique", len(factor_map) == len(value["factors"]), f"{len(factor_map)} factors")
    add("estimandIdsUnique", len(estimand_ids) == len(set(estimand_ids)), f"{len(estimand_ids)} estimands")
    add("primaryFamilyComplete", set(value["multiplicity"]["primaryFamily"]) <= set(estimand_ids), "12/12 primary estimands named")

    factor_contracts = all(
        row.get("interventionContract") and row.get("levels") and
        len({level["id"] for level in row["levels"]}) == len(row["levels"]) and
        all(level.get("implementationStatus") for level in row["levels"])
        for row in value["factors"]
    )
    add("factorInterventionContractsComplete", factor_contracts, "all factors have unique levels, semantics, and implementation status")

    primary = [row for row in value["estimands"] if row["tier"].startswith("primary")]
    defined = True
    level_errors: list[str] = []
    for row in primary:
        base_treatments = row["treatment"].split("*")
        defined &= all(item in factor_map for item in base_treatments)
        defined &= row["primaryOutcome"] in outcome_map
        defined &= row["population"] in population_map
        defined &= bool(row.get("identification")) and bool(row.get("analysisPlan"))
        defined &= row.get("executableIntervention") is True
        for factor_id, level_id in row["fixedInterventions"].items():
            if factor_id not in factor_levels or level_id not in factor_levels[factor_id]:
                level_errors.append(f"{row['id']}:fixed:{factor_id}={level_id}")
        if "factorialContrast" in row:
            if set(row["factorialContrast"]) != set(base_treatments):
                level_errors.append(f"{row['id']}:factorial-treatment-mismatch")
            for factor_id, contrast in row["factorialContrast"].items():
                for arm in ("active", "reference"):
                    if contrast[arm] not in factor_levels.get(factor_id, set()):
                        level_errors.append(f"{row['id']}:{factor_id}:{arm}={contrast[arm]}")
        elif len(base_treatments) == 1:
            factor_id = base_treatments[0]
            for arm in ("active", "reference"):
                if row["contrast"][arm] not in factor_levels.get(factor_id, set()):
                    level_errors.append(f"{row['id']}:{factor_id}:{arm}={row['contrast'][arm]}")
    add("primaryEstimandsFullyDefined", defined, f"{len(primary)}/{len(primary)} include intervention, contrast, outcome, population, identification, and analysis")
    add("estimandLevelsResolve", not level_errors, "none" if not level_errors else ",".join(level_errors))

    baseline_adjustments = True
    forbidden_adjustments: list[str] = []
    for row in primary:
        for item in row["adjustmentSet"]:
            node_id = item.split(":", 1)[0]
            if roles.get(node_id) not in {"baseline", "design_constant"}:
                baseline_adjustments = False
                forbidden_adjustments.append(f"{row['id']}:{item}")
    add("noPostTreatmentAdjustment", baseline_adjustments, "none" if not forbidden_adjustments else ",".join(forbidden_adjustments))

    margins = all(outcome_map[row["primaryOutcome"]].get("equivalenceMargin") is not None for row in primary)
    add("primaryEquivalenceMarginsFrozen", margins, "all primary outcomes have scale-specific margins")
    add("legacyContrastNotMisidentified", value["estimands"][-1]["identification"].startswith("not identified"), "E02-S01-D01 explicitly remains descriptive")
    add("naturalMediationProhibited", any("natural direct" in item for item in value["mediatorPolicy"]["prohibited"]), "cross-world effects excluded")

    e01_checks = {
        "semanticsVersion": SEMANTICS_VERSION == "E01-reference-v1",
        "policies": {item.value for item in E01Policy} == {"Bubble", "Insertion", "Selection"},
        "architectures": {item.value for item in E01Architecture} == {"cell_view", "traditional"},
        "faults": {item.value for item in E01FaultMode} == {"normal", "passive", "stuck"},
        "ledger": tuple(LEDGER_FIELDS) == (
            "activations", "observationReads", "valueComparisons", "proposals", "noOps",
            "rejections", "memoryUpdates", "acceptedSwaps", "displacedCells", "conflictLosses",
        ),
    }
    add("e01VocabularyAligned", all(e01_checks.values()), json.dumps(e01_checks, sort_keys=True))

    unsupported = [
        (factor["id"], level["id"])
        for factor in value["factors"]
        for level in factor["levels"]
        if "not_implemented" in level["implementationStatus"]
    ]
    gates_complete = all(
        level.get("implementationGate")
        for factor in value["factors"]
        for level in factor["levels"]
        if "not_implemented" in level["implementationStatus"]
    )
    add("unsupportedLevelsHaveImplementationGates", gates_complete, f"{len(unsupported)} future levels explicitly gated")
    add("selectionGlobalPrimitiveEscalated", any(row["id"] == "ESC-S02-01" for row in value["implementationEscalations"]), "legacy global Selection is not treated as matched")

    success = all(row["passed"] for row in checks.values())
    return {
        "schemaVersion": "e02.s01.validation_summary.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "success": success,
        "checks": checks,
        "counts": {
            "dagNodes": len(node_ids),
            "dagEdges": len(graph["edges"]),
            "factors": len(factor_map),
            "factorLevels": sum(len(row["levels"]) for row in value["factors"]),
            "primaryEstimands": len(primary),
            "descriptiveEstimands": len(value["estimands"]) - len(primary),
            "analysisPopulations": len(population_map),
            "outcomes": len(outcome_map),
            "futureImplementationGates": len(unsupported),
        },
        "topologicalOrder": topo,
        "e01SemanticsVersion": SEMANTICS_VERSION,
        "outcomeClassification": "supportive",
        "caveatsOrBlockers": [
            "The global traditional controller remains a bundled descriptive comparator, not a pure architecture intervention.",
            "Matched central-local, scheduler, retry, sensing, action-failure, and coordinator levels are contracts awaiting S02-S05 implementation.",
            "Natural direct and indirect effects are not identified; mediator traces are descriptive unless their generating factor is randomized.",
        ],
    }


def _escape_svg(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write_dag_svg(path: Path, graph: Mapping[str, Any]) -> None:
    """Write a deterministic, dependency-free layered SVG."""
    roles = {
        "baseline": (70, "#e8eef7", "#365a7a"),
        "design_constant": (70, "#f1f1f1", "#666666"),
        "treatment": (285, "#e3f3e8", "#2f6b43"),
        "mediator": (570, "#fff2cf", "#8a6719"),
        "outcome": (855, "#f8dfdf", "#8b3e3e"),
    }
    grouped: dict[str, list[dict[str, str]]] = {role: [] for role in roles}
    for node in graph["nodes"]:
        grouped[node["role"]].append(node)
    positions: dict[str, tuple[float, float]] = {}
    for role, rows in grouped.items():
        x = roles[role][0]
        if role == "baseline":
            start = 105
        elif role == "design_constant":
            start = 335
        elif role == "outcome":
            start = 105
        else:
            start = 55
        gap = 82 if role == "treatment" else (68 if role == "mediator" else 115)
        for index, node in enumerate(rows):
            positions[node["id"]] = (x, start + index * gap)
    width, height = 1120, 820
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">E02 S01 causal DAG</title>',
        '<desc id="desc">Randomized implementation factors act through observations, opportunities, faults, proposals, actions, trajectories, and terminal classification to task, time, and cost outcomes.</desc>',
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L8,3 z" fill="#68737d"/></marker></defs>',
        '<rect x="0" y="0" width="1120" height="820" fill="#ffffff"/>',
        '<text x="32" y="28" font-family="sans-serif" font-size="20" font-weight="700">E02 S01 — frozen causal model</text>',
        '<text x="32" y="48" font-family="sans-serif" font-size="11" fill="#555">Treatments are assigned; yellow nodes are post-treatment mediators and are prohibited primary-model adjustments.</text>',
    ]
    node_lookup = {row["id"]: row for row in graph["nodes"]}
    for edge in graph["edges"]:
        x1, y1 = positions[edge["from"]]
        x2, y2 = positions[edge["to"]]
        parts.append(
            f'<path d="M{x1 + 205},{y1 + 22} C{x1 + 235},{y1 + 22} {x2 - 30},{y2 + 22} {x2},{y2 + 22}" fill="none" stroke="#a0a8af" stroke-width="1.1" marker-end="url(#arrow)"/>'
        )
    for node_id, (x, y) in positions.items():
        node = node_lookup[node_id]
        _, fill, stroke = roles[node["role"]]
        parts.append(f'<rect x="{x}" y="{y}" width="205" height="44" rx="7" fill="{fill}" stroke="{stroke}" stroke-width="1.3"/>')
        parts.append(f'<text x="{x + 9}" y="{y + 17}" font-family="sans-serif" font-size="10" font-weight="700" fill="{stroke}">{_escape_svg(node_id)}</text>')
        parts.append(f'<text x="{x + 9}" y="{y + 33}" font-family="sans-serif" font-size="11" fill="#202428">{_escape_svg(node["label"])}</text>')
    parts.extend([
        '<rect x="845" y="670" width="240" height="118" rx="7" fill="#fafafa" stroke="#b7bdc2"/>',
        '<text x="858" y="690" font-family="sans-serif" font-size="11" font-weight="700">Identification policy</text>',
        '<text x="858" y="710" font-family="sans-serif" font-size="10">• Randomize/fix green factors.</text>',
        '<text x="858" y="728" font-family="sans-serif" font-size="10">• Block only on baseline scenario.</text>',
        '<text x="858" y="746" font-family="sans-serif" font-size="10">• Do not adjust for yellow mediators.</text>',
        '<text x="858" y="764" font-family="sans-serif" font-size="10">• Keep wall time descriptive.</text>',
        '<text x="858" y="782" font-family="sans-serif" font-size="10">• Preserve censoring and stop class.</text>',
        '</svg>',
    ])
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def analysis_population_markdown(value: Mapping[str, Any]) -> str:
    lines = [
        "# E02 S01 Analysis-Population Specification",
        "",
        "## Top summary",
        "",
        "- **Research step ID:** S01.",
        "- **Completion status:** Complete and frozen for S02 implementation.",
        "- **Artifacts written:** `causal_dag.svg`, `estimand_registry.yaml`, this analysis-population specification, `intervention_compatibility.csv`, and validation/provenance records.",
        "- **Validation result:** Pass; all primary estimands have an intervention, contrast, outcome, population, identification statement, baseline-only adjustment set, and analysis plan.",
        "- **Outcome classification:** Supportive for the S01 completion criterion.",
        "- **Caveats or blockers:** The E01 global traditional controller is a bundled descriptive comparator; matched/future levels require S02–S05 implementation; natural mediation is not identified.",
        "- **Recommended next action:** Chief Scientist may authorize S02 to implement the common action interface and first satisfy exact no-fault parity E02-S01-E02. Do not begin S02 from this step.",
        "",
        "## Population rules shared by all analyses",
        "",
        "The analysis unit is an immutable scenario-condition run. Pairing is by the frozen scenario/pairing-block identity, not by worker order. The primary principle is intention-to-simulate: every assigned arm is represented, including completion, lawful quiescence, event-budget exhaustion, invariant error, and infrastructure failure. Outcome-dependent removal is prohibited.",
        "",
        "Baseline descriptors (policy, size, input structure, initial disorder, fault map, placement descriptors, and pairing block) may be used for design balance, precision, and prespecified heterogeneity. Observations, activation opportunities, failures, retries, proposals, accepted swaps, trajectory features, costs, and terminal class are post-treatment and may not enter primary total-effect adjustment sets.",
        "",
        "## Named populations",
        "",
    ]
    for population in value["analysisPopulations"]:
        lines.extend([
            f"### {population['id']}: {population['purpose']}",
            "",
            "Inclusion:",
            "",
            *[f"- {item}" for item in population["inclusion"]],
            "",
            "Exclusion/retention boundary:",
            "",
            *[f"- {item}" for item in population["exclusion"]],
            "",
            f"Terminal handling: {population['terminalHandling']}",
            "",
        ])
    lines.extend([
        "## Outcome-specific terminal handling",
        "",
        "| Outcome | Complete | Quiescent | Event budget | Invariant/infrastructure failure |",
        "| --- | --- | --- | --- | --- |",
        "| Success by budget | 1 | 0 unless already complete (terminal precedence prevents ambiguity) | 0 | Missing with named failure; denominator/accounting retained |",
        "| Normalized residual error | Include terminal state | Include terminal state | Include state at budget | Missing with named failure |",
        "| Completion time | Event time | Competing failure terminal | Right-censored at budget | Administrative failure, reported separately |",
        "| Cost ledger | Include full ledger | Include full ledger | Include through budget | Include partial diagnostic ledger but exclude from causal cost mean |",
        "",
        "Quiescence is a lawful semantic result, not censoring. Budget exhaustion is not evidence of equilibrium. Conditioning task comparisons on completion is prohibited because completion is post-treatment.",
        "",
        "## Pairing and missingness",
        "",
        "All arms assigned to a pairing block must be launched. A paired scientific contrast uses complete assigned pairs for the outcome when both outcomes are defined; infrastructure/invariant failures produce an explicit missingness and run-accounting sensitivity rather than conversion to task failure. If random-stream consumption differs, pairing still shares baseline scenario/fault identities but uses condition-addressed streams with verified equal marginals; it is not described as exact common-random-number coupling.",
        "",
        "## Frozen confirmatory scope and support",
        "",
        "S07 must construct sizes 20, 50, 100, 200, and 500 across the planned input structures, and may prune only through its documented compute gate before any confirmatory outcome is opened. S10 may select interactions under frozen adaptive rules. S11 then defines the exact P_CONFIRMATORY_ITS support. S12 may estimate heterogeneity only inside that support; unsupported cells are reported, not extrapolated.",
        "",
        "## Multiplicity and equivalence",
        "",
        "The 12 primary estimands form one family. Simultaneous familywise 95% intervals use a max-t paired bootstrap or Holm-adjusted paired randomization tests. Equivalence of the matched topology effect requires both task intervals within ±0.02 and both completion-time and full-cost ratios within 0.90–1.10. Exact parity E02-S01-E02 is an engineering all-or-none prerequisite, not a statistical equivalence test.",
        "",
        "## Evidence and claim boundary",
        "",
        "This specification targets E01 reference layer R. It does not convert the frozen public comparator C into the publication backend, and it does not reinterpret the global traditional control as a pure architecture intervention. Computational outcomes remain measurements of the transparent simulator, not biological or causal evidence about morphogenesis.",
    ])
    return "\n".join(lines) + "\n"


def compatibility_rows(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for factor in value["factors"]:
        for level in factor["levels"]:
            rows.append({
                "factor": factor["id"],
                "level": level["id"],
                "intervention_contract_defined": True,
                "implementation_status": level["implementationStatus"],
                "implementation_gate": level.get("implementationGate", "none"),
                "current_e01_direct_support": level["implementationStatus"] in {
                    "available_in_E01", "available_for_E01_cell_view", "available_for_E01_traditional_bundle",
                    "implicit_in_E01_reference", "partially_available_in_E01",
                },
            })
    return rows


def write_compatibility(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generate(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    value = registry()
    validation = validate_design(value)
    if not validation["success"]:
        raise RuntimeError("S01 causal design validation failed")
    registry_path = output_dir / "estimand_registry.yaml"
    registry_path.write_text(
        yaml.safe_dump(value, sort_keys=False, allow_unicode=True, width=110),
        encoding="utf-8",
    )
    write_dag_svg(output_dir / "causal_dag.svg", value["dag"])
    (output_dir / "analysis_population.md").write_text(
        analysis_population_markdown(value), encoding="utf-8"
    )
    write_compatibility(output_dir / "intervention_compatibility.csv", compatibility_rows(value))
    write_json(output_dir / "validation_summary.json", validation)
    input_paths = [
        "/workspace/FULL_PLAN.md",
        "/workspace/RESEARCH_PLAN.md",
        "/previous-artifacts/E01/release/baseline/release_manifest.json",
        "/previous-artifacts/E01/research_steps/S14/downstream_readiness_audit.json",
        "/previous-artifacts/E01/research_steps/S01/claim_registry.parquet",
        "/previous-artifacts/E01/research_steps/S03/transition_spec.md",
        "/previous-artifacts/E01/research_steps/S06/event_schema.json",
        "/previous-artifacts/E01/research_steps/S08/scenario_bank_schema.json",
        "/previous-artifacts/E01/research_steps/S10/cost_ledger.parquet",
        "/previous-artifacts/E01/research_steps/S11/validation_summary.json",
        "/workspace/input-attachments/21c2278b-9950-4e39-a2c8-df578a2508ec/pdf-markdown.md",
    ]
    provenance = {
        "schemaVersion": "e02.s01.provenance.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "generatedDate": FROZEN_DATE,
        "generator": str(Path(__file__).relative_to(REPOSITORY)),
        "generatorSha256": sha256_file(Path(__file__)),
        "repository": "https://github.com/Eidosoma/cell_research.git",
        "branch": "eidosoma/groups/28",
        "repositoryCommit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
        ).strip(),
        "e01SemanticsVersion": SEMANTICS_VERSION,
        "inputs": [input_record(path) for path in input_paths],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pyyaml": yaml.__version__,
            "workerCount": 1,
            "gpuUsed": False,
        },
        "networkUsed": False,
        "newDependenciesInstalled": [],
    }
    write_json(output_dir / "provenance_manifest.json", provenance)
    return validation


def finalize_manifest(output_dir: Path) -> dict[str, Any]:
    records = []
    for path in sorted(output_dir.iterdir()):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        records.append({
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    manifest = {
        "schemaVersion": "e02.s01.artifact_manifest.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "success": True,
        "artifactCount": len(records),
        "artifacts": records,
    }
    write_json(output_dir / "artifact_manifest.json", manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--finalize-manifest", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.finalize_manifest:
        manifest = finalize_manifest(args.output_dir)
        print(json.dumps(manifest, sort_keys=True))
        return 0
    value = registry()
    validation = validate_design(value)
    if args.validate_only:
        print(json.dumps(validation, sort_keys=True))
        return 0 if validation["success"] else 1
    validation = generate(args.output_dir)
    print(json.dumps(validation, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
