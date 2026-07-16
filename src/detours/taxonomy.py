"""Operational category rules for the E03 S14 detour taxonomy.

The rules deliberately classify evidence, not latent mental states.  The order is
the frozen S14 strength order.  A caller may retain all evidence flags while
publishing exactly one strongest supported category.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Mapping


CATEGORY_ORDER = (
    "local_observable_backtracking",
    "global_regression",
    "barrier_correlated_detour",
    "necessary_detour",
    "adaptive_detour",
)

UNSUPPORTED = "not_supported_in_E03"


@dataclass(frozen=True)
class EvidenceFlags:
    """Boolean gates used by the frozen strongest-category rule."""

    replayable_local_worsening: bool = False
    named_global_worsening: bool = False
    explicit_goal_projection: bool = False
    isolated_barrier_contrast: bool = False
    matched_pre_state: bool = False
    valid_stream_scope: bool = False
    exact_goal_reachable: bool = False
    exact_minimum_excursion_positive: bool = False
    observed_successful_recovered_global_excursion: bool = False
    intervention_relative_utility: bool = False
    no_created_impossibility_as_benefit: bool = False
    no_censoring_as_efficiency: bool = False
    matched_null_exceedance: bool = False
    same_support_metric_goal: bool = False
    replay_and_provenance_pass: bool = False


def supported_categories(flags: EvidenceFlags) -> tuple[str, ...]:
    """Return every category whose complete frozen gate is satisfied."""

    categories: list[str] = []
    if flags.replayable_local_worsening:
        categories.append("local_observable_backtracking")
    if flags.named_global_worsening and flags.explicit_goal_projection:
        categories.append("global_regression")
    if (
        flags.isolated_barrier_contrast
        and flags.matched_pre_state
        and flags.valid_stream_scope
    ):
        categories.append("barrier_correlated_detour")
    if flags.exact_goal_reachable and flags.exact_minimum_excursion_positive:
        categories.append("necessary_detour")
    if (
        flags.named_global_worsening
        and flags.explicit_goal_projection
        and flags.observed_successful_recovered_global_excursion
        and flags.intervention_relative_utility
        and flags.no_created_impossibility_as_benefit
        and flags.no_censoring_as_efficiency
        and flags.matched_null_exceedance
        and flags.same_support_metric_goal
        and flags.replay_and_provenance_pass
    ):
        categories.append("adaptive_detour")
    return tuple(categories)


def strongest_supported_category(flags: EvidenceFlags) -> str:
    """Return exactly one strongest category, or the explicit unsupported label."""

    categories = supported_categories(flags)
    if not categories:
        return UNSUPPORTED
    rank = {category: index for index, category in enumerate(CATEGORY_ORDER, start=1)}
    return max(categories, key=rank.__getitem__)


def validate_claim_assignment(
    assignment: Mapping[str, object], flags: EvidenceFlags
) -> list[str]:
    """Return contract violations for one claim-to-category assignment."""

    errors: list[str] = []
    expected = strongest_supported_category(flags)
    observed = assignment.get("strongest_supported_category")
    if observed != expected:
        errors.append(f"strongest category {observed!r} != frozen-rule result {expected!r}")
    supported = list(supported_categories(flags))
    declared = assignment.get("supported_categories")
    if declared is not None and list(declared) != supported:
        errors.append(f"supported categories {declared!r} != {supported!r}")
    return errors


_ANTHROPOMORPHIC_ASSERTION_PATTERNS = {
    "anticipation": re.compile(r"\banticipat(?:e|es|ed|ing|ion)\b", re.I),
    "intention": re.compile(r"\bintent(?:ion|ional|ionally)?\b", re.I),
    "desire": re.compile(r"\bdesir(?:e|es|ed|ing)\b", re.I),
    "subjective_valuation": re.compile(r"\bsubjective(?:ly)?\b|\bvalues? the future\b", re.I),
    "foresight": re.compile(r"\bforesight\b", re.I),
    "cognition": re.compile(r"\bcogniti(?:on|ve)\b", re.I),
    "wanting": re.compile(r"\bwants?\b", re.I),
}


def anthropomorphic_assertion_hits(texts: Iterable[str]) -> list[dict[str, object]]:
    """Find disallowed mental-state assertions in publication claim text.

    This scanner is applied to claim assertions and lay conclusions, not to
    caveat text that explicitly says such inferences are not established.
    """

    hits: list[dict[str, object]] = []
    for index, text in enumerate(texts):
        for label, pattern in _ANTHROPOMORPHIC_ASSERTION_PATTERNS.items():
            match = pattern.search(text)
            if match:
                hits.append(
                    {
                        "text_index": index,
                        "pattern": label,
                        "match": match.group(0),
                    }
                )
    return hits
