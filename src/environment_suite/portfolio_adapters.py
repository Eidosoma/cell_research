"""Frozen S08P multi-policy dispatch semantics for train-only native adapters.

This module owns no native state transition.  It chooses one already-typed DSL
member at a native policy opportunity and records portfolio-only structural and
coordination costs.  Native runners remain responsible for observations,
legality, commit, clocks, failures, censoring, metrics, and stopping.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

from src.policy_dsl import CompiledPolicy

from .contracts import EvaluationAction, SuiteValidationError, canonical_sha256


PORTFOLIO_ADAPTER_VERSION = "e07.s08b.portfolio-adapters.v2"
PORTFOLIO_MODES = {
    "fixed_balanced_identity",
    "random_static_identity",
    "random_dynamic_opportunity",
    "environment_conditioned",
}
ALLOWED_SELECTOR_SIGNALS = {
    "e07_s02_sorting_1d": {"last_action.rejected"},
    "e07_s02_faults_1d": {"last_action.rejected"},
    "e07_s02_detour_1d": {"last_action.rejected"},
    "e07_s02_chimera_1d": {"last_action.rejected"},
    "e07_s02_regeneration_1d": {
        "last_action.rejected",
        "repair.nudge_count",
    },
    "e07_s02_target_change_1d": {"last_action.rejected"},
    "e07_s02_spatial2d_local": {"last_action.rejected"},
    "e07_s02_spatial2d_memory": {"last_action.rejected"},
}


def portfolio_action(
    policy_documents: Sequence[Mapping[str, Any]],
    portfolio_definition: Mapping[str, Any],
) -> EvaluationAction:
    """Build an action whose hash commits to member order and frozen dispatch."""

    from src.policy_dsl import compile_policy

    documents = tuple(policy_documents)
    policies = tuple(compile_policy(item) for item in documents)
    if len(policies) < 2:
        raise SuiteValidationError("portfolio action needs at least two members")
    definition = dict(portfolio_definition)
    digest = canonical_sha256(
        "E07/S08A/portfolio-action/v1",
        {
            "definition": definition,
            "policies": [item.policy_sha256 for item in policies],
        },
    )
    return EvaluationAction(
        "dsl_portfolio",
        digest,
        mode="dsl_episode",
        policy_documents=documents,
        portfolio_definition=definition,
    )


def _counter_index(domain: str, parts: Sequence[str], size: int) -> int:
    if size <= 0:
        raise SuiteValidationError("portfolio carrier has no members")
    payload = "\x1f".join((domain, *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % size


def _selector_costs(mode: str, selector: Mapping[str, Any] | None) -> dict[str, int]:
    if mode != "environment_conditioned":
        return {
            "selectorBranchCount": 0,
            "selectorExpressionNodes": 0,
            "selectorCanonicalBytes": 0,
        }
    canonical = json.dumps(
        dict(selector or {}), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "selectorBranchCount": 2,
        "selectorExpressionNodes": 3,
        "selectorCanonicalBytes": len(canonical),
    }


@dataclass(frozen=True, slots=True)
class PortfolioMember:
    carrier: str
    policy: CompiledPolicy


class PortfolioDispatcher:
    """Deterministic actor/member dispatcher with no native outcome access."""

    def __init__(
        self,
        definition: Mapping[str, Any],
        policies: Mapping[str, CompiledPolicy],
    ) -> None:
        self.definition = dict(definition)
        if self.definition.get("schemaVersion") not in {
            "e07.s08p.portfolio-seed-row.v1",
            "e07.s08a.qualification-portfolio.v1",
        }:
            raise SuiteValidationError("unsupported portfolio definition schema")
        self.configuration_id = str(self.definition.get("configurationId", ""))
        self.task_id = str(self.definition.get("taskId", ""))
        self.mode = str(self.definition.get("mode", ""))
        if len(self.configuration_id) != 64:
            raise SuiteValidationError("portfolio configuration ID must be SHA-256")
        if self.task_id not in ALLOWED_SELECTOR_SIGNALS:
            raise SuiteValidationError("portfolio task is not in frozen S08P registry")
        if self.mode not in PORTFOLIO_MODES:
            raise SuiteValidationError("unsupported bounded portfolio mode")
        if self.definition.get("s07ArmMembershipUsed", False):
            raise SuiteValidationError("S07 arm membership cannot promote a portfolio")
        if self.definition.get("rejectedModelOrEmbeddingUsed", False):
            raise SuiteValidationError("rejected learned artifacts are prohibited")

        members: list[PortfolioMember] = []
        seen: set[str] = set()
        for raw in self.definition.get("members", []):
            policy_id = str(raw.get("policyId", ""))
            carrier = str(raw.get("nativeCarrier", ""))
            if policy_id in seen or policy_id not in policies or not carrier:
                raise SuiteValidationError("portfolio member registry is incomplete")
            policy = policies[policy_id]
            declared_hash = raw.get("policySha256")
            if declared_hash is not None and declared_hash != policy.policy_sha256:
                raise SuiteValidationError("portfolio member hash mismatch")
            seen.add(policy_id)
            members.append(PortfolioMember(carrier, policy))
        if seen != set(policies) or len(members) < 2:
            raise SuiteValidationError(
                "portfolio members must exactly match action policies"
            )
        self.members_by_carrier: dict[str, tuple[CompiledPolicy, ...]] = {}
        for carrier in sorted({item.carrier for item in members}):
            self.members_by_carrier[carrier] = tuple(
                item.policy for item in members if item.carrier == carrier
            )
        # A heterogeneous portfolio can legally contain a native carrier with
        # exactly one compatible frozen member (the S08P Chimera registry has
        # Bubble + Insertion carriers and some 1+N compositions).  Native
        # carrier authority still determines which member set is eligible;
        # cardinality one merely makes dispatch a deterministic no-choice.
        self.singleton_members_by_carrier = {
            carrier: items[0].policy_sha256
            for carrier, items in self.members_by_carrier.items()
            if len(items) == 1
        }
        if int(self.definition.get("portfolioSize", len(members))) != len(members):
            raise SuiteValidationError("portfolio size mismatch")

        selector = self.definition.get("selector")
        self.selector = dict(selector) if isinstance(selector, Mapping) else None
        self.selector_signal: str | None = None
        if self.mode == "environment_conditioned":
            if self.selector is None:
                raise SuiteValidationError("conditioned portfolio requires selector")
            exact = {
                "profile": "bounded_memoryless_two_branch_v1",
                "falseBranch": "current_or_first_compatible_member",
                "trueBranch": "next_compatible_member",
                "persistentMemoryBits": 0,
            }
            if any(self.selector.get(key) != value for key, value in exact.items()):
                raise SuiteValidationError("selector exceeds frozen two-branch profile")
            self.selector_signal = str(self.selector.get("signal", ""))
            if self.selector_signal not in ALLOWED_SELECTOR_SIGNALS[self.task_id]:
                raise SuiteValidationError("selector signal is not authorized for task")
            expected_test = (
                ("gt", 0)
                if self.selector_signal == "repair.nudge_count"
                else ("eq", True)
            )
            if (
                self.selector.get("operator"),
                self.selector.get("threshold"),
            ) != expected_test:
                raise SuiteValidationError(
                    "selector predicate differs from frozen task rule"
                )
        elif self.selector is not None:
            raise SuiteValidationError("non-conditioned mode cannot carry selector")

        self.counter_domain = str(
            self.definition.get(
                "assignmentCounterDomain",
                f"E07/S08/{self.mode}/identity-opportunity/v1",
            )
        )
        if not self.counter_domain.startswith(f"E07/S08/{self.mode}/"):
            raise SuiteValidationError(
                "assignment counter domain is outside frozen mode"
            )
        self.identities_by_carrier: dict[str, tuple[str, ...]] = {}
        self.current_member: dict[str, str] = {}
        self.static_member: dict[str, str] = {}
        self.assignment_counts: Counter[str] = Counter()
        self.member_activation_counts: Counter[str] = Counter()
        self.coordination = {
            "selectorReads": 0,
            "selectorDecisions": 0,
            "selectorCounterReads": 0,
            "memberAssignments": 0,
            "memberSwitches": 0,
            "memberNamespaceInitializations": 0,
        }
        costs = _selector_costs(self.mode, self.selector)
        complexities = [item.policy.complexity for item in members]
        self.structural = {
            "uniqueMemberCount": len(members),
            "totalCanonicalMemberBytes": sum(
                item.canonical_bytes for item in complexities
            ),
            "totalRuleCount": sum(item.rule_count for item in complexities),
            "totalExpressionNodes": sum(item.expression_nodes for item in complexities),
            "totalPersistentMemoryBits": sum(
                item.persistent_memory_bits for item in complexities
            ),
            "totalOutboundSignalBits": sum(
                item.outbound_signal_bits for item in complexities
            ),
            **costs,
        }
        declared = self.definition.get("portfolioStructuralCosts")
        if declared is not None and dict(declared) != self.structural:
            raise SuiteValidationError("portfolio structural cost commitment mismatch")

    def register_identities(self, by_carrier: Mapping[str, Sequence[str]]) -> None:
        canonical = {
            carrier: tuple(sorted(str(actor) for actor in actors))
            for carrier, actors in by_carrier.items()
        }
        if set(canonical) != set(self.members_by_carrier):
            raise SuiteValidationError("native carriers and portfolio carriers differ")
        if self.identities_by_carrier and canonical != self.identities_by_carrier:
            raise SuiteValidationError(
                "portfolio identity population changed mid-episode"
            )
        self.identities_by_carrier = canonical

    def _base_index(self, carrier: str, actor_id: str, size: int) -> int:
        identities = self.identities_by_carrier.get(carrier)
        if not identities or actor_id not in identities:
            raise SuiteValidationError("portfolio actor was not registered")
        if self.mode == "fixed_balanced_identity":
            return identities.index(actor_id) % size
        if self.mode == "random_static_identity":
            return _counter_index(
                self.counter_domain,
                (self.configuration_id, carrier, actor_id),
                size,
            )
        return 0

    def select(
        self,
        carrier: str,
        actor_id: str,
        opportunity_index: int,
        *,
        selector_value: bool | int = False,
        mutate: bool,
    ) -> CompiledPolicy:
        try:
            members = self.members_by_carrier[carrier]
        except KeyError as exc:
            raise SuiteValidationError("portfolio lacks the native carrier") from exc
        identities = self.identities_by_carrier.get(carrier)
        if not identities or actor_id not in identities:
            raise SuiteValidationError("portfolio actor was not registered")
        current = self.current_member.get(actor_id)
        if len(members) == 1:
            # Do not consume assignment randomness for a singleton carrier.
            # Conditioned selectors remain observed/accounted below so the
            # frozen selector contract and coordination costs are preserved.
            chosen = members[0]
        elif self.mode in {"fixed_balanced_identity", "random_static_identity"}:
            index = self._base_index(carrier, actor_id, len(members))
            chosen = members[index]
        elif self.mode == "random_dynamic_opportunity":
            index = _counter_index(
                self.counter_domain,
                (
                    self.configuration_id,
                    carrier,
                    actor_id,
                    str(int(opportunity_index)),
                ),
                len(members),
            )
            chosen = members[index]
        else:
            if current is None:
                index = 0
            else:
                index = next(
                    i
                    for i, member in enumerate(members)
                    if member.policy_sha256 == current
                )
            selector_true = (
                int(selector_value) > 0
                if self.selector_signal == "repair.nudge_count"
                else selector_value is True
            )
            if selector_true:
                index = (index + 1) % len(members)
            chosen = members[index]
        if mutate:
            self.coordination["memberAssignments"] += 1
            if (
                self.mode == "random_static_identity"
                and len(members) > 1
                and actor_id not in self.static_member
            ):
                self.coordination["selectorCounterReads"] += 1
                self.static_member[actor_id] = chosen.policy_sha256
            elif self.mode == "random_dynamic_opportunity" and len(members) > 1:
                self.coordination["selectorCounterReads"] += 1
            if self.mode == "environment_conditioned":
                self.coordination["selectorReads"] += 1
                self.coordination["selectorDecisions"] += 1
            if current is not None and current != chosen.policy_sha256:
                self.coordination["memberSwitches"] += 1
            self.current_member[actor_id] = chosen.policy_sha256
            self.assignment_counts[chosen.policy_sha256] += 1
            self.member_activation_counts[f"{actor_id}::{chosen.policy_sha256}"] += 1
        return chosen

    def note_namespace_initialization(self) -> None:
        self.coordination["memberNamespaceInitializations"] += 1

    def ledger_snapshot(self) -> dict[str, Any]:
        return {
            "adapterVersion": PORTFOLIO_ADAPTER_VERSION,
            "configurationId": self.configuration_id,
            "mode": self.mode,
            "selectorSignal": self.selector_signal,
            "singletonCarrierPolicySha256": dict(
                sorted(self.singleton_members_by_carrier.items())
            ),
            "portfolioStructuralLedger": dict(self.structural),
            "portfolioCoordinationLedger": dict(self.coordination),
            "assignmentCountsByPolicySha256": dict(
                sorted(self.assignment_counts.items())
            ),
            "memberActivationCounts": dict(
                sorted(self.member_activation_counts.items())
            ),
            "currentMemberByIdentity": dict(sorted(self.current_member.items())),
        }

    def definition_sha256(self) -> str:
        return canonical_sha256("E07/S08A/portfolio-definition/v1", self.definition)
