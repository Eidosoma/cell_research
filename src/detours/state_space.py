"""Canonical finite structural state space for E03 S04.

The E01 runtime state contains execution history (activation and ledger counters)
as well as the finite policy-visible state.  Exact legal-action graphs need the
latter: immutable family semantics, identity occupancy, and Selection cursors.
This module defines that explicit projection and a lossless rank-based encoder.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import math
import struct
from typing import Iterable, Iterator, Mapping, Sequence

from reference_simulator.model import (
    Architecture,
    Cell,
    Direction,
    FaultMode,
    Policy,
    RunState,
    Scenario,
    canonical_json_bytes,
)


FAMILY_SCHEMA = "e03.s04.state_family.v1"
STATE_SCHEMA = "e03.s04.structural_state.v1"
SCHEDULER_PROJECTION = "fair_serial_legal_opportunity_set_v1"
FAMILY_DOMAIN = b"E03/S04/family/v1\x00"
STATE_DOMAIN = b"E03/S04/state/v1\x00"
POLICY_ORDER = (Policy.BUBBLE, Policy.INSERTION, Policy.SELECTION)
FAULT_ORDER = (FaultMode.PASSIVE, FaultMode.STUCK)


def cell_id(index: int) -> str:
    if index < 0:
        raise ValueError("cell index must be nonnegative")
    return f"c{index}"


def cell_index(value: str) -> int:
    if not value.startswith("c") or not value[1:].isdigit():
        raise ValueError(f"invalid S04 cell ID {value!r}")
    return int(value[1:])


def rank_permutation(permutation: Sequence[int]) -> int:
    """Return the zero-based lexicographic Lehmer rank."""

    n = len(permutation)
    if sorted(permutation) != list(range(n)):
        raise ValueError("occupancy must be a permutation of 0..n-1")
    available = list(range(n))
    rank = 0
    for position, value in enumerate(permutation):
        index = available.index(value)
        rank += index * math.factorial(n - position - 1)
        available.pop(index)
    return rank


def unrank_permutation(n: int, rank: int) -> tuple[int, ...]:
    """Invert :func:`rank_permutation` for a fixed size."""

    if n < 1:
        raise ValueError("n must be positive")
    count = math.factorial(n)
    if not 0 <= rank < count:
        raise ValueError(f"permutation rank must be in [0, {count})")
    available = list(range(n))
    result: list[int] = []
    remainder = rank
    for width in range(n, 0, -1):
        factor = math.factorial(width - 1)
        index, remainder = divmod(remainder, factor)
        result.append(available.pop(index))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class FamilySpec:
    """Immutable graph-family semantics shared by structural nodes."""

    n: int
    architecture: Architecture
    direction: Direction
    policies: tuple[Policy, ...]
    faults: tuple[FaultMode, ...]
    scheduler_projection: str = SCHEDULER_PROJECTION

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError("n must be positive")
        if len(self.policies) != self.n or len(self.faults) != self.n:
            raise ValueError("policy and fault vectors must each have length n")
        if self.architecture == Architecture.TRADITIONAL and len(set(self.policies)) != 1:
            raise ValueError("traditional families require one controller policy")
        if self.scheduler_projection != SCHEDULER_PROJECTION:
            raise ValueError("unsupported structural scheduler projection")

    @property
    def selection_owners(self) -> tuple[int, ...]:
        if self.architecture == Architecture.TRADITIONAL:
            return ()
        return tuple(index for index, policy in enumerate(self.policies) if policy == Policy.SELECTION)

    @property
    def selection_owner_count(self) -> int:
        return len(self.selection_owners)

    @property
    def cursor_state_count(self) -> int:
        return (self.n + 1) ** self.selection_owner_count

    @property
    def occupancy_state_count(self) -> int:
        return math.factorial(self.n)

    @property
    def state_count(self) -> int:
        return self.occupancy_state_count * self.cursor_state_count

    @property
    def fault_count(self) -> int:
        return sum(value != FaultMode.NORMAL for value in self.faults)

    @property
    def fault_mode(self) -> str:
        modes = {value.value for value in self.faults if value != FaultMode.NORMAL}
        if not modes:
            return "none"
        if len(modes) != 1:
            raise ValueError("S04 families allow only one non-normal fault mode")
        return next(iter(modes))

    @property
    def policy_code(self) -> int:
        code = 0
        for policy in self.policies:
            code = 3 * code + POLICY_ORDER.index(policy)
        return code

    @property
    def fault_code(self) -> int:
        code = 0
        digits = (FaultMode.NORMAL, FaultMode.PASSIVE, FaultMode.STUCK)
        for fault in self.faults:
            code = 3 * code + digits.index(fault)
        return code

    @property
    def policy_profile(self) -> str:
        unique = set(self.policies)
        if len(unique) == 1:
            return f"pure_{next(iter(unique)).value}"
        counts = {policy.value: self.policies.count(policy) for policy in POLICY_ORDER}
        return "+".join(f"{name}:{counts[name]}" for name in sorted(counts) if counts[name])

    def canonical_dict(self) -> dict[str, object]:
        return {
            "schemaVersion": FAMILY_SCHEMA,
            "semanticsVersion": "E01-reference-v1",
            "architecture": self.architecture.value,
            "n": self.n,
            "cellIds": [cell_id(index) for index in range(self.n)],
            "valuesByIdentity": list(range(self.n)),
            "policiesByIdentity": [value.value for value in self.policies],
            "directionsByIdentity": [self.direction.value] * self.n,
            "faultsByIdentity": [value.value for value in self.faults],
            "sourceScheduler": (
                "traditional_controller"
                if self.architecture == Architecture.TRADITIONAL
                else "serial_counter_addressed"
            ),
            "schedulerProjection": self.scheduler_projection,
            "initialOccupancyConvention": "identity_order_only_for_scenario_anchor",
            "selectionCursorOwnership": "identity_owned_positional_integer",
        }

    @classmethod
    def from_canonical_dict(cls, value: Mapping[str, object]) -> "FamilySpec":
        """Reconstruct a family while rejecting incompatible canonical metadata."""

        if value.get("schemaVersion") != FAMILY_SCHEMA:
            raise ValueError("unsupported S04 family schema")
        n = int(value["n"])
        directions = tuple(Direction(item) for item in value["directionsByIdentity"])
        if len(directions) != n or len(set(directions)) != 1:
            raise ValueError("S04 families require one homogeneous direction")
        expected_ids = [cell_id(index) for index in range(n)]
        if value.get("cellIds") != expected_ids:
            raise ValueError("noncanonical S04 cell IDs")
        if value.get("valuesByIdentity") != list(range(n)):
            raise ValueError("noncanonical S04 identity/value profile")
        family = cls(
            n=n,
            architecture=Architecture(value["architecture"]),
            direction=directions[0],
            policies=tuple(Policy(item) for item in value["policiesByIdentity"]),
            faults=tuple(FaultMode(item) for item in value["faultsByIdentity"]),
            scheduler_projection=str(value["schedulerProjection"]),
        )
        if family.canonical_dict() != dict(value):
            raise ValueError("family metadata is not canonical")
        return family

    @property
    def canonical_bytes(self) -> bytes:
        return FAMILY_DOMAIN + canonical_json_bytes(self.canonical_dict())

    @property
    def family_digest(self) -> bytes:
        return hashlib.sha256(self.canonical_bytes).digest()

    @property
    def family_id(self) -> str:
        return "s04f:" + self.family_digest.hex()

    def dual(self) -> "FamilySpec":
        """Return the ascending/descending value-complement symmetry family."""

        return FamilySpec(
            n=self.n,
            architecture=self.architecture,
            direction=(
                Direction.DESCENDING
                if self.direction == Direction.ASCENDING
                else Direction.ASCENDING
            ),
            policies=tuple(reversed(self.policies)),
            faults=tuple(reversed(self.faults)),
            scheduler_projection=self.scheduler_projection,
        )

    def scenario(self) -> Scenario:
        """Build the E01 anchor scenario used for semantic cross-checks."""

        cells = [
            Cell(
                cell_id=cell_id(index),
                value=index,
                policy=self.policies[index],
                direction=self.direction,
                fault=self.faults[index],
            )
            for index in range(self.n)
        ]
        return Scenario.create(
            cells,
            initial_occupancy=[cell_id(index) for index in range(self.n)],
            seed=0,
            max_activations=1_000_000,
            architecture=self.architecture,
            batch_width=1,
            traditional_policy=(
                self.policies[0]
                if self.architecture == Architecture.TRADITIONAL
                else None
            ),
            generation_key=f"E03-S04-{self.family_digest.hex()}",
            fault_placement="explicit",
            requested_fault_count=self.fault_count,
        )


@dataclass(frozen=True, slots=True)
class StructuralState:
    """Lossless rank/cursor coordinates within one graph family."""

    family: FamilySpec
    occupancy_rank: int
    selection_cursor_code: int = 0

    def __post_init__(self) -> None:
        if not 0 <= self.occupancy_rank < self.family.occupancy_state_count:
            raise ValueError("occupancy rank is outside the family domain")
        if not 0 <= self.selection_cursor_code < self.family.cursor_state_count:
            raise ValueError("Selection cursor code is outside the family domain")

    @property
    def occupancy(self) -> tuple[int, ...]:
        return unrank_permutation(self.family.n, self.occupancy_rank)

    @property
    def cursor_digits(self) -> tuple[int, ...]:
        base = self.family.n + 1
        code = self.selection_cursor_code
        digits: list[int] = []
        for _ in self.family.selection_owners:
            code, digit = divmod(code, base)
            digits.append(digit)
        if code:
            raise AssertionError("cursor code exceeds declared owner radix")
        return tuple(digits)

    @property
    def selection_cursors(self) -> dict[int, int]:
        cursors: dict[int, int] = {}
        for owner, digit in zip(self.family.selection_owners, self.cursor_digits):
            cursors[owner] = (
                digit
                if self.family.direction == Direction.ASCENDING
                else self.family.n - 1 - digit
            )
        return cursors

    @property
    def canonical_bytes(self) -> bytes:
        return (
            STATE_DOMAIN
            + self.family.family_digest
            + struct.pack(">QQ", self.occupancy_rank, self.selection_cursor_code)
        )

    @property
    def state_digest(self) -> bytes:
        return hashlib.sha256(self.canonical_bytes).digest()

    @property
    def state_id(self) -> str:
        return "s04s:" + self.state_digest.hex()

    def to_run_state(self) -> RunState:
        return RunState(
            occupancy=[cell_id(index) for index in self.occupancy],
            selection_cursors={
                cell_id(owner): cursor for owner, cursor in self.selection_cursors.items()
            },
        )

    def dual(self) -> "StructuralState":
        dual_family = self.family.dual()
        n = self.family.n
        dual_occupancy = tuple(n - 1 - value for value in self.occupancy)
        original_cursors = self.selection_cursors
        dual_actual = {
            n - 1 - owner: n - 1 - cursor
            for owner, cursor in original_cursors.items()
        }
        return StructuralState.from_components(
            dual_family,
            dual_occupancy,
            dual_actual,
        )

    @classmethod
    def from_components(
        cls,
        family: FamilySpec,
        occupancy: Sequence[int],
        selection_cursors: Mapping[int, int] | None = None,
    ) -> "StructuralState":
        expected = set(family.selection_owners)
        provided = set((selection_cursors or {}).keys())
        if provided != expected:
            raise ValueError("Selection cursor owners do not match the family")
        base = family.n + 1
        code = 0
        multiplier = 1
        for owner in family.selection_owners:
            actual = (selection_cursors or {})[owner]
            digit = actual if family.direction == Direction.ASCENDING else family.n - 1 - actual
            if not 0 <= digit < base:
                raise ValueError("Selection cursor is outside its direction-specific domain")
            code += digit * multiplier
            multiplier *= base
        return cls(
            family=family,
            occupancy_rank=rank_permutation(occupancy),
            selection_cursor_code=code,
        )

    @classmethod
    def from_run_state(cls, family: FamilySpec, state: RunState) -> "StructuralState":
        occupancy = tuple(cell_index(value) for value in state.occupancy)
        cursors = {cell_index(key): value for key, value in state.selection_cursors.items()}
        return cls.from_components(family, occupancy, cursors)


def fault_maps(n: int, *, maximum_count: int = 3) -> Iterator[tuple[FaultMode, ...]]:
    """Yield no-fault then every passive/stuck identity placement."""

    if n < 1 or maximum_count < 0:
        raise ValueError("invalid fault-map domain")
    yield (FaultMode.NORMAL,) * n
    for mode in FAULT_ORDER:
        for count in range(1, min(maximum_count, n) + 1):
            for indices in itertools.combinations(range(n), count):
                values = [FaultMode.NORMAL] * n
                for index in indices:
                    values[index] = mode
                yield tuple(values)


def nonzero_fault_maps(n: int, *, maximum_count: int = 3) -> Iterator[tuple[FaultMode, ...]]:
    iterator = fault_maps(n, maximum_count=maximum_count)
    next(iterator)
    yield from iterator


def all_policy_assignments(n: int) -> Iterator[tuple[Policy, ...]]:
    if n < 1:
        raise ValueError("n must be positive")
    yield from itertools.product(POLICY_ORDER, repeat=n)


def pure_policy(policy: Policy, n: int) -> tuple[Policy, ...]:
    if n < 1:
        raise ValueError("n must be positive")
    return (policy,) * n


def structural_state_count(n: int, selection_owner_count: int) -> int:
    if not 0 <= selection_owner_count <= n:
        raise ValueError("invalid Selection-owner count")
    return math.factorial(n) * (n + 1) ** selection_owner_count


def all_policy_state_count(n: int) -> int:
    """Sum structural states over all 3^n cell-view policy assignments."""

    if n < 1:
        raise ValueError("n must be positive")
    # Each identity contributes two memoryless labels or one Selection label
    # with n+1 cursor values: 2 + (n+1) = n+3.
    return math.factorial(n) * (n + 3) ** n


def family_lookup_key(family: FamilySpec) -> tuple[object, ...]:
    return (
        family.n,
        family.architecture.value,
        family.direction.value,
        tuple(value.value for value in family.policies),
        tuple(value.value for value in family.faults),
        family.scheduler_projection,
    )
