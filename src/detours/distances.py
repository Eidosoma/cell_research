"""Goal-relative distances for E03 trajectory analysis.

The public functions accept a non-empty finite numeric sequence and either
``"ascending"`` or ``"descending"``.  Equal-valued cells are exchangeable:
global displacement scores minimize over every identity-level ordering that
has the requested non-strict value order.

The module intentionally keeps the paper's strict Sortedness proxy separate
from proper distance-to-goal quantities.  A value-sorted array containing
ties can have paper Sortedness below one.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import math
from typing import Any, Literal, Sequence


Direction = Literal["ascending", "descending"]
Numeric = int | float


def _direction_value(direction: Direction | Any) -> Direction:
    value = getattr(direction, "value", direction)
    if value not in {"ascending", "descending"}:
        raise ValueError("direction must be 'ascending' or 'descending'")
    return value


def _validated_values(values: Sequence[Numeric]) -> tuple[Numeric, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("values must be a numeric sequence, not text")
    try:
        result = tuple(values)
    except TypeError as exc:
        raise TypeError("values must be a finite numeric sequence") from exc
    if not result:
        raise ValueError("distance metrics require at least one value")
    for value in result:
        if isinstance(value, (str, bytes, bool)):
            raise TypeError("values must contain finite real numbers")
        try:
            numeric = float(value)
            hash(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("values must contain finite real numbers") from exc
        if not math.isfinite(numeric):
            raise ValueError("values must not contain NaN or infinity")
    try:
        sorted(result)
    except TypeError as exc:
        raise TypeError("values must be mutually orderable") from exc
    return result


def _strict_violation(left: Numeric, right: Numeric, direction: Direction) -> bool:
    return left > right if direction == "ascending" else left < right


def _strict_order(left: Numeric, right: Numeric, direction: Direction) -> bool:
    return left < right if direction == "ascending" else left > right


def adjacent_descents(
    values: Sequence[Numeric], *, direction: Direction = "ascending"
) -> int:
    """Count strict adjacent violations of the requested value order."""

    sequence = _validated_values(values)
    resolved = _direction_value(direction)
    return sum(
        _strict_violation(left, right, resolved)
        for left, right in zip(sequence, sequence[1:])
    )


def normalized_adjacent_descents(
    values: Sequence[Numeric], *, direction: Direction = "ascending"
) -> float:
    """Adjacent violations divided by ``n - 1`` (zero for a singleton)."""

    sequence = _validated_values(values)
    resolved = _direction_value(direction)
    if len(sequence) == 1:
        return 0.0
    return adjacent_descents(sequence, direction=resolved) / (len(sequence) - 1)


def paper_sortedness_strict(
    values: Sequence[Numeric], *, direction: Direction = "ascending"
) -> float:
    """Implement the paper's printed strict Sortedness fraction.

    The first position contributes one and every strictly ordered adjacent
    pair contributes one.  Consequently, ties do not count as ordered.
    """

    sequence = _validated_values(values)
    resolved = _direction_value(direction)
    strictly_ordered = sum(
        _strict_order(left, right, resolved)
        for left, right in zip(sequence, sequence[1:])
    )
    return (1 + strictly_ordered) / len(sequence)


def paper_sortedness_distance(
    values: Sequence[Numeric], *, direction: Direction = "ascending"
) -> float:
    """Return ``1 - paper_sortedness_strict``.

    This is a trajectory proxy, not a proper distance to the non-strict goal
    set when duplicate values occur.
    """

    return 1.0 - paper_sortedness_strict(values, direction=direction)


def maximum_inversion_count(values: Sequence[Numeric]) -> int:
    """Number of unequal pairs, the multiset-specific inversion maximum."""

    sequence = _validated_values(values)
    counts = Counter(sequence)
    total_pairs = len(sequence) * (len(sequence) - 1) // 2
    tied_pairs = sum(count * (count - 1) // 2 for count in counts.values())
    return total_pairs - tied_pairs


def inversion_count(
    values: Sequence[Numeric], *, direction: Direction = "ascending"
) -> int:
    """Count strict unequal-value inversions in ``O(n log k)`` time."""

    sequence = _validated_values(values)
    resolved = _direction_value(direction)
    goal_order = sorted(set(sequence), reverse=resolved == "descending")
    ranks = {value: index + 1 for index, value in enumerate(goal_order)}
    tree = [0] * (len(ranks) + 1)

    def prefix_count(index: int) -> int:
        count = 0
        while index:
            count += tree[index]
            index -= index & -index
        return count

    def add(index: int) -> None:
        while index < len(tree):
            tree[index] += 1
            index += index & -index

    inversions = 0
    for seen, value in enumerate(sequence):
        rank = ranks[value]
        inversions += seen - prefix_count(rank)
        add(rank)
    return inversions


def normalized_kendall_distance(
    values: Sequence[Numeric], *, direction: Direction = "ascending"
) -> float:
    """Strict inversions divided by the number of unequal pairs.

    All-equal arrays and singletons use the continuous convention ``0 / 0 =
    0`` because they already belong to both non-strict sorting goal sets.
    """

    sequence = _validated_values(values)
    resolved = _direction_value(direction)
    maximum = maximum_inversion_count(sequence)
    if maximum == 0:
        return 0.0
    return inversion_count(sequence, direction=resolved) / maximum


def _goal_displacements(
    values: tuple[Numeric, ...], direction: Direction
) -> tuple[int, ...]:
    current_positions: dict[Numeric, list[int]] = defaultdict(list)
    for position, value in enumerate(values):
        current_positions[value].append(position)

    target_positions: dict[Numeric, list[int]] = defaultdict(list)
    target = sorted(values, reverse=direction == "descending")
    for position, value in enumerate(target):
        target_positions[value].append(position)

    # Sorted-to-sorted matching is optimal for both L1 sum and L-infinity
    # bottleneck costs on a line.  Each target block represents every valid
    # identity ordering among cells with the same value.
    return tuple(
        abs(current - target_position)
        for value, positions in current_positions.items()
        for current, target_position in zip(positions, target_positions[value])
    )


def _opposite_goal(values: tuple[Numeric, ...], direction: Direction) -> tuple[Numeric, ...]:
    return tuple(sorted(values, reverse=direction == "ascending"))


def spearman_footrule(
    values: Sequence[Numeric], *, direction: Direction = "ascending"
) -> int:
    """Minimum total absolute position displacement to the valid goal set."""

    sequence = _validated_values(values)
    resolved = _direction_value(direction)
    return sum(_goal_displacements(sequence, resolved))


def maximum_spearman_footrule(
    values: Sequence[Numeric], *, direction: Direction = "ascending"
) -> int:
    """Multiset-specific maximum footrule, attained by opposite value order."""

    sequence = _validated_values(values)
    resolved = _direction_value(direction)
    return sum(_goal_displacements(_opposite_goal(sequence, resolved), resolved))


def normalized_spearman_footrule(
    values: Sequence[Numeric], *, direction: Direction = "ascending"
) -> float:
    """Footrule divided by its multiset-specific reverse-order maximum."""

    sequence = _validated_values(values)
    maximum = maximum_spearman_footrule(sequence, direction=direction)
    if maximum == 0:
        return 0.0
    return spearman_footrule(sequence, direction=direction) / maximum


def maximum_rank_error(
    values: Sequence[Numeric], *, direction: Direction = "ascending"
) -> int:
    """Minimum possible largest cell displacement over the valid goal set."""

    sequence = _validated_values(values)
    resolved = _direction_value(direction)
    return max(_goal_displacements(sequence, resolved), default=0)


def maximum_possible_rank_error(
    values: Sequence[Numeric], *, direction: Direction = "ascending"
) -> int:
    """Multiset-specific bottleneck maximum, attained by opposite order."""

    sequence = _validated_values(values)
    resolved = _direction_value(direction)
    return max(
        _goal_displacements(_opposite_goal(sequence, resolved), resolved),
        default=0,
    )


def normalized_maximum_rank_error(
    values: Sequence[Numeric], *, direction: Direction = "ascending"
) -> float:
    """Bottleneck displacement divided by its multiset-specific maximum."""

    sequence = _validated_values(values)
    maximum = maximum_possible_rank_error(sequence, direction=direction)
    if maximum == 0:
        return 0.0
    return maximum_rank_error(sequence, direction=direction) / maximum


def duplicate_aware_earth_movers_distance(
    values: Sequence[Numeric], *, direction: Direction = "ascending"
) -> float:
    """Mean one-dimensional transport distance per cell.

    For each value class, unit masses at current positions are transported to
    that value's positions in the sorted target.  Monotone matching is the
    exact one-dimensional optimal transport plan.  With unit cells this is
    ``spearman_footrule / n``; the duplicate-aware name makes the goal-set
    semantics explicit for downstream analyses.
    """

    sequence = _validated_values(values)
    return spearman_footrule(sequence, direction=direction) / len(sequence)


def maximum_duplicate_aware_earth_movers_distance(
    values: Sequence[Numeric], *, direction: Direction = "ascending"
) -> float:
    """Maximum mean transport for the sequence's value multiset."""

    sequence = _validated_values(values)
    return maximum_spearman_footrule(sequence, direction=direction) / len(sequence)


def normalized_duplicate_aware_earth_movers_distance(
    values: Sequence[Numeric], *, direction: Direction = "ascending"
) -> float:
    """Duplicate-aware transport divided by its reverse-order maximum."""

    sequence = _validated_values(values)
    maximum = maximum_duplicate_aware_earth_movers_distance(
        sequence, direction=direction
    )
    if maximum == 0:
        return 0.0
    return duplicate_aware_earth_movers_distance(
        sequence, direction=direction
    ) / maximum


# Concise aliases for callers that already know the duplicate goal semantics.
earth_movers_distance = duplicate_aware_earth_movers_distance
maximum_earth_movers_distance = maximum_duplicate_aware_earth_movers_distance
normalized_earth_movers_distance = normalized_duplicate_aware_earth_movers_distance


@dataclass(frozen=True)
class DistanceProfile:
    """Complete S01 metric record for one array state."""

    n: int
    direction: Direction
    adjacent_descents: int
    normalized_adjacent_descents: float
    paper_sortedness_strict: float
    paper_sortedness_distance: float
    inversion_count: int
    maximum_inversion_count: int
    normalized_kendall_distance: float
    spearman_footrule: int
    maximum_spearman_footrule: int
    normalized_spearman_footrule: float
    maximum_rank_error: int
    maximum_possible_rank_error: int
    normalized_maximum_rank_error: float
    duplicate_aware_earth_movers_distance: float
    maximum_duplicate_aware_earth_movers_distance: float
    normalized_duplicate_aware_earth_movers_distance: float

    def to_dict(self) -> dict[str, int | float | str]:
        """Return a stable JSON-compatible field mapping."""

        return asdict(self)


def distance_profile(
    values: Sequence[Numeric], *, direction: Direction = "ascending"
) -> DistanceProfile:
    """Calculate every S01 local proxy and global distance for one state."""

    sequence = _validated_values(values)
    resolved = _direction_value(direction)
    n = len(sequence)
    adjacent = adjacent_descents(sequence, direction=resolved)
    paper = paper_sortedness_strict(sequence, direction=resolved)
    inversions = inversion_count(sequence, direction=resolved)
    maximum_inversions = maximum_inversion_count(sequence)
    displacements = _goal_displacements(sequence, resolved)
    footrule = sum(displacements)
    rank_error = max(displacements, default=0)
    reverse_displacements = _goal_displacements(
        _opposite_goal(sequence, resolved), resolved
    )
    maximum_footrule = sum(reverse_displacements)
    maximum_rank = max(reverse_displacements, default=0)
    normalized_footrule = footrule / maximum_footrule if maximum_footrule else 0.0

    return DistanceProfile(
        n=n,
        direction=resolved,
        adjacent_descents=adjacent,
        normalized_adjacent_descents=adjacent / (n - 1) if n > 1 else 0.0,
        paper_sortedness_strict=paper,
        paper_sortedness_distance=1.0 - paper,
        inversion_count=inversions,
        maximum_inversion_count=maximum_inversions,
        normalized_kendall_distance=(
            inversions / maximum_inversions if maximum_inversions else 0.0
        ),
        spearman_footrule=footrule,
        maximum_spearman_footrule=maximum_footrule,
        normalized_spearman_footrule=normalized_footrule,
        maximum_rank_error=rank_error,
        maximum_possible_rank_error=maximum_rank,
        normalized_maximum_rank_error=(
            rank_error / maximum_rank if maximum_rank else 0.0
        ),
        duplicate_aware_earth_movers_distance=footrule / n,
        maximum_duplicate_aware_earth_movers_distance=maximum_footrule / n,
        normalized_duplicate_aware_earth_movers_distance=normalized_footrule,
    )
