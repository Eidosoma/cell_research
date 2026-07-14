"""S06 exact-count fault placement and outcome-access contracts.

Placement generation occurs before simulator construction. It receives only a
frozen initial occupancy and immutable values; the four structural generators
have no outcome callback. Outcome access exists only on the separately typed
exploratory search entry point, whose products are rejected by the
confirmatory-selection guard.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Callable, Iterable, Mapping, Sequence

from reference_simulator.model import (
    Architecture,
    Cell,
    Direction,
    FaultMode,
    Policy,
    Scenario,
    canonical_json_bytes,
    sha256_json,
)
from reference_simulator.rng import bounded


PLACEMENT_INTERFACE_VERSION = "E02-fault-placement-v1"
PLACEMENT_RNG_PROFILE = "E02/S06/pre-scenario-counter/v1"
PLACEMENT_ID_PREFIX = "fp1:"
STRUCTURAL_CLASSES = (
    "uniform_exact",
    "clustered_exact",
    "boundary_exact",
    "median_rank_exact",
)


class PlacementClass(str, Enum):
    UNIFORM_EXACT = "uniform_exact"
    CLUSTERED_EXACT = "clustered_exact"
    BOUNDARY_EXACT = "boundary_exact"
    MEDIAN_RANK_EXACT = "median_rank_exact"
    SEARCH_DERIVED_EXPLORATORY = "search_derived_exploratory"


class OutcomeAccess(str, Enum):
    NONE = "none"
    EXPLORATORY_TRAINING_ONLY = "exploratory_training_only"


class AnalysisRole(str, Enum):
    OUTCOME_BLIND_STRUCTURAL_CANDIDATE = "outcome_blind_structural_candidate"
    EXPLORATORY_SEARCH_DERIVED = "exploratory_search_derived"


class ConfirmatoryLeakageError(ValueError):
    """Raised when an outcome-accessed placement reaches a confirmatory surface."""


@dataclass(frozen=True, slots=True)
class PlacementContext:
    """Immutable pre-outcome identity/value state in initial-position order."""

    context_id: str
    identity_ids: tuple[str, ...]
    values: tuple[int | float, ...]

    def __post_init__(self) -> None:
        if not self.context_id:
            raise ValueError("placement context ID must be nonempty")
        if len(self.identity_ids) < 2 or len(self.values) != len(self.identity_ids):
            raise ValueError("placement context requires aligned identity/value arrays")
        if len(set(self.identity_ids)) != len(self.identity_ids):
            raise ValueError("placement context identities must be unique")
        if any(not item for item in self.identity_ids):
            raise ValueError("placement context identities must be nonempty")
        for value in self.values:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("placement context values must be numeric")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("placement context values must be finite")

    @property
    def n(self) -> int:
        return len(self.identity_ids)

    @property
    def value_ranks(self) -> tuple[float, ...]:
        """One-based average ranks, with identity providing stable tie ordering."""

        ordered = sorted(
            range(self.n),
            key=lambda position: (self.values[position], self.identity_ids[position]),
        )
        result = [0.0] * self.n
        cursor = 0
        while cursor < self.n:
            end = cursor + 1
            value = self.values[ordered[cursor]]
            while end < self.n and self.values[ordered[end]] == value:
                end += 1
            average_rank = ((cursor + 1) + end) / 2.0
            for ordinal in range(cursor, end):
                result[ordered[ordinal]] = average_rank
            cursor = end
        return tuple(result)

    @property
    def context_sha256(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "contextId": self.context_id,
            "identityIds": list(self.identity_ids),
            "values": list(self.values),
        }


def _validate_fault_count(context: PlacementContext, fault_count: int) -> None:
    if not isinstance(fault_count, int) or isinstance(fault_count, bool):
        raise TypeError("fault count must be an integer")
    if not 1 <= fault_count < context.n:
        raise ValueError("S06 fault count must satisfy 1 <= f < n")


def _generation_address(
    context: PlacementContext,
    placement_class: PlacementClass,
    fault_count: int,
    replicate_ordinal: int,
    collision_attempt: int,
) -> str:
    return (
        f"E02/S06/{context.context_sha256}/{placement_class.value}/"
        f"f{fault_count}/r{replicate_ordinal}/a{collision_attempt}"
    )


def _permutation(
    positions: Sequence[int], seed: int, address: str, stream: str
) -> tuple[int, ...]:
    result = list(positions)
    for index in range(len(result) - 1, 0, -1):
        selected, _ = bounded(seed, address, stream, index, index + 1)
        result[index], result[selected] = result[selected], result[index]
    return tuple(result)


def boundary_band_positions(
    context: PlacementContext, fault_count: int
) -> tuple[int, ...]:
    _validate_fault_count(context, fault_count)
    width = max(fault_count + 2, math.ceil(context.n / 3))
    ordered = sorted(
        range(context.n),
        key=lambda position: (min(position, context.n - 1 - position), position),
    )
    return tuple(ordered[:width])


def median_rank_band_positions(
    context: PlacementContext, fault_count: int
) -> tuple[int, ...]:
    _validate_fault_count(context, fault_count)
    width = max(fault_count + 2, math.ceil(context.n / 3))
    midpoint = (context.n + 1) / 2.0
    ranks = context.value_ranks
    ordered = sorted(
        range(context.n),
        key=lambda position: (
            abs(ranks[position] - midpoint),
            ranks[position],
            context.identity_ids[position],
        ),
    )
    return tuple(ordered[:width])


@dataclass(frozen=True, slots=True)
class FaultPlacement:
    placement_id: str
    context_id: str
    context_sha256: str
    n: int
    placement_class: PlacementClass
    fault_count: int
    positions: tuple[int, ...]
    identity_ids: tuple[str, ...]
    value_ranks: tuple[float, ...]
    seed: int
    replicate_ordinal: int
    collision_attempt: int
    rng_stream: str
    generation_address: str
    outcome_access: OutcomeAccess
    analysis_role: AnalysisRole
    confirmatory_eligible: bool
    search_tail: str | None = None
    search_score: int | None = None
    local_loss: int | None = None
    weak_loss: int | None = None
    search_round_discovered: int | None = None

    @classmethod
    def create(
        cls,
        context: PlacementContext,
        placement_class: PlacementClass,
        fault_count: int,
        positions: Iterable[int],
        *,
        seed: int,
        replicate_ordinal: int,
        collision_attempt: int,
        rng_stream: str,
        generation_address: str,
        search_tail: str | None = None,
        search_score: int | None = None,
        local_loss: int | None = None,
        weak_loss: int | None = None,
        search_round_discovered: int | None = None,
    ) -> "FaultPlacement":
        _validate_fault_count(context, fault_count)
        frozen_positions = tuple(sorted(positions))
        if len(frozen_positions) != fault_count or len(set(frozen_positions)) != fault_count:
            raise ValueError("placement must contain exactly f unique positions")
        if any(not 0 <= position < context.n for position in frozen_positions):
            raise ValueError("placement position outside context")
        identities = tuple(context.identity_ids[position] for position in frozen_positions)
        ranks = tuple(context.value_ranks[position] for position in frozen_positions)
        is_search = placement_class == PlacementClass.SEARCH_DERIVED_EXPLORATORY
        outcome_access = (
            OutcomeAccess.EXPLORATORY_TRAINING_ONLY if is_search else OutcomeAccess.NONE
        )
        analysis_role = (
            AnalysisRole.EXPLORATORY_SEARCH_DERIVED
            if is_search
            else AnalysisRole.OUTCOME_BLIND_STRUCTURAL_CANDIDATE
        )
        if is_search:
            if search_tail not in {"max_weak_minus_local", "min_weak_minus_local"}:
                raise ValueError("search-derived placement requires a declared search tail")
            if None in {search_score, local_loss, weak_loss, search_round_discovered}:
                raise ValueError("search-derived placement requires complete training provenance")
        elif any(
            item is not None
            for item in (
                search_tail,
                search_score,
                local_loss,
                weak_loss,
                search_round_discovered,
            )
        ):
            raise ValueError("outcome-blind placements cannot carry search outcomes")
        content = {
            "interfaceVersion": PLACEMENT_INTERFACE_VERSION,
            "contextId": context.context_id,
            "contextSha256": context.context_sha256,
            "n": context.n,
            "placementClass": placement_class.value,
            "faultCount": fault_count,
            "positions": list(frozen_positions),
            "identityIds": list(identities),
            "valueRanks": list(ranks),
            "seed": str(seed),
            "replicateOrdinal": replicate_ordinal,
            "collisionAttempt": collision_attempt,
            "rngStream": rng_stream,
            "generationAddress": generation_address,
            "outcomeAccess": outcome_access.value,
            "analysisRole": analysis_role.value,
            "confirmatoryEligible": not is_search,
            "searchTail": search_tail,
            "searchScore": search_score,
            "localLoss": local_loss,
            "weakLoss": weak_loss,
            "searchRoundDiscovered": search_round_discovered,
        }
        return cls(
            placement_id=PLACEMENT_ID_PREFIX + sha256_json(content),
            context_id=context.context_id,
            context_sha256=context.context_sha256,
            n=context.n,
            placement_class=placement_class,
            fault_count=fault_count,
            positions=frozen_positions,
            identity_ids=identities,
            value_ranks=ranks,
            seed=seed,
            replicate_ordinal=replicate_ordinal,
            collision_attempt=collision_attempt,
            rng_stream=rng_stream,
            generation_address=generation_address,
            outcome_access=outcome_access,
            analysis_role=analysis_role,
            confirmatory_eligible=not is_search,
            search_tail=search_tail,
            search_score=search_score,
            local_loss=local_loss,
            weak_loss=weak_loss,
            search_round_discovered=search_round_discovered,
        )

    def content_dict(self) -> dict[str, Any]:
        return {
            "interfaceVersion": PLACEMENT_INTERFACE_VERSION,
            "contextId": self.context_id,
            "contextSha256": self.context_sha256,
            "n": self.n,
            "placementClass": self.placement_class.value,
            "faultCount": self.fault_count,
            "positions": list(self.positions),
            "identityIds": list(self.identity_ids),
            "valueRanks": list(self.value_ranks),
            "seed": str(self.seed),
            "replicateOrdinal": self.replicate_ordinal,
            "collisionAttempt": self.collision_attempt,
            "rngStream": self.rng_stream,
            "generationAddress": self.generation_address,
            "outcomeAccess": self.outcome_access.value,
            "analysisRole": self.analysis_role.value,
            "confirmatoryEligible": self.confirmatory_eligible,
            "searchTail": self.search_tail,
            "searchScore": self.search_score,
            "localLoss": self.local_loss,
            "weakLoss": self.weak_loss,
            "searchRoundDiscovered": self.search_round_discovered,
        }

    def validate_against(self, context: PlacementContext) -> None:
        if self.context_id != context.context_id or self.context_sha256 != context.context_sha256:
            raise ValueError("placement/context provenance mismatch")
        if self.n != context.n:
            raise ValueError("placement/context size mismatch")
        if len(self.positions) != self.fault_count or len(set(self.positions)) != self.fault_count:
            raise ValueError("placement does not preserve exact unique fault count")
        expected_ids = tuple(context.identity_ids[position] for position in self.positions)
        expected_ranks = tuple(context.value_ranks[position] for position in self.positions)
        if self.identity_ids != expected_ids or self.value_ranks != expected_ranks:
            raise ValueError("placement identity or value-rank binding changed")
        if self.placement_id != PLACEMENT_ID_PREFIX + sha256_json(self.content_dict()):
            raise ValueError("placement ID does not match canonical content")

    def descriptor_dict(self) -> dict[str, Any]:
        positions = self.positions
        n_scale = max(self.n - 1, 1)
        edges = tuple(min(position, self.n - 1 - position) for position in positions)
        pair_distances = tuple(
            positions[right] - positions[left]
            for left in range(len(positions))
            for right in range(left + 1, len(positions))
        )
        runs: list[int] = []
        current = 1
        for left, right in zip(positions, positions[1:]):
            if right == left + 1:
                current += 1
            else:
                runs.append(current)
                current = 1
        runs.append(current)
        midpoint = (self.n + 1) / 2.0
        rank_scale = max((self.n - 1) / 2.0, 1.0)
        boundary_width = max(self.fault_count + 2, math.ceil(self.n / 3))
        ordered_edges = sorted(
            range(self.n),
            key=lambda position: (min(position, self.n - 1 - position), position),
        )
        boundary_band = set(ordered_edges[:boundary_width])
        return {
            "placementId": self.placement_id,
            "contextId": self.context_id,
            "placementClass": self.placement_class.value,
            "faultCount": self.fault_count,
            "meanNormalizedPosition": sum(positions) / len(positions) / n_scale,
            "minNormalizedEdgeDistance": min(edges) / n_scale,
            "meanNormalizedEdgeDistance": sum(edges) / len(edges) / n_scale,
            "normalizedPositionSpan": (positions[-1] - positions[0]) / n_scale,
            "meanNormalizedPairwiseDistance": (
                sum(pair_distances) / len(pair_distances) / n_scale
                if pair_distances
                else 0.0
            ),
            "longestContiguousRun": max(runs),
            "contiguousRunCount": len(runs),
            "boundaryBandWidth": boundary_width,
            "boundaryBandOccupancy": sum(
                position in boundary_band for position in positions
            )
            / len(positions),
            "meanNormalizedMedianRankDistance": sum(
                abs(rank - midpoint) for rank in self.value_ranks
            )
            / len(self.value_ranks)
            / rank_scale,
            "minValueRank": min(self.value_ranks),
            "maxValueRank": max(self.value_ranks),
            "outcomeAccess": self.outcome_access.value,
            "analysisRole": self.analysis_role.value,
            "confirmatoryEligible": self.confirmatory_eligible,
        }

    def to_record(self) -> dict[str, Any]:
        duplicate_keys = {
            "placementId",
            "contextId",
            "placementClass",
            "faultCount",
            "outcomeAccess",
            "analysisRole",
            "confirmatoryEligible",
        }
        return {
            "schemaVersion": "e02.s06.fault_placement.v1",
            "placementInterfaceVersion": PLACEMENT_INTERFACE_VERSION,
            "placementId": self.placement_id,
            **self.content_dict(),
            **{
                key: value
                for key, value in self.descriptor_dict().items()
                if key not in duplicate_keys
            },
        }

    def to_json_bytes(self) -> bytes:
        return canonical_json_bytes(
            {"placementId": self.placement_id, **self.content_dict()}
        )


def generate_structural_placement(
    context: PlacementContext,
    placement_class: PlacementClass,
    fault_count: int,
    *,
    seed: int,
    replicate_ordinal: int,
    collision_attempt: int = 0,
) -> FaultPlacement:
    """Generate one outcome-blind exact-count placement."""

    _validate_fault_count(context, fault_count)
    if placement_class.value not in STRUCTURAL_CLASSES:
        raise ValueError("structural generator cannot create search-derived placements")
    if not 0 <= seed < (1 << 128):
        raise ValueError("placement seed must be unsigned uint128")
    if replicate_ordinal < 0 or collision_attempt < 0:
        raise ValueError("generation ordinals must be nonnegative")
    address = _generation_address(
        context, placement_class, fault_count, replicate_ordinal, collision_attempt
    )
    stream = f"placement_{placement_class.value}_s06_v1"
    if placement_class == PlacementClass.UNIFORM_EXACT:
        positions = _permutation(
            tuple(range(context.n)), seed, address, stream
        )[:fault_count]
    elif placement_class == PlacementClass.CLUSTERED_EXACT:
        start, _ = bounded(
            seed, address, stream, 0, context.n - fault_count + 1
        )
        positions = tuple(range(start, start + fault_count))
    elif placement_class == PlacementClass.BOUNDARY_EXACT:
        band = boundary_band_positions(context, fault_count)
        positions = _permutation(band, seed, address, stream)[:fault_count]
    else:
        band = median_rank_band_positions(context, fault_count)
        positions = _permutation(band, seed, address, stream)[:fault_count]
    return FaultPlacement.create(
        context,
        placement_class,
        fault_count,
        positions,
        seed=seed,
        replicate_ordinal=replicate_ordinal,
        collision_attempt=collision_attempt,
        rng_stream=stream,
        generation_address=address,
    )


def generate_structural_bank(
    contexts: Sequence[PlacementContext],
    fault_counts: Sequence[int],
    *,
    seed: int,
    maps_per_class: int,
    max_collision_attempts: int = 4096,
) -> tuple[FaultPlacement, ...]:
    """Generate a cross-class-unique outcome-blind placement bank."""

    if maps_per_class < 1:
        raise ValueError("maps_per_class must be positive")
    placements: list[FaultPlacement] = []
    for context in contexts:
        for fault_count in fault_counts:
            _validate_fault_count(context, fault_count)
            used: set[tuple[int, ...]] = set()
            for placement_class in (
                PlacementClass.UNIFORM_EXACT,
                PlacementClass.CLUSTERED_EXACT,
                PlacementClass.BOUNDARY_EXACT,
                PlacementClass.MEDIAN_RANK_EXACT,
            ):
                for replicate in range(maps_per_class):
                    for attempt in range(max_collision_attempts):
                        placement = generate_structural_placement(
                            context,
                            placement_class,
                            fault_count,
                            seed=seed,
                            replicate_ordinal=replicate,
                            collision_attempt=attempt,
                        )
                        if placement.positions not in used:
                            used.add(placement.positions)
                            placements.append(placement)
                            break
                    else:
                        raise RuntimeError(
                            "unable to construct cross-class-unique structural placement bank"
                        )
    return tuple(placements)


def assert_confirmatory_safe(placements: Sequence[FaultPlacement]) -> None:
    for item in placements:
        if (
            item.outcome_access != OutcomeAccess.NONE
            or item.analysis_role
            != AnalysisRole.OUTCOME_BLIND_STRUCTURAL_CANDIDATE
            or not item.confirmatory_eligible
            or item.placement_class
            == PlacementClass.SEARCH_DERIVED_EXPLORATORY
        ):
            raise ConfirmatoryLeakageError(
                f"placement {item.placement_id} accessed outcomes and is not confirmatory-safe"
            )


def structural_lock_record(
    placements: Sequence[FaultPlacement],
) -> dict[str, Any]:
    if not placements:
        raise ValueError("structural lock cannot be empty")
    assert_confirmatory_safe(placements)
    ordered = sorted(placements, key=lambda item: item.placement_id)
    payload = [
        {
            "placementId": item.placement_id,
            "contextId": item.context_id,
            "faultCount": item.fault_count,
            "placementClass": item.placement_class.value,
            "positions": list(item.positions),
            "contentSha256": sha256_json(item.content_dict()),
        }
        for item in ordered
    ]
    return {
        "schemaVersion": "e02.s06.structural_lock.v1",
        "placementInterfaceVersion": PLACEMENT_INTERFACE_VERSION,
        "placementCount": len(payload),
        "outcomesAccessedBeforeLock": False,
        "lockedPlacementSetSha256": sha256_json(payload),
        "placements": payload,
    }


def select_confirmatory_placements(
    placements: Sequence[FaultPlacement],
) -> tuple[FaultPlacement, ...]:
    """Return only when the entire caller-supplied set is outcome-blind.

    Deliberately raises instead of silently filtering a mixed set, because
    silent filtering can hide an upstream confirmatory-design leak.
    """

    assert_confirmatory_safe(placements)
    return tuple(placements)


def generate_search_initial_positions(
    context: PlacementContext,
    fault_count: int,
    *,
    seed: int,
    count: int,
    forbidden: set[tuple[int, ...]],
    max_collision_attempts: int = 16384,
) -> tuple[tuple[int, ...], ...]:
    """Generate a separate exploratory pool without evaluating structural maps."""

    _validate_fault_count(context, fault_count)
    if count < 1:
        raise ValueError("search initialization count must be positive")
    selected: list[tuple[int, ...]] = []
    used = set(forbidden)
    attempt = 0
    while len(selected) < count and attempt < max_collision_attempts:
        address = (
            f"E02/S06/{context.context_sha256}/search_initial/f{fault_count}/a{attempt}"
        )
        positions = tuple(
            sorted(
                _permutation(
                    tuple(range(context.n)),
                    seed,
                    address,
                    "placement_search_initial_s06_v1",
                )[:fault_count]
            )
        )
        if positions not in used:
            used.add(positions)
            selected.append(positions)
        attempt += 1
    if len(selected) != count:
        raise RuntimeError("unable to generate disjoint search initialization pool")
    return tuple(selected)


@dataclass(frozen=True, slots=True)
class SearchEvaluation:
    positions: tuple[int, ...]
    score: int
    local_loss: int
    weak_loss: int
    round_discovered: int
    source: str

    def to_dict(self, context_id: str, fault_count: int) -> dict[str, Any]:
        return {
            "schemaVersion": "e02.s06.search_evaluation.v1",
            "contextId": context_id,
            "faultCount": fault_count,
            "positions": list(self.positions),
            "scoreWeakMinusLocal": self.score,
            "localLoss": self.local_loss,
            "weakLoss": self.weak_loss,
            "roundDiscovered": self.round_discovered,
            "source": self.source,
            "outcomeAccess": OutcomeAccess.EXPLORATORY_TRAINING_ONLY.value,
            "confirmatoryEligible": False,
        }


SearchScorer = Callable[[tuple[int, ...]], tuple[int, int, int]]
SearchBatchScorer = Callable[
    [tuple[tuple[int, ...], ...]],
    Mapping[tuple[int, ...], tuple[int, int, int]],
]


def search_exploratory_placements(
    context: PlacementContext,
    fault_count: int,
    *,
    seed: int,
    forbidden_structural: set[tuple[int, ...]],
    scorer: SearchScorer,
    batch_scorer: SearchBatchScorer | None = None,
    initial_candidate_count: int = 24,
    beam_width_per_tail: int = 4,
    rounds: int = 4,
    retained_per_tail: int = 3,
) -> tuple[tuple[FaultPlacement, ...], tuple[SearchEvaluation, ...]]:
    """Deterministic two-tailed exploratory beam search.

    The scorer is deliberately available only here. Structural generators do
    not accept a scorer or outcome-bearing object.
    """

    if (
        initial_candidate_count,
        beam_width_per_tail,
        rounds,
        retained_per_tail,
    ) != (24, 4, 4, 3):
        raise ValueError("S06 exploratory search parameters are frozen")
    initial = generate_search_initial_positions(
        context,
        fault_count,
        seed=seed,
        count=initial_candidate_count,
        forbidden=forbidden_structural,
    )
    evaluated: dict[tuple[int, ...], SearchEvaluation] = {}

    def evaluate_many(
        candidates: Iterable[tuple[int, ...]], round_discovered: int, source: str
    ) -> None:
        pending = tuple(
            sorted(
                {
                    tuple(sorted(positions))
                    for positions in candidates
                    if tuple(sorted(positions)) not in evaluated
                }
            )
        )
        if any(positions in forbidden_structural for positions in pending):
            raise ConfirmatoryLeakageError(
                "exploratory search attempted a structural map"
            )
        if not pending:
            return
        results = (
            dict(batch_scorer(pending))
            if batch_scorer is not None
            else {positions: scorer(positions) for positions in pending}
        )
        if set(results) != set(pending):
            raise ValueError("batch scorer must return every requested candidate exactly once")
        for positions in pending:
            score, local_loss, weak_loss = results[positions]
            if score != weak_loss - local_loss:
                raise ValueError("search score must equal weak loss minus local loss")
            evaluated[positions] = SearchEvaluation(
                positions,
                score,
                local_loss,
                weak_loss,
                round_discovered,
                source,
            )

    evaluate_many(initial, 0, "separate_uniform_initialization")

    for round_index in range(1, rounds + 1):
        ordered = list(evaluated.values())
        positive = sorted(
            ordered, key=lambda item: (-item.score, item.positions)
        )[:beam_width_per_tail]
        negative = sorted(
            ordered, key=lambda item: (item.score, item.positions)
        )[:beam_width_per_tail]
        seeds = sorted({item.positions for item in (*positive, *negative)})
        neighbors: set[tuple[int, ...]] = set()
        universe = set(range(context.n))
        for positions in seeds:
            selected = set(positions)
            for removed in positions:
                for added in sorted(universe - selected):
                    candidate = tuple(sorted((selected - {removed}) | {added}))
                    if (
                        candidate not in forbidden_structural
                        and candidate not in evaluated
                    ):
                        neighbors.add(candidate)
        evaluate_many(neighbors, round_index, "one_for_one_beam_neighbor")

    all_items = list(evaluated.values())
    maxima = sorted(
        all_items, key=lambda item: (-item.score, item.positions)
    )[:retained_per_tail]
    used = {item.positions for item in maxima}
    minima = [
        item
        for item in sorted(
            all_items, key=lambda item: (item.score, item.positions)
        )
        if item.positions not in used
    ][:retained_per_tail]
    if len(maxima) != retained_per_tail or len(minima) != retained_per_tail:
        raise RuntimeError("search did not produce enough distinct tail placements")
    selected = [
        *(("max_weak_minus_local", item) for item in maxima),
        *(("min_weak_minus_local", item) for item in minima),
    ]
    placements: list[FaultPlacement] = []
    for replicate, (tail, item) in enumerate(selected):
        address = (
            f"E02/S06/{context.context_sha256}/search_derived/"
            f"f{fault_count}/{tail}/r{replicate}"
        )
        placements.append(
            FaultPlacement.create(
                context,
                PlacementClass.SEARCH_DERIVED_EXPLORATORY,
                fault_count,
                item.positions,
                seed=seed,
                replicate_ordinal=replicate,
                collision_attempt=0,
                rng_stream="placement_search_beam_s06_v1",
                generation_address=address,
                search_tail=tail,
                search_score=item.score,
                local_loss=item.local_loss,
                weak_loss=item.weak_loss,
                search_round_discovered=item.round_discovered,
            )
        )
    return tuple(placements), tuple(
        sorted(evaluated.values(), key=lambda item: item.positions)
    )


def materialize_fault_scenario(
    context: PlacementContext,
    placement: FaultPlacement,
    *,
    fault_mode: FaultMode | str,
    policy: Policy | str,
    direction: Direction | str = Direction.ASCENDING,
    seed: int,
    max_activations: int,
    generation_key: str,
) -> Scenario:
    """Apply an S06 map without changing any S05 execution semantics."""

    placement.validate_against(context)
    mode = FaultMode(fault_mode)
    if mode == FaultMode.NORMAL:
        raise ValueError("S06 placement requires passive or stuck mobility")
    faulty = set(placement.identity_ids)
    cells = tuple(
        Cell(
            cell_id=identity,
            value=value,
            policy=Policy(policy),
            direction=Direction(direction),
            fault=mode if identity in faulty else FaultMode.NORMAL,
        )
        for identity, value in zip(context.identity_ids, context.values)
    )
    scenario = Scenario.create(
        cells,
        initial_occupancy=context.identity_ids,
        seed=seed,
        max_activations=max_activations,
        architecture=Architecture.CELL_VIEW,
        batch_width=1,
        generation_key=f"{generation_key}/{placement.placement_id}",
        fault_placement="explicit",
        requested_fault_count=placement.fault_count,
    )
    if (
        scenario.requested_fault_count != placement.fault_count
        or scenario.realized_fault_count != placement.fault_count
    ):
        raise AssertionError("materialized scenario did not preserve exact fault count")
    return scenario
