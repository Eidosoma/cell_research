"""Counter-addressed random streams frozen by the S03 transition contract."""

from __future__ import annotations

import hashlib
from functools import lru_cache

RNG_DOMAIN = b"E01/RNG/v1\x00"
UINT64_SPACE = 1 << 64


@lru_cache(maxsize=8192)
def _root(seed: int, scenario_id: str) -> bytes:
    if not 0 <= seed < (1 << 128):
        raise ValueError("seed must be an unsigned 128-bit integer")
    return hashlib.sha256(
        RNG_DOMAIN + seed.to_bytes(16, "big") + b"\x00" + scenario_id.encode("utf-8")
    ).digest()


def block(seed: int, scenario_id: str, stream: str, event_index: int, draw_index: int) -> bytes:
    if not 0 <= event_index < (1 << 64):
        raise ValueError("event_index must fit uint64")
    if not 0 <= draw_index < (1 << 32):
        raise ValueError("draw_index must fit uint32")
    return hashlib.sha256(
        _root(seed, scenario_id)
        + b"\x00"
        + stream.encode("ascii")
        + b"\x00"
        + event_index.to_bytes(8, "big")
        + draw_index.to_bytes(4, "big")
    ).digest()


def u64(seed: int, scenario_id: str, stream: str, event_index: int, draw_index: int = 0) -> int:
    return int.from_bytes(block(seed, scenario_id, stream, event_index, draw_index)[:8], "big")


def bounded(
    seed: int,
    scenario_id: str,
    stream: str,
    event_index: int,
    upper: int,
    draw_index: int = 0,
) -> tuple[int, int]:
    """Return an unbiased integer in ``[0, upper)`` and number of blocks consumed."""
    if upper <= 0 or upper > UINT64_SPACE:
        raise ValueError("upper must be in [1, 2**64]")
    limit = UINT64_SPACE - (UINT64_SPACE % upper)
    consumed = 0
    while True:
        value = u64(seed, scenario_id, stream, event_index, draw_index + consumed)
        consumed += 1
        if value < limit:
            return value % upper, consumed


def permutation(items: tuple[str, ...], seed: int, generation_key: str) -> tuple[str, ...]:
    """Fisher-Yates permutation keyed before final scenario-ID derivation.

    This intentionally resolves the S03 self-reference: the scenario permutation
    uses the caller-supplied stable generation key, while runtime streams use the
    final post-permutation scenario ID.
    """
    result = list(items)
    for index in range(len(result) - 1, 0, -1):
        selected, _ = bounded(seed, generation_key, "scenario_permutation", index, index + 1)
        result[index], result[selected] = result[selected], result[index]
    return tuple(result)
