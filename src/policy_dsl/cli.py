"""Command-line validation and canonicalization for E07 policy documents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .core import load_policy


def _record(path: Path, *, include_canonical: bool) -> dict[str, object]:
    policy = load_policy(path)
    result: dict[str, object] = {
        "path": str(path),
        "policyId": policy.policy_id,
        "environment": policy.environment,
        "policySha256": policy.policy_sha256,
        "valid": True,
        "complexity": policy.complexity.to_dict(),
    }
    if include_canonical:
        result["canonicalPolicy"] = json.loads(policy.canonical_json)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("validate", "canonicalize"),
        help="Validate only, or also emit the canonical policy object.",
    )
    parser.add_argument("policies", nargs="+", type=Path)
    args = parser.parse_args(argv)
    records = [
        _record(path, include_canonical=args.command == "canonicalize")
        for path in args.policies
    ]
    print(json.dumps(records, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
