# E02 common action interface

This package is the S02 interface layer over the deterministic E01 reference
kernel. It implements only the frozen matched profile:

- `distributed_local` and `central_local_proposal_k1`;
- one counter-addressed actor opportunity and one proposal;
- `policy_native_local` observations;
- exact sensing, no action failure, no retry, and skip-and-continue;
- the common E01 validator, atomic commit, and ten-field cost ledger.

Topology is an immutable execution contract kept outside the E01 scenario and
event payload. That separation preserves a common scenario ID, random stream,
state hash, and event schema for the exact E02-S01-E02 parity gate. Condition
metadata is retained by `TopologyRun.contract`.

The policy endpoint never receives `Scenario` or `RunState`. Typed views expose
only self information plus Bubble's selected neighbor, Insertion's strict
prefix, or Selection's cursor target. Other identities and analysis labels are
absent from those views. A trusted envelope carries only the observed swap
target identity needed for stale-target validation.

`central_global_legacy`, additional coordinator rules, alternative schedulers,
retry/failure mechanisms, and broader information permissions are explicitly
outside S02 and remain blocked for later separately authorized steps.

Validation:

```bash
UV_CACHE_DIR=/cache/uv uv run --with pytest==9.0.2 \
  python -m pytest -q tests/test_action_interface_parity.py tests/test_action_interface.py

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  python scripts/validate_action_interface.py --output /artifacts/research_steps/S02
```
