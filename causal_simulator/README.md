# E02 common action interface

This package is the S02 action layer and S03 architecture layer over the
deterministic E01 reference kernel. The S02 frozen matched profile remains:

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

S03 adds four explicitly named architecture contracts:

- `central_global_legacy`, the existing E01 global traditional controller,
  registered as an unmatched descriptive anchor only;
- `central_local_proposal_k1`, the S02 one-envelope relay;
- `distributed_local`, the S02 direct route; and
- `distributed_weak_coordinator`, with either no coordinator or the frozen
  `weak_nonlocal_veto_p8_v1` budget.

The weak coordinator receives only a typed event index plus a one-bit
`is_nonlocal_swap` signal every eighth opportunity and returns one allow/veto
bit. A veto selects the already-legal `NoOp`; it cannot add candidates, retry,
reschedule, inspect raw state, or change the policy-native information surface.
Message, bit, eligible-decision, and intervention costs are retained in a
supplementary architecture ledger pending the separately planned S09 outcome
schema. Central candidate pools with k greater than one remain structurally
unsupported until S04 freezes compatible opportunity semantics.

S04 adds deterministic scans, uniform random activations, random-permutation
sweeps, synchronous proposals with deterministic conflicts, and a bounded fair
adversary over the same one-proposal boundary. S05 then composes scenario-owned
normal/passive/stuck mobility, continuation rules, charged actor-selected
bounded retries, counter-addressed Bernoulli/transient action failures, and
policy-visible value/target-status sensing noise. The trusted validator keeps
ground truth, and the full-global legacy controller remains non-comparable.
See `faults.py` and `scripts/validate_fault_semantics.py`.

Validation:

```bash
UV_CACHE_DIR=/cache/uv uv run --with pytest==9.0.2 \
  python -m pytest -q tests/test_action_interface_parity.py tests/test_action_interface.py

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  python scripts/validate_action_interface.py --output /artifacts/research_steps/S02

UV_CACHE_DIR=/cache/uv uv run --with pytest==9.0.2 \
  python -m pytest -q tests/test_architectures.py tests/test_action_interface.py \
  tests/test_action_interface_parity.py

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  python scripts/validate_architectures.py --output /artifacts/research_steps/S03

UV_CACHE_DIR=/cache/uv uv run --with pytest==9.0.2 \
  python -m pytest -q tests/test_faults.py

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  python scripts/validate_fault_semantics.py \
  --output /artifacts/research_steps/S05/fault_package
```
