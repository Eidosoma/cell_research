# E01 deterministic reference simulator

This package implements the clean-room **R** semantics frozen by research step
S03. It is independent of the quarantined public C source. The frozen public
backend remains a scheduler-sensitive comparator and is not imported here.

## API

```python
from reference_simulator import Architecture, Policy, create_scenario, run_scenario

scenario = create_scenario(
    [3, 1, 2],
    policy=Policy.BUBBLE,
    architecture=Architecture.CELL_VIEW,
    generation_key="example-v1",  # stable key used before final scenario-ID derivation
    seed=7,
)
result = run_scenario(scenario, trace_mode="digest")
print(result.summary)
```

`Scenario.to_json_bytes()` and `RunResult.to_json_bytes()` use sorted-key,
whitespace-free UTF-8 JSON. `Scenario.from_json_bytes()` verifies the content
hash embedded in the scenario ID. `exact_replay(result)` reruns the scenario and
requires byte-identical result serialization.

`run_many(scenarios, workers=k)` parallelizes independent replicates only,
preserves input order, and rejects worker counts outside 1–8. Randomness is
addressed by scenario ID, stream name, event index, and draw index, so output is
independent of worker completion order.

`create_scenario(..., fault_count=f, fault_mode="passive")` samples exactly
`f` distinct identities without replacement. Explicit `{index: fault_mode}`
maps are also accepted. The stable generation key is domain-separated for
fault placement and occupancy permutation before the final scenario ID exists.

## Architectures and policies

- `cell_view` implements identity-owned Bubble, Insertion, and Selection policies
  from S03, including passive/stuck faults, exact quiescence, and Selection cursor
  memory.
- `traditional` implements separately named conventional clean-room controls. It
  is not evidence of the missing historical traditional generators; see
  `TRADITIONAL_CONTROLS.md` for the exact primary-action and fault rules.
- Selection deliberately uses `target.value <= actor.value` in both directions;
  direction changes only cursor initialization and cursor advance. This unusual
  behavior is the frozen S03 contract and is not silently corrected.

The event dictionary is explicitly marked `pre-S06`: it retains the minimum S03
semantic fields but is not the shared cross-backend event schema planned for S06.

## CLI

```bash
python -m reference_simulator.cli scenario.json --trace-mode full --output result.json
```

## Validation

```bash
python -m unittest tests.test_reference_simulator -v
python scripts/validate_reference_backend.py --output-dir /artifacts/research_steps/S05
```
