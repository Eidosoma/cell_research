# Conventional clean-room controller controls

Version: `traditional_global_primary_action_v1`

These controllers are new R specifications for paired controls. They are not
paper prose (P), frozen-public-commit behavior (C), or recovered publication
generators. S02 found no traditional generator in reachable history.

All controller scenarios require one homogeneous direction and one matching
policy label. Each controller activation performs at most one primary action:

- Bubble scans adjacent edges from left to right and swaps the first strict
  inversion.
- Insertion scans from left to right and moves the right occupant of the first
  strict adjacent inversion one position left. Repeated activations produce the
  conventional insertion trajectory.
- Selection scans boundaries from left to right. At the first boundary whose
  suffix contains a strictly more extreme value, it swaps that extreme identity
  nonlocally into the boundary. Ties preserve the first encountered identity.

Passive identities never own an activation in this centralized architecture,
but the controller may displace them. A primary swap touching a stuck identity
is rejected. The primary-action rule is canonical: the controller does not skip
a blocked primary action to search for an alternative. Therefore a blocked
primary action is quiescent under this named control profile. These fault rules
are explicit clean-room decisions needed for eventual Figure 5/7 controls; they
must not be described as historical behavior.

Traditional controllers use `batch_width=1`, no stochastic actor stream, the
same atomic swap validator and terminal precedence as cell-view R, and the full
reference operation ledger. Their deterministic trajectories are controlled by
the exact initial occupancy rather than scheduler randomness.
