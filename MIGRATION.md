# Migration status

`v1.0.0` preserves the paper code and published dataset workflow. Current
development uses schema 3 and one MILP formulation with `unrestricted` and
`binary_rank1` options. Earlier dataset formats are not supported here.

## Package boundary

- `handover-optim-milp` owns the MILP, recorded-data pipeline, reports, and
  `ho-optim` CLI.
- `mobile-network-sim` owns radio generation and the RRC used by A3 and
  no-handover replay. The optimizer depends on its tagged GitHub release.
- The private development repository remains unchanged; it is a source for
  selected algorithms, not a runtime dependency.

## Defaults and data

- Raw SINR feeds capacity; the newer constraints have one implementation.
- Defaults: `mip_gap=0.0`, N310=N311=`1`, RRC step `40 ms`.
- A3 TTT and MILP T310 must align with the configured step.
- The builder requires A3 profiles and a selection split. It writes MILP, A3,
  no-handover, event, figure, and manifest artifacts from stored `network_data`.

## Verification

A shortened trajectory built and validated a schema-3 dataset. Stored
`unrestricted` and `binary_rank1` trajectories from the v3 dataset replayed
through the simulator RRC for all 1,500 steps. The full local suite passed
with Gurobi (76 tests).
No full campaign was rerun. The clean-install CI job requires the simulator's
`v0.2.0` tag to be published first.
