# Migration status

`v1.0.0` is the immutable release for the published paper. The current
development version is `1.1.0.dev0`; it keeps the paper workflow available
while recorded network data can also be processed by the newer formulation.

## Current package boundary

- This repository owns `ho_optim_milp` and the `ho-optim` command.
- `mobile-network-sim` owns radio generation and the simulator RRC state
  machine. The former combined implementation is temporarily named
  `ho_optim_milp_legacy` with the `ho-optim-legacy` command in that repository.
- The core package accepts recorded HDF5 trajectories without importing the
  simulator. Simulator-backed A3 replay and radio generation import it only
  when called.

## Temporary compatibility

The `paper_v1` MILP profile remains the 1.x default. The `current` profile
selects raw signal inputs, the newer constraints, and optional initial access,
target selection, and objective discounting. Both old and new network-data
HDF5 metadata are accepted. Output rows record the chosen model profile and
A3 backend.

The two MILP implementations are a migration aid. Before `v2.0.0`, run the
full recorded-data pipeline on the new dataset, compare the scientific metrics
and figures, and retain a reproducible manifest of that run. Then remove the
paper constraint builder, the profile switch, and the old HDF5 reader. The
standalone recorded-data pipeline remains. Reproduction of the paper stays
available from the `v1.0.0` tag.

## Verification

The standard CI job runs without the simulator and publishes its test coverage.
The simulator CI job requires access to the separate repository. Small MILP
models run when Gurobi can create a licensed model; a full comparison against
the published dataset requires that dataset and a suitable license separately.
`ho-optim validate_dataset` verifies output hashes and the completion marker.
