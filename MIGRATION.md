# Migration status

`v1.0.0` preserves the published paper code. The development version targets
the schema-3 `dataset_fr2_mobility_v3` release and uses one MILP formulation.
There is no `paper_v1`/`current` implementation switch in the development code.
The published dataset remains available through the `v1.0.0` tag; the new
dataset builder does not support its release layout.

## Package boundary

- This repository owns `ho_optim_milp` and the `ho-optim` command.
- The separate `mobile-network-sim` repository owns radio generation and the
  simulator RRC implementation. Its bundled optimizer is temporarily named
  `ho_optim_milp_legacy` with the `ho-optim-legacy` command.
- MILP, standalone A3, metrics, reports, and dataset construction from saved
  `network_data` run without the simulator. Simulator-dependent functions
  import it when called.

## Current development state

- The solver uses the newer constraints, raw radio input, configurable initial
  access and handover target selection, and explicit `mip_gap=0.0` by default.
  `unrestricted` and `binary_rank1` are options of this single formulation.
- N310 and N311 default to `1`, as specified in 3GPP TS 38.331 section 9.2.3.
  Explicit values in configuration files override those defaults.
- The default RRC step is `40 ms`. A3 TTT values must align with the configured
  step. The v3 search protocol also lists `240` and `960 ms`, which satisfy
  this alignment but are not shown in the cited TS 38.331 V18.2.0
  TimeToTrigger excerpt. No fixed standards value list is enforced, and the
  recorded v3 search definition is retained for parity.
- The MILP rejects T310 durations that cannot be represented exactly on the
  configured RRC step grid; it does not silently truncate them.
- A recorded-data builder creates schema-3 unrestricted and binary-rank-one
  MILP results, A3 selections, a no-handover baseline, event tables, figures,
  and a release manifest. The builder requires explicit A3 profiles and a
  selection split. It creates a new output directory and has no resume mode.
- The schema-3 reader has been checked against the existing v3 release. Its
  stored HDF5 files retain their existing internal layout; supporting that
  layout is necessary to read v3 and does not imply support for earlier
  dataset releases.

## Verification

- A shortened recorded trajectory builds and validates a complete schema-3
  release, including unrestricted and binary-rank-one MILP results, A3
  selection, events, figures, metadata, and artifact hashes.
- The existing v3 release was inspected read-only. Its manifest lists 340
  artifacts; the reader loads the saved MILP, A3, and event tables.
- Local tests with the Gurobi licence passed: 69 passed, 6 skipped. The
  skipped cases are optional or environment-dependent. No full-size
  optimization campaign was rerun.
- The public CLI provides direct recorded-data optimization, standalone or
  simulator-backed A3, dataset construction and validation, and plotting from
  saved results. Obsolete paper-only entry points were removed from the
  development branch; `v1.0.0` remains unchanged.

The public package no longer has parallel MILP formulations or a required
simulator import. The separate simulator repository still contains
`ho_optim_milp_legacy` for its existing RL and campaign callers. Retiring that
package requires migrating those callers; it is a separate repository change.
