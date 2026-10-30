![Pylint](https://img.shields.io/endpoint?url=https://kit-cel.github.io/handover-optim-milp/pylint-badge.json)
![Test coverage](https://img.shields.io/endpoint?url=https://kit-cel.github.io/handover-optim-milp/coverage-badge.json)

# MILP-Based Optimal Handover Decisions: A Benchmark for Mobility Management Algorithms

This framework determines when a user equipment (UE) should change its serving cell.
Given recorded SINR and RSRP measurements for candidate cells, a mixed-integer linear program (MILP) chooses the serving cell at each time step.
Its objective trades achievable data rate against outage.
The model includes RRC counters and timers, handover preparation and execution, radio link failure, and recovery.

The same measurements can be replayed with an A3-based handover policy or with handover disabled.
A3 uses the step-based RRC implementation from [`mobile-network-sim`](https://github.com/kit-cel/mobile-network-sim).
From stored HDF5 network data, the framework can produce decisions, event logs, metrics, comparison figures, and a complete dataset without generating new radio trajectories.

The [v1.0.0 release](https://github.com/kit-cel/handover-optim-milp/releases/tag/v1.0.0) preserves the code for the [paper](https://doi.org/10.1109/LCOMM.2026.3701321) and [published dataset](https://dx.doi.org/10.21227/9zk6-vb30).
The current code also supports R1-MILP, which starts at the strongest RSRP cell and limits each handover target to the strongest RSRP neighbor.

Example comparison on held-out trajectories from the newer dataset:

![Rate-outage comparison on dataset v3 ID-Test: MILP, R1-MILP, and A3 selection levels](assets/rate_outage_frontier_v3_id_test.png)

## Install

```bash
python -m pip install -e .
```

Installation includes the tagged `mobile-network-sim` GitHub release.

## Use stored network data

```bash
ho-optim optimize --network-data /path/to/id_val.h5 \
  --config config/optimization_config.yaml --episode-index 0 --ue-index 0 \
  --max-steps 20 --output-dir /path/to/optimization

ho-optim run-a3 --network-data /path/to/id_val.h5 \
  --config config/reference_config.yaml --episode-index 0 \
  --max-steps 20 --output-dir /path/to/a3
```

`--max-steps` limits local checks to a trajectory prefix.

## Build and inspect a dataset

```bash
ho-optim build-dataset --network-data id_val=/path/to/id_val.h5 \
  --config config/optimization_config.yaml \
  --a3-config config/reference_config.yaml --selection-split id_val \
  --lambda 0.0 --lambda 1.0 --output-root /path/to/dataset

ho-optim validate-dataset --dataset-root /path/to/dataset
ho-optim plot-dataset --dataset-root /path/to/dataset \
  --split id_val --output-root /path/to/plots
```

Repeat `--network-data` for more splits and `--a3-config` for more A3 profiles.
Use `--a3-search` with one A3 template profile for the coarse-to-fine search.
The plot command reads saved results and writes outside the dataset.

## Citation

```bibtex
@ARTICLE{11554293,
  author={Voigt, Johannes and Rost, Peter M.},
  journal={IEEE Communications Letters},
  title={{MILP}-Based Optimal Handover Decisions: A Benchmark for Mobility Management Algorithms},
  year={2026},
  volume={30},
  pages={2193-2197},
  doi={10.1109/LCOMM.2026.3701321}}
```

MIT licensed; see [LICENSE](LICENSE).
