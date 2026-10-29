![Pylint](https://img.shields.io/endpoint?url=https://kit-cel.github.io/handover-optim-milp/pylint-badge.json)
![Test coverage](https://img.shields.io/endpoint?url=https://kit-cel.github.io/handover-optim-milp/coverage-badge.json)

# MILP-based Optimal Handover Decisions: A Benchmark for Mobility Management Algorithms [1]

## Description
This repository accompanies the paper *MILP-Based Optimal Handover Decisions: A Benchmark for Mobility Management Algorithms* [1]. The immutable `v1.0.0` release contains the paper's code. Current development targets schema-3 datasets and provides one MILP formulation for saved radio trajectories, with unrestricted and rank-one handover-target configurations.

The formulation includes handover execution and key radio resource control (RRC) mechanisms: N310/N311 counters, the T310 timer, and radio link failure (RLF) recovery. An A3 event-based reference simulation provides a baseline on the same trajectories. The [published IEEE DataPort dataset](https://dx.doi.org/10.21227/9zk6-vb30) belongs to `v1.0.0`; the current development code reads schema-3 datasets and builds new releases from saved `network_data` without the simulator.

The figure shows the rate-outage trade-off for the published 60 km/h trajectories, including the MILP frontier and A3 reference curves for three quantile selections.

![Rate-outage Pareto frontiers for the MILP benchmark and A3 reference](assets/rate_outage_pareto_frontiers_q-90-95-99.png)

---

## Installation
To install the handover-optim-milp package, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/kit-cel/handover-optim-milp
    ```

2. Navigate to the project directory:
    ```bash
    cd handover-optim-milp
    ```

3. Install the package:
    ```bash
    python -m pip install .
    ```
    i.e., to install it in editable mode/develop mode:
    ```bash
    python -m pip install -e .
    ```

You are now ready to use the handover-optim-milp framework for your projects.

Install the optional simulator integration only when generating new radio
measurements or using its RRC implementation:

```bash
python -m pip install -e ../mobile-network-sim
python -m pip install ".[simulator]"
```

The simulator is maintained in its own repository. Until its new package
layout is published, install it from a local checkout first. The core package
and its dataset builder do not import it.

## Build a dataset from recorded network data

The core package can build a dataset without the simulator. Supply one or more
HDF5 files, an optimization configuration including an RRC profile, and the
lambda values to solve:

```bash
ho-optim build-dataset \
  --network-data id_val=/path/to/network_data/id_val.h5 \
  --config config/optimization_config.yaml \
  --a3-config config/reference_config.yaml \
  --selection-split id_val \
  --lambda 0.0 --lambda 0.1 --lambda 1.0 \
  --output-root /path/to/output_dataset
```

Repeat `--a3-config` for additional A3 parameter profiles. Use
`--a3-backend simulator` only when the simulator is installed. The two A3
backends are labeled separately in the results. The builder creates a new
schema-3 dataset and refuses to overwrite an existing output directory.

Check the dataset structure with
`ho-optim validate-dataset --dataset-root /path/to/output_dataset`. Add
`--verify-artifact-hashes` to check every recorded artifact hash.

The development version uses one MILP formulation with raw signal inputs.
Its `unrestricted` and `binary_rank1` configurations use the same model with
different initial access and handover target options. The `v1.0.0` tag
preserves the paper code and published-data workflow. N310 and N311 default
to `1` as specified in 3GPP TS 38.331 section 9.2.3;
explicit configuration values take precedence. The default RRC step is
`40 ms`; configured A3 TTT values must be multiples of that step.

The coverage badge measures the standalone CI job, including its small MILP
cases when the runner has a Gurobi licence. Simulator integration has a
separate CI job enabled by the repository variable `SIMULATOR_CI_ENABLED` and
access secret `SIMULATOR_REPO_TOKEN`; it is excluded from the badge until
that job is configured. Coverage is informational and has no minimum threshold.
Existing full-size datasets are inspected without rerunning their
optimization campaigns.

---

## Run on stored network data

The commands below take a recorded HDF5 file directly. For a short local
check, `--max-steps` reads only a prefix of the trajectory.

```bash
ho-optim optimize --network-data /path/to/network_data/id_val.h5 \
  --config config/optimization_config.yaml --episode-index 0 --ue-index 0 \
  --max-steps 20 --output-dir /path/to/single_optimization
ho-optim run-a3 --network-data /path/to/network_data/id_val.h5 \
  --config config/reference_config.yaml --episode-index 0 --max-steps 20 \
  --output-dir /path/to/a3_reference
```

To regenerate result figures from a completed dataset, choose a split and a
separate output directory:

```bash
ho-optim plot-dataset --dataset-root /path/to/dataset \
  --split id_val --output-root /path/to/result_plots
```

The command reads saved metrics and event tables; it does not rerun MILP or A3.

Figures and metrics from the published paper are available with the
`v1.0.0` release. The development branch targets the schema-3 dataset.

## Citation [1]
If you use the **handover-optim-milp** framework in your work, please cite our paper:
```
@ARTICLE{11554293,
  author={Voigt, Johannes and Rost, Peter M.},
  journal={IEEE Communications Letters}, 
  title={{MILP}-Based Optimal Handover Decisions: A Benchmark for Mobility Management Algorithms}, 
  year={2026},
  volume={30},
  number={},
  pages={2193-2197},
  doi={10.1109/LCOMM.2026.3701321}}

```

## **License**
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

