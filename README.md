![Pylint](https://img.shields.io/endpoint?url=https://kit-cel.github.io/handover-optim-milp/pylint-badge.json)

# MILP-based Optimal Handover Decisions: A Benchmark for Mobility Management Algorithms [1]

## Description
MILP benchmark for UE RRC handover optimization.
Provides a fully linearized (pure MILP) formulation of handover state machines (N310/N311/T310/RLF), enabling reproducible optimization and benchmarking against simulation traces.

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

---

## Getting Started
### **Get the Dataset**
Download the corresponding dataset at [https://ieee-dataport.org/](https://dx.doi.org/10.21227/9zk6-vb30) and place it in the `handover-optim-milp` directory.

Note regarding the availability of the dataset:
Please note that due to the size of the dataset and the individual results, it is not possible to make the data available in this repository.
Upon acceptance and/or publication of the associated paper, the relevant datasets and detailed optimization results (per-UE results) will be **published on IEEE Dataport** to provide access via a persistent link (DOI).

### Run the MILP Optimization and the RRC Reference Simulation
1. **Run the optimization**:
   ```bash
   python -m ho_optim_milp.run run_optimization --ep-idx=0 --ue-idx=0
   ```
   where `--ep-idx` specifies the episode (0-5) of the dataset and `--ue-idx` defines the UE trajectory that should be used (0-99).
2. **Run the RRC reference simulation**:
   ```bash
   python -m ho_optim_milp.run run_reference --ep-idx=0
   ```
   where `--ep-idx` specifies the episode (0-5) of the dataset. The reference simulation is automatically performed for all UEs in the dataset.

### Results
You can plot the results stored in the dataset and reproduce the figures in [1] using the included plotting functionality.

1. **Plot the rate-outage Pareto fronts of the optimization and the reference.**:
   ```bash
   python -m ho_optim_milp.run plot_pareto_fronts
   ```

2. **Plot the trade-off between the mean achieved rate and the relative connected time versus the Lagrangian multiplier lambda.**:
   ```bash
   python -m ho_optim_milp.run plot_tradeoff
   ```

## Citation [1]
If you use the **handover-optim-milp** framework in your work, please cite our paper:
```
@article{11554293,
  author={Voigt, Johannes and Rost, Peter M.},
  journal={IEEE Communications Letters}, 
  title={{MILP-Based Optimal Handover Decisions: A Benchmark for Mobility Management Algorithms}},
  year={2026},
  volume={30},
  number={},
  pages={2193-2197},
  keywords={Optimization;Radio access networks;Regional area networks;Timing;Cells (biology);Modeling;Handover;Joining processes;3GPP;Interrupters;Handover;mixed-integer linear programming;mobility management;mobile network optimization},
  doi={10.1109/LCOMM.2026.3701321}}
```

## **License**
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

