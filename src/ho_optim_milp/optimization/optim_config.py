"""Config for MILP optimization configuration class."""

import os
from typing import Literal

from pydantic import BaseModel, ConfigDict, StrictBool, StrictFloat, Field, model_validator

from ..common.base_config import BaseConfig
from ..common.type_aliases import StrictPositiveInt
from ..reference.rrc_config import RRCConfig


class GurobiSolverConfig(BaseModel):
    """Portable solver settings with no machine-specific defaults."""

    threads: int | None = Field(default=None, ge=1)
    time_limit_s: float | None = Field(default=None, gt=0)
    mip_gap: float | None = Field(default=None, ge=0)
    log_to_console: bool = True

    model_config = ConfigDict(extra="forbid")


class WarmStartSolverConfig(BaseModel):
    """Optional candidate-based MIP start for the final unpruned model."""

    enabled: bool = False
    candidate_k: int = Field(default=0, ge=0)
    candidate_delta_db: float = Field(default=0.0, ge=0)
    candidate_metric: Literal["sinr", "rsrp"] = "sinr"
    candidate_include_prev: bool = True
    safe_pruning_time_limit_s: float | None = Field(default=None, gt=0)
    safe_pruning_mip_gap: float | None = Field(default=0.0, ge=0)

    model_config = ConfigDict(extra="forbid")


class OptimConfig(BaseConfig):
    """MILP optimization configuration class."""

    seed: StrictPositiveInt | None = Field(
        default=None, description="Random seed for reproducibility."
    )

    load_dataset: StrictBool = Field(
        default=False,
        description="Whether to load an existing dataset or create a new one.",
    )
    dataset_path: str = Field(
        default="sim_dataset.h5", description="Path to the dataset file."
    )
    save_dir: str = Field(
        default="gurobi_optim_results",
        description="Directory to save optimization results.",
    )

    show_plots: StrictBool = Field(default=False, description="Show plots.")
    save_plots: StrictBool = Field(default=False, description="Save plots.")

    use_l3_filtered_sinr: StrictBool = Field(
        default=True,
        description="Whether to use L3 filtered SINR values for optimization.",
    )

    lambda_r: StrictFloat = Field(
        default=1.0,
        description="Discount given in bit/s/Hz for each outage step that is avoided.",
    )

    model_profile: Literal["paper_v1", "current"] = Field(
        default="paper_v1",
        description="Paper formulation and L3 inputs, or the current raw-SINR formulation.",
    )
    discount_gamma: StrictFloat = Field(default=1.0, gt=0, le=1)
    initial_access: Literal["free", "strongest_rsrp"] = "free"
    handover_target_mode: Literal["free", "rank1_neighbor"] = "free"
    gurobi: GurobiSolverConfig = Field(default_factory=GurobiSolverConfig)
    warm_start: WarmStartSolverConfig = Field(default_factory=WarmStartSolverConfig)

    rrc: "RRCConfig"

    @model_validator(mode="after")
    def _validate_model_profile(self) -> "OptimConfig":
        expected_l3 = self.model_profile == "paper_v1"
        if ("use_l3_filtered_sinr" in self.model_fields_set
                and self.use_l3_filtered_sinr != expected_l3):
            raise ValueError(
                "use_l3_filtered_sinr conflicts with model_profile; "
                "paper_v1 uses L3 inputs and current uses raw inputs."
            )
        self.use_l3_filtered_sinr = expected_l3
        if self.model_profile == "paper_v1" and (
            self.discount_gamma != 1.0
            or self.initial_access != "free"
            or self.handover_target_mode != "free"
        ):
            raise ValueError(
                "New model options require model_profile='current'."
            )
        if self.warm_start.enabled and (
            self.warm_start.candidate_k == 0
            and self.warm_start.candidate_delta_db == 0
        ):
            raise ValueError("Enabled warm start requires a candidate criterion.")
        return self

    @classmethod
    def from_yaml(
        cls, path: str | None, simulation_id: str | None = None
    ) -> "OptimConfig":
        """Load an :class:`OptimConfig` from a YAML file.

        Parameters
        ----------
        path:
            Path to the YAML configuration file.
        simulation_id:
            Optional override for the simulation ID.  Raises if the file
            already contains a non-``None`` simulation ID.
        """
        if path is None:
            raise ValueError("Path to YAML configuration file must be provided.")
        config_dict = cls._from_yaml(path)
        config = cls(**config_dict)

        file_sim_id = config_dict.get("simulation_id", None)
        if simulation_id is not None:
            if file_sim_id is not None:
                raise ValueError(
                    f"Simulation ID in yaml file is not None ({file_sim_id}), "
                    "but a new simulation ID was provided."
                )
            if not isinstance(simulation_id, str):
                raise TypeError("Simulation ID must be a string.")
            config.update_recursively({"simulation_id": simulation_id})
        config.path_to_config = str(path)

        return config

    @model_validator(mode="after")
    def _set_log_directories(self) -> "OptimConfig":
        """Create the log directory on disk after the model has been validated."""
        log_dir = os.path.join(self.base_dir, self.log_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        return self
