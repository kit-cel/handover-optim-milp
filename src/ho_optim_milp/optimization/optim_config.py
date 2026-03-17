"""Config for MILP optimization configuration class."""

import os

from pydantic import StrictBool, StrictFloat, Field, model_validator

from ..common.base_config import BaseConfig
from ..common.type_aliases import StrictPositiveInt
from ..reference.rrc_config import RRCConfig


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

    rrc: "RRCConfig"

    @classmethod
    def from_yaml(
        cls, path: str | None, simulation_id: str | None = None
    ) -> "OptimConfig":
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
    def _set_seeds(self) -> "OptimConfig":
        """Set the seed for the environment configuration."""
        if self.seed is not None:
            # Forward the seed to lower-level configurations
            pass
        return self

    @model_validator(mode="after")
    def _set_log_directories(self) -> "OptimConfig":
        log_dir = os.path.join(self.base_dir, self.log_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        return self
