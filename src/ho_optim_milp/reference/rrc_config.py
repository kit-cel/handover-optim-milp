"""RRC configuration module."""

from typing import Any
from pydantic import Field, model_validator, StrictFloat, StrictStr

from ..common.base_config import BaseConfig
from ..common.type_aliases import (
    StrictNonNegativeInt,
    StrictPositiveInt,
)


class RRCConfig(BaseConfig):
    """RRC configuration class."""

    # General RRC parameters
    t_res_ms: StrictPositiveInt = Field(
        default=0, description="RRC time resolution in milliseconds."
    )

    # Resource management
    max_num_ues: StrictNonNegativeInt = Field(
        default=100, description="Maximum number of UEs the RRC can handle."
    )
    num_prb: StrictNonNegativeInt = Field(
        default=10**32 - 1, description="Number of physical resource blocks (PRBs)."
    )

    # Counters and timers for radio link monitoring
    n310: StrictNonNegativeInt = Field(
        default=10,
        description="Number of consecutive out-of-sync measurements for N310.",
    )
    n311: StrictNonNegativeInt = Field(
        default=10,
        description="Number of consecutive out-of-sync measurements for N311.",
    )
    t301_ms: StrictNonNegativeInt = Field(
        default=1_000, description="Timer T301 duration in ms."
    )
    t310_ms: StrictNonNegativeInt = Field(
        default=1_000, description="Timer T310 duration in ms."
    )
    t311_ms: StrictNonNegativeInt = Field(
        default=30_000, description="Timer T311 duration in ms."
    )

    # RLM QoS thresholds and filter parameters
    q_in_db: StrictFloat = Field(
        default=-6.0,
        description="Radio link outage threshold Q-in parameter in dB.",
    )
    q_out_db: StrictFloat = Field(
        default=-8.0,
        description="Radio link recovery threshold Q-out parameter in dB.",
    )
    l1_filter_sample_interval_ms: StrictPositiveInt = Field(
        default=10, description="L1 filter sample interval in ms."
    )
    l1_q_in_filter_len_ms: StrictPositiveInt = Field(
        default=100, description="L1 filter length in ms for Q-in signal."
    )
    l1_q_out_filter_len_ms: StrictPositiveInt = Field(
        default=200, description="L1 filter length in ms for Q-out signal."
    )

    # HO event triggering and L3 filter parameters
    l3_filter_coef: StrictPositiveInt = Field(
        default=1,
        description="L3 filter coefficient for measurements triggering HO events.",
    )
    l3_filter_sample_reference_interval_ms: StrictPositiveInt = Field(
        default=200,
        description="L3 filter sample reference interval in ms.",
    )
    ho_trigger_quantity: StrictStr = Field(
        default="rsrp", description="HO trigger quantity."
    )
    ho_event_config: dict[str, Any] = Field(
        default_factory=dict, description="HO event configuration."
    )

    # Other UE related RRC parameters
    cell_selection_quality_criterion: StrictStr = Field(
        default="sinr", description="Cell selection quality criterion (RSRQ or SINR)."
    )
    use_bs_blacklist: bool = Field(
        default=False, description="Whether to use a blacklist for BSs."
    )

    # Simulation times for RRC procedures in ms
    simulated_t_rach_ms: StrictNonNegativeInt = Field(
        default=0, description="RACH simulation duration in ms."
    )
    t_ho_prep_ms_simulated: StrictNonNegativeInt = Field(
        default=40, description="HO preparation simulation duration in ms."
    )
    t_ho_exec_ms_simulated: StrictNonNegativeInt = Field(
        default=40, description="HO execution simulation duration in ms."
    )
    t_rlfr_ms_simulated: StrictNonNegativeInt = Field(
        default=1000, description="Re-establishment simulation duration in ms."
    )
    t_mts_ms: StrictNonNegativeInt = Field(
        default=1000, description="Minimum-time-of-stay for ping-pong detection."
    )

    @classmethod
    def from_yaml(
        cls, path: str | None, simulation_id: str | None = None
    ) -> "RRCConfig":
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
    def _validate(self) -> "RRCConfig":
        return self
