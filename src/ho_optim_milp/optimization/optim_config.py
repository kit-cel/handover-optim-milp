"""Config for MILP optimization configuration class."""

from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from mobile_network_sim.rrc import RRCConfig
from pydantic import BaseModel, ConfigDict, Field, model_validator



class GurobiSolverConfig(BaseModel):
    """Portable solver settings with no machine-specific defaults."""

    threads: int | None = Field(default=None, ge=1)
    time_limit_s: float | None = Field(default=None, gt=0)
    mip_gap: float | None = Field(
        default=0.0, ge=0,
        description="Relative optimality gap; zero overrides Gurobi's nonzero default.",
    )
    log_to_console: bool = True
    concurrent_mip: int = Field(default=1, ge=1)
    parameter_file: str | None = None
    gurobi_log_file: str | None = None

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


class OptimConfig(BaseModel):
    """MILP options with the simulator's RRC configuration."""

    lambda_r: float = Field(default=1.0, ge=0)
    discount_gamma: float = Field(default=1.0, gt=0, le=1)
    initial_access: Literal["free", "strongest_rsrp"] = "free"
    handover_target_mode: Literal["free", "rank1_neighbor"] = "free"
    gurobi: GurobiSolverConfig = Field(default_factory=GurobiSolverConfig)
    warm_start: WarmStartSolverConfig = Field(default_factory=WarmStartSolverConfig)

    rrc: "RRCConfig"
    log_dir: Path = Path("logs")
    simulation_id: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M%S%f")
    )

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate_optimizer(self) -> "OptimConfig":
        if self.rrc.handover_mode != "standard":
            raise ValueError("The MILP supports standard handover only")
        if self.warm_start.enabled and (
            self.warm_start.candidate_k == 0
            and self.warm_start.candidate_delta_db == 0
        ):
            raise ValueError("Enabled warm start requires a candidate criterion.")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "OptimConfig":
        """Load and validate an optimization YAML file."""
        with Path(path).open("r", encoding="utf-8") as stream:
            payload = yaml.safe_load(stream)
        if not isinstance(payload, dict):
            raise ValueError("Optimization config must be a YAML mapping")
        return cls.model_validate(payload)
