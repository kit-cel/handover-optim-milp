"""Module defining helper classes for the handover optimization MILP model."""

from dataclasses import dataclass

import gurobipy as gp
import numpy as np

from .milp_variables import ModelVars


@dataclass(frozen=True)
class ProblemData:
    """Structured problem data for optimization."""

    ue_idx: int
    n_bs: int
    n_steps: int
    rsrp_dbm: np.ndarray
    sinr_db: np.ndarray
    q_in_mat: np.ndarray
    q_out_mat: np.ndarray

    @property
    def capacity(self) -> np.ndarray:
        """Calculate capacity in Mbps from SINR."""
        return np.log2(1 + 10 ** (self.sinr_db / 10))


@dataclass(frozen=True)
class RRCConstants:
    """
    RRC constants derived from configuration.

    Note: timer values must be converted from ms to integer steps based on t_res_ms.
    """

    t_res_ms: int
    q_in_db: float
    q_out_db: float
    n310: int
    n311: int
    t310: int
    t_ho_prep_sim: int
    t_ho_exec_sim: int
    t_rlfr_sim: int
    t_mts: int

    @property
    def t_ho_sim(self) -> int:
        """Total simulated HO time (prep + exec)."""
        return self.t_ho_prep_sim + self.t_ho_exec_sim


@dataclass(frozen=True)
class WarmStartConfig:
    """Configuration for safe candidate pruning warm start."""

    candidate_k: int = 0
    candidate_delta_db: float = 0.0
    candidate_metric: str = "sinr"
    candidate_include_prev: bool = True
    candidate_include_forced: bool = True
    safe_pruning_time_limit_s: float | None = None
    safe_pruning_mip_gap: float | None = 0.0


@dataclass(frozen=True)
class ConstraintContext:
    """Context object containing all necessary information for building constraints."""

    model: gp.Model
    vars: ModelVars
    data: ProblemData
    consts: RRCConstants
    debug: bool | None
    ref_results: dict[str, np.ndarray] | None
