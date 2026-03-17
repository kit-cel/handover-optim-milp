"""Abstract Gurobi base optimizer class."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import gurobipy as gp
import numpy as np

if TYPE_CHECKING:
    from .optim_config import OptimConfig
    from ..result_manager.episode_result import EpisodeResult


class GurobiBaseOptimizer(ABC):
    """Abstract base class for Gurobi optimizers."""

    model: gp.Model
    obj: gp.LinExpr | gp.QuadExpr | gp.MLinExpr | gp.MQuadExpr
    minimize_or_maximize: int

    _variables_added: bool = False
    _constraints_added: bool = False
    _optimal_solution_found: bool = False
    _optimization_finished: bool = False

    def __init__(self, config: "OptimConfig") -> None:
        """Initialize the Gurobi base optimizer."""
        self.debug = config.debug
        self.config = config
        self.rrc_config = config.rrc

    def setup_model(self, name: str | None = None) -> None:
        """Set up the Gurobi optimization model."""
        name = name if name is not None else "GurobiBaseOptimizerModel"
        self.model = gp.Model(name)

    def optimize(self) -> None:
        """Run the optimization process."""
        if self._variables_added is False:
            raise RuntimeError("Variables have not been added to the model.")
        if self._constraints_added is False:
            raise RuntimeError("Constraints have not been added to the model.")
        if self.obj is None or self.minimize_or_maximize is None:
            raise RuntimeError("Objective function has not been set.")

        if self.debug:
            self.model.setObjective(0.0, self.minimize_or_maximize)
        else:
            self.model.setObjective(self.obj, self.minimize_or_maximize)

        self.model.optimize()

        self._optimal_solution_found = self.model.status == gp.GRB.OPTIMAL
        self._optimization_finished = True

    @abstractmethod
    def load_data(self, data: "EpisodeResult") -> None:
        """Load necessary data for optimization."""

    @abstractmethod
    def extract_results(self) -> dict[str, np.ndarray]:
        """Retrieve optimization results."""

    @abstractmethod
    def get_result_metrics(self) -> dict[str, float]:
        """Get the optimization results in a structured format."""

    @abstractmethod
    def _add_variables(self) -> None:
        """Add variables to the Gurobi model."""

    @abstractmethod
    def _add_constraints(self) -> None:
        """Add constraints to the Gurobi model."""

    @abstractmethod
    def _set_objective(self) -> None:
        """Set the objective function for the Gurobi model."""
