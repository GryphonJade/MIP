# Copyright 2021 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Common utilities for Solver."""

import abc
import enum
from typing import Any, Dict, Optional
import ml_collections
from mip_utils import MPModel, MPVariable, MPConstraint, MPSolverResponseStatus
from pyscipopt import Model, quicksum
import numpy as np

class SolverState(enum.Enum):
    INIT = 0
    MODEL_LOADED = 1
    FINISHED = 2

class Solver(abc.ABC):
    """Wrapper around a given classical MIP solver.

    This class contains the API needed to communicate with a MIP solver, e.g.
    SCIP.
    """

    def load_model(self, mip: Any) -> SolverState:
        """Loads a MIP model into the solver."""
        raise NotImplementedError('load_model method should be implemented')

    def solve(
        self, solving_params: ml_collections.ConfigDict
    ) -> MPSolverResponseStatus:
        """Solves the loaded MIP model."""
        raise NotImplementedError('solve method should be implemented')

    def get_best_solution(self) -> Optional[Any]:
        """Returns the best solution found from the last solve call."""
        raise NotImplementedError('get_best_solution method should be implemented')

    def add_solution(self, solution: Any) -> bool:
        """Adds a known solution to the solver."""
        raise NotImplementedError('add_solution method should be implemented')

    def extract_lp_features_at_root(
        self, solving_params: ml_collections.ConfigDict) -> Dict[str, Any]:
        """Returns a dictionary of root node features."""
        raise NotImplementedError(
            'extract_lp_features_at_root method should be implemented')

def is_var_binary(variable: Any) -> bool:
    """Checks whether a given variable is binary."""
    lb_is_zero = np.isclose(variable.lower_bound, 0)
    ub_is_one = np.isclose(variable.upper_bound, 1)
    return variable.is_integer and lb_is_zero and ub_is_one

class SCIPSolver(Solver):
    def __init__(self):
        self.model = None
        self.state = SolverState.INIT

    def load_model(self, mip: MPModel) -> SolverState:
        self.model = Model()
        # 添加变量
        scip_vars = {}
        for var in mip.variable:
            scip_var = self.model.addVar(
                name=var.name,
                lb=var.lower_bound,
                ub=var.upper_bound,
                vtype='BINARY' if is_var_binary(var) else 'INTEGER' if var.is_integer else 'CONTINUOUS'
            )
            scip_vars[var.name] = scip_var
        # 添加约束
        for constr in mip.constraint:
            expr = quicksum(scip_vars[mip.variable[idx].name] * coeff for idx, coeff in zip(constr.var_index, constr.coefficient))
            self.model.addCons(expr >= constr.lower_bound)
            self.model.addCons(expr <= constr.upper_bound)
        # 设置目标
        obj = quicksum(var.objective_coefficient * scip_vars[var.name] for var in mip.variable)
        self.model.setObjective(obj, sense='maximize' if mip.maximize else 'minimize')
        self.state = SolverState.MODEL_LOADED
        return self.state

    def extract_lp_features_at_root(self, params) -> Dict[str, Any]:
        # 提取根节点LP特征的实现（示例）
        self.model.optimize()
        return {
            'variable_names': [var.name for var in self.model.getVars()],
            'variable_features': np.array([])  # 实际特征数据
        }
