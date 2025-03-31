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
from pyscipopt import Model, quicksum, SCIP_PARAMSETTING
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
        self._mip = None
        self._scip_vars = []
        self._static_features = None

    def load_model(self, mip: MPModel) -> MPSolverResponseStatus:
        """加载MIP模型到SCIP求解器"""
        self._mip = mip
        self.model = Model()
        
        # 添加变量
        for var in mip.variable:
            scip_var = self.model.addVar(
                name=var.name,
                vtype='BINARY' if is_var_binary(var) else 'INTEGER' if var.is_integer else 'CONTINUOUS',
                lb=var.lower_bound,
                ub=var.upper_bound,
                obj=var.objective_coefficient
            )
            self._scip_vars.append(scip_var)
            
        # 添加约束
        for cons in mip.constraint:
            expr = quicksum(
                self._scip_vars[idx] * coeff 
                for idx, coeff in zip(cons.var_index, cons.coefficient)
            )
            if cons.lower_bound == cons.upper_bound:
                self.model.addCons(expr == cons.lower_bound, name=cons.name)
            else:
                if cons.lower_bound > -np.inf:
                    self.model.addCons(expr >= cons.lower_bound, name=f"{cons.name}_lb")
                if cons.upper_bound < np.inf:
                    self.model.addCons(expr <= cons.upper_bound, name=f"{cons.name}_ub")
                    
        # 设置目标函数方向
        if mip.maximize:
            self.model.setMaximize()
        else:
            self.model.setMinimize()
            
        self.state = SolverState.MODEL_LOADED
        return MPSolverResponseStatus.NOT_SOLVED
        
    def extract_lp_features_at_root(self, params=None):
        """在根节点提取LP特征。"""
        if params is None:
            params = {}
        
        # 设置SCIP参数
        self.model.setParam('display/verblevel', 0)
        self.model.setParam('limits/time', 60)  # 限制为1分钟
        self.model.setParam('lp/iterlim', 1000)  # 限制LP迭代次数
        self.model.setParam('presolving/maxrounds', 0)  # 禁用预求解
        self.model.setParam('separating/maxroundsroot', 0)  # 禁用根节点切割
        
        # 优化模型
        try:
            self.model.optimize()
            
            # 检查求解状态
            status = self.model.getStatus()
            if status != 'optimal':
                print(f"警告: LP求解未达到最优状态，当前状态: {status}")
                return self._mip.extract_static_features()
            
            # 获取求解结果
            solution = np.array([var.getLPSol() for var in self.model.getVars()])
            if len(solution) == 0:
                print("警告: 无法获取LP列数据")
                return self._mip.extract_static_features()
                
            # 提取动态特征
            return self._mip.extract_dynamic_features(solution)
            
        except Exception as e:
            print(f"LP求解过程中发生错误: {str(e)}")
            return self._mip.extract_static_features()
