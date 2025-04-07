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
        self._mip = None
        self.state = SolverState.INIT
        
    def load_model(self, mip):
        self._mip = mip
        self.state = SolverState.MODEL_LOADED
        return MPSolverResponseStatus.NOT_SOLVED
        
    def extract_lp_features_at_root(self, solving_params):
        """提取LP根节点特征
        Returns:
            features: {
                'V': {  # 变量特征
                    'type': shape=(n_vars, 4),      # [binary, integer, impl_integer, continuous]
                    'coef': shape=(n_vars, 1),      # 归一化的目标系数
                    'has_lb': shape=(n_vars, 1),    # 是否有下界
                    'has_ub': shape=(n_vars, 1),    # 是否有上界
                    'sol_is_at_lb': shape=(n_vars, 1),    # 解是否在下界
                    'sol_is_at_ub': shape=(n_vars, 1),    # 解是否在上界
                    'sol_frac': shape=(n_vars, 1),        # 解的分数部分
                    'basis_status': shape=(n_vars, 4),    # 基状态的one-hot编码
                    'reduced_cost': shape=(n_vars, 1),    # 归一化的简化成本
                    'age': shape=(n_vars, 1),             # 归一化的LP年龄
                    'sol_val': shape=(n_vars, 1),         # 解值
                },
                'C': {  # 约束特征
                    'obj_cos_sim': shape=(n_cons_total, 1),  # 与目标函数的余弦相似度
                    'bias': shape=(n_cons_total, 1),         # 归一化的约束右端项
                    'is_tight': shape=(n_cons_total, 1),     # 约束是否紧的
                    'dualsol_val': shape=(n_cons, 1),        # 归一化的对偶值
                    'age': shape=(n_cons_total, 1),          # 归一化的LP年龄
                },
                'E': {  # 边特征
                    'coef': shape=(n_edges, 1),        # 归一化的约束系数
                    'indices': shape=(n_edges, 2),     # [cons_idx, var_idx]
                }
            }
        """
        try:
            # 提取静态特征
            static_features = self._mip.extract_static_features()
            if not self._mip.check_feature_completeness(static_features, check_dynamic=False):
                print("警告:静态特征提取不完整")
                return None
                
            # 获取LP解信息
            solution, basis_status, reduced_costs, is_tight, dual_values, age = self._get_lp_solution()
            
            # 提取动态特征
            features = self._mip.extract_dynamic_features(
                static_features=static_features,
                solution=solution,
                basis_status=basis_status,
                reduced_costs=reduced_costs,
                is_tight=is_tight,
                dual_values=dual_values,
                age=age
            )
            
            # 检查特征完整性
            if not self._mip.check_feature_completeness(features, check_dynamic=True):
                print("警告:动态特征提取不完整")
                return None
                
            return features
            
        except Exception as e:
            print(f"LP特征提取过程中发生错误: {str(e)}")
            return None
            
    def _get_lp_solution(self):
        """获取LP解信息"""
        try:
            from pyscipopt import Model, quicksum
            
            # 创建SCIP求解器实例
            model = Model()
            
            # 添加变量
            vars = []
            for var in self._mip.variable:
                x = model.addVar(
                    lb=var.lower_bound,
                    ub=var.upper_bound,
                    vtype='C',  # 连续变量
                    name=var.name
                )
                vars.append(x)
                
            # 设置目标函数
            obj = quicksum(var.objective_coefficient * x 
                          for var, x in zip(self._mip.variable, vars))
            if self._mip.maximize:
                model.setObjective(obj, sense='maximize')
            else:
                model.setObjective(obj, sense='minimize')
                
            # 添加约束
            constrs = []
            for cons in self._mip.constraint:
                expr = quicksum(coef * vars[idx] 
                              for idx, coef in zip(cons.var_index, cons.coefficient))
                
                if cons.lower_bound > float('-inf'):
                    c = model.addCons(expr >= cons.lower_bound, name=f"{cons.name}_lb")
                    constrs.append(c)
                if cons.upper_bound < float('inf'):
                    c = model.addCons(expr <= cons.upper_bound, name=f"{cons.name}_ub")
                    constrs.append(c)
                    
            # 求解LP
            model.hideOutput()  # 隐藏求解输出
            model.optimize()
            
            n_vars = len(self._mip.variable)
            n_cons = len(self._mip.constraint)
            
            if model.getStatus() == 'optimal':
                # 获取解值
                solution = np.array([model.getVal(x) for x in vars])
                
                # 获取基状态 (0=LOWER, 1=BASIC, 2=UPPER, 3=ZERO)
                basis_status = np.zeros(n_vars, dtype=np.int32)
                for i, x in enumerate(vars):
                    val = model.getVal(x)
                    lb = self._mip.variable[i].lower_bound
                    ub = self._mip.variable[i].upper_bound
                    if abs(val - lb) < 1e-6:
                        basis_status[i] = 0  # LOWER
                    elif abs(val - ub) < 1e-6:
                        basis_status[i] = 2  # UPPER
                    elif abs(val) < 1e-6:
                        basis_status[i] = 3  # ZERO
                    else:
                        basis_status[i] = 1  # BASIC
                
                # 获取简化成本
                reduced_costs = np.zeros(n_vars)
                cols = model.getLPColsData()
                for i, col in enumerate(cols):
                    if i < n_vars:
                        reduced_costs[i] = col.getObjCoeff()
                
                # 获取约束紧致性和对偶值
                is_tight = np.zeros(n_cons)
                dual_values = np.zeros(n_cons)
                
                # 获取所有约束的活动度
                activities = []
                lhss = []
                rhss = []
                for cons in constrs:
                    activities.append(model.getActivity(cons))
                    lhss.append(model.getLhs(cons))
                    rhss.append(model.getRhs(cons))
                
                # 处理每个原始约束
                cons_idx = 0
                for i, cons in enumerate(self._mip.constraint):
                    # 如果约束有下界
                    if cons.lower_bound > float('-inf'):
                        if cons_idx < len(activities):
                            if abs(activities[cons_idx] - lhss[cons_idx]) < 1e-6:
                                is_tight[i] = 1
                                dual_values[i] = model.getDualSolVal(constrs[cons_idx])
                        cons_idx += 1
                    
                    # 如果约束有上界
                    if cons.upper_bound < float('inf'):
                        if cons_idx < len(activities):
                            if abs(activities[cons_idx] - rhss[cons_idx]) < 1e-6:
                                is_tight[i] = 1
                                dual_values[i] = model.getDualSolVal(constrs[cons_idx])
                        cons_idx += 1
                
                # 获取LP迭代次数
                age = model.getNLPIterations()
                
                return solution, basis_status, reduced_costs, is_tight, dual_values, age
                
            else:
                print(f"LP求解失败: {model.getStatus()}")
                return self._get_default_solution()
            
        except Exception as e:
            print(f"获取LP解时出错: {str(e)}")
            return self._get_default_solution()
        
    def _get_default_solution(self):
        """返回默认的零解"""
        n_vars = len(self._mip.variable)
        n_cons = len(self._mip.constraint)
        return (np.zeros(n_vars), np.zeros(n_vars, dtype=np.int32),
                np.zeros(n_vars), np.zeros(n_cons), np.zeros(n_cons), 0)
            
    def solve(self, solving_params):
        # 简化实现
        return MPSolverResponseStatus.OPTIMAL
        
    def get_best_solution(self):
        # 简化实现,返回全零解
        solution = MPSolutionResponse(
            objective_value=0.0,
            variable_value=[0.0] * len(self._mip.variable),
            status_str="Optimal",
            status=MPSolverResponseStatus.OPTIMAL
        )
        return solution
