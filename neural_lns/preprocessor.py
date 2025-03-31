#ad474fc1-d6be-4e4e-bf77-662a4f69c706
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
"""Wrapper APIs for MIP preprocessing."""

import abc
from typing import Optional, Tuple
from mip_utils import MPModel, MPVariable, MPConstraint, MPSolverResponseStatus, MPSolutionResponse, is_var_binary
from pyscipopt import Model, SCIP_RESULT, SCIP_PARAMSETTING, quicksum
import numpy as np
import math


class Preprocessor(abc.ABC):
  """Class describing the API used to access a MIP presolver.

  This class should be used as a wrapper around any general presolving method
  MIPs, e.g. the presolver used in SCIP. The API only needs to expose a
  presolve method that turns a MPModel into a presolved MPModel, as well as a
  get_original_solution method that turns a solution to the presolved model to
  one a solution to the original.
  """

  def __init__(self, *args, **kwargs):
    """Initializes the preprocessor."""

  def presolve(
      self, mip: MPModel
  ) -> Tuple[MPSolverResponseStatus, Optional[MPModel]]:
    """Presolve the given MIP as MPModel.

    Args:
      mip: MPModel for MIP instance to presolve.

    Returns:
      status: A Status returned by the presolver.
      result: The MPModel of the presolved problem.
    """
    raise NotImplementedError('presolve method has to be implemented')

  def get_original_solution(
      self,
      solution: MPSolutionResponse) -> MPSolutionResponse:
    raise NotImplementedError(
        'get_original_solution method has to be implemented')


class SCIPPreprocessor(Preprocessor):
    """SCIP-based MIP预处理器，支持模型简化解映射"""
    
    def __init__(self):
        self.orig_to_presolved = {}  # 原始变量名到预求解变量的映射
        self.fixed_vars = {}         # 被固定的变量名及其值
        self.aggregated_vars = {}    # 被聚合变量的线性表达式


    def _scip_model_to_mpmodel(self, scip_model: Model) -> MPModel:
        """将SCIP模型转换为自定义MPModel结构（兼容旧版PySCIPOpt）"""
        mip = MPModel()
        mip.maximize = scip_model.getObjectiveSense() == 'maximize'
        
        scip_vars = scip_model.getVars()
        
        # === 1. 解析目标函数系数 ===
        objective = scip_model.getObjective()
        
        # 检查目标是否为线性（兼容方法）
        # try:
        #     terms = scip_model.getTermsLinear(objective)
        # except Exception as e:
        #     raise ValueError("仅支持线性目标函数") from e
        
        var_coeff_map = {var: coeff for var, coeff in zip(terms['variables'], terms['coeffs'])}
        mip.objective_offset = terms['constant']
        
        # === 2. 添加变量 ===
        for scip_var in scip_vars:
            mp_var = MPVariable(
                name=scip_var.name,
                lower_bound=scip_var.getLbOriginal(),
                upper_bound=scip_var.getUbOriginal(),
                is_integer=scip_var.vtype() in ['INTEGER', 'BINARY'],
                objective_coefficient=var_coeff_map.get(scip_var, 0.0)
            )
            mip.variable.append(mp_var)
        
        # === 3. 添加约束 ===
        for cons in scip_model.getConss():
            if not cons.isInitial():
                continue
            
            # 处理约束边界
            lhs = cons.getLhs()
            rhs = cons.getRhs()
            lhs = lhs if lhs != -scip_model.infinity() else -np.inf
            rhs = rhs if rhs != scip_model.infinity() else np.inf
            
            mp_cons = MPConstraint(
                lower_bound=lhs,
                upper_bound=rhs,
                name=cons.name
            )
            
            # 检查约束是否为线性（兼容方法）
            try:
                terms = scip_model.getValsLinear(cons)
            except Exception as e:
                raise ValueError("仅支持线性约束") from e
            
            # 添加变量索引和系数
            for var, coeff in terms.items():
                mp_cons.var_index.append(scip_vars.index(var))
                mp_cons.coefficient.append(coeff)
            
            mip.constraint.append(mp_cons)
        
        return mip

    def presolve(self, original_mip: MPModel) -> Tuple[MPSolverResponseStatus, Optional[MPModel]]:
        """执行SCIP预求解并记录变量映射"""
        self.scip_model = Model()
        self.scip_model.setPresolve(SCIP_PARAMSETTING.AGGRESSIVE)
        
        # 1. 加载原始模型到SCIP
        var_map = {}
        for var in original_mip.variable:
            scip_var = self.scip_model.addVar(
                name=var.name,
                lb=var.lower_bound,
                ub=var.upper_bound,
                vtype='BINARY' if is_var_binary(var) else 'INTEGER' if var.is_integer else 'CONTINUOUS'
            )
            var_map[var.name] = scip_var
        
        for cons in original_mip.constraint:
            expr = quicksum(
            var_map[original_mip.variable[idx].name] * coeff
            for idx, coeff in zip(cons.var_index, cons.coefficient)
        )
        if cons.lower_bound == cons.upper_bound:
            self.scip_model.addCons(expr == cons.lower_bound, name=cons.name)
        
        # 处理区间约束（lb < ub）
        else:
            # 添加下界约束（如果存在）
            if cons.lower_bound > -math.inf:
                self.scip_model.addCons(
                    expr >= cons.lower_bound, 
                    name=f"{cons.name}_lb"
                )
            # 添加上界约束（如果存在）
            if cons.upper_bound < math.inf:
                self.scip_model.addCons(
                    expr <= cons.upper_bound, 
                    name=f"{cons.name}_ub"
                )
        
        # 2. 执行预求解
        self.scip_model.presolve()
        
        # 3. 检查预求解结果状态
        if self.scip_model.getStatus() in ['infeasible', 'unbounded']:
            return MPSolverResponseStatus.INFEASIBLE_OR_UNBOUNDED, None
        
        # 4. 提取预求解后模型
        presolved_mip = self._scip_model_to_mpmodel(self.scip_model)
        
        # 5. 记录变量映射关系（关键步骤）
        self._record_var_mappings(original_mip, presolved_mip)
        
        return MPSolverResponseStatus.OPTIMAL, presolved_mip

    def _record_var_mappings(self, original_mip: MPModel, presolved_mip: MPModel):
        """记录变量如何被预求解修改"""
        orig_names = {var.name for var in original_mip.variable}
        presolved_names = {var.name for var in presolved_mip.variable}
        
        # 查找被固定的变量
        for var in self.scip_model.getVars():
            if var.name not in presolved_names and var.name in orig_names:
                if var.getLb() == var.getUb():
                    self.fixed_vars[var.name] = var.getLb()
        
        # 记录变量聚合关系（需SCIP插件支持，此处简化）
        # 实际项目需通过SCIP回调获取详细信息

    def get_original_solution(self, presolved_solution: MPSolutionResponse) -> MPSolutionResponse:
        """将预求解模型的解映射回原始模型"""
        orig_solution = MPSolutionResponse(
            objective_value=presolved_solution.objective_value,
            variable_value=[],
            status_str=presolved_solution.status_str,
            status=presolved_solution.status
        )
        
        # 1. 处理被固定的变量
        for var in self.original_mip.variable:
            if var.name in self.fixed_vars:
                orig_solution.variable_value.append(self.fixed_vars[var.name])
            else:
                # 假设预求解后的变量名与原始模型一致（实际需更复杂处理）
                presolved_idx = next(i for i, v in enumerate(self.presolved_mip.variable) 
                                 if v.name == var.name)
                orig_solution.variable_value.append(
                    presolved_solution.variable_value[presolved_idx])
        
        return orig_solution
