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
"""MIP utility functions."""

import copy
import dataclasses
import enum
import math
from typing import Any, Dict, List, Optional, Tuple

from absl import logging
import numpy as np
from scipy import sparse

# from neural_lns import sampling


class MPSolverResponseStatus(enum.Enum):
  """Enum of solver statuses."""
  OPTIMAL = 0
  FEASIBLE = 1
  NOT_SOLVED = 2
  INFEASIBLE = 3
  UNBOUNDED = 4
  INFEASIBLE_OR_UNBOUNDED = 5
  STOPPED = 6
  UNKNOWN = 7
  FAILED = 8
  BESTSOLLIMIT = 9


@dataclasses.dataclass
class MPVariable:
  """MPVariable contains all the information related to a single variable."""
  # Lower and upper bounds; lower_bound must be <= upper_bound.
  lower_bound: float = -math.inf
  upper_bound: float = math.inf
  # The coefficient of the variable in the objective. Must be finite.
  objective_coefficient: float = 0.0
  # True if the variable is constrained to be integer.
  is_integer: bool = True
  # The name of the variable.
  name: str = ""
  # The age of the variable (LP迭代次数)
  age: int = 0


@dataclasses.dataclass
class MPConstraint:
  """MPConstraint contains all the information related to a single constraint."""
  # var_index[i] is the variable index (w.r.t. to "variable" field of
  # MPModel) of the i-th linear term involved in this constraint, and
  # coefficient[i] is its coefficient. Only the terms with non-zero
  # coefficients need to appear. var_index may not contain duplicates.
  var_index: List[int] = dataclasses.field(default_factory=list)
  coefficient: List[float] = dataclasses.field(default_factory=list)
  # lower_bound must be <= upper_bound.
  lower_bound: float = -math.inf
  upper_bound: float = math.inf
  # The name of the constraint.
  name: str = ""


@dataclasses.dataclass
class MPModel:
  """MPModel fully encodes a Mixed-Integer Linear Programming model."""
  
  def __init__(self):
    self.variable = []
    self.constraint = []
    self.objective_offset = 0.0
    self.maximize = False
    self.historical_incumbents = []  # 用于存储历史incumbent解
    
  def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
    """归一化向量"""
    norm = np.linalg.norm(vector)
    if norm > 0:
      return vector / norm
    return vector
    
  def _get_constraint_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
    """获取约束矩阵及其范数
    Returns:
        cons_matrix: 约束矩阵, shape=(n_cons, n_vars)
        cons_norms: 每个约束的范数, shape=(n_cons,)
    """
    n_cons = len(self.constraint)
    n_vars = len(self.variable)
    cons_matrix = np.zeros((n_cons, n_vars))
    
    for i, cons in enumerate(self.constraint):
      for var_idx, coeff in zip(cons.var_index, cons.coefficient):
        cons_matrix[i, var_idx] = coeff
        
    cons_norms = np.linalg.norm(cons_matrix, axis=1)
    cons_norms[cons_norms == 0] = 1.0  # 避免除零
    
    return cons_matrix, cons_norms
    
  def add_incumbent(self, incumbent: np.ndarray):
    """添加新的incumbent解到历史记录中
    Args:
        incumbent: incumbent解的值, shape=(n_vars,)
    """
    self.historical_incumbents.append(incumbent.copy())
    # 只保留最近的NUM_PAST_INCUMBENTS个incumbent
    if len(self.historical_incumbents) > 3:  # NUM_PAST_INCUMBENTS = 3
      self.historical_incumbents.pop(0)
      
  def check_feature_completeness(self, features: Dict[str, Dict[str, np.ndarray]], check_dynamic: bool = False) -> bool:
    """检查特征是否完整
    Args:
        features: 特征字典
        check_dynamic: 是否检查动态特征
    Returns:
        是否完整
    """
    # 静态特征
    static_features = {
        'V': ['type', 'coef', 'has_lb', 'has_ub'],
        'C': ['obj_cos_sim', 'bias'],
        'E': ['coef', 'indices']
    }
    
    # 动态特征
    dynamic_features = {
        'V': ['sol_is_at_lb', 'sol_is_at_ub', 'sol_frac', 'basis_status', 
             'reduced_cost', 'age', 'sol_val'],
        'C': ['is_tight', 'dualsol_val', 'age']
    }
    
    # 选择要检查的特征
    required_features = static_features.copy()
    if check_dynamic:
        for tensor in required_features:
            if tensor in dynamic_features:
                required_features[tensor].extend(dynamic_features[tensor])
    
    # 检查特征完整性
    for tensor, required in required_features.items():
        if tensor not in features:
            print(f"缺少{tensor}特征")
            return False
        for feat in required:
            if feat not in features[tensor]:
                print(f"缺少{tensor}.{feat}特征")
                return False
    return True

  def extract_static_features(self) -> Dict[str, Dict[str, np.ndarray]]:
    """提取静态特征(不需要求解信息的特征)
    Returns:
        features: {
            'V': {  # 变量特征
                'type': shape=(n_vars, 4),      # [binary, integer, impl_integer, continuous]
                'coef': shape=(n_vars, 1),      # 归一化的目标系数
                'has_lb': shape=(n_vars, 1),    # 是否有下界
                'has_ub': shape=(n_vars, 1),    # 是否有上界
            },
            'C': {  # 约束特征
                'obj_cos_sim': shape=(n_cons_total, 1),  # 与目标函数的余弦相似度
                'bias': shape=(n_cons_total, 1),         # 归一化的约束右端项
            },
            'E': {  # 边特征
                'coef': shape=(n_edges, 1),        # 归一化的约束系数
                'indices': shape=(n_edges, 2),     # [cons_idx, var_idx]
            }
        }
    """
    features = {'V': {}, 'C': {}, 'E': {}}
    n_vars = len(self.variable)
    
    # === 变量特征(V) ===
    # 变量类型的one-hot编码
    var_types = np.zeros((n_vars, 4))
    for i, var in enumerate(self.variable):
      if var.is_integer:
        if var.lower_bound == 0 and var.upper_bound == 1:
          var_types[i, 0] = 1  # binary
        else:
          var_types[i, 1] = 1  # integer
      else:
        var_types[i, 3] = 1  # continuous
    features['V']['type'] = var_types
    
    # 目标系数(归一化)
    obj_coeffs = np.array([var.objective_coefficient for var in self.variable])
    obj_norm = np.linalg.norm(obj_coeffs)
    if obj_norm > 0:
      features['V']['coef'] = (obj_coeffs / obj_norm).reshape(-1, 1)
    else:
      features['V']['coef'] = obj_coeffs.reshape(-1, 1)
    
    # 边界指示器
    features['V']['has_lb'] = np.array([var.lower_bound > -np.inf for var in self.variable]).reshape(-1, 1)
    features['V']['has_ub'] = np.array([var.upper_bound < np.inf for var in self.variable]).reshape(-1, 1)
    
    # === 约束特征(C) ===
    # 分别处理上下界约束
    cons_rows = []  # 约束矩阵行
    cons_rhs = []  # 约束右端项
    cons_idx = 0  # 约束索引
    
    for cons in self.constraint:
      row = np.zeros(n_vars)
      for var_idx, coef in zip(cons.var_index, cons.coefficient):
        row[var_idx] = coef
        
      # 处理下界约束
      if cons.lower_bound > -np.inf:
        cons_rows.append(-row)  # 转换为标准形式 Ax >= b
        cons_rhs.append(-cons.lower_bound)
        cons_idx += 1
        
      # 处理上界约束
      if cons.upper_bound < np.inf:
        cons_rows.append(row)  # 标准形式 Ax <= b
        cons_rhs.append(cons.upper_bound)
        cons_idx += 1
    
    if cons_rows:
      cons_matrix = np.array(cons_rows)
      cons_rhs = np.array(cons_rhs)
      
      # 计算约束范数
      cons_norms = np.linalg.norm(cons_matrix, axis=1)
      cons_norms[cons_norms == 0] = 1
      
      # 计算与目标函数的余弦相似度
      obj_vec = obj_coeffs.reshape(-1)
      cos_sims = np.zeros(len(cons_matrix))
      for i, cons_vec in enumerate(cons_matrix):
        cos_sims[i] = np.dot(cons_vec, obj_vec) / (cons_norms[i] * (np.linalg.norm(obj_vec) + 1e-10))
      
      # 合并约束特征
      features['C']['obj_cos_sim'] = cos_sims.reshape(-1, 1)
      features['C']['bias'] = (cons_rhs / cons_norms).reshape(-1, 1)
    else:
      features['C']['obj_cos_sim'] = np.zeros((0, 1))
      features['C']['bias'] = np.zeros((0, 1))
    
    # === 边特征(E) ===
    edge_indices = []
    edge_coefs = []
    cons_idx = 0
    
    for cons in self.constraint:
      # 获取约束范数
      row = np.zeros(n_vars)
      for var_idx, coef in zip(cons.var_index, cons.coefficient):
        row[var_idx] = coef
      cons_norm = np.linalg.norm(row)
      if cons_norm == 0:
        cons_norm = 1
      
      # 处理下界约束
      if cons.lower_bound > -np.inf:
        for var_idx, coef in zip(cons.var_index, cons.coefficient):
          edge_indices.append([cons_idx, var_idx])
          edge_coefs.append(-coef / cons_norm)  # 转换为标准形式
        cons_idx += 1
      
      # 处理上界约束
      if cons.upper_bound < np.inf:
        for var_idx, coef in zip(cons.var_index, cons.coefficient):
          edge_indices.append([cons_idx, var_idx])
          edge_coefs.append(coef / cons_norm)
        cons_idx += 1
    
    if edge_indices:
      features['E']['indices'] = np.array(edge_indices)
      features['E']['coef'] = np.array(edge_coefs).reshape(-1, 1)
    else:
      features['E']['indices'] = np.zeros((0, 2), dtype=np.int32)
      features['E']['coef'] = np.zeros((0, 1))
    
    return features

  def extract_dynamic_features(
      self, 
      static_features: Dict[str, Dict[str, np.ndarray]],
      solution: np.ndarray,
      basis_status: np.ndarray,
      reduced_costs: np.ndarray,
      is_tight: np.ndarray,
      dual_values: np.ndarray,
      age: int,
      incumbent_values: Optional[np.ndarray] = None
  ) -> Dict[str, Dict[str, np.ndarray]]:
    """提取动态特征(需要求解信息的特征)
    Args:
        static_features: 静态特征字典
        solution: 变量的当前解值, shape=(n_vars,)
        basis_status: 变量的基状态(LOWER=0, BASIC=1, UPPER=2, ZERO=3), shape=(n_vars,)
        reduced_costs: 变量的简化成本, shape=(n_vars,)
        is_tight: 约束是否紧的, shape=(n_cons_total,)
        dual_values: 约束的对偶值, shape=(n_cons,)
        age: LP迭代次数
        incumbent_values: 当前最优解中的变量值, shape=(n_vars,), 可选
        
    Returns:
        features: {
            'V': {
                'sol_is_at_lb': shape=(n_vars, 1),    # 解是否在下界
                'sol_is_at_ub': shape=(n_vars, 1),    # 解是否在上界
                'sol_frac': shape=(n_vars, 1),        # 解的分数部分
                'basis_status': shape=(n_vars, 4),    # 基状态的one-hot编码
                'reduced_cost': shape=(n_vars, 1),    # 归一化的简化成本
                'age': shape=(n_vars, 1),             # 归一化的LP年龄
                'sol_val': shape=(n_vars, 1),         # 解值
            },
            'C': {
                'is_tight': shape=(n_cons_total, 1),  # 约束是否紧的
                'dualsol_val': shape=(n_cons, 1),     # 归一化的对偶值
                'age': shape=(n_cons_total, 1),       # 归一化的LP年龄
            }
        }
    """
    features = static_features.copy()
    n_vars = len(self.variable)
    
    # === 变量特征(V) ===
    # 解在边界的指示器
    lb_array = np.array([var.lower_bound for var in self.variable])
    ub_array = np.array([var.upper_bound for var in self.variable])
    
    features['V']['sol_is_at_lb'] = np.isclose(
        solution,
        lb_array,
        rtol=1e-6, atol=1e-6
    ).reshape(-1, 1)
    
    features['V']['sol_is_at_ub'] = np.isclose(
        solution,
        ub_array,
        rtol=1e-6, atol=1e-6
    ).reshape(-1, 1)
    
    # 解的分数部分(对整数变量)
    integer_mask = np.array([var.is_integer for var in self.variable])
    sol_frac = np.abs(solution - np.round(solution)) * integer_mask
    features['V']['sol_frac'] = sol_frac.reshape(-1, 1)
    
    # 基状态的one-hot编码
    basis_onehot = np.zeros((n_vars, 4))  # LOWER BASIC UPPER ZERO
    basis_onehot[np.arange(n_vars), basis_status] = 1
    features['V']['basis_status'] = basis_onehot
    
    # 简化成本归一化
    obj_coeffs = np.array([var.objective_coefficient for var in self.variable])
    obj_norm = np.linalg.norm(obj_coeffs)
    if obj_norm > 0:
      features['V']['reduced_cost'] = (reduced_costs / obj_norm).reshape(-1, 1)
    else:
      features['V']['reduced_cost'] = reduced_costs.reshape(-1, 1)
    
    # LP年龄归一化
    features['V']['age'] = np.full((n_vars, 1), age / (age + 5))
    
    # 解值
    features['V']['sol_val'] = solution.reshape(-1, 1)
    
    # === 约束特征(C) ===
    if 'obj_cos_sim' in features['C']:
        n_cons_total = len(features['C']['obj_cos_sim'])
        
        # 紧致性指标
        features['C']['is_tight'] = is_tight.reshape(-1, 1)
        
        # 对偶值归一化
        if len(dual_values) > 0:
            cons_matrix = []
            for cons in self.constraint:
                row = np.zeros(n_vars)
                for var_idx, coef in zip(cons.var_index, cons.coefficient):
                    row[var_idx] = coef
                cons_matrix.append(row)
            cons_matrix = np.array(cons_matrix)
            cons_norms = np.linalg.norm(cons_matrix, axis=1)
            cons_norms[cons_norms == 0] = 1
            
            dual_values = dual_values / (cons_norms * (np.linalg.norm(obj_coeffs) + 1e-10))
            features['C']['dualsol_val'] = dual_values.reshape(-1, 1)
        
        # LP年龄归一化
        features['C']['age'] = np.full((n_cons_total, 1), age / (age + 5))
    
    return features


@dataclasses.dataclass
class MPSolutionResponse:
  """Class for solution response from the solver."""
  # Objective value corresponding to the "variable_value" below, taking into
  # account the source "objective_offset" and "objective_coefficient".
  objective_value: float
  # Variable values in the same order as the MPModel.variable field.
  # This is a dense representation. These are set iff 'status' is OPTIMAL or
  # FEASIBLE.
  variable_value: List[float]
  # Human-readable status string.
  status_str: str
  # Result of the optimization.
  status: MPSolverResponseStatus = MPSolverResponseStatus.UNKNOWN


def tighten_variable_bounds(mip: Any,
                            names: List[str],
                            lbs: List[float],
                            ubs: List[float]):
  """Tightens variables of the given MIP in-place.

  Args:
    mip: Input MIP.
    names: List of variable names to tighten.
    lbs: List of lower bounds, in same order as names.
    ubs: List of lower bounds, in same order as names.
  """
  if len(names) != len(lbs) or len(lbs) != len(ubs):
    raise ValueError(
        "Names, lower and upper bounds should have the same length")

  name_to_bounds = {}
  for name, lb, ub in zip(names, lbs, ubs):
    name = name.decode() if isinstance(name, bytes) else name
    name_to_bounds[name] = (lb, ub)

  c = 0
  for v in mip.variable:
    name = v.name.decode() if isinstance(v.name, bytes) else v.name
    if name in name_to_bounds:
      lb, ub = name_to_bounds[name]
      v.lower_bound = max(lb, v.lower_bound)
      v.upper_bound = min(ub, v.upper_bound)
      c += 1

  logging.info("Tightened %s vars", c)


def is_var_binary(variable: Any) -> bool:
  """Checks whether a given variable is binary."""
  lb_is_zero = np.isclose(variable.lower_bound, 0)
  ub_is_one = np.isclose(variable.upper_bound, 1)
  return variable.is_integer and lb_is_zero and ub_is_one


def add_binary_invalid_cut(mip: Any,
                           names: List[str],
                           values: List[int],
                           weights: List[float],
                           depth: float):
  """Adds a weighted binary invalid cut to the given MIP in-place.

  Given a binary assignment for all or some of the binary variables, adds
  a constraint in the form:

  sum_{i in zeros} w_i * x_i + sum_{j in ones} w_j * (1-x_j) <= d

  The first summation is over variables predicted to be zeros, the second
  summation is over variables predicted to be ones. d is the maximum distance
  allowed for a solution to be away from predicted assignment.

  Args:
    mip: Input MIP.
    names: Binary variable names.
    values: Predicted values of binary variables.
    weights: Weights associated with cost inccured by reversing prediction.
    depth: The amount of cost allowed to be incurred by flipping
      assignments.
  """
  assert len(names) == len(values) == len(weights)

  name_to_idx = {}
  for i, v in enumerate(mip.variable):
    name = v.name.decode() if isinstance(v.name, bytes) else v.name
    name_to_idx[name] = i

  ub = depth
  var_index = []
  coeffs = []

  for name, val, w in zip(names, values, weights):
    name = name.decode() if isinstance(name, bytes) else name
    assert is_var_binary(mip.variable[name_to_idx[name]])
    var_index.append(name_to_idx[name])

    if val == 1:
      ub -= w
      coeffs.append(-w)
    else:
      coeffs.append(w)

  constraint = mip.constraint.add()
  constraint.var_index.extend(var_index)
  constraint.coefficient.extend(coeffs)
  constraint.upper_bound = ub
  constraint.name = "weighted_invalid_cut"


# def make_sub_mip(mip: Any, assignment: sampling.Assignment):
#   """Creates a sub-MIP by tightening variables and applying cut."""
#   sub_mip = copy.deepcopy(mip)
#   tighten_variable_bounds(sub_mip, assignment.names,
#                           assignment.lower_bounds, assignment.upper_bounds)
#   return sub_mip
