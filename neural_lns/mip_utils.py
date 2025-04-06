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
  # All the variables appearing in the model.
  variable: List[MPVariable] = dataclasses.field(default_factory=list)
  # All the constraints appearing in the model.
  constraint: List[MPConstraint] = dataclasses.field(default_factory=list)
  # True if the problem is a maximization problem. Minimize by default.
  maximize: bool = False
  # Offset for the objective function. Must be finite.
  objective_offset: float = 0.0
  # Name of the model.
  name: str = ""
  # 历史incumbent值列表,每个元素是一个numpy数组,shape=(n_vars,)
  historical_incumbents: List[np.ndarray] = dataclasses.field(default_factory=list)

  def add_incumbent(self, incumbent_values: np.ndarray) -> None:
    """添加新的incumbent值到历史记录中
    
    Args:
        incumbent_values: shape=(n_vars,) 当前incumbent解
    """
    if len(incumbent_values) != len(self.variable):
      raise ValueError(f"Incumbent values length {len(incumbent_values)} "
                      f"does not match number of variables {len(self.variable)}")
    self.historical_incumbents.append(incumbent_values.copy())

  def _normalize_vector(self, vec):
    """更稳定的向量归一化方法
    
    Args:
        vec: 需要归一化的向量
        
    Returns:
        归一化后的向量
    """
    # 处理零向量
    if np.all(vec == 0):
        return vec
        
    # 计算范数
    norm = np.linalg.norm(vec)
    
    # 处理极小值
    if norm < 1e-10:
        return vec
        
    # 归一化并添加小量防止除零
    normalized = vec / (norm + 1e-10)
    
    # 处理异常值
    normalized = np.clip(normalized, -1e6, 1e6)
    
    return normalized

  def _get_constraint_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
    """获取约束矩阵和范数
    Returns:
        cons_matrix: shape=(n_cons, n_vars)
        cons_norms: shape=(n_cons,)
    """
    n_cons = len(self.constraint)
    n_vars = len(self.variable)
    cons_matrix = np.zeros((n_cons, n_vars))
    
    for i, cons in enumerate(self.constraint):
      for var_idx, coeff in zip(cons.var_index, cons.coefficient):
        cons_matrix[i, var_idx] = coeff
        
    cons_norms = np.linalg.norm(cons_matrix, axis=1)
    cons_norms[cons_norms == 0] = 1
    
    return cons_matrix, cons_norms

  def check_feature_completeness(self, features: Dict[str, Dict[str, np.ndarray]]) -> bool:
    """检查特征是否完整
    
    Args:
        features: 特征字典
        
    Returns:
        bool: 特征是否完整
    """
    # 检查必需的特征键
    required_keys = ['V', 'C', 'E', 'model_maximize']
    for key in required_keys:
        if key not in features:
            print(f"警告：缺少必需的特征键 '{key}'")
            return False
            
    # 检查变量特征
    if not isinstance(features['V'], dict):
        print("警告：变量特征格式不正确")
        return False
        
    # 检查约束特征
    if not isinstance(features['C'], dict):
        print("警告：约束特征格式不正确")
        return False
        
    # 检查边特征
    if not isinstance(features['E'], dict):
        print("警告：边特征格式不正确")
        return False
        
    # 检查边特征的必需字段
    if 'names' not in features['E'] or 'indices' not in features['E']:
        print("警告：边特征缺少必需字段")
        return False
        
    # 检查特征维度
    try:
        n_vars = len(self.variable)
        n_cons = len(self.constraint)
        
        # 检查变量特征维度
        for key, value in features['V'].items():
            if isinstance(value, np.ndarray) and value.shape[0] != n_vars:
                print(f"警告：变量特征 '{key}' 维度不匹配")
                return False
                
        # 检查约束特征维度
        for key, value in features['C'].items():
            if isinstance(value, np.ndarray) and value.shape[0] != n_cons:
                print(f"警告：约束特征 '{key}' 维度不匹配")
                return False
                
    except Exception as e:
        print(f"警告：特征维度检查出错: {str(e)}")
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
                'indices': shape=(n_edges, 3),     # [edge_idx, cons_idx, var_idx]
            }
        }
    """
    features = {'V': {}, 'C': {}, 'E': {}}
    n_vars = len(self.variable)
    
    # === 变量特征(V) ===
    # 变量类型的one-hot编码
    var_types = np.zeros((n_vars, 4))  # [binary, integer, impl_integer, continuous]
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
    features['V']['coef'] = self._normalize_vector(obj_coeffs).reshape(-1, 1)
    
    # 边界指示器
    features['V']['has_lb'] = np.array([var.lower_bound > -np.inf for var in self.variable]).reshape(-1, 1)
    features['V']['has_ub'] = np.array([var.upper_bound < np.inf for var in self.variable]).reshape(-1, 1)
    
    # === 约束特征(C) ===
    # 获取约束矩阵
    cons_matrix, cons_norms = self._get_constraint_matrix()
    
    # 分离左右约束
    has_lhs = []  # 有下界的约束
    has_rhs = []  # 有上界的约束
    for i, cons in enumerate(self.constraint):
      if cons.lower_bound > -np.inf:
        has_lhs.append(i)
      if cons.upper_bound < np.inf:
        has_rhs.append(i)
    has_lhs = np.array(has_lhs)
    has_rhs = np.array(has_rhs)
    
    # 计算与目标函数的余弦相似度
    obj_vec = obj_coeffs.reshape(-1)
    cos_sims = np.zeros(len(self.constraint))
    for i, cons_vec in enumerate(cons_matrix):
      cos_sims[i] = np.dot(cons_vec, obj_vec) / (cons_norms[i] * np.linalg.norm(obj_vec) + 1e-10)
    
    # 合并左右约束的余弦相似度(添加正负号)
    if len(has_lhs) > 0 or len(has_rhs) > 0:
      cos_sims_combined = np.concatenate([
        -cos_sims[has_lhs],
        +cos_sims[has_rhs]
      ]).reshape(-1, 1)
      features['C']['obj_cos_sim'] = cos_sims_combined
    
    # 约束偏置(bias)归一化
    if len(has_lhs) > 0 or len(has_rhs) > 0:
      bias = np.concatenate([
        -np.array([self.constraint[i].lower_bound for i in has_lhs]),
        +np.array([self.constraint[i].upper_bound for i in has_rhs])
      ])
      # 对偏置进行归一化
      bias = self._normalize_vector(bias)
      features['C']['bias'] = bias.reshape(-1, 1)
    
    # === 边特征(E) ===
    # 约束系数归一化
    rows = []  # 约束索引
    cols = []  # 变量索引
    data = []  # 归一化的系数值
    for i, cons in enumerate(self.constraint):
      for var_idx, coeff in zip(cons.var_index, cons.coefficient):
        if i in has_lhs:
          rows.append(i)
          cols.append(var_idx)
          data.append(-coeff / cons_norms[i])
        if i in has_rhs:
          rows.append(i)
          cols.append(var_idx)
          data.append(+coeff / cons_norms[i])
    
    if data:
      # 创建稀疏矩阵
      edge_matrix = sparse.coo_matrix((data, (rows, cols)))
      # 提取所需格式的特征
      features['E']['names'] = edge_matrix.data.reshape(-1, 1).tolist()
      features['E']['indices'] = np.vstack([edge_matrix.row, edge_matrix.col]).T
    
    # 检查特征完整性
    if not self.check_feature_completeness(features):
        print("警告：特征提取不完整")
        
    # 添加模型类型特征
    features['model_maximize'] = self.maximize
        
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
                'inc_val': shape=(n_vars, 1),         # incumbent值(可选)
                'avg_inc_val': shape=(n_vars, 1),     # 平均incumbent值(可选)
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
    features['V']['reduced_cost'] = self._normalize_vector(reduced_costs).reshape(-1, 1)
    
    # 获取每个变量的年龄
    var_ages = np.array([var.age for var in self.variable])
    # LP年龄归一化 - 对每个变量使用其实际年龄
    features['V']['age'] = (var_ages / (var_ages + 5)).reshape(-1, 1)
    
    # 解值
    features['V']['sol_val'] = solution.reshape(-1, 1)
    
    # incumbent相关特征(如果提供)
    if incumbent_values is not None:
      features['V']['inc_val'] = incumbent_values.reshape(-1, 1)
      # 计算历史incumbent的平均值
      if len(self.historical_incumbents) > 0:
        # 添加当前incumbent到历史记录
        self.add_incumbent(incumbent_values)
        # 计算所有历史incumbent的平均值
        avg_incumbent = np.mean(self.historical_incumbents, axis=0)
        features['V']['avg_inc_val'] = avg_incumbent.reshape(-1, 1)
      else:
        # 如果没有历史记录,使用当前incumbent值
        features['V']['avg_inc_val'] = incumbent_values.reshape(-1, 1)
        self.add_incumbent(incumbent_values)
    
    # === 约束特征(C) ===
    if 'obj_cos_sim' in features['C']:
      n_cons_total = len(features['C']['obj_cos_sim'])
      
      # 紧致性指标
      features['C']['is_tight'] = is_tight.reshape(-1, 1)
      
      # 对偶值归一化
      if len(dual_values) > 0:
        cons_matrix, cons_norms = self._get_constraint_matrix()
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
