from mip import Model
import math
import numpy as np
# 读取MPS文件
# # 提取模型信息
# variables = model.vars  # 获取所有变量
# constraints = model.constrs  # 获取所有约束

from mip_utils import MPModel, MPVariable, MPConstraint, MPSolverResponseStatus
from mip import Model as mip_model, MAXIMIZE
from solving_utils import SCIPSolver, SolverState
from preprocessor import SCIPPreprocessor
import ml_collections
from typing import Dict, Any, Optional

SCIP_FEATURE_EXTRACTION_PARAMS = ml_collections.ConfigDict({
    'seed': 42,
    'time_limit_seconds': 60 * 10,
    'separating_maxroundsroot': 0,   # No cuts
    'conflict_enable': False,        # No additional cuts
    'heuristics_emphasis': 'off',    # No heuristics
})

def convert_mip_to_mpmodel(mip_model: mip_model) -> MPModel:
    mp_model = MPModel()
    var_index_map = {var: idx for idx, var in enumerate(mip_model.vars)}

    # 处理变量
    for var in mip_model.vars:
        mp_var = MPVariable(
            lower_bound=var.lb,
            upper_bound=var.ub,
            is_integer=var.var_type in ('I', 'B'),
            name=var.name
        )
        mp_model.variable.append(mp_var)

    # 处理目标函数（字典结构）
    obj_expr = mip_model.objective.expr  # 获取字典 {Var: coeff}
    for var, coeff in obj_expr.items():
        idx = var_index_map[var]
        mp_model.variable[idx].objective_coefficient = coeff
    
    # 处理目标偏移量（需要从模型中单独获取）
    mp_model.objective_offset = mip_model.objective_const
    mp_model.maximize = (mip_model.sense == MAXIMIZE)

    # 处理约束（需考虑表达式结构）
    for con in mip_model.constrs:
        expr = con.expr._LinExpr__expr  # 访问私有属性
        
        # 分解变量和系数
        if isinstance(expr, dict):  # 字典形式
            var_indices = [var_index_map[var] for var in expr.keys()]
            coefficients = list(expr.values())
            const_term = con.expr._LinExpr__const  # 假设字典不包含常数项

        # 调整约束边界
        adjusted_rhs = con.rhs - const_term
        if con.expr._LinExpr__sense == '<=' or con.expr._LinExpr__sense == '<':
            lower, upper = -math.inf, adjusted_rhs
        elif con.expr._LinExpr__sense == '>=' or con.expr._LinExpr__sense == '>':
            lower, upper = adjusted_rhs, math.inf
        elif con.expr._LinExpr__sense == '==' or con.expr._LinExpr__sense == '=':
            lower = upper = adjusted_rhs
        
        else:
            raise ValueError(f"未知约束方向: {con.expr._LinExpr__sense}")

        mp_model.constraint.append(
            MPConstraint(
                var_index=var_indices,
                coefficient=coefficients,
                lower_bound=lower,
                upper_bound=upper,
                name=con.name
            )
        )

    mp_model.name = mip_model.name
    return mp_model

def get_features(
    mip: MPModel,
    solver_params: ml_collections.ConfigDict = SCIP_FEATURE_EXTRACTION_PARAMS
    ) -> Optional[Dict[str, Any]]:
  """Extracts and preprocesses the features from the root of B&B tree."""
  mip_solver = SCIPSolver()
  status = mip_solver.load_model(mip)
  
  # 只在模型未被求解时提取特征
  features = None
  if status == MPSolverResponseStatus.NOT_SOLVED:
    features = mip_solver.extract_lp_features_at_root(solver_params)
    features['model_maximize'] = mip.maximize
    
  return features

if __name__ == "__main__":
    mps_file_path = "./data/air05.mps"  # Replace with your MPS file path
    print(f"尝试读取MPS文件: {mps_file_path}")
    
    model = mip_model()
    model.read(mps_file_path)
    print("MPS文件读取成功")
    
    mp_model = convert_mip_to_mpmodel(model)
    print("模型转换完成")
    
    print("\n开始提取特征...")
    features = get_features(mp_model)
    if features is not None:
        print("\n特征提取成功！")
        
        # 打印features字典的键
        print("\n可用特征键:")
        for key in features.keys():
            print(f"- {key}")
            
        # 变量特征 (V)
        print("\n变量特征 (V):")
        if 'V' in features:
            v_features = features['V']
            if isinstance(v_features, dict):
                for sub_key, sub_value in v_features.items():
                    if isinstance(sub_value, np.ndarray):
                        print(f"- {sub_key}: 形状={sub_value.shape}")
                        print(f"  范围: [{sub_value.min():.4f}, {sub_value.max():.4f}]")
                    else:
                        print(f"- {sub_key}: {sub_value}")
            else:
                print(f"V特征值: {v_features}")
        else:
            print("未找到V特征")
            
        # 约束特征 (C)
        print("\n约束特征 (C):")
        if 'C' in features:
            c_features = features['C']
            if isinstance(c_features, dict):
                for sub_key, sub_value in c_features.items():
                    if isinstance(sub_value, np.ndarray):
                        print(f"- {sub_key}: 形状={sub_value.shape}")
                        print(f"  范围: [{sub_value.min():.4f}, {sub_value.max():.4f}]")
                    else:
                        print(f"- {sub_key}: {sub_value}")
            else:
                print(f"C特征值: {c_features}")
        else:
            print("未找到C特征")
                
        # 边特征 (E)
        print("\n边特征 (E):")
        if 'E' in features:
            e_features = features['E']
            if isinstance(e_features, dict):
                for sub_key, sub_value in e_features.items():
                    if isinstance(sub_value, np.ndarray):
                        print(f"- {sub_key}: 形状={sub_value.shape}")
                        print(f"  范围: [{sub_value.min():.4f}, {sub_value.max():.4f}]")
                        if sub_key == 'indices':
                            print(f"  前3行: {sub_value[:3]}")
                    else:
                        print(f"- {sub_key}: {sub_value}")
            else:
                print(f"E特征值: {e_features}")
        else:
            print("未找到E特征")
                
        # 其他特征
        print("\n其他特征:")
        print(f"- 目标函数方向: {'最大化' if features['model_maximize'] else '最小化'}")
    else:
        print("\n特征提取失败") 