from mip import Model
import math
# 读取MPS文件
# # 提取模型信息
# variables = model.vars  # 获取所有变量
# constraints = model.constrs  # 获取所有约束

#from data_utils import get_features
from neural_lns.mip_utils import MPModel, MPVariable, MPConstraint, MPSolverResponseStatus
from mip import Model as mip_model, MAXIMIZE
# from neural_lns.data_utils import get_features
from neural_lns.solving_utils import SCIPSolver
from neural_lns.preprocessor import SCIPPreprocessor
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
# Example usage

def get_features(
    mip: MPModel,
    solver_params: ml_collections.ConfigDict = SCIP_FEATURE_EXTRACTION_PARAMS
    ) -> Optional[Dict[str, Any]]:
  """Extracts and preprocesses the features from the root of B&B tree."""
  mip_solver = SCIPSolver()
#   presolver = SCIPPreprocessor()
#   _, mip = presolver.presolve(mip)
  status = mip_solver.load_model(mip)
#   features = None
#   if status == MPSolverResponseStatus.NOT_SOLVED:
  features = mip_solver.extract_lp_features_at_root(solver_params)

#   if features is not None and mip is not None:
  features['model_maximize'] = mip.maximize

  return features

if __name__ == "__main__":
    mps_file_path = "./neural_lns/data/assign1-5-8.mps"  # Replace with your MPS file path
    model = mip_model()
    model.read(mps_file_path)
    mp_model = convert_mip_to_mpmodel(model)
    features = get_features(mp_model)
    print(features)

    # # Print the converted model
    # print("Model Name:", mp_model.name)
    # print("Maximize:", mp_model.maximize)
    # print("Objective Offset:", mp_model.objective_offset)
    # print("\nVariables:")
    # for var in mp_model.variable:
    #     print(f"  {var.name}: [{var.lower_bound}, {var.upper_bound}], Type: {var.is_integer}")
    # print("\nConstraints:")
    # for constr in mp_model.constraint:
    #     print(f"  {constr.name}: [{constr.lower_bound}, {constr.upper_bound}]")
    #     print("    Coefficients:", constr.coefficient)


