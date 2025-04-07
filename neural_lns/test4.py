from mip import Model
import math
import numpy as np
import pandas as pd  # 导入pandas用于Excel操作
import os  # 导入os模块用于处理文件路径
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

def export_features_to_excel(features, output_file):
    """将特征导出到Excel文件
    Args:
        features: 特征字典
        output_file: 输出Excel文件路径
    """
    # 创建输出目录
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 定义必需的特征
    required_features = {
        'V': ['type', 'coef', 'has_lb', 'has_ub', 'sol_is_at_lb', 'sol_is_at_ub',
              'sol_frac', 'basis_status', 'reduced_cost', 'age', 'sol_val'],
        'C': ['obj_cos_sim', 'bias', 'is_tight', 'dualsol_val', 'age'],
        'E': ['coef', 'indices']
    }
    
    # 检查缺失的特征
    missing_features = {}
    for tensor, required in required_features.items():
        if tensor not in features:
            missing_features[tensor] = required
            continue
        missing = []
        for feat in required:
            if feat not in features[tensor]:
                missing.append(feat)
        if missing:
            missing_features[tensor] = missing
    
    if missing_features:
        print("\n警告: 以下特征缺失:")
        for tensor, missing in missing_features.items():
            print(f"{tensor}: {', '.join(missing)}")
    
    # 准备特征摘要数据
    summary_data = []
    
    with pd.ExcelWriter(output_file) as writer:
        # 导出变量特征(V)
        if 'V' in features and isinstance(features['V'], dict):
            v_data = {}
            for feat_name in required_features['V']:
                if feat_name in features['V']:
                    values = features['V'][feat_name]
                    if feat_name == 'type':
                        # 对type特征进行特殊处理
                        type_names = ['binary', 'integer', 'impl_integer', 'continuous']
                        for i, name in enumerate(type_names):
                            v_data[f'type_{name}'] = values[:, i]
                    elif feat_name == 'basis_status':
                        # 对basis_status特征进行特殊处理
                        status_names = ['LOWER', 'BASIC', 'UPPER', 'ZERO']
                        for i, name in enumerate(status_names):
                            v_data[f'basis_status_{name}'] = values[:, i]
                    else:
                        v_data[feat_name] = values.flatten()
                    
                    # 添加到摘要
                    if values.size > 0:
                        summary_data.append({
                            '特征类别': '变量特征(V)',
                            '特征名称': feat_name,
                            '形状': str(values.shape),
                            '最小值': values.min(),
                            '最大值': values.max(),
                            '平均值': values.mean(),
                            '标准差': values.std(),
                            '是否缺失': feat_name in (missing_features.get('V', []))
                        })
            
            if v_data:
                v_df = pd.DataFrame(v_data)
                v_df.to_excel(writer, sheet_name='变量特征', index=True)
        
        # 导出约束特征(C)
        if 'C' in features and isinstance(features['C'], dict):
            c_data = {}
            # 获取约束特征的长度
            c_lens = [v.shape[0] for v in features['C'].values() if isinstance(v, np.ndarray)]
            if c_lens:
                max_len = max(c_lens)
                for feat_name in required_features['C']:
                    if feat_name in features['C']:
                        values = features['C'][feat_name]
                        if values.shape[0] < max_len:
                            # 如果特征长度不足,用0填充
                            pad_width = ((0, max_len - values.shape[0]), (0, 0))
                            values = np.pad(values, pad_width, mode='constant')
                        c_data[feat_name] = values.flatten()
                        
                        # 添加到摘要
                        if values.size > 0:
                            summary_data.append({
                                '特征类别': '约束特征(C)',
                                '特征名称': feat_name,
                                '形状': str(values.shape),
                                '最小值': values.min(),
                                '最大值': values.max(),
                                '平均值': values.mean(),
                                '标准差': values.std(),
                                '是否缺失': feat_name in (missing_features.get('C', []))
                            })
            
            if c_data:
                c_df = pd.DataFrame(c_data)
                c_df.to_excel(writer, sheet_name='约束特征', index=True)
        
        # 导出边特征(E)
        if 'E' in features and isinstance(features['E'], dict):
            e_data = {}
            if 'indices' in features['E'] and 'coef' in features['E']:
                indices = features['E']['indices']
                coefs = features['E']['coef']
                
                if indices.size > 0 and coefs.size > 0:
                    e_data = {
                        'cons_idx': indices[:, 0],
                        'var_idx': indices[:, 1],
                        'coef': coefs.flatten()
                    }
                    
                    # 添加到摘要
                    summary_data.append({
                        '特征类别': '边特征(E)',
                        '特征名称': 'indices',
                        '形状': str(indices.shape),
                        '最小值': indices.min(),
                        '最大值': indices.max(),
                        '平均值': indices.mean(),
                        '标准差': indices.std(),
                        '是否缺失': 'indices' in (missing_features.get('E', []))
                    })
                    
                    summary_data.append({
                        '特征类别': '边特征(E)',
                        '特征名称': 'coef',
                        '形状': str(coefs.shape),
                        '最小值': coefs.min(),
                        '最大值': coefs.max(),
                        '平均值': coefs.mean(),
                        '标准差': coefs.std(),
                        '是否缺失': 'coef' in (missing_features.get('E', []))
                    })
            
            if e_data:
                e_df = pd.DataFrame(e_data)
                e_df.to_excel(writer, sheet_name='边特征', index=True)
        
        # 导出特征摘要
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='特征摘要', index=False)
        
        # 导出其他信息
        other_data = {
            '目标函数方向': '最大化' if features.get('model_maximize', False) else '最小化'
        }
        other_df = pd.DataFrame([other_data])
        other_df.to_excel(writer, sheet_name='其他信息', index=False)
    
    print(f"\n特征已成功导出到: {output_file}")
    if missing_features:
        print("注意: 部分特征缺失,请查看特征摘要sheet了解详情")

if __name__ == "__main__":
    import argparse
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='提取MIP模型特征并导出到Excel')
    parser.add_argument('--mps', type=str, default="./data/air05.mps", help='MPS文件路径')
    parser.add_argument('--output', type=str, default=None, help='Excel输出文件路径')
    
    args = parser.parse_args()
    mps_file_path = args.mps
    
    # 如果未指定输出路径，则根据输入文件名生成
    if args.output is None:
        model_name = os.path.splitext(os.path.basename(mps_file_path))[0]
        output_file = f"./output/{model_name}_features.xlsx"
    else:
        output_file = args.output
    
    print(f"尝试读取MPS文件: {mps_file_path}")
    
    try:
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
            
            # 导出特征到Excel
            export_features_to_excel(features, output_file)
            
        else:
            print("\n特征提取失败")
    
    except Exception as e:
        print(f"错误: {str(e)}") 