import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from pyscipopt import Model
from neural_lns import data_utils
from neural_lns.mip_utils import MPModel, MPVariable, MPConstraint, MPSolverResponseStatus, MPSolutionResponse
from neural_lns.solving_utils import SCIPSolver, SolverState
from neural_lns.preprocessor import SCIPPreprocessor
import ml_collections

# 特征序列化辅助函数
def _bytes_feature(value):
    """将字节值序列化为 TFRecord 特征"""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """将浮点值序列化为 TFRecord 特征"""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    """将整数值序列化为 TFRecord 特征"""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bool_feature(value):
    """将布尔值序列化为 TFRecord 特征"""
    return tf.train.Feature(bool_list=tf.train.BoolList(value=[value]))

def _bytes_list_feature(value):
    """将字节列表序列化为 TFRecord 特征"""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_list_feature(value):
    """将浮点列表序列化为 TFRecord 特征"""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_list_feature(value):
    """将整数列表序列化为 TFRecord 特征"""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def parse_mps_file(mps_file):
    """解析MPS文件并返回MPModel对象
    
    注意：为了确保特征提取的准确性，在解析过程中会临时禁用预处理。
    这可能会影响性能，但可以保证获取原始问题的完整结构。
    """
    model = Model()
    
    # 禁用预处理和启发式搜索
    model.setPresolve(False)
    model.setHeuristics(False)
    model.disablePropagation()
    
    # 禁用约束升级
    model.setParam('constraints/linear/upgrade/logicor', False)
    model.setParam('constraints/linear/upgrade/indicator', False)
    model.setParam('constraints/linear/upgrade/knapsack', False)
    model.setParam('constraints/linear/upgrade/setppc', False)
    model.setParam('constraints/linear/upgrade/xor', False)
    model.setParam('constraints/linear/upgrade/varbound', False)
    
    model.readProblem(mps_file)
    print(f"Original problem has {model.getNVars()} variables (all binary) and {model.getNConss()} constraints")
    
    mp_model = MPModel()
    
    # Add variables
    var_name_to_idx = {}  # 添加变量名到索引的映射
    for i, var in enumerate(model.getVars()):
        mp_var = MPVariable()
        mp_var.name = var.name
        mp_var.lower_bound = var.getLbLocal()
        mp_var.upper_bound = var.getUbLocal()
        mp_var.is_integer = var.vtype() in ('BINARY', 'INTEGER')
        mp_model.variable.append(mp_var)
        var_name_to_idx[var.name] = i  # 记录变量名到索引的映射
    
    # Add constraints
    for con in model.getConss():
        if con.isLinear():
            mip_constraint = MPConstraint()
            mip_constraint.name = con.name
            
            try:
                # Get coefficients for each variable
                coeff_dict = model.getValsLinear(con)
                if coeff_dict is None:
                    print(f"Warning: Could not get coefficients for constraint {con.name}")
                    continue
                
                # Add variables and coefficients
                for var_name, coef in coeff_dict.items():
                    if abs(coef) > 1e-10:  # Only add non-zero coefficients
                        if var_name in var_name_to_idx:
                            mip_constraint.var_index.append(var_name_to_idx[var_name])
                            mip_constraint.coefficient.append(coef)
                        else:
                            print(f"Warning: Variable {var_name} not found in mapping")
                
                # Get bounds
                lhs = model.getLhs(con)
                rhs = model.getRhs(con)
                if lhs is not None and lhs > -model.infinity():
                    mip_constraint.lower_bound = lhs
                if rhs is not None and rhs < model.infinity():
                    mip_constraint.upper_bound = rhs
                
                mp_model.constraint.append(mip_constraint)
            except Exception as e:
                print(f"Warning: Could not process constraint {con.name}: {str(e)}")
        else:
            print(f"Warning: Constraint {con.name} is not linear")
    
    return mp_model

def extract_features(mip):
    """从MIP模型中提取特征"""
    try:
        # 创建SCIP求解器
        solver = SCIPSolver()
        
        # 加载模型
        solver.load_model(mip)
            
        # 提取特征
        features = solver.extract_lp_features_at_root(
            data_utils.SCIP_FEATURE_EXTRACTION_PARAMS
        )
        if features is None:
            print("特征提取失败")
            return None
            
        # 添加模型类型
        features['model_maximize'] = mip.maximize
        
        # 添加变量边界
        features['variable_lbs'] = np.array(
            [var.lower_bound for var in mip.variable], dtype=np.float32
        )
        features['variable_ubs'] = np.array(
            [var.upper_bound for var in mip.variable], dtype=np.float32
        )
        
        # 添加二进制变量和整数变量索引
        binary_indices = []
        integer_indices = []
        for i, var in enumerate(mip.variable):
            if var.is_integer:
                integer_indices.append(i)
                if var.lower_bound == 0 and var.upper_bound == 1:
                    binary_indices.append(i)
        
        features['binary_variable_indices'] = np.array(binary_indices, dtype=np.int64)
        features['all_integer_variable_indices'] = np.array(integer_indices, dtype=np.int64)
        
        # 添加最优解标签
        solver.solve(data_utils.SCIP_FEATURE_EXTRACTION_PARAMS)
        solution = solver.get_best_solution()
        if solution and solution.status in {
            MPSolverResponseStatus.OPTIMAL,
            MPSolverResponseStatus.FEASIBLE
        }:
            features['best_solution_labels'] = np.array(
                solution.variable_value, dtype=np.float32
            )
        else:
            features['best_solution_labels'] = np.zeros(
                len(mip.variable), dtype=np.float32
            )
        
        print("特征提取成功")
        return features
        
    except Exception as e:
        print(f"特征提取过程中出错: {e}")
        return None

def serialize_features(variable_features, constraint_features, edge_features, edge_indices):
    """序列化特征到TFRecord格式。"""
    print("\n=== 特征形状 ===")
    print("变量特征:")
    for name, feat in variable_features.items():
        print(f"{name}: {feat.shape}")
    
    print("\n约束特征:")
    for name, feat in constraint_features.items():
        print(f"{name}: {feat.shape}")
    
    print("\n边特征:")
    print(f"indices: {edge_indices.shape}")
    print(f"coef: {edge_features.shape}")

    # 连接变量特征
    var_feats = []
    for feat in variable_features.values():
        if len(feat.shape) == 1:
            feat = feat.reshape(-1, 1)
        var_feats.append(feat.astype(np.float32))
    variable_features_concat = np.concatenate(var_feats, axis=1)

    # 连接约束特征
    # 将is_tight和dualsol_val扩展到852维
    is_tight = constraint_features['is_tight']
    dualsol_val = constraint_features['dualsol_val']
    is_tight_expanded = np.repeat(is_tight, 2, axis=0)
    dualsol_val_expanded = np.repeat(dualsol_val, 2, axis=0)

    con_feats = [
        constraint_features['obj_cos_sim'].astype(np.float32),
        constraint_features['bias'].astype(np.float32),
        is_tight_expanded.astype(np.float32),
        dualsol_val_expanded.astype(np.float32),
        constraint_features['age'].astype(np.float32)
    ]
    constraint_features_concat = np.concatenate(con_feats, axis=1)

    # 确保边特征的形状正确
    edge_features = edge_features.reshape(-1, 1).astype(np.float32)
    edge_indices = edge_indices.astype(np.int64)

    print("\n=== 连接后的形状 ===")
    print(f"variable_features: {variable_features_concat.shape}")
    print(f"constraint_features: {constraint_features_concat.shape}")
    print(f"edge_features: {edge_features.shape}")
    print(f"edge_indices: {edge_indices.shape}")

    # 创建特征字典
    feature = {
        'variable_features': _bytes_feature([tf.io.serialize_tensor(variable_features_concat).numpy()]),
        'constraint_features': _bytes_feature([tf.io.serialize_tensor(constraint_features_concat).numpy()]),
        'edge_features': _bytes_feature([tf.io.serialize_tensor(edge_features).numpy()]),
        'edge_indices': _bytes_feature([tf.io.serialize_tensor(edge_indices).numpy()]),
        'binary_variable_indices': _int64_list_feature(list(range(len(variable_features_concat)))),
        'model_maximize': _bool_feature(False),
        'variable_names': _bytes_list_feature([f"x{i}".encode() for i in range(len(variable_features_concat))]),
        'best_solution_labels': _float_list_feature([0.0] * len(variable_features_concat)),
        'variable_lbs': _float_list_feature([0.0] * len(variable_features_concat)),
        'variable_ubs': _float_list_feature([1.0] * len(variable_features_concat)),
        'all_integer_variable_indices': _int64_list_feature(list(range(len(variable_features_concat)))),
        'edge_features_names': _bytes_feature([b"coef"]),
        'variable_feature_names': _bytes_feature([b"features"]),
        'constraint_feature_names': _bytes_feature([b"features"])
    }

    # 创建Example
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()

def mps_to_tfrecord(mps_file_path, tfrecord_file_path):
    """将MPS文件转换为TFRecord文件"""
    try:
        # 1. 解析MPS文件
        mip = parse_mps_file(mps_file_path)
        if mip is None:
            return False
            
        # 2. 提取特征
        features = extract_features(mip)
        if features is None:
            return False
            
        # 3. 序列化特征
        example = serialize_features(features['V'], features['C'], features['E']['coef'], features['E']['indices'])
        if example is None:
            return False
            
        # 4. 写入TFRecord文件
        os.makedirs(os.path.dirname(tfrecord_file_path), exist_ok=True)
        with tf.io.TFRecordWriter(tfrecord_file_path) as writer:
            writer.write(example)
        
        print(f"TFRecord文件已保存至: {tfrecord_file_path}")
        return True
        
    except Exception as e:
        print(f"转换过程中出错: {e}")
        return False

def batch_process_mps_files(mps_dir, output_dir):
    """批量处理MPS文件并转换为TFRecord"""
    import glob
    
    # 获取所有MPS文件
    mps_files = glob.glob(os.path.join(mps_dir, "*.mps"))
    
    if not mps_files:
        print(f"在{mps_dir}中未找到MPS文件")
        return
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 记录转换结果
    success_count = 0
    failed_count = 0
    
    # 处理每个MPS文件
    for i, mps_file in enumerate(mps_files):
        try:
            # 生成输出文件名
            base_name = os.path.basename(mps_file)
            name_without_ext = os.path.splitext(base_name)[0]
            tfrecord_file = os.path.join(output_dir, f"{name_without_ext}.tfrecord")
            
            print(f"处理 [{i+1}/{len(mps_files)}]: {base_name}")
            if mps_to_tfrecord(mps_file, tfrecord_file):
                success_count += 1
            else:
                failed_count += 1
                
        except Exception as e:
            print(f"处理{mps_file}时出错: {e}")
            failed_count += 1
    
    print(f"\n批量处理完成: 成功 {success_count}, 失败 {failed_count}")

def verify_tfrecord(tfrecord_file):
    """验证TFRecord文件是否可以正确读取"""
    try:
        # 创建数据集
        dataset = data_utils.get_dataset(tfrecord_file)
        
        # 尝试读取一个样本
        for data in dataset.take(1):
            print("TFRecord验证成功! 数据结构:")
            print(f"图结构: {data.graphs_tuple}")
            print(f"节点数量: {data.graphs_tuple.n_node}")
            print(f"边数量: {data.graphs_tuple.n_edge}")
            print(f"标签形状: {data.integer_labels.shape}")
            return True
            
    except Exception as e:
        print(f"TFRecord验证失败: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='将MPS文件转换为TFRecord')
    parser.add_argument('--mps_file', help='单个MPS文件的路径')
    parser.add_argument('--mps_dir', help='包含多个MPS文件的目录')
    parser.add_argument('--output_file', help='输出TFRecord文件路径（用于单个文件）')
    parser.add_argument('--output_dir', help='输出TFRecord文件目录（用于批量处理）')
    parser.add_argument('--verify', action='store_true', help='验证生成的TFRecord文件')
    
    args = parser.parse_args()
    
    try:
        # 单个文件处理
        if args.mps_file and args.output_file:
            success = mps_to_tfrecord(args.mps_file, args.output_file)
            if success and args.verify:
                verify_tfrecord(args.output_file)
        
        # 批量处理
        elif args.mps_dir and args.output_dir:
            batch_process_mps_files(args.mps_dir, args.output_dir)
            if args.verify:
                # 验证第一个生成的文件
                import glob
                tfrecord_files = glob.glob(os.path.join(args.output_dir, "*.tfrecord"))
                if tfrecord_files:
                    verify_tfrecord(tfrecord_files[0])
        
        else:
            print("请指定--mps_file和--output_file（单个文件处理）")
            print("或--mps_dir和--output_dir（批量处理）")
            
    except Exception as e:
        print(f"转换过程中发生错误: {e}")