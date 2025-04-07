import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from pyscipopt import Model
from neural_lns import data_utils
from neural_lns.mip_utils import MPModel, MPVariable, MPConstraint, MPSolverResponseStatus
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

def parse_mps_file(mps_file):
    """解析MPS文件并返回MPModel对象
    
    注意：为了确保特征提取的准确性，在解析过程中会临时禁用预处理。
    这可能会影响性能，但可以保证获取原始问题的完整结构。
    """
    model = Model()
    
    try:
        # 临时禁用预处理和启发式搜索，以确保获取原始问题结构
        model.setPresolve(False)  # 禁用预处理
        model.setHeuristics(False)  # 禁用启发式搜索
        
        # 读取MPS文件
        model.readProblem(mps_file)
        print(f"Original problem has {model.getNVars()} variables (all binary) and {model.getNConss()} constraints")
        
        # 获取所有变量和约束
        vars = model.getVars()
        cons = model.getConss()
        
        # 创建MPModel对象
        mp_model = MPModel()
        
        # 设置变量
        for var in vars:
            mp_variable = MPVariable()
            mp_variable.name = var.name
            mp_variable.lb = var.getLbLocal()
            mp_variable.ub = var.getUbLocal()
            mp_variable.type = var.vtype()
            mp_model.variable.append(mp_variable)
        
        # 设置约束
        for con in cons:
            mip_constraint = MPConstraint()
            try:
                row = model.getRowLinear(con)
                if row is not None:
                    cols = row.getCols()
                    vals = row.getVals()
                    for i in range(len(cols)):
                        var_idx = cols[i].getIndex()
                        coef = vals[i]
                        if abs(coef) > 1e-10:  # 只添加非零系数
                            mip_constraint.var_idx.append(var_idx)
                            mip_constraint.coef.append(coef)
                    mip_constraint.lb = row.getLhs()
                    mip_constraint.ub = row.getRhs()
                    mp_model.constraint.append(mip_constraint)
                else:
                    print(f"Warning: Could not get row for constraint {con.name}")
            except Exception as e:
                print(f"Warning: Could not process constraint {con.name}: {str(e)}")
        
        return mp_model
        
    finally:
        # 恢复原始设置
        model.setPresolve(True)
        model.setHeuristics(True)

def extract_features(mip):
    """从MIP模型中提取特征"""
    try:
        # 创建SCIP求解器
        solver = SCIPSolver()
        
        # 加载模型
        status = solver.load_model(mip)
        if status != MPSolverResponseStatus.NOT_SOLVED:
            print("模型已被求解，无法提取特征")
            return None
            
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
        if solution and solution.status in [
            MPSolverResponseStatus.OPTIMAL,
            MPSolverResponseStatus.FEASIBLE
        ]:
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

def serialize_features(features):
    """将特征序列化为TFRecord格式"""
    try:
        # 序列化张量特征
        variable_features_serialized = tf.io.serialize_tensor(
            tf.convert_to_tensor(features['variable_features'])
        ).numpy()
        constraint_features_serialized = tf.io.serialize_tensor(
            tf.convert_to_tensor(features['constraint_features'])
        ).numpy()
        edge_indices_serialized = tf.io.serialize_tensor(
            tf.convert_to_tensor(features['edge_indices'])
        ).numpy()
        edge_features_serialized = tf.io.serialize_tensor(
            tf.convert_to_tensor(features['edge_features'])
        ).numpy()
        
        # 创建特征字典
        feature_dict = {
            'variable_features': _bytes_feature(variable_features_serialized),
            'constraint_features': _bytes_feature(constraint_features_serialized),
            'edge_indices': _bytes_feature(edge_indices_serialized),
            'edge_features': _bytes_feature(edge_features_serialized),
            'variable_lbs': _float_feature(features['variable_lbs']),
            'variable_ubs': _float_feature(features['variable_ubs']),
            'binary_variable_indices': _int64_feature(features['binary_variable_indices']),
            'all_integer_variable_indices': _int64_feature(features['all_integer_variable_indices']),
            'model_maximize': _int64_feature([int(features['model_maximize'])]),
            'best_solution_labels': _float_feature(features['best_solution_labels']),
            'constraint_feature_names': _bytes_feature(tf.io.serialize_tensor(
                tf.constant([""], dtype=tf.string)
            ).numpy()),
            'variable_feature_names': _bytes_feature(tf.io.serialize_tensor(
                tf.constant([""], dtype=tf.string)
            ).numpy()),
            'edge_features_names': _bytes_feature(tf.io.serialize_tensor(
                tf.constant([""], dtype=tf.string)
            ).numpy()),
            'variable_names': _bytes_feature(tf.io.serialize_tensor(
                tf.constant([""], dtype=tf.string)
            ).numpy()),
        }
        
        return tf.train.Example(features=tf.train.Features(feature=feature_dict))
        
    except Exception as e:
        print(f"特征序列化过程中出错: {e}")
        return None

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
        example = serialize_features(features)
        if example is None:
            return False
            
        # 4. 写入TFRecord文件
        os.makedirs(os.path.dirname(tfrecord_file_path), exist_ok=True)
        with tf.io.TFRecordWriter(tfrecord_file_path) as writer:
            writer.write(example.SerializeToString())
        
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