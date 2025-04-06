# 特征提取修改记录

## 1. 边索引格式调整 (2024-03-xx)

### 问题描述
- 代码中的 `edge_indices` 格式为 `[edge_idx, cons_idx, var_idx]`（三列）
- 而论文中 `indices` 是 `[cons_idx, var_idx]`（两列，COO格式的 `row` 和 `col`）
- 多出的 `edge_idx`（边的序号）在论文实现中并未使用
- GCNN只需知道约束-变量对即可

### 修改方案
1. 将 `edge_indices` 的格式从三列改为两列，移除不必要的 `edge_idx`
2. 保持与论文一致的COO格式：`[cons_idx, var_idx]`

### 代码修改
文件：`neural_lns/mip_utils.py`
```diff
- edge_indices.append([len(edge_coeffs), i, var_idx])
+ edge_indices.append([i, var_idx])  # 仅保留约束索引和变量索引
```

### 修改影响
- 简化了边特征的表示方式
- 与论文保持一致，更容易理解和维护
- 不影响模型的功能，因为GCNN只需要知道约束和变量的连接关系

### 验证方法
1. 运行 `test3.py` 检查特征提取
2. 确认边特征的 `indices` 现在是两列格式
3. 验证其他功能不受影响

## 2. 稀疏矩阵支持 (2024-03-xx)

### 问题描述
- 当前代码使用列表 `edge_indices` 和 `edge_coeffs` 存储边特征
- 最终转为稠密 NumPy 数组，不适合大规模稀疏 MILP
- 论文使用 `scipy.sparse.coo_matrix` 存储边特征，更适合大规模稀疏问题

### 修改方案
1. 使用 `scipy.sparse.coo_matrix` 替代稠密数组存储边特征
2. 保持接口不变，仅修改内部实现
3. 在返回特征之前将稀疏矩阵转换为所需格式

### 代码修改
文件：`neural_lns/mip_utils.py`
```diff
+ from scipy import sparse
...
- edge_indices = []
- edge_coeffs = []
+ rows = []
+ cols = []
+ data = []
  for i, cons in enumerate(self.constraint):
    for var_idx, coeff in zip(cons.var_index, cons.coefficient):
      if i in has_lhs:
-       edge_indices.append([i, var_idx])
-       edge_coeffs.append(-coeff / cons_norms[i])
+       rows.append(i)
+       cols.append(var_idx)
+       data.append(-coeff / cons_norms[i])
      if i in has_rhs:
-       edge_indices.append([i, var_idx])
-       edge_coeffs.append(+coeff / cons_norms[i])
+       rows.append(i)
+       cols.append(var_idx)
+       data.append(+coeff / cons_norms[i])

- if edge_coeffs:
-   features['E']['coef'] = np.array(edge_coeffs).reshape(-1, 1)
-   features['E']['indices'] = np.array(edge_indices)
+ if data:
+   # 创建稀疏矩阵
+   edge_matrix = sparse.coo_matrix((data, (rows, cols)))
+   # 提取所需格式的特征
+   features['E']['coef'] = edge_matrix.data.reshape(-1, 1)
+   features['E']['indices'] = np.vstack([edge_matrix.row, edge_matrix.col]).T
```

### 修改影响
- 提高大规模稀疏MILP的内存效率
- 与论文实现保持一致
- 保持API兼容性，不影响外部调用
- 为后续优化提供基础

### 验证方法
1. 运行 `test3.py` 检查特征提取
2. 确认边特征格式与之前相同
3. 验证稀疏矩阵转换的正确性
4. 测试大规模MILP的内存使用情况

## 3. 特征存储格式统一 (2024-03-xx)

### 问题描述
- 代码中边特征的键名为 `coef` 和 `indices`
- 论文中使用 `names` 和 `indices` 存储在 `edge_features` 字典中
- 当前代码缺少 `names` 键，且存储结构略有不同

### 修改方案
1. 将边特征的键名从 `coef` 改为 `names`，保持与论文一致
2. 保持特征的数据类型：`names` 为列表，`indices` 为 numpy 数组
3. 维持现有的特征计算逻辑不变

### 代码修改
文件：`neural_lns/mip_utils.py`
```diff
  if data:
    # 创建稀疏矩阵
    edge_matrix = sparse.coo_matrix((data, (rows, cols)))
    # 提取所需格式的特征
-   features['E']['coef'] = edge_matrix.data.reshape(-1, 1)
+   features['E']['names'] = edge_matrix.data.reshape(-1, 1).tolist()
    features['E']['indices'] = np.vstack([edge_matrix.row, edge_matrix.col]).T
```

### 修改影响
- 统一了特征命名规范
- 与论文保持一致，提高代码可读性
- 不影响特征的实际内容和计算逻辑
- 便于后续与论文代码对比和维护

### 验证方法
1. 运行 `test3.py` 检查特征提取
2. 确认边特征现在使用 `names` 而不是 `coef`
3. 验证特征值保持不变
4. 确认数据类型符合要求 