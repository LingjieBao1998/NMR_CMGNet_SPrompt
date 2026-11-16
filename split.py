import numpy as np
import ot
from scipy.optimize import linprog

def target_split_transport(u_values, v_values, u_weights=None, v_weights=None):
    """
    强制目标点分裂：让数量少的v_values进行分裂
    u_values有5个点，v_values有4个点
    让v点分裂来匹配更多的u点
    """
    u = np.asarray(u_values)
    v = np.asarray(v_values)
    
    n, m = len(u), len(v)
    
    if u_weights is None:
        u_weights = np.ones(n) / n
    else:
        u_weights = np.asarray(u_weights) / np.sum(u_weights)
    
    if v_weights is None:
        v_weights = np.ones(m) / m
    else:
        v_weights = np.asarray(v_weights) / np.sum(v_weights)
    
    # 成本矩阵
    C = np.abs(u[:, np.newaxis] - v[np.newaxis, :])
    
    # 我们想要v点分裂，所以设置约束：
    # 1. 每个u点只能连接到一个v点（不允许u分裂）
    # 2. 每个v点可以连接到多个u点（允许v分裂）
    
    # 线性规划求解
    # 变量: x[i,j] 表示从u[i]到v[j]的传输量
    # 目标: 最小化 sum(C[i,j] * x[i,j])
    
    c = C.flatten()
    
    # 约束矩阵
    A_eq = []
    b_eq = []
    
    # 约束1: 每个u点的输出等于其权重（不允许u分裂）
    for i in range(n):
        row = np.zeros((n, m))
        row[i, :] = 1
        A_eq.append(row.flatten())
        b_eq.append(u_weights[i])
    
    # 约束2: 每个v点的输入等于其权重（但允许v接收多个u点）
    for j in range(m):
        col = np.zeros((n, m))
        col[:, j] = 1
        A_eq.append(col.flatten())
        b_eq.append(v_weights[j])
    
    A_eq = np.array(A_eq)
    b_eq = np.array(b_eq)
    
    # 边界: x >= 0
    bounds = [(0, None)] * (n * m)
    
    # 求解
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if res.success:
        transport = res.x.reshape(n, m)
        return transport
    else:
        # 如果失败，使用标准最优传输
        return ot.emd(u_weights, v_weights, C)

# 测试
u_values = [0, 1, 3, 5, 7]
v_values = [0, 2, 4, 6]

transport_v_split = target_split_transport(u_values, v_values)
print("目标点分裂传输计划 (v_values分裂):")
print(transport_v_split)
print("\n详细对应关系:")
for i in range(len(u_values)):
    for j in range(len(v_values)):
        if transport_v_split[i, j] > 0.001:
            print(f"u[{i}]={u_values[i]} -> v[{j}]={v_values[j]} (质量: {transport_v_split[i, j]:.3f})")
