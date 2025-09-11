import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from data.mi import calculate_mi_matrix, find_top_correlated_features

def create_test_data():
    """创建测试数据"""
    np.random.seed(42)
    T, D = 200, 6
    
    # 创建6个变量的测试数据
    data = np.random.randn(T, D)
    
    # 让变量1与电压变量(最后一个)有强相关性
    data[:, -1] = 0.8 * data[:, 1] + 0.2 * np.random.randn(T)
    
    # 让变量3与电压变量有中等相关性
    data[:, 3] = 0.5 * data[:, -1] + 0.5 * np.random.randn(T)
    
    return data

def main():
    print("=== MI模块测试 ===")
    
    # 创建测试数据
    data = create_test_data()
    print(f"数据形状: {data.shape}")
    
    # 计算互信息矩阵
    print("\n1. 计算互信息矩阵:")
    mi_vector, results = calculate_mi_matrix(data, voltage_index=-1)
    
    print(f"电压变量索引: {results['voltage_index']}")
    print("各变量与电压的互信息:")
    for i, (name, mi_value) in enumerate(zip(results['variable_names'], mi_vector)):
        marker = " (电压变量)" if i == results['voltage_index'] else ""
        print(f"  {name} vs {results['voltage_name']}: {mi_value:.4f}{marker}")
    
    # 找到与电压最相关的前3个特征
    print("\n2. 找到与电压最相关的前3个特征:")
    top_features, top_results = find_top_correlated_features(data, voltage_index=-1, top_k=3)
    
    print(f"前{top_results['top_k']}个相关特征:")
    for idx, mi_value in top_features:
        print(f"  {results['variable_names'][idx]}: {mi_value:.4f}")
    
    # 验证结果
    print("\n3. 结果验证:")
    # 变量1应该有最高的互信息（我们构造的数据中与电压最相关）
    var1_mi = mi_vector[1]
    voltage_mi = mi_vector[-1]  # 电压与自身互信息为0
    
    print(f"变量1与电压的互信息: {var1_mi:.4f}")
    print(f"电压与自身的互信息: {voltage_mi:.4f}")
    
    if var1_mi > 0.1:  # 阈值检查
        print("✓ 变量1与电压有较强相关性，结果合理")
    else:
        print("✗ 变量1与电压相关性较弱，可能需要检查实现")

if __name__ == "__main__":
    main()