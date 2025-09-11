import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from data.data_preprocessing import preprocess_data, load_and_preprocess_data

def create_test_data():
    """创建测试数据"""
    np.random.seed(42)
    T, D = 300, 7
    
    # 创建7个变量的测试数据
    data = np.random.randn(T, D)
    
    # 让变量2与电压变量(最后一个)有强相关性
    data[:, -1] = 0.75 * data[:, 2] + 0.25 * np.random.randn(T)
    
    # 让变量4与电压变量有中等相关性
    data[:, 4] = 0.5 * data[:, -1] + 0.5 * np.random.randn(T)
    
    return data

def main():
    print("=== 数据预处理流程测试 ===")
    
    # 创建测试数据
    data = create_test_data()
    print(f"数据形状: {data.shape}")
    
    # 测试完整的预处理流程
    print("\n1. 执行完整的数据预处理流程:")
    results = preprocess_data(data, voltage_index=-1, method='all')
    
    # 验证结果
    print("预处理完成，结果包含以下键:")
    for key in results.keys():
        print(f"  - {key}")
    
    # 详细查看互信息结果
    if 'mi' in results:
        print("\n2. 互信息计算结果:")
        mi_results = results['mi']
        print(f"电压变量索引: {mi_results['voltage_index']}")
        
        # 找到互信息最大的变量（除了电压本身）
        mi_values = mi_results['mi_values']
        voltage_idx = mi_results['voltage_index']
        
        # 创建一个副本并设置电压变量的互信息为负无穷，以便找到最大值
        mi_copy = mi_values.copy()
        mi_copy[voltage_idx] = -np.inf
        max_mi_idx = np.argmax(mi_copy)
        
        print(f"与电压相关性最高的变量: Variable_{max_mi_idx} (MI={mi_values[max_mi_idx]:.4f})")
    
    # 测试单个方法
    print("\n3. 测试单独的互信息计算:")
    mi_only_results = preprocess_data(data, voltage_index=-1, method='mi')
    print(f"互信息结果键: {list(mi_only_results.keys())}")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    main()