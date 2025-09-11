"""
数据预处理主流程
整合MI, P-M Scores, LOWESS, Wavelet等处理方法
"""

import numpy as np
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.mi import calculate_mi_matrix, find_top_correlated_features
from data.pm_scores import calculate_pm_scores
from data.lowess import lowess_smooth
from data.wavelet import wavelet_denoising

def preprocess_data(data, voltage_index=None, method='all'):
    """
    数据预处理主流程
    
    Parameters:
    data (np.array): T*D的二维信号
    voltage_index (int): 电压变量索引
    method (str): 预处理方法 ('mi', 'pm', 'lowess', 'wavelet', 'all')
    
    Returns:
    dict: 预处理结果
    """
    if data.ndim != 2:
        raise ValueError("输入数据必须是二维数组")
    
    results = {}
    
    if method == 'mi' or method == 'all':
        # 互信息计算
        print("执行互信息计算...")
        mi_vector, mi_results = calculate_mi_matrix(data, voltage_index)
        results['mi'] = mi_results
        
        # 找到最相关的特征
        top_features, _ = find_top_correlated_features(data, voltage_index, top_k=3)
        results['top_features'] = top_features
    
    if method == 'pm' or method == 'all':
        # P-M分数计算
        print("执行P-M分数计算...")
        pm_scores = calculate_pm_scores(data, voltage_index)
        results['pm_scores'] = pm_scores
    
    if method == 'lowess' or method == 'all':
        # LOWESS平滑处理
        print("执行LOWESS平滑处理...")
        smoothed_data = lowess_smooth(data, voltage_index)
        results['lowess'] = smoothed_data
    
    if method == 'wavelet' or method == 'all':
        # 小波去噪处理
        print("执行小波去噪处理...")
        denoised_data = wavelet_denoising(data, voltage_index)
        results['wavelet'] = denoised_data
    
    return results

def load_and_preprocess_data(file_path, voltage_index=None):
    """
    加载并预处理数据
    
    Parameters:
    file_path (str): 数据文件路径
    voltage_index (int): 电压变量索引
    
    Returns:
    dict: 预处理结果
    """
    # TODO: 实现数据加载逻辑
    # 这里只是一个示例实现
    print(f"从 {file_path} 加载数据...")
    
    # 生成示例数据
    np.random.seed(42)
    T, D = 200, 6
    data = np.random.randn(T, D)
    
    # 让变量1与电压变量(最后一个)有强相关性
    data[:, -1] = 0.8 * data[:, 1] + 0.2 * np.random.randn(T)
    
    print("执行完整的数据预处理流程...")
    results = preprocess_data(data, voltage_index, method='all')
    
    return results

# 示例用法
if __name__ == "__main__":
    print("=== 数据预处理主流程示例 ===")
    
    # 创建示例数据
    np.random.seed(42)
    T, D = 200, 6
    data = np.random.randn(T, D)
    
    # 让变量1与电压变量(最后一个)有强相关性
    data[:, -1] = 0.8 * data[:, 1] + 0.2 * np.random.randn(T)
    
    print(f"数据形状: {data.shape}")
    
    # 执行完整的预处理流程
    results = preprocess_data(data, voltage_index=-1, method='all')
    
    # 输出结果
    if 'mi' in results:
        print("\n互信息计算结果:")
        mi_results = results['mi']
        print(f"电压变量: {mi_results['voltage_name']}")
        for i, (name, mi_value) in enumerate(zip(mi_results['variable_names'], mi_results['mi_values'])):
            marker = " (电压变量)" if i == mi_results['voltage_index'] else ""
            print(f"  {name} vs {mi_results['voltage_name']}: {mi_value:.4f}{marker}")
    
    if 'top_features' in results:
        print(f"\n与电压最相关的前3个特征:")
        for idx, mi_value in results['top_features']:
            print(f"  Variable_{idx}: {mi_value:.4f}")
    
    print("\n数据预处理完成!")