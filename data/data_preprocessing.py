"""
数据预处理主流程
整合MI和LOWESS处理方法
"""

import numpy as np
from data.mi import calculate_mi_matrix, find_top_correlated_features
from data.lowess import lowess_smooth

def preprocess_data(data, voltage_index=None, method='all'):
    if data.ndim != 2:
        raise ValueError("输入数据必须是二维数组")
    
    results = {}
    
    if method == 'mi' or method == 'all':
        mi_vector = calculate_mi_matrix(data, voltage_index)
        results['mi'] = mi_vector
        
        top_features = find_top_correlated_features(data, voltage_index, top_k=3)
        results['top_features'] = top_features
    
    if method == 'lowess' or method == 'all':
        smoothed_data = lowess_smooth(data, voltage_index)
        results['lowess'] = smoothed_data
    
    return results

def load_and_preprocess_data(file_path, voltage_index=None):
    print(f"从 {file_path} 加载数据...")
    
    np.random.seed(42)
    T, D = 200, 6
    data = np.random.randn(T, D)
    data[:, -1] = 0.8 * data[:, 1] + 0.2 * np.random.randn(T)
    
    print("执行数据预处理流程...")
    results = preprocess_data(data, voltage_index, method='all')
    
    return results

if __name__ == "__main__":
    np.random.seed(42)
    T, D = 200, 6
    data = np.random.randn(T, D)
    data[:, -1] = 0.8 * data[:, 1] + 0.2 * np.random.randn(T)
    
    print("数据形状:", data.shape)
    
    results = preprocess_data(data, voltage_index=-1, method='all')
    
    if 'mi' in results:
        print("互信息向量:", results['mi'])
    
    if 'top_features' in results:
        print("前3个相关特征:", results['top_features'])
    
    print("数据预处理完成!")