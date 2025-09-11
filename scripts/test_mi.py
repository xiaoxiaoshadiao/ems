import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from data.mi import calculate_mi_matrix, find_top_correlated_features

def create_test_data():
    np.random.seed(42)
    T, D = 200, 6
    data = np.random.randint(0, 10, size=(T, D))
    data[:, -1] = np.where(np.random.rand(T) < 0.8, data[:, 1], np.random.randint(0, 10, T))
    data[:, 3] = np.where(np.random.rand(T) < 0.5, data[:, -1], np.random.randint(0, 10, T))
    return data

def main():
    data = create_test_data()
    print("数据形状:", data.shape)
    
    mi_vector = calculate_mi_matrix(data, voltage_index=-1)
    print("互信息向量:", mi_vector)
    
    top_features = find_top_correlated_features(data, voltage_index=-1, top_k=3)
    print("前3个相关特征:", top_features)

if __name__ == "__main__":
    main()
