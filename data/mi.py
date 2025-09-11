import numpy as np
from sklearn.metrics import mutual_info_score

def compute_mutual_information(x, y):
    return mutual_info_score(x, y)

def calculate_mi_matrix(data, voltage_index=None):
    if data.ndim != 2:
        raise ValueError("输入数据必须是二维数组")
    
    T, D = data.shape
    if voltage_index is None:
        voltage_index = D - 1
    if voltage_index < 0:
        voltage_index = D + voltage_index
    if voltage_index < 0 or voltage_index >= D:
        raise ValueError("电压变量索引超出范围")
    
    voltage_signal = data[:, voltage_index]
    mi_vector = np.zeros(D)
    
    for i in range(D):
        if i == voltage_index:
            mi_vector[i] = 0
        else:
            mi_vector[i] = compute_mutual_information(data[:, i], voltage_signal)
    
    return mi_vector

def find_top_correlated_features(data, voltage_index=None, top_k=5):
    mi_vector = calculate_mi_matrix(data, voltage_index)
    
    if voltage_index is None:
        voltage_index = data.shape[1] - 1
    if voltage_index < 0:
        voltage_index = data.shape[1] + voltage_index
    
    mi_without_voltage = np.delete(mi_vector, voltage_index)
    indices_without_voltage = np.delete(np.arange(len(mi_vector)), voltage_index)
    
    top_indices = np.argsort(mi_without_voltage)[::-1][:top_k]
    top_features = [(indices_without_voltage[i], mi_without_voltage[i]) for i in top_indices]
    
    return top_features

if __name__ == "__main__":
    np.random.seed(42)
    T, D = 100, 5
    data = np.random.randint(0, 10, size=(T, D))
    data[:, -1] = np.where(np.random.rand(T) < 0.7, data[:, 0], np.random.randint(0, 10, T))
    
    mi_vector = calculate_mi_matrix(data, voltage_index=-1)
    top_features = find_top_correlated_features(data, voltage_index=-1, top_k=3)
    
    print("互信息向量:", mi_vector)
    print("前3个相关特征:", top_features)