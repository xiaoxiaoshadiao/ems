import numpy as np
from sklearn.metrics import mutual_info_score
from scipy.spatial.distance import jaccard
import warnings

def discretize_signal(signal, bins=10):
    """
    将连续信号离散化
    
    Parameters:
    signal (np.array): 输入信号
    bins (int): 离散化的箱数
    
    Returns:
    np.array: 离散化后的信号
    """
    # 处理常数信号的情况
    if np.std(signal) == 0:
        return np.zeros_like(signal, dtype=int)
    
    # 使用等宽分箱进行离散化
    discretized = np.digitize(signal, np.linspace(signal.min(), signal.max(), bins))
    return discretized

def compute_mutual_information(x, y, bins=10):
    """
    计算两个变量之间的互信息
    
    Parameters:
    x (np.array): 第一个变量
    y (np.array): 第二个变量
    bins (int): 离散化的箱数
    
    Returns:
    float: 互信息值
    """
    # 离散化信号
    x_disc = discretize_signal(x, bins)
    y_disc = discretize_signal(y, bins)
    
    # 计算互信息
    mi = mutual_info_score(x_disc, y_disc)
    return mi

def calculate_mi_matrix(data, voltage_index=None, bins=10):
    """
    计算T*D二维信号中各变量与电压变量的互信息矩阵
    
    Parameters:
    data (np.array): T*D的二维信号，T为时间步，D为变量维度
    voltage_index (int): 电压变量在D维度中的索引，如果为None则默认为最后一个变量
    bins (int): 离散化的箱数
    
    Returns:
    np.array: 1*D的互信息向量，表示各变量与电压变量的互信息
    dict: 包含详细信息的字典
    """
    # 检查输入数据
    if data.ndim != 2:
        raise ValueError("输入数据必须是二维数组")
    
    T, D = data.shape
    
    # 如果没有指定电压索引，默认为最后一个变量
    if voltage_index is None:
        voltage_index = D - 1
    
    # 处理负索引
    if voltage_index < 0:
        voltage_index = D + voltage_index
    
    # 检查电压索引是否有效
    if voltage_index < 0 or voltage_index >= D:
        raise ValueError("电压变量索引超出范围")
    
    # 提取电压信号
    voltage_signal = data[:, voltage_index]
    
    # 初始化互信息向量
    mi_vector = np.zeros(D)
    variable_names = [f"Variable_{i}" for i in range(D)]
    
    # 计算每个变量与电压信号的互信息
    for i in range(D):
        if i == voltage_index:
            mi_vector[i] = 0  # 电压与自身的互信息为0
        else:
            mi_vector[i] = compute_mutual_information(data[:, i], voltage_signal, bins)
    
    # 创建结果字典
    results = {
        'mi_values': mi_vector,
        'voltage_index': voltage_index,
        'variable_names': variable_names,
        'voltage_name': variable_names[voltage_index]
    }
    
    return mi_vector, results

def find_top_correlated_features(data, voltage_index=None, top_k=5, bins=10):
    """
    找到与电压变量互信息最高的前K个特征
    
    Parameters:
    data (np.array): T*D的二维信号
    voltage_index (int): 电压变量索引
    top_k (int): 返回前K个相关特征
    bins (int): 离散化的箱数
    
    Returns:
    list: 包含(特征索引, 互信息值)的元组列表
    dict: 详细结果信息
    """
    mi_vector, results = calculate_mi_matrix(data, voltage_index, bins)
    
    # 获取除电压变量外的所有互信息值
    mi_without_voltage = np.delete(mi_vector, results['voltage_index'])
    indices_without_voltage = np.delete(np.arange(len(mi_vector)), results['voltage_index'])
    
    # 获取前K个最大的互信息值及其索引
    top_indices = np.argsort(mi_without_voltage)[::-1][:top_k]
    top_features = [(indices_without_voltage[i], mi_without_voltage[i]) for i in top_indices]
    
    # 添加详细信息到结果
    results['top_features'] = top_features
    results['top_k'] = top_k
    
    return top_features, results

# 示例用法
if __name__ == "__main__":
    # 生成示例数据 (100个时间步, 5个变量)
    np.random.seed(42)
    T, D = 100, 5
    data = np.random.randn(T, D)
    
    # 假设电压是最后一个变量，并与第一个变量有较强相关性
    data[:, -1] = 0.7 * data[:, 0] + 0.3 * np.random.randn(T)
    
    print("=== 互信息计算示例 ===")
    print(f"数据形状: {data.shape}")
    
    # 计算互信息矩阵
    mi_vector, results = calculate_mi_matrix(data, voltage_index=-1)
    
    print(f"\n电压变量: {results['voltage_name']}")
    print("各变量与电压的互信息:")
    for i, (name, mi_value) in enumerate(zip(results['variable_names'], mi_vector)):
        print(f"  {name} vs {results['voltage_name']}: {mi_value:.4f}")
    
    # 找到与电压最相关的前3个特征
    top_features, top_results = find_top_correlated_features(data, voltage_index=-1, top_k=3)
    
    print(f"\n与电压最相关的前{top_results['top_k']}个特征:")
    for idx, mi_value in top_features:
        print(f"  {results['variable_names'][idx]}: {mi_value:.4f}")