"""
LOWESS平滑处理模块
用于对电力负荷数据进行LOWESS(局部加权散点图平滑)处理
"""

import numpy as np
from scipy import linalg

def lowess_smooth(data, voltage_index=None, frac=0.3, iter=3):
    """
    对数据进行LOWESS平滑处理
    
    Parameters:
    data (np.array): T*D的二维信号
    voltage_index (int): 电压变量索引
    frac (float): 用于平滑的窗口大小比例
    iter (int): robustifying迭代次数
    
    Returns:
    np.array: 平滑后的数据
    """
    if data.ndim != 2:
        raise ValueError("输入数据必须是二维数组")
    
    T, D = data.shape
    
    # 如果没有指定电压索引，默认为最后一个变量
    if voltage_index is None:
        voltage_index = D - 1
    
    # 对每一列分别进行LOWESS平滑处理
    smoothed_data = np.zeros_like(data)
    for i in range(D):
        smoothed_data[:, i] = _lowess_single_variable(data[:, i], frac, iter)
    
    return smoothed_data

def _lowess_single_variable(y, frac=0.3, iter=3):
    """
    对单个变量进行LOWESS平滑处理
    
    Parameters:
    y (np.array): 一维信号
    frac (float): 用于平滑的窗口大小比例
    iter (int): robustifying迭代次数
    
    Returns:
    np.array: 平滑后的数据
    """
    n = len(y)
    # 计算窗口大小
    window_size = int(np.ceil(frac * n))
    
    # 初始化输出数组
    smoothed_y = np.zeros(n)
    
    # 对每个点进行局部回归
    for i in range(n):
        # 确定局部窗口的范围
        start = max(0, i - window_size // 2)
        end = min(n, i + window_size // 2 + 1)
        
        # 获取局部数据
        x_local = np.arange(start, end)
        y_local = y[start:end]
        
        # 计算权重（基于距离的高斯权重）
        distances = np.abs(x_local - i)
        max_distance = np.max(distances)
        if max_distance == 0:
            weights = np.ones(len(distances))
        else:
            # 使用三立方权重函数
            weights = np.power(1 - np.power(distances / max_distance, 3), 3)
            weights[distances >= max_distance] = 0
        
        # 执行局部加权线性回归
        if len(x_local) > 1:
            smoothed_y[i] = _weighted_linear_regression(x_local, y_local, weights)
        else:
            smoothed_y[i] = y_local[0]
    
    # Robustifying迭代
    if iter > 0:
        residuals = np.abs(y - smoothed_y)
        median_res = np.median(residuals)
        if median_res > 0:
            # 计算鲁棒权重
            robust_weights = np.power(1 - np.power(residuals / (6 * median_res), 2), 2)
            robust_weights[residuals >= (6 * median_res)] = 0
            
            # 重新进行加权平滑
            for j in range(iter):
                for i in range(n):
                    start = max(0, i - window_size // 2)
                    end = min(n, i + window_size // 2 + 1)
                    
                    x_local = np.arange(start, end)
                    y_local = y[start:end]
                    distances = np.abs(x_local - i)
                    max_distance = np.max(distances)
                    if max_distance == 0:
                        weights = np.ones(len(distances))
                    else:
                        weights = np.power(1 - np.power(distances / max_distance, 3), 3)
                        weights[distances >= max_distance] = 0
                    
                    # 应用鲁棒权重
                    if i >= start and i < end:
                        idx = i - start
                        weights *= robust_weights[start:end]
                    
                    if len(x_local) > 1:
                        smoothed_y[i] = _weighted_linear_regression(x_local, y_local, weights)
                    else:
                        smoothed_y[i] = y_local[0]
    
    return smoothed_y

def _weighted_linear_regression(x, y, weights):
    """
    执行加权线性回归
    
    Parameters:
    x (np.array): 自变量
    y (np.array): 因变量
    weights (np.array): 权重
    
    Returns:
    float: 在x中心点的预测值
    """
    # 构建加权设计矩阵
    W = np.diag(weights)
    X = np.column_stack([np.ones(len(x)), x])
    
    # 计算加权最小二乘解
    try:
        # (X'WX)^(-1)X'Wy
        XtWX = X.T @ W @ X
        if np.linalg.det(XtWX) != 0:
            beta = np.linalg.solve(XtWX, X.T @ W @ y)
            # 返回中心点的预测值
            center_x = x[len(x)//2]
            prediction = beta[0] + beta[1] * center_x
            return prediction
        else:
            # 如果矩阵奇异，返回加权平均值
            if np.sum(weights) > 0:
                return np.average(y, weights=weights)
            else:
                return y[len(y)//2]
    except np.linalg.LinAlgError:
        # 如果求解失败，返回加权平均值
        if np.sum(weights) > 0:
            return np.average(y, weights=weights)
        else:
            return y[len(y)//2]

# 示例用法
if __name__ == "__main__":
    # 生成示例数据（包含趋势和噪声）
    np.random.seed(42)
    T, D = 100, 5
    
    # 创建具有趋势的数据
    t = np.linspace(0, 4*np.pi, T)
    data = np.zeros((T, D))
    for i in range(D):
        # 添加正弦波趋势
        data[:, i] = np.sin(t + i*np.pi/4) + 0.5*np.cos(2*t + i*np.pi/2)
        # 添加噪声
        data[:, i] += 0.3 * np.random.randn(T)
    
    # LOWESS平滑处理
    smoothed_data = lowess_smooth(data, frac=0.3, iter=3)
    
    print("=== LOWESS平滑处理示例 ===")
    print(f"原始数据形状: {data.shape}")
    print(f"平滑后数据形状: {smoothed_data.shape}")
    print("原始数据前5行:")
    print(data[:5])
    print("平滑后数据前5行:")
    print(smoothed_data[:5])
    
    # 计算平滑效果
    mse = np.mean((data - smoothed_data)**2)
    print(f"\n平滑前后均方误差: {mse:.4f}")