"""
LOWESS平滑处理模块
用于对电力负荷数据进行LOWESS(局部加权散点图平滑)处理
"""

import numpy as np
from scipy import linalg

def lowess_smooth(data, voltage_index=None, frac=0.3, iter=3):
    if data.ndim != 2:
        raise ValueError("输入数据必须是二维数组")
    
    T, D = data.shape
    if voltage_index is None:
        voltage_index = D - 1
    
    smoothed_data = np.zeros_like(data)
    for i in range(D):
        smoothed_data[:, i] = _lowess_single_variable(data[:, i], frac, iter)
    
    return smoothed_data

def _lowess_single_variable(y, frac=0.3, iter=3):
    n = len(y)
    window_size = int(np.ceil(frac * n))
    smoothed_y = np.zeros(n)
    
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
        
        if len(x_local) > 1:
            smoothed_y[i] = _weighted_linear_regression(x_local, y_local, weights)
        else:
            smoothed_y[i] = y_local[0]
    
    if iter > 0:
        residuals = np.abs(y - smoothed_y)
        median_res = np.median(residuals)
        if median_res > 0:
            robust_weights = np.power(1 - np.power(residuals / (6 * median_res), 2), 2)
            robust_weights[residuals >= (6 * median_res)] = 0
            
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
                    
                    if i >= start and i < end:
                        idx = i - start
                        weights *= robust_weights[start:end]
                    
                    if len(x_local) > 1:
                        smoothed_y[i] = _weighted_linear_regression(x_local, y_local, weights)
                    else:
                        smoothed_y[i] = y_local[0]
    
    return smoothed_y

def _weighted_linear_regression(x, y, weights):
    W = np.diag(weights)
    X = np.column_stack([np.ones(len(x)), x])
    
    try:
        XtWX = X.T @ W @ X
        if np.linalg.det(XtWX) != 0:
            beta = np.linalg.solve(XtWX, X.T @ W @ y)
            center_x = x[len(x)//2]
            prediction = beta[0] + beta[1] * center_x
            return prediction
        else:
            if np.sum(weights) > 0:
                return np.average(y, weights=weights)
            else:
                return y[len(y)//2]
    except np.linalg.LinAlgError:
        if np.sum(weights) > 0:
            return np.average(y, weights=weights)
        else:
            return y[len(y)//2]

if __name__ == "__main__":
    np.random.seed(42)
    T, D = 100, 5
    
    t = np.linspace(0, 4*np.pi, T)
    data = np.zeros((T, D))
    for i in range(D):
        data[:, i] = np.sin(t + i*np.pi/4) + 0.5*np.cos(2*t + i*np.pi/2)
        data[:, i] += 0.3 * np.random.randn(T)
    
    smoothed_data = lowess_smooth(data, frac=0.3, iter=3)
    
    print("原始数据形状:", data.shape)
    print("平滑后数据形状:", smoothed_data.shape)
    mse = np.mean((data - smoothed_data)**2)
    print("平滑前后均方误差:", mse)