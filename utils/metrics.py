#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估指标模块
实现常用的回归评估指标：RMSE、MAE、R²
"""

import numpy as np
from typing import Union, Optional


def rmse(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
    """
    计算均方根误差 (Root Mean Square Error)
    
    RMSE = sqrt(mean((y_true - y_pred)²))
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        float: RMSE 值
        
    Example:
        >>> y_true = [3, -0.5, 2, 7]
        >>> y_pred = [2.5, 0.0, 2, 8]
        >>> rmse(y_true, y_pred)
        0.6123724356957945
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError("真实值和预测值的形状必须相同")
    
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
    """
    计算平均绝对误差 (Mean Absolute Error)
    
    MAE = mean(|y_true - y_pred|)
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        float: MAE 值
        
    Example:
        >>> y_true = [3, -0.5, 2, 7]
        >>> y_pred = [2.5, 0.0, 2, 8]
        >>> mae(y_true, y_pred)
        0.5
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError("真实值和预测值的形状必须相同")
    
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
    """
    计算决定系数 R² (Coefficient of Determination)
    
    R² = 1 - (SS_res / SS_tot)
    SS_res = Σ(y_true - y_pred)²
    SS_tot = Σ(y_true - mean(y_true))²
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        float: R² 值，范围通常在 (-∞, 1] 之间
               R² = 1 表示完美预测
               R² = 0 表示与均值预测相同
               R² < 0 表示比均值预测更差
        
    Example:
        >>> y_true = [3, -0.5, 2, 7]
        >>> y_pred = [2.5, 0.0, 2, 8]
        >>> r2_score(y_true, y_pred)
        0.9486081370449679
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError("真实值和预测值的形状必须相同")
    
    # 计算总平方和 SS_tot
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    # 计算残差平方和 SS_res
    ss_res = np.sum((y_true - y_pred) ** 2)
    
    # 避免除零错误
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    
    return 1 - (ss_res / ss_tot)


def all_metrics(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> dict:
    """
    一次性计算所有评估指标
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        dict: 包含所有评估指标的字典
        
    Example:
        >>> y_true = [3, -0.5, 2, 7]
        >>> y_pred = [2.5, 0.0, 2, 8]
        >>> all_metrics(y_true, y_pred)
        {'rmse': 0.6123724356957945, 'mae': 0.5, 'r2': 0.9486081370449679}
    """
    return {
        'rmse': rmse(y_true, y_pred),
        'mae': mae(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }


# 测试代码
if __name__ == "__main__":
    # 示例数据
    y_true = [3, -0.5, 2, 7]
    y_pred = [2.5, 0.0, 2, 8]
    
    print("评估指标计算示例:")
    print(f"真实值: {y_true}")
    print(f"预测值: {y_pred}")
    print()
    
    # 分别计算各项指标
    print(f"RMSE: {rmse(y_true, y_pred):.6f}")
    print(f"MAE:  {mae(y_true, y_pred):.6f}")
    print(f"R²:   {r2_score(y_true, y_pred):.6f}")
    print()
    
    # 一次性计算所有指标
    metrics = all_metrics(y_true, y_pred)
    print("所有评估指标:")
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.6f}")