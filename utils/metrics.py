#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估指标模块
实现常用的回归评估指标：RMSE、MAE、R²
"""

import numpy as np

def rmse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("真实值和预测值的形状必须相同")
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("真实值和预测值的形状必须相同")
    return np.mean(np.abs(y_true - y_pred))

def r2_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("真实值和预测值的形状必须相同")
    
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    
    return 1 - (ss_res / ss_tot)

def all_metrics(y_true, y_pred):
    return {
        'rmse': rmse(y_true, y_pred),
        'mae': mae(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }

if __name__ == "__main__":
    y_true = [3, -0.5, 2, 7]
    y_pred = [2.5, 0.0, 2, 8]
    
    print("RMSE:", rmse(y_true, y_pred))
    print("MAE:", mae(y_true, y_pred))
    print("R²:", r2_score(y_true, y_pred))
    print("所有指标:", all_metrics(y_true, y_pred))