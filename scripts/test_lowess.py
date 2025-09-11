import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from data.lowess import lowess_smooth

def create_test_data():
    np.random.seed(42)
    T, D = 200, 4
    
    t = np.linspace(0, 6*np.pi, T)
    data = np.zeros((T, D))
    
    data[:, 0] = np.sin(t) + 0.3 * np.random.randn(T)
    data[:, 1] = np.cos(t) + 0.3 * np.random.randn(T)
    data[:, 2] = 0.02 * t + 0.2 * np.random.randn(T)
    data[:, 3] = np.sin(t) + 0.5*np.cos(2*t) + 0.2 * np.random.randn(T)
    
    return data

def test_lowess_functionality():
    data = create_test_data()
    T, D = data.shape
    print("数据形状:", data.shape)
    
    test_cases = [
        {"frac": 0.1, "iter": 1},
        {"frac": 0.3, "iter": 3},
        {"frac": 0.5, "iter": 5}
    ]
    
    for i, case in enumerate(test_cases):
        smoothed_data = lowess_smooth(data, frac=case['frac'], iter=case['iter'])
        mse = np.mean((data - smoothed_data)**2)
        diff = np.mean(np.abs(data - smoothed_data))
        print(f"测试 {i+1}: frac={case['frac']}, iter={case['iter']}")
        print(f"  均方误差: {mse:.4f}, 平均差异: {diff:.4f}")

def main():
    test_lowess_functionality()
    print("LOWESS平滑功能测试完成")

if __name__ == "__main__":
    main()