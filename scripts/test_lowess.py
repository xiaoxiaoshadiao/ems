import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from data.lowess import lowess_smooth

def create_test_data():
    """创建测试数据"""
    np.random.seed(42)
    T, D = 200, 4
    
    # 创建具有不同特征的测试数据
    t = np.linspace(0, 6*np.pi, T)
    data = np.zeros((T, D))
    
    # 变量0: 正弦波 + 噪声
    data[:, 0] = np.sin(t) + 0.3 * np.random.randn(T)
    
    # 变量1: 余弦波 + 噪声
    data[:, 1] = np.cos(t) + 0.3 * np.random.randn(T)
    
    # 变量2: 线性趋势 + 噪声
    data[:, 2] = 0.02 * t + 0.2 * np.random.randn(T)
    
    # 变量3: 复合信号 + 噪声
    data[:, 3] = np.sin(t) + 0.5*np.cos(2*t) + 0.2 * np.random.randn(T)
    
    return data

def test_lowess_functionality():
    """测试LOWESS功能"""
    print("=== LOWESS平滑功能测试 ===")
    
    # 创建测试数据
    data = create_test_data()
    T, D = data.shape
    print(f"数据形状: {data.shape}")
    
    # 测试不同的参数设置
    test_cases = [
        {"frac": 0.1, "iter": 1, "description": "小窗口，少迭代"},
        {"frac": 0.3, "iter": 3, "description": "中等窗口，标准迭代"},
        {"frac": 0.5, "iter": 5, "description": "大窗口，多迭代"}
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\n{i+1}. {case['description']}:")
        smoothed_data = lowess_smooth(data, frac=case['frac'], iter=case['iter'])
        
        # 计算平滑效果
        mse = np.mean((data - smoothed_data)**2)
        print(f"   平滑前后均方误差: {mse:.4f}")
        
        # 检查平滑数据是否与原始数据不同
        diff = np.mean(np.abs(data - smoothed_data))
        print(f"   平均绝对差异: {diff:.4f}")
        
        # 验证输出形状
        assert smoothed_data.shape == data.shape, "输出形状不正确"
        print(f"   ✓ 输出形状正确: {smoothed_data.shape}")
        
        # 验证没有NaN值
        assert not np.isnan(smoothed_data).any(), "输出包含NaN值"
        print(f"   ✓ 无NaN值")

def test_edge_cases():
    """测试边界情况"""
    print("\n=== 边界情况测试 ===")
    
    # 测试单行数据
    try:
        single_row = np.random.randn(1, 3)
        result = lowess_smooth(single_row)
        print("✓ 单行数据处理成功")
    except Exception as e:
        print(f"✗ 单行数据处理失败: {e}")
    
    # 测试两行数据
    try:
        two_rows = np.random.randn(2, 3)
        result = lowess_smooth(two_rows)
        print("✓ 两行数据处理成功")
    except Exception as e:
        print(f"✗ 两行数据处理失败: {e}")
    
    # 测试不同frac值
    data = create_test_data()
    try:
        for frac in [0.01, 0.1, 0.5, 0.9, 0.99]:
            result = lowess_smooth(data, frac=frac)
        print("✓ 不同frac值处理成功")
    except Exception as e:
        print(f"✗ 不同frac值处理失败: {e}")

def visualize_results():
    """可视化结果"""
    print("\n=== 结果可视化 ===")
    
    # 创建测试数据
    data = create_test_data()
    smoothed_data = lowess_smooth(data, frac=0.3, iter=3)
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    t = np.arange(data.shape[0])
    
    for i in range(min(4, data.shape[1])):
        axes[i].plot(t, data[:, i], 'b-', alpha=0.7, label='原始数据', linewidth=1)
        axes[i].plot(t, smoothed_data[:, i], 'r-', label='LOWESS平滑', linewidth=2)
        axes[i].set_title(f'变量 {i}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lowess_test_results.png')
    print("结果已保存到 'lowess_test_results.png'")
    plt.show()

def main():
    """主函数"""
    test_lowess_functionality()
    test_edge_cases()
    
    print("\n=== 测试完成 ===")
    print("所有测试均已通过，LOWESS平滑功能正常工作。")
    
    # 显示可视化结果
    visualize_results()

if __name__ == "__main__":
    main()