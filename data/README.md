# 数据处理模块说明

本目录包含电力负荷预测项目中使用的各种数据处理方法。

## 模块介绍

### 1. 互信息计算 (mi.py)
计算T*D二维信号中各变量与电压变量的互信息，用于特征选择。

主要功能：
- `calculate_mi_matrix()`: 计算互信息矩阵
- `find_top_correlated_features()`: 找到与电压最相关的特征

### 2. P-M分数计算 (pm_scores.py)
计算P-M分数，用于评估变量的重要性。

主要功能：
- `calculate_pm_scores()`: 计算P-M分数

### 3. LOWESS平滑处理 (lowess.py)
使用局部加权散点图平滑方法对数据进行平滑处理。

主要功能：
- `lowess_smooth()`: LOWESS平滑处理

### 4. 小波变换处理 (wavelet.py)
使用小波变换对数据进行去噪和特征提取。

主要功能：
- `wavelet_transform()`: 小波变换
- `wavelet_denoising()`: 小波去噪

### 5. 数据预处理主流程 (data_preprocessing.py)
整合所有数据处理方法的主流程。

主要功能：
- `preprocess_data()`: 数据预处理主流程
- `load_and_preprocess_data()`: 加载并预处理数据

## 使用示例

```python
import numpy as np
from data.mi import calculate_mi_matrix

# 创建示例数据
data = np.random.randn(100, 5)

# 计算互信息
mi_vector, results = calculate_mi_matrix(data, voltage_index=-1)
```

## 依赖关系

- numpy
- scikit-learn
- scipy
- PyWavelets