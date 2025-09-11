import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.metrics import rmse, mae, r2_score, all_metrics

# 示例数据
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

# 单独计算各项指标
rmse_value = rmse(y_true, y_pred)
mae_value = mae(y_true, y_pred)
r2_value = r2_score(y_true, y_pred)

# 或者一次性计算所有指标
metrics = all_metrics(y_true, y_pred)

# 打印结果
print("评估指标计算示例:")
print(f"真实值: {y_true}")
print(f"预测值: {y_pred}")
print()
print(f"RMSE: {rmse_value:.6f}")
print(f"MAE:  {mae_value:.6f}")
print(f"R²:   {r2_value:.6f}")
print()
print("所有评估指标:")
for metric, value in metrics.items():
    print(f"{metric.upper()}: {value:.6f}")