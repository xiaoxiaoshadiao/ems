import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.metrics import rmse, mae, r2_score, all_metrics

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

print("RMSE:", rmse(y_true, y_pred))
print("MAE:", mae(y_true, y_pred))
print("R²:", r2_score(y_true, y_pred))
print("所有指标:", all_metrics(y_true, y_pred))