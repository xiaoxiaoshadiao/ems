import numpy as np
from sklearn.linear_model import LogisticRegression
from models.famm import FAMM
from models.ramm import RAMM


class AM2FFramework:
    def __init__(self, theta_high=0.7, theta_low=0.3):
        self.theta_high = theta_high
        self.theta_low = theta_low
        self.classifier = LogisticRegression()
        self.famm = FAMM()
        self.ramm = RAMM()
        self.is_trained = False

    def extract_features(self, data_window):
        """提取分类特征"""
        features = {}
        # 1. 波动性特征
        features['variance'] = np.var(data_window)
        features['max_min_diff'] = np.max(data_window) - np.min(data_window)

        # 2. 趋势性特征
        time_indices = np.arange(len(data_window))
        slope, _ = np.polyfit(time_indices, data_window, 1)
        features['trend_slope'] = slope

        # 3. 事件标志特征
        features['has_start_stop'] = self._detect_start_stop(data_window)
        features['high_load_ratio'] = self._calc_high_load_ratio(data_window)

        return np.array([list(features.values())])

    def _detect_start_stop(self, data):
        """检测启停事件"""
        # 实现启停检测逻辑
        return 0  # 简化示例

    def _calc_high_load_ratio(self, data):
        """计算高负载比例"""
        threshold = np.percentile(data, 80)
        return np.sum(data > threshold) / len(data)

    def train_classifier(self, X_train, y_train):
        """训练逻辑回归分类器"""
        self.classifier.fit(X_train, y_train)
        self.is_trained = True

    def predict(self, real_time_data):
        """AM²F核心预测流程"""
        if not self.is_trained:
            raise ValueError("Classifier not trained!")

        # 1. 特征提取
        features = self.extract_features(real_time_data)

        # 2. 分类器决策
        prob = self.classifier.predict_proba(features)[0, 1]  # P(FAMM)

        # 3. 决策规则
        if prob > self.theta_high:
            # FAMM主导
            prediction = self.famm.predict(real_time_data)
            decision = "FAMM"
            alpha = 1.0
        elif prob < self.theta_low:
            # RAMM主导
            prediction = self.ramm.predict(real_time_data)
            decision = "RAMM"
            alpha = 0.0
        else:
            # 加权融合
            pred_famm = self.famm.predict(real_time_data)
            pred_ramm = self.ramm.predict(real_time_data)
            alpha = prob  # 线性映射
            prediction = alpha * pred_famm + (1 - alpha) * pred_ramm
            decision = f"Fusion(alpha={alpha:.2f})"

        return prediction, decision, alpha

    def evaluate_framework(self, test_data):
        """框架性能评估"""
        results = []
        for data in test_data:
            pred, decision, alpha = self.predict(data)
            results.append({
                'prediction': pred,
                'decision': decision,
                'alpha': alpha,
                'actual': data['target']  # 假设数据中包含真实值
            })
        return results