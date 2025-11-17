# D:\xiaoxiaoshadiao\ems\demo.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
import os

# 设置matplotlib中文字体
plt.rcParams['axes.unicode_minus'] = False


# 数据集类
class TongjiDataset(Dataset):
    """同济大学PEMFC数据集加载器"""

    def __init__(self, data_path=None, sequence_length=100, prediction_horizon=1,
                 mode='single_step', train_ratio=0.7, target_column='voltage'):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.mode = mode
        self.target_column = target_column

        # 设置默认数据路径
        if data_path is None:
            data_path = os.path.join(os.path.dirname(__file__), 'data', 'processed', 'tongji',
                                     'Durability_test_dataset', 'classified_current_data',
                                     'all_representative_rows.csv')

        # 加载数据
        self.data = pd.read_csv(data_path)
        if 'time' in self.data.columns:
            self.data = self.data.drop('time', axis=1)

        # 划分数据集
        train_size = int(len(self.data) * train_ratio)
        self.train_data = self.data.iloc[:train_size]
        self.test_data = self.data.iloc[train_size:]

        # 归一化
        self.scaler = StandardScaler()
        self.train_data_scaled = pd.DataFrame(self.scaler.fit_transform(self.train_data),
                                              columns=self.train_data.columns)
        self.test_data_scaled = pd.DataFrame(self.scaler.transform(self.test_data),
                                             columns=self.test_data.columns)

        # 创建序列
        self.train_sequences = self._create_sequences(self.train_data_scaled)
        self.test_sequences = self._create_sequences(self.test_data_scaled)

        print(f"训练集序列: {len(self.train_sequences[0])}, 测试集序列: {len(self.test_sequences[0])}")

    def _create_sequences(self, data):
        """创建输入输出序列"""
        X, y = [], []
        data_values = data.values
        target_idx = list(data.columns).index(self.target_column)

        if self.mode == 'single_step':
            # 单步预测
            for i in range(len(data_values) - self.sequence_length):
                X.append(data_values[i:i + self.sequence_length])
                y.append(data_values[i + self.sequence_length, target_idx])
        else:
            # 多步预测
            for i in range(len(data_values) - self.sequence_length - self.prediction_horizon + 1):
                X.append(data_values[i:i + self.sequence_length])
                y.append(data_values[
                             i + self.sequence_length:i + self.sequence_length + self.prediction_horizon, target_idx])

        return np.array(X), np.array(y)

    def inverse_transform(self, data):
        """将归一化数据反归一化"""
        dummy_data = np.zeros((len(data), len(self.data.columns)))
        target_idx = list(self.data.columns).index(self.target_column)
        dummy_data[:, target_idx] = data

        inverted = self.scaler.inverse_transform(dummy_data)
        return inverted[:, target_idx]

    def get_train_data(self):
        """获取训练数据"""
        X_train = torch.FloatTensor(self.train_sequences[0])
        y_train = torch.FloatTensor(self.train_sequences[1])
        if self.mode == 'single_step' and y_train.dim() == 1:
            y_train = y_train.unsqueeze(1)
        return X_train, y_train

    def get_test_data(self):
        """获取测试数据"""
        X_test = torch.FloatTensor(self.test_sequences[0])
        y_test = torch.FloatTensor(self.test_sequences[1])
        if self.mode == 'single_step' and y_test.dim() == 1:
            y_test = y_test.unsqueeze(1)
        return X_test, y_test


# 数据加载器类
class TongjiDataLoader:
    def __init__(self, batch_size=32, shuffle=True, **dataset_args):
        self.dataset = TongjiDataset(**dataset_args)
        self.batch_size = batch_size

        X_train, y_train = self.dataset.get_train_data()
        X_test, y_test = self.dataset.get_test_data()

        self.train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=batch_size, shuffle=shuffle)
        self.test_loader = DataLoader(list(zip(X_test, y_test)), batch_size=batch_size, shuffle=False)

    def get_loaders(self):
        return {'train': self.train_loader, 'test': self.test_loader}

    def inverse_transform(self, data):
        return self.dataset.inverse_transform(data)


# 简单的LSTM模型
class SimpleLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(SimpleLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        output = self.fc(self.dropout(last_output))
        return output


# 训练器类
class Trainer:
    def __init__(self, model, train_loader, test_loader, inverse_transform_func, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.inverse_transform = inverse_transform_func
        self.device = device

        # 将模型移动到设备
        self.model.to(device)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)

        self.train_losses = []
        self.test_losses = []

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            # 将数据移动到设备
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss

    def evaluate(self):
        """在测试集上评估模型"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for data, target in self.test_loader:
                # 将数据移动到设备
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()

                # 将数据移回CPU进行后续处理
                all_predictions.extend(output.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        avg_loss = total_loss / len(self.test_loader)
        self.test_losses.append(avg_loss)

        # 转换为numpy数组
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        # 反归一化
        predictions_original = self.inverse_transform(all_predictions.flatten())
        targets_original = self.inverse_transform(all_targets.flatten())

        return avg_loss, predictions_original, targets_original

    def train(self, epochs=10):
        """训练模型"""
        print("开始训练...")

        for epoch in range(epochs):
            train_loss = self.train_epoch()
            test_loss, _, _ = self.evaluate()

            if (epoch + 1) % 5 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}')

        print("训练完成!")


def calculate_metrics(predictions, targets):
    """计算评估指标"""
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    mape = np.mean(np.abs((targets - predictions) / targets)) * 100

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2
    }


def plot_results(train_losses, test_losses, predictions, targets, metrics):
    """绘制结果"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. 训练损失曲线
    axes[0, 0].plot(train_losses, label='Train Loss')
    axes[0, 0].plot(test_losses, label='Test Loss')
    axes[0, 0].set_title('Training and Test Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 预测 vs 真实值散点图
    axes[0, 1].scatter(targets, predictions, alpha=0.5)
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    axes[0, 1].set_xlabel('True Values')
    axes[0, 1].set_ylabel('Predictions')
    axes[0, 1].set_title(f'Predictions vs True Values (R2 = {metrics["R2"]:.4f})')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 预测值和真实值的时间序列对比
    sample_size = min(200, len(predictions))
    axes[1, 0].plot(range(sample_size), targets[:sample_size], label='True', linewidth=1)
    axes[1, 0].plot(range(sample_size), predictions[:sample_size], label='Predicted', linewidth=1, alpha=0.7)
    axes[1, 0].set_xlabel('Sample Index')
    axes[1, 0].set_ylabel('Voltage (V)')
    axes[1, 0].set_title('Predictions vs True Values Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. 误差分布直方图
    errors = predictions - targets
    axes[1, 1].hist(errors, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Prediction Error')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Error Distribution Histogram')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    """主函数：完整的训练和测试流程"""
    print("=== PEMFC Voltage Prediction Demo ===")

    # 检查GPU可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    if torch.cuda.is_available():
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")

    # 1. 加载数据
    print("\n1. Loading data...")
    loader = TongjiDataLoader(
        batch_size=32,
        sequence_length=50,  # 使用较短的序列加快训练
        prediction_horizon=1,
        mode='single_step',
        train_ratio=0.7
    )

    # 获取数据加载器
    loaders = loader.get_loaders()
    train_loader = loaders['train']
    test_loader = loaders['test']

    # 检查数据形状
    X_sample, y_sample = next(iter(train_loader))
    print(f"Input shape: {X_sample.shape}")
    print(f"Output shape: {y_sample.shape}")

    # 2. 创建模型
    print("\n2. Creating model...")
    input_size = X_sample.shape[2]  # 特征数量
    output_size = y_sample.shape[1]  # 输出维度

    model = SimpleLSTMModel(
        input_size=input_size,
        hidden_size=64,
        output_size=output_size,
        num_layers=2
    )

    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")

    # 3. 训练模型
    print("\n3. Training model...")
    trainer = Trainer(model, train_loader, test_loader, loader.inverse_transform, device)
    trainer.train(epochs=20)  # 减少训练轮数以加快演示

    # 4. 最终评估
    print("\n4. Model evaluation...")
    test_loss, predictions, targets = trainer.evaluate()

    # 计算评估指标
    metrics = calculate_metrics(predictions, targets)

    print("\n=== Evaluation Results ===")
    print(f"Test MSE: {metrics['MSE']:.6f}")
    print(f"Test RMSE: {metrics['RMSE']:.6f}")
    print(f"Test MAE: {metrics['MAE']:.6f}")
    print(f"Test MAPE: {metrics['MAPE']:.2f}%")
    print(f"Test R²: {metrics['R2']:.4f}")

    # 5. 可视化结果
    print("\n5. Generating visualization...")
    plot_results(trainer.train_losses, trainer.test_losses, predictions, targets, metrics)

    print("\n=== Demo Completed ===")


if __name__ == "__main__":
    main()
