import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import os
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class IEEEDataset(Dataset):
    def __init__(self, data_path, sequence_length=50, train_ratio=0.7, target_column='V'):
        # 选择的列和重命名映射
        selected_columns = ['Utot (V)', 'J (A/cmｲ)', 'I (A)', 'TinH2 (ｰC)', 'ToutH2 (ｰC)', 'TinAIR (ｰC)',
                            'DoutH2 (l/mn)', 'DWAT (l/mn)']
        rename_mapping = {'Utot (V)': 'V', 'J (A/cmｲ)': 'J', 'I (A)': 'I', 'TinH2 (ｰC)': 'TinH2',
                          'TinAIR (ｰC)': 'ToutH2', 'TinAIR (°C)': 'TinAIR', 'DoutH2 (l/mn)': 'DoutH2',
                          'DWAT (l/mn)': 'Dwat'}

        df = pd.read_csv(data_path)
        df = df[selected_columns]
        df = df.rename(columns=rename_mapping)

        train_size = int(len(df) * train_ratio)
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]

        self.scaler = StandardScaler()
        train_scaled = pd.DataFrame(self.scaler.fit_transform(train_df), columns=df.columns)
        test_scaled = pd.DataFrame(self.scaler.transform(test_df), columns=df.columns)

        self.train_X, self.train_y = self.make_seq(train_scaled, sequence_length, target_column)
        self.test_X, self.test_y = self.make_seq(test_scaled, sequence_length, target_column)

        self.target_idx = list(df.columns).index(target_column)

    def make_seq(self, df, seq_len, target_column):
        data = df.values
        X, y = [], []
        target_idx = list(df.columns).index(target_column)

        for i in range(len(data) - seq_len):
            X.append(data[i:i + seq_len])
            y.append(data[i + seq_len, target_idx])

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        return torch.from_numpy(X), torch.from_numpy(y).unsqueeze(1)

    def inverse_transform(self, x):
        dummy = np.zeros((len(x), 8))
        dummy[:, self.target_idx] = x
        return self.scaler.inverse_transform(dummy)[:, self.target_idx]


class LSTM(nn.Module):
    def __init__(self, input_size=8, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, 2, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.fc(o[:, -1, :])


class Trainer:
    def __init__(self, model, train_loader, test_loader, inverse_transform_func, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.inverse_transform = inverse_transform_func
        self.opt = optim.Adam(model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.device = device
        self.train_loss = []
        self.test_loss = []

    def train_epoch(self):
        self.model.train()
        total = 0
        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            self.opt.zero_grad()
            pred = self.model(x)
            loss = self.loss_fn(pred, y)
            loss.backward()
            self.opt.step()
            total += loss.item()
        avg = total / len(self.train_loader)
        self.train_loss.append(avg)
        return avg

    def eval(self):
        self.model.eval()
        total = 0
        preds, tars = [], []
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                loss = self.loss_fn(pred, y)
                total += loss.item()
                preds.extend(pred.cpu().numpy())
                tars.extend(y.cpu().numpy())

        avg = total / len(self.test_loader)
        self.test_loss.append(avg)

        preds = np.array(preds).flatten()
        tars = np.array(tars).flatten()
        return avg, self.inverse_transform(preds), self.inverse_transform(tars)

    def train(self, epochs):
        for e in range(1, epochs + 1):
            start = time.time()
            train_l = self.train_epoch()
            test_l, _, _ = self.eval()
            print(f"Epoch {e}: Train {train_l:.6f}, Test {test_l:.6f}, Time {time.time() - start:.2f}s")


def calculate_metrics(predictions, targets):
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    mape = np.mean(np.abs((targets - predictions) / targets)) * 100

    return {
        'MSE': mse, 'RMSE': rmse, 'MAE': mae,
        'MAPE': mape, 'R2': r2
    }


def plot_results(train_losses, test_losses, predictions, targets, metrics, fc_name):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 训练损失
    axes[0, 0].plot(train_losses, label='Train Loss')
    axes[0, 0].plot(test_losses, label='Test Loss')
    axes[0, 0].set_title(f'{fc_name} - Training and Test Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 预测vs真实值
    axes[0, 1].scatter(targets, predictions, alpha=0.5)
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    axes[0, 1].set_xlabel('True Values')
    axes[0, 1].set_ylabel('Predictions')
    axes[0, 1].set_title(f'{fc_name} - Predictions vs True Values (R2 = {metrics["R2"]:.4f})')
    axes[0, 1].grid(True, alpha=0.3)

    # 时间序列对比
    sample_size = min(200, len(predictions))
    axes[1, 0].plot(range(sample_size), targets[:sample_size], label='True', linewidth=1)
    axes[1, 0].plot(range(sample_size), predictions[:sample_size], label='Predicted', linewidth=1, alpha=0.7)
    axes[1, 0].set_xlabel('Sample Index')
    axes[1, 0].set_ylabel('Voltage (V)')
    axes[1, 0].set_title(f'{fc_name} - Predictions vs True Values Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 误差分布
    errors = predictions - targets
    axes[1, 1].hist(errors, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Prediction Error')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title(f'{fc_name} - Error Distribution Histogram')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def train_single_model(file_path, fc_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    ds = IEEEDataset(file_path, sequence_length=50)
    train_loader = DataLoader(list(zip(ds.train_X, ds.train_y)), batch_size=32, shuffle=True)
    test_loader = DataLoader(list(zip(ds.test_X, ds.test_y)), batch_size=32, shuffle=False)

    model = LSTM(input_size=8)
    trainer = Trainer(model, train_loader, test_loader, ds.inverse_transform, device)

    trainer.train(20)

    test_loss, preds, tars = trainer.eval()
    metrics = calculate_metrics(preds, tars)

    print(f"\n=== {fc_name} Evaluation Results ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.6f}")

    plot_results(trainer.train_loss, trainer.test_loss, preds, tars, metrics, fc_name)

    return trainer, metrics


def main():
    base_path = r"D:\xiaoxiaoshadiao\ems\data\processed\ieee\Durability_test_dataset"
    fc1_path = os.path.join(base_path, "FC1_aging_durability_data.csv")
    fc2_path = os.path.join(base_path, "FC2_aging_durability_data.csv")

    # print("Training FC1 model...")
    # trainer_fc1, metrics_fc1 = train_single_model(fc1_path, "FC1")

    print("\nTraining FC2 model...")
    trainer_fc2, metrics_fc2 = train_single_model(fc2_path, "FC2")

    print("\n=== Model Comparison ===")
    # print(f"FC1 - R²: {metrics_fc1['R2']:.4f}, RMSE: {metrics_fc1['RMSE']:.6f}")
    print(f"FC2 - R²: {metrics_fc2['R2']:.4f}, RMSE: {metrics_fc2['RMSE']:.6f}")


if __name__ == "__main__":
    main()
