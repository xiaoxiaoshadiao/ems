import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import os
import time
import matplotlib.pyplot as plt  # 导入绘图库


class IEEEDataset(Dataset):
    def __init__(self, data_path, sequence_length=50, train_ratio=0.7, target_column='V'):
        # 变量筛选与重命名
        selected_columns = ['Utot (V)', 'J (A/cmｲ)', 'I (A)', 'TinH2 (ｰC)', 'ToutH2 (ｰC)', 'TinAIR (ｰC)',
                            'DoutH2 (l/mn)', 'DWAT (l/mn)']
        rename_mapping = {'Utot (V)': 'V', 'J (A/cmｲ)': 'J', 'I (A)': 'I', 'TinH2 (ｰC)': 'TinH2',
                          'ToutH2 (ｰC)': 'ToutH2', 'TinAIR (ｰC)': 'TinAIR', 'DoutH2 (l/mn)': 'DoutH2',
                          'DWAT (l/mn)': 'Dwat'}

        df = pd.read_csv(data_path)
        df = df[selected_columns].rename(columns=rename_mapping)

        # LOWESS平滑处理
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smooth_frac = 20 / len(df)
        idx = np.arange(len(df))
        for col in df.columns:
            df[col] = lowess(df[col].values, idx, frac=smooth_frac, it=0, return_sorted=False)

        # 划分训练/测试集并标准化
        train_size = int(len(df) * train_ratio)
        self.scaler = StandardScaler()
        train_scaled = self.scaler.fit_transform(df.iloc[:train_size])
        test_scaled = self.scaler.transform(df.iloc[train_size:])

        # 构建序列数据
        self.train_X, self.train_y = self.make_seq(train_scaled, sequence_length, df.columns.get_loc(target_column))
        self.test_X, self.test_y = self.make_seq(test_scaled, sequence_length, df.columns.get_loc(target_column))
        self.target_idx = df.columns.get_loc(target_column)

    def make_seq(self, data, seq_len, target_idx):
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i + seq_len])
            y.append(data[i + seq_len, target_idx])
        return torch.from_numpy(np.array(X, dtype=np.float32)), torch.from_numpy(
            np.array(y, dtype=np.float32)).unsqueeze(1)

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
    def __init__(self, model, train_loader, test_loader, inverse_func, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.inverse = inverse_func
        self.opt = optim.Adam(model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.device = device
        self.train_loss = []
        self.test_loss = []

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            self.opt.zero_grad()
            loss = self.loss_fn(self.model(x), y)
            loss.backward()
            self.opt.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(self.train_loader)
        self.train_loss.append(avg_loss)
        return avg_loss

    def eval(self):
        self.model.eval()
        total_loss = 0
        preds, tars = [], []
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                total_loss += self.loss_fn(pred, y).item()
                preds.extend(pred.cpu().numpy())
                tars.extend(y.cpu().numpy())
        avg_loss = total_loss / len(self.test_loader)
        self.test_loss.append(avg_loss)
        return avg_loss, self.inverse(np.array(preds).flatten()), self.inverse(np.array(tars).flatten())

    def train(self, epochs):
        for e in range(1, epochs + 1):
            start = time.time()
            train_l = self.train_epoch()
            test_l, _, _ = self.eval()
            print(f"Epoch {e}: Train {train_l:.6f}, Test {test_l:.6f}, Time {time.time() - start:.2f}s")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_path = os.path.join(r"D:\xiaoxiaoshadiao\ems\data\processed\ieee\Durability_test_dataset",
                             "FC2_aging_durability_data.csv")

    # 初始化数据集和加载器
    ds = IEEEDataset(data_path, sequence_length=50)
    train_loader = DataLoader(list(zip(ds.train_X, ds.train_y)), batch_size=32, shuffle=True)
    test_loader = DataLoader(list(zip(ds.test_X, ds.test_y)), batch_size=32, shuffle=False)

    # 训练模型
    model = LSTM()
    trainer = Trainer(model, train_loader, test_loader, ds.inverse_transform, device)
    trainer.train(20)

    # 最终评估
    final_loss, preds, tars = trainer.eval()
    mse = np.mean((preds - tars) ** 2)
    rmse = np.sqrt(mse)
    print(f"\nFinal Test Loss: {final_loss:.6f}, MSE: {mse:.6f}, RMSE: {rmse:.6f}")

    # 绘制真实值与预测值对比图（取前500个点更清晰）
    sample_size = min(500, len(preds))
    plt.figure(figsize=(12, 4))
    plt.plot(range(sample_size), tars[:sample_size], label='True Value', linewidth=1.5)
    plt.plot(range(sample_size), preds[:sample_size], label='Predicted Value', linewidth=1.5, alpha=0.8)
    plt.xlabel('Sample Index')
    plt.ylabel('Voltage (V)')
    plt.title('FC2 - True vs Predicted Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # 可选：绘制损失曲线
    plt.figure(figsize=(10, 3))
    plt.plot(trainer.train_loss, label='Train Loss')
    plt.plot(trainer.test_loss, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Test Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()