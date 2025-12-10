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
from sklearn.metrics import mean_absolute_error, r2_score
import pywt


# 小波去噪函数
def wavelet_denoise(signal, wavelet='db3', level=4, alpha=0.25):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = alpha * sigma * np.sqrt(2 * np.log(len(signal)))
    coeffs_thresh = [coeffs[0]]
    for c in coeffs[1:]:
        coeffs_thresh.append(pywt.threshold(c, threshold, mode='soft'))
    denoised = pywt.waverec(coeffs_thresh, wavelet)
    return denoised[:len(signal)]


class TongjiDataset(Dataset):
    def __init__(self, data_path=None, sequence_length=100, train_ratio=0.7):
        if data_path is None:
            data_path = os.path.join(os.path.dirname(__file__), 'data', 'processed',
                                     'tongji',
                                     'Durability_test_dataset',
                                     'classified_current_data',
                                     'all_representative_rows.csv')

        df = pd.read_csv(data_path)

        df = df[[
            'voltage', 'power', 'current', 'temp_anode_endplate', 'pressure_anode_outlet',
            'temp_cathode_outlet', 'temp_cathode_inlet', 'temp_anode_outlet'
        ]]
        df.columns = ['V', 'P', 'I', 'T', 'Pao', 'Tco', 'Tcin', 'Tao']

        # 切分训练/测试
        train_size = int(len(df) * train_ratio)
        train_df = df.iloc[:train_size].copy()
        test_df = df.iloc[train_size:].copy()

        # 分别去噪，避免数据泄露
        train_df = self.denoise_dataframe(train_df)
        test_df = self.denoise_dataframe(test_df)

        # 标准化
        self.scaler = StandardScaler()
        train_scaled = pd.DataFrame(self.scaler.fit_transform(train_df), columns=df.columns)
        test_scaled = pd.DataFrame(self.scaler.transform(test_df), columns=df.columns)

        self.train_X, self.train_y = self.make_seq(train_scaled, sequence_length)
        self.test_X, self.test_y = self.make_seq(test_scaled, sequence_length)

    def denoise_dataframe(self, df):
        """对单个数据集做小波去噪"""
        df_denoised = df.copy()
        for c in df.columns:
            df_denoised[c] = wavelet_denoise(df[c].values, wavelet='db3', level=4, alpha=0.25)
        return df_denoised

    def make_seq(self, df, seq_len):
        data = df.values
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i + seq_len])
            y.append(data[i + seq_len, 0])
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        return torch.from_numpy(X), torch.from_numpy(y).unsqueeze(1)

    def inverse(self, x):
        dummy = np.zeros((len(x), 8))
        dummy[:, 0] = x
        return self.scaler.inverse_transform(dummy)[:, 0]


class LSTM(nn.Module):
    def __init__(self, input_size=8, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, 2, batch_first=True, dropout=0.0)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.fc(o[:, -1, :])


class Trainer:
    def __init__(self, model, train_loader, test_loader, inverse, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.inverse = inverse
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

        # 计算额外指标
        mse = np.mean((preds - tars) ** 2)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(tars, preds)
        r2 = r2_score(tars, preds)

        return avg, self.inverse(preds), self.inverse(tars), mse, rmse, mae, r2

    def train(self, epochs):
        for e in range(1, epochs + 1):
            start = time.time()
            train_l = self.train_epoch()
            results = self.eval()
            test_l = results[0]
            print(f"Epoch {e}: Train {train_l:.6f}, Test {test_l:.6f}, Time {time.time() - start:.2f}s")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds = TongjiDataset(sequence_length=100)
    train_loader = DataLoader(list(zip(ds.train_X, ds.train_y)), batch_size=32, shuffle=True)
    test_loader = DataLoader(list(zip(ds.test_X, ds.test_y)), batch_size=32, shuffle=False)

    model = LSTM()
    trainer = Trainer(model, train_loader, test_loader, ds.inverse, device)

    trainer.train(20)

    final_loss, preds, tars, mse, rmse, mae, r2 = trainer.eval()
    print(f"\nFinal MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}")

    # 损失曲线
    plt.figure(figsize=(10, 3))
    plt.plot(trainer.train_loss, label='Train Loss', linewidth=1.5)
    plt.plot(trainer.test_loss, label='Test Loss', linewidth=1.5)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Test Loss Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # 真实 vs 预测
    sample_size = min(500, len(preds))
    plt.figure(figsize=(12, 4))
    plt.plot(range(sample_size), tars[:sample_size], label='True Voltage', linewidth=1.5)
    plt.plot(range(sample_size), preds[:sample_size], label='Predicted Voltage', linewidth=1.5, alpha=0.8)
    plt.xlabel('Sample Index')
    plt.ylabel('Voltage (V)')
    plt.title('Tongji - True vs Predicted Voltage')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()
