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
from statsmodels.nonparametric.smoothers_lowess import lowess


# 数据集
class TongjiDataset(Dataset):
    def __init__(self, data_path=None, sequence_length=100, train_ratio=0.7):
        if data_path is None:
            data_path = os.path.join(os.path.dirname(__file__), 'data', 'processed', 'tongji',
                                     'Durability_test_dataset',
                                     'classified_current_data', 'all_representative_rows.csv')

        df = pd.read_csv(data_path)
        df = df[['voltage', 'power', 'current', 'temp_anode_endplate', 'pressure_anode_outlet',
                 'temp_cathode_outlet', 'temp_cathode_inlet', 'temp_anode_outlet']]
        df.columns = ['V', 'P', 'I', 'T', 'Pao', 'Tco', 'Tcin', 'Tao']

        # 切分训练/测试
        train_size = int(len(df) * train_ratio)
        train_df = df.iloc[:train_size].copy()
        test_df = df.iloc[train_size:].copy()

        # 分别平滑，避免数据泄露
        train_df = self.smooth_dataframe(train_df)
        test_df = self.smooth_dataframe(test_df)

        # 标准化
        self.scaler = StandardScaler()
        train_scaled = pd.DataFrame(self.scaler.fit_transform(train_df), columns=df.columns)
        test_scaled = pd.DataFrame(self.scaler.transform(test_df), columns=df.columns)

        self.train_X, self.train_y = self.make_seq(train_scaled, sequence_length)
        self.test_X, self.test_y = self.make_seq(test_scaled, sequence_length)

    def smooth_dataframe(self, df):
        idx = np.arange(len(df))
        df_smoothed = df.copy()
        frac = min(1.0, 25 / len(df))  # 保证窗口约 25 点
        for c in df.columns:
            df_smoothed[c] = lowess(df[c].values, idx, frac=frac, it=0, return_sorted=False)
        return df_smoothed

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


# 配置
class Configs:
    def __init__(self):
        self.seq_len = 100
        self.pred_len = 1
        self.enc_in = 8
        self.d_model = 64
        self.d_ff = 128
        self.dropout = 0.1
        self.moving_avg = 25
        self.e_layers = 1


# 模块
class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride=1):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=1, padding_mode='circular', bias=False)
        nn.init.kaiming_normal_(self.tokenConv.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)


# RAMM 模型
class RAMM(nn.Module):
    def __init__(self, configs):
        super(RAMM, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.dropout = configs.dropout

        self.value_embedding = TokenEmbedding(c_in=self.enc_in, d_model=self.d_model)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.decomp = series_decomp(configs.moving_avg)
        self.trend_fc = nn.Linear(self.enc_in, self.d_model)
        self.final_fc = nn.Linear(self.d_model, 1)

    def forecast(self, x_enc):
        seasonal, trend = self.decomp(x_enc)

        enc_out = self.value_embedding(seasonal)  # [B, T, 64]
        enc_out = self.dropout_layer(enc_out)

        trend_proj = self.trend_fc(trend)  # [B, T, 64]

        dec_out = enc_out + trend_proj  # [B, T, 64]
        dec_out = self.final_fc(dec_out)  # [B, T, 1]

        return dec_out[:, -self.pred_len:, :]


# Trainer
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
            pred = self.model.forecast(x)
            loss = self.loss_fn(pred.squeeze(-1), y)
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
                pred = self.model.forecast(x)
                loss = self.loss_fn(pred.squeeze(-1), y)
                total += loss.item()
                preds.extend(pred.cpu().numpy())
                tars.extend(y.cpu().numpy())
        avg = total / len(self.test_loader)
        self.test_loss.append(avg)
        preds = np.array(preds).reshape(-1)
        tars = np.array(tars).reshape(-1)
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


# main
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds = TongjiDataset(sequence_length=100)
    train_loader = DataLoader(list(zip(ds.train_X, ds.train_y)), batch_size=32, shuffle=True)
    test_loader = DataLoader(list(zip(ds.test_X, ds.test_y)), batch_size=32, shuffle=False)

    configs = Configs()
    model = RAMM(configs)

    trainer = Trainer(model, train_loader, test_loader, ds.inverse, device)
    trainer.train(20)

    final_loss, preds, tars, mse, rmse, mae, r2 = trainer.eval()
    print(f"\nFinal MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}")

    # 1. 损失曲线
    plt.figure(figsize=(10, 3))
    plt.plot(trainer.train_loss, label='Train Loss', linewidth=1.5)
    plt.plot(trainer.test_loss, label='Test Loss', linewidth=1.5)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('RAMM - Training & Test Loss Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # 2. 真实值 vs 预测值
    sample_size = min(500, len(preds))
    plt.figure(figsize=(12, 4))
    plt.plot(range(sample_size), tars[:sample_size], label='True Voltage', linewidth=1.5)
    plt.plot(range(sample_size), preds[:sample_size], label='Predicted Voltage', linewidth=1.5, alpha=0.8)
    plt.xlabel('Sample Index')
    plt.ylabel('Voltage (V)')
    plt.title('RAMM - True vs Predicted Voltage')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()
