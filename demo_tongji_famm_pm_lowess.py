import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import os
import time
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score


# ======================
# 数据集
# ======================
class TongjiDataset(Dataset):
    def __init__(self, data_path=None, sequence_length=100, train_ratio=0.7):
        if data_path is None:
            data_path = os.path.join(os.path.dirname(__file__), 'data', 'processed', 'tongji',
                                     'Durability_test_dataset',
                                     'classified_current_data', 'all_representative_rows.csv')

        df = pd.read_csv(data_path)

        # 仅保留需要的列，且 voltage 在第一列
        df = df[['voltage', 'power', 'current', 'temp_anode_endplate', 'pressure_anode_outlet',
                 'temp_cathode_outlet', 'temp_cathode_inlet', 'temp_anode_outlet']]

        df.columns = ['V', 'P', 'I', 'T', 'Pao', 'Tco', 'Tcin', 'Tao']

        # LOWESS 平滑
        from statsmodels.nonparametric.smoothers_lowess import lowess
        frac = 25 / len(df)
        idx = np.arange(len(df))
        for c in df.columns:
            df[c] = lowess(df[c].values, idx, frac=frac, it=0, return_sorted=False)

        # 切分训练/测试
        train_size = int(len(df) * train_ratio)
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]

        self.scaler = StandardScaler()
        train_scaled = pd.DataFrame(self.scaler.fit_transform(train_df), columns=df.columns)
        test_scaled = pd.DataFrame(self.scaler.transform(test_df), columns=df.columns)

        self.train_X, self.train_y = self.make_seq(train_scaled, sequence_length)
        self.test_X, self.test_y = self.make_seq(test_scaled, sequence_length)

    def make_seq(self, df, seq_len):
        data = df.values
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i + seq_len])  # [seq_len, 8]
            y.append(data[i + seq_len, 0])  # 下一时刻电压（第一列）
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        return torch.from_numpy(X), torch.from_numpy(y).unsqueeze(1)

    def inverse(self, x):
        dummy = np.zeros((len(x), 8))
        dummy[:, 0] = x
        return self.scaler.inverse_transform(dummy)[:, 0]


# ======================
# 模型基础组件
# ======================
class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class series_decomp_multi(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.kernel_size = kernel_size
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]

    def forward(self, x):
        moving_mean = []
        res = []
        for func in self.moving_avg:
            m = func(x)
            moving_mean.append(m)
            res.append(x - m)
        sea = sum(res) / len(res)
        moving_mean = sum(moving_mean) / len(moving_mean)
        return sea, moving_mean


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.layer1 = nn.Linear(hidden_size, filter_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(filter_size, hidden_size)
        nn.init.xavier_uniform_(self.layer1.weight);
        nn.init.constant_(self.layer1.bias, 0)
        nn.init.xavier_uniform_(self.layer2.weight);
        nn.init.constant_(self.layer2.bias, 0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x


class MIC(nn.Module):
    def __init__(self, feature_size=128, n_heads=4, dropout=0.05, decomp_kernel=[25], conv_kernel=[8],
                 isometric_kernel=[6], device='cuda'):
        super(MIC, self).__init__()
        self.conv_kernel = conv_kernel
        self.isometric_kernel = isometric_kernel
        self.device = device

        self.isometric_conv = nn.ModuleList([
            nn.Conv1d(in_channels=feature_size, out_channels=feature_size, kernel_size=i, padding=0, stride=1)
            for i in isometric_kernel
        ])

        self.conv = nn.ModuleList([
            nn.Conv1d(in_channels=feature_size, out_channels=feature_size, kernel_size=i, padding=i // 2, stride=i)
            for i in conv_kernel
        ])

        self.conv_trans = nn.ModuleList([
            nn.ConvTranspose1d(in_channels=feature_size, out_channels=feature_size, kernel_size=i, padding=0, stride=i)
            for i in conv_kernel
        ])

        self.decomp = nn.ModuleList([series_decomp(k) for k in decomp_kernel])
        self.merge = nn.Conv2d(in_channels=feature_size, out_channels=feature_size,
                               kernel_size=(len(self.conv_kernel), 1))

        self.fnn = FeedForwardNetwork(feature_size, feature_size * 4, dropout)
        self.fnn_norm = nn.LayerNorm(feature_size)

        self.norm = nn.LayerNorm(feature_size)
        self.act = nn.Tanh()
        self.drop = nn.Dropout(dropout)

    def conv_trans_conv(self, input, conv1d, conv1d_trans, isometric):
        batch, seq_len, channel = input.shape
        x = input.permute(0, 2, 1)

        # downsample
        x1 = self.drop(self.act(conv1d(x)))
        x = x1

        # isometric conv
        zeros = torch.zeros((x.shape[0], x.shape[1], max(x.shape[2] - 1, 0)), device=self.device)
        x = torch.cat((zeros, x), dim=-1) if zeros.shape[-1] > 0 else x
        x = self.drop(self.act(isometric(x)))

        # align length before residual add
        min_len = min(x.shape[2], x1.shape[2])
        x = x[:, :, :min_len]
        x1 = x1[:, :, :min_len]

        x = self.norm((x + x1).permute(0, 2, 1)).permute(0, 2, 1)

        # upsample and truncate to original seq_len
        x = self.drop(self.act(conv1d_trans(x)))
        x = x[:, :, :seq_len]
        x = self.norm(x.permute(0, 2, 1) + input)
        return x

    def forward(self, src):
        multi = []
        for i in range(len(self.conv_kernel)):
            src_out, _ = self.decomp[i](src)
            src_out = self.conv_trans_conv(src_out, self.conv[i], self.conv_trans[i], self.isometric_conv[i])
            multi.append(src_out)

        # merge multi-scale
        mg = torch.tensor([], device=self.device)
        for i in range(len(self.conv_kernel)):
            mg = torch.cat((mg, multi[i].unsqueeze(1)), dim=1)
        mg = self.merge(mg.permute(0, 3, 1, 2)).squeeze(-2).permute(0, 2, 1)
        return self.fnn_norm(mg + self.fnn(mg))


class Seasonal_Prediction(nn.Module):
    def __init__(self, embedding_size=128, n_heads=4, dropout=0.05, d_layers=1, decomp_kernel=[25], c_out=8,
                 conv_kernel=[8], isometric_kernel=[6], device='cuda'):
        super(Seasonal_Prediction, self).__init__()
        self.mic = nn.ModuleList([
            MIC(feature_size=embedding_size, n_heads=n_heads, dropout=dropout,
                decomp_kernel=decomp_kernel, conv_kernel=conv_kernel,
                isometric_kernel=isometric_kernel, device=device)
            for _ in range(d_layers)
        ])
        self.projection = nn.Linear(embedding_size, c_out)

    def forward(self, dec):
        for mic_layer in self.mic:
            dec = mic_layer(dec)
        return self.projection(dec)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular')
        nn.init.kaiming_normal_(self.tokenConv.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


# ======================
# FAMM 模型 简化输入仅 x
# ======================
class FAMM(nn.Module):
    def __init__(self, dec_in, seq_len, out_len=1, d_model=128, n_heads=4, d_layers=1, dropout=0.1, decomp_kernel=[25],
                 conv_kernel=[8], isometric_kernel=[6], device='cuda'):
        super(FAMM, self).__init__()
        self.pred_len = out_len
        self.seq_len = seq_len
        self.device = device

        # 分解
        self.decomp_multi = series_decomp_multi(decomp_kernel)

        # embedding（仅 value + position）
        self.value_embedding = TokenEmbedding(dec_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

        # MIC 多尺度卷积编码 + projection 到 dec_in 通道
        self.conv_trans = Seasonal_Prediction(
            embedding_size=d_model, n_heads=n_heads, dropout=dropout,
            d_layers=d_layers, decomp_kernel=decomp_kernel, c_out=dec_in,
            conv_kernel=conv_kernel, isometric_kernel=isometric_kernel, device=device
        )

        # 趋势回归：seq_len -> out_len
        self.regression = nn.Linear(seq_len, out_len)
        self.regression.weight = nn.Parameter((1 / out_len) * torch.ones([out_len, seq_len]), requires_grad=True)

        # 投影到单一输出
        self.final_fc = nn.Linear(dec_in, 1)

    def forward(self, x):
        # 分解为季节项 + 趋势
        seasonal_init_enc, trend = self.decomp_multi(x)
        trend = self.regression(trend.permute(0, 2, 1)).permute(0, 2, 1)

        # embedding
        dec_out = self.value_embedding(seasonal_init_enc) + self.position_embedding(seasonal_init_enc)
        dec_out = self.dropout(dec_out)

        # MIC 编码 + projection 到 dec_in 通道
        dec_out = self.conv_trans(dec_out)

        # 取最后 pred_len 步并加上趋势
        dec_out = dec_out[:, -self.pred_len:, :] + trend[:, -self.pred_len:, :]

        dec_out = self.final_fc(dec_out)  # [B, pred_len, 1]
        return dec_out.squeeze(-1)  # [B, pred_len]，pred_len=1 => [B, 1]


# ======================
# 训练器
# ======================
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
            pred = self.model(x)  # 仅输入 x
            loss = self.loss_fn(pred, y)  # [B,1]对齐
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
            test_l = results[0]  # 只取第一个 avg
            print(f"Epoch {e}: Train {train_l:.6f}, Test {test_l:.6f}, Time {time.time() - start:.2f}s")


# ======================
# 主函数
# ======================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = TongjiDataset(sequence_length=100)
    train_loader = DataLoader(list(zip(ds.train_X, ds.train_y)), batch_size=32, shuffle=True)
    test_loader = DataLoader(list(zip(ds.test_X, ds.test_y)), batch_size=32, shuffle=False)

    X_sample, y_sample = next(iter(train_loader))
    model = FAMM(dec_in=X_sample.shape[2], seq_len=X_sample.shape[1], out_len=1, d_model=64, n_heads=4, d_layers=1,
                 dropout=0.1, decomp_kernel=[25], conv_kernel=[6], isometric_kernel=[4], device=device)

    trainer = Trainer(model, train_loader, test_loader, ds.inverse, device)
    trainer.train(10)

    final_loss, preds, tars, mse, rmse, mae, r2 = trainer.eval()
    print(f"\nFinal MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}")

    # Loss 曲线
    plt.figure(figsize=(10, 3))
    plt.plot(trainer.train_loss, label='Train Loss', linewidth=1.5)
    plt.plot(trainer.test_loss, label='Test Loss', linewidth=1.5)
    plt.xlabel('Epoch');
    plt.ylabel('Loss')
    plt.title('Training & Test Loss Curve')
    plt.legend();
    plt.grid(True, alpha=0.3)
    plt.show()

    # 真实 vs 预测
    sample_size = min(500, len(preds))
    plt.figure(figsize=(12, 4))
    plt.plot(range(sample_size), tars[:sample_size], label='True Voltage', linewidth=1.5)
    plt.plot(range(sample_size), preds[:sample_size], label='Predicted Voltage', linewidth=1.5, alpha=0.8)
    plt.xlabel('Sample Index');
    plt.ylabel('Voltage (V)')
    plt.title('Tongji - True vs Predicted Voltage')
    plt.legend();
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()
