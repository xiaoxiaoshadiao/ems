import os
import math
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pywt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ==============================
# 配置（集中可调参数）
# ==============================
CONFIG = {
    # 数据相关
    "data_path": 'D:/xiaoxiaoshadiao/ems\data\processed/tongji/Durability_test_dataset\classified_current_data/14.85A_representative_rows.csv',
    "train_ratio": 0.7,

    # 去噪（小波）
    "wavelet": "db3",
    "wavelet_level": 4,
    "wavelet_alpha": 0.5,

    # 序列滑窗与预测长度
    "seq_len": 100,
    "pred_len": 8,  # 改这里即可一键切换单步/多步

    # 批量与训练
    "batch_size": 64,
    "epochs": 5,
    "lr": 1e-3,

    # 模型结构（FAMM：多尺度分解 + 等距卷积 + 趋势GRU + 投影）
    "d_model": 64,
    "d_layers": 1,
    "dropout": 0.1,
    "decomp_kernel": [25],
    "conv_kernel": [6],
    "isometric_kernel": [4],

    # 设备与绘图
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "show_plots": True,
    "compare_samples": 500,
}


# ==============================
# 小波去噪（软阈值）
# ==============================
def wavelet_denoise(signal, wavelet='db3', level=4, alpha=0.5):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = alpha * sigma * np.sqrt(2 * np.log(len(signal)))
    coeffs_thresh = [coeffs[0]]
    for c in coeffs[1:]:
        coeffs_thresh.append(pywt.threshold(c, threshold, mode='soft'))
    denoised = pywt.waverec(coeffs_thresh, wavelet)
    return denoised[:len(signal)]


# ==============================
# 数据集（滑窗生成多步标签）
# ==============================
class TongjiDataset(Dataset):
    def __init__(self, data_path=None, sequence_length=100, pred_len=1, train_ratio=0.7,
                 wavelet='db3', wavelet_level=4, wavelet_alpha=0.5):
        if data_path is None:
            data_path = os.path.join(os.path.dirname(__file__), 'data', 'processed', 'tongji',
                                     'Durability_test_dataset', 'classified_current_data',
                                     'all_representative_rows.csv')

        df = pd.read_csv(data_path)
        # 电压置于首列，便于反变换
        df = df[['voltage', 'power', 'current', 'temp_anode_endplate', 'pressure_anode_outlet',
                 'temp_cathode_outlet', 'temp_cathode_inlet', 'temp_anode_outlet']]
        df.columns = ['V', 'P', 'I', 'Pao', 'Tco', 'Tcin', 'Tao', 'Tan']

        # 小波去噪（逐列）
        for c in df.columns:
            df[c] = wavelet_denoise(df[c].values, wavelet=wavelet, level=wavelet_level, alpha=wavelet_alpha)

        # 时序切分（避免信息泄漏）
        train_size = int(len(df) * train_ratio)
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]

        # 标准化（仅训练集拟合）
        self.scaler = StandardScaler()
        train_scaled = pd.DataFrame(self.scaler.fit_transform(train_df), columns=df.columns)
        test_scaled = pd.DataFrame(self.scaler.transform(test_df), columns=df.columns)

        self.train_X, self.train_y = self._make_seq(train_scaled, sequence_length, pred_len)
        self.test_X, self.test_y = self._make_seq(test_scaled, sequence_length, pred_len)
        self.pred_len = pred_len

    def _make_seq(self, df, seq_len, pred_len):
        data = df.values
        X, y = [], []
        max_i = len(data) - (seq_len + pred_len)
        for i in range(max_i):
            X.append(data[i:i + seq_len])  # [seq_len, F]
            y.append(data[i + seq_len:i + seq_len + pred_len, 0])  # 仅电压通道
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        return torch.from_numpy(X), torch.from_numpy(y)  # y: [B, pred_len]

    def inverse_voltage(self, x_flat):
        dummy = np.zeros((len(x_flat), 8), dtype=np.float32)
        dummy[:, 0] = x_flat
        return self.scaler.inverse_transform(dummy)[:, 0]


# ==============================
# 模型组件（多尺度分解 + 等距卷积 + 投影）
# ==============================
class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x: [B, L, C]
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        mean = self.moving_avg(x)
        res = x - mean
        return res, mean


class series_decomp_multi(nn.Module):
    def __init__(self, kernel_size_list):
        super().__init__()
        self.moving_avg = nn.ModuleList([moving_avg(k, stride=1) for k in kernel_size_list])

    def forward(self, x):
        means, ress = [], []
        for f in self.moving_avg:
            m = f(x)
            means.append(m)
            ress.append(x - m)
        sea = sum(ress) / len(ress)  # 局部项（短期波动）
        mean = sum(means) / len(means)  # 背景项（趋势+恢复）
        return sea, mean


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate=0.1):
        super().__init__()
        self.l1 = nn.Linear(hidden_size, filter_size)
        self.l2 = nn.Linear(filter_size, hidden_size)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout_rate)
        nn.init.xavier_uniform_(self.l1.weight);
        nn.init.constant_(self.l1.bias, 0)
        nn.init.xavier_uniform_(self.l2.weight);
        nn.init.constant_(self.l2.bias, 0)

    def forward(self, x):
        return self.l2(self.drop(self.relu(self.l1(x))))


class MIC(nn.Module):
    # 等距卷积块（ICB）：下采样-等距卷积-上采样，并行多尺度
    def __init__(self, feature_size=128, dropout=0.05, decomp_kernel=[25],
                 conv_kernel=[8], isometric_kernel=[6], device='cuda'):
        super().__init__()
        self.device = device
        self.conv = nn.ModuleList([
            nn.Conv1d(feature_size, feature_size, kernel_size=k, padding=k // 2, stride=k) for k in conv_kernel
        ])
        self.conv_trans = nn.ModuleList([
            nn.ConvTranspose1d(feature_size, feature_size, kernel_size=k, padding=0, stride=k) for k in conv_kernel
        ])
        self.iso = nn.ModuleList([
            nn.Conv1d(feature_size, feature_size, kernel_size=k, padding=0, stride=1) for k in isometric_kernel
        ])
        self.decomp = nn.ModuleList([series_decomp(k) for k in decomp_kernel])
        self.merge = nn.Conv2d(feature_size, feature_size, kernel_size=(len(conv_kernel), 1))
        self.fnn = FeedForwardNetwork(feature_size, feature_size * 4, dropout)
        self.fnn_norm = nn.LayerNorm(feature_size)
        self.norm = nn.LayerNorm(feature_size)
        self.act = nn.Tanh()
        self.drop = nn.Dropout(dropout)

    def _icb_once(self, input, c_down, c_up, c_iso):
        # input: [B, L, C]
        B, L, C = input.shape
        x = input.permute(0, 2, 1)  # [B, C, L]
        x1 = self.drop(self.act(c_down(x)))  # 下采样
        x = x1
        zeros = torch.zeros((x.shape[0], x.shape[1], max(x.shape[2] - 1, 0)), device=self.device)
        x = torch.cat((zeros, x), dim=-1) if zeros.shape[-1] > 0 else x
        x = self.drop(self.act(c_iso(x)))  # 等距卷积（同尺度全局相关）
        min_len = min(x.shape[2], x1.shape[2])
        x, x1 = x[:, :, :min_len], x1[:, :, :min_len]
        x = self.norm((x + x1).permute(0, 2, 1)).permute(0, 2, 1)
        x = self.drop(self.act(c_up(x)))  # 上采样
        x = x[:, :, :L]
        x = self.norm(x.permute(0, 2, 1) + input)  # 跳跃融合（稳定梯度）
        return x

    def forward(self, src):
        outs = []
        for i in range(len(self.conv)):
            sea, _ = self.decomp[i](src)  # 用局部项驱动卷积特征提取
            outs.append(self._icb_once(sea, self.conv[i], self.conv_trans[i], self.iso[i]))
        mg = torch.cat([o.unsqueeze(1) for o in outs], dim=1)  # [B, S, L, C]
        mg = self.merge(mg.permute(0, 3, 1, 2)).squeeze(-2).permute(0, 2, 1)  # 通道-时间联合压缩
        return self.fnn_norm(mg + self.fnn(mg))  # 前馈反馈（非线性校正）


class Seasonal_Prediction(nn.Module):
    # 多层 ICB 堆叠 + 投影到原始特征通道
    def __init__(self, embedding_size=128, dropout=0.05, d_layers=1,
                 decomp_kernel=[25], c_out=8, conv_kernel=[8], isometric_kernel=[6], device='cuda'):
        super().__init__()
        self.mic = nn.ModuleList([
            MIC(feature_size=embedding_size, dropout=dropout, decomp_kernel=decomp_kernel,
                conv_kernel=conv_kernel, isometric_kernel=isometric_kernel, device=device)
            for _ in range(d_layers)
        ])
        self.proj = nn.Linear(embedding_size, c_out)

    def forward(self, x):
        for m in self.mic:
            x = m(x)
        return self.proj(x)


class PositionalEmbedding(nn.Module):
    # 位置编码（PE）
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    # 数值编码（VE）
    def __init__(self, c_in, d_model):
        super().__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.conv = nn.Conv1d(c_in, d_model, kernel_size=3, padding=padding, padding_mode='circular')
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        return self.conv(x.permute(0, 2, 1)).transpose(1, 2)


class FAMM(nn.Module):
    def __init__(self, dec_in, seq_len, out_len=1, d_model=128, d_layers=1, dropout=0.1,
                 decomp_kernel=[25], conv_kernel=[8], isometric_kernel=[6], device='cuda'):
        super().__init__()
        self.pred_len = out_len
        self.decomp_multi = series_decomp_multi(decomp_kernel)
        self.value_emb = TokenEmbedding(dec_in, d_model)
        self.pos_emb = PositionalEmbedding(d_model)
        self.drop = nn.Dropout(dropout)
        self.conv_trans = Seasonal_Prediction(embedding_size=d_model, dropout=dropout, d_layers=d_layers,
                                              decomp_kernel=decomp_kernel, c_out=dec_in,
                                              conv_kernel=conv_kernel, isometric_kernel=isometric_kernel, device=device)
        # 背景趋势通道：GRU
        self.trend_gru = nn.GRU(input_size=dec_in, hidden_size=d_model, batch_first=True)
        self.trend_proj = nn.Linear(d_model, dec_in)
        # 门控融合
        self.gate_fc = nn.Linear(dec_in * 2, dec_in)
        # 投影到电压
        self.final_fc = nn.Linear(dec_in, 1)

    def forward(self, x):
        seasonal, trend = self.decomp_multi(x)
        trend_out, _ = self.trend_gru(trend)
        trend_out = self.trend_proj(trend_out)
        trend_out = trend_out[:, -self.pred_len:, :]

        dec = self.value_emb(seasonal) + self.pos_emb(seasonal)
        dec = self.drop(dec)
        dec = self.conv_trans(dec)
        dec = dec[:, -self.pred_len:, :]

        # 门控融合
        fusion_input = torch.cat([dec, trend_out], dim=-1)
        gate = torch.sigmoid(self.gate_fc(fusion_input))
        dec = gate * dec + (1 - gate) * trend_out

        # 投影到电压
        dec = self.final_fc(dec).squeeze(-1)
        return dec


# ==============================
# 训练器（使用库函数评估指标）
# ==============================
class Trainer:
    def __init__(self, model, train_loader, test_loader, inverse_fn, device, lr):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.inverse = inverse_fn
        self.opt = optim.Adam(model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.device = device
        self.train_loss = []
        self.test_loss = []
        self.test_metrics = []  # (MAE, MAPE, RMSE, R2)

    @staticmethod
    def mape(y_true, y_pred, eps=1e-8):
        # sklearn 没有内置 MAPE；做除零保护
        denom = np.where(np.abs(y_true) < eps, eps, np.abs(y_true))
        return float(100.0 * np.mean(np.abs((y_pred - y_true) / denom)))

    def _step(self, batch, train=True):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)  # y: [B, pred_len]
        if train:
            self.opt.zero_grad()
        pred = self.model(x)  # [B, pred_len]
        loss = self.loss_fn(pred, y)
        if train:
            loss.backward()
            self.opt.step()
        return loss.item(), pred.detach().cpu().numpy(), y.detach().cpu().numpy()

    def train_epoch(self):
        self.model.train()
        total = 0.0
        for batch in self.train_loader:
            loss, _, _ = self._step(batch, train=True)
            total += loss
        avg = total / len(self.train_loader)
        self.train_loss.append(avg)
        return avg

    def eval_epoch(self):
        self.model.eval()
        total = 0.0
        preds_all, tars_all = [], []
        with torch.no_grad():
            for batch in self.test_loader:
                loss, preds, tars = self._step(batch, train=False)
                total += loss
                preds_all.append(preds)
                tars_all.append(tars)
        avg = total / len(self.test_loader)
        self.test_loss.append(avg)

        # 展平序列，并反变换到原始电压尺度
        preds_all = np.concatenate(preds_all, axis=0).reshape(-1)
        tars_all = np.concatenate(tars_all, axis=0).reshape(-1)
        preds_inv = self.inverse(preds_all)
        tars_inv = self.inverse(tars_all)

        mae = mean_absolute_error(tars_inv, preds_inv)
        rmse = mean_squared_error(tars_inv, preds_inv)
        mape = self.mape(tars_inv, preds_inv)
        r2 = r2_score(tars_inv, preds_inv)
        self.test_metrics.append((mae, mape, rmse, r2))
        return avg, mae, mape, rmse, r2

    def train(self, epochs):
        for e in range(1, epochs + 1):
            t0 = time.time()
            train_l = self.train_epoch()
            test_l, mae, mape, rmse, r2 = self.eval_epoch()
            print(f"第 {e:03d} 轮 | 训练损失 {train_l:.6f} | 测试损失 {test_l:.6f} | "
                  f"MAE {mae:.6f} | MAPE {mape:.3f}% | RMSE {rmse:.6f} | R² {r2:.6f} | "
                  f"用时 {time.time() - t0:.2f}s")

    def final_eval(self):
        # 返回用于绘图的完整预测与标签（原始尺度）
        self.model.eval()
        preds_all, tars_all = [], []
        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(self.device)
                preds = self.model(x).cpu().numpy()
                tars = y.cpu().numpy()
                preds_all.append(preds)
                tars_all.append(tars)
        preds_all = np.concatenate(preds_all, axis=0).reshape(-1)
        tars_all = np.concatenate(tars_all, axis=0).reshape(-1)
        preds_inv = self.inverse(preds_all)
        tars_inv = self.inverse(tars_all)

        mae = mean_absolute_error(tars_inv, preds_inv)
        rmse = mean_squared_error(tars_inv, preds_inv)
        mape = self.mape(tars_inv, preds_inv)
        r2 = r2_score(tars_inv, preds_inv)
        return preds_inv, tars_inv, mae, mape, rmse, r2


# ==============================
# 主函数
# ==============================
def main(cfg=CONFIG):
    device = torch.device(cfg["device"])
    ds = TongjiDataset(
        data_path=cfg["data_path"],
        sequence_length=cfg["seq_len"],
        pred_len=cfg["pred_len"],
        train_ratio=cfg["train_ratio"],
        wavelet=cfg["wavelet"],
        wavelet_level=cfg["wavelet_level"],
        wavelet_alpha=cfg["wavelet_alpha"],
    )

    train_loader = DataLoader(list(zip(ds.train_X, ds.train_y)), batch_size=cfg["batch_size"], shuffle=True)
    test_loader = DataLoader(list(zip(ds.test_X, ds.test_y)), batch_size=cfg["batch_size"], shuffle=False)

    dec_in = ds.train_X.shape[-1]
    model = FAMM(
        dec_in=dec_in, seq_len=cfg["seq_len"], out_len=cfg["pred_len"],
        d_model=cfg["d_model"], d_layers=cfg["d_layers"], dropout=cfg["dropout"],
        decomp_kernel=cfg["decomp_kernel"], conv_kernel=cfg["conv_kernel"],
        isometric_kernel=cfg["isometric_kernel"], device=device.type
    )

    trainer = Trainer(model, train_loader, test_loader, ds.inverse_voltage, device, lr=cfg["lr"])
    trainer.train(cfg["epochs"])

    preds, tars, mae, mape, rmse, r2 = trainer.final_eval()
    print("\n最终指标（原始电压尺度）：")
    print(f"MAE:  {mae:.6f}")
    print(f"MAPE: {mape:.3f}%")
    print(f"RMSE: {rmse:.6f}")
    print(f"R²:   {r2:.6f}")

    # 图 1：训练/测试损失曲线
    plt.figure(figsize=(10, 3))
    plt.plot(trainer.train_loss, label='训练损失', linewidth=1.5)
    plt.plot(trainer.test_loss, label='测试损失', linewidth=1.5)
    plt.xlabel('轮次');
    plt.ylabel('损失');
    plt.title('训练与测试损失曲线')
    plt.legend();
    plt.grid(True, alpha=0.3)
    if cfg["show_plots"]: plt.show()

    # 图 2：真实 vs 预测
    n = min(cfg["compare_samples"], len(preds))
    plt.figure(figsize=(12, 4))
    plt.plot(range(n), tars[:n], label='真实电压', linewidth=1.5)
    plt.plot(range(n), preds[:n], label='预测电压', linewidth=1.5, alpha=0.8)
    plt.xlabel('样本索引');
    plt.ylabel('电压 (V)')
    plt.title(f'真实 vs 预测（预测步长={cfg["pred_len"]}）')
    plt.legend();
    plt.grid(True, alpha=0.3)
    if cfg["show_plots"]: plt.show()


if __name__ == "__main__":
    main()
