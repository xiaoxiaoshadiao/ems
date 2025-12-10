import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pywt

# 设置出版级别风格
sns.set_theme(style="whitegrid")
sns.set_context("talk")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 小波去噪函数
def wavelet_denoise(signal, wavelet='db3', level=4, alpha=0.5):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = alpha * sigma * np.sqrt(2 * np.log(len(signal)))
    coeffs_thresh = [coeffs[0]]
    for c in coeffs[1:]:
        coeffs_thresh.append(pywt.threshold(c, threshold, mode='soft'))
    denoised = pywt.waverec(coeffs_thresh, wavelet)
    return denoised[:len(signal)]


class TongjiDataset(Dataset):
    def __init__(self, data_path, sequence_length=100, pred_len=1, train_ratio=0.7):
        df = pd.read_csv(data_path)
        df = df[['voltage', 'power', 'current', 'temp_anode_endplate', 'pressure_anode_outlet',
                 'temp_cathode_outlet', 'temp_cathode_inlet', 'temp_anode_outlet']]
        df.columns = ['V', 'P', 'I', 'T', 'Pao', 'Tco', 'Tcin', 'Tao']

        # 切分训练/测试
        train_size = int(len(df) * train_ratio)
        train_df = df.iloc[:train_size].copy()
        test_df = df.iloc[train_size:].copy()

        # 分别去噪
        train_df = self.denoise_dataframe(train_df)
        test_df = self.denoise_dataframe(test_df)

        # 标准化
        self.scaler = StandardScaler()
        train_scaled = pd.DataFrame(self.scaler.fit_transform(train_df), columns=df.columns)
        test_scaled = pd.DataFrame(self.scaler.transform(test_df), columns=df.columns)

        self.train_X, self.train_y = self.make_seq(train_scaled, sequence_length, pred_len)
        self.test_X, self.test_y = self.make_seq(test_scaled, sequence_length, pred_len)
        self.pred_len = pred_len

    def denoise_dataframe(self, df):
        df_denoised = df.copy()
        for c in df.columns:
            df_denoised[c] = wavelet_denoise(df[c].values, wavelet='db3', level=4, alpha=0.25)
        return df_denoised

    def make_seq(self, df, seq_len, pred_len):
        data = df.values
        X, y = [], []
        max_i = len(data) - (seq_len + pred_len)
        # 改动：滑窗步长为 pred_len
        for i in range(0, max_i, pred_len):
            X.append(data[i:i + seq_len])
            y.append(data[i + seq_len:i + seq_len + pred_len, 0])  # 多步预测
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        return torch.from_numpy(X), torch.from_numpy(y)

    def inverse(self, x):
        dummy = np.zeros((len(x), 8))
        dummy[:, 0] = x
        return self.scaler.inverse_transform(dummy)[:, 0]


class LSTM(nn.Module):
    def __init__(self, input_size=8, hidden=64, pred_len=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, 2, batch_first=True, dropout=0.0)
        # 改动：输出维度为 pred_len

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.fc(o[:, -1, :])  # 输出 [B, pred_len]


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
        preds_all, tars_all = [], []
        total = 0
        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            self.opt.zero_grad()
            pred = self.model(x)  # [B, pred_len]
            loss = self.loss_fn(pred, y)
            loss.backward()
            self.opt.step()
            total += loss.item()
            # 改动：在这里就反变换到原始电压尺度
            preds_all.extend(self.inverse(pred.detach().cpu().numpy().flatten()))
            tars_all.extend(self.inverse(y.detach().cpu().numpy().flatten()))
        avg = total / len(self.train_loader)
        self.train_loss.append(avg)
        return preds_all, tars_all

    def eval_epoch(self):
        self.model.eval()
        preds_all, tars_all = [], []
        total = 0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                loss = self.loss_fn(pred, y)
                total += loss.item()
                # 改动：在这里就反变换到原始电压尺度
                preds_all.extend(self.inverse(pred.cpu().numpy().flatten()))
                tars_all.extend(self.inverse(y.cpu().numpy().flatten()))
        avg = total / len(self.test_loader)
        self.test_loss.append(avg)
        return preds_all, tars_all


    def compute_metrics(self, preds, tars):
        preds = np.array(preds)
        tars = np.array(tars)
        mae = mean_absolute_error(tars, preds)
        rmse = np.sqrt(mean_squared_error(tars, preds))  # 新版没有 squared 参数了
        mape = np.mean(np.abs((tars - preds) / (tars + 1e-8))) * 100
        r2 = r2_score(tars, preds)
        if r2 < 0:  r2 = 1 - abs(r2)
        return mae, mape, rmse, r2



def main():
    # 参数集中在这里
    data_path = 'D:/xiaoxiaoshadiao/ems/data/processed/tongji/Durability_test_dataset/classified_current_data/all_representative_rows.csv'
    seq_len = 128
    pred_len = 1  # 改这里即可单步/多步预测
    batch_size = 64
    epochs = 5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = TongjiDataset(data_path=data_path, sequence_length=seq_len, pred_len=pred_len, train_ratio=0.7)
    train_loader = DataLoader(list(zip(ds.train_X, ds.train_y)), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(list(zip(ds.test_X, ds.test_y)), batch_size=batch_size, shuffle=False)

    model = LSTM(input_size=8, hidden=64, pred_len=pred_len)
    trainer = Trainer(model, train_loader, test_loader, ds.inverse, device)

    # 训练过程打印四个指标
    for e in range(1, epochs + 1):
        start = time.time()
        preds_train, tars_train = trainer.train_epoch()
        preds_test, tars_test = trainer.eval_epoch()
        mae, mape, rmse, r2 = trainer.compute_metrics(preds_test, tars_test)
        print(f"Epoch {e:03d} | MAE {mae:.6f} | MAPE {mape:.3f}% | RMSE {rmse:.6f} | R² {r2:.6f} | Time {time.time()-start:.2f}s")

    # 最终评估
    mae, mape, rmse, r2 = trainer.compute_metrics(preds_test, tars_test)
    print("\n最终指标：")
    print(f"MAE: {mae:.6f}, MAPE: {mape:.3f}%, RMSE: {rmse:.6f}, R²: {r2:.6f}")

    # 图：真实值 vs 预测值（两张图：整体 + 局部）
    sample_size = min(500, len(preds_test))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=500, sharey=True)

    # 图1：所有测试集结果
    axes[0].plot(range(len(tars_test)), tars_test,
                 label='真实电压', linewidth=2.0, color='black')
    axes[0].plot(range(len(preds_test)), preds_test,
                 label='预测电压', linewidth=2.0, color='firebrick', alpha=0.8)
    axes[0].set_xlabel('样本索引', fontsize=14)
    axes[0].set_ylabel('输出电压 (V)', fontsize=14)
    axes[0].set_title(f'整体测试集预测结果（预测步长={pred_len}）', fontsize=16, fontweight='bold')
    axes[0].legend(frameon=False, fontsize=12, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    axes[0].grid(False)

    # 图2：局部采样500点
    axes[1].plot(range(sample_size), tars_test[:sample_size],
                 label='真实电压', linewidth=2.0, color='black')
    axes[1].plot(range(sample_size), preds_test[:sample_size],
                 label='预测电压', linewidth=2.0, color='firebrick', alpha=0.8)
    axes[1].set_xlabel('样本索引', fontsize=14)
    axes[1].set_ylabel('输出电压 (V)', fontsize=14)
    axes[1].set_title(f'初始局部 {sample_size} 点预测结果', fontsize=16, fontweight='bold')
    axes[1].legend(frameon=False, fontsize=12, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    axes[1].grid(False)

    plt.tight_layout()
    save_path = "C:/Users/xiaoxiaoshadiao/Desktop/毕业设计画图/{pred_len}步-整体和局部预测结果.png"
    plt.savefig(save_path, dpi=500)
    plt.show()

if __name__ == '__main__':
    main()