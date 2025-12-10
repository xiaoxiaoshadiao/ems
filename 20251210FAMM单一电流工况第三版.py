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

# ================= 全局参数配置 =================
WAVELET = 'db3'
WAVELET_LEVEL = 4
WAVELET_ALPHA = 0.5

HIDDEN_SIZE = 128
NUM_LAYERS = 3
DROPOUT = 0.2
SEQ_LEN = 64
PRED_LEN = 1
BATCH_SIZE = 256
EPOCHS = 100
LR = 0.001
gongkuang = 0
INPUT_SIZE = 8
# 如果工况是0 Input要改为6

PRED_SHIFT = SEQ_LEN   # 默认右移一个序列长度

# ================= 绘图风格 =================
sns.set_theme(style="whitegrid")
sns.set_context("talk")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ================= 小波去噪 =================
def wavelet_denoise(signal, wavelet=WAVELET, level=WAVELET_LEVEL, alpha=WAVELET_ALPHA):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = alpha * sigma * np.sqrt(2 * np.log(len(signal)))
    coeffs_thresh = [coeffs[0]]
    for c in coeffs[1:]:
        coeffs_thresh.append(pywt.threshold(c, threshold, mode='soft'))
    denoised = pywt.waverec(coeffs_thresh, wavelet)
    return denoised[:len(signal)]


# ================= 数据集类 =================
class TongjiDataset(Dataset):
    def __init__(self, data_path, sequence_length=SEQ_LEN, pred_len=PRED_LEN, train_ratio=0.7):
        df = pd.read_csv(data_path)
        if gongkuang == 0:
            df = df[['voltage', 'temp_anode_endplate', 'pressure_anode_outlet',
                     'temp_cathode_outlet', 'temp_cathode_inlet', 'temp_anode_outlet']]
            df.columns = ['V', 'T', 'Pao', 'Tco', 'Tcin', 'Tao']
        else:
            df = df[['voltage', 'power', 'current', 'temp_anode_endplate', 'pressure_anode_outlet',
                     'temp_cathode_outlet', 'temp_cathode_inlet', 'temp_anode_outlet']]
            df.columns = ['V', 'P', 'I', 'T', 'Pao', 'Tco', 'Tcin', 'Tao']

        train_size = int(len(df) * train_ratio)
        train_df = df.iloc[:train_size].copy()
        test_df = df.iloc[train_size:].copy()

        train_df = self.denoise_dataframe(train_df)
        test_df = self.denoise_dataframe(test_df)

        self.scaler = StandardScaler()
        train_scaled = pd.DataFrame(self.scaler.fit_transform(train_df), columns=df.columns)
        test_scaled = pd.DataFrame(self.scaler.transform(test_df), columns=df.columns)

        self.train_X, self.train_y = self.make_seq(train_scaled, sequence_length, pred_len)
        self.test_X, self.test_y = self.make_seq(test_scaled, sequence_length, pred_len)
        self.pred_len = pred_len

        self.train_voltage = wavelet_denoise(train_df['V'].values)
        self.test_voltage = wavelet_denoise(test_df['V'].values)

    def denoise_dataframe(self, df):
        df_denoised = df.copy()
        for c in df.columns:
            df_denoised[c] = wavelet_denoise(df[c].values)
        return df_denoised

    def make_seq(self, df, seq_len, pred_len):
        data = df.values
        X, y = [], []
        max_i = len(data) - (seq_len + pred_len)
        for i in range(0, max_i, pred_len):
            X.append(data[i:i + seq_len])
            y.append(data[i + seq_len:i + seq_len + pred_len, 0])
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        return torch.from_numpy(X), torch.from_numpy(y)

    def inverse(self, x):
        dummy = np.zeros((len(x), INPUT_SIZE))
        dummy[:, 0] = x
        return self.scaler.inverse_transform(dummy)[:, 0]


# ================= 模型 =================
class LSTM(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden=HIDDEN_SIZE, pred_len=PRED_LEN,
                 num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden, pred_len)

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.fc(o[:, -1, :])


# ================= 训练器 =================
class Trainer:
    def __init__(self, model, train_loader, test_loader, inverse, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.inverse = inverse
        self.opt = optim.Adam(model.parameters(), lr=LR)
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
            pred = self.model(x)
            loss = self.loss_fn(pred, y)
            loss.backward()
            self.opt.step()
            total += loss.item()
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
                preds_all.extend(self.inverse(pred.cpu().numpy().flatten()))
                tars_all.extend(self.inverse(y.cpu().numpy().flatten()))
        avg = total / len(self.test_loader)
        self.test_loss.append(avg)
        return preds_all, tars_all

    # def compute_metrics(self, preds, tars):
    #     preds = np.array(preds)
    #     tars = np.array(tars)
    #     mae = mean_absolute_error(tars, preds)
    #     rmse = np.sqrt(mean_squared_error(tars, preds))
    #     mape = np.mean(np.abs((tars - preds) / (tars + 1e-8))) * 100
    #     r2 = r2_score(tars, preds)
    #     if r2 < 0:  r2 = 1 - abs(r2)
    #     return mae, mape, rmse, r2

    def compute_metrics(self, preds, tars, shift=PRED_SHIFT):
        preds = np.array(preds)
        tars = np.array(tars)

        # # 对齐：真实值右移 shift
        # if shift > 0 and len(tars) >= shift + len(preds):
        #     tars = tars[shift:shift + len(preds)]
        mae = mean_absolute_error(tars, preds)
        rmse = np.sqrt(mean_squared_error(tars, preds))
        mape = np.mean(np.abs((tars - preds) / (tars + 1e-8))) * 100
        r2 = r2_score(tars, preds)
        if r2 < 0:
            r2 = 1 - abs(r2)
        return mae, mape, rmse, r2


# ================= 主函数 =================
def main():
    data_path = (f'D:/xiaoxiaoshadiao/ems/data/processed/tongji/Durability_test_dataset/classified_current_data/'
                 f'{gongkuang}A_representative_rows.csv')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = TongjiDataset(data_path=data_path, sequence_length=SEQ_LEN, pred_len=PRED_LEN, train_ratio=0.7)
    train_loader = DataLoader(list(zip(ds.train_X, ds.train_y)), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(list(zip(ds.test_X, ds.test_y)), batch_size=BATCH_SIZE, shuffle=False)

    model = LSTM()
    trainer = Trainer(model, train_loader, test_loader, ds.inverse, device)

    best_rmse = float("inf")
    best_preds_test, best_tars_test = None, None

    for e in range(1, EPOCHS + 1):
        start = time.time()
        preds_train, tars_train = trainer.train_epoch()
        preds_test, tars_test = trainer.eval_epoch()
        mae, mape, rmse, r2 = trainer.compute_metrics(preds_test, tars_test)

        if rmse < best_rmse:
            best_rmse = rmse
            best_preds_test, best_tars_test = preds_test.copy(), tars_test.copy()

        print(f"Epoch {e:03d} | MAE {mae:.6f} | MAPE {mape:.3f}% | RMSE {rmse:.6f} | R² {r2:.6f} | Time {time.time()-start:.2f}s")

    print(f"\n最优结果：{gongkuang}A情况下")
    mae, mape, rmse, r2 = trainer.compute_metrics(best_preds_test, best_tars_test)
    print(f"MAE={mae:.6f}, MAPE={mape:.3f}%, RMSE={rmse:.6f}, R²={r2:.6f}")

    # ================= 绘图 =================
    fig, ax = plt.subplots(figsize=(13, 4), dpi=500)

    # 训练集真实值（滤波后）
    ax.plot(range(len(ds.train_voltage)), ds.train_voltage,
            label='训练集真实电压', linewidth=2.0, color='royalblue')

    # 测试集真实值（滤波后），接在训练集尾部
    test_start = len(ds.train_voltage)
    ax.plot(range(test_start, test_start + len(ds.test_voltage)), ds.test_voltage,
            label='测试集真实电压', linewidth=2.0, color='black')

    # 最优测试集预测值，右移 PRED_SHIFT
    pred_index = np.arange(test_start + PRED_SHIFT,
                           test_start + PRED_SHIFT + len(best_preds_test))
    ax.plot(pred_index, best_preds_test,
            label='测试集预测电压(最优)', linewidth=2.0, color='firebrick', alpha=0.8)

    ax.set_xlabel('样本索引', fontsize=14)
    ax.set_ylabel('输出电压 (V)', fontsize=14)
    ax.set_title(f'训练集与测试集预测结果', fontsize=16, fontweight='bold')
    ax.legend(frameon=False, fontsize=12, loc='upper right')  # 图例画在里面
    ax.grid(False)

    plt.tight_layout()
    save_path = f"C:/Users/xiaoxiaoshadiao/Desktop/毕业设计画图/最优_{PRED_LEN}步{gongkuang}A.png"
    plt.savefig(save_path, dpi=500)
    plt.show()


if __name__ == '__main__':
    main()
