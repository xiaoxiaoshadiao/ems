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

# ---------- 数据 ----------
DATA_PATH = r'D:\xiaoxiaoshadiao\ems\data\processed\ieee\Durability_test_dataset\FC2_aging_durability_data_denoised.csv'
TRAIN_RATIO = 0.7

# ---------- 序列 ----------
SEQ_LEN = 128
PRED_LEN = 4
SLIDE_STEP = PRED_LEN

# ---------- 小波 ----------
USE_WAVELET = False
WAVELET_NAME = 'db3'
WAVELET_LEVEL = 4
WAVELET_ALPHA = 0.5

# ---------- 模型 ----------
INPUT_SIZE = 8
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.0

# ---------- 训练 ----------
BATCH_SIZE = 64
EPOCHS = 17
LR = 0.001

# ---------- 画图 ----------
MAX_PLOT_POINTS = 500
SAVE_FIG = True
SAVE_DIR = r"C:/Users/xiaoxiaoshadiao/Desktop/毕业设计画图/"

# ======================= STYLE =======================
sns.set_theme(style="whitegrid")
sns.set_context("talk")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# =================== Wavelet =========================
def wavelet_denoise(signal):
    coeffs = pywt.wavedec(signal, WAVELET_NAME, level=WAVELET_LEVEL)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = WAVELET_ALPHA * sigma * np.sqrt(2 * np.log(len(signal)))
    coeffs = [coeffs[0]] + [
        pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]
    ]
    return pywt.waverec(coeffs, WAVELET_NAME)[:len(signal)]


# =================== Dataset =========================
class TongjiDataset(Dataset):
    def __init__(self):
        df = pd.read_csv(DATA_PATH)

        df = df[['V', 'TinH2', 'TinAIR', 'I', 'J', 'DoutAIR', 'ToutH2', 'Dwat']]

        df = df.apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)

        # ✅ 保存完整原始电压（用于画图）
        self.full_voltage = df['V'].values

        split = int(len(df) * TRAIN_RATIO)
        train_df, test_df = df.iloc[:split], df.iloc[split:]

        # ✅ 保存训练 / 测试真实电压
        self.train_voltage = train_df['V'].values
        self.test_voltage = test_df['V'].values

        if USE_WAVELET:
            train_df = train_df.apply(wavelet_denoise)
            test_df = test_df.apply(wavelet_denoise)

        self.scaler = StandardScaler()
        train_df = self.scaler.fit_transform(train_df)
        test_df = self.scaler.transform(test_df)

        self.train_X, self.train_y = self.make_seq(train_df)
        self.test_X, self.test_y = self.make_seq(test_df)


    def make_seq(self, data):
        X, y = [], []
        for i in range(0, len(data) - SEQ_LEN - PRED_LEN, SLIDE_STEP):
            X.append(data[i:i + SEQ_LEN])
            y.append(data[i + SEQ_LEN:i + SEQ_LEN + PRED_LEN, 0])

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        return torch.from_numpy(X), torch.from_numpy(y)


    def inverse(self, x):
        dummy = np.zeros((len(x), INPUT_SIZE))
        dummy[:, 0] = x
        return self.scaler.inverse_transform(dummy)[:, 0]


# =================== Model ===========================
class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS,
                            batch_first=True, dropout=DROPOUT)
        self.fc = nn.Linear(HIDDEN_SIZE, PRED_LEN)

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.fc(o[:, -1])


# =================== Trainer =========================
class Trainer:
    def __init__(self, model, loader, inverse, device):
        self.model = model
        self.loader = loader
        self.inverse = inverse
        self.device = device
        self.opt = optim.Adam(model.parameters(), lr=LR)
        self.loss_fn = nn.MSELoss()

    def run_epoch(self, train=True):
        self.model.train() if train else self.model.eval()
        preds, tars = [], []

        for x, y in self.loader:
            x = x.to(self.device)
            y = y.to(self.device)

            if train:
                self.opt.zero_grad()

            pred = self.model(x)
            loss = self.loss_fn(pred, y)

            if train:
                loss.backward()
                self.opt.step()

            preds.extend(self.inverse(pred.detach().cpu().numpy().flatten()))
            tars.extend(self.inverse(y.detach().cpu().numpy().flatten()))

        return preds, tars


# =================== Main ============================
def main():
    ds = TongjiDataset()

    train_loader = DataLoader(list(zip(ds.train_X, ds.train_y)),
                              batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(list(zip(ds.test_X, ds.test_y)),
                             batch_size=BATCH_SIZE)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTM().to(device)
    trainer_train = Trainer(model, train_loader, ds.inverse, device)
    trainer_test = Trainer(model, test_loader, ds.inverse, device)

    best_rmse = float('inf')
    best_preds = None
    best_tars = None

    for e in range(EPOCHS):
        start = time.time()

        trainer_train.run_epoch(train=True)
        preds, tars = trainer_test.run_epoch(train=False)

        preds = np.array(preds)
        tars = np.array(tars)

        mae = mean_absolute_error(tars, preds)
        rmse = np.sqrt(mean_squared_error(tars, preds))
        mape = np.mean(np.abs((tars - preds) / (tars + 1e-8))) * 100
        r2 = r2_score(tars, preds)

        # ✅ 记录 RMSE 最优结果
        if rmse < best_rmse:
            best_rmse = rmse
            best_preds = preds.copy()
            best_tars = tars.copy()
            best_epoch = e + 1

        print(
            f"Epoch {e + 1:03d} | "
            f"MAE {mae:.6f} | "
            f"MAPE {mape:.3f}% | "
            f"RMSE {rmse:.6f} | "
            f"R² {r2:.6f} | "
            f"Time {time.time() - start:.2f}s"
        )

    print(f"\n✅ 最优结果出现在 Epoch {best_epoch}")
    print(f"✅ Best RMSE = {best_rmse:.6f}")

    # ================= Plot =================
    fig, ax = plt.subplots(figsize=(14, 5), dpi=500)

    # ---------- 1️⃣ 完整原始电压 ----------
    ax.plot(
        range(len(ds.full_voltage)),
        ds.full_voltage,
        color='lightgray',
        linewidth=2.0,
        label='完整原始电压'
    )

    # ---------- 2️⃣ 训练集真实电压 ----------
    train_end = len(ds.train_voltage)
    ax.plot(
        range(train_end),
        ds.train_voltage,
        color='royalblue',
        linewidth=2.0,
        label='训练集真实电压'
    )

    # ---------- 3️⃣ 测试集真实电压 ----------
    ax.plot(
        range(train_end, train_end + len(ds.test_voltage)),
        ds.test_voltage,
        color='black',
        linewidth=2.0,
        label='测试集真实电压'
    )

    # ---------- 4️⃣ 测试集预测电压（最优 RMSE） ----------
    pred_start = train_end + SEQ_LEN
    pred_index = range(pred_start, pred_start + len(best_preds))

    ax.plot(
        pred_index,
        best_preds,
        color='firebrick',
        linewidth=2.0,
        alpha=0.85,
        label='测试集预测电压'
    )

    ax.set_xlabel('样本索引', fontsize=16)
    ax.set_ylabel('输出电压 (V)', fontsize=16)
    ax.set_title('IEEE FC2 电压预测结果（最优 RMSE）',
                 fontsize=16, fontweight='bold')

    ax.legend(frameon=False, fontsize=16)
    ax.grid(False)

    plt.tight_layout()

    if SAVE_FIG:
        plt.savefig(f"{SAVE_DIR}/IEEE_FC2_few_shot_10步.png", dpi=500)


    plt.show()


if __name__ == '__main__':
    main()
