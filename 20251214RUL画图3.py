import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# ===================== 全局风格 =====================
sns.set_theme(style="white")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 路径 =====================
DATA_PATH = r"D:\xiaoxiaoshadiao\ems\data\processed\lvdong\330_100_8_denoised.csv"
MODEL_PATH = r"D:\xiaoxiaoshadiao\ems\data\processed\lvdong\215_best_model.pth"
SCALER_PATH = r"D:\xiaoxiaoshadiao\ems\data\processed\lvdong\215_scaler.pkl"
SAVE_DIR = r"C:\Users\xiaoxiaoshadiao\Desktop\毕业设计画图"
os.makedirs(SAVE_DIR, exist_ok=True)

# ===================== 模型参数（必须与训练一致） =====================
INPUT_SIZE = 8
HIDDEN_SIZE = 128
NUM_LAYERS = 2
SEQ_LEN = 128
PRED_LEN = 1

# ===================== RUL 参数 =====================
RUL_START_INDEX = 2000        # ✅ 本次指定
INIT_WINDOW = 200
FAILURE_RATIOS = [0.035, 0.04, 0.045]
CONSECUTIVE_POINTS = 5
DELTA_T_MIN = 3.605

# ===================== 模型定义 =====================
class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS,
                            batch_first=True)
        self.fc = nn.Linear(HIDDEN_SIZE, PRED_LEN)

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.fc(o[:, -1])

# ===================== 工具函数 =====================
def compute_v_init(voltage, window=200):
    v_init = np.mean(voltage[:window])
    print(f"\n✅ 初始电压 V_init = {v_init:.6f} V")
    return v_init


def find_failure_index(voltage, threshold, consecutive=5):
    count = 0
    for i, v in enumerate(voltage):
        if v < threshold:
            count += 1
            if count == consecutive:
                return i
        else:
            count = 0
    return None


def one_step_predict_with_true_v(model, data_scaled, start_idx, seq_len, max_steps):
    """
    单步预测 + 真值更新（在线 RUL 标准做法）
    """
    preds = []
    for k in range(max_steps):
        cur_seq = data_scaled[start_idx + k - seq_len : start_idx + k].copy()
        x = torch.tensor(cur_seq, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x).cpu().numpy()[0, 0]  # ✅ 只取第 1 步
        preds.append(pred)
    return np.array(preds)

# ===================== 主流程 =====================
def main():
    # ---------- 1️⃣ 读取数据 ----------
    df = pd.read_csv(DATA_PATH)
    df = df[['U', 'TinH2', 'ToutAIR', 'RH', 'PoutAIR', 'PinAIR', 'QAIR', 'TinAIR']]
    df = df.apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)

    features = df.values
    v_true_full = features[:, 0]

    print(f"✅ 数据总长度：{len(v_true_full)}")
    print(f"✅ RUL 评估起点 t0 = {RUL_START_INDEX}")

    # ---------- 2️⃣ 加载 scaler ----------
    scaler = joblib.load(SCALER_PATH)
    features_scaled = scaler.transform(features)

    # ---------- 3️⃣ 加载模型 ----------
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTM().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    print("✅ 已成功加载 215 通道最优模型")

    # ---------- 4️⃣ 构造 RUL 区间 ----------
    v_true_rul = v_true_full[RUL_START_INDEX:]

    # ---------- 5️⃣ 单步滚动预测 ----------
    pred_scaled = one_step_predict_with_true_v(
        model,
        features_scaled,
        start_idx=RUL_START_INDEX,
        seq_len=SEQ_LEN,
        max_steps=len(v_true_rul)
    )

    # ---------- 6️⃣ 反标准化 ----------
    base = features_scaled[RUL_START_INDEX:RUL_START_INDEX + len(pred_scaled)].copy()
    base[:, 0] = pred_scaled
    v_pred_rul = scaler.inverse_transform(base)[:, 0]

    # ---------- 7️⃣ 画整体预测 ----------
    fig, ax = plt.subplots(figsize=(14,5), dpi=600)
    ax.plot(v_true_rul, color='black', linewidth=2, label='真实电压')
    ax.plot(v_pred_rul, color='firebrick', linewidth=2, alpha=0.85, label='预测电压')

    ax.set_title(f'215 通道 RUL 预测（起点 t0 = {RUL_START_INDEX}）',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('采样点索引')
    ax.set_ylabel('输出电压 (V)')
    ax.legend(frameon=False)
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"215_RUL_roll_t0_{RUL_START_INDEX}.png"), dpi=600)
    plt.show()

    # ---------- 8️⃣ RUL 计算 ----------
    v_init = compute_v_init(v_true_full, INIT_WINDOW)
    rul_records = []

    for ratio in FAILURE_RATIOS:
        v_ft = (1 - ratio) * v_init

        idx_true = find_failure_index(v_true_rul, v_ft, consecutive=CONSECUTIVE_POINTS)
        idx_pred = find_failure_index(v_pred_rul, v_ft, consecutive=CONSECUTIVE_POINTS)

        if idx_true is None or idx_pred is None:
            print(f"⚠️ 阈值 {ratio * 100:.1f}% 未检测到失效点")
            continue

        rul_ae = abs(idx_pred - idx_true)
        rul_re = rul_ae / idx_true
        time_ae = rul_ae * DELTA_T_MIN

        print("\n" + "=" * 60)
        print(f"🔻 阈值 {ratio * 100:.1f}%")
        print(f"T_true = {idx_true}, T_pred = {idx_pred}")
        print(f"RUL_AE = {rul_ae} 点 ({time_ae:.2f} min)")
        print(f"RUL_RE = {rul_re:.4%}")

        rul_records.append([
            f"{ratio * 100:.1f}%",
            idx_true,
            idx_pred,
            rul_ae,
            time_ae,
            rul_re
        ])

        # ---------- 画图：每个阈值一张 ----------
        fig, ax = plt.subplots(figsize=(14, 5), dpi=600)

        ax.plot(v_true_rul, color='black', linewidth=2, label='真实电压')
        ax.plot(v_pred_rul, color='firebrick', linewidth=2, alpha=0.85, label='预测电压')

        ax.axhline(v_ft, linestyle='--', color='gray', label='失效阈值')
        ax.axvline(idx_true, linestyle=':', color='royalblue', label='真实失效点')
        ax.axvline(idx_pred, linestyle=':', color='darkorange', label='预测失效点')

        ax.set_title(f'215 通道 RUL 滚动预测结果（阈值 {ratio * 100:.1f}%）',
                     fontsize=16, fontweight='bold')
        ax.set_xlabel('采样点索引')
        ax.set_ylabel('输出电压 (V)')
        ax.legend(frameon=False)
        ax.grid(False)

        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, f"215_RUL_roll_{int(ratio * 1000)}.png"), dpi=600)
        plt.show()


    # ---------- 9️⃣ 保存结果 ----------
    rul_df = pd.DataFrame(
        rul_records,
        columns=["失效阈值","T_true","T_pred","RUL_AE(点)","RUL_AE(分钟)","RUL_RE"]
    )

    print("\n✅ RUL 结果汇总：")
    print(rul_df)

    # rul_df.to_csv(os.path.join(SAVE_DIR, "215_RUL_results.csv"), index=False)
    # print("✅ RUL 结果已保存")

if __name__ == "__main__":
    main()
