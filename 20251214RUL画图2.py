import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler

# ===================== 全局风格 =====================
sns.set_theme(style="white")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 路径 =====================
RAW_DATA_PATH = r"D:\xiaoxiaoshadiao\ems\data\processed\ieee\Durability_test_dataset\FC2_aging_durability_data_denoised.csv"
MODEL_PATH = r"D:\xiaoxiaoshadiao\ems\data\processed\ieee\Durability_test_dataset\FC2_best_model.pth"
SAVE_DIR = r"C:\Users\xiaoxiaoshadiao\Desktop\毕业设计画图"

# ===================== 模型参数（必须和训练一致） =====================
INPUT_SIZE = 8
HIDDEN_SIZE = 128
NUM_LAYERS = 2
SEQ_LEN = 128
PRED_LEN = 1

# ===================== RUL 参数 =====================
RUL_START_INDEX = 5000        # ✅ 尚未失效的预测起点（改大一点 否则看不到失效  也不嫩太大  否则全失效了）
INIT_WINDOW = 200
FAILURE_RATIOS = [0.035, 0.04, 0.045]
CONSECUTIVE_POINTS = 5
DELTA_T_MIN = 4.805           # FC2 时间分辨率（min / sample）

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
    每一步都使用真实V（不把pred喂回去），只做单步预测：
    - 输入窗口完全来自真实序列（已标准化）
    - 输出为下一时刻V的预测（标准化域）
    """
    preds = []
    for k in range(max_steps):
        cur_seq = data_scaled[start_idx + k - seq_len : start_idx + k].copy()
        x = torch.tensor(cur_seq, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_v_scaled = model(x).cpu().numpy()[0, 0]
        preds.append(pred_v_scaled)
    return np.array(preds)


# ===================== 主流程 =====================
def main():
    # ---------- 1️⃣ 读取原始数据 ----------
    df = pd.read_csv(RAW_DATA_PATH)
    features = df[['V','TinH2','TinAIR','I','J','DoutAIR','ToutH2','Dwat']].values
    v_true_full = features[:, 0]

    print(f"✅ 原始数据长度：{len(v_true_full)}")

    # ---------- 2️⃣ 标准化 ----------
    TRAIN_RATIO = 0.7
    split = int(len(features) * TRAIN_RATIO)

    scaler = StandardScaler()
    scaler.fit(features[:split])          # ✅ 只用训练段拟合
    features_scaled = scaler.transform(features)


    # ---------- 3️⃣ 加载模型 ----------
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTM().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    print("✅ 已成功加载 FC2 最优模型")

    # ---------- 4️⃣ 构造真实 RUL 序列 ----------
    v_true_rul = v_true_full[RUL_START_INDEX:]
    print(f"✅ RUL 评估起点 t0 = {RUL_START_INDEX}")

    # ---------- 5️⃣ 滚动预测 ----------
    pred_scaled = one_step_predict_with_true_v(
        model,
        features_scaled,
        start_idx=RUL_START_INDEX,
        seq_len=SEQ_LEN,
        max_steps=len(v_true_rul)
    )


    # 用真实特征的标准化值做基底，只替换V那一列为预测值，再整体 inverse_transform
    base = features_scaled[RUL_START_INDEX:RUL_START_INDEX + len(pred_scaled)].copy()
    base[:, 0] = pred_scaled
    v_pred_rul = scaler.inverse_transform(base)[:, 0]
    print("\n📌 前10个点对比：")
    for i in range(10):
        print(i, "true=", float(v_true_rul[i]), "pred=", float(v_pred_rul[i]))


    # ======== 先画整体滚动预测结果（不依赖失效） ========
    fig, ax = plt.subplots(figsize=(14,5), dpi=600)

    ax.plot(v_true_rul, color='black', linewidth=2, label='真实电压')
    ax.plot(v_pred_rul, color='firebrick', linewidth=2, alpha=0.85, label='滚动预测电压')

    ax.set_title(
        f'FC2 滚动预测结果（起点 t0 = {RUL_START_INDEX}）',
        fontsize=16, fontweight='bold'
    )
    ax.set_xlabel('采样点索引')
    ax.set_ylabel('输出电压 (V)')
    ax.legend(frameon=False)
    ax.grid(False)

    plt.tight_layout()
    # plt.savefig(os.path.join(SAVE_DIR, f"FC2_roll_prediction_t0_{RUL_START_INDEX}.png"), dpi=600)
    plt.show()


    # ---------- 6️⃣ RUL 计算 ----------
    v_init = compute_v_init(v_true_full, INIT_WINDOW)
    rul_records = []

    for ratio in FAILURE_RATIOS:
        v_ft = (1 - ratio) * v_init

        idx_true = find_failure_index(v_true_rul, v_ft)
        idx_pred = find_failure_index(v_pred_rul, v_ft)

        if idx_true is None or idx_pred is None:
            continue

        rul_ae = abs(idx_pred - idx_true)
        rul_re = rul_ae / idx_true
        time_ae = rul_ae * DELTA_T_MIN

        print("\n" + "="*60)
        print(f"🔻 阈值 {ratio*100:.1f}%")
        print(f"T_true = {idx_true}, T_pred = {idx_pred}")
        print(f"RUL_AE = {rul_ae} 点 ({time_ae:.2f} min)")
        print(f"RUL_RE = {rul_re:.4%}")

        rul_records.append([
            f"{ratio*100:.1f}%",
            idx_true,
            idx_pred,
            rul_ae,
            time_ae,
            rul_re
        ])

        # ---------- 7️⃣ 画图 ----------
        fig, ax = plt.subplots(figsize=(14,5), dpi=600)

        ax.plot(v_true_rul, color='black', linewidth=2, label='真实电压')
        ax.plot(v_pred_rul, color='firebrick', linewidth=2, alpha=0.85, label='预测电压')

        ax.axhline(v_ft, linestyle='--', color='gray', label='失效阈值')
        ax.axvline(idx_true, linestyle=':', color='royalblue', label='真实失效点')
        ax.axvline(idx_pred, linestyle=':', color='darkorange', label='预测失效点')

        ax.set_title(f'FC2 RUL 滚动预测结果（阈值 {ratio*100:.1f}%）', fontsize=16, fontweight='bold')
        ax.set_xlabel('采样点索引')
        ax.set_ylabel('输出电压 (V)')
        ax.legend(frameon=False)
        ax.grid(False)

        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, f"FC2_RUL_roll_{int(ratio*1000)}.png"), dpi=600)
        plt.show()

    # ---------- 8️⃣ 保存结果 ----------
    rul_df = pd.DataFrame(
        rul_records,
        columns=["失效阈值","T_true","T_pred","RUL_AE(点)","RUL_AE(分钟)","RUL_RE"]
    )

    print("\n✅ RUL 结果汇总：")
    print(rul_df)


if __name__ == "__main__":
    main()
