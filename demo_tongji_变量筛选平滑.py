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


class TongjiDataset(Dataset):
    def __init__(self, data_path=None, sequence_length=50, train_ratio=0.7):
        if data_path is None:
            data_path = os.path.join(os.path.dirname(__file__), 'data', 'processed', 'tongji',
                                     'Durability_test_dataset',
                                     'classified_current_data', 'all_representative_rows.csv')

        df = pd.read_csv(data_path)

        df = df[[
            'voltage', 'power', 'current', 'temp_anode_endplate', 'pressure_anode_outlet',
            'temp_cathode_outlet', 'temp_cathode_inlet', 'temp_anode_outlet'
        ]]

        df.columns = ['V', 'P', 'I', 'T', 'Pao', 'Tco', 'Tcin', 'Tao']

        # from statsmodels.nonparametric.smoothers_lowess import lowess
        # frac = 25 / len(df)
        # idx = np.arange(len(df))
        # for c in df.columns:
        #     df[c] = lowess(df[c].values, idx, frac=frac, it=0, return_sorted=False)

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
        return avg, self.inverse(preds), self.inverse(tars)

    def train(self, epochs):
        for e in range(1, epochs + 1):
            start = time.time()
            train_l = self.train_epoch()
            test_l, _, _ = self.eval()
            print(f"Epoch {e}: Train {train_l:.6f}, Test {test_l:.6f}, Time {time.time() - start:.2f}s")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds = TongjiDataset(sequence_length=50)
    train_loader = DataLoader(list(zip(ds.train_X, ds.train_y)), batch_size=32, shuffle=True)
    test_loader = DataLoader(list(zip(ds.test_X, ds.test_y)), batch_size=32, shuffle=False)

    model = LSTM()
    trainer = Trainer(model, train_loader, test_loader, ds.inverse, device)

    trainer.train(20)

    preds, tars = trainer.eval()[1:]
    mse = np.mean((preds - tars) ** 2)
    print("Final MSE:", mse)

    plt.figure(figsize=(12, 4))
    plt.plot(trainer.train_loss, label='Train')
    plt.plot(trainer.test_loss, label='Test')
    plt.legend();
    plt.title("Loss Curve");
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(tars, label='True')
    plt.plot(preds, label='Pred')
    plt.legend();
    plt.title("Prediction Result");
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
