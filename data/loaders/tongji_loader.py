# D:\xiaoxiaoshadiao\ems\data\loaders\tongji_loader.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import os


class TongjiDataset(Dataset):
    """同济大学PEMFC数据集加载器"""

    def __init__(self, data_path=None, sequence_length=100, prediction_horizon=1,
                 mode='single_step', train_ratio=0.7, target_column='voltage'):
        """
        数据集加载器
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.mode = mode
        self.target_column = target_column

        # 设置默认数据路径
        if data_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(current_dir, '..', 'processed', 'tongji',
                                     'Durability_test_dataset', 'classified_current_data',
                                     'all_representative_rows.csv')

        # 加载数据
        self.data = pd.read_csv(data_path)
        if 'time' in self.data.columns:
            self.data = self.data.drop('time', axis=1)

        # 划分数据集
        train_size = int(len(self.data) * train_ratio)
        self.train_data = self.data.iloc[:train_size]
        self.test_data = self.data.iloc[train_size:]

        # 归一化
        self.scaler = StandardScaler()
        self.train_data_scaled = pd.DataFrame(self.scaler.fit_transform(self.train_data),
                                              columns=self.train_data.columns)
        self.test_data_scaled = pd.DataFrame(self.scaler.transform(self.test_data),
                                             columns=self.test_data.columns)

        # 创建序列
        self.train_sequences = self._create_sequences(self.train_data_scaled)
        self.test_sequences = self._create_sequences(self.test_data_scaled)

        print(f"训练集序列: {len(self.train_sequences[0])}, 测试集序列: {len(self.test_sequences[0])}")

    def _create_sequences(self, data):
        """创建输入输出序列"""
        X, y = [], []
        data_values = data.values
        target_idx = list(data.columns).index(self.target_column)

        if self.mode == 'single_step':
            # 单步预测
            for i in range(len(data_values) - self.sequence_length):
                X.append(data_values[i:i + self.sequence_length])
                y.append(data_values[i + self.sequence_length, target_idx])
        else:
            # 多步预测
            for i in range(len(data_values) - self.sequence_length - self.prediction_horizon + 1):
                X.append(data_values[i:i + self.sequence_length])
                y.append(data_values[
                             i + self.sequence_length:i + self.sequence_length + self.prediction_horizon, target_idx])

        return np.array(X), np.array(y)

    def inverse_transform(self, data):
        """将归一化数据反归一化"""
        # 创建与原始数据相同形状的数组，只填充目标列
        dummy_data = np.zeros((len(data), len(self.data.columns)))
        target_idx = list(self.data.columns).index(self.target_column)
        dummy_data[:, target_idx] = data

        inverted = self.scaler.inverse_transform(dummy_data)
        return inverted[:, target_idx]

    def get_train_data(self):
        """获取训练数据"""
        X_train = torch.FloatTensor(self.train_sequences[0])
        y_train = torch.FloatTensor(self.train_sequences[1])
        # 确保单步预测的输出形状为 [batch_size, 1]
        if self.mode == 'single_step' and y_train.dim() == 1:
            y_train = y_train.unsqueeze(1)
        return X_train, y_train

    def get_test_data(self):
        """获取测试数据"""
        X_test = torch.FloatTensor(self.test_sequences[0])
        y_test = torch.FloatTensor(self.test_sequences[1])
        # 确保单步预测的输出形状为 [batch_size, 1]
        if self.mode == 'single_step' and y_test.dim() == 1:
            y_test = y_test.unsqueeze(1)
        return X_test, y_test

    def get_feature_names(self):
        """获取特征名称"""
        return list(self.data.columns)


# noinspection PyTypeChecker
class TongjiDataLoader:
    """数据加载器"""

    def __init__(self, batch_size=32, shuffle=True, **dataset_args):
        self.dataset = TongjiDataset(**dataset_args)
        self.batch_size = batch_size

        X_train, y_train = self.dataset.get_train_data()
        X_test, y_test = self.dataset.get_test_data()

        self.train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=batch_size, shuffle=shuffle)
        self.test_loader = DataLoader(list(zip(X_test, y_test)), batch_size=batch_size, shuffle=False)

    def get_loaders(self):
        return {'train': self.train_loader, 'test': self.test_loader}

    def inverse_transform(self, data):
        return self.dataset.inverse_transform(data)


# 测试函数
def test_simple_loader():
    print("=== 测试数据集加载器 ===")

    # 单步预测
    print("\n单步预测:")
    loader = TongjiDataLoader(
        batch_size=16,
        sequence_length=50,
        prediction_horizon=1,
        mode='single_step'
    )

    train_loader, test_loader = loader.get_loaders().values()
    X_batch, y_batch = next(iter(train_loader))
    print(f"训练数据形状: X={X_batch.shape}, y={y_batch.shape}")

    X_batch, y_batch = next(iter(test_loader))
    print(f"测试数据形状: X={X_batch.shape}, y={y_batch.shape}")

    # 多步预测
    print("\n多步预测:")
    loader_multi = TongjiDataLoader(
        batch_size=16,
        sequence_length=50,
        prediction_horizon=5,
        mode='multi_step'
    )

    train_loader, test_loader = loader_multi.get_loaders().values()
    X_batch, y_batch = next(iter(train_loader))
    print(f"训练数据形状: X={X_batch.shape}, y={y_batch.shape}")

    # 反归一化测试
    print("\n反归一化测试:")
    sample_data = np.array([0.5, -0.3, 1.2])
    original = loader.inverse_transform(sample_data)
    print(f"归一化: {sample_data} -> 原始: {original}")


# 运行检查
if __name__ == "__main__":
    test_simple_loader()
