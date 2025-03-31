import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, random_split
import torch

def load_class_feature_data(file_path, test_size=0.2, random_state=42):
    data = pd.read_csv(file_path)
    X = data.iloc[:, 1:-2]  # 特征数据
    y = data["type"]  # 目标变量
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test

def load_regress_feature_data(file_path, test_size=0.2, random_state=42, log_base=None):
    data = pd.read_csv(file_path)
    X = data.iloc[:, 1:-2]  # 特征数据
    y = data["MIC"]  # 目标变量
    
    # 对 MIC 进行 log 变换（如果 log_base 指定）
    if log_base:
        y = np.log(y) / np.log(log_base)  # 计算 log_base(y)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test


class PeptideDataset(Dataset):
    def __init__(self, sequences, logMIC_values, max_len=50):
        """
        sequences: list[str] -> 多肽序列列表
        logMIC_values: list[float] -> 对应的 logMIC 值
        max_len: int -> 设定所有序列的最大长度（短的填充，长的截断）
        """
        self.sequences = sequences
        self.logMIC_values = torch.tensor(logMIC_values, dtype=torch.float32)
        self.max_len = max_len
        self.char_to_idx = {aa: i+1 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}  # 仅考虑20种氨基酸

    def encode_sequence(self, seq):
        """把氨基酸序列转换成索引"""
        encoded = [self.char_to_idx.get(aa, 0) for aa in seq]  # 0 代表未知氨基酸
        encoded = encoded[:self.max_len]  # 截断
        encoded += [0] * (self.max_len - len(encoded))  # 填充
        return torch.tensor(encoded, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.encode_sequence(self.sequences[idx]), self.logMIC_values[idx]

def load_regress_seq_data(file_path, batch_size=32, max_len=50, test_size=0.2):
    data = pd.read_csv(file_path)
    sequences = data["sequence"].tolist()
    logMIC_values = np.log10(data["MIC"]).tolist()

    dataset = PeptideDataset(sequences, logMIC_values, max_len)
    # 按 8:2 划分训练集和测试集
    test_size = int(len(dataset) * test_size)
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


class PredictDataset(Dataset):
    def __init__(self, sequences, max_len=50):
        self.sequences = sequences
        self.max_len = max_len
        self.char_to_idx = {aa: i+1 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}  # 仅考虑20种氨基酸
    def encode_sequence(self, seq):
        encoded = [self.char_to_idx.get(aa, 0) for aa in seq]  # 0 代表未知氨基酸
        encoded = encoded[:self.max_len]  # 截断
        encoded += [0] * (self.max_len - len(encoded))  # 填充
        return torch.tensor(encoded, dtype=torch.long)
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        return self.encode_sequence(self.sequences[idx])
        


if __name__ == '__main__':
    # 示例用法
    file_path = "data/processed/all_feature.csv"  # 替换为你的数据文件路径
    X_train, X_test, y_train, y_test = load_class_feature_data(file_path)
    
    print("训练集特征形状:", X_train.shape)
    print("测试集特征形状:", X_test.shape)
    print("训练集标签形状:", y_train.shape)
    print("测试集标签形状:", y_test.shape)
    
    