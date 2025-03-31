from torch import nn
import torch

class LSTMRegressor(nn.Module):
    def __init__(self, input_dim=21, embedding_dim=16, hidden_dim=64, num_layers=2, bidirectional=True, dropout=0.5):
        """
        input_dim: 词汇表大小（氨基酸种类 + padding）
        embedding_dim: 嵌入层维度
        hidden_dim: LSTM 隐藏层大小
        num_layers: LSTM 层数
        """
        super(LSTMRegressor, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)  # 输出回归值

    def forward(self, x):
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        _, (hn, _) = self.lstm(x)  # 只取最终隐藏状态 hn
        out = self.fc(hn[-1])  # 取 LSTM 最后一层的输出
        return out.squeeze(1)  # 输出 (batch_size,)

