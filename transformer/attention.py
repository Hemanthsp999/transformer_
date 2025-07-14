import torch
import math
import torch.nn as nn


vocab = {
    "who": 1,
    "are": 2,
    "you": 3,
    "what": 4,
    "and": 5,
    "how": 6,
    "is": 7,
    "that": 8,
    "hi": 9,
    "bye": 10,
    "?": 11,
    "your": 12,
    "name": 13
}


class InputEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, self.d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_length: int, drop_out: float):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(drop_out)

        # create a matrix of shape(seq_length, d_model)
        pe = torch.zeros(seq_length, d_model)

        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :]
        return self.droput(x)


class LayerNorm(nn.Module):
    def __init__(self, eps: float = 10**-6):
        super().__init__()
        self.eps = eps

        self.alpha = nn.Parameter(torch.ones(1))  # Multiply
        self.bias = nn.Parameter(torch.zeros(1))  # Additive

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()

        self.layer1 = nn.Linear(d_model, d_ff)  # d_ff -> 2048
        self.dropout = nn.Dropout(dropout)
        self.layer2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # input -> batch_size, seq, d_model -> layer1: d_model - d_ff -> layer2: d_ff - d_model
        return self.layer2(self.dropout(torch.relu(self.layer1(x))))


class Attention(nn.Module):
    def __init__(self, d_model: int, seq_length: int):
        super().__init__()

        Q = torch.tensor(seq_length, d_model)
        K = torch.tensor(seq_length, d_model).t()

        V = torch.tensor(seq_length, seq_length)
