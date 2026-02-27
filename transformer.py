import torch
import torch.nn as nn
import tiktoken
import math

encoder = tiktoken.get_encoding("gpt2")

class InputTokens:
    def __init__(self, x):
        self.x = x

    def encode(self):
        return encoder.encode(self.x)

    def decode(self):
        tokens = self.encode()
        return encoder.decode(tokens)

    def length(self):
        return len(self.x)


class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size, d_size):
        super().__init__()
        self.d_model = d_size
        self.embeddings = nn.Embedding(vocab_size, d_size) # original papper shows dimension of vocab_size * 512 

    def forward(self, indices):
        return self.embeddings(indices) * math.sqrt(self.d_model) # matrix * d_model


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float, seq_length: int) -> None:
        super().__init__()

        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.seq_length = seq_length

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # 10000 ^ (2 * i)/d_model

        self.pe = torch.zeros(seq_length, d_model) # create tensor of size seq_lenth * d_model
        self.pos = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1) # create a tensor of seq_length * 1

        self.pe[:, ::2] = torch.sin(pos * div_term) # for even pos
        self.pe[:, 1::2] = torch.cos(pos * div_term) # for odd pos


        self.pe = self.pe.unsqueeze(0) # (1, seq_length, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[: ,:x.shape[1], :]).requires_grad_(False) # (batch, seq_length, d_model)
        return self.dropout(x)

class MultiHeadAttention(nn.Module):

    def __init__(self, seq_length: int, d_model: int, batch_size: int, dropout: float):

        super().__init__()

class AddAndNorm(nn.Module):

    def __init__(self, d_model: int, epsilon: float=1e-9):

        super().__init__()
        self.d_model = d_model
        self.epsilon = epsilon

        self.beta = nn.Parameter(torch.zeros(d_model)) # set to 0
        self.gama = nn.Parameter(torch.ones(d_model)) # set to 1 


    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.std_mean(dim=1, keepdim=True)

        layer_val = (x - mean)/torch.sqrt(std**2 + self.epsilon)
        return self.gama * layer_val + self.beta


class FFN(nn.Module):

    def __init__(self, dropout: float, d_model: int):

        super().__init__()
        self.d_model = d_model

        self.d_ff = 4 * self.d_model # 2048 which is used in original paper
        self.net = nn.Sequential(
                nn.Linear(self.d_model, self.d_ff), # First Layer 
                nn.ReLU(), # First Layer output passed to relu
                nn.Dropout(dropout), # dropout is applied on relu output
                nn.Linear(self.d_ff, self.d_model) # Second Layer
        ) 

    def forward(self, x):
        return self.net(x)


class ResidualConnection(nn.Module):

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_normalization = AddAndNorm() # Layer Normalization


    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.layer_normalization(x)))


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float)->None:
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        assert self.d_model == 0, "d_model is not divisible by h"

        self.d_h = self.d_model // h

        self.w_q = nn.Linear(self.d_model, self.d_model)
        self.w_k = nn.Linear(self.d_model, self.d_model)
        self.w_v = nn.Linear(self.d_model, self.d_model)

        self.w_o = nn.Linear(self.d_model, self.d_model)

    @staticmethod()
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]


    def forward(self, query, key, value, masked_val):

        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

