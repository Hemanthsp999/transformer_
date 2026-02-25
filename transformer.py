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


class AddAndNorm(nn.Module):

    def __init__(self, d_model: int, dropout: float):

        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        self.Layers = 
