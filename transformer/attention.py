import torch
import math
import torch.nn as nn


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
        # alpha * (x - mean / std + epsilon) + beta
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


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, seq_length: int, head: int, dropout: float):
        super().__init__()

        self.dk = d_model // head

        # Learnable parameters
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def single_head(query, key, value, dropout: nn.Dropout, mask):
        d_k = query.shape[-1]

        # (batch, h, seq_len, d_k) @ (batch, h, d_k, seq_len) => (batch_size, h, seq_len, seq_len)
        attention_score = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_score.masked_fill_(mask == 0, -1e9)

        attention_score = attention_score.softmax(dim=-1)

        # (batch_size, h, seq_len, seq_len) @ (batch_size, h, seq_len, d_k) => (batch_size, h, seq_len, d_k)
        attention_weights = (attention_score @ value)
        if dropout is not None:
            attention_score = dropout(attention_score)

        return attention_weights, attention_score

    def forward(self, q, k, v, mask):
        # x -> seq_length, d_model * d_model, d_model -> seq_length, d_model
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # batch_size, seq_length, d_model -> batch_size, seq_length,h, d_k -> batch_size, h, seq_length, d_k
        query = query.view(query.shape[0], query.shape[1], self.head, self.dk).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.head, self.dk).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.head, self.dk).transpose(1, 2)

        # (batch_size, )
        x, self.attentionScore = MultiHeadAttention.single_head(
            query, key, value, self.dropout, mask)

        # x (batch_size, h, seq_len, d_k) => (batch_size, seq_len, d_model)
        # x = x.view(query.shape[0], query.shape[1], self.h * self.dk)
        x = x.transpose(1, 2).contiguous().view(query.shape[0], -1, self.h * self.dk)
        return self.W_o(x)


class ResNet(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderModel(nn.Module):

    def __init__(self, Multiattention: MultiHeadAttention, feed_forward_network: FeedForwardNetwork, dropout: float):
        super().__init__()
        self.multiHead_Attention = Multiattention
        self.Feed_ForwardNetwork = feed_forward_network
        self.ResNet = nn.ModuleList([ResNet(dropout) for _ in range(2)])

    def forward(self, x, mask):
        x = self.ResNet[0](x, lambda x: self.multiHead_Attention(x, x, x, mask))
        x = self.ResNet[1](x, self.Feed_ForwardNetwork)

        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x, mask):

        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


class DecoderModel(nn.Module):

    def __init__(self, multiattention: MultiHeadAttention, crossattention: MultiHeadAttention, ffn: FeedForwardNetwork, dropout: float):
        super().__init__()
        self.crossAttention = crossattention
        self.multi_head_attention = multiattention
        self.forwardNetwork = ffn

        self.Resnet = nn.ModuleList([ResNet(dropout) for _ in range(3)])

    def forward(self, x, mask):
        x = self.Resnet[0](x, lambda x: self.crossAttention(x, x, x, mask))
        x = self.Resnet[1](x, lambda x: self.multi_head_attention(x, x, x, mask))
        x = self.Resnet[3](x, self.forwardNetwork)

        return x
