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


class PositionalEncoder(nn.Module):

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

    def __init__(self, d_model: int, epsilon: float=1e-9):

        super().__init__()
        self.d_model = d_model
        self.epsilon = epsilon

        self.beta = nn.Parameter(torch.zeros(d_model)) # set to 0
        self.gama = nn.Parameter(torch.ones(d_model)) # set to 1 


    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

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
        self.h = h

        assert self.d_model % h == 0, "d_model is not divisible by h"

        self.d_k = self.d_model // h

        self.w_q = nn.Linear(self.d_model, self.d_model)
        self.w_k = nn.Linear(self.d_model, self.d_model)
        self.w_v = nn.Linear(self.d_model, self.d_model)

        self.w_o = nn.Linear(self.d_model, self.d_model)

    @staticmethod()
    def attention(query, key, value, mask, dropout: nn.Dropout):
        # last dimension of the query matrix i.e d_model
        d_k = query.shape[-1]

        # transpose last two dimensions of the key matrix -> (batch, h, seq_length, d_k) -> (batch, h, d_k, seq_length)
        attention_score = ((query @ key.transpose(-2, -1)) / math.sqrt(d_k))

        if mask is not None:
            attention_score.masked_fill(mask == 0, -1e9)

        attention_score = attention_score.softmax(dim = -1)

        if dropout is not None:
            attention_score = dropout(attention_score)

        return (attention_score @ value), attention_score


    def forward(self, query, key, value, masked_val):

        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        # batch, seq_length, d_model -> batch, seq_length, h, d_k -> batch, head, seq_length, d_k
        Q = Q.view(query[0], query[1], self.h, self.d_k).transpose(1, 2)
        K = K.view(key[0], key[1], self.h, self.d_k).transpose(1, 2)
        V = V.view(value[0], value[1], self.h, self.d_k).transpose(1, 2)

        x, attention_score = MultiHeadAttention.attention(Q, K, V, masked_val, dropout)

        # batch, head, seq_len, d_k =? batch, seq_len, head, d_k => batch, seq_len, d_model
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # (Batch, seq_len, d_model) -> (batch, seq_len, d_model)
        return self.w_o(x)


class EncoderBlock(nn.Module):

    def __init__(self, self_attention: MultiHeadAttention, FeedForwardNetwork: FeedForwardNetwork, dropout: float):

        super().__init__()

        self.mulit_attention = self_attention
        self.feed_forward_block = FeedForwardNetwork
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, mask):

        x = self.residual_connection[0](x, lambda x: self.self_attention(x, x, x, mask))
        x = self.residual_connection[1](x, lambda x: self.feed_forward_block)

        return x


class Encoder(nn.Module):

    def __init__(self, layer: nn.ModuleList) -> None:

        self.layers = layer
        self.layer_norm = AddAndNorm()

    def forward(self, x, mask):

        for layer in self.layers:
            x = layer(x, mask)

        return self.layer_norm(x)


class DecoderBlock(nn.Module):

    def __init__(self, self_attention: MultiHeadAttention, fnn: FeedForwardNetwork, cross_attention: MultiHeadAttention, dropout: float):

        super().__init__()
        self.multi_attention = self_attention
        self.fnn = fnn
        self.cross_attention = cross_attention
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])


    def forward(self, x, src_mask, encoder_output, target_mask):
        
        x = self.residual_connection[0](x, lambda x: self.multi_attention(x, x, x, target_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection[2](x, self.fnn)

        return x


class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.layer_norm = AddAndNorm()

    def forward(self, x, encoder_output, src_mask, target_mask):

        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, target_mask)

        return self.layer_norm(x)


class ProjectionLayer(nn.Module):

    def __init__(self, seq_length: int, d_model: int, vocab_size: int):

        super().__init__()
        self.seq_length = seq_length
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Final Neural Layer
        self.linear = nn.Linear(self.d_model, self.vocab_size) # (batch_size, seq_length, d_model) @ (d_model, vocab_size) i.e (512 * 512)
        # (b, m, n) * (n, v) -> (m, v)
        


    def forward(self, input):
        return torch.log_softmax(self.linear(input), dim=-1)



class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embedding: InputEmbeddings, positionalEncoder: PositionalEncoder trgt_embedding: InputEmbeddings, projectionLayer: ProjectionLayer):

        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_embeddings = src_embedding
        self.targt_embeddings = targt_embddings
        self.pos_encoder = positionalEncoder
        self.projectionlayer = projectionLayer



    def encoder(self, src, src_mask):

        embeddings = self.src_embeddings(src)
        positional_encoding = self.pos_encoder(embeddings)

        return self.encoder(positional_encoding, src_mask)


    def decoder(self, encoder_output, src_mask, targt, targt_mask):

        targt = self.targt_embeddings(targt)
        targt = self.pos_encoder(targt)

        return self.decoder(target, encoder_output, src_mask, targt_mask)


    def Projection(self, x):

        return self.projectionlayer(x)



def build_transformer(vocab_size: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1):
