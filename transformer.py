import torch
import torch.nn as nn
import tiktoken
import math

class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.embeddings = nn.Embedding(vocab_size, self.d_model)

    def forward(self, indices):
        return self.embeddings(indices) * math.sqrt(self.d_model)


class PositionalEncoder(nn.Module):

    def __init__(self, d_model: int, dropout: float, seq_length: int) -> None:
        super().__init__()

        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.seq_length = seq_length

        # BUG FIX: was correct, kept as-is
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(seq_length, d_model)
        pos = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)

        pe[:, ::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)

        pe = pe.unsqueeze(0)  # (1, seq_length, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class AddAndNorm(nn.Module):

    def __init__(self, d_model: int, epsilon: float = 1e-9):
        super().__init__()
        self.d_model = d_model
        self.epsilon = epsilon

        self.beta = nn.Parameter(torch.zeros(d_model))
        self.gama = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        layer_val = (x - mean) / torch.sqrt(std ** 2 + self.epsilon)
        return self.gama * layer_val + self.beta


class FFN(nn.Module):

    def __init__(self, dropout: float, d_model: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        # BUG FIX: was `dff` (undefined variable), should be `d_ff`
        self.d_ff = d_ff
        self.net = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_ff, self.d_model)
        )

    def forward(self, x):
        return self.net(x)


class ResidualConnection(nn.Module):

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # BUG FIX: AddAndNorm() requires d_model argument
        self.layer_normalization = AddAndNorm(d_model)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.layer_normalization(x)))


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.h = h

        assert self.d_model % h == 0, "d_model is not divisible by h"

        self.d_k = self.d_model // h

        self.w_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_v = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_o = nn.Linear(self.d_model, self.d_model, bias=False)

    # BUG FIX: @staticmethod() has no parentheses
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        attention_score = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            # BUG FIX: must use masked_fill_ (in-place) or assign result
            attention_score = attention_score.masked_fill(mask == 0, -1e9)

        attention_score = attention_score.softmax(dim=-1)

        if dropout is not None:
            attention_score = dropout(attention_score)

        return (attention_score @ value), attention_score

    def forward(self, query, key, value, masked_val):
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        # BUG FIX: query[0] doesn't work on tensors; use query.shape[0], query.shape[1]
        Q = Q.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        K = K.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        V = V.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttention.attention(Q, K, V, masked_val, self.dropout)

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        return self.w_o(x)


class EncoderBlock(nn.Module):

    # BUG FIX: FFN was referred to as FeedForwardNetwork (undefined)
    def __init__(self, d_model: int, self_attention: MultiHeadAttention, feed_forward_network: FFN, dropout: float):
        super().__init__()

        # BUG FIX: was `self.mulit_attention` (typo), and later referenced as `self.self_attention`
        self.self_attention = self_attention
        self.feed_forward_block = feed_forward_network
        # BUG FIX: ResidualConnection now requires d_model
        self.residual_connection = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(2)])

    def forward(self, x, mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention(x, x, x, mask))
        # BUG FIX: was `lambda x: self.feed_forward_block` — missing call ()
        x = self.residual_connection[1](x, lambda x: self.feed_forward_block(x))
        return x


class Encoder(nn.Module):

    def __init__(self, d_model: int, layer: nn.ModuleList) -> None:
        # BUG FIX: missing super().__init__()
        super().__init__()
        self.layers = layer
        # BUG FIX: AddAndNorm() requires d_model
        self.layer_norm = AddAndNorm(d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.layer_norm(x)


class DecoderBlock(nn.Module):

    # BUG FIX: FFN was referred to as FeedForwardNetwork (undefined)
    def __init__(self, d_model: int, self_attention: MultiHeadAttention, fnn: FFN, cross_attention: MultiHeadAttention, dropout: float):
        super().__init__()
        self.self_attention = self_attention
        self.fnn = fnn
        self.cross_attention = cross_attention
        # BUG FIX: ResidualConnection now requires d_model
        self.residual_connection = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, target_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention(x, x, x, target_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask))
        # BUG FIX: was `self.fnn` without calling it as a function
        x = self.residual_connection[2](x, lambda x: self.fnn(x))
        return x


class Decoder(nn.Module):

    def __init__(self, d_model: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        # BUG FIX: AddAndNorm() requires d_model
        self.layer_norm = AddAndNorm(d_model)

    def forward(self, x, encoder_output, src_mask, target_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, target_mask)
        return self.layer_norm(x)


class ProjectionLayer(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.linear = nn.Linear(self.d_model, self.vocab_size)

    def forward(self, input):
        return torch.log_softmax(self.linear(input), dim=-1)


class Transformer(nn.Module):

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embedding: InputEmbeddings,
        src_pos: PositionalEncoder,
        target_pos: PositionalEncoder,
        trgt_embedding: InputEmbeddings,
        projectionLayer: ProjectionLayer
    ) -> None:
        super().__init__()

        self.encoder_block = encoder
        self.decoder_block = decoder
        self.src_embeddings = src_embedding
        # BUG FIX: was `targt_embddings` (undefined variable)
        self.trgt_embeddings = trgt_embedding
        self.src_pos = src_pos
        self.target_pos = target_pos
        self.projectionlayer = projectionLayer

    # BUG FIX: methods named `encoder`/`decoder` shadowed the stored nn.Module attributes.
    # Renamed to encode() / decode() / project()
    def encode(self, src, src_mask):
        src = self.src_embeddings(src)
        src = self.src_pos(src)
        return self.encoder_block(src, src_mask)

    def decode(self, encoder_output, src_mask, targt, targt_mask):
        targt = self.trgt_embeddings(targt)
        targt = self.target_pos(targt)
        # BUG FIX: was `target` (undefined), should be `targt`
        return self.decoder_block(targt, encoder_output, src_mask, targt_mask)

    def project(self, x):
        return self.projectionlayer(x)


def build_transformer(
    src_vocab_size: int,
    target_vocab_size: int,
    src_seq_length: int,
    targt_seq_length: int,
    N: int = 6,
    h: int = 8,
    d_model: int = 512,
    dropout: float = 0.1,
    d_ff: int = 2048
) -> Transformer:

    src_embedding = InputEmbeddings(src_vocab_size, d_model)
    target_embedding = InputEmbeddings(target_vocab_size, d_model)

    src_pos = PositionalEncoder(d_model, dropout, src_seq_length)
    target_pos = PositionalEncoder(d_model, dropout, targt_seq_length)

    # BUG FIX: `ecoder_block` typo; also the loop reused `encoder_block` as both
    # the list name AND the newly created block — overwrote the list each iteration.
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FFN(dropout, d_model, d_ff)
        block = EncoderBlock(d_model, encoder_self_attention, feed_forward_block, dropout)
        encoder_blocks.append(block)

    # BUG FIX: same list-vs-instance reuse bug in decoder loop
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention = MultiHeadAttention(d_model, h, dropout)
        decoder_cross_attention = MultiHeadAttention(d_model, h, dropout)
        decoder_fnn = FFN(dropout, d_model, d_ff)
        block = DecoderBlock(d_model, decoder_self_attention, decoder_fnn, decoder_cross_attention, dropout)
        decoder_blocks.append(block)

    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    projection_layer = ProjectionLayer(d_model, target_vocab_size)

    # BUG FIX: was `ecoder=encoder` (typo kwarg), now matches __init__ param `encoder`
    transformer = Transformer(
        encoder=encoder,
        decoder=decoder,
        src_embedding=src_embedding,
        src_pos=src_pos,
        target_pos=target_pos,
        trgt_embedding=target_embedding,
        projectionLayer=projection_layer
    )

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
