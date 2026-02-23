import torch
import torch.nn as nn
import tiktoken

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
        self.embeddings = nn.Embedding(vocab_size, d_size)

    def forward(self, indices):
        return self.embeddings(indices)


query = "what is your name"

query_encoder = InputTokens(query)
tokens = query_encoder.encode()  # Returns a list of ints

token_tensor = torch.tensor(tokens, dtype=torch.long)

embeddings = InputEmbeddings(50257, 412)
print(embeddings.forward(token_tensor))
