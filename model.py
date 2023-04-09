import torch
from torch import nn
from torch.nn import functional as F
import math


class Block(nn.Module):
    def __init__(self, n_embeds, n_heads, dropout, block_size):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(n_embeds, n_heads, dropout, block_size)
        self.linear = FeedForward(n_embeds, dropout)
        self.ln1 = nn.LayerNorm(n_embeds)
        self.ln2 = nn.LayerNorm(n_embeds)
    
    def forward(self, x):
        x = x + self.multi_head_attention(self.ln1(x))
        x = x + self.linear(self.ln2(x))

        return x

class FeedForward(nn.Module):
    def __init__(self, n_embeds, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embeds, 4*n_embeds),
            nn.ReLU(),
            nn.Linear(4*n_embeds, n_embeds),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embeds, n_heads, dropout, block_size):
        super().__init__()
        self.n_heads = n_heads
        self.n_embeds = n_embeds
        
        self.projection = nn.Linear(n_embeds, n_embeds)
        self.dropout = nn.Dropout(dropout)
        self.linear_heads = nn.Linear(n_embeds, 3 * n_embeds) # combine key, query and value into single linear model
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape

        k, q, v = self.linear_heads(x).split(self.n_embeds, dim=-1)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)

        wei = q @ k.transpose(-2,-1) * (1.0 / math.sqrt(k.size(-1)))
        wei = wei.masked_fill(self.tril[:,:,:T,:T]  == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        out = wei @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.projection(out)
        out = self.dropout(out)
        return out

class CheekGPT(nn.Module):
    def __init__(self, vocab_size, n_embeds, n_layers, block_size, n_heads, dropout):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_embeds) # basically (VOCAB_SIZE x VOCAB_SIZE) tensor
        self.positional_embedding = nn.Embedding(num_embeddings=block_size, embedding_dim=n_embeds)
        self.blocks = nn.Sequential(*[Block(n_embeds, n_heads, dropout, block_size) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(n_embeds)

        self.linear_head = nn.Linear(n_embeds, vocab_size)

        self.block_size = block_size

    def forward(self, idx, target=None):
        # Embedding returns "scores" for each character
        B, T = idx.shape
        embeddings = self.embedding(idx) # (B, T, C) 
        pos_embeddings = self.positional_embedding(torch.arange(T))
        x = embeddings + pos_embeddings
        x = self.blocks(x)
        x = self.ln(x)
        y = self.linear_head(x)

        if target is None:
            loss = None
        else:
            B, T, C = y.shape
            y = y.view(B*T, C)
            target = target.view(B*T)
            loss = F.cross_entropy(y, target)
        return y, loss

    def generate(self, idx, max_new_tokens):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            pred, loss = self(idx_cond)
            pred = pred[:, -1, :]
            probs = F.softmax(pred, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        self.train()
        return idx
