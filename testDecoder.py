import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(CausalSelfAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.layernorm = nn.LayerNorm(embed_dim)

    def forward(self, x): 
        # x: (seq_len, batch_size, embed_dim)
        seq_len, batch_size, embed_dim = x.size()
        mask = torch.triu(torch.ones((seq_len, seq_len), device=x.device), diagonal=1).bool()
        attn_output, _ = self.mha(x, x, x, attn_mask=mask)
        x = self.layernorm(x + attn_output)
        return x


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.layernorm = nn.LayerNorm(embed_dim)

    def forward(self, x, y):
        attn_output, self.last_attention_scores = self.mha(x, y, y)
        x = self.layernorm(x + attn_output)
        return x


class FeedForward(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(embed_dim, embed_dim * 2)
        self.linear2 = nn.Linear(embed_dim * 2, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.layernorm(x + self.dropout(self.linear2(F.relu(self.linear1(x)))))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=1, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = CausalSelfAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.cross_attention = CrossAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.ff = FeedForward(embed_dim=embed_dim, dropout=dropout)

    def forward(self, in_seq, out_seq):
        in_seq = in_seq.unsqueeze(0)
        out_seq = self.self_attention(out_seq)
        out_seq = self.cross_attention(out_seq, in_seq)
        out_seq = self.ff(out_seq)
        return out_seq


if __name__ == "__main__":
    embed_dim = 512
    num_heads = 8
    seq_len = 10
    batch_size = 1 
    decoder_layer = DecoderLayer(embed_dim=embed_dim, num_heads=num_heads)

    in_seq = torch.rand(batch_size, embed_dim)
    out_seq = torch.rand(seq_len, batch_size, embed_dim)
    out_seq = decoder_layer(in_seq, out_seq)

    print("Output Sequence:", out_seq)
    print("Output Shape:", out_seq.shape)