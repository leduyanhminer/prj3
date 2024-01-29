import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryAdapterLayer(nn.Module):
    def __init__(self, dim_query, dim_mem):
        super(MemoryAdapterLayer, self).__init__()

        self.query_transform = nn.Linear(dim_query, dim_mem)
        self.memory_transform = nn.Linear(dim_mem, dim_query)

    def forward(self, x, memory):
        # x: (batch_size, seq_len, dim_query)
        # memory: (batch_size, 1, dim_mem)

        query = self.query_transform(x) # query: (batch_size, seq_len, dim_mem)

        attention_scores = torch.matmul(query, memory.transpose(-2, -1)) # attention_scores: (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_scores, dim=-1) # attention_weights: (batch_size, seq_len, 1)
        attended_memory = torch.matmul(attention_weights, memory) # attended_memory: (batch_size, seq_len, dim_mem)

        transformed_memory = self.memory_transform(attended_memory) # transformed_memory: (batch_size, seq_len, dim_query)

        return x, transformed_memory
    

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
        #x (b, 1, 768) y(b, 50, 768)
        super(CrossAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.layernorm = nn.LayerNorm(embed_dim)

    def forward(self, out_seq, in_seq):
        out_seq = out_seq.transpose(0, 1)
        in_seq = in_seq.transpose(0, 1)
        attn_output, self.last_attention_scores = self.mha(out_seq, in_seq, in_seq)
        out_seq = self.layernorm(out_seq + attn_output)
        out_seq = out_seq.transpose(0, 1)
        return out_seq


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

    def forward(self, out_seq, in_seq):
        out_seq = self.self_attention(out_seq)
        out_seq = self.cross_attention(out_seq, in_seq)
        out_seq = self.ff(out_seq)
        return out_seq


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, nhead, dropout) for _ in range(num_layers)])

    def forward(self, x, memory):
        self_attention_weights_list = []
        cross_attention_weights_list = []

        for layer in self.layers:
            x = layer(x, memory)
            # self_attention_weights_list.append(self_attention_weights)
            # cross_attention_weights_list.append(cross_attention_weights)

        return x
    
if __name__ == "__main__":
    model = TransformerDecoder(num_layers=3, d_model=768, nhead=8)
    batch_size = 2
    image_embeds = torch.rand(batch_size, 1, 768)
    caption_embeds = torch.rand(batch_size, 50, 768)
    
    output = model(caption_embeds, image_embeds)
    print(output)
    print(output.shape)