import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import SwiGLUFFN


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads, alibi_alpha=1.0, start_i_increment=0, use_alibi=True):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        self.use_alibi = use_alibi

        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

        self.alibi_alpha = alibi_alpha
        self.start_i_increment = start_i_increment

        if self.use_alibi:
            # Precompute ALiBi slopes
            self.slopes = torch.tensor(
                [2 ** (-self.alibi_alpha * (i + self.start_i_increment)) for i in range(1, self.heads + 1)],
                dtype=torch.float32).view(1, self.heads, 1, 1)

    def forward(self, values, keys, queries, mask=None):
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Compute energy using einsum, simplifying matrix multiplication across batches and heads
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        # Apply ALiBi positional encodings if enabled
        if self.use_alibi:
            t_q = torch.arange(query_len, device=self.slopes.device)
            t_k = torch.arange(key_len, device=self.slopes.device)
            alibi_bias = (t_q.view(1, 1, -1, 1) - t_k.view(1, 1, 1, -1)).abs()
            alibi_bias = -alibi_bias * self.slopes
            energy += alibi_bias.to(energy.device)

        if mask is not None:                        #-1e4 for numerical stability with fp16
            energy = energy.masked_fill(mask == 0, float("-1e4"))

        attention = F.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out






# pre-LN transformer Encoder with SwiGLUFFN
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, alibi_alpha=1.0, start_i_increment=0):
        super(TransformerEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.attention = MultiHeadAttention(embed_size, heads, alibi_alpha=alibi_alpha, start_i_increment=start_i_increment)
        self.feed_forward = SwiGLUFFN(
            in_features=embed_size,
            hidden_features=forward_expansion * embed_size,
            out_features=embed_size
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        # Normalize inputs
        query_norm = self.norm1(query)
        key_norm = self.norm1(key)
        value_norm = self.norm1(value)

        # Multi-head attention using normalized values
        x = self.attention(value_norm, key_norm, query_norm, mask)
        # Apply dropout and add the residual (skip connection)
        x = query + self.dropout(x)

        # Normalize before the feed-forward network
        x = self.norm2(x)
        # Feed-forward network
        x = self.feed_forward(x)
        # Apply dropout and add the residual (skip connection)
        x = query + self.dropout(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_size, heads, num_layers, forward_expansion, dropout, alibi_alpha=1.0, start_i = 0):
        super(TransformerEncoder, self).__init__()
        self.encoder_layers = nn.ModuleList([                                                               # Index-Ramped ALiBi
            TransformerEncoderLayer(embed_size, heads, forward_expansion, dropout, alibi_alpha=alibi_alpha, start_i_increment=start_i + i)
            for i in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_length, embed_size).
            mask: Mask tensor of shape (batch_size, 1, seq_length, seq_length) or similar.
        Returns:
            The output of the last encoder layer.
        """
        # Pass the input through each encoder layer in sequence
        for layer in self.encoder_layers:
            x = layer(x, x, x, mask)  # Here x serves as query, key, and value

        return x