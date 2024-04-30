import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm



class SwiGLUConvFFN(nn.Module):
    """ A two-layer module with SwiGLU activation in a position-wise manner using convolutional layers """

    def __init__(self, d_in, d_hid, kernel_size, dropout=0.1):
        super(SwiGLUConvFFN, self).__init__()

        # First convolution expands the channel dimension and prepares for gating
        self.conv1 = nn.Conv1d(
            d_in,
            2 * d_hid,  # Doubling for gating purposes
            kernel_size=kernel_size[0],
            padding=(kernel_size[0] - 1) // 2,
        )

        # Second convolution brings the channel dimension back to the original
        self.conv2 = nn.Conv1d(
            d_hid,
            d_in,
            kernel_size=kernel_size[1],
            padding=(kernel_size[1] - 1) // 2,
        )

        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x  # Save for the residual connection
        x = x.transpose(1, 2)  # (Batch, Channels, Seq_Length) for convolution

        # Apply first convolution and split for gating
        x = self.conv1(x)
        x1, x2 = x.chunk(2, dim=1)  # Split along the channel dimension for gating
        x = F.silu(x1) * x2  # Apply SwiGLU activation

        # Apply second convolution
        x = self.conv2(x)

        # Restore shape to (Batch, Seq_Length, Channels)
        x = x.transpose(1, 2)

        # Apply dropout, residual connection, and layer normalization
        x = self.dropout(x)
        x = self.layer_norm(x + residual)

        return x


class SwiGLUFFN(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=None,
            drop=0.0,
            bias=True,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


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


class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, alibi_alpha, start_i_index, mode="linear", kernel_sizes=[2, 2]):
        super(TransformerDecoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)

        self.self_attention = MultiHeadAttention(embed_size, heads, alibi_alpha, start_i_index)
        self.encoder_decoder_attention = MultiHeadAttention(embed_size, heads, alibi_alpha, start_i_index)  # Not used in isolation

        if mode == "linear":
            self.feed_forward = nn.Sequential(
                SwiGLUFFN(embed_size, forward_expansion * embed_size, embed_size),
                nn.Dropout(dropout)
            )
        elif mode == "conv":
            self.feed_forward = SwiGLUConvFFN(embed_size, forward_expansion * embed_size, kernel_sizes, dropout) #SwiGLUConvFFN already integrates dropout
        else:
            raise TypeError(f"Invalid FFN type for TransformerDecoderLayer: {mode}. Valid are linear and conv")

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, tgt_mask):
        # Self-attention with look-ahead mask
        x = self.self_attention(x, x, x, tgt_mask)
        x = self.dropout(self.norm1(x))

        # Encoder-decoder attention (if you have encoder context)
        x = self.encoder_decoder_attention(x, key, value, src_mask)
        x = self.dropout(self.norm2(x))

        # Feed-forward network
        x = self.feed_forward(x)
        x = self.dropout(self.norm3(x))

        return x


class TransformerDecoder(nn.Module):
    def __init__(self, embed_size, heads, num_layers, forward_expansion, dropout, alibi_alpha, mode="linear", kernel_sizes=[2, 2]):
        super(TransformerDecoder, self).__init__()

        self.layers = nn.ModuleList([
            TransformerDecoderLayer(embed_size, heads, forward_expansion, dropout, alibi_alpha, i, mode, kernel_sizes)
            for i in range(num_layers)
        ])

    def forward(self, x, src_encodings, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, src_encodings, src_encodings, src_mask, tgt_mask)

        return x




class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
