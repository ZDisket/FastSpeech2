from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from rotary_embedding_torch import RotaryEmbedding

import torchbnn
from torchbnn import BayesConv1d


class SwiGLU(nn.Module):
    def __init__(self, dim):
        super(SwiGLU, self).__init__()
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        """
        Pass through simple SwiGLU
        :param x:
        :return:
        """
        x = x.transpose(1, 2)
        x_proj = self.fc1(x)
        x_proj, x_gate = x_proj.chunk(2, dim=-1)
        x = x_proj * torch.sigmoid(x_gate)
        return x.transpose(1,2)

class SwiGLUConvFFN(nn.Module):
    def __init__(
            self,
            in_features: int,
            hidden_features: int = None,
            out_features: int = None,
            kernel_size: int = 3,
            drop: float = 0.0,
            bias: bool = True,
            causal: bool = False
    ):
        """
        Initializes the SwiGLU feed-forward network with Conv1D layers.

        Parameters:
            in_features (int): Input dimension of the FFN.
            hidden_features (int, optional): Inner dimension of the FFN. Defaults to in_features.
            out_features (int, optional): Output dimension of the FFN. Defaults to in_features.
            kernel_size (int, optional): Kernel size for convolution layers. Defaults to 3.
            drop (float, optional): Dropout rate. Defaults to 0.0.
            bias (bool, optional): Whether to use bias in convolution layers. Defaults to True.
            causal (bool, optional): Whether to use causal padding. Defaults to False.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.kernel_size = kernel_size
        self.causal = causal
        self.drop = nn.Dropout(drop)

        if causal:
            self.padding = self._causal_padding
        else:
            self.padding = self._same_padding

        self.conv1 = nn.Conv1d(in_features, 2 * hidden_features, kernel_size, bias=bias)
        self.conv2 = nn.Conv1d(hidden_features, out_features, kernel_size, bias=bias)

    def _causal_padding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies causal padding to the input tensor.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, seq_length).

        Returns:
            torch.Tensor: Padded tensor.
        """
        if self.kernel_size == 1:
            return x
        pad_left = self.kernel_size - 1
        pad_right = 0
        return F.pad(x, (pad_left, pad_right))

    def _same_padding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies same padding to the input tensor.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, seq_length).

        Returns:
            torch.Tensor: Padded tensor.
        """
        if self.kernel_size == 1:
            return x
        pad_left = (self.kernel_size - 1) // 2
        pad_right = self.kernel_size // 2
        return F.pad(x, (pad_left, pad_right))

    def apply_mask(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Applies a mask to the input tensor.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, seq_length).
            mask (torch.Tensor): Mask tensor of shape (batch_size, 1, 1, seq_length).

        Returns:
            torch.Tensor: Masked input tensor of shape (batch_size, channels, seq_length).
        """
        batch_size, channels, seq_length = x.shape
        if mask is not None:
            assert mask.shape == (batch_size, 1, 1, seq_length), f"Mask shape mismatch: {mask.shape}"
            mask = mask.squeeze(1)  # Reduce to (batch_size, 1, seq_length)
            x = x * mask
        return x

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through the SwiGLU Conv1D feed-forward network.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, in_features).
            mask (torch.Tensor, optional): Mask tensor of shape (batch_size, 1, seq_length, seq_length), where True is include and False exclude.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, out_features).
        """
        # Transpose for Conv1D (batch_size, channels, seq_length)
        x = x.transpose(1, 2)

        # Apply mask before the first convolution
        x = self.apply_mask(x, mask)

        x12 = self.conv1(self.padding(x))
        x1, x2 = x12.chunk(2, dim=1)

        hidden = F.silu(x1) * x2
        hidden = self.drop(hidden)

        # Apply mask before the second convolution
        hidden = self.apply_mask(hidden, mask)

        out = self.conv2(self.padding(hidden))
        out = self.drop(out)

        # Transpose back to (batch_size, seq_length, out_features)
        return out.transpose(1, 2)


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
    """
    Modern Multi Head Attention. Contains:

    num_persistent: "Augmenting Self-attention with Persistent Memory" (https://arxiv.org/abs/1907.01470)
    use_talking_heads: "Talking-Heads Attention" (https://arxiv.org/abs/2003.02436)
    use_alibi: "Attention with Linear Biases" (https://ofir.io/train_short_test_long.pdf)

    If num_persistent > 0, we call this an AllAttention layer.

    """

    def __init__(self, embed_size, heads, alibi_alpha=1.0, start_i_increment=0, use_alibi=True, use_talking_heads=True,
                 num_persistent=0):
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
        self.use_talking_heads = use_talking_heads
        self.start_i_increment = start_i_increment
        self.num_persistent = num_persistent

        if self.use_alibi:
            # Precompute ALiBi slopes
            self.slopes = torch.tensor(
                [2 ** (-self.alibi_alpha * (i + self.start_i_increment)) for i in range(1, self.heads + 1)],
                dtype=torch.float32).view(1, self.heads, 1, 1)

        if self.use_talking_heads:  # Talking heads: x-transformers version (using Conv2d instead of Linear)
            self.pre_softmax_talking_heads = nn.Conv2d(heads, heads, 1, bias=False)
            self.post_softmax_talking_heads = nn.Conv2d(heads, heads, 1, bias=False)

        if self.num_persistent > 0:
            # persistent vectors:
            # (num_persistent, heads, head_dim)
            # Could shaping the persistent vectors like this also result in inter-head communication?
            self.persistent_keys = nn.Parameter(torch.randn(self.num_persistent, self.heads, self.head_dim))
            self.persistent_values = nn.Parameter(torch.randn(self.num_persistent, self.heads, self.head_dim))

            # Initialize persistent vectors
            nn.init.kaiming_uniform_(self.persistent_keys, a=sqrt(self.num_persistent))
            nn.init.kaiming_uniform_(self.persistent_values, a=sqrt(self.num_persistent))

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

        if self.num_persistent > 0:
            expanded_persistent_keys = self.persistent_keys.unsqueeze(0).expand(N, -1, -1, -1)
            expanded_persistent_values = self.persistent_values.unsqueeze(0).expand(N, -1, -1, -1)

            # Concatenate persistent vectors to keys and values
            keys = torch.cat([keys, expanded_persistent_keys], dim=1)
            values = torch.cat([values, expanded_persistent_values], dim=1)

        # Compute energy using einsum, simplifying matrix multiplication across batches and heads
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        # Apply ALiBi positional encodings if enabled
        if self.use_alibi:
            t_q = torch.arange(query_len, device=self.slopes.device)
            t_k = torch.arange(key_len, device=self.slopes.device)
            alibi_bias = (t_q.view(1, 1, -1, 1) - t_k.view(1, 1, 1, -1)).abs()
            alibi_bias = -alibi_bias * self.slopes

            if self.num_persistent > 0:
                # Extend ALiBi bias for persistent vectors with zero bias (so that it is allowed to attend to everything)
                extended_alibi_bias = F.pad(alibi_bias, (0, self.num_persistent), "constant", 0)
                extended_alibi_bias = extended_alibi_bias.to(energy.device)
                alibi_bias = extended_alibi_bias

            energy += alibi_bias.to(energy.device)

        if self.use_talking_heads:
            energy = self.pre_softmax_talking_heads(energy)

        if mask is not None:
            if self.num_persistent > 0:
                # Extend mask to include persistent vectors (always unmasked)
                extended_mask = F.pad(mask, (0, self.num_persistent), value=1)
                extended_mask = extended_mask.expand(N, self.heads, query_len, key_len + self.num_persistent)
                mask = extended_mask
                # -1e4 for numerical stability with fp16
            energy = energy.masked_fill(mask == 0, float("-1e4"))

        attention = F.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        if self.use_talking_heads:
            attention = self.post_softmax_talking_heads(attention)

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
        self.attention = MultiHeadAttention(embed_size, heads, alibi_alpha=alibi_alpha,
                                            start_i_increment=start_i_increment)
        self.feed_forward = SwiGLUConvFFN(
            in_features=embed_size,
            hidden_features=forward_expansion * embed_size,
            out_features=embed_size,
            kernel_size=3,
            drop=0.1,
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
        x = self.feed_forward(x, mask)
        # Apply dropout and add the residual (skip connection)
        x = query + self.dropout(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_size, heads, num_layers, forward_expansion, dropout, alibi_alpha=1.0, start_i=0):
        super(TransformerEncoder, self).__init__()
        self.encoder_layers = nn.ModuleList([  # Index-Ramped ALiBi
            TransformerEncoderLayer(embed_size, heads, forward_expansion, dropout, alibi_alpha=alibi_alpha,
                                    start_i_increment=start_i + (i * heads))
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
    def __init__(self, embed_size, heads, forward_expansion, dropout, alibi_alpha, start_i_index, mode="linear",
                 kernel_size=3):
        super(TransformerDecoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)

        self.self_attention = MultiHeadAttention(embed_size, heads, alibi_alpha, start_i_index)
        self.encoder_decoder_attention = MultiHeadAttention(embed_size, heads, alibi_alpha,
                                                            start_i_index)  # Not used in isolation

        if mode == "linear":
            self.feed_forward = nn.Sequential(
                SwiGLUFFN(embed_size, forward_expansion * embed_size, embed_size),
                nn.Dropout(dropout)
            )
        elif mode == "conv":
            self.feed_forward = SwiGLUConvFFN(
                in_features=embed_size,
                hidden_features=forward_expansion * embed_size,
                out_features=embed_size,
                kernel_size=kernel_size,
                drop=0.1,
                causal=True,
            )
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
        x = self.feed_forward(x, src_mask)
        x = self.dropout(self.norm3(x))

        return x


class TransformerDecoder(nn.Module):
    def __init__(self, embed_size, heads, num_layers, forward_expansion, dropout, alibi_alpha, mode="linear",
                 kernel_size=3, start_i=0):
        super(TransformerDecoder, self).__init__()

        self.layers = nn.ModuleList([
            TransformerDecoderLayer(embed_size, heads, forward_expansion, dropout, alibi_alpha, start_i + (i * heads),
                                    mode, kernel_size)
            for i in range(num_layers)
        ])

    def forward(self, x, src_encodings, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, src_encodings, src_encodings, src_mask, tgt_mask)

        return x

class APTx(nn.Module):
    """
    APTx: Alpha Plus Tanh Times, an activation function that behaves like Mish,
    but is 2x faster.

    https://arxiv.org/abs/2209.06119
    """
    def __init__(self, alpha=1, beta=1, gamma=0.5):
        super(APTx, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, x):
        return (self.alpha + torch.tanh(self.beta * x)) * self.gamma * x

class SEBlock1D(nn.Module):
    """
    Lightweight Squeeze-Excite attention.
    """

    def __init__(self, in_channels, reduction=16):
        super(SEBlock1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
        self.x_mask = torch.zeros((1, 1 ,1))

    def set_mask(self, in_mask):
        self.x_mask = in_mask
    def forward(self, x):
        x = x[:, :, :-self.chomp_size].contiguous()
        x = x.masked_fill(self.x_mask, 0)
        return x


class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed


class TransposeRMSNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(TransposeRMSNorm, self).__init__()
        self.ln = RMSNorm(num_features, eps=eps)

    def forward(self, x):
        # Transpose from (batch, channels, seq_len) to (batch, seq_len, channels)
        x = x.transpose(1, 2)
        # Apply RMSNorm
        x = self.ln(x)
        # Transpose back from (batch, seq_len, channels) to (batch, channels, seq_len)
        x = x.transpose(1, 2)
        return x


class TransposeLayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(TransposeLayerNorm, self).__init__()
        self.ln = nn.LayerNorm(num_features, eps, affine)

    def forward(self, x):
        # Transpose from (batch, channels, seq_len) to (batch, seq_len, channels)
        x = x.transpose(1, 2)
        # Apply LayerNorm
        x = self.ln(x)
        # Transpose back from (batch, seq_len, channels) to (batch, channels, seq_len)
        x = x.transpose(1, 2)
        return x


def make_conv(bayesian, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
    return BayesConv1d(0.0, 0.1, in_channels, out_channels, kernel_size,
                       stride=stride, padding=padding, dilation=dilation) \
        if bayesian else weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size,
                                               stride=stride, padding=padding, dilation=dilation))

class ReluSquared(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2


class SwiGLUCNN(nn.Module):
    def __init__(self):
        super(SwiGLUCNN, self).__init__()

    def forward(self, x):
        """
        :param x: input tensor of shape (batch_size, dim, seq_length)
        :return: output tensor of shape (batch_size, dim // 2, seq_length)
        """
        # Split the input tensor into two equal parts along the last dimension
        x = x.transpose(1,2)
        x_proj, x_gate = x.chunk(2, dim=-1)
        # Apply the SwiGLU activation function
        x = x_proj * torch.sigmoid(x_gate)
        x = x.transpose(1, 2)
        return x

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention = SEBlock1D(in_channels, reduction)
        self.spatial_attention = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // reduction, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_out = self.channel_attention(x)
        y = self.spatial_attention(x_out)
        return x_out * y.expand_as(x_out)


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, use_se=False,
                 reduction=16, bayesian=False, use_swiglu=False, use_aptx=False, use_cbam=False):
        """
        Initialize TemporalBlock for TCN
        :param n_inputs: Number of input channels
        :param n_outputs: Output channels
        :param kernel_size: Kernel size
        :param stride: Stride of convs
        :param dilation: Dilation
        :param padding: Padding
        :param dropout: Dropout
        :param use_se: Use Squeeze-Excite attention
        :param reduction: Reduction for Squeeze-Excite, if enabled
        :param bayesian: Use Bayesian convs, for nondeterminism. Will use LayerNorm instead of weight normalization
        :param use_swiglu: Use SwiGLU for the final activation
        :param use_aptx: Use APTx for the acts
        :param use_cbam: Use CBAM at the final and at residual.
        """
        super(TemporalBlock, self).__init__()

        self.use_swiglu = use_swiglu

        self.conv1 = make_conv(bayesian, n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.ln1 = TransposeLayerNorm(n_outputs) if bayesian else nn.Identity()
        self.relu1 = APTx() if use_aptx else nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        n_outputs_orig = n_outputs
        n_final = n_outputs

        if self.use_swiglu and n_inputs == n_outputs:
            n_final = n_inputs * 2

        self.conv2 = make_conv(bayesian, n_outputs, n_final, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.ln2 = TransposeLayerNorm(n_final) if bayesian else nn.Identity()
        self.relu2 = APTx() if use_aptx else nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.ln1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.ln2, self.relu2, self.dropout2)

        n_outputs = n_final

        self.downsample = make_conv(bayesian, n_inputs, n_outputs, 1) if n_inputs != n_outputs else nn.Identity()

        if use_cbam:
            if use_se:
                print("Cannot use SE and CBAM. Using CBAM")
            use_se = False

        self.se_block = SEBlock1D(n_outputs, reduction) if use_se else nn.Identity()
        self.cbam_block = CBAM(n_outputs, reduction) if use_cbam else nn.Identity()
        self.res_cbam = CBAM(n_outputs, reduction) if use_cbam else nn.Identity()
        self.drop = nn.Dropout(0.1)

        if use_swiglu and n_inputs == n_outputs_orig:
            self.relu = SwiGLUCNN()
        else:
            self.relu = APTx() if use_aptx else nn.ReLU()

    def forward(self, x, mask=None):
        """
        Forward pass through the Temporal Block
        :param x: Tensor size (batch, in_channels, seq_len)
        :param mask: Bool mask size (batch, 1, seq_len), where True is padded and False is valid.
                    If not passed, will assume all sequence is valid.
        :return: Processed tensor size (batch, out_channels, seq_len)
        """

        if mask is None:
            mask = torch.zeros((x.size(0), 1, x.size(2))).bool().to(x.device)

        self.chomp1.set_mask(mask)
        self.chomp2.set_mask(mask)

        out = self.net(x).masked_fill(mask, 0)

        res = self.downsample(x).masked_fill(mask, 0)
        out = out + self.res_cbam(res).masked_fill(mask, 0)
        out = self.drop(out)

        # Only one of these will be valid
        out = self.se_block(out).masked_fill(mask, 0)
        out = self.cbam_block(out).masked_fill(mask, 0)
        out = self.drop(out)

        out = self.relu(out).masked_fill(mask, 0)
        return out


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, dilation_growth="exp", use_se=False,
                 bayesian=False):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)

        if not isinstance(kernel_size, list):
            kernel_size = [kernel_size] * num_levels

        for i in range(num_levels):

            if dilation_growth == "exp":
                dilation_size = 2 ** i
            elif dilation_growth == "mul":
                dilation_size = max(1, 2 * i)
            elif dilation_growth == "add":
                dilation_size = i + 1
            else:
                raise RuntimeError(f"Unknown dilation growth type {dilation_growth}")

            k_size = kernel_size[i]
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, k_size, stride=1, dilation=dilation_size,
                                     padding=(k_size - 1) * dilation_size, dropout=dropout, use_se=use_se,
                                     bayesian=bayesian)]

        self.network = nn.Sequential(*layers)

    def forward(self, x, mask):
        """
        :param x: Tensor size (batch, in_channels, seq_len)
        :param mask: Bool mask size (batch, 1, seq_len), where True is padded and False is valid.
                    If not passed, will assume all sequence is valid.
        :return: Processed tensor size (batch, out_channels, seq_len)
        """

        # TODO: Refactor the Sequential into a ModuleList; we're doing this because transfer learning
        for layer in self.network:
            x = layer(x, mask)

        return x


def mask_to_causal_attention_mask(mask):
    """
    Turn a bool mask into a causal attention mask
    :param mask: Bool sequence mask, True=padding size (batch, max_length)
    :return: Causal attention mask size (batch, 1, seq_len, seq_len), True=valid
    """
    batch_size, seq_len = mask.shape
    # Create a lower triangular matrix of ones
    causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=0).to(mask.device)
    # Expand dimensions to fit the attention mask shape (batch, 1, seq_len, seq_len)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)
    # Combine the causal mask with the input mask
    attention_mask = mask.unsqueeze(1).unsqueeze(2) & mask.unsqueeze(1).unsqueeze(3) & causal_mask
    # Flip the mask, our attention uses True=valid
    attention_mask = ~attention_mask
    return attention_mask


def reduce_mask(mask):
    """
    Reduce an attention mask to a normal one
    :param mask: Attention mask shape (batch, 1, seq_length, seq_length)

    :return: Reduced mask size (batch, 1, seq_length)
    """
    reduced_mask = mask[:, 0, :, 0].unsqueeze(1)
    return reduced_mask


class GatedRetention(nn.Module):
    """
    Allows the model to selectively retain or discard information based on the learned gate values.

    https://github.com/Mr-Twave/YOCO-Groq-BitNet-KV-cache/tree/main?tab=readme-ov-file#the-math
    """
    def __init__(self, in_channels, hidden_size):
        super(GatedRetention, self).__init__()
        self.proj = nn.Linear(in_channels, hidden_size) if in_channels != hidden_size else nn.Identity()
        self.gate = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        x = self.proj(x)
        gated_output = torch.sigmoid(self.gate(x)) * x
        return gated_output

class TCNAttentionBlock(nn.Module):
    """
    Transformer-inspired TCNAttentionBlock:

    x + Drop(AllAttention(x)) => LayerNorm => TemporalBlock => Gated Skip => Drop(LayerNorm)
    Optionally, cross-attention between context and x
    """

    def __init__(self, in_channels, out_channels, kernel_size, heads, att_dropout, dropout, dilation, alibi_alpha,
                 start_i_increment=0, bayesian=False, context_size=0, cross_att_heads=0):
        """
        Initialize the TCNAttentionBlock
        :param in_channels: Input channels
        :param out_channels: Output channels
        :param kernel_size: Kernel size of convolution
        :param heads: Attention heads. Set to 0 for no attention
        :param att_dropout: Dropout for attention
        :param dropout: General dropout
        :param dilation: Dilation in the conv kernel
        :param alibi_alpha: Alpha for ALiBi
        :param start_i_increment: Starting increment of ALiBi
        :param cross_att_heads: Heads for cross-attention between x and context. Set to 0 for no cross-att
        :param context_size: Size, in channels, of context. Will use projection if different from in_channels
        """
        super(TCNAttentionBlock, self).__init__()

        self.heads = heads
        self.cross_att_heads = cross_att_heads

        if self.heads > 0:
            self.attention = MultiHeadAttention(in_channels, heads, alibi_alpha=alibi_alpha,
                                                start_i_increment=start_i_increment,
                                                num_persistent=16)
            self.dropout1 = nn.Dropout(att_dropout)  # Dropout for attention

        if self.cross_att_heads > 0:
            self.context_proj = nn.Linear(context_size, in_channels) if context_size != in_channels else nn.Identity()

            self.cross_attention = MultiHeadAttention(in_channels, cross_att_heads, alibi_alpha=alibi_alpha,
                                                      start_i_increment=start_i_increment,
                                                      num_persistent=16)
        # A touch of insanity is all you need
        self.rotary_emb = RotaryEmbedding(dim=in_channels // 2)

        padding = (kernel_size - 1) * dilation  # Calculate padding based on dilation
        self.temporal_block = TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation,
                                            padding=padding, dropout=dropout, use_se=True, bayesian=bayesian,
                                            use_swiglu=False, use_aptx=True, use_cbam=True)

        # Gated skip connection, increases naturalness a bit
        self.gate = GatedRetention(in_channels, out_channels)
        self.post_cross_att_norm = nn.LayerNorm(in_channels) if self.cross_att_heads > 0 else nn.Identity()
        self.pre_norm = nn.LayerNorm(in_channels)
        self.norm = nn.LayerNorm(out_channels)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask, att_mask, context):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_length, channels).
            mask: Mask tensor of shape (batch_size, 1, seq_length), where True is invalid (padding) and False is valid
            att_mask: Attention mask of shape (batch_size, 1, seq_len, seq_len) where True is valid and False is invalid
            (ideally, causal)
            context: Context tensor for cross-attention, same shape and lengths as x
        """
        x_orig = x

        if self.heads > 0:
            x_att = self.attention(x, x, x, att_mask)
            x_att = self.dropout1(x_att)
            x = x + x_att  # Residual connection

        x = self.pre_norm(x)

        if self.cross_att_heads > 0:
            context = self.context_proj(context)
            x_cross_att = self.cross_attention(context, context, x, att_mask)
            x_cross_att = self.dropout1(x_cross_att)
            x = x + x_cross_att
            x = self.post_cross_att_norm(x)

        # Apply rotary embeddings before TCN
        # Yes, this actually works.
        x = self.rotary_emb.rotate_queries_or_keys(
            x.unsqueeze(1) # (batch, 1, seq_len, channels)
        ).squeeze(1) # (batch, seq_len, channels)

        x = x.transpose(1, 2)  # Switch dimensions for convolution
        x = x.masked_fill(mask, 0)

        # x = (batch, channels, seq_len)
        x = self.temporal_block(x, mask)

        x = x.masked_fill(mask, 0)
        x = x.transpose(1, 2)  # (batch, channels, seq_len) => (batch, seq_len, channels)

        x = x + self.gate(x_orig)
        x = self.norm(x)
        x = self.dropout2(x)

        return x


class ResidualBlock1D(nn.Module):
    """
    Conv1D+Squeeze-Excite+RMSNorm residual block for sequence modeling
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.3, act="relu"):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding="same")
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation, padding="same")
        self.norm1 = TransposeRMSNorm(out_channels)
        self.norm2 = TransposeRMSNorm(out_channels)
        self.se = SEBlock1D(out_channels)
        self.relu = APTx() if act == "aptx" else nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.se(out)
        out += residual
        out = self.relu(out)
        out = self.dropout(out)
        return out



class TCNAttention(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=[2, 2, 2], dropout=0.2, att_dropout=0.3, heads=[2, 2, 2],
                 alibi_alpha=1.25, start_i_increment=1, bayesian=False):
        super(TCNAttention, self).__init__()
        self.layers = nn.ModuleList()

        if len(heads) != len(num_channels):
            raise ValueError("The length of heads must be equal to the length of num_channels")
        if len(kernel_size) != len(num_channels):
            raise ValueError("The length of kernel_size must be equal to the length of num_channels")

        current_channels = num_inputs
        for level, (out_channels, num_heads, k_size) in enumerate(zip(num_channels, heads, kernel_size)):
            dilation = 1  # we want max precision, dilation is detrimental.
            is_last = level == len(num_channels) - 1

            self.layers.append(TCNAttentionBlock(current_channels, out_channels, k_size, num_heads,
                                                 att_dropout, dropout, dilation, alibi_alpha=alibi_alpha,
                                                 start_i_increment=start_i_increment + (level * num_heads),
                                                 bayesian=bayesian, cross_att_heads=2 if is_last else 0,
                                                 context_size=num_inputs,
                                                 )
                               )
            current_channels = out_channels  # The output of the current block is the input for the next

    def forward(self, x, mask):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_length, channels).
            mask: Mask tensor of shape (batch_size, seq_length), where True is invalid (padding) and False is valid
        """

        att_mask = mask_to_causal_attention_mask(mask)
        mask = mask.unsqueeze(1)
        context = x.clone()

        for layer in self.layers:
            x = layer(x, mask, att_mask, context)
        return x
