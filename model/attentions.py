from math import sqrt
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from rotary_embedding_torch import RotaryEmbedding

import torchbnn
from torchbnn import BayesConv1d
from .attblocks import *
from .subatts import *




class PartialConv1d(torch.nn.Conv1d):
    """
    Zero padding creates a unique identifier for where the edge of the data is, such that the model can almost always identify
    exactly where it is relative to either edge given a sufficient receptive field. Partial padding goes to some lengths to remove
    this affect.
    """

    def __init__(self, *args, **kwargs):
        super(PartialConv1d, self).__init__(*args, **kwargs)
        weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0])
        self.register_buffer("weight_maskUpdater", weight_maskUpdater, persistent=False)
        slide_winsize = torch.tensor(self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2])
        self.register_buffer("slide_winsize", slide_winsize, persistent=False)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1)
            self.register_buffer('bias_view', bias_view, persistent=False)
        # caching part
        self.last_size = (-1, -1, -1)

        update_mask = torch.ones(1, 1, 1)
        self.register_buffer('update_mask', update_mask, persistent=False)
        mask_ratio = torch.ones(1, 1, 1)
        self.register_buffer('mask_ratio', mask_ratio, persistent=False)
        self.partial: bool = True

    def calculate_mask(self, input: torch.Tensor, mask_in: Optional[torch.Tensor]):
        with torch.no_grad():
            if mask_in is None:
                mask = torch.ones(1, 1, input.shape[2], dtype=input.dtype, device=input.device)
            else:
                mask = mask_in
            update_mask = F.conv1d(
                mask,
                self.weight_maskUpdater,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=1,
            )
            # for mixed precision training, change 1e-8 to 1e-6
            mask_ratio = self.slide_winsize / (update_mask + 1e-6)
            update_mask = torch.clamp(update_mask, 0, 1)
            mask_ratio = torch.mul(mask_ratio.to(update_mask), update_mask)
            return torch.mul(input, mask), mask_ratio, update_mask

    def forward_aux(self, input: torch.Tensor, mask_ratio: torch.Tensor, update_mask: torch.Tensor) -> torch.Tensor:
        assert len(input.shape) == 3

        raw_out = self._conv_forward(input, self.weight, self.bias)

        if self.bias is not None:
            output = torch.mul(raw_out - self.bias_view, mask_ratio) + self.bias_view
            output = torch.mul(output, update_mask)
        else:
            output = torch.mul(raw_out, mask_ratio)

        return output

    @torch.jit.ignore
    def forward_with_cache(self, input: torch.Tensor, mask_in: Optional[torch.Tensor] = None) -> torch.Tensor:
        use_cache = not (torch.jit.is_tracing() or torch.onnx.is_in_onnx_export())
        cache_hit = use_cache and mask_in is None and self.last_size == input.shape
        if cache_hit:
            mask_ratio = self.mask_ratio
            update_mask = self.update_mask
        else:
            input, mask_ratio, update_mask = self.calculate_mask(input, mask_in)
            if use_cache:
                # if a mask is input, or tensor shape changed, update mask ratio
                self.last_size = tuple(input.shape)
                self.update_mask = update_mask
                self.mask_ratio = mask_ratio
        return self.forward_aux(input, mask_ratio, update_mask)

    def forward_no_cache(self, input: torch.Tensor, mask_in: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.partial:
            input, mask_ratio, update_mask = self.calculate_mask(input, mask_in)
            return self.forward_aux(input, mask_ratio, update_mask)
        else:
            if mask_in is not None:
                input = torch.mul(input, mask_in)
            return self._conv_forward(input, self.weight, self.bias)

    def forward(self, input: torch.Tensor, mask_in: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.partial:
            return self.forward_with_cache(input, mask_in)
        else:
            if mask_in is not None:
                input = torch.mul(input, mask_in).to(input.device)
            return self._conv_forward(input, self.weight, self.bias)


def reduce_mask(mask):
    """
    Reduce an attention mask to a normal one
    :param mask: Attention mask shape (batch, 1, seq_length, seq_length)

    :return: Reduced mask size (batch, 1, seq_length)
    """
    reduced_mask = mask[:, 0, :, 0].unsqueeze(1)
    return reduced_mask


class SwiGLUConvFFN(nn.Module):
    def __init__(
            self,
            in_features: int,
            hidden_features: int = None,
            out_features: int = None,
            kernel_size: int = 3,
            drop: float = 0.0,
            bias: bool = True,
            causal: bool = False,
            act="swiglu",
            conv_att=False,
    ):
        """
        Initializes the SwiGLU feed-forward network with Conv1D layers.

        Parameters:
            in_features (int): Input dimension of the FFN.
            hidden_features (int, optional): Inner dimension of the FFN. Defaults to in_features.
            out_features (int, optional): Output dimension of the FFN. Defaults to in_features.
            kernel_size (int, optional): Kernel size for convolution layers. Defaults to 3. Can also pass a 2-elem list
            drop (float, optional): Dropout rate. Defaults to 0.0.
            bias (bool, optional): Whether to use bias in convolution layers. Defaults to True.
            causal (bool, optional): Whether to use causal padding. Defaults to False.
            act: What activation to use. Options are "swiglu", "relu2", "relu", and "aptx"
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        valid_acts = ["swiglu", "relu2", "aptx", "relu", "dprelu", "aptxs1"]

        if act not in valid_acts:
            raise ValueError(f"Unknown activation {act}. Valid activations are {valid_acts}")

        if not isinstance(kernel_size, list):
            kernel_size = [kernel_size] * 2

        self.kernel_size = kernel_size
        self.causal = causal
        self.drop = nn.Dropout(drop)
        self.act = act

        # wall of if statements
        # I swear im not yanderedev
        if act == "swiglu":
            self.act_fn = self._swiglu
        elif act == "relu2":
            self.act_fn = self._relu2
        elif act == "aptx" or act == "aptxs1":
            self.aptx = APTx(trainable=True) if act == "aptx" else APTxS1(trainable=True)
            self.act_fn = self._aptx
        elif act == "relu":
            self.act_fn = self._relu
        elif act == "dprelu":
            self.dprelu = DPReLU()
            self.act_fn = self._dprelu

        if causal:
            self.padding = self._causal_padding
        else:
            self.padding = self._same_padding

        expand = 2 * hidden_features if act == "swiglu" else hidden_features

        self.conv1 = nn.Conv1d(in_features, expand, kernel_size[0], bias=bias)
        self.conv2 = nn.Conv1d(hidden_features, out_features, kernel_size[1], bias=bias)
        self.lwa = nn.Sequential(CBAM(expand), nn.Dropout(0.1)) if conv_att else nn.Identity()

    def _swiglu(self, x):
        x1, x2 = x.chunk(2, dim=1)
        x = F.silu(x1) * x2
        return x

    def _relu2(self, x):
        return F.relu(x) ** 2

    def _aptx(self, x):
        return self.aptx(x)

    def _relu(self, x):
        return F.relu(x)

    def _dprelu(self, x):
        return self.dprelu(x)

    def _causal_padding(self, x: torch.Tensor, kernel_size) -> torch.Tensor:
        """
        Applies causal padding to the input tensor.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, seq_length).

        Returns:
            torch.Tensor: Padded tensor.
        """
        if kernel_size == 1:
            return x
        pad_left = kernel_size - 1
        pad_right = 0
        return F.pad(x, (pad_left, pad_right))

    def _same_padding(self, x: torch.Tensor, kernel_size) -> torch.Tensor:
        """
        Applies same padding to the input tensor.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, seq_length).

        Returns:
            torch.Tensor: Padded tensor.
        """
        if kernel_size == 1:
            return x
        pad_left = (kernel_size - 1) // 2
        pad_right = kernel_size // 2
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

        if mask.shape == (batch_size, 1, seq_length):
            x = x.masked_fill(mask, 0)
            return x

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

        x12 = self.conv1(self.padding(x, self.kernel_size[0]))
        x12 = self.lwa(x12)

        hidden = self.act_fn(x12)

        hidden = self.drop(hidden)

        # Apply mask before the second convolution
        hidden = self.apply_mask(hidden, mask)

        out = self.conv2(self.padding(hidden, self.kernel_size[1]))
        out = self.drop(out)

        # Transpose back to (batch_size, seq_length, out_features)
        return out.transpose(1, 2)






class MultiHeadAttention(nn.Module):
    """
    Modern Multi Head Attention. Contains:

    num_persistent: "Augmenting Self-attention with Persistent Memory" (https://arxiv.org/abs/1907.01470)
    use_talking_heads: "Talking-Heads Attention" (https://arxiv.org/abs/2003.02436)
    use_alibi: "Attention with Linear Biases" (https://ofir.io/train_short_test_long.pdf)
    rma_inp_dim: Recurrent Memory Attention (my invention). Per-head dim for projection, if necessary.

    If num_persistent > 0, we call this an AllAttention layer.

    """

    def __init__(self, embed_size, heads, alibi_alpha=1.0, start_i_increment=0, use_alibi=True, use_talking_heads=True,
                 num_persistent=0, rma_inp_dim=None):
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
            # (num_persistent, 1, head_dim)
            self.persistent_keys = nn.Parameter(torch.randn(self.num_persistent, 1, self.head_dim))
            self.persistent_values = nn.Parameter(torch.randn(self.num_persistent, 1, self.head_dim))

            # Initialize persistent vectors
            nn.init.kaiming_uniform_(self.persistent_keys, a=sqrt(self.num_persistent))
            nn.init.kaiming_uniform_(self.persistent_values, a=sqrt(self.num_persistent))

            if rma_inp_dim is not None:
                self.rma_k_proj = GatedRetention(rma_inp_dim, self.head_dim)
                self.rma_v_proj = GatedRetention(rma_inp_dim, self.head_dim)


    def forward(self, values, keys, queries, mask=None, recurr_persistent=None):
        """
        Do attention
        :param values: Values
        :param keys: Keys
        :param queries: Queries
        :param mask: Attention mask
        :param recurr_persistent: Packed tuple (keys, values) of recurrent persistent memory
        :return: Attentioned tensor
        """
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        current_persistent = self.num_persistent

        if current_persistent > 0:
            p_keys = self.persistent_keys
            p_values = self.persistent_values

            if recurr_persistent is not None:
                recurr_keys, recurr_values = recurr_persistent

                recurr_keys = self.rma_k_proj(recurr_keys)
                recurr_values = self.rma_v_proj(recurr_values)

                # Concat the recurrent ones before ours along the seq dim
                p_keys = torch.cat([recurr_keys, p_keys], dim=0)
                p_values = torch.cat([recurr_values, p_values], dim=0)

                current_persistent = p_keys.size(0)

            expanded_persistent_keys = p_keys.unsqueeze(0).expand(N, -1, self.heads, -1)
            expanded_persistent_values = p_values.unsqueeze(0).expand(N, -1, self.heads, -1)

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

            if current_persistent > 0:
                # Extend ALiBi bias for persistent vectors with zero bias (so that it is allowed to attend to everything)
                extended_alibi_bias = F.pad(alibi_bias, (0, current_persistent), "constant", 0)
                extended_alibi_bias = extended_alibi_bias.to(energy.device)
                alibi_bias = extended_alibi_bias

            energy += alibi_bias.to(energy.device)

        if self.use_talking_heads:
            energy = self.pre_softmax_talking_heads(energy)

        if mask is not None:
            if current_persistent > 0:
                # Extend mask to include persistent vectors (always unmasked)
                extended_mask = F.pad(mask, (0, current_persistent), value=1)
                extended_mask = extended_mask.expand(N, self.heads, query_len, key_len + current_persistent)
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
    def __init__(self, embed_size, heads, forward_expansion, dropout, alibi_alpha=1.0, start_i_increment=0,
                 kernel_size=3, act="swiglu", rma_mem_dim=0, conv_att=False):
        super(TransformerEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.use_rma = rma_mem_dim > 0
        self.attention = MultiHeadAttention(embed_size, heads, alibi_alpha=alibi_alpha,
                                            start_i_increment=start_i_increment, num_persistent=rma_mem_dim,
                                            rma_inp_dim=embed_size // heads if self.use_rma else 0)
        self.feed_forward = SwiGLUConvFFN(
            in_features=embed_size,
            hidden_features=forward_expansion * embed_size,
            out_features=embed_size,
            kernel_size=kernel_size,
            drop=0.1,
            act=act,
            conv_att=conv_att,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask, conv_mask=None, mem_kv=None):
        # Normalize inputs
        query_norm = self.norm1(query)
        key_norm = self.norm1(key)
        value_norm = self.norm1(value)

        # Multi-head attention using normalized values
        x = self.attention(value_norm, key_norm, query_norm, mask, mem_kv)
        # Apply dropout and add the residual (skip connection)
        x = query + self.dropout(x)

        # Normalize before the feed-forward network
        x = self.norm2(x)
        # Feed-forward network
        x = self.feed_forward(x, mask if conv_mask is None else conv_mask)
        # Apply dropout and add the residual (skip connection)
        x = query + self.dropout(x)

        kv_ret = (self.attention.persistent_keys, self.attention.persistent_values) if self.use_rma else None
        return x, kv_ret


class TransformerEncoder(nn.Module):
    def __init__(self, embed_size, heads, num_layers, forward_expansion, dropout, alibi_alpha=1.0, start_i=0,
                 kernel_size=3, act="swiglu", rma_mem_dim=0, conv_att=False, multi_scale=False):
        super(TransformerEncoder, self).__init__()
        self.use_conv_att = conv_att
        self.encoder_layers = nn.ModuleList([  # Index-Ramped ALiBi
            TransformerEncoderLayer(embed_size, heads, forward_expansion, dropout, alibi_alpha=alibi_alpha,
                                    start_i_increment=start_i + (i * heads), kernel_size=[kernel_size[i], 1] if multi_scale else kernel_size, act=act,
                                    rma_mem_dim=rma_mem_dim, conv_att=self.use_conv_att and i == num_layers - 1)
            for i in range(num_layers)
        ])
        self.head_dim = embed_size // heads
        self.rma_mem_dim = rma_mem_dim
        self.dropout = nn.Dropout(dropout)
        self.use_rma = self.rma_mem_dim > 0

        if self.use_rma:
            self.kv_proj = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
            )

    def forward(self, x, mask, conv_mask=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_length, embed_size).
            mask: Mask tensor of shape (batch_size, 1, seq_length, seq_length) or similar.
            conv_mask: Convolutional mask size (batch, 1, seq_length) where True is padded and False is valid
        Returns:
            The output of the last encoder layer.
        """
        # Pass the input through each encoder layer in sequence

        recurr_keys = None
        recurr_values = None

        for layer in self.encoder_layers:
            x, current_kv = layer(x, x, x,
                                  mask, conv_mask, (recurr_keys, recurr_values) if recurr_keys is not None else None)  # Here x serves as query, key, and value

            if self.use_rma:
                key_r, val_r = current_kv

                # prevent backpropagation into the previous layers
                # otherwise, it tries to optimize each attention layer for the next
                # faster and better loss
                key_r = self.kv_proj(key_r.detach())
                val_r = self.kv_proj(val_r.detach())

                key_r = reduce_sequence_length(key_r)
                val_r = reduce_sequence_length(val_r)

                # Collect recurrent key-values
                recurr_keys = key_r if recurr_keys is None else torch.cat([recurr_keys, key_r], dim=0)
                recurr_values = val_r if recurr_values is None else torch.cat([recurr_values, val_r], dim=0)

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
                act="relu2"
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


def make_conv(bayesian, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
    return BayesConv1d(0.0, 0.1, in_channels, out_channels, kernel_size,
                       stride=stride, padding=padding, dilation=dilation) \
        if bayesian else weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size,
                                               stride=stride, padding=padding, dilation=dilation))

def perturb(x, mul=0.1):
    return F.dropout(x, mul, training=True)


class SuperTemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, bayesian=False,
                 use_aptx=False,
                 reduction=16, heads=4, start_i_increment=0, cross_att_heads=0, alibi_alpha=1.2, rma_head_dim=0,
                 att_dropout=0.1,
                 context_size=0, noise_scale=0.3):
        """
        Initialize SuperTemporalBlock for TCN
        :param n_inputs: Number of input channels
        :param n_outputs: Output channels
        :param kernel_size: Kernel size
        :param stride: Stride of convs
        :param dilation: Dilation
        :param padding: Padding
        :param dropout: Dropout
        :param use_se: Use Squeeze-Excite attention
        :param bayesian: Use Bayesian convs, for nondeterminism. Will use LayerNorm instead of weight normalization
        :param use_aptx: Use APTx for the acts
        :param noise_scale: Scale of noise to perturb inputs for increased variability. Set to 0 if not using Bayesian
        (Not functional)
        """
        super(SuperTemporalBlock, self).__init__()

        self.noise_scale = noise_scale

        self.conv1 = make_conv(bayesian, n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.ln1 = TransposeLayerNorm(n_outputs) if bayesian else nn.Identity()
        self.relu1 = APTx() if use_aptx else nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        n_final = n_outputs

        self.conv2 = make_conv(bayesian, n_outputs, n_final, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.ln2 = TransposeLayerNorm(n_final) if bayesian else nn.Identity()
        self.relu2 = APTx() if use_aptx else nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net_part_1 = nn.Sequential(self.conv1, self.chomp1, self.ln1, self.relu1, self.dropout1)
        self.net_part_2 = nn.Sequential(self.conv2, self.chomp2, self.ln2, self.relu2, self.dropout2)

        n_outputs = n_final

        self.downsample = make_conv(False, n_inputs, n_outputs, 1) if n_inputs != n_outputs else nn.Identity()

        self.cbam_block = CBAM(n_outputs, reduction)
        self.res_cbam = CBAM(n_outputs, reduction)
        self.drop = nn.Dropout(0.1)

        self.relu = APTx(trainable=True) if use_aptx else nn.ReLU()

        self.heads = heads
        self.cross_att_heads = cross_att_heads

        if self.heads > 0:
            self.attention = MultiHeadAttention(n_outputs, heads, alibi_alpha=alibi_alpha,
                                                start_i_increment=start_i_increment,
                                                num_persistent=16, rma_inp_dim=rma_head_dim)
            self.att_drop = nn.Dropout(att_dropout)  # Dropout for attention
            self.att_norm = nn.LayerNorm(n_outputs)

        if self.cross_att_heads > 0:
            self.context_proj = nn.Linear(context_size, n_outputs) if context_size != n_outputs else nn.Identity()

            self.cross_attention = MultiHeadAttention(n_outputs, cross_att_heads, alibi_alpha=alibi_alpha,
                                                      start_i_increment=start_i_increment,
                                                      num_persistent=16)


    def forward(self, x, mask=None, att_mask=None, context=None, packed_kv=None):
        """
        Forward pass through the Temporal Block
        :param x: Tensor size (batch, seq_len, in_channels)
        :param mask: Bool mask size (batch, 1, seq_len), where True is padded and False is valid.
                    If not passed, will assume all sequence is valid.
        :param context: Context tensor same shape as x for cross attention
        :packed_kv: (key, value) packed for RMA
        :return: Processed tensor size (batch, seq_len, in_channels)
        """

        if mask is None:
            mask = torch.zeros((x.size(0), 1, x.size(2))).bool().to(x.device)

        self.chomp1.set_mask(mask)
        self.chomp2.set_mask(mask)

        x = x.transpose(1, 2)

        # keep orig for residual block
        x0 = x

        x = self.net_part_1(x).masked_fill(mask, 0)

        if self.heads > 0:
            x = x.transpose(1, 2)
            x_att = self.attention(x, x, x, att_mask, packed_kv)
            x_att = self.att_drop(x_att)
            x = x + x_att
            x = self.att_norm(x)
            x = x.transpose(1, 2)

        x = x.masked_fill(mask, 0)
        x = self.net_part_2(x)

        res = self.downsample(x0).masked_fill(mask, 0)
        x = x + self.res_cbam(res).masked_fill(mask, 0)
        x = self.drop(x)

        x = self.cbam_block(x).masked_fill(mask, 0)

        x = self.drop(x)

        if self.cross_att_heads > 0:
            x = x.transpose(1, 2)
            context = self.context_proj(context)
            x_cross_att = self.cross_attention(context, context, x, att_mask)
            x_cross_att = self.att_drop(x_cross_att)
            x = x + x_cross_att
            x = x.transpose(1, 2)

        x = self.relu(x).masked_fill(mask, 0)
        x = x.transpose(1, 2)
        return x, (self.attention.persistent_keys, self.attention.persistent_values)


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

        # Must do: Refactor the Sequential into a ModuleList; we're doing this because transfer learning
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


class TCNAttentionBlock(nn.Module):
    """
    Transformer-inspired TCNAttentionBlock:

    x + Drop(AllAttention(x)) => LayerNorm => TemporalBlock => Gated Skip => Drop(LayerNorm)
    Optionally, cross-attention between context and x
    """

    def __init__(self, in_channels, out_channels, kernel_size, heads, att_dropout, dropout, dilation, alibi_alpha,
                 start_i_increment=0, bayesian=False, context_size=0, cross_att_heads=0, rma_head_dim=None):
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
        :param rma_head_dim: Head dim of last layer for Recurrent Memory Attention
        """
        super(TCNAttentionBlock, self).__init__()

        self.heads = heads
        self.cross_att_heads = cross_att_heads

        if self.heads > 0:
            self.attention = MultiHeadAttention(in_channels, heads, alibi_alpha=alibi_alpha,
                                                start_i_increment=start_i_increment,
                                                num_persistent=16, rma_inp_dim=rma_head_dim)
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

    def forward(self, x, mask, att_mask, context, recurr_kv):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_length, channels).
            mask: Mask tensor of shape (batch_size, 1, seq_length), where True is invalid (padding) and False is valid
            att_mask: Attention mask of shape (batch_size, 1, seq_len, seq_len) where True is valid and False is invalid
            (ideally, causal)
            recurr_kv: Tuple of recurrent consistent tokens for the attention (key, values)
            context: Context tensor for cross-attention, same shape and lengths as x
        """
        x_orig = x

        if self.heads > 0:
            x_att = self.attention(x, x, x, att_mask, recurr_kv)
            x_att = self.dropout1(x_att)
            x = x + x_att  # Residual connection

        x = self.pre_norm(x)

        if self.cross_att_heads > 0:
            context = self.context_proj(context)
            x_cross_att = self.cross_attention(context, context, x, att_mask)
            x_cross_att = self.dropout1(x_cross_att)
            x = x + x_cross_att
            x = self.post_cross_att_norm(x)

        x = x.transpose(1, 2)  # Switch dimensions for convolution
        x = x.masked_fill(mask, 0)

        # x = (batch, channels, seq_len)
        x = self.temporal_block(x, mask)

        x = x.masked_fill(mask, 0)
        x = x.transpose(1, 2)  # (batch, channels, seq_len) => (batch, seq_len, channels)

        x = x + self.gate(x_orig)
        x = self.norm(x)
        x = self.dropout2(x)

        return x, (self.attention.persistent_keys, self.attention.persistent_values)


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

        self.residual = nn.Conv1d(in_channels, out_channels,
                                  kernel_size=1) if in_channels != out_channels else nn.Identity()

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


def reduce_sequence_length(input_tensor):
    """
    Reduces the sequence length of a tensor by half using 1D max pooling.

    Args:
    input_tensor (torch.Tensor): Input tensor of shape (seq_len, N, dim).

    Returns:
    torch.Tensor: Output tensor with reduced sequence length, shape (seq_len//2, N, dim).
    """
    seq_len, N, dim = input_tensor.shape

    # Ensure seq_len is even for the 1D max pooling to reduce by half
    assert seq_len % 2 == 0, "Sequence length should be even for halving with max pooling"

    # Permute the tensor to (N, dim, seq_len) for 1D max pooling
    input_tensor_permuted = input_tensor.permute(1, 2, 0)  # (N, dim, seq_len)

    # Apply 1D max pooling with kernel size 2 and stride 2
    output_tensor_permuted = F.max_pool1d(input_tensor_permuted, kernel_size=2, stride=2)

    # Permute back to original dimensions (seq_len//2, N, dim)
    output_tensor = output_tensor_permuted.permute(2, 0, 1)  # (seq_len//2, N, dim)

    return output_tensor


class TCNAttention(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=[2, 2, 2], dropout=0.2, att_dropout=0.3, heads=[2, 2, 2],
                 alibi_alpha=1.25, start_i_increment=1, bayesian=False, integrated=False):
        super(TCNAttention, self).__init__()
        self.layers = nn.ModuleList()

        self.key_projs = nn.ModuleList()
        self.val_projs = nn.ModuleList()

        if len(heads) != len(num_channels):
            raise ValueError("The length of heads must be equal to the length of num_channels")
        if len(kernel_size) != len(num_channels):
            raise ValueError("The length of kernel_size must be equal to the length of num_channels")

        current_channels = num_inputs
        # Keep a global head dim so that we don't have to deal with varying head dimensions when collecting the recurrent states
        self.global_head_dim = 64

        if integrated:
            print("Using SuperTemporalBlocks")

        for level, (out_channels, num_heads, k_size) in enumerate(zip(num_channels, heads, kernel_size)):
            dilation = 1  # we want max precision, dilation is detrimental.
            is_last = level == len(num_channels) - 1
            curr_i_increment = start_i_increment + (level * num_heads)
            c_att_heads = 2 if is_last else 0

            if not integrated:
                self.layers.append(TCNAttentionBlock(current_channels, out_channels, k_size, num_heads,
                                                     att_dropout, dropout, dilation, alibi_alpha=alibi_alpha,
                                                     start_i_increment=curr_i_increment,
                                                     bayesian=bayesian, cross_att_heads=2 if is_last else 0,
                                                     context_size=num_inputs, rma_head_dim=self.global_head_dim
                                                     )
                                   )
            else:
                self.layers.append(SuperTemporalBlock(current_channels, out_channels, k_size, 1, dilation,
                                                      padding=(k_size - 1) * dilation,
                                                      dropout=dropout, bayesian=bayesian, use_aptx=True, reduction=16,
                                                      heads=num_heads,
                                                      start_i_increment=curr_i_increment, cross_att_heads=c_att_heads,
                                                      alibi_alpha=alibi_alpha,
                                                      rma_head_dim=self.global_head_dim, att_dropout=att_dropout,
                                                      context_size=num_inputs)
                                   )

            current_channels = out_channels  # The output of the current block is the input for the next
            layer_head_dim = self.layers[-1].attention.head_dim

            self.key_projs.append(
                nn.Sequential(
                    nn.Linear(layer_head_dim,
                              self.global_head_dim) if layer_head_dim != self.global_head_dim else nn.Identity(),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1),
                ),
            )
            self.val_projs.append(
                nn.Sequential(
                    nn.Linear(layer_head_dim,
                              self.global_head_dim) if layer_head_dim != self.global_head_dim else nn.Identity(),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1),
                ),
            )

    def forward(self, x, mask):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_length, channels).
            mask: Mask tensor of shape (batch_size, seq_length), where True is invalid (padding) and False is valid
        """

        att_mask = mask_to_causal_attention_mask(mask)
        mask = mask.unsqueeze(1)
        context = x.clone()

        recurr_keys = None
        recurr_values = None

        for i, layer in enumerate(self.layers):
            x, current_kv = layer(x, mask, att_mask, context,
                                  (recurr_keys, recurr_values) if recurr_keys is not None else None)

            key_r, val_r = current_kv

            # prevent backpropagation into the previous layers
            # otherwise, it tries to optimize each attention layer for the next
            # faster and better loss
            key_r = self.key_projs[i](key_r.detach())
            val_r = self.val_projs[i](val_r.detach())

            key_r = reduce_sequence_length(key_r)
            val_r = reduce_sequence_length(val_r)

            # Collect recurrent key-values
            recurr_keys = key_r if recurr_keys is None else torch.cat([recurr_keys, key_r], dim=0)
            recurr_values = val_r if recurr_values is None else torch.cat([recurr_values, val_r], dim=0)

        return x
