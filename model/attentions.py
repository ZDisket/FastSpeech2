from math import sqrt
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
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

        valid_acts = ["swiglu", "relu2", "aptx", "relu", "dprelu", "aptxs1", "relugtz", "relugt"]

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
        elif act == "relugt":
            self.dprelu = ReLUGT()
            self.act_fn = self._dprelu
        elif act == "relugtz":
            self.dprelu = ReLUGT()
            self.act_fn = self._relugtz

        if causal:
            self.padding = self._causal_padding
        else:
            self.padding = self._same_padding

        expand = 2 * hidden_features if act in ["swiglu", "relugtz"] else hidden_features

        self.conv1 = nn.Conv1d(in_features, expand, kernel_size[0], bias=bias)
        self.conv2 = nn.Conv1d(hidden_features, out_features, kernel_size[1], bias=bias)
        self.lwa = MaskedCBAM1d(expand) if conv_att else None

    def _swiglu(self, x):
        x1, x2 = x.chunk(2, dim=1)
        x = F.silu(x1) * x2
        return x

    def _relugtz(self, x):
        x1, x2 = x.chunk(2, dim=1)
        x = self.dprelu(x1) * x2
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

    def apply_mask(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[Tensor, Tensor]:
        """
        Applies a mask to the input tensor.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, seq_length).
            mask (torch.Tensor): Mask tensor of shape (batch_size, 1, 1, seq_length).

        Returns:
            torch.Tensor: Masked input tensor of shape (batch_size, channels, seq_length).
            torch.Tensor: Mask.
        """
        batch_size, channels, seq_length = x.shape

        if mask.shape == (batch_size, 1, seq_length):
            x = x.masked_fill(mask, 0)
            return x, mask

        if mask is not None:
            assert mask.shape == (batch_size, 1, 1, seq_length), f"Mask shape mismatch: {mask.shape}"
            mask = mask.squeeze(1)  # Reduce to (batch_size, 1, seq_length)
            x = x * mask

            mask = mask.bool()
            mask = ~mask
        return x, mask

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through the SwiGLU Conv1D feed-forward network.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, in_features).
            mask (torch.Tensor, optional): Mask tensor of shape (batch_size, 1, seq_length, seq_length), where True is include and False exclude.
            OR  (batch_size, 1, seq_length) where True is exclude and False include

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, out_features).
        """
        # Transpose for Conv1D (batch_size, channels, seq_length)
        x = x.transpose(1, 2)

        # Apply mask before the first convolution
        x, c_mask = self.apply_mask(x, mask)

        x12 = self.conv1(self.padding(x, self.kernel_size[0]))

        if self.lwa is not None:
            x12 = self.lwa(x12, c_mask)

        hidden = self.act_fn(x12)

        hidden = self.drop(hidden)

        # Apply mask before the second convolution
        hidden, _ = self.apply_mask(hidden, mask)

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
    weighted_heads: Weighted Heads Attention. Keep trainable scalar weights for each head, which are used to multiply just
    before the final projection, in order to allow the model to dynamically prioritize heads (Decreases performance, don't use)
    dynamic_alibi: Dynamic ALiBi. Keep per-head trainable multipliers to dynamically adjust the slopes as it trains.


    If num_persistent > 0, we call this an AllAttention layer.

    """

    def __init__(self, embed_size, heads, alibi_alpha=1.0, start_i_increment=0, use_alibi=True, use_talking_heads=True,
                 num_persistent=0, rma_inp_dim=None, weighted_heads=False, dynamic_alibi=False):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        self.use_alibi = use_alibi
        self.dynamic_alibi = dynamic_alibi

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
        self.weighted_heads = weighted_heads

        if self.use_alibi:
            # Precompute ALiBi slopes
            self.slopes = torch.tensor(
                [2 ** (-self.alibi_alpha * (i + self.start_i_increment)) for i in range(1, self.heads + 1)],
                dtype=torch.float32).view(1, self.heads, 1, 1)

            if self.dynamic_alibi:
                self.alibi_betas = nn.Parameter(torch.ones(self.heads).view(1, self.heads, 1, 1))

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

        if self.weighted_heads:
            self.head_weights = nn.Parameter(torch.ones(self.heads))


    def forward(self, values, keys, queries, mask=None, recurr_persistent=None, return_weights=False):
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
            self.slopes = self.slopes.to(energy.device)

            t_q = torch.arange(query_len, device=energy.device)
            t_k = torch.arange(key_len, device=energy.device)
            alibi_bias = (t_q.view(1, 1, -1, 1) - t_k.view(1, 1, 1, -1)).abs()

            if self.dynamic_alibi:
                alibi_bias = -alibi_bias * (self.slopes * self.alibi_betas)
            else:
                alibi_bias = -alibi_bias * self.slopes

            if current_persistent > 0:
                # Extend ALiBi bias for persistent vectors with zero bias (so that it is allowed to attend to everything)
                extended_alibi_bias = F.pad(alibi_bias, (0, current_persistent), "constant", 0)
                extended_alibi_bias = extended_alibi_bias.to(energy.device)
                alibi_bias = extended_alibi_bias

            energy += alibi_bias

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

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values])

        if self.weighted_heads: # (batch, len, n_heads, head_dim)
            out = out * self.head_weights.view(1, 1, -1, 1)

        out = out.reshape(N,query_len, self.heads * self.head_dim)

        out = self.fc_out(out)

        if not return_weights:
            return out
        else:
            return out, attention


def expand_masks(x_mask, y_mask):
    """
    Expand True=padded masks into an attention mask.
    Inputs can be different or the same
    :param x_mask: Mask of x size (batch, seq_len), where True is padded
    :param y_mask: Mask of y size (batch, seq_2_len), where True is padded
    :return: Attention mask for MultiHeadAttention
    """
    x_mask_expanded = x_mask.unsqueeze(1).unsqueeze(3)  # Shape: (batch_size, 1, mel_len, 1)
    y_mask_expanded = y_mask.unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1, duration_len)
    # Combine masks using broadcasting
    attention_mask = x_mask_expanded & y_mask_expanded  # Shape: (batch_size, 1, mel_len, duration_len)
    attention_mask = ~attention_mask  # True=padded => True=valid
    return attention_mask


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, alibi_alpha=1.0, start_i_increment=0,
                 kernel_size=3, act="swiglu", rma_mem_dim=0, conv_att=False, talking_heads=True, coarse_fine=False, dynamic_alibi=False):
        super(TransformerEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.use_rma = rma_mem_dim > 0
        self.coarse_fine = coarse_fine

        self.attention = MultiHeadAttention(embed_size, heads, alibi_alpha=alibi_alpha,
                                            start_i_increment=start_i_increment, num_persistent=rma_mem_dim,
                                            rma_inp_dim=embed_size // heads if self.use_rma else 0, use_talking_heads=talking_heads,
                                            dynamic_alibi=dynamic_alibi)

        if self.coarse_fine:
            self.coarse_attention = MultiHeadAttention(embed_size, 1, alibi_alpha=alibi_alpha,
                                                start_i_increment=start_i_increment, num_persistent=0,
                                                rma_inp_dim=0,
                                                use_talking_heads=False)
            self.norm3 = nn.LayerNorm(embed_size)

        self.feed_forward = SwiGLUConvFFN(
            in_features=embed_size,
            hidden_features=forward_expansion * embed_size,
            out_features=embed_size,
            kernel_size=kernel_size,
            drop=dropout,
            act=act,
            conv_att=conv_att,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, conv_mask=None, mem_kv=None, coarse_features=None, coarse_mask=None):
        # Compute normalized x for coarse attention if needed, using the original x
        if self.coarse_fine:
            norm_x_coarse = self.norm3(x)

        # Primary attention
        norm_x_primary = self.norm1(x)
        attn_output = self.attention(norm_x_primary, norm_x_primary, norm_x_primary, mask, mem_kv)
        x = x + self.dropout(attn_output)

        # Coarse attention (if applicable)
        if self.coarse_fine:
            coarse_fine_attn_mask = expand_masks(conv_mask.squeeze(1), coarse_mask.squeeze(1))
            coarse_attn_output = self.coarse_attention(coarse_features, coarse_features, norm_x_coarse,
                                                       coarse_fine_attn_mask)
            x = x + self.dropout(coarse_attn_output)

        # Feed-forward
        norm_x_ff = self.norm2(x)
        ff_output = self.feed_forward(norm_x_ff, mask if conv_mask is None else conv_mask)
        x = x + self.dropout(ff_output)

        # Return persistent key-value pairs if using RMA
        kv_ret = (self.attention.persistent_keys, self.attention.persistent_values) if self.use_rma else None
        return x, kv_ret


class TransformerEncoder(nn.Module):
    def __init__(self, embed_size, heads, num_layers, forward_expansion, dropout, alibi_alpha=1.0, start_i=0,
                 kernel_size=3, act="swiglu", rma_mem_dim=0, conv_att=False, multi_scale=False, talking_heads=True, coarse_fine=False,
                 dynamic_alibi=False):
        super(TransformerEncoder, self).__init__()
        self.use_conv_att = conv_att
        self.coarse_fine = coarse_fine

        # Our design is coarse fine attention for all layers except the first.
        coarse_fine_vec = [self.coarse_fine] * num_layers
        # if coarse_fine=True, this will be all True except for the first layer (what we want)
        coarse_fine_vec[0] = False

        self.encoder_layers = nn.ModuleList([  # Layer-Scaled ALiBi
            TransformerEncoderLayer(embed_size, heads, forward_expansion, dropout, alibi_alpha=alibi_alpha,
                                    start_i_increment=start_i + (i * heads), kernel_size=[kernel_size[i], 1] if multi_scale else kernel_size, act=act,
                                    rma_mem_dim=rma_mem_dim, conv_att=self.use_conv_att and i == num_layers - 1, talking_heads=talking_heads, coarse_fine=coarse_fine_vec[i],
                                    dynamic_alibi=dynamic_alibi)
            for i in range(num_layers)
        ])
        self.head_dim = embed_size // heads
        self.rma_mem_dim = rma_mem_dim
        self.dropout = nn.Dropout(dropout)
        self.use_rma = self.rma_mem_dim > 0

        if self.use_rma:
            self.kv_proj = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
            )

        if self.coarse_fine:
            self.coarse_projs = nn.ModuleList(
                [
                    nn.Sequential(nn.Conv1d(embed_size, embed_size, 5, 2),
                                  nn.ReLU(),
                                  nn.Dropout(0.1))
                    for _ in range(num_layers - 1)
                ]
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
        coarse_x, coarse_mask = x, conv_mask

        for i, layer in enumerate(self.encoder_layers):
            x, current_kv = layer(x, mask, conv_mask, (recurr_keys, recurr_values) if recurr_keys is not None else None,
                                  coarse_x, coarse_mask)  # Here x serves as query, key, and value

            # Break at the last layer after processing;
            # due to the design of coarse_fine, we have n_layers - 1 projections, which will trigger an IndexError
            # when it tries to process coarse features after the last layer
            if i == len(self.encoder_layers) - 1:
                break

            if self.coarse_fine:
                # Neat 'lil trick: Max pooling with same args as the conv will compress the mask for us
                coarse_mask = F.max_pool1d(conv_mask.float(), kernel_size=5, stride=2).bool()

                coarse_x = self.coarse_projs[i](
                    x.transpose(1,2)
                ).masked_fill(coarse_mask, 0).transpose(1,2)

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

def make_conv(bayesian, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, WN=True):
    if bayesian:
        return BayesConv1d(0.0, 0.1, in_channels, out_channels, kernel_size,
                           stride=stride, padding=padding, dilation=dilation)
    else:
        if WN:
            return weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size,
                                stride=stride, padding=padding, dilation=dilation))
        else:
            return nn.Conv1d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, dilation=dilation)


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


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=self.padding, dilation=dilation)

    def forward(self, x):
        return self.conv(x)[:, :, :-self.padding]


class ConvReluNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, normalization='layer', act="aptx", dropout=0.5, causal=True):
        super(ConvReluNorm, self).__init__()
        self.causal = causal

        if self.causal:
            self.causal_conv = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        else:
            self.causal_conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation, padding="same")

        if act == "relu":
            self.act = nn.ReLU()
        elif act == "aptx":
            self.act = APTx()
        elif act == "taptx":
            self.act = APTx(trainable=True)
        elif act == "dprelu":
            self.act = DPReLU()

        self.drop = nn.Dropout(dropout)

        if normalization == 'batch':
            self.norm = nn.BatchNorm1d(out_channels)
        elif normalization == 'layer':
            self.norm = TransposeLayerNorm(out_channels)
        elif normalization == "":
            self.norm = nn.Identity()
        else:
            raise ValueError("Normalization type must be either 'batch', 'layer', or an empty string (none)")

    def forward(self, x, x_mask):
        x = self.causal_conv(x).masked_fill(x_mask, 0)
        x = self.act(x).masked_fill(x_mask, 0)
        x = self.norm(x).masked_fill(x_mask, 0)
        x = self.drop(x)
        return x


class NeoTCNAttention(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=[2, 2, 2], dropout=0.2, att_dropout=0.3, heads=[2, 2, 2],
                 alibi_alpha=1.25, start_i_increment=1, dilation_growth="", act="aptx", bayesian=False, integrated=False, conv_att="se"):
        super(NeoTCNAttention, self).__init__()

        self.layers = nn.ModuleList()
        self.att_layers = nn.ModuleList()
        self.out_channels = num_channels[-1]
        self.n_heads = heads
        self.attn_dropout = nn.Dropout(att_dropout)

        if len(heads) != len(num_channels):
            raise ValueError("The length of heads must be equal to the length of num_channels")
        if len(kernel_size) != len(num_channels):
            raise ValueError("The length of kernel_size must be equal to the length of num_channels")

        current_channels = num_inputs

        for level, (out_channels, num_heads, k_size) in enumerate(zip(num_channels, heads, kernel_size)):
            is_last = level == len(num_channels) - 1
            curr_i_increment = start_i_increment + (level * num_heads)

            if dilation_growth == "exp":
                dilation_size = 2 ** level
            elif dilation_growth == "mul":
                dilation_size = max(1, 2 * level)
            elif dilation_growth == "add":
                dilation_size = level + 1
            elif dilation_growth == "":
                dilation_size = 1
            else:
                raise RuntimeError(f"Unknown dilation growth type {dilation_growth}")

            # pre-attention arrangement
            self.att_layers.append(
                MultiHeadAttention(current_channels, num_heads, alibi_alpha=alibi_alpha,
                                   start_i_increment=curr_i_increment,
                                   num_persistent=16)
                if num_heads > 0 else nn.Identity()  # append an identity so it still occupies an index i in the list

            )
            self.layers.append(ConvReluNorm(current_channels, out_channels, k_size, dilation_size, act=act, dropout=dropout, causal=False))

            current_channels = out_channels  # The output of the current block is the input for the next

        if conv_att == "se":
            self.conv_att = MaskedSEBlock1D(out_channels)
        elif conv_att == "cbam":
            self.conv_att = MaskedCBAM1d(out_channels)
        else:
            self.conv_att = None

    def forward(self, x, mask, inp_channel_last=True):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_length, channels) if inp_channel_last=True, else (batch, channels, seq)
            mask: Mask tensor of shape (batch_size, seq_length), where True is invalid (padding) and False is valid
        """

        att_mask = mask_to_causal_attention_mask(mask)
        mask = mask.unsqueeze(1)

        if inp_channel_last:
            x = x.transpose(1, 2)  # (batch, channels, seq)

        for i, layer in enumerate(self.layers):

            if self.n_heads[i] > 0:
                x = x.transpose(1, 2)  # (batch, seq, channels)
                x_att = self.att_layers[i](x, x, x, att_mask)
                x_att = self.attn_dropout(x_att)
                x += x_att
                x = x.transpose(1, 2)  # (batch, channels, seq)

            x = layer(x, mask)
        
        if self.conv_att is not None:
            x = self.conv_att(x, mask)

        if inp_channel_last:
            x = x.transpose(1, 2)  # (batch, seq, channels)

        return x


