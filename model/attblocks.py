from math import sqrt
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

# attblocks.py: blocks for attns


class ReduceSequenceLength(nn.Module):
    def __init__(self):
        super(ReduceSequenceLength, self).__init__()

    def forward(self, input_tensor):
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

class GatedRetention(nn.Module):
    """
    Allows the model to selectively retain or discard information based on the learned gate values.

    https://github.com/Mr-Twave/YOCO-Groq-BitNet-KV-cache/tree/main?tab=readme-ov-file#the-math
    """

    def __init__(self, in_channels, hidden_size, drop=0.1):
        super(GatedRetention, self).__init__()
        self.proj = nn.Linear(in_channels, hidden_size) if in_channels != hidden_size else nn.Identity()
        self.gate = nn.Linear(hidden_size, hidden_size)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.proj(x)
        gated_output = torch.sigmoid(self.gate(x)) * x

        return self.drop(gated_output)


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


class MaskedGlobalAvgPool1d(nn.Module):
    """
    Masked Global Average Pooling for 1D inputs.
    """

    def __init__(self):
        super(MaskedGlobalAvgPool1d, self).__init__()

    def forward(self, x, mask):
        # Invert the mask: True -> False (padded), False -> True (valid)
        inverted_mask = ~mask
        inverted_mask = inverted_mask.float()

        # Expand mask to match the shape of x
        inverted_mask = inverted_mask.expand_as(x)

        # Apply inverted mask to input
        masked_x = x * inverted_mask

        # Calculate masked global average pooling
        sum_masked_x = masked_x.sum(dim=-1)
        sum_inverted_mask = inverted_mask.sum(dim=-1).clamp(min=1)  # Avoid division by zero
        y = sum_masked_x / sum_inverted_mask

        return y


class MaskedSEBlock1D(nn.Module):
    """
    Lightweight Squeeze-Excite attention with masked global average pooling, or attention pooling
    """

    def __init__(self, in_channels, reduction=16, pooling="avg"):
        super(MaskedSEBlock1D, self).__init__()

        if pooling not in ["avg", "att"]:
            raise RuntimeError(f"Unknown pooling type {pooling}, must be either avg or att")

        self.pooling = pooling
        self.avg_pool = MaskedGlobalAvgPool1d() if pooling == "avg" else AttentionPooling(in_channels)

        # attn
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)

    def _attn_forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

    def forward(self, x, mask):
        """
        Forward pass through the Masked SE Block
        :param x: Hidden tensor shape (batch, channels, seq)
        :param mask: Bool mask where True=padded, size (batch, 1, seq)
        :return: Same shape as x, SE-attentioned tensor.
        """
        # Compute masked global average pooling
        if self.pooling == "att":
            # att pooling returns attention weights
            y, _ = self.avg_pool(x.transpose(1,2), # AttentionPooling takes (batch, seq), maskedavg takes (batch, 1, seq)
                          mask.squeeze(1))
        else:
            y = self.avg_pool(x, mask)

        y = self._attn_forward(y)

        y = y.view(x.size(0), x.size(1), 1).masked_fill(mask, 0)

        # Apply the excitation
        return x * y.expand_as(x)


def masked_fill_(x: torch.Tensor, mask: torch.Tensor, fill_value: float):
    """
    In-place masked fill. Where mask == True, fill `x` with `fill_value`.

    x:    (B, C, L)
    mask: (B, 1, L)  (True means invalid/padded)
    """
    # We broadcast the mask across the channel dimension:
    if mask.shape[1] == 1 and x.shape[1] != 1:
        mask = mask.expand(-1, x.shape[1], -1)  # shape: (B, C, L)
    x.masked_fill_(mask, fill_value)


def masked_max_pool1d(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Computes max over the last dimension (seq_len) while ignoring masked (padded) positions.
    Returns a tensor of shape (B, C, 1).

    x:    (B, C, L)
    mask: (B, 1, L) (True means invalid/padded)
    """
    # Temporarily fill invalid positions with a very small number so they won't dominate max
    x_clone = x.clone()
    masked_fill_(x_clone, mask, float('-inf'))  # fill padded positions with -inf
    # Max over seq_len dimension
    max_vals, _ = x_clone.max(dim=-1, keepdim=True)  # shape: (B, C, 1)
    return max_vals


def masked_avg_pool1d(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Computes average over the last dimension (seq_len) while ignoring masked (padded) positions.
    Returns a tensor of shape (B, C, 1).

    x:    (B, C, L)
    mask: (B, 1, L) (True means invalid/padded)
    """
    # We will sum only valid positions and then divide by count of valid positions
    x_clone = x.clone()

    # Invert the mask to get a "valid" mask where True = valid
    valid_mask = ~mask  # shape: (B, 1, L)
    if valid_mask.shape[1] == 1 and x_clone.shape[1] != 1:
        valid_mask = valid_mask.expand(-1, x_clone.shape[1], -1)  # (B, C, L)

    # Fill invalid positions with zero, so they don't contribute to sum
    x_clone.masked_fill_(~valid_mask, 0.0)

    # Sum across seq_len
    sum_vals = x_clone.sum(dim=-1, keepdim=True)  # shape: (B, C, 1)

    # Count of valid positions per (B, C)
    counts = valid_mask.sum(dim=-1, keepdim=True).clamp(min=1)  # shape: (B, C, 1)

    # Weighted average
    avg_vals = sum_vals / counts
    return avg_vals


class SAM1D(nn.Module):
    """
    Spatial Attention Module for 1D sequences.
    - Takes (B, C, L) as input
    - Produces a spatial attention map of shape (B, 1, L)
    - Then multiplies it (elementwise) by the original input (while respecting the mask).
    """

    def __init__(self, bias=False):
        super(SAM1D, self).__init__()
        self.bias = bias
        # We take 2 'channels' as input (max, avg along channel dim),
        # and produce a 1-channel output (spatial attention).
        self.conv = nn.Conv1d(
            in_channels=2,
            out_channels=1,
            kernel_size=7,
            stride=1,
            padding=3,
            dilation=1,
            bias=self.bias
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x:    (B, C, L)
        mask: (B, 1, L) (True means invalid/padded)
        """
        # Masked max & avg across channel dimension
        max_out = x.max(dim=1, keepdim=True)[0]  # shape: (B, 1, L)
        avg_out = x.mean(dim=1, keepdim=True)  # shape: (B, 1, L)

        # Expand the mask to match shape (B, 1, L):
        # We skip that because max_out and avg_out already have shape (B, 1, L).
        # If needed, we can do:
        masked_fill_(max_out, mask, 0.0)  # or a safe fill for the 'max' map
        masked_fill_(avg_out, mask, 0.0)  # fill with 0.0 for 'avg'

        # Concatenate along the channel dimension => shape (B, 2, L)
        concat = torch.cat((max_out, avg_out), dim=1)

        # Convolution over the sequence dimension
        output = self.conv(concat)  # shape: (B, 1, L)

        output = output.masked_fill(mask, -10.0) # prevent masked positions from activating the sigmoid.

        # Sigmoid
        output = torch.sigmoid(output)

        # Multiply elementwise with the original x (but also mask out padded positions)
        # We'll broadcast output from (B, 1, L) to (B, C, L)
        output = output * x

        # Finally, we can mask out padded positions so that they remain zero if desired
        masked_fill_(output, mask, 0.0)

        return output


class CAM1D(nn.Module):
    """
    Channel Attention Module for 1D sequences.
    - Takes (B, C, L) as input
    - Pools across the L dimension (with masked max & avg) to get (B, C, 1)
    - Then uses an MLP (two linear layers) to compute channel attention
    - Finally multiplies it with the input (while respecting the mask).
    """

    def __init__(self, channels, reduction_ratio):
        super(CAM1D, self).__init__()
        self.channels = channels
        self.r = reduction_ratio
        self.linear = nn.Sequential(
            nn.Linear(self.channels, self.channels // self.r, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(self.channels // self.r, self.channels, bias=True)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x:    (B, C, L)
        mask: (B, 1, L) (True means invalid/padded)
        """
        # We do masked max pooling & masked avg pooling across the length dimension
        max_pool = masked_max_pool1d(x, mask)  # (B, C, 1)
        avg_pool = masked_avg_pool1d(x, mask)  # (B, C, 1)

        # Flatten for linear => shape (B, C)
        # (We remove the last dimension which is 1)
        max_pool_flat = max_pool.squeeze(-1)  # (B, C)
        avg_pool_flat = avg_pool.squeeze(-1)  # (B, C)

        # Feed through MLP
        mlp_max = self.linear(max_pool_flat)  # (B, C)
        mlp_avg = self.linear(avg_pool_flat)  # (B, C)

        # Sum
        attn = mlp_max + mlp_avg  # (B, C)

        # Channel attention map
        attn = torch.sigmoid(attn).unsqueeze(-1)  # shape (B, C, 1)

        # Multiply by the original input (broadcast across L dimension)
        output = attn * x

        # We can optionally mask out the padded positions
        masked_fill_(output, mask, 0.0)

        return output


class CBAM1D(nn.Module):
    """
    Convolutional Block Attention Module for 1D sequences.
    - Applies Channel Attention (CAM1D)
    - Then applies Spatial Attention (SAM1D)
    - Adds the result to the original input as a residual connection.
    """

    def __init__(self, channels, reduction_ratio=8):
        super(CBAM1D, self).__init__()
        self.cam = CAM1D(channels, reduction_ratio)
        self.sam = SAM1D(bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x:    (B, C, L)
        mask: (B, 1, L) (True means invalid/padded)
        """
        # Channel Attention
        out = self.cam(x, mask)
        # Spatial Attention
        out = self.sam(out, mask)
        # Residual connection
        return out + x


class MaskedCBAM1d(nn.Module):
    def __init__(self, in_channels, reduction=16, pooling="avg"):
        super(MaskedCBAM1d, self).__init__()
        self.channel_attention = MaskedSEBlock1D(in_channels, reduction, pooling)

        # spatial attn
        self.conv1 = nn.Conv1d(in_channels, in_channels // reduction, kernel_size=7, padding=3)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels // reduction, 1, kernel_size=7, padding=3)

    def _spatial_forward(self, x, mask):
        x = self.conv1(x).masked_fill(mask, 0)
        x = self.act(x)
        x = self.conv2(x).masked_fill(mask, -10) # for sigmoid
        x = torch.sigmoid(x)
        return x

    def forward(self, x, mask):
        x_out = self.channel_attention(x, mask)
        y = self._spatial_forward(x_out, mask)
        return x_out * y.expand_as(x_out)



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
        self.x_mask = torch.zeros((1, 1, 1))
        self.mask_value = 0

    def set_mask(self, in_mask):
        self.x_mask = in_mask

    def forward(self, x):
        x = x[:, :, :-self.chomp_size].contiguous()
        x = x.masked_fill(self.x_mask, 0)
        return x


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionPooling, self).__init__()
        self.attention_weights = nn.Parameter(torch.Tensor(hidden_dim, 1))
        nn.init.xavier_uniform_(self.attention_weights)

    def forward(self, x, mask):
        # x: (batch, seq_len, hidden_dim)
        # mask: (batch, seq_len)
        attn_scores = torch.matmul(x, self.attention_weights).squeeze(-1)  # (batch, seq_len)
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=-1).unsqueeze(-1)  # (batch, seq_len, 1)
        context = torch.sum(attn_weights * x, dim=1)  # (batch, hidden_dim)
        return context, attn_weights


class ChannelAttention2d(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention2d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention2d(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention2d, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM2d(nn.Module):
    def __init__(self, planes, reduction_ratio=16, no_spatial=False):
        super(CBAM2d, self).__init__()
        self.ChannelAttention = ChannelAttention2d(planes, reduction_ratio)
        self.SpatialAttention = SpatialAttention2d() if not no_spatial else None

    def forward(self, x):
        x = x * self.ChannelAttention(x)
        if self.SpatialAttention:
            x = x * self.SpatialAttention(x)
        return x



# Note: Not actually lightweight
class LightweightConvAttention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super(LightweightConvAttention, self).__init__()
        self.depthwise_conv = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size,
                                        groups=in_channels, padding=kernel_size // 2)
        self.pointwise_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(out_channels, out_channels // 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(out_channels // 16, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.depthwise_conv(x)
        out = self.pointwise_conv(out)

        # Channel attention
        ca = self.channel_attention(out)
        out = out * ca

        # Spatial attention
        max_pool = torch.max(out, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(out, dim=1, keepdim=True)
        sa = self.spatial_attention(torch.cat([max_pool, avg_pool], dim=1))
        out = out * sa

        return out