from math import sqrt
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

# attblocks.py: blocks for attns

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