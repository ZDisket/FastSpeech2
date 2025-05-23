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

    Args:
        x (torch.Tensor): Tensor to fill. Shape can be (B, C, L) or other shapes
                          broadcastable with the mask.
        mask (torch.Tensor): Boolean mask. Common shapes are (B, 1, L) or (B, L).
                             (True means invalid/padded position).
        fill_value (float): Value to fill with.
    """
    # Determine the target mask shape for broadcasting based on x's dimensions
    if x.ndim == 3 and mask.ndim == 3:  # e.g., x is (B, C, L), mask is (B, 1, L)
        if mask.shape[1] == 1 and x.shape[1] != 1:
            mask_expanded = mask.expand(-1, x.shape[1], -1)
        else:
            mask_expanded = mask  # Assume mask is already (B,C,L) or correctly broadcastable
    elif x.ndim == 3 and mask.ndim == 2:  # e.g., x is (B, C, L), mask is (B, L)
        mask_expanded = mask.unsqueeze(1).expand(-1, x.shape[1], -1)
    elif x.ndim == mask.ndim:  # e.g. x is (B,L) and mask is (B,L), or x is (B,C,1) and mask is (B,1,1)
        mask_expanded = mask  # No expansion needed if dimensions match or broadcasting handles it
    else:
        # Fallback or error for unhandled mask/data shape combinations if necessary
        # For now, assume broadcasting will work or mask is already correct
        mask_expanded = mask

    x = x.masked_fill(mask_expanded.bool(), fill_value)
    return x


def masked_max_pool1d(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Computes max over the last dimension (seq_len) while ignoring masked (padded) positions.
    Returns a tensor of shape (B, C, 1).

    Args:
        x (torch.Tensor): Input tensor. Shape: (B, C, L).
        mask (torch.Tensor): Boolean mask. Shape: (B, 1, L) (True means invalid/padded).
    Returns:
        torch.Tensor: Max pooled tensor. Shape: (B, C, 1).
    """
    x_clone = x.clone()
    # Fill invalid positions with a very small number so they won't dominate max
    masked_fill_(x_clone, mask, float('-inf'))
    max_vals, _ = x_clone.max(dim=-1, keepdim=True)  # shape: (B, C, 1)
    return max_vals


def masked_avg_pool1d(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Computes average over the last dimension (seq_len) while ignoring masked (padded) positions.
    Returns a tensor of shape (B, C, 1).

    Args:
        x (torch.Tensor): Input tensor. Shape: (B, C, L).
        mask (torch.Tensor): Boolean mask. Shape: (B, 1, L) (True means invalid/padded).
    Returns:
        torch.Tensor: Average pooled tensor. Shape: (B, C, 1).
    """
    x_clone = x.clone()

    # Invert the mask to get a "valid" mask where True = valid
    valid_mask = ~mask  # shape: (B, 1, L)

    # Expand valid_mask to (B, C, L) if necessary, for element-wise operations
    if valid_mask.shape[1] == 1 and x_clone.shape[1] != 1:
        expanded_valid_mask = valid_mask.expand(-1, x_clone.shape[1], -1)  # (B, C, L)
    else:
        expanded_valid_mask = valid_mask  # Handles cases where C=1 or mask is already expanded

    # Fill invalid positions with zero, so they don't contribute to sum
    # Use the inverse of expanded_valid_mask (i.e., the original mask concept for invalid positions)
    x_clone.masked_fill_(~expanded_valid_mask, 0.0)

    # Sum across seq_len
    sum_vals = x_clone.sum(dim=-1, keepdim=True)  # shape: (B, C, 1)

    # Count of valid positions per (B, C)
    # Sum the expanded_valid_mask along L to get (B,C,1) counts directly.
    counts = expanded_valid_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)  # shape: (B, C, 1)

    avg_vals = sum_vals / counts
    return avg_vals


# New Causal Helper Functions

def causal_masked_max_pool1d(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Computes causal max over the last dimension (seq_len) while ignoring masked (padded) positions.
    For each position i, the max is taken over x[:, :, 0:i+1].
    Returns a tensor of shape (B, C, L).

    Args:
        x (torch.Tensor): Input tensor. Shape: (B, C, L).
        mask (torch.Tensor): Boolean mask. Shape: (B, 1, L) (True means invalid/padded).
    Returns:
        torch.Tensor: Causal max pooled tensor. Shape: (B, C, L).
    """
    x_clone = x.clone()
    # Fill padded positions with -inf so they are ignored by cummax unless all previous are -inf.
    masked_fill_(x_clone, mask, float('-inf'))
    # cummax computes cumulative max along the specified dimension
    causal_max_vals, _ = torch.cummax(x_clone, dim=-1)  # shape: (B, C, L)
    # If a position was originally masked (and thus -inf), and all preceding elements were also masked (or smaller),
    # it will remain -inf. This is generally desired for attention mechanisms.
    return causal_max_vals


def causal_masked_avg_pool1d(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Computes causal average over the last dimension (seq_len) while ignoring masked (padded) positions.
    For each position i, the average is taken over x[:, :, 0:i+1].
    Returns a tensor of shape (B, C, L).

    Args:
        x (torch.Tensor): Input tensor. Shape: (B, C, L).
        mask (torch.Tensor): Boolean mask. Shape: (B, 1, L) (True means invalid/padded).
    Returns:
        torch.Tensor: Causal average pooled tensor. Shape: (B, C, L).
    """
    x_clone = x.clone()

    # Invert mask: True means valid
    valid_mask = ~mask  # Shape: (B, 1, L)

    # Fill invalid positions with zero for summation
    masked_fill_(x_clone, mask, 0.0)

    # Cumulative sum of values
    sum_causal = torch.cumsum(x_clone, dim=-1)  # Shape: (B, C, L)

    # Cumulative sum of valid counts
    # valid_mask (B, 1, L) is broadcasted during division.
    counts_causal = torch.cumsum(valid_mask.float(), dim=-1)  # Shape: (B, 1, L)

    # Clamp counts to avoid division by zero. If counts_causal is 0, avg is 0.
    counts_causal_clamped = counts_causal.clamp(min=1.0)

    avg_causal = sum_causal / counts_causal_clamped  # Broadcasts counts_causal over C dim

    # Where counts_causal was originally 0 (all preceding steps masked), avg_causal will be 0.
    # We need to ensure that these positions are correctly handled, e.g. if they should be 0.
    # If all elements up to 't' are masked, sum_causal is 0, counts_causal is 0 (clamped to 1). Result is 0.
    # This is generally a safe default.
    avg_causal.masked_fill_(counts_causal == 0, 0.0)  # Explicitly set to 0 if no valid elements

    return avg_causal


class CAM1D(nn.Module):
    """
    Channel Attention Module for 1D sequences.
    - Takes (B, C, L) as input.
    - If not causal: Pools across the L dimension (with masked max & avg) to get (B, C, 1).
    - If causal: Pools causally across L to get (B, C, L).
    - Then uses an MLP (two linear layers) to compute channel attention.
    - Finally multiplies it with the input (while respecting the mask).
    """

    def __init__(self, channels: int, reduction_ratio: int, causal: bool = False):
        super(CAM1D, self).__init__()
        self.channels = channels
        self.r = reduction_ratio
        self.causal = causal

        self.mlp = nn.Sequential(  # Renamed from 'linear' to 'mlp' for clarity
            nn.Linear(self.channels, self.channels // self.r, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(self.channels // self.r, self.channels, bias=True)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor. Shape: (B, C, L).
            mask (torch.Tensor): Boolean mask. Shape: (B, 1, L) (True means invalid/padded).
        Returns:
            torch.Tensor: Output tensor after channel attention. Shape: (B, C, L).
        """
        B, C, L_dim = x.shape  # Use L_dim to avoid conflict with L in permute

        if self.causal:
            # Causal masked pooling: Output shape (B, C, L_dim)
            max_pool_out = causal_masked_max_pool1d(x, mask)
            avg_pool_out = causal_masked_avg_pool1d(x, mask)

            # Permute for MLP: (B, C, L_dim) -> (B, L_dim, C) to apply MLP on C features per time step
            max_pool_perm = max_pool_out.permute(0, 2, 1)  # (B, L_dim, C)
            avg_pool_perm = avg_pool_out.permute(0, 2, 1)  # (B, L_dim, C)

            # Feed through MLP
            mlp_max = self.mlp(max_pool_perm)  # (B, L_dim, C)
            mlp_avg = self.mlp(avg_pool_perm)  # (B, L_dim, C)

            # Sum
            attn_logits = mlp_max + mlp_avg  # (B, L_dim, C)

            # Channel attention map
            attn_map = torch.sigmoid(attn_logits)  # (B, L_dim, C)

            # Permute back: (B, L_dim, C) -> (B, C, L_dim)
            attn_map = attn_map.permute(0, 2, 1)  # (B, C, L_dim)
        else:
            # Global masked pooling: Output shape (B, C, 1)
            max_pool_out = masked_max_pool1d(x, mask)
            avg_pool_out = masked_avg_pool1d(x, mask)

            # Flatten for MLP: (B, C, 1) -> (B, C)
            max_pool_flat = max_pool_out.squeeze(-1)
            avg_pool_flat = avg_pool_out.squeeze(-1)

            # Feed through MLP
            mlp_max = self.mlp(max_pool_flat)  # (B, C)
            mlp_avg = self.mlp(avg_pool_flat)  # (B, C)

            # Sum
            attn_logits = mlp_max + mlp_avg  # (B, C)

            # Channel attention map, unsqueeze to (B, C, 1) for broadcasting
            attn_map = torch.sigmoid(attn_logits).unsqueeze(-1)

        # Multiply by the original input (broadcasts attn_map if non-causal)
        output = attn_map * x

        # Mask out padded positions in the final output
        masked_fill_(output, mask, 0.0)

        return output


class SAM1D(nn.Module):
    """
    Spatial Attention Module for 1D sequences.
    - Takes (B, C, L) as input.
    - Produces a spatial attention map of shape (B, 1, L).
    - If causal, uses causal convolution.
    - Then multiplies it (elementwise) by the original input (while respecting the mask).
    """

    def __init__(self, kernel_size: int = 7, bias: bool = False, causal: bool = False):
        super(SAM1D, self).__init__()
        self.kernel_size_val = kernel_size  # Renamed to avoid conflict
        self.bias = bias
        self.causal = causal

        if self.causal:
            # For causal convolution, we pad (kernel_size - 1) on the left.
            # The Conv1d layer itself will have padding=0.
            self.causal_padding_amount = self.kernel_size_val - 1
            conv_padding = 0
        else:
            # Standard symmetric padding
            assert self.kernel_size_val % 2 == 1, "Kernel size must be odd for symmetric padding for SAM1D"
            conv_padding = self.kernel_size_val // 2

        self.conv = nn.Conv1d(
            in_channels=2,  # Max and Avg pooled features along channel dim
            out_channels=1,  # Output is a single attention map
            kernel_size=self.kernel_size_val,
            stride=1,
            padding=conv_padding,
            dilation=1,
            bias=self.bias
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor. Shape: (B, C, L).
            mask (torch.Tensor): Boolean mask. Shape: (B, 1, L) (True means invalid/padded).
                                 This mask refers to the original sequence positions.
        Returns:
            torch.Tensor: Output tensor after spatial attention. Shape: (B, C, L).
        """
        # Max & Avg pooling across the channel dimension. Output shape: (B, 1, L)
        # These operations are inherently "spatially local" or "causal" in the sense that
        # pooling at time 't' only uses data from x[:,:,t].
        max_out_ch = x.max(dim=1, keepdim=True)[0]
        avg_out_ch = x.mean(dim=1, keepdim=True)

        # Apply mask to pooled features before concatenation and convolution.
        # This ensures that convolution doesn't operate on values from padded regions
        # if those regions influenced the channel pooling (e.g. mean).
        # Filling with 0.0 is a common choice.
        masked_fill_(max_out_ch, mask, 0.0)
        masked_fill_(avg_out_ch, mask, 0.0)

        # Concatenate along the channel dimension => shape (B, 2, L)
        concat_features = torch.cat((max_out_ch, avg_out_ch), dim=1)

        # Apply causal padding if needed before convolution
        if self.causal:
            # F.pad format for 1D (last dim): (pad_left, pad_right)
            concat_features = F.pad(concat_features, (self.causal_padding_amount, 0))
            # After padding, shape is (B, 2, L + causal_padding_amount)
            # Conv1d with padding=0 will then produce output of length L.

        # Convolution over the sequence dimension
        attn_logits = self.conv(concat_features)  # Expected shape: (B, 1, L)

        # Fill attention logits at masked positions with a large negative number.
        # This ensures that the sigmoid output for these positions is close to 0.
        # The 'mask' is (B, 1, L) and corresponds to original sequence length.
        masked_fill_(attn_logits, mask, -1e+4)

        # Apply sigmoid to get attention scores
        spatial_attn_map = torch.sigmoid(attn_logits)  # shape: (B, 1, L)

        # Ensure attention at padded positions is strictly zero after sigmoid.
        # This is somewhat redundant if logits were set to -1e9, but ensures exact zeros.
        masked_fill_(spatial_attn_map, mask, 0.0)

        # Multiply the spatial attention map with the original input tensor x.
        # The map (B, 1, L) broadcasts across the channel dimension of x (B, C, L).
        output = spatial_attn_map * x

        # Final explicit masking of the output tensor.
        # This ensures that any padded positions in the original x remain zeroed out.
        masked_fill_(output, mask, 0.0)

        return output


class CBAM1D(nn.Module):
    """
    Convolutional Block Attention Module for 1D sequences.
    - Applies Channel Attention (CAM1D).
    - Then applies Spatial Attention (SAM1D).
    - Adds the result to the original input as a residual connection.
    - Includes a `causal` option for both attention mechanisms.
    """

    def __init__(self, channels: int, reduction_ratio: int = 8, causal: bool = False, sam_kernel_size: int = 7):
        super(CBAM1D, self).__init__()
        self.causal = causal

        self.channel_attention = CAM1D(
            channels=channels,
            reduction_ratio=reduction_ratio,
            causal=self.causal
        )

        # Bias is typically False for SAM's convolution layer in many CBAM implementations.
        self.spatial_attention = SAM1D(
            kernel_size=sam_kernel_size,
            bias=False,  # Consistent with common practice
            causal=self.causal
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor. Shape: (B, C, L).
            mask (torch.Tensor): Boolean mask. Shape: (B, 1, L) (True means invalid/padded).
                                 This mask indicates padded regions in the input 'x'.
        Returns:
            torch.Tensor: Output tensor after CBAM. Shape: (B, C, L).
        """
        # Apply Channel Attention
        # The mask is passed to ensure CAM handles padded sequences correctly.
        x_after_cam = self.channel_attention(x, mask)

        # Apply Spatial Attention
        # The mask is passed to ensure SAM handles padded sequences correctly.
        x_after_sam = self.spatial_attention(x_after_cam, mask)

        # Residual connection: Add the attention-modified features to the original input
        output = x_after_sam + x

        # Ensure that padded positions in the final output are zeroed out.
        # Although sub-modules also apply masking, this acts as a final safeguard,
        # especially important for the residual connection if 'x' had non-zero values
        # in its padded regions.
        masked_fill_(output, mask, 0.0)

        return output


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