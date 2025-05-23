import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from typing import Union, Tuple, List # Added typing imports

# Import sequence_mask from the preencoder module within the same package
from .preencoder import sequence_mask

class ChannelSELayerMasked(nn.Module):
    """
    Squeeze-and-Excitation that supports a padding mask.

    Parameters
    ----------
    num_channels : int
        Number of channels in the input.
    reduction_ratio : int, default=2
        Channel‐reduction factor (same as the classic SE block).

    Notes
    -----
    * `padding_mask` is expected to be a **bool tensor** with shape
      (B, 1, H, W).  `True` = padded, `False` = valid.
    * If `padding_mask` is omitted (or None), the layer behaves exactly
      like a standard SE block.
    """

    def __init__(self, num_channels: int, reduction_ratio: int = 2):
        super().__init__()
        reduced = max(1, num_channels // reduction_ratio)

        self.fc1 = nn.Linear(num_channels, reduced, bias=True)
        self.fc2 = nn.Linear(reduced,   num_channels, bias=True)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,              # (B, C, H, W)
        padding_mask=None  # (B, 1, H, W) -- True = padded
    ):
        B, C, H, W = x.shape

        # ---------- SQUEEZE (masked global average) ------------------
        if padding_mask is None:
            # Vanilla SE: mean over spatial dims
            squeeze = x.view(B, C, -1).mean(dim=2)           # (B, C)
        else:
            # Exclude padded positions
            #   mask_valid : (B, 1, H, W)   True = valid
            mask_valid = ~padding_mask.bool()
            # prevent div-by-zero
            denom = mask_valid.sum(dim=(2, 3), keepdim=False).clamp(min=1)  # (B,1)
            # spatial sum over valid positions
            summed = (x * mask_valid).view(B, C, -1).sum(dim=2)             # (B,C)
            squeeze = summed / denom                                        # (B,C)

        # ---------- EXCITATION ---------------------------------------
        excite = self.sigmoid(self.fc2(self.relu(self.fc1(squeeze))))  # (B,C)

        # ---------- SCALE --------------------------------------------
        y = x * excite.view(B, C, 1, 1)

        return y


class MelSpectrogramPatchDiscriminator2D(nn.Module):
    """
    2-D PatchGAN discriminator for (time × mel) spectrograms.

    Input  : (B, T, F)  – time major
    Output : logits      (B, 1, H, W)
             patch_mask  (B, 1, H, W)  -- True = *valid* patch

    Args
    ----
    mel_channels   : number of mel bins (F)
    hidden_channels: List[int] – output channels per conv block
    kernel_sizes   : List[int] – square kernels, len = len(hidden_channels)+1
    stride         : int       – stride for down-sampling conv blocks
    """

    def __init__(
        self,
        mel_channels: int,
        hidden_channels: List[int] = (64, 128, 256, 512), # Changed to List[int]
        kernel_sizes: List[int] = (7, 5, 5, 3, 3),       # Changed to List[int]
        stride: int = 2,
    ):
        super().__init__()
        assert len(kernel_sizes) == len(hidden_channels) + 1, (
            "kernel_sizes must be hidden_channels len + 1"
        )

        self.convs = nn.ModuleList()
        self.mel_channels = mel_channels

        self.ret_features_map = [True] * len(hidden_channels)
        if len(hidden_channels) > 0:
            self.ret_features_map[0] = False
            if len(hidden_channels) > 1:
                self.ret_features_map[1] = False
            self.ret_features_map[-1] = False

        in_ch = 1
        current_in_channels = in_ch
        for i, (out_ch, k) in enumerate(zip(hidden_channels, kernel_sizes[:-1])):
            self.convs.append(
                spectral_norm(
                    nn.Conv2d(
                        current_in_channels,
                        out_ch,
                        kernel_size=(k, k),
                        stride=(stride, stride),
                        padding=(k - 1) // 2,
                    )
                )
            )
            current_in_channels = out_ch

        self.convs.append(
            spectral_norm(
                nn.Conv2d(
                    current_in_channels,
                    1,
                    kernel_size=(kernel_sizes[-1], kernel_sizes[-1]),
                    stride=1,
                    padding=(kernel_sizes[-1] - 1) // 2,
                )
            )
        )

        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.se_block = ChannelSELayerMasked(current_in_channels, 8)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @torch.no_grad()
    def _build_mask(self, T: int, Freq: int, lengths: torch.Tensor) -> torch.Tensor:
        tmask = sequence_mask(T, lengths)
        return tmask.unsqueeze(1).unsqueeze(2).expand(-1, 1, Freq, -1)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        return_features: bool = False,
    ):
        B, T, _ = x.shape
        padded_mask = self._build_mask(T, self.mel_channels, x_lengths)
        out = x.transpose(1, 2).unsqueeze(1)

        features = []
        for i, conv_layer in enumerate(self.convs):
            is_last_conv = (i == len(self.convs) - 1)

            if is_last_conv:
                out = self.se_block(out, padded_mask)

            out = conv_layer(out)

            if not is_last_conv:
                out = self.activation(out)

            stride_h, stride_w = conv_layer.stride
            if stride_h > 1 or stride_w > 1:
                padded_mask = F.max_pool2d(
                    padded_mask.float(),
                    kernel_size=(stride_h, stride_w),
                    stride=(stride_h, stride_w),
                    ceil_mode=True,
                ).bool()

            out = out.masked_fill(padded_mask, 0.0)

            if return_features and i < len(self.ret_features_map) and self.ret_features_map[i]:
                features.append((out, padded_mask.clone()))

        patch_mask = ~padded_mask

        if return_features:
            return out, patch_mask, features
        return out, patch_mask


class MultiBinDiscriminator(nn.Module):
    """
    Splits the mel axis into `n_bins` equal bands and runs an independent
    MelSpectrogramPatchDiscriminator2D on each band.
    """
    def __init__(
        self,
        mel_channels: int,
        n_bins: int = 4,
        hidden_channels: List[int] = (64, 128, 256, 512), # Changed to List[int]
        kernel_sizes:   List[int] = (7, 5, 5, 3, 3),       # Changed to List[int]
    ):
        super().__init__()
        assert mel_channels % n_bins == 0, "mel_channels must divide n_bins"

        sub_hidden_channels = []
        for h in hidden_channels:
            assert h % n_bins == 0, f"hidden size {h} must divide n_bins"
            sub_hidden_channels.append(h // n_bins)

        self.n_bins = n_bins
        bin_size   = mel_channels // n_bins

        self.discriminators = nn.ModuleList(
            [
                MelSpectrogramPatchDiscriminator2D(
                    mel_channels=bin_size,
                    hidden_channels=list(sub_hidden_channels),
                    kernel_sizes=list(kernel_sizes),
                    stride=2,
                )
                for _ in range(n_bins)
            ]
        )

    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor,
                return_features: bool = False):
        splits = torch.split(x, x.size(-1) // self.n_bins, dim=-1)

        outs, masks, feats = [], [], []
        for disc, sub_x_bin in zip(self.discriminators, splits):
            if return_features:
                o, m, f_list = disc(sub_x_bin, x_lengths, True)
                outs.append(o); masks.append(m); feats.extend(f_list)
            else:
                o, m = disc(sub_x_bin, x_lengths, False)
                outs.append(o); masks.append(m)

        if return_features:
            return outs, masks, feats
        return outs, masks
