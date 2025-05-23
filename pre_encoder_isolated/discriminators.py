import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

# Assuming sequence_mask is in pre_encoder_isolated/preencoder.py
# Adjust the relative import path if the structure is different
# For a package pre_encoder_isolated containing preencoder.py and discriminators.py,
# the import from discriminators.py to preencoder.py would be:
from .preencoder import sequence_mask
# If preencoder.py is one level up, it would be from ..preencoder import sequence_mask

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
    hidden_channels: list[int] – output channels per conv block
    kernel_sizes   : list[int] – square kernels, len = len(hidden_channels)+1
    stride         : int       – stride for down-sampling conv blocks
    norm_bin_size  : int       – freq-group size for GroupActNorm2d
    """

    def __init__(
        self,
        mel_channels: int,
        hidden_channels: list = (64, 128, 256, 512),
        kernel_sizes: list = (7, 5, 5, 3, 3),
        stride: int = 2,
        # norm_bin_size: int = 8, # This parameter is not used in the provided code
    ):
        super().__init__()
        assert len(kernel_sizes) == len(hidden_channels) + 1, (
            "kernel_sizes must be hidden_channels len + 1"
        )

        # --- Convolutional backbone ------------------------------------
        self.convs = nn.ModuleList()
        self.mel_channels = mel_channels
        ret_features_map = [True] * len(hidden_channels)
        if len(ret_features_map) > 0: # Make sure list is not empty
            ret_features_map[0] = False
        if len(ret_features_map) > 1: # Make sure list has at least 2 elements
            ret_features_map[1] = False
        if len(ret_features_map) > 0: # Make sure list is not empty
            ret_features_map[-1] = False
        self.ret_features_map = ret_features_map


        in_ch = 1  # we keep a single input channel and treat (F,T) as H×W

        for out_ch, k in zip(hidden_channels, kernel_sizes[:-1]):
            self.convs.append(
                spectral_norm(
                    nn.Conv2d(
                        in_ch,
                        out_ch,
                        kernel_size=(k, k),
                        stride=(stride, stride),
                        padding=(k - 1) // 2,
                    )
                )
            )
            in_ch = out_ch

        # final 1-channel logits layer
        self.convs.append(
            spectral_norm(
                nn.Conv2d(
                    in_ch,
                    1,
                    kernel_size=(kernel_sizes[-1], kernel_sizes[-1]),
                    stride=1,
                    padding=(kernel_sizes[-1] - 1) // 2,
                )
            )
        )

        self.activation = nn.LeakyReLU(0.2, inplace=True)
        # Ensure ChannelSELayerMasked is defined or imported correctly
        self.se_block = ChannelSELayerMasked(in_ch, 8)


        self._initialize_weights()

    # ------------------------------------------------------------------
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _build_mask(self, T: int, Freq: int, lengths: torch.Tensor) -> torch.Tensor:
        """
        lengths : (B,) – number of valid time frames
        returns : (B, 1, F, T) – True = padded
        """
        # time mask (B, T)
        tmask = sequence_mask(T, lengths)           # True = padded
        # broadcast across frequency bins and channel dim
        return tmask.unsqueeze(1).unsqueeze(2).expand(-1, 1, Freq, -1)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,           # (B, T, F)
        x_lengths: torch.Tensor,   # (B,)
        return_features: bool = False,
    ):
        B, T, _ = x.shape

        # build padded mask BEFORE any conv
        padded_mask = self._build_mask(T, self.mel_channels, x_lengths)  # (B,1,F,T)

        # bring to 2-D conv layout
        out = x.transpose(1, 2).unsqueeze(1)  # (B,1,F,T)

        features = []

        for i, conv in enumerate(self.convs):
            if i == len(self.convs) - 1: # Apply SE block before the last conv layer
                out = self.se_block(out, padded_mask)

            out = conv(out) # Apply convolution
            # Apply activation only if it's not the final layer outputting logits
            if i < len(self.convs) -1:
                 out = self.activation(out)


            # ----- down-sample mask to match feature map --------------
            stride_h, stride_w = conv.stride
            if stride_h > 1 or stride_w > 1:
                padded_mask = F.max_pool2d(
                    padded_mask.float(),
                    kernel_size=(stride_h, stride_w),
                    stride=(stride_h, stride_w),
                    ceil_mode=True,
                ).bool()

            # zero-out fully padded patches
            out = out.masked_fill(padded_mask, 0.0)

            if return_features and i < len(self.ret_features_map) and self.ret_features_map[i]:
                features.append((out, padded_mask))

        # -------- flip mask so True = valid for loss -------------------
        patch_mask = ~padded_mask  # (B,1,H,W)

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
        hidden_channels: list = (64, 128, 256, 512),
        kernel_sizes:   list = (7, 5, 5, 3, 3),
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
                    stride=2, #  stride
                    # norm_bin_size= bin_size // 2, # This parameter is not used in MelSpectrogramPatchDiscriminator2D
                )
                for _ in range(n_bins)
            ]
        )

    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor,
                return_features: bool = False):
        """
        x: (B, T, C_mel) – full spectrogram
        returns lists of per-bin outputs.
        """
        # The original code had x.transpose(1,2) in the loop for each discriminator.
        # However, MelSpectrogramPatchDiscriminator2D expects (B, T, F)
        # and transposes internally. So we should feed (B, T, bin_size) to each.
        splits = torch.split(x, x.size(-1) // self.n_bins, dim=-1)


        outs, masks, feats = [], [], []
        for disc, sub_x in zip(self.discriminators, splits):
            # sub_x is already (B, T, bin_size)
            if return_features:
                o, m, f = disc(sub_x, x_lengths, True)
                outs.append(o); masks.append(m); feats.append(f)
            else:
                o, m = disc(sub_x, x_lengths, False)
                outs.append(o); masks.append(m)

        if return_features:
            return outs, masks, feats
        return outs, masks
