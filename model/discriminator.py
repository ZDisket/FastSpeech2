from IPython.utils import text
import torch
import torch.nn as nn
import torch.optim as optim
from .attentions import SEBlock1D, TransposeRMSNorm, AttentionPooling, TransposeLayerNorm, MultiHeadAttention, APTx, \
    ResidualBlock1D, CBAM1D, CAM1D
from .submodels import sequence_mask, mask_to_attention_mask, Prenet, GroupActNorm1d
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .s4 import S4Block as S4 
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class SequenceNormalization(nn.Module):
    """
    Masked sequence normalization with learned alpha and beta values
    """

    def __init__(self, num_features):
        super(SequenceNormalization, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, num_features, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1))

    def forward(self, x, seq_lens):
        """
        Forward pass through the learnable normalization layer
        :param x: Tensor size (batch, seq_len, num_features)
        :param seq_lens: Int sequence lengths tensor size (batch,)
        :return: Normalized x, same shape
        """
        x = x.transpose(1, 2)  # (batch, seq_len, 1) => (batch, 1, seq_len)

        # Create mask based on sequence lengths
        batch_size, max_len = x.size(0), x.size(2)
        mask = torch.arange(max_len, device=x.device).expand(batch_size, max_len) < seq_lens.unsqueeze(1)

        # Masked min and max calculations
        masked_x = x.masked_fill(~mask.unsqueeze(1), float('inf'))
        min_vals = masked_x.min(dim=2, keepdim=True).values
        masked_x = x.masked_fill(~mask.unsqueeze(1), float('-inf'))
        max_vals = masked_x.max(dim=2, keepdim=True).values

        # Normalize to [0, 1]
        normalized_x = (x - min_vals) / (max_vals - min_vals + 1e-8)

        # Apply learnable scaling and shifting
        scaled_x = normalized_x * self.alpha + self.beta
        scaled_x = scaled_x.transpose(1, 2)  # (batch, 1, seq_len) => (batch, seq_len, 1)
        return scaled_x


class ConvBlock1D(nn.Module):
    """
    Simple block for sequence modeling with configurable normalization.

    If norm=="layer", uses TransposeLayerNorm.
    If norm=="spectral", applies spectral normalization to the Conv1d and uses nn.Identity.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.3, act="relu", norm="layer"):
        super(ConvBlock1D, self).__init__()

        if norm == "layer":
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding="same")
            self.norm1 = TransposeLayerNorm(out_channels)
        elif norm == "spectral":
            self.conv1 = spectral_norm(
                nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding="same"))
            self.norm1 = nn.Identity()
        else:
            raise ValueError("norm must be either 'layer' or 'spectral'")

        self.act = APTx() if act == "aptx" else nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mask):
        # x: (B, C, L)
        out = self.conv1(x).masked_fill(x_mask, 0)
        out = self.norm1(out).masked_fill(x_mask, 0)
        out = self.act(out).masked_fill(x_mask, 0)
        out = self.dropout(out)
        return out


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
        norm_bin_size: int = 8,
    ):
        super().__init__()
        assert len(kernel_sizes) == len(hidden_channels) + 1, (
            "kernel_sizes must be hidden_channels len + 1"
        )

        # --- Convolutional backbone ------------------------------------
        self.convs = nn.ModuleList()
        self.mel_channels = mel_channels
        ret_features_map = [True] * len(hidden_channels)
        ret_features_map[0] = False
        ret_features_map[1] = False
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
            if i == len(self.convs) - 1:
                out = self.se_block(out, padded_mask)

            out = self.activation(conv(out))

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

            if return_features and self.ret_features_map[i]:
                features.append((out, padded_mask))

        # -------- flip mask so True = valid for loss -------------------
        patch_mask = ~padded_mask  # (B,1,H,W)

        if return_features:
            return out, patch_mask, features
        return out, patch_mask





class MelSpectrogramPatchDiscriminator(nn.Module):
    """
    PatchGAN-style discriminator for mel-spectrogram inputs with strided convolutions (optional)

    Args:
        mel_channels (int): number of mel frequency bins.
        hidden_channels (List[int]): channels for each conv layer.
        kernel_sizes (List[int]): kernel sizes (len = len(hidden_channels)+1).

    Forward returns a patch mask where True indicates padded patches.
    """
    def __init__(
        self,
        mel_channels: int,
        hidden_channels: list = [64, 128, 256, 512],
        kernel_sizes: list = [7, 5, 5, 3, 3],
        stride = 2,
        norm_bin_size = 8,
    ):
        super().__init__()
        assert len(kernel_sizes) == len(hidden_channels) + 1, \
            "kernel_sizes must be hidden_channels len + 1"

        self.pre_norm = GroupActNorm1d(mel_channels, group_size=norm_bin_size)
        self.convs = nn.ModuleList()
        self.proj = nn.Linear(mel_channels, hidden_channels[0])
        in_ch = hidden_channels[0]
        # feature layers
        for out_ch, k in zip(hidden_channels, kernel_sizes[:-1]):
            conv = spectral_norm(
                nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=stride, padding=(k-1)//2)
            )
            self.convs.append(conv)
            in_ch = out_ch
        # final layer
        final_conv = spectral_norm(
            nn.Conv1d(in_ch, 1, kernel_size=kernel_sizes[-1], stride=1, padding=(kernel_sizes[-1]-1)//2)
        )
        self.convs.append(final_conv)

        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.se_block = CAM1D(in_ch, 8)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor, return_features=False):
        """
        Args:
            x: (batch, time, mel_channels)
            x_lengths: (batch,) lengths of each sequence
        Returns:
            out: (batch, 1, L_out) patch logits
            patch_mask: (batch, 1, L_out) bool mask where True indicates padded patches
        """

        x = self.pre_norm(
            x.transpose(1,2)
        ).transpose(1,2)
        B, T, C = x.size()
        # build padded mask (True indicates padding)
        padded_mask = sequence_mask(T, x_lengths)  # (B, T)
        x = self.proj(x)
        # zero out padded input
        x = x.masked_fill(padded_mask.unsqueeze(-1), 0.0)
        out = x.transpose(1, 2)  # (B, C, T)
        features = []

        # propagate through convs, downsampling padded_mask
        for i, conv in enumerate(self.convs):
            if i == len(self.convs) - 1:
                out = self.se_block(out, padded_mask.unsqueeze(1))

            out = self.activation(conv(out))
            stride = conv.stride[0]
            if stride > 1:
                # downsample padded_mask to align with out
                padded_mask = F.max_pool1d(
                    padded_mask.float().unsqueeze(1),
                    kernel_size=stride,
                    stride=stride,
                    ceil_mode=True
                ).squeeze(1).bool()
            # mask out fully-padded patches
            out = out.masked_fill(padded_mask.unsqueeze(1), 0.0)
            if return_features:
                features.append((out, padded_mask))


        # final patch mask: True=padded
        patch_mask = padded_mask.unsqueeze(1)
        patch_mask = ~patch_mask # True=valid, loss uses that
        if return_features:
            return out, patch_mask, features
        return out, patch_mask


class MultiBinDiscriminator(nn.Module):
    """
    Splits the mel axis into `n_bins` equal bands and runs an independent
    MelSpectrogramPatchDiscriminator on each band.
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
                    norm_bin_size= bin_size // 2, # half norm for each bin
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
        splits = torch.split(x, x.size(-1) // self.n_bins, dim=-1)

        outs, masks, feats = [], [], []
        for disc, sub_x in zip(self.discriminators, splits):
            # (B, Lmel, subCmel) => (B, subCmel, Lmel)
            # We want Cmel to be the 1st dim after batch because our subbin Ds
            # stide along the 2nd dim only -- it must only reduce the seq len and keep the bin dim intact.
            sub_x = sub_x.transpose(1,2)

            if return_features:
                o, m, f = disc(sub_x, x_lengths, True)
                outs.append(o); masks.append(m); feats.append(f)
            else:
                o, m = disc(sub_x, x_lengths, False)
                outs.append(o); masks.append(m)

        if return_features:
            return outs, masks, feats
        return outs, masks




class S4Block1D(nn.Module):
    """
    S4D + ReLU + configurable normalization + Dropout.

    For norm=="layer", uses TransposeLayerNorm.
    For norm=="spectral", applies spectral normalization to the S4 layer and replaces
    the normalization layer with nn.Identity.

    Note: We assume the S4 module is compatible with spectral_norm.
    """

    def __init__(self, in_channels, out_channels, dropout=0.3, act="relu", d_state=64, norm="layer"):
        super(S4Block1D, self).__init__()
        # For simplicity, in_channels should equal out_channels
        assert in_channels == out_channels, "in_channels should match out_channels."

        if norm == "layer":
            self.s4 = S4(d_model=out_channels,
                         d_state=d_state,
                         dropout=dropout,
                         transposed=True)  # Expects input shape (B, C, L)
            self.norm1 = TransposeLayerNorm(out_channels)
        elif norm == "spectral":
            self.s4 = spectral_norm(S4(d_model=out_channels,
                                       d_state=d_state,
                                       dropout=dropout,
                                       transposed=True))
            self.norm1 = nn.Identity()
        else:
            raise ValueError("norm must be either 'layer' or 'spectral'")

        self.act = APTx() if act == "aptx" else nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mask):
        # x: (B, C, L)
        out, _ = self.s4(x.float())  # S4 returns (output, state); state is ignored.
        out = out.masked_fill(x_mask, 0)
        out = self.norm1(out).masked_fill(x_mask, 0)
        out = self.act(out).masked_fill(x_mask, 0)
        out = self.dropout(out)
        return out

class AdvSeqDiscriminatorS4(nn.Module):
    """
    Conv+S4 Discriminator.
    ConvBlocks -> S4Blocks

    I tried residual design but found that it's hard to optimize.
    """
    def __init__(self, hidden_dim=1024, num_ssm_layers=6, conv_kernel_size=[3, 7, 11], conv_dropout=0.5, ssm_dropout=0.3, use_cbam=True, norm="layer"):
        """
        Init S4+Conv D
        :param hidden_dim: Hidden dimension size for all layers
        :param num_ssm_layers: Number of S4 blocks
        :param conv_kernel_size: List detailing conv kernel sizes, which also serves as the amount
        :param conv_dropout: Dropout for conv layers
        :param ssm_dropout: Dropout for S4 layers
        :param use_cbam: Use a CBAM at the end of the conv layers.
        """
        super(AdvSeqDiscriminatorS4, self).__init__()

        self.use_cbam = use_cbam
        self.num_ssm_layers = num_ssm_layers

        self.convs = nn.ModuleList(
            [ConvBlock1D(hidden_dim, hidden_dim, kernel_size=ks, dropout=conv_dropout, norm=norm) for ks in conv_kernel_size]
        )

        self.ssms = nn.ModuleList(
            [S4Block1D(hidden_dim, hidden_dim, dropout=ssm_dropout, norm="layer") for _ in range(num_ssm_layers)]
        )
        
        # Optional CBAM block (unchanged)
        if self.use_cbam:
            self.cbam = CBAM1D(hidden_dim)

        # Attention-based pooling (unchanged)
        self.att_pooling = AttentionPooling(hidden_dim)

        # Final linear (unchanged)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, x_mask, x_mask_conv):
        """
        Forward pass through the discriminator.

        :param x: Input tensor of shape (batch, seq_len, 1): durations
        :param x_mask: Boolean mask (batch, seq_len) for padding.
        :param x_mask_conv: Boolean mask (batch, 1, seq_len) for convolutional operations.
        :return: Tensor of shape (batch, 1) representing logits.
        """
        # (batch, seq_len, hidden_dim) => (batch, hidden_dim, seq_len)
        x = x.transpose(1, 2)

        # Apply convolutional blocks with masking
        for layer in self.convs:
            x = x.masked_fill(x_mask_conv, 0)
            x = layer(x, x_mask_conv)

        # Apply CBAM if used
        if self.use_cbam:
            x = self.cbam(x, x_mask_conv)

        # Apply S4 blocks with masking
        if self.num_ssm_layers > 0:
            for layer in self.ssms:
                x = x.masked_fill(x_mask_conv, 0)
                x = layer(x, x_mask_conv)

        # Attention pooling
        # (batch, hidden_dim, seq_len) => (batch, seq_len, hidden_dim)
        x_pooled, _ = self.att_pooling(x.transpose(1, 2), x_mask)

        # Final linear
        out = self.fc(x_pooled)  # (batch, 1)
        return out



########################################
# Updated MultiLengthDiscriminator
########################################

class MultiLengthDiscriminator(nn.Module):
    """
    Multi-Length Discriminator that:
    - Takes in raw input sequences and optional text/emotion features.
    - Applies a single shared projection layer to raw input.
    - Optionally processes the concatenated features (x + text_hidden + emotion) via a shared GRU.
    - Optionally applies multi-head attention.
    - Then computes x_mask and x_mask_conv once and passes them to multiple AdvSeqDiscriminators of different kernel sizes.
    - Aggregates their outputs.
    """

    def __init__(self,
                 text_hidden=256, num_channels=1, hidden_dim=1024,
                 n_heads=0, dropout=0.5, kernel_size=[[3, 3, 5], [7, 7, 9, 11]],
                 emotion_hidden=0,use_cbam=True, att_dropout=0.3,
                 ssm_dropout=0.3, ssm_depths = [ 0, 0 ], norm="layer"):
        super(MultiLengthDiscriminator, self).__init__()

        self.text_hidden = text_hidden
        self.emotion_hidden = emotion_hidden
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads

        # Projection layer for input features
        self.proj = nn.Linear(num_channels, hidden_dim)

        # Compress text features if provided
        if text_hidden > 0:
            if text_hidden != hidden_dim:
                self.text_compress = nn.Linear(text_hidden, hidden_dim)
            else:
                self.text_compress = nn.Identity()
        else:
            self.text_compress = None

        # Project emotion features if provided
        if self.emotion_hidden > 0:
            self.em_proj = nn.Sequential(
                nn.Linear(emotion_hidden, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
            )
        else:
            self.em_proj = None


        # Optional Attention mechanism at the parent level
        if self.n_heads > 0:
            self.attention = MultiHeadAttention(hidden_dim, self.n_heads, alibi_alpha=1.5, start_i_increment=4,
                                                num_persistent=16)
            self.norm = nn.LayerNorm(hidden_dim)
            self.att_drop = nn.Dropout(att_dropout)
            self.drop1 = nn.Dropout(0.1)
        else:
            self.attention = None

        # Instantiate discriminators
        self.discriminators = nn.ModuleList()
        for i, ks in enumerate(kernel_size):
            d = AdvSeqDiscriminatorS4(
                hidden_dim=hidden_dim,
                conv_kernel_size=ks,
                num_ssm_layers=ssm_depths[i],
                conv_dropout=dropout,
                ssm_dropout=ssm_dropout,
                use_cbam=use_cbam,
                norm=norm,
            )
            self.discriminators.append(d)

    def forward(self, x, seq_lens, text_hidden=None, em_hidden=None):
        """
        Forward pass
        :param x: Input tensor of shape (batch_size, seq_len, num_channels)
        :param seq_lens: Sequence lengths tensor of shape (batch_size,)
        :param text_hidden: Optional text features of shape (batch_size, seq_len, text_hidden)
        :param em_hidden: Optional emotion features of shape (batch_size, emotion_hidden)
        :return: Combined discriminator scores of shape (batch_size, num_discriminators)
        """

        x = x.unsqueeze(-1)  # (batch, seq_len) => (batch, seq_len, 1)

        # Project the base input
        x = self.proj(x)  # (batch, seq_len, hidden_dim)

        # Add text features if present
        if text_hidden is not None and self.text_hidden > 0:
            text_hidden = self.text_compress(text_hidden)
            x = x + text_hidden

        # Add emotion features if present
        if em_hidden is not None and self.emotion_hidden > 0:
            em_h = self.em_proj(em_hidden).unsqueeze(1)  # (batch, 1, hidden_dim)
            x = x + em_h

        # Apply attention if enabled
        if self.n_heads > 0 and self.attention is not None:
            x_mask = sequence_mask(x.size(1), seq_lens)  # (batch, seq_len)
            att_mask = mask_to_attention_mask(x_mask)  # (batch, seq_len)

            x_att = self.attention(x, x, x, mask=att_mask)
            x = x + self.att_drop(x_att)
            x = self.norm(x)
            x = self.drop1(x)
        else:
            # If no attention, still need masks for convolutional ops
            x_mask = sequence_mask(x.size(1), seq_lens)

        # Prepare masks for convolutional blocks
        x_mask_conv = x_mask.unsqueeze(1)  # (batch, 1, seq_len)

        # Pass through each discriminator
        scores = []
        for discriminator in self.discriminators:
            score = discriminator(x, x_mask, x_mask_conv)  # (batch, 1)
            scores.append(score)

        # Concatenate all scores along the last dimension
        combined_score = torch.cat(scores, dim=1)  # (batch, num_discriminators)

        return combined_score


