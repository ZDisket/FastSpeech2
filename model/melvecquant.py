import torch
import torch.nn as nn
import torch.nn.functional as F
from .subatts import APTx
from torch.nn.utils import weight_norm
from .finite_scalar_quantization import FSQ

def sequence_mask(max_length, x_lengths):
    """
    Make a bool sequence mask
    :param max_length: Max length of sequences
    :param x_lengths: Tensor (batch,) indicating sequence lengths
    :return: Bool tensor size (batch, max_length) where True is padded and False is valid
    """
    mask = torch.arange(max_length).expand(len(x_lengths), max_length).to(x_lengths.device)
    mask = mask >= x_lengths.unsqueeze(1)
    return mask

# -------- helpers -------- #
def _safe_avg(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """average over H×W while ignoring masked (invalid) positions"""
    num = (x * valid).sum(dim=(2, 3), keepdim=True)
    den = valid.sum(dim=(2, 3), keepdim=True).clamp(min=1.0)
    return num / den


def _safe_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """max over H×W while ignoring masked (True) positions"""
    x_masked = x.masked_fill(mask, float("-inf"))
    m = x_masked.amax(dim=(2, 3), keepdim=True)  # (B,C,1,1)
    m = m.masked_fill(torch.isinf(m), 0.0)       # if all −inf → 0
    return m


# ---------------- SAM ---------------- #
class SAM(nn.Module):
    """Spatial Attention Module (2-D) with optional padding mask"""

    def __init__(self, bias: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            pooled_max = x.max(dim=1, keepdim=True)[0]
            pooled_avg = x.mean(dim=1, keepdim=True)
        else:
            mask_exp = mask.expand_as(x)
            valid = (~mask).float()
            pooled_max = _safe_max(x, mask_exp)
            pooled_avg = _safe_avg(x, valid)

        concat = torch.cat((pooled_max, pooled_avg), dim=1)
        attn = torch.sigmoid(self.conv(concat))
        out = attn * x
        if mask is not None:
            out = out.masked_fill(mask, 0.0)
        return out


# ---------------- CAM ---------------- #
class CAM(nn.Module):
    """Channel Attention Module (2-D) with optional padding mask"""

    def __init__(self, channels: int, r: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // r, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channels // r, channels, bias=True),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            max_pool = F.adaptive_max_pool2d(x, 1)   # (B,C,1,1)
            avg_pool = F.adaptive_avg_pool2d(x, 1)
        else:
            mask_exp = mask.expand_as(x)
            valid = (~mask).float()
            max_pool = _safe_max(x, mask_exp)
            avg_pool = _safe_avg(x, valid)

        b, c, _, _ = x.size()
        attn = self.mlp(max_pool.view(b, c)) + self.mlp(avg_pool.view(b, c))
        attn = torch.sigmoid(attn).view(b, c, 1, 1)

        out = attn * x
        if mask is not None:
            out = out.masked_fill(mask, 0.0)
        return out


# ---------------- CBAM2D ---------------- #
class CBAM2D(nn.Module):
    """CBAM block (Channel ➔ Spatial) with mask support"""

    def __init__(self, channels: int, r: int = 16):
        super().__init__()
        self.cam = CAM(channels, r)
        self.sam = SAM(bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        out = self.cam(x, mask)
        out = self.sam(out, mask)
        if mask is not None:
            out = out.masked_fill(mask, 0.0)
        return out + x






# assumes CBAM2D and (optionally) APTx are defined elsewhere

class ResidualBlock2D(nn.Module):
    """
    Conv2D + CBAM residual block for image-like tensors.

    Features kept from the 1-D version
    ----------------------------------
    • weight-norm option
    • instance-norm option
    • optional CBAM attention (now CBAM2D)
    • APTx / ReLU activation selector
    • dropout on the residual output
    • 1×1 projection when in/out channels differ
    • boolean mask support (B, 1, H, W) to zero-out padded areas

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
        act: str = "aptx",          # "relu" | "aptx"
        norm: str = "weight"      # "weight" | "instance"
    ):
        super().__init__()

        assert norm in ("weight", "instance"), (
            f"Unknown norm '{norm}'. Use 'weight' or 'instance'."
        )

        # ---------------- Convolutions ---------------- #
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding="same",
            dilation=dilation,
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size,
            padding="same",
            dilation=dilation,
        )

        # ---------------- Normalisation ---------------- #
        if norm == "weight":
            self.conv1 = weight_norm(self.conv1)
            self.conv2 = weight_norm(self.conv2)
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        else:  # instance norm
            self.norm1 = nn.InstanceNorm2d(out_channels, affine=True)
            self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)

        # ---------------- Attention ---------------- #
        self.cbam = CBAM2D(out_channels)

        # ---------------- Misc ---------------- #
        self.act = APTx() if act == "aptx" else nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout)
        self.proj = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, x_mask = None) -> torch.Tensor:
        """
        Args
        ----
        x       : (B, C_in, H, W) tensor
        x_mask  : Optional bool tensor (B, 1, H, W) where True = padding

        Returns
        -------
        (B, C_out, H, W) tensor
        """
        residual = self.proj(x)

        out = self.conv1(x)
        out = self.norm1(out)
        if x_mask is not None:
            out = out.masked_fill(x_mask, 0)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.cbam(out, x_mask)
        out = out + residual

        if x_mask is not None:
            out = out.masked_fill(x_mask, 0)
        out = self.act(out)
        out = self.dropout(out)
        return out


class MVQGenerator(nn.Module):
    def __init__(self, mel_channels, channels, kernel_sizes, fsq_levels=[8, 8, 5, 5, 5], dropout=0.1):
        """
        Spectrogram Pre-Encoder.
        ResNet-based autoencoder with configurable encoder and decoder blocks.

        Parameters:
          - mel_channels (int): number of channels in the input spectrogram.
          - channels (list of ints): list of channel dimensions for encoder blocks.
            * The first element is the projected input dimension.
            * The last element is the latent dimension.
          - kernel_sizes (list of ints): list of kernel sizes for each ResidualBlock1D.
            Length should be len(channels) - 1. The decoder will use these lists in reverse.
        """
        super(MVQGenerator, self).__init__()
        # Project input from 1 to first channels
        self.proj = nn.Conv2d(1, channels[0], 1)
        self.quantizer_dim = len(fsq_levels)
        # Encoder: build a sequence of ResidualBlock2D modules
        self.encoder_blocks = nn.ModuleList([
            ResidualBlock2D(channels[i], channels[i + 1], kernel_size=kernel_sizes[i], dropout=dropout, act="aptx", norm="weight")
            for i in range(len(channels) - 1)
        ])

        # Quantization stage: here we use the latent dimension as the last element of channels.
        latent_dim = channels[-1]

        # we collapse the Conv2d channels and treat mel channels as channels for quant
        self.pre_q_proj = nn.Conv2d(latent_dim, 1, 1)
        self.q_in_proj = nn.Linear(mel_channels, self.quantizer_dim)

        self.quantizer = FSQ(levels=fsq_levels)
        self.q_out_proj = nn.Linear(self.quantizer_dim, mel_channels)
        self.post_q_proj = nn.Conv2d(1, latent_dim, 1)

        self.codebook_size = 8010  # TODO: dyn calculate this
        self.bos_token_id = 8001
        self.eos_token_id = 8002

        # Decoder: use the reversed lists so that the decoder mirrors the encoder.
        rev_channels = list(reversed(channels))
        rev_kernel_sizes = list(reversed(kernel_sizes))
        self.decoder_blocks = nn.ModuleList([
            ResidualBlock2D(rev_channels[i], rev_channels[i + 1], kernel_size=rev_kernel_sizes[i], dropout=dropout,
                            act="aptx", norm="weight")
            for i in range(len(rev_channels) - 1)
        ])

        # Output projection: map from the decoder’s final channel (channels[0]) to 1.
        self.out_proj = nn.Conv2d(channels[0], 1, 1)

    def forward(self, x, x_lengths):
        """
        Forward pass.

        Parameters:
          - x: Tensor of shape (batch, mel_len, mel_channels)
          - x_lengths: (batch,), int lengths of each thing
        Returns:
          - Reconstructed tensor of shape (batch, mel_len, mel_channels)
        """

        x_mask = sequence_mask(x.size(1), x_lengths) # (B, T)
        x_mask = x_mask.reshape(x_mask.size(0), 1, 1, x_mask.size(1)) # (B, 1, 1, T)

        x = self.project_in(x, x_mask)

        # Pass through the encoder blocks
        for block in self.encoder_blocks:
            x = block(x, x_mask=x_mask)

        xhat, indices = self.quantize(x)

        x = self.dequantize(xhat)

        # Pass through the decoder blocks
        for block in self.decoder_blocks:
            x = block(x, x_mask=x_mask)

        x = self.project_out(x)

        return x

    def project_out(self, x):
        x = self.out_proj(x)  # (B, 1, Cmel, Tmel)
        x = x.squeeze(1).transpose(1, 2)  # (B, Tmel, Cmel)
        return x

    def dequantize(self, xhat):
        x = self.q_out_proj(xhat)  # (B, Tmel, Cmel)
        x = x.transpose(1, 2)  # (B, Cmel, Tmel)
        x = x.unsqueeze(1)  # (B, 1, Cmel, Tmel)
        x = self.post_q_proj(x)  # (B, C, Cmel, Tmel)
        return x

    def quantize(self, x):
        x = self.pre_q_proj(x)  # (B, 1, Cmel, Tmel)
        x = x.squeeze(1)  # (B, Cmel, Tmel)
        x = x.transpose(1, 2)  # (B, Tmel, Cmel)
        x = self.q_in_proj(x)
        xhat, indices = self.quantizer(x)
        return xhat, indices

    def project_in(self, x, x_mask):
        # (B, MelLen, MelChannels)
        # (B, C, MelChannels, MelLen)
        x = x.transpose(1, 2)  # (B, Cmel, Lmel)
        x = x.unsqueeze(1)  # (B, 1, Cmel, Lmel)
        x = self.proj(x)  # (B, C, Cmel, Lmel)
        x = x.masked_fill(x_mask, 0.0)
        return x

    def encode(self, x, x_mask=None):
        """
        Encodes the input spectrogram into discrete latent indices.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, mel_len, mel_channels).
          - x_mask: Tensor of shape (batch, mel_len), bool where padded positions are True.
                   (This mask will be passed to each ResidualBlock2D, which is assumed to apply
                   .masked_fill(x_mask, 0) before its activation calls.)
        Returns:
            indices (torch.Tensor): Discrete token indices from the vector quantizer.
        """
        if x_mask is None:
            x_mask = torch.zeros((x.size(0), 1, 1, x.size(1)), device=x.device).bool()

        x = self.project_in(x, x_mask)

        # Pass through the encoder blocks
        for block in self.encoder_blocks:
            x = block(x, x_mask=x_mask)

        _, indices = self.quantize(x)

        return indices.long()  # otherwise cross entropy loss bitches later

    def decode(self, indices, x_mask=None):
        """
        Decodes discrete latent indices into a reconstructed spectrogram.

        Args:
            indices (torch.Tensor): Discrete token indices from the vector quantizer.

        Returns:
            x (torch.Tensor): Reconstructed spectrogram of shape (batch, mel_len, mel_channels).
        """
        # Convert indices to quantized latent codes (shape: (batch, mel_len, 4))
        xhat = self.quantizer.indices_to_codes(indices)
        x = self.dequantize(xhat)

        if x_mask is None:
            x_mask = torch.zeros((x.size(0), 1, 1, x.size(3)), device=x.device).bool()

        # Pass through the decoder blocks
        for block in self.decoder_blocks:
            x = block(x, x_mask=x_mask)

        x = self.project_out(x)
        return x

