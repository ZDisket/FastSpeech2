from IPython.utils import text
import torch
import torch.nn as nn
import torch.optim as optim
from .attentions import SEBlock1D, TransposeRMSNorm, AttentionPooling, TransposeLayerNorm, MultiHeadAttention, APTx, \
    ResidualBlock1D, CBAM1D
from .submodels import sequence_mask, mask_to_attention_mask, Prenet
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .s4 import S4Block as S4 
from torch.cuda.amp import GradScaler, autocast

class ConvBlock1D(nn.Module):
    """
    Simple Conv1D+ReLU+TransposeRMSNorm block for sequence modeling.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.3, act="relu"):
        super(ConvBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding="same")
        self.norm1 = TransposeLayerNorm(out_channels)  # Using TransposeRMSNorm
        self.relu = APTx() if act == "aptx" else nn.ReLU()  # Activation logic
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mask):
        out = self.conv1(x).masked_fill(x_mask, 0)
        out = self.norm1(out).masked_fill(x_mask, 0)
        out = self.relu(out).masked_fill(x_mask, 0)
        out = self.dropout(out)
        return out


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


class S4Block1D(nn.Module):
    """
    S4D + ReLU + TransposeLayerNorm + Dropout
    """
    def __init__(self, in_channels, out_channels, dropout=0.3, act="relu", d_state=64):
        super(S4Block1D, self).__init__()
        # Make sure in_channels == out_channels for simplicity (as in the original code with hidden_dim)
        assert in_channels == out_channels, "For simplicity, in_channels should match out_channels."
        
        # S4 layer
        self.s4 = S4(
            d_model=out_channels,
            d_state=d_state,
            dropout=dropout,
            transposed=True  # Input/Output are in (B, H, L) format
        )

        self.norm1 = TransposeLayerNorm(out_channels)
        self.relu = APTx() if act == "aptx" else nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mask):
        # x: (B, C, L) with C=out_channels
        
        out, _ = self.s4(x.float())  # S4D returns (output, state), ignore state here

        
        out = out.masked_fill(x_mask, 0)
        out = self.norm1(out).masked_fill(x_mask, 0)
        out = self.relu(out).masked_fill(x_mask, 0)
        out = self.dropout(out)
        return out


class AdvSeqDiscriminatorS4(nn.Module):
    """
    Conv+S4 Discriminator.
    ConvBlocks -> S4Blocks

    I tried residual design but found that it's hard to optimize.
    """
    def __init__(self, hidden_dim=1024, num_ssm_layers=6, conv_kernel_size=[3, 7, 11], conv_dropout=0.5, ssm_dropout=0.3, use_cbam=True):
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

        self.use_cbam = True
        self.num_ssm_layers = num_ssm_layers

        self.convs = nn.ModuleList(
            [ConvBlock1D(hidden_dim, hidden_dim, kernel_size=ks, dropout=conv_dropout) for ks in conv_kernel_size]
        )

        self.ssms = nn.ModuleList(
            [S4Block1D(hidden_dim, hidden_dim, dropout=ssm_dropout) for _ in range(num_ssm_layers)]
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
                 ssm_dropout=0.3, ssm_depths = [ 0, 0 ]):
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
                use_cbam=use_cbam
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


