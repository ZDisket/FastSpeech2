from IPython.utils import text
import torch
import torch.nn as nn
import torch.optim as optim
from .attentions import SEBlock1D, TransposeRMSNorm, AttentionPooling, TransposeLayerNorm, MultiHeadAttention, APTx, \
    ResidualBlock1D, MaskedCBAM1d
from .submodels import sequence_mask, mask_to_attention_mask
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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


########################################
# Updated AdvSeqDiscriminator
########################################

class AdvSeqDiscriminator(nn.Module):
    """
    A sequence-level discriminator that:
    - Uses convolutional blocks (with masked convolution)
    - Optionally uses a CBAM block
    - Applies attention pooling at the end
    - Outputs a single logit (no sigmoid at the end)

    Note:
    - No attention or GRU layers here. Those are handled in the parent.
    - Requires x_mask and x_mask_conv from the parent for efficiency.
    """

    def __init__(self, hidden_dim=128, num_conv_layers=3, conv_kernel_size=[3, 3, 3], dropout=0.3, use_cbam=True):
        super(AdvSeqDiscriminator, self).__init__()

        self.use_cbam = use_cbam

        # Convolutional blocks
        self.blocks = nn.ModuleList(
            [ConvBlock1D(hidden_dim, hidden_dim, kernel_size=ks, dropout=dropout) for ks in conv_kernel_size]
        )

        # Optional CBAM block
        if self.use_cbam:
            self.cbam = MaskedCBAM1d(hidden_dim, pooling='att')

        # Attention-based pooling
        self.att_pooling = AttentionPooling(hidden_dim)

        # Final linear for classification
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, x_mask, x_mask_conv):
        """
        Forward pass through the discriminator.

        :param x: Input tensor of shape (batch, seq_len, hidden_dim),
                  already pre-processed by parent.
        :param x_mask: Boolean mask (batch, seq_len) for padding.
        :param x_mask_conv: Boolean mask (batch, 1, seq_len) for convolutional operations.
        :return: Tensor of shape (batch, 1) representing logits.
        """

        # (batch, seq_len, hidden_dim) => (batch, hidden_dim, seq_len)
        x = x.transpose(1, 2)

        # Apply convolutional blocks with masking
        for layer in self.blocks:
            x = x.masked_fill(x_mask_conv, 0)
            x = layer(x, x_mask_conv)

        # Apply CBAM if used
        if self.use_cbam:
            x = self.cbam(x, x_mask_conv)

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
                 text_hidden=256, num_channels=1, hidden_dim=768,
                 n_heads=0, dropout=0.5, kernel_size=[[3, 3], [7, 9]],
                 gru_size=256, emotion_hidden=256,
                 use_gru=True, use_cbam=True, att_dropout=0.3):
        super(MultiLengthDiscriminator, self).__init__()

        self.text_hidden = text_hidden
        self.emotion_hidden = emotion_hidden
        self.hidden_dim = hidden_dim
        self.use_gru = use_gru and gru_size > 0
        self.n_heads = n_heads

        # Projection layer for input features
        self.proj = nn.Linear(num_channels, hidden_dim)

        self.pre = ConvBlock1D(hidden_dim, hidden_dim, 5, dropout=0.5)

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

        # Optional shared GRU layer
        if self.use_gru:
            self.gru_channels = gru_size
            self.gru = nn.GRU(input_size=hidden_dim, hidden_size=self.gru_channels, batch_first=True,
                              bidirectional=True)
            self.gru_proj = nn.Sequential(nn.Linear(self.gru_channels * 2, hidden_dim), nn.Dropout(0.35))
        else:
            self.gru = None
            self.gru_proj = None

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
        for ks in kernel_size:
            d = AdvSeqDiscriminator(
                hidden_dim=hidden_dim,
                num_conv_layers=len(ks),
                conv_kernel_size=ks,
                dropout=dropout,
                use_cbam=use_cbam
            )
            self.discriminators.append(d)

    def run_rnn(self, x, seq_lens):
        """
        Run the GRU with packed sequences to handle variable-length inputs.

        :param x: Tensor (batch, seq_len, hidden_dim)
        :param seq_lens: Tensor (batch,) with lengths
        """
        x_seq_len_orig = x.size(1)
        x_packed = pack_padded_sequence(x, seq_lens.cpu(), batch_first=True, enforce_sorted=False)
        x_packed, _ = self.gru(x_packed)
        x_unpacked, _ = pad_packed_sequence(x_packed, batch_first=True, total_length=x_seq_len_orig)
        return x_unpacked

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

        # Run GRU if specified
        if self.use_gru:
            x_rnn = self.run_rnn(x, seq_lens)
            x = x + self.gru_proj(x_rnn)

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

        # Discriminator pre-net
        x = x + self.pre(x.transpose(1, 2), x_mask_conv).transpose(1, 2)

        # Pass through each discriminator
        scores = []
        for discriminator in self.discriminators:
            score = discriminator(x, x_mask, x_mask_conv)  # (batch, 1)
            scores.append(score)

        # Concatenate all scores along the last dimension
        combined_score = torch.cat(scores, dim=1)  # (batch, num_discriminators)

        return combined_score


class DualDiscriminator(nn.Module):
    def __init__(self, text_hidden=256, num_channels=1, num_blocks=3, hidden_dim=128, n_heads=4, dropout=0.3,
                 kernel_size=[3, 3, 3], emotion_hidden=256):
        super(DualDiscriminator, self).__init__()

        if text_hidden > 0:
            self.text_compress = nn.Conv1d(text_hidden, hidden_dim, 3,
                                           padding="same") if text_hidden != hidden_dim else nn.Identity()

        self.em_proj = nn.Sequential(nn.Linear(emotion_hidden, hidden_dim),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(0.5), )

        self.sequence_discriminator = AdvSeqDiscriminator(num_channels, num_conv_layers=num_blocks,
                                                          hidden_dim=hidden_dim, n_heads=n_heads, dropout=dropout,
                                                          conv_kernel_size=kernel_size)
        self.difference_discriminator = AdvSeqDiscriminator(num_channels, num_conv_layers=num_blocks,
                                                            hidden_dim=hidden_dim, n_heads=n_heads, dropout=dropout,
                                                            conv_kernel_size=kernel_size)

    def forward(self, x, seq_lens, text_hidden=None, em_hidden=None):
        """
        Forward pass
        :param x:
        :param seq_lens:
        :param text_hidden:
        :return:
        """
        if text_hidden is not None:
            text_hidden = self.text_compress(
                text_hidden.transpose(1, 2)
            ).transpose(1, 2)

        if em_hidden is not None:
            em_hidden = self.em_proj(em_hidden)

        # Forward pass through sequence discriminator
        sequence_score = self.sequence_discriminator(x, seq_lens, text_hidden, em_hidden)

        # Compute consecutive differences for the difference discriminator
        diff_x = x[:, :1] - x[:, :-1]
        diff_seq_lens = seq_lens - 1  # Adjust sequence lengths for differences

        # Forward pass through difference discriminator
        difference_score = self.difference_discriminator(diff_x, diff_seq_lens, text_hidden, em_hidden)

        # Concatenate both scores
        combined_score = torch.cat((sequence_score, difference_score), dim=1)

        return combined_score


class SeqDiscriminator(nn.Module):
    """
    Sequence-level discriminator with masked global average pooling.
    Conv1Ds => Masked Global Pooling => Fully Connected => Out
    """

    def __init__(self, num_channels=1, num_conv_layers=3, conv_kernel_size=3, hidden_dim=128, dropout=0.3):
        super(SeqDiscriminator, self).__init__()

        self.conv_layers = nn.ModuleList()
        in_channels = num_channels
        for _ in range(num_conv_layers):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, hidden_dim, conv_kernel_size, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(dropout)
                )
            )
            in_channels = hidden_dim

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, seq_lens):
        """
        Forward pass through the sequence-level discriminator with masked global average pooling.
        :param x: Durations (real or fake), size (batch, seq_len)
        :param seq_lens: Int tensor of length for each batch, size (batch,)
        :return: Probability that each sequence is real, size (batch, 1).
        """
        # x should have shape (batch_size, seq_len)
        x = x.unsqueeze(1)  # Add a channel dimension: (batch_size, 1, seq_len)

        for layer in self.conv_layers:
            x = layer(x)

        # Masked global average pooling
        x = self.masked_global_avg_pool1d(x, seq_lens)  # (batch_size, hidden_dim)

        # Fully connected layer
        x = self.fc(x)  # (batch_size, 1)

        return x

    def masked_global_avg_pool1d(self, x, seq_lens):
        """
        Perform masked global average pooling.
        :param x: Input tensor of shape (batch_size, hidden_dim, seq_len)
        :param seq_lens: Int tensor of shape (batch_size,) indicating the valid length of each sequence
        :return: Pooled tensor of shape (batch_size, hidden_dim)
        """
        batch_size, hidden_dim, max_len = x.size()

        # Create mask based on sequence lengths
        mask = torch.arange(max_len, device=x.device).expand(batch_size, max_len) < seq_lens.unsqueeze(1)
        mask = mask.unsqueeze(1).float()  # Shape: (batch_size, 1, seq_len)

        # Apply mask to the input
        x = x * mask

        # Sum over the sequence length dimension and divide by the actual lengths
        x = x.sum(dim=2) / seq_lens.unsqueeze(1).float()  # Shape: (batch_size, hidden_dim)

        return x


class PatchDiscriminator(nn.Module):
    """
    Duration-Patch discriminator.
    Conv1Ds => Compress => Out
    Very slow.
    """

    def __init__(self, num_channels=1, num_conv_layers=3, conv_kernel_size=3, hidden_dim=128, chunk_size=3,
                 dropout=0.3):
        super(PatchDiscriminator, self).__init__()
        self.chunk_size = chunk_size

        self.conv_layers = nn.ModuleList()
        in_channels = num_channels
        for _ in range(num_conv_layers):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, hidden_dim, conv_kernel_size, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(dropout)
                )
            )
            in_channels = hidden_dim
        self.pre_fc_conv = nn.Conv1d(hidden_dim, hidden_dim, chunk_size)  # process each chunk into 1
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, seq_lens):
        """
        Forward pass through the patch duration discriminator
        :param x: Durations (real or fake), size (batch, seq_len)
        :param seq_lens: Int tensor of length for each batch, size (batch,)
        :return: Probabilities that each chunk size chunk_size (defined in constructor) are true, size (n_chunks,).
        No batch dim.
        """
        # x should have shape (batch_size, seq_len)
        x = x.unsqueeze(1)  # Add a channel dimension: (batch_size, 1, seq_len)
        for layer in self.conv_layers:
            x = layer(x)

        # Transpose to (batch_size, seq_len, hidden_dim) for linear layers
        x = x.transpose(1, 2)

        # Process each chunk independently
        all_chunks = []
        batch_size = x.size(0)
        for b in range(batch_size):
            max_len = seq_lens[b]
            chunks = [x[b:b + 1, i:i + self.chunk_size, :] for i in
                      range(0, max_len - self.chunk_size + 1, self.chunk_size)]
            all_chunks.extend(chunks)

        if all_chunks:
            chunks = torch.cat(all_chunks, dim=0)  # Concatenate all chunks for batch processing
            chunks = self.pre_fc_conv(chunks.transpose(1, 2)).transpose(1, 2)
            chunks = self.fc(chunks)

            x = chunks.squeeze(2).squeeze(1)
            return x
        else:
            return torch.tensor([]).to(x.device)
