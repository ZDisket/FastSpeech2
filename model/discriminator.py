import torch
import torch.nn as nn
import torch.optim as optim
from .attentions import SEBlock1D, TransposeRMSNorm, MultiHeadAttention
from .submodels import sequence_mask, mask_to_attention_mask


class ResidualBlock1D(nn.Module):
    """
    Conv1D+Squeeze-Excite+RMSNorm residual block for sequence modeling
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.3):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.norm1 = TransposeRMSNorm(out_channels)
        self.norm2 = TransposeRMSNorm(out_channels)
        self.se = SEBlock1D(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

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


class AdvSeqDiscriminator(nn.Module):
    """
    Sequence-level discriminator with masked global average pooling.
    Proj+MultiHeadAttn => ResidualBlocks => Masked Global Pooling => Fully Connected => Out

    No sigmoid at the end, please use BCEWithLogitsLoss
    """

    def __init__(self, num_channels=1, num_conv_layers=3, conv_kernel_size=3, hidden_dim=128, dropout=0.3, n_heads=4, att_dropout=0.3):
        super(AdvSeqDiscriminator, self).__init__()

        self.proj = nn.Linear(num_channels, hidden_dim)
        self.att_drop = nn.Dropout(att_dropout)
        self.drop1 = nn.Dropout(0.1)
        self.attention = MultiHeadAttention(hidden_dim, n_heads, alibi_alpha=1.5, start_i_increment=4, num_persistent=16)
        self.norm = nn.LayerNorm(hidden_dim)

        in_channels = hidden_dim

        self.blocks = nn.ModuleList(
            [
                ResidualBlock1D(in_channels, in_channels, kernel_size=conv_kernel_size, dropout=dropout) for _ in range(num_conv_layers)
            ])

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
        x_mask = sequence_mask(x.size(1), seq_lens)
        att_mask = mask_to_attention_mask(x_mask)
        x = x.unsqueeze(-1)  # (batch, seq_len) => (batch, seq_len, 1)

        x = self.proj(x) # (batch, seq_len, 1) => (batch, seq_len, hidden_dim)
        x_att = self.attention(x, x, x, mask=att_mask)
        x = x + self.att_drop(x_att)
        x = self.norm(x)
        x = self.drop1(x)

        # Prepare for convs
        x = x.transpose(1,2) # (batch, seq_len, hidden_dim) => (batch, hidden_dim, seq_len)
        x_mask = x_mask.unsqueeze(1) # (batch, seq_len) => (batch, 1, seq_len)

        for layer in self.blocks:
            x.masked_fill(x_mask, 0)
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
    def __init__(self, num_channels=1, num_conv_layers=3, conv_kernel_size=3, hidden_dim=128, chunk_size=3, dropout=0.3):
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
            chunks = [x[b:b + 1, i:i + self.chunk_size, :] for i in range(0, max_len - self.chunk_size + 1, self.chunk_size)]
            all_chunks.extend(chunks)

        if all_chunks:
            chunks = torch.cat(all_chunks, dim=0)  # Concatenate all chunks for batch processing
            chunks = self.pre_fc_conv(chunks.transpose(1, 2)).transpose(1, 2)
            chunks = self.fc(chunks)

            x = chunks.squeeze(2).squeeze(1)
            return x
        else:
            return torch.tensor([]).to(x.device)