import torch
import torch.nn as nn
import torch.optim as optim
from .attentions import SEBlock1D, TransposeRMSNorm, MultiHeadAttention, APTx, ResidualBlock1D
from .submodels import sequence_mask, mask_to_attention_mask

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
        x = x.transpose(1,2) # (batch, seq_len, 1) => (batch, 1, seq_len)

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
        scaled_x = scaled_x.transpose(1, 2) # (batch, 1, seq_len) => (batch, seq_len, 1)
        return scaled_x

class AdvSeqDiscriminator(nn.Module):
    """
    Sequence-level discriminator with masked global average pooling.
    Proj+MultiHeadAttn => ResidualBlocks => Masked Global Pooling => Fully Connected => Out

    No sigmoid at the end, please use BCEWithLogitsLoss
    Use n_heads=0 for no attention
    """

    def __init__(self, num_channels=1, num_conv_layers=3, conv_kernel_size=[3, 3, 3], hidden_dim=128, dropout=0.3, n_heads=4, att_dropout=0.3):
        super(AdvSeqDiscriminator, self).__init__()

        self.proj = nn.Linear(num_channels, hidden_dim)
        self.att_drop = nn.Dropout(att_dropout)
        self.drop1 = nn.Dropout(0.1)
        self.n_heads = n_heads


        if self.n_heads > 0:
            self.attention = MultiHeadAttention(hidden_dim, self.n_heads, alibi_alpha=1.5, start_i_increment=4, num_persistent=16)
            self.norm = nn.LayerNorm(hidden_dim)

        in_channels = hidden_dim

        self.blocks = nn.ModuleList(
            [
                ResidualBlock1D(in_channels, in_channels, kernel_size=conv_kernel_size[i], dropout=dropout) for i in range(num_conv_layers)
            ])

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, seq_lens, hidden_in=None, emotion_in=None):
        """
        Forward pass through the sequence-level discriminator with masked global average pooling.
        :param x: Durations (real or fake), size (batch, seq_len)
        :param seq_lens: Int tensor of length for each batch, size (batch,)
        :param hidden_in: Text hidden states, (batch, seq_len, channels)
        :param emotion_in: Emotion hidden states, (batch, 1, channels)
        :return: Probability that each sequence is real, size (batch, 1).
        """
        x_mask = sequence_mask(x.size(1), seq_lens)
        att_mask = mask_to_attention_mask(x_mask)
        x = x.unsqueeze(-1)  # (batch, seq_len) => (batch, seq_len, 1)

        x = self.proj(x) # (batch, seq_len, 1) => (batch, seq_len, hidden_dim)

        if hidden_in is not None:
            x = x + hidden_in[:, :x.size(1), :]

        if emotion_in is not None:
            x = x + emotion_in

        if self.n_heads > 0:
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


class DualDiscriminator(nn.Module):
    def __init__(self, text_hidden=256, num_channels=1, num_blocks=3, hidden_dim=128, n_heads=4, dropout=0.3, kernel_size=[3, 3, 3], emotion_hidden=256):
        super(DualDiscriminator, self).__init__()

        if text_hidden > 0:
            self.text_compress = nn.Conv1d(text_hidden, hidden_dim, 3, padding="same") if text_hidden != hidden_dim else nn.Identity()

        self.em_proj = nn.Sequential(nn.Linear(emotion_hidden, hidden_dim),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(0.5),)

        self.sequence_discriminator = AdvSeqDiscriminator(num_channels, num_conv_layers=num_blocks,
                                                          hidden_dim=hidden_dim, n_heads=n_heads, dropout=dropout, conv_kernel_size=kernel_size)
        self.difference_discriminator = AdvSeqDiscriminator(num_channels, num_conv_layers=num_blocks,
                                                            hidden_dim=hidden_dim, n_heads=n_heads, dropout=dropout,  conv_kernel_size=kernel_size)

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