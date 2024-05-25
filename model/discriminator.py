import torch
import torch.nn as nn
import torch.optim as optim


class PatchDiscriminator(nn.Module):
    """
    Duration-Patch discriminator.
    Conv1Ds => Compress => Out
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
            nn.Sigmoid()
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
        chunks = []
        batch_size = x.size(0)
        for b in range(batch_size):
            max_len = seq_lens[b]
            for i in range(0, max_len - self.chunk_size + 1, self.chunk_size):
                chunk = x[b:b + 1, i:i + self.chunk_size, :]  # Extract chunk of size (1, chunk_size, hidden_dim)
                chunk = self.pre_fc_conv(chunk.transpose(1, 2)).transpose(1, 2)
                chunk = self.fc(chunk)
                chunks.append(chunk)

        x = torch.cat(chunks, dim=0)
        x = x.squeeze(2).squeeze(1)
        # returns (n_chunks,)
        return x