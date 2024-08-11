import torch
import torch.nn as nn
import torch.optim as optim
from .attentions import MultiHeadAttention, APTx, ResidualBlock1D, RMSNorm, SwiGLUConvFFN, AttentionPooling
from .submodels import sequence_mask, mask_to_attention_mask
import torch.nn.functional as F



class Zephyr(nn.Module):
    """
    Emotion classification model for assisting TTS.
    NormalizedEmbedding => MHA(4H) => 2-Residual => MHA(2H) => 2-Residual => Attention => Out

    Increasing dilation in order to capture more context, and to encourage each block to be "wider" in its
    reception due to the nature of our multi-head attention

    """
    def __init__(self, vocab_size, n_classes, num_conv_layers=4, kernel_sizes=[3, 3, 4, 5], dilations=[1, 2, 4, 6], hidden_dim=256, dropout=0.1, n_heads=4,
                 n_inter_heads=2, att_dropout=0.3, act="relu"):
        super(Zephyr, self).__init__()

        self.att_drop = nn.Dropout(att_dropout)
        self.drop1 = nn.Dropout(0.1)
        self.n_heads = n_heads

        # Normalized embedding
        self.emb = nn.Sequential(
            nn.Embedding(vocab_size, hidden_dim),
            RMSNorm(hidden_dim),
            nn.Dropout(0.1),
        )

        if self.n_heads > 0:
            self.attention = MultiHeadAttention(hidden_dim, self.n_heads, alibi_alpha=1.5, start_i_increment=4, num_persistent=16)
            self.norm = RMSNorm(hidden_dim)

        in_channels = hidden_dim

        self.blocks = nn.ModuleList(
            [
                ResidualBlock1D(in_channels, in_channels, kernel_size=kernel_sizes[i], dilation=dilations[i], dropout=dropout, act=act) for i in range(num_conv_layers)
            ])

        self.inter_att = MultiHeadAttention(hidden_dim, n_inter_heads, alibi_alpha=1.5, start_i_increment=6, num_persistent=16)

        # Attention pooling layer
        self.att_pooling = AttentionPooling(hidden_dim)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x, seq_lens):
        """
        Forward pass
        :param x: Indices of text size (batch, seq_len)
        :param seq_lens: Int tensor of length for each batch, size (batch,)
        :return: Tensor size (batch, n_classes), tuple: (detached hidden states size (batch, n_blocks, seq_len, hidden_dim), final hidden states (batch, 1, hidden_dim), attention weights)
        """

        x = self.emb(x)

        x_mask = sequence_mask(x.size(1), seq_lens)
        att_mask = mask_to_attention_mask(x_mask)

        if self.n_heads > 0:
            x_att = self.attention(x, x, x, mask=att_mask)
            x = x + self.att_drop(x_att)
            x = self.norm(x)
            x = self.drop1(x)

        # Prepare for convs
        x = x.transpose(1,2) # (batch, seq_len, hidden_dim) => (batch, hidden_dim, seq_len)
        x_mask = x_mask.unsqueeze(1) # (batch, seq_len) => (batch, 1, seq_len)

        # (batch, n_blocks, hidden_dim, seq_len)
        x_blocks = torch.zeros(x.size(0), len(self.blocks), x.size(1), x.size(2)).to(x.device)
        x_blocks.requires_grad = False

        for l_i, layer in enumerate(self.blocks):
            if l_i == (len(self.blocks) // 2) - 1:
                x = x.transpose(1,2)
                x_att = self.inter_att(x, x, x, mask=att_mask)
                x_att = self.att_drop(x_att)
                x = x + x_att
                x = x.transpose(1, 2)

            x.masked_fill(x_mask, 0)
            x = layer(x)

            x_blocks[:, l_i, :, :] = x

        # Attention pooling
        x = x.transpose(1, 2)  # (batch, hidden_dim, seq_len) => (batch, seq_len, hidden_dim)

        final_hid, attention_weights = self.att_pooling(x, x_mask.squeeze(1))

        # Fully connected layer
        x = self.fc(final_hid)  # (batch_size, n_classes)

        # (batch, n_blocks, hidden_dim, seq_len) => (batch, n_blocks, seq_len, hidden_dim)
        x_blocks = x_blocks.transpose(2, 3)

        return x, (x_blocks, final_hid, attention_weights)
