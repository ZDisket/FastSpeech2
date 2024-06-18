import torch
import torch.nn as nn
import torch.optim as optim
from .attentions import MultiHeadAttention, APTx, ResidualBlock1D, RMSNorm, SwiGLUConvFFN
from .submodels import sequence_mask, mask_to_attention_mask
import torch.nn.functional as F



class ReduceSequenceLength(nn.Module):
    def __init__(self):
        super(ReduceSequenceLength, self).__init__()

    def forward(self, input_tensor):
        """
        Reduces the sequence length of a tensor by half using 1D max pooling.

        Args:
        input_tensor (torch.Tensor): Input tensor of shape (seq_len, N, dim).

        Returns:
        torch.Tensor: Output tensor with reduced sequence length, shape (seq_len//2, N, dim).
        """
        seq_len, N, dim = input_tensor.shape

        # Ensure seq_len is even for the 1D max pooling to reduce by half
        assert seq_len % 2 == 0, "Sequence length should be even for halving with max pooling"

        # Permute the tensor to (N, dim, seq_len) for 1D max pooling
        input_tensor_permuted = input_tensor.permute(1, 2, 0)  # (N, dim, seq_len)

        # Apply 1D max pooling with kernel size 2 and stride 2
        output_tensor_permuted = F.max_pool1d(input_tensor_permuted, kernel_size=2, stride=2)

        # Permute back to original dimensions (seq_len//2, N, dim)
        output_tensor = output_tensor_permuted.permute(2, 0, 1)  # (seq_len//2, N, dim)

        return output_tensor

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionPooling, self).__init__()
        self.attention_weights = nn.Parameter(torch.Tensor(hidden_dim, 1))
        nn.init.xavier_uniform_(self.attention_weights)

    def forward(self, x, mask):
        # x: (batch, seq_len, hidden_dim)
        # mask: (batch, seq_len)
        attn_scores = torch.matmul(x, self.attention_weights).squeeze(-1)  # (batch, seq_len)
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=-1).unsqueeze(-1)  # (batch, seq_len, 1)
        context = torch.sum(attn_weights * x, dim=1)  # (batch, hidden_dim)
        return context, attn_weights


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
        final_hid = x.transpose(1, 2)  # (batch, hidden_dim, seq_len) => (batch, seq_len, hidden_dim)

        x, attention_weights = self.att_pooling(final_hid, x_mask.squeeze(1))

        # Fully connected layer
        x = self.fc(x)  # (batch_size, n_classes)

        # (batch, n_blocks, hidden_dim, seq_len) => (batch, n_blocks, seq_len, hidden_dim)
        x_blocks = x_blocks.transpose(2, 3)

        return x, (x_blocks, final_hid, attention_weights)
