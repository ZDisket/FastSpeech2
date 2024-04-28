import torch
import torch.nn as nn
from .attentions import TransformerEncoder, TransformerDecoder
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F



class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, forward_expansion, dropout, alibi_alpha=1.0):
        super(TextEncoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.emb_norm = nn.LayerNorm(embed_size)
        self.encoder = TransformerEncoder(embed_size, num_heads, num_layers, forward_expansion, dropout, alibi_alpha=alibi_alpha)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_size)

    def forward(self, token_ids, seq_lens):
        # Embed token_ids
        x = self.embed(token_ids)  # Shape: (batch, max_seq_len, embed_size)
        x = self.emb_norm(x)
        x = self.dropout(x)

        # Create a mask based on sequence lengths
        max_len = token_ids.size(1)
        mask = torch.arange(max_len, device=seq_lens.device).expand(len(seq_lens), max_len) >= seq_lens.unsqueeze(1)

        # Pass through the transformer encoder
        x = self.encoder(x, mask.unsqueeze(1).unsqueeze(2))

        # Apply dropout and LayerNorm after the encoder
        x = self.dropout(x)
        x = self.layer_norm(x)

        return x


class SConvNorm(nn.Module):
    def __init__(self, channels, num_layers=2, kernel_size=3):
        super(SConvNorm, self).__init__()
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(channels),
                nn.ReLU()
            ) for _ in range(num_layers)
        ])
        # x = tensor size (batch, channels, seq_len)
    def forward(self, x):
        for layer in self.conv_layers:
            residual = x
            x = layer(x)
            x = x + residual  # Adding residual connection
        return x

def lens_to_sequence_mask(lens_unpacked, max_seq_len):
    """
    Create a sequence mask for padded sequences.

    Args:
    lens_unpacked (Tensor): A 1D tensor containing the lengths of each sequence in the batch.
    max_seq_len (int): The maximum sequence length in the batch.

    Returns:
    Tensor: A binary mask of size (batch, 1, seq_len) where 1 indicates valid positions and 0 indicates padding.
    """
    batch_size = lens_unpacked.size(0)
    mask = torch.arange(max_seq_len).to(lens_unpacked.device).expand(batch_size, max_seq_len) < lens_unpacked.unsqueeze(1)
    mask = mask.unsqueeze(1).float()  # Add channel dimension and convert to float
    return mask.to(lens_unpacked.device)


def pad_to_original_length(lstm_output, original_seq_len, current_seq_len):
    """
    Pad the LSTM output back to the original input sequence length.

    Args:
    lstm_output (Tensor): The output tensor from the LSTM.
    original_seq_len (int): The original sequence length before any padding was applied.
    current_seq_len (int): The current sequence length after the LSTM processing.

    Returns:
    Tensor: Padded LSTM output tensor.
    """
    padding_length = original_seq_len - current_seq_len
    if padding_length > 0:
        # Pad the sequences to match the original input length
        padded_output = F.pad(lstm_output, [0, 0, 0, padding_length])
    else:
        padded_output = lstm_output

    return padded_output


def generate_masks_from_float_mask(float_mask):
    # Ensure src_mask is in the correct format for the Transformer which expects (batch, 1, 1, seq_len)
    src_mask = float_mask.bool().unsqueeze(1)  # Shape: (batch, 1, 1, seq_len)

    # Generate target mask combining src_mask with subsequent mask for causal attention
    seq_len = float_mask.size(2)
    subsequent_mask = torch.tril(torch.ones((seq_len, seq_len), device=float_mask.device, dtype=torch.bool))
    tgt_mask = src_mask & subsequent_mask.unsqueeze(0)  # Expand subsequent_mask to batch size

    return src_mask, tgt_mask


class VariantDurationPredictor(nn.Module):
    def __init__(self, text_channels, filter_channels=512, depth=4, heads=4, kernel_size=3, p_dropout=0.1,
                 final_dropout=0.2, conv_depth=2):
        super(VariantDurationPredictor, self).__init__()

        print("Using Variant Duration Predictor")
        self.use_dual_proj = False
        # Transformer Decoder
        self.decoder = TransformerDecoder(embed_size=filter_channels, heads=heads, num_layers=depth,
                                          forward_expansion=4, dropout=p_dropout)

        self.pre_convs = SConvNorm(filter_channels, num_layers=conv_depth)

        lstm_channels = filter_channels // 2

        self.lstm_channels = lstm_channels

        self.lstm = nn.LSTM(input_size=filter_channels, hidden_size=lstm_channels, batch_first=True)

        self.post_convs = SConvNorm(lstm_channels, num_layers=conv_depth // 2)

        self.out_proj = nn.Conv1d(in_channels=lstm_channels, out_channels=1, kernel_size=1)

        self.final_dropout = final_dropout
        self.use_pre_proj = False

        if text_channels != filter_channels:
            self.pre_proj = nn.Conv1d(text_channels, filter_channels, 1)
            self.use_pre_proj = True

    # x = Encoder hidden states size (batch, seq_len, text_channels)
    # x_lengths = Lengths of x size (batch_size,)
    def forward(self, x, x_lengths):
        x = x.transpose(1,2) # (batch, seq_len, text_channels) => (batch, text_channels, seq_len)

        x_mask = lens_to_sequence_mask(x_lengths, x.size(2))
        if self.use_pre_proj:
            x = self.pre_proj(x)

        x = self.pre_convs(x)  # x = (b, channels, seq_len)
        # Apply mask after convolutions
        x = x * x_mask

        x = F.dropout(x, self.final_dropout, training=True)

        # Perform transformer stuff

        # Transpose for decoder
        x = x.transpose(1, 2)  # (b, text_channels, seq_len) -> (b, seq_len, channels)

        src_mask, tgt_mask = generate_masks_from_float_mask(x_mask)

        # Transformer pass

        x = self.decoder(x, x, src_mask,
                         tgt_mask)  #x enc = (b, seq_len, channels)
        # Apply dropout even in inference
        x = F.dropout(x, self.final_dropout, training=True)

        # LSTM Portion
        # input must be (batch, seq_len, channels)

        # Pack padded sequence

        # max(x_lengths) is less than the seq_len dimension in x, often by 5 or 10
        # and the LSTM outputs a tensor of max(x_lengths) in its seq_len dimension
        # therefore, we save the original seq_len dimension for padding back later
        x_seq_len_orig = x.size(1)

        x = pack_padded_sequence(x, x_lengths.detach().cpu(),
                                 # pack_padded_sequence demands that the lengths tensor be on the CPU
                                 batch_first=True, enforce_sorted=False)

        # LSTM pass
        x, (hn, cn) = self.lstm(x)

        # Unpack the sequence
        x, lens_unpacked = pad_packed_sequence(x, batch_first=True)  # x_lstm:  (batch, seq_len, lstm_channels)
        x = F.dropout(x, self.final_dropout, training=True)
        # pad back to pre-LSTM seq_len
        x = pad_to_original_length(x, x_seq_len_orig, x.size(1))

        # Transpose dimensions for the post-convolution
        x = x.transpose(1, 2)  # (b, seq_len, channels) -> (b, channels, seq_len)
        x = self.post_convs(x)  # x = (b, channels, seq_len)
        x = x * x_mask
        x = F.dropout(x, self.final_dropout, training=True)

        # Project using 1D convolution
        log_durations = self.out_proj(x)

        log_durations *= x_mask

        log_durations = log_durations.squeeze(1) # (batch, 1, seq_len) => (batch, seq_len)

        return log_durations, x_mask