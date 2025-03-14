import torch
import torch.nn as nn
from .attentions import TransformerEncoder, TemporalConvNet, MultiHeadAttention, \
    mask_to_causal_attention_mask, TransposeLayerNorm, AttentionPooling, APTxS1, APTx, SwiGLUConvFFN, NeoTCNAttention, \
    ConvReluNorm, TransformerDecoder
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from .attblocks import CBAM2d, MaskedSEBlock1D, CBAM1D
from .subatts import init_weights_he
import monotonic_align, math
from torchbnn import BayesLinear

# Applying LayerNorm + Dropout on embeddings increases performance, probably due to the regularizing effect
# Thanks dathudeptrai from TensorFlowTTS for discovering this.
class NormalizedEmbedding(nn.Module):
    """
    Embedding + LayerNorm + Dropout
    """

    def __init__(self, num_embeddings, embedding_dim, dropout=0.1, norm=True):
        super(NormalizedEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim) if norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x


class StochasticDropout(nn.Module):
    """
    Also known as Monte Carlo dropout, StochasticDropout keeps a lower dropout rate to apply during inference, for stochasticity.
    If not specified, it will be dropout / 3, with the minimum being 0.1
    Note: Currently unused
    """

    def __init__(self, p=0.5, p_inference=None, min_p_inference=0.1, stochastic=False):
        super(StochasticDropout, self).__init__()
        self.p = p  # Dropout probability during training

        if p_inference is None:
            p_inference = max(min_p_inference, p / 3)  # ensure stochastic dropout is at least 0.1

        self.p_inference = p_inference  # Dropout probability during inference
        self.stochastic = stochastic

    def forward(self, x):
        if self.training:
            return F.dropout(x, self.p, self.training)
        else:
            if self.stochastic:
                return F.dropout(x, self.p, True)
            else:
                return F.dropout(x, self.p, self.training)


class ResNetEmProj(nn.Module):
    def __init__(self, emotion_channels, embed_size, cond_heads, kernel_size, drop=0.5):
        super(ResNetEmProj, self).__init__()
        self.emotion_channels = emotion_channels
        self.cond_heads = cond_heads
        self.cond_head_size = embed_size // self.cond_heads
        inter_size = emotion_channels // 2

        self.conv1 = nn.Conv2d(emotion_channels, inter_size, kernel_size, padding="same")
        self.ln1 = nn.LayerNorm([self.cond_heads, inter_size])
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(drop)

        self.conv2 = nn.Conv2d(inter_size, self.cond_head_size, kernel_size, padding="same")
        self.ln2 = nn.LayerNorm([self.cond_heads, self.cond_head_size])
        self.dropout2 = nn.Dropout(drop)

        self.cbam = CBAM2d(self.cond_head_size)

        # Use 1x1 convolution to match dimensions for the residual connection
        self.residual_conv = nn.Conv2d(emotion_channels, self.cond_head_size,
                                       1) if emotion_channels != self.cond_head_size else nn.Identity()
        self.residual_ln = nn.LayerNorm(
            [self.cond_heads, self.cond_head_size]) if emotion_channels != self.cond_head_size else nn.Identity()

        self.final_act = nn.ReLU()

    def forward(self, x, mask):
        # Mask should be of shape (batch, max_len) and needs to be expanded
        # to match the shape of x: (batch, n_blocks, seq_len, hidden)
        mask = mask.unsqueeze(-1).unsqueeze(-1)  # (batch, max_len, 1, 1)
        mask = mask.transpose(1, 2)  # (batch, 1, max_len, 1)

        x = x.masked_fill(mask, 0)

        # Permute to (batch, hidden, n_blocks, seq_len) for convolution
        x = x.permute(0, 3, 1, 2)

        residual = self.residual_conv(x)
        # (batch, hidden, n_blocks, seq_len) => (batch, seq_len, n_blocks, hidden)
        residual = residual.permute(0, 3, 2, 1)
        residual = self.residual_ln(residual)
        residual = residual.permute(0, 3, 2, 1)

        # (batch, 1, max_len, 1) => (batch, 1, 1, max_len)
        inter_mask = mask.transpose(2, 3)

        out = self.conv1(x)
        out = out.masked_fill(inter_mask, 0)
        out = out.permute(0, 3, 2, 1)
        out = self.ln1(out)
        out = out.permute(0, 3, 2, 1)
        out = self.relu(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = out.masked_fill(inter_mask, 0)
        out = out.permute(0, 3, 2, 1)
        out = self.ln2(out)
        out = out.permute(0, 3, 2, 1)
        out = self.relu(out)
        out = self.dropout2(out)

        out += residual

        out = self.cbam(out)

        out = self.relu(out)

        # Permute back to original shape (batch, n_blocks, seq_len, hidden)
        out = out.permute(0, 2, 3, 1)

        return out


class SimpleEmProj(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(SimpleEmProj, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding="same")
        self.relu = nn.ReLU()

    def forward(self, x, mask):
        # Mask should be of shape (batch, max_len) and needs to be expanded
        # to match the shape of x: (batch, n_blocks, seq_len, hidden)
        mask = mask.unsqueeze(-1).unsqueeze(-1)  # (batch, max_len, 1, 1)
        mask = mask.transpose(1, 2)  # (batch, 1, max_len, 1)

        # Apply the mask to zero out positions
        x = x.masked_fill(mask, 0)

        # Permute to (batch, hidden, n_blocks, seq_len) for convolution
        x = x.permute(0, 3, 1, 2)

        x = self.conv(x)
        x = self.relu(x)

        # Permute back to original shape (batch, n_blocks, seq_len, hidden)
        x = x.permute(0, 2, 3, 1)

        return x



class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, forward_expansion, dropout, kernel_sizes, alibi_alpha=1.0,
                 start_i=0, emotion_channels=256, speaker_channels=0):
        super().__init__()
        self.embed = NormalizedEmbedding(vocab_size, embed_size)
        self.encoder = TransformerEncoder(embed_size, num_heads, num_layers, forward_expansion, dropout,
                                          alibi_alpha=alibi_alpha, start_i=start_i, multi_scale=True, kernel_size=kernel_sizes,
                                          act="relugtz")
        self.use_prenet = False
        if self.use_prenet:
            self.pre = Prenet(embed_size, 384, embed_size, 5, 3, 0.5, "aptx")

        self.emotion_channels = emotion_channels
        self.speaker_channels = speaker_channels

        if self.speaker_channels > 0:
            self.spk_cond = nn.Linear(speaker_channels, embed_size)

    def forward(self, token_ids, seq_lens, encoded_em, spk_emb=None):
        # Embed token_ids
        x = self.embed(token_ids)  # Shape: (batch, max_seq_len, embed_size)

        # Create a mask based on sequence lengths
        max_len = token_ids.size(1)
        mask = torch.arange(max_len, device=seq_lens.device).expand(len(seq_lens), max_len) >= seq_lens.unsqueeze(1)
        x_mask = sequence_mask(max_len, seq_lens)

        if self.speaker_channels > 0:
            x = x + self.spk_cond(spk_emb)

        if self.use_prenet:
            x = self.pre(x, x_mask.unsqueeze(1))

        if self.emotion_channels > 0:
            x[:, :, :self.emotion_channels] = encoded_em.unsqueeze(1)

        # Pass through the transformer encoder
        x = self.encoder(x, mask.unsqueeze(1).unsqueeze(2))

        return x


class SConvNorm(nn.Module):
    def __init__(self, channels, num_layers=2, kernel_size=3, p_dropout=0.1):
        super(SConvNorm, self).__init__()
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(channels),
                nn.Dropout(p_dropout),
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
    mask = torch.arange(max_seq_len).to(lens_unpacked.device).expand(batch_size, max_seq_len) < lens_unpacked.unsqueeze(
        1)
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
    def __init__(self, text_channels, filter_channels=256, depth=4, heads=4, kernel_size=3, p_dropout=0.2,
                 final_dropout=0.2, conv_depth=2, lstm_bidirectional=True, start_i=0, bayesian=False, use_cbam=True):
        super(VariantDurationPredictor, self).__init__()

        print("Using Variant Duration Predictor")
        self.use_dual_proj = False
        self.lstm_bidirectional = lstm_bidirectional
        self.conv_depth = conv_depth
        self.use_cbam = use_cbam
        bayesian = False

        self.conv_layers = nn.ModuleList()

        for _ in range(conv_depth):
            self.conv_layers.append(
                ConvReluNorm(filter_channels, filter_channels, kernel_size, 1, act="relu", causal=False,
                             dropout=p_dropout))

        if self.use_cbam:
            self.cbam = CBAM1D(filter_channels)

        self.lstm_channels = filter_channels

        self.lstm = nn.GRU(input_size=filter_channels, hidden_size=self.lstm_channels, batch_first=True,
                            bidirectional=self.lstm_bidirectional)
        if self.lstm_bidirectional:
            print("BiGRU")

        if not bayesian:
            self.out_proj = nn.Linear(self.lstm_channels * 2 if self.lstm_bidirectional else self.lstm_channels, 1)
        else:
            self.out_proj = BayesLinear(prior_mu=0.0, prior_sigma=0.01, # very low prior sigma because we operate in the log domain
                                        in_features=self.lstm_channels * 2 if self.lstm_bidirectional else self.lstm_channels,
                                        out_features=1)



        self.final_dropout = nn.Dropout(final_dropout)
        self.drop = nn.Dropout(0.1)
        self.use_pre_proj = False


        if text_channels != filter_channels:
            self.pre_proj = nn.Conv1d(text_channels, filter_channels, 1)
            self.use_pre_proj = True

    # x = Encoder hidden states size (batch, seq_len, text_channels)
    # x_lengths = Lengths of x size (batch_size,)
    def forward(self, x, x_lengths, in_em, in_spk):
        x = x.transpose(1, 2)  # (batch, seq_len, text_channels) => (batch, text_channels, seq_len)

        conv_mask = sequence_mask(x.size(2), x_lengths).unsqueeze(1)

        if self.use_pre_proj:
            x = self.pre_proj(x)

        for i, layer in enumerate(self.conv_layers):
            x = layer(x, conv_mask)

        # Apply mask after convolutions
        x = x.masked_fill(conv_mask, 0)

        if self.use_cbam:
            x = self.cbam(x, conv_mask)


        # Transpose for LSTM
        x = x.transpose(1, 2)  # (b, text_channels, seq_len) -> (b, seq_len, channels)

        # LSTM Portion
        # input must be (batch, seq_len, channels)

        x = self.run_rnn(x, x_lengths)

        # Transpose dimensions for the post-convolution
        x = x.transpose(1, 2)  # (b, seq_len, channels) -> (b, channels, seq_len)

        x = self.drop(x)

        # Project using 1D convolution
        log_durations = self.out_proj(x.transpose(1,2)).transpose(1,2)

        log_durations = log_durations.masked_fill(conv_mask, 0)

        log_durations = log_durations.squeeze(1)  # (batch, 1, seq_len) => (batch, seq_len)

        return log_durations, conv_mask.squeeze(1), x.transpose(1,2)

    def run_rnn(self, x, x_lengths):
        # Pack padded sequence
        # max(x_lengths) is less than the seq_len dimension in x, often by 5 or 10
        # and the LSTM outputs a tensor of max(x_lengths) in its seq_len dimension
        # therefore, we save the original seq_len dimension for padding back later
        x_seq_len_orig = x.size(1)
        x = pack_padded_sequence(x, x_lengths.detach().cpu(),
                                 # pack_padded_sequence demands that the lengths tensor be on the CPU
                                 batch_first=True, enforce_sorted=False)
        # LSTM pass
        x, _ = self.lstm(x)
        # Unpack the sequence
        x, lens_unpacked = pad_packed_sequence(x, batch_first=True, total_length=x_seq_len_orig)  # x_lstm:  (batch, seq_len, lstm_channels)

        return x


def expand_masks(x_mask, y_mask):
    """
    Expand True=padded masks into an attention mask.
    Inputs can be different or the same
    :param x_mask: Mask of x size (batch, seq_len), where True is padded
    :param y_mask: Mask of y size (batch, seq_2_len), where True is padded
    :return: Attention mask for MultiHeadAttention
    """
    x_mask_expanded = x_mask.unsqueeze(1).unsqueeze(3)  # Shape: (batch_size, 1, mel_len, 1)
    y_mask_expanded = y_mask.unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1, duration_len)
    # Combine masks using broadcasting
    attention_mask = x_mask_expanded & y_mask_expanded  # Shape: (batch_size, 1, mel_len, duration_len)
    attention_mask = ~attention_mask  # True=padded => True=valid
    return attention_mask

# VariancePredictor but using TCNs for cheaper-than-RNN temporal dependencies.
class TemporalVariancePredictor(nn.Module):

    def __init__(self, input_channels, num_channels, kernel_size=2, dropout=0.2, cond_input_size=None):
        super(TemporalVariancePredictor, self).__init__()
        # Temporal Convolutional Network
        self.tcn = NeoTCNAttention(input_channels, num_channels, kernel_size, dropout, dropout, [0] * len(num_channels), dilation_growth="", act="relu")

        self.cond_input_size = cond_input_size
        self.input_channels = input_channels

        self.output_layer = nn.Linear(num_channels[-1], 1)

        # Duration-Spectrogram pre-conditioning
        # Conv1D => ReLU => AllAttention(duration_hidden,spec) => LayerNorm => ReLU => Dropout
        if self.cond_input_size is not None:

            self.cond_proj = nn.Conv1d(self.cond_input_size, self.input_channels,
                                       3, padding="same")

            # For some reason, this case benefits from having 4 heads instead of 2
            self.cond_attention = MultiHeadAttention(self.input_channels, 4, alibi_alpha=1.5,
                                                     start_i_increment=4, num_persistent=16)

            self.cond_act = nn.ReLU()
            self.inter_cond_drop = nn.Dropout(0.25)
            self.cond_drop = nn.Dropout(0.5)

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def make_cond_vector(self, x, y, x_mask, y_mask):
        """
        Create conditioning vector
        :param x: Input size (batch, seq_len, channels)
        :param y: Conditioning size  (batch, cond_len, cond_channels)
        :param x_mask: Mask of x size (batch, seq_len), where True is padded
        :param y_mask: Mask of y size (batch, cond_len), where True is padded

        :return: Conditioning vector, ready to add to hidden states
        """
        attention_mask = expand_masks(x_mask, y_mask)

        # (batch, seq_len, channels) <<=> (batch, channels, seq_len)
        y = self.cond_proj(
            y.transpose(1, 2)
        ).transpose(1,2).masked_fill(y_mask.unsqueeze(-1), 0)

        y = self.cond_act(y)
        y = self.inter_cond_drop(y)

        z = self.cond_attention(y, y, x, mask=attention_mask)
        z = self.inter_cond_drop(z)

        return z

    def forward(self, x, x_mask, y, y_mask):
        """
        Forward pass through the Temporal Variance Predictor

        :param x: Input hidden size (batch, seq_len, channels)
        :param x_mask: Mask of x size (batch, seq_len), where True is padded
        :param y: Conditioning hidden size (batch, cond_len, cond_channels)
        :param y_mask: Mask of y size (batch, cond_len), where True is padded

        :return: Predictions
        """

        if self.cond_input_size is not None:
            cond = self.make_cond_vector(x, y, x_mask, y_mask)
            cond = self.cond_drop(cond)
            x = x + cond

        # Pass input through the TCN
        x = x.transpose(1, 2)  # (batch, seq_len, channels) => (batch, channels, seq_len)
        x_mask = x_mask.unsqueeze(1)  # (batch, seq_len) => (batch, 1, seq_len)

        x = x.masked_fill(x_mask, 0.0)

        x = self.tcn(x, x_mask.squeeze(1), inp_channel_last=False)  # x = (batch, channels, seq_len)

        if x_mask is not None:
            x = x.masked_fill(x_mask, 0.0)

        # linear pass
        x = x.transpose(1, 2)  # x = (batch, seq_len, channels)

        # output
        x = self.output_layer(x)  # x = (batch, seq_len, 1)

        # squeeze 4 output
        x = x.squeeze(-1)  # (batch, seq_len, 1) => (batch, seq_len)

        if x_mask is not None:  # x_mask: (batch, 1, seq_len) => (batch, seq_len)
            x = x.masked_fill(x_mask.squeeze(1), 0.0)

        return x


class DecoderPrenet(nn.Module):
    """
    Tacotron2-style decoder prenet.

    This prenet applies a sequence of linear layers followed by ReLU activations
    and dropout. Dropout is applied even during inference.

    Args:
        in_dim (int): Dimensionality of the input features.
        sizes (list of int): List of output sizes for each linear layer.
        dropout (float): Dropout probability.
    """

    def __init__(self, in_dim, sizes=[256, 256], dropout=0.5):
        super(DecoderPrenet, self).__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        prev_size = in_dim
        for size in sizes:
            self.layers.append(nn.Linear(prev_size, size))
            prev_size = size

    def forward(self, x, x_mask):
        """
        Forward pass for the prenet.

        Args:
            x (torch.Tensor): Input tensor of shape (B, L, in_dim) or (B, in_dim).
            x_mask: Mask for x shape (B, L, 1)
        Returns:
            torch.Tensor: Processed tensor.
        """
        for linear in self.layers:
            x = F.relu(linear(x).masked_fill(x_mask, 0))
            # Force dropout to be active at all times by setting training=True.
            x = F.dropout(x, p=self.dropout, training=True)
        return x



class SpectrogramDecoderAR(nn.Module):
    def __init__(self, encoder_channels, mel_channels, filter_channels, depth, heads, dropout=0.1,
                 alibi_alpha=1.0, forward_expansion=4):
        super().__init__()


        self.filter_channels = filter_channels
        self.mel_channels = mel_channels

        if encoder_channels != filter_channels:
            self.y_proj = nn.Linear(encoder_channels, filter_channels)
        else:
            self.y_proj = nn.Identity()

        self.prenet = DecoderPrenet(mel_channels, [filter_channels, filter_channels])

        self.dec = TransformerDecoder(filter_channels, heads=heads, num_layers=depth,
                                      forward_expansion=forward_expansion, dropout=dropout,
                                      alibi_alpha=alibi_alpha, start_i=0, act="relugt", talking_heads=True,
                                      dynamic_alibi=True)

        self.mel_proj = nn.Linear(filter_channels, mel_channels)
        self.gate_proj = nn.Linear(filter_channels, 1) # no sigmoid, we use BCEWithLogitsLoss


    def forward(self, x, x_mask, y, y_mask):
        """
        Forward pass, decode mel spectrogram

        :param x: Melspectrogram size (batch, mel_len, mel_channels)
        :param x_mask: True=padded mask size (batch, mel_lens)
        :param y: Text hidden states size (batch, text_len, text_channels)
        :param y_mask: True=padded mask size (batch, text_lens)

        :return: Mel pred (batch, mel_len, mel_channels); gate (batch, mel_len, 1)
        """
        x_mask_b = x_mask.bool()

        lin_x_mask = x_mask_b.unsqueeze(-1) # (B, L, 1)
        conv_x_mask = x_mask_b.unsqueeze(1) # (B, 1, L)

        sa_mask = mask_to_attention_mask(x_mask_b)
        ca_mask = expand_masks(x_mask_b,
                               y_mask.bool())

        x = self.prenet(x, lin_x_mask)
        y = self.y_proj(y)

        x = self.dec(x, y, sa_mask, ca_mask, conv_x_mask)

        mel, gate = self.mel_proj(x), self.gate_proj(x)

        return mel, gate

    def infer(self, y, y_mask, max_length=1000, gate_threshold=0.5):
        """
        Autoregressive inference function.

        Args:
            y (torch.Tensor): Encoded text features of shape (B, text_len, text_channels).
            y_mask (torch.Tensor): Boolean mask of shape (B, text_len) where True indicates padded.
            max_length (int): Maximum number of decoder steps.
            gate_threshold (float): Threshold on gate prediction to stop decoding.

        Returns:
            mel_outputs (torch.Tensor): Generated mel spectrogram of shape (B, mel_len, mel_channels).
            gate_outputs (torch.Tensor): Gate predictions for each frame (B, mel_len, 1).
        """
        B = y.size(0)
        device = y.device

        # Initialize mel spectrogram with a single "go" frame: zeros.
        # Shape: (B, 1, mel_channels)
        decoder_input = torch.zeros(B, 1, self.mel_channels, device=device)

        # To store all predictions
        mel_outputs = []
        gate_outputs = []

        # A flag to indicate which examples have finished decoding.
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for t in range(max_length):
            if t == max_length - 1:
                print("Warning! Reached max decoder steps.")

            current_length = decoder_input.size(1)
            # In autoregressive inference, no positions are padded so x_mask is all False.
            x_mask = torch.zeros(B, current_length, dtype=torch.bool, device=device)

            # Run forward pass to get predictions for the current sequence.
            mel_pred, gate_pred = self.forward(decoder_input, x_mask, y, y_mask)
            # mel_pred: (B, current_length, mel_channels)
            # gate_pred: (B, current_length, 1)

            # Get the last time-step's predictions.
            last_mel = mel_pred[:, -1:, :]   # shape: (B, 1, mel_channels)
            last_gate = gate_pred[:, -1, :]    # shape: (B, 1)

            mel_outputs.append(last_mel)
            gate_outputs.append(last_gate)

            # Append the last mel frame to the decoder input for the next step.
            decoder_input = torch.cat([decoder_input, last_mel], dim=1)

            # Check which examples have predicted a gate value exceeding the threshold.
            finished = finished | (last_gate.squeeze(1) > gate_threshold)
            # If all sequences are finished, stop inference.
            if finished.all():
                break

        # Concatenate all predicted mel frames along time.
        # Note that the first frame (initial zero "go" frame) is removed.
        mel_outputs = torch.cat(mel_outputs, dim=1)  # shape: (B, mel_len, mel_channels)
        gate_outputs = torch.cat(gate_outputs, dim=1)  # shape: (B, mel_len, 1)
        return mel_outputs, gate_outputs


class SpectrogramDecoder(nn.Module):
    def __init__(self, input_size, filter_channels, mel_channels, depth, heads, kernel_sizes, dropout=0.1,
                 alibi_alpha=1.0, forward_expansion=4, emotion_size=256, speaker_channels=0, causal=True):
        super(SpectrogramDecoder, self).__init__()

        self.input_size = input_size
        self.filter_channels = filter_channels
        self.speaker_channels = speaker_channels

        if input_size != filter_channels:
            self.pre_fc = nn.Linear(input_size, filter_channels)

        self.dec = TransformerEncoder(filter_channels, heads=heads, num_layers=depth,
                                      forward_expansion=forward_expansion, dropout=dropout,
                                      alibi_alpha=alibi_alpha, start_i=4, kernel_size=kernel_sizes,
                                      act="relugt", rma_mem_dim=0, conv_att=False, multi_scale=True, talking_heads=True,
                                      dynamic_alibi=True, coarse_fine=False)

        self.mel_fc = nn.Linear(filter_channels, mel_channels)
        self.do_em_cond = False

        if self.do_em_cond:
            self.em_cond = nn.Sequential(nn.Linear(emotion_size, filter_channels),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(0.5),)

        if self.speaker_channels > 0:
            self.spk_cond = nn.Sequential(nn.Linear(speaker_channels, filter_channels),)


    # x_mask : True=exclude mask size (batch, mel_lens)
    # x: (batch, mel_lens, channels)
    def forward(self, x, x_mask, in_em, in_spk=None):
        """
        Forward pass, decode hidden states into a mel spectrogram.

        :param x: Hidden states size (batch, ,mel_len, channels)
        :param x_mask: True=padded mask size (batch, mel_lens)
        :param in_em:
        :param in_spk:

        :return: Mel spectrogram size (batch, mel_len, mel_channels)
        """
        orig_mask = x_mask.clone()

        conv_mask = x_mask.bool()
        attn_mask = mask_to_attention_mask(conv_mask)

        # (batch, mel_lens) -> (batch, 1, mel_lens)
        # True=exclude => True=Include
        x_mask = ~conv_mask
        x_mask = x_mask.float().unsqueeze(1)

        # Decoder pass
        if self.input_size != self.filter_channels:
            x = self.pre_fc(x)

        if self.speaker_channels > 0:
            x = x + self.spk_cond(in_spk)

        if self.do_em_cond:
            x = x + self.em_cond(in_em)

        x_mask = x_mask.transpose(1, 2)

        x = x * x_mask

        x = self.dec(x, attn_mask, conv_mask.unsqueeze(1))

        x = x * x_mask

        spec = self.mel_fc(x)

        spec = spec * x_mask

        return spec, x, orig_mask


def mask_to_attention_mask(mask):
    """
    Turn a bool mask into an attention mask
    :param mask: Bool sequence mask, True=padding size (batch, max_length)
    :return: Attention mask size (batch, 1, seq_len, seq_len), True=valid
    """
    attention_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
    attention_mask = attention_mask.unsqueeze(1)
    # Flip the mask, our attention uses True=valid
    attention_mask = ~attention_mask
    return attention_mask


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


class DynamicDurationPredictor(nn.Module):
    """
    DynamicDurationPredictor:

    (optionally bidirectional) TCN-Attention with increasing kernel sizes => Projection
    """

    def __init__(self, num_inputs, num_channels, kernel_sizes=[2, 2, 3], dropout=0.2, att_dropout=0.3, alibi_alpha=1.5,
                 start_i=0, heads=2, bidirectional=False, backwards_channels=[256, 256], backwards_heads=[2, 2],
                 backwards_kernel_sizes=[2, 3], emotion_size=256, speaker_channels=0, bayesian=True):
        super(DynamicDurationPredictor, self).__init__()

        self.tcn_output_channels = num_channels[-1]
        self.bidirectional = bidirectional
        self.speaker_channels = speaker_channels
        self.emotion_size = 0
        self.em_replace_channels = 16
        self.bayesian = bayesian
        self.use_mul = False

        self.tcn_attention = NeoTCNAttention(num_inputs, num_channels, kernel_sizes, dropout, att_dropout, heads,
                                             alibi_alpha=alibi_alpha, start_i_increment=start_i,
                                             bayesian=self.bayesian, integrated=True, act="taptx", conv_att="cbam")
        if self.bidirectional:
            # Widen the backwards attention bias in order to compensate for the lesser heads
            backwards_start_i = start_i * ((sum(heads) - sum(backwards_heads)) // 2)

            if backwards_start_i < 0:
                raise ValueError(
                    "DynamicDurationPredictor::Cannot have more backwards attention heads than forward heads.")

            self.backwards_tcn_attention = NeoTCNAttention(num_inputs, backwards_channels, backwards_kernel_sizes,
                                                           dropout, att_dropout,
                                                           backwards_heads,
                                                           alibi_alpha=alibi_alpha,
                                                           start_i_increment=backwards_start_i, bayesian=self.bayesian,
                                                           integrated=True, conv_att="cbam")

            self.bw_tcn_output_channels = backwards_channels[-1]

            # prevent model from overrelying on backwards features
            self.backwards_drop = nn.Dropout(0.1)
            self.fw_projection = nn.Linear(self.tcn_output_channels + self.bw_tcn_output_channels,
                                           self.tcn_output_channels)

    #    self.refiner = ConvReluNorm(self.tcn_output_channels + self.bw_tcn_output_channels, self.tcn_output_channels + self.bw_tcn_output_channels, 5, 1, "layer", dropout=0.5)
        self.linear_projection = nn.Linear(self.tcn_output_channels, 1)

        if self.use_mul:
            self.multiplier_proj = nn.Conv1d(self.tcn_output_channels, 1, 5, padding="same")



        if self.speaker_channels > 0:
            self.spk_cond = nn.Sequential(nn.Linear(self.speaker_channels, num_inputs),
                                          nn.Dropout(0.1), )

        if self.emotion_size > 0:
            self.em_proj = nn.Sequential(nn.Linear(self.emotion_size, self.em_replace_channels),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(0.1), )

    def forward(self, x, x_lengths, in_em, in_spk=None):
        """
        Forward pass through the DynamicDurationPredictor

        :param x: Encoded text size (batch, seq_len, channels)
        :param x_lengths: Int tensor of the lengths of x size (batch,)

        :return: Predicted durations size (batch, seq_len), True=padded mask of x size (batch, seq_len), duration hidden states
        """
        # Generate the appropriate mask for attention
        mask = sequence_mask(x.size(1), x_lengths)

        if self.speaker_channels > 0:
            x = x + self.spk_cond(in_spk)

        if self.emotion_size > 0:
            x_e = self.em_proj(in_em)  # (b, 1, 16)
            x[:, :, -self.em_replace_channels:] = x_e  # auto-broadcasting takes care of the seq dim

        if self.bidirectional:
            x_orig = x.clone()

        # Pass input through the TCNAttention layer
        x = self.tcn_attention(x, mask)

        if self.bidirectional:
            # Reverse input and mask for backward processing
            x_reversed = x_orig.flip(dims=[1])
            mask_reversed = mask.flip(dims=[1])

            x_reversed = self.backwards_tcn_attention(x_reversed, mask_reversed)
            x_reversed = self.backwards_drop(x_reversed)

            # flip back to align with x
            x_reversed = x_reversed.flip(dims=[1])

            # cat and project back to normal
            x = torch.cat((x, x_reversed), dim=-1)

     #       x = x + self.refiner(x.transpose(1,2), mask.unsqueeze(1)).transpose(1,2)
            x = x.masked_fill(mask.unsqueeze(-1), 0)

            x = self.fw_projection(x)
            x = x.masked_fill(mask.unsqueeze(-1), 0)

        durations = self.linear_projection(x)

        if self.use_mul:
            mul = self.multiplier_proj(
                x.transpose(1, 2)
            ).transpose(1, 2).masked_fill(mask.unsqueeze(-1), -10)

            mul = torch.sigmoid(mul).masked_fill(mask.unsqueeze(-1), 0)
            mul = mul.squeeze(-1)
        else:
            mul = 1.0


        durations = durations.squeeze(-1) * mul
        durations = durations.masked_fill(mask, 0)

        return durations, mask, x


class EmotionEncoder(nn.Module):
    def __init__(self, enc_sizes, dropout_prob=0.5):
        super().__init__()
        layers = []
        # Build layers from enc_sizes list: each pair defines a linear layer
        for i in range(1, len(enc_sizes)):
            layers.append(nn.Linear(enc_sizes[i - 1], enc_sizes[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))

        # Normally we use LayerNorm for 1D; but our final channel dim tends to be VERY small
        # so BatchNorm is better.
        layers.append(nn.BatchNorm1d(enc_sizes[-1]))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.squeeze(1) # (batch, 1, channels) = (batch, channels)
        return self.net(x)


def safe_log(tensor, epsilon=1e-6):
    """
    Computes the logarithm of the input tensor, adding epsilon for numerical stability
    and replacing NaNs with a large negative value.

    :param tensor: Input tensor for which the log is to be computed
    :param epsilon: Small value to add to the tensor for numerical stability
    :return: Tensor with the logarithm applied
    """
    tensor_with_epsilon = tensor + epsilon
    log_tensor = torch.log(tensor_with_epsilon)
    safe_log_tensor = torch.nan_to_num(log_tensor, nan=epsilon)  # Replace NaNs with -inf
    return safe_log_tensor

from rotary_embedding_torch import RotaryEmbedding

class SimpleAttention(nn.Module):
    def __init__(self, input_dim, attention_dim, use_positional_encoding=False):
        super(SimpleAttention, self).__init__()
        self.query_layer = nn.Linear(input_dim, attention_dim)
        self.key_layer = nn.Linear(input_dim, attention_dim)
        self.value_layer = nn.Linear(input_dim, attention_dim)
        self.attention_dim = attention_dim
        self.use_positional_encoding = use_positional_encoding
        self.rotary_emb = RotaryEmbedding(dim=attention_dim // 2)
            #self.positional_encoding = PositionalEncoding(attention_dim)
    def forward(self, query, key, value, mask=None):
        # Compute the query, key, and value
        query = self.query_layer(query)
        key = self.key_layer(key)
        value = self.value_layer(value)

        if self.use_positional_encoding:
            query_seq_length = query.size(1)
            key_seq_length = key.size(1)
            query = self.rotary_emb.rotate_queries_or_keys(query.unsqueeze(1)).squeeze(1)
            key = self.rotary_emb.rotate_queries_or_keys(key.unsqueeze(1)).squeeze(1)
         #   query += self.positional_encoding(query_seq_length)
          #  key += self.positional_encoding(key_seq_length)

        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.attention_dim ** 0.5)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-1e-9'))

        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Compute the context vector as the weighted sum of values
        context_vector = torch.matmul(attention_weights, value)

        return context_vector, attention_weights


class PositionalEncoding(nn.Module):
    def __init__(self, attention_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.attention_dim = attention_dim

        # Create a matrix of [seq_length, attention_dim] representing the positional encodings
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, attention_dim, 2).float() * (-math.log(10000.0) / attention_dim))

        pe = torch.zeros(max_len, attention_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, length):
        return self.pe[:length, :].transpose(0, 1)


class Aligner(nn.Module):
    def __init__(self, mel_channels, text_channels, mas_channels, heads, num_persistent=16, speaker_channels=0):
        super(Aligner, self).__init__()
        self.proj_type = "conv"
        self.attn_type = "simple" # mha is BROKEN (you could probably fix it with a LR warmup)
        self.n_heads = heads
        self.spk_cond = nn.Linear(speaker_channels, mas_channels) if speaker_channels > 0 else None

        self.mel_proj = SwiGLUConvFFN(mel_channels, mas_channels * 2, mas_channels, 5, 0.1, act="relugt")
        self.text_proj = SwiGLUConvFFN(text_channels, mas_channels * 2, mas_channels, 3, 0.1, act="relugt")

        self.num_persistent = num_persistent

        if self.attn_type == "simple":
            self.attn = SimpleAttention(mas_channels, mas_channels)
            self.n_heads = 1
        elif self.attn_type == "mha":
            self.attn = MultiHeadAttention(mas_channels, heads, 1.5, 4, num_persistent=num_persistent)

        if self.proj_type == "conv" and self.n_heads > 1:
            self.mha_proj = nn.Conv2d(heads, 1, 3, padding="same")

    def forward(self, mel_hidden_states, text_hidden_states, x_lens, y_lens, in_spk=None):

        x_mask = sequence_mask(text_hidden_states.size(1), x_lens)
        y_mask = sequence_mask(mel_hidden_states.size(1), y_lens)

        # Project mel and text hidden states to mas_channels
        text_proj = self.text_proj(text_hidden_states, x_mask.unsqueeze(1))
        mel_proj = self.mel_proj(mel_hidden_states, y_mask.unsqueeze(1))

        if self.spk_cond is not None:
            cond_spk = self.spk_cond(in_spk)
            mel_proj += cond_spk
            text_proj += cond_spk

        mha_mask = expand_masks(y_mask, x_mask)

        # Run through MultiHeadAttention
        if self.attn_type == "simple":
            # query, key, value
            _, attention_weights = self.attn(mel_proj, text_proj, text_proj, mask=mha_mask.squeeze(1))
            attention_weights = attention_weights.unsqueeze(1)
        else:
            # values, keys, queries
            _, attention_weights = self.attn(text_proj, text_proj, mel_proj, mha_mask, return_weights=True)
            attention_weights = attention_weights[:, :, :, :-self.num_persistent]

        # Average attention weights across heads
        if self.n_heads > 1:
            if self.proj_type == "conv":
                average_attention_weights = self.mha_proj(attention_weights).masked_fill(mha_mask, float("-1e4"))
                average_attention_weights = F.softmax(average_attention_weights, dim=3)
            else:
                average_attention_weights = torch.mean(attention_weights, 1, keepdim=True)
        else:
            average_attention_weights = attention_weights

        # Compute log probabilities for MAS and CTC loss
        attn_logprob = safe_log(average_attention_weights)

        # prepare for MAS
        x_mask = ~x_mask
        y_mask = ~y_mask

        x_mask = x_mask.float().unsqueeze(1)
        y_mask = y_mask.float().unsqueeze(1)

        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)

        # Apply monotonic alignment search (MAS)
        # MAS works with (batch, text, mel) ; CTC loss works with (batch, mel, text)
        with torch.no_grad():                       # Cython demands a contiguous tensor
            attn_hard = monotonic_align.maximum_path(attn_logprob.squeeze(1).transpose(1,2).contiguous(), attn_mask.squeeze(1))

        attn_hard_dur = attn_hard.sum(2)

        return average_attention_weights, attn_logprob, attn_hard, attn_hard_dur

# taken from glow-tts
class Prenet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, n_layers, p_dropout, act="relu", conv_att=False):
        super(Prenet, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout

        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        self.conv_layers.append(nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=kernel_size // 2))
        self.norm_layers.append(nn.LayerNorm(hidden_channels))

        self.mask_value = 0
        act_fn = nn.Identity()
        if act == "relu":
            act_fn = nn.ReLU()
        elif act == "aptx":
            act_fn = APTx()
            self.mask_value = -3
        elif act == "aptxs1":
            act_fn = APTxS1()
            self.mask_value = -1
        else:
            raise RuntimeError(f"Unknown activation name {act}")

        self.act_drop = nn.Sequential(act_fn, StochasticDropout(p_dropout, p_dropout, stochastic=False))

        for _ in range(n_layers - 1):
            self.conv_layers.append(nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2))
            self.norm_layers.append(nn.LayerNorm(hidden_channels))

        self.conv_att = MaskedSEBlock1D(hidden_channels) if conv_att else None

        self.proj = nn.Conv1d(hidden_channels, out_channels, 1) if hidden_channels != out_channels else nn.Identity()
        if hidden_channels != out_channels:
            self.proj.weight.data.zero_()
            self.proj.bias.data.zero_()

    def forward(self, x, x_mask):
        """
        Prenet pass
        :param x: Hidden states (batch, seq, channels)
        :param x_mask: Bool mask size (batch, 1, seq) where True is padded
        :return: x + prenet
        """
        x = x.transpose(1, 2)  # Transpose to (batch_size, in_channels, seq_length)
        x_org = x

        for i in range(self.n_layers):
            x = self.conv_layers[i](x)
            x = x.masked_fill(x_mask, 0)
            x = x.transpose(1, 2)  # Transpose to (batch_size, seq_length, hidden_channels)
            x = self.norm_layers[i](x)
            x = x.transpose(1, 2)  # Transpose back to (batch_size, hidden_channels, seq_length)
            x = x.masked_fill(x_mask, self.mask_value)
            x = self.act_drop(x)

        if self.conv_att is not None:
            x = self.conv_att(x, x_mask)

        x = self.proj(x)
        x = x_org + x

        x = x.masked_fill(x_mask, 0)
        x = x.transpose(1, 2)  # Transpose back to (batch_size, seq_length, out_channels)

        return x
