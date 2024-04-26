import torch
import torch.nn as nn
from .attentions import TransformerEncoder

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