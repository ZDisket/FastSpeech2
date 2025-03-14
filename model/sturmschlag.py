import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from .submodels import TextEncoder, SpectrogramDecoderAR, EmotionEncoder, NormalizedEmbedding
from text.symbols import symbols
from .submodels import sequence_mask as seq_mask2


class Sturmschlag(nn.Module):
    """ Sturmschlag:
        Transformer-TTS based model: NAR Encoder, AR Decoder"""

    def __init__(self, preprocess_config, model_config):
        super(Sturmschlag, self).__init__()
        self.model_config = model_config
        self.emotion_channels = model_config["em_enc_sizes"][-1]
        self.speaker_channels = model_config["speaker_channels"]
        self.aligner_type = model_config["aligner"]

        self.encoder = TextEncoder(
            len(symbols) + 1,
            model_config["transformer"]["encoder_hidden"],
            model_config["transformer"]["encoder_head"],
            model_config["transformer"]["encoder_layer"],
            2,
            model_config["transformer"]["encoder_dropout"],
            model_config["transformer"]["encoder_kernel_sizes"],
            1.0,
            0,
            emotion_channels=self.emotion_channels,
            speaker_channels=self.speaker_channels,
        )
        self.emotion_encoder = EmotionEncoder(model_config["em_enc_sizes"], 0.5)

        self.decoder = SpectrogramDecoderAR(model_config["transformer"]["encoder_hidden"],
                                            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
                                            model_config["transformer"]["decoder_hidden"],
                                            model_config["transformer"]["decoder_layer"],
                                            model_config["transformer"]["decoder_head"],
                                            model_config["transformer"]["decoder_dropout"],
                                            alibi_alpha=1.0,
                                            forward_expansion=4)

        self.speaker_emb = None
        if model_config["multi_speaker"]:
            with open(
                    os.path.join(
                        preprocess_config["path"]["preprocessed_path"], "speakers.json"
                    ),
                    "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(n_speaker, self.speaker_channels)


    def forward(
            self,
            speakers,
            texts,
            src_lens,
            mels,
            mel_lens,
            em_hidden=None,
    ):
        """
        Forward pass of Sturmschlag, training. For inference see .infer()
        :param speakers: Int tensor size (batch,), speaker IDs. Can pass None if not multi speaker
        :param texts: Int tensor size (batch, text_len), phoneme IDs
        :param src_lens: Int tensor size (batch,), text lengths
        :param mels: Float tensor size (batch, mel_len, mel_channels), mel spectrograms
        :param mel_lens: Int tensor size (batch,), mel lengths
        :param em_hidden: Float tensor size (batch, bert_hidden), pooled BERT embeds

        :return: Tuple mel: (batch, mel_len, mel_channels), gate: (batch, mel_len, 1)
                (text_mask: True=padded (batch, text_len), mel_mask (batch, mel_len)
        """

        # (batch, 1, spk_channels)
        spk_emb = self.speaker_emb(speakers).unsqueeze(1) if self.speaker_emb is not None else torch.zeros(1)
        encoded_emotion = self.emotion_encoder(em_hidden) if self.emotion_channels > 0 else torch.zeros(1)

        encoded_text = self.encoder(texts, src_lens, encoded_emotion, spk_emb)

        text_mask, mel_mask = seq_mask2(encoded_text.size(1), src_lens), seq_mask2(mels.size(1), mel_lens)

        mel, gate = self.decoder(mels, mel_mask, encoded_text, text_mask)

        return (
            mel,
            gate,
            text_mask,
            mel_mask
        )

    def infer(self, speakers, texts, src_lens, em_hidden=None, max_length=1000, gate_threshold=0.5):
        """
        Autoregressive inference for the Sturmschlag model.

        Args:
            speakers (torch.Tensor): Speaker IDs (or similar) for the batch.
            texts (torch.Tensor): Input text token sequences (batch, text_length).
            src_lens (torch.Tensor): Lengths of each text sequence (batch,).
            em_hidden (torch.Tensor, optional): Emotion features if available.
            max_length (int): Maximum mel spectrogram steps to generate.
            gate_threshold (float): Threshold on the gate prediction to stop decoding.

        Returns:
            mel (torch.Tensor): Generated mel spectrogram (batch, mel_length, mel_channels).
            gate (torch.Tensor): Gate predictions (batch, mel_length, 1).
        """
        device = texts.device

        # Process speaker embedding (if using multi-speaker)
        if self.speaker_emb is not None:
            spk_emb = self.speaker_emb(speakers).unsqueeze(1)  # (B, 1, speaker_channels)
        else:
            spk_emb = torch.zeros(1, device=device)

        # Process emotion encoding if applicable
        if self.emotion_channels > 0:
            encoded_emotion = self.emotion_encoder(em_hidden)
        else:
            encoded_emotion = torch.zeros(1, device=device)

        # Encode the text using the text encoder.
        encoded_text = self.encoder(texts, src_lens, encoded_emotion, spk_emb)

        # Create text mask: using a helper like seq_mask2 which produces a mask of shape (B, seq_length)
        text_mask = seq_mask2(encoded_text.size(1), src_lens)  # True indicates padded

        # Now hand off the encoded text to the decoder's inference method.
        # The decoder autoregressively generates mel spectrogram frames.
        mel, gate = self.decoder.infer(encoded_text, text_mask, max_length=max_length,
                                           gate_threshold=gate_threshold)

        return mel, gate



