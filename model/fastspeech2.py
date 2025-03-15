import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Encoder, Decoder, PostNet
from .modules import VarianceAdaptor, AlignmentEncoder, sequence_mask, binarize_attention_parallel
from utils.tools import get_mask_from_lengths
from .submodels import TextEncoder, SpectrogramDecoder, EmotionEncoder, NormalizedEmbedding, Aligner
from text.symbols import symbols
from .submodels import sequence_mask as seq_mask2
import math
import monotonic_align

class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config
        self.emotion_channels = model_config["em_enc_sizes"][-1]
        self.speaker_channels = model_config["speaker_channels"]
        self.aligner_type = model_config["aligner"]

        self.text_encoder = TextEncoder(
            len(symbols) + 1,
            model_config["transformer"]["encoder_hidden"],
            model_config["transformer"]["encoder_head"],
            model_config["transformer"]["encoder_layer"],
            2,
            model_config["transformer"]["encoder_dropout"],
            model_config["transformer"]["encoder_kernel_sizes"],
            1.5,
            3,
            emotion_channels=self.emotion_channels,
            speaker_channels=self.speaker_channels,
        )
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)

        self.decoder = SpectrogramDecoder(model_config["transformer"]["encoder_hidden"],
                                          model_config["transformer"]["decoder_hidden"],
                                          preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
                                          model_config["transformer"]["decoder_layer"],
                                          model_config["transformer"]["decoder_head"],
                                          model_config["transformer"]["decoder_kernel_sizes"],
                                          model_config["transformer"]["decoder_dropout"],
                                          alibi_alpha=1.25,
                                          emotion_size=0,
                                          speaker_channels=self.speaker_channels)

        self.emotion_encoder = EmotionEncoder(model_config["em_enc_sizes"],0.5)
        self.mel_linear = nn.Identity()

        self.postnet = PostNet(n_mel_channels=preprocess_config["preprocessing"]["mel"]["n_mel_channels"])

        if self.speaker_channels > 0 and self.aligner_type == "rad":
            self.alenc_spk_cond = nn.Sequential(nn.Linear(self.speaker_channels, model_config["transformer"]["encoder_hidden"]),
                                                nn.Dropout(0.1),)
        if self.aligner_type == "rad":
            self.aligner = AlignmentEncoder(n_mel_channels=preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
                                            n_text_channels=model_config["transformer"]["encoder_hidden"],
                                            n_att_channels=preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
                                            temperature=0.0005)

        self.mas_mean_only = True
        if self.aligner_type == "mas":
            self.mas_channels = model_config["mas_channels"]

            self.aligner = Aligner(preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
                                   model_config["transformer"]["encoder_hidden"], self.mas_channels, 1, speaker_channels=self.speaker_channels)


        self.speaker_emb = None
        if model_config["multi_speaker"]:
            with open(
                    os.path.join(
                        preprocess_config["path"]["preprocessed_path"], "speakers.json"
                    ),
                    "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = NormalizedEmbedding(
                n_speaker,
                self.speaker_channels,
            )

    @torch.jit.unused
    def run_aligner(self, text_emb, text_len, text_mask, spect, spect_len, attn_prior, spk_emb):
        text_emb = text_emb.permute(0, 2, 1)
        text_mask = text_mask.permute(0, 2, 1)  # [b, 1, mxlen] => [b, mxlen, 1]
        spect = spect.permute(0, 2, 1)  # [b, mel_len, channels] => [b, channels, mel_len]
        cond = None

        if self.speaker_channels > 0:
            # need to be B x T2 x 1 (batch, text_len, text_channels)
            cond = self.alenc_spk_cond(spk_emb)
            cond = cond.expand(-1, text_emb.size(2), -1)

        attn_soft, attn_logprob = self.aligner(
            # note: text_mask is MASK=TRUE, do NOT invert it!!!!
            spect, text_emb, mask=text_mask, attn_prior=attn_prior, conditioning=cond
        )
        attn_hard = binarize_attention_parallel(attn_soft, text_len, spect_len)
        attn_hard_dur = attn_hard.sum(2)
        return attn_soft, attn_logprob, attn_hard, attn_hard_dur

    def forward(
            self,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels=None,
            mel_lens=None,
            max_mel_len=None,
            p_targets=None,
            e_targets=None,
            em_blocks=None,
            em_hidden=None,
            em_lens=None,
            p_control=1.0,
            e_control=1.0,
            d_control=1.0,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        # (batch, 1, spk_channels)
        spk_emb = self.speaker_emb(speakers).unsqueeze(1) if self.speaker_emb is not None else torch.zeros(1)
        encoded_emotion = self.emotion_encoder(em_hidden) if self.emotion_channels > 0 else torch.zeros(1)

        output = self.text_encoder(texts, src_lens, encoded_emotion, spk_emb)
        encoded_text = output



        # src_masks -> [batch, mxlen] => [batch, 1, mxlen]
        if self.aligner_type == "rad":
            if mels is not None:
                attn_soft, attn_logprob, attn_hard, attn_hard_dur = self.run_aligner(output, src_lens,
                                                                                     src_masks.unsqueeze(1), mels,
                                                                                     mel_lens, None, spk_emb)
                total_durs = attn_hard_dur.sum(1)
            else:
                attn_soft, attn_logprob, attn_hard, attn_hard_dur, total_durs = torch.zeros(1), torch.zeros(
                    1), torch.zeros(
                    1), torch.zeros(1), None
        else:
            attn_soft, attn_logprob, attn_hard, attn_hard_dur, total_durs = torch.zeros(1), torch.zeros(
                1), torch.zeros(
                1), torch.zeros(1), None

        if self.aligner_type == "mas":
            # (self, mel_hidden_states, text_hidden_states, x_lens, y_lens):
                                                                            # text encoder shouldn't optimize for alignment; we discard the aligner during inference
                                                                        # also helps prevent NaN loss by simplifying gradient flow (maybe?)
            attn_soft, attn_logprob, attn_hard, attn_hard_dur, _ = self.aligner(mels, encoded_text.detach(), src_lens,
                                                                             mel_lens, spk_emb.detach())

            total_durs = attn_hard_dur.squeeze(1)

            attn_hard = attn_hard.transpose(1, 2).unsqueeze(1)

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            src_lens,
            mel_lens,
            encoded_emotion,
            spk_emb,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            total_durs,
            p_control,
            e_control,
            d_control,
        )

        output, dec_hid, mel_masks = self.decoder(output, mel_masks, encoded_emotion, spk_emb)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
            attn_logprob,
            attn_hard,
            attn_soft,
            total_durs,
            encoded_text,
        )

    def infer(
            self,
            speakers,
            texts,
            src_lens,
            max_src_len,
            em_blocks=None,
            em_hidden=None,
            em_lens=None,
            p_control=1.0,
            e_control=1.0,
            d_control=1.0,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = None

        spk_emb = self.speaker_emb(speakers).unsqueeze(1) if self.speaker_emb is not None else torch.zeros(1)
        encoded_emotion = self.emotion_encoder(em_hidden) if self.emotion_channels > 0 else torch.zeros(1) # (batch, hidden)

        output = self.text_encoder(texts, src_lens, encoded_emotion, spk_emb)

        attn_soft, attn_logprob, attn_hard, attn_hard_dur = torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1)

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            src_lens,
            None,
            encoded_emotion,
            spk_emb,
            mel_masks,
            None,
            None,
            None,
            None,
            p_control,
            e_control,
            d_control,
        )

        output, dec_hid, mel_masks = self.decoder(output, mel_masks, encoded_emotion, spk_emb)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
            attn_logprob,
            attn_hard,
            attn_soft,
            torch.zeros(1),
        )
