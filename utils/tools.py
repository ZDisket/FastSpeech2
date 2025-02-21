import os
import json

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
from scipy.io import wavfile
from matplotlib import pyplot as plt


matplotlib.use("Agg")


def compute_phoneme_level_features_optimized(duration, mel_pitch):
    """
    Convert mel-level features (mostly pitch/energy) into phoneme-level using durations to average.
    Note that masking is not taken into account, apply it in your loss function

    :param duration: Predicted int/long durations tensor shape (batch, text_len)
    :param mel_pitch: Predicted mel-level float features tensor shape (batch, mel_len)
    :return: Phoneme-level features float tensor shape (batch, text_len)
    """
    batch_size, phoneme_len = duration.shape

    # Compute the cumulative sum of durations to get the end indices of each phoneme
    cumsum_durations = torch.cumsum(duration, dim=1)
    start_indices = torch.cat(
        [torch.zeros((batch_size, 1), device=duration.device, dtype=duration.dtype), cumsum_durations[:, :-1]], dim=1)

    # Create a mask to select pitch values for each phoneme
    mask = (torch.arange(mel_pitch.size(1), device=mel_pitch.device).view(1, 1, -1) < cumsum_durations.unsqueeze(2)) & (
                torch.arange(mel_pitch.size(1), device=mel_pitch.device).view(1, 1, -1) >= start_indices.unsqueeze(2))

    # Compute phoneme-level pitch
    masked_pitch = mel_pitch.unsqueeze(1) * mask.float()
    sum_pitch = masked_pitch.sum(dim=2)
    count_pitch = mask.sum(dim=2).float()

    phoneme_pitch = sum_pitch / (count_pitch + 1e-9)  # Add a small value to avoid division by zero

    return phoneme_pitch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from text import text_to_sequence, sequence_to_text, cleaned_text_to_sequence


def to_device(data, device):
    if len(data) == 11 + 3:
        (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            energies,
            emotion_blocks,
            emotion_hiddens,
            emotion_lens,
        ) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)
        mels = torch.from_numpy(mels).float().to(device)
        mel_lens = torch.from_numpy(mel_lens).to(device)
        pitches = torch.from_numpy(pitches).float().to(device)
        energies = torch.from_numpy(energies).to(device)
        emotion_blocks = torch.from_numpy(emotion_blocks).to(device)
        emotion_hiddens = torch.from_numpy(emotion_hiddens).to(device)
        emotion_lens = torch.from_numpy(emotion_lens).to(device)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            energies,
            emotion_blocks,
            emotion_hiddens,
            emotion_lens,
        )

    if len(data) == 6:
        (ids, raw_texts, speakers, texts, src_lens, max_src_len) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)

        return (ids, raw_texts, speakers, texts, src_lens, max_src_len)


import matplotlib.pyplot as plt
import torch


def log_attention_maps(logger, attention_tensor, widths, heights, step, tag_prefix=""):
    """
    Log each attention map from the attention_tensor to TensorBoard, with given widths and heights.

    :param logger: The TensorBoard SummaryWriter instance.
    :param attention_tensor: A tensor of shape (batch, max_w, max_h) with attention maps.
    :param widths: A 1D tensor or array of shape (batch) with widths for each attention map.
    :param heights: A 1D tensor or array of shape (batch) with heights for each attention map.
    :param step: The current step in training for logging.
    :param tag_prefix: Prefix for the tag to categorize the logs in TensorBoard.
    """
    batch_size = attention_tensor.size(0)

    for i in range(batch_size):
        fig_width = 5
        fig_height = fig_width * (heights[i] / widths[i])  # Adjust the height based on the aspect ratio
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))  # Create a matplotlib figure and axes.
        # Slice the attention tensor to the specified width and height
        attention_map = attention_tensor[i, :heights[i], :widths[i]].cpu().numpy()
        im = ax.imshow(attention_map, cmap='viridis', interpolation='nearest')
        # Adjust colorbar size by changing fraction and pad
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.title(f'Attention Map {i + 1}')
        plt.xlabel('Decoder Timestep')
        plt.ylabel('Encoder Timestep')
        plt.close(fig)  # Close the figure to prevent it from displaying

        # Log the figure to TensorBoard
        logger.add_figure(f"{tag_prefix}/Soft Attention {i + 1}", fig, global_step=step)


def log(
    logger, step=None, losses=None, fig=None, audio=None, sampling_rate=22050, tag=""
):
    if losses is not None:
        logger.add_scalar("Loss/total_loss", losses[0], step)
        logger.add_scalar("Loss/mel_loss", losses[1], step)
        logger.add_scalar("Loss/mel_postnet_loss", losses[2], step)
        logger.add_scalar("Loss/pitch_loss", losses[3], step)
        logger.add_scalar("Loss/energy_loss", losses[4], step)
        logger.add_scalar("Loss/duration_loss", losses[5], step)
        logger.add_scalar("Loss/attention_loss", losses[6], step)
        logger.add_scalar("Loss/duration_temporal_loss", losses[7], step)
        logger.add_scalar("Loss/total_temporal_loss", losses[8], step)
        logger.add_scalar("Loss/duration_kl_loss", losses[9], step)
        logger.add_scalar("Loss/pitch_energy_kl_loss", losses[10], step)
        if len(losses) > 11:
            logger.add_scalar("Loss/dur_discriminator_loss", losses[11], step)
            logger.add_scalar("Loss/dur_gan_loss", losses[12], step)

    if fig is not None:
        logger.add_figure(tag, fig)

    if audio is not None:
        logger.add_audio(
            tag,
            audio / max(abs(audio)),
            sample_rate=sampling_rate,
            global_step=step,
        )


def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask


def expand(values, durations):
    out = list()
    for value, d in zip(values, durations):
        out += [value] * max(0, int(d))
    return np.array(out)


def synth_one_sample(targets, predictions, vocoder, model_config, preprocess_config):

    basename = targets[0][0]
    src_len = predictions[8][0].item()
    mel_len = predictions[9][0].item()
    mel_target = targets[6][0, :mel_len].detach().transpose(0, 1)
    mel_prediction = predictions[1][0, :mel_len].detach().transpose(0, 1)
    attn_soft = predictions[12].squeeze(1)
    if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
        pitch = targets[9][0, :src_len].detach().cpu().numpy()
    else:
        pitch = targets[9][0, :mel_len].detach().cpu().numpy()
    if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
        energy = targets[10][0, :src_len].detach().cpu().numpy()
    else:
        energy = targets[10][0, :mel_len].detach().cpu().numpy()

    with open(
        os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
    ) as f:
        stats = json.load(f)
        stats = stats["pitch"] + stats["energy"][:2]

    fig = plot_mel(
        [
            (mel_prediction.cpu().numpy(), pitch, energy),
            (mel_target.cpu().numpy(), pitch, energy),
        ],
        stats,
        ["Synthetized Spectrogram", "Ground-Truth Spectrogram"],
    )

    if vocoder is not None:
        from .model import vocoder_infer

        wav_reconstruction = vocoder_infer(
            mel_target.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
        wav_prediction = vocoder_infer(
            mel_prediction.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
    else:
        wav_reconstruction = wav_prediction = None

    return fig, wav_reconstruction, wav_prediction, basename, attn_soft


def synth_samples(targets, predictions, vocoder, model_config, preprocess_config, path):

    basenames = targets[0]
    for i in range(len(predictions[0])):
        basename = basenames[i]
        src_len = predictions[8][i].item()
        mel_len = predictions[9][i].item()
        mel_prediction = predictions[1][i, :mel_len].detach().transpose(0, 1)
        duration = predictions[5][i, :src_len].detach().cpu().numpy()
        if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
            pitch = predictions[2][i, :src_len].detach().cpu().numpy()
            pitch = expand(pitch, duration)
        else:
            pitch = predictions[2][i, :mel_len].detach().cpu().numpy()
        if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
            energy = predictions[3][i, :src_len].detach().cpu().numpy()
            energy = expand(energy, duration)
        else:
            energy = predictions[3][i, :mel_len].detach().cpu().numpy()

        with open(
            os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
        ) as f:
            stats = json.load(f)
            stats = stats["pitch"] + stats["energy"][:2]

        fig = plot_mel(
            [
                (mel_prediction.cpu().numpy(), pitch, energy),
            ],
            stats,
            ["Synthetized Spectrogram"],
        )
        plt.savefig(os.path.join(path, "{}.png".format(basename)))
        plt.close()

    from .model import vocoder_infer

    mel_predictions = predictions[1].transpose(1, 2)
    lengths = predictions[9] * preprocess_config["preprocessing"]["stft"]["hop_length"]
    wav_predictions = vocoder_infer(
        mel_predictions, vocoder, model_config, preprocess_config, lengths=lengths
    )

    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    for wav, basename in zip(wav_predictions, basenames):
        wavfile.write(os.path.join(path, "{}.wav".format(basename)), sampling_rate, wav)


def plot_mel(data, stats, titles):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]
    pitch_min, pitch_max, pitch_mean, pitch_std, energy_min, energy_max = stats
    pitch_min = pitch_min * pitch_std + pitch_mean
    pitch_max = pitch_max * pitch_std + pitch_mean

    def add_axis(fig, old_ax):
        ax = fig.add_axes(old_ax.get_position(), anchor="W")
        ax.set_facecolor("None")
        return ax

    for i in range(len(data)):
        mel, pitch, energy = data[i]
        pitch = pitch * pitch_std + pitch_mean
        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")

        ax1 = add_axis(fig, axes[i][0])
        ax1.plot(pitch, color="tomato")
        ax1.set_xlim(0, mel.shape[1])
        ax1.set_ylim(0, pitch_max)
        ax1.set_ylabel("F0", color="tomato")
        ax1.tick_params(
            labelsize="x-small", colors="tomato", bottom=False, labelbottom=False
        )

        ax2 = add_axis(fig, axes[i][0])
        ax2.plot(energy, color="darkviolet")
        ax2.set_xlim(0, mel.shape[1])
        ax2.set_ylim(energy_min, energy_max)
        ax2.set_ylabel("Energy", color="darkviolet")
        ax2.yaxis.set_label_position("right")
        ax2.tick_params(
            labelsize="x-small",
            colors="darkviolet",
            bottom=False,
            labelbottom=False,
            left=False,
            labelleft=False,
            right=True,
            labelright=True,
        )

    return fig


def pad_zephyr_outputs(hidden_blocks_list):
    """
    Zero-pads lists of numpy arrays along the sequence length dimension and returns their lengths.

    Args:
        hidden_blocks_list (list of np.ndarray): List of arrays with shape (1, 4, seq_len, hidden_dim).

    Returns:
        tuple: A numpy array, zero-padded along the sequence length dimension, and an array of lengths.
    """
    # Determine the maximum sequence length in the list
    max_seq_len = max(arr.shape[2] for arr in hidden_blocks_list)

    # Get lengths of each sequence
    lengths = np.array([arr.shape[2] for arr in hidden_blocks_list])

    # Pad hidden_blocks_list
    padded_hidden_blocks = []
    for arr in hidden_blocks_list:
        pad_width = ((0, 0), (0, 0), (0, max_seq_len - arr.shape[2]), (0, 0))
        padded_arr = np.pad(arr, pad_width, mode='constant', constant_values=0)
        padded_hidden_blocks.append(padded_arr)

    # Convert list to numpy array
    padded_hidden_blocks = np.concatenate(padded_hidden_blocks, axis=0)

    return padded_hidden_blocks, lengths


def pad_bert_outputs(hidden_blocks_list):
    """
    Zero-pads a list of numpy arrays along the sequence length dimension.

    Args:
        hidden_blocks_list (list of np.ndarray): Each array has shape (1, seq_len, hidden_dim).

    Returns:
        tuple: A numpy array of shape (batch_size, max_seq_len, hidden_dim) with zero-padding,
               and a numpy array of the original sequence lengths.
    """
    # Calculate the sequence lengths and maximum sequence length
    seq_lengths = np.array([arr.shape[1] for arr in hidden_blocks_list])
    max_seq_len = seq_lengths.max()

    # Pad each array along the sequence dimension and concatenate along the batch axis
    padded_arrays = [
        np.pad(arr, pad_width=((0, 0), (0, max_seq_len - arr.shape[1]), (0, 0)), mode='constant')
        for arr in hidden_blocks_list
    ]
    padded_hidden_blocks = np.concatenate(padded_arrays, axis=0)

    return padded_hidden_blocks, seq_lengths


def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded


from text import _clean_text
def preproc_text(in_txt, cleaners = ["english_cleaners2"]):
    txt_ipa = _clean_text(in_txt, cleaners)
    txt_arr = np.array(cleaned_text_to_sequence(txt_ipa, cleaners))
    return txt_arr


def fs2_infer(inmodel, text, in_blocks, in_hid):
    src_len = torch.from_numpy(np.array([text.shape[1]])).to(device)
    text = torch.IntTensor(text).to(device)
    em_len = torch.from_numpy(np.array([in_hid.shape[1]])).to(device)
    in_blocks = torch.from_numpy(in_blocks).to(device)
    in_hid = torch.from_numpy(in_hid).to(device).unsqueeze(0)
    speakers = torch.IntTensor([0])

    predictions = inmodel.infer(speakers, text, src_len, src_len[0], in_blocks, in_hid, em_len)
    mel, mel_postnet = predictions[0], predictions[1]

    mel_torch = mel.transpose(1, 2).detach()
    mel_postnet_torch = mel_postnet.transpose(1, 2).detach()
    mel = mel[0].cpu().transpose(0, 1).detach()
    mel_postnet = mel_postnet[0].cpu().transpose(0, 1).detach()

    return mel, mel_postnet, mel_torch, mel_postnet_torch


def test_one_fs2(inmodel, invocoder, in_txt, in_blocks, in_hid):
    with torch.no_grad():
        txt = preproc_text(in_txt)
        # [text_len] => [1, text_len]

        txt = np.expand_dims(txt, 0)
        try:
            mel, mel_postnet, mel_torch, mel_postnet_torch = fs2_infer(inmodel, txt, in_blocks, in_hid)
        except RuntimeError:
            print(f"Error inferring {txt}")
            return None



        audio = invocoder.infer(mel_postnet_torch.to(device))
    return audio