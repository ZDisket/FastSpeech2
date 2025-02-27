import os

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

from text import _clean_text


def prepare_align(config):
    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]
    speaker = "LJSpeech"
    printed_ms_notice = False
    unique_speakers = set()  # Set to keep track of unique speakers
    with open(os.path.join(in_dir, "filelist.txt"), encoding="utf-8") as f:
        for line in tqdm(f):
            parts = line.strip().split("|")
            wav_name = parts[0]
            #wavs/myname.wav

            base_name = wav_name.split("/")[-1].split(".")[0]

            text = parts[1]
            text = _clean_text(text, cleaners)
            if len(parts) == 3:
                if not printed_ms_notice:
                    print("Multispeaker filelist detected. Third part will be used as speaker name")

                speaker = parts[2]
                printed_ms_notice = True
                # Add the current speaker to the set of unique speakers
                unique_speakers.add(speaker)



            wav_path = os.path.join(in_dir, wav_name)
            if os.path.exists(wav_path):
                os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
                wav, _ = librosa.load(wav_path, sr=sampling_rate)
                wav = wav / max(abs(wav)) * max_wav_value
                wavfile.write(
                    os.path.join(out_dir, speaker, "{}.wav".format(base_name)),
                    sampling_rate,
                    wav.astype(np.int16),
                )
                with open(
                    os.path.join(out_dir, speaker, "{}.lab".format(base_name)),
                    "w",encoding="utf-8"
                ) as f1:
                    f1.write(text)

    print(f"Number of speakers (unique): {len(unique_speakers)}")