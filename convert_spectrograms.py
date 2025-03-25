import os
import librosa
import numpy as np
from tqdm import tqdm

# !/usr/bin/env python3
import argparse
import os


# Assume that Audio.tools.get_mel_from_wav and Audio.stft.TacotronSTFT are available elsewhere.
# For example:
import audio as Audio
from audio import stft
from audio.tools import get_mel_from_wav
from audio.stft import TacotronSTFT  # Assumes this is defined elsewhere

class MelSpectrogramConverter:
    """
    Recursively converts audio files in an input folder to mel spectrograms and saves them as .npy files,
    preserving the folder structure in the output folder.

    Attributes:
        input_folder (str): Path to the root directory containing .wav files.
        output_folder (str): Path to the root directory where the .npy spectrograms will be saved.
        stft: An instance of a STFT object (e.g., TacotronSTFT) used to compute the mel spectrogram.
        sampling_rate (int): Sampling rate to use when loading audio files.
    """

    def __init__(self, input_folder, output_folder, stft, sampling_rate):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.stft = stft
        self.sampling_rate = sampling_rate

        # Create the output folder if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)

    def process_file(self, file_path, output_dir):
        """
        Process a single .wav file: load, compute mel spectrogram, and save as .npy in the given output directory.

        Args:
            file_path (str): The full path to the .wav file.
            output_dir (str): The directory where the processed file should be saved.

        Returns:
            bool: True if processing succeeded, False otherwise.
        """
        try:
            # Load the audio file at the desired sampling rate
            wav, _ = librosa.load(file_path, sr=self.sampling_rate)

            # Optional: Check duration and skip files that are too short (or too long, if desired)
            duration = len(wav) / self.sampling_rate
            if duration < 1.0:
                print(f"Skipping {file_path}: duration too short ({duration:.2f} s)")
                return False

            # Compute mel spectrogram (energy output is ignored here)
            mel_spectrogram, _ = Audio.tools.get_mel_from_wav(wav, self.stft)

            # Transpose if needed to match expected dimensions
            mel_spectrogram = mel_spectrogram.T

            # Construct output file name and path
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_file = os.path.join(output_dir, f"{base_name}_mel.npy")

            # Save the mel spectrogram as a .npy file
            np.save(output_file, mel_spectrogram)
            return True

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return False

    def convert_folder(self):
        """
        Recursively searches for .wav files in the input folder, processes them, and saves their mel spectrograms
        while mimicking the input folder structure in the output folder.
        """
        # Use os.walk to traverse directories recursively
        for root, dirs, files in os.walk(self.input_folder):
            # Compute relative path from the root of input_folder
            rel_path = os.path.relpath(root, self.input_folder)
            # Determine corresponding output subfolder and create it if it doesn't exist
            output_subfolder = os.path.join(self.output_folder, rel_path)
            os.makedirs(output_subfolder, exist_ok=True)

            # Process each .wav file in the current directory
            wav_files = [f for f in files if f.lower().endswith(".wav")]
            for wav_file in tqdm(wav_files, desc=f"Processing {rel_path}"):
                file_path = os.path.join(root, wav_file)
                self.process_file(file_path, output_subfolder)


def main():
    parser = argparse.ArgumentParser(
        description="Recursively convert WAV files to mel spectrogram .npy files while preserving folder structure."
    )
    parser.add_argument("--input_folder", type=str, required=True,
                        help="Path to the root directory containing WAV files.")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Path to the root directory where output .npy files will be saved.")
    parser.add_argument("--sampling_rate", type=int, default=22050,
                        help="Sampling rate for loading audio files. Default: 22050")
    parser.add_argument("--filter_length", type=int, default=1024,
                        help="Filter length for the STFT. Default: 1024")
    parser.add_argument("--hop_length", type=int, default=256,
                        help="Hop length for the STFT. Default: 256")
    parser.add_argument("--win_length", type=int, default=1024,
                        help="Window length for the STFT. Default: 1024")
    parser.add_argument("--n_mel_channels", type=int, default=80,
                        help="Number of mel channels. Default: 80")
    parser.add_argument("--mel_fmin", type=float, default=0.0,
                        help="Minimum frequency for the mel filter. Default: 0.0")
    parser.add_argument("--mel_fmax", type=float, default=8000.0,
                        help="Maximum frequency for the mel filter. Default: 8000.0")

    args = parser.parse_args()

    # Instantiate the TacotronSTFT object with the provided parameters.
    stft = TacotronSTFT(
        filter_length=args.filter_length,
        hop_length=args.hop_length,
        win_length=args.win_length,
        n_mel_channels=args.n_mel_channels,
        sampling_rate=args.sampling_rate,
        mel_fmin=args.mel_fmin,
        mel_fmax=args.mel_fmax
    )

    # Create the mel spectrogram converter.
    converter = MelSpectrogramConverter(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        stft=stft,
        sampling_rate=args.sampling_rate
    )

    # Execute the conversion.
    converter.convert_folder()


if __name__ == "__main__":
    main()