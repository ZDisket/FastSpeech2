# Isolated PreEncoder Model

This directory contains an isolated version of the `PreEncoder` model and its necessary dependencies.
It is designed to be a self-contained Python package.

## Structure

- `preencoder.py`: Contains the main `PreEncoder` class, `ConvBlock2D`, and `sequence_mask`.
- `attentions.py`: Contains `ResidualBlock1D` and its dependencies (`CausalConv1da`, `APTx`, `TransposeLayerNorm`, `CBAM1D`, etc.).
- `quantizer.py`: Contains the `FSQ` (Finite Scalar Quantizer) class and its helpers.
- `discriminators.py`: Contains `MelSpectrogramPatchDiscriminator2D`, `MultiBinDiscriminator`, and their helpers.
- `losses.py`: Contains `LSGANLoss`.
- `requirements.txt`: Lists the external Python package dependencies.
- `__init__.py`: Makes this directory usable as a Python package and exports `PreEncoder`, `MelSpectrogramPatchDiscriminator2D`, `MultiBinDiscriminator`, and `LSGANLoss`.

## Usage

You should be able to import the classes as follows, assuming `pre_encoder_isolated` is in your Python path:

```python
from pre_encoder_isolated import PreEncoder, MelSpectrogramPatchDiscriminator2D, MultiBinDiscriminator, LSGANLoss

# Example instantiation (replace with actual parameters)
# pre_encoder = PreEncoder(mel_channels=80, channels=[...], kernel_sizes=[...])
# disc_2d = MelSpectrogramPatchDiscriminator2D(mel_channels=80)
# disc_multi_bin = MultiBinDiscriminator(mel_channels=80)
# lsgan_loss = LSGANLoss()
```
