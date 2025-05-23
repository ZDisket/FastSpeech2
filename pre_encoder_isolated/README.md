# Isolated PreEncoder Model

This directory contains an isolated version of the `PreEncoder` model and its necessary dependencies.
It is designed to be a self-contained Python package.

## Structure

- `preencoder.py`: Contains the main `PreEncoder` class, `ConvBlock2D`, and `sequence_mask`.
- `attentions.py`: Contains `ResidualBlock1D` and its dependencies (`CausalConv1da`, `APTx`, `TransposeLayerNorm`, `CBAM1D`, etc.).
- `quantizer.py`: Contains the `FSQ` (Finite Scalar Quantizer) class and its helpers.
- `requirements.txt`: Lists the external Python package dependencies.
- `__init__.py`: Makes this directory usable as a Python package and exports `PreEncoder`.

## Usage

You should be able to import the `PreEncoder` class as follows, assuming `pre_encoder_isolated` is in your Python path:

```python
from pre_encoder_isolated import PreEncoder

# Example instantiation (replace with actual parameters)
# pre_encoder = PreEncoder(mel_channels=80, channels=[...], kernel_sizes=[...])
```
