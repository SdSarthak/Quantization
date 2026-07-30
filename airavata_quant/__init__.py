"""Quantized inference service for the AI4Bharat Airavata model.

The package is split so that the pure logic (configuration, request schemas,
quantization presets, benchmark statistics) can be imported and tested without
touching torch or downloading model weights.
"""

__version__ = "0.2.0"

__all__ = ["__version__"]
