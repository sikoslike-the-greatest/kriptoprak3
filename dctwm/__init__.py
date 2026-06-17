"""Robust blind DCT inter-block coefficient watermarking (Practical work 5)."""

from .watermark import (
    EmbedParams,
    arnold_scramble,
    arnold_unscramble,
    embed,
    extract,
)

__all__ = [
    "EmbedParams",
    "arnold_scramble",
    "arnold_unscramble",
    "embed",
    "extract",
]
