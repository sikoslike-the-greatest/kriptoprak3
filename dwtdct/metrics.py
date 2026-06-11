"""Imperceptibility and watermark-fidelity metrics."""

from __future__ import annotations

import math

import numpy as np
from skimage.metrics import structural_similarity as sk_ssim


def mse(cover: np.ndarray, stego: np.ndarray) -> float:
    a = cover.astype(np.float64)
    b = stego.astype(np.float64)
    return float(np.mean((a - b) ** 2))


def rmse(cover: np.ndarray, stego: np.ndarray) -> float:
    return math.sqrt(mse(cover, stego))


def psnr(cover: np.ndarray, stego: np.ndarray) -> float:
    err = mse(cover, stego)
    if err == 0:
        return float("inf")
    return 10.0 * math.log10((255.0 ** 2) / err)


def ssim(cover: np.ndarray, stego: np.ndarray) -> float:
    if cover.ndim == 3:
        return float(sk_ssim(cover, stego, channel_axis=-1, data_range=255))
    return float(sk_ssim(cover, stego, data_range=255))


def ber(original: np.ndarray, extracted: np.ndarray) -> float:
    """Bit-error rate between two binary watermarks."""
    a = (np.asarray(original) > 0).astype(np.uint8).ravel()
    b = (np.asarray(extracted) > 0).astype(np.uint8).ravel()
    n = min(a.size, b.size)
    if n == 0:
        return 0.0
    return float(np.mean(a[:n] != b[:n]))


def bit_accuracy(original: np.ndarray, extracted: np.ndarray) -> float:
    """Fraction of correctly recovered watermark bits."""
    return 1.0 - ber(original, extracted)


def ncc(original: np.ndarray, extracted: np.ndarray) -> float:
    """Normalized cross-correlation between two watermarks (1.0 = identical)."""
    a = (np.asarray(original) > 0).astype(np.float64).ravel()
    b = (np.asarray(extracted) > 0).astype(np.float64).ravel()
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    den = math.sqrt(float(np.sum(a * a)) * float(np.sum(b * b)))
    if den == 0:
        return 0.0
    return float(np.sum(a * b)) / den
