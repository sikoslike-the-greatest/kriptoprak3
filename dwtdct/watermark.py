"""Hybrid DCT + DWT robust watermarking (algorithm from Abdulrahman & Ozturk, 2019).

Pipeline (embedding):

1. The grayscale cover image (N x N) is transformed with a full-frame 2-D DCT.
2. A single level of the Haar DWT is applied to the DCT-coefficient array,
   producing four sub-bands LL, HL, LH, HH (each N/2 x N/2).
3. The binary watermark (m x m) is scrambled with the Arnold cat map and then
   transformed with a 2-D DCT.
4. The watermark DCT array is split into four equal m/2 x m/2 blocks; each block
   is added to the top-left corner of one sub-band with scaling factor ``alpha``.
5. The inverse DWT followed by the inverse DCT yields the watermarked image.

Extraction is *non-blind*: it requires the original cover image. The four
watermark DCT blocks are recovered from the difference of the sub-bands of the
cover and the stego image, recombined, inverse-DCT'd, and finally unscrambled
with the inverse Arnold transform.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy.fft import dctn, idctn

SQRT2 = np.sqrt(2.0)


# --------------------------------------------------------------------------- #
# 1-level separable Haar DWT (orthonormal)                                    #
# --------------------------------------------------------------------------- #

def _dwt_axis(x: np.ndarray, axis: int) -> np.ndarray:
    x = np.moveaxis(np.asarray(x, dtype=np.float64), axis, 0)
    even, odd = x[0::2], x[1::2]
    low = (even + odd) / SQRT2
    high = (even - odd) / SQRT2
    return np.moveaxis(np.concatenate([low, high], axis=0), 0, axis)


def _idwt_axis(low: np.ndarray, high: np.ndarray, axis: int) -> np.ndarray:
    low = np.moveaxis(low, axis, 0)
    high = np.moveaxis(high, axis, 0)
    even = (low + high) / SQRT2
    odd = (low - high) / SQRT2
    out = np.empty((low.shape[0] * 2,) + low.shape[1:], dtype=np.float64)
    out[0::2] = even
    out[1::2] = odd
    return np.moveaxis(out, 0, axis)


def haar_dwt2(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One level of the 2-D Haar transform. Returns (LL, HL, LH, HH)."""
    t = _dwt_axis(_dwt_axis(x, 1), 0)
    h0, h1 = t.shape[0] // 2, t.shape[1] // 2
    return t[:h0, :h1], t[:h0, h1:], t[h0:, :h1], t[h0:, h1:]


def haar_idwt2(ll: np.ndarray, hl: np.ndarray, lh: np.ndarray, hh: np.ndarray) -> np.ndarray:
    """Inverse of :func:`haar_dwt2`."""
    top = np.concatenate([ll, hl], axis=1)
    bottom = np.concatenate([lh, hh], axis=1)
    t = np.concatenate([top, bottom], axis=0)
    n0 = t.shape[0]
    t = _idwt_axis(t[: n0 // 2], t[n0 // 2 :], 0)
    n1 = t.shape[1]
    t = _idwt_axis(t[:, : n1 // 2], t[:, n1 // 2 :], 1)
    return t


# --------------------------------------------------------------------------- #
# Arnold cat-map scrambling                                                   #
# --------------------------------------------------------------------------- #

def arnold_scramble(img: np.ndarray, iters: int) -> np.ndarray:
    """Apply the Arnold transform ``iters`` times to a square matrix."""
    m = img.shape[0]
    if img.shape[0] != img.shape[1]:
        raise ValueError("Arnold transform requires a square matrix")
    xs, ys = np.meshgrid(np.arange(m), np.arange(m), indexing="ij")
    out = img.copy()
    for _ in range(iters):
        nx = (xs + ys) % m
        ny = (xs + 2 * ys) % m
        new = np.empty_like(out)
        new[nx, ny] = out[xs, ys]
        out = new
    return out


def arnold_unscramble(img: np.ndarray, iters: int) -> np.ndarray:
    """Inverse Arnold transform (``iters`` iterations)."""
    m = img.shape[0]
    if img.shape[0] != img.shape[1]:
        raise ValueError("Arnold transform requires a square matrix")
    xs, ys = np.meshgrid(np.arange(m), np.arange(m), indexing="ij")
    out = img.copy()
    for _ in range(iters):
        px = (2 * xs - ys) % m
        py = (-xs + ys) % m
        new = np.empty_like(out)
        new[px, py] = out[xs, ys]
        out = new
    return out


# --------------------------------------------------------------------------- #
# Embedding parameters                                                        #
# --------------------------------------------------------------------------- #

@dataclass
class EmbedParams:
    """Side information shared between embedding and extraction."""

    cover_shape: tuple[int, int]
    wm_size: int
    alpha: float
    arnold_iters: int

    def to_dict(self) -> dict:
        d = asdict(self)
        d["cover_shape"] = list(self.cover_shape)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "EmbedParams":
        return cls(
            cover_shape=tuple(d["cover_shape"]),
            wm_size=int(d["wm_size"]),
            alpha=float(d["alpha"]),
            arnold_iters=int(d["arnold_iters"]),
        )


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _split_quadrants(a: np.ndarray) -> list[np.ndarray]:
    h, w = a.shape[0] // 2, a.shape[1] // 2
    return [a[:h, :w], a[:h, w:], a[h:, :w], a[h:, w:]]


def _join_quadrants(blocks: list[np.ndarray]) -> np.ndarray:
    top = np.concatenate([blocks[0], blocks[1]], axis=1)
    bottom = np.concatenate([blocks[2], blocks[3]], axis=1)
    return np.concatenate([top, bottom], axis=0)


def _to_gray_float(image: np.ndarray) -> np.ndarray:
    a = np.asarray(image)
    if a.ndim == 3:
        a = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2])
    return a.astype(np.float64)


# --------------------------------------------------------------------------- #
# Embedding / extraction                                                      #
# --------------------------------------------------------------------------- #

def embed(
    cover: np.ndarray,
    watermark: np.ndarray,
    alpha: float = 20.0,
    arnold_iters: int = 10,
) -> tuple[np.ndarray, EmbedParams]:
    """Embed a binary watermark into a grayscale cover image.

    Parameters
    ----------
    cover : ndarray
        Square grayscale (or RGB, converted to luminance) image, uint8.
    watermark : ndarray
        Square binary image (values 0/1 or 0/255). Its side ``m`` must be even
        and not larger than the cover side ``N``.
    alpha : float
        Embedding strength (scaling coefficient).
    arnold_iters : int
        Number of Arnold scrambling iterations applied to the watermark.

    Returns
    -------
    (stego_uint8, params)
    """
    gray = _to_gray_float(cover)
    n = gray.shape[0]
    if gray.shape[0] != gray.shape[1]:
        raise ValueError("cover image must be square")

    wm = (np.asarray(watermark) > 0).astype(np.float64)
    m = wm.shape[0]
    if wm.shape[0] != wm.shape[1]:
        raise ValueError("watermark must be square")
    if m % 2 != 0 or m > n:
        raise ValueError("watermark side must be even and <= cover side")

    # cover: full-frame DCT, then one level of Haar DWT
    cover_dct = dctn(gray, norm="ortho")
    subbands = list(haar_dwt2(cover_dct))

    # watermark: Arnold scramble, then DCT, then split into 4 blocks
    wm_scr = arnold_scramble(wm, arnold_iters)
    wm_dct = dctn(wm_scr, norm="ortho")
    wm_blocks = _split_quadrants(wm_dct)
    half = m // 2

    # additive embedding into the top-left corner of each sub-band
    for sb, blk in zip(subbands, wm_blocks):
        sb[:half, :half] += alpha * blk

    stego_dct = haar_idwt2(*subbands)
    stego = idctn(stego_dct, norm="ortho")
    stego = np.clip(np.rint(stego), 0, 255).astype(np.uint8)

    params = EmbedParams(
        cover_shape=(n, n),
        wm_size=m,
        alpha=float(alpha),
        arnold_iters=int(arnold_iters),
    )
    return stego, params


def extract(
    stego: np.ndarray,
    cover: np.ndarray,
    params: EmbedParams,
) -> np.ndarray:
    """Recover the binary watermark (non-blind: requires the original cover)."""
    gray_stego = _to_gray_float(stego)
    gray_cover = _to_gray_float(cover)

    cover_sub = list(haar_dwt2(dctn(gray_cover, norm="ortho")))
    stego_sub = list(haar_dwt2(dctn(gray_stego, norm="ortho")))

    m = params.wm_size
    half = m // 2
    blocks = []
    for sc, ss in zip(cover_sub, stego_sub):
        blocks.append((ss[:half, :half] - sc[:half, :half]) / params.alpha)

    wm_dct = _join_quadrants(blocks)
    wm_scr = idctn(wm_dct, norm="ortho")
    wm = arnold_unscramble(wm_scr, params.arnold_iters)
    return (wm > 0.5).astype(np.uint8)
