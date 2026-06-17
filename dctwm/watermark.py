"""Robust blind image watermarking in the DCT domain using inter-block
coefficient correlation (algorithm of Ko, Huang, Horng & Wang, 2020).

Pipeline (embedding):

1. Pixel values are shifted to the range ``[-128, 127]`` (each pixel minus 128)
   and the image is split into non-overlapping ``8 x 8`` blocks; a 2-D DCT is
   applied to every block.
2. One bit of the (Arnold-scrambled) binary watermark is embedded into a single
   predefined mid-frequency coefficient of each *carrier* block. The bit is
   encoded by the difference ``delta`` between the chosen coefficient of the
   carrier block and the same coefficient of an adjacent *reference* block.
3. The difference axis is partitioned into regions of width ``T``; even regions
   carry bit 0, odd regions carry bit 1. The carrier coefficient is offset in
   steps proportional to an adaptive, block-dependent strength ``S`` (derived
   from the block DC value and the median of its first nine AC coefficients)
   until ``delta`` enters the region that corresponds to the watermark bit, with
   a guard ``t`` to the region boundary.
4. The inverse DCT is applied block-wise, values are rounded and ``128`` is
   added back, yielding the watermarked image.

Extraction is **blind**: it needs only the watermarked image and the embedding
parameters. The per-block difference ``delta`` is recomputed and the bit is read
from the region it falls into; the inverse Arnold transform restores the mark.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np
from scipy.fft import dctn, idctn

BLOCK = 8


# --------------------------------------------------------------------------- #
# Zig-zag ordering of an 8x8 block                                            #
# --------------------------------------------------------------------------- #

def _zigzag_order() -> list[tuple[int, int]]:
    """Coordinates of the 8x8 coefficients in zig-zag (JPEG) order."""
    coords: list[tuple[int, int]] = []
    for s in range(2 * BLOCK - 1):
        if s % 2 == 0:
            r = min(s, BLOCK - 1)
            c = s - r
            while r >= 0 and c < BLOCK:
                coords.append((r, c))
                r -= 1
                c += 1
        else:
            c = min(s, BLOCK - 1)
            r = s - c
            while c >= 0 and r < BLOCK:
                coords.append((r, c))
                r += 1
                c -= 1
    return coords


ZIGZAG = _zigzag_order()
# first nine AC coefficients (skip the DC term at index 0)
AC9 = ZIGZAG[1:10]
AC9_R = np.array([p[0] for p in AC9])
AC9_C = np.array([p[1] for p in AC9])
# default mid-frequency carrier coefficient (zig-zag index 18)
DEFAULT_COEF = (3, 2)


# --------------------------------------------------------------------------- #
# Arnold cat-map scrambling (square matrices)                                 #
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
    """Side information shared between embedding and (blind) extraction."""

    cover_shape: tuple[int, int]
    wm_size: int
    T: float
    t: float
    alpha: float
    arnold_iters: int
    coef: tuple[int, int]
    direction: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["cover_shape"] = list(self.cover_shape)
        d["coef"] = list(self.coef)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "EmbedParams":
        return cls(
            cover_shape=tuple(d["cover_shape"]),
            wm_size=int(d["wm_size"]),
            T=float(d["T"]),
            t=float(d["t"]),
            alpha=float(d["alpha"]),
            arnold_iters=int(d["arnold_iters"]),
            coef=tuple(d["coef"]),
            direction=str(d.get("direction", "LR")),
        )


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _to_gray_float(image: np.ndarray) -> np.ndarray:
    a = np.asarray(image)
    if a.ndim == 3:
        a = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    return a.astype(np.float64)


def _image_to_blocks(gray: np.ndarray) -> np.ndarray:
    """Return a (nb, nb, 8, 8) array of block-wise DCT coefficients."""
    n = gray.shape[0]
    nb = n // BLOCK
    blocks = (
        gray[: nb * BLOCK, : nb * BLOCK]
        .reshape(nb, BLOCK, nb, BLOCK)
        .transpose(0, 2, 1, 3)
    )
    return dctn(blocks - 128.0, axes=(-2, -1), norm="ortho")


def _blocks_to_image(coeffs: np.ndarray) -> np.ndarray:
    nb = coeffs.shape[0]
    spatial = idctn(coeffs, axes=(-2, -1), norm="ortho") + 128.0
    img = spatial.transpose(0, 2, 1, 3).reshape(nb * BLOCK, nb * BLOCK)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def _strength(dc: float, med: float, alpha: float) -> float:
    """Adaptive per-block modification strength ``S`` (step size)."""
    if abs(dc) > 1000.0 or abs(dc) < 1.0:
        s = alpha * abs(med)
    else:
        s = alpha * abs((dc - med) / dc)
    return max(1.0, s)


def _carrier_indices(nb: int, m: int, direction: str):
    """Yield ``(carrier, reference)`` block indices for ``m x m`` carriers.

    The block grid is ``nb x nb``; the last block row/column acts as a pure
    reference so that blind extraction is exact. Carriers are processed in an
    order that keeps every reference final at the moment a carrier is embedded.
    """
    if direction == "LR":
        for bu in range(m):
            for bv in range(m - 1, -1, -1):  # right-to-left
                yield (bu, bv), (bu, bv + 1)
    elif direction == "UD":
        for bv in range(m):
            for bu in range(m - 1, -1, -1):  # bottom-to-top
                yield (bu, bv), (bu + 1, bv)
    else:
        raise ValueError("direction must be 'LR' or 'UD'")


def _target_center(delta: float, bit: int, T: float) -> float:
    """Centre of the region (multiple of ``T``) of correct parity nearest delta."""
    base = int(round(delta / T))
    best = None
    for k in (base - 2, base - 1, base, base + 1, base + 2):
        if k % 2 == bit:
            if best is None or abs(delta - k * T) < abs(delta - best * T):
                best = k
    return best * T


# --------------------------------------------------------------------------- #
# Embedding / extraction                                                      #
# --------------------------------------------------------------------------- #

def embed(
    cover: np.ndarray,
    watermark: np.ndarray,
    T: float = 60.0,
    t: float = 12.0,
    alpha: float = 2.0,
    arnold_iters: int = 10,
    coef: tuple[int, int] = DEFAULT_COEF,
    direction: str = "LR",
) -> tuple[np.ndarray, EmbedParams]:
    """Embed a binary watermark into a grayscale cover image (blind scheme).

    Parameters
    ----------
    cover : ndarray
        Square grayscale (or RGB, converted to luminance) image, uint8, with a
        side that is a multiple of 8.
    watermark : ndarray
        Square binary image (values 0/1 or 0/255). Its side ``m`` must equal
        ``N/8 - 1`` (the last block row/column is reserved as a reference).
    T : float
        Width of the difference region used to encode one bit.
    t : float
        Guard distance kept to the region boundary after embedding.
    alpha : float
        Scale of the adaptive per-block modification strength.
    arnold_iters : int
        Number of Arnold scrambling iterations applied to the watermark.
    coef : (int, int)
        Mid-frequency coefficient position used as the carrier.

    Returns
    -------
    (stego_uint8, params)
    """
    gray = _to_gray_float(cover)
    n = gray.shape[0]
    if gray.shape[0] != gray.shape[1]:
        raise ValueError("cover image must be square")
    if n % BLOCK != 0:
        raise ValueError("cover side must be a multiple of 8")
    nb = n // BLOCK

    wm = (np.asarray(watermark) > 0).astype(int)
    m = wm.shape[0]
    if wm.shape[0] != wm.shape[1]:
        raise ValueError("watermark must be square")
    if m != nb - 1:
        raise ValueError(f"watermark side must be N/8 - 1 = {nb - 1}, got {m}")

    coeffs = _image_to_blocks(gray)
    cx, cy = coef

    wm_scr = arnold_scramble(wm, arnold_iters)

    for (bu, bv), (ru, rv) in _carrier_indices(nb, m, direction):
        bit = int(wm_scr[bu, bv])
        block = coeffs[bu, bv]
        dc = block[0, 0]
        med = float(np.median(block[AC9_R, AC9_C]))
        s = _strength(dc, med, alpha)

        ref = coeffs[ru, rv, cx, cy]
        delta = block[cx, cy] - ref
        center = _target_center(delta, bit, T)

        # offset the carrier coefficient by +-S until delta is within t of center
        step = math.copysign(s, center - delta)
        guard = max(0.0, min(t, T / 2.0 - 1e-6))
        max_iter = int(abs(center - delta) / s) + 4
        it = 0
        while abs(delta - center) > guard and it < max_iter:
            block[cx, cy] += step
            delta = block[cx, cy] - ref
            it += 1
        # land exactly inside the [center - t, center + t] band, minimal change
        final = min(max(delta, center - guard), center + guard)
        block[cx, cy] = ref + final

    stego = _blocks_to_image(coeffs)
    params = EmbedParams(
        cover_shape=(n, n),
        wm_size=m,
        T=float(T),
        t=float(t),
        alpha=float(alpha),
        arnold_iters=int(arnold_iters),
        coef=(int(cx), int(cy)),
        direction=direction,
    )
    return stego, params


def extract(stego: np.ndarray, params: EmbedParams) -> np.ndarray:
    """Recover the binary watermark (blind: only the stego image is needed)."""
    gray = _to_gray_float(stego)
    n = gray.shape[0]
    nb = n // BLOCK
    m = params.wm_size
    cx, cy = params.coef
    T = params.T

    coeffs = _image_to_blocks(gray)

    wm_scr = np.zeros((m, m), dtype=np.uint8)
    for (bu, bv), (ru, rv) in _carrier_indices(nb, m, params.direction):
        delta = coeffs[bu, bv, cx, cy] - coeffs[ru, rv, cx, cy]
        wm_scr[bu, bv] = int(round(delta / T)) % 2

    wm = arnold_unscramble(wm_scr, params.arnold_iters)
    return (wm > 0).astype(np.uint8)
