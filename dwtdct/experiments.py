"""Computational experiments for the hybrid DCT+DWT watermarking report.

Run as ``python -m dwtdct.experiments``. Produces, in ``kriptoprak3/images/``:

* the cover image, the binary watermark, the stego image and a difference map;
* an imperceptibility study versus the embedding strength ``alpha``;
* a robustness study (clean / JPEG at several qualities / brightness / noise /
  scaling / cropping) with the recovered watermarks;
* a robustness-versus-alpha study under JPEG compression;
* CSV tables, a JSON summary and a Markdown digest used to fill the report.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from skimage import data

from . import watermark as wm
from . import metrics as M

OUT_DIR = Path(__file__).resolve().parents[1] / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WM_SIZE = 64
DEFAULT_ALPHA = 20.0
DEFAULT_ITERS = 10


# --------------------------------------------------------------------------- #
# Inputs                                                                      #
# --------------------------------------------------------------------------- #

def make_cover() -> np.ndarray:
    """512x512 grayscale standard test image (`camera`)."""
    img = data.camera()
    return img.astype(np.uint8)


def make_watermark(size: int = WM_SIZE) -> np.ndarray:
    """Recognisable binary logo (the letters 'HSE' + frame), size x size."""
    scale = 8
    big = Image.new("L", (size * scale, size * scale), color=0)
    draw = ImageDraw.Draw(big)
    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
            size=int(size * scale * 0.55),
        )
    except OSError:
        font = ImageFont.load_default()
    text = "HSE"
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(
        ((size * scale - tw) / 2 - bbox[0], (size * scale - th) / 2 - bbox[1]),
        text, fill=255, font=font,
    )
    border = int(size * scale * 0.06)
    draw.rectangle(
        [border, border, size * scale - border, size * scale - border],
        outline=255, width=int(size * scale * 0.04),
    )
    small = big.resize((size, size), Image.LANCZOS)
    return (np.array(small) > 100).astype(np.uint8)


def save_image(arr: np.ndarray, path: Path) -> None:
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


# --------------------------------------------------------------------------- #
# Attacks                                                                     #
# --------------------------------------------------------------------------- #

def attack_jpeg(stego: np.ndarray, quality: int) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(stego).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf).convert("L"), dtype=np.uint8)


def attack_brightness(stego: np.ndarray, delta: int) -> np.ndarray:
    return np.clip(stego.astype(np.int16) + delta, 0, 255).astype(np.uint8)


def attack_gaussian(stego: np.ndarray, sigma: float, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = stego.astype(np.float64) + rng.normal(0, sigma, stego.shape)
    return np.clip(np.rint(noisy), 0, 255).astype(np.uint8)


def attack_saltpepper(stego: np.ndarray, amount: float, seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = stego.copy()
    mask = rng.random(out.shape)
    out[mask < amount / 2] = 0
    out[mask > 1 - amount / 2] = 255
    return out


def attack_rescale(stego: np.ndarray, factor: float) -> np.ndarray:
    n = stego.shape[0]
    small = Image.fromarray(stego).resize(
        (max(1, int(n * factor)), max(1, int(n * factor))), Image.BILINEAR
    )
    return np.array(small.resize((n, n), Image.BILINEAR), dtype=np.uint8)


def attack_crop(stego: np.ndarray, frac: float) -> np.ndarray:
    """Zero out a border of width ``frac`` of the image (cropping then padding)."""
    out = stego.copy()
    n = out.shape[0]
    b = int(n * frac)
    if b > 0:
        out[:b, :] = 0
        out[-b:, :] = 0
        out[:, :b] = 0
        out[:, -b:] = 0
    return out


# --------------------------------------------------------------------------- #
# Experiments                                                                 #
# --------------------------------------------------------------------------- #

def experiment_example(cover, mark):
    print("[exp 1] example embedding (default parameters) ...")
    stego, params = wm.embed(cover, mark, alpha=DEFAULT_ALPHA, arnold_iters=DEFAULT_ITERS)
    rec = wm.extract(stego, cover, params)
    save_image(stego, OUT_DIR / "stego.png")
    save_image((rec * 255).astype(np.uint8), OUT_DIR / "recovered_clean.png")

    diff = np.abs(cover.astype(np.int16) - stego.astype(np.int16)).astype(np.float64)
    scale = max(1.0, 255.0 / max(1.0, diff.max()))
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.imshow(np.clip(diff * scale, 0, 255).astype(np.uint8), cmap="gray")
    ax.set_title(f"|cover - stego| x {scale:.0f}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "diffmap.png", dpi=130)
    plt.close(fig)

    metrics = {
        "alpha": DEFAULT_ALPHA,
        "arnold_iters": DEFAULT_ITERS,
        "psnr_db": M.psnr(cover, stego),
        "mse": M.mse(cover, stego),
        "rmse": M.rmse(cover, stego),
        "ssim": M.ssim(cover, stego),
        "clean_ber": M.ber(mark, rec),
        "clean_ncc": M.ncc(mark, rec),
    }
    print(f"        PSNR={metrics['psnr_db']:.2f} dB  SSIM={metrics['ssim']:.4f}  "
          f"clean BER={metrics['clean_ber']:.4f}")
    return metrics


def experiment_alpha(cover, mark):
    print("[exp 2] imperceptibility vs alpha ...")
    alphas = [5, 10, 20, 30, 40, 60]
    rows = []
    stegos = {}
    for a in alphas:
        stego, params = wm.embed(cover, mark, alpha=a, arnold_iters=DEFAULT_ITERS)
        rec = wm.extract(stego, cover, params)
        rows.append({
            "alpha": a,
            "psnr_db": M.psnr(cover, stego),
            "mse": M.mse(cover, stego),
            "ssim": M.ssim(cover, stego),
            "clean_ber": M.ber(mark, rec),
            "clean_ncc": M.ncc(mark, rec),
        })
        stegos[a] = stego
        print(f"        alpha={a:>3}  PSNR={rows[-1]['psnr_db']:.2f} dB  "
              f"SSIM={rows[-1]['ssim']:.4f}")

    fig, ax1 = plt.subplots(figsize=(6.5, 4.0))
    xs = [r["alpha"] for r in rows]
    ax1.plot(xs, [r["psnr_db"] for r in rows], "o-", color="navy", label="PSNR")
    ax1.set_xlabel("alpha")
    ax1.set_ylabel("PSNR, dB", color="navy")
    ax1.tick_params(axis="y", labelcolor="navy")
    ax2 = ax1.twinx()
    ax2.plot(xs, [r["ssim"] for r in rows], "s--", color="darkred", label="SSIM")
    ax2.set_ylabel("SSIM", color="darkred")
    ax2.tick_params(axis="y", labelcolor="darkred")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "psnr_ssim_vs_alpha.png", dpi=130)
    plt.close(fig)

    save_image(stegos[5], OUT_DIR / "stego_alpha05.png")
    save_image(stegos[60], OUT_DIR / "stego_alpha60.png")
    return rows


def experiment_robustness(cover, mark, alpha=DEFAULT_ALPHA):
    print(f"[exp 3] robustness to attacks (alpha={alpha}) ...")
    stego, params = wm.embed(cover, mark, alpha=alpha, arnold_iters=DEFAULT_ITERS)

    attacks = [
        ("Без атаки", lambda s: s),
        ("JPEG q=90", lambda s: attack_jpeg(s, 90)),
        ("JPEG q=70", lambda s: attack_jpeg(s, 70)),
        ("JPEG q=50", lambda s: attack_jpeg(s, 50)),
        ("JPEG q=30", lambda s: attack_jpeg(s, 30)),
        ("Яркость +20", lambda s: attack_brightness(s, 20)),
        ("Гаусс. шум s=5", lambda s: attack_gaussian(s, 5)),
        ("Соль-перец 1%", lambda s: attack_saltpepper(s, 0.01)),
        ("Масштаб 0.5x", lambda s: attack_rescale(s, 0.5)),
        ("Кадрир. 10%", lambda s: attack_crop(s, 0.10)),
    ]

    rows = []
    recovered = []
    for name, fn in attacks:
        attacked = fn(stego)
        rec = wm.extract(attacked, cover, params)
        rows.append({
            "attack": name,
            "psnr_attacked": M.psnr(stego, attacked) if attacked.shape == stego.shape else float("nan"),
            "ber": M.ber(mark, rec),
            "ncc": M.ncc(mark, rec),
        })
        recovered.append((name, rec))
        print(f"        {name:18s}  BER={rows[-1]['ber']:.4f}  NCC={rows[-1]['ncc']:.4f}")

    n = len(recovered)
    cols = 5
    rcount = (n + cols - 1) // cols
    fig, axes = plt.subplots(rcount, cols, figsize=(2.1 * cols, 2.35 * rcount))
    axes = np.atleast_2d(axes)
    for idx, (name, rec) in enumerate(recovered):
        ax = axes[idx // cols, idx % cols]
        ax.imshow((rec * 255).astype(np.uint8), cmap="gray", vmin=0, vmax=255)
        ncc_v = next(r["ncc"] for r in rows if r["attack"] == name)
        ax.set_title(f"{name}\nNCC={ncc_v:.3f}", fontsize=8)
        ax.axis("off")
    for idx in range(n, rcount * cols):
        axes[idx // cols, idx % cols].axis("off")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "recovered_attacks.png", dpi=130)
    plt.close(fig)
    return rows


def experiment_robustness_vs_alpha(cover, mark, quality=50):
    print(f"[exp 4] robustness vs alpha under JPEG q={quality} ...")
    alphas = [5, 10, 20, 30, 40, 60]
    rows = []
    for a in alphas:
        stego, params = wm.embed(cover, mark, alpha=a, arnold_iters=DEFAULT_ITERS)
        attacked = attack_jpeg(stego, quality)
        rec = wm.extract(attacked, cover, params)
        rows.append({
            "alpha": a,
            "psnr_db": M.psnr(cover, stego),
            "jpeg_ber": M.ber(mark, rec),
            "jpeg_ncc": M.ncc(mark, rec),
        })
        print(f"        alpha={a:>3}  PSNR={rows[-1]['psnr_db']:.2f} dB  "
              f"JPEG-NCC={rows[-1]['jpeg_ncc']:.4f}  BER={rows[-1]['jpeg_ber']:.4f}")

    fig, ax1 = plt.subplots(figsize=(6.5, 4.0))
    xs = [r["alpha"] for r in rows]
    ax1.plot(xs, [r["jpeg_ncc"] for r in rows], "o-", color="green", label="NCC")
    ax1.set_xlabel("alpha")
    ax1.set_ylabel(f"NCC after JPEG q={quality}", color="green")
    ax1.set_ylim(0, 1.02)
    ax1.tick_params(axis="y", labelcolor="green")
    ax2 = ax1.twinx()
    ax2.plot(xs, [r["psnr_db"] for r in rows], "s--", color="navy", label="PSNR")
    ax2.set_ylabel("PSNR (stego), dB", color="navy")
    ax2.tick_params(axis="y", labelcolor="navy")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ncc_vs_alpha_jpeg.png", dpi=130)
    plt.close(fig)
    return rows


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    cover = make_cover()
    mark = make_watermark()
    save_image(cover, OUT_DIR / "cover.png")
    save_image((mark * 255).astype(np.uint8), OUT_DIR / "watermark.png")
    print(f"cover: {cover.shape}  watermark: {mark.shape}")

    m1 = experiment_example(cover, mark)
    rows_alpha = experiment_alpha(cover, mark)
    rows_rob = experiment_robustness(cover, mark)
    rows_rob_alpha = experiment_robustness_vs_alpha(cover, mark)

    write_csv(OUT_DIR / "alpha.csv", rows_alpha)
    write_csv(OUT_DIR / "robustness.csv", rows_rob)
    write_csv(OUT_DIR / "robustness_vs_alpha.csv", rows_rob_alpha)

    summary = {
        "cover_shape": list(cover.shape),
        "wm_size": int(mark.shape[0]),
        "experiment_1_example": m1,
        "experiment_2_alpha": rows_alpha,
        "experiment_3_robustness": rows_rob,
        "experiment_4_robustness_vs_alpha": rows_rob_alpha,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    lines = ["# DCT+DWT watermarking — experiments summary", ""]
    lines.append(f"Cover {cover.shape}, watermark {mark.shape}, "
                 f"alpha={DEFAULT_ALPHA}, arnold={DEFAULT_ITERS}")
    lines += ["", "## Exp 1 — example"]
    for k, v in m1.items():
        lines.append(f"* {k}: {v}")
    lines += ["", "## Exp 2 — imperceptibility vs alpha",
              "| alpha | PSNR dB | SSIM | clean BER |", "|---|---|---|---|"]
    for r in rows_alpha:
        lines.append(f"| {r['alpha']} | {r['psnr_db']:.2f} | {r['ssim']:.4f} | {r['clean_ber']:.4f} |")
    lines += ["", "## Exp 3 — robustness (alpha=20)",
              "| attack | PSNR att. | BER | NCC |", "|---|---|---|---|"]
    for r in rows_rob:
        lines.append(f"| {r['attack']} | {r['psnr_attacked']:.2f} | {r['ber']:.4f} | {r['ncc']:.4f} |")
    lines += ["", "## Exp 4 — robustness vs alpha (JPEG q=50)",
              "| alpha | PSNR dB | JPEG BER | JPEG NCC |", "|---|---|---|---|"]
    for r in rows_rob_alpha:
        lines.append(f"| {r['alpha']} | {r['psnr_db']:.2f} | {r['jpeg_ber']:.4f} | {r['jpeg_ncc']:.4f} |")
    (OUT_DIR / "summary.md").write_text("\n".join(lines))
    print(f"\nDone. Output in {OUT_DIR}")


if __name__ == "__main__":
    main()
