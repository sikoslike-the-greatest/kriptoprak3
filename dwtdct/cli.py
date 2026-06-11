"""Command-line interface for the hybrid DCT+DWT watermarking method.

Usage examples::

    python -m dwtdct embed   --cover cover.png --watermark wm.png \\
                             --stego stego.png --params params.json
    python -m dwtdct extract --stego stego.png --cover cover.png \\
                             --params params.json --output recovered.png \\
                             [--watermark wm.png]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from . import watermark as wm
from . import metrics as M


def _load_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)


def _load_binary(path: Path) -> np.ndarray:
    a = np.array(Image.open(path).convert("L"))
    return (a > 127).astype(np.uint8)


def _save_image(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(arr).save(path)


def cmd_embed(args: argparse.Namespace) -> int:
    cover = _load_gray(Path(args.cover))
    mark = _load_binary(Path(args.watermark))

    stego, params = wm.embed(cover, mark, alpha=args.alpha, arnold_iters=args.iters)

    _save_image(stego, Path(args.stego))
    Path(args.params).write_text(json.dumps(params.to_dict(), indent=2))

    print(f"cover     : {args.cover}  shape={cover.shape}")
    print(f"watermark : {args.watermark}  {mark.shape}")
    print(f"stego     : {args.stego}")
    print(f"params    : {args.params}  (alpha={args.alpha}, arnold={args.iters})")
    print(f"PSNR      : {M.psnr(cover, stego):.4f} dB")
    print(f"MSE       : {M.mse(cover, stego):.4f}")
    print(f"RMSE      : {M.rmse(cover, stego):.4f}")
    print(f"SSIM      : {M.ssim(cover, stego):.6f}")
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    stego = _load_gray(Path(args.stego))
    cover = _load_gray(Path(args.cover))
    params = wm.EmbedParams.from_dict(json.loads(Path(args.params).read_text()))

    recovered = wm.extract(stego, cover, params)
    _save_image((recovered * 255).astype(np.uint8), Path(args.output))
    print(f"extracted watermark -> {args.output}  {recovered.shape}")

    if args.watermark:
        original = _load_binary(Path(args.watermark))
        print(f"BER       : {M.ber(original, recovered):.4f}")
        print(f"NCC       : {M.ncc(original, recovered):.4f}")
        print(f"bit-acc   : {M.bit_accuracy(original, recovered):.4f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dwtdct",
        description="Hybrid DCT+DWT robust image watermarking",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("embed", help="embed a binary watermark into a cover image")
    pe.add_argument("--cover", required=True)
    pe.add_argument("--watermark", required=True, help="binary watermark image")
    pe.add_argument("--stego", required=True)
    pe.add_argument("--params", required=True)
    pe.add_argument("--alpha", type=float, default=20.0)
    pe.add_argument("--iters", type=int, default=10, help="Arnold iterations")
    pe.set_defaults(func=cmd_embed)

    px = sub.add_parser("extract", help="extract a watermark (non-blind)")
    px.add_argument("--stego", required=True)
    px.add_argument("--cover", required=True, help="original cover image")
    px.add_argument("--params", required=True)
    px.add_argument("--output", required=True)
    px.add_argument("--watermark", help="original watermark, to report BER/NCC")
    px.set_defaults(func=cmd_extract)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
