"""
Build the data/{real,tampered,ai_generated} layout from public HF mirrors.

Sources (download the parquet files first, e.g. with curl):
  * CASIA v2.0  — real + tampered
      https://huggingface.co/datasets/KhangTruong/Casia-v2.0
      data/cross_test-00000-of-00001.parquet  (~3.5 GB, 12.5k images + masks)
      Authentic images (Au_*) have an all-black mask; tampered (Tp_*) do not.
  * Tiny-GenImage — ai_generated (label == fake; 8 generators: ADM, BigGAN,
      GLIDE, Midjourney, SD1.4, SD1.5, VQDM, Wukong)
      https://huggingface.co/datasets/TheKernel01/Tiny-GenImage
      data/train-0000{0..3}-of-00014.parquet

Original encoded bytes are written unchanged so ELA sees the real
compression history.  The one exception: GenImage fakes are almost all PNG
while CASIA is JPEG/TIFF, which would let the model learn "PNG => AI".
To weaken that shortcut, a fraction of AI images (--ai-jpeg-frac) is
re-encoded as JPEG at a random quality in [70, 95].

Usage:
    python scripts/prepare_data.py --casia raw/casia.parquet \
        --genimage raw/genimage_*.parquet --out data --ai-max 5000
"""

from __future__ import annotations

import argparse
import glob
import io
import os
import random
from collections import Counter

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

_SNIFF = {b"\xff\xd8": ".jpg", b"\x89P": ".png", b"II": ".tif", b"MM": ".tif",
          b"BM": ".bmp", b"RI": ".webp"}


def _ext(data: bytes, path: str | None) -> str:
    if path:
        e = os.path.splitext(path)[1].lower()
        if e in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}:
            return ".tif" if e == ".tiff" else e
    return _SNIFF.get(data[:2], ".png")


def _write(data: bytes, ext: str, out_dir: str, name: str) -> str:
    # TIFF isn't in ForgeryDataset's accepted extensions — convert losslessly.
    if ext == ".tif":
        buf = io.BytesIO()
        Image.open(io.BytesIO(data)).convert("RGB").save(buf, format="PNG")
        data, ext = buf.getvalue(), ".png"
    path = os.path.join(out_dir, name + ext)
    with open(path, "wb") as f:
        f.write(data)
    return path


def prepare_casia(parquet: str, out: str) -> Counter:
    counts: Counter = Counter()
    for d in ("real", "tampered"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    pf = pq.ParquetFile(parquet)
    i = 0
    for batch in pf.iter_batches(batch_size=64, columns=["image", "mask"]):
        for img, mask in zip(batch.column("image").to_pylist(),
                             batch.column("mask").to_pylist()):
            i += 1
            data, path = img["bytes"], img.get("path")
            name = os.path.splitext(os.path.basename(path))[0] if path else f"casia_{i:05d}"
            if path and os.path.basename(path).startswith(("Au_", "Tp_")):
                tampered = os.path.basename(path).startswith("Tp_")
            else:
                m = None if mask is None or mask.get("bytes") is None else \
                    np.asarray(Image.open(io.BytesIO(mask["bytes"])).convert("L"))
                tampered = m is not None and bool((m > 127).any())
            cls = "tampered" if tampered else "real"
            try:
                _write(data, _ext(data, path), os.path.join(out, cls), name)
                counts[cls] += 1
            except Exception as exc:  # corrupt sample
                print(f"  skip {name}: {exc}")
                counts["skipped"] += 1
    return counts


def prepare_genimage(parquets: list[str], out: str, max_n: int,
                     jpeg_frac: float, seed: int) -> Counter:
    rng = random.Random(seed)
    out_dir = os.path.join(out, "ai_generated")
    os.makedirs(out_dir, exist_ok=True)

    # Collect fake samples, balanced across generators.
    per_gen: dict[int, list[tuple[bytes, str | None]]] = {}
    for p in parquets:
        pf = pq.ParquetFile(p)
        for batch in pf.iter_batches(batch_size=64,
                                     columns=["image", "label", "generator"]):
            for img, label, gen in zip(batch.column("image").to_pylist(),
                                       batch.column("label").to_pylist(),
                                       batch.column("generator").to_pylist()):
                if label == 1:
                    per_gen.setdefault(gen, []).append((img["bytes"], img.get("path")))
    print("  fakes per generator:", {g: len(v) for g, v in sorted(per_gen.items())})

    for v in per_gen.values():
        rng.shuffle(v)
    chosen: list[tuple[int, bytes, str | None]] = []
    while len(chosen) < max_n and any(per_gen.values()):
        for g, v in sorted(per_gen.items()):
            if v and len(chosen) < max_n:
                b, p = v.pop()
                chosen.append((g, b, p))

    counts: Counter = Counter()
    for i, (gen, data, path) in enumerate(chosen):
        name = f"gen{gen}_{i:05d}"
        try:
            if rng.random() < jpeg_frac:
                buf = io.BytesIO()
                Image.open(io.BytesIO(data)).convert("RGB").save(
                    buf, format="JPEG", quality=rng.randint(70, 95))
                _write(buf.getvalue(), ".jpg", out_dir, name)
                counts["ai_generated(jpeg)"] += 1
            else:
                _write(data, _ext(data, path), out_dir, name)
                counts["ai_generated(orig)"] += 1
        except Exception as exc:
            print(f"  skip {name}: {exc}")
            counts["skipped"] += 1
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--casia", required=True)
    ap.add_argument("--genimage", nargs="+", required=True)
    ap.add_argument("--out", default="data")
    ap.add_argument("--ai-max", type=int, default=5000)
    ap.add_argument("--ai-jpeg-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    gen_files = sorted(f for pat in a.genimage for f in glob.glob(pat))
    print("CASIA v2.0 ...")
    print(" ", dict(prepare_casia(a.casia, a.out)))
    print("GenImage ...")
    print(" ", dict(prepare_genimage(gen_files, a.out, a.ai_max, a.ai_jpeg_frac, a.seed)))


if __name__ == "__main__":
    main()
