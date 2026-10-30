"""Summarize the dataset: per-class counts, image sizes, color mode, and
train/test near-duplicates. Run before any training so decisions about
resolution, color and the validation split are grounded in the real data.

Usage:
    python scripts/inspect_dataset.py --data data
"""
from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


def iter_images(split_dir: Path):
    for cls_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        for f in sorted(cls_dir.iterdir()):
            if f.suffix.lower() in IMG_EXT:
                yield cls_dir.name, f


def dhash(img: Image.Image, size: int = 8) -> str:
    """Difference hash: robust to resizing/JPEG re-encoding, good for dedup."""
    g = img.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    px = list(g.get_flattened_data()) if hasattr(g, "get_flattened_data") else list(g.getdata())
    bits = []
    for r in range(size):
        row = px[r * (size + 1):(r + 1) * (size + 1)]
        bits.extend(int(row[c] > row[c + 1]) for c in range(size))
    return "".join(str(b) for b in bits)


def is_color(img: Image.Image, sample: int = 64, tol: int = 8) -> bool:
    """True if the image has meaningful chroma (not a gray image stored as RGB)."""
    if img.mode in ("L", "1", "LA", "I;16", "I"):
        return False
    rgb = img.convert("RGB").resize((sample, sample))
    for r, g, b in (rgb.get_flattened_data() if hasattr(rgb, "get_flattened_data") else rgb.getdata()):
        if max(r, g, b) - min(r, g, b) > tol:
            return True
    return False


def summarize(split: str, split_dir: Path):
    counts: Counter = Counter()
    sizes: Counter = Counter()
    modes: Counter = Counter()
    color_by_cls: dict = defaultdict(lambda: [0, 0])
    hashes: dict = {}
    md5s: dict = {}
    widths, heights = [], []
    for cls, f in iter_images(split_dir):
        counts[cls] += 1
        with Image.open(f) as im:
            w, h = im.size
            widths.append(w); heights.append(h)
            sizes[(w, h)] += 1
            modes[im.mode] += 1
            color_by_cls[cls][int(is_color(im))] += 1
            hashes[f] = dhash(im)
        md5s[f] = hashlib.md5(f.read_bytes()).hexdigest()
    n = sum(counts.values())
    print(f"\n=== {split}: {n} images, {len(counts)} classes ===")
    for cls in sorted(counts):
        gray, col = color_by_cls[cls]
        print(f"  {cls:<14} {counts[cls]:>5}   color={col:>4} gray={gray:>4}")
    print(f"  modes: {dict(modes)}")
    if widths:
        print(f"  width  min/median/max: {min(widths)}/{sorted(widths)[len(widths)//2]}/{max(widths)}")
        print(f"  height min/median/max: {min(heights)}/{sorted(heights)[len(heights)//2]}/{max(heights)}")
    print(f"  top sizes: {sizes.most_common(5)}")
    return hashes, md5s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    args = ap.parse_args()
    tr_h, tr_md5 = summarize("train", args.data / "train")
    te_h, te_md5 = summarize("test", args.data / "test")

    # exact duplicates
    inv = defaultdict(list)
    for f, m in {**tr_md5, **te_md5}.items():
        inv[m].append(f)
    exact = [v for v in inv.values() if len(v) > 1]
    cross = [v for v in exact if any("/train/" in str(p) for p in v) and any("/test/" in str(p) for p in v)]
    print(f"\n=== duplicates ===")
    print(f"  exact duplicate groups: {len(exact)}  (train<->test: {len(cross)})")
    for grp in cross[:10]:
        print("   ", [str(p.relative_to(args.data)) for p in grp])

    # near duplicates train<->test by dhash hamming distance <= 2
    te_by_hash = defaultdict(list)
    for f, h in te_h.items():
        te_by_hash[h].append(f)
    near = 0
    examples = []
    te_items = list(te_h.items())
    for f, h in tr_h.items():
        for g, h2 in te_items:
            if sum(a != b for a, b in zip(h, h2)) <= 2:
                near += 1
                if len(examples) < 10:
                    examples.append((str(f.relative_to(args.data)), str(g.relative_to(args.data))))
                break
    print(f"  train images with a near-duplicate (dhash dist<=2) in test: {near}")
    for a, b in examples:
        print("   ", a, "~", b)


if __name__ == "__main__":
    main()
