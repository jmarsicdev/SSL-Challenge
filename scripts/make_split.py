"""Create the fixed, stratified train/val split (committed to splits/val_split.json).

- 30 validation images per class (20%), seed 0.
- Byte-identical duplicate images are kept on the same side of the split so the
  validation set never contains a copy of a training image.

    python scripts/make_split.py --data data --out splits/val_split.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="splits/val_split.json")
    ap.add_argument("--val-per-class", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    train_root = Path(args.data) / "train"

    train, val, groups_split = [], [], 0
    for cls_dir in sorted(p for p in train_root.iterdir() if p.is_dir()):
        files = sorted(f for f in cls_dir.iterdir() if f.suffix.lower() in IMG_EXT)
        # group exact duplicates
        by_md5 = defaultdict(list)
        for f in files:
            by_md5[hashlib.md5(f.read_bytes()).hexdigest()].append(f"{cls_dir.name}/{f.name}")
        groups = list(by_md5.values())
        rng.shuffle(groups)
        n_val, cls_val = 0, []
        while n_val < args.val_per_class and groups:
            g = groups.pop()
            cls_val += g; n_val += len(g)
            groups_split += len(g) > 1
        val += cls_val
        train += [x for g in groups for x in g]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"seed": args.seed, "val_per_class": args.val_per_class,
                                          "train": sorted(train), "val": sorted(val)}, indent=0))
    print(f"train {len(train)}  val {len(val)}  (duplicate groups landing in val: {groups_split})")
    assert not set(train) & set(val)


if __name__ == "__main__":
    main()
