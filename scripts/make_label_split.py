"""Hide most training labels to simulate a low-label regime.

Starting from the fixed train portion of splits/val_split.json (120 images/class),
keep `--labels-per-class` labeled images per class and treat the rest as an
unlabeled pool. The validation set is untouched, so low-label runs are directly
comparable with the fully supervised ones. Byte-identical duplicates stay together.

    python scripts/make_label_split.py --labels-per-class 20   # -> splits/semisup_k20.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


def make_label_split(data: Path, val_split: Path, k: int, seed: int, out: Path) -> dict:
    rng = random.Random(seed)
    train_rels = json.loads(val_split.read_text())["train"]
    by_cls = defaultdict(list)
    for r in train_rels:
        by_cls[r.split("/")[0]].append(r)

    labeled, unlabeled = [], []
    for cls in sorted(by_cls):
        groups = defaultdict(list)
        for r in by_cls[cls]:
            groups[hashlib.md5((data / "train" / r).read_bytes()).hexdigest()].append(r)
        gs = list(groups.values())
        rng.shuffle(gs)
        n, picked = 0, []
        while n < k and gs:
            g = gs.pop(); picked += g; n += len(g)
        labeled += picked
        unlabeled += [r for g in gs for r in g]

    d = {"seed": seed, "labels_per_class": k, "source": str(val_split),
         "labeled": sorted(labeled), "unlabeled": sorted(unlabeled)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(d, indent=0))
    assert not set(labeled) & set(unlabeled)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--val-split", default="splits/val_split.json")
    ap.add_argument("--labels-per-class", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out or f"splits/semisup_k{args.labels_per_class}.json")
    d = make_label_split(Path(args.data), Path(args.val_split), args.labels_per_class, args.seed, out)
    print(f"labeled {len(d['labeled'])}  unlabeled {len(d['unlabeled'])}  -> {out}")


if __name__ == "__main__":
    main()
