#!/usr/bin/env python
"""Evaluate a trained classifier checkpoint on a labeled split.

    python evaluate.py --checkpoint runs/best/s0/best.pt --split test [--tta] [--out docs/final/test_metrics.json]

The test split is meant to be evaluated once per reported number.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from src.data import build_eval_loader, build_stores
from src.engine import evaluate, per_class_accuracy
from src.models import build_classifier
from src.utils import save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--val-split", default="splits/val_split.json")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--out", default=None, help="write metrics JSON here")
    ap.add_argument("--confusion-png", default=None)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg, classes = ck["config"], ck["classes"]
    cfg = {**cfg, "init": None, "freeze_backbone": False}
    model = build_classifier(cfg, len(classes))
    model.load_state_dict(ck["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    stores, classes2 = build_stores(Path(args.data), Path(args.val_split))
    assert classes == classes2
    loader = build_eval_loader(cfg, stores[args.split])
    res = evaluate(model, loader, device, tta=args.tta)
    pc = per_class_accuracy(res["preds"], res["targets"], len(classes))
    n = len(res["targets"])
    print(f"{args.split}: acc {res['acc']*100:.2f}% ({int(round(res['acc']*n))}/{n}) loss {res['loss']:.4f}"
          + (" [flip-TTA]" if args.tta else ""))
    for c, a in zip(classes, pc):
        print(f"  {c:14s} {a*100:6.1f}%")
    out = {"checkpoint": args.checkpoint, "split": args.split, "n": n, "acc": res["acc"], "loss": res["loss"],
           "tta": args.tta, "per_class": dict(zip(classes, pc)), "model": cfg["model"],
           "train_val_acc": ck.get("val_acc"), "epoch": ck.get("epoch")}
    if args.out:
        save_json(out, Path(args.out))
    if args.confusion_png:
        from sklearn.metrics import confusion_matrix
        from train import plot_confusion
        cm = confusion_matrix(res["targets"], res["preds"], labels=range(len(classes)))
        plot_confusion(cm, classes, args.confusion_png, f"{args.split} acc {res['acc']:.4f}")


if __name__ == "__main__":
    main()
