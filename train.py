#!/usr/bin/env python
"""Supervised training: baseline from scratch, fine-tuning from an SSL checkpoint, linear probe, or
joint SSL + classification.

    python train.py --config configs/sup_baseline.yaml --seed 0
    python train.py --config configs/ft_simclr.yaml --seed 0 --set init=runs/ssl_simclr/s0/ssl.pt
    python train.py --config configs/sup_baseline.yaml --label-split splits/semisup_k20.json --name sup_k20

Writes runs/<name>/s<seed>/{config.json, env.json, history.csv, metrics.json, best.pt, per_class.json,
confusion.npy, confusion.png}.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from src.data import build_ssl_loader, build_stores, build_sup_loaders
from src.engine import evaluate, per_class_accuracy, train
from src.models import Classifier, build_classifier, count_params, load_ssl_backbone
from src.ssl import build_ssl
from src.utils import environment_info, load_config, save_json, set_seed


def plot_confusion(cm, classes, path, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 7))
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=60, ha="right", fontsize=8); ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=6,
                        color="white" if cmn[i, j] > 0.5 else "black")
    fig.colorbar(im, fraction=0.046); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides")
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="splits/val_split.json")
    ap.add_argument("--label-split", default=None, help="splits/semisup_k<k>.json: train on the labeled subset only")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    seed = args.seed if args.seed is not None else cfg.get("seed", 0)
    cfg["seed"] = seed
    name = args.name or Path(args.config).stem
    run_dir = Path(args.runs) / name / f"s{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_f = (run_dir / "train.log").open("w")

    def log(msg):
        print(msg, flush=True); log_f.write(msg + "\n"); log_f.flush()

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_split = Path(args.label_split) if args.label_split else None
    stores, classes = build_stores(Path(args.data), Path(args.split), label_split)
    cfg["num_classes"] = len(classes)
    if label_split:
        cfg["label_split"] = str(label_split)
        cfg["labels_per_class"] = len(stores["train"]) // len(classes)

    # init may point at a run directory or a checkpoint file; a seed-matched path is substituted when {seed} is used
    init = cfg.get("init")
    if init:
        init = str(init).format(seed=seed)
        if Path(init).is_dir():
            init = str(Path(init) / "ssl.pt")
        cfg["init"] = init

    ssl_module = None
    if cfg.get("mode", "supervised") == "joint":
        ssl_module = build_ssl(cfg)
        if init:
            load_ssl_backbone(ssl_module.backbone, init)
        model = Classifier(ssl_module.backbone, len(classes), cfg.get("dropout", 0.0))  # shared encoder
        train_loader = build_ssl_loader(cfg, stores["train"], with_sup_view=True)
        _, val_loader, _ = build_sup_loaders(cfg, stores)
    else:
        model = build_classifier(cfg, len(classes))
        train_loader, val_loader, _ = build_sup_loaders(cfg, stores)

    n_params, n_train = count_params(model), count_params(model, trainable_only=True)
    save_json(cfg, run_dir / "config.json"); save_json(environment_info(), run_dir / "env.json")
    log(f"run {name} seed {seed} | {cfg['model']} init={init or 'random'} mode={cfg.get('mode','supervised')} "
        f"frozen={cfg.get('freeze_backbone', False)} | params {n_params/1e6:.2f}M (trainable {n_train/1e6:.2f}M) "
        f"| train {len(train_loader.dataset)} val {len(val_loader.dataset)} | {device}")
    log(f"config: {cfg}")

    best = train(cfg, model, train_loader, val_loader, device, run_dir, log=log, ssl_module=ssl_module)

    model.load_state_dict(best["state"])
    val = evaluate(model, val_loader, device)
    pc = per_class_accuracy(val["preds"], val["targets"], len(classes))
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(val["targets"], val["preds"], labels=range(len(classes)))
    np.save(run_dir / "confusion.npy", cm)
    plot_confusion(cm, classes, run_dir / "confusion.png", f"{name} s{seed} val acc {val['acc']:.3f}")
    save_json(dict(zip(classes, pc)), run_dir / "per_class.json")

    # final-epoch train/val gap from the history (overfitting diagnostic)
    import csv
    rows = [r for r in csv.DictReader((run_dir / "history.csv").open()) if r["val_acc"]]
    last = rows[-1]
    metrics = {"name": name, "seed": seed, "model": cfg["model"], "init": init or "random",
               "init_method": (torch.load(init, map_location="cpu", weights_only=False)["config"].get("ssl")
                               if init else "-"),
               "mode": cfg.get("mode", "supervised"), "ssl": cfg.get("ssl", "-") if cfg.get("mode") == "joint" else "-",
               "frozen": bool(cfg.get("freeze_backbone", False)), "aug": cfg["aug"], "img_size": cfg["img_size"],
               "epochs": cfg["epochs"], "lr": cfg["lr"], "label_smoothing": cfg.get("label_smoothing", 0.0),
               "weight_decay": cfg.get("weight_decay", 0.0), "ema": bool(cfg.get("ema_decay")),
               "mixup": cfg.get("mixup", 0.0), "cutmix": cfg.get("cutmix", 0.0),
               "labels_per_class": cfg.get("labels_per_class", len(stores["train"]) // len(classes)),
               "params_M": n_params / 1e6, "trainable_M": n_train / 1e6,
               "best_val_acc": best["acc"], "best_epoch": best["epoch"], "best_val_loss": best["val_loss"],
               "final_train_acc": float(last["train_acc"]), "final_val_acc": float(last["val_acc"]),
               "final_train_loss": float(last["train_loss"]), "final_val_loss": float(last["val_loss"]),
               "minutes": best["minutes"], "worst_classes": sorted(zip(pc, classes))[:3]}
    save_json(metrics, run_dir / "metrics.json")
    torch.save({"model": best["state"], "config": cfg, "classes": classes, "epoch": best["epoch"],
                "val_acc": best["acc"]}, run_dir / "best.pt")
    log(f"BEST val acc {best['acc']:.4f} @ epoch {best['epoch']} | final train {float(last['train_acc']):.3f} / "
        f"val {float(last['val_acc']):.3f} | {best['minutes']:.1f} min | worst: {metrics['worst_classes']}")


if __name__ == "__main__":
    main()
