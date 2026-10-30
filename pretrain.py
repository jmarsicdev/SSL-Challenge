#!/usr/bin/env python
"""Self-supervised pretraining on the 1,920 training-split images (labels are never read).

    python pretrain.py --config configs/ssl_simclr.yaml [--seed 0] [--set epochs=800]

Writes runs/<name>/s<seed>/{config.json, env.json, history.csv, metrics.json, ssl.pt, snapshots/ep*.pt}.
A weighted kNN classifier on frozen features (train split -> val split) is logged every `knn_every`
epochs as a label-free-training / label-using-monitor of representation quality.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from src.data import build_eval_loader, build_ssl_loader, build_stores
from src.engine import build_scheduler
from src.models import count_params
from src.ssl import build_ssl, extract_features, knn_accuracy
from src.utils import environment_info, load_config, save_json, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="splits/val_split.json")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    seed = args.seed if args.seed is not None else cfg.get("seed", 0)
    cfg["seed"] = seed
    name = args.name or Path(args.config).stem
    run_dir = Path(args.runs) / name / f"s{seed}"
    (run_dir / "snapshots").mkdir(parents=True, exist_ok=True)
    log_f = (run_dir / "train.log").open("w")

    def log(msg):
        print(msg, flush=True); log_f.write(msg + "\n"); log_f.flush()

    set_seed(seed)
    device = torch.device("cuda")
    stores, classes = build_stores(Path(args.data), Path(args.split))
    loader = build_ssl_loader(cfg, stores["pretrain"])
    knn_train = build_eval_loader(cfg, stores["pretrain"])
    knn_val = build_eval_loader(cfg, stores["val"])

    ssl = build_ssl(cfg).to(device)
    ssl = ssl.to(memory_format=torch.channels_last)
    params = [p for p in ssl.parameters() if p.requires_grad]
    decay = [p for p in params if p.ndim > 1]
    no_decay = [p for p in params if p.ndim <= 1]
    opt_name = cfg.get("optimizer", "adamw")
    groups = [{"params": decay, "weight_decay": cfg.get("weight_decay", 1e-5)}, {"params": no_decay, "weight_decay": 0.0}]
    if opt_name == "adamw":
        opt = torch.optim.AdamW(groups, lr=cfg["lr"], betas=(0.9, 0.999))
    else:
        opt = torch.optim.SGD(groups, lr=cfg["lr"], momentum=0.9, nesterov=True)
    sched = build_scheduler(cfg, opt, len(loader))
    total_steps = cfg["epochs"] * len(loader)
    save_json(cfg, run_dir / "config.json"); save_json(environment_info(), run_dir / "env.json")
    log(f"pretrain {name} seed {seed} | {cfg['ssl']} on {cfg['model']} | params {count_params(ssl)/1e6:.2f}M "
        f"| {len(stores['pretrain'])} images, {len(loader)} steps/epoch x {cfg['epochs']} epochs | {device}")
    log(f"config: {cfg}")

    fields = ["epoch", "lr", "loss", "stat", "knn_acc", "sec"]
    with (run_dir / "history.csv").open("w", newline="") as f:
        csv.DictWriter(f, fields).writeheader()
    knn_every = int(cfg.get("knn_every", 25))
    snaps = set(cfg.get("snapshots", []))
    best_knn, step, t0 = 0.0, 0, time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        ssl.train()
        tot, stat_sum, n, te = 0.0, 0.0, 0, time.time()
        for batch in loader:
            batch = tuple(b.to(device, non_blocking=True) for b in batch)
            batch = (batch[0].contiguous(memory_format=torch.channels_last),) + batch[1:] if batch[0].ndim == 4 else batch
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, stats = ssl.loss(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
            opt.step()
            if sched is not None:
                sched.step()
            ssl.after_step(step, total_steps)
            step += 1
            tot += loss.item(); stat_sum += next(iter(stats.values())); n += 1
        knn = ""
        if epoch % knn_every == 0 or epoch == cfg["epochs"]:
            trf, trl = extract_features(ssl.backbone, knn_train, device)
            vaf, val_ = extract_features(ssl.backbone, knn_val, device)
            knn = knn_accuracy(trf, trl, vaf, val_, num_classes=len(classes))
            best_knn = max(best_knn, knn)
        row = {"epoch": epoch, "lr": opt.param_groups[0]["lr"], "loss": tot / n, "stat": stat_sum / n,
               "knn_acc": knn, "sec": time.time() - te}
        with (run_dir / "history.csv").open("a", newline="") as f:
            csv.DictWriter(f, fields).writerow(row)
        if knn != "" or epoch % 10 == 0 or epoch == 1:
            sk = next(iter(stats.keys()))
            log(f"ep {epoch:04d}/{cfg['epochs']} lr {row['lr']:.2e} | loss {row['loss']:.4f} {sk} {row['stat']:.3f}"
                + (f" | kNN val {knn:.4f}" if knn != "" else "") + f" | {row['sec']:.1f}s")
        if epoch in snaps:
            torch.save({"backbone": ssl.backbone.state_dict(), "config": cfg, "epoch": epoch, "knn_acc": knn},
                       run_dir / "snapshots" / f"ep{epoch}.pt")

    minutes = (time.time() - t0) / 60
    torch.save({"backbone": ssl.backbone.state_dict(), "ssl_state": ssl.extra_state(), "config": cfg,
                "epoch": cfg["epochs"], "knn_acc": knn}, run_dir / "ssl.pt")
    save_json({"name": name, "seed": seed, "ssl": cfg["ssl"], "model": cfg["model"], "views": cfg.get("views", "-"),
               "img_size": cfg["img_size"], "epochs": cfg["epochs"], "batch_size": cfg["batch_size"], "lr": cfg["lr"],
               "final_knn_acc": knn, "best_knn_acc": best_knn, "final_loss": tot / n, "minutes": minutes,
               "params_M": count_params(ssl.backbone) / 1e6}, run_dir / "metrics.json")
    log(f"DONE kNN val {knn:.4f} (best {best_knn:.4f}) | {minutes:.1f} min | saved {run_dir/'ssl.pt'}")


if __name__ == "__main__":
    main()
