"""Figures from run logs (no retraining).

    python scripts/plot_curves.py            # writes docs/figures/*.png

1. overfitting.png : train/val accuracy and val loss per epoch, scratch vs SSL-initialised fine-tunes
                     (mean over seeds, shaded = min/max).
2. knn_pretraining.png : kNN val accuracy over pretraining epochs per objective (mean over seeds).
3. aug_ablation.png : probe and fine-tune accuracy per SSL view preset, if those runs exist.
"""
from __future__ import annotations

import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNS, OUT = Path("runs"), Path("docs/figures")
OUT.mkdir(parents=True, exist_ok=True)
COLORS = {"sup_baseline": "#444444", "sup_baseline_long": "#999999", "ft_rotnet": "#d95f02",
          "ft_simclr": "#1b9e77", "ft_byol": "#7570b3", "joint_byol": "#e7298a"}
LABELS = {"sup_baseline": "supervised (scratch)", "sup_baseline_long": "supervised, 300 ep", "ft_rotnet": "RotNet → fine-tune",
          "ft_simclr": "SimCLR → fine-tune", "ft_byol": "BYOL → fine-tune", "joint_byol": "joint CE + BYOL"}


def histories(name):
    out = []
    for h in sorted(RUNS.glob(f"{name}/s*/history.csv")):
        rows = [r for r in csv.DictReader(h.open()) if r.get("val_acc", "") != ""]
        if rows:
            out.append(rows)
    return out


def stack(hists, key):
    n = min(len(h) for h in hists)
    a = np.array([[float(r[key]) for r in h[:n]] for h in hists])
    ep = np.array([int(r["epoch"]) for r in hists[0][:n]])
    return ep, a


def overfitting():
    names = [n for n in ("sup_baseline", "ft_rotnet", "ft_simclr", "ft_byol") if histories(n)]
    if not names:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for n in names:
        hs = histories(n)
        ep, tr = stack(hs, "train_acc"); _, va = stack(hs, "val_acc"); _, vl = stack(hs, "val_loss")
        c = COLORS[n]
        axes[0].plot(ep, va.mean(0) * 100, color=c, label=f"{LABELS[n]} (n={len(hs)})")
        axes[0].fill_between(ep, va.min(0) * 100, va.max(0) * 100, color=c, alpha=0.15)
        axes[1].plot(ep, (tr.mean(0) - va.mean(0)) * 100, color=c, label=LABELS[n])
        axes[2].plot(ep, vl.mean(0), color=c, label=LABELS[n])
    axes[0].set_ylabel("val accuracy %"); axes[1].set_ylabel("train − val accuracy (points)"); axes[2].set_ylabel("val loss")
    for ax in axes:
        ax.set_xlabel("epoch"); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Same 100-epoch supervised recipe, only the initialisation differs (ResNet-18, 160 px)")
    fig.tight_layout(); fig.savefig(OUT / "overfitting.png", dpi=150); plt.close(fig)


def knn_pretraining():
    fig, ax = plt.subplots(figsize=(6, 3.8))
    any_ = False
    for n, lab, c in (("ssl_rotnet", "RotNet", COLORS["ft_rotnet"]), ("ssl_simclr", "SimCLR", COLORS["ft_simclr"]),
                      ("ssl_byol", "BYOL", COLORS["ft_byol"])):
        hs = []
        for h in sorted(RUNS.glob(f"{n}/s*/history.csv")):
            rows = [r for r in csv.DictReader(h.open()) if r["knn_acc"] != ""]
            if rows:
                hs.append(rows)
        if not hs:
            continue
        any_ = True
        ep, a = stack(hs, "knn_acc")
        ax.plot(ep, a.mean(0) * 100, color=c, marker="o", ms=3, label=f"{lab} (n={len(hs)})")
        ax.fill_between(ep, a.min(0) * 100, a.max(0) * 100, color=c, alpha=0.15)
    if not any_:
        return
    pr = [json.loads(p.read_text())["best_val_acc"] * 100 for p in RUNS.glob("probe_random/s*/metrics.json")]
    if pr:
        ax.axhline(st.mean(pr), color="k", ls="--", lw=1, label=f"random encoder, linear probe ({st.mean(pr):.1f})")
    ax.set_xlabel("pretraining epoch"); ax.set_ylabel("kNN val accuracy % (frozen features)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("Representation quality during self-supervised pretraining")
    fig.tight_layout(); fig.savefig(OUT / "knn_pretraining.png", dpi=150); plt.close(fig)


def aug_ablation():
    groups = defaultdict(dict)  # method -> views -> {probe:[], ft:[]}
    for mf in RUNS.glob("*/s*/metrics.json"):
        m = json.loads(mf.read_text())
        n = m["name"]
        for method in ("simclr", "byol"):
            for kind in ("probe", "ft"):
                if n == f"{kind}_{method}" or n.startswith(f"{kind}_{method}_"):
                    views = "photoblur" if n == f"{kind}_{method}" else n.split("_", 2)[2]
                    groups[method].setdefault(views, {"probe": [], "ft": []})[kind].append(m["best_val_acc"] * 100)
    groups = {k: v for k, v in groups.items() if len(v) > 1}
    if not groups:
        return
    fig, axes = plt.subplots(1, len(groups), figsize=(5.5 * len(groups), 3.8), squeeze=False)
    order = ["crop", "photo", "photoblur"]
    for ax, (method, d) in zip(axes[0], groups.items()):
        vs = [v for v in order if v in d]
        x = np.arange(len(vs))
        for i, kind in enumerate(("probe", "ft")):
            means = [st.mean(d[v][kind]) if d[v][kind] else np.nan for v in vs]
            errs = [st.stdev(d[v][kind]) if len(d[v][kind]) > 1 else 0 for v in vs]
            ax.bar(x + (i - 0.5) * 0.38, means, 0.38, yerr=errs, capsize=3, label="linear probe" if kind == "probe" else "fine-tune")
        ax.set_xticks(x); ax.set_xticklabels(["crop + flip", "+ brightness/contrast\n+ solarize", "+ blur"][:len(vs)], fontsize=8)
        ax.set_ylabel("val accuracy %"); ax.set_title(f"{method.upper()} views"); ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8)
        lo = min(min(d[v][k]) for v in vs for k in ("probe", "ft") if d[v][k])
        ax.set_ylim(max(0, lo - 10), 100)
    fig.tight_layout(); fig.savefig(OUT / "aug_ablation.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    overfitting(); knn_pretraining(); aug_ablation()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
