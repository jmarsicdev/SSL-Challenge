"""Supervised training / evaluation loops (baseline, fine-tuning, linear probe, joint SSL+CE)."""
from __future__ import annotations

import copy
import csv
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.utils import ModelEmaV3
from torch.utils.tensorboard import SummaryWriter

from .models import param_groups


def build_optimizer(cfg, model, lr_key="lr"):
    groups = param_groups(model, cfg[lr_key], cfg.get("weight_decay", 0.0), cfg.get("head_lr_mult", 1.0))
    name = cfg.get("optimizer", "adamw").lower()
    if name == "adamw":
        return torch.optim.AdamW(groups, betas=(0.9, 0.999))
    if name == "sgd":
        return torch.optim.SGD(groups, momentum=0.9, nesterov=True)
    raise ValueError(name)


def build_scheduler(cfg, optimizer, steps_per_epoch):
    """Per-step linear warmup then cosine to ~0."""
    if cfg.get("scheduler", "cosine") == "none":
        return None
    total = cfg["epochs"] * steps_per_epoch
    warm = int(cfg.get("warmup_epochs", 1) * steps_per_epoch)

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / max(1, warm)
        p = (step - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.inference_mode()
def evaluate(model, loader, device, tta: bool = False):
    model.eval()
    losses, preds, targets, probs = [], [], [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(x)
            if tta:
                logits = (logits.softmax(1) + model(torch.flip(x, dims=[3])).softmax(1)).log()
        logits = logits.float()
        losses.append(F.cross_entropy(logits, y, reduction="sum").item())
        probs.append(logits.softmax(1).cpu()); preds.append(logits.argmax(1).cpu()); targets.append(y.cpu())
    preds, targets, probs = torch.cat(preds), torch.cat(targets), torch.cat(probs)
    return {"loss": sum(losses) / len(targets), "acc": (preds == targets).float().mean().item(),
            "preds": preds.numpy(), "targets": targets.numpy(), "probs": probs.numpy()}


def per_class_accuracy(preds, targets, num_classes):
    return [float((preds[targets == c] == c).mean()) if (targets == c).any() else float("nan")
            for c in range(num_classes)]


def train(cfg: dict, model: nn.Module, train_loader, val_loader, device, run_dir: Path, log=print,
          ssl_module=None):
    """Supervised loop. If `ssl_module` is given, `train_loader` yields (x_sup, v1, v2, y) and the loss is
    CE(x_sup) + ssl_weight * ssl_module.loss((v1, v2, y)); the SSL module shares model.backbone."""
    model.to(device)
    if ssl_module is not None:
        ssl_module.to(device)
        # optimise classifier + SSL heads (+ shared backbone once)
        extra = [p for n, p in ssl_module.named_parameters() if not n.startswith("backbone.") and p.requires_grad]
        opt = build_optimizer(cfg, model)
        if extra:
            opt.add_param_group({"params": extra, "lr": cfg["lr"], "weight_decay": cfg.get("weight_decay", 0.0),
                                 "name": "ssl_heads"})
    else:
        opt = build_optimizer(cfg, model)
    sched = build_scheduler(cfg, opt, len(train_loader))
    epochs, num_classes = cfg["epochs"], cfg["num_classes"]
    total_steps = epochs * len(train_loader)
    frozen = cfg.get("freeze_backbone", False)

    mix = None
    if cfg.get("mixup", 0.0) > 0 or cfg.get("cutmix", 0.0) > 0:
        mix = Mixup(mixup_alpha=cfg.get("mixup", 0.0), cutmix_alpha=cfg.get("cutmix", 0.0),
                    label_smoothing=cfg.get("label_smoothing", 0.0), num_classes=num_classes)
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))

    ema = ModelEmaV3(model, decay=cfg["ema_decay"]) if cfg.get("ema_decay") else None
    use_amp = device.type == "cuda" and cfg.get("amp", True)
    ssl_w = float(cfg.get("ssl_weight", 1.0))
    ssl_warm = int(cfg.get("ssl_warmup_epochs", 0) * len(train_loader))

    best = {"acc": -1.0, "epoch": 0, "state": None, "val_loss": None}
    fields = ["epoch", "lr", "train_loss", "train_acc", "ssl_loss", "val_loss", "val_acc", "ema_val_acc", "sec"]
    hist_path = run_dir / "history.csv"
    with hist_path.open("w", newline="") as f:
        csv.DictWriter(f, fields).writeheader()
    tb = SummaryWriter(log_dir=str(run_dir / "tb"))
    eval_every = int(cfg.get("eval_every", 1))
    t0, step = time.time(), 0
    for epoch in range(1, epochs + 1):
        model.train()
        if frozen:
            model.backbone.eval()  # linear probe: BN statistics of the frozen encoder stay fixed
        if ssl_module is not None:
            ssl_module.train()
        tot_loss = tot_ssl = 0.0
        tot_correct = n = 0
        te = time.time()
        for batch in train_loader:
            if ssl_module is not None:
                x, v1, v2, y = (b.to(device, non_blocking=True) for b in batch)
            else:
                x, y = (b.to(device, non_blocking=True) for b in batch)
            y_in = y
            if mix is not None:
                x, y_in = mix(x, y)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y_in)
                if ssl_module is not None:
                    ssl_loss, _ = ssl_module.loss((v1, v2, y))
                    w = ssl_w * (min(1.0, step / ssl_warm) if ssl_warm else 1.0)
                    loss = loss + w * ssl_loss
                    tot_ssl += ssl_loss.item() * y.size(0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            if sched is not None:
                sched.step()
            if ema is not None:
                ema.update(model)
            if ssl_module is not None:
                ssl_module.after_step(step, total_steps)
            step += 1
            tot_loss += loss.item() * y.size(0)
            tot_correct += (logits.argmax(1) == y).sum().item()
            n += y.size(0)

        val = ema_val = None
        if (epoch % eval_every == 0) or epoch == epochs:
            val = evaluate(model, val_loader, device)
            ema_val = evaluate(ema.module, val_loader, device) if ema is not None else None
            sel_acc, sel_loss = (ema_val["acc"], ema_val["loss"]) if ema is not None else (val["acc"], val["loss"])
            if sel_acc > best["acc"]:
                src = ema.module if ema is not None else model
                best = {"acc": sel_acc, "epoch": epoch, "val_loss": sel_loss,
                        "state": copy.deepcopy(src.state_dict())}

        row = {"epoch": epoch, "lr": opt.param_groups[0]["lr"], "train_loss": tot_loss / n,
               "train_acc": tot_correct / n, "ssl_loss": tot_ssl / n if ssl_module is not None else "",
               "val_loss": val["loss"] if val else "", "val_acc": val["acc"] if val else "",
               "ema_val_acc": ema_val["acc"] if ema_val else "", "sec": time.time() - te}
        with hist_path.open("a", newline="") as f:
            csv.DictWriter(f, fields).writerow(row)
        for k in ("train_loss", "train_acc", "lr"):
            tb.add_scalar(k, row[k], epoch)
        if val:
            tb.add_scalar("val_loss", val["loss"], epoch); tb.add_scalar("val_acc", val["acc"], epoch)
            tb.add_scalar("gap/train_minus_val_acc", row["train_acc"] - val["acc"], epoch)
        if ema_val:
            tb.add_scalar("ema_val_acc", ema_val["acc"], epoch)
        tb.flush()
        if val or epoch % 10 == 0:
            msg = f"ep {epoch:03d}/{epochs} lr {row['lr']:.2e} | train loss {row['train_loss']:.3f} acc {row['train_acc']:.3f}"
            if ssl_module is not None:
                msg += f" ssl {row['ssl_loss']:.3f}"
            if val:
                msg += f" | val loss {val['loss']:.3f} acc {val['acc']:.4f}" + (f" | ema {ema_val['acc']:.4f}" if ema_val else "")
            log(msg + f" | {row['sec']:.1f}s")

    best["minutes"] = (time.time() - t0) / 60
    tb.close()
    return best
