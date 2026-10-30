"""Self-supervised objectives. Each wraps a backbone and exposes

    loss(batch) -> (loss, stats)     batch as yielded by data.build_ssl_loader
    after_step()                      EMA / centre updates (no-op where unused)
    backbone                          the encoder to transfer

Objectives
    rotnet  : predict the 90-degree rotation applied to the crop (Gidaris et al. 2018).
    simclr  : NT-Xent contrastive loss between two views, other images in the batch are negatives
              (Chen et al. 2020).
    byol    : online network predicts the EMA target network's projection of the other view,
              no negatives (Grill et al. 2020). Collapse is avoided by the predictor + EMA asymmetry.
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import build_backbone, mlp


class SSLBase(nn.Module):
    name = "base"

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg["model"], cfg.get("drop_path", 0.0))
        self.dim = self.backbone.num_features

    def after_step(self, step: int, total_steps: int):
        pass

    def extra_state(self) -> dict:
        return {}


# ---------------------------------------------------------------------------
class RotNet(SSLBase):
    name = "rotnet"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.head = nn.Linear(self.dim, 4)

    def loss(self, batch):
        x, k, _ = batch
        logits = self.head(self.backbone(x))
        loss = F.cross_entropy(logits, k)
        return loss, {"rot_acc": (logits.argmax(1) == k).float().mean().item()}


# ---------------------------------------------------------------------------
class SimCLR(SSLBase):
    name = "simclr"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.proj = mlp(self.dim, cfg.get("proj_hidden", 2048), cfg.get("proj_dim", 128), layers=2)
        self.temp = float(cfg.get("temperature", 0.2))

    def loss(self, batch):
        v1, v2, _ = batch
        n = v1.size(0)
        z = F.normalize(self.proj(self.backbone(torch.cat([v1, v2]))).float(), dim=1)
        sim = z @ z.t() / self.temp
        sim.fill_diagonal_(float("-inf"))
        target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
        loss = F.cross_entropy(sim, target)
        with torch.no_grad():
            pos = (sim.argmax(1) == target).float().mean().item()  # fraction where the positive is the nearest
        return loss, {"pos_top1": pos}


# ---------------------------------------------------------------------------
def cosine_momentum(base: float, step: int, total: int) -> float:
    return 1 - (1 - base) * (math.cos(math.pi * step / max(1, total)) + 1) / 2


class BYOL(SSLBase):
    name = "byol"

    def __init__(self, cfg):
        super().__init__(cfg)
        h, d = cfg.get("proj_hidden", 2048), cfg.get("proj_dim", 256)
        self.proj = mlp(self.dim, h, d, layers=2)
        self.pred = mlp(d, h, d, layers=2)
        self.t_backbone = copy.deepcopy(self.backbone)
        self.t_proj = copy.deepcopy(self.proj)
        for p in list(self.t_backbone.parameters()) + list(self.t_proj.parameters()):
            p.requires_grad = False
        self.base_m = float(cfg.get("ema_base", 0.99))
        self.m = self.base_m

    @torch.no_grad()
    def _target(self, x):
        return F.normalize(self.t_proj(self.t_backbone(x)).float(), dim=1)

    def _online(self, x):
        return F.normalize(self.pred(self.proj(self.backbone(x))).float(), dim=1)

    def loss(self, batch):
        v1, v2, _ = batch
        p = self._online(torch.cat([v1, v2]))
        with torch.no_grad():
            t = self._target(torch.cat([v2, v1]))  # swapped: predict the other view
        loss = (2 - 2 * (p * t).sum(1)).mean()
        with torch.no_grad():
            # per-dimension std of normalised targets: collapse -> 0, healthy -> ~1/sqrt(d)
            std = t.std(0).mean().item() * math.sqrt(t.size(1))
        return loss, {"target_std": std, "momentum": self.m}

    @torch.no_grad()
    def after_step(self, step, total_steps):
        self.m = cosine_momentum(self.base_m, step, total_steps)
        for o, t in zip(list(self.backbone.parameters()) + list(self.proj.parameters()),
                        list(self.t_backbone.parameters()) + list(self.t_proj.parameters())):
            t.mul_(self.m).add_(o.detach(), alpha=1 - self.m)
        for o, t in zip(list(self.backbone.buffers()) + list(self.proj.buffers()),
                        list(self.t_backbone.buffers()) + list(self.t_proj.buffers())):
            t.copy_(o)


METHODS = {"rotnet": RotNet, "simclr": SimCLR, "byol": BYOL}


def build_ssl(cfg: dict) -> SSLBase:
    return METHODS[cfg["ssl"]](cfg)


# ---------------------------------------------------------------------------
@torch.inference_mode()
def extract_features(backbone: nn.Module, loader, device) -> tuple[torch.Tensor, torch.Tensor]:
    backbone.eval()
    feats, ys = [], []
    for x, y in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            f = backbone(x.to(device, non_blocking=True))
        feats.append(F.normalize(f.float(), dim=1).cpu())
        ys.append(y)
    return torch.cat(feats), torch.cat(ys)


def knn_accuracy(train_f, train_y, val_f, val_y, k: int = 20, temp: float = 0.07, num_classes: int = 16) -> float:
    """Weighted kNN classifier on L2-normalised features (the standard SSL monitor)."""
    sim = val_f @ train_f.t()
    w, idx = sim.topk(k, dim=1)
    w = (w / temp).exp()
    votes = torch.zeros(val_f.size(0), num_classes).scatter_add_(1, train_y[idx], w)
    return (votes.argmax(1) == val_y).float().mean().item()
