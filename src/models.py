"""Backbones (always randomly initialised: the assignment forbids external pretraining),
classifier wrapper, projection heads, and loading of self-supervised checkpoints."""
from __future__ import annotations

import torch
import torch.nn as nn
import timm


class ConvBNAct(nn.Sequential):
    def __init__(self, cin, cout, k=3, s=1, groups=1):
        super().__init__(nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False),
                         nn.BatchNorm2d(cout), nn.SiLU(inplace=True))


class DSBlock(nn.Module):
    """Depthwise-separable inverted-residual block (MobileNet-style), optional stride-2 downsample."""

    def __init__(self, cin, cout, stride=1, expand=4):
        super().__init__()
        mid = cin * expand
        self.body = nn.Sequential(ConvBNAct(cin, mid, 1), ConvBNAct(mid, mid, 3, stride, groups=mid),
                                  nn.Conv2d(mid, cout, 1, bias=False), nn.BatchNorm2d(cout))
        self.skip = stride == 1 and cin == cout

    def forward(self, x):
        y = self.body(x)
        return x + y if self.skip else y


class SceneNetS(nn.Module):
    """0.86M-parameter edge CNN from the Assignment 1 distillation study (same architecture, random init):
    stem /4, four stages of depthwise-separable blocks (/4, /8, /16, /32), 512-d head conv, global pool.
    Exposes `num_features` like a timm backbone; forward returns the pooled 512-d feature."""

    def __init__(self, widths=(24, 48, 96, 192), depths=(2, 3, 4, 2)):
        super().__init__()
        self.stem = nn.Sequential(ConvBNAct(3, 16, 3, 2), ConvBNAct(16, widths[0], 3, 2))
        stages, cin = [], widths[0]
        for i, (w, d) in enumerate(zip(widths, depths)):
            blocks = []
            for j in range(d):
                blocks.append(DSBlock(cin, w, 2 if (j == 0 and i > 0) else 1))
                cin = w
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)
        self.head_conv = ConvBNAct(widths[3], 512, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.num_features = 512

    def forward(self, x):
        return self.pool(self.head_conv(self.stages(self.stem(x)))).flatten(1)


def build_backbone(name: str, drop_path: float = 0.0) -> nn.Module:
    """Randomly initialised encoder: forward(x) -> pooled feature vector, `num_features` attribute."""
    if name == "scenenet_s":
        return SceneNetS()
    kwargs = dict(pretrained=False, num_classes=0, in_chans=3)
    if drop_path:
        kwargs["drop_path_rate"] = drop_path
    assert kwargs["pretrained"] is False, "assignment rule: no externally pretrained weights"
    bb = timm.create_model(name, **kwargs)
    assert hasattr(bb, "num_features")
    return bb


class Classifier(nn.Module):
    """backbone -> (dropout) -> linear."""

    def __init__(self, backbone: nn.Module, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.backbone = backbone
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(backbone.num_features, num_classes)

    def forward(self, x):
        return self.fc(self.drop(self.backbone(x)))

    def features(self, x):
        return self.backbone(x)

    def get_classifier(self):
        return self.fc


def mlp(in_dim: int, hidden: int, out: int, layers: int = 2, last_bn: bool = False) -> nn.Sequential:
    """SimCLR/BYOL-style projection or prediction MLP: Linear-BN-ReLU blocks then Linear."""
    mods: list[nn.Module] = []
    d = in_dim
    for _ in range(layers - 1):
        mods += [nn.Linear(d, hidden, bias=False), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True)]
        d = hidden
    mods.append(nn.Linear(d, out, bias=not last_bn))
    if last_bn:
        mods.append(nn.BatchNorm1d(out, affine=False))
    return nn.Sequential(*mods)


def build_classifier(cfg: dict, num_classes: int) -> nn.Module:
    """Random init, or backbone weights copied from a self-supervised checkpoint (cfg['init'])."""
    bb = build_backbone(cfg["model"], cfg.get("drop_path", 0.0))
    model = Classifier(bb, num_classes, cfg.get("dropout", 0.0))
    init = cfg.get("init")
    if init:
        load_ssl_backbone(model.backbone, init)
    if cfg.get("freeze_backbone", False):
        for p in model.backbone.parameters():
            p.requires_grad = False
    return model


def load_ssl_backbone(backbone: nn.Module, ckpt_path: str) -> dict:
    """Copy backbone weights out of a pretrain.py checkpoint. Returns the pretraining config."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["backbone"]
    missing, unexpected = backbone.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    assert not missing, missing
    return ck.get("config", {})


def param_groups(model: nn.Module, lr: float, weight_decay: float, head_lr_mult: float = 1.0):
    """No weight decay on biases / norm params; optional separate LR for the classifier head."""
    head_ids = {id(p) for p in model.get_classifier().parameters()} if hasattr(model, "get_classifier") else set()
    buckets: dict[tuple[str, bool], list] = {}
    for p in model.parameters():
        if not p.requires_grad:
            continue
        key = ("head" if id(p) in head_ids else "body", p.ndim > 1)
        buckets.setdefault(key, []).append(p)
    out = []
    for (part, decay), ps in buckets.items():
        out.append({"params": ps, "lr": lr * (head_lr_mult if part == "head" else 1.0),
                    "weight_decay": weight_decay if decay else 0.0, "name": f"{part}_{'decay' if decay else 'nodecay'}"})
    return out


def count_params(m: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in m.parameters() if (p.requires_grad or not trainable_only))
