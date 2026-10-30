"""Datasets and transform presets.

Images are decoded ONCE into RAM (2,400 images at <=288 px is ~0.2 GB as uint8),
so long SSL schedules are not bottlenecked by JPEG decoding. Everything is
converted to grayscale and replicated to 3 channels (the Flower colour shortcut
from Assignment 1: 15 of 16 classes are stored as grayscale JPEGs).

Supervised augmentation presets (train only)
    none   : eval transform (resize shorter side, centre crop)
    light  : RandomResizedCrop(scale 0.35-1) + horizontal flip        (A1 recipe)
    medium : light + TrivialAugmentWide
    strong : medium + RandomErasing

SSL view presets (two independent views per image)
    crop      : RandomResizedCrop(scale 0.2-1) + flip                  (geometry only)
    photo     : crop + brightness/contrast jitter + random solarize    (+ photometric)
    photoblur : photo + Gaussian blur p=0.5                            (SimCLR/BYOL style)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MAX_SIDE = 288  # cache resolution: native detail tops out at 256 px; 288 keeps a margin for crops


def list_classes(train_root: Path) -> list[str]:
    return sorted(p.name for p in train_root.iterdir() if p.is_dir())


# ---------------------------------------------------------------------------
# in-memory image store
# ---------------------------------------------------------------------------
_CACHE: dict[str, np.ndarray] = {}


def load_gray(path: Path) -> np.ndarray:
    """Decode to single-channel uint8 HxW, downscaling so the longer side <= MAX_SIDE."""
    key = str(path)
    arr = _CACHE.get(key)
    if arr is None:
        with Image.open(path) as im:
            im = im.convert("L")
            w, h = im.size
            s = MAX_SIDE / max(w, h)
            if s < 1:
                im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
            arr = np.asarray(im, dtype=np.uint8).copy()
        _CACHE[key] = arr
    return arr


def to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr, mode="L").convert("RGB")


class CachedImages:
    """List of (relative_path, label) with eager decoding into RAM."""

    def __init__(self, root: Path, items: list[tuple[str, int]]):
        self.root = Path(root)
        self.items = items
        self.arrays = [load_gray(self.root / rel) for rel, _ in items]

    def __len__(self):
        return len(self.items)

    @property
    def labels(self) -> list[int]:
        return [y for _, y in self.items]


# ---------------------------------------------------------------------------
# transforms
# ---------------------------------------------------------------------------
def eval_transform(img_size: int) -> T.Compose:
    return T.Compose([T.Resize(int(round(img_size / 0.875))), T.CenterCrop(img_size),
                      T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def sup_transform(img_size: int, aug: str, rrc_scale=(0.35, 1.0)) -> T.Compose:
    if aug == "none":
        return eval_transform(img_size)
    ops: list = [T.RandomResizedCrop(img_size, scale=tuple(rrc_scale)), T.RandomHorizontalFlip()]
    if aug in ("medium", "strong"):
        ops.append(T.TrivialAugmentWide())
    elif aug != "light":
        raise ValueError(f"unknown aug preset {aug}")
    ops += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if aug == "strong":
        ops.append(T.RandomErasing(p=0.25))
    return T.Compose(ops)


def ssl_view_transform(img_size: int, preset: str, rrc_scale=(0.2, 1.0), blur_p: float = 0.5) -> T.Compose:
    """One SSL view. Colour jitter is reduced to brightness/contrast because the input is gray."""
    ops: list = [T.RandomResizedCrop(img_size, scale=tuple(rrc_scale), interpolation=InterpolationMode.BICUBIC),
                 T.RandomHorizontalFlip()]
    if preset in ("photo", "photoblur"):
        ops += [T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4)], p=0.8),
                T.RandomSolarize(threshold=128, p=0.2)]
    elif preset != "crop":
        raise ValueError(f"unknown ssl view preset {preset}")
    if preset == "photoblur":
        k = max(3, int(0.1 * img_size) // 2 * 2 + 1)
        ops.append(T.RandomApply([T.GaussianBlur(k, sigma=(0.1, 2.0))], p=blur_p))
    ops += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(ops)


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------
class SupDataset(Dataset):
    """(image, label)."""

    def __init__(self, store: CachedImages, transform):
        self.store, self.transform = store, transform

    def __len__(self):
        return len(self.store)

    def __getitem__(self, i):
        return self.transform(to_pil(self.store.arrays[i])), self.store.items[i][1]


class TwoViewDataset(Dataset):
    """(view1, view2, label). The label is only used for diagnostics (kNN monitor) or joint training."""

    def __init__(self, store: CachedImages, tf_view, tf_sup=None):
        self.store, self.tf_view, self.tf_sup = store, tf_view, tf_sup

    def __len__(self):
        return len(self.store)

    def __getitem__(self, i):
        im = to_pil(self.store.arrays[i])
        y = self.store.items[i][1]
        if self.tf_sup is not None:  # joint training: a supervised view + two SSL views
            return self.tf_sup(im), self.tf_view(im), self.tf_view(im), y
        return self.tf_view(im), self.tf_view(im), y


class RotationDataset(Dataset):
    """(image rotated by k*90 deg, k, label). Rotation applied after the crop so the pretext is not trivial."""

    def __init__(self, store: CachedImages, img_size: int, rrc_scale=(0.35, 1.0)):
        self.store = store
        self.tf = T.Compose([T.RandomResizedCrop(img_size, scale=tuple(rrc_scale)), T.RandomHorizontalFlip(),
                             T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    def __len__(self):
        return len(self.store)

    def __getitem__(self, i):
        x = self.tf(to_pil(self.store.arrays[i]))
        k = int(torch.randint(0, 4, (1,)))
        return torch.rot90(x, k, dims=(1, 2)), k, self.store.items[i][1]


# ---------------------------------------------------------------------------
# split handling
# ---------------------------------------------------------------------------
def folder_items(split_root: Path, classes: list[str]) -> list[tuple[str, int]]:
    c2i = {c: i for i, c in enumerate(classes)}
    out = []
    for c in classes:
        for f in sorted((split_root / c).iterdir()):
            if f.suffix.lower() in IMG_EXT:
                out.append((f"{c}/{f.name}", c2i[c]))
    return out


def load_split(split_file: Path, classes: list[str]):
    d = json.loads(Path(split_file).read_text())
    c2i = {c: i for i, c in enumerate(classes)}
    to_items = lambda rels: [(r, c2i[r.split("/")[0]]) for r in rels]
    return to_items(d["train"]), to_items(d["val"])


def load_label_split(path: Path, classes: list[str]):
    d = json.loads(Path(path).read_text())
    c2i = {c: i for i, c in enumerate(classes)}
    to_items = lambda rels: [(r, c2i[r.split("/")[0]]) for r in rels]
    return to_items(d["labeled"]), to_items(d["unlabeled"])


def make_loader(ds, batch_size: int, shuffle: bool, num_workers: int, drop_last: bool = False):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0,
                      drop_last=drop_last, prefetch_factor=4 if num_workers > 0 else None)


def build_stores(data_root: Path, split_file: Path, label_split: Path | None = None):
    """Returns dict(train=CachedImages, val=..., test=..., [unlabeled=...]), classes.

    With `label_split`, `train` holds only the labeled subset (low-label experiments) and
    `unlabeled` the rest of the training split. SSL pretraining uses `pretrain`, which is
    always the full 1,920-image training split (labels never read)."""
    classes = list_classes(data_root / "train")
    tr_items, va_items = load_split(split_file, classes)
    stores = {"val": CachedImages(data_root / "train", va_items),
              "test": CachedImages(data_root / "test", folder_items(data_root / "test", classes)),
              "pretrain": CachedImages(data_root / "train", tr_items)}
    if label_split is not None:
        lab, unl = load_label_split(label_split, classes)
        stores["train"] = CachedImages(data_root / "train", lab)
        stores["unlabeled"] = CachedImages(data_root / "train", unl)
    else:
        stores["train"] = stores["pretrain"]
    return stores, classes


def build_sup_loaders(cfg: dict, stores: dict):
    """Supervised / fine-tune / linear-probe loaders: (train, val, test)."""
    nw, bs, s = cfg.get("num_workers", 8), cfg["batch_size"], cfg["img_size"]
    tf_train = sup_transform(s, cfg["aug"], cfg.get("rrc_scale", (0.35, 1.0)))
    tf_eval = eval_transform(s)
    train = make_loader(SupDataset(stores["train"], tf_train), bs, True, nw)
    val = make_loader(SupDataset(stores["val"], tf_eval), bs * 2, False, nw)
    test = make_loader(SupDataset(stores["test"], tf_eval), bs * 2, False, nw)
    return train, val, test


def build_eval_loader(cfg: dict, store: CachedImages):
    return make_loader(SupDataset(store, eval_transform(cfg["img_size"])), cfg["batch_size"] * 2, False,
                       cfg.get("num_workers", 8))


def build_ssl_loader(cfg: dict, store: CachedImages, with_sup_view: bool = False, batch_size: int | None = None):
    """Two-view (or rotation) loader for self-supervised pretraining / joint training."""
    nw, bs, s = cfg.get("num_workers", 8), batch_size or cfg["batch_size"], cfg["img_size"]
    if cfg["ssl"] == "rotnet":
        ds = RotationDataset(store, s, cfg.get("rrc_scale", (0.35, 1.0)))
    else:
        tf_view = ssl_view_transform(s, cfg.get("views", "photoblur"), cfg.get("ssl_rrc_scale", (0.2, 1.0)))
        tf_sup = sup_transform(s, cfg["aug"], cfg.get("rrc_scale", (0.35, 1.0))) if with_sup_view else None
        ds = TwoViewDataset(store, tf_view, tf_sup)
    return make_loader(ds, bs, True, nw, drop_last=True)
