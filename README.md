# SSL-Challenge

ITCS 6169/8169 Computer Vision, Assignment 2: self-supervised visual representation learning on the
16-class scene dataset (2,400 training images), with every model trained from random initialisation
on the provided images only.

**Final result: TBD % test accuracy** (TBD/400). Details in the results section below.

| | |
|---|---|
| Final config | `configs/best.yaml` (TBD) |
| Final checkpoint | GitHub Release (TBD) |
| All experiments | [`docs/results.md`](docs/results.md) (generated, mean ± std over seeds) |
| Figures | [`docs/figures/`](docs/figures/) |

## Rules compliance

- No externally pretrained weights. The only model constructor is `build_backbone` in
  `src/models.py`, which calls `timm.create_model(..., pretrained=False)` and asserts it. Nothing in
  the repository downloads weights.
- No external images. Self-supervised pretraining reads the 1,920 images of the training split only
  (labels are never loaded). The 480-image validation split is used for model selection, and the
  400-image labeled test set is evaluated once per reported number. `data/test2` is never used.
- Methods (RotNet, SimCLR, BYOL) are re-implemented in `src/ssl.py`; their names refer to the
  training objectives, not to released checkpoints.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and an NVIDIA GPU (developed on an RTX 5080, CUDA 12.8
wheels). Exact package versions are pinned in `uv.lock`.

```bash
git clone https://github.com/jmarsicdev/SSL-Challenge && cd SSL-Challenge
uv sync
source .venv/bin/activate
```

**Data.** Extract the handout dataset at the repo root (or symlink it) so that `data/train/<class>/`
and `data/test/<class>/` exist. `python scripts/inspect_dataset.py --data data` prints the summary.

## Reproducing the final result

```bash
# evaluate the released checkpoint on the 400-image test set
python evaluate.py --checkpoint checkpoints/<final>.pt --split test
```

Retraining the final model is a two-stage recipe (self-supervised pretraining, then supervised
fine-tuning from those weights); the exact commands are listed in the results section once final.

## Running any experiment

```bash
python pretrain.py --config configs/ssl_byol.yaml   --seed 0      # SSL pretraining -> runs/ssl_byol/s0/ssl.pt
python train.py    --config configs/ft_byol.yaml    --seed 0      # fine-tune from the seed-0 SSL weights
python train.py    --config configs/probe_byol.yaml --seed 0      # frozen-encoder linear probe
python train.py    --config configs/sup_baseline.yaml --seed 0    # supervised baseline from scratch
python train.py    --config configs/joint_byol.yaml --seed 0      # joint CE + BYOL, shared encoder
python scripts/summarize_runs.py                                  # regenerates docs/results.md
python scripts/plot_curves.py                                     # regenerates docs/figures/
scripts/run_queue.sh <name> queues/<file>                         # sequential queue of the above
```

Configs inherit via `_base_`; `init: runs/<ssl-run>/s{seed}/ssl.pt` ties fine-tune seed k to
pretraining seed k. Every run writes `config.json`, `env.json`, `history.csv`, `metrics.json`
and a checkpoint under `runs/<config>/s<seed>/`.

## Layout

```
configs/         one YAML per experiment (sup_* baselines, ssl_* pretraining, probe_*, ft_*, joint_*)
src/data.py      in-RAM grayscale image cache, supervised / SSL-view / rotation datasets, split loading
src/models.py    random-init backbones (timm ResNet-18, MobileNetV3-Small; SceneNet-S), classifier, heads
src/ssl.py       RotNet, SimCLR (NT-Xent), BYOL objectives; kNN monitor
src/engine.py    supervised loop: fine-tune / linear probe / joint CE+SSL, EMA weights, label smoothing
pretrain.py      self-supervised pretraining  -> runs/<name>/s<seed>/ssl.pt
train.py         supervised training from random or SSL init -> runs/<name>/s<seed>/best.pt
evaluate.py      labeled-split evaluation (test used once per reported number)
scripts/         summarize_runs, plot_curves, benchmark_latency, run_queue, dataset inspection, splits
splits/          committed 1,920/480 train/val split
docs/            generated results table, figures, final test metrics
```
