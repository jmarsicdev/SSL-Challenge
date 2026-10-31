# SSL-Challenge

ITCS 6169/8169 Computer Vision, Assignment 2: self-supervised visual representation learning on the
16-class scene dataset (2,400 training images), with every model trained from random initialisation
on the provided images only.

**Final result: 87.00 % test accuracy** (348/400) with a ResNet-18 that was pretrained with SimCLR on
the 1,920 training-split images (no labels) and then fine-tuned at 224 px. Validation accuracy of the
recipe: 90.6 ± 0.2 % over 3 seeds; the identical recipe from random initialisation reaches 86.2 ± 0.1 %
validation / 82.25 % test. Plain supervised baseline: 80.4 ± 0.9 % validation.

| | |
|---|---|
| Final config | [`configs/best.yaml`](configs/best.yaml) (stage 2) on top of [`configs/ssl_simclr.yaml`](configs/ssl_simclr.yaml) (stage 1) |
| Final checkpoint | GitHub Release v1.0: `resnet18_simclr_ft224_s0.pt` (fine-tuned, 45 MB) and `resnet18_simclr_pretrain_s0.pt` (SSL encoder); SHA256 in [`docs/final/checkpoint.sha256`](docs/final/checkpoint.sha256) |
| Test metrics | [`docs/final/test_metrics_s0.json`](docs/final/test_metrics_s0.json), confusion matrix [`docs/final/test_confusion.png`](docs/final/test_confusion.png); sibling seeds `test_metrics_s1/s2.json`; random-init control `test_metrics_control_s0.json` |
| All experiments | [`docs/results.md`](docs/results.md) (generated, mean ± std over seeds) |
| Figures | [`docs/figures/`](docs/figures/): learning curves, kNN during pretraining, augmentation ablation |
| Inference cost | [`docs/latency.json`](docs/latency.json) |

## Rules compliance

- No externally pretrained weights. The only model constructor is `build_backbone` in
  `src/models.py`, which calls `timm.create_model(..., pretrained=False)` and asserts it. Nothing in
  the repository downloads weights.
- No external images. Self-supervised pretraining reads the 1,920 images of the training split only
  (labels are never loaded). The 480-image validation split is used for model selection, and the
  400-image labeled test set is evaluated once per reported number. `data/test2` is never used.
- RotNet, SimCLR and BYOL are re-implemented in `src/ssl.py`; the names refer to the training
  objectives, not to released checkpoints.

## Controls shared by every experiment

Same 1,920/480 split (`splits/val_split.json`, 30 validation images per class), grayscale input
replicated to 3 channels (15 of the 16 classes are stored as grayscale JPEGs; this removes the
colour shortcut for `Flower`), 160 px for every controlled comparison. One supervised recipe is used
for the baseline and for every fine-tune: AdamW lr 1e-3, weight decay 0.05, batch 64, 3 warm-up epochs
then cosine, RandomResizedCrop(0.35–1) + flip, 100 epochs, best-validation-epoch selection. Only the
initialisation differs between arms. One validation image is 0.21 %; the standard error at 85 % is
about 1.6 %, so main-table arms are run with 3 seeds and single-seed ablations are labeled as such.

## Self-supervised pretraining

800 epochs (12k updates) on the 1,920 training-split images, batch 128, AdamW 1e-3, cosine schedule,
two views per image: RandomResizedCrop(0.2–1) + flip + brightness/contrast jitter (p 0.8) + solarize
(p 0.2) + Gaussian blur (p 0.5). Colour jitter is reduced to brightness/contrast because the input is
grayscale. A weighted kNN classifier on frozen features (train split → validation) is logged every 25
epochs as a label-free-training / label-using-monitor of representation quality.

- **RotNet**: predict which of four 90° rotations was applied to the crop. Simplest pretext; scenes
  have a canonical orientation.
- **SimCLR**: NT-Xent contrastive loss (temperature 0.2) between the two views; the other 254 images
  in the batch are negatives. 2-layer projection MLP to 128-d.
- **BYOL**: an online network (encoder + projector + predictor) regresses the embedding of the other
  view produced by an EMA copy of itself (momentum 0.99 → 1, cosine). No negatives; the
  predictor + stop-gradient asymmetry prevents collapse.

## Results (validation, 480 images, mean ± std over 3 seeds)

### Required comparison (ResNet-18, 160 px)

| Method | Linear probe (frozen encoder) | Classifier |
|---|---|---|
| Supervised baseline, 100 epochs | – | 80.4 ± 0.9 |
| Supervised baseline, 300 epochs | – | 81.2 ± 0.1 |
| RotNet pretraining → fine-tune | 69.5 ± 0.3 | 84.9 ± 0.8 |
| SimCLR pretraining → fine-tune | 82.3 ± 0.6 | **86.0 ± 0.4** |
| BYOL pretraining → fine-tune | 83.2 ± 0.9 | 84.8 ± 0.9 |
| Joint supervised + BYOL (shared encoder, 300 epochs) | – | 83.1 ± 0.4 |
| Random encoder, linear probe (control) | 28.5 | – |
| **Best model** (SimCLR → fine-tune at 224 px, LS 0.1, EMA, 300 ep, lr 3e-4) | – | **90.6 ± 0.2** (test 87.0) |
| Random init with the best model's recipe (control) | – | 86.2 ± 0.1 (test 82.25) |

- SSL pretraining on the same 1,920 images is worth +3.6 to +5.6 points with the identical
  fine-tuning recipe; tripling the supervised schedule is worth +0.8, so the gain is not extra training.
- A linear layer on a frozen SimCLR/BYOL encoder (82–83 %) outperforms the whole network trained
  end-to-end on the labels (80.4 %).
- Objective complexity matters for the representation (RotNet probe 69.5 vs 82–83) but barely for the
  fine-tuned classifier (84.9 vs 84.8–86.0): fine-tuning erases most of the gap.
- Pretrain-then-fine-tune beats joint optimisation with the same objective (84.8 vs 83.1).
- kNN accuracy saturates by pretraining epoch ~200 (SimCLR) / ~400 (BYOL)
  (`docs/figures/knn_pretraining.png`); SSL-initialised fine-tunes reach 80 % by epoch 20 where scratch
  needs 60 (`docs/figures/overfitting.png`).

### Best-recipe ladder (seed 0 unless noted)

| Change on top of SimCLR → fine-tune (85.8) | Val |
|---|---|
| + label smoothing 0.1, EMA weights, 300 epochs (160 px) | 85.8 |
| + fine-tune lr 3e-4 | 87.3 |
| + fine-tune at 224 px (encoder pretrained at 160 px) | 89.8 (3 seeds: 90.2 ± 0.4) |
| + both | **90.8** (3 seeds: 90.6 ± 0.2) |
| same knobs, random init, 160 px | 84.0 |
| same knobs, random init, 224 px, lr 3e-4 | 86.0 (3 seeds: 86.2 ± 0.1) |

Test (one evaluation per number): final model 87.00 %; sibling seeds 85.00 / 87.50 %; random-init
control 82.25 %. Per-class test accuracy is 100 % for Flower, Forest, Highway, InsideCity and
Suburb and lowest for Kitchen (68 %), LivingRoom, Store and Industrial (72 %).

### Which augmentations matter on grayscale scenes? (seed 0, 160 px)

| Views | SimCLR probe / fine-tune | BYOL probe / fine-tune |
|---|---|---|
| crop + flip only | 78.8 / 84.6 | 78.1 / 85.4 |
| + brightness/contrast + solarize | 81.9 / 86.0 | 84.8 / 85.2 |
| + Gaussian blur (default) | 82.9 / 85.8 | 84.2 / 85.2 |

Cropping alone already yields a representation that fine-tunes to within ~1 point of the full
recipe; photometric distortion adds 3–7 probe points, blur adds nothing measurable. See
`docs/figures/aug_ablation.png`.

### Does the backbone change the benefit of SSL? Edge models (BYOL, 160 px, 3 seeds)

| Backbone | Params | MMACs @160 | GPU img/s (bf16, b64) | Supervised | BYOL → fine-tune | Gain |
|---|---|---|---|---|---|---|
| ResNet-18 | 11.2 M | 925 | 21.8k | 80.4 ± 0.9 | 84.8 ± 0.9 | +4.4 |
| SceneNet-S (custom) | 0.86 M | 92 | 54.7k | 82.5 ± 1.1 (300 ep: 83.1 ± 0.2) | 85.4 ± 1.1 | +2.9 |
| MobileNetV3-Small | 1.5 M | 29 | 53.2k | 75.1 ± 1.5 | 80.2 ± 2.0 | +5.2 |

All three backbones gain from SSL pretraining on 1,920 images. The 0.86 M-parameter SceneNet-S with
BYOL pretraining matches the SSL-pretrained ResNet-18 at 10× fewer MACs and 2.5× the GPU throughput.
Inference numbers are in `docs/latency.json` (`scripts/benchmark_latency.py`).

## Setup

Requires [uv](https://docs.astral.sh/uv/) and an NVIDIA GPU (developed on an RTX 5080, CUDA 12.8
wheels). Exact package versions are pinned in `uv.lock`.

```bash
git clone https://github.com/jmarsicdev/SSL-Challenge && cd SSL-Challenge
uv sync
source .venv/bin/activate
```

**Data.** Extract the handout dataset at the repo root (or symlink it) so that `data/train/<class>/`
and `data/test/<class>/` exist (2,400 and 400 images). `python scripts/inspect_dataset.py --data data`
prints the summary.

## Reproducing the final result

```bash
gh release download v1.0 -p '*.pt' -D checkpoints/          # or download from the Releases page
sha256sum -c docs/final/checkpoint.sha256
python evaluate.py --checkpoint checkpoints/resnet18_simclr_ft224_s0.pt --split test   # -> 87.00 %
```

Retraining (about 16 min + 8 min on an RTX 5080; GPU nondeterminism gives ±0.5 % on validation):

```bash
python pretrain.py --config configs/ssl_simclr.yaml --seed 0      # stage 1 -> runs/ssl_simclr/s0/ssl.pt
python train.py    --config configs/best.yaml       --seed 0      # stage 2 -> runs/best/s0/best.pt
python evaluate.py --checkpoint runs/best/s0/best.pt --split test
```

## Running any experiment

```bash
python pretrain.py --config configs/ssl_byol.yaml   --seed 0      # SSL pretraining -> runs/ssl_byol/s0/ssl.pt
python train.py    --config configs/ft_byol.yaml    --seed 0      # fine-tune from the seed-0 SSL weights
python train.py    --config configs/probe_byol.yaml --seed 0      # frozen-encoder linear probe
python train.py    --config configs/sup_baseline.yaml --seed 0    # supervised baseline from scratch
python train.py    --config configs/joint_byol.yaml --seed 0      # joint CE + BYOL, shared encoder
python scripts/summarize_runs.py                                  # regenerates docs/results.md
python scripts/plot_curves.py                                     # regenerates docs/figures/
scripts/run_queue.sh <name> queues/<file>                         # sequential queue (queues/ holds the ones used)
```

Configs inherit via `_base_`; `init: runs/<ssl-run>/s{seed}/ssl.pt` ties fine-tune seed k to
pretraining seed k. Every run writes `config.json`, `env.json`, `history.csv`, `metrics.json`
and a checkpoint under `runs/<config>/s<seed>/`.

## Layout

```
configs/         one YAML per experiment: sup_* baselines, ssl_* pretraining, probe_*, ft_*, joint_*, best_*
src/data.py      in-RAM grayscale image cache, supervised / SSL-view / rotation datasets, split loading
src/models.py    random-init backbones (timm ResNet-18, MobileNetV3-Small; SceneNet-S), classifier, heads
src/ssl.py       RotNet, SimCLR (NT-Xent), BYOL objectives; kNN monitor
src/engine.py    supervised loop: fine-tune / linear probe / joint CE+SSL, EMA weights, label smoothing
pretrain.py      self-supervised pretraining  -> runs/<name>/s<seed>/ssl.pt
train.py         supervised training from random or SSL init -> runs/<name>/s<seed>/best.pt
evaluate.py      labeled-split evaluation (test used once per reported number)
scripts/         summarize_runs, plot_curves, benchmark_latency, run_queue, dataset inspection, splits
splits/          committed 1,920/480 train/val split
docs/            generated results table, figures, latency, final test metrics
```
