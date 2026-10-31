# AI Usage

## What I used

**Claude Code** running as a terminal agent inside this repo, the same setup as the CNN Challenge.
I used it for the pipeline, running experiment queues while I was travelling, result tables, figures
and documentation drafts.

## How AI helped

1. **Setup and the training pipeline.** It ported the config-driven `train.py` / `evaluate.py` layout
   from the CNN Challenge and added the self-supervised pieces: an in-RAM grayscale image cache (JPEG
   decoding was the bottleneck for 800-epoch pretraining), two-view and rotation datasets, the RotNet /
   SimCLR / BYOL objectives in `src/ssl.py`, a kNN monitor on frozen features, and linear-probe and
   joint-training modes in the supervised loop. Every path was smoke-tested for 2–3 epochs and
   `evaluate.py` was checked to reproduce the training loop's validation number before any real run.

2. **Sizing the budget.** A throughput benchmark at 128 / 160 / 224 px showed the data loader, not the
   GPU, was the limit (an SSL epoch is about 1 s at 160 px and 3 s at 224 px). That is why every
   controlled experiment runs at 160 px and 224 px only appears during fine-tuning.

3. **Running controlled experiments unattended.** It turned the plan into config files and two detached
   queues (`scripts/run_queue.sh`), about 90 runs over six hours, and I followed progress and made
   decisions remotely.

4. **Tables and figures.** `scripts/summarize_runs.py` builds the mean ± std table in `docs/results.md`,
   `scripts/plot_curves.py` the learning curves, kNN-over-pretraining and augmentation figures, and
   `scripts/benchmark_latency.py` the inference-cost numbers. The report table comes from that output.

## Where the AI was wrong or not helpful

### 1. The time estimates
- **What it did:** From the benchmark it estimated 15 minutes per pretraining run and a 2:15 pm finish
  for the three-seed table.
- **Why it seemed fine:** The benchmark measured the training step directly.
- **What was actually wrong:** Real runs took 16–21 minutes (BYOL's EMA copy, the kNN monitor and
  worker start-up), and the SceneNet-S pretraining took 25 minutes because the depthwise network is
  slower per step than its MAC count suggests. The table finished about an hour later than first stated.
- **How I caught it:** I asked for an updated timetable twice; the second one was built from the queue
  log instead of the benchmark and held.
- **Fix:** Later estimates used measured run times from the queue log, and the second queue was started
  in parallel once it was clear the GPU was mostly idle during CPU-bound pretraining.

### 2. A latency number that didn't make sense
- **What it did:** The CPU batch-1 benchmark reported ResNet-18 as faster at 224 px (5.6 ms) than at
  160 px (9.4 ms).
- **Why it seemed fine:** The numbers came from a timing loop with warm-up and matched the GPU numbers'
  ordering for the small models.
- **What was actually wrong:** A model cannot be faster on twice the MACs. The CPU batch-1 timings are
  dominated by which convolution kernel the backend picks, not by the math, so they are not a clean
  cost measure.
- **How it was caught:** Re-running the benchmark in a different model order reproduced the inversion,
  so it was not noise from a busy machine.
- **Fix:** The README and the report use MMACs and GPU throughput as the cost measures and treat the
  CPU batch-1 timings as indicative only.

### 3. The "best recipe" credit
- **What it did:** The candidate sweep on top of the SimCLR init found +4 points from fine-tuning at
  224 px and +1.5 from a lower learning rate, and the first summary read as a +9 point SSL gain over
  the plain baseline.
- **Why it seemed fine:** Each knob was changed one at a time against the SSL-initialised model.
- **What was actually wrong:** Nothing in that sweep said how much the same knobs help a model trained
  from random init; the recipe and the resolution turned out to help a scratch model by about the same
  amount.
- **How I caught it:** The matched random-init control (same knobs, same resolution, same learning
  rate) was part of the plan from the start, the CNN Challenge habit, and it came in at 86.2 against 90.6.
- **Fix:** The report states the SSL gain at the final recipe as +4.4 (and +4.75 on test), with the
  control in the table, and the failure-analysis section is about exactly this.

### 4. Things it could not do
- It could not create the GitHub release. It also could not commit without a worktree, so the code
  lives on a branch that I merged myself.

## Decisions I made

- **The three questions.** The edge-model framing that I was interested in because it connects to my
  semester project (an edge classifier).
- **Keeping the CNN Challenge controls:** the same validation split, grayscale input, 3 seeds for every
  main-table arm, test evaluated once per reported number, and a matched random-init control for the
  final recipe.
- **Dropping the DINO-style piece**, because three objectives already gave the complexity spread and it
  was the piece I felt least confident about.
- **BYOL for the augmentation and edge arms**, as the label-free continuation of my CNN Challenge EMA
  teacher-student work, even though SimCLR won the ResNet-18 table by about a point. Noted as a
  limitation in the report.
- **Submitting the best-validation seed** and reporting the sibling seeds' test numbers next to it,
  rather than the seed that happened to score highest on test.
