"""Inference cost per backbone: parameters, MACs, GPU throughput (batch 64, bf16) and CPU batch-1 latency
(fp32, 4 threads, an edge-device proxy).

    python scripts/benchmark_latency.py --models resnet18 scenenet_s mobilenetv3_small_100 --img-size 160 224
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models import Classifier, build_backbone, count_params


def macs(model, x):
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as fc:
            model(x)
        return fc.get_total_flops() / 2 / 1e6
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["resnet18", "scenenet_s", "mobilenetv3_small_100"])
    ap.add_argument("--img-size", nargs="+", type=int, default=[160])
    ap.add_argument("--out", default="docs/latency.json")
    ap.add_argument("--cpu-threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.cpu_threads)
    rows = []
    for name in args.models:
        for s in args.img_size:
            m = Classifier(build_backbone(name), 16).eval()
            x1 = torch.randn(1, 3, s, s)
            with torch.inference_mode():
                mm = macs(m, x1)
                for _ in range(5): m(x1)
                t = time.perf_counter()
                for _ in range(30): m(x1)
                cpu_ms = (time.perf_counter() - t) / 30 * 1000
                gpu_ips = float("nan")
                if torch.cuda.is_available():
                    mg = m.cuda().to(memory_format=torch.channels_last)
                    xb = torch.randn(64, 3, s, s, device="cuda").contiguous(memory_format=torch.channels_last)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        for _ in range(10): mg(xb)
                        torch.cuda.synchronize(); t = time.perf_counter()
                        for _ in range(50): mg(xb)
                        torch.cuda.synchronize()
                    gpu_ips = 50 * 64 / (time.perf_counter() - t)
            row = {"model": name, "img_size": s, "params_M": count_params(m) / 1e6, "MMACs": mm,
                   "cpu_batch1_ms": cpu_ms, "gpu_img_per_s_bf16_b64": gpu_ips}
            rows.append(row)
            print(f"{name:24s} {s}px  {row['params_M']:.2f}M params  {mm:7.0f} MMACs  CPU b1 {cpu_ms:6.1f} ms  GPU {gpu_ips:8.0f} img/s", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
