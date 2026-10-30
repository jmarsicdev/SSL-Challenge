"""Throughput benchmark: how long is one SSL epoch on this GPU at each resolution? Used to size the budget."""
import sys, time, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import build_stores, build_ssl_loader
from src.ssl import build_ssl

t = time.time()
stores, classes = build_stores(Path("data"), Path("splits/val_split.json"))
print(f"cache load {time.time()-t:.1f}s, {len(stores['pretrain'])} imgs", flush=True)
dev = torch.device("cuda")
sizes = [int(s) for s in sys.argv[1:]] or [128, 160, 224]
for img in sizes:
    for method in ("simclr", "byol"):
        cfg = dict(model="resnet18", ssl=method, img_size=img, batch_size=256, num_workers=16, views="photoblur", lr=1e-3, aug="light")
        loader = build_ssl_loader(cfg, stores["pretrain"])
        ssl = build_ssl(cfg).to(dev).to(memory_format=torch.channels_last)
        opt = torch.optim.AdamW([p for p in ssl.parameters() if p.requires_grad], lr=1e-3)
        t = time.time(); nb = 0
        for b in loader:
            nb += 1
        tdata = time.time() - t
        t = time.time(); step = 0
        for ep in range(2):
            for b in loader:
                b = tuple(x.to(dev, non_blocking=True) for x in b)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, _ = ssl.loss(b)
                opt.zero_grad(); loss.backward(); opt.step(); ssl.after_step(step, 1000); step += 1
        torch.cuda.synchronize(); tt = (time.time() - t) / 2
        print(f"{method} {img}px: data-only epoch {tdata:.2f}s | train epoch {tt:.2f}s ({nb} steps) | "
              f"800 ep = {800*tt/60:.0f} min | mem {torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)
        del ssl, opt, loader; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
