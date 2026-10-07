"""Pilot: simulator speed, healthy-tissue check and inducibility per S2 interval. usage: pilot.py N_TISSUES SIZE"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reentrynet.sim import run_protocol  # noqa: E402
from reentrynet.tissue import make_tissue  # noqa: E402

n, size = int(sys.argv[1]), int(sys.argv[2])
dev = "cuda"
cis = [float(x) for x in range(40, 180, 10)]
rng = np.random.default_rng(0)
healthy = [np.ones((size, size), np.float32), np.clip(1 - 0.2 * rng.random((size, size)), 0, 1).astype(np.float32)]
fams = ["blobs"] * (n // 2) + ["strokes"] * (n - n // 2)
tissues = healthy + [make_tissue(rng, (size, size), f) for f in fams]
c = torch.from_numpy(np.stack(tissues)).to(dev)
torch.cuda.synchronize()
t0 = time.time()
out = run_protocol(c, cis)
torch.cuda.synchronize()
el = time.time() - t0
steps = int(round((max(cis) + 450) / 0.05))
print(f"{len(tissues)} tissues x {len(cis)} intervals at {size}x{size}: {el:.1f}s, {1e3 * el / steps:.2f} ms/step")
s = out["sustained"].cpu().numpy()
print("healthy:", s[0].any(), "| healthy+noise:", s[1].any())
for k, ci in enumerate(cis):
    print(f"  S2 at {ci:5.0f}: blobs {s[2:2 + n // 2, k].mean():.2f}  strokes {s[2 + n // 2:, k].mean():.2f}")
ind = out["inducible"].cpu().numpy()
print(f"inducible: blobs {ind[2:2 + n // 2].mean():.2f}, strokes {ind[2 + n // 2:].mean():.2f}, all {ind[2:].mean():.2f}")
scar_frac = (c[2:] == 0).float().mean((1, 2)).cpu().numpy()
print("scar fraction quartiles", np.round(np.quantile(scar_frac, [0.25, 0.5, 0.75]), 3),
      "| inducible vs scar-fraction corr", np.round(np.corrcoef(scar_frac, ind[2:])[0, 1], 3))
np.savez_compressed(sys.argv[3] if len(sys.argv) > 3 else "pilot.npz", c=c.cpu().numpy(), sustained=s,
                    reentry=out["reentry_map"].cpu().numpy(), act=out["s1_act"].cpu().numpy())
