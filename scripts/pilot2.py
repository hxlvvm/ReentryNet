"""Pilot 2: inducibility on a fine S2 grid, and label stability under dt and S2-site changes."""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reentrynet.sim import run_protocol  # noqa: E402
from reentrynet.tissue import make_tissue  # noqa: E402

n = int(sys.argv[1])
cis = [30.0, 35.0, 40.0, 45.0, 50.0, 55.0]
rng = np.random.default_rng(1)
fams = ["blobs"] * (n // 2) + ["strokes"] * (n - n // 2)
c = torch.from_numpy(np.stack([make_tissue(rng, (128, 128), f) for f in fams])).cuda()
runs = {}
for name, kw in {"base": {}, "dt_half": {"dt": 0.025}, "s2_shift3": {"s2_shift": 3}}.items():
    t0 = time.time()
    runs[name] = run_protocol(c, cis, **kw)
    torch.cuda.synchronize()
    print(f"{name}: {time.time() - t0:.0f}s", flush=True)
s = runs["base"]["sustained"].cpu().numpy()
ind = s.any(1)
print("per interval:", {int(ci): round(float(s[:, k].mean()), 3) for k, ci in enumerate(cis)})
print(f"inducible: blobs {ind[: n // 2].mean():.3f}, strokes {ind[n // 2:].mean():.3f}, all {ind.mean():.3f}")
for name in ("dt_half", "s2_shift3"):
    other = runs[name]["inducible"].cpu().numpy()
    print(f"label flips vs {name}: {(other != ind).mean():.3f} ({(other != ind).sum()} of {n})")
np.savez_compressed(sys.argv[2], c=c.cpu().numpy(), sustained=s)
