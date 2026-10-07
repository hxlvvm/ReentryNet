"""Pilot 3: inducibility at the safe S2 intervals, with healthy controls in the same batch."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reentrynet.sim import run_protocol  # noqa: E402
from reentrynet.tissue import make_tissue  # noqa: E402

CIS = [41.0, 43.0, 45.0, 48.0, 52.0, 60.0]
n = int(sys.argv[1])
rng = np.random.default_rng(3)
healthy = [np.ones((128, 128), np.float32)] + [np.clip(1 - a * rng.random((128, 128)), 0, 1).astype(np.float32) for a in (0.1, 0.2, 0.3)]
fams = ["blobs"] * (n // 2) + ["strokes"] * (n - n // 2)
c = torch.from_numpy(np.stack(healthy + [make_tissue(rng, (128, 128), f) for f in fams])).cuda()
s = run_protocol(c, CIS)["sustained"].cpu().numpy()
print("healthy controls inducible:", s[:4].any(1).tolist())
s = s[4:]
ind = s.any(1)
print("per interval:", {int(ci): round(float(s[:, k].mean()), 3) for k, ci in enumerate(CIS)})
print(f"inducible: blobs {ind[: n // 2].mean():.3f}, strokes {ind[n // 2:].mean():.3f}, all {ind.mean():.3f}")
cs = c[4:].cpu().numpy()
scar = (cs == 0).mean((1, 2))
bz = ((cs > 0) & (cs < 1)).mean((1, 2))
print("corr with scar fraction", round(float(np.corrcoef(scar, ind)[0, 1]), 3), "| with border-zone fraction", round(float(np.corrcoef(bz, ind)[0, 1]), 3))
