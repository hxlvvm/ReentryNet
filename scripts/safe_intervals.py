"""Find S2 intervals at which healthy tissue (with and without conductivity noise) is never inducible."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reentrynet.sim import run_protocol  # noqa: E402

rng = np.random.default_rng(5)
variants = [np.ones((128, 128), np.float32)]
for amp in (0.1, 0.2, 0.3):
    for _ in range(2):
        variants.append(np.clip(1 - amp * rng.random((128, 128)), 0, 1).astype(np.float32))
variants.append(np.full((128, 128), 0.6, np.float32))
cis = [float(x) for x in range(30, 71)]
c = torch.from_numpy(np.stack(variants)).cuda()
res = {}
for shift in (0, 3, -3):
    s = run_protocol(c, cis, s2_shift=shift)["sustained"].cpu().numpy()
    res[shift] = s
bad = np.zeros(len(cis), bool)
for shift, s in res.items():
    bad |= s.any(0)
    print(f"shift {shift:+d}: healthy inducible at", [int(ci) for ci, b in zip(cis, s.any(0)) if b])
print("safe intervals:", [int(ci) for ci, b in zip(cis, bad) if not b])
