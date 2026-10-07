"""Inducibility rate for a few generator settings (healthy controls included)."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reentrynet.sim import run_protocol  # noqa: E402
from reentrynet.tissue import make_tissue  # noqa: E402

CIS = [41.0, 43.0, 45.0, 48.0, 52.0, 60.0]
SETTINGS = {
    "current": {},
    "more_channels": {"channels": (1, 5), "bz": (0.05, 0.35)},
    "more_fibrosis": {"fibrosis_p": 0.9, "fib_density": (0.1, 0.45)},
    "both_bigger": {"channels": (1, 5), "bz": (0.05, 0.35), "fibrosis_p": 0.8, "fib_density": (0.1, 0.4), "scar": (0.08, 0.3)},
}
n = 256
for name, kw in SETTINGS.items():
    rng = np.random.default_rng(11)
    fams = ["blobs"] * (n // 2) + ["strokes"] * (n - n // 2)
    c = torch.from_numpy(np.stack([np.ones((128, 128), np.float32)] + [make_tissue(rng, (128, 128), f, **kw) for f in fams])).cuda()
    s = run_protocol(c, CIS)["sustained"].cpu().numpy()
    ind = s[1:].any(1)
    print(f"{name:15s} healthy={bool(s[0].any())} inducible all {ind.mean():.3f} blobs {ind[: n // 2].mean():.3f} strokes {ind[n // 2:].mean():.3f}", flush=True)
