"""Simulate a block of tissues and save inputs and labels. usage: make_dataset.py OUT.npz N SEED FAMILIES [BATCH]"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reentrynet.sim import run_protocol  # noqa: E402
from reentrynet.tissue import GENERATOR, make_tissue  # noqa: E402

INTERVALS = [41.0, 43.0, 45.0, 48.0, 52.0, 60.0]
out, n, seed, families = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4].split(",")
batch = int(sys.argv[5]) if len(sys.argv) > 5 else 256
rng = np.random.default_rng(seed)
fams = [families[i % len(families)] for i in range(n)]
C = np.stack([make_tissue(rng, (128, 128), f) for f in fams])
keys = ["inducible", "sustained", "reentry_map", "s1_act"]
res = {k: [] for k in keys}
t0 = time.time()
for s in range(0, n, batch):
    o = run_protocol(torch.from_numpy(C[s:s + batch]).cuda(), INTERVALS)
    for k in keys:
        res[k].append(o[k].cpu().numpy())
    print(f"{s + len(C[s:s + batch])}/{n} tissues, {time.time() - t0:.0f}s, inducible so far {np.concatenate(res['inducible']).mean():.3f}", flush=True)
np.savez_compressed(out, c=C.astype(np.float16), family=np.array(fams), inducible=np.concatenate(res["inducible"]),
                    sustained=np.concatenate(res["sustained"]), reentry_map=np.concatenate(res["reentry_map"]).astype(np.float16),
                    s1_act=np.concatenate(res["s1_act"]).astype(np.float16), intervals=np.array(INTERVALS), seed=seed,
                    generator=str(GENERATOR))
print("saved", out)
