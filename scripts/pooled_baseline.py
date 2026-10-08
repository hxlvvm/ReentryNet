"""E4: position-aware baseline, gradient boosting on global features plus pooled conductivity and scar maps."""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reentrynet.features import features  # noqa: E402

d = Path(sys.argv[1])


def load(names):
    parts = [np.load(d / f"{n}.npz") for n in names]
    return np.concatenate([p["c"] for p in parts]).astype(np.float32), np.concatenate([p["inducible"] for p in parts])


def feats(c, g):
    b = 128 // g
    pooled_c = c.reshape(len(c), g, b, g, b).mean((2, 4)).reshape(len(c), -1)
    pooled_s = (c == 0).reshape(len(c), g, b, g, b).mean((2, 4)).reshape(len(c), -1)
    return np.hstack([np.stack([features(x) for x in c]), pooled_c, pooled_s])


ctr, ytr = load([f"block_{s}" for s in range(1000, 1007)])
sets = {"test": load(["block_1008", "block_1009"]), "rings": load(["rings_2000"])}
res = {}
for g in (8, 16):
    model = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, random_state=0).fit(feats(ctr, g), ytr)
    res[f"pooled_{g}x{g}"] = {k: float(roc_auc_score(y, model.predict_proba(feats(c, g))[:, 1])) for k, (c, y) in sets.items()}
    print(g, res[f"pooled_{g}x{g}"], flush=True)
json.dump(res, open(sys.argv[2], "w"), indent=1)
