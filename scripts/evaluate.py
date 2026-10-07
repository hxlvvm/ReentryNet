"""Score the U-Net (ensemble of checkpoints) and the baselines. usage: evaluate.py DATA_DIR SPLIT(val|test|ood) OUT.json MODEL.pt [...]"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reentrynet.features import features  # noqa: E402
from reentrynet.net import ReentryUNet  # noqa: E402
from train import load, predict  # noqa: E402

SPLITS = {"val": ["block_1007"], "test": ["block_1008", "block_1009"], "ood": ["rings_2000"]}


def load_named(d, names):
    parts = [np.load(Path(d) / f"{n}.npz") for n in names]
    get = lambda k: np.concatenate([p[k] for p in parts]).astype(np.float32)  # noqa: E731
    return get("c"), get("inducible"), get("reentry_map"), get("s1_act")


def dice(a, b):
    s = a.sum() + b.sum()
    return 2 * (a & b).sum() / s if s else np.nan


def main():
    d, split, out, ckpts = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    c_tr, y_tr, _, act_tr = load(d, range(1000, 1007))
    c, y, rm, act = load_named(d, SPLITS[split])
    F_tr = np.stack([features(x) for x in c_tr])
    F = np.stack([features(x) for x in c])
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(F_tr, y_tr)
    gbm = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(F_tr, y_tr)
    preds = []
    for ckpt in ckpts:
        st = torch.load(ckpt, map_location="cuda")
        model = ReentryUNet().cuda()
        model.load_state_dict(st["state"])
        preds.append(predict(model, c, "cuda", maps=True))
    p, rmap, apred = (np.mean([q[k] for q in preds], axis=0) for k in range(3))
    if len(ckpts) > 1:
        cv_, yv_, _, _ = load(d, [1007])
        pv = []
        for ckpt in ckpts:
            m = ReentryUNet().cuda()
            m.load_state_dict(torch.load(ckpt, map_location="cuda")["state"])
            pv.append(predict(m, cv_, "cuda"))
        st = dict(st, threshold_spec90=float(np.quantile(np.mean(pv, 0)[yv_ == 0], 0.9)))
    res = {"split": split, "checkpoints": [Path(k).name for k in ckpts], "n": int(len(y)), "inducible_fraction": float(y.mean()),
           "auroc": {"cnn": roc_auc_score(y, p), "logistic": roc_auc_score(y, lr.predict_proba(F)[:, 1]),
                     "gbm": roc_auc_score(y, gbm.predict_proba(F)[:, 1])}}
    thr = st["threshold_spec90"]
    res["cnn_at_validation_threshold"] = {"sensitivity": float((p[y == 1] >= thr).mean()), "specificity": float((p[y == 0] < thr).mean())}
    pos = y == 1
    res["dice"] = {"cnn": float(np.nanmean([dice(rmap[i] > 0.5, rm[i] > 0.2) for i in np.where(pos)[0]])),
                   "border_zone_mask": float(np.nanmean([dice((c[i] > 0) & (c[i] < 1), rm[i] > 0.2) for i in np.where(pos)[0]]))}
    valid = np.isfinite(act)
    xs = np.broadcast_to(np.arange(128, dtype=np.float32), act.shape)
    vtr = np.isfinite(act_tr) & (act_tr > 1)
    speed = np.median(np.broadcast_to(np.arange(128, dtype=np.float32), act_tr.shape)[vtr] / act_tr[vtr])
    res["activation_mae"] = {"cnn": float(np.abs(apred * st["act_scale"] - act)[valid].mean()),
                             "distance_over_speed": float(np.abs(xs / speed - act)[valid].mean()), "median_speed": float(speed)}
    cpu = ReentryUNet().eval()
    cpu.load_state_dict({k: v.cpu() for k, v in st["state"].items()})
    torch.set_num_threads(1)
    with torch.no_grad():
        x = torch.from_numpy(c[:32])
        cpu(x)
        t0 = time.time()
        cpu(x)
    res["cnn_ms_per_tissue_cpu_1thread"] = 1000 * (time.time() - t0) / 32
    res = json.loads(json.dumps(res, default=float))
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
