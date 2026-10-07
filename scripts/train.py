"""Train the U-Net on blocks 1000-1006 and pick the threshold on block 1007. usage: train.py DATA_DIR OUT.pt [EPOCHS] [SEED]"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reentrynet.net import ReentryUNet  # noqa: E402

ACT_SCALE = 100.0


def load(d, seeds):
    parts = [np.load(Path(d) / f"block_{s}.npz") for s in seeds]
    c = np.concatenate([p["c"] for p in parts]).astype(np.float32)
    y = np.concatenate([p["inducible"] for p in parts]).astype(np.float32)
    rm = np.concatenate([p["reentry_map"] for p in parts]).astype(np.float32)
    act = np.concatenate([p["s1_act"] for p in parts]).astype(np.float32)
    return c, y, rm, act


def main():
    d, out = sys.argv[1], sys.argv[2]
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    c, y, rm, act = load(d, range(1000, 1007))
    cv, yv, _, _ = load(d, [1007])
    print(f"train {len(y)} tissues ({y.mean():.3f} inducible), validation {len(yv)}", flush=True)
    dev = "cuda"
    T = {k: torch.from_numpy(v) for k, v in dict(c=c, y=y, rm=rm, act=np.nan_to_num(act / ACT_SCALE, nan=-1.0)).items()}
    model = ReentryUNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    steps = epochs * (len(y) // 64)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 5e-4, total_steps=steps, pct_start=0.1)
    bce = torch.nn.BCEWithLogitsLoss()
    t0 = time.time()
    best, best_auc, history = None, -1.0, []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(y))
        for s in range(0, len(y) - 63, 64):
            i = perm[s:s + 64]
            cb, yb, rmb, ab = (T[k][i].to(dev) for k in ("c", "y", "rm", "act"))
            with torch.autocast(dev, dtype=torch.bfloat16):
                logit, rlog, apred = model(cb)
            logit, rlog, apred = logit.float(), rlog.float(), apred.float()
            loss = bce(logit, yb)
            pos = yb > 0
            if pos.any():
                loss = loss + 0.5 * bce(rlog[pos], (rmb[pos] > 0.2).float())
            valid = ab >= 0
            loss = loss + 0.5 * (apred - ab).abs()[valid].mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        p = predict(model, cv, dev)
        auc = float(roc_auc_score(yv, p))
        history.append(auc)
        if auc > best_auc:
            best_auc, best = auc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch {ep + 1}: validation AUROC {auc:.4f} ({time.time() - t0:.0f}s)", flush=True)
    model.load_state_dict(best)
    p = predict(model, cv, dev)
    thr = float(np.quantile(p[yv == 0], 0.9))
    torch.save({"state": best, "threshold_spec90": thr, "act_scale": ACT_SCALE, "seed": seed,
                "best_epoch": int(np.argmax(history)) + 1, "val_history": history}, out)
    print("saved", out, "best validation AUROC", round(best_auc, 4), "at epoch", int(np.argmax(history)) + 1)


@torch.no_grad()
def predict(model, c, dev, maps=False):
    model.eval()
    outs = []
    for s in range(0, len(c), 256):
        logit, rlog, apred = model(torch.from_numpy(c[s:s + 256]).to(dev))
        outs.append((torch.sigmoid(logit).cpu().numpy(), torch.sigmoid(rlog).cpu().numpy(), apred.cpu().numpy()) if maps
                    else torch.sigmoid(logit).cpu().numpy())
    if maps:
        return tuple(np.concatenate([o[k] for o in outs]) for k in range(3))
    return np.concatenate(outs)


if __name__ == "__main__":
    main()
