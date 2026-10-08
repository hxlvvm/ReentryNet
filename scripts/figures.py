"""README figures: spiral animation, example predictions, ROC curves. usage: figures.py DATA CKPT [CKPT ...]"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from reentrynet.features import features  # noqa: E402
from reentrynet.net import ReentryUNet  # noqa: E402
from reentrynet.sim import AlievPanfilov, diffusion, disc, face_conductances  # noqa: E402

d, ckpts = Path(sys.argv[1]), sys.argv[2:]
dev = "cuda"
models = []
for k in ckpts:
    m = ReentryUNet().to(dev).eval()
    m.load_state_dict(torch.load(k, map_location=dev)["state"])
    models.append(m)
scale = torch.load(ckpts[0], map_location="cpu")["act_scale"]


@torch.no_grad()
def predict(c):
    outs = []
    for s in range(0, len(c), 256):
        x = torch.from_numpy(c[s:s + 256]).to(dev)
        o = [m(x) for m in models]
        outs.append([torch.stack([torch.sigmoid(z[0]) for z in o]).mean(0).cpu().numpy(),
                     torch.stack([torch.sigmoid(z[1]) for z in o]).mean(0).cpu().numpy(),
                     torch.stack([z[2] for z in o]).mean(0).cpu().numpy() * scale])
    return [np.concatenate([o[k] for o in outs]) for k in range(3)]


def load(names):
    parts = [np.load(d / f"{n}.npz") for n in names]
    return {k: np.concatenate([p[k] for p in parts]) for k in ("c", "inducible", "reentry_map", "s1_act", "sustained")}


test, rings = load(["block_1008", "block_1009"]), load(["rings_2000"])
train = load([f"block_{s}" for s in range(1000, 1007)])
for D in (test, rings, train):
    D["c"] = D["c"].astype(np.float32)

# ROC curves
gbm = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(np.stack([features(x) for x in train["c"]]), train["inducible"])
fig, ax = plt.subplots(figsize=(4.6, 4.2))
for D, name, col in ((test, "test (seen shape families)", "#c0392b"), (rings, "rings (never seen)", "#2e86de")):
    p = predict(D["c"])[0]
    g = gbm.predict_proba(np.stack([features(x) for x in D["c"]]))[:, 1]
    for score, style, lab in ((p, "-", "U-Net"), (g, "--", "global features")):
        fpr, tpr, _ = roc_curve(D["inducible"], score)
        ax.plot(fpr, tpr, style, color=col, lw=1.6, label=f"{lab}, {name}: {roc_auc_score(D['inducible'], score):.3f}")
ax.plot([0, 1], [0, 1], color="#bbb", lw=0.8)
ax.set_xlabel("false positive rate")
ax.set_ylabel("true positive rate")
ax.set_title("Predicting inducible re-entry", fontsize=10)
ax.legend(fontsize=6.5, frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(ROOT / "assets/roc.png", dpi=160)

# example predictions on inducible test tissues
rng = np.random.default_rng(3)
idx = rng.choice(np.where(test["inducible"])[0], 4, replace=False)
p, rmap, act = predict(test["c"][idx])
fig, axes = plt.subplots(4, 5, figsize=(10, 8.4))
for r, i in enumerate(idx):
    c = test["c"][i]
    true_act = test["s1_act"][i].astype(np.float32)
    panels = [(c, "conductivity", "gray", 0, 1), (test["reentry_map"][i].astype(np.float32), "re-entry (simulated)", "magma", 0, 1),
              (rmap[r], f"re-entry (U-Net), p={p[r]:.2f}", "magma", 0, 1), (true_act, "S1 activation (simulated)", "viridis", 0, np.nanmax(true_act)),
              (np.where(c > 0, act[r], np.nan), "S1 activation (U-Net)", "viridis", 0, np.nanmax(true_act))]
    for a, (img, title, cmap, lo, hi) in zip(axes[r], panels):
        a.imshow(img, cmap=cmap, vmin=lo, vmax=hi, origin="lower")
        a.set_xticks([]), a.set_yticks([])
        if r == 0:
            a.set_title(title, fontsize=8)
fig.tight_layout()
fig.savefig(ROOT / "assets/examples.png", dpi=130)

# spiral animation for one inducible test tissue at its first inducing interval
INTERVALS = [41.0, 43.0, 45.0, 48.0, 52.0, 60.0]
i = int(idx[0])
c = torch.from_numpy(test["c"][i]).to(dev)[None]
k = int(np.argmax(test["sustained"][i]))
t_s2, dt = INTERVALS[k], 0.05
alive = c > 0
cx, cy = face_conductances(c)
u = torch.zeros_like(c)
v = torch.zeros_like(c)
u[..., :3] = 1.0
u = u * alive
s2 = disc((128, 128), (64, 3), 4.0, dev)
model = AlievPanfilov()
frames = []
with torch.no_grad():
    for step in range(int((t_s2 + 450) / dt)):
        if step == int(round(t_s2 / dt)):
            u[0][s2 & alive[0]] = 1.0
        du, dv = model.reaction(u, v)
        u = (u + dt * (diffusion(u, cx, cy) + du)) * alive
        v = (v + dt * dv) * alive
        if step % 60 == 0:
            frames.append(u[0].cpu().numpy().copy())
fig, ax = plt.subplots(figsize=(3.6, 3.6))
base = np.where(test["c"][i] == 0, 0.35, np.nan)
im = ax.imshow(frames[0], cmap="inferno", vmin=0, vmax=1, origin="lower")
ax.imshow(base, cmap="Blues", vmin=0, vmax=1, origin="lower", alpha=0.9)
ax.set_xticks([]), ax.set_yticks([])
txt = ax.set_title("", fontsize=8)


def draw(f):
    im.set_data(frames[f])
    txt.set_text(f"t = {f * 60 * dt:.0f}   (premature beat at {t_s2:.0f})")
    return im, txt


FuncAnimation(fig, draw, frames=len(frames), blit=False).save(ROOT / "assets/reentry.gif", writer=PillowWriter(fps=14), dpi=80)
print("figures written; example tissue", i, "inducing interval", t_s2)
