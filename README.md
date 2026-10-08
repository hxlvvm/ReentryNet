# ReentryNet

**A neural surrogate for a cardiac tissue simulator: it predicts whether a scarred piece of heart tissue
will sustain re-entry, and searches for an ablation line that stops it. Every suggestion is checked against
the simulator, including the ones that expose the surrogate's shortcuts.**

![re-entry](assets/reentry.gif)

## The question

Ventricular tachycardia after a heart attack usually runs as a re-entrant circuit around and through scar.
Research groups simulate patient-specific heart models to test whether a scar can sustain re-entry and
where to ablate. Each simulation is slow, so testing many candidate ablations is expensive.

This project builds a small, fully reproducible version of that pipeline in 2D:
1. **A simulator** that knows the physics.
2. **A neural network** that learns its answers.
3. **The network used to search ablation lines** much faster than the simulator could.

Every claim is checked against the simulator, and the targets were written down before the data existed.

## How it works

**Physics** ([`sim.py`](src/reentrynet/sim.py)):
- **Model.** Aliev-Panfilov excitable tissue on a 128 x 128 grid, from
  [CardioGrad](https://github.com/hxlvvm/CardioGrad), with anisotropic conduction along x.
- **Insulating scar.** Scar is electrically insulating: face conductances use the harmonic mean, and scar
  cells are inert. Unit tests check that a one-cell scar line blocks a wave and that a one-cell gap lets it
  through.

**Protocol:**
- An S1 plane wave from the left edge, then a premature S2 stimulus at the same edge after 41, 43, 45, 48,
  52 or 60 time units.
- A tissue is **inducible** if any of these intervals leaves activity running through the whole final window.
- Healthy tissue re-enters for S2 at 34-38, the classic vulnerable window. Those intervals are excluded.
- **Checks:** removing the scar makes all 400 tested positives non-inducible, and 63 of 64 positives are
  still active 1,200 time units after S2.

**Data:**
- 20,480 random tissues: blob and stroke scars, border zones, conduction channels and patchy fibrosis.
- Plus 1,024 **ring-shaped scars the network never trains on**.
- Simulating all of it took 1.4 GPU-hours.

**Model** ([`net.py`](src/reentrynet/net.py)): a U-Net with coordinate channels (the protocol is not
translation invariant), three seeds averaged.

## Results (frozen model; details in [`GATE_RESULT.md`](GATE_RESULT.md))

| inducibility AUROC | U-Net | global scar features + boosting | position-aware features + boosting |
|---|---|---|---|
| test, 4,096 tissues | **0.853** | 0.778 | 0.818 |
| ring scars never seen, 1,024 tissues | **0.829** | 0.806 | 0.828 |

- **Against the pre-registered baseline**, global scar descriptors, the U-Net leads clearly and met its target
  of +0.05.
- **Against a stronger, position-aware baseline added after an adversarial review**, it leads by 0.035 on
  familiar shapes and ties on unseen shapes.

![roc](assets/roc.png)

**Activation map.** The U-Net also predicts the early S1 activation map: mean absolute error 0.72 time units,
against 1.68 for a per-pixel average. This covers the first 41 time units, about 30 % of the tissue.

**Negative result.** Its "where does re-entry run" map loses to simply predicting all tissue (Dice 0.51 vs
0.61), so it is not used for any claim.

## Ablation planning

For each of 400 inducible test tissues, the frozen ensemble scores 1,520 candidate ablation lines (20 px long,
3 px wide) in 0.54 s and picks the one it predicts is least inducible. The simulator then checks the result.

| one line per tissue | made non-inducible (re-simulated) |
|---|---|
| **surrogate's pick** | **47.8 %** (191/400) |
| surrogate's pick, kept 20 px away from the pacing site | 28.5 % |
| line closest to the pacing site | 22.3 % |
| line joining the most scar pieces ("isthmus") | 12.8 % |
| line through the most border zone | 2.8 % |
| random line | 3.0 % |
| *surrogate's top 5 checked by simulation* | *63.5 %* (5 random lines: 9.3 %) |

- **The surrogate beats every simple heuristic.** Against the border-zone line, 184 tissues are fixed only by
  the surrogate and 4 only by the heuristic.
- **But much of what it learned is a shortcut.** 57 % of its picks lie within 20 px of the single pacing site,
  where a line partly shields the premature beat. Clinical protocols pace from several sites for exactly
  this reason, so a multi-site protocol is the obvious next step.
- **It is far from optimal.** On 40 tissues, simulating every line on a coarser grid finds a working one for
  31, against 11 for the surrogate's pick.
- **Side effects.** The median lesion changes the activated tissue by 0.03 %, but the worst one cuts off 28 %
  of it.
- **Speed.** About 500 times faster than brute-force simulation of every candidate on the same GPU (0.54 s vs
  about 270 s per tissue), not counting the 1.4 GPU-hours of training data.

![ablation](assets/ablation.png)

## Limits

- **Not spatially converged.** Halving the grid spacing changes 27 % of the inducibility labels, and lowers
  inducibility from 42 % to 18 %. All results describe this discretised model; they are not converged
  tissue physics.
- **A simplified model.** 2D monolayer, a phenomenological cell model, synthetic scars and one pacing site.
  Times are model units.
- **Not a planning tool.** Nothing here is patient-specific or clinically validated. It is a testbed for
  surrogate-guided search, and the shielding shortcut shows why such searches must be checked carefully.

## Related work

- [Halfar et al. 2021](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8415115/) used machine learning to find
  pro-arrhythmic structures in simulated cardiac fibrosis.
- [Prakosa et al. 2018](https://pubmed.ncbi.nlm.nih.gov/30847259/) showed that personalised virtual-heart
  simulation can identify ablation targets for infarct-related VT.
- [AutoVARP](https://arxiv.org/abs/2510.18635) is a reproducible framework for inducibility testing in
  cardiac digital twins.
- This repository is a small open testbed that puts a learned surrogate in the loop of an ablation search
  and verifies every suggestion by simulation.

## Process

- **Targets set in advance.** `GATE.md` was written before the data were simulated, and
  `ABLATION_PLAN.md` before the ablation runs.
- **Disclosed development.** Two development rounds on the validation set are logged in `NOTES.md`.
- **Review.** An adversarial review led to the verification round in `VERIFY_PLAN.md`; its findings,
  including the ones that weaken the claims, are reported above.

## Run it

```bash
pip install torch numpy scipy scikit-learn matplotlib git+https://github.com/hxlvvm/CardioGrad
python -m pytest -q tests                                     # physics tests (CPU)
python scripts/make_dataset.py data/block_1000.npz 2048 1000 blobs,strokes   # one block (GPU)
python scripts/train.py data unet_s0.pt 30 0                  # blocks 1000-1007 needed
python scripts/evaluate.py data test results.json unet_s*.pt
python scripts/ablation.py data ablation.json unet_s*.pt
```

The three trained U-Nets are attached to the GitHub release `v1`. Simulation seeds are fixed (blocks
1000-1009, rings 2000), so `scripts/make_dataset.py` regenerates the dataset exactly.

## Licence

MIT.
