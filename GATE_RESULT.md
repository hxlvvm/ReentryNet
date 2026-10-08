# Result (2026-10-08)

The plan is in `GATE.md`, written before the main dataset was simulated.
- **Development** used only the training blocks (1000-1006) and the validation block (1007), and is
  logged in `NOTES.md`. There were two rounds.
- **Round 2** was frozen on validation and then run once on the test blocks (1008-1009) and on the
  held-out ring family.
- **Raw outputs** are in `results/`.

## Inducibility (does a premature beat start sustained re-entry?)

| set | tissues | inducible | U-Net ensemble AUROC | logistic AUROC | GBM AUROC | target |
|---|---|---|---|---|---|---|
| validation (development) | 2,048 | 38.8 % | 0.871 | 0.781 | 0.780 | — |
| **test** (blobs and strokes) | 4,096 | 39.6 % | **0.853** | 0.775 | 0.778 | best baseline + 0.05: **met (+0.075)** |
| **rings** (shape family never trained on) | 1,024 | 44.8 % | **0.829** | 0.795 | 0.806 | reported: +0.023 |

At the threshold fixed on validation (90 % specificity), the test set gives sensitivity 57.8 % and
specificity 88.7 %; the rings give 59.3 % and 85.7 %.

**Label-noise ceiling.** Re-simulating flips 1-3 % of labels, so the remaining error is mostly not noise.

**Reading.**
- On the shape families it was trained on, the network beats global scar descriptors clearly.
- On a new shape family its advantage shrinks to 0.02. Scar geometry that it has not seen is the main
  limitation.

## Where the re-entry runs, and the activation map

| set | Dice, U-Net | Dice, border-zone mask (pre-registered) | Dice, all living tissue (added later) | activation MAE, U-Net | activation MAE, distance / speed (pre-registered) | activation MAE, per-pixel training mean (added later) |
|---|---|---|---|---|---|---|
| test | 0.51 | 0.06 | **0.61** | **0.72** | 1.81 | 1.68 |
| rings | 0.49 | 0.04 | **0.59** | **0.68** | 1.58 | 1.62 |

**The re-entry map is a negative result.**
- The pre-registered border-zone baseline was weak. The simulated re-entry map, defined as cells depolarised
  in more than 20 % of the final window, covers about half the living tissue.
- So simply predicting all living tissue beats the U-Net's Dice.
- As defined, the map says where activity wanders rather than where a circuit is anchored. It is not used
  for any claim.

**The activation map holds up against the stronger baseline.** It covers only the first 41 time units, before
the earliest S2: about 30 % of the tissue.

The trivial baselines were computed after the frozen evaluation (`scripts/trivial_baselines.py`).

## Speed

| | time per tissue |
|---|---|
| simulator: 6 protocol runs, batch of 256, one GPU | 0.18 s (H100) to 0.28 s (A100) |
| U-Net ensemble, one CPU thread | 0.05 s |

## Development history (validation only)

- **Round 1:** one model, last epoch. AUROC 0.790 against a baseline of 0.781, so the target was missed.
- **Round 2:** coordinate input channels (the protocol is not translation invariant), best-epoch selection,
  a lower learning rate and three seeds. Validation AUROC 0.871.

## Ablation suggestion (plan in `ABLATION_PLAN.md`, written before it ran)

**Setup.**
- 400 inducible test tissues, one insulating line per tissue: 20 px long, 3 px wide, chosen from 1,520
  candidates.
- Success means the tissue is no longer inducible on re-simulation with the same protocol.
- The unablated tissues re-simulate as inducible in 400 of 400 cases.

| strategy | made non-inducible |
|---|---|
| **surrogate** (lowest predicted inducibility) | **191/400 = 47.8 %** [42.9, 52.6] |
| border-zone heuristic | 11/400 = 2.8 % [1.5, 4.9] |
| random candidate | 12/400 = 3.0 % [1.7, 5.2] |

**Paired comparisons (exact McNemar):**
- surrogate vs heuristic: 184 tissues where only the surrogate succeeds, 4 where only the heuristic does,
  p = 3e-49;
- surrogate vs random: 185 vs 6, p = 4e-47.

**Target** (surrogate at least 10 points above the heuristic, p < 0.05): **met**.

**Search cost.** 0.54 s per tissue on one GPU for 1,520 candidates. Simulating every candidate would take
about 270 s.

**Side effect.** The S1 beat fails to reach a median of 0.5 % more of the tissue after the surrogate's lesion.

**Winner's curse.** The surrogate predicted a mean inducibility of 0.37 for its own choices, but 52 % of them
stayed inducible. Optimising against the model exploits its errors.

**Upper bound** (`scripts/oracle.py`, 40 of the 400 tissues). Every candidate on a 12 px grid (median 398 per
tissue) was simulated:

| quantity | result |
|---|---|
| tissues where at least one line works | 31/40 |
| average share of lines that work | 1.9 % |
| surrogate's pick within this coarser set works | 11/40 |
| one of the surrogate's top 5 works | 18/40 |

**Reading** (corrected; an earlier version said "15-25 times").
- On the full grid the surrogate's pick works 16 times as often as a random line (47.8 % vs 3.0 %).
- On the coarse oracle grid it works 14 times as often as the average line (27.5 % vs 1.9 %).
- It falls well short of what exhaustive simulation can find.
- The side-effect figure of 0.5 % above covered only the early-activated part of the tissue; the
  verification round below measures it over the whole tissue.

## Verification round (plan in `VERIFY_PLAN.md`, written before running; raw numbers in `results/verify.json`)

An adversarial review questioned several claims. These experiments do not change the frozen model.

**Labels**

| check | result | reading |
|---|---|---|
| E3a: all c < 1 set to 1 for the 400 inducible test tissues | 0/400 inducible | positives depend on the scar |
| E3b: 64 positives run to 1,200 units after S2 | 63/64 still active in the last 150 units | persistent activity, not a slow transient |
| E6: half time step (512 validation tissues) | 11 flips (2.1 %) | time integration converged |
| **E6: half grid spacing** (256 tissues on a 256 x 256 grid) | **70 flips (27 %)**; inducible 42 % -> 18 % | **not spatially converged** |

The labels describe this discretised model at dx = 1. They are not converged continuum physics. Everything
here, including the surrogate, is a surrogate of the dx = 1 simulator.

**A position-aware baseline (E4)**

Gradient boosting on global features plus pooled 8 x 8 / 16 x 16 conductivity and scar maps:

| set | AUROC |
|---|---|
| test | 0.817 / 0.818 |
| rings | 0.823 / 0.828 |

- Against this stronger baseline, the U-Net leads by +0.035 on test and ties on the rings (0.829 vs 0.828).
- The pre-registered target (+0.05 over the global-feature baseline) was met. Against the position-aware
  baseline, added afterwards, it would not be.

**What drives ablation success (E1, E1b, E2, E5)**

| strategy, 400 inducible test tissues | made non-inducible |
|---|---|
| surrogate's pick | 191 (47.8 %) |
| surrogate's pick, excluding lesions within 20 px of the S2 site | 114 (28.5 %) |
| line closest to the S2 site | 89 (22.3 %) |
| line touching the most scar components ("isthmus") | 51 (12.8 %) |
| line covering the most border zone | 11 (2.8 %) |
| random line | 11-12 (about 3 %) |
| surrogate's top 5, simulated; success if any works | 254 (63.5 %) |
| 5 random lines, simulated; success if any works | 37 (9.3 %) |

**Shielding the stimulus.** 229 of the surrogate's 400 picks lie within 20 px of the S2 site, and they work
53 % of the time. The picks further away work 41 % of the time (70/171).

**Reading.**
- A large part of what the surrogate learned is to shield the single S2 pacing site. That is a shortcut
  specific to this protocol. Clinical inducibility protocols pace from several sites for exactly this reason.
- It still beats every heuristic, including the shielding heuristic, and succeeds well above chance far
  from the stimulus.
- The decision rule set in advance (>= 30 % with near-S2 lesions excluded) is narrowly missed at 28.5 %. The
  claim is therefore reworded to include the shielding effect.

**Best achievable (E5).** On tissues 0-39 the fine-grid surrogate pick works for 11, while simulating every
line on a coarser grid finds a working one for 31. The surrogate is far from optimal.

**Side effects (E2).** S1-only simulation over the whole tissue:
- the median change in the activated share is -0.03 %;
- the 5th percentile is -0.5 %;
- in the worst tissue, 28 % of the tissue is no longer reached;
- across tissues, the 95th percentile of the change in 95th-percentile activation time is 10 time units.
