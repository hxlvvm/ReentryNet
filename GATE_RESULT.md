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

**Reading.** The surrogate finds working lesions 15-25 times more often than chance. It still falls well
short of what exhaustive simulation can find.
