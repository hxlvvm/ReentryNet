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

| set | Dice, U-Net | Dice, border-zone mask | activation MAE, U-Net (time units) | activation MAE, distance / speed |
|---|---|---|---|---|
| test | **0.51** | 0.06 | **0.72** | 1.81 |
| rings | **0.49** | 0.04 | **0.68** | 1.58 |

Both maps generalise to the ring family.

## Speed

| | time per tissue |
|---|---|
| simulator: 6 protocol runs, batch of 256, one GPU | 0.18 s (H100) to 0.28 s (A100) |
| U-Net ensemble, one CPU thread | 0.05 s |

## Development history (validation only)

- **Round 1:** one model, last epoch. AUROC 0.790 against a baseline of 0.781, so the target was missed.
- **Round 2:** coordinate input channels (the protocol is not translation invariant), best-epoch selection,
  a lower learning rate and three seeds. Validation AUROC 0.871.
