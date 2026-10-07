# Evaluation plan (written 2026-10-07, before the main dataset was simulated)

## Data

- **Simulator.** 2D Aliev-Panfilov tissue, 128 x 128, with insulating scar (`src/reentrynet/sim.py`).
- **Protocol.** S1 plane wave from the left edge, then a small S2 disc at the same edge after
  41, 43, 45, 48, 52 or 60 time units. A tissue counts as **inducible** if any interval gives activity
  that persists through the whole final window.
- **Healthy controls.** Uniform and noisy-conductivity healthy tissue is never inducible at these
  intervals. The vulnerable window of healthy tissue is 34-38, excluded with a 2-unit margin.
- **Label noise.** Re-simulating 256 pilot tissues with half the time step flips 1.2 % of labels; moving
  the S2 site 3 px flips 3.1 %. Accuracy above about 97 % is therefore not meaningful.
- **Splits.**
  - Training and validation: blobs and strokes, simulation seeds 1000-1007.
  - In-distribution test: seeds 1008-1009, same families.
  - **Out-of-distribution test:** annular scars with gaps ("rings", seed 2000), never seen in training.

## Tasks and targets

| task | metric | baseline | target |
|---|---|---|---|
| inducibility | AUROC, in-distribution test | logistic regression and gradient boosting on global features (scar, border-zone and fibrosis fractions, scar components, scar perimeter) | CNN >= best baseline + 0.05 |
| inducibility | AUROC, rings (out of distribution) | same | reported; the gap to in-distribution is the generalisation result |
| re-entry location | Dice of the thresholded map, inducible test tissues | border-zone mask | CNN > baseline |
| S1 activation map | mean absolute error, time units | straight-line distance from the pacing edge / median speed | CNN < baseline |
| speed | ms per tissue, CPU | simulator (GPU, seconds per tissue) | reported |

## Rules

- Test and out-of-distribution files are not opened until the model and its threshold are frozen on validation.
- If the CNN does not beat the global-feature baseline by 0.05 AUROC, the README says so.
- An ablation-suggestion stage comes later, under its own written plan, and is judged only by re-simulation.
