# Status (paused 2026-10-07)

- Simulator: insulating scar (harmonic-mean faces, inert scar cells); physics tests pass.
- Protocol: S1 plane wave from the left edge, S2 disc (r = 4) at the same edge.
- Healthy tissue is inducible for S2 at 34-38 (vulnerable window), at every S2 site tested. Use S2 in
  41-60 only: [41, 43, 45, 48, 52, 60]. Healthy controls are non-inducible there.
- Label noise: 1.2 % flips at dt/2, 3.1 % with the S2 site moved by 3 px (256 tissues).
- Generator tuning (256 tissues each):
  - current 20.7 %;
  - more_channels 22.3 %;
  - more_fibrosis 41.0 % (fibrosis_p 0.9, fib_density 0.1-0.45);
  - both_bigger 39.8 %.
  Choose more_fibrosis for the full run.
- Timing: about 4.5 ms per step on an A100 for 924 runs at 128x128; the full 20k-tissue run is about 1.5 GPU-hours.
- Next: freeze GATE.md (trivial baseline = logistic regression on scar, border-zone and fibrosis
  fractions), run the full set, train the U-Net, then the ablation suggestion verified by re-simulation.

## Model development log (validation block 1007 only; test and rings untouched)

- **Run 1** (2026-10-07):
  - **Setup.** U-Net with inputs c and scar mask, lr 1e-3, 40 epochs, last epoch saved.
  - **Inducibility.** Validation AUROC 0.790 against logistic 0.781 and GBM 0.780. That misses the +0.05 target.
  - **Training curve.** It peaked at 0.837 (epoch 10) and then decayed.
  - **Maps.** Dice 0.505 against a border-zone mask at 0.063; activation MAE 0.48 against 1.81.
- **Run 2** (planned, one round only):
  - coordinate channels, because the protocol is not translation invariant (S1 from the left edge, S2 at mid-height);
  - best-epoch selection on validation;
  - lr 5e-4 with weight decay 1e-3;
  - three seeds averaged.
  Whatever run 2 gives is what goes to the test set.
- **Run 2 result** (validation): ensemble AUROC 0.871 against logistic 0.781 and GBM 0.780, i.e. +0.09.
  - **Maps.** Dice 0.527 against 0.063; activation MAE 0.73 against 1.81.
  - **Selection.** Best epochs were 11-12.
  - **Freeze.** The model is frozen here and run on the test and rings sets once.
