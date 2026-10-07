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
