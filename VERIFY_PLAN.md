# Verification round (written before running; prompted by an adversarial review)

None of these runs changes the frozen model. They test whether the published claims hold, and the
decision rules below fix in advance what each outcome means for the README.

| id | question | experiment | decision rule |
|---|---|---|---|
| E1 | Do ablation successes come from shielding the stimulus rather than changing the substrate? | Re-run the surrogate search saving every pick. Add a variant that excludes candidates with any lesion pixel within 20 px of the S2 centre, and a "closest to S2" heuristic. | If the excluded variant still succeeds >= 30 % (heuristics about 3 %), report "not explained by shielding". Otherwise reword the claim to include the shielding effect. |
| E1b | Is the border-zone heuristic a strawman? | Add an isthmus heuristic: the line touching the most distinct scar components. Add same-budget comparisons: simulate 5 random lines vs the surrogate's top 5, success if any works. | Report all of them; the headline compares against the strongest heuristic. |
| E2 | Does the lesion damage normal conduction? | S1-only simulation long enough to cover the whole tissue, before vs after each lesion: share activated, and the change in 95th-percentile activation time. | Report the medians and the 95th percentiles of both. |
| E3 | Are positives caused by the scar, and is the activity persistent? | (a) Set every c < 1 to 1 for the 400 inducible test tissues and re-simulate. (b) Extend 64 positives to 1,200 time units after S2. | (a) Expect about 0 % inducible; otherwise drop "scar-caused". (b) If < 80 % still active in the last 150 units, call the label "persistent activity over 450 units", not sustained re-entry. |
| E4 | Is the AUROC gain only position-aware vs position-blind? | Gradient boosting on global features plus 8 x 8 pooled conductivity and scar maps. | If within 0.02 AUROC of the U-Net on test, the headline becomes the ring result plus this baseline. |
| E5 | How close is the surrogate to the best achievable, on comparable grids? | The fine-grid surrogate pick for tissues 0-39, compared with the coarse-grid oracle (31/40, a lower bound). | Report both on the same tissues. |
| E6 | Label noise on the final protocol | Half time step (512 validation tissues) and half grid spacing (256 tissues, 256 x 256 grid, dt 0.025). | Replace the old 1-3 % figure with these numbers. |
