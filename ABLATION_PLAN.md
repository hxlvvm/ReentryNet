# Ablation-suggestion plan (written 2026-10-08, before any ablation was simulated)

**Question.** Can the frozen surrogate choose an ablation line that makes an inducible tissue
non-inducible, verified by re-simulating with the same protocol, more often than simple strategies?

## Setup

- **Tissues.** 400 tissues drawn at random (seed 7) from the inducible tissues of the test blocks
  (1008-1009). The surrogate never trained on these blocks.
- **Lesion.** A straight line 20 px long and 3 px wide, set to conductivity 0 (insulating, like an ablation
  lesion).
- **Candidates.** Centres on a 6 px grid with x >= 14 (the pacing strip stays intact), at 4 orientations:
  about 1,500 candidates per tissue. Candidates lying entirely inside existing scar are discarded.
- **Strategies** (one lesion each):
  1. **Surrogate:** the candidate with the lowest predicted inducibility (mean of the three frozen U-Nets).
  2. **Border-zone heuristic:** the candidate covering the most border-zone cells (0 < conductivity < 1),
     a stand-in for substrate-guided ablation of slow-conducting channels.
  3. **Random:** a uniformly random candidate.
- **Sanity check.** Re-simulating the unablated tissues must reproduce "inducible" for all 400.

## Endpoints

| endpoint | definition |
|---|---|
| primary | share of tissues made non-inducible: surrogate vs heuristic, paired, with an exact McNemar test |
| secondary | surrogate vs random; Wilson 95 % intervals for each strategy |
| side effect | share of living tissue the S1 beat no longer reaches after ablation |
| target | surrogate >= heuristic + 10 percentage points, McNemar p < 0.05 |

**Optional upper bound.** On a 40-tissue subset, simulate a thinned candidate set (every second centre)
and report the best achievable success rate, to show how close the surrogate's choice comes.
