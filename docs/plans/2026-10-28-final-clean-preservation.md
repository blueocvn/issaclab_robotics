# Final clean-preservation attempt — 2026-10-28

**Authorization:** User requested one final change/attempt. No further training
attempts or automatic continuations after this candidate.
**Single experimental change:** Replace augmented successful2200 preservation
trajectories with clean-observation trajectories at the same seed63 poses.
Disable observation corruption in every observation group, exactly as the base
play_mode does, while retaining256 collection envs and identical replay poses.
Record actual flags. Successful-episode membership may change as a consequence.

All other experimental settings remain from offline100:base2200 weights+Adam,
teacher/student Wide improvement datasets unchanged by hash,64+64 minibatch,
phase/population balance, seed67, LR1e-5, same weighted objective, frozen normalizer,
100 optimizer steps. This is the final standalone research spike, not a new
production trainer. No camera/physics/success changes or checkpoint replacement.

- [x] Add collector flag and explicit observation-corruption provenance; recollect
  the same256 seed63 poses into final_clean/collection. Verify identical poses,
  all corruption flags false and successful phase/population coverage.
- [x] Parameterize preservation input/output paths only; verify unchanged teacher
  improvement data hashes and normalizer/teacher/Adam invariants. Train exactly100
  steps into final_clean/training. No continuation candidate.
- [x] Screen nominal256 seed42 and Wide256 seed45 with unchanged standard playback.
  Compare exact protocols/poses/checkpoint identities to recorded2200 screens.
- [x] If Wide gains>=10pp and Nominal loses<=3pp, confirm once on independent1024
  starts (52/49) against2200. Otherwise stop and retain2200. No further attempt.
- [x] Record evidence and final disposition in handoff/report.

Execution:clean collection114 Nominal+40 Wide successes; train88+31 and
holdout26+9. Identical poses and unchanged improvement hashes checked. One
100-step candidate completed, Adam77184->77284, teacher/normalizer unchanged,
finite gradients. Independent review confirmed play_mode corruption semantics.
Nominal/Wide256-start screens running. No continuation authorized.


Final outcome:Nominal167/256,Wide81/256. Matched protocol/pose comparisons
verified; candidate fails both adoption requirements. Independent1024-start
confirmation skipped because ineligible. Retain2200. No continuation or further
training attempts. Evidence:logs/camera_wide_focus_20261028/final_clean/REPORT.md.

