# Final clean-preservation attempt — completed2026-10-28

**Final disposition:retain camera2200. Stop; no further training attempts.**

| Policy | Nominal | Wide |
| --- | ---: | ---: |
| Recorded camera2200 screen | 229/256 (89.45%) | 80/256 (31.25%) |
| Final clean-preservation candidate | 167/256 (65.23%) | 81/256 (31.64%) |
| Change | -24.22 percentage points | +0.39 percentage points |

The candidate fails the <=3-point Nominal-loss limit and the >=10-point Wide-gain
milestone. Wide improved by only one episode; the paired descriptive95% interval
is[-5.08,+5.86] percentage points. This does not demonstrate improved Wide
accuracy. Nominal's paired interval is[-30.47,-17.97] points. No continuation,
additional candidate,1024-start confirmation or policy replacement was performed.

## The single experimental change

Recollected successful2200 preservation trajectories with all observation
corruption disabled, matching standard evaluation observation settings, at the
identical256 seed63 poses from the preceding experiment. The teacher improvement
datasets remain exactly unchanged by hash. Objective,64+64 batch composition,
phase/population balance, seed67, LR1e-5, Adam restoration, frozen normalizer,
CNN, camera, physics and physical-success conditions remain unchanged.

Clean collection:114/128 Nominal and40/128 Wide successes. Successful preservation
training episodes:88 Nominal+31 Wide. Episode holdout:26 Nominal+9 Wide.
Success membership changes are a consequence of clean collection; preservation
uses only successful2200 episodes and targets2200's own actions. Both clean and
augmented collections are data-collection protocols rather than population
accuracy benchmarks.

Exactly100 optimizer steps from2200+original Adam:77184 ->77284. No failed
checkpoint was resumed. Teacher and normalizer remain unchanged; all gradients
finite. Heldout own-action MSE:Nominal0.002272,Wide0.003004. These errors again
failed to predict preserved task success. Matching clean preservation observations
alone was insufficient; no claim of a complete root-cause diagnosis is made.

Candidate stages:Nominal250 grasp/171 lift/169 place/167 success; Wide187 grasp/
85 lift/82 place/81 success. The main Nominal regression remains grasp-to-lift.

## Evidence and verification

- comparison.json:identical collection poses, unchanged improvement hashes,
  optimizer settings matched to the preceding experiment, all corruption flags
  false, paired screen protocols/poses, checkpoint identities and confidence
  intervals. Baseline is the recorded2200 screen, not a fresh rerun.
- training/verification_0100.json:Adam steps, frozen teacher/normalizer, finite
  gradients, successful episode counts, heldout errors and dataset hashes.
- collection/baseline_collection.json and .pt:full first-episode data/provenance.
- screen_nominal/ and screen_wide/:raw first-episode audits and logs. Both planned
  256-start jobs completed. insufficient_coverage refers to stricter deployment
  coverage requirements, not an incomplete selection run.
- REVIEW.md:independent read-only review; the corruption loop matches installed
  ManagerBasedRLEnvCfg.play_mode. No blocking correctness issue found.

Baseline SHA256:3f5ad0ea7dabe634c40b1c0c82d4eee8a88b3ff8624aa88ce1801686a0c817b6.
Candidate SHA256:a8ac6e9f0370343c6b585a7af7f696678a3f6e2c4ef092b34705b01666e712c7.
The failed candidate is retained only as experimental evidence. No production
trainer source or success criteria changed. No new production test suite was
needed for the standalone research spike; verification used numerical invariants
and actual standard student-only simulator playback. No commit or push.
