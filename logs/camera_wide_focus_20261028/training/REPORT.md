# Offline Wide preservation screen — 2026-10-28

**Decision: stop at100 optimizer steps and retain camera2200.** No200/300-step
continuation or1024-start confirmation is eligible.

| Checkpoint | Nominal | Wide |
| --- | ---: | ---: |
| Camera2200, recorded matched screen | 229/256 (89.45%) | 80/256 (31.25%) |
| Offline100 | 161/256 (62.89%) | 79/256 (30.86%) |
| Change | -26.56 percentage points | -0.39 percentage points |

The candidate fails the Nominal guard and supplies no Wide gain. It is an
experimental artifact, not a replacement policy. Source2200 SHA256 remains
3f5ad0ea7dabe634c40b1c0c82d4eee8a88b3ff8624aa88ce1801686a0c817b6.
Candidate SHA256 is1c116c3056d762b5b9a0b17fc2ff9a3514af2e3bb9affde884c2638f84530108.

## Data and training

Expanded student-controlled seed63 collection:128 fresh Wide and128 fresh
Nominal starts, no overlap with existing selection poses. Successful episodes:
24 Wide and75 Nominal. Preservation training uses15 Wide+63 Nominal episodes;
episode holdout uses9 Wide+12 Nominal. Only physically successful episodes supply
preservation targets, labelled with camera2200's own pre-action outputs.

Wide improvement data comes from the preceding seed61 collection: successful
teacher demonstrations and teacher labels on student-visited Wide states, kept
as distinct sources. Minibatch:64 improvement+64 preservation; populations and
available phases balanced within preservation, sources/phases balanced within
improvement. Every improvement input also has an output-deviation penalty to2200.
Objective:teacher MSE+4*preservation MSE+improvement-input deviation MSE.

Initialized2200 weights and Adam state. LR1e-5,100 optimizer steps; recorded Adam
counter77184 ->77284. Normalizer statistics frozen, teacher unchanged, all
gradients finite. Snapshot RGB stored FP16 and converted to FP32 for inference.
Heldout own-action MSE after training:Nominal0.002194,Wide0.003503. These small
errors did not preserve evaluated task success.

## Stage outcomes

Candidate Nominal:250 grasped,162 lifted,161 placed,161 success. Candidate Wide:
185 grasped,89 lifted,82 placed,79 success. Baseline Wide:168 grasped,89 lifted,
83 placed,80 success. More Wide grasps did not yield more lifts or successes.
Nominal again regressed principally at the grasp-to-lift transition.

## Additional collection/evaluation mismatch

The standalone collector constructs BoxCameraEnvCfg directly and does not call
play_mode. WristImageCfg enables observation corruption; collection therefore
includes image photometric augmentation and configured observation noise. The
standard playback path calls ManagerBasedRLEnvCfg.play_mode, which disables
observation corruption in all groups. Both candidate and baseline selection
screens use that same playback protocol.

Consequently, successful-action preservation here was trained/checked on
augmented collection inputs, not directly on the clean observation distribution
used by the selection screen. The earlier48-start and expanded collection scores
must not be interpreted as clean-policy accuracy estimates. This is an identified
coverage gap and possible contributor, not an established cause of the entire
regression. The experiment does not test whether adding clean-view preservation
would remedy it. Matching clean inference observations and retaining augmented
coverage is required before another preservation experiment; also verify this
setting for any future diagnostic comparison.

## Verification and limits

verification_0100.json records dataset hashes, frozen teacher/normalizer, Adam
steps, episode counts, finite gradients and offline holdout errors. Normal
student-only playback successfully loads the generated checkpoint. iter2200 is
retained for schema compatibility; offline_preservation.optimizer_updates=100
records actual work, and the filename explicitly identifies offline optimizer
steps. Dataset labels are not silently repackaged into the earlier teacher-only
reference bank.

comparison_0100.json records exact protocol/pose pairing, checkpoint identities,
paired outcomes and descriptive confidence intervals. Baseline is the existing
recorded2200 selection screen, not a newly rerun baseline. The small256-start
screens are not deployment acceptance. Audit status insufficient_coverage is
expected for this selection size; both planned jobs completed.

Independent scoped review found no blocking code issues. Its two provenance
recommendations were added:reject reused pose files with mismatched seed/count,
and require collection/episode controller identities. No production trainer or
physical-success criterion was changed. No broad production test suite was run
for these standalone research scripts; checks are the recorded numerical
invariants and real student-only simulator screens.
