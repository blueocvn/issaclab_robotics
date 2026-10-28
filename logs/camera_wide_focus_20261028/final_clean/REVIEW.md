# Final attempt review and execution invariants

Fresh-context read-only review found no concrete correctness issues in the
clean-preservation additions. The corruption loop matches installed
ManagerBasedRLEnvCfg.play_mode exactly. All configured observation groups are
turned off before environment creation; actual flags are saved in provenance.

Verified before training:baseline-controlled preservation collection, same
seed63 initial poses as the earlier augmented collection, no resume, unchanged
hashes for student-visited and teacher-demonstration improvement datasets.
Sampling/objective/LR/Adam restoration/100-step budget remain unchanged.

Actual execution:114/128 successful clean Nominal episodes and40/128 Wide.
Training preservation88 Nominal+31 Wide; episode holdout26 Nominal+9 Wide.
Adam77184->77284; teacher and normalizer unchanged; finite gradients. Candidate
SHA256:a8ac6e9f0370343c6b585a7af7f696678a3f6e2c4ef092b34705b01666e712c7.
All four observation-corruption flags are false. The outcome remains subject
to independent student-only simulator screening; offline errors are not accuracy.
