# Scoped review — Wide preservation experiment

Fresh-context read-only reviewer found no blocking correctness issues in
collect.py/train_offline.py against the experiment plan and installed RSL-RL.
Reviewed target separation, successful episode filtering, sampling sizes,
normalizer immutability, optimizer restore/counters and default playback schema.
Baseline has no stale replay/assistance metadata. Keeping iter2200 is intentional;
actual new work is in offline_preservation.optimizer_updates and checkpoint name.

Two robustness gaps were identified and fixed after the first100 steps:
- Reused pose files now require matching requested seed and episode count.
- Training requires baseline-controlled expanded/student data and teacher-
  controlled demonstration data, at both collection and episode metadata levels.

Current collection JSON provenance was independently checked: expanded seed63,
256 poses, baseline/baseline/teacher controller identities. These guards change
input validation only; current checkpoint training inputs were already correct.

Recorded first-block optimizer counter:77184 ->77284. Teacher and normalizer
unchanged, finite gradients and heldout action preservation checked. Real
student-only playback screens remain the required performance evidence.
