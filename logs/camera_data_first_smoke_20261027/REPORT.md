# Data-first trainer GPU smoke

This is compatibility and collection verification, not an accuracy comparison.
No 400-update experimental arm was launched.

- Eight environments, two updates from legacy camera2200, save, then two
  resumed updates. Relative counters2→4, replay512→1024 rows, Adam77440→77696.
  Normalizer count increased by exactly256 per update (fresh data only).
- An additional32 updates from that smoke checkpoint exercised a full episode
  horizon. Relative counter36; Adam81792; total normalizer increment9216.
  Replay phase occupancies were4096 approach,1786 grasp,1426 lift,395 placed.
  Every phase was represented; the approach ring reached its configured bound.
- Terminal metrics recorded28 teacher episodes,28 teacher physical successes,
  zero student episodes and teacher control fraction1. These assisted episodes
  must not be reported as student accuracy.
- All embedded teacher tensors remained exactly equal to baseline2200.
  Student tensors were finite at every checked checkpoint.
- Standard default-agent student-only playback loaded the new checkpoint and
  completed all8 first episodes (0 successes after only4 smoke updates).
  This confirms format/inference compatibility, not learned improvement.

Commands and raw logs: commands.json, phase_command.json, a.log, b.log,
phase.log, play_plan.json, play.log, play.json.
Assertions and hashes: verification.json and phase_verification.json.

CPU verification at this point:200 tests passed,2 optional-dependency skips;
25 new trainer/comparison tests had observed RED missing-module baselines and
then passed. Changed-file Ruff checks and formatting passed. The comparison CLI
also validated real matched audit files and reproduced the8.984375-point gain
that correctly failed the10-point adoption gate.

Final verification after independent review:203 tests passed,2 skipped.
The review's single Important finding (mixed-checkpoint nominal/wide inputs)
was reproduced by3 failing regression cases and fixed; all7 comparison tests
and the full suite then passed. Lint passed. The corrected CLI's real-audit
result includes exact checkpoint hashes in comparison_verified.json under the
gate's comparison_cli folder. Documented Bash syntax and both resolved Hydra
training configurations also passed (command_verification.json).
