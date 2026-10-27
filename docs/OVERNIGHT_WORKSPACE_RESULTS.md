# Overnight workspace training results —2026-10-27

The run stopped on a **Stage F convergence limit**, not a software crash.
Stages A-E passed their screening gates. Camera distillation did not start
because the full-range state teacher did not pass its stricter promotion gate.
The launcher requested graceful shutdown at03:16 local time after preserving
the outcome and checkpoints.

## Training and exact screening

The controller completed16 blocks of200 PPO updates (3,200 updates in total),
starting from nominal teacher2196. All recorded training scalars were finite.
The original success rules, nominal physics and20% nominal training mixture
were preserved. Audit starts used100% each stage's wider distribution.

| Stage | Last/best screening result | Training blocks |
| --- | ---: | ---: |
| A: +/-2 cm, small yaw | 1,022/1,024 (99.80%) | 2 |
| B: +/-5 cm, small yaw | 1,011/1,024 (98.73%) | 1 |
| C: +/-10 cm, small yaw | 977/1,024 (95.41%) | 2 |
| D: +/-10 cm, +/-45 degrees | 983/1,024 (96.00%) | 1 |
| E: +/-10 cm, +/-90 degrees | 999/1,024 (97.56%) | 2 |
| F: +/-10 cm, full upright yaw | best1,009/1,024 (98.54%) | 8 |

Stage F results by block:845,960,982,987,1002,1009,1004,1006 successes, each
out of1,024. Saved initial poses match across all eight F audits within1e-7.
The comparison uses selection seed43; it does not establish a globally best
checkpoint. A separate, unused-seed45 diagnostic of4982 passed1,013/1,024
(98.93%), with11 timeouts and no physical-failure terminations. Its weakest
spatial bin was33/38 (86.84%) and weakest yaw bin104/107 (97.20%); it also
misses the strict Stage F promotion gate. This is an additional screen, not
the complete final grid/random/nominal/regression acceptance suite.

Best observed checkpoint:

```text
logs/rsl_rl/so101_box_state/2026-10-26_19-41-12_overnight_20261027_ppo_camera_ppo_F_block6_try0/model_4982.pt
```

Latest checkpoint5380 scored1,006/1,024 (98.24%). It is not the best observed
screening result, although the difference is too small to establish a reliable
global ranking. Continue from4982 for the proposed controlled experiment.

![Stage F results](../logs/overnight_workspace_results.png)

## Why the teacher was not promoted

The generic stage report labels>=95% as pass. The controller's Stage F gate
additionally requires>=99% overall and>=95% in every spatial/yaw bin before
the four final acceptance populations run. Checkpoint4982 missed both:

- Overall1,009/1,024 (98.54%);15 timeouts, no drop/out-of-bounds/unstable
  robot terminations.
- Worst spatial bin35/42 (83.33%), at pickup offsets X=[+6,+10] cm and
  Y=[-10,-6] cm.
- Worst yaw bin112/118 (94.92%), covering[135,180] degrees.
- Failure stages:4 never grasped,3 grasped without lifting,8 lifted without
  completing placement. Ten of15 failed starts had absolute yaw>135 degrees.
  This is broader than the old nominal release-only failure.

There was no single environment failing all eight F checkpoints. Failure
patterns and small score changes require trajectory inspection if another
controlled refinement does not help. Final grid/random/nominal/case527 gates
were not reached, so no accepted broader teacher or camera student exists yet.

## Next: user-launched PPO refinement

Use400 updates from4982 with a fresh optimizer and a **fixed learning rate
0.00005**, keeping entropy, rewards, stage F reset ranges and nominal mixture
unchanged. This tests whether smaller, stable policy updates consolidate the
remaining behavior. It is an experiment, not an established fix: the prior
adaptive F6 rate varied from0.000050625 to0.000256289 and ended0.000113906.

Run in PowerShell:

```powershell
wsl -d Ubuntu -- bash -lc 'cd /mnt/d/work/robotics/IsaacLabTutorial; uv run --locked isaaclab train --rl_library rsl_rl --task IsaacTutorial-Pick-Box-SO101 --num_envs 512 --max_iterations 400 --seed 42 --run_name so101-box-workspace-F-refine --device cuda:0 --checkpoint logs/rsl_rl/so101_box_state/2026-10-26_19-41-12_overnight_20261027_ppo_camera_ppo_F_block6_try0/model_4982.pt --no-resume_optimizer_state --visualizer none presets=newton_mjwarp agent.algorithm.learning_rate=0.00005 agent.algorithm.schedule=fixed env.events.reset_home.params.position_noise=0.1 env.events.reset_home.params.yaw_noise=3.141592653589793 env.events.reset_home.params.nominal_fraction=0.2'
```

The negative optimizer flag is necessary so the checkpoint's previous optimizer
and learning rate do not override this experiment. Actor, critic, normalization
and iteration are restored. The installed RSL code accepts the fixed schedule.
No new training has been launched by the agent. Do not restart the overnight
controller from A. After this block, compare the actual saved checkpoint on
original seed43 and held-out starts, then run the final acceptance populations
before camera distillation. If it plateaus, inspect failures before adding
more identical blocks or changing the task contract.

## Evidence

- Original controller report: `logs/overnight_20261027_ppo_camera/SUMMARY.md`,
  `status.json`, `journal.jsonl` and per-block audit JSON.
- Recomputed unique episode counts, all stage rates, matching F starts, failure
  classifications and finite scalar checks: `logs/overnight_workspace_analysis.json`.
- Analysis/figure scripts: `logs/analyze_overnight_workspace.py` and
  `logs/plot_overnight_workspace.py`.
- Additional frozen-policy evaluation:
  `logs/workspace_4982_holdout45_20261027/`; recomputed counts and strict gate:
  `logs/workspace_4982_holdout45_verification.json`.
