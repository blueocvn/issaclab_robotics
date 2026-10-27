# Release-v3 trained results — 2026-10-26

The nominal state teacher **passes the agreed 90% full-placement threshold**.
Its latest checkpoint achieved **1,023/1,024 successes (99.9023%)** in an exact
first-episode audit with the original success rules and deterministic policy
actions. No scripted opening, withdrawal or other assistance was used in this
audit. Physics determinism was not enabled.

**Follow-up:** the user raised concern about the remaining failed episode. It
was reproduced and the teacher received 200 additional PPO updates. The refined
`model_2196.pt` passes the old failure and 3,072/3,072 audited first episodes
across three seeds. See [RELEASE_V3_REFINEMENT_RESULTS.md](RELEASE_V3_REFINEMENT_RESULTS.md)
for current evidence and the camera command using the refined teacher. This
document records the earlier `model_1997.pt` results.

**Run analyzed**

`logs/rsl_rl/so101_box_state/2026-10-26_14-21-24_so101-box-release-v3/`

Training resumed lift-v2 `model_1598.pt` for 400 PPO updates, global iterations
1598–1997, with 512 environments and seed 42. The numerically latest saved
checkpoint is `model_1997.pt`. TensorBoard events span about 14.28 minutes.
All recorded scalar values are finite. Training used the existing nominal
Newton MJWarp configuration; sim2real randomization was not enabled.

**Independent exact audit**

Evaluated the latest checkpoint with 1,024 environments, recording exactly one
first episode per environment. Verified 1,024 unique environment IDs and
recounted successes from the saved individual outcomes.

| Outcome | Count | Rate |
| --- | ---: | ---: |
| Sustained grasp history | 1,024 | 100% |
| Physical lift history | 1,024 | 100% |
| On target with table support at episode end | 1,024 | 100% |
| Upright on target at episode end | 1,024 | 100% |
| Released / withdrawn / full placement success | 1,023 each | 99.9023% each |
| Timeout | 1 | 0.0977% |
| Dropped / out of bounds / unstable robot | 0 each | 0% each |

Environment 527 was the only failure: it grasped, lifted and reached the paper
upright, but did not satisfy the release/withdrawal checks before timeout.
The saved outcome does not identify whether residual jaw contact or insufficient
jaw opening prevented release. No reward or success-rule changes were made for
this analysis.

Compared with lift-v2's historical 0/1,024 full successes, release-v3 has learned
the missing release/withdrawal behavior while retaining grasp and physical lift.
This acceptance result covers the nominal simulated task and configured reset
distribution, not real-hardware transfer or arbitrary paper detection.

Evidence: [audit JSON](../logs/box_release_v3_1997_evaluation.json) and
[audit log](../logs/box_release_v3_1997_evaluation.log).

**Training interpretation**

The run spent most of its budget without full successes, then improved sharply
near the end. Training's final episode-success estimate was 95.14%, with a peak
of 97.80%. Mean training reward rose from about 1.19 in the first 50 updates to
6.79 in the last 50, ending at 8.73. Episode lengths fell from the 900-step
timeout plateau as successful episodes began ending earlier.

The last-50-update success-estimate mean was only 54.80% because this window
includes the late learning transition. It is not the final policy's acceptance
score. Likewise, `Metrics/success_rate` is a per-control-step fraction and
`on_target/released/withdrawn` are live batch occupancy fractions; successful
environments immediately reset. Their small or falling values near the end
must not be interpreted as episode failure rates. The independent audit above
is the acceptance result.

The value loss increased as success rewards appeared but remained finite.
The later failure diagnosis supports a short PPO refinement experiment for
the remaining release/withdrawal case, without changing the reward design or
weakening task success.

![Training curves](../logs/box_release_v3_training_curves.png)

Full extracted data: [scalar summary](../logs/box_release_v3_training_summary.json),
[CSV](../logs/box_release_v3_training_scalars.csv), and
[vector figure](../logs/box_release_v3_training_curves.svg).

**Next action: train the wrist-camera student**

Keep the accepted state checkpoint as the frozen teacher. Launch the separate
nominal camera-distillation run yourself. From PowerShell:

```powershell
wsl -d Ubuntu -- bash -lc 'cd /mnt/d/work/robotics/IsaacLabTutorial; uv run --locked isaaclab train --rl_library rsl_rl --task IsaacTutorial-Pick-Box-SO101-Camera --num_envs 128 --max_iterations 1600 --seed 42 --run_name so101-box-camera-student --device cuda:0 --checkpoint logs/rsl_rl/so101_box_state/2026-10-26_14-21-24_so101-box-release-v3/model_1997.pt --visualizer none presets=newton_mjwarp,newton_renderer'
```

The student learns from wrist RGB history and proprioception; the frozen teacher
labels the states visited by the student. Checkpoints go to
`logs/rsl_rl/so101_box_camera_distillation/`. This command is prepared from the
existing documented configuration; the long camera run has not been launched
as part of this analysis. A 1,600-update budget does not itself establish a
successful student: audit its final camera checkpoint over 1,024 first episodes
with the camera task and matching renderer preset, using the same full-placement
criterion.

For real-robot transfer, follow the separate sim2real workflow in the box guide:
adapt and audit the state teacher under `presets=newton_mjwarp,sim2real`, then
distill the matching camera student under
`presets=newton_mjwarp,newton_renderer,sim2real`. Measure actual mass and calibrate
the workcell before supervised hardware use.
