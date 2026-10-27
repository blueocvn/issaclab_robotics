# PPO refinement results — 2026-10-26

The refined state teacher, `model_2196.pt`, passed **3,072/3,072 full-placement
episodes** across seeds 42, 43 and 44. The previously failed episode 527 now
succeeds from the same initial pose. The next nominal stage is **camera
distillation** using this refined checkpoint as the frozen teacher.

**Run and training analysis**

Run: `logs/rsl_rl/so101_box_state/2026-10-26_14-59-07_so101-box-release-v3-refine/`.
It resumed `model_1997.pt` for 200 additional PPO updates, iterations 1997–2196,
with 512 environments, seed 42 and unchanged nominal Newton MJWarp physics.
The numerically latest saved checkpoint is `model_2196.pt`.

Training consolidated the release/withdrawal behavior rather than discovering
it only near the end as in the previous run. All recorded scalar series were
finite. TensorBoard events span about 7.39 minutes.

| Training statistic | Refined run |
| --- | ---: |
| Final episode-success estimate | 99.12% |
| Mean episode-success estimate, last 50 updates | 98.43% |
| Mean reward, last 50 updates | 9.47 |
| Mean episode length, last 50 updates | 225.42 steps, about 7.51 simulated seconds |

Training uses stochastic policy actions; the exact playback audits below use
deterministic policy actions. Live `Metrics/*` values are occupancy fractions,
not episode success probabilities, and fall when successful environments reset
sooner. They must not be used to infer loss of grasp or lift ability.

![Training comparison](../logs/box_release_v3_refinement_curves.png)

**Exact evaluation and old failure regression**

Ran three separate audits, each counting exactly one first episode per each of
1,024 environments, with the original success definition. No scripted release,
withdrawal assistance, physical parameter changes or physics determinism flag
were used.

| Seed | Successful placements | Timeouts | Drops / out-of-bounds / unstable robot |
| --- | ---: | ---: | ---: |
| 42, original starts | 1,024/1,024 | 0 | 0 / 0 / 0 |
| 43, different starts | 1,024/1,024 | 0 | 0 / 0 / 0 |
| 44, different starts | 1,024/1,024 | 0 | 0 / 0 / 0 |
| Total | **3,072/3,072 (100% tested)** | **0** | **0 / 0 / 0** |

Every episode achieved grasp, physical lift, on-target upright placement,
release, withdrawal and full stable success. Each audit contains 1,024 unique
environment IDs and every summary rate was recomputed from individual outcomes.

All 1,024 seed-42 initial poses match the earlier audit within 1e-7. Thus the
refined teacher retains the earlier 1,023 successes and passes the known failed
case, environment 527. That case's recorded flags now include `released`,
`withdrawn`, `placed` and `success`, with no timeout. For seeds 43 and 44, all
1,024 corresponding starts differ from the original seed-42 starts.

These results establish success on the tested nominal reset distribution; they
do not establish universal reliability or real-robot transfer. There is no
remaining observed failure in these audits that calls for another nominal PPO
continuation before the camera stage.

Evidence:

- [Combined validation and case 527 comparison](../logs/box_release_v3_refinement_validation.json)
- [Seed 42 outcomes](../logs/box_release_v3_refine_2196_seed42_evaluation.json)
- [Seed 43 outcomes](../logs/box_release_v3_refine_2196_seed43_evaluation.json)
- [Seed 44 outcomes](../logs/box_release_v3_refine_2196_seed44_evaluation.json)
- [Training scalar summary](../logs/box_release_v3_refine_training_summary.json)
- [Training scalars CSV](../logs/box_release_v3_refine_training_scalars.csv)
- [Vector training figure](../logs/box_release_v3_refinement_curves.svg)

**Next action: launch camera distillation**

Keep the refined checkpoint as the frozen teacher. The student acts using
wrist RGB history and joint feedback; the teacher labels the student's visited
states. This next run uses the existing distillation configuration, not PPO.

Run in PowerShell:

```powershell
wsl -d Ubuntu -- bash -lc 'cd /mnt/d/work/robotics/IsaacLabTutorial; uv run --locked isaaclab train --rl_library rsl_rl --task IsaacTutorial-Pick-Box-SO101-Camera --num_envs 128 --max_iterations 1600 --seed 42 --run_name so101-box-camera-student --device cuda:0 --checkpoint logs/rsl_rl/so101_box_state/2026-10-26_14-59-07_so101-box-release-v3-refine/model_2196.pt --visualizer none presets=newton_mjwarp,newton_renderer'
```

The long camera run has not been started by the agent. Checkpoints will be
written under `logs/rsl_rl/so101_box_camera_distillation/`. After it finishes,
evaluate its actual numerically latest checkpoint with the camera task, matching
`newton_mjwarp,newton_renderer` presets and the exact episode callback. Report
full placement outcomes and inspect any remaining student failures rather than
accepting it from imitation loss alone.

For eventual hardware transfer, follow the separate sim2real teacher adaptation,
audit and matching camera-distillation workflow in `BOX_PICK_PLACE.md`. No
hardware operation or source behavior change was made during this analysis.
