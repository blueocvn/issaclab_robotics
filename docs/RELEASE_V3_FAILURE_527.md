# Reproduced release/withdrawal failure — episode 527

**Follow-up result:** the user completed the recommended PPO refinement.
`model_2196.pt` now passes this same initial pose and all 3,072 audited episodes
across three seeds. The current next action is camera distillation; see
[RELEASE_V3_REFINEMENT_RESULTS.md](RELEASE_V3_REFINEMENT_RESULTS.md). The diagnosis
and initial recommendation below are retained as history.

The 99.90% result meets the original 90% teacher threshold, but the remaining
failure is real and reproducible. After the user raised this concern, the next
recommendation is to refine the PPO state teacher before camera distillation.

Replayed `model_1997.pt` through the same 1,024-environment RSL-RL playback
pipeline and seed 42. Verified that environment 527's initial pose matches its
saved audit pose within 1e-7. No policy, reset distribution, physics, episode
timeout or success criteria were changed. Instrumentation only records the
first episode, including its terminal state before automatic reset.

| Terminal condition, step 900 | Measured | Requirement |
| --- | ---: | --- |
| Upright tilt | 6.39 degrees | At most 10 degrees |
| Bottom relative to table | -0.0022 mm | Within 3 mm, with table contact |
| Jaw gap | 58.07 mm | Greater than 50 mm |
| Fixed jaw contact | 1.01 N | No contact (task threshold 0.05 N) |
| Moving jaw contact | 1.03 N | No contact (task threshold 0.05 N) |
| Pad midpoint distance | 68.86 mm | Greater than 80 mm |
| Linear speed | 0.000286 m/s | Below 0.03 m/s |
| Angular speed | 0.00237 rad/s | Below 0.3 rad/s |

Grasp, lift, on-target containment/support and uprightness were achieved. The
failure is residual grasp plus insufficient withdrawal, not an incorrect
upright/table/velocity test. Stable placement count was zero at timeout.

The sampled trajectory briefly lost both jaw contacts on target around steps
210–270, but did not complete withdrawal and subsequently contacted the box
again. At steps 700–900 it remained gripped near the same pose. All sampled
gripper actions in that late interval were negative (mean -0.0525), commanding
small closing offsets rather than further opening. The jaw gap alone cannot
prove release: both jaws were still physically touching the box.

This is a narrow learned-policy weakness, not evidence that all release-v3
behavior is broken. Camera distillation trains imitation and is not a targeted
remedy for this teacher failure. The release reward already permits and has
learned full release in the other 1,023 audited episodes; there is not yet
evidence requiring a reward or success-rule change.

**Initial recommendation, now completed:** run an initial 200-update PPO refinement from `model_1997.pt`,
then re-evaluate the numerically latest checkpoint on the original 1,024 starts
and additional seeds. Confirm that this specific first episode succeeds and
that the original successes are retained before proceeding to the camera
student. Further PPO updates are a proposed experiment, not a guaranteed fix.

PowerShell command used for the completed refinement:

```powershell
wsl -d Ubuntu -- bash -lc 'cd /mnt/d/work/robotics/IsaacLabTutorial; uv run --locked isaaclab train --rl_library rsl_rl --task IsaacTutorial-Pick-Box-SO101 --num_envs 512 --max_iterations 200 --seed 42 --run_name so101-box-release-v3-refine --device cuda:0 --checkpoint logs/rsl_rl/so101_box_state/2026-10-26_14-21-24_so101-box-release-v3/model_1997.pt --visualizer none presets=newton_mjwarp'
```

Evidence: [terminal and sampled trace](../logs/box_release_v3_failure_527_trace.json),
[playback log](../logs/box_release_v3_failure_527_trace.log), and
[diagnostic callback](../logs/diagnose_release_failure.py). All training remains
user-launched. No camera training, long refinement, source behavior changes or
hardware operations were performed during this diagnosis.
