# Wider pickup workspace: frozen-policy screening

The user requested a box-center pickup region of **+/-10 cm in X and Y**
around the current pickup point, with box rotation. The working interpretation
is any upright tabletop yaw; the paper stays fixed.

The current models do not meet this goal. Resume and broaden the **state PPO
teacher first**, then distill a camera student using the accepted wider teacher.

## Completed evaluations

Each run used the existing exact-first-episode callback with unchanged success,
physics, rewards and30-second episode limits. Only reset position/yaw noise was
overridden. No scripted assistance, policy training or hardware action occurred.

| Policy / reset distribution | Full success | Grasp | Lift | Timeouts |
| --- | ---: | ---: | ---: | ---: |
| Teacher2196: +/-5 cm, yaw +/-0.1 rad | 110/256 (42.97%) | 165 | 157 | 146 |
| Teacher2196: +/-10 cm, yaw +/-pi | 36/1,024 (3.52%) | 260 | 75 | 988 |
| Camera1599: +/-10 cm, yaw +/-pi | 49/1,024 (4.79%) | 173 | 109 | 975 |

No run recorded drop, out-of-bounds or unstable-robot terminations. Most wide
failures never achieved a sustained grasp:764 teacher episodes and851 camera
episodes. An additional185 teacher and64 camera episodes grasped without lifting.
Thus wider task performance cannot be recovered solely by improving nominal
release/withdrawal or distilling the existing teacher longer.

These are **uniform random starts, grouped into a5x5 spatial map after testing**,
not predetermined grid points. Yaw is mixed within each spatial cell. Every
run used seed42, but teacher and camera initialization consume randomness
differently: none of the1,024 corresponding initial poses match within1e-7.
Their scores describe the same requested distribution, not a paired comparison
proving one model better. Use a controlled start set for future comparisons.
The combined wide test also does not isolate yaw's effect from position's.

![Workspace screening](../logs/box_workspace_screening.png)

Each cell reports full successes / actual trials. Low sample counts, especially
the256-episode intermediate screen, do not establish a reliable operating region.
Failure to solve a start also does not establish mechanical unreachability.

## Reproducibility and validation

Saved under `logs/`:

- `box_workspace_teacher_2196_pm5cm_seed42_evaluation.json` / `.log`.
- `box_workspace_teacher_2196_pm10cm_full_yaw_seed42_evaluation.json` / `.log`.
- `box_workspace_camera_1599_pm10cm_full_yaw_seed42_evaluation.json` / `.log`.
- `box_workspace_screening_summary.json` with bins, ranges and failure stages.
- `box_workspace_screening.png` / `.svg`.
- `analyze_box_workspace.py`, the analysis script.

The script verified each audit contains the requested number of unique
environment IDs, recomputed every summary rate from individual outcomes,
checked saved XY/yaw bounds and reconciled spatial/yaw bin totals and failure
stages. The camera audit's runner resolved student inputs to wrist RGB and
proprioception. Original checkpoints were evaluated without modification.

Runtime overrides for the requested full region were:

```text
env.events.reset_home.params.position_noise=0.1
env.events.reset_home.params.yaw_noise=3.141592653589793
```

Full command arguments are embedded in each audit JSON. Existing nominal
defaults remain+/-1 cm and+/-0.1 rad; no source behavior change has been made.

## Proposed next work

[The design](superpowers/specs/2026-10-26-box-workspace-design.md) proposes
position stages2/5/10 cm, then yaw stages45/90/180 degrees, with nominal
experience retained, repeatable grid audits and separate camera acceptance.
Long runs remain user-launched. The design is pending approval before changes
to training behavior or new task tooling.
