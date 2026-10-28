# Regression diagnosis — 2026-10-28

The observed task regression is predominantly a failure to lift after grasp.
Data-first also demonstrates forgetting of earlier replay examples while fitting
its current replay. This supports a distribution/retention problem; it does not
establish the complete causal explanation for both trainers' success collapse.
No training, policy modification or installed-library modification was performed
during this diagnosis. Camera2200 remains selected.

## Task failure location

Counts are completed independent student-only screens,256 episodes each.

| Arm/population | Updates | Grasped | Lifted | Placed | Success |
| --- | ---: | ---: | ---: | ---: | ---: |
| Control nominal |200|255|242|228|168|
| Control nominal |400|256|52|52|45|
| Data-first nominal |200|254|182|164|158|
| Data-first nominal |400|239|5|5|4|
| Control wide |200|126|81|75|61|
| Control wide |400|181|39|31|29|
| Data-first wide |200|208|83|75|73|
| Data-first wide |400|87|20|13|11|

For data-first400 nominal,234 episodes grasped without lifting,17 never grasped,
and only one failed after lifting. Thus the nominal collapse is not mainly a
release/withdrawal failure. Control400 also primarily fails between grasp and lift.
Source: each audit's summary.json and validated first-episode result files.

## Checks against implementation/state explanations

- Exact checkpoint teacher weights, finite student weights, optimizer-step counts,
  LR0.0001, matched reset poses and playback protocols were verified previously
  in checkpoint_verification.json and comparison_{200,400}.json.
- Normalizer counts are9023488 baseline,9842688 after200,10661888 after400,
  exactly819200 fresh transitions per200-update block in both arms. Replay was
  not counted as fresh normalization data.
- Maximum normalizer mean shift is less than0.017 baseline standard deviations.
  Standard-deviation ratios remain approximately0.990–1.016 across checkpoints.
- Replacing each model's normalizer with the baseline normalizer barely changes
  its fixed-replay imitation errors. For data400 on earlier replay, phase errors
  change0.36391/0.33812/0.24562/0.26758 to0.36360/0.33666/0.24641/0.26622.
  This weakens normalization drift as the main explanation. No simulator
  normalizer-swap success audit was run; small output changes can still matter
  in closed-loop contact tasks.
- The installed upstream Distillation and local DataFirstDistillation were read.
  Both update the normalizer after optimization. No shared numerical failure
  or checkpoint-load defect was identified by these checks.

## Replay retention and measured forgetting

The diagnostic script samples256 examples per phase from each saved data-first
replay with seed0. These are training replay examples, not held-out task accuracy.
All models see exactly the same observations/labels. Four phases are equally
weighted in the following average MSE values:

| Model | Replay saved at200 | Replay saved at400 |
| --- | ---: | ---: |
| Baseline2200 |0.39791|0.41487|
| Control200 |0.42362|0.44214|
| Control400 |0.40248|0.37943|
| Data-first200 |0.13602|0.31287|
| Data-first400 |0.30381|0.13061|

Data-first400 is better on its current replay but worse on the fixed older set.
Earlier-phase errors also rise: approach0.13103→0.36391 and grasp0.14113→0.33812.
This is measured forgetting on these examples, not proof that their error alone
causes the nominal success decline. Baseline task success despite higher replay
MSE further shows that this loss is not a task-success metric.

Every4096-capacity phase ring replaced all rows present at200 updates. Exact
fingerprints of proprioception plus labels have zero intersection between the
two checkpoints in every phase. A retained row would necessarily share that
fingerprint. Block2 inserted454659 approach,320180 grasp,33087 lift and11274
placed transitions, each above its ring capacity. Phase balancing therefore
preserves phase occupancy, but does not preserve earlier demonstrations, nominal
examples or successful trajectories within each phase.

Data-first replay loss remains around0.12–0.13 late in training despite poor
independent success. Neither a full ring nor low current-replay MSE establishes
retention of an earlier behavior or task competence.

## Interpretation and next diagnostic

The strongest supported mechanism is replacement of earlier examples plus
forgetting as the visited-state distribution changes. It remains a hypothesis
for the task-level collapse; control has no persistent replay and also regresses.
Possible shared causes such as ambiguous visual observations, changed visitation,
optimization drift and teacher-label conflict have not been isolated.

Before another training experiment, use fixed baseline/teacher trajectories as
a diagnostic reference and compare per-joint student errors specifically around
the grasp-to-lift transition. Any later retention experiment should distinguish
fixed reference examples from rolling phase replay and compare against the same
baseline under a bounded budget. This report does not authorize an automatic
extension of the completed experiment or claim a validated fix.

Evidence: diagnose_checkpoints.py (reproducible read-only offline diagnostic),
checkpoint_diagnostics.json (normalization, parameter change, sampled replay
errors and training-loss summaries), replay_turnover.json (exact retention checks).
