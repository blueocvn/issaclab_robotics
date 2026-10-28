# Action drift and wrist intervention — 2026-10-28

The protected-reference method does not directly preserve2200's output actions:
its fixed targets come from the state teacher. It can fit those targets while
changing successful baseline behavior on other trajectories. The paired action
diagnostic demonstrates that mismatch, but does not establish the full cause of
the256-start performance regression. A wrist-only intervention did not help.

No training ran; both checkpoints remain unchanged. Camera2200 stays selected.

## Baseline-generated trajectories

Ran32 nominal first episodes(seed42) and32 wide first episodes(seed45) under
baseline2200 control. Both camera policies and frozen teacher predicted actions
from identical observations every4 control steps. All actions were clipped to
the task's[-1,1] range. Baseline successes were19/32 nominal and7/32 wide.
These new32-env cohorts are diagnostic; their pose draws and scores must not be
substituted for the256-env selection screens.

On successful baseline nominal episodes during grasp-before-lift, teacher-label
MSE was0.09971 for baseline and0.15895 for anchored50. Paired action MSE between
the camera policies was0.04734. Wrist-flex action RMSE was0.34013, the largest
component in that phase. These are normalized actions, not joint-angle errors.
Nominal pre-grasp teacher MSE was0.09099 baseline and0.08709 anchored50.

On successful baseline wide episodes, anchored50 teacher-label error was
slightly higher before grasp(0.31682 vs0.31002), but lower during grasp-before-
lift(0.14160 vs0.16922). Across all wide trajectories, anchored50 also had lower
teacher error in all phases. Thus improved teacher-action MSE does not reliably
predict preserved closed-loop success, and the nominal finding is not a universal
phase/joint explanation. Sampled steps are correlated within episodes.

## Focused intervention

Compared three controllers on exactly the same32 nominal starting poses:

| Controller | Grasped | Lifted | Placed | Success |
| --- | ---: | ---: | ---: | ---: |
| Baseline2200 |26|22|19|19/32|
| Anchored50 |31|20|19|18/32|
| Anchored50 with baseline wrist-flex action |30|23|18|17/32|

The intervention substitutes only action index3 with2200's prediction at the
currently visited state; all other action channels come from anchored50.
No policy is exported or adopted. Exact starting poses and checkpoint identities
are validated in action_drift_comparison.json, which includes paired outcomes and
descriptive bootstrap intervals. Restoring this joint did not improve success.
The small sample does not prove that wrist actions are irrelevant; it supplies
no support for treating a wrist-only patch as a remedy.

## Next experiment to design

Preserve successful baseline behavior explicitly: collect fixed successful2200
nominal/wide trajectories with2200's own actions as preservation targets. Keep
teacher-labelled improvement data separate and identify both target sources in
metadata. A preservation loss or policy-deviation constraint should be evaluated
on those baseline targets and independent success screens, rather than relying
on teacher-reference training error. This is a proposed experiment, not a verified
fix. Consider target conflict and holdout coverage
before choosing weights; no additional training is queued here.

## Evidence

- action_drift.py: diagnostic-only script, unchanged production source.
- drift_nominal/report.json and drift_wide/report.json: phase/action statistics,
  initial poses, per-env terminal flags, checkpoint hashes.
- Each directory's trace.json: sampled baseline/candidate/teacher actions.
- Each directory's fixed_baseline_observations.pt: bounded fixed observations
  and teacher labels for repeatable offline analysis; not a training reference
  bank or holdout acceptance set. Diagnostic phase rings cap128 examples/phase.
- intervention_candidate/report.json and intervention_wrist/report.json: runtime
  candidate and hybrid conditions. Their legacy baseline_stage_counts field
  refers to the executed controller; controller and controller_successes identify
  the condition explicitly.
- action_drift_comparison.json: matched-condition validation and paired counts.
