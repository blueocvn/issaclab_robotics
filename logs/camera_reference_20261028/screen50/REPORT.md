# Protected-reference50-update screen — 2026-10-28

Decision: retain2200. The bounded50-update experiment completed, but failed
the nominal retention gate. No continuation or1024-start confirmation is run.

| Checkpoint | Nominal seed42 | Wide seed45 |
| --- | --- | --- |
| Baseline2200 |229/256 (89.45%)|80/256 (31.25%)|
| Protected-reference +50,model2249 |181/256 (70.70%)|20/256 (7.81%)|
| Change |−18.75 percentage points|−23.44 percentage points|

The comparison validates all256 first episodes/population, same playback
protocols, checkpoint identities and exact paired starting poses. See
comparison.json for paired outcomes and descriptive bootstrap intervals.
Standalone audit exit2 reflects acceptance/coverage gates; both jobs completed.

Run: logs/rsl_rl/so101_box_camera_anchored/20261028_anchored_50_b1/model_2249.pt.
SHA256:43392107407f2980d95590238aa523e371f6aec4553838dc5119275ef680637b.
Started from exact2200 weights/optimizer,128 envs,LR0.0001,seed/reset seed43,
full initial episodes,current crop,20% nominal/80% wide reset mixture.
The original teacher5381,physics and physical success contract were unchanged.
All50 collection updates use the previously specified teacher assistance1.0;
evaluation is always student-only. Training took101.82 seconds.

## Retention mechanism checks

- Exactly50 iterations2200–2249,relative update50,optimizer83584.
- Normalizer9228288: exactly204800 fresh rollout rows added to baseline9023488.
- Teacher tensors unchanged and student tensors finite.
- All3277 reference examples exactly equal the original bank; private RNGs advanced.
- Optimizer sample totals:204800 protected,200704 rolling,413696 fresh.
- Bank SHA256 unchanged:e79f69ff7754b42fd8dadd4f24e7b672dac45f43bcfaa1dcd7db059ecb983fcf.

The reference population/phase MSE improved in all eight groups. Equal-weighted
average MSE fell from approximately0.37278 to0.02080 on the fixed bank.
This is training-reference error, not holdout success or preservation of every
earlier baseline behavior. See verification.json and reproducible verify.py.

## Interpretation

Protecting old examples and fitting them succeeded mechanically, but this bank
and mixing rule did not preserve task performance. The50-update result does not
support adopting the anchored trainer or extending training. It also does not
prove that protected references are inherently ineffective: bank diversity,
labels, sampling weights and optimization remain potential factors, not isolated
causes. Reference fitting alone cannot establish closed-loop competence.

Nominal stages:250 grasps,197 lifts,184 placed,181 successes. Wide stages:
77 grasps,33 lifts,20 placed,20 successes. The wide failure is not only final
release:179 episodes never recorded a grasp. All failed episodes timed out;
there were no dropped/out-of-bounds/unstable-robot terminations.

Next useful diagnostic: compare actions on fixed baseline-generated nominal and
wide trajectories, especially before grasp and through lift, to locate policy
drift outside the successful-teacher reference bank. Do not launch another
long run or select model2249 on its reference loss. Camera2200 stays selected.
