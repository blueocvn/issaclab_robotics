# Release-v3 handoff review — 2026-10-26

The release-v3 handoff review and fixes were performed after local setup and
GPU runtime verification on the current PC. The current launch command is in
[BOX_PICK_PLACE.md](BOX_PICK_PLACE.md#release-v3-continuation).

**Review scope and result**

Reviewed the actual untracked box sources and tests, not only the tracked Git
diff. An independent read-only review checked signed potential deltas, selected
environment resets, contact/support gating, unchanged success rules, terminal
snapshots, first-episode accounting and 54-dimensional checkpoint compatibility.

One diagnostic defect was found and fixed. Isaac Lab replaces `extras["log"]`
during automatic resets, losing the stage metrics on terminal steps. RSL-RL also
buffers that dictionary by reference, allowing later steps to overwrite earlier
metric samples. `BoxEnv._reset_idx` now preserves only `Metrics/*` across the
framework reset while retaining its newly generated episode logs. `BoxSuccess`
publishes a fresh dictionary each step so buffered samples retain their values.

No reward values, physics, observations, success thresholds or success duration
were changed by this review fix. Independent re-review found the diagnostic
issue resolved and no further concrete defects in the reviewed scope.

**Regression evidence**

Before the fix, terminal-metric preservation failed for both partial and full
resets, and the buffered-success regression demonstrated a recorded success
changing from 1 to 0 on the following step. After the fix:

| Check | Result | Evidence under `logs/` |
| --- | --- | --- |
| Reset regression before fix | 2 failed, 2 passed (initial empty cases) | `release_v3_metrics_reset_red.log` |
| Buffered metrics regression before fix | 1 failed | `release_v3_metrics_snapshot_red.log` |
| All box tests after fix | 37 passed | `release_v3_review_targeted_green.log` |
| Full suite after fix | 132 passed, 2 skipped | `release_v3_review_full_pytest.log` |
| Box Ruff check | All checks passed | `release_v3_review_ruff.log` |

The two skips are optional `cv2` and `leapp`; the suite emits 35 existing
TorchScript deprecation warnings. A full `git diff --check` also reports
pre-existing blank lines at EOF in `vial.usda` and `vial_labeled.usda`; those
user assets were retained without edits.

**Physical release and continuation**

Reproduced the lift-v2 scripted release on the current RTX 5070: **8/8 first
episodes succeeded**, at steps 563–568. The policy acts for 450 control steps,
then the diagnostic opens the jaws and slowly withdraws. The box stays dynamic
under the original physics and success criterion. This is scripted assistance,
not learned-policy success. See `logs/local_pc_scripted_release_summary.json`,
`logs/local_pc_scripted_release.json` and its `.log`.

The separate run
`logs/rsl_rl/so101_box_state/2026-10-26_14-14-12_local-pc-release-v3-integration/`
successfully loaded the original `model_1598.pt` and completed exactly two PPO
updates with 512 environments. Actor and critic each received 54 inputs. Both
updates had finite value, surrogate and entropy losses, and all four release
stage metrics were present. The run wrote `model_1599.pt`; it is a smoke
checkpoint, not the continuation starting point or an accepted teacher.
See `logs/local_pc_release_v3_integration.log`.

The smoke checkpoint's eight-episode audit is recorded separately in
`logs/local_pc_release_v3_smoke_audit.json` and its `.log`. It counted eight
unique first episodes and saved all four terminal stage flags/rates: 100% grasp,
lift and on-target, 0% released/withdrawn/full success, 25% upright, 100% timeout,
and no drop/out-of-bounds/unstable-robot failures. This validates the audit
pipeline; the smoke policy remains unaccepted for learned release.

**Next action**

The user launches the separate 400-update, 512-environment continuation from
the preserved lift-v2 `model_1598.pt`, using the command in the box guide.
The long run has not been started. Monitor lift preservation and the live
on-target/release/withdrawal/upright rates, then run a full 1,024-first-episode
audit on the numerically latest checkpoint. Require at least 90% full placement
success before camera distillation. No hardware operation, commit, merge or push
was performed; all pre-existing untracked feature files and checkpoints remain.
