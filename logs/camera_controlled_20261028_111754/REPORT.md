# Controlled camera experiment — 2026-10-28

Decision: retain camera2200. Both arms completed 400 new updates in two
200-update blocks; neither produced a nominal-eligible screened endpoint.
No new trainer is adopted and no additional training is queued.

| Checkpoint | Nominal seed42 | Wide seed45 |
| --- | --- | --- |
| Common baseline2200 | 229/256 (89.45%) | 80/256 (31.25%) |
| Control +200, model2399 | 168/256 (65.63%) | 61/256 (23.83%) |
| Data-first +200, model2399 | 158/256 (61.72%) | 73/256 (28.52%) |
| Control +400, model2598 | 45/256 (17.58%) | 29/256 (11.33%) |
| Data-first +400, model2598 | 4/256 (1.56%) | 11/256 (4.30%) |

All eight screens completed and were validated by camera_comparison with
256 episodes, exact paired initial poses, matching playback protocols and
consistent nominal/wide checkpoint identities. See comparison_200.json and
comparison_400.json for paired outcomes and descriptive bootstrap intervals.
Audit exit2 reflects standalone acceptance/coverage thresholds, not a crashed
or incomplete screen. These are selection screens, not deployment acceptance.

The protocol selects the best nominal-eligible endpoint per arm before final
1024-start confirmation. Every endpoint lost more than the allowed3 nominal
percentage points, so neither arm has an eligible candidate. The 1024-start
confirmation was therefore not run. There is no final holdout gain claim.

Both arms started from the exact2200 checkpoint and its optimizer state.
Baseline SHA256: 3f5ad0ea7dabe634c40b1c0c82d4eee8a88b3ff8624aa88ce1801686a0c817b6.
128 envs, seed/reset seed43, LR0.0001, full initial episodes, current wrist
crop, teacher5381, original physics/success criteria and20% nominal/80% full
workspace reset mixture were retained. Runs were sequential on RTX5060 Ti
16GB in WSL Ubuntu-24.04, using the existing .venv entrypoints directly.

Checkpoint verification: checkpoint_verification.json. Optimizer steps were
77184 at baseline,102784 after200 updates and128384 after400 for both arms.
Final learning rates were0.0001, student tensors finite and teacher tensors
exactly unchanged. Both second blocks logged exactly200 iterations2399–2598.
Data-first relative updates restored200 and reached400. Its persistent replay
and assistance schedule were resumed; teacher control reached zero at the end.

Run directories under logs/rsl_rl:

- so101_box_camera_distillation/20261028_111754_control_b1
- so101_box_camera_distillation/20261028_111754_control_b2
- so101_box_camera_data_first/20261028_111754_data_b1
- so101_box_camera_data_first/20261028_111754_data_b2

Audit directories: logs/camera_20261028_111754_{control,data}_{200,400}_{nominal,wide}.
Baseline audits: logs/camera_800_gate_20261027_170933/comparison_cli/2200_{nominal,stage}.
Raw training and audit logs remain alongside this report. No source code,
installed libraries, physical success gates or original checkpoints were changed.

Next action: diagnose the large nominal regression before proposing another
bounded experiment. These results establish regression, not its cause. Preserve
the raw evidence and2200 baseline; do not extend training automatically.
