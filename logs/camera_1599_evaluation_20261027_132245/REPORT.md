# Wider-workspace camera student evaluation

Training completed: 1,600 iterations from frozen state teacher 5381.

Checkpoint: `logs/rsl_rl/so101_box_camera_distillation/2026-10-27_12-56-22_so101-box-workspace-camera-5381/model_1599.pt`

SHA256: `4982ebd41b447b50909090d707d04ca008b85a3180674693333963d94620b8b5`

| Population | Camera successes | Camera rate | Teacher 5381 rate |
| --- | ---: | ---: | ---: |
| grid | 455/3000 | 15.17% | 96.37% |
| random | 673/3328 | 20.22% | 98.62% |
| nominal | 2152/3072 | 70.05% | 100.00% |
| regression | 3/3 | 100.00% | 100.00% |

All four audits completed with sufficient coverage. Every student start matches its recorded teacher comparison start exactly (maximum component difference: 0.0).

The camera student remains substantially behind the teacher on both the wider workspace and nominal starts. The historical regression case passed. Training completion and finite loss do not establish teacher-level task performance.

## Training verification

- All 1,600 iterations recorded; all logged scalars and final student tensors finite.
- Embedded frozen teacher matches teacher 5381 exactly.
- Mean behavior loss: first 50 iterations 0.118697; last 50 0.079409.

## Scope and next action

The user deferred edge-case inspection and additional training. Neither was started. Preserve the current checkpoints and results. This student is not selected for robot transfer on the strength of these results. Teacher 5381 remains accepted by the user at its observed performance; per-episode physical success criteria were unchanged.

No per-run launch script was generated. Evidence is in training_summary.json, results.json, comparison.json and each population directory.
