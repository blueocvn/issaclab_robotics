# Completed400-update camera distillation experiment

The planned two200-update blocks completed at learning rate0.0001. Block1 reset the optimizer; block2 correctly resumed its state (25,600 to51,200 optimizer steps). All logged scalars and candidate student tensors were finite; frozen teacher weights matched5381 exactly.

| Checkpoint | Nominal | Wide workspace |
| --- | ---: | ---: |
| 1599 | 171/256 (66.80%) | 54/256 (21.09%) |
| 1700 | 0/256 (0.00%) | 23/256 (8.98%) |
| 1798 | 231/256 (90.23%) | 54/256 (21.09%) |
| 1900 | 17/256 (6.64%) | 27/256 (10.55%) |
| 1997 | 164/256 (64.06%) | 14/256 (5.47%) |

All initial poses match the original1599 screen exactly. These256-start cohorts are checkpoint-selection evidence, not full acceptance.

## Selection

Retain **1798**: `logs/rsl_rl/so101_box_camera_distillation/2026-10-27_14-16-20_so101-box-workspace-camera-1599-lr1e-4-block1/model_1798.pt`

SHA256: `92e88f27e5f8e9b5817e42742d779f6c41d96fe50b983e9585212cf5529aebf6`

It tied1599 on wide successes and improved nominal success from171/256 to231/256. Neither1900 nor1997 improved this result; the latest checkpoint1997 is not selected. The400-update experiment improved the best saved nominal result but demonstrated no wide-workspace improvement.

## Remaining work

The large performance swings and weak wider-workspace results remain unresolved. Preserve the checkpoints and raw evidence. The next useful diagnostic is ordinary wrist-camera visibility and training-data coverage before further identical training. Edge-case trajectory inspection remains deferred by the user. No training beyond the planned400 additional updates was launched.

Evidence: training_summary.json, results.json, selection.json and each population directory. No per-run launch script was created.
