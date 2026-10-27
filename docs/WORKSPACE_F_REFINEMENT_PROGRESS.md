# Stage F refinement: in-progress analysis

Snapshot:2026-10-27,10:28 Asia/Saigon. This is a live training snapshot, not
an acceptance evaluation or a completed-run result.

Run: `logs/rsl_rl/so101_box_state/2026-10-27_03-19-50_so101-box-workspace-F-refine`.
It continues teacher4982 for400 PPO updates with512 environments, a fresh
optimizer and the intended fixed learning rate0.00005. Full stage F bounds and
20% nominal reset mixture are confirmed in the saved configuration/process.

The analyzed event file contained235 updates, iterations4982-5216 (58.75%).
Latest saved checkpoint at that snapshot: `model_5200.pt`. Observed event span
was8.24 minutes; remaining time at that pace was approximately6 minutes.

| Statistic | Updates51-100 mean | Last50 updates mean |
| --- | ---: | ---: |
| Reported training episode success | 99.07% | 99.22% |
| Timeout estimate | 0.93% | 0.78% |
| Mean reward | 9.439 | 9.489 |
| Mean episode length | 235.23 steps | 229.37 steps (7.65 seconds) |
| Value loss | 0.01965 | 0.01541 |

All recorded scalar series were finite. Learning rate stayed fixed at0.00005.
No drops, out-of-bounds or unstable-robot terminations were recorded. Policy
action standard deviation remained stable near0.295 within the configured
0.30 cap. These observations support continuing the current run; they do not
establish improved deterministic policy acceptance.

The first50 updates include reset/warmup effects, so the table uses51-100 for
the early comparison. Differences in these overlapping training estimates are
modest and should not be treated as independent statistical evidence.

Next:let the400-update run finish without interruption or setting changes.
Then evaluate the actual saved checkpoint on100% wider starts, comparing
selection seed43 and held-out starts, and check nominal regression. The prior
teacher's98.54%/98.93% exact audit scores cannot be directly compared with this
run's99.22% training estimate: the distributions and action sampling differ.
Camera distillation remains gated on independent teacher validation.

Evidence: `logs/box_workspace_F_refine_training_summary.json`,
`box_workspace_F_refine_training_series.json` and
`box_workspace_F_refine_training_analysis.log`. Reading these artifacts did
not interrupt or modify the user's active training process.
