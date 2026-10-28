# Wide failure map and fresh collection — 2026-10-28

Subsequent offline100-step training failed:62.89% Nominal,30.86% Wide. See [training report](training/REPORT.md). Camera2200 remains selected. This diagnostic completed the failure map and a
fresh paired collection; it did not train or change a policy.

## Existing 256-start selection evidence

| Outcome | Wide | Nominal |
| --- | ---: | ---: |
| Never grasped | 88 | 0 |
| Grasped, never lifted | 79 | 26 |
| Lifted, unsuccessful | 9 | 1 |
| Success | 80 | 229 |

The lowest local-Y strip (-0.20 <= Y < -0.15 m) succeeds4/61 (6.56%).
The other starts succeed76/195 (38.97%). Across the historical home-plane
projection, estimated in-crop centers succeed71/187 (37.97%), versus9/69
(13.04%) estimated outside. Never-grasp counts are46/187 and42/69 respectively.
The initial-view boundary is an estimate from the prior camera diagnostic,
not measured occlusion or per-frame visibility on these2200 episodes. Position,
reach and orientation are confounded; this association does not establish cause.

Absolute-yaw-bin success:0–45deg27/74;45–90deg11/52;90–135deg20/66;
135–180deg22/64. These bins are descriptive and have different XY samples.

![Failure map](wide_failure_map.png)

Map inputs, SHA256 identities, per-episode classifications,16 XY cells and all
bin counts are in analysis.json and episodes.json. The map generator checks
unique episode IDs, totals, expected position bounds and recorded success rates.
The existing audit poses remain evaluation-only.

## Fresh paired collection

Seed61 generates48 new poses:16 low-Y Wide,8 other estimated-outside Wide,
8 other estimated-inside Wide,16 Nominal. All Wide yaw draws span[-pi,pi].
Exact overlap with the existing nominal/Wide audit poses is rejected. The saved
fresh_poses.json defines the collection, which intentionally overweights weak
regions and is not a representative accuracy benchmark.

| Fresh stratum | Camera2200 | State teacher5381 |
| --- | ---: | ---: |
| Low Y | 3/16 | 16/16 |
| Other estimated outside | 1/8 | 8/8 |
| Other estimated inside | 2/8 | 8/8 |
| Nominal | 5/16 | 16/16 |

Both controllers use matching initial poses and unchanged physical success
criteria. The collector keeps training observation corruption enabled; playback audits disable it. This input-distribution difference, the small48-env cohort and changed reset protocol differ from the
256-start screens; its low Nominal score must not replace the89.45% screen.
The teacher succeeding at all48 starts demonstrates feasibility on these starts,
without proving that the camera observations contain sufficient information.

Pre-action snapshots every8 control steps retain two-frame policy RGB,
proprioception, phase, both camera2200 actions and frozen teacher actions.
First episodes only; automatically reset later episodes are excluded.

- baseline_collection.pt: student-visited states, including11 successful episodes
  (6 Wide,5 Nominal) with camera2200's own actions as preservation targets.
  Teacher labels on failed student trajectories are correction candidates.
- teacher_collection.pt:48 successful teacher-controlled demonstrations at the
  same starts, with both policies' predicted actions retained separately.
- Label source, controller, population, stratum, initial pose, flags and model
  identity are explicit. These raw bundles are not inputs for the existing
  teacher-only reference trainer and must not be merged silently into it.

Teacher-controlled demonstrations begin at the initial poses. Recovery from
saved failed intermediate student states has not been tested. Snapshot labels
alone do not demonstrate that executing those corrections will recover a task.
The11 successful baseline episodes are a pilot preservation set, not comprehensive
coverage. Broader fresh successful2200 coverage is needed before training.

## Next experiment

Prioritize approach/grasp and grasp-to-lift in the weak regions. Expand successful
2200 preservation coverage; keep baseline-action preservation distinct from
teacher improvement targets. Any training design must address conflicting
labels, protect independent evaluation sets and bound optimizer steps between
screens. Retain the Wide +10pp / Nominal loss <=3pp selection rule and evaluate
an eligible candidate on larger independent cohorts. No new training is queued.

## Verification

collection_verification.json records matched starts, finite observations/actions,
bounded actions, valid phases, consistent snapshot lengths, snapshot counts and
bundle hashes. initial_policy_views.png shows the actual first policy inputs.
Both collection runs checked the unchanged2200 checkpoint hash before/after,
performed zero optimizer updates and preserved the current camera configuration.
No production source, model weights, exports or physical-success criteria were
modified by this diagnostic. Logs and scripts are confined to this directory.

