# Wrist-camera visibility and phase-coverage diagnostic

Selected policy1798 was evaluated without training or policy/source changes. Only ordinary nominal and interior starts were replayed; no edge-case trajectory audit was performed.

![Diagnostic summary](diagnostic_summary.png)

## Paired ordinary rollouts

| Distribution | Teacher | Student1798 |
| --- | ---: | ---: |
| Nominal: +/-1cm XY, +/-0.1rad yaw | 8/8 | 6/8 |
| Interior: +/-6cm XY, +/-90deg yaw | 8/8 | 4/8 |

Each teacher/student pair used exactly matching initial poses. The interior sample actually reached5.94cm and52.9deg yaw. Eight episodes per actor/distribution are diagnostic examples, not a benchmark replacing the larger audits.

## Camera framing

- Rendered images are valid, and the box center was geometrically inside the policy view in every sampled teacher frame for both distributions. The camera is not blank.
- The box is heavily cropped. All box corners were inside the policy frame in0% of sampled nominal teacher frames and4.0% of interior teacher frames. That geometric measurement alone does not establish inadequate visual information.
- A fixed-height home-pose planar projection, fitted to8 simulator camera projections with maximum error0.000025pixels, estimates that72.4% of uniformly distributed full-workspace box centers fall inside the64x48 policy crop.87.9% lie inside the raw80x60 render. Approximately27.6% fall below the policy frame.
- This estimate covers initial geometric center visibility only. It does not measure occlusion, identity recognition, complete-box visibility or episode success. It cannot account for all observed policy failures by itself.

## Phase coverage

In the corrected interior paired rollouts, the teacher spent30.0% of control steps after lifting (including placement), while the student spent5.6%. The student spent56.1% grasped without lifting. This demonstrates different phase visitation in these rollouts; it is not a reconstruction of historical training batches.

The installed training implementation is student-controlled: the teacher labels visited states. It trains on32 control steps per environment for4 epochs, then resets the rollout buffer. There is no persistent cross-update replay or teacher-controlled demonstration collection in this configuration.

This creates a plausible coverage/forgetting mechanism when the student seldom reaches lift/placement states. Causality has not been established by this diagnostic. The [DAgger paper](https://arxiv.org/abs/1011.0686) describes retaining aggregated teacher-labelled data and optionally mixing expert actions during data collection; those are candidate changes to test, not changes made here.

## Recommended next experiments

1. Validate improved camera framing against the intended physical wrist-camera calibration: cover the pickup region and preserve useful box/paper context. Compare camera settings geometrically and in rendered inputs before retraining.
2. Separately test teacher-assisted data collection with retained, phase-balanced examples so successful grasp/lift/placement states remain represented. Keep teacher5381 and the existing physical success contract.
3. Change one mechanism at a time and compare with the same nominal/wide screens. Preserve1798 as the original baseline. No new training was started.

## Valid evidence and exclusions

- Valid nominal evidence: nominal_teacher/ and nominal_student/.
- Valid interior evidence: interior_corrected/; configured before scene creation and checked against actual poses.
- The original top-level interior_teacher/ and interior_student/ profiles accidentally reused nominal event-manager parameters; they are explicitly excluded from summary.json and this report.
- An earlier setup attempt stopped before rollouts because it bypassed Isaac Lab configuration migration. The valid runs used the standard runner factory and migration helpers.
- summary.json, validation.json and home_workspace_projection.json record the quantitative checks and limitations.
- profiles.json/observations.json store original nominal evidence; interior_corrected/ stores corrected wider-interior evidence.
- The selected checkpoint SHA256 is unchanged. No optimizer updates, checkpoint writes or per-run launch scripts were produced.
