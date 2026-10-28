# Protected reference implementation verification — 2026-10-28

Implemented an opt-in anchored trainer with64 fresh,32 rolling and32 protected
examples per128-row minibatch. The first update uses96 fresh/32 references while
rolling replay is empty. Existing default and data-first trainers remain available.

Collected32/32 successful nominal and32/32 successful wide teacher episodes,
seed43,8 environments, current crop/nominal physics, frozen teacher embedded in
camera2200. First episodes only; copied pre-action observations/labels every4
control steps. Failed episodes are excluded by bank construction.

Prepared bank: reference.pt, SHA256
e79f69ff7754b42fd8dadd4f24e7b672dac45f43bcfaa1dcd7db059ecb983fcf.
3277 examples: nominal phase sizes[662,227,255,108],wide[1024,612,279,110].
The raw nominal.pt and wide.pt preserve episode flags, observations/labels,
phases, initial poses, reset/teacher/checkpoint provenance. References are training
data, not acceptance evidence. Sparse late phases have limited diversity.

## Automated checks

- Initial baseline:203 tests passed,2 optional skips.
- New module/collector tests were observed failing before implementation.
- Independent review found one Important missing-bank-path resume defect.
  A regression test reproduced FileNotFoundError, then passed after deferred
  external loading. No other Critical/Important/Minor findings were reported.
- Final suite:214 passed,2 skipped(cv2/leapp),35 upstream TorchScript deprecation
  warnings,55.67s. Changed-file Ruff check and format passed.
- Documented Bash fences passed bash -n.

## GPU initialization/resume

Eight-env two-update initialization from2200 and its optimizer, then two-update
resume with deliberately missing original reference_path. Both completed.

| State | Relative updates | Optimizer steps | Normalizer count |
| --- | ---: | ---: | ---: |
| Baseline |0|77184|9023488|
| First smoke |2|77440|9024000|
| Resumed smoke |4|77696|9024512|

All reference tensors exactly equal the original bank in both checkpoints.
Private reference RNGs advanced and persisted; teacher tensors unchanged,
student tensors finite. Normalization added only256 fresh rollout rows/update.
Actual optimizer batches logged4096 reference samples/update; rolling samples
0 then4096; fresh samples12288 then8192. Resume logged4096 reference,4096
rolling and8192 fresh per update. See smoke/verification.json.

## Student-only compatibility and limits

Default camera playback loaded model_2202.pt from the anchored resume and
completed8 nominal first episodes. Result0/8 successful,2 grasps,0 lifts,8
timeouts. Audit summary is insufficient_coverage and fails acceptance, as expected
for a compatibility smoke of an unselected checkpoint. The result is recorded;
it is not evidence that the retention change improves accuracy. See
smoke/playback/{manifest,summary,group42_batch0}.json.

Camera2200 remains selected. No long training or50-update screening block was
launched. Fixed-bank immutability is verified; a task-performance fix is not.
Next bounded screening command and source/state compatibility are documented in
docs/CAMERA_PROTECTED_REFERENCE.md. Preserve original checkpoints/evidence.
