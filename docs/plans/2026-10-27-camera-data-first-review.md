# Review package

Review the explicit implementation file set, not all workspace changes:

- src/isaaclab_tutorial/tasks/place_box/distillation.py (new)
- src/isaaclab_tutorial/tasks/place_box/camera_comparison.py (new)
- src/isaaclab_tutorial/tasks/place_box/agents.py (new opt-in config)
- src/isaaclab_tutorial/tasks/place_box/__init__.py (camera-only extra key)
- tests/test_box_distillation.py (new)
- tests/test_box_camera_comparison.py (new)
- docs/CAMERA_DATA_FIRST.md (new commands/protocol)

Plan: docs/plans/2026-10-27-camera-data-first.md.
Ledger: docs/plans/2026-10-27-camera-data-first-progress.md (including Ruling).
Existing workspace/runtime/models and unrelated modifications are preserved;
there are no new commits. Branch camera-data-first. Compare named new files
in full; agents/__init__ diff against HEAD is scoped to these additions.

Evidence:200 tests passed,2 skipped; new25 tests had missing-module RED before
implementation. Ruff passed. Real-audit comparison CLI passed.8-env GPU
save/resume verified, plus standard student-only playback. A32-update full
horizon smoke is finishing; results will be added to the ledger.

Review focus:

- Pre-action phase/observation/label alignment, including reset buffer mutation.
- Episode controller lifetime across done/schedule transitions and resume.
- Replay phase balance/capacity/cursors/FP16 noise and checkpoint roundtrip.
- Frozen teacher and absence of privileged inputs from student training/inference.
- Optimizer budget128/update, fresh-only normalization and insertion ordering.
- Mixed training metrics versus pure student evaluation semantics.
- Legacy load compatibility and standard playback of new checkpoints.
- Direct commands preserve identical starting model/optimizer/reset protocol,
 200/400 endpoint labels, and relative assistance schedule across two blocks.
- Paired comparison rejects mismatched provenance/poses and implements the
 agreed10pp/3pp gates and conservative preference for unchanged trainer.

Report Critical/Important/Minor findings with file/line references; explicitly
list anything you declined to judge. Do not edit files or launch long training.
