# Horizon audit review and verification

Five new cases failed before implementation (unsupported episode_length_s),
then the full suite passed227 tests with2 optional dependency skips and35
upstream TorchScript warnings. Ruff passed after import sorting; format checks
passed. Independent read-only review found no blocker:default commands unchanged,
finite positive duration required,override appended to initial and supplemental
jobs,default task horizon remains30. Comparisons preserve the duration argument
in protocol checks,preventing silent30-versus60 comparisons.

The new training run uses only per-run env.episode_length_s=60.0. Saved env.yaml
confirms60; training mean episode length exceeding900 control steps provides
additional runtime evidence that episodes can continue beyond30 seconds.
