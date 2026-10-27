# Checkpoint 5381 independent evaluation

Original strict acceptance gates: FAIL

The user subsequently accepted teacher 5381 at the observed grid/random scores for camera distillation.

| Population | Successes | Rate | Result |
| --- | ---: | ---: | --- |
| stage | 2020/2048 | 98.6328% | fails Stage F promotion |
| grid | 2891/3000 | 96.3667% | fail |
| random | 3282/3328 | 98.6178% | fail |
| nominal | 3072/3072 | 100.0000% | pass |
| regression | 3/3 | 100.0000% | pass |

## Comparison with archived checkpoint 4982

| Seed | 4982 | 5381 | Recovered failures | New failures |
| --- | ---: | ---: | ---: | ---: |
| 43 | 1009 | 1008 | 14 | 15 |
| 45 | 1013 | 1012 | 6 | 7 |

Both comparison seeds have exactly matching initial poses. The baseline results are archived;
a one-episode difference does not establish a meaningful performance difference.

## Findings

- Teacher 5381 is user-approved for camera distillation at the observed scores.
- The grid's far pickup corner and yaw near -180 degrees need targeted trajectory diagnosis.
- Preserve the physical success contract. No training or hardware operation was launched.

## Audit repair and provenance

The original random run rejected its first supplement because the job plan recorded
start index 256 while the command passed 0. The validator was kept intact. The source
fix propagates the planned start index into the launch command, with a regression test.
The full suite passed: 175 passed, 2 optional skips. Lint passed.

The recovered random audit reuses only the three validated original batches, with
SHA256 provenance in random_recovered/reused_batches.json. Supplements are rerun
with correct metadata. Original output and failed-attempt logs are preserved.

Per-population directories, failure starts, bins, and gates are in analysis.json.
