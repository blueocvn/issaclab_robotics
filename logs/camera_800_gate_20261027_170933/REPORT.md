# Completed 800-update run gate

| Checkpoint | Nominal success | Wide success | Nominal eligible | Wide gain vs1798 |
| --- | --- | --- | --- | --- |
| 1798 | 225/256 (87.89%) | 57/256 (22.27%) | True | +0.00 pp |
| 2000 | 3/256 (1.17%) | 27/256 (10.55%) | False | -11.72 pp |
| 2200 | 229/256 (89.45%) | 80/256 (31.25%) | True | +8.98 pp |
| 2400 | 34/256 (13.28%) | 62/256 (24.22%) | False | +1.95 pp |
| 2597 | 98/256 (38.28%) | 48/256 (18.75%) | False | -3.52 pp |

Select **2200** as the common experimental baseline. No candidate combined a wide gain of at least10 percentage points with a nominal drop of at most3 points. Therefore no1024-start confirmation was triggered; implement the opt-in data-first trainer.

All10 fresh play processes exited0; workspace_audit.read_result validated every result against the saved command/reset provenance. All256 initial poses matched exactly per population, student inference only, frozen teacher, current crop and unchanged physical success criteria. Checkpoint SHA256 hashes are unchanged. These256-start screens are selection evidence, not full deployment acceptance.

The completed800-update run had finite behavior loss throughout, last100 mean0.122597 and minimum0.094176. This loss is not episode accuracy. Checkpoint tensors/teacher equality/optimizer steps are recorded in checkpoint_verification.json.
