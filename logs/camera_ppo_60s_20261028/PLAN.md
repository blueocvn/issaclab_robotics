#60-second Camera PPO continuation

User authorized60-second episode limit and500 additional PPO iterations.
Start from completed1000-iteration model1048 with native actor/critic/Adam.
Only episode_length_s changes from30 to60; retain other PPO/reset/camera/physics
and physical-success settings. Resume counter1048 through1547 (500 updates),
expected Adam16800->24800 and actor normalizer17625088->21721088. Full initial
episodes,128envs,seed67,20%nominal80%Wide. Save every100 iterations.
Remaining prior evaluation jobs were stopped on user steering; retain completed
raw results and mark prior selection incomplete. Final source screen was255/256
Nominal and94/256 Wide at30 seconds; do not compare that directly with60-second
success as a training gain. Evaluate source and endpoint using the same60-second
horizon. No change to default30-second task configuration or hardware actions.
