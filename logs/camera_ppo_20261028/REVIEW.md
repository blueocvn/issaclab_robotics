# Camera PPO review and verification record

Independent read-only review covered student-to-actor transfer, fresh critic/
Adam, actor/critic observation separation, native checkpoints, observation-buffer
ownership, normalizer update counts and optional audit-agent propagation. The
review found no remaining blocking issue in the final implementation.

The first GPU launch exposed a configuration-boundary gap:raw agent.to_dict
included obsolete stochastic fields. It failed before any optimizer update.
A new actual native-runner construction regression failed before the fix, then
passed after using installed-version migration, the standard runner factory and
canonical actor-config serialization. The reviewer checked that exact wiring.

Final verification:222 tests passed,2 optional dependency skips (cv2/leapp),
35 upstream TorchScript warnings. Changed-file Ruff passed. Actual GPU trial_v2
initialized with every actor tensor equal to2200 and maximum action error0.0.
Critic stayed unchanged during transfer and Adam was empty. It completed50 new
PPO iterations/800 optimizer steps with finite actor/critic. Actor normalizer
count9023488->9433088 equals409600 new rollout transitions.

Native pre-training checkpoint is trial_v2/initial.pt, separate from all training
saves. Source2200 remains immutable. Model49 is the50-iteration endpoint. These
mechanics checks do not establish performance; matched native initial/endpoint
screens and comparison.json record actual success and disposition.
