"""Read-only capture of the first reset batch during the two-update smoke."""
import json
from pathlib import Path


def install():
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    original = RslRlVecEnvWrapper.step
    captured = False

    def step(self, actions):
        nonlocal captured
        if not captured:
            env = self.unwrapped
            Path('logs/workspace_A_smoke_reset.json').write_text(json.dumps(dict(
                parameters=env._box_reset_parameters,
                poses=env._box_initial_pose.cpu().tolist(),
                sources=env._box_reset_source.cpu().tolist(),
            ), indent=2))
            captured = True
        return original(self, actions)

    RslRlVecEnvWrapper.step = step
