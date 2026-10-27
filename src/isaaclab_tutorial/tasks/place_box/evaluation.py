"""Exact first-episode-per-environment accounting for box policy playback."""

import json
import os
import sys
from pathlib import Path

import torch

EXACT_EVALUATION_ACTIVE = False


def install_episode_counter():
    """Install before scene creation; stop after one episode per requested audit environment."""
    global EXACT_EVALUATION_ACTIVE
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    target = int(os.environ.get("SO101_EVALUATION_EPISODES", "1024"))
    if target <= 0:
        raise ValueError("SO101_EVALUATION_EPISODES must be positive")
    EXACT_EVALUATION_ACTIVE = True
    original_step = RslRlVecEnvWrapper.step
    argv = sys.argv[1:].copy()
    counted = None
    outcomes = []
    initial = None
    provenance = None
    reset_parameters = None

    def step(self, actions):
        nonlocal counted, initial, provenance, reset_parameters
        if self.num_envs != target:
            raise ValueError(f"Exact box evaluation requires --num_envs {target}")
        if counted is None:
            counted = torch.zeros(target, device=self.unwrapped.device, dtype=torch.bool)
            initial = self.unwrapped._box_initial_pose.clone().cpu().tolist()
            env = self.unwrapped
            if hasattr(env, "_box_reset_source"):
                from .workspace import SOURCES

                provenance = list(zip(
                    [SOURCES[int(x)] for x in env._box_reset_source.cpu().tolist()],
                    env._box_reset_case.cpu().tolist(), strict=True,
                ))
            reset_parameters = dict(getattr(env, "_box_reset_parameters", {}))
        result = original_step(self, actions)
        ids = (result[2].bool() & ~counted).nonzero().flatten()
        env = self.unwrapped
        for idx in ids.tolist():
            flags = env._box_terminal_progress[idx].tolist()
            outcomes.append(
                dict(
                    env_id=idx,
                    initial_box_pose=initial[idx],
                    **(dict(reset_source=provenance[idx][0], case_id=provenance[idx][1]) if provenance else {}),
                    **dict(zip(("grasped", "lifted", "placed", "success"), flags, strict=True)),
                    **{key: bool(value[idx]) for key, value in getattr(env, '_box_terminal_details', {}).items()},
                    dropped=bool(env.termination_manager.get_term("box_dropped")[idx]),
                    out_of_bounds=bool(env.termination_manager.get_term("box_out_of_bounds")[idx]),
                    unstable_robot=bool(env.termination_manager.get_term("unstable_robot")[idx]),
                    timed_out=bool(env.termination_manager.get_term("time_out")[idx]),
                )
            )
        counted[ids] = True
        if len(outcomes) == target:
            summary = {"episodes": target}
            for key in (
                "grasped",
                "lifted",
                "placed",
                "success",
                "dropped",
                "out_of_bounds",
                "unstable_robot",
                "timed_out",
            ):
                summary[key + "_rate"] = sum(row[key] for row in outcomes) / target
            for key in ('on_target', 'released', 'withdrawn', 'upright'):
                if key in outcomes[0]:
                    summary[key + '_rate'] = sum(row[key] for row in outcomes) / target
            print("SO101_BOX_EVAL_RESULT=" + json.dumps(summary, sort_keys=True), flush=True)
            if output := os.environ.get("SO101_EVALUATION_OUTPUT"):
                path = Path(output)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(dict(
                    summary=summary, argv=argv, reset_parameters=reset_parameters, episodes=outcomes
                ), indent=2) + "\n")
            raise SystemExit(0)
        return result

    RslRlVecEnvWrapper.step = step
    return argv
