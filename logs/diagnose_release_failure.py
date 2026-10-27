"""Read-only terminal trace for the failed first episode, using the actual playback pipeline."""
import json
import math
from pathlib import Path

import torch


def install():
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tutorial.tasks.place_box import evaluation
    from isaaclab_tutorial.tasks.place_box.mdp import progress, quantities
    from isaaclab_tutorial.tasks.place_vial.mdp.task import contact_force

    evaluation.EXACT_EVALUATION_ACTIVE = True
    source = json.loads(Path('logs/box_release_v3_1997_evaluation.json').read_text())
    failed = next(row for row in source['episodes'] if not row['success'])
    index = failed['env_id']
    original_step = RslRlVecEnvWrapper.step
    rows = []
    state = {}

    def snapshot(env):
        q, _, lowest = quantities(env)
        p = progress(env)
        def item(tensor):
            return tensor[index].detach().cpu().tolist()
        alignment = 1 - 2 * q['quaternion'][index, :2].square().sum()
        return dict(step=int(env.common_step_counter),
                    box_position=item(q['position'] - env.scene.env_origins),
                    quaternion=item(q['quaternion']),
                    bottom_height_m=item(lowest),
                    tilt_deg=math.degrees(math.acos(max(-1, min(1, float(alignment))))),
                    jaw_gap_m=item(q['jaw_distance']), pad_distance_m=item(q['gripper_distance']),
                    contacts=item(q['contacts']),
                    fixed_contact_n=item(contact_force(env, 'fixed_jaw_contact')),
                    moving_contact_n=item(contact_force(env, 'moving_jaw_contact')),
                    table_contact_n=item(contact_force(env, 'box_table_contact')),
                    linear_speed_m_s=item(q['linear_velocity'].norm(dim=-1)),
                    angular_speed_rad_s=item(q['angular_velocity'].norm(dim=-1)),
                    grasped=item(p.grasped), lifted=item(p.lifted), placed=item(p.placed),
                    stable_steps=item(p.stable_count),
                    **{key: item(value) for key, value in env._box_terminal_details.items()},
                    success=bool(env.termination_manager.get_term('success')[index]),
                    timed_out=bool(env.termination_manager.get_term('time_out')[index]))

    def step(self, actions):
        env = self.unwrapped
        if not state:
            initial = env._box_initial_pose[index].detach().cpu()
            expected = torch.tensor(failed['initial_box_pose'])
            state['initial_pose'] = initial.tolist()
            state['matches_audited_initial_pose'] = torch.allclose(initial, expected, atol=1e-7, rtol=0)
            if not state['matches_audited_initial_pose']:
                raise RuntimeError(f'Initial pose mismatch: {initial.tolist()} vs {expected.tolist()}')
            original_reset = env._reset_idx

            def reset(ids):
                if bool((ids == index).any()):
                    state['terminal'] = snapshot(env)
                original_reset(ids)

            env._reset_idx = reset
        result = original_step(self, actions)
        terminal = state.get('terminal')
        if env.common_step_counter % 5 == 0 or terminal:
            row = terminal or snapshot(env)
            row['actions'] = actions[index].detach().cpu().tolist()
            rows.append(row)
        if terminal:
            payload = dict(env_id=index, num_envs=self.num_envs, source_audit=str(Path('logs/box_release_v3_1997_evaluation.json')),
                           **state, trace=rows)
            Path('logs/box_release_v3_failure_527_trace.json').write_text(json.dumps(payload, indent=2)+'\n')
            print('BOX_FAILED_EPISODE_TRACE=' + json.dumps(dict(env_id=index, initial_pose_matches=True, terminal=terminal)), flush=True)
            raise SystemExit(0)
        return result

    RslRlVecEnvWrapper.step = step
