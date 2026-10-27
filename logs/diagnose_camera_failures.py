"""Trace the latest camera student's four failed first episodes without changing the task."""
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
    source=json.loads(Path('logs/box_camera_1599_seed42_evaluation.json').read_text())
    failed={r['env_id']:r for r in source['episodes'] if not r['success']}
    original_step=RslRlVecEnvWrapper.step
    state={}
    terminal={}
    traces={i:[] for i in failed}

    def capture(env, index):
        q,_,lowest=quantities(env)
        p=progress(env)
        def item(t):
            return t[index].detach().cpu().tolist()
        alignment=1-2*q['quaternion'][index,:2].square().sum()
        return dict(step=int(env.common_step_counter),
                    box_position=item(q['position']-env.scene.env_origins),
                    tilt_deg=math.degrees(math.acos(max(-1,min(1,float(alignment))))),
                    bottom_height_m=item(lowest), jaw_gap_m=item(q['jaw_distance']),
                    pad_distance_m=item(q['gripper_distance']), contacts=item(q['contacts']),
                    fixed_contact_n=item(contact_force(env,'fixed_jaw_contact')),
                    moving_contact_n=item(contact_force(env,'moving_jaw_contact')),
                    table_contact_n=item(contact_force(env,'box_table_contact')),
                    linear_speed_m_s=item(q['linear_velocity'].norm(dim=-1)),
                    angular_speed_rad_s=item(q['angular_velocity'].norm(dim=-1)),
                    grasped=item(p.grasped),lifted=item(p.lifted),placed=item(p.placed),stable_steps=item(p.stable_count),
                    **{key:item(value) for key,value in env._box_terminal_details.items()},
                    success=bool(env.termination_manager.get_term('success')[index]),
                    timed_out=bool(env.termination_manager.get_term('time_out')[index]))

    def step(self, actions):
        env=self.unwrapped
        if not state:
            state['initial_poses']={}
            for index,row in failed.items():
                actual=env._box_initial_pose[index].detach().cpu()
                assert torch.allclose(actual,torch.tensor(row['initial_box_pose']),atol=1e-7,rtol=0),index
                state['initial_poses'][index]=actual.tolist()
            original_reset=env._reset_idx

            def reset(ids):
                for index in failed:
                    if index not in terminal and bool((ids==index).any()):
                        terminal[index]=capture(env,index)
                original_reset(ids)

            env._reset_idx=reset
        result=original_step(self,actions)
        for index in failed:
            if traces[index] and traces[index][-1].get('terminal'):
                continue
            if index in terminal or env.common_step_counter%10==0:
                row=dict(terminal[index]) if index in terminal else capture(env,index)
                row.update(actions=actions[index].detach().cpu().tolist(),terminal=index in terminal)
                traces[index].append(row)
        if len(terminal)==len(failed):
            output=dict(initial_poses_match=True,**state,terminal=terminal,traces=traces)
            Path('logs/box_camera_1599_failed_episodes_trace.json').write_text(json.dumps(output,indent=2)+'\n')
            print('CAMERA_FAILURE_TERMINALS='+json.dumps(terminal,sort_keys=True),flush=True)
            raise SystemExit(0)
        return result

    RslRlVecEnvWrapper.step=step
