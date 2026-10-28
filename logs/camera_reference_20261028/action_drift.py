"""Read-only paired predictions on fixed baseline-generated first episodes."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_tasks.utils.hydra import resolve_presets
from rsl_rl.models import CNNModel, MLPModel

from isaaclab_tutorial.tasks.place_box.distillation import PhaseReplay, phase_ids
from isaaclab_tutorial.tasks.place_box.env import BoxEnv
from isaaclab_tutorial.tasks.place_box.env_cfg import BoxCameraEnvCfg
from isaaclab_tutorial.tasks.place_box.mdp import progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--population',choices=['nominal','wide'],required=True)
    parser.add_argument('--num-envs',type=int,default=32)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--controller',choices=['baseline','candidate','candidate_baseline_wrist'],default='baseline')
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Use a new output directory')
    args.output.mkdir(parents=True)
    base_path = Path('logs/rsl_rl/so101_box_camera_distillation/2026-10-27_16-26-29_so101-box-workspace-camera-1798-lr1e-4-seed43/model_2200.pt')
    end_path = Path('logs/rsl_rl/so101_box_camera_anchored/20261028_anchored_50_b1/model_2249.pt')
    base = torch.load(base_path,map_location='cpu',weights_only=False)
    end = torch.load(end_path,map_location='cpu',weights_only=False)
    seed = 42 if args.population=='nominal' else 45
    cfg = resolve_presets(BoxCameraEnvCfg(),['newton_mjwarp','newton_renderer'])
    cfg.scene.num_envs=args.num_envs
    cfg.sim.device=args.device or 'cuda:0'
    cfg.seed=seed
    cfg.configure_layout()
    cfg.events.reset_home.params.update(position_noise=0.01 if args.population=='nominal' else 0.1,
        yaw_noise=0.1 if args.population=='nominal' else 3.141592653589793,
        nominal_fraction=0.0,sampling_seed=seed)
    rows=[]
    fixed=PhaseReplay(128,seed=0)
    with launch_simulation(cfg,args):
        env=BoxEnv(cfg)
        try:
            obs,_=env.reset()
            starts=env.scene['box'].data.root_link_pose_w.torch.cpu().tolist()
            def student(state):
                model=CNNModel(obs,{'student':['wrist_rgb','proprioception']},'student',6,
                    hidden_dims=[256,128],activation='elu',obs_normalization=True,
                    distribution_cfg={'class_name':'GaussianDistribution','init_std':0.05,'std_type':'log'},
                    cnn_cfg={'output_channels':[16,32,32],'kernel_size':[5,3,3],'stride':[2,2,2],'activation':'elu'}).to(env.device)
                model.load_state_dict(state)
                return model.eval()
            baseline=student(base['student_state_dict'])
            candidate=student(end['student_state_dict'])
            teacher=MLPModel(obs,{'teacher':['teacher_state']},'teacher',6,
                hidden_dims=[256,256,128],activation='elu',obs_normalization=True,
                distribution_cfg={'class_name':'GaussianDistribution','init_std':0.2,'std_type':'log','std_range':(0.05,0.3)}).to(env.device)
            teacher.load_state_dict(base['teacher_state_dict'])
            teacher.eval()
            active=torch.ones(env.num_envs,dtype=torch.bool,device=env.device)
            terminal=torch.zeros(env.num_envs,4,dtype=torch.bool,device=env.device)
            with torch.inference_mode():
                for step in range(int(env.max_episode_length)+2):
                    phases=phase_ids(progress(env))
                    base_actions=baseline(obs).clamp(-1,1)
                    candidate_actions=candidate(obs).clamp(-1,1)
                    actions=base_actions if args.controller=='baseline' else candidate_actions.clone()
                    if args.controller=='candidate_baseline_wrist':
                        actions[:,3]=base_actions[:,3]
                    if step%4==0:
                        other=candidate_actions
                        label=teacher(obs).clamp(-1,1)
                        indices=active.nonzero().flatten()
                        fixed.add({k:v[indices] for k,v in obs.items()},label[indices],phases[indices])
                        for i in indices.tolist():
                            rows.append({'step':step,'env_id':i,'phase':int(phases[i]),
                                'baseline':base_actions[i].cpu().tolist(),'candidate':other[i].cpu().tolist(),
                                'teacher':label[i].cpu().tolist()})
                    obs,_,terminated,truncated,_=env.step(actions)
                    done=(terminated|truncated)&active
                    terminal[done]=env._box_terminal_progress[done,:4].bool()
                    active[done]=False
                    if not active.any(): break
            assert not active.any(), 'Incomplete first episodes'
            terminal=terminal.cpu().tolist()
        finally:
            env.close()
    groups={}
    for successful_only in (False,True):
        for phase in range(4):
            selected=[r for r in rows if r['phase']==phase and (not successful_only or terminal[r['env_id']][3])]
            if not selected: continue
            a=torch.tensor([r['baseline'] for r in selected])
            b=torch.tensor([r['candidate'] for r in selected])
            t=torch.tensor([r['teacher'] for r in selected])
            groups[f'{"successful" if successful_only else "all"}_phase{phase}']={
                'sampled_steps':len(selected),'episodes':len(set(r['env_id'] for r in selected)),
                'candidate_baseline_joint_rmse':(b-a).square().mean(0).sqrt().tolist(),
                'candidate_baseline_mse':float((b-a).square().mean()),
                'baseline_teacher_mse':float((a-t).square().mean()),
                'candidate_teacher_mse':float((b-t).square().mean()),
                'baseline_mean_action':a.mean(0).tolist(),'candidate_mean_action':b.mean(0).tolist()}
    report={'population':args.population,'episodes':args.num_envs,'seed':seed,
        'baseline_successes':sum(t[3] for t in terminal) if args.controller=='baseline' else None,
        'controller_successes':sum(t[3] for t in terminal),
        'baseline_stage_counts':[sum(t[i] for t in terminal) for i in range(4)],
        'initial_poses':starts,'terminal_flags':terminal,'groups':groups,
        'baseline_sha256':hashlib.sha256(base_path.read_bytes()).hexdigest(),
        'candidate_sha256':hashlib.sha256(end_path.read_bytes()).hexdigest(),
        'controller':args.controller,'candidate_actions_executed':args.controller!='baseline',
        'joint_order':['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper'],
        'interpretation':'Read-only diagnostic first episodes; hybrid wrist substitution is an intervention, not an adopted policy. Correlated sampled action errors are descriptive.'}
    torch.save(fixed.state_dict(),args.output/'fixed_baseline_observations.pt')
    (args.output/'trace.json').write_text(json.dumps(rows)+'\n')
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('initial_poses','terminal_flags')},indent=2))


if __name__=='__main__': main()
