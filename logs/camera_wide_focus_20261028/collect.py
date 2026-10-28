"""Fresh diagnostic/training-candidate collection; no gradient or optimizer updates."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_tasks.utils.hydra import resolve_presets
from rsl_rl.models import CNNModel, MLPModel
from isaaclab_tutorial.tasks.place_box.env import BoxEnv
from isaaclab_tutorial.tasks.place_box.env_cfg import BoxCameraEnvCfg
from isaaclab_tutorial.tasks.place_box.distillation import phase_ids
from isaaclab_tutorial.tasks.place_box.mdp import progress

P=Path(__file__).resolve().parent
BASE=Path('logs/rsl_rl/so101_box_camera_distillation/2026-10-27_16-26-29_so101-box-workspace-camera-1798-lr1e-4-seed43/model_2200.pt')

def make_poses(seed=61, random_per_population=0):
    rng=np.random.default_rng(seed)
    h=np.array(json.loads(Path('logs/camera_visibility_diagnostic_20261027_153417/home_workspace_projection.json').read_text())['home_projection_homography'])
    poses=[]; strata=[]
    strata_spec=[('random_wide',random_per_population),('nominal',random_per_population)] if random_per_population else [('low_y',16),('other_estimated_outside',8),('other_estimated_inside',8),('nominal',16)]
    for name,count in strata_spec:
        for _ in range(count):
            while True:
                nominal=name=='nominal'
                x=.17+rng.uniform(-.01 if nominal else -.1,.01 if nominal else .1)
                y=-.1+rng.uniform(-.01 if nominal else -.1,.01 if nominal else .1)
                uv=h@np.array([x,y,1.]); u,v=uv[:2]/uv[2]
                inside=0<=u<64 and 0<=v<48
                if nominal or name=='random_wide' or (name=='low_y' and y<-.15) or (name=='other_estimated_outside' and y>=-.15 and not inside) or (name=='other_estimated_inside' and y>=-.15 and inside): break
            yaw=rng.uniform(-.1 if nominal else -np.pi,.1 if nominal else np.pi)
            poses.append([x,y,.0805,0.,0.,float(np.sin(yaw/2)),float(np.cos(yaw/2))]); strata.append(name)
    # Prohibit overlap with known selection/confirmation poses.
    audits=[Path('logs/camera_800_gate_20261027_170933')/f for f in ('2200_stage.json','2200_nominal.json')]
    for audit in audits:
        old=np.array([e['initial_box_pose'] for e in json.loads(audit.read_text())['episodes']])
        assert np.min(np.max(np.abs(np.array(poses)[:,None,:]-old[None,:,:]),axis=2))>1e-6
    return dict(poses=poses,strata=strata,seed=seed,purpose='Fresh biased training-candidate collection, not evaluation accuracy')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller',choices=['baseline','teacher'],required=True)
    parser.add_argument('--seed',type=int,default=61)
    parser.add_argument('--random-per-population',type=int,default=0)
    parser.add_argument('--output-dir',type=Path,default=P)
    parser.add_argument('--clean-observations',action='store_true')
    add_launcher_args(parser); args=parser.parse_args()
    destination=args.output_dir; destination.mkdir(parents=True,exist_ok=True)
    output=destination/f'{args.controller}_collection.pt'
    assert not output.exists(), output
    posefile=destination/'fresh_poses.json'
    if not posefile.exists(): posefile.write_text(json.dumps(make_poses(args.seed,args.random_per_population),indent=2)+'\n')
    specification=json.loads(posefile.read_text())
    assert specification['seed']==args.seed, 'Existing pose file uses another seed'
    assert len(specification['poses'])==(2*args.random_per_population if args.random_per_population else 48), 'Existing pose count differs'
    assert len(specification['strata'])==len(specification['poses'])
    checkpoint=torch.load(BASE,map_location='cpu',weights_only=False)
    before_hash=hashlib.sha256(BASE.read_bytes()).hexdigest()
    assert before_hash=='3f5ad0ea7dabe634c40b1c0c82d4eee8a88b3ff8624aa88ce1801686a0c817b6'
    cfg=resolve_presets(BoxCameraEnvCfg(),['newton_mjwarp','newton_renderer'])
    cfg.scene.num_envs=len(specification['poses']); cfg.sim.device=args.device or 'cuda:0'; cfg.seed=args.seed
    cfg.configure_layout()
    cfg.events.reset_home.params.update(position_noise=.1,yaw_noise=np.pi,nominal_fraction=0.,sampling_seed=args.seed,reset_mode='replay',replay_path=str(posefile),grid_start_index=0)
    if args.clean_observations:
        for name in cfg.observations.__dataclass_fields__:
            group=getattr(cfg.observations,name)
            if hasattr(group,'enable_corruption'): group.enable_corruption=False
    corruption={name:bool(getattr(cfg.observations,name).enable_corruption) for name in cfg.observations.__dataclass_fields__ if hasattr(getattr(cfg.observations,name),'enable_corruption')}
    if args.clean_observations: assert corruption and not any(corruption.values())
    rows=[[] for _ in specification['poses']]
    with launch_simulation(cfg,args):
        env=BoxEnv(cfg)
        try:
            obs,_=env.reset()
            starts=env._box_initial_pose.detach().cpu()
            assert torch.allclose(starts,torch.tensor(specification['poses']),atol=2e-6,rtol=0)
            baseline=CNNModel(obs,{'student':['wrist_rgb','proprioception']},'student',6,hidden_dims=[256,128],activation='elu',obs_normalization=True,distribution_cfg={'class_name':'GaussianDistribution','init_std':.05,'std_type':'log'},cnn_cfg={'output_channels':[16,32,32],'kernel_size':[5,3,3],'stride':[2,2,2],'activation':'elu'}).to(env.device)
            baseline.load_state_dict(checkpoint['student_state_dict']); baseline.eval()
            teacher=MLPModel(obs,{'teacher':['teacher_state']},'teacher',6,hidden_dims=[256,256,128],activation='elu',obs_normalization=True,distribution_cfg={'class_name':'GaussianDistribution','init_std':.2,'std_type':'log','std_range':(.05,.3)}).to(env.device)
            teacher.load_state_dict(checkpoint['teacher_state_dict']); teacher.eval()
            active=torch.ones(env.num_envs,dtype=torch.bool,device=env.device)
            terminal=torch.zeros(env.num_envs,4,dtype=torch.bool,device=env.device)
            with torch.inference_mode():
                for step in range(int(env.max_episode_length)+2):
                    a=baseline(obs).clamp(-1,1); t=teacher(obs).clamp(-1,1)
                    if step%8==0:
                        phases=phase_ids(progress(env))
                        values={k:obs[k].detach().to('cpu',dtype=torch.float16 if k=='wrist_rgb' else torch.float32) for k in ('wrist_rgb','proprioception')}
                        values.update(baseline_actions=a.cpu(),teacher_actions=t.cpu(),phase=phases.cpu())
                        for i in active.nonzero().flatten().tolist():
                            rows[i].append(dict(step=step,**{k:v[i].clone() for k,v in values.items()}))
                    obs,_,terminated,truncated,_=env.step(a if args.controller=='baseline' else t)
                    done=(terminated|truncated)&active
                    terminal[done]=env._box_terminal_progress[done,:4].bool()
                    active[done]=False
                    if step%150==0: print(f'{args.controller}: step {step}, active {int(active.sum())}',flush=True)
                    if not active.any(): break
            assert not active.any()
            flags=terminal.cpu().tolist()
        finally: env.close()
    episodes=[]
    for i,items in enumerate(rows):
        episodes.append(dict(env_id=i,stratum=specification['strata'][i],population='nominal' if specification['strata'][i]=='nominal' else 'wide',initial_pose=starts[i],terminal_flags=flags[i],controller=args.controller,steps=torch.tensor([r['step'] for r in items]),**{k:torch.stack([r[k] for r in items]) for k in items[0] if k!='step'}))
    assert before_hash==hashlib.sha256(BASE.read_bytes()).hexdigest()
    report=dict(observation_corruption=corruption,controller=args.controller,seed=args.seed,checkpoint=str(BASE),checkpoint_sha256=before_hash,initial_poses=starts.tolist(),strata=specification['strata'],terminal_flags=flags,first_episodes_only=True,stride=8,optimizer_updates=0,counts={s:dict(n=specification['strata'].count(s),stages=[sum(flags[i][j] for i,v in enumerate(specification['strata']) if v==s) for j in range(4)]) for s in dict.fromkeys(specification['strata'])},limitations=['Biased fresh collection, not a population success estimate.','Teacher labels on baseline-visited states are correction candidates; recovery from those states has not been demonstrated.','Teacher-controlled trajectories start from matched initial poses, not saved failed intermediate states.'])
    torch.save(dict(episodes=episodes,provenance=report,label_sources={'baseline_actions':'camera2200 preservation targets on its successful episodes','teacher_actions':'frozen privileged teacher5381 labels'}),output)
    (destination/f'{args.controller}_collection.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['counts'],indent=2),flush=True)

if __name__=='__main__': main()




