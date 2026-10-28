"""Bounded offline research experiment; preserve own-policy targets explicitly."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import torch
from tensordict import TensorDict
from rsl_rl.models import CNNModel

P=Path(__file__).resolve().parent
BASE=Path('logs/rsl_rl/so101_box_camera_distillation/2026-10-27_16-26-29_so101-box-workspace-camera-1798-lr1e-4-seed43/model_2200.pt')
KEYS=('wrist_rgb','proprioception')

def fingerprint(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def pool(episodes,kind):
    assert episodes,kind
    value={k:torch.cat([e[k] for e in episodes]) for k in (*KEYS,'baseline_actions','teacher_actions','phase')}
    assert all(torch.isfinite(value[k]).all() for k in (*KEYS,'baseline_actions','teacher_actions'))
    assert all(value[k].abs().max()<=1 for k in ('baseline_actions','teacher_actions'))
    assert all((value['phase']==i).any() for i in range(4)),kind
    value['indices']=[(value['phase']==i).nonzero().flatten() for i in range(4)]
    return value

def sample(pool,rng):
    ids=torch.cat([idx[torch.randint(len(idx),(8,),generator=rng)] for idx in pool['indices']])
    return {k:pool[k][ids].to('cuda',dtype=torch.float32) for k in (*KEYS,'baseline_actions','teacher_actions')}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',type=int,default=100)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--preservation-collection',type=Path,default=P/'expanded/baseline_collection.pt')
    parser.add_argument('--output-dir',type=Path,default=P/'training')
    args=parser.parse_args()
    assert args.steps==100,'Exactly100 optimizer steps per block'
    torch.set_num_threads(4); torch.manual_seed(67); torch.cuda.manual_seed_all(67)
    paths=[args.preservation_collection,P/'baseline_collection.pt',P/'teacher_collection.pt']
    identities={str(p):fingerprint(p) for p in paths}
    expanded,visited,demos=[torch.load(p,map_location='cpu',weights_only=False) for p in paths]
    assert expanded['provenance']['controller']==visited['provenance']['controller']=='baseline'
    assert demos['provenance']['controller']=='teacher'
    assert all(e['controller']=='baseline' for data in (expanded,visited) for e in data['episodes'])
    assert all(e['controller']=='teacher' for e in demos['episodes'])
    baseline=torch.load(BASE,map_location='cpu',weights_only=False)
    base_hash=fingerprint(BASE)
    assert all(d['provenance']['checkpoint_sha256']==base_hash for d in (expanded,visited,demos))
    if args.preservation_collection != P/'expanded/baseline_collection.pt':
        assert args.resume is None, 'Final attempt starts from2200; no continuation'
        assert expanded['provenance']['observation_corruption'] and not any(expanded['provenance']['observation_corruption'].values())
        prior=json.loads((P/'training/verification_0100.json').read_text())['dataset_sha256']
        assert all(identities[str(path)]==prior[str(path)] for path in paths[1:]), 'Improvement data changed'
        old=json.loads((P/'expanded/baseline_collection.json').read_text())
        assert expanded['provenance']['initial_poses']==old['initial_poses'], 'Preservation poses changed'
    successful=[e for e in expanded['episodes'] if e['terminal_flags'][3]]
    training=[e for e in successful if e['env_id']%5!=0]
    heldout=[e for e in successful if e['env_id']%5==0]
    preservation={pop:pool([e for e in training if e['population']==pop],pop) for pop in ('nominal','wide')}
    # Successful teacher demonstrations and labels on student-visited Wide states are distinct sources.
    improvement={name:pool([e for e in data['episodes'] if e['population']=='wide' and e['env_id']%5!=0 and (name!='teacher' or e['terminal_flags'][3])],name) for name,data in [('student',visited),('teacher',demos)]}
    example=next(iter(preservation.values()))
    obs=TensorDict({k:example[k][:1].float().cuda() for k in KEYS},[1])
    model=CNNModel(obs,{'student':list(KEYS)},'student',6,hidden_dims=[256,128],activation='elu',obs_normalization=True,distribution_cfg={'class_name':'GaussianDistribution','init_std':.05,'std_type':'log'},cnn_cfg={'output_channels':[16,32,32],'kernel_size':[5,3,3],'stride':[2,2,2],'activation':'elu'}).cuda().eval()
    checkpoint=torch.load(args.resume,map_location='cpu',weights_only=False) if args.resume else baseline
    if args.resume: assert checkpoint['offline_preservation']['dataset_sha256']==identities
    model.load_state_dict(checkpoint['student_state_dict'])
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-5)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    for g in optimizer.param_groups:g['lr']=1e-5
    rng=torch.Generator().manual_seed(67)
    previous=0
    if args.resume:
        rng.set_state(checkpoint['offline_preservation']['rng'])
        previous=checkpoint['offline_preservation']['optimizer_updates']
    before_steps=sorted({int(s['step']) for s in optimizer.state.values() if 'step' in s})
    metrics=[]
    for step in range(args.steps):
        chunks=[sample(improvement[s],rng) for s in ('student','teacher')]+[sample(preservation[p],rng) for p in ('nominal','wide')]
        batch=TensorDict({k:torch.cat([c[k] for c in chunks]) for k in KEYS},[128])
        target=torch.cat([c['teacher_actions'] for c in chunks[:2]])
        prior=torch.cat([c['baseline_actions'] for c in chunks[:2]])
        protected=torch.cat([c['baseline_actions'] for c in chunks[2:]])
        predicted=model(batch).clamp(-1,1)
        teach=(predicted[:64]-target).square().mean()
        preserve=(predicted[64:]-protected).square().mean()
        deviation=(predicted[:64]-prior).square().mean()
        loss=teach+4*preserve+deviation
        assert torch.isfinite(loss)
        optimizer.zero_grad(); loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        gradnorm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        metrics.append(dict(step=previous+step+1,teacher_mse=float(teach.detach()),preservation_mse=float(preserve.detach()),deviation_mse=float(deviation.detach()),gradient_norm=float(gradnorm)))
    state=model.state_dict()
    assert all(torch.isfinite(v).all() for v in state.values())
    assert all(torch.equal(v.cpu(),baseline['student_state_dict'][k]) for k,v in state.items() if 'normalizer' in k)
    steps=sorted({int(s['step']) for s in optimizer.state.values() if 'step' in s})
    assert steps==[v+100 for v in before_steps]
    holdout={}
    with torch.inference_mode():
        for pop in ('nominal','wide'):
            rows=[e for e in heldout if e['population']==pop]
            errors=[]
            for e in rows:
                for start in range(0,len(e['steps']),128):
                    observation=TensorDict({k:e[k][start:start+128].float().cuda() for k in KEYS},[len(e['steps'][start:start+128])])
                    errors.append((model(observation).clamp(-1,1)-e['baseline_actions'][start:start+128].cuda()).square().cpu())
            assert errors,pop
            holdout[pop]=dict(episodes=len(rows),mse=float(torch.cat(errors).mean()))
    output=copy.deepcopy(checkpoint)
    output.update(student_state_dict={k:v.cpu() for k,v in state.items()},optimizer_state_dict=optimizer.state_dict())
    output['offline_preservation']=dict(version=1,optimizer_updates=previous+100,rng=rng.get_state(),dataset_sha256=identities,baseline_sha256=base_hash,lr=1e-5,objective='teacher_mse +4*successful_baseline_mse +improvement_baseline_deviation_mse',normalizer_frozen=True,preservation_observation_corruption=expanded['provenance'].get('observation_corruption','legacy_augmented'))
    # Iteration label is retained; exact new work is recorded separately as optimizer_updates.
    directory=args.output_dir; directory.mkdir(parents=True,exist_ok=True)
    destination=directory/f'model_offline_{previous+100:04d}.pt'
    assert not destination.exists()
    torch.save(output,destination)
    saved=torch.load(destination,map_location='cpu',weights_only=False)
    assert all(torch.equal(v,baseline['teacher_state_dict'][k]) for k,v in saved['teacher_state_dict'].items())
    assert fingerprint(BASE)==base_hash
    report=dict(checkpoint=str(destination),checkpoint_sha256=fingerprint(destination),dataset_sha256=identities,optimizer_updates=previous+100,optimizer_steps_before=before_steps,optimizer_steps_after=steps,preservation_observation_corruption=expanded['provenance'].get('observation_corruption','legacy_augmented'),normalizer_unchanged=True,teacher_unchanged=True,finite_gradients=True,successful_preservation_episodes={p:sum(e['population']==p for e in training) for p in ('nominal','wide')},heldout_action_preservation=holdout,metrics=metrics,interpretation='Offline error is not closed-loop success; screen with default student-only playback.')
    (directory/f'verification_{previous+100:04d}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='metrics'},indent=2))

if __name__=='__main__':main()


