"""Verify the bounded screen and measure fixed-reference fit, not holdout accuracy."""
import hashlib
import json
import re
from pathlib import Path

import torch
from rsl_rl.models import CNNModel
from tensordict import TensorDict

torch.set_num_threads(4)
p = Path(__file__).resolve().parent
base_path = Path('logs/rsl_rl/so101_box_camera_distillation/2026-10-27_16-26-29_so101-box-workspace-camera-1798-lr1e-4-seed43/model_2200.pt')
end_path = Path('logs/rsl_rl/so101_box_camera_anchored/20261028_anchored_50_b1/model_2249.pt')
base = torch.load(base_path, map_location='cpu', weights_only=False)
end = torch.load(end_path, map_location='cpu', weights_only=False)
bank_path = p.parent/'reference.pt'
original = torch.load(bank_path, map_location='cpu', weights_only=False)
assert hashlib.sha256(base_path.read_bytes()).hexdigest() == '3f5ad0ea7dabe634c40b1c0c82d4eee8a88b3ff8624aa88ce1801686a0c817b6'
assert hashlib.sha256(bank_path.read_bytes()).hexdigest() == 'e79f69ff7754b42fd8dadd4f24e7b672dac45f43bcfaa1dcd7db059ecb983fcf'
assert end['box_data_first']['num_updates'] == 50
steps = sorted(set(int(v['step']) for v in end['optimizer_state_dict']['state'].values() if 'step' in v))
assert steps == [83584]
assert int(end['student_state_dict']['obs_normalizer.count']) == 9228288
assert all(torch.equal(v, base['teacher_state_dict'][k]) for k,v in end['teacher_state_dict'].items())
assert all(torch.isfinite(v).all() for v in end['student_state_dict'].values())
for population in ('nominal','wide'):
    before = original['populations'][population]
    after = end['box_reference']['populations'][population]
    assert before['sizes'] == after['sizes']
    assert not torch.equal(before['rng'],after['rng'])
    for a,b in zip(before['buckets'],after['buckets'],strict=True):
        for key in a: assert torch.equal(a[key],b[key])
text = (p/'train.log').read_text()
assert [int(v) for v in re.findall(r'Learning iteration (\d+)/',text)] == list(range(2200,2250))
metrics = {name:[int(float(v)) for v in re.findall(r'Mean '+name+r' loss:\s+([-+0-9.eE]+)',text)] for name in ('reference_samples','rolling_samples','fresh_samples')}
assert metrics['reference_samples'] == [4096]*50
assert metrics['rolling_samples'] == [0]+[4096]*49
assert metrics['fresh_samples'] == [12288]+[8192]*49

example = original['populations']['nominal']['buckets'][0]
obs = TensorDict({k:example[k][:1].float() for k in ('wrist_rgb','proprioception')},[1])
model = CNNModel(obs,{'student':['wrist_rgb','proprioception']},'student',6,
    hidden_dims=[256,128],activation='elu',obs_normalization=True,
    distribution_cfg={'class_name':'GaussianDistribution','init_std':0.05,'std_type':'log'},
    cnn_cfg={'output_channels':[16,32,32],'kernel_size':[5,3,3],'stride':[2,2,2],'activation':'elu'})
model.eval()
fit = {}
with torch.inference_mode():
    for name,checkpoint in [('baseline',base),('anchored50',end)]:
        model.load_state_dict(checkpoint['student_state_dict'])
        results = {}
        for population in ('nominal','wide'):
            for phase,bucket in enumerate(original['populations'][population]['buckets']):
                errors = []
                for start in range(0,len(bucket['labels']),128):
                    observation = TensorDict({k:bucket[k][start:start+128].float() for k in ('wrist_rgb','proprioception')},[len(bucket['labels'][start:start+128])])
                    errors.append((model(observation)-bucket['labels'][start:start+128]).square())
                errors = torch.cat(errors)
                results[f'{population}_phase{phase}'] = {'count':len(errors),'mse':float(errors.mean()),'joint_mse':errors.mean(0).tolist()}
        fit[name] = results
report = {'relative_updates':50,'optimizer_steps':83584,'normalizer_count':9228288,
    'reference_tensors_unchanged':True,'reference_rngs_advanced':True,'teacher_unchanged':True,'student_finite':True,
    'checkpoint_sha256':hashlib.sha256(end_path.read_bytes()).hexdigest(),
    'sample_totals':{k:sum(v) for k,v in metrics.items()},'fixed_reference_fit':fit,
    'interpretation':'Reference fit is training-data error, not held-out success or an adoption criterion.'}
(p/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='fixed_reference_fit'},indent=2))
for name,result in fit.items():
    print(name,{k:round(v['mse'],6) for k,v in result.items()})
