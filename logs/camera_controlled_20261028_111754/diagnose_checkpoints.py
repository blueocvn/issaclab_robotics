"""Read-only checkpoint diagnostics; replay samples are not held-out accuracy."""
import json
import re
from pathlib import Path

import torch
from rsl_rl.models import CNNModel
from tensordict import TensorDict

torch.set_num_threads(4)
torch.manual_seed(0)
folder = Path(__file__).resolve().parent
root = Path('logs/rsl_rl')
paths = {
    'baseline': root / 'so101_box_camera_distillation/2026-10-27_16-26-29_so101-box-workspace-camera-1798-lr1e-4-seed43/model_2200.pt',
    'control200': root / 'so101_box_camera_distillation/20261028_111754_control_b1/model_2399.pt',
    'control400': root / 'so101_box_camera_distillation/20261028_111754_control_b2/model_2598.pt',
    'data200': root / 'so101_box_camera_data_first/20261028_111754_data_b1/model_2399.pt',
    'data400': root / 'so101_box_camera_data_first/20261028_111754_data_b2/model_2598.pt',
}
checkpoints = {k: torch.load(v, map_location='cpu', weights_only=False) for k, v in paths.items()}
base = checkpoints['baseline']['student_state_dict']
report = {'checkpoints': {}, 'replay_diagnostics': {}, 'training_loss': {}}
datasets = {}
for source in ('data200', 'data400'):
    replay = checkpoints[source]['box_data_first']['replay']
    for phase, bucket in enumerate(replay['buckets']):
        ids = torch.randperm(replay['sizes'][phase])[:256]
        datasets[f'{source}_phase{phase}'] = {k: v[ids].float() for k, v in bucket.items()}
    report['checkpoints'][source] = {'replay_sizes': replay['sizes'], 'replay_cursors': replay['cursors']}

example = next(iter(datasets.values()))
obs = TensorDict({k: example[k] for k in ('wrist_rgb', 'proprioception')}, [len(example['labels'])])
model = CNNModel(obs, {'student': ['wrist_rgb', 'proprioception']}, 'student', 6,
    hidden_dims=[256, 128], activation='elu', obs_normalization=True,
    distribution_cfg={'class_name':'GaussianDistribution','init_std':0.05,'std_type':'log'},
    cnn_cfg={'output_channels':[16,32,32],'kernel_size':[5,3,3],'stride':[2,2,2],'activation':'elu'})
model.eval()
with torch.inference_mode():
    for name, checkpoint in checkpoints.items():
        state = checkpoint['student_state_dict']
        stats = report['checkpoints'].setdefault(name, {})
        stats.update({
            'normalizer_count': int(state['obs_normalizer.count']),
            'normalizer_mean_shift_max_base_std': float(((state['obs_normalizer._mean']-base['obs_normalizer._mean']).abs()/(base['obs_normalizer._std']+0.01)).max()),
            'normalizer_std_ratio_range': [float((state['obs_normalizer._std']/base['obs_normalizer._std']).min()), float((state['obs_normalizer._std']/base['obs_normalizer._std']).max())],
            'parameter_relative_l2_change': float(torch.sqrt(sum((v-base[k]).square().sum() for k,v in state.items() if not k.startswith('obs_normalizer'))/sum(v.square().sum() for k,v in base.items() if not k.startswith('obs_normalizer')))),
        })
        for mode in ('own_normalizer', 'baseline_normalizer'):
            model.load_state_dict(state)
            if mode == 'baseline_normalizer':
                model.obs_normalizer.load_state_dict({k.removeprefix('obs_normalizer.'):v for k,v in base.items() if k.startswith('obs_normalizer.')})
            results = {}
            for source, batch in datasets.items():
                n = len(batch['labels'])
                observation = TensorDict({k:batch[k] for k in ('wrist_rgb','proprioception')}, [n])
                prediction = model(observation)
                error = (prediction-batch['labels']).square()
                shuffled = observation.clone()
                shuffled['wrist_rgb'] = shuffled['wrist_rgb'][torch.arange(n).roll(1)]
                shuffled_error = (model(shuffled)-batch['labels']).square().mean()
                results[source] = {'mse':float(error.mean()), 'joint_mse':error.mean(0).tolist(), 'rgb_shuffled_mse':float(shuffled_error), 'raw_action_abs_gt1_fraction':float((prediction.abs()>1).float().mean()),'label_abs_mean':batch['labels'].abs().mean(0).tolist()}
            report['replay_diagnostics'][name+'_'+mode] = results
for arm in ('control', 'data'):
    for block in (1,2):
        text=(folder/f'{arm}_b{block}.log').read_text()
        for metric in ('behavior', 'fresh_behavior', 'replay_behavior', 'student_success_rate', 'student_episodes'):
            values=[float(v) for v in re.findall(r'Mean '+metric+r' loss:\s+([-+0-9.eE]+)',text)]
            if values:
                report['training_loss'][f'{arm}_b{block}_{metric}']={'first20':sum(values[:20])/len(values[:20]),'last20':sum(values[-20:])/len(values[-20:]),'min':min(values),'max':max(values)}
(folder/'checkpoint_diagnostics.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'checkpoints':report['checkpoints'],'training_loss':report['training_loss']},indent=2))
for name, results in report['replay_diagnostics'].items():
    print(name, {k: round(v['mse'],5) for k,v in results.items()})
