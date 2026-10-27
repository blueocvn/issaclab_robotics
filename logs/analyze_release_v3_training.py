"""Read completed release-v3 TensorBoard scalars without modifying the training run."""
import csv
import argparse
import json
import math
from pathlib import Path
from statistics import mean

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

parser = argparse.ArgumentParser()
parser.add_argument('--run', type=Path, default=Path('logs/rsl_rl/so101_box_state/2026-10-26_14-21-24_so101-box-release-v3'))
parser.add_argument('--prefix', default='box_release_v3_training')
args = parser.parse_args()
run = args.run
prefix = Path('logs') / args.prefix
events = EventAccumulator(str(run), size_guidance={'scalars': 0}).Reload()
tags = events.Tags()['scalars']
summary = {}
series = {}
for tag in tags:
    values = events.Scalars(tag)
    series[tag] = [dict(iteration=v.step, value=v.value, wall_time=v.wall_time) for v in values]
    numbers = [v.value for v in values]
    summary[tag] = dict(count=len(values), first_iteration=values[0].step, last_iteration=values[-1].step,
                        first=numbers[0], last=numbers[-1], min=min(numbers), max=max(numbers),
                        finite=all(math.isfinite(v) for v in numbers),
                        first50_mean=mean(numbers[:50]), last50_mean=mean(numbers[-50:]),
                        windows=[dict(first_iteration=values[start].step,
                                      last_iteration=values[min(start+49,len(values)-1)].step,
                                      mean=mean(numbers[start:start+50]))
                                 for start in range(0, len(values), 50)])
Path(str(prefix)+'_summary.json').write_text(json.dumps(dict(run=str(run), scalars=summary), indent=2)+'\n')
Path(str(prefix)+'_series.json').write_text(json.dumps(series)+'\n')
with Path(str(prefix)+'_scalars.csv').open('w', newline='') as file:
    writer=csv.writer(file)
    writer.writerow(['metric','iteration','value','wall_time'])
    for tag, points in series.items():
        writer.writerows([tag,p['iteration'],p['value'],p['wall_time']] for p in points)
print('TAGS=' + json.dumps(tags))
for tag, data in summary.items():
    if tag.startswith(('Episode_Termination/', 'Metrics/', 'Train/', 'Loss/', 'Policy/')):
        print(json.dumps(dict(metric=tag, **{k:v for k,v in data.items() if k!='windows'})))
print('DURATION_MIN=' + str((max(v['wall_time'] for points in series.values() for v in points)-min(v['wall_time'] for points in series.values() for v in points))/60))
