"""Summarize completed trials without treating consistency as known accuracy."""
from pathlib import Path
import json
import re
import sys
import numpy as np

ROOT=Path(__file__).resolve().parent
mode=sys.argv[1] if len(sys.argv)>1 else 'full'
rows={}
for value in (0,1):
    prefix=ROOT/f'{mode}_{value:02d}'
    row=json.loads(prefix.with_suffix('.json').read_text())
    log=prefix.with_suffix('.log').read_text()
    rss=re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)',log)
    if rss: row['maximum_rss_gib']=int(rss[1])/1024**2
    phase=re.search(r'BA landmark selection complete selected=(\d+) omitted_processed=(\d+) selection_s=([\d.]+) joint_s=([\d.]+) point_only_s=([\d.]+)',log)
    if phase:
        row['phases']=dict(zip(('selected','omitted_processed','selection_seconds','joint_seconds','point_only_seconds'),
            [int(phase[1]),int(phase[2]),float(phase[3]),float(phase[4]),float(phase[5])]))
        row['selected_fraction']=int(phase[1])/row['tracks']
    elif value:
        raise RuntimeError('Selection never completed; do not present this as a selector trial')
    row['pipeline_seconds']=row['correspondence_seconds']+row['solve_seconds']
    rows[str(value)]=row
if rows['0']['input_hash']!=rows['1']['input_hash']:
    raise RuntimeError('Unmatched landmark inputs')
base=np.loadtxt(ROOT/f'{mode}_00_camera_poses.txt')
test=np.loadtxt(ROOT/f'{mode}_01_camera_poses.txt')
if not np.array_equal(base[:,-1],test[:,-1]): raise RuntimeError('Frame mismatch')
distance=np.linalg.norm(base[:,1:4]-test[:,1:4],axis=1)
report=dict(mode=mode,rows=rows,
    solver_speedup=rows['0']['solve_seconds']/rows['1']['solve_seconds'],
    pipeline_speedup=rows['0']['pipeline_seconds']/rows['1']['pipeline_seconds'],
    pose_difference_m=dict(median=float(np.median(distance)),p95=float(np.quantile(distance,.95)),maximum=float(distance.max())),
    caveat='Pose differences and track retention are not ground-truth accuracy measurements.')
repeat=ROOT/f'{mode}_00_repeat_camera_poses.txt'
if repeat.exists():
    report['baseline_repeat_poses_identical']=repeat.read_bytes()==(ROOT/f'{mode}_00_camera_poses.txt').read_bytes()
    report['baseline_repeat']=json.loads((ROOT/f'{mode}_00_repeat.json').read_text())
if mode=='full' and (ROOT/'geometry_quality.json').exists():
    quality=json.loads((ROOT/'geometry_quality.json').read_text())
    report['geometry_summary']=quality['summary']
    report['geometry_limitations']=quality['limitations']
(ROOT/f'{mode}_results.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
