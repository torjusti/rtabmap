from pathlib import Path
import json,re,hashlib
import numpy as np
r=Path(__file__).resolve().parent
ansi=re.compile(r'\x1b\[[0-9;]*m')
summary={'source_db':'/media/torjusti/Files/Esquirol/merged.db',
 'input_poses_sha256':hashlib.sha256((r/'input_camera_poses.txt').read_bytes()).hexdigest(),
 'scope':'Incremental BA from the same frozen existing camera trajectory; graph optimization and meshing excluded from BA timings.',
 'settings':{'iterations':50,'threads':8,'BAProgressive':False,'BADepth':0,'window_graph_depth':2,'match_cache':False},'runs':[]}
for name in ('pilot_00','pilot_08','pilot_16','pilot_00_repeat','full_00','full_08','full_16'):
 path=r/f'{name}.json'
 if not path.exists():continue
 row=json.loads(path.read_text());row['name']=name;row['ba_seconds']=row['correspondence_seconds']+row['solve_seconds']
 row['mean_observations_per_track_before_rejection']=row['observations']/row['tracks']
 row['observations_after_rejection']=row['observations']-row['rejected_observations']
 row['rejected_observation_fraction']=row['rejected_observations']/row['observations']
 row['timing_valid']=not (name=='full_16' and (r/'resource_pause.json').exists() and not (r/'paused_warmup/full_16.json').exists())
 log=ansi.sub('',(r/f'{name}.log').read_text(errors='replace'))
 for pattern,keys in [
  (r'BA coverage: budget=\d+ proposed_pairs=(\d+) matched_pairs=(\d+) fallback_pairs=(\d+) three_view_tracks=\d+ frames_with_three_view_tracks=(\d+) frames_with_20_tracks=(\d+) total_frames=\d+', ['proposed_pairs','matched_pairs','fallback_pairs','frames_with_three_view_tracks','frames_with_20_tracks']),
  (r'Maximum resident set size \(kbytes\): (\d+)', ['peak_rss_kib']),
  (r'BA tracks: \d+ tracks / \d+ observations.*?rejected: \d+ too short, (\d+) frame conflict', ['conflicting_tracks']),
 ]:
  m=re.search(pattern,log)
  if m:row.update(zip(keys,map(int,m.groups())))
 for key,pattern in {
  'matching_loop_seconds': r'BA pairs: \d+ matched candidates[^\n]*? frames, ([0-9.]+)s',
  'track_build_seconds': r'BA tracks: \d+ tracks / \d+ observations[^\n]*? pairs in ([0-9.]+)s',
  'pair_selection_seconds': r'BA pair budget: [^\n]*?selection_seconds=([0-9.]+)',
  'fallback_seconds': r'BA pair budget fallback: [^\n]*?coverage_seconds=([0-9.]+)',
 }.items():
  m=re.search(pattern,log)
  if m:row[key]=float(m.group(1))
 row['shared_track_counts_before_ba']={}
 for category in ('odom','loop'):
  m=re.search(r'BA lever \['+category+r'\]: links=(\d+) \((\d+) with shared tracks\) \| shared tracks/link mean=([0-9.]+) median=([0-9.]+) \| track length mean=([0-9.]+) median=([0-9.]+)',log)
  if m:
   row['shared_track_counts_before_ba'][category]=dict(links=int(m[1]),links_with_tracks=int(m[2]),
    shared_tracks_mean=float(m[3]),shared_tracks_median=float(m[4]),track_length_mean=float(m[5]),track_length_median=float(m[6]))
 iterations=re.findall(r'PROGRESS Bundle adjustment iteration (\d+)/(\d+)',log)
 if iterations:
  row['solver_last_reported_iteration']=int(iterations[-1][0])
  row['solver_configured_iteration_cap']=int(iterations[-1][1])
 baseline=r/f'{name.split("_")[0]}_00_camera_poses.txt'
 current=r/f'{name}_camera_poses.txt'
 if baseline.exists() and current.exists():
  a=np.loadtxt(baseline);b=np.loadtxt(current)
  assert np.array_equal(a[:,-1],b[:,-1])
  d=np.linalg.norm(a[:,1:4]-b[:,1:4],axis=1)
  qa=a[:,4:8];qb=b[:,4:8];qa=qa/np.linalg.norm(qa,axis=1)[:,None];qb=qb/np.linalg.norm(qb,axis=1)[:,None]
  qb=qb*np.where(np.sum(qa*qb,axis=1)<0,-1.0,1.0)[:,None]
  angles=4*np.degrees(np.arctan2(np.linalg.norm(qa-qb,axis=1),np.linalg.norm(qa+qb,axis=1)))
  row['translation_difference_vs_unlimited_m']={'median':float(np.median(d)),'p95':float(np.quantile(d,.95)),'max':float(d.max())}
  row['rotation_difference_vs_unlimited_deg']={'median':float(np.median(angles)),'p95':float(np.quantile(angles,.95)),'max':float(angles.max())}
 summary['runs'].append(row)
(r/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
for row in summary['runs']:
 print(row['name'],round(row['ba_seconds'],2),'s',row.get('matched_pairs'),'pairs',row['three_view_tracks'],'three-view tracks',row.get('frames_with_20_tracks'),'well-supported frames',row.get('translation_difference_vs_unlimited_m'))
