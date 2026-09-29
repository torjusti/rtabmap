from pathlib import Path
import os,subprocess,json,time,sys
r=Path(__file__).resolve().parent
while (r/'hold_for_native_build').exists():time.sleep(5)
mode=sys.argv[1] if len(sys.argv)>1 else 'pilot'
ids=str(r/'pilot_ids.txt') if mode=='pilot' else 'all'
budgets=[0,8,16,0] if mode=='pilot' else [16,0,8]
results=[]
env=dict(os.environ,OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='1',OMP_DYNAMIC='FALSE',LD_LIBRARY_PATH=str(r/'frozen_runtime')+':'+os.environ.get('LD_LIBRARY_PATH',''))
for j,budget in enumerate(budgets):
 name=f'{mode}_{budget:02d}'+('_repeat' if mode=='pilot' and j==3 else '')
 prefix=r/name
 if prefix.with_suffix('.json').exists():
  results.append(json.loads(prefix.with_suffix('.json').read_text()));continue
 cmd=[str(r/'benchmark'),'/media/torjusti/Files/Esquirol/merged.db',str(r/'input_camera_poses.txt'),ids,str(budget),str(prefix),'50']
 state=dict(mode=mode,current=name,status='running',command=cmd,started=time.time(),finished=results)
 (r/f'{mode}_status.json').write_text(json.dumps(state,indent=2))
 print('START',name,flush=True)
 with prefix.with_suffix('.log').open('w') as log:
  run=subprocess.run(['/usr/bin/time','-v',*cmd],env=env,stdout=log,stderr=subprocess.STDOUT)
 if run.returncode:
  state.update(status='failed',returncode=run.returncode);(r/f'{mode}_status.json').write_text(json.dumps(state,indent=2));raise SystemExit(run.returncode)
 row=json.loads(prefix.with_suffix('.json').read_text());results.append(row)
 print('DONE',name,row,flush=True)
(r/f'{mode}_status.json').write_text(json.dumps(dict(mode=mode,status='complete',finished=results),indent=2))
