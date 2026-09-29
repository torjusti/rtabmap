from pathlib import Path
import subprocess,os,json,time,sys
r=Path(__file__).resolve().parent
env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
if not (r/'geometry_quality.json').exists():
 with (r/'geometry_quality.log').open('w') as log:
  subprocess.run([sys.executable,'-B',str(r/'evaluate_geometry.py')],stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
for budget in (0,8,16):
 done=r/f'mesh_{budget:02d}.json'
 if done.exists():continue
 (r/'mesh_status.json').write_text(json.dumps(dict(status='running',budget=budget,started=time.time())))
 with (r/f'mesh_{budget:02d}.log').open('w') as log:
  result=subprocess.run([sys.executable,'-B',str(r/'mesh.py'),str(budget)],stdout=log,stderr=subprocess.STDOUT,env=env)
 if result.returncode:
  (r/'mesh_status.json').write_text(json.dumps(dict(status='failed',budget=budget,returncode=result.returncode)))
  raise SystemExit(result.returncode)
 print('DONE MESH',budget,flush=True)
(r/'mesh_status.json').write_text(json.dumps(dict(status='complete')))
