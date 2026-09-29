from pathlib import Path
import time,json,subprocess,sys,shutil
r=Path(__file__).resolve().parent
while True:
    state=json.loads((r/'full_status.json').read_text())
    if state['status']=='failed':raise RuntimeError('Full BA comparison failed; see full_status.json')
    if state['status']=='complete':break
    time.sleep(20)
# Preserve, but exclude, the run whose data-loading phase was paused.
archive=r/'paused_warmup';archive.mkdir(exist_ok=True)
if not (archive/'full_16.json').exists():
    for suffix in ('.json','.log','_camera_poses.txt'):
        source=r/('full_16'+suffix)
        source.rename(archive/source.name)
with (r/'repeat_full_16.log').open('w') as log:
    subprocess.run([sys.executable,'-u',str(r/'run_bench.py'),'full'],stdout=log,stderr=subprocess.STDOUT,check=True)
subprocess.run([sys.executable,str(r/'summarize.py')],check=True)
with (r/'meshes_run.log').open('w') as log:
    subprocess.run([sys.executable,'-u',str(r/'run_meshes.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
(r/'complete.json').write_text(json.dumps(dict(status='complete',time=time.time())))
print('COMPLETE: three timed full BA variants and whole OBJs',flush=True)
