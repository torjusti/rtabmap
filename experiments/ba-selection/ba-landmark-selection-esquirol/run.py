"""Run isolated baseline/selection trials, then whole meshes and common scoring.

Source DB and source exported poses are read-only. Each variant runs in a fresh
process, serially. Timings include omitted-point refinement, not just joint BA.
"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
MODE=sys.argv[1] if len(sys.argv)>1 else 'pilot'
if MODE not in ('pilot','full'):
    raise SystemExit('Use pilot or full')
ENV=dict(os.environ,OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='1',OMP_DYNAMIC='FALSE',
         LD_LIBRARY_PATH=str(ROOT/'frozen_runtime')+':'+os.environ.get('LD_LIBRARY_PATH',''))

def write_state(**state):
    temp=ROOT/f'{MODE}_status.tmp'
    temp.write_text(json.dumps(dict(mode=MODE,updated=time.time(),**state),indent=2))
    temp.replace(ROOT/f'{MODE}_status.json')

def run(name,cmd):
    write_state(status='running',current=name,command=cmd)
    print('START',name,flush=True)
    with (ROOT/f'{name}.log').open('w') as log:
        subprocess.run(cmd,env=ENV,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('DONE',name,flush=True)

def available_memory():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'): return int(line.split()[1])*1024
    return 0

try:
    cases=[(0,''),(1,''),(0,'_repeat')] if MODE=='pilot' else [(0,''),(1,'')]
    results=[]
    for value,suffix in cases:
        name=f'{MODE}_{value:02d}{suffix}'
        if not (ROOT/f'{name}.json').exists():
            db=Path('/media/torjusti/Files/Esquirol/merged.db')
            signature=json.loads((ROOT/'input_poses_provenance.json').read_text())['source'][db.name]
            stat=db.stat()
            if [stat.st_size,stat.st_mtime_ns]!=signature:
                raise RuntimeError('Source DB changed since the input trajectory was frozen')
            # The previous full baseline needed ~28 GiB RSS. Leave room for
            # the desktop and do not start a run into a known memory shortage.
            while MODE=='full' and available_memory()<32*1024**3:
                write_state(status='waiting_for_memory',current=name,
                            required_available_gib=32,available_gib=available_memory()/1024**3)
                time.sleep(20)
            run(name,['/usr/bin/time','-v',str(ROOT/'benchmark'),
                '/media/torjusti/Files/Esquirol/merged.db',str(ROOT/'input_camera_poses.txt'),
                str(ROOT/'pilot_ids.txt') if MODE=='pilot' else 'all', str(value),str(ROOT/name),'50'])
        results.append(json.loads((ROOT/f'{name}.json').read_text()))
    if len({row['input_hash'] for row in results})!=1:
        raise RuntimeError('Correspondence inputs differed; comparison is invalid')
    if MODE=='full':
        for value in (0,1):
            if not (ROOT/f'mesh_{value:02d}.json').exists():
                run(f'mesh_{value:02d}',['/usr/bin/python3',str(ROOT/'mesh.py'),str(value)])
        if not (ROOT/'geometry_quality.json').exists():
            run('geometry_quality',['/usr/bin/python3',str(ROOT/'evaluate_geometry.py')])
    write_state(status='complete',results=results)
except BaseException as exc:
    write_state(status='failed',error=str(exc))
    raise
