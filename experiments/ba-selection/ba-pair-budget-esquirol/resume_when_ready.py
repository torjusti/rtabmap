"""Resume only this experiment after enough RAM is available."""
from pathlib import Path
import os,signal,time,json
root=Path(__file__).resolve().parent
pid=2627183
while True:
    command=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')[0].decode()
    if command != str(root/'benchmark'):raise RuntimeError('Benchmark PID no longer matches')
    mem={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if len(line.split())>=2}
    available=mem['MemAvailable']/1024**2
    state=dict(pid=pid,available_gib=available,status='waiting',time=time.time())
    if available >= 55:
        os.kill(pid,signal.SIGCONT)
        state['status']='resumed'
        (root/'resource_status.json').write_text(json.dumps(state,indent=2))
        print('Resumed BA:',state,flush=True)
        break
    (root/'resource_status.json').write_text(json.dumps(state,indent=2))
    time.sleep(20)
