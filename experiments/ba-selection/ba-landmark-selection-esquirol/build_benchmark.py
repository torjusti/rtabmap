from pathlib import Path
import shlex,subprocess
base=Path('/home/torjusti/Projects/rtabmap/build/tools/Export');r=Path(__file__).resolve().parent
flags={a:b.strip() for l in (base/'CMakeFiles/export.dir/flags.make').read_text().splitlines() if ' = ' in l for a,b in [l.split(' = ',1)]}
cmd=shlex.split((base/'CMakeFiles/export.dir/link.txt').read_text());cmd[cmd.index('CMakeFiles/export.dir/main.cpp.o')]=str(r/'benchmark.cpp');cmd[cmd.index('-o')+1]=str(r/'benchmark');cmd[1:1]=shlex.split(flags['CXX_DEFINES']+' '+flags['CXX_INCLUDES']);subprocess.run(cmd,cwd=base,check=True)
