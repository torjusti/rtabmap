#!/usr/bin/env bash
# Quick stage profiling of reprocess on the perf branch.
set -euo pipefail
ROOT=/home/torjusti/Projects/rtabmap
BIN=$ROOT/build/bin
INPUT="${INPUT_DB:-/media/torjusti/Files/July 2026/Gravelle/merged.db}"
OUTDIR="/media/torjusti/Files/rtabmap-bench/profile-stages"
STOP=${1:-2000}
LABEL=${2:-stages}
shift $(( $# > 2 ? 2 : $# ))
EXTRA=("$@")   # extra --Param value pairs
export LD_LIBRARY_PATH=$BIN${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

PARAMS=(
  -stop "$STOP"
  --Kp/DetectorStrategy 11
  --Kp/MaxFeatures 500
  --Mem/STMSize 20
  --Mem/UseOdomFeatures false
  --RGBD/MarkerDetection false
  --RGBD/OptimizeMaxError 6
  --RGBD/ProximityBySpace true
  --Rtabmap/MemoryThr 2000
  --SuperPoint/ModelPath "$ROOT/superpoint.pt"
  --Vis/FeatureType 11
  --Vis/PnPVarianceMedianRatio 2
  --Vis/DepthAsMask false
)

mkdir -p "$OUTDIR"
DB="$OUTDIR/profile_${LABEL}_${STOP}.db"
LOG="$OUTDIR/profile_${LABEL}_${STOP}.log"
rm -f "$DB"
echo "======== $(date -Is) PROFILE reprocess stop=$STOP label=$LABEL extra='${EXTRA[*]}' ========"
/usr/bin/time -v "$BIN/rtabmap-reprocess" "${PARAMS[@]}" "${EXTRA[@]}" \
  "$INPUT" "$DB" > "$LOG" 2>&1
grep -E "Elapsed \(wall clock\)|Total loop|Processed ${STOP}" "$LOG" || true

python3 - "$DB" "$STOP" <<'PY'
import sqlite3, sys, zlib, statistics
db, stop = sys.argv[1], int(sys.argv[2])

def uncompress(blob):
    if not blob:
        return ""
    try:
        raw = zlib.decompress(blob[:-12])
    except zlib.error:
        raw = zlib.decompress(blob)
    return raw.split(b"\x00", 1)[0].decode("utf-8", "replace")

con = sqlite3.connect(db)
acc = {}
rows = 0
for (blob,) in con.execute("SELECT data FROM Statistics WHERE id <= ?", (stop,)):
    rows += 1
    for part in uncompress(blob).split(";"):
        if ":" not in part:
            continue
        k, v = part.rsplit(":", 1)
        if k.startswith("Timing/") or k.startswith("TimingMem/"):
            try:
                acc.setdefault(k, []).append(float(v))
            except ValueError:
                pass
con.close()

total_key = "Timing/Total/ms"
grand = sum(acc.get(total_key, [])) / 1000.0
print(f"\nStatistics rows: {rows}   Timing/Total sum: {grand:.1f}s\n")
print(f"{'stage':45s} {'total_s':>9s} {'%oftot':>7s} {'mean_ms':>9s} {'p95_ms':>9s}")
items = []
for k, vals in acc.items():
    items.append((sum(vals)/1000.0, k, statistics.mean(vals),
                  statistics.quantiles(vals, n=20)[18] if len(vals) >= 20 else max(vals)))
for tot, k, mean, p95 in sorted(items, reverse=True):
    pct = 100.0 * tot / grand if grand else 0.0
    print(f"{k:45s} {tot:9.1f} {pct:6.1f}% {mean:9.2f} {p95:9.1f}")
PY

echo "======== $(date -Is) PROFILE_DONE ========"
