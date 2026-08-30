#!/usr/bin/env bash
# Does a larger optimization window (MemoryThr=8000) change the iSAM2 verdict?
# Koda, run past the window (-stop 12000), iSAM2 off vs on.
set -euo pipefail
ROOT=/home/torjusti/Projects/rtabmap
export INPUT_DB="/media/torjusti/Files/Chaos - Koda/merged.db"
OUTDIR="/media/torjusti/Files/rtabmap-bench/profile-stages"
STOP=12000
mkdir -p "$OUTDIR"

run() {
  local label="$1"; shift
  echo ">>> $(date -Is) starting $label ($*)"
  "$ROOT/bench/profile_stages.sh" "$STOP" "$label" --Rtabmap/MemoryThr 8000 "$@" \
    | grep -E "Timing/Total/ms|Map_optimization|Proximity_by_space_visual|Add_loop_closure_link|Memory_update" || true
  echo "--- $label wall / closures ---"
  grep -E "Total loop closures|Elapsed \(wall|Percent of CPU" "$OUTDIR/profile_${label}_${STOP}.log" || true
  grep -E "iSAM2 root stats|RESET iSAM2" "$OUTDIR/profile_${label}_${STOP}.log" | tail -3 || true
  echo
}

echo "======== ISAM_MEM8K_AB START $(date -Is) ========"
run mem8koff --GTSAM/Incremental false
run mem8kon  --GTSAM/Incremental true
echo "======== ISAM_MEM8K_AB DONE $(date -Is) ========"
