#!/usr/bin/env bash
# Full Koda A/B: iSAM2 off vs on, with GPU utilization sampling and
# iSAM2 root reanchor/reset instrumentation.
set -euo pipefail
ROOT=/home/torjusti/Projects/rtabmap
export INPUT_DB="/media/torjusti/Files/Chaos - Koda/merged.db"
OUTDIR="/media/torjusti/Files/rtabmap-bench/profile-stages"
STOP=19700   # > node count (19606) => full run
mkdir -p "$OUTDIR"

run() {
  local label="$1"; shift
  local gpucsv="$OUTDIR/gpu_${label}.csv"; rm -f "$gpucsv"
  ( while true; do nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader >> "$gpucsv"; sleep 2; done ) &
  local smipid=$!
  echo ">>> $(date -Is) starting $label ($*)"
  "$ROOT/bench/profile_stages.sh" "$STOP" "$label" "$@" \
    | grep -E "Timing/Total/ms|Map_optimization|Proximity_by_space_visual|Add_loop_closure_link|Memory_update|Add_new_words|Likelihood" || true
  kill $smipid 2>/dev/null || true
  echo "--- $label wall / closures / iSAM2 root stats ---"
  grep -E "Total loop closures|Elapsed \(wall|Percent of CPU" "$OUTDIR/profile_${label}_${STOP}.log" || true
  grep -E "iSAM2 root stats|RESET iSAM2" "$OUTDIR/profile_${label}_${STOP}.log" | tail -5 || true
  awk -F, '{u=$1;gsub(/[^0-9]/,"",u); s+=u;n++; if(u+0>mx)mx=u} END{if(n)printf "GPU: mean_util=%.1f%% max=%d%% samples=%d\n", s/n, mx, n}' "$gpucsv" || true
  echo
}

echo "======== ISAM_KODA_AB START $(date -Is) ========"
run kodaoff --GTSAM/Incremental false
run kodaon  --GTSAM/Incremental true
echo "======== ISAM_KODA_AB DONE $(date -Is) ========"
