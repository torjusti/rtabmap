#!/usr/bin/env bash
# Full Koda reprocess with the user's real/default parameters, to reproduce
# the occasional >1s per-node stalls and confirm the culprit (Pre_update /
# FLANN vocabulary rebuild vs LTM retrieval vs optimization).
set -euo pipefail
ROOT=/home/torjusti/Projects/rtabmap
BIN=$ROOT/build/bin
INPUT="/media/torjusti/Files/Chaos - Koda/merged.db"
OUTDIR="/media/torjusti/Files/rtabmap-bench/profile-stages"
STOP=${1:-19700}
LABEL=${2:-koda_userparams}
shift $(( $# > 2 ? 2 : $# ))
EXTRA=("$@")   # extra --Param value pairs
export LD_LIBRARY_PATH=$BIN${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
mkdir -p "$OUTDIR"
DB="$OUTDIR/profile_${LABEL}_${STOP}.db"
LOG="$OUTDIR/profile_${LABEL}_${STOP}.log"
rm -f "$DB"

# User's default parameters (as shared).
PARAMS=(
  -stop "$STOP"
  --RGBD/ProximityBySpace true
  --Mem/STMSize 50
  --Rtabmap/MemoryThr 8000
  --Kp/MaxFeatures 1000
  --Vis/MaxFeatures 1000
  --RGBD/OptimizeRobust false
  --RGBD/OptimizeMaxError 0
  --Vis/PnPVarianceMedianRatio 2
  --Vis/DepthAsMask false
  --Mem/DepthAsMask false
  --Reg/RepeatOnce false
  --Vis/CorGuessWinSize 0
  --Kp/DetectorStrategy 11
  --Vis/FeatureType 11
  --Mem/UseOdomFeatures false
  --Vis/CorNNType 6
  --Rtabmap/LoopThr 0.11
  --Vis/MinInliers 25
  --Vis/PnPReprojError 2.0
  --RGBD/LinearUpdate 0
  --RGBD/AngularUpdate 0
  --Kp/DepthConfidenceThr 40
  --RGBD/MarkerDetection false
  --Rtabmap/TimeThr 0
  --Rtabmap/MaxRetrieved 10
  --RGBD/MaxLocalRetrieved 5
  --Optimizer/LoopRedundancyRadius 5
  --SuperPoint/ModelPath "$ROOT/superpoint.pt"
)

echo "======== $(date -Is) KODA USERPARAMS reprocess stop=$STOP ========"
/usr/bin/time -v "$BIN/rtabmap-reprocess" "${PARAMS[@]}" ${EXTRA[@]+"${EXTRA[@]}"} "$INPUT" "$DB" > "$LOG" 2>&1
echo "--- wall / closures ---"
grep -aE "Elapsed \(wall clock\)|Total loop closures|Percent of CPU" "$LOG" || true
echo "======== $(date -Is) DONE ========"
