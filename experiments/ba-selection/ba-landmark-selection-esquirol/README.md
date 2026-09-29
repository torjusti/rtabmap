# COLMAP-style landmark selection experiment

Experimental flag, **off by default**: `--ba-landmark-selection` in the
weighted-Poisson / rtabmap-cloud CLI and “BA landmark selection [experimental]”
in their TUI forms. Requires bundle adjustment to be enabled separately.

Native settings:

* `Optimizer/BALandmarkSelection=false`
* `Optimizer/BALandmarkMinCoverageGain=0.05`

Each physical camera has an 8×8 image grid. A track's marginal gain is the sum
over its observations of `1/sqrt(c+1) - 1/sqrt(c+2)`, where `c` counts already
selected landmarks in that image cell. A lazy max-heap repeatedly selects the
largest-gain track and updates coverage. Equal gains prefer the larger track ID.
Selection stops when the largest marginal gain is **at most** the cutoff.
The cutoff is neither a percentage nor a fixed landmark count.

The joint solve contains selected landmarks, all cameras, and existing metric
constraints. Omitted finite points with valid observations are then refined with
the resulting cameras fixed. This second stage uses the same projection edges,
variance, robust loss, rejection gate and active-support rule. Sparse points are
retained; selecting fewer sparse landmarks does not directly remove depth pixels
from the dense mesh. Cancellation/failure publishes neither partial points nor
partial outlier masks.

RTAB-map adaptations:

* Select after existing correspondence/triangulation gates and initial support.
* Keep existing metric pose-graph/gravity/rig constraints in the joint solve.
* Process omitted points in batches of 16,384 using dynamic blocks and
  non-marginalized points. All camera parameters remain fixed in this stage.
  This bounds graph allocation. The optional progressive P80 gate is computed
  per batch; ordinary rejection uses its unchanged absolute gate. LM damping
  and stopping are also shared within a batch, so this is not a numerically
  identical reproduction of COLMAP's Ceres solve.
* Selection needs at least ten poses and known image dimensions. Missing
  dimensions or an empty selection falls back to the full solver.
* No new parallax, loop-bridge or weak-camera coverage protection is claimed.
  The full COLMAP track-completion/retriangulation pipeline is not implemented.
* Joint BA retains optional per-iteration diagnostics; the fixed-camera stage
  reports batch progress and total wall time without per-iteration statistics.

Algorithm sources inspected 2026-09-29:
[coverage selector](https://github.com/colmap/colmap/blob/main/src/colmap/scene/reconstruction_pruning.cc),
[global BA orchestration](https://github.com/colmap/colmap/blob/main/src/colmap/sfm/incremental_mapper.cc).

## Evaluation

Read-only source: `/media/torjusti/Files/Esquirol/merged.db`.
The input trajectory is copied from the preceding pair-budget experiment; this
is an **additional BA pass on an existing exported trajectory**, not raw odometry.
Both arms use unlimited pair matching, ordinary rejection, minimum support 2,
mono residuals, 50 final iterations plus warmup, and the same metric factors.
Per-iteration diagnostic statistics are off in both timed arms.

First run the 800-frame pilot, including a repeated full-landmark baseline.
Then compare all 43,971 frames in separate, serial processes. Input point and
observation hashes must match. Compare total solver time (including coverage
selection and fixed-camera refinement), phase timings, RSS, and common LiDAR
point-to-plane residuals. The latter measure local consistency, **not known
absolute pose accuracy**.

Whole weighted-Poisson OBJs are written under `objs/`. Each mesh uses its own
frozen trajectory and isolated artifact index; source exports are only linked
read-only. Mesh settings and high-precision OBJ verification match the preceding
experiment. Logs/status JSON files report actual completion, not a predicted
result. Default behavior should not change based solely on speed or track counts.

## Initial validation and pilot

Build/install and installed-parameter discovery passed. Focused native tests:
23 passed, 20 skipped (unavailable backends or g2o-specific behavior). Python
CLI/TUI/provenance and meshing regressions: 132 passed.

The 800-frame pilot selected 46,789 of 72,625 tracks (64.4%). Total solver time
was 10.07 s with selection, versus 9.54 and 9.60 s for repeated baselines. The
selected joint solve took 9.14 s; selection took 0.067 s and omitted-point
processing 0.845 s. The three landmark-input hashes matched and repeated
baseline pose files were byte-identical. Selection changed camera centers by
4.0 mm median / 12.2 mm p95 relative to baseline; these differences do not say
which trajectory is more accurate. Compilation was still running during this
pilot, so use the subsequent full experiment for stronger timing evidence.

## Full-scan results

Complete: both 43,971-frame trials, whole meshes and the common depth evaluation.
The landmark input hashes matched. Selection kept 3,325,807 of 5,067,653 tracks
(65.6%). Total BA time increased from 613.46 s to 698.67 s (**13.9% slower**).
The selected joint solve took 580.00 s, selection 10.09 s and omitted-point
processing 107.36 s. Peak process RSS decreased from 27.90 to 26.80 GiB.

Matching took 631.85 s in the first run and 452.94 s in the second. The selector
does not run during matching and the matching inputs/settings were identical:
do not attribute that difference (or the lower combined pipeline time) to
landmark selection. These are serial single-run timings, subject to cache and
system-load effects.

The common geometry evaluation used 214 pairs and 307,170 bidirectional samples.
Nearby-pair mean absolute point-to-plane residual was 33.9713 mm for baseline
and 33.9813 mm with selection. Revisit residual was 36.31936 versus 36.31953 mm.
These differences show **no meaningful improvement in this consistency test**;
the test does not measure absolute pose accuracy. Camera-center differences
between the two outputs were 0.70 mm median, 2.01 mm p95 and 7.82 mm maximum.

Both weighted-Poisson meshes used identical settings (depth 12, depth subsampling
4, 50 normal neighbors, 2 cm voxels, medium confidence, incidence weighting on,
50% decimation) and frozen per-variant camera trajectories:

* `objs/00_all_landmarks_full.obj`: 760,697 vertices, 1,520,778 triangles.
* `objs/01_colmap_selection_full.obj`: 760,760 vertices, 1,520,898 triangles.

The exporter verified exact OBJ text geometry against each source mesh, then
verified oriented triangles after the reader's expected float32 conversion.
The experiment leaves landmark selection **off by default**. See
`full_results.json`, `geometry_quality.json` and per-mesh metadata for details.
