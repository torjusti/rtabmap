# Esquirol BA pair-budget experiment

This experiment compares unlimited extra pair matching with initial budgets of
8 and 16 extra partners per frame. All existing graph-link matching pairs and
all metric pose constraints remain. A budget applies to both ends of an extra
pair; it is not a cap on the frame's total graph degree.

## Reproduction

- Database: `/media/torjusti/Files/Esquirol/merged.db`, opened read-only by the
  native benchmark.
- Initial trajectory: `input_camera_poses.txt`, a frozen copy of the existing
  exported trajectory. Its original provenance is saved alongside it.
- This is a controlled **additional BA pass from that same existing trajectory**,
  not a rerun of graph optimization from raw odometry.
- Native g2o, mono residuals, 50 configured final iterations plus the existing
  five-iteration warmup, eight matching threads. Progressive rejection is OFF; active-support pruning is OFF (the historical
  ordinary baseline). The benchmark library is frozen in `frozen_runtime/` to
  isolate this comparison from the subsequently requested support-control edits.
- Features/descriptors, calibration, graph constraints, matching settings and
  meshing settings are common to all three variants.
- `run_bench.py pilot` runs a fixed connected 800-frame loop-rich subset and
  repeats the unlimited baseline to expose timing variability/cache effects.
- `run_bench.py full` runs every one of the 43,971 exported frames sequentially.
- `run_meshes.py` builds whole-scan weighted-Poisson OBJs in `objs/`, using fixed
  Poisson depth 12, depth subsampling 4, PCA neighbours 50, and current remaining
  defaults. Separate workspaces/indexes keep the source project's poses intact.
- `summarize.py` produces `results.json`, including pose differences against the
  unlimited variant. Mesh metadata records the exact reconstruction parameters.

Matching timings include geometry/descriptor loading, pair selection and track
construction. Solver timings are separate. These are local single-machine
measurements, not a claim of general speedup; filesystem caching and system load
can affect runtime. Track counts and pose differences do not establish accuracy
without ground truth or visual inspection.

## Selection and fallback

The new production parameter is `Optimizer/BAExtraPairBudget` (default **0**, off).
The Python CLI/TUI exposes it as `--ba-extra-pair-budget` / **BA extra pair budget
(0 = unlimited)**, in weighted-Poisson and rtabmap-cloud. Pose cache provenance
includes the value. It is independent of `--ba-refinement`.

For each frame, geometry ranking samples at most 48 finite stored 3D features,
one per occupied cell of an 8×6 image grid. Extra pairs get a score from symmetric
projected overlap and median parallax, saturated at 1.5 degrees. Projected overlap
is a ranking heuristic; it does not perform depth-buffer occlusion testing.
Unknown geometry is not hard-rejected. Selection greedily favours useful pairs
and camera positions different from already selected partners, including protected
graph neighbours. A lazy priority queue handles the decreasing diversity scores.
The diversity distance scale is 10% of the source frame's median sampled scene
range, with a 1 cm minimum. First-camera geometry ranks multi-camera frames;
actual matching and the BA solver retain their existing calibration handling.

After matching the initial selection, the code counts conflict-free tracks that
span at least three frames. A frame is weak if fewer than 20 such tracks remain,
or they occupy fewer than four cells of a 4×4 image grid. One fallback pass restores
all deferred pairs touching a weak frame. Therefore final extra-partner counts
can exceed the initial budget, but never exceed the original proposal set. This
pre-triangulation support test is not a guarantee of geometric observability.

A budget of zero follows the original pair-selection path. The new logging reports
proposal/selection/fallback counts and surviving three-view track coverage.

## Validation and caveats

Python CLI/TUI, provenance, and existing meshing tests: 93 passed. Focused native
selection/track/BA tests: 16 passed, 10 backend-specific cases skipped. Native
checks cover unlimited selection, symmetric degree caps, viewpoint diversity,
and retention of shared tracks through the actual correspondence builder.

`preliminary_single_direction/` contains superseded pilot outputs and an aborted
full run from an initial benchmark harness that removed reverse pose constraints.
Those are excluded from `results.json`. The corrected harness preserves both
stored directions exactly as the production export BA caller does. No production
constraint handling was changed.

## Geometry evaluation

`evaluate_geometry.py` scores all trajectories using the same depth samples and
fixed correspondences selected from the frozen input trajectory. It chooses
150 candidate nearby pairs (~2 seconds apart) and 150 revisit pairs (>=30 seconds
apart), tests both directions, and scores signed point-to-plane residuals without
per-variant matching, ICP or residual rejection. Sensor-only gates use medium-or-
higher confidence, 0.2–5 m depth, valid 3×3 neighborhoods with <=10 cm depth span,
and every fourth pixel. Initial projective depth disagreement must be <=20 cm;
each direction needs 50 samples. Pair selection also requires camera separation
0.1–2 m and optical-axis dot product >=0.7. Seed and trajectory hashes are saved.

`geometry_quality.json` records every selected/skipped pair, accepted sample
counts, per-pair errors and equal-pair-weight aggregate differences against
budget 0. It measures **local geometric consistency, not absolute pose accuracy**.
Depth informed upstream matching and graph constraints, so it is not independent
ground truth. Frozen initial correspondences exclude gross initial misalignment;
planes do not constrain tangential drift; sampled pairs can be correlated.
Small metric differences should not be sold as meaningful quality gains without
whole-mesh inspection and replication on other scans.

The printed track/observation counts are **before BA outlier rejection**. The
benchmark also counts rejected observations, but neither quantity certifies
landmark quality. A budget can prevent conflicting unions and retain useful
observations, or fragment a track into weaker pieces. More landmarks alone is
not a reason to change the default.

## Completed comparison (2026-09-29)

Full BA took 1268.511 seconds with budget 0, 1349.207 with budget 8 and
1349.282 with budget 16. Both budgeted variants were about 6.4% slower here.
The fixed-correspondence evaluation retained 123 nearby pairs and 91 revisit
pairs, totaling 307,170 directional samples. Average point-to-plane disagreement
was approximately 34 mm nearby and 36 mm on revisits; differences between
budgeted variants and unlimited were below 0.06 mm in those aggregate means.
This test does not establish a meaningful quality improvement or absolute pose
accuracy. More shared tracks around loops remain an interesting structural
change, not proof of improved reconstruction. Budget 0 remains the default.

Whole OBJs are in `objs/`, with per-mesh manifests `mesh_00.json`, `mesh_08.json`
and `mesh_16.json`. `verify_obj_exports.py` exports 17-significant-digit vertex
coordinates and checks every vertex and face against the source PLY. The local
Open3D OBJ importer reduces coordinates to float32; re-import is checked at that
precision separately, with the maximum coordinate rounding saved in each
manifest. `obj_precision_complete.json` marks completion of these extra checks.
The source PLY and the OBJ text preserve full source geometry.
