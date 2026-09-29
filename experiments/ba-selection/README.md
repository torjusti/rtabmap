# Archived BA selection experiments

This branch preserves the extra-pair budget and COLMAP-style landmark selector,
including their CLI/TUI controls. Neither justified adoption on Esquirol:

* Pair budgets 8/16: about 6.4% slower overall, no meaningful measured geometry improvement.
* COLMAP-style selection: 65.6% of landmarks selected, total BA 698.7 s versus
  613.5 s baseline (13.9% slower), no meaningful consistency improvement.

The active checkouts remove these two experiments while retaining normal
estimation improvements, established BA paths, support policy and diagnostics.
Both repositories have a companion branch named
`archive/ba-selection-experiments-2026-09-29`.

Native source: https://github.com/torjusti/rtabmap/tree/archive/ba-selection-experiments-2026-09-29
Python CLI/TUI: https://github.com/torjusti/rtab-scripts/tree/archive/ba-selection-experiments-2026-09-29

Benchmark source and compact result manifests are archived in the native repo
under `experiments/ba-selection/`. Raw scans, images, trajectories, native
binaries and large meshes remain local in the Scani analysis folders.
The native archive also includes existing source prerequisites referenced by
its parent commit that had not yet been tracked. No unrelated local models,
build outputs or application data are included.
