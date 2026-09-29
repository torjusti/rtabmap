"""Compare frozen BA trajectories on identical, bidirectional LiDAR correspondences.

This measures local geometric consistency, NOT absolute pose accuracy. Depth was
used upstream for matching/metric constraints, so it is not independent ground
truth. Selection uses only the frozen INPUT trajectory, never a tested variant.
No per-variant ICP, re-matching, residual rejection, or best-subset selection.
"""
from pathlib import Path
from functools import lru_cache
import argparse
import hashlib
import json
import os
import sys

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
sys.path.insert(0, '/home/torjusti/Projects/3d')
import cv2
import numpy as np
from scipy.spatial import cKDTree
from rtab.project import Project
from rtab.utils.geometry import iter_camera_poses
from rtab.utils.depth_filters import intrinsics_for_depth

ROOT = Path(__file__).resolve().parent
DB = Path('/media/torjusti/Files/Esquirol/merged.db')


def load_poses(path):
    poses = {p.image_id: p for p in iter_camera_poses(path.read_text().splitlines())}
    if not poses:
        raise ValueError(f'Empty trajectory: {path}')
    return poses


def camera_to_camera(points, a, b):
    # Subtract camera centers before applying rotation, avoiding cancellation
    # from this scan's large absolute world coordinates.
    return points @ (b.rotation.T @ a.rotation).T + (a.translation-b.translation) @ b.rotation


def point_plane_residual(points, targets, normals, a, b):
    return np.einsum('ij,ij->i', camera_to_camera(points, a, b)-targets, normals)


def select_pairs(poses, per_class=150):
    ordered = sorted(poses.values(), key=lambda p: (p.stamp, p.image_id))
    stamps = np.array([p.stamp for p in ordered])
    positions = np.array([p.translation for p in ordered])
    tree = cKDTree(positions)
    rng = np.random.default_rng(73421)
    picked = {'nearby': [], 'revisit': []}
    seen = set()
    for i in rng.permutation(len(ordered)):
        a = ordered[i]
        for category in picked:
            if len(picked[category]) >= per_class:
                continue
            if category == 'nearby':
                j = int(np.searchsorted(stamps, a.stamp+2.0))
                candidates = [j] if j < len(ordered) and 1 <= stamps[j]-a.stamp <= 4 else []
            else:
                candidates = tree.query_ball_point(positions[i], 2.0)
                # Wider temporal separation examines independently revisited
                # surfaces. Randomize so dense revisit sequences don't always
                # pick the temporally first frame.
                rng.shuffle(candidates)
            for j in candidates:
                b = ordered[j]
                dt = abs(a.stamp-b.stamp)
                distance = np.linalg.norm(a.translation-b.translation)
                pair = tuple(sorted((a.image_id, b.image_id)))
                if (pair in seen or distance < .1 or distance > 2.0 or
                    np.dot(a.rotation[:, 2], b.rotation[:, 2]) < .7 or
                    (category == 'revisit' and dt < 30)):
                    continue
                seen.add(pair)
                picked[category].append(pair)
                break
        if all(len(v) >= per_class for v in picked.values()):
            break
    return picked


def make_loader(project):
    @lru_cache(maxsize=12)
    def load(frame):
        raw = cv2.imread(str(project.depth_dir/f'{frame}.png'), cv2.IMREAD_UNCHANGED)
        confidence = cv2.imread(str(project.confidence_dir/f'{frame}.png'), cv2.IMREAD_UNCHANGED)
        calib = project.read_calibration(frame)
        if raw is None or confidence is None or calib is None or raw.shape != confidence.shape:
            return None
        if raw.dtype != np.uint16 or raw.ndim != 2:
            raise ValueError('Expected exported uint16 millimetre depth')
        z = raw.astype(np.float64)*.001
        fx, fy, cx, cy = intrinsics_for_depth(calib, z.shape)
        y, x = np.indices(z.shape)
        xyz = np.stack(((x-cx)*z/fx, (y-cy)*z/fy, z), axis=-1)
        valid = (z >= .2) & (z <= 5.0) & (confidence >= 50)
        # Fixed sensor-only edge gate: all pixels in a 3x3 neighborhood valid
        # and its depth range <= 10 cm. Avoid normal fits across silhouettes.
        kernel = np.ones((3, 3), np.uint8)
        valid = cv2.erode(valid.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        valid &= cv2.dilate(z, kernel)-cv2.erode(z, kernel) <= .1
        normals = np.zeros_like(xyz)
        normals[1:-1, 1:-1] = np.cross(xyz[1:-1, 2:]-xyz[1:-1, :-2], xyz[2:, 1:-1]-xyz[:-2, 1:-1])
        norm = np.linalg.norm(normals, axis=-1)
        valid &= norm > 1e-9
        normals /= np.maximum(norm[..., None], 1e-9)
        return xyz, normals, valid, (fx, fy, cx, cy)
    return load


def correspondences(source, target, a, b):
    xyz, _, valid, _ = source
    target_xyz, normals, target_valid, (fx, fy, cx, cy) = target
    sampling = np.zeros(valid.shape, bool)
    sampling[2::4, 2::4] = True
    points = xyz[valid & sampling]
    projected = camera_to_camera(points, a, b)
    in_front = projected[:, 2] > .05
    points, projected = points[in_front], projected[in_front]
    uv = np.rint(projected[:, :2]/projected[:, 2, None]*[fx, fy]+[cx, cy]).astype(np.int64)
    h, w = target_valid.shape
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    points, projected, uv = points[inside], projected[inside], uv[inside]
    u, v = uv.T
    gate = target_valid[v, u] & (abs(projected[:, 2]-target_xyz[v, u, 2]) <= .2)
    return points[gate], target_xyz[v, u][gate], normals[v, u][gate]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--per-class', type=int, default=150)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    project = Project(DB)
    expected = json.loads((ROOT/'input_poses_provenance.json').read_text())['source']
    if project.database_file_signature() != expected:
        raise RuntimeError('Source DB changed since benchmark input was frozen')
    paths = {'input': ROOT/'input_camera_poses.txt', **{f'budget_{b}': ROOT/f'full_{b:02d}_camera_poses.txt' for b in (0, 8, 16)}}
    trajectories = {name: load_poses(path) for name, path in paths.items()}
    original = trajectories['input']
    if any(set(p) != set(original) for p in trajectories.values()):
        raise ValueError('Trajectories must contain exactly the same frames')
    pairs = select_pairs(original, args.per_class)
    load = make_loader(project)
    rows, skipped = [], []
    for category, selection in pairs.items():
        for a, b in selection:
            source, target = load(a), load(b)
            if source is None or target is None:
                skipped.append({'pair': [a, b], 'reason': 'missing depth/confidence/calibration'})
                continue
            data = []
            for first, second, aa, bb in ((source, target, a, b), (target, source, b, a)):
                points, targets, normals = correspondences(first, second, original[aa], original[bb])
                data.append((aa, bb, points, targets, normals))
            counts = [len(d[2]) for d in data]
            if min(counts) < 50:
                skipped.append({'pair': [a, b], 'reason': 'fewer than 50 correspondences in a direction', 'counts': counts})
                continue
            values = {}
            for name, poses in trajectories.items():
                residuals = np.concatenate([point_plane_residual(p, q, n, poses[aa], poses[bb]) for aa, bb, p, q, n in data])
                absolute = abs(residuals)
                values[name] = {'mean_abs_m': float(absolute.mean()), 'median_abs_m': float(np.median(absolute)),
                                'p90_abs_m': float(np.quantile(absolute, .9)), 'rms_m': float(np.sqrt(np.mean(residuals**2)))}
            rows.append({'class': category, 'pair': [a, b], 'samples': sum(counts), 'directions': counts,
                         'time_gap_s': abs(original[a].stamp-original[b].stamp), 'metrics': values})
            if len(rows) % 25 == 0:
                print('Scored pairs', len(rows), flush=True)
    summaries = {}
    for category in pairs:
        subset = [r for r in rows if r['class'] == category]
        if not subset:
            continue
        summaries[category] = {}
        for name in trajectories:
            # Each pair has equal weight: dense textured patches don't dominate.
            errors = np.array([r['metrics'][name]['mean_abs_m'] for r in subset])
            baseline = np.array([r['metrics']['budget_0']['mean_abs_m'] for r in subset])
            summaries[category][name] = {'pairs': len(subset), 'samples': sum(r['samples'] for r in subset),
                'pair_mean_abs_m': float(errors.mean()), 'median_pair_mean_abs_m': float(np.median(errors)),
                'mean_change_vs_budget0_m': float((errors-baseline).mean()),
                'median_change_vs_budget0_m': float(np.median(errors-baseline)),
                'fraction_pairs_better_than_budget0': float(np.mean(errors < baseline))}
    output = dict(method='fixed_input_correspondences_symmetric_point_to_plane',
                  limitations=['Local consistency, not ground-truth trajectory accuracy.',
                      'Depth informed upstream matching and metric links; it is not held-out ground truth.',
                      'Frozen input correspondence gate excludes gross initial misalignments.',
                      'Planar surfaces do not constrain tangential displacement.',
                      'Pair samples are correlated; win fractions are descriptive, not significance tests.'],
                  settings=dict(seed=73421, per_class=args.per_class, range_m=[.2, 5], confidence_min=50,
                      sample_step=4, normal_window=3, max_neighborhood_depth_span_m=.1,
                      input_projective_gate_m=.2, minimum_direction_samples=50),
                  input_hashes={k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()},
                  selected_pairs=pairs, skipped=skipped, summary=summaries, pairs=rows)
    (ROOT/'geometry_quality.json').write_text(json.dumps(output, indent=2))
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == '__main__':
    main()
