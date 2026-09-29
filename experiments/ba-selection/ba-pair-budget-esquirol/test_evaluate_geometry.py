from types import SimpleNamespace
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from evaluate_geometry import camera_to_camera, point_plane_residual, correspondences


def pose(translation=(0, 0, 0), rotation=None):
    return SimpleNamespace(translation=np.array(translation, float),
                           rotation=np.eye(3) if rotation is None else rotation)


class GeometryEvaluationTest(unittest.TestCase):
    def test_transform_matches_explicit_world_route_and_global_frame_invariance(self):
        a = pose((4000, -5000, 2), Rotation.from_euler('xyz', [.1, .2, .3]).as_matrix())
        b = pose((4001, -4999, 3), Rotation.from_euler('xyz', [-.3, .2, .1]).as_matrix())
        p = np.array([[.2, .1, 2], [-.4, .3, 3]])
        expected = ((p@a.rotation.T+a.translation)-b.translation)@b.rotation
        np.testing.assert_allclose(camera_to_camera(p, a, b), expected, atol=1e-11)
        R = Rotation.from_euler('xyz', [.3, -.5, .9]).as_matrix()
        t = np.array([7, 8, 9])
        aa = pose(R@a.translation+t, R@a.rotation)
        bb = pose(R@b.translation+t, R@b.rotation)
        np.testing.assert_allclose(camera_to_camera(p, aa, bb), expected, atol=1e-11)

    def test_known_normal_displacement_and_planar_tangential_blindness(self):
        p = np.array([[.2, .3, 2], [-.2, -.1, 2]])
        n = np.tile([0, 0, 1], (2, 1))
        np.testing.assert_allclose(point_plane_residual(p, p, n, pose(), pose()), 0)
        np.testing.assert_allclose(point_plane_residual(p, p, n, pose((0, 0, .01)), pose()), .01)
        np.testing.assert_allclose(point_plane_residual(p, p, n, pose((.01, 0, 0)), pose()), 0)

    def test_correspondence_gate_ignores_invalid_and_occluded_samples(self):
        y, x = np.indices((12, 12))
        xyz = np.stack(((x-6)*.02, (y-6)*.02, np.full_like(x, 2)), axis=-1)
        normals = np.zeros_like(xyz); normals[..., 2] = 1
        valid = np.ones((12, 12), bool)
        frame = (xyz, normals, valid, (100, 100, 6, 6))
        points, targets, _ = correspondences(frame, frame, pose(), pose())
        self.assertEqual(len(points), 9)
        np.testing.assert_allclose(points, targets)
        invalid = (xyz, normals, ~valid, frame[3])
        self.assertEqual(len(correspondences(frame, invalid, pose(), pose())[0]), 0)
        self.assertEqual(len(correspondences(frame, frame, pose((0, 0, .3)), pose())[0]), 0)


if __name__ == '__main__':
    unittest.main()
