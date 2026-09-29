"""Preserve small geometry differences at large world-coordinate offsets."""
import numpy as np


def write_precise_obj(path, mesh):
    with path.open('w') as f:
        f.write('# Full precision world coordinates; metres\n')
        np.savetxt(f, np.asarray(mesh.vertices), fmt='v %.17g %.17g %.17g')
        faces = np.asarray(mesh.triangles, dtype=np.int64)+1
        if mesh.has_vertex_normals():
            np.savetxt(f, np.asarray(mesh.vertex_normals), fmt='vn %.17g %.17g %.17g')
            np.savetxt(f, np.repeat(faces, 2, axis=1), fmt='f %d//%d %d//%d %d//%d')
        else:
            np.savetxt(f, faces, fmt='f %d %d %d')


def verify_text(path, mesh):
    """Check actual file coordinates/faces, independently of importer precision."""
    vertices = np.asarray(mesh.vertices)
    with path.open() as f:
        assert f.readline().startswith('# Full precision')
        written = np.loadtxt(f, max_rows=len(vertices), usecols=(1, 2, 3))
        assert np.array_equal(written, vertices)
        indices = np.fromiter((int(part.split('//')[0])-1 for line in f
                               if line.startswith('f ') for part in line.split()[1:]), dtype=np.int64)
    assert np.array_equal(indices, np.asarray(mesh.triangles).ravel())


def verify(source, imported):
    """Check oriented triangles at the Open3D OBJ importer's float32 precision."""
    x, y = np.asarray(source.vertices), np.asarray(imported.vertices)
    assert np.isfinite(y).all()
    assert len(x) == len(y) and len(source.triangles) == len(imported.triangles)
    # Open3D's Assimp OBJ path reads coordinates as float32, even though its
    # Python mesh container uses float64. The file itself is checked separately.
    rounded = x.astype(np.float32).astype(np.float64)
    max_reader_rounding_m = float(np.max(abs(x-rounded)))
    x = rounded
    _, inverse = np.unique(np.concatenate((x, y)), axis=0, return_inverse=True)
    def canonical(mesh, ids):
        faces = ids[np.asarray(mesh.triangles)]
        smallest = faces.argmin(axis=1)
        faces = faces[np.arange(len(faces))[:, None], (smallest[:, None]+np.arange(3)) % 3]
        return faces[np.lexsort(faces.T[::-1])]
    assert np.array_equal(canonical(source, inverse[:len(x)]), canonical(imported, inverse[len(x):]))
    return max_reader_rounding_m
