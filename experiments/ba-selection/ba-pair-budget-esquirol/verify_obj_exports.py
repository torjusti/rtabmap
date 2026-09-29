"""Validate finished whole meshes; repair only text export precision, not meshes."""
from pathlib import Path
import time,json,hashlib
import numpy as np
import open3d as o3d
from obj_io import write_precise_obj,verify,verify_text

r=Path(__file__).resolve().parent
for budget in (0,8,16):
    manifest=r/f'mesh_{budget:02d}.json'
    while not manifest.exists():
        status=json.loads((r/'mesh_status.json').read_text())
        if status['status']=='failed':raise RuntimeError('Meshing failed')
        time.sleep(5)
    d=json.loads(manifest.read_text())
    if d.get('obj_exact_oriented_geometry'):continue
    mesh=o3d.io.read_triangle_mesh(d['ply'])
    assert len(mesh.triangles) and np.isfinite(np.asarray(mesh.vertices)).all()
    mesh.compute_vertex_normals()
    obj=Path(d['obj']); temporary=obj.with_name(obj.stem+'.verifying.obj')
    write_precise_obj(temporary,mesh)
    verify_text(temporary,mesh)
    reader_rounding=verify(mesh,o3d.io.read_triangle_mesh(str(temporary)))
    temporary.replace(obj)
    digest=hashlib.sha256()
    with obj.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
    d['obj_sha256']=digest.hexdigest()
    d.update(obj_bytes=obj.stat().st_size,obj_coordinate_significant_digits=17,obj_exact_oriented_geometry=True,
             obj_text_vertices_faces_exact=True,obj_importer_float32_geometry_verified=True,
             obj_importer_max_coordinate_rounding_m=reader_rounding)
    manifest.write_text(json.dumps(d,indent=2)+'\n')
    print('VERIFIED',obj,flush=True)
(r/'obj_precision_complete.json').write_text(json.dumps(dict(status='complete',time=time.time())))
