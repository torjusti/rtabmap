"""Whole Esquirol OBJs, with isolated indexes and frozen per-variant BA poses."""
from pathlib import Path
import sys,os,json,shutil,time,subprocess
ROOT=Path(__file__).resolve().parent
DB=Path('/media/torjusti/Files/Esquirol/merged.db')
sys.path.insert(0,'/home/torjusti/Projects/3d')
from rtab.project import Project as BaseProject
SOURCE=BaseProject(DB)
BUDGET=int(sys.argv[1]) if len(sys.argv)>1 else 0
WORK=ROOT/'mesh_work'/f'selection_{BUDGET:02d}'

class EvaluationProject(BaseProject):
    def __init__(self,db):
        super().__init__(db)
        old=self.db_dir
        for key,value in list(vars(self).items()):
            if key!='db_path' and isinstance(value,Path) and value.is_relative_to(old):
                setattr(self,key,WORK/value.relative_to(old))
    def ensure_exports(self,**kwargs):
        if kwargs.get('cloud'):raise RuntimeError('No source cloud exports in this experiment')
        original=json.loads((ROOT/'input_poses_provenance.json').read_text())
        if self.database_file_signature()!=original['source']:
            raise RuntimeError('Source database changed since the frozen input trajectory')
        self.exports_dir.mkdir(parents=True,exist_ok=True)
        for kind in ('rgb','depth','calib','confidence'):
            src=SOURCE.exports_dir/f'{self.rtabmap_stem}_{kind}'
            dst=self.exports_dir/src.name
            if not src.is_dir():raise RuntimeError(f'Missing input: {src}')
            if not dst.exists():dst.symlink_to(src,target_is_directory=True)
        trajectory=ROOT/f'full_{BUDGET:02d}_camera_poses.txt'
        if not trajectory.is_file():raise RuntimeError('BA run is not complete')
        content=trajectory.read_bytes()
        if self.poses_file.exists() and self.poses_file.read_bytes()!=content:
            raise RuntimeError('Frozen variant poses changed')
        self.poses_file.write_bytes(content)
        options=['--ba','--Optimizer/BAMinActiveObservations','2','--Optimizer/BAProgressive','false','--Optimizer/BAExtraPairBudget','0','--Optimizer/BALandmarkSelection',str(bool(BUDGET)).lower(),'--Optimizer/BALandmarkMinCoverageGain','0.05']
        self.pose_export_options_path.write_text(json.dumps(options))
        self.poses_provenance_path.write_text(json.dumps({'source':original['source'],'export_flags':self.camera_pose_export_flags(options)}))
        link=WORK/DB.name
        if not link.exists():link.symlink_to(DB)

if __name__=='__main__':
    from rtab.commands import fast_cloud
    from rtab.cli import build_parser
    import open3d as o3d,numpy as np
    fast_cloud.Project=EvaluationProject
    # Matches the user's current Esquirol weighted-Poisson settings. Fixed depth
    # avoids a pose-bounds change accidentally selecting a different octree depth.
    args=build_parser().parse_args(['weighted-poisson',str(DB),'--ba-extra-pair-budget','0', '--ba-landmark-selection' if BUDGET else '--no-ba-landmark-selection',
        '--ba-min-observations','2','--poisson-depth','12','--downsample','4','--prefetch','8','--yes'])
    start=time.perf_counter();code=fast_cloud.run_cli(args)
    if code:raise SystemExit(code)
    project=EvaluationProject(DB)
    items=project.index.list_by_command('cloud-poisson')
    assert len(items)==1
    path=project.artifact_abspath(items[0].path)
    mesh=o3d.io.read_triangle_mesh(str(path));vertices=np.asarray(mesh.vertices)
    assert len(vertices)>0 and len(mesh.triangles)>0 and np.isfinite(vertices).all()
    mesh.compute_vertex_normals()
    out=ROOT/'objs';out.mkdir(exist_ok=True)
    name='00_all_landmarks' if BUDGET==0 else '01_colmap_selection'
    obj=out/f'{name}_full.obj'
    from obj_io import write_precise_obj, verify, verify_text
    write_precise_obj(obj, mesh)
    verify_text(obj, mesh)
    verify(mesh, o3d.io.read_triangle_mesh(str(obj)))
    meta=dict(landmark_selection=bool(BUDGET),mesh_seconds=time.perf_counter()-start,obj=str(obj),ply=str(path),
        vertices=len(vertices),triangles=len(mesh.triangles),bounds=[vertices.min(0).tolist(),vertices.max(0).tolist()],
        parameters=items[0].params)
    (ROOT/f'mesh_{BUDGET:02d}.json').write_text(json.dumps(meta,indent=2))
    print('OBJ_READY',obj,flush=True)
