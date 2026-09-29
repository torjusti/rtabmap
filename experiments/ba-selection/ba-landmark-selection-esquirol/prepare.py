"""Prepare a separate, reproducible landmark-selection experiment."""
from pathlib import Path
import shutil

root=Path(__file__).resolve().parent
old=root.parent/'ba-pair-budget-esquirol'
for name in ('input_camera_poses.txt','input_poses_provenance.json','pilot_ids.txt',
             'pilot_selection.json','build_benchmark.py','obj_io.py'):
    shutil.copyfile(old/name,root/name)
s=(old/'benchmark.cpp').read_text()
s=s.replace('budget output_prefix iterations','landmark_selection output_prefix iterations')
s=s.replace('p[Parameters::kOptimizerBAExtraPairBudget()]=argv[4];', '''p[Parameters::kOptimizerBAExtraPairBudget()]="0";
    p[Parameters::kOptimizerBAMinActiveObservations()]="2";
    p[Parameters::kOptimizerBALandmarkSelection()]=std::string(argv[4])=="1"?"true":"false";
    p[Parameters::kOptimizerBALandmarkMinCoverageGain()]="0.05";
    p[Parameters::kOptimizerBADiagnostics()]="false";''')
s=s.replace(' // isolate pair-selection effect',' // isolate landmark-selection effect')
s=s.replace(' budget=', ' landmark_selection=')
s=s.replace('"{\\"budget\\":', '"{\\"landmark_selection\\":')
# Record a reproducibility hash over exactly the point/residual input for each
# independent process. This is a local deterministic equality check, not a
# cryptographic hash; source/provenance files also get SHA-256 in run.py.
s=s.replace('#include <iomanip>','#include <iomanip>\n#include <cstdint>')
s=s.replace('    BAOutliers outliers;', '''    uint64_t hash=14695981039346656037ULL;
    auto addHash=[&hash](const auto & value) {
        const unsigned char * bytes=reinterpret_cast<const unsigned char *>(&value);
        for(size_t i=0;i<sizeof(value);++i) { hash^=bytes[i]; hash*=1099511628211ULL; }
    };
    for(const auto & track:refs) {
        addHash(track.first);
        const auto point=points.at(track.first);
        addHash(point.x);addHash(point.y);addHash(point.z);
        for(const auto & observation:track.second) {
            addHash(observation.first);addHash(observation.second.cameraIndex);
            addHash(observation.second.kpt.pt.x);addHash(observation.second.kpt.pt.y);
            addHash(observation.second.depth);
        }
    }
    std::cout<<"BA_INPUT_HASH "<<hash<<std::endl;
    // Exclude hashing and census from the reported optimization wall time.
    timer.restart();
    BAOutliers outliers;''')
s=s.replace('<<",\\"tracks\\":"', '<<",\\"input_hash\\":\\""<<hash<<"\\""<<",\\"tracks\\":"')
(root/'benchmark.cpp').write_text(s)
s=(old/'evaluate_geometry.py').read_text()
s=s.replace("**{f'budget_{b}': ROOT/f'full_{b:02d}_camera_poses.txt' for b in (0, 8, 16)}", "**{name: ROOT/f'full_{value:02d}_camera_poses.txt' for name,value in [('all_landmarks',0),('selected_landmarks',1)]}")
s=s.replace("'budget_0'", "'all_landmarks'").replace('budget0', 'all_landmarks')
(root/'evaluate_geometry.py').write_text(s)
s=(old/'mesh.py').read_text()
s=s.replace('budget_', 'selection_')
s=s.replace("'0','--Optimizer/BAProgressive','false','--Optimizer/BAExtraPairBudget',str(BUDGET)", "'2','--Optimizer/BAProgressive','false','--Optimizer/BAExtraPairBudget','0','--Optimizer/BALandmarkSelection',str(bool(BUDGET)).lower(),'--Optimizer/BALandmarkMinCoverageGain','0.05'")
s=s.replace("'--ba-extra-pair-budget',str(BUDGET)","'--ba-extra-pair-budget','0', '--ba-landmark-selection' if BUDGET else '--no-ba-landmark-selection'")
s=s.replace("'--ba-min-observations','0'", "'--ba-min-observations','2'")
s=s.replace("'00_unlimited' if BUDGET==0 else f'{BUDGET:02d}_extra_pairs'", "'00_all_landmarks' if BUDGET==0 else '01_colmap_selection'")
s=s.replace('budget=BUDGET','landmark_selection=bool(BUDGET)')
(root/'mesh.py').write_text(s)
