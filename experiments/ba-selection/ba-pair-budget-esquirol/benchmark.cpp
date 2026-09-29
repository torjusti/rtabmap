#include <rtabmap/core/DBDriver.h>
#include <rtabmap/core/Optimizer.h>
#include <rtabmap/core/ProgressState.h>
#include <rtabmap/core/Signature.h>
#include <rtabmap/core/Graph.h>
#include <rtabmap/core/CameraModel.h>
#include <rtabmap/utilite/ULogger.h>
#include <rtabmap/utilite/UTimer.h>
#include <fstream>
#include <sstream>
#include <memory>
#include <iomanip>
using namespace rtabmap;
struct Progress : ProgressState {
    mutable UTimer clock;
    bool callback(const std::string & msg) const override {
        if(clock.elapsed()>20.0 || msg.find("iteration")!=std::string::npos) {
            std::cout<<"PROGRESS "<<msg<<std::endl;clock.restart();
        }
        return true;
    }
};
int main(int argc,char **argv)
{
    if(argc!=7) {std::cerr<<"benchmark DB input_camera_poses ids_file_or_all budget output_prefix iterations\n";return 2;}
    ULogger::setType(ULogger::kTypeConsole);ULogger::setLevel(ULogger::kInfo);
    std::unique_ptr<DBDriver> db(DBDriver::create());
    if(!db->openConnection(argv[1],false,true)) return 3;
    std::set<int> selected;
    if(std::string(argv[3])!="all") {std::ifstream f(argv[3]);int id;while(f>>id)selected.insert(id);if(selected.empty())return 4;}
    std::map<int,Transform> poses;std::map<int,double> stamps;
    std::map<int,std::vector<CameraModel> > models;
    std::ifstream input(argv[2]);std::string line;
    while(std::getline(input,line))
    {
        if(line.empty()||line[0]=='#') continue;
        std::istringstream in(line);double stamp,x,y,z,qx,qy,qz,qw;int id;
        if(!(in>>stamp>>x>>y>>z>>qx>>qy>>qz>>qw>>id)) return 5;
        if(id<=0||(!selected.empty()&&!selected.count(id)))continue;
        std::vector<CameraModel> camera;std::vector<StereoCameraModel> stereo;
        if(!db->getCalibration(id,camera,stereo)||camera.size()!=1) return 6;
        poses[id]=Transform(x,y,z,qx,qy,qz,qw)*camera[0].localTransform().inverse();
        models[id]=camera;stamps[id]=stamp;
    }
    std::multimap<int,Link> allLinks,links;db->getAllLinks(allLinks,true,false);
    // Preserve every stored metric factor, including reverse directions,
    // exactly as the export BA caller does. Only feature matching deduplicates.
    for(const auto & entry:allLinks)
        if(poses.count(entry.second.from()) && poses.count(entry.second.to())) links.insert(entry);
    ParametersMap p=db->getLastParameters();
    p[Parameters::kOptimizerBAStrategy()]="1";
    p[Parameters::kOptimizerBADepth()]="0";
    p[Parameters::kOptimizerBAExtraPairBudget()]=argv[4];
    p[Parameters::kOptimizerBAProgressive()]="false"; // isolate pair-selection effect
    p[Parameters::kOptimizerBAProposePairs()]="false";
    p[Parameters::kOptimizerBAWindowGraphDepth()]="2";
    p[Parameters::kOptimizerBAMatchCache()]="false";
    p[Parameters::kOptimizerIterations()]=argv[6];
    Optimizer::applyBundleAdjustmentKernel(p);
    std::unique_ptr<Optimizer> opt(Optimizer::create(Optimizer::kTypeG2O,p));
    Progress progress;opt->setProgressState(&progress);
    std::cout<<"BENCH frames="<<poses.size()<<" links="<<links.size()<<" budget="<<argv[4]<<std::endl;
    UTimer timer;std::map<int,cv::Point3f> points;
    std::map<int,std::map<int,FeatureBA> > refs;
    opt->computeBACorrespondences(poses,links,{},points,refs,true,false,{},db.get());
    double match=timer.ticks();
    std::map<int,int> coverage;size_t observations=0,three=0;
    for(const auto & track:refs) { observations+=track.second.size();if(track.second.size()>=3)++three;for(const auto & v:track.second)++coverage[v.first]; }
    BAOutliers outliers;
    const auto result=opt->optimizeBA(poses.begin()->first,poses,links,models,points,refs,&outliers);
    double solve=timer.ticks();
    if(result.size()!=poses.size()) {std::cerr<<"incomplete BA output"<<std::endl;return 7;}
    std::ofstream output(std::string(argv[5])+"_camera_poses.txt");
    output<<"# stamp x y z qx qy qz qw id\n"<<std::setprecision(12);
    double displacement=0,maxDisplacement=0;size_t rejected=0;
    for(const auto & item:result)
    {
        const auto t=item.second*models.at(item.first)[0].localTransform();
        const auto q=t.getQuaterniond();
        output<<stamps.at(item.first)<<" "<<t.x()<<" "<<t.y()<<" "<<t.z()<<" "<<q.x()<<" "<<q.y()<<" "<<q.z()<<" "<<q.w()<<" "<<item.first<<"\n";
        double d=(poses.at(item.first).inverse()*item.second).getNorm();displacement+=d*d;maxDisplacement=std::max(d,maxDisplacement);
    }
    for(const auto & o:outliers)rejected+=o.second.size();
    std::ofstream json(std::string(argv[5])+".json");
    json<<"{\"budget\":"<<argv[4]<<",\"frames\":"<<poses.size()<<",\"correspondence_seconds\":"<<match<<",\"solve_seconds\":"<<solve
        <<",\"tracks\":"<<refs.size()<<",\"three_view_tracks\":"<<three<<",\"observations\":"<<observations<<",\"covered_frames\":"<<coverage.size()
        <<",\"rejected_observations\":"<<rejected<<",\"pose_rms_shift_m\":"<<std::sqrt(displacement/poses.size())<<",\"pose_max_shift_m\":"<<maxDisplacement<<"}\n";
    std::cout<<"BENCH_DONE match_seconds="<<match<<" solve_seconds="<<solve<<std::endl;
}
