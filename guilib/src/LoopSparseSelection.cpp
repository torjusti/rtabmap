#include "LoopSparseSelection.h"
#include <Eigen/Eigenvalues>

namespace rtabmap { namespace loopSparseSelection {

double informationScore(const Signature & from, const Signature & to,
        const Transform & transform, const RegistrationInfo & info)
{
    if(transform.isNull()) return 0;
    std::vector<CameraModel> models=to.sensorData().cameraModels();
    if(models.empty()) for(const auto & s : to.sensorData().stereoCameraModels()) models.push_back(s.left());
    typedef Eigen::Matrix<double,6,6> Matrix6;
    Matrix6 h=Matrix6::Zero();
    std::vector<double> residuals;
    const Eigen::Affine3d inverse=transform.inverse().toEigen3d();
    for(int id : info.inliersIDs)
    {
        if(from.getWords().count(id)!=1 || to.getWords().count(id)!=1) continue;
        size_t i=from.getWords().find(id)->second, j=to.getWords().find(id)->second;
        if(i>=from.getWords3().size() || j>=to.getWordsKpts().size()) continue;
        const cv::Point3f & p=from.getWords3()[i];
        Eigen::Vector3d q=inverse*Eigen::Vector3d(p.x,p.y,p.z);
        if(!q.allFinite()) continue;
        cv::Point2f pixel=to.getWordsKpts()[j].pt;
        const CameraModel * model=0;
        for(const auto & m : models)
        {
            if(pixel.x>=m.imageWidth() && models.size()>1) { pixel.x-=m.imageWidth(); continue; }
            model=&m; break;
        }
        if(!model || !model->isValidForProjection() || model->localTransform().isNull()) continue;
        const auto local=model->localTransform().inverse().toEigen3d();
        Eigen::Vector3d c=local*q;
        if(!c.allFinite() || c.z()<=0) continue;
        const double z=c.z(), fx=model->fx(), fy=model->fy();
        Eigen::Matrix<double,2,3> projection;
        projection << fx/z,0,-fx*c.x()/(z*z),0,fy/z,-fy*c.y()/(z*z);
        Eigen::Matrix3d skew;
        skew << 0,-q.z(),q.y(),q.z(),0,-q.x(),-q.y(),q.x(),0;
        Eigen::Matrix<double,3,6> motion;
        motion.leftCols<3>().setIdentity(); motion.rightCols<3>()=-skew;
        Eigen::Matrix<double,2,6> jacobian=projection*local.linear()*motion;
        // Reference uncertainty: 10 cm translation and 5 degrees rotation.
        jacobian.leftCols<3>()*=0.1;
        jacobian.rightCols<3>()*=5.0*CV_PI/180.0;
        double dx=fx*c.x()/z+model->cx()-pixel.x, dy=fy*c.y()/z+model->cy()-pixel.y;
        if(!jacobian.allFinite() || !std::isfinite(dx*dx+dy*dy)) continue;
        h.noalias()+=jacobian.transpose()*jacobian;
        residuals.push_back(dx*dx+dy*dy);
    }
    if(residuals.size()<6) return 0;
    auto median=residuals.begin()+residuals.size()/2;
    std::nth_element(residuals.begin(),median,residuals.end());
    // Median squared 2D Gaussian residual = 2*ln(2)*sigma^2.
    // A two-pixel floor avoids rewarding implausibly tiny fitted residuals.
    double variance=std::max(4.0,*median/(2.0*std::log(2.0)));
    Eigen::SelfAdjointEigenSolver<Matrix6> eigen(h/variance,Eigen::EigenvaluesOnly);
    if(eigen.info()!=Eigen::Success) return 0;
    double score=0;
    for(int i=0;i<6;++i) score+=0.5*std::log1p(std::max(0.0,eigen.eigenvalues()[i]));
    return std::isfinite(score) ? score : 0;
}

} }
