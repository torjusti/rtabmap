#ifndef RTABMAP_CORELIB_SRC_OPENCV_SOLVEPNP_MSAC_H_
#define RTABMAP_CORELIB_SRC_OPENCV_SOLVEPNP_MSAC_H_

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#if CV_MAJOR_VERSION >= 3
#include <opencv2/calib3d/calib3d_c.h>
#endif

#include "solvepnp.h"

using namespace cv;

namespace cv_custom {

cv::Mat poseCovarianceRodriguesToRPY(const cv::Mat& rvec, const cv::Mat& cov_rodriguez);

std::vector<float> computeMahalanobisReprojErrors(
   const std::vector<cv::Point3f>& opoints,
   const std::vector<cv::Point2f>& ipoints,
   const cv::Mat & cameraMatrix,
   const cv::Mat & distCoeffs,
   const cv::Mat & rvec,
   const cv::Mat & tvec,
   const std::vector<cv::Matx33f> & cov3D,
   float pixelVariance,
   float unsquaredThreshold,
   std::vector<int> & inliers);

bool solvePnPMsac(cv::InputArray objectPoints, cv::InputArray imagePoints,
                  cv::InputArray cameraMatrix, cv::InputArray distCoeffs,
                  const std::vector<cv::Matx33f>& covariances3A,
                  cv::OutputArray rvec, cv::OutputArray tvec,
                  bool useExtrinsicGuess = false, int iterationsCount = 100,
                  float chi2Threshold = 2.44765f, float confidence = 0.99, 
                  float pixelVariance = 1.0f, bool use_prosac_ordering = false,
                  cv::OutputArray inliers = cv::noArray(), int flags = cv::SOLVEPNP_ITERATIVE);

void solvePnPMsacRefineLM(
    cv::Mat & rvec,
    cv::Mat & tvec,
    const std::vector<cv::Point3f> & opoints_inliers,
    const std::vector<cv::Point2f> & ipoints_inliers,
    const std::vector<cv::Matx33f> & cov_inliers,
    float pixelVariance,
    const cv::Mat & cameraMatrix,
    const cv::Mat & distCoeffs,
    int maxIterations = 20);

cv::Ptr<cv3::PointSetRegistrator> createMSACPointSetRegistrator(
    const cv::Ptr<cv3::PointSetRegistrator::Callback>& cb,
    int modelPoints, double threshold, double confidence,
    int maxIters, bool prosacOrdering);

} // namespace cv_custom

#endif /* RTABMAP_CORELIB_SRC_OPENCV_SOLVEPNP_MSAC_H_ */