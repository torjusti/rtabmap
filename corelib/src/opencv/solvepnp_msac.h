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

// 1. OpenCV-Style Core Layer
bool solvePnPMsac(cv::InputArray objectPoints, cv::InputArray imagePoints,
                  cv::InputArray cameraMatrix, cv::InputArray distCoeffs,
                  const std::vector<cv::Matx33f>& covariances3A,
                  cv::OutputArray rvec, cv::OutputArray tvec,
                  bool useExtrinsicGuess = false, int iterationsCount = 100,
                  float confidence = 0.99, float pixelVariance = 1.0f,
                  bool use_prosac_ordering = false,
                  cv::OutputArray inliers = cv::noArray(), int flags = cv::SOLVEPNP_ITERATIVE);

// 2. RTAB-Map Wrapper / Refinement Layer
void solvePnPMsac(const std::vector<cv::Point3f> & objectPoints,
                  const std::vector<cv::Point2f> & imagePoints,
                  const cv::Mat & cameraMatrix,
                  const cv::Mat & distCoeffs,
                  const std::vector<cv::Matx33f> & covariances3A,
                  cv::Mat & rvec,
                  cv::Mat & tvec,
                  bool useExtrinsicGuess,
                  int iterationsCount,
                  float reprojectionError, // Unused internally, overridden by Chi2
                  int minInliersCount,
                  float confidence,
                  float pixelVariance,
                  std::vector<int> & inliers,
                  int flags,
                  int refineIterations,
                  float refineSigma,
                  bool use_prosac_ordering,
                  cv::Mat & outPoseCovariance);

cv::Ptr<cv3::PointSetRegistrator> createMSACPointSetRegistrator(
    const cv::Ptr<cv3::PointSetRegistrator::Callback>& cb,
    int modelPoints, double threshold, double confidence,
    int maxIters, bool prosacOrdering);

} // namespace cv_custom

#endif /* RTABMAP_CORELIB_SRC_OPENCV_SOLVEPNP_MSAC_H_ */