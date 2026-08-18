/*
Copyright (c) 2010-2016, Mathieu Labbe - IntRoLab - Universite de Sherbrooke
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Universite de Sherbrooke nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "rtabmap/core/util3d_motion_estimation.h"

#include "rtabmap/utilite/UStl.h"
#include "rtabmap/utilite/UMath.h"
#include "rtabmap/utilite/ULogger.h"
#include "rtabmap/utilite/UTimer.h"
#include "rtabmap/core/util3d_transforms.h"
#include "rtabmap/core/util3d_registration.h"
#include "rtabmap/core/util3d_correspondences.h"
#include "rtabmap/core/util3d.h"

#include <pcl/common/common.h>

#include "opencv/solvepnp.h"
#include "opencv/solvepnp_msac.h"

#ifdef RTABMAP_OPENGV
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/absolute_pose/NoncentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/NoncentralAbsoluteMultiAdapter.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac/MultiRansac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#include <opengv/sac_problems/absolute_pose/MultiNoncentralAbsolutePoseSacProblem.hpp>
#endif

namespace rtabmap
{

namespace util3d
{

namespace {
// When true, every newly-constructed OpenGV SAC problem inside this
// translation unit gets its RNG reseeded with a fixed constant so that
// RANSAC is bit-for-bit reproducible. Tests can flip this on via
// setRansacDeterministicSeed(true); production code leaves it off.
bool g_ransacDeterministicSeed = false;
} // namespace

void setRansacDeterministicSeed(bool enable)
{
	g_ransacDeterministicSeed = enable;
}

bool ransacDeterministicSeedEnabled()
{
	return g_ransacDeterministicSeed;
}

Transform estimateMotion3DTo2D(
    const std::map<int, cv::Point3f> & words3A,
    const std::map<int, cv::KeyPoint> & words2B,
    const CameraModel & cameraModel,
    int minInliers,
    int iterations,
    double reprojError,
    int flagsPnP,
    int refineIterations,
    int varianceMedianRatio,
    float maxLinVariance,
	float maxAngVariance,
    const Transform & guess,
    const std::map<int, cv::Point3f> & words3B,
    cv::Mat * covariance,
    std::vector<int> * matchesOut,
    std::vector<int> * inliersOut,
    bool splitLinearCovarianceComponents,
    const std::map<int, cv::Matx33f>& covariances3A,
    const std::map<int, cv::Matx33f>& covariances3B,
    bool useMsac,
    float pixelVariance,
    float depthVariance,
    bool computeFullCovariance, 
    bool useInlierVariance)
{
	UASSERT(cameraModel.isValidForProjection());
	UASSERT(!guess.isNull());
	UASSERT(varianceMedianRatio>1);

    Transform transform;
	std::vector<int> matches, inliers;
    
    if(covariance)
    {
        *covariance = cv::Mat::eye(6,6,CV_64FC1);
    }

    // find correspondences
	std::vector<int> ids = uKeys(words2B);
	std::vector<cv::Point3f> objectPoints(ids.size());
	std::vector<cv::Point2f> imagePoints(ids.size());
	int oi=0;
	matches.resize(ids.size());
	for(unsigned int i=0; i<ids.size(); ++i)
	{
		std::map<int, cv::Point3f>::const_iterator iter=words3A.find(ids[i]);
		if(iter != words3A.end() && util3d::isFinite(iter->second))
		{
			const cv::Point3f & pt = iter->second;
			objectPoints[oi].x = pt.x;
			objectPoints[oi].y = pt.y;
			objectPoints[oi].z = pt.z;
			imagePoints[oi] = words2B.find(ids[i])->second.pt;
			matches[oi++] = ids[i];
		}
	}

	objectPoints.resize(oi);
	imagePoints.resize(oi);
	matches.resize(oi);

	UDEBUG("words3A=%d words2B=%d matches=%d words3B=%d guess=%s reprojError=%f iterations=%d useMsac=%d",
		(int)words3A.size(), (int)words2B.size(), (int)matches.size(), (int)words3B.size(),
		guess.prettyPrint().c_str(), reprojError, iterations, useMsac?1:0);

	if((int)matches.size() >= minInliers)
    {
		//PnPRansac - PnPMsac
        cv::Mat K = cameraModel.K();
        cv::Mat D = cameraModel.D();
        Transform guessCameraFrame = (guess * cameraModel.localTransform()).inverse();
        cv::Mat R = (cv::Mat_<double>(3,3) <<
                (double)guessCameraFrame.r11(), (double)guessCameraFrame.r12(), (double)guessCameraFrame.r13(),
                (double)guessCameraFrame.r21(), (double)guessCameraFrame.r22(), (double)guessCameraFrame.r23(),
                (double)guessCameraFrame.r31(), (double)guessCameraFrame.r32(), (double)guessCameraFrame.r33());

        cv::Mat rvec(3,1, CV_64FC1);
        cv::Rodrigues(R, rvec);
        cv::Mat tvec = (cv::Mat_<double>(3,1) <<
                (double)guessCameraFrame.x(), (double)guessCameraFrame.y(), (double)guessCameraFrame.z());

		if(useMsac)
		{
			std::vector<cv::Matx33f> objectCovariances;
			if(useInlierVariance)
			{
				int covAvailableCount = 0;
				objectCovariances.resize(objectPoints.size());
				double fx = K.at<double>(0,0);
				double fy = K.at<double>(1,1);
				double d_pixelVar = static_cast<double>(pixelVariance);
				double d_depthVar = static_cast<double>(depthVariance);

				for(size_t i = 0; i < objectPoints.size(); ++i)
				{
					auto iter = covariances3A.find(matches[i]);
					if(iter != covariances3A.end())
					{
						objectCovariances[i] = iter->second;
						covAvailableCount++;
					}
					else
					{
						double Z = static_cast<double>(objectPoints[i].z);
						double var_x = Z*Z * d_pixelVar / (fx*fx);
						double var_y = Z*Z * d_pixelVar / (fy*fy);
						double var_z = d_depthVar;
						objectCovariances[i] = cv::Matx33f(
							static_cast<float>(var_x), 0.0f, 0.0f, 
							0.0f, static_cast<float>(var_y), 0.0f, 
							0.0f, 0.0f, static_cast<float>(var_z));
					}
				}
				if(covAvailableCount != (int)objectPoints.size())
				{
					UDEBUG("Covariance-aware PnPMsac: Not all object points have covariance available from sensor data! %d/%d points have covariance, others will be given default covariances", 
						covAvailableCount, (int)objectPoints.size());
				}
				util3d::solvePnPMsac(
					objectPoints,
					imagePoints,
					K, D,
					objectCovariances,
					rvec, tvec, 
					!guessCameraFrame.isNull(), 
					iterations, 0, minInliers, // 0 reprojection error : covariance aware mode of solvePnPMsac
					0.99, pixelVariance, inliers, flagsPnP, refineIterations, 3.0f, false);
			}
			else
			{
				// Identity covariance for each point, so that the Mahalanobis distance is equal to the Euclidean distance
				// Computationally unoptimal, a dedicated solvePnPMsac without covariance could also be implemented.
				std::vector<cv::Matx33f> objectCovariances(objectPoints.size(), cv::Matx33f::eye());
				util3d::solvePnPMsac(
					objectPoints,
					imagePoints,
					K, D,
					objectCovariances,
					rvec, tvec, 
					!guessCameraFrame.isNull(), 
					iterations, reprojError, minInliers, 
					0.99, pixelVariance, inliers, flagsPnP, refineIterations, 3.0f, false);
			}
			

		}
		else
		{
			util3d::solvePnPRansac(
				objectPoints,
				imagePoints,
				K,
				D,
				rvec,
				tvec,
				true,
				iterations,
				reprojError,
				minInliers, // min inliers
				inliers,
				flagsPnP,
				refineIterations);
		}

		if((int)inliers.size() >= minInliers)
		{
			cv::Rodrigues(rvec, R);
			Transform pnp(R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2), tvec.at<double>(0),
							R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), tvec.at<double>(1),
							R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2), tvec.at<double>(2));

			transform = (cameraModel.localTransform() * pnp).inverse();

			if(covariance)
			{
				*covariance = computePoseCovariance(
					computeFullCovariance, 
					useInlierVariance,
					objectPoints, imagePoints, inliers, matches,
					cameraModel, transform, rvec, tvec,
					covariances3A, covariances3B, words3B,
					splitLinearCovarianceComponents,
					varianceMedianRatio, 
					pixelVariance
				);
			}

			if(covariance->empty())
			{
				UWARN("Rejected PnP transform, degenerate covariance matrix!");
				*covariance = cv::Mat::eye(6,6,CV_64FC1);
				transform.setNull();
			}
			else
			{
				double max_var_lin = uMax3(
					covariance->at<double>(0,0), 
					covariance->at<double>(1,1), 
					covariance->at<double>(2,2)
				);

				double max_var_ang = uMax3(
					covariance->at<double>(3,3), 
					covariance->at<double>(4,4), 
					covariance->at<double>(5,5)
				);
				if(maxLinVariance > 0 && max_var_lin > maxLinVariance)
				{
					UWARN("Rejected PnP transform, linear variance is too high! %f > %f!", max_var_lin, maxLinVariance);
					*covariance = cv::Mat::eye(6,6,CV_64FC1);
					transform.setNull();
				}
				if(maxAngVariance > 0 && max_var_ang > maxAngVariance)
				{
					UWARN("Rejected PnP transform, angular variance is too high! %f > %f!", max_var_ang, maxAngVariance);
					*covariance = cv::Mat::eye(6,6,CV_64FC1);
					transform.setNull();
				}
			}
		}
	}

	if(matchesOut)
	{
		*matchesOut = matches;
	}
	if(inliersOut)
	{
		inliersOut->resize(inliers.size());
		for(unsigned int i=0; i<inliers.size(); ++i)
		{
			inliersOut->at(i) = matches[inliers[i]];
		}
	}

    return transform;
}

// Computes the 6-DoF pose covariance matrix for PnPRansac and PnPMsac
cv::Mat computePoseCovariance(
    bool computeFullCovariance, 
    bool useInlierCovariance,
    const std::vector<cv::Point3f>& objectPoints,
    const std::vector<cv::Point2f>& imagePoints,
    const std::vector<int>& inliers,
    const std::vector<int>& matches,
    const CameraModel& cameraModel,
    const Transform& transform,
    const cv::Mat& rvec,
    const cv::Mat& tvec,
    const std::map<int, cv::Matx33f>& covariances3A,
    const std::map<int, cv::Matx33f>& covariances3B,
    const std::map<int, cv::Point3f>& words3B,
    bool splitLinearCovarianceComponents,
    int varianceMedianRatio,
    float pixelVariance)
{
    cv::Mat covariance = cv::Mat::eye(6, 6, CV_64FC1);
    if(inliers.size() < 4)
    {
        return covariance;
    }

    cv::Mat R_mat;
    cv::Rodrigues(rvec, R_mat);
    cv::Matx33d R_x33(R_mat);
    cv::Mat K = cameraModel.K();
    
    double fx = K.at<double>(0,0);
    double fy = K.at<double>(1,1);

    UDEBUG("Computing pose covariance (computeFullCovariance=%d, useInlierCovariance=%d, inliers=%d)", 
        computeFullCovariance?1:0, useInlierCovariance?1:0, (int)inliers.size());

	if(useInlierCovariance && covariances3A.empty())
	{
		UWARN("Inlier covariance requested but no covariances3A provided! Ignoring inlier covariance.");
		useInlierCovariance = false;
	}

    // Compute the covariance matrix using the Jacobian of the reprojection errors
    if(computeFullCovariance)
 	{
 	 	// Compute full projection Jacobians
 	 	std::vector<cv::Point3f> inlierObjPts(inliers.size());
 	 	for(size_t i = 0; i < inliers.size(); ++i)
 	 	 	inlierObjPts[i] = objectPoints[inliers[i]];

 	 	std::vector<cv::Point2f> reprojectedPts;
 	 	cv::Mat jacobianFull, J_pose;
 	 	cv::projectPoints(inlierObjPts, rvec, tvec, K, cameraModel.D(), reprojectedPts, jacobianFull);
 	 	jacobianFull.colRange(0, 6).convertTo(J_pose, CV_64FC1);

 	 	// Build Hessian matrix H = sum( J_i^T * inv(Sigma_i) * J_i )
 	 	cv::Mat H = cv::Mat::zeros(6, 6, CV_64FC1);

 	 	for(size_t i = 0; i < inliers.size(); ++i)
 	 	{
 	 	 	int idx = inliers[i];
 	 	 	cv::Point3f pt3d = objectPoints[idx];

 	 	 	cv::Matx31d ptB = R_x33 * cv::Matx31d(pt3d.x, pt3d.y, pt3d.z) + cv::Matx31d(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
 	 	 	double Z = std::max(ptB(2,0), 1e-5); 

 	 	 	cv::Matx22d cov2D = cv::Matx22d::eye() * (double)pixelVariance;
 	 	 	cv::Matx23d J_pi(fx/Z, 0.0, -fx*ptB(0,0)/(Z*Z),
 	 	 	 	 	 	 	 0.0, fy/Z, -fy*ptB(1,0)/(Z*Z));

 	 	 	// Incorporate 3D noise (MSAC or RANSAC with 3D covariance)
 	 	 	if(useInlierCovariance)
 	 	 	{
 	 	 	 	UASSERT(covariances3A.find(matches[idx]) != covariances3A.end());
 	 	 	 	cv::Matx33d cov3D(covariances3A.at(matches[idx]));
 	 	 	 	cov2D += J_pi * (R_x33 * cov3D * R_x33.t()) * J_pi.t();
 	 	 	}

 	 	 	cv::Mat J_i = J_pose.rowRange(static_cast<int>(i)*2, static_cast<int>(i)*2 + 2);

 	 	 	if((cov2D(0,0) + cov2D(1,1)) < 1e-12)
 	 	 	{
 	 	 	 	cov2D(0,0) += 1e-6;
 	 	 	 	cov2D(1,1) += 1e-6;
 	 	 	}

 	 	 	cv::Matx22d cov2D_inv = cov2D.inv(cv::DECOMP_SVD);
 	 	 	H += J_i.t() * cv::Mat(cov2D_inv) * J_i;
 	 	}

 	 	// Hessian stabilization / inversion
    
		H = (H + H.t()) / 2.0;

		for(int i = 0; i < 6; ++i) {
			H.at<double>(i, i) += 1e-8; 
		}

		cv::Mat cov;
		// Prefer Cholesky decomposition, fallback to SVD only if Cholesky fails.
		if(cv::invert(H, cov, cv::DECOMP_CHOLESKY) != 0 || cv::invert(H, cov, cv::DECOMP_SVD) > 1e-12)
		{
			cv_custom::poseCovarianceRodriguesToRPY(rvec, cov).copyTo(covariance);
			covariance = (covariance + covariance.t()) / 2.0;
		}
		else
		{
			return cv::Mat()
		}
 	}
	// Original RTABMap strategy: like in PCL computeVariance() method of sac_model.h
    else if(!words3B.empty() || cameraModel.imageSize() != cv::Size())
    {
        std::vector<float> errorSqrdDists(inliers.size());
        std::vector<float> errorSqrdX;
        std::vector<float> errorSqrdY;
        std::vector<float> errorSqrdZ;
        if(splitLinearCovarianceComponents)
        {
            errorSqrdX.resize(inliers.size());
            errorSqrdY.resize(inliers.size());
            errorSqrdZ.resize(inliers.size());
        }
        std::vector<float> errorSqrdAngles(inliers.size());
        Transform localTransformInv = cameraModel.localTransform().inverse();
        Transform transformCameraFrame = transform * cameraModel.localTransform();
        Transform transformCameraFrameInv = transformCameraFrame.inverse();
        for(unsigned int i=0; i<inliers.size(); ++i)
        {
            cv::Point3f objPt = objectPoints[inliers[i]];

            // Get 3D point from cameraB base frame in cameraA base frame
            std::map<int, cv::Point3f>::const_iterator iter = words3B.find(matches[inliers[i]]);
            cv::Point3f newPt;
            if(iter!=words3B.end() && util3d::isFinite(iter->second))
            {
                newPt = util3d::transformPoint(iter->second, transform);
            }
            else
            {

                // Project obj point from base frame of cameraA in cameraB frame (z+ in front of the cameraB)
                cv::Point3f objPtCamBFrame = util3d::transformPoint(objPt, transformCameraFrameInv);

                //compute from projection
                Eigen::Vector3f ray = projectDepthTo3DRay(
                        cameraModel.imageSize(),
                        imagePoints.at(inliers[i]).x,
                        imagePoints.at(inliers[i]).y,
                        cameraModel.cx(),
                        cameraModel.cy(),
                        cameraModel.fx(),
                        cameraModel.fy());
                // transform in camera B frame
                newPt = cv::Point3f(ray.x(), ray.y(), ray.z()) * objPtCamBFrame.z*1.1; // Add 10 % error

                //transform back into cameraA base frame
                newPt = util3d::transformPoint(newPt, transformCameraFrame);
            }

            Eigen::Vector4f v1(objPt.x, objPt.y, objPt.z, 0);
            Eigen::Vector4f v2(newPt.x, newPt.y, newPt.z, 0);
            float angle = pcl::getAngle3D(v1, v2);

            if(useInlierCovariance)
			{
				UASSERT(covariances3A.find(matches[inliers[i]]) != covariances3A.end());

				// Mahalanobis errors, made metric (m^2) using average 3D variance (Trace/3)
				cv::Matx31d diff(objPt.x - newPt.x, objPt.y - newPt.y, objPt.z - newPt.z);
				cv::Matx33d cov3D_total(covariances3A.at(matches[inliers[i]]));
				
				if(!covariances3B.empty()) 
				{
					UASSERT(covariances3B.find(matches[inliers[i]]) != covariances3B.end());
					cv::Matx33d cov3D_B(covariances3B.at(matches[inliers[i]]));
					cov3D_total += R_x33.t() * cov3D_B * R_x33;
				}
				else
				{
					cov3D_total *= 2; // we suppose that the 3d points have the same covariance in both frames
				}

				cv::Matx33d cov3D_inv = cov3D_total.inv(cv::DECOMP_SVD);
				cv::Matx31d v = cov3D_inv * diff; // Sigma^-1 * diff
				double mahal = (diff.t() * v)(0,0);
				
				double trace3D = (cov3D_total(0,0) + cov3D_total(1,1) + cov3D_total(2,2)) / 3.0;
				double errSq_metric = mahal * trace3D;

				errorSqrdDists[i] = (float)errSq_metric;

				if(splitLinearCovarianceComponents)
				{
					errorSqrdX[i] = (float)std::max(0.0, diff(0,0) * v(0,0) * trace3D);
					errorSqrdY[i] = (float)std::max(0.0, diff(1,0) * v(1,0) * trace3D);
					errorSqrdZ[i] = (float)std::max(0.0, diff(2,0) * v(2,0) * trace3D);
				}

                // Angular Mahalanobis error, made metric (rad^2) using isotropic angular variance (trace3D / r1^2)
                cv::Matx31d p1(objPt.x, objPt.y, objPt.z);
                double r1_sq = p1.dot(p1);
                if(r1_sq > 1e-12)
                {
                    cv::Matx31d u1 = p1 * (1.0 / std::sqrt(r1_sq));
                    cv::Matx31d diff_perp = diff - u1 * u1.dot(diff);
                    double perp_sq = diff_perp.dot(diff_perp);
                    
                    cv::Matx31d n = (perp_sq > 1e-12) ? diff_perp * (1.0 / std::sqrt(perp_sq)) : cv::Matx31d(0, 1, 0);
                    double var_trans = (n.t() * cov3D_total * n)(0,0);
                    double var_angle = std::max(1e-12, var_trans / r1_sq);

                    double mahal_angle = (angle * angle) / var_angle;
                    double avg_var_angle = trace3D / r1_sq;
                    errorSqrdAngles[i] = static_cast<float>(mahal_angle * avg_var_angle);
                }
                else
                {
                    errorSqrdAngles[i] = angle * angle;
                }
            }
            else
            {
                // Classical 3D errors
                if(splitLinearCovarianceComponents)
                {
                    double errorX = objPt.x-newPt.x;
                    double errorY = objPt.y-newPt.y;
                    double errorZ = objPt.z-newPt.z;
                    errorSqrdX[i] = errorX * errorX;
                    errorSqrdY[i] = errorY * errorY;
                    errorSqrdZ[i] = errorZ * errorZ;
                }

                errorSqrdDists[i] = uNormSquared(objPt.x-newPt.x, objPt.y-newPt.y, objPt.z-newPt.z);
                errorSqrdAngles[i] = angle * angle;
            }
        }

        std::sort(errorSqrdDists.begin(), errorSqrdDists.end());
        //divide by 4 instead of 2 to ignore very very far features (stereo)
        double median_error_sqr_lin = 2.1981 * (double)errorSqrdDists[errorSqrdDists.size () / varianceMedianRatio];
        UASSERT(uIsFinite(median_error_sqr_lin));
        covariance(cv::Range(0,3), cv::Range(0,3)) *= median_error_sqr_lin + 1e-6;
        std::sort(errorSqrdAngles.begin(), errorSqrdAngles.end());
        double median_error_sqr_ang = 2.1981 * (double)errorSqrdAngles[errorSqrdAngles.size () / varianceMedianRatio];
        UASSERT(uIsFinite(median_error_sqr_ang));
        covariance(cv::Range(3,6), cv::Range(3,6)) *= median_error_sqr_ang + 1e-6;

        if(splitLinearCovarianceComponents)
        {
            std::sort(errorSqrdX.begin(), errorSqrdX.end());
            double median_error_sqr_x = 2.1981 * (double)errorSqrdX[errorSqrdX.size () / varianceMedianRatio];
            std::sort(errorSqrdY.begin(), errorSqrdY.end());
            double median_error_sqr_y = 2.1981 * (double)errorSqrdY[errorSqrdY.size () / varianceMedianRatio];
            std::sort(errorSqrdZ.begin(), errorSqrdZ.end());
            double median_error_sqr_z = 2.1981 * (double)errorSqrdZ[errorSqrdZ.size () / varianceMedianRatio];
            
            UASSERT(uIsFinite(median_error_sqr_x));
            UASSERT(uIsFinite(median_error_sqr_y));
            UASSERT(uIsFinite(median_error_sqr_z));
            covariance.at<double>(0,0) = median_error_sqr_x + 1e-6;
            covariance.at<double>(1,1) = median_error_sqr_y + 1e-6;
            covariance.at<double>(2,2) = median_error_sqr_z + 1e-6;
        }
    }
    else
    {
        // compute variance, which is the rms of reprojection errors
        std::vector<cv::Point2f> imagePointsReproj;
        cv::projectPoints(objectPoints, rvec, tvec, K, cv::Mat(), imagePointsReproj);
        float err = 0.0f;

        for(unsigned int i=0; i<inliers.size(); ++i)
        {
            int idx = inliers[i];
            double ex = static_cast<double>(imagePoints.at(idx).x) - static_cast<double>(imagePointsReproj.at(idx).x);
            double ey = static_cast<double>(imagePoints.at(idx).y) - static_cast<double>(imagePointsReproj.at(idx).y);
            
            // Depth of A projected in frame B using the transform (rvec/tvec)
            cv::Matx31d pt3d_mat(objectPoints[idx].x, objectPoints[idx].y, objectPoints[idx].z);
            cv::Matx31d ptB = R_x33 * pt3d_mat + cv::Matx31d(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
            double Z = std::max(ptB(2,0), 1e-5);
            
			bool originalRTABMapStrategy = false;
            if(!originalRTABMapStrategy && useInlierCovariance && !covariances3A.empty())
            {
				UASSERT(covariances3A.find(matches[idx]) != covariances3A.end());
                cv::Matx33d cov3D(covariances3A.at(matches[idx]));
                cv::Matx23d J_pi(fx/Z, 0.0, -fx*ptB(0,0)/(Z*Z),
                                 0.0, fy/Z, -fy*ptB(1,0)/(Z*Z));
                
                cv::Matx22d cov2D = J_pi * (R_x33 * cov3D * R_x33.t()) * J_pi.t();
                
                if((cov2D(0,0) + cov2D(1,1)) < 1e-12)
                {
                    cov2D(0,0) += 1e-6;
                    cov2D(1,1) += 1e-6;
                }

                cv::Matx22d cov2D_inv = cov2D.inv(cv::DECOMP_SVD);
                
                cv::Matx21d diff(ex, ey);
                double mahal = (diff.t() * cov2D_inv * diff)(0,0);
                
                double trace3D = (cov3D(0,0) + cov3D(1,1) + cov3D(2,2)) / 3.0;
                err += (float)(mahal * trace3D);
            }
            else if(!originalRTABMapStrategy)
            {
                // Metric 2D error fallback
                double metricEx = ex * Z / fx;
                double metricEy = ey * Z / fy;
                err += (float)(metricEx * metricEx + metricEy * metricEy);
            }
			else // original RTABMap calculation: covariance is in pixel space...
			{
				err += (float)(ex * ex + ey * ey);
			}
        }
        
        UASSERT(uIsFinite(err));
        covariance *= std::sqrt(err/float(inliers.size())) + 1e-6;
    }
    
    return covariance;
}

Transform estimateMotion3DTo2D(
	const std::map<int, cv::Point3f> & words3A,
	const std::map<int, cv::KeyPoint> & words2B,
	const std::vector<CameraModel> & cameraModels,
	unsigned int samplingPolicy,
	int minInliers,
	int iterations,
	double reprojError,
	int flagsPnP,
	int refineIterations,
	int varianceMedianRatio,
	float maxLinVariance,
	float maxAngVariance,
	const Transform & guess,
	const std::map<int, cv::Point3f> & words3B,
	cv::Mat * covariance,
	std::vector<int> * matchesOut,
	std::vector<int> * inliersOut,
	bool splitLinearCovarianceComponents)
{
	std::vector<std::vector<int> > matchesPerCamera;
	std::vector<std::vector<int> > inliersPerCamera;
	Transform t = estimateMotion3DTo2D(
		words3A,
		words2B,
		cameraModels,
		samplingPolicy,
		minInliers,
		iterations,
		reprojError,
		flagsPnP,
		refineIterations,
		varianceMedianRatio,
		maxLinVariance,
		maxAngVariance,
		guess,
		words3B,
		covariance,
		matchesOut?&matchesPerCamera:0,
		inliersOut?&inliersPerCamera:0,
		splitLinearCovarianceComponents);
	if(matchesOut)
	{
		for(size_t i=0; i<matchesPerCamera.size(); ++i)
		{
			matchesOut->insert(matchesOut->end(), matchesPerCamera[i].begin(), matchesPerCamera[i].end());
		}
	}
	if(inliersOut)
	{
		for(size_t i=0; i<inliersPerCamera.size(); ++i)
		{
			inliersOut->insert(inliersOut->end(), inliersPerCamera[i].begin(), inliersPerCamera[i].end());
		}
	}
	return t;
}

Transform estimateMotion3DTo2D(
			const std::map<int, cv::Point3f> & words3A,
			const std::map<int, cv::KeyPoint> & words2B,
			const std::vector<CameraModel> & cameraModels,
			unsigned int samplingPolicy,
			int minInliers,
			int iterations,
			double reprojError,
			int flagsPnP,
			int refineIterations,
			int varianceMedianRatio,
			float maxLinVariance,
			float maxAngVariance,
			const Transform & guess,
			const std::map<int, cv::Point3f> & words3B,
			cv::Mat * covariance,
			std::vector<std::vector<int> > * matchesOut,
			std::vector<std::vector<int> > * inliersOut,
			bool splitLinearCovarianceComponents)
{
	Transform transform;
#ifndef RTABMAP_OPENGV
	UERROR("This function is only available if rtabmap is built with OpenGV dependency.");
#else
	UASSERT(!cameraModels.empty() && cameraModels[0].imageWidth() > 0);
	int subImageWidth = cameraModels[0].imageWidth();
	for(size_t i=0; i<cameraModels.size(); ++i)
	{
		UASSERT(cameraModels[i].isValidForProjection());
		UASSERT(subImageWidth  == cameraModels[i].imageWidth());
	}

	UASSERT(!guess.isNull());
	UASSERT(varianceMedianRatio > 1);

	std::vector<int> matches, inliers;

	if(covariance)
	{
		*covariance = cv::Mat::eye(6,6,CV_64FC1);
	}

	// find correspondences
	std::vector<int> ids = uKeys(words2B);
	std::vector<cv::Point3f> objectPoints(ids.size());
	std::vector<cv::Point2f> imagePoints(ids.size());
	int oi=0;
	matches.resize(ids.size());
	std::vector<int> cameraIndexes(ids.size());
	for(unsigned int i=0; i<ids.size(); ++i)
	{
		std::map<int, cv::Point3f>::const_iterator iter=words3A.find(ids[i]);
		if(iter != words3A.end() && util3d::isFinite(iter->second))
		{
			const cv::Point2f & kpt = words2B.find(ids[i])->second.pt;
			int cameraIndex = int(kpt.x / subImageWidth);
			UASSERT_MSG(cameraIndex >= 0 && cameraIndex < (int)cameraModels.size(),
					uFormat("cameraIndex=%d, models=%d, kpt.x=%f, subImageWidth=%f (Camera model image width=%d)",
							cameraIndex, (int)cameraModels.size(), kpt.x, (double)subImageWidth, cameraModels[cameraIndex].imageWidth()).c_str());

			const cv::Point3f & pt = iter->second;
			objectPoints[oi] = pt;
			imagePoints[oi] = kpt;
			// convert in image space
			imagePoints[oi].x = imagePoints[oi].x - (cameraIndex*subImageWidth);
			cameraIndexes[oi] = cameraIndex;
			matches[oi++] = ids[i];
		}
	}

	objectPoints.resize(oi);
	imagePoints.resize(oi);
	cameraIndexes.resize(oi);
	matches.resize(oi);

	UDEBUG("words3A=%d words2B=%d matches=%d words3B=%d guess=%s reprojError=%f iterations=%d samplingPolicy=%ld",
			(int)words3A.size(), (int)words2B.size(), (int)matches.size(), (int)words3B.size(),
			guess.prettyPrint().c_str(), reprojError, iterations, (long)samplingPolicy);

	if((int)matches.size() >= minInliers)
	{
		if(samplingPolicy == 0 || samplingPolicy == 2)
		{
			std::vector<int> cc;
			cc.resize(cameraModels.size());
			std::fill(cc.begin(), cc.end(),0);
			for(size_t i=0; i<cameraIndexes.size(); ++i)
			{
				cc[cameraIndexes[i]] = cc[cameraIndexes[i]] + 1;
			}

			for (size_t i=0; i<cameraModels.size(); ++i)
			{
				UDEBUG("Matches in Camera %d: %d", (int)i, cc[i]);
				// opengv multi ransac needs at least 2 matches/camera
				if (cc[i] < 2)
				{
					if(samplingPolicy==2) {
						UERROR("Not enough matches in camera %ld to do "
							  "homogenoeus random sampling, returning null "
							  "transform. Consider using AUTO sampling "
							  "policy to fallback to ANY policy.", i);
						return Transform();
					}
					else { // samplingPolicy==0
						samplingPolicy = 1;
						UWARN("Not enough matches in camera %ld to do "
							  "homogenoeus random sampling, falling back to ANY policy.", i);
						break;
					}
				}
			}
		}

		if(samplingPolicy == 0)
		{
			samplingPolicy = 2;
		}

		// convert cameras
		opengv::translations_t camOffsets;
		opengv::rotations_t camRotations;
		for(size_t i=0; i<cameraModels.size(); ++i)
		{
			camOffsets.push_back(opengv::translation_t(
					cameraModels[i].localTransform().x(),
					cameraModels[i].localTransform().y(),
					cameraModels[i].localTransform().z()));
			camRotations.push_back(cameraModels[i].localTransform().toEigen4d().block<3,3>(0, 0));
		}

		Transform pnp;
		if(samplingPolicy == 2) // Homogenoeus random sampling
		{
			// convert 3d points
			std::vector<std::shared_ptr<opengv::points_t>> multiPoints;
			multiPoints.resize(cameraModels.size());
			// convert 2d-3d correspondences into bearing vectors
			std::vector<std::shared_ptr<opengv::bearingVectors_t>> multiBearingVectors;
			multiBearingVectors.resize(cameraModels.size());
			std::vector<std::vector<int> > localIndexToGlobalIndex(cameraModels.size());
			for(size_t i=0; i<cameraModels.size();++i)
			{
				multiPoints[i] = std::make_shared<opengv::points_t>();
				multiBearingVectors[i] = std::make_shared<opengv::bearingVectors_t>();
			}

			for(size_t i=0; i<objectPoints.size(); ++i)
			{
				int cameraIndex = cameraIndexes[i];
				multiPoints[cameraIndex]->push_back(opengv::point_t(objectPoints[i].x,objectPoints[i].y,objectPoints[i].z));
				cv::Vec3f pt;
				cameraModels[cameraIndex].project(imagePoints[i].x, imagePoints[i].y, 1, pt[0], pt[1], pt[2]);
				pt = cv::normalize(pt);
				multiBearingVectors[cameraIndex]->push_back(opengv::bearingVector_t(pt[0], pt[1], pt[2]));
				localIndexToGlobalIndex[cameraIndex].push_back(i);
			}

			//create a non-central absolute multi adapter
			opengv::absolute_pose::NoncentralAbsoluteMultiAdapter adapter(
					multiBearingVectors,
					multiPoints,
					camOffsets,
					camRotations );

			adapter.setR(guess.toEigen4d().block<3,3>(0, 0));
			adapter.sett(opengv::translation_t(guess.x(), guess.y(), guess.z()));

			//Create a MultiNoncentralAbsolutePoseSacProblem and MultiRansac
			//The method is set to GP3P
			opengv::sac::MultiRansac<opengv::sac_problems::absolute_pose::MultiNoncentralAbsolutePoseSacProblem> ransac;
			std::shared_ptr<opengv::sac_problems::absolute_pose::MultiNoncentralAbsolutePoseSacProblem> absposeproblem_ptr(
					new opengv::sac_problems::absolute_pose::MultiNoncentralAbsolutePoseSacProblem(adapter));

			// Opt-in deterministic RANSAC: OpenGV's default constructor seeds
			// its internal mt19937 from the system clock, so without this
			// override two calls with identical inputs can produce different
			// inlier sets / covariances. Tests flip the toggle via
			// setRansacDeterministicSeed(true).
			if(g_ransacDeterministicSeed)
			{
				absposeproblem_ptr->rng_alg_.seed(12345u);
				absposeproblem_ptr->rng_gen_.reset(new std::function<int()>(
						std::bind(*absposeproblem_ptr->rng_dist_, absposeproblem_ptr->rng_alg_)));
			}

			ransac.sac_model_ = absposeproblem_ptr;
			ransac.threshold_ = 1.0 - cos(atan(reprojError/cameraModels[0].fx()));
			ransac.max_iterations_ = iterations;
			UDEBUG("Ransac params: threshold = %f (reprojError=%f fx=%f), max iterations=%d", ransac.threshold_, reprojError, cameraModels[0].fx(), ransac.max_iterations_);

			//Run the experiment
			ransac.computeModel();

			pnp = Transform::fromEigen3d(ransac.model_coefficients_);

			UDEBUG("Ransac result: %s", pnp.prettyPrint().c_str());
			UDEBUG("Ransac iterations done: %d", ransac.iterations_);
			for (size_t i=0; i < cameraModels.size(); ++i) {
				for (size_t j=0; j < ransac.inliers_[i].size(); ++j) {
					inliers.push_back(localIndexToGlobalIndex[i][ransac.inliers_[i][j]]);
				}
			}
		}
		else
		{
			// convert 3d points
			opengv::points_t points;

			// convert 2d-3d correspondences into bearing vectors
			opengv::bearingVectors_t bearingVectors;
			opengv::absolute_pose::NoncentralAbsoluteAdapter::camCorrespondences_t camCorrespondences;

			for(size_t i=0; i<objectPoints.size(); ++i)
			{
				int cameraIndex = cameraIndexes[i];
				points.push_back(opengv::point_t(objectPoints[i].x,objectPoints[i].y,objectPoints[i].z));
				cv::Vec3f pt;
				cameraModels[cameraIndex].project(imagePoints[i].x, imagePoints[i].y, 1, pt[0], pt[1], pt[2]);
				pt = cv::normalize(pt);
				bearingVectors.push_back(opengv::bearingVector_t(pt[0], pt[1], pt[2]));
				camCorrespondences.push_back(cameraIndex);
			}

			//create a non-central absolute adapter
			opengv::absolute_pose::NoncentralAbsoluteAdapter adapter(
					bearingVectors,
					camCorrespondences,
					points,
					camOffsets,
					camRotations );

			adapter.setR(guess.toEigen4d().block<3,3>(0, 0));
			adapter.sett(opengv::translation_t(guess.x(), guess.y(), guess.z()));

			//Create a AbsolutePoseSacProblem and Ransac
			//The method is set to GP3P
			opengv::sac::Ransac<opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem> ransac;
			std::shared_ptr<opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem> absposeproblem_ptr(
					new opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem(adapter, opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem::GP3P));

			// Opt-in deterministic RANSAC (see comment on MultiRansac above).
			if(g_ransacDeterministicSeed)
			{
				absposeproblem_ptr->rng_alg_.seed(12345u);
				absposeproblem_ptr->rng_gen_.reset(new std::function<int()>(
						std::bind(*absposeproblem_ptr->rng_dist_, absposeproblem_ptr->rng_alg_)));
			}

			ransac.sac_model_ = absposeproblem_ptr;
			ransac.threshold_ = 1.0 - cos(atan(reprojError/cameraModels[0].fx()));
			ransac.max_iterations_ = iterations;
			UDEBUG("Ransac params: threshold = %f (reprojError=%f fx=%f), max iterations=%d", ransac.threshold_, reprojError, cameraModels[0].fx(), ransac.max_iterations_);

			//Run the experiment
			ransac.computeModel();

			pnp = Transform::fromEigen3d(ransac.model_coefficients_);

			UDEBUG("Ransac result: %s", pnp.prettyPrint().c_str());
			UDEBUG("Ransac iterations done: %d", ransac.iterations_);
			inliers = ransac.inliers_;
		}

		UDEBUG("Ransac inliers: %ld", inliers.size());

		if((int)inliers.size() >= minInliers && !pnp.isNull())
		{
			transform = pnp;

			// compute variance (like in PCL computeVariance() method of sac_model.h)
			if(covariance)
			{
				std::vector<float> errorSqrdX;
				std::vector<float> errorSqrdY;
				std::vector<float> errorSqrdZ;
				if(splitLinearCovarianceComponents)
				{
					errorSqrdX.resize(inliers.size());
					errorSqrdY.resize(inliers.size());
					errorSqrdZ.resize(inliers.size());
				}
				std::vector<float> errorSqrdDists(inliers.size());
				std::vector<float> errorSqrdAngles(inliers.size());

				std::vector<Transform> transformsCameraFrame(cameraModels.size());
				std::vector<Transform> transformsCameraFrameInv(cameraModels.size());
				for(size_t i=0; i<cameraModels.size(); ++i)
				{
					transformsCameraFrame[i] = transform * cameraModels[i].localTransform();
					transformsCameraFrameInv[i] = transformsCameraFrame[i].inverse();
				}

				for(unsigned int i=0; i<inliers.size(); ++i)
				{
					cv::Point3f objPt = objectPoints[inliers[i]];

					// Get 3D point from cameraB base frame in cameraA base frame
					std::map<int, cv::Point3f>::const_iterator iter = words3B.find(matches[inliers[i]]);
					cv::Point3f newPt;
					if(iter!=words3B.end() && util3d::isFinite(iter->second))
					{
						newPt = util3d::transformPoint(iter->second, transform);
					}
					else
					{
						int cameraIndex = cameraIndexes[inliers[i]];

						// Project obj point from base frame of cameraA in cameraB frame (z+ in front of the cameraB)
						cv::Point3f objPtCamBFrame = util3d::transformPoint(objPt, transformsCameraFrameInv[cameraIndex]);

						//compute from projection
						Eigen::Vector3f ray = projectDepthTo3DRay(
								cameraModels[cameraIndex].imageSize(),
								imagePoints.at(inliers[i]).x,
								imagePoints.at(inliers[i]).y,
								cameraModels[cameraIndex].cx(),
								cameraModels[cameraIndex].cy(),
								cameraModels[cameraIndex].fx(),
								cameraModels[cameraIndex].fy());
						// transform in camera B frame
						newPt = cv::Point3f(ray.x(), ray.y(), ray.z()) * objPtCamBFrame.z*1.1; // Add 10 % error

						//transfor back into cameraA base frame
						newPt = util3d::transformPoint(newPt, transformsCameraFrame[cameraIndex]);
					}

					if(splitLinearCovarianceComponents)
					{
						double errorX = objPt.x-newPt.x;
						double errorY = objPt.y-newPt.y;
						double errorZ = objPt.z-newPt.z;
						errorSqrdX[i] = errorX * errorX;
						errorSqrdY[i] = errorY * errorY;
						errorSqrdZ[i] = errorZ * errorZ;
					}

					errorSqrdDists[i] = uNormSquared(objPt.x-newPt.x, objPt.y-newPt.y, objPt.z-newPt.z);

					Eigen::Vector4f v1(objPt.x, objPt.y, objPt.z, 0);
					Eigen::Vector4f v2(newPt.x, newPt.y, newPt.z, 0);
					errorSqrdAngles[i] = pcl::getAngle3D(v1, v2);
				}

				std::sort(errorSqrdDists.begin(), errorSqrdDists.end());
				//divide by 4 instead of 2 to ignore very very far features (stereo)
				double median_error_sqr_lin = 2.1981 * (double)errorSqrdDists[errorSqrdDists.size () / varianceMedianRatio];
				UASSERT(uIsFinite(median_error_sqr_lin));
				(*covariance)(cv::Range(0,3), cv::Range(0,3)) *= median_error_sqr_lin;
				std::sort(errorSqrdAngles.begin(), errorSqrdAngles.end());
				double median_error_sqr_ang = 2.1981 * (double)errorSqrdAngles[errorSqrdAngles.size () / varianceMedianRatio];
				UASSERT(uIsFinite(median_error_sqr_ang));
				(*covariance)(cv::Range(3,6), cv::Range(3,6)) *= median_error_sqr_ang;

				if(splitLinearCovarianceComponents)
				{
					std::sort(errorSqrdX.begin(), errorSqrdX.end());
					double median_error_sqr_x = 2.1981 * (double)errorSqrdX[errorSqrdX.size () / varianceMedianRatio];
					std::sort(errorSqrdY.begin(), errorSqrdY.end());
					double median_error_sqr_y = 2.1981 * (double)errorSqrdY[errorSqrdY.size () / varianceMedianRatio];
					std::sort(errorSqrdZ.begin(), errorSqrdZ.end());
					double median_error_sqr_z = 2.1981 * (double)errorSqrdZ[errorSqrdZ.size () / varianceMedianRatio];
					
					
					UASSERT(uIsFinite(median_error_sqr_x));
					UASSERT(uIsFinite(median_error_sqr_y));
					UASSERT(uIsFinite(median_error_sqr_z));
					covariance->at<double>(0,0) = median_error_sqr_x;
					covariance->at<double>(1,1) = median_error_sqr_y;
					covariance->at<double>(2,2) = median_error_sqr_z;

					median_error_sqr_lin = uMax3(median_error_sqr_x, median_error_sqr_y, median_error_sqr_z);
				}

				if(maxLinVariance > 0 && median_error_sqr_lin > maxLinVariance)
				{
					UWARN("Rejected PnP transform, linear variance is too high! %f > %f!", median_error_sqr_lin, maxLinVariance);
					*covariance = cv::Mat::eye(6,6,CV_64FC1);
					transform.setNull();
				}
				if(maxAngVariance > 0 && median_error_sqr_ang > maxAngVariance)
				{
					UWARN("Rejected PnP transform, angular variance is too high! %f > %f!", median_error_sqr_ang, maxAngVariance);
					*covariance = cv::Mat::eye(6,6,CV_64FC1);
					transform.setNull();
				}
			}
		}
	}

	if(matchesOut)
	{
		matchesOut->clear();
		matchesOut->resize(cameraModels.size());
		UASSERT(matches.size() == cameraIndexes.size());
		for(size_t i=0; i<matches.size(); ++i)
		{
			UASSERT(cameraIndexes[i]>=0 && cameraIndexes[i] < (int)cameraModels.size());
			matchesOut->at(cameraIndexes[i]).push_back(matches[i]);
		}
	}
	if(inliersOut)
	{
		inliersOut->clear();
		inliersOut->resize(cameraModels.size());
		for(unsigned int i=0; i<inliers.size(); ++i)
		{
			UASSERT(inliers[i]>=0 && inliers[i] < (int)cameraIndexes.size());
			UASSERT(cameraIndexes[inliers[i]]>=0 && cameraIndexes[inliers[i]] < (int)cameraModels.size());
			inliersOut->at(cameraIndexes[inliers[i]]).push_back(matches[inliers[i]]);
		}
	}
#endif
	return transform;
}

Transform estimateMotion3DTo3D(
			const std::map<int, cv::Point3f> & words3A,
			const std::map<int, cv::Point3f> & words3B,
			int minInliers,
			double inliersDistance,
			int iterations,
			int refineIterations,
			cv::Mat * covariance,
			std::vector<int> * matchesOut,
			std::vector<int> * inliersOut)
{
	Transform transform;
	std::vector<cv::Point3f> inliers1; // previous
	std::vector<cv::Point3f> inliers2; // new

	std::vector<int> matches;
	util3d::findCorrespondences(
			words3A,
			words3B,
			inliers1,
			inliers2,
			0,
			&matches);
	UASSERT(inliers1.size() == inliers2.size());
	UDEBUG("Unique correspondences = %d", (int)inliers1.size());

	if(covariance)
	{
		*covariance = cv::Mat::eye(6,6,CV_64FC1);
	}

	std::vector<int> inliers;
	if((int)inliers1.size() >= minInliers)
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr inliers1cloud(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr inliers2cloud(new pcl::PointCloud<pcl::PointXYZ>);
		inliers1cloud->resize(inliers1.size());
		inliers2cloud->resize(inliers1.size());
		for(unsigned int i=0; i<inliers1.size(); ++i)
		{
			(*inliers1cloud)[i].x = inliers1[i].x;
			(*inliers1cloud)[i].y = inliers1[i].y;
			(*inliers1cloud)[i].z = inliers1[i].z;
			(*inliers2cloud)[i].x = inliers2[i].x;
			(*inliers2cloud)[i].y = inliers2[i].y;
			(*inliers2cloud)[i].z = inliers2[i].z;
		}
		Transform t = util3d::transformFromXYZCorrespondences(
				inliers2cloud,
				inliers1cloud,
				inliersDistance,
				iterations,
				refineIterations,
				3.0,
				&inliers,
				covariance);

		if(!t.isNull() && (int)inliers.size() >= minInliers)
		{
			transform = t;
		}
	}

	if(matchesOut)
	{
		*matchesOut = matches;
	}
	if(inliersOut)
	{
		inliersOut->resize(inliers.size());
		for(unsigned int i=0; i<inliers.size(); ++i)
		{
			inliersOut->at(i) = matches[inliers[i]];
		}
	}

	return transform;
}


std::vector<float> computeReprojErrors(
		std::vector<cv::Point3f> opoints,
		std::vector<cv::Point2f> ipoints,
		const cv::Mat & cameraMatrix,
		const cv::Mat & distCoeffs,
		const cv::Mat & rvec,
		const cv::Mat & tvec,
		float reprojErrorThreshold,
		std::vector<int> & inliers)
{
	UASSERT(opoints.size() == ipoints.size());
	int count = (int)opoints.size();

	std::vector<cv::Point2f> projpoints;
	projectPoints(opoints, rvec, tvec, cameraMatrix, distCoeffs, projpoints);

	inliers.resize(count,0);
	std::vector<float> err(count);
	int oi=0;
	for (int i = 0; i < count; ++i)
	{
		float e = (float)cv::norm( ipoints[i] - projpoints[i]);
		if(e <= reprojErrorThreshold)
		{
			inliers[oi] = i;
			err[oi++] = e;
		}
	}
	inliers.resize(oi);
	err.resize(oi);
	return err;
}

void solvePnPRansac(
		const std::vector<cv::Point3f> & objectPoints,
		const std::vector<cv::Point2f> & imagePoints,
		const cv::Mat & cameraMatrix,
		const cv::Mat & distCoeffs,
		cv::Mat & rvec,
		cv::Mat & tvec,
		bool useExtrinsicGuess,
        int iterationsCount,
        float reprojectionError,
        int minInliersCount,
        std::vector<int> & inliers,
        int flags,
        int refineIterations,
        float refineSigma)
{
	if(minInliersCount < 4)
	{
		minInliersCount = 4;
	}

	// Use OpenCV3 version of solvePnPRansac in OpenCV2.
	// FIXME: we should use this version of solvePnPRansac in newer 3.3.1 too, which seems a lot less stable!?!? Why!?
	cv3::solvePnPRansac(
			objectPoints,
			imagePoints,
			cameraMatrix,
			distCoeffs,
			rvec,
			tvec,
			useExtrinsicGuess,
			iterationsCount,
			reprojectionError,
			0.99, // confidence
			inliers,
			flags);

	float inlierThreshold = reprojectionError;
	if((int)inliers.size() >= minInliersCount && refineIterations>0)
	{
		float error_threshold = inlierThreshold;
		int refine_iterations = 0;
		bool inlier_changed = false, oscillating = false;
		std::vector<int> new_inliers, prev_inliers = inliers;
		std::vector<int> final_inliers = inliers;
		std::vector<size_t> inliers_sizes;
		//Eigen::VectorXf new_model_coefficients = model_coefficients;
		cv::Mat new_model_rvec = rvec;
		cv::Mat new_model_tvec = tvec;

		do
		{
			// Get inliers from the current model
			std::vector<cv::Point3f> opoints_inliers(prev_inliers.size());
			std::vector<cv::Point2f> ipoints_inliers(prev_inliers.size());
			for(unsigned int i=0; i<prev_inliers.size(); ++i)
			{
				opoints_inliers[i] = objectPoints[prev_inliers[i]];
				ipoints_inliers[i] = imagePoints[prev_inliers[i]];
			}

			UDEBUG("inliers=%d refine_iterations=%d, rvec=%f,%f,%f tvec=%f,%f,%f", (int)prev_inliers.size(), refine_iterations,
					*new_model_rvec.ptr<double>(0), *new_model_rvec.ptr<double>(1), *new_model_rvec.ptr<double>(2),
					*new_model_tvec.ptr<double>(0), *new_model_tvec.ptr<double>(1), *new_model_tvec.ptr<double>(2));

			// Optimize the model coefficients
			cv::solvePnP(opoints_inliers, ipoints_inliers, cameraMatrix, distCoeffs, new_model_rvec, new_model_tvec, true, flags);
			inliers_sizes.push_back(prev_inliers.size());

			UDEBUG("rvec=%f,%f,%f tvec=%f,%f,%f",
					*new_model_rvec.ptr<double>(0), *new_model_rvec.ptr<double>(1), *new_model_rvec.ptr<double>(2),
					*new_model_tvec.ptr<double>(0), *new_model_tvec.ptr<double>(1), *new_model_tvec.ptr<double>(2));

			// Select the new inliers based on the optimized coefficients and new threshold
			std::vector<float> err = computeReprojErrors(objectPoints, imagePoints, cameraMatrix, distCoeffs, new_model_rvec, new_model_tvec, error_threshold, new_inliers);
			UDEBUG("RANSAC refineModel: Number of inliers found (before/after): %d/%d, with an error threshold of %f.",
					(int)prev_inliers.size (), (int)new_inliers.size (), error_threshold);

			final_inliers = new_inliers;

			if ((int)new_inliers.size() < minInliersCount)
			{
				++refine_iterations;
				if (refine_iterations >= refineIterations)
				{
					break;
				}
				continue;
			}

			// Estimate the variance and the new threshold
			float m = uMean(err.data(), err.size());
			float variance = uVariance(err.data(), err.size());
			error_threshold = std::min(inlierThreshold, refineSigma * float(sqrt(variance)));

			UDEBUG ("RANSAC refineModel: New estimated error threshold: %f (variance=%f mean=%f) on iteration %d out of %d.",
				  error_threshold, variance, m, refine_iterations, refineIterations);
			inlier_changed = false;
			std::swap (prev_inliers, new_inliers);

			// If the number of inliers changed, then we are still optimizing
			if (new_inliers.size () != prev_inliers.size ())
			{
				// Check if the number of inliers is oscillating in between two values
				if ((int)inliers_sizes.size () >= minInliersCount)
				{
					if (inliers_sizes[inliers_sizes.size () - 1] == inliers_sizes[inliers_sizes.size () - 3] &&
					inliers_sizes[inliers_sizes.size () - 2] == inliers_sizes[inliers_sizes.size () - 4])
					{
						oscillating = true;
						break;
					}
				}
				inlier_changed = true;
				continue;
			}

			// Check the values of the inlier set
			for (size_t i = 0; i < prev_inliers.size (); ++i)
			{
				// If the value of the inliers changed, then we are still optimizing
				if (prev_inliers[i] != new_inliers[i])
				{
					inlier_changed = true;
					break;
				}
			}
		}
		while (inlier_changed && ++refine_iterations < refineIterations);

		// If the new set of inliers is empty, we didn't do a good job refining
		if ((int)prev_inliers.size() < minInliersCount)
		{
			UWARN ("RANSAC refineModel: Refinement failed: got very low inliers (%d)!", (int)prev_inliers.size());
		}

		if (oscillating)
		{
			UDEBUG("RANSAC refineModel: Detected oscillations in the model refinement.");
		}

		std::swap(inliers, final_inliers);
		rvec = new_model_rvec;
		tvec = new_model_tvec;
	}

}

void solvePnPMsac(const std::vector<cv::Point3f> & objectPoints,
                  const std::vector<cv::Point2f> & imagePoints,
                  const cv::Mat & cameraMatrix,
                  const cv::Mat & distCoeffs,
                  const std::vector<cv::Matx33f> & covariances3A,
                  cv::Mat & rvec, cv::Mat & tvec,
                  bool useExtrinsicGuess, int iterationsCount,
                  float reprojectionError, int minInliersCount,
                  float pixelVariance, std::vector<int> & inliers, 
				  int flags, int refineIterations, float refineSigma,
                  bool use_prosac_ordering)
{
    if(minInliersCount < 4)
    {
        minInliersCount = 4;
    }

    UDEBUG("MSAC input points=%d useExtrinsicGuess=%d iterations=%d minInliers=%d flags=%d refineIterations=%d refineSigma=%f",
           (int)objectPoints.size(), useExtrinsicGuess, iterationsCount, minInliersCount, flags, refineIterations, refineSigma);

    // 1. Call the custom OpenCV-style function
	const float chi2_95_unsquared = 2.44765f;
	float inlierThreshold = reprojectionError ? reprojectionError : chi2_95_unsquared;

    cv_custom::solvePnPMsac(
            objectPoints, imagePoints, cameraMatrix, distCoeffs, covariances3A,
            rvec, tvec, useExtrinsicGuess, iterationsCount, inlierThreshold, 
			0.99, pixelVariance, use_prosac_ordering, inliers, flags);


    // 2. Exact mimic of the Refinement loop
    if((int)inliers.size() >= minInliersCount && refineIterations > 0)
    {
        float error_threshold = inlierThreshold;
        int refine_iterations = 0;
        bool inlier_changed = false, oscillating = false;
		std::vector<int> final_inliers = inliers;
        std::vector<int> new_inliers, prev_inliers = inliers;
        std::vector<size_t> inliers_sizes;
        
        cv::Mat new_model_rvec = rvec.clone();
        cv::Mat new_model_tvec = tvec.clone();

        do
        {
            // Get inliers from the current model
            std::vector<cv::Point3f> opoints_inliers(prev_inliers.size());
            std::vector<cv::Point2f> ipoints_inliers(prev_inliers.size());
            std::vector<cv::Matx33f> cov_inliers(prev_inliers.size());
            for(unsigned int i = 0; i < prev_inliers.size(); ++i)
            {
                opoints_inliers[i] = objectPoints[prev_inliers[i]];
                ipoints_inliers[i] = imagePoints[prev_inliers[i]];
                cov_inliers[i] = covariances3A[prev_inliers[i]];
            }

            UDEBUG("inliers=%d refine_iterations=%d, rvec=%f,%f,%f tvec=%f,%f,%f", (int)prev_inliers.size(), refine_iterations,
                   *new_model_rvec.ptr<double>(0), *new_model_rvec.ptr<double>(1), *new_model_rvec.ptr<double>(2),
                   *new_model_tvec.ptr<double>(0), *new_model_tvec.ptr<double>(1), *new_model_tvec.ptr<double>(2));

            cv_custom::solvePnPMsacRefineLM(
					new_model_rvec, new_model_tvec,
					opoints_inliers,
					ipoints_inliers,
					cov_inliers,
					pixelVariance,
					cameraMatrix,
					distCoeffs
			);
			inliers_sizes.push_back(prev_inliers.size());

            // Select the new inliers based on the optimized coefficients and new threshold
            std::vector<float> err = cv_custom::computeMahalanobisReprojErrors(
                objectPoints, imagePoints, cameraMatrix, distCoeffs, 
                new_model_rvec, new_model_tvec, covariances3A, pixelVariance, error_threshold, new_inliers
            );

            UDEBUG("MSAC refineModel: Number of inliers found (before/after): %d/%d, with an error threshold of %f.",
                   (int)prev_inliers.size (), (int)new_inliers.size (), error_threshold);

			final_inliers = new_inliers;

            if ((int)new_inliers.size() < minInliersCount)
            {
                ++refine_iterations;
                if (refine_iterations >= refineIterations)
                {
                    break;
                }
                continue;
            }

            // Estimate the variance and the new threshold
            float m = uMean(err.data(), err.size());
            float variance = uVariance(err.data(), err.size());
            error_threshold = std::min(inlierThreshold, refineSigma * float(sqrt(variance)));

            UDEBUG ("MSAC refineModel: New estimated error threshold: %f (variance=%f mean=%f) on iteration %d out of %d.",
                  error_threshold, variance, m, refine_iterations, refineIterations);
            
            inlier_changed = false;
            std::swap (prev_inliers, new_inliers);

            // If the number of inliers changed, then we are still optimizing
            if (new_inliers.size () != prev_inliers.size ())
            {
                // Check if the number of inliers is oscillating in between two values
                if ((int)inliers_sizes.size () >= minInliersCount)
                {
                    if (inliers_sizes[inliers_sizes.size () - 1] == inliers_sizes[inliers_sizes.size () - 3] &&
                        inliers_sizes[inliers_sizes.size () - 2] == inliers_sizes[inliers_sizes.size () - 4])
                    {
                        oscillating = true;
                        break;
                    }
                }
                inlier_changed = true;
                continue;
            }

            // Check the values of the inlier set
            for (size_t i = 0; i < prev_inliers.size (); ++i)
            {
                // If the value of the inliers changed, then we are still optimizing
                if (prev_inliers[i] != new_inliers[i])
                {
                    inlier_changed = true;
                    break;
                }
            }
        }
        while (inlier_changed && ++refine_iterations < refineIterations);

        // If the new set of inliers is empty, we didn't do a good job refining
        if ((int)prev_inliers.size() < minInliersCount)
        {
            UWARN ("MSAC refineModel: Refinement failed: got very low inliers (%d)!", (int)prev_inliers.size());
        }

        if (oscillating)
        {
            UDEBUG("MSAC refineModel: Detected oscillations in the model refinement.");
        }

        std::swap (inliers, final_inliers);
        rvec = new_model_rvec;
        tvec = new_model_tvec;
    }

}

}

}