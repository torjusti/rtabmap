/**
 * Point observed in a camera/body frame.
 *
 * Error is pose.transformTo(point) - measured, in metres. This is the
 * cartesian constraint g2o's EdgeSE3PointXYZ applies. GTSAM's
 * BearingRangeFactor reads the same 3x3 block as two angles plus a range,
 * which turns a metre uncertainty into a heading torque.
 */

#pragma once

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/geometry/Pose3.h>

namespace rtabmap {

class BodyPointFactor : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Point3> {
	gtsam::Point3 measured_;

public:
	BodyPointFactor(gtsam::Key poseKey, gtsam::Key pointKey, const gtsam::Point3 & measured,
			const gtsam::SharedNoiseModel & model) :
		gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Point3>(model, poseKey, pointKey),
		measured_(measured) {}

	gtsam::Vector evaluateError(const gtsam::Pose3 & pose, const gtsam::Point3 & point,
#if GTSAM_VERSION_NUMERIC >= 40300
			gtsam::OptionalMatrixType H1,
			gtsam::OptionalMatrixType H2) const override
#else
			boost::optional<gtsam::Matrix &> H1,
			boost::optional<gtsam::Matrix &> H2) const
#endif
	{
		return pose.transformTo(point, H1, H2) - measured_;
	}
};

} // namespace rtabmap
