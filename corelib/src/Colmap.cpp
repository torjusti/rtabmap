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

#include <rtabmap/core/Colmap.h>
#include <rtabmap/core/DBDriver.h>
#include <rtabmap/core/GPS.h>
#include <rtabmap/core/SensorData.h>
#include <rtabmap/core/Signature.h>
#include <rtabmap/core/ProgressState.h>
#include <rtabmap/core/util3d_transforms.h>
#include <rtabmap/utilite/UConversion.h>
#include <rtabmap/utilite/UMath.h>
#include <rtabmap/utilite/UStl.h>
#include <rtabmap/utilite/UTimer.h>
#include <rtabmap/utilite/ULogger.h>
#include <rtabmap/utilite/UDirectory.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <fstream>
#include <limits>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <cstdio>
#include <memory>

namespace rtabmap {

namespace {

const char * colmapModelName(int colmapModel)
{
	if(colmapModel == kColmapOpenCV) return "OPENCV";
	if(colmapModel == kColmapSimpleRadial) return "SIMPLE_RADIAL";
	return "PINHOLE";
}

std::string fixed(double v, int precision)
{
	std::ostringstream ss;
	ss << std::fixed << std::setprecision(precision) << v;
	return ss.str();
}

double coeff(const cv::Mat & d, int i)
{
	if(d.empty() || i < 0 || i >= d.rows * d.cols)
	{
		return 0.0;
	}
	if(d.type() == CV_64F)
	{
		return d.reshape(1, 1).at<double>(0, i);
	}
	return (double)d.reshape(1, 1).at<float>(0, i);
}

void cameraIntrinsics(const CameraModel & model, int colmapModel, double & fx, double & fy, double & cx, double & cy)
{
	fx = model.fx();
	fy = model.fy();
	cx = model.cx();
	cy = model.cy();
	// OPENCV / SIMPLE_RADIAL describe the stored image, so prefer the
	// pre-rectification K when it is present. PINHOLE matches the
	// pinhole model RTAB-Map's BA actually used (fx() reads P when set).
	if(colmapModel != kColmapPinhole)
	{
		const cv::Mat K = model.K_raw();
		if(!K.empty() && K.rows == 3 && K.cols == 3)
		{
			const double rfx = K.at<double>(0, 0);
			const double rfy = K.at<double>(1, 1);
			if(rfx > 0.0 && rfy > 0.0)
			{
				fx = rfx;
				fy = rfy;
				cx = K.at<double>(0, 2);
				cy = K.at<double>(1, 2);
			}
		}
	}
}

std::string colmapCameraParams(const CameraModel & model, int colmapModel)
{
	double fx = 0, fy = 0, cx = 0, cy = 0;
	cameraIntrinsics(model, colmapModel, fx, fy, cx, cy);
	const cv::Mat d = model.D_raw();
	const double k1 = coeff(d, 0);
	const double k2 = coeff(d, 1);
	const double p1 = coeff(d, 2);
	const double p2 = coeff(d, 3);

	if(colmapModel == kColmapOpenCV)
	{
		return fixed(fx,10)+" "+fixed(fy,10)+" "+
			   fixed(cx,10)+" "+fixed(cy,10)+" "+
			   fixed(k1,10)+" "+fixed(k2,10)+" "+fixed(p1,10)+" "+fixed(p2,10);
	}
	if(colmapModel == kColmapSimpleRadial)
	{
		return fixed((fx+fy)/2.0,10)+" "+fixed(cx,10)+" "+
			   fixed(cy,10)+" "+fixed(k1,10);
	}
	return fixed(fx,10)+" "+fixed(fy,10)+" "+fixed(cx,10)+" "+fixed(cy,10);
}

std::string cameraKey(const CameraModel & model, int colmapModel)
{
	std::ostringstream ss;
	ss << model.imageWidth() << "|" << model.imageHeight() << "|"
	   << colmapModelName(colmapModel) << "|" << colmapCameraParams(model, colmapModel);
	return ss.str();
}

bool cameraFromSignature(const Signature & s, CameraModel & model)
{
	if(!s.sensorData().cameraModels().empty())
	{
		model = s.sensorData().cameraModels()[0];
	}
	else if(!s.sensorData().stereoCameraModels().empty())
	{
		model = s.sensorData().stereoCameraModels()[0].left();
	}
	else
	{
		return false;
	}
	return model.isValidForProjection() && !model.localTransform().isNull();
}

bool cameraFromDb(DBDriver * dbDriver, int id, CameraModel & model)
{
	std::vector<CameraModel> models;
	std::vector<StereoCameraModel> stereoModels;
	if(!dbDriver->getCalibration(id, models, stereoModels))
	{
		return false;
	}
	if(!models.empty())
	{
		model = models[0];
	}
	else if(!stereoModels.empty())
	{
		model = stereoModels[0].left();
	}
	else
	{
		return false;
	}
	return model.isValidForProjection() && !model.localTransform().isNull();
}

bool frameCameraModel(
		int id,
		const std::map<int, Signature> & signatures,
		DBDriver * dbDriver,
		CameraModel & model)
{
	std::map<int, Signature>::const_iterator siter = signatures.find(id);
	if(siter != signatures.end() && cameraFromSignature(siter->second, model))
	{
		return true;
	}
	return dbDriver && cameraFromDb(dbDriver, id, model);
}

cv::Mat frameImage(
		int id,
		const std::map<int, Signature> & signatures,
		DBDriver * dbDriver)
{
	std::map<int, Signature>::const_iterator siter = signatures.find(id);
	if(siter != signatures.end())
	{
		cv::Mat image = siter->second.sensorData().imageRaw();
		if(image.empty())
		{
			siter->second.sensorData().uncompressDataConst(&image, 0, 0, 0);
		}
		if(!image.empty())
		{
			return image;
		}
	}
	if(dbDriver)
	{
		SensorData data;
		dbDriver->getNodeData(id, data, true, false, false, false);
		cv::Mat image;
		data.uncompressDataConst(&image, 0, 0, 0);
		return image;
	}
	return cv::Mat();
}

std::vector<float> frameVelocity(
		int id,
		const std::map<int, Signature> & signatures,
		DBDriver * dbDriver)
{
	std::map<int, Signature>::const_iterator siter = signatures.find(id);
	if(siter != signatures.end() && siter->second.getVelocity().size() == 6)
	{
		return siter->second.getVelocity();
	}
	if(dbDriver)
	{
		Transform pose, groundTruth;
		int mapId = 0, weight = 0;
		std::string label;
		double stamp = 0.0;
		std::vector<float> velocity;
		GPS gps;
		EnvSensors sensors;
		dbDriver->getNodeInfo(id, pose, mapId, weight, label, stamp, groundTruth, velocity, gps, sensors);
		return velocity;
	}
	return std::vector<float>();
}

std::vector<cv::KeyPoint> frameKeypoints(
		int id,
		const std::map<int, Signature> & signatures,
		DBDriver * dbDriver)
{
	std::map<int, Signature>::const_iterator siter = signatures.find(id);
	if(siter != signatures.end() && !siter->second.getWordsKpts().empty())
	{
		return siter->second.getWordsKpts();
	}
	if(dbDriver)
	{
		std::multimap<int, int> words;
		std::vector<cv::KeyPoint> kpts;
		std::vector<cv::Point3f> words3;
		cv::Mat descriptors;
		dbDriver->getLocalFeatures(id, words, kpts, words3, descriptors);
		return kpts;
	}
	return std::vector<cv::KeyPoint>();
}

void stripOutliers(
		const BAOutliers & outliers,
		std::map<int, cv::Point3f> & points3DMap,
		std::map<int, std::map<int, FeatureBA> > & wordReferences,
		int & removedObservations)
{
	removedObservations = 0;
	for(BAOutliers::const_iterator iter=outliers.begin(); iter!=outliers.end(); ++iter)
	{
		std::map<int, std::map<int, FeatureBA> >::iterator wIter = wordReferences.find(iter->first);
		if(wIter == wordReferences.end())
		{
			continue;
		}
		for(std::set<int>::const_iterator pIter=iter->second.begin(); pIter!=iter->second.end(); ++pIter)
		{
			if(wIter->second.erase(*pIter))
			{
				++removedObservations;
			}
		}
		if(wIter->second.size() < 2)
		{
			points3DMap.erase(iter->first);
			wordReferences.erase(wIter);
		}
	}
}

typedef std::map<int, std::vector<std::pair<int, int> > > FrameObservations;

} // namespace

ColmapExportOptions::ColmapExportOptions() :
	cameraModel(kColmapOpenCV),
	exportPoints(true),
	exportImages(true),
	allKeypoints(true),
	sbaType(Optimizer::kTypeG2O),
	sbaIterations(20),
	sbaVariance(1.0),
	sbaRematchFeatures(true),
	maxLinearVelocity(0.0),
	maxAngularVelocity(0.0),
	laplacianThr(0.0)
{
}

bool progress(const ProgressState * state, const std::string & msg)
{
	if(!state)
	{
		return true;
	}
	if(state->isCanceled())
	{
		return false;
	}
	return state->callback(msg);
}

ColmapExportResult exportColmap(
		std::map<int, Transform> & poses,
		const std::multimap<int, Link> & links,
		const std::map<int, Signature> & signaturesIn,
		DBDriver * dbDriver,
		const ParametersMap & parameters,
		const ColmapExportOptions & options,
		const std::string & outputPath,
		const ProgressState * progressState)
{
	ColmapExportResult result;
	result.posesTotal = (int)poses.size();
	UTimer timer;

	if(poses.empty())
	{
		result.error = "No poses to export.";
		return result;
	}
	if(!UDirectory::exists(outputPath) && !UDirectory::makeDir(outputPath))
	{
		result.error = "Failed creating directory " + outputPath + ".";
		return result;
	}

	const std::map<int, Signature> * signatures = &signaturesIn;
	if(signaturesIn.empty() && !dbDriver)
	{
		result.error = "No signatures available (empty cache and no database).";
		return result;
	}

	std::map<int, cv::Point3f> points3DMap;
	std::map<int, std::map<int, FeatureBA> > wordReferences;
	if(options.exportPoints)
	{
		if(options.sbaType == Optimizer::kTypeTORO || !Optimizer::isAvailable(options.sbaType))
		{
			result.error = "Selected bundle adjustment back-end is not available.";
			return result;
		}

		ParametersMap parametersSBA = parameters;
		uInsert(parametersSBA, ParametersPair(Parameters::kOptimizerIterations(), uNumber2Str(options.sbaIterations)));
		uInsert(parametersSBA, ParametersPair(Parameters::kOptimizerPixelVariance(), uNumber2Str(options.sbaVariance)));

		std::shared_ptr<Optimizer> sba(Optimizer::create(options.sbaType, parametersSBA));
		sba->setProgressState(progressState);
		std::map<int, Transform> posesOut;
		std::multimap<int, Link> linksOut;
		sba->getConnectedGraph(poses.begin()->first, poses, links, posesOut, linksOut);
		for(std::map<int, Transform>::iterator iter=posesOut.begin(); iter!=posesOut.end(); ++iter)
		{
			iter->second = poses.at(iter->first);
		}

		UINFO("Colmap export: %s (%d poses, %d links)...",
				options.sbaIterations>0?"running bundle adjustment":"building correspondences",
				(int)posesOut.size(), (int)linksOut.size());

		if(options.sbaIterations > 0)
		{
			BAOutliers outliers;
			std::map<int, Transform> posesBA = sba->optimizeBA(
					posesOut.begin()->first,
					posesOut,
					linksOut,
					*signatures,
					points3DMap,
					wordReferences,
					options.sbaRematchFeatures,
					parametersSBA,
					&outliers,
					dbDriver);
			if(posesBA.empty())
			{
				result.error = (progressState && progressState->isCanceled()) ?
						"Canceled." : "Bundle adjustment failed! Nothing was exported.";
				return result;
			}
			poses = posesBA;
			stripOutliers(outliers, points3DMap, wordReferences, result.outliersRemoved);
		}
		else
		{
			poses = posesOut;
			sba->computeBACorrespondences(
					poses,
					linksOut,
					*signatures,
					points3DMap,
					wordReferences,
					options.sbaRematchFeatures,
					false,
					parametersSBA,
					dbDriver);
		}
		UINFO("Colmap export: %d points / %d poses after bundle adjustment (%.0fs).",
				(int)points3DMap.size(), (int)poses.size(), timer.ticks());
	}

	FrameObservations frameObservations;
	for(std::map<int, std::map<int, FeatureBA> >::const_iterator iter=wordReferences.begin(); iter!=wordReferences.end(); ++iter)
	{
		for(std::map<int, FeatureBA>::const_iterator jter=iter->second.begin(); jter!=iter->second.end(); ++jter)
		{
			frameObservations[jter->first].push_back(
					std::make_pair(jter->second.featureIndex, iter->first));
		}
	}

	if(options.exportImages)
	{
		UDirectory::makeDir(outputPath + "/images");
	}
	std::set<int> keptCameras;
	std::map<int, CameraModel> frameModels;

	const int maxWordId = points3DMap.empty()? 0 : points3DMap.rbegin()->first;
	std::vector<cv::Vec3b> pointColors((size_t)std::max(0, maxWordId)+1, cv::Vec3b(255,255,255));
	std::vector<bool> pointColorKnown((size_t)std::max(0, maxWordId)+1, false);

	{
		UINFO("Colmap export: streaming images and colours (%d frames)...", (int)poses.size());
		int frameIdx = 0;
		for(std::map<int, Transform>::const_iterator iter=poses.begin(); iter!=poses.end(); ++iter)
		{
			++frameIdx;
			if((frameIdx == 1 || frameIdx == (int)poses.size() || frameIdx % 50 == 0) &&
			   !progress(progressState, uFormat("Streaming images %d/%d", frameIdx, (int)poses.size())))
			{
				result.error = "Canceled.";
				return result;
			}
			const int id = iter->first;
			if(id <= 0)
			{
				continue;
			}
			CameraModel model;
			if(!frameCameraModel(id, *signatures, dbDriver, model))
			{
				UWARN("Missing calibration for node %d, skipping", id);
				continue;
			}

			bool rejected = false;
			if(options.maxLinearVelocity > 0.0 || options.maxAngularVelocity > 0.0)
			{
				const std::vector<float> velocity = frameVelocity(id, *signatures, dbDriver);
				if(velocity.size() == 6)
				{
					const float transVel = uMax3(fabs(velocity[0]), fabs(velocity[1]), fabs(velocity[2]));
					const float rotVel = uMax3(fabs(velocity[3]), fabs(velocity[4]), fabs(velocity[5]));
					if(options.maxLinearVelocity > 0.0 && transVel > options.maxLinearVelocity)
					{
						UWARN("Fast motion detected for camera %d (%f m/s > %f m/s), excluded.", id, transVel, options.maxLinearVelocity);
						rejected = true;
					}
					else if(options.maxAngularVelocity > 0.0 && rotVel > options.maxAngularVelocity)
					{
						UWARN("Fast motion detected for camera %d (%f rad/s > %f rad/s), excluded.", id, rotVel, options.maxAngularVelocity);
						rejected = true;
					}
				}
				else
				{
					UWARN("Camera motion filtering is set, but velocity of camera %d is not available.", id);
				}
			}

			FrameObservations::const_iterator oIter = frameObservations.find(id);
			const bool needImage = !rejected &&
					(options.laplacianThr > 0.0 || options.exportImages || oIter != frameObservations.end());
			cv::Mat image;
			if(needImage)
			{
				image = frameImage(id, *signatures, dbDriver);
			}

			if(!rejected && options.laplacianThr > 0.0 && !image.empty())
			{
				cv::Mat imgLaplacian;
				cv::Laplacian(image, imgLaplacian, CV_16S);
				cv::Mat m, s;
				cv::meanStdDev(imgLaplacian, m, s);
				const double stddev = s.at<double>(0);
				const double var = stddev*stddev;
				if(var < options.laplacianThr)
				{
					UWARN("Camera's image %d is blurry (var=%f < thr=%f), excluded.", id, var, options.laplacianThr);
					rejected = true;
				}
			}

			if(rejected)
			{
				continue;
			}
			if(model.imageWidth() <= 0 && !image.empty())
			{
				model.setImageSize(image.size());
			}
			keptCameras.insert(keptCameras.end(), id);
			frameModels.insert(frameModels.end(), std::make_pair(id, model));

			if(options.exportImages && !image.empty())
			{
				char nameBuf[32];
				snprintf(nameBuf, sizeof(nameBuf), "%d.jpg", id);
				const std::string p = outputPath + "/images/" + nameBuf;
				if(!cv::imwrite(p, image))
				{
					UERROR("Failed to save image %s", p.c_str());
				}
			}

			if(oIter != frameObservations.end() && !image.empty() &&
			   (image.type() == CV_8UC3 || image.type() == CV_8UC1))
			{
				for(size_t i=0; i<oIter->second.size(); ++i)
				{
					const int wordId = oIter->second[i].second;
					if(wordId < 0 || wordId >= (int)pointColorKnown.size() || pointColorKnown[wordId])
					{
						continue;
					}
					const std::map<int, FeatureBA> & refs = wordReferences.at(wordId);
					const std::map<int, FeatureBA>::const_iterator rIter = refs.find(id);
					if(rIter == refs.end())
					{
						continue;
					}
					const cv::Point2f & pt = rIter->second.kpt.pt;
					if(pt.x >= 0.0f && (int)pt.x < image.cols && pt.y >= 0.0f && (int)pt.y < image.rows)
					{
						if(image.channels() == 3)
						{
							pointColors[wordId] = image.at<cv::Vec3b>((int)pt.y, (int)pt.x);
						}
						else
						{
							const unsigned char g = image.at<unsigned char>((int)pt.y, (int)pt.x);
							pointColors[wordId] = cv::Vec3b(g, g, g);
						}
						pointColorKnown[wordId] = true;
					}
				}
			}
		}
	}
	UINFO("Colmap export: %d cameras kept (%.0fs).", (int)keptCameras.size(), timer.ticks());

	if(keptCameras.empty())
	{
		result.error = "No camera survived the filters, nothing was exported.";
		return result;
	}

	int droppedPoints = 0;
	for(std::map<int, std::map<int, FeatureBA> >::iterator iter=wordReferences.begin(); iter!=wordReferences.end(); )
	{
		for(std::map<int, FeatureBA>::iterator jter=iter->second.begin(); jter!=iter->second.end(); )
		{
			if(keptCameras.find(jter->first) == keptCameras.end())
			{
				iter->second.erase(jter++);
			}
			else
			{
				++jter;
			}
		}
		if(iter->second.size() < 2)
		{
			points3DMap.erase(iter->first);
			wordReferences.erase(iter++);
			++droppedPoints;
		}
		else
		{
			++iter;
		}
	}

	// Dense COLMAP POINT3D_IDs, used in both images.txt and points3D.txt.
	std::map<int, int> wordIdToPointId;
	{
		int pointId = 1;
		for(std::map<int, cv::Point3f>::const_iterator iter=points3DMap.begin(); iter!=points3DMap.end(); ++iter)
		{
			if(wordReferences.find(iter->first) != wordReferences.end())
			{
				wordIdToPointId[iter->first] = pointId++;
			}
		}
	}

	frameObservations.clear();
	for(std::map<int, std::map<int, FeatureBA> >::const_iterator iter=wordReferences.begin(); iter!=wordReferences.end(); ++iter)
	{
		for(std::map<int, FeatureBA>::const_iterator jter=iter->second.begin(); jter!=iter->second.end(); ++jter)
		{
			frameObservations[jter->first].push_back(
					std::make_pair(jter->second.featureIndex, iter->first));
		}
	}

	// <frame, <wordId, POINT2D_IDX>>
	std::map<int, std::vector<std::pair<int, int> > > point2DIndex;
	std::map<int, std::vector<cv::KeyPoint> > frameKpts;
	for(std::set<int>::const_iterator iter=keptCameras.begin(); iter!=keptCameras.end(); ++iter)
	{
		const int id = *iter;
		if(options.allKeypoints)
		{
			frameKpts[id] = frameKeypoints(id, *signatures, dbDriver);
		}
		FrameObservations::iterator oIter = frameObservations.find(id);
		if(oIter == frameObservations.end())
		{
			continue;
		}
		std::sort(oIter->second.begin(), oIter->second.end());
		std::vector<std::pair<int, int> > & index = point2DIndex[id];
		index.reserve(oIter->second.size());
		for(size_t i=0; i<oIter->second.size(); ++i)
		{
			const int wordId = oIter->second[i].second;
			int point2dIdx;
			if(options.allKeypoints)
			{
				point2dIdx = oIter->second[i].first;
				const std::vector<cv::KeyPoint> & kpts = frameKpts[id];
				const cv::Point2f & pt = wordReferences.at(wordId).at(id).kpt.pt;
				if(point2dIdx < 0 || point2dIdx >= (int)kpts.size() ||
				   fabs(kpts[point2dIdx].pt.x - pt.x) > 0.5f ||
				   fabs(kpts[point2dIdx].pt.y - pt.y) > 0.5f)
				{
					point2dIdx = -1;
					for(size_t k=0; k<kpts.size(); ++k)
					{
						if(fabs(kpts[k].pt.x - pt.x) <= 0.5f && fabs(kpts[k].pt.y - pt.y) <= 0.5f)
						{
							point2dIdx = (int)k;
							break;
						}
					}
				}
			}
			else
			{
				point2dIdx = (int)i;
			}
			if(point2dIdx >= 0)
			{
				index.push_back(std::make_pair(wordId, point2dIdx));
			}
		}
		std::sort(index.begin(), index.end());
	}

	UINFO("Colmap export: writing model (%d cameras, %d points) to %s...",
			(int)keptCameras.size(), (int)wordIdToPointId.size(), outputPath.c_str());
	if(!progress(progressState, "Writing COLMAP model..."))
	{
		result.error = "Canceled.";
		return result;
	}

	std::map<std::string, int> cameraIds;
	std::map<int, int> frameToCamera;
	{
		std::ofstream file((outputPath + "/cameras.txt").c_str());
		if(!file.is_open())
		{
			result.error = "Failed opening cameras.txt for writing.";
			return result;
		}
		file << "# Camera list with one line of data per camera:\n";
		file << "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n";

		std::vector<std::string> lines;
		for(std::set<int>::const_iterator iter=keptCameras.begin(); iter!=keptCameras.end(); ++iter)
		{
			const CameraModel & model = frameModels.at(*iter);
			const std::string key = cameraKey(model, options.cameraModel);
			std::map<std::string, int>::const_iterator cIter = cameraIds.find(key);
			int camId;
			if(cIter == cameraIds.end())
			{
				camId = (int)cameraIds.size() + 1;
				cameraIds.insert(std::make_pair(key, camId));
				std::ostringstream ln;
				ln << camId << " " << colmapModelName(options.cameraModel) << " "
				   << model.imageWidth() << " " << model.imageHeight() << " "
				   << colmapCameraParams(model, options.cameraModel);
				lines.push_back(ln.str());
			}
			else
			{
				camId = cIter->second;
			}
			frameToCamera.insert(frameToCamera.end(), std::make_pair(*iter, camId));
		}
		file << "# Number of cameras: " << lines.size() << "\n";
		for(size_t i=0; i<lines.size(); ++i)
		{
			file << lines[i] << "\n";
		}
	}

	std::map<int, int> frameToImageId;
	{
		int imageId = 1;
		for(std::set<int>::const_iterator iter=keptCameras.begin(); iter!=keptCameras.end(); ++iter)
		{
			frameToImageId.insert(frameToImageId.end(), std::make_pair(*iter, imageId++));
		}
	}

	{
		std::ofstream file((outputPath + "/images.txt").c_str());
		if(!file.is_open())
		{
			result.error = "Failed opening images.txt for writing.";
			return result;
		}
		file << "# Image list with two lines of data per image:\n";
		file << "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n";
		file << "#   POINTS2D[] as (X, Y, POINT3D_ID)\n";
		file << "# Number of images: " << (int)keptCameras.size() << "\n";

		int written = 0;
		for(std::set<int>::const_iterator iter=keptCameras.begin(); iter!=keptCameras.end(); ++iter)
		{
			const int id = *iter;
			const CameraModel & model = frameModels.at(id);
			const Transform cameraToWorld = poses.at(id) * model.localTransform();
			const Transform worldToCamera = cameraToWorld.inverse();
			const Eigen::Quaterniond q = worldToCamera.getQuaterniond();

			file << frameToImageId.at(id) << " "
				<< fixed(q.w(),10) << " " << fixed(q.x(),10) << " " << fixed(q.y(),10) << " " << fixed(q.z(),10) << " "
				<< fixed(worldToCamera.x(),10) << " " << fixed(worldToCamera.y(),10) << " " << fixed(worldToCamera.z(),10) << " "
				<< frameToCamera.at(id) << " " << id << ".jpg\n";

			if(options.allKeypoints)
			{
				const std::vector<cv::KeyPoint> & kpts = frameKpts[id];
				std::vector<int> ptIds(kpts.size(), -1);
				std::map<int, std::vector<std::pair<int, int> > >::const_iterator idxIter = point2DIndex.find(id);
				if(idxIter != point2DIndex.end())
				{
					for(size_t i=0; i<idxIter->second.size(); ++i)
					{
						const int wordId = idxIter->second[i].first;
						const int point2dIdx = idxIter->second[i].second;
						std::map<int, int>::const_iterator pIter = wordIdToPointId.find(wordId);
						if(pIter != wordIdToPointId.end() && point2dIdx >= 0 && point2dIdx < (int)ptIds.size())
						{
							ptIds[point2dIdx] = pIter->second;
						}
					}
				}
				for(size_t i=0; i<kpts.size(); ++i)
				{
					if(i)
					{
						file << " ";
					}
					file << fixed(kpts[i].pt.x,6) << " " << fixed(kpts[i].pt.y,6) << " " << ptIds[i];
				}
			}
			else
			{
				FrameObservations::const_iterator oIter = frameObservations.find(id);
				if(oIter != frameObservations.end())
				{
					for(size_t i=0; i<oIter->second.size(); ++i)
					{
						const int wordId = oIter->second[i].second;
						const cv::Point2f & pt = wordReferences.at(wordId).at(id).kpt.pt;
						std::map<int, int>::const_iterator pIter = wordIdToPointId.find(wordId);
						const int point3dId = pIter == wordIdToPointId.end() ? -1 : pIter->second;
						if(i)
						{
							file << " ";
						}
						file << fixed(pt.x,6) << " " << fixed(pt.y,6) << " " << point3dId;
					}
				}
			}
			file << "\n";
			if(++written % 2000 == 0)
			{
				UINFO("Colmap export: images.txt %d/%d (%.0fs)...", written, (int)keptCameras.size(), timer.elapsed());
			}
		}
	}

	{
		std::ofstream file((outputPath + "/points3D.txt").c_str());
		if(!file.is_open())
		{
			result.error = "Failed opening points3D.txt for writing.";
			return result;
		}
		file << "# 3D point list with one line of data per point:\n";
		file << "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n";
		file << "# Number of points: " << (int)wordIdToPointId.size() << "\n";

		for(std::map<int, int>::const_iterator idIter=wordIdToPointId.begin(); idIter!=wordIdToPointId.end(); ++idIter)
		{
			const int wordId = idIter->first;
			const int pointId = idIter->second;
			const cv::Point3f & xyz = points3DMap.at(wordId);
			const std::map<int, FeatureBA> & refs = wordReferences.at(wordId);

			cv::Vec3b color(255, 255, 255);
			if(wordId >= 0 && wordId < (int)pointColors.size())
			{
				color = pointColors[wordId];
			}

			double fx, fy, cx, cy;
			double errorSum = 0.0;
			int errorCount = 0;
			for(std::map<int, FeatureBA>::const_iterator jter=refs.begin(); jter!=refs.end(); ++jter)
			{
				std::map<int, CameraModel>::const_iterator mIter = frameModels.find(jter->first);
				if(mIter == frameModels.end())
				{
					continue;
				}
				const Transform worldToCamera = (poses.at(jter->first) * mIter->second.localTransform()).inverse();
				const cv::Point3f pc = util3d::transformPoint(xyz, worldToCamera);
				if(pc.z <= 0.0f)
				{
					continue;
				}
				cameraIntrinsics(mIter->second, options.cameraModel, fx, fy, cx, cy);
				const double du = fx*pc.x/pc.z + cx - jter->second.kpt.pt.x;
				const double dv = fy*pc.y/pc.z + cy - jter->second.kpt.pt.y;
				errorSum += std::sqrt(du*du + dv*dv);
				++errorCount;
			}

			file << pointId << " "
				<< fixed(xyz.x,8) << " " << fixed(xyz.y,8) << " " << fixed(xyz.z,8) << " "
				<< (int)color[2] << " " << (int)color[1] << " " << (int)color[0] << " "
				<< fixed(errorCount? errorSum/(double)errorCount : -1.0, 6);

			for(std::map<int, FeatureBA>::const_iterator jter=refs.begin(); jter!=refs.end(); ++jter)
			{
				std::map<int, std::vector<std::pair<int, int> > >::const_iterator fIter =
						point2DIndex.find(jter->first);
				if(fIter == point2DIndex.end())
				{
					continue;
				}
				const std::vector<std::pair<int, int> >::const_iterator idxIter =
						std::lower_bound(fIter->second.begin(), fIter->second.end(),
								std::make_pair(wordId, std::numeric_limits<int>::min()));
				if(idxIter == fIter->second.end() || idxIter->first != wordId)
				{
					continue;
				}
				file << " " << frameToImageId.at(jter->first) << " " << idxIter->second;
			}
			file << "\n";
		}

		result.pointsExported = (int)wordIdToPointId.size();
	}

	result.ok = true;
	result.camerasExported = (int)keptCameras.size();
	result.droppedPoints = droppedPoints;
	UINFO("Colmap export: done. %d cameras, %d points exported to %s (%.0fs writing).",
			result.camerasExported, result.pointsExported, outputPath.c_str(), timer.ticks());
	return result;
}

} // namespace rtabmap
