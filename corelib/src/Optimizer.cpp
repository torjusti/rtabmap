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

#include <rtabmap/utilite/ULogger.h>
#include <rtabmap/utilite/UStl.h>
#include <rtabmap/utilite/UMath.h>
#include <rtabmap/utilite/UConversion.h>
#include <rtabmap/utilite/UTimer.h>
#include <rtabmap/core/Optimizer.h>
#include <rtabmap/core/ProgressState.h>
#include <rtabmap/core/OdometryArcLength.h>
#include <tuple>
#include <algorithm>
#include <rtabmap/core/Graph.h>
#include <rtabmap/core/util3d_transforms.h>
#include <rtabmap/core/RegistrationVis.h>
#include <rtabmap/core/DBDriver.h>
#include <rtabmap/core/GPS.h>
#include <rtabmap/core/EnvSensor.h>
#include <set>
#include <queue>
#include <utility>
#include <vector>
#include <list>
#include <mutex>
#ifdef _OPENMP
#include <omp.h>
#endif

#include <rtabmap/core/optimizer/OptimizerTORO.h>
#include <rtabmap/core/optimizer/OptimizerG2O.h>
#include <rtabmap/core/optimizer/OptimizerGTSAM.h>
#include <rtabmap/core/optimizer/OptimizerCVSBA.h>
#include <rtabmap/core/optimizer/OptimizerCeres.h>

namespace rtabmap {

bool Optimizer::isAvailable(Optimizer::Type type)
{
	if(type == Optimizer::kTypeG2O)
	{
		return OptimizerG2O::available();
	}
	else if(type == Optimizer::kTypeGTSAM)
	{
		return OptimizerGTSAM::available();
	}
	else if(type == Optimizer::kTypeCVSBA)
	{
		return OptimizerCVSBA::available();
	}
	else if(type == Optimizer::kTypeTORO)
	{
		return OptimizerTORO::available();
	}
	else if(type == Optimizer::kTypeCeres)
	{
		return OptimizerCeres::available();
	}
	return false;
}

Optimizer * Optimizer::create(const ParametersMap & parameters)
{
	int optimizerTypeInt = Parameters::defaultOptimizerStrategy();
	Parameters::parse(parameters, Parameters::kOptimizerStrategy(), optimizerTypeInt);
	return create((Optimizer::Type)optimizerTypeInt, parameters);
}

Optimizer * Optimizer::create(Optimizer::Type type, const ParametersMap & parameters)
{
	UASSERT_MSG(OptimizerG2O::available() || OptimizerGTSAM::available() || OptimizerTORO::available() || OptimizerCeres::available(),
			"RTAB-Map is not built with any graph optimization approach!");

	if(!OptimizerTORO::available() && type == Optimizer::kTypeTORO)
	{
		if(OptimizerGTSAM::available())
		{
			UWARN("TORO optimizer not available. GTSAM will be used instead.");
					type = Optimizer::kTypeGTSAM;
		}
		else if(OptimizerG2O::available())
		{
			UWARN("TORO optimizer not available. g2o will be used instead.");
					type = Optimizer::kTypeG2O;
		}
		else if(OptimizerCeres::available())
		{
			UWARN("TORO optimizer not available. ceres will be used instead.");
					type = Optimizer::kTypeCeres;
		}
	}
	if(!OptimizerG2O::available() && type == Optimizer::kTypeG2O)
	{
		if(OptimizerGTSAM::available())
		{
			UWARN("g2o optimizer not available. GTSAM will be used instead.");
					type = Optimizer::kTypeGTSAM;
		}
		else if(OptimizerTORO::available())
		{
			UWARN("g2o optimizer not available. TORO will be used instead.");
					type = Optimizer::kTypeTORO;
		}
		else if(OptimizerCeres::available())
		{
			UWARN("g2o optimizer not available. ceres will be used instead.");
					type = Optimizer::kTypeCeres;
		}
	}
	if(!OptimizerGTSAM::available() && type == Optimizer::kTypeGTSAM)
	{
		if(OptimizerG2O::available())
		{
			UWARN("GTSAM optimizer not available. g2o will be used instead.");
					type = Optimizer::kTypeG2O;
		}
		else if(OptimizerTORO::available())
		{
			UWARN("GTSAM optimizer not available. TORO will be used instead.");
					type = Optimizer::kTypeTORO;
		}
		else if(OptimizerCeres::available())
		{
			UWARN("GTSAM optimizer not available. ceres will be used instead.");
					type = Optimizer::kTypeCeres;
		}
	}
	if(!OptimizerCVSBA::available() && type == Optimizer::kTypeCVSBA)
	{
		if(OptimizerG2O::available())
		{
			UWARN("CVSBA optimizer not available. g2o will be used instead.");
					type = Optimizer::kTypeG2O;
		}
	}
	if(!OptimizerCeres::available() && type == Optimizer::kTypeCeres)
	{
		if(OptimizerGTSAM::available())
		{
			UWARN("Ceres optimizer not available. gtsam will be used instead.");
					type = Optimizer::kTypeGTSAM;
		}
		else if(OptimizerG2O::available())
		{
			UWARN("Ceres optimizer not available. g2o will be used instead.");
					type = Optimizer::kTypeG2O;
		}
		else if(OptimizerTORO::available())
		{
			UWARN("Ceres optimizer not available. TORO will be used instead.");
					type = Optimizer::kTypeTORO;
		}
	}
	Optimizer * optimizer = 0;
	switch(type)
	{
	case Optimizer::kTypeGTSAM:
		optimizer = new OptimizerGTSAM(parameters);
		break;
	case Optimizer::kTypeG2O:
		optimizer = new OptimizerG2O(parameters);
		break;
	case Optimizer::kTypeCVSBA:
		optimizer = new OptimizerCVSBA(parameters);
		break;
	case Optimizer::kTypeCeres:
		optimizer = new OptimizerCeres(parameters);
		break;
	case Optimizer::kTypeTORO:
	default:
		optimizer = new OptimizerTORO(parameters);
		break;

	}
	return optimizer;
}

class LinkIdKey
{
public:
LinkIdKey(int id, Link::Type type) :
		id_(id),
		type_(type) {}
	bool operator<(const LinkIdKey & k) const
	{
		// landmark, sort by smallest to largest landmark id, after normal links
		if(id_ < 0 && k.id_ < 0)
		{
			return id_ > k.id_;
		}
		else if(id_ < 0) {
			return false;
		}
		else if(k.id_ < 0) {
			return true;
		}

		if(type_ == Link::kNeighbor && k.type_ != Link::kNeighbor)
		{
			return true;
		}
		else if(type_ != Link::kNeighbor && k.type_ == Link::kNeighbor)
		{
			return false;
		}
		else if(type_ == Link::kNeighborMerged && k.type_ != Link::kNeighbor && k.type_ != Link::kNeighborMerged)
		{
			return true;
		}
		else if(k.type_ == Link::kNeighborMerged && type_ != Link::kNeighbor && type_ != Link::kNeighborMerged)
		{
			return false;
		}
		else
		{
			// normal link, sort by smallest to largest id
			return id_ < k.id_;
		}
	}
	int id_;
	Link::Type type_;
};

void Optimizer::getConnectedGraph(
		int fromId,
		const std::map<int, Transform> & posesIn,
		const std::multimap<int, Link> & linksIn,
		std::map<int, Transform> & posesOut,
		std::multimap<int, Link> & linksOut) const
{
	UDEBUG("IN: fromId=%d poses=%d links=%d priorsIgnored=%d landmarksIgnored=%d", fromId, (int)posesIn.size(), (int)linksIn.size(), priorsIgnored()?1:0, landmarksIgnored()?1:0);
	UASSERT(fromId>0);
	UASSERT_MSG(uContains(posesIn, fromId), uFormat("poses=%ld (first=%d last=%d) fromId=%d",
		posesIn.size(), posesIn.empty()?0:posesIn.begin()->first, posesIn.empty()?0:posesIn.rbegin()->first, fromId).c_str());

	posesOut.clear();
	linksOut.clear();

	std::map<LinkIdKey, Transform> nextPoses;
	nextPoses.insert(std::make_pair(LinkIdKey(fromId, Link::kUndef), posesIn.find(fromId)->second));
	std::multimap<int, std::pair<int, Link::Type> > biLinks;
	for(std::multimap<int, Link>::const_iterator iter=linksIn.begin(); iter!=linksIn.end(); ++iter)
	{
		if(iter->second.from() != iter->second.to())
		{
			if(graph::findLink(biLinks, iter->second.from(), iter->second.to(), true, iter->second.type()) == biLinks.end())
			{
				biLinks.insert(std::make_pair(iter->second.from(), std::make_pair(iter->second.to(), iter->second.type())));
				biLinks.insert(std::make_pair(iter->second.to(), std::make_pair(iter->second.from(), iter->second.type())));
			}
		}
	}

	while(!nextPoses.empty())
	{
		// Fill up all nodes before landmarks
		// For nodes, fill up all neightbor nodes before loop closure ones
		int currentId = nextPoses.begin()->first.id_; 
		Transform currentPose = nextPoses.begin()->second;
		nextPoses.erase(nextPoses.begin());

		if(posesOut.find(currentId) != posesOut.end()) {
			// Already added from priority list
			continue;
		}

		posesOut.insert(std::make_pair(currentId, currentPose));

		// add prior links
		for(std::multimap<int, Link>::const_iterator pter=linksIn.lower_bound(currentId); pter!=linksIn.end() && pter->first==currentId; ++pter)
		{
			if(pter->second.from() == pter->second.to() && (!priorsIgnored() || pter->second.type() != Link::kPosePrior))
			{
				linksOut.insert(*pter);
			}
		}

		for(std::multimap<int, std::pair<int, Link::Type> >::const_iterator iter=biLinks.lower_bound(currentId); iter!=biLinks.end() && iter->first==currentId; ++iter)
		{
			int toId = iter->second.first;
			Link::Type type = iter->second.second;
			if(posesIn.find(toId) != posesIn.end() && (!landmarksIgnored() || toId>0))
			{
				std::multimap<int, Link>::const_iterator kter = graph::findLink(linksIn, currentId, toId, true, type);
				UASSERT(kter!=linksIn.end());
				if(!uContains(posesOut, toId))
				{
					const Transform & poseToIn = posesIn.at(toId);
					Transform t = kter->second.from()==currentId?kter->second.transform():kter->second.transform().inverse();
					Transform pose;
					if(isSlam2d() && kter->second.type() == Link::kLandmark && toId>0 && (poseToIn.is3DoF() || poseToIn.is4DoF()))
					{
						if(poseToIn.is3DoF())
						{
							pose = (posesOut.at(currentId) * t).to3DoF();
						}
						else
						{
							pose = (posesOut.at(currentId) * t).to4DoF();
						}
					}
					else
					{
						pose = posesOut.at(currentId)* t;
					}

					nextPoses.insert(std::make_pair(LinkIdKey(toId, type), pose));
				}

				// only add unique links
				if(graph::findLink(linksOut, currentId, toId, true, kter->second.type()) == linksOut.end())
				{
					if(kter->second.to() < 0)
					{
						// For landmarks, make sure fromId is the landmark
						linksOut.insert(std::make_pair(kter->second.to(), kter->second.inverse()));
					}
					else
					{
						linksOut.insert(*kter);
					}
				}
			}
		}
	}
	UDEBUG("OUT: poses=%d links=%d", (int)posesOut.size(), (int)linksOut.size());
}

Optimizer::Optimizer(int iterations, bool slam2d, bool covarianceIgnored, double epsilon, bool robust, bool priorsIgnored, bool landmarksIgnored, float gravitySigma) :
		iterations_(iterations),
		slam2d_(slam2d),
		covarianceIgnored_(covarianceIgnored),
		epsilon_(epsilon),
		robust_(robust),
		priorsIgnored_(priorsIgnored),
		landmarksIgnored_(landmarksIgnored),
		gravitySigma_(gravitySigma),
		loopRedundancyRadius_(Parameters::defaultOptimizerLoopRedundancyRadius()),
		neighborLinearStdPerMeter_(Parameters::defaultOptimizerNeighborLinearStdPerMeter()),
		neighborYawStdPerMeter_(Parameters::defaultOptimizerNeighborYawStdPerMeter()),
		loopLinearStdMin_(Parameters::defaultOptimizerLoopLinearStdMin()),
		loopAngularStdMin_(Parameters::defaultOptimizerLoopAngularStdMin()),
		canceled_(false),
		progressState_(0),
		baStrategy_(Parameters::defaultOptimizerBAStrategy()),
		baMinTrackLength_(Parameters::defaultOptimizerBAMinTrackLength()),
		baTriangulationMinAngle_(Parameters::defaultOptimizerBATriangulationMinAngle()),
		baTriangulationMaxReprojError_(Parameters::defaultOptimizerBATriangulationMaxReprojError()),
		baTriangulationMaxDepthError_(Parameters::defaultOptimizerBATriangulationMaxDepthError()),
		baTriangulationDepthSeed_(Parameters::defaultOptimizerBATriangulationDepthSeed()),
		baWindowBaseline_(Parameters::defaultOptimizerBAWindowBaseline()),
		baWindowBaselineRatio_(Parameters::defaultOptimizerBAWindowBaselineRatio()),
		baGuidedMatching_(Parameters::defaultOptimizerBAGuidedMatching()),
		baGuidedWinSize_(Parameters::defaultOptimizerBAGuidedWinSize()),
		baGuidedNNDR_(Parameters::defaultOptimizerBAGuidedNNDR()),
		baGuidedSkipPnP_(Parameters::defaultOptimizerBAGuidedSkipPnP()),
		baMatchCache_(Parameters::defaultOptimizerBAMatchCache()),
		baFrameCacheMemory_(Parameters::defaultOptimizerBAFrameCacheMemory()),
		matchStore_(0),
		baProposePairs_(Parameters::defaultOptimizerBAProposePairs()),
		baProposeMaxRange_(Parameters::defaultOptimizerBAProposeMaxRange()),
		baProposeMaxAngle_(Parameters::defaultOptimizerBAProposeMaxAngle()),
		baProposeMinOverlap_(Parameters::defaultOptimizerBAProposeMinOverlap()),
		baProposeMaxPairs_(Parameters::defaultOptimizerBAProposeMaxPairs()),
		baLoopScope_(Parameters::defaultOptimizerBALoopScope()),
		baLoopScopeMinTracks_(Parameters::defaultOptimizerBALoopScopeMinTracks()),
		baWindowGraphDepth_(Parameters::defaultOptimizerBAWindowGraphDepth()),
		baLoopLinkAsEdge_(Parameters::defaultOptimizerBALoopLinkAsEdge()),
		baLoopLinkMinObservations_(Parameters::defaultOptimizerBALoopLinkMinObservations()),
		baDepth_(Parameters::defaultOptimizerBADepth())
{
}

Optimizer::Optimizer(const ParametersMap & parameters) :
		iterations_(Parameters::defaultOptimizerIterations()),
		slam2d_(Parameters::defaultRegForce3DoF()),
		covarianceIgnored_(Parameters::defaultOptimizerVarianceIgnored()),
		epsilon_(Parameters::defaultOptimizerEpsilon()),
		robust_(Parameters::defaultOptimizerRobust()),
		priorsIgnored_(Parameters::defaultOptimizerPriorsIgnored()),
		landmarksIgnored_(Parameters::defaultOptimizerLandmarksIgnored()),
		gravitySigma_(Parameters::defaultOptimizerGravitySigma()),
		loopRedundancyRadius_(Parameters::defaultOptimizerLoopRedundancyRadius()),
		neighborLinearStdPerMeter_(Parameters::defaultOptimizerNeighborLinearStdPerMeter()),
		neighborYawStdPerMeter_(Parameters::defaultOptimizerNeighborYawStdPerMeter()),
		loopLinearStdMin_(Parameters::defaultOptimizerLoopLinearStdMin()),
		loopAngularStdMin_(Parameters::defaultOptimizerLoopAngularStdMin()),
		canceled_(false),
		progressState_(0),
		baStrategy_(Parameters::defaultOptimizerBAStrategy()),
		baMinTrackLength_(Parameters::defaultOptimizerBAMinTrackLength()),
		baTriangulationMinAngle_(Parameters::defaultOptimizerBATriangulationMinAngle()),
		baTriangulationMaxReprojError_(Parameters::defaultOptimizerBATriangulationMaxReprojError()),
		baTriangulationMaxDepthError_(Parameters::defaultOptimizerBATriangulationMaxDepthError()),
		baTriangulationDepthSeed_(Parameters::defaultOptimizerBATriangulationDepthSeed()),
		baWindowBaseline_(Parameters::defaultOptimizerBAWindowBaseline()),
		baWindowBaselineRatio_(Parameters::defaultOptimizerBAWindowBaselineRatio()),
		baGuidedMatching_(Parameters::defaultOptimizerBAGuidedMatching()),
		baGuidedWinSize_(Parameters::defaultOptimizerBAGuidedWinSize()),
		baGuidedNNDR_(Parameters::defaultOptimizerBAGuidedNNDR()),
		baGuidedSkipPnP_(Parameters::defaultOptimizerBAGuidedSkipPnP()),
		baMatchCache_(Parameters::defaultOptimizerBAMatchCache()),
		baFrameCacheMemory_(Parameters::defaultOptimizerBAFrameCacheMemory()),
		matchStore_(0),
		baProposePairs_(Parameters::defaultOptimizerBAProposePairs()),
		baProposeMaxRange_(Parameters::defaultOptimizerBAProposeMaxRange()),
		baProposeMaxAngle_(Parameters::defaultOptimizerBAProposeMaxAngle()),
		baProposeMinOverlap_(Parameters::defaultOptimizerBAProposeMinOverlap()),
		baProposeMaxPairs_(Parameters::defaultOptimizerBAProposeMaxPairs()),
		baLoopScope_(Parameters::defaultOptimizerBALoopScope()),
		baLoopScopeMinTracks_(Parameters::defaultOptimizerBALoopScopeMinTracks()),
		baWindowGraphDepth_(Parameters::defaultOptimizerBAWindowGraphDepth()),
		baLoopLinkAsEdge_(Parameters::defaultOptimizerBALoopLinkAsEdge()),
		baLoopLinkMinObservations_(Parameters::defaultOptimizerBALoopLinkMinObservations()),
		baDepth_(Parameters::defaultOptimizerBADepth())
{
	parseParameters(parameters);
}

void Optimizer::parseParameters(const ParametersMap & parameters)
{
	Parameters::parse(parameters, Parameters::kOptimizerIterations(), iterations_);
	Parameters::parse(parameters, Parameters::kOptimizerVarianceIgnored(), covarianceIgnored_);
	Parameters::parse(parameters, Parameters::kRegForce3DoF(), slam2d_);
	Parameters::parse(parameters, Parameters::kOptimizerEpsilon(), epsilon_);
	Parameters::parse(parameters, Parameters::kOptimizerRobust(), robust_);
	Parameters::parse(parameters, Parameters::kOptimizerPriorsIgnored(), priorsIgnored_);
	Parameters::parse(parameters, Parameters::kOptimizerLandmarksIgnored(), landmarksIgnored_);
	Parameters::parse(parameters, Parameters::kOptimizerGravitySigma(), gravitySigma_);
	Parameters::parse(parameters, Parameters::kOptimizerLoopRedundancyRadius(), loopRedundancyRadius_);
	UASSERT(loopRedundancyRadius_ >= 0.0);
	Parameters::parse(parameters, Parameters::kOptimizerNeighborLinearStdPerMeter(), neighborLinearStdPerMeter_);
	Parameters::parse(parameters, Parameters::kOptimizerNeighborYawStdPerMeter(), neighborYawStdPerMeter_);
	UASSERT(neighborLinearStdPerMeter_ >= 0.0);
	UASSERT(neighborYawStdPerMeter_ >= 0.0);
	Parameters::parse(parameters, Parameters::kOptimizerLoopLinearStdMin(), loopLinearStdMin_);
	Parameters::parse(parameters, Parameters::kOptimizerLoopAngularStdMin(), loopAngularStdMin_);
	UASSERT(loopLinearStdMin_ >= 0.0);
	UASSERT(loopAngularStdMin_ >= 0.0);
	Parameters::parse(parameters, Parameters::kOptimizerBAStrategy(), baStrategy_);
	Parameters::parse(parameters, Parameters::kOptimizerBAMinTrackLength(), baMinTrackLength_);
	Parameters::parse(parameters, Parameters::kOptimizerBATriangulationMinAngle(), baTriangulationMinAngle_);
	Parameters::parse(parameters, Parameters::kOptimizerBATriangulationMaxReprojError(), baTriangulationMaxReprojError_);
	Parameters::parse(parameters, Parameters::kOptimizerBATriangulationMaxDepthError(), baTriangulationMaxDepthError_);
	Parameters::parse(parameters, Parameters::kOptimizerBATriangulationDepthSeed(), baTriangulationDepthSeed_);
	Parameters::parse(parameters, Parameters::kOptimizerBAWindowBaseline(), baWindowBaseline_);
	Parameters::parse(parameters, Parameters::kOptimizerBAWindowBaselineRatio(), baWindowBaselineRatio_);
	Parameters::parse(parameters, Parameters::kOptimizerBAGuidedMatching(), baGuidedMatching_);
	Parameters::parse(parameters, Parameters::kOptimizerBAGuidedWinSize(), baGuidedWinSize_);
	Parameters::parse(parameters, Parameters::kOptimizerBAGuidedNNDR(), baGuidedNNDR_);
	Parameters::parse(parameters, Parameters::kOptimizerBAGuidedSkipPnP(), baGuidedSkipPnP_);
	Parameters::parse(parameters, Parameters::kOptimizerBAMatchCache(), baMatchCache_);
	Parameters::parse(parameters, Parameters::kOptimizerBAFrameCacheMemory(), baFrameCacheMemory_);
	Parameters::parse(parameters, Parameters::kOptimizerBAProposePairs(), baProposePairs_);
	Parameters::parse(parameters, Parameters::kOptimizerBAProposeMaxRange(), baProposeMaxRange_);
	Parameters::parse(parameters, Parameters::kOptimizerBAProposeMaxAngle(), baProposeMaxAngle_);
	Parameters::parse(parameters, Parameters::kOptimizerBAProposeMinOverlap(), baProposeMinOverlap_);
	Parameters::parse(parameters, Parameters::kOptimizerBAProposeMaxPairs(), baProposeMaxPairs_);
	Parameters::parse(parameters, Parameters::kOptimizerBALoopScope(), baLoopScope_);
	Parameters::parse(parameters, Parameters::kOptimizerBALoopScopeMinTracks(), baLoopScopeMinTracks_);
	Parameters::parse(parameters, Parameters::kOptimizerBAWindowGraphDepth(), baWindowGraphDepth_);
	Parameters::parse(parameters, Parameters::kOptimizerBALoopLinkAsEdge(), baLoopLinkAsEdge_);
	Parameters::parse(parameters, Parameters::kOptimizerBALoopLinkMinObservations(), baLoopLinkMinObservations_);
	Parameters::parse(parameters, Parameters::kOptimizerBADepth(), baDepth_);
}

bool Optimizer::reportProgress(const std::string & msg) const
{
	if(canceled_)
	{
		return false;
	}
	if(progressState_)
	{
		if(progressState_->isCanceled())
		{
			return false;
		}
		if(!progressState_->callback(msg))
		{
			return false;
		}
		if(progressState_->isCanceled())
		{
			return false;
		}
	}
	return true;
}

std::multimap<int, Link> Optimizer::adjustLinkCovariances(
		const std::multimap<int, Link> & linksIn) const
{
	const bool scaleOdometry = neighborLinearStdPerMeter_ > 0.0 || neighborYawStdPerMeter_ > 0.0;
	const bool floorLoops = loopLinearStdMin_ > 0.0 || loopAngularStdMin_ > 0.0;
	if(covarianceIgnored_ || (!scaleOdometry && !floorLoops))
	{
		return linksIn;
	}

	std::multimap<int, Link> links = linksIn;
	int scaled = 0;
	int floored = 0;
	for(std::multimap<int, Link>::iterator iter=links.begin(); iter!=links.end(); ++iter)
	{
		Link & link = iter->second;
		const bool neighbor = link.type() == Link::kNeighbor || link.type() == Link::kNeighborMerged;
		const bool loop = link.type() == Link::kGlobalClosure ||
				link.type() == Link::kLocalSpaceClosure ||
				link.type() == Link::kLocalTimeClosure ||
				link.type() == Link::kUserClosure;
		if(!(neighbor && scaleOdometry) && !(loop && floorLoops))
		{
			continue;
		}
		if(link.transform().isNull() || link.from() == link.to())
		{
			continue;
		}
		const cv::Mat & stored = link.infMatrix();
		if(stored.rows != 6 || stored.cols != 6 || stored.type() != CV_64FC1)
		{
			continue;
		}

		bool diagonal = true;
		for(int r=0; r<6 && diagonal; ++r)
		{
			for(int c=0; c<6; ++c)
			{
				if(r != c && stored.at<double>(r, c) != 0.0)
				{
					diagonal = false;
					break;
				}
			}
		}
		cv::Mat inf = stored.clone();
		cv::Mat cov = diagonal ? cv::Mat() : cv::Mat(stored.inv());
		auto variance = [&](int i) {
			if(diagonal)
			{
				const double information = inf.at<double>(i,i);
				return information > 0.0 && uIsFinite(information) ? 1.0 / information : 0.0;
			}
			return cov.at<double>(i,i);
		};
		auto setVariance = [&](int i, double value) {
			if(diagonal)
			{
				inf.at<double>(i,i) = 1.0 / value;
			}
			else
			{
				cov.at<double>(i,i) = value;
			}
		};
		// 9999 is the sentinel for an ignored axis (first pose, or a degree
		// of freedom the link does not constrain).
		auto adjustable = [&](int i) {
			const double v = variance(i);
			return v > 0.0 && v < 9999.0 && uIsFinite(v);
		};

		bool changed = false;
		if(neighbor)
		{
			const double d = link.transform().getNorm();
			if(!(d > 0.0) || !uIsFinite(d))
			{
				continue;
			}
			const double addLin = neighborLinearStdPerMeter_ * neighborLinearStdPerMeter_ * d * d;
			const double addYaw = neighborYawStdPerMeter_ * neighborYawStdPerMeter_ * d * d;
			for(int i=0; i<3; ++i)
			{
				if(addLin > 0.0 && adjustable(i))
				{
					setVariance(i, variance(i) + addLin);
					changed = true;
				}
			}
			if(addYaw > 0.0 && adjustable(5))
			{
				setVariance(5, variance(5) + addYaw);
				changed = true;
			}
			scaled += changed ? 1 : 0;
		}
		else
		{
			const double minLin = loopLinearStdMin_ * loopLinearStdMin_;
			const double minAng = loopAngularStdMin_ * loopAngularStdMin_;
			for(int i=0; i<6; ++i)
			{
				const double minimum = i < 3 ? minLin : minAng;
				if(minimum > 0.0 && adjustable(i) && variance(i) < minimum)
				{
					setVariance(i, minimum);
					changed = true;
				}
			}
			floored += changed ? 1 : 0;
		}

		if(changed)
		{
			link.setInfMatrix(diagonal ? inf : cv::Mat(cov.inv()));
		}
	}

	UINFO("Link covariances adjusted: %d neighbor links scaled (linear std/m=%g, yaw std/m=%g), "
		  "%d loop closures floored (min linear std=%g m, min angular std=%g rad)",
			scaled, neighborLinearStdPerMeter_, neighborYawStdPerMeter_,
			floored, loopLinearStdMin_, loopAngularStdMin_);
	return links;
}

std::multimap<int, Link> Optimizer::downweightRedundantLoopClosures(
		const std::map<int, Transform> & poses,
		const std::multimap<int, Link> & linksIn) const
{
	if(loopRedundancyRadius_ <= 0.0 || covarianceIgnored_)
	{
		return linksIn;
	}
	UTimer timer;

	// Share the detector's actual-odometry-chain coordinates. Optimizers have
	// no session IDs: connectivity comes only from Neighbor/NeighborMerged
	// edges. Loop closures never join chains for redundancy calculations.
	const odometryGraph::Trajectory arcLength = odometryGraph::trajectory(poses, linksIn, std::map<int,int>());

	std::multimap<int, Link> links = linksIn;

	// Collect loop closures with their endpoint positions along the trajectory,
	// canonicalized by chain ID, then distance (handles swapped directions).
	struct LoopRef
	{
		int chainA;
		int chainB;
		double sa;
		double sb;
		Link * link;
	};
	std::vector<LoopRef> loops;
	for(std::multimap<int, Link>::iterator iter=links.begin(); iter!=links.end(); ++iter)
	{
		if(iter->second.type() == Link::kGlobalClosure ||
		   iter->second.type() == Link::kLocalSpaceClosure ||
		   iter->second.type() == Link::kLocalTimeClosure ||
		   iter->second.type() == Link::kUserClosure)
		{
			auto fromIter = arcLength.find(iter->second.from());
			auto toIter = arcLength.find(iter->second.to());
			if(fromIter != arcLength.end() && toIter != arcLength.end())
			{
				LoopRef ref;
				const std::pair<std::pair<int,double>,std::pair<int,double> > endpoints = std::minmax(
					std::make_pair(fromIter->second.chain, fromIter->second.metres),
					std::make_pair(toIter->second.chain, toIter->second.metres));
				ref.chainA = endpoints.first.first;
				ref.chainB = endpoints.second.first;
				ref.sa = endpoints.first.second;
				ref.sb = endpoints.second.second;
				ref.link = &iter->second;
				loops.push_back(ref);
			}
		}
	}
	if(loops.size() < 2)
	{
		return links;
	}

	// Per-loop density: count for each loop closure how many loop closures
	// (including itself) have both endpoints within the radius of its own
	// endpoints along the trajectory, then divide its information by that
	// count. Total information over a dense group thus saturates at about
	// one loop closure per radius window.
	std::vector<size_t> order(loops.size());
	for(size_t i=0; i<order.size(); ++i)
	{
		order[i] = i;
	}
	std::sort(order.begin(), order.end(), [&loops](size_t a, size_t b) {
		return std::tie(loops[a].chainA, loops[a].chainB, loops[a].sa, loops[a].sb) <
			std::tie(loops[b].chainA, loops[b].chainB, loops[b].sa, loops[b].sb);
	});

	std::vector<int> counts(loops.size(), 1);
	for(size_t i=0; i<order.size(); ++i)
	{
		for(size_t j=i+1; j<order.size(); ++j)
		{
			if(loops[order[j]].chainA != loops[order[i]].chainA ||
			   loops[order[j]].chainB != loops[order[i]].chainB ||
			   loops[order[j]].sa - loops[order[i]].sa > loopRedundancyRadius_)
			{
				break; // sorted by sa: no further candidates for i
			}
			if(fabs(loops[order[j]].sb - loops[order[i]].sb) <= loopRedundancyRadius_)
			{
				++counts[order[i]];
				++counts[order[j]];
			}
		}
	}

	int downweighted = 0;
	int largest = 1;
	for(size_t i=0; i<loops.size(); ++i)
	{
		if(counts[i] > 1)
		{
			loops[i].link->setInfMatrix(loops[i].link->infMatrix() / double(counts[i]));
			++downweighted;
			largest = counts[i] > largest ? counts[i] : largest;
		}
	}

	UINFO("Loop closure redundancy down-weighting: %d/%d loop closures down-weighted "
		  "(max neighbors=%d, radius=%.2f m, %.3fs)",
			downweighted, (int)loops.size(), largest,
			loopRedundancyRadius_, timer.ticks());

	return links;
}

std::map<int, Transform> Optimizer::optimizeIncremental(
		int rootId,
		const std::map<int, Transform> & poses,
		const std::multimap<int, Link> & constraints,
		std::list<std::map<int, Transform> > * intermediateGraphes,
		double * finalError,
		int * iterationsDone)
{
	std::map<int, Transform> incGraph;
	std::multimap<int, Link> incGraphLinks;
	incGraph.insert(*poses.begin());
	int i=0;
	std::multimap<int, Link> constraintsCpy = constraints;
	UDEBUG("Incremental optimization... poses=%d constraints=%d", (int)poses.size(), (int)constraints.size());
	for(std::map<int, Transform>::const_iterator iter=poses.begin(); iter!=poses.end(); ++iter)
	{
		incGraph.insert(*iter);
		bool hasLoopClosure = false;
		for(std::multimap<int, Link>::iterator jter=constraintsCpy.lower_bound(iter->first); jter!=constraintsCpy.end() && jter->first==iter->first; ++jter)
		{
			UDEBUG("%d: %d -> %d type=%d", iter->first, jter->second.from(), jter->second.to(), jter->second.type());
			if(jter->second.type() == Link::kNeighbor || jter->second.type() == Link::kNeighborMerged)
			{
				UASSERT(uContains(incGraph, iter->first));
				incGraph.insert(std::make_pair(jter->second.to(), incGraph.at(iter->first) * jter->second.transform()));
				incGraphLinks.insert(*jter);
			}
			else
			{
				if(!uContains(incGraph, jter->second.to()) && jter->second.to() > iter->first)
				{
					// node not yet in graph, switch link direction
					constraintsCpy.insert(std::make_pair(jter->second.to(), jter->second.inverse()));
				}
				else
				{
					UASSERT(uContains(incGraph, jter->second.to()));
					incGraphLinks.insert(*jter);
					hasLoopClosure = true;
				}
			}
		}
		if(hasLoopClosure)
		{
			incGraph = this->optimize(incGraph.begin()->first, incGraph, incGraphLinks);
			if(incGraph.empty())
			{
				UWARN("Failed incremental optimization... last pose added is %d", iter->first);
				break;
			}
		}
		UDEBUG("Iteration %d/%d %s", ++i, (int)poses.size(), hasLoopClosure?"*":"");
	}
	if(!incGraph.empty() && incGraph.size() == poses.size())
	{
		UASSERT(incGraphLinks.size() == constraints.size());
		UASSERT(uContains(poses, rootId) && uContains(incGraph, rootId));
		incGraph.at(rootId) = poses.at(rootId);
		return this->optimize(rootId, incGraph, incGraphLinks, intermediateGraphes, finalError, iterationsDone);
	}

	UDEBUG("Failed incremental optimization");
	return std::map<int, Transform>();
}

std::map<int, Transform> Optimizer::optimize(
		int rootId,
		const std::map<int, Transform> & poses,
		const std::multimap<int, Link> & edgeConstraints,
		std::list<std::map<int, Transform> > * intermediateGraphes,
		double * finalError,
		int * iterationsDone)
{
	struct CovarianceNotRequested
	{
		explicit CovarianceNotRequested(bool & requested) : requested_(requested) {requested_ = false;}
		~CovarianceNotRequested() {requested_ = true;}
		bool & requested_;
	} guard(covarianceRequested_);
	cv::Mat covariance;
	return optimize(rootId,
			poses,
			edgeConstraints,
			covariance,
			intermediateGraphes,
			finalError,
			iterationsDone);
}

std::map<int, Transform> Optimizer::optimize(
		int rootId,
		const std::map<int, Transform> & poses,
		const std::multimap<int, Link> & constraints,
		cv::Mat & outputCovariance,
		std::list<std::map<int, Transform> > * intermediateGraphes,
		double * finalError,
		int * iterationsDone)
{
	UERROR("Optimizer %d doesn't implement optimize() method.", (int)this->type());
	return std::map<int, Transform>();
}

std::map<int, Transform> Optimizer::optimizeBA(
		int rootId,
		const std::map<int, Transform> & poses,
		const std::multimap<int, Link> & links,
		const std::map<int, std::vector<CameraModel> > & models,
		std::map<int, cv::Point3f> & points3DMap,
		const std::map<int, std::map<int, FeatureBA> > & wordReferences,
		BAOutliers * outliers)
{
	if(outliers)
	{
		outliers->clear();
	}
	UERROR("Optimizer %d doesn't implement optimizeBA() method.", (int)this->type());
	return std::map<int, Transform>();
}

static bool cameraModelsFromData(
		const std::vector<CameraModel> & cameraModels,
		const std::vector<StereoCameraModel> & stereoModels,
		std::vector<CameraModel> & models)
{
	models.clear();
	if(cameraModels.size() >= 1 && cameraModels.at(0).isValidForProjection())
	{
		models = cameraModels;
		return true;
	}
	if(!stereoModels.empty() && stereoModels[0].isValidForProjection())
	{
		for(size_t i = 0; i < stereoModels.size(); ++i)
		{
			CameraModel model = stereoModels[i].left();
			models.push_back(CameraModel(
					model.fx(),
					model.fy(),
					model.cx(),
					model.cy(),
					model.localTransform(),
					-stereoModels[i].baseline() * model.fx(),
					model.imageSize()));
		}
		return true;
	}
	return false;
}

std::map<int, Transform> Optimizer::optimizeBA(
		int rootId,
		const std::map<int, Transform> & posesIn,
		const std::multimap<int, Link> & links,
		const std::map<int, Signature> & signatures,
		std::map<int, cv::Point3f> & points3DMap,
		std::map<int, std::map<int, FeatureBA> > & wordReferences,
		bool rematchFeatures,
		const ParametersMap & registrationParameters,
		BAOutliers * outliers,
		DBDriver * dbDriver)
{
	UDEBUG("");
	resetCancel();
	std::map<int, std::vector<CameraModel> > multiModels;
	std::map<int, Transform> poses;
	for(std::map<int, Transform>::const_iterator iter=posesIn.lower_bound(1); iter!=posesIn.end(); ++iter)
	{
		std::vector<CameraModel> models;
		bool found = false;
		if(uContains(signatures, iter->first))
		{
			const SensorData & s = signatures.at(iter->first).sensorData();
			found = cameraModelsFromData(s.cameraModels(), s.stereoCameraModels(), models);
		}
		else if(dbDriver)
		{
			std::vector<CameraModel> cameraModels;
			std::vector<StereoCameraModel> stereoModels;
			if(dbDriver->getCalibration(iter->first, cameraModels, stereoModels))
			{
				found = cameraModelsFromData(cameraModels, stereoModels, models);
			}
		}
		if(!found)
		{
			UERROR("Missing calibration for node %d", iter->first);
			return std::map<int, Transform>();
		}

		multiModels.insert(std::make_pair(iter->first, models));
		poses.insert(*iter);
	}

	this->computeBACorrespondences(poses, links, signatures, points3DMap, wordReferences, rematchFeatures, false, registrationParameters, dbDriver);
	if(isCanceled() || (progressState_ && progressState_->isCanceled()))
	{
		return std::map<int, Transform>();
	}

	return optimizeBA(rootId, poses, links, multiModels, points3DMap, wordReferences, outliers);
}

std::map<int, Transform> Optimizer::optimizeBA(
		int rootId,
		const std::map<int, Transform> & poses,
		const std::multimap<int, Link> & links,
		const std::map<int, Signature> & signatures,
		bool rematchFeatures,
		const ParametersMap & registrationParameters,
		DBDriver * dbDriver)
{
	std::map<int, cv::Point3f> points3DMap;
	std::map<int, std::map<int, FeatureBA> > wordReferences;
	return optimizeBA(rootId, poses, links, signatures, points3DMap, wordReferences, rematchFeatures, registrationParameters, 0, dbDriver);
}

Transform Optimizer::optimizeBA(
		const Link & link,
		const CameraModel & model,
		std::map<int, cv::Point3f> & points3DMap,
		const std::map<int, std::map<int, FeatureBA> > & wordReferences,
		BAOutliers * outliers)
{
	std::map<int, Transform> poses;
	poses.insert(std::make_pair(link.from(), Transform::getIdentity()));
	poses.insert(std::make_pair(link.to(), link.transform()));
	std::multimap<int, Link> links;
	links.insert(std::make_pair(link.from(), link));
	std::map<int, std::vector<CameraModel> > models;
	std::vector<CameraModel> tmp;
	tmp.push_back(model);
	models.insert(std::make_pair(link.from(), tmp));
	models.insert(std::make_pair(link.to(), tmp));
	poses = optimizeBA(link.from(), poses, links, models, points3DMap, wordReferences, outliers);
	if(poses.size() == 2)
	{
		return poses.rbegin()->second;
	}
	else
	{
		return link.transform();
	}
}

void Optimizer::applyBundleAdjustmentKernel(ParametersMap & parameters)
{
	double delta = Parameters::defaultOptimizerBARobustKernelDelta();
	Parameters::parse(parameters, Parameters::kOptimizerBARobustKernelDelta(), delta);
	if(delta >= 0.0)
	{
		uInsert(parameters, ParametersPair(Parameters::kOptimizerRobustKernelDelta(), uNumber2Str(delta)));
	}
}

} /* namespace rtabmap */
