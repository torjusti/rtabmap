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
		canceled_(false),
		progressState_(0)
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
		canceled_(false),
		progressState_(0)
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

struct KeyPointCompare
{
   bool operator() (const cv::KeyPoint& lhs, const cv::KeyPoint& rhs) const
   {
       return lhs.pt.x < rhs.pt.x || (lhs.pt.x == rhs.pt.x && lhs.pt.y < rhs.pt.y);
   }
};

namespace {

bool isVisualBACorrespondenceLink(const Link & link)
{
	if(link.from() <= 0 || link.to() <= 0 || link.from() == link.to())
	{
		return false;
	}
	switch(link.type())
	{
	case Link::kPosePrior:
	case Link::kLandmark:
	case Link::kGravity:
	case Link::kVirtualClosure:
		return false;
	default:
		return true;
	}
}

bool hasCameraModel(const Signature & s)
{
	return !s.sensorData().cameraModels().empty() || !s.sensorData().stereoCameraModels().empty();
}

struct BACorrCandidate
{
	Link link;
	int fromId;
	int toId;
};

struct BACorrInlier
{
	int indexFrom;
	int indexTo;
	cv::KeyPoint ptFrom;
	cv::KeyPoint ptTo;
	cv::Point3f pFrom;
	int cameraIndexFrom;
	int cameraIndexTo;
	float depthFrom;
	float depthTo;
};

struct BACorrMatch
{
	bool ok;
	int fromId;
	int toId;
	std::vector<BACorrInlier> inliers;
	BACorrMatch() : ok(false), fromId(0), toId(0) {}
};

// Words + calibration only. Copying a full Signature also duplicates every
// graph link on the node and, after RegistrationVis::setWords(), pins a
// cloned descriptor matrix for as long as the copy lives.
Signature makeBAWorkSignature(const Signature & src, bool keepDescriptors)
{
	Signature s(src.id(), src.mapId(), src.getWeight(), src.getStamp());
	s.setWords(
			src.getWords(),
			src.getWordsKpts(),
			src.getWords3(),
			keepDescriptors ? src.getWordsDescriptors() : cv::Mat(),
			keepDescriptors);
	if(!src.sensorData().cameraModels().empty())
	{
		s.sensorData().setCameraModels(src.sensorData().cameraModels());
	}
	if(!src.sensorData().stereoCameraModels().empty())
	{
		s.sensorData().setStereoCameraModels(src.sensorData().stereoCameraModels());
	}
	return s;
}

void observationGeometry(
		const Signature & s,
		cv::KeyPoint & pt,
		const cv::Point3f & p,
		int & cameraIndex,
		float & depth)
{
	cameraIndex = 0;
	if(s.sensorData().cameraModels().size() > 1 || s.sensorData().stereoCameraModels().size() > 1)
	{
		const float subImageWidth = s.sensorData().cameraModels().size() > 1 ?
				s.sensorData().cameraModels()[0].imageWidth() :
				s.sensorData().stereoCameraModels()[0].left().imageWidth();
		cameraIndex = int(pt.pt.x / subImageWidth);
		pt.pt.x = pt.pt.x - (subImageWidth * float(cameraIndex));
	}
	depth = 0.0f;
	if(!s.sensorData().cameraModels().empty())
	{
		depth = util3d::transformPoint(p, s.sensorData().cameraModels()[cameraIndex].localTransform().inverse()).z;
	}
	else
	{
		UASSERT(!s.sensorData().stereoCameraModels().empty());
		depth = util3d::transformPoint(p, s.sensorData().stereoCameraModels()[cameraIndex].localTransform().inverse()).z;
	}
}

void extractBAInliers(
		const Signature & sFrom,
		const Signature & sTo,
		const RegistrationInfo & info,
		std::vector<BACorrInlier> & out)
{
	out.clear();
	out.reserve(info.inliersIDs.size());
	for(unsigned int i = 0; i < info.inliersIDs.size(); ++i)
	{
		const int indexFrom = sFrom.getWords().lower_bound(info.inliersIDs[i])->second;
		const cv::Point3f & p = sFrom.getWords3()[indexFrom];
		if(p.x <= 0.0f)
		{
			continue;
		}
		BACorrInlier inlier;
		inlier.indexFrom = indexFrom;
		inlier.indexTo = sTo.getWords().lower_bound(info.inliersIDs[i])->second;
		inlier.ptFrom = sFrom.getWordsKpts()[indexFrom];
		inlier.ptTo = sTo.getWordsKpts()[inlier.indexTo];
		inlier.pFrom = p;
		observationGeometry(sFrom, inlier.ptFrom, p, inlier.cameraIndexFrom, inlier.depthFrom);
		cv::Point3f pTo(0, 0, 0);
		if(!sTo.getWords3().empty())
		{
			UASSERT(inlier.indexTo < (int)sTo.getWords3().size());
			pTo = sTo.getWords3()[inlier.indexTo];
		}
		observationGeometry(sTo, inlier.ptTo, pTo, inlier.cameraIndexTo, inlier.depthTo);
		if(sTo.getWords3().empty())
		{
			inlier.depthTo = 0.0f;
		}
		out.push_back(inlier);
	}
}

class BAFeatureCache
{
public:
	BAFeatureCache(
			const std::map<int, Signature> * loaded,
			DBDriver * db,
			unsigned long maxBytes) :
		loaded_(loaded),
		db_(db),
		maxBytes_(maxBytes),
		bytes_(0)
	{}

	bool get(int id, Signature & out)
	{
		std::lock_guard<std::mutex> lock(mutex_);
		std::map<int, CacheEntry>::iterator it = cache_.find(id);
		if(it != cache_.end())
		{
			lru_.erase(it->second.lruIt);
			lru_.push_front(id);
			it->second.lruIt = lru_.begin();
			out = it->second.signature;
			return true;
		}
		Signature loaded;
		if(!loadUnlocked(id, loaded))
		{
			return false;
		}
		const unsigned long sz = std::max(1ul, loaded.getMemoryUsed(true));
		evictUnlocked(sz);
		lru_.push_front(id);
		CacheEntry entry;
		entry.signature = loaded;
		entry.lruIt = lru_.begin();
		entry.bytes = sz;
		cache_.insert(std::make_pair(id, entry));
		bytes_ += sz;
		out = loaded;
		return true;
	}

private:
	struct CacheEntry
	{
		Signature signature;
		std::list<int>::iterator lruIt;
		unsigned long bytes;
	};

	bool loadUnlocked(int id, Signature & out)
	{
		if(loaded_)
		{
			std::map<int, Signature>::const_iterator it = loaded_->find(id);
			if(it != loaded_->end())
			{
				out = makeBAWorkSignature(it->second, true);
				return true;
			}
		}
		if(!db_)
		{
			return false;
		}
		std::multimap<int, int> words;
		std::vector<cv::KeyPoint> kpts;
		std::vector<cv::Point3f> pts;
		cv::Mat desc;
		db_->getLocalFeatures(id, words, kpts, pts, desc);
		std::vector<CameraModel> models;
		std::vector<StereoCameraModel> stereo;
		db_->getCalibration(id, models, stereo);
		Transform pose;
		Transform gt;
		int mapId = 0;
		int weight = 0;
		std::string label;
		double stamp = 0.0;
		std::vector<float> vel;
		GPS gps;
		EnvSensors sensors;
		db_->getNodeInfo(id, pose, mapId, weight, label, stamp, gt, vel, gps, sensors);
		out = Signature(id, mapId, weight, stamp);
		out.setWords(words, kpts, pts, desc, false);
		if(!models.empty())
		{
			out.sensorData().setCameraModels(models);
		}
		if(!stereo.empty())
		{
			out.sensorData().setStereoCameraModels(stereo);
		}
		return true;
	}

	void evictUnlocked(unsigned long incoming)
	{
		while(!cache_.empty() && bytes_ + incoming > maxBytes_ && cache_.size() > 2)
		{
			const int old = lru_.back();
			lru_.pop_back();
			std::map<int, CacheEntry>::iterator it = cache_.find(old);
			if(it != cache_.end())
			{
				if(bytes_ >= it->second.bytes)
				{
					bytes_ -= it->second.bytes;
				}
				else
				{
					bytes_ = 0;
				}
				cache_.erase(it);
			}
		}
	}

	const std::map<int, Signature> * loaded_;
	DBDriver * db_;
	unsigned long maxBytes_;
	unsigned long bytes_;
	std::mutex mutex_;
	std::list<int> lru_;
	std::map<int, CacheEntry> cache_;
};

void mergeBACorrMatches(
		std::vector<BACorrMatch> & matches,
		const std::map<int, Transform> & poses,
		std::map<int, cv::Point3f> & points3DMap,
		std::map<int, std::map<int, FeatureBA> > & wordReferences,
		std::map<int, std::map<cv::KeyPoint, int, KeyPointCompare> > & frameToWordMap,
		int & wordCount,
		int & edgeWithWordsAdded)
{
	for(size_t mi = 0; mi < matches.size(); ++mi)
	{
		BACorrMatch & match = matches[mi];
		if(!match.ok || match.inliers.empty())
		{
			continue;
		}

		std::map<int, Transform>::const_iterator poseIter = poses.find(match.fromId);
		if(poseIter == poses.end())
		{
			continue;
		}
		Transform pose = poseIter->second;
		UASSERT(!pose.isNull());
		for(size_t i = 0; i < match.inliers.size(); ++i)
		{
			BACorrInlier & inlier = match.inliers[i];
			int wordId = -1;

			std::map<int, std::map<cv::KeyPoint, int, KeyPointCompare> >::iterator fromIter = frameToWordMap.find(match.fromId);
			std::map<int, std::map<cv::KeyPoint, int, KeyPointCompare> >::iterator toIter = frameToWordMap.find(match.toId);
			bool fromAlreadyAdded = false;
			bool toAlreadyAdded = false;
			if( fromIter != frameToWordMap.end() &&
				fromIter->second.find(inlier.ptFrom) != fromIter->second.end())
			{
				wordId = fromIter->second.at(inlier.ptFrom);
				fromAlreadyAdded = true;
			}
			if( toIter != frameToWordMap.end() &&
				toIter->second.find(inlier.ptTo) != toIter->second.end())
			{
				wordId = toIter->second.at(inlier.ptTo);
				toAlreadyAdded = true;
			}

			if(wordId == -1)
			{
				wordId = ++wordCount;
				wordReferences.insert(std::make_pair(wordId, std::map<int, FeatureBA>()));
				points3DMap.insert(std::make_pair(wordId, util3d::transformPoint(inlier.pFrom, pose)));
			}
			else
			{
				UASSERT(wordReferences.find(wordId) != wordReferences.end());
				UASSERT(points3DMap.find(wordId) != points3DMap.end());
			}

			if(!fromAlreadyAdded)
			{
				wordReferences.at(wordId).insert(std::make_pair(match.fromId, FeatureBA(inlier.ptFrom, inlier.depthFrom, cv::Mat(), inlier.cameraIndexFrom, inlier.indexFrom)));
				frameToWordMap.insert(std::make_pair(match.fromId, std::map<cv::KeyPoint, int, KeyPointCompare>()));
				frameToWordMap.at(match.fromId).insert(std::make_pair(inlier.ptFrom, wordId));
			}

			if(!toAlreadyAdded)
			{
				wordReferences.at(wordId).insert(std::make_pair(match.toId, FeatureBA(inlier.ptTo, inlier.depthTo, cv::Mat(), inlier.cameraIndexTo, inlier.indexTo)));
				frameToWordMap.insert(std::make_pair(match.toId, std::map<cv::KeyPoint, int, KeyPointCompare>()));
				frameToWordMap.at(match.toId).insert(std::make_pair(inlier.ptTo, wordId));
			}
		}
		++edgeWithWordsAdded;
		match.inliers.clear();
		match.inliers.shrink_to_fit();
	}
}

} // namespace

void Optimizer::computeBACorrespondences(
		const std::map<int, Transform> & poses,
		const std::multimap<int, Link> & links,
		const std::map<int, Signature> & signatures,
		std::map<int, cv::Point3f> & points3DMap,
		std::map<int, std::map<int, FeatureBA> > & wordReferences,
		bool rematchFeatures,
		bool useLinkTransformAsGuess,
		ParametersMap registrationParameters,
		DBDriver * dbDriver)
{
	UDEBUG("rematchFeatures=%d dbDriver=%d", rematchFeatures?1:0, dbDriver?1:0);
	int wordCount = 0;
	int edgeWithWordsAdded = 0;

	registrationParameters.insert(ParametersPair(Parameters::kVisEstimationType(), "1"));
	registrationParameters.insert(ParametersPair(Parameters::kVisPnPReprojError(), "5"));
	registrationParameters.insert(ParametersPair(Parameters::kVisMinInliers(), "6"));
	registrationParameters.insert(ParametersPair(Parameters::kVisCorNNDR(), "0.6"));

	BAFeatureCache cache(&signatures, dbDriver, 512ul * 1024ul * 1024ul);
	std::map<int, int> weightCache;
	auto weightOf = [&](int id) -> int
	{
		std::map<int, int>::const_iterator it = weightCache.find(id);
		if(it != weightCache.end())
		{
			return it->second;
		}
		int weight = 0;
		std::map<int, Signature>::const_iterator siter = signatures.find(id);
		if(siter != signatures.end())
		{
			weight = siter->second.getWeight();
		}
		else if(dbDriver)
		{
			dbDriver->getWeight(id, weight);
		}
		else
		{
			weight = -1;
		}
		weightCache.insert(std::make_pair(id, weight));
		return weight;
	};

	std::vector<BACorrCandidate> candidates;
	std::set<std::pair<int, int> > seenPairs;
	candidates.reserve(links.size());
	for(std::multimap<int, Link>::const_iterator iter=links.lower_bound(1); iter!=links.end(); ++iter)
	{
		Link link = iter->second;
		if(!isVisualBACorrespondenceLink(link))
		{
			continue;
		}
		if(link.to() < link.from())
		{
			link = link.inverse();
		}
		if(!seenPairs.insert(std::make_pair(link.from(), link.to())).second)
		{
			continue;
		}
		if(poses.find(link.from()) == poses.end())
		{
			continue;
		}
		if(weightOf(link.from()) < 0)
		{
			continue;
		}
		int toId = link.to();
		if(weightOf(toId) < 0)
		{
			bool resolved = false;
			for(std::multimap<int, Link>::const_iterator jter=links.find(toId);
				jter != links.end();
				++jter)
			{
				if(weightOf(jter->second.to()) >= 0)
				{
					toId = jter->second.to();
					resolved = true;
					break;
				}
			}
			if(!resolved)
			{
				continue;
			}
		}

		BACorrCandidate candidate;
		candidate.link = link;
		candidate.fromId = link.from();
		candidate.toId = toId;
		candidates.push_back(candidate);
	}
	std::sort(candidates.begin(), candidates.end(), [](const BACorrCandidate & a, const BACorrCandidate & b)
	{
		if(a.fromId != b.fromId)
		{
			return a.fromId < b.fromId;
		}
		return a.toId < b.toId;
	});

	const int totalLinks = (int)candidates.size();
	if(!reportProgress(uFormat("Correspondences 0/%d", totalLinks)))
	{
		points3DMap.clear();
		wordReferences.clear();
		canceled_ = true;
		return;
	}

	std::atomic<int> processed(0);
	std::atomic<bool> abort(false);
	UTimer progressTimer;
	int nnType = Parameters::defaultVisCorNNType();
	Parameters::parse(registrationParameters, Parameters::kVisCorNNType(), nnType);
	int threads = 1;
#ifdef _OPENMP
	if(nnType != 4)
	{
		threads = std::max(1, omp_get_max_threads());
	}
#endif

	std::map<int, std::map<cv::KeyPoint, int, KeyPointCompare> > frameToWordMap;
	const int chunkSize = 256;
	for(int start = 0; start < totalLinks && !abort.load(); start += chunkSize)
	{
		const int end = std::min(start + chunkSize, totalLinks);
		std::vector<BACorrMatch> matches(end - start);

#ifdef _OPENMP
#pragma omp parallel num_threads(threads)
#endif
		{
			RegistrationVis reg(registrationParameters);
#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
			for(int i = start; i < end; ++i)
			{
				if(abort.load())
				{
					continue;
				}

				const BACorrCandidate & candidate = candidates[i];
				Signature fromSrc;
				Signature toSrc;
				if(!cache.get(candidate.fromId, fromSrc) || !cache.get(candidate.toId, toSrc))
				{
					continue;
				}
				if(!hasCameraModel(fromSrc) || !hasCameraModel(toSrc) ||
				   fromSrc.getWordsKpts().empty() || toSrc.getWordsKpts().empty() || fromSrc.getWords3().empty())
				{
					continue;
				}

				BACorrMatch & out = matches[i - start];
				out.fromId = candidate.fromId;
				out.toId = candidate.toId;
				Signature sFrom = makeBAWorkSignature(fromSrc, rematchFeatures);
				Signature sTo = makeBAWorkSignature(toSrc, rematchFeatures);
				if(rematchFeatures && sFrom.getWordsDescriptors().empty() && sTo.getWordsDescriptors().empty())
				{
					UWARN("Rematching features is enabled but signatures (%d and %d) don't have word descriptors!? "
						  "Features won't be rematched. If it is an old database, do rtabmap-reprocess "
						  "so that signatures contain word desriptors.",
						  sFrom.id(), sTo.id());
				}

				Transform guess;
				if(useLinkTransformAsGuess)
				{
					guess = candidate.link.transform();
				}
				else
				{
					std::map<int, Transform>::const_iterator pFrom = poses.find(candidate.fromId);
					std::map<int, Transform>::const_iterator pTo = poses.find(candidate.toId);
					if(pFrom != poses.end() && pTo != poses.end() &&
					   !pFrom->second.isNull() && !pTo->second.isNull())
					{
						guess = pFrom->second.inverse() * pTo->second;
					}
				}

				RegistrationInfo info;
				Transform t = reg.computeTransformationMod(sFrom, sTo, guess, &info);
				UDEBUG("%d->%d, inliers=%d", sFrom.id(), sTo.id(), (int)info.inliersIDs.size());
				if(!t.isNull())
				{
					out.ok = true;
					extractBAInliers(sFrom, sTo, info, out.inliers);
				}
				else
				{
					UWARN("Not enough inliers (%d) between %d and %d", (int)info.inliersIDs.size(), sFrom.id(), sTo.id());
				}

				const int n = ++processed;
				if(n == 1 || n == totalLinks || (n % 16) == 0)
				{
#ifdef _OPENMP
#pragma omp critical(ba_corr_progress)
#endif
					{
						if(!abort.load() &&
						   (n == 1 || n == totalLinks || progressTimer.getElapsedTime() >= 0.05))
						{
							if(!reportProgress(uFormat("Correspondences %d/%d", n, totalLinks)))
							{
								abort = true;
								canceled_ = true;
							}
							progressTimer.restart();
						}
					}
				}
			}
		}

		if(abort.load() || isCanceled() || (progressState_ && progressState_->isCanceled()))
		{
			points3DMap.clear();
			wordReferences.clear();
			canceled_ = true;
			return;
		}
		mergeBACorrMatches(matches, poses, points3DMap, wordReferences, frameToWordMap, wordCount, edgeWithWordsAdded);
	}
	UDEBUG("Added %d words (edges with words=%d/%d)", wordCount, edgeWithWordsAdded, totalLinks);
	if(links.empty())
	{
		UERROR("No links found for BA?!");
	}
	else if(wordCount == 0)
	{
		UERROR("No words added for BA?!");
	}
}

} /* namespace rtabmap */
