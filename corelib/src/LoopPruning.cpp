/*
Copyright (c) 2010-2026, Mathieu Labbe - IntRoLab - Universite de Sherbrooke
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

#include <rtabmap/core/LoopPruning.h>
#include <rtabmap/core/Graph.h>
#include <rtabmap/core/Optimizer.h>
#include <rtabmap/core/ProgressState.h>
#include <rtabmap/utilite/ULogger.h>
#include <rtabmap/utilite/UMath.h>
#include <rtabmap/utilite/UStl.h>
#include <rtabmap/utilite/UTimer.h>
#include <rtabmap/utilite/UConversion.h>
#include <algorithm>
#include <cmath>
#include <deque>
#include <functional>
#include <iterator>
#include <limits>
#include <queue>
#include <tuple>
#include <unordered_map>

namespace rtabmap {
namespace loopPruning {

namespace {

std::pair<int, int> pairKey(const Link & link)
{
	return std::minmax(link.from(), link.to());
}

bool report(const ProgressState * progress, const std::string & msg)
{
	UINFO("%s", msg.c_str());
	if(progress)
	{
		progress->callback(msg);
		return !progress->isCanceled();
	}
	return true;
}

int countLoops(const std::multimap<int, Link> & links, bool includeUserClosures)
{
	std::set<std::pair<int, int> > loops;
	for(std::multimap<int, Link>::const_iterator iter=links.begin(); iter!=links.end(); ++iter)
	{
		if(isPrunableLoop(iter->second, includeUserClosures))
		{
			loops.insert(pairKey(iter->second));
		}
	}
	return (int)loops.size();
}

void eraseLink(std::multimap<int, Link> & links, const Link & link)
{
	const std::pair<int, int> key = pairKey(link);
	for(int id : {key.first, key.second})
	{
		for(std::multimap<int, Link>::iterator iter=links.lower_bound(id); iter!=links.end() && iter->first==id;)
		{
			if(iter->second.from() != iter->second.to() &&
			   iter->second.type() == link.type() &&
			   pairKey(iter->second) == key)
			{
				iter = links.erase(iter);
			}
			else
			{
				++iter;
			}
		}
	}
}

// Undirected multigraph over all non-self links, for bridge checks: a link
// is a bridge iff its endpoints are disconnected once it is removed.
class Connectivity
{
public:
	explicit Connectivity(const std::multimap<int, Link> & links)
	{
		for(std::multimap<int, Link>::const_iterator iter=links.begin(); iter!=links.end(); ++iter)
		{
			const Link & link = iter->second;
			if(link.from() == link.to())
			{
				continue;
			}
			int a = index(link.from());
			int b = index(link.to());
			alive_.push_back(true);
			edges_.push_back(std::make_pair(a, b));
			adjacency_[a].push_back((int)edges_.size()-1);
			adjacency_[b].push_back((int)edges_.size()-1);
			byKey_[std::make_tuple(std::min(link.from(), link.to()), std::max(link.from(), link.to()), (int)link.type())].push_back((int)edges_.size()-1);
		}
	}

	// Removes the link unless that disconnects its endpoints. Returns false
	// (link kept) if it is a bridge.
	bool removeIfNotBridge(const Link & link)
	{
		std::map<std::tuple<int,int,int>, std::vector<int> >::iterator found = byKey_.find(
				std::make_tuple(std::min(link.from(), link.to()), std::max(link.from(), link.to()), (int)link.type()));
		if(found == byKey_.end())
		{
			return true;
		}
		for(int e : found->second)
		{
			alive_[e] = false;
		}
		if(!connected(edges_[found->second.front()].first, edges_[found->second.front()].second))
		{
			for(int e : found->second)
			{
				alive_[e] = true;
			}
			return false;
		}
		byKey_.erase(found);
		return true;
	}

private:
	int index(int id)
	{
		std::unordered_map<int, int>::iterator iter = ids_.find(id);
		if(iter != ids_.end())
		{
			return iter->second;
		}
		int i = (int)adjacency_.size();
		ids_.insert(std::make_pair(id, i));
		adjacency_.push_back(std::vector<int>());
		return i;
	}

	bool connected(int from, int to) const
	{
		if(from == to)
		{
			return true;
		}
		std::vector<bool> visited(adjacency_.size(), false);
		std::deque<int> queue;
		queue.push_back(from);
		visited[from] = true;
		while(!queue.empty())
		{
			int current = queue.front();
			queue.pop_front();
			for(int e : adjacency_[current])
			{
				if(!alive_[e])
				{
					continue;
				}
				int next = edges_[e].first == current ? edges_[e].second : edges_[e].first;
				if(next == to)
				{
					return true;
				}
				if(!visited[next])
				{
					visited[next] = true;
					queue.push_back(next);
				}
			}
		}
		return false;
	}

	std::unordered_map<int, int> ids_;
	std::vector<std::vector<int> > adjacency_;
	std::vector<std::pair<int, int> > edges_;
	std::vector<bool> alive_;
	std::map<std::tuple<int,int,int>, std::vector<int> > byKey_;
};

float closestEndpointDistance(const Link & a, const Link & b, const std::map<int, Transform> & poses)
{
	float best = std::numeric_limits<float>::max();
	for(int i : {a.from(), a.to()})
	{
		std::map<int, Transform>::const_iterator pi = poses.find(i);
		if(pi == poses.end())
		{
			continue;
		}
		for(int j : {b.from(), b.to()})
		{
			std::map<int, Transform>::const_iterator pj = poses.find(j);
			if(pj != poses.end())
			{
				best = std::min(best, pi->second.getDistance(pj->second));
			}
		}
	}
	return best;
}

} // namespace

bool isPrunableLoop(const Link & link, bool includeUserClosures)
{
	if(link.from() == link.to() || link.from() <= 0 || link.to() <= 0)
	{
		return false;
	}
	return link.type() == Link::kGlobalClosure ||
		   link.type() == Link::kLocalSpaceClosure ||
		   link.type() == Link::kLocalTimeClosure ||
		   (includeUserClosures && link.type() == Link::kUserClosure);
}

std::vector<LoopResidual> computeLoopResiduals(
		const std::map<int, Transform> & optimizedPoses,
		const std::multimap<int, Link> & links,
		bool includeUserClosures,
		bool force3DoF)
{
	std::vector<LoopResidual> residuals;
	std::set<std::pair<int, int> > seen;
	for(std::multimap<int, Link>::const_iterator iter=links.begin(); iter!=links.end(); ++iter)
	{
		const Link & link = iter->second;
		if(!isPrunableLoop(link, includeUserClosures) || !seen.insert(pairKey(link)).second)
		{
			continue;
		}
		std::map<int, Transform>::const_iterator fromIter = optimizedPoses.find(link.from());
		std::map<int, Transform>::const_iterator toIter = optimizedPoses.find(link.to());
		if(fromIter == optimizedPoses.end() || toIter == optimizedPoses.end() || link.transform().isNull())
		{
			continue;
		}
		const Transform t = fromIter->second.inverse() * toIter->second;
		const Transform & linkT = link.transform();

		LoopResidual r;
		r.link = link;
		r.linear = uMax3(
				fabs(linkT.x() - t.x()),
				fabs(linkT.y() - t.y()),
				force3DoF?0.0f:fabs(linkT.z() - t.z()));
		if(force3DoF)
		{
			float angle = fabs(t.theta() - linkT.theta());
			r.angular = angle > M_PI ? 2*M_PI - angle : angle;
		}
		else
		{
			r.angular = t.getAngle(linkT);
		}
		const double transVariance = link.transVariance(false);
		const double rotVariance = link.rotVariance(false);
		r.linearRatio = transVariance > 0.0 ? r.linear / sqrt(transVariance) : 0.0f;
		r.angularRatio = rotVariance > 0.0 ? r.angular / sqrt(rotVariance) : 0.0f;
		residuals.push_back(r);
	}
	return residuals;
}

std::vector<Link> findRedundantLoops(
		const std::multimap<int, Link> & links,
		float minGraphDistance,
		bool includeUserClosures,
		const std::map<std::pair<int, int>, float> & rank)
{
	std::vector<Link> redundant;
	if(minGraphDistance <= 0.0f)
	{
		return redundant;
	}

	// Undirected graph weighted by link translation, loop closures added as
	// they are kept.
	std::unordered_map<int, std::vector<std::pair<int, float> > > adjacency;
	auto addEdge = [&adjacency](const Link & link) {
		const float length = link.transform().getNorm();
		adjacency[link.from()].push_back(std::make_pair(link.to(), length));
		adjacency[link.to()].push_back(std::make_pair(link.from(), length));
	};

	struct Candidate
	{
		float rank;
		Link link;
	};
	std::vector<Candidate> candidates;
	std::set<std::tuple<int, int, int> > seen;
	for(std::multimap<int, Link>::const_iterator iter=links.begin(); iter!=links.end(); ++iter)
	{
		const Link & link = iter->second;
		if(link.from() == link.to() || link.transform().isNull() ||
		   !seen.insert(std::make_tuple(std::min(link.from(), link.to()), std::max(link.from(), link.to()), (int)link.type())).second)
		{
			continue;
		}
		if(isPrunableLoop(link, includeUserClosures))
		{
			std::map<std::pair<int, int>, float>::const_iterator rankIter = rank.find(pairKey(link));
			Candidate c;
			c.rank = rankIter != rank.end() ? rankIter->second : std::numeric_limits<float>::max();
			c.link = link;
			candidates.push_back(c);
		}
		else
		{
			addEdge(link);
		}
	}

	// Ties broken by node ids so the result does not depend on link order.
	std::sort(candidates.begin(), candidates.end(), [](const Candidate & a, const Candidate & b) {
		return std::make_tuple(a.rank, a.link.from(), a.link.to()) < std::make_tuple(b.rank, b.link.from(), b.link.to());
	});

	// Dijkstra from one node, not expanding past minGraphDistance.
	auto joinedWithin = [&adjacency, minGraphDistance](int from, int to) {
		std::unordered_map<int, float> distance;
		typedef std::pair<float, int> Entry;
		std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry> > queue;
		distance[from] = 0.0f;
		queue.push(Entry(0.0f, from));
		while(!queue.empty())
		{
			Entry current = queue.top();
			queue.pop();
			if(current.second == to)
			{
				return true;
			}
			if(current.first > distance[current.second])
			{
				continue;
			}
			std::unordered_map<int, std::vector<std::pair<int, float> > >::const_iterator neighbors = adjacency.find(current.second);
			if(neighbors == adjacency.end())
			{
				continue;
			}
			for(const std::pair<int, float> & edge : neighbors->second)
			{
				const float d = current.first + edge.second;
				if(d >= minGraphDistance)
				{
					continue;
				}
				std::unordered_map<int, float>::iterator known = distance.find(edge.first);
				if(known == distance.end() || d < known->second)
				{
					distance[edge.first] = d;
					queue.push(Entry(d, edge.first));
				}
			}
		}
		return false;
	};

	for(size_t i=0; i<candidates.size(); ++i)
	{
		const Link & link = candidates[i].link;
		if(joinedWithin(link.from(), link.to()))
		{
			redundant.push_back(link);
		}
		else
		{
			addEdge(link);
		}
	}
	return redundant;
}

int largestComponentRoot(
		const std::map<int, Transform> & odomPoses,
		const std::multimap<int, Link> & links)
{
	std::unordered_map<int, int> parent;
	std::function<int(int)> find = [&](int id) {
		std::unordered_map<int, int>::iterator iter = parent.find(id);
		if(iter == parent.end())
		{
			parent.insert(std::make_pair(id, id));
			return id;
		}
		int root = id;
		while(parent.at(root) != root)
		{
			root = parent.at(root);
		}
		while(parent.at(id) != root)
		{
			int next = parent.at(id);
			parent[id] = root;
			id = next;
		}
		return root;
	};
	for(std::map<int, Transform>::const_iterator iter=odomPoses.lower_bound(1); iter!=odomPoses.end(); ++iter)
	{
		find(iter->first);
	}
	for(std::multimap<int, Link>::const_iterator iter=links.begin(); iter!=links.end(); ++iter)
	{
		const Link & link = iter->second;
		if(link.from() == link.to())
		{
			continue;
		}
		// Only follow links whose nodes have a pose, as getConnectedGraph() does.
		if((link.from() > 0 && odomPoses.find(link.from()) == odomPoses.end()) ||
		   (link.to() > 0 && odomPoses.find(link.to()) == odomPoses.end()))
		{
			continue;
		}
		int a = find(link.from());
		int b = find(link.to());
		if(a != b)
		{
			parent[a] = b;
		}
	}

	std::map<int, std::pair<int, int> > components; // root -> (node count, largest node id)
	for(std::map<int, Transform>::const_iterator iter=odomPoses.lower_bound(1); iter!=odomPoses.end(); ++iter)
	{
		std::pair<int, int> & c = components[find(iter->first)];
		++c.first;
		c.second = std::max(c.second, iter->first);
	}
	std::pair<int, int> best(0, 0);
	for(std::map<int, std::pair<int, int> >::iterator iter=components.begin(); iter!=components.end(); ++iter)
	{
		if(iter->second > best)
		{
			best = iter->second;
		}
	}
	return best.second;
}

namespace {
struct StageTimes
{
	double seed = 0.0;
	double connectedGraph = 0.0;
	double guess = 0.0;
	double optimize = 0.0;
	int iterations = 0;
	bool fullSolve = false;
	double residuals = 0.0;
	double select = 0.0;
	double connectivity = 0.0;
	double bridges = 0.0;
	double erase = 0.0;
};

std::map<int, Transform> optimizeComponentTimed(
		Optimizer & optimizer,
		int rootId,
		const std::map<int, Transform> & odomPoses,
		const std::multimap<int, Link> & links,
		const std::map<int, Transform> & guess,
		std::multimap<int, Link> * linksOut,
		StageTimes * times)
{
	UTimer timer;
	StageTimes unused;
	StageTimes & t = times ? *times : unused;
	std::map<int, Transform> posesIn = odomPoses;
	if(posesIn.find(rootId) == posesIn.end())
	{
		return std::map<int, Transform>();
	}
	// getConnectedGraph() only follows landmarks that already have a pose.
	for(std::multimap<int, Link>::const_iterator iter=links.begin(); iter!=links.end(); ++iter)
	{
		const Link & link = iter->second;
		if(link.type() != Link::kLandmark)
		{
			continue;
		}
		if(link.from() > 0 && link.to() < 0 &&
		   posesIn.find(link.from()) != posesIn.end() &&
		   posesIn.find(link.to()) == posesIn.end())
		{
			posesIn.insert(std::make_pair(link.to(), posesIn.at(link.from())*link.transform()));
		}
		else if(link.from() < 0 && link.to() > 0 &&
		   posesIn.find(link.to()) != posesIn.end() &&
		   posesIn.find(link.from()) == posesIn.end())
		{
			posesIn.insert(std::make_pair(link.from(), posesIn.at(link.to())*link.transform().inverse()));
		}
	}

	t.seed = timer.ticks();

	std::map<int, Transform> poses;
	std::multimap<int, Link> optLinks;
	optimizer.getConnectedGraph(rootId, posesIn, links, poses, optLinks);
	t.connectedGraph = timer.ticks();
	if(poses.empty())
	{
		return poses;
	}

	if(!guess.empty())
	{
		for(std::map<int, Transform>::iterator iter=poses.begin(); iter!=poses.end(); ++iter)
		{
			std::map<int, Transform>::const_iterator found = guess.find(iter->first);
			if(found != guess.end())
			{
				iter->second = found->second;
			}
		}
	}
	else if(!optimizer.priorsIgnored())
	{
		// Iterative optimizers can diverge when the initial guess is far from
		// the priors (e.g. site CRS coordinates).
		Transform align = graph::alignPosesToLandmarkPriors(poses, optLinks);
		if(!align.isNull() && (align.getNorm() > 0.5 || fabs(align.theta()) > 0.05))
		{
			for(std::map<int, Transform>::iterator iter=poses.begin(); iter!=poses.end(); ++iter)
			{
				iter->second = align * iter->second;
			}
		}
	}

	t.guess = timer.ticks();

	std::map<int, Transform> optimized = optimizer.optimize(rootId, poses, optLinks, 0, 0, &t.iterations);
	t.optimize = timer.ticks();
	if(linksOut)
	{
		*linksOut = optLinks;
	}
	return optimized;
}
} // namespace

std::map<int, Transform> optimizeComponent(
		Optimizer & optimizer,
		int rootId,
		const std::map<int, Transform> & odomPoses,
		const std::multimap<int, Link> & links,
		const std::map<int, Transform> & guess,
		std::multimap<int, Link> * linksOut)
{
	return optimizeComponentTimed(optimizer, rootId, odomPoses, links, guess, linksOut, 0);
}

Result prune(
		Optimizer & optimizer,
		const std::map<int, Transform> & odomPoses,
		const std::multimap<int, Link> & linksIn,
		const Options & options,
		const ProgressState * progress)
{
	Result result;
	UTimer timer;

	// One entry per unordered node pair and type, self links as they are.
	std::multimap<int, Link> working;
	std::set<std::tuple<int, int, int> > seen;
	for(std::multimap<int, Link>::const_iterator iter=linksIn.begin(); iter!=linksIn.end(); ++iter)
	{
		const Link & link = iter->second;
		if(seen.insert(std::make_tuple(std::min(link.from(), link.to()), std::max(link.from(), link.to()), (int)link.type())).second)
		{
			working.insert(std::make_pair(link.from(), link));
		}
	}
	result.loopsBefore = countLoops(working, options.includeUserClosures);

	result.rootId = largestComponentRoot(odomPoses, working);
	if(result.rootId == 0)
	{
		UERROR("No node with a pose to optimize from.");
		return result;
	}

	const bool greedyEnabled =
			options.maxLinearError > 0.0f ||
			options.maxAngularError > 0.0f ||
			options.maxErrorRatio > 0.0f;

	std::map<int, Transform> lastOptimized;
	std::vector<LoopResidual> lastResiduals;
	bool lastOptimizedIsCurrent = false;

	auto score = [&options](const LoopResidual & r) {
		float s = 0.0f;
		if(options.maxLinearError > 0.0f)
		{
			s = std::max(s, r.linear / options.maxLinearError);
		}
		if(options.maxAngularError > 0.0f)
		{
			s = std::max(s, r.angular / options.maxAngularError);
		}
		if(options.maxErrorRatio > 0.0f)
		{
			s = std::max(s, std::max(r.linearRatio, r.angularRatio) / options.maxErrorRatio);
		}
		return s;
	};

	// Returns false on optimization failure or cancel.
	StageTimes times;
	optimizer.resetIncremental();
	int optimizationsSinceFullSolve = 0;
	auto optimizeOnce = [&](bool fullSolve) {
		times = StageTimes();
		if(optimizer.isIncremental())
		{
			if(optimizationsSinceFullSolve > 0 &&
			   (fullSolve || (options.fullSolveInterval > 0 && optimizationsSinceFullSolve >= options.fullSolveInterval)))
			{
				optimizer.resetIncremental();
				optimizationsSinceFullSolve = 0;
			}
			times.fullSolve = optimizationsSinceFullSolve == 0;
			++optimizationsSinceFullSolve;
		}
		std::multimap<int, Link> optLinks;
		std::map<int, Transform> optimized = optimizeComponentTimed(
				optimizer, result.rootId, odomPoses, working, lastOptimized, &optLinks, &times);
		if(optimized.empty())
		{
			UERROR("Optimization failed (root=%d, %d links).", result.rootId, (int)working.size());
			return false;
		}
		lastOptimized = optimized;
		UTimer residualTimer;
		lastResiduals = computeLoopResiduals(optimized, optimizer.adjustLinkCovariances(optLinks), options.includeUserClosures, optimizer.isSlam2d());
		times.residuals = residualTimer.ticks();
		lastOptimizedIsCurrent = true;
		result.componentPoses = (int)std::distance(optimized.lower_bound(1), optimized.end());
		return true;
	};

	std::set<std::pair<int, int> > unresolvedKeys;
	auto greedy = [&]() {
		std::vector<std::pair<int, float> > worstHistory; // (round, worst score)
		for(int i=0; i<options.maxRounds; ++i)
		{
			if(!optimizeOnce(false))
			{
				return false;
			}
			++result.rounds;
			UTimer roundTimer;

			std::vector<std::pair<float, size_t> > inconsistent;
			for(size_t j=0; j<lastResiduals.size(); ++j)
			{
				float s = score(lastResiduals[j]);
				if(s > 1.0f &&
				   result.bridges.find(pairKey(lastResiduals[j].link)) == result.bridges.end() &&
				   unresolvedKeys.find(pairKey(lastResiduals[j].link)) == unresolvedKeys.end())
				{
					inconsistent.push_back(std::make_pair(s, j));
				}
			}
			std::sort(inconsistent.begin(), inconsistent.end(), [](const std::pair<float, size_t> & a, const std::pair<float, size_t> & b) {
				return a.first > b.first;
			});

			if(inconsistent.empty())
			{
				report(progress, uFormat("Round %d: %d loop closures in the optimized graph, none inconsistent.",
						result.rounds, (int)lastResiduals.size()));
				return true;
			}
			const LoopResidual & worst = lastResiduals[inconsistent.front().second];

			worstHistory.push_back(std::make_pair(result.rounds, inconsistent.front().first));
			if(options.stallRounds > 0 && (int)worstHistory.size() > options.stallRounds)
			{
				const std::pair<int, float> & before = worstHistory[worstHistory.size()-1-options.stallRounds];
				const float improvement = 1.0f - inconsistent.front().first / before.second;
				if(improvement < options.stallImprovement)
				{
					for(size_t j=0; j<inconsistent.size(); ++j)
					{
						result.unresolved.push_back(lastResiduals[inconsistent[j].second]);
						unresolvedKeys.insert(pairKey(lastResiduals[inconsistent[j].second].link));
					}
					int restored = 0;
					for(std::vector<RemovedLoop>::iterator iter=result.removed.begin(); iter!=result.removed.end();)
					{
						if(iter->reason == kInconsistent && iter->round >= before.first)
						{
							working.insert(std::make_pair(iter->link.from(), iter->link));
							result.unresolved.push_back(iter->residual);
							unresolvedKeys.insert(pairKey(iter->link));
							iter = result.removed.erase(iter);
							++restored;
						}
						else
						{
							++iter;
						}
					}
					if(!report(progress, uFormat("Round %d: worst error improved %.1f%% over the last %d rounds (less than %.0f%%): "
							"stopping and putting back the %d loop closures removed since round %d. "
							"%d inconsistent loop closures left (worst %d->%d: %.3f m, %.2f deg), check odometry and priors there.",
							result.rounds, improvement*100.0f, options.stallRounds, options.stallImprovement*100.0f,
							restored, before.first, (int)result.unresolved.size(),
							worst.link.from(), worst.link.to(), worst.linear, worst.angular*180.0/M_PI)))
					{
						return false;
					}
					if(restored)
					{
						lastOptimizedIsCurrent = false;
						return optimizeOnce(true);
					}
					return true;
				}
			}

			std::vector<size_t> selected;
			for(size_t j=0; j<inconsistent.size(); ++j)
			{
				const LoopResidual & candidate = lastResiduals[inconsistent[j].second];
				if(!selected.empty())
				{
					if(options.batchRadius < 0.0f)
					{
						break;
					}
					if(options.batchRadius > 0.0f)
					{
						bool near = false;
						for(size_t k : selected)
						{
							if(closestEndpointDistance(candidate.link, lastResiduals[k].link, lastOptimized) < options.batchRadius)
							{
								near = true;
								break;
							}
						}
						if(near)
						{
							continue;
						}
					}
				}
				selected.push_back(inconsistent[j].second);
			}
			times.select = roundTimer.ticks();

			Connectivity connectivity(working);
			times.connectivity = roundTimer.ticks();
			int removedThisRound = 0;
			int bridgesThisRound = 0;
			for(size_t j : selected)
			{
				const LoopResidual & r = lastResiduals[j];
				UTimer bridgeTimer;
				const bool removable = connectivity.removeIfNotBridge(r.link);
				times.bridges += bridgeTimer.ticks();
				if(!removable)
				{
					result.bridges.insert(pairKey(r.link));
					++bridgesThisRound;
					continue;
				}
				RemovedLoop removed;
				removed.link = r.link;
				removed.reason = kInconsistent;
				removed.round = result.rounds;
				removed.residual = r;
				result.removed.push_back(removed);
				bridgeTimer.start();
				eraseLink(working, r.link);
				times.erase += bridgeTimer.ticks();
				++removedThisRound;
			}

			if(!report(progress, uFormat("Round %d: %d loop closures, %d inconsistent (worst %d->%d: %.3f m, %.2f deg, ratio %.1f), "
					"removed %d%s.",
					result.rounds, (int)lastResiduals.size(), (int)inconsistent.size(),
					worst.link.from(), worst.link.to(), worst.linear, worst.angular*180.0/M_PI,
					std::max(worst.linearRatio, worst.angularRatio),
					removedThisRound,
					bridgesThisRound?uFormat(", kept %d that would split the graph", bridgesThisRound).c_str():"")))
			{
				return false;
			}
			if(!report(progress, uFormat("  timing: seed %.2fs, connected graph %.2fs, guess %.2fs, optimize %.2fs (%d iterations%s), "
					"residuals %.2fs, select %.2fs, connectivity %.2fs, bridge checks %.2fs, erase %.2fs",
					times.seed, times.connectedGraph, times.guess, times.optimize, times.iterations, times.fullSolve&&optimizer.isIncremental()?", full solve":"",
					times.residuals, times.select, times.connectivity, times.bridges, times.erase)))
			{
				return false;
			}

			if(removedThisRound == 0)
			{
				// Only bridges left: the graph did not change.
				return true;
			}
			lastOptimizedIsCurrent = false;
		}
		UWARN("Greedy pruning stopped after %d rounds with inconsistent loop closures left.", options.maxRounds);
		return true;
	};

	bool ok = greedyEnabled ? greedy() : optimizeOnce(false);
	if(!ok)
	{
		return result;
	}

	if(options.minGraphDistance > 0.0f)
	{
		// Prefer the loop closure that agrees best with the optimized graph;
		// rotation counted at a 1 m lever arm.
		std::map<std::pair<int, int>, float> rank;
		std::map<std::pair<int, int>, LoopResidual> residualByKey;
		for(size_t i=0; i<lastResiduals.size(); ++i)
		{
			rank[pairKey(lastResiduals[i].link)] = lastResiduals[i].linear + lastResiduals[i].angular;
			residualByKey[pairKey(lastResiduals[i].link)] = lastResiduals[i];
		}
		std::vector<Link> redundant = findRedundantLoops(working, options.minGraphDistance, options.includeUserClosures, rank);
		for(size_t i=0; i<redundant.size(); ++i)
		{
			RemovedLoop removed;
			removed.link = redundant[i];
			removed.reason = kRedundant;
			std::map<std::pair<int, int>, LoopResidual>::iterator found = residualByKey.find(pairKey(redundant[i]));
			if(found != residualByKey.end())
			{
				removed.residual = found->second;
			}
			result.removed.push_back(removed);
			eraseLink(working, redundant[i]);
		}
		if(!report(progress, uFormat("Redundancy (min graph distance %.2f m): removed %d of %d loop closures.",
				options.minGraphDistance, (int)redundant.size(), countLoops(working, options.includeUserClosures) + (int)redundant.size())))
		{
			return result;
		}
		if(!redundant.empty())
		{
			lastOptimizedIsCurrent = false;
			// Removing copies changes how strongly each remaining loop closure
			// pulls, so check consistency again.
			optimizer.resetIncremental();
			optimizationsSinceFullSolve = 0;
			ok = greedyEnabled ? greedy() : optimizeOnce(false);
			if(!ok)
			{
				return result;
			}
		}
	}

	if((!lastOptimizedIsCurrent || (optimizer.isIncremental() && optimizationsSinceFullSolve != 1)) && !optimizeOnce(true))
	{
		return result;
	}

	result.optimizedPoses = lastOptimized;
	result.loopsAfter = countLoops(working, options.includeUserClosures);
	result.ok = true;
	UINFO("Loop pruning done: %d -> %d loop closures, %d rounds (%.1fs)",
			result.loopsBefore, result.loopsAfter, result.rounds, timer.ticks());
	return result;
}

} // namespace loopPruning
} // namespace rtabmap
