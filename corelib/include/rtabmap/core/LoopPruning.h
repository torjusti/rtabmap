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

#ifndef RTABMAP_CORE_LOOP_PRUNING_H
#define RTABMAP_CORE_LOOP_PRUNING_H

#include "rtabmap/core/rtabmap_core_export.h" // DLL export/import defines

#include <rtabmap/core/Link.h>
#include <rtabmap/core/Transform.h>
#include <map>
#include <set>
#include <utility>
#include <vector>

namespace rtabmap {

class Optimizer;
class ProgressState;

/**
 * @namespace rtabmap::loopPruning
 * @brief Offline removal of inconsistent and redundant loop closures.
 *
 * Works on a whole pose graph (odometry poses and all links, as stored in a
 * database) after loop closure detection has finished. Two passes:
 *
 * - Greedy pruning: optimize, score every loop closure by its own residual,
 *   remove the worst ones, re-optimize, repeat until no loop closure exceeds
 *   the thresholds.
 * - Redundancy removal: a loop closure is redundant if its two nodes are
 *   already joined by a path shorter than a minimum graph distance, the same
 *   rule detect-more loop closures uses to skip candidates. Loop closures
 *   agreeing best with the optimized graph are kept first.
 *
 * Only loop closure links are ever removed. Odometry, priors, gravity and
 * landmark links are left alone.
 */
namespace loopPruning {

struct RTABMAP_CORE_EXPORT Options
{
	/// Also consider @ref Link::kUserClosure links for removal. Detect-more
	/// loop closures are stored with that type, like manually added ones.
	bool includeUserClosures = true;

	/// @name Greedy pruning. A loop closure is inconsistent when any enabled threshold is exceeded.
	/// @{
	float maxLinearError = 0.25f;            ///< Largest per-axis translation residual (m). 0 disables.
	float maxAngularError = 0.0872665f;      ///< Rotation residual (rad, default 5 deg). 0 disables.
	float maxErrorRatio = 0.0f;              ///< Residual over the link's own standard deviation. 0 disables.
	/**
	 * Per round, the worst inconsistent loop closure is removed, plus every
	 * other one whose endpoints are all farther than this (m, in the optimized
	 * map) from the endpoints of those already removed this round. 0 removes
	 * all inconsistent loop closures at once, negative removes one per round.
	 */
	float batchRadius = 10.0f;
	int maxRounds = 200;                     ///< Safety bound on optimizations for one greedy pass.
	/**
	 * Stop rule: the greedy pass stops when the worst inconsistent loop
	 * closure's score (error over threshold) improved by less than
	 * @ref stallImprovement (fraction) over the last @c stallRounds rounds.
	 * The loop closures removed during those rounds are put back: removing
	 * them did not resolve the disagreement, which then usually comes from
	 * odometry or priors rather than from loop closures. 0 disables.
	 */
	int stallRounds = 10;
	float stallImprovement = 0.1f;
	/// @}

	/// With an incremental optimizer (GTSAM iSAM2), solve from scratch every
	/// this many optimizations, and always for the final one. 0 only solves
	/// from scratch for the final optimization.
	int fullSolveInterval = 20;

	/**
	 * Redundancy removal (m): a loop closure is dropped if its nodes are
	 * already joined by a path shorter than this through odometry, links that
	 * are never removed, and better loop closures already kept. Path length is
	 * the sum of link translations. 0 disables.
	 */
	float minGraphDistance = 5.0f;
};

enum Reason
{
	kInconsistent, ///< Removed by greedy pruning.
	kRedundant     ///< Removed by redundancy removal.
};

struct RTABMAP_CORE_EXPORT LoopResidual
{
	Link link;
	float linear = 0.0f;       ///< Largest per-axis translation residual (m).
	float angular = 0.0f;      ///< Rotation residual (rad).
	float linearRatio = 0.0f;  ///< @ref linear over the link's translation standard deviation.
	float angularRatio = 0.0f; ///< @ref angular over the link's rotation standard deviation.
};

struct RTABMAP_CORE_EXPORT RemovedLoop
{
	Link link;
	Reason reason = kInconsistent;
	int round = 0;             ///< Greedy round, 0 for redundancy removal.
	LoopResidual residual;     ///< Residual in the optimization that led to the removal (zeros if unknown).
};

struct RTABMAP_CORE_EXPORT Result
{
	bool ok = false;                          ///< False if the first optimization failed or was canceled.
	std::vector<RemovedLoop> removed;
	std::set<std::pair<int, int> > bridges;   ///< Inconsistent loop closures kept because removing them would split the graph.
	std::map<int, Transform> optimizedPoses;  ///< Poses optimized with the remaining links.
	int loopsBefore = 0;
	int loopsAfter = 0;
	int rounds = 0;                           ///< Optimizations run by greedy pruning.
	/// Inconsistent loop closures left when the stop rule ended greedy pruning (empty otherwise).
	std::vector<LoopResidual> unresolved;
	int rootId = 0;                           ///< Node the optimized component was grown from.
	int componentPoses = 0;                   ///< Node count of that component.
};

/** True for loop closure link types this module may remove. */
bool RTABMAP_CORE_EXPORT isPrunableLoop(const Link & link, bool includeUserClosures);

/**
 * Residual of every prunable loop closure in @p links whose two endpoints
 * are in @p optimizedPoses. Each unordered node pair is reported once.
 */
std::vector<LoopResidual> RTABMAP_CORE_EXPORT computeLoopResiduals(
		const std::map<int, Transform> & optimizedPoses,
		const std::multimap<int, Link> & links,
		bool includeUserClosures,
		bool force3DoF = false);

/**
 * Redundant loop closures. Starting from @p links without any prunable loop
 * closure, candidates are visited by increasing @p rank (keyed by the
 * unordered pair, lower is better, missing ranks last). A candidate is kept
 * and added to the graph unless its nodes are already joined by a path
 * shorter than @p minGraphDistance (m, sum of link translations), otherwise
 * it is returned. Every returned loop closure is thus covered by links that
 * are kept.
 */
std::vector<Link> RTABMAP_CORE_EXPORT findRedundantLoops(
		const std::multimap<int, Link> & links,
		float minGraphDistance,
		bool includeUserClosures,
		const std::map<std::pair<int, int>, float> & rank = std::map<std::pair<int, int>, float>());

/**
 * Node (positive id) with the largest id in the largest connected component
 * of @p links, or 0 if there is none.
 */
int RTABMAP_CORE_EXPORT largestComponentRoot(
		const std::map<int, Transform> & odomPoses,
		const std::multimap<int, Link> & links);

/**
 * Optimize the component of @p rootId the same way the database viewer does
 * with anchors: landmark poses seeded from their first observer, connected
 * graph extraction, rigid pre-alignment to landmark priors, then full
 * optimization. Poses found in @p guess replace the chained initial poses.
 * @param linksOut if not null, receives the links that were optimized.
 * @return optimized poses, empty on failure.
 */
std::map<int, Transform> RTABMAP_CORE_EXPORT optimizeComponent(
		Optimizer & optimizer,
		int rootId,
		const std::map<int, Transform> & odomPoses,
		const std::multimap<int, Link> & links,
		const std::map<int, Transform> & guess = std::map<int, Transform>(),
		std::multimap<int, Link> * linksOut = 0);

/**
 * Greedy pruning, then redundancy removal, then greedy pruning again when
 * redundancy removal removed something. Either pass is skipped when all of
 * its thresholds are 0.
 * @param odomPoses odometry poses of the nodes (landmarks are seeded internally).
 * @param links all links, in one or both directions.
 */
Result RTABMAP_CORE_EXPORT prune(
		Optimizer & optimizer,
		const std::map<int, Transform> & odomPoses,
		const std::multimap<int, Link> & links,
		const Options & options,
		const ProgressState * progress = 0);

} // namespace loopPruning
} // namespace rtabmap

#endif /* RTABMAP_CORE_LOOP_PRUNING_H */
