#include <gtest/gtest.h>
#include <rtabmap/core/LoopPruning.h>
#include <rtabmap/core/Optimizer.h>
#include <rtabmap/utilite/UConversion.h>
#include <memory>

using namespace rtabmap;

namespace {

// Two walks down the same 10 m corridor, 0.25 m steps. Walk A is nodes
// 1..41, walk B is nodes 101..141, started 5 m to the side in its own
// odometry frame. Node i of A and node 100+i of B are the same place.
struct TwoWalks
{
	std::map<int, Transform> poses;
	std::multimap<int, Link> links;
	static constexpr int kNodes = 41;
	static constexpr float kStep = 0.25f;

	TwoWalks()
	{
		for(int i=1; i<=kNodes; ++i)
		{
			poses[i] = Transform(kStep*(i-1), 0, 0, 0, 0, 0);
			poses[100+i] = Transform(kStep*(i-1), 5, 0, 0, 0, 0);
			if(i > 1)
			{
				add(Link(i-1, i, Link::kNeighbor, Transform(kStep, 0, 0, 0, 0, 0), information(0.01)));
				add(Link(99+i, 100+i, Link::kNeighbor, Transform(kStep, 0, 0, 0, 0, 0), information(0.01)));
			}
		}
	}
	static cv::Mat information(double sigma)
	{
		return cv::Mat::eye(6, 6, CV_64FC1) / (sigma*sigma);
	}
	void add(const Link & link)
	{
		links.insert(std::make_pair(link.from(), link));
	}
	// A loop as confident as one odometry step would bend the walks to hide
	// most of its own error.
	static cv::Mat loopInformation()
	{
		return information(0.05);
	}
	void loop(int i, const Transform & t = Transform::getIdentity())
	{
		add(Link(i, 100+i, Link::kGlobalClosure, t, loopInformation()));
	}
	void allLoops()
	{
		for(int i=1; i<=kNodes; ++i)
		{
			loop(i);
		}
	}
};

std::pair<int, int> key(int a, int b)
{
	return std::make_pair(std::min(a, b), std::max(a, b));
}

bool removedPair(const loopPruning::Result & result, int a, int b, loopPruning::Reason reason)
{
	for(const loopPruning::RemovedLoop & r : result.removed)
	{
		if(key(r.link.from(), r.link.to()) == key(a, b) && r.reason == reason)
		{
			return true;
		}
	}
	return false;
}

std::unique_ptr<Optimizer> priorCapableOptimizer()
{
	for(Optimizer::Type type : {Optimizer::kTypeGTSAM, Optimizer::kTypeG2O})
	{
		if(Optimizer::isAvailable(type))
		{
			ParametersMap parameters;
			parameters.insert(ParametersPair(Parameters::kOptimizerPriorsIgnored(), "false"));
			parameters.insert(ParametersPair(Parameters::kOptimizerIterations(), "50"));
			return std::unique_ptr<Optimizer>(Optimizer::create(type, parameters));
		}
	}
	return std::unique_ptr<Optimizer>();
}

} // namespace

TEST(LoopPruning, OnlyLoopClosureTypesArePrunable)
{
	Transform t = Transform::getIdentity();
	cv::Mat inf = cv::Mat::eye(6, 6, CV_64FC1);
	EXPECT_TRUE(loopPruning::isPrunableLoop(Link(1, 2, Link::kGlobalClosure, t, inf), false));
	EXPECT_TRUE(loopPruning::isPrunableLoop(Link(1, 2, Link::kLocalSpaceClosure, t, inf), false));
	EXPECT_TRUE(loopPruning::isPrunableLoop(Link(1, 2, Link::kLocalTimeClosure, t, inf), false));
	EXPECT_FALSE(loopPruning::isPrunableLoop(Link(1, 2, Link::kUserClosure, t, inf), false));
	EXPECT_TRUE(loopPruning::isPrunableLoop(Link(1, 2, Link::kUserClosure, t, inf), true));
	EXPECT_FALSE(loopPruning::isPrunableLoop(Link(1, 2, Link::kNeighbor, t, inf), true));
	EXPECT_FALSE(loopPruning::isPrunableLoop(Link(1, -1, Link::kLandmark, t, inf), true));
	EXPECT_FALSE(loopPruning::isPrunableLoop(Link(1, 1, Link::kPosePrior, t, inf), true));
}

TEST(LoopPruning, ResidualsReportEachPairOnceWithRatios)
{
	std::map<int, Transform> poses;
	poses[1] = Transform::getIdentity();
	poses[2] = Transform(1, 0, 0, 0, 0, 0);
	std::multimap<int, Link> links;
	Link loop(1, 2, Link::kGlobalClosure, Transform(1.3, 0, 0, 0, 0, 0), TwoWalks::information(0.1));
	links.insert(std::make_pair(1, loop));
	links.insert(std::make_pair(2, loop.inverse()));

	std::vector<loopPruning::LoopResidual> residuals = loopPruning::computeLoopResiduals(poses, links, false);
	ASSERT_EQ(residuals.size(), 1u);
	EXPECT_NEAR(residuals[0].linear, 0.3f, 1e-5);
	EXPECT_NEAR(residuals[0].linearRatio, 3.0f, 1e-4);
	EXPECT_NEAR(residuals[0].angular, 0.0f, 1e-5);
}

TEST(LoopPruning, RedundancyKeepsLoopsSpacedByMinGraphDistance)
{
	TwoWalks g;
	g.allLoops();
	// Loops i and j are joined by i -> j -> 100+j -> 100+i, 0.5 m per step of
	// |i-j|. Equal ranks must not remove each loop because of the next one
	// and leave a single loop for the whole corridor.
	const float distance = 2.5f;
	std::vector<Link> removed = loopPruning::findRedundantLoops(g.links, distance, false);

	std::set<int> removedIds;
	for(const Link & link : removed)
	{
		removedIds.insert(std::min(link.from(), link.to()));
	}
	std::vector<int> kept;
	for(int i=1; i<=TwoWalks::kNodes; ++i)
	{
		if(!removedIds.count(i))
		{
			kept.push_back(i);
		}
	}
	EXPECT_EQ(kept, std::vector<int>({1, 6, 11, 16, 21, 26, 31, 36, 41}));
	for(int id : removedIds)
	{
		bool covered = false;
		for(int k : kept)
		{
			covered = covered || 2*TwoWalks::kStep*std::abs(k-id) < distance;
		}
		EXPECT_TRUE(covered) << id;
	}
}

TEST(LoopPruning, RedundancyPrefersBestRankedLoop)
{
	TwoWalks g;
	for(int i : {1, 2, 3, 4})
	{
		g.loop(i);
	}
	std::map<std::pair<int, int>, float> rank;
	rank[std::make_pair(1, 101)] = 0.3f;
	rank[std::make_pair(2, 102)] = 0.2f;
	rank[std::make_pair(3, 103)] = 0.01f;
	rank[std::make_pair(4, 104)] = 0.1f;
	std::vector<Link> removed = loopPruning::findRedundantLoops(g.links, 2.5f, false, rank);
	ASSERT_EQ(removed.size(), 3u);
	for(const Link & link : removed)
	{
		EXPECT_NE(std::min(link.from(), link.to()), 3);
	}
}

TEST(LoopPruning, RedundancyCountsUserClosuresAndOdometryAsPaths)
{
	TwoWalks g;
	g.loop(1);
	g.loop(2);
	// Same node on walk A, far along walk B: 9.5 m away through the user loop.
	g.add(Link(1, 141, Link::kGlobalClosure, Transform::getIdentity(), TwoWalks::loopInformation()));
	// Never removed, but a shortcut for loops 1 and 2 (1.0 m and 0.5 m away).
	g.add(Link(3, 103, Link::kUserClosure, Transform::getIdentity(), TwoWalks::loopInformation()));
	// Within walk A, 1 m along odometry.
	g.add(Link(10, 14, Link::kLocalSpaceClosure, Transform(4*TwoWalks::kStep, 0, 0, 0, 0, 0), TwoWalks::loopInformation()));

	std::vector<Link> removed = loopPruning::findRedundantLoops(g.links, 1.25f, false);
	std::set<std::pair<int, int> > removedPairs;
	for(const Link & link : removed)
	{
		removedPairs.insert(key(link.from(), link.to()));
	}
	const std::set<std::pair<int, int> > expected = {key(1, 101), key(2, 102), key(10, 14)};
	EXPECT_EQ(removedPairs, expected);
}

TEST(LoopPruning, GreedyRemovesInconsistentLoopOnly)
{
	TwoWalks g;
	g.allLoops();
	// Node 125 is 5 steps past node 20. This loop claims 0.8 m more than that.
	g.add(Link(20, 125, Link::kGlobalClosure, Transform(5*TwoWalks::kStep + 0.8f, 0, 0, 0, 0, 0), TwoWalks::loopInformation()));

	std::unique_ptr<Optimizer> optimizer(Optimizer::create(ParametersMap()));
	loopPruning::Options options;
	options.minGraphDistance = 0.0f;
	loopPruning::Result result = loopPruning::prune(*optimizer, g.poses, g.links, options);

	ASSERT_TRUE(result.ok);
	EXPECT_EQ(result.loopsBefore, 42);
	EXPECT_EQ(result.loopsAfter, 41);
	ASSERT_EQ(result.removed.size(), 1u);
	EXPECT_TRUE(removedPair(result, 20, 125, loopPruning::kInconsistent));
	EXPECT_EQ(result.componentPoses, 2*TwoWalks::kNodes);
	EXPECT_NEAR(result.optimizedPoses.at(1).getDistance(result.optimizedPoses.at(101)), 0.0f, 0.02f);
}

TEST(LoopPruning, GreedyStopsAndRestoresLoopsWhenRemovalsDoNotHelpEnough)
{
	TwoWalks g;
	// Odometry of walk A jumps 0.6 m between nodes 20 and 21, so every loop
	// closure disagrees a little and removing one at a time helps slowly.
	for(std::multimap<int, Link>::iterator iter=g.links.begin(); iter!=g.links.end(); ++iter)
	{
		if(iter->second.from() == 20 && iter->second.to() == 21)
		{
			iter->second.setTransform(Transform(TwoWalks::kStep + 0.6f, 0, 0, 0, 0, 0));
		}
	}
	g.allLoops();

	std::unique_ptr<Optimizer> optimizer(Optimizer::create(ParametersMap()));
	loopPruning::Options options;
	options.minGraphDistance = 0.0f;
	options.batchRadius = -1.0f;
	options.maxLinearError = 0.1f;
	options.stallRounds = 3;
	options.stallImprovement = 0.99f;
	loopPruning::Result result = loopPruning::prune(*optimizer, g.poses, g.links, options);

	ASSERT_TRUE(result.ok);
	EXPECT_EQ(result.rounds, options.stallRounds + 1);
	EXPECT_TRUE(result.removed.empty());
	EXPECT_EQ(result.loopsAfter, result.loopsBefore);
	EXPECT_FALSE(result.unresolved.empty());

	options.stallRounds = 0;
	result = loopPruning::prune(*optimizer, g.poses, g.links, options);
	ASSERT_TRUE(result.ok);
	EXPECT_TRUE(result.unresolved.empty());
	EXPECT_GT(result.removed.size(), 3u);
}

TEST(LoopPruning, IncrementalGtsamRemovesTheSameLoopsAsBatch)
{
	if(!Optimizer::isAvailable(Optimizer::kTypeGTSAM))
	{
		GTEST_SKIP() << "Needs GTSAM";
	}
	TwoWalks g;
	g.allLoops();
	g.add(Link(5, 110, Link::kGlobalClosure, Transform(5*TwoWalks::kStep + 0.8f, 0, 0, 0, 0, 0), TwoWalks::loopInformation()));
	g.add(Link(20, 125, Link::kGlobalClosure, Transform(5*TwoWalks::kStep - 0.7f, 0, 0, 0, 0, 0), TwoWalks::loopInformation()));
	g.add(Link(35, 138, Link::kGlobalClosure, Transform(3*TwoWalks::kStep, 0.9f, 0, 0, 0, 0), TwoWalks::loopInformation()));

	loopPruning::Options options;
	options.minGraphDistance = 0.0f;
	options.batchRadius = -1.0f;
	options.fullSolveInterval = 2;

	std::map<int, loopPruning::Result> results;
	for(bool incremental : {false, true})
	{
		ParametersMap parameters;
		parameters.insert(ParametersPair(Parameters::kGTSAMIncremental(), uBool2Str(incremental)));
		std::unique_ptr<Optimizer> optimizer(Optimizer::create(Optimizer::kTypeGTSAM, parameters));
		ASSERT_EQ(optimizer->isIncremental(), incremental);
		results[incremental] = loopPruning::prune(*optimizer, g.poses, g.links, options);
		ASSERT_TRUE(results[incremental].ok);
	}

	std::set<std::pair<int, int> > removed[2];
	for(int i=0; i<2; ++i)
	{
		for(const loopPruning::RemovedLoop & r : results[i].removed)
		{
			removed[i].insert(key(r.link.from(), r.link.to()));
		}
	}
	std::set<std::pair<int, int> > expected = {key(5, 110), key(20, 125), key(35, 138)};
	EXPECT_EQ(removed[0], expected);
	EXPECT_EQ(removed[1], expected);
	EXPECT_EQ(results[1].rounds, 4);
	for(const auto & pose : results[0].optimizedPoses)
	{
		EXPECT_NEAR(pose.second.getDistance(results[1].optimizedPoses.at(pose.first)), 0.0f, 0.001f) << pose.first;
	}
}

TEST(LoopPruning, GreedyKeepsLoopThatWouldSplitTheGraph)
{
	std::unique_ptr<Optimizer> optimizer = priorCapableOptimizer();
	if(!optimizer)
	{
		GTEST_SKIP() << "Needs g2o or GTSAM for priors";
	}
	TwoWalks g;
	// The only link between the walks disagrees with priors on both walks by 1 m.
	g.loop(1);
	g.add(Link(1, 1, Link::kPosePrior, Transform::getIdentity(), TwoWalks::information(0.01)));
	g.add(Link(101, 101, Link::kPosePrior, Transform(0, 1, 0, 0, 0, 0), TwoWalks::information(0.01)));

	loopPruning::Options options;
	options.minGraphDistance = 0.0f;
	loopPruning::Result result = loopPruning::prune(*optimizer, g.poses, g.links, options);

	ASSERT_TRUE(result.ok);
	EXPECT_TRUE(result.removed.empty());
	ASSERT_EQ(result.bridges.size(), 1u);
	EXPECT_EQ(*result.bridges.begin(), std::make_pair(1, 101));
	EXPECT_EQ(result.loopsAfter, 1);
}

TEST(LoopPruning, PruneThenRemoveRedundant)
{
	TwoWalks g;
	g.allLoops();
	g.add(Link(20, 125, Link::kGlobalClosure, Transform(5*TwoWalks::kStep + 0.8f, 0, 0, 0, 0, 0), TwoWalks::loopInformation()));

	std::unique_ptr<Optimizer> optimizer(Optimizer::create(ParametersMap()));
	loopPruning::Options options;
	options.minGraphDistance = 2.5f;
	loopPruning::Result result = loopPruning::prune(*optimizer, g.poses, g.links, options);

	ASSERT_TRUE(result.ok);
	EXPECT_TRUE(removedPair(result, 20, 125, loopPruning::kInconsistent));
	// The good loops have near-equal residuals, so which ones survive depends
	// on solver noise; any maximal set at least 5 steps apart is valid.
	EXPECT_GE(result.loopsAfter, 5);
	EXPECT_LE(result.loopsAfter, 9);
	int redundant = 0;
	std::set<int> removedIds;
	for(const loopPruning::RemovedLoop & r : result.removed)
	{
		if(r.reason == loopPruning::kRedundant)
		{
			++redundant;
			removedIds.insert(std::min(r.link.from(), r.link.to()));
		}
	}
	EXPECT_EQ(redundant, 41 - result.loopsAfter);
	std::vector<int> kept;
	for(int i=1; i<=TwoWalks::kNodes; ++i)
	{
		if(!removedIds.count(i))
		{
			kept.push_back(i);
		}
	}
	for(size_t i=1; i<kept.size(); ++i)
	{
		EXPECT_GE(kept[i]-kept[i-1], 5);
	}
	EXPECT_NEAR(result.optimizedPoses.at(1).getDistance(result.optimizedPoses.at(101)), 0.0f, 0.02f);
}
