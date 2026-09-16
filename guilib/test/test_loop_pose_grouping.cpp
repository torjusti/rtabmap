#include "../src/LoopPoseGrouping.h"
#include <gtest/gtest.h>

using namespace rtabmap;
using namespace rtabmap::loopPoseGrouping;

namespace {
struct Trajectory
{
	std::map<int, Transform> poses;
	std::map<int, int> maps;
	std::multimap<int, Link> links;
	std::set<int> eligible;
	void add(int id, float x, int session=0, bool connect=true, float roll=0, float yaw=0)
	{
		Transform pose(x,0,0,roll,0,yaw);
		if(!poses.empty() && connect)
		{
			auto previous=poses.rbegin();
			Link l(previous->first,id,Link::kNeighbor,previous->second.inverse()*pose);
			links.emplace(l.from(),l);
		}
		poses[id]=pose; maps[id]=session; eligible.insert(id);
	}
	Groups groups() const { return makeGroups(poses,links,maps,eligible,0.10,5.0*3.141592653589793/180.0); }
};
}

TEST(LoopPoseGrouping, StationaryFramesUseMiddleAndBoundedAlternatives)
{
	Trajectory t;
	for(int id=1; id<=40; ++id) t.add(id,0);
	auto groups=t.groups();
	ASSERT_EQ(groups.size(),1u);
	EXPECT_EQ(groups.begin()->first,21);
	EXPECT_EQ(alternatives(21,groups.begin()->second),(std::vector<int>{21,1,40}));
}

TEST(LoopPoseGrouping, FixedAnchorDoesNotAbsorbSlowMotion)
{
	Trajectory t;
	t.add(1,0); t.add(2,0.06); t.add(3,0.12); t.add(4,0.18);
	auto groups=t.groups();
	ASSERT_EQ(groups.size(),2u);
	EXPECT_EQ(groups.at(2),(std::vector<int>{1,2}));
	EXPECT_EQ(groups.at(4),(std::vector<int>{3,4}));
}

TEST(LoopPoseGrouping, RotationIncludesRollAndYaw)
{
	Trajectory t;
	t.add(1,0); t.add(2,0,0,true,0.11); t.add(3,0,0,true,0.11,0.11);
	EXPECT_EQ(t.groups().size(),3u);
}

TEST(LoopPoseGrouping, SessionsGapsSelectionsAndRevisitsStaySeparate)
{
	Trajectory t;
	t.add(1,0); t.add(2,0,1); t.add(3,0,1,false);
	t.add(4,0,1); t.add(5,0,1); t.eligible.erase(4);
	t.add(6,1,1); t.add(7,0,1);
	EXPECT_EQ(t.groups().size(),6u);
}

TEST(LoopPoseGrouping, NonconsecutiveIdsCanShareARealOdometryChain)
{
	Trajectory t;
	t.add(10,0); t.add(20,0.01); t.add(30,0.02);
	EXPECT_EQ(t.groups().size(),1u);
}

TEST(LoopPoseGrouping, GroupsReduceCartesianPairsToNineAndRepresentativeFirst)
{
	Trajectory t;
	for(int id=1; id<=40; ++id) t.add(id,0);
	for(int id=101; id<=140; ++id) t.add(id,0.5,1,id!=101);
	auto groups=t.groups();
	auto proposals=candidates(groups,t.poses,0,1,0.5);
	ASSERT_EQ(proposals.size(),1u);
	const auto & pairs=proposals.begin()->second;
	ASSERT_EQ(pairs.size(),9u);
	EXPECT_EQ(pairs.front(),Pair(21,121));
	EXPECT_EQ(std::set<Pair>(pairs.begin(),pairs.end()).size(),9u);
}

TEST(LoopPoseGrouping, AlternativeInsideRadiusSurvivesRepresentativeOutside)
{
	Trajectory t;
	t.add(1,0); t.add(2,0.09); t.add(3,0.09);
	t.add(10,-0.95,1,false);
	auto proposals=candidates(t.groups(),t.poses,0,1,0.5);
	ASSERT_EQ(proposals.size(),1u);
	EXPECT_EQ(proposals.begin()->second,(std::vector<Pair>{Pair(1,10)}));
}

TEST(LoopPoseGrouping, ExactRadiusMinimumAndViewingAngleStillApply)
{
	Trajectory t;
	t.add(1,0); t.add(2,0.5,1,false,0,1.0);
	EXPECT_TRUE(candidates(t.groups(),t.poses,0,1,0.5).empty());
	EXPECT_TRUE(candidates(t.groups(),t.poses,0.6,1,0).empty());
	EXPECT_TRUE(candidates(t.groups(),t.poses,0,0.5,0).empty());
	EXPECT_EQ(candidates(t.groups(),t.poses,0,1,0).size(),1u);
}
