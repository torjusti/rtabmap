#include "../src/LoopSparseSelection.h"
#include <gtest/gtest.h>
#include <random>

using namespace rtabmap;
using namespace rtabmap::loopSparseSelection;

TEST(LoopSparseSelection, FollowsOdometryAcrossInterleavedSessionsAndUsesEdgeLengths)
{
    std::map<int,Transform> poses;
    for(int id : {1,2,3,4,5,6,7}) poses[id]=Transform::getIdentity();
    std::map<int,int> maps={{1,0},{3,0},{5,0},{2,1},{4,1},{6,1},{7,0}};
    std::multimap<int,Link> links;
    for(Pair p : {Pair(1,3),Pair(3,5),Pair(2,4),Pair(4,6)})
        links.emplace(p.first,Link(p.first,p.second,Link::kNeighbor,Transform(2,0,0,0,0,0)));
    links.emplace(5,Link(5,7,Link::kGlobalClosure,Transform::getIdentity()));
    auto t=trajectory(poses,links,maps);
    EXPECT_EQ(t.at(1).chain,t.at(5).chain);
    EXPECT_NE(t.at(1).chain,t.at(2).chain);
    EXPECT_NE(t.at(1).chain,t.at(7).chain);
    EXPECT_DOUBLE_EQ(t.at(5).metres,4);
    EXPECT_EQ(t.at(5).ordinal,2);
}

TEST(LoopSparseSelection, SplitsBranchesConservatively)
{
    std::map<int,Transform> poses;
    std::multimap<int,Link> links;
    for(int id=1;id<=4;++id) poses[id]=Transform::getIdentity();
    for(int id=2;id<=4;++id) links.emplace(1,Link(1,id,Link::kNeighbor,Transform::getIdentity()));
    auto t=trajectory(poses,links,{});
    EXPECT_NE(t.at(1).chain,t.at(2).chain);
    EXPECT_NE(t.at(2).chain,t.at(3).chain);
}

TEST(LoopSparseSelection, SpacingCrossesBinsRequiresBothEndsAndPreservesOtherRoutes)
{
    Trajectory t={{1,{1,4.9,0}},{2,{2,4.9,0}},{3,{1,5.1,1}},{4,{2,5.1,1}},
        {5,{2,10.0,2}},{6,{3,5.1,0}}};
    SpacingIndex index(t,5);
    index.insert({2,1});
    EXPECT_TRUE(index.covered({3,4}));
    EXPECT_TRUE(index.covered({4,3}));
    EXPECT_FALSE(index.covered({3,5}));
    EXPECT_FALSE(index.covered({3,6}));
}

TEST(LoopSparseSelection, IndexedSpacingMatchesBruteForceIncludingSameChainLoops)
{
    Trajectory t;
    std::mt19937 rng(123);
    for(int id=1;id<=300;++id) t[id]={int(rng()%5),double(rng()%1000)/10,0};
    SpacingIndex index(t,5);
    std::vector<Pair> fixed;
    for(int i=0;i<100;++i)
    {
        Pair p={int(rng()%300)+1,int(rng()%300)+1}; fixed.push_back(p); index.insert(p);
    }
    for(int i=0;i<1000;++i)
    {
        Pair p={int(rng()%300)+1,int(rng()%300)+1}; bool expected=false;
        for(Pair q : fixed) expected=expected || conflicts(p,q,t,5);
        EXPECT_EQ(index.covered(p),expected);
    }
}

TEST(LoopSparseSelection, BoundedPoolIsOrderIndependentAndSpreadsAttempts)
{
    Trajectory t; Pool forward(128), reverse(128);
    for(int i=1;i<=100;++i)
    {
        t[i]={1,double(i)/20,i}; t[i+100]={101,double(i)/20,i};
        forward.offer({{i,i+100},double(i)});
    }
    for(int i=100;i>=1;--i) reverse.offer({{i,i+100},double(i)});
    auto chosen=forward.choose(4,t);
    EXPECT_EQ(chosen,reverse.choose(4,t));
    ASSERT_EQ(chosen.size(),4u);
    EXPECT_EQ(chosen[0],Pair(1,101));
    EXPECT_EQ(chosen[1],Pair(100,200));
    EXPECT_GT(chosen[2].first,40);
    EXPECT_LT(chosen[2].first,60);
    Pool bounded(8);
    for(int i=1;i<=100;++i) bounded.offer({{i,i+100},0});
    EXPECT_EQ(bounded.choose(32,t).size(),8u);
}

TEST(LoopSparseSelection, StreamingCandidatesMatchExactMemberGates)
{
    loopPoseGrouping::Groups groups={{2,{1,2,3}},{5,{4,5,6}},{8,{7,8,9}}};
    std::map<int,Transform> poses;
    for(int i=1;i<=9;++i) poses[i]=Transform(float(i)*0.2f,0,0,0,0,float(i)*0.1f);
    std::set<Pair> streamed, expected;
    loopPoseGrouping::visitCandidates(groups,poses,0.3,1.2,0.6,
        [&](Pair p,double){streamed.insert(p);},[](){return false;});
    for(const auto & e : loopPoseGrouping::candidates(groups,poses,0.3,1.2,0.6))
        expected.insert(e.second.begin(),e.second.end());
    EXPECT_EQ(streamed,expected);
    EXPECT_FALSE(streamed.empty());
}

namespace {
double score(bool spread, double noise=0, double offset=0)
{
    Signature from,to;
    CameraModel camera(400,400,320,240,Transform::getIdentity(),0,cv::Size(640,480));
    to.sensorData().setCameraModels({camera});
    std::multimap<int,int> words;
    std::vector<cv::KeyPoint> pixels;
    std::vector<cv::Point3f> points;
    RegistrationInfo info;
    for(int i=0;i<100;++i)
    {
        double x=spread?(i%10-4.5)*0.2:0, y=spread?(i/10-4.5)*0.2:0;
        words.emplace(i,i); points.emplace_back(x+offset,y,4);
        pixels.emplace_back(cv::Point2f(400*x/4+320+noise,400*y/4+240),1);
        info.inliersIDs.push_back(i);
    }
    from.setWords(words,pixels,points,cv::Mat()); to.setWords(words,pixels,{},cv::Mat());
    return informationScore(from,to,Transform(offset,0,0,0,0,0),info);
}
}

TEST(LoopSparseSelection, InformationRewardsGeometryAndPenalizesResiduals)
{
    EXPECT_GT(score(true),score(false));
    EXPECT_GT(score(true),score(true,10));
    EXPECT_NEAR(score(true),score(true,0,10),1e-4);
    EXPECT_GT(score(true),0);
}

TEST(LoopSparseSelection, MissingQualityDataIsExplicitlyUnscored)
{
    EXPECT_EQ(informationScore(Signature(),Signature(),Transform::getIdentity(),RegistrationInfo()),0);
}
