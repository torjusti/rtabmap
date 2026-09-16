#ifndef RTABMAP_GUI_LOOP_SPARSE_SELECTION_H
#define RTABMAP_GUI_LOOP_SPARSE_SELECTION_H

#include "LoopPoseGrouping.h"
#include <rtabmap/core/Signature.h>
#include <rtabmap/core/OdometryArcLength.h>
#include <rtabmap/core/RegistrationInfo.h>
#include <array>
#include <cstdint>
#include <limits>

namespace rtabmap { namespace loopSparseSelection {

using loopPoseGrouping::Pair;
using loopPoseGrouping::ordered;
using odometryGraph::Position;
using odometryGraph::Trajectory;
using odometryGraph::trajectory;
typedef std::pair<int, long long> Cell;
typedef std::pair<Cell, Cell> Neighborhood;

inline Cell cell(const Position & p, double spacing)
{ return {p.chain, (long long)std::floor(p.metres/spacing)}; }
inline Neighborhood neighborhood(Pair p, const Trajectory & t, double spacing)
{ return std::minmax(cell(t.at(p.first),spacing),cell(t.at(p.second),spacing)); }
inline bool near(const Position & a, const Position & b, double spacing)
{ return a.chain==b.chain && std::abs(a.metres-b.metres)<spacing; }
inline bool conflicts(Pair a, Pair b, const Trajectory & t, double spacing)
{
    return (near(t.at(a.first),t.at(b.first),spacing) && near(t.at(a.second),t.at(b.second),spacing)) ||
        (near(t.at(a.first),t.at(b.second),spacing) && near(t.at(a.second),t.at(b.first),spacing));
}

// Two-dimensional arc-length hash grid. Check neighboring bins on BOTH
// trajectories, including reversed endpoints and loops straddling bin edges.
class SpacingIndex
{
public:
    SpacingIndex(const Trajectory & t, double spacing) : t_(t), spacing_(spacing) {}
    void insert(Pair p) { bins_[neighborhood(p,t_,spacing_)].push_back(p); }
    bool covered(Pair p) const
    {
        Cell a=cell(t_.at(p.first),spacing_), b=cell(t_.at(p.second),spacing_);
        for(int i=-1;i<=1;++i) for(int j=-1;j<=1;++j)
        {
            auto it=bins_.find(std::minmax(Cell(a.first,a.second+i),Cell(b.first,b.second+j)));
            if(it!=bins_.end()) for(Pair q : it->second) if(conflicts(p,q,t_,spacing_)) return true;
        }
        return false;
    }
private:
    const Trajectory & t_; double spacing_;
    std::map<Neighborhood,std::vector<Pair> > bins_;
};

inline uint64_t mix(uint64_t x)
{
    x+=0x9e3779b97f4a7c15ULL;
    x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;
    x=(x^(x>>27))*0x94d049bb133111ebULL;
    return x^(x>>31);
}
struct Candidate { Pair nodes; double poseCost; };
// Bounded deterministic reservoir prevents candidate storage growing with all
// spatial overlaps. The hash avoids always retaining the earliest node IDs.
class Pool
{
public:
    explicit Pool(size_t capacity=128) : capacity_(capacity) {}
    void offer(Candidate c)
    {
        auto key=std::make_pair(mix((uint64_t(uint32_t(c.nodes.first))<<32)|uint32_t(c.nodes.second)),c.nodes);
        if(items_.size()<capacity_ || key<items_.rbegin()->first) items_[key]=c;
        if(items_.size()>capacity_) items_.erase(std::prev(items_.end()));
    }
    std::vector<Pair> choose(size_t budget, const Trajectory & t) const
    {
        std::vector<Candidate> pool;
        for(const auto & e : items_) pool.push_back(e.second);
        std::vector<Pair> out;
        std::vector<double> distance(pool.size(),std::numeric_limits<double>::infinity());
        while(out.size()<budget && !pool.empty())
        {
            size_t best=0;
            for(size_t i=1;i<pool.size();++i)
                if(distance[i]>distance[best] || (distance[i]==distance[best] &&
                    std::make_pair(pool[i].poseCost,pool[i].nodes)<std::make_pair(pool[best].poseCost,pool[best].nodes))) best=i;
            Pair p=pool[best].nodes;
            out.push_back(p);
            pool.erase(pool.begin()+best); distance.erase(distance.begin()+best);
            auto coordinates=[&](Pair q) {
                const Position & a=t.at(q.first), & b=t.at(q.second);
                return std::make_pair(a.chain<b.chain || (a.chain==b.chain && a.metres<=b.metres) ? a.metres : b.metres,
                    a.chain<b.chain || (a.chain==b.chain && a.metres<=b.metres) ? b.metres : a.metres);
            };
            const auto a=coordinates(p);
            for(size_t i=0;i<pool.size();++i)
            {
                const auto b=coordinates(pool[i].nodes);
                // Farthest-point coverage on the product of the two paths.
                double d=std::max(std::abs(a.first-b.first),std::abs(a.second-b.second));
                distance[i]=std::min(distance[i],d);
            }
        }
        return out;
    }
private:
    size_t capacity_;
    std::map<std::pair<uint64_t,Pair>,Candidate> items_;
};

// Returns a local, dimensionless D-optimal information score from inlier
// reprojection Jacobians. This is a ranking proxy, never an edge covariance.
double informationScore(const Signature & from, const Signature & to,
        const Transform & transform, const RegistrationInfo & info);

} }
#endif
