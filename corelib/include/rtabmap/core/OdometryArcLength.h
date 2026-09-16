#ifndef RTABMAP_CORE_ODOMETRY_ARC_LENGTH_H
#define RTABMAP_CORE_ODOMETRY_ARC_LENGTH_H

#include <rtabmap/core/Link.h>
#include <algorithm>
#include <cmath>
#include <map>
#include <set>

namespace rtabmap { namespace odometryGraph {

struct Position { int chain; double metres; int ordinal; };
typedef std::map<int, Position> Trajectory;

// Split at missing links, session boundaries and branches, never at selection
// boundaries. Follow actual odometry edges, not differences between node IDs.
inline Trajectory trajectory(const std::map<int, Transform> & poses,
        const std::multimap<int, Link> & links, const std::map<int, int> & maps)
{
    std::map<int, std::map<int, double> > edges;
    for(const auto & p : poses) if(p.first>0 && !p.second.isNull()) edges[p.first];
    for(const auto & e : links)
    {
        const Link & l=e.second;
        if((l.type()!=Link::kNeighbor && l.type()!=Link::kNeighborMerged) ||
            l.from()==l.to() || !edges.count(l.from()) || !edges.count(l.to()) ||
            (maps.count(l.from()) && maps.count(l.to()) && maps.at(l.from())!=maps.at(l.to()))) continue;
        // A merged edge has only a displacement, not the removed path's length.
        double length=l.transform().isNull() ? poses.at(l.from()).getDistance(poses.at(l.to())) : l.transform().getNorm();
        if(!std::isfinite(length)) continue;
        edges[l.from()][l.to()]=length;
        edges[l.to()][l.from()]=length;
    }
    std::set<int> branches;
    for(const auto & e : edges) if(e.second.size()>2) branches.insert(e.first);
    for(int id : branches)
    {
        for(const auto & e : edges[id]) edges[e.first].erase(id);
        edges[id].clear();
    }
    Trajectory out;
    auto walk = [&](int start) {
        int id=start, ordinal=0; double metres=0;
        while(!out.count(id))
        {
            out[id]={start,metres,ordinal++};
            int next=0;
            for(const auto & e : edges.at(id))
                if(!out.count(e.first)) { next=e.first; metres+=e.second; break; }
            if(!next) break;
            id=next;
        }
    };
    for(const auto & e : edges) if(e.second.size()<2 && !out.count(e.first)) walk(e.first);
    // A rare closed odometry cycle is cut at the smallest remaining node ID.
    for(const auto & e : edges) if(!out.count(e.first)) walk(e.first);
    return out;
}

} }
#endif
