#ifndef RTABMAP_GUI_LOOP_POSE_GROUPING_H
#define RTABMAP_GUI_LOOP_POSE_GROUPING_H

#include <rtabmap/core/Graph.h>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <vector>
#include <array>

namespace rtabmap {
namespace loopPoseGrouping {

typedef std::pair<int, int> Pair;
typedef std::map<int, std::vector<int> > Groups;

inline Pair ordered(int a, int b) { return std::minmax(a, b); }

// Full camera rotation, including roll.
inline double rotationDistance(const Transform & a, const Transform & b)
{
	return a.getAngle(b);
}

// Walk in acquisition/node order, but only extend over an actual odometry
// edge. A missing/unselected node or session boundary starts another group.
// Always compare to the first pose: neighbor-to-neighbor clustering would
// otherwise absorb an arbitrarily long, slowly moving trajectory.
inline Groups makeGroups(
		const std::map<int, Transform> & odometry,
		const std::multimap<int, Link> & links,
		const std::map<int, int> & mapIds,
		const std::set<int> & eligible,
		double translation, double angle)
{
	std::set<Pair> neighbors;
	for(const auto & entry : links)
	{
		const Link & l = entry.second;
		if(l.type() == Link::kNeighbor || l.type() == Link::kNeighborMerged)
			neighbors.insert(ordered(l.from(), l.to()));
	}
	Groups groups;
	std::vector<int> members;
	Transform anchor;
	int previous = 0;
	auto flush = [&]() {
		if(!members.empty())
			groups.emplace(members[members.size()/2], members);
		members.clear();
	};
	for(const auto & entry : odometry)
	{
		const int id = entry.first;
		if(id <= 0 || !eligible.count(id) || entry.second.isNull())
		{
			flush();
			previous = 0;
			continue;
		}
		bool sameMap = mapIds.count(id) && mapIds.count(previous) && mapIds.at(id) == mapIds.at(previous);
		if(members.empty() || !sameMap || !neighbors.count(ordered(previous, id)) ||
			anchor.getDistance(entry.second) > translation || rotationDistance(anchor, entry.second) > angle)
		{
			flush();
			anchor = entry.second;
		}
		members.push_back(id);
		previous = id;
	}
	flush();
	return groups;
}

inline std::vector<int> alternatives(int representative, const std::vector<int> & members)
{
	std::vector<int> out(1, representative);
	for(int id : {members.front(), members.back()})
		if(std::find(out.begin(), out.end(), id) == out.end()) out.push_back(id);
	return out;
}

// Stream local group pairs without materializing the quadratic list returned
// by radiusPosesClustering in dense stationary recordings. A uniform spatial
// hash is sufficient for this fixed-radius search; exact tests follow below.
template<class Visitor, class Canceled>
inline void visitCandidates(const Groups & groups, const std::map<int, Transform> & poses,
        double radiusMin, double radiusMax, double angle, Visitor visit, Canceled canceled)
{
    double extent=0;
    for(const auto & g : groups) if(poses.count(g.first))
        for(int id : alternatives(g.first,g.second)) if(poses.count(id))
            extent=std::max(extent,double(poses.at(g.first).getDistance(poses.at(id))));
    double width=radiusMax+2*extent+1e-5;
    if(width<=0) return;
    typedef std::array<long long,3> Cell;
    auto cell=[&](const Transform & p) -> Cell {
        return {{(long long)std::floor(p.x()/width),(long long)std::floor(p.y()/width),(long long)std::floor(p.z()/width)}};
    };
    std::map<Cell,std::vector<int> > grid;
    for(const auto & g : groups) if(poses.count(g.first)) grid[cell(poses.at(g.first))].push_back(g.first);
    size_t checked=0;
    for(const auto & g : groups)
    {
        if(canceled()) return;
        if(!poses.count(g.first)) continue;
        Cell c=cell(poses.at(g.first));
        for(int x=-1;x<=1;++x) for(int y=-1;y<=1;++y) for(int z=-1;z<=1;++z)
        {
            auto bin=grid.find({{c[0]+x,c[1]+y,c[2]+z}});
            if(bin==grid.end()) continue;
            for(int other : bin->second)
            {
                if(++checked%4096==0 && canceled()) return;
                if(other<=g.first || poses.at(g.first).getDistance(poses.at(other))>width) continue;
                for(int a : alternatives(g.first,g.second)) for(int b : alternatives(other,groups.at(other)))
                {
                    if(!poses.count(a) || !poses.count(b)) continue;
                    const Transform & p=poses.at(a), & q=poses.at(b);
                    double distance=p.getDistance(q);
                    double dot=p.toEigen3d().linear().col(0).dot(q.toEigen3d().linear().col(0));
                    double viewAngle=std::acos(std::max(-1.0,std::min(1.0,dot)));
                    if(distance>=radiusMin && distance<radiusMax && (angle<=0 || viewAngle<=angle))
                        visit(ordered(a,b),distance/radiusMax+viewAngle/CV_PI);
                }
            }
        }
    }
}

// At most nine distinct attempts per group pair, with the representative
// pair first. Only representatives enter the spatial tree. Enlarge its query
// radius by the actual optimized group extents; exact member gates below
// prevent false acceptance and preserve alternatives near a radius boundary.
inline std::map<Pair, std::vector<Pair> > candidates(
		const Groups & groups, const std::map<int, Transform> & poses,
		double radiusMin, double radiusMax, double angle)
{
	std::map<int, Transform> representatives;
	double extent = 0.0;
	for(const auto & group : groups)
	{
		if(!poses.count(group.first)) continue;
		representatives.emplace(group.first, poses.at(group.first));
		for(int id : alternatives(group.first, group.second))
			if(poses.count(id)) extent = std::max(extent, double(poses.at(group.first).getDistance(poses.at(id))));
	}
	std::map<Pair, std::vector<Pair> > out;
	const auto near = graph::radiusPosesClustering(representatives, radiusMax + 2.0*extent + 1e-5, 0.0f);
	std::set<Pair> visited;
	for(const auto & entry : near)
	{
		Pair key = ordered(entry.first, entry.second);
		if(!visited.insert(key).second) continue;
		const auto a = alternatives(key.first, groups.at(key.first));
		const auto b = alternatives(key.second, groups.at(key.second));
		std::vector<Pair> pairs;
		for(size_t rank=0; rank<a.size()+b.size()-1; ++rank)
			for(size_t i=0; i<a.size(); ++i)
			{
				if(rank<i || rank-i>=b.size()) continue;
				int from = a[i], to = b[rank-i];
				if(!poses.count(from) || !poses.count(to)) continue;
				const Transform & p = poses.at(from), & q = poses.at(to);
				double distance = p.getDistance(q);
				// Match radiusPosesClustering's viewing-axis angle semantics.
				double dot = p.toEigen3d().linear().col(0).dot(q.toEigen3d().linear().col(0));
				if(distance >= radiusMin && distance < radiusMax &&
					(angle <= 0.0 || std::acos(std::max(-1.0, std::min(1.0, dot))) <= angle))
					pairs.push_back(ordered(from, to));
			}
		if(!pairs.empty()) out.emplace(key, pairs);
	}
	return out;
}

} // namespace loopPoseGrouping
} // namespace rtabmap
#endif
