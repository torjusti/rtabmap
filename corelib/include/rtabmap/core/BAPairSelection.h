#ifndef RTABMAP_BA_PAIR_SELECTION_H
#define RTABMAP_BA_PAIR_SELECTION_H

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <queue>
#include <utility>
#include <vector>

namespace rtabmap { namespace ba {
struct PairGeometry
{
    std::array<double, 3> center;
    double sceneDepth;
};
struct ExtraPair
{
    int from, to;
    double quality; // Symmetric projected overlap times a saturated parallax score.
};

// Deterministic, undirected budget: a selected extra consumes one slot at BOTH
// endpoints. Existing graph pairs only seed viewpoint diversity, never consume
// budget or get removed. A lazy priority queue avoids rescoring every edge after
// each choice: adding partners can only lower the score of the remaining edges.
inline std::vector<bool> selectExtraPairs(
    const std::vector<ExtraPair> & extras,
    const std::vector<std::pair<int, int> > & protectedPairs,
    const std::map<int, PairGeometry> & geometry, int budget)
{
    std::vector<bool> selected(extras.size(), budget <= 0);
    if(budget <= 0) return selected;
    std::map<int, std::vector<int> > partners;
    std::map<int, int> counts;
    for(const auto & p : protectedPairs)
    {
        partners[p.first].push_back(p.second);
        partners[p.second].push_back(p.first);
    }
    auto novelty = [&](int from, int to) {
        auto a = geometry.find(from), b = geometry.find(to);
        if(a == geometry.end() || b == geometry.end()) return 1.0;
        const double scale = std::max(0.01, 0.1*a->second.sceneDepth);
        double value = 1.0;
        for(int previous : partners[from])
        {
            auto c = geometry.find(previous);
            if(c == geometry.end()) continue;
            double d2 = 0;
            for(int k=0; k<3; ++k)
                d2 += std::pow(b->second.center[k]-c->second.center[k], 2);
            value = std::min(value, std::sqrt(d2)/scale);
        }
        return value;
    };
    struct Item { double score; size_t index; };
    struct Less {
        bool operator()(const Item & a, const Item & b) const {
            return a.score != b.score ? a.score < b.score : a.index > b.index;
        }
    };
    std::priority_queue<Item, std::vector<Item>, Less> queue;
    for(size_t i=0; i<extras.size(); ++i)
        queue.push({std::isfinite(extras[i].quality) ? std::max(0.0, extras[i].quality) : 0.0, i});
    while(!queue.empty())
    {
        const Item item = queue.top(); queue.pop();
        const auto & p = extras[item.index];
        if(p.from == p.to || counts[p.from] >= budget || counts[p.to] >= budget) continue;
        const double quality = std::isfinite(p.quality) ? std::max(0.0, p.quality) : 0.0;
        const double score = quality*(0.25 + 0.75*std::min(novelty(p.from,p.to), novelty(p.to,p.from)));
        if(!queue.empty() && Less()({score,item.index}, queue.top()))
        {
            queue.push({score,item.index});
            continue;
        }
        selected[item.index] = true;
        ++counts[p.from]; ++counts[p.to];
        partners[p.from].push_back(p.to);
        partners[p.to].push_back(p.from);
    }
    return selected;
}
}} // namespace rtabmap::ba
#endif
