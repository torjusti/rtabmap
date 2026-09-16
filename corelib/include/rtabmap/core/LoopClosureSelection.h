#ifndef RTABMAP_CORE_LOOP_CLOSURE_SELECTION_H
#define RTABMAP_CORE_LOOP_CLOSURE_SELECTION_H

#include <rtabmap/core/OdometryArcLength.h>
#include <vector>
#include <functional>
#include <array>

namespace rtabmap { namespace loopSelection {

typedef std::pair<int,int> Pair;
inline Pair ordered(int a,int b) { return std::minmax(a,b); }
inline bool isLoop(const Link & l)
{
    return l.from()>0 && l.to()>0 && l.from()!=l.to() &&
        (l.type()==Link::kGlobalClosure || l.type()==Link::kLocalSpaceClosure ||
         l.type()==Link::kLocalTimeClosure || l.type()==Link::kUserClosure);
}
enum Reason { Quality, Connectivity, Protected, MissingGeometry, Redundant };
struct Candidate { Pair nodes; double quality; };
struct Result { std::map<Pair,Reason> reasons; bool canceled=false; };

// Greedy quality/spacing selection with a connectivity invariant: a loop may
// be omitted only if its endpoints are already connected by retained edges.
// Non-loop constraints are always retained. Input links can be bidirectional.
inline Result select(const std::map<int,Transform> & poses,
        const std::multimap<int,Link> & links, const std::map<int,int> & maps,
        const std::map<Pair,double> & scores, const std::set<Pair> & protectedLinks,
        double spacing, std::function<bool()> canceled = [](){return false;})
{
    Result out;
    if(!(spacing>0) || !std::isfinite(spacing)) return out;
    const auto paths=odometryGraph::trajectory(poses,links,maps);
    std::map<int,int> parent,rank;
    auto root=[&](int id) {
        if(!parent.count(id)) parent[id]=id;
        int r=id;
        while(parent[r]!=r) r=parent[r];
        while(id!=r) { int next=parent[id]; parent[id]=r; id=next; }
        return r;
    };
    auto join=[&](int a,int b) {
        a=root(a); b=root(b);
        if(a==b) return;
        if(rank[a]<rank[b]) std::swap(a,b);
        parent[b]=a;
        if(rank[a]==rank[b]) ++rank[a];
    };
    std::set<Pair> seen;
    std::vector<Candidate> candidates;
    for(const auto & e : links)
    {
        const Link & l=e.second;
        if(!isLoop(l)) { join(l.from(),l.to()); continue; }
        Pair p=ordered(l.from(),l.to());
        if(!seen.insert(p).second) continue;
        if(protectedLinks.count(p)) { out.reasons[p]=Protected; join(p.first,p.second); }
        else if(!paths.count(p.first) || !paths.count(p.second))
        { out.reasons[p]=MissingGeometry; join(p.first,p.second); }
        else
        {
            auto score=scores.find(p);
            double q=score==scores.end() || !std::isfinite(score->second) ? 0 : std::max(0.0,score->second);
            candidates.push_back({p,q});
        }
    }
    typedef std::pair<int,long long> Cell;
    typedef std::pair<Cell,Cell> Bin;
    std::map<Bin,std::vector<Pair> > occupied;
    auto cell=[&](int id) { const auto & p=paths.at(id); return Cell(p.chain,(long long)std::floor(p.metres/spacing)); };
    auto insert=[&](Pair p) { occupied[std::minmax(cell(p.first),cell(p.second))].push_back(p); };
    auto near=[&](int a,int b) {
        return paths.at(a).chain==paths.at(b).chain && std::abs(paths.at(a).metres-paths.at(b).metres)<spacing;
    };
    auto covered=[&](Pair p) {
        Cell a=cell(p.first),b=cell(p.second);
        for(int i=-1;i<=1;++i) for(int j=-1;j<=1;++j)
        {
            auto it=occupied.find(std::minmax(Cell(a.first,a.second+i),Cell(b.first,b.second+j)));
            if(it!=occupied.end()) for(Pair q : it->second)
                if((near(p.first,q.first) && near(p.second,q.second)) ||
                   (near(p.first,q.second) && near(p.second,q.first))) return true;
        }
        return false;
    };
    for(const auto & e : out.reasons)
        if(paths.count(e.first.first) && paths.count(e.first.second)) insert(e.first);
    std::sort(candidates.begin(),candidates.end(),[](const Candidate & a,const Candidate & b) {
        return a.quality!=b.quality ? a.quality>b.quality : a.nodes<b.nodes;
    });
    size_t count=0;
    for(const auto & c : candidates)
    {
        if(++count%128==0 && canceled()) { out.canceled=true; return out; }
        const bool redundant=covered(c.nodes);
        Reason reason=!redundant ? Quality : root(c.nodes.first)!=root(c.nodes.second) ? Connectivity : Redundant;
        out.reasons[c.nodes]=reason;
        if(reason!=Redundant) { join(c.nodes.first,c.nodes.second); insert(c.nodes); }
    }
    out.canceled=canceled();
    return out;
}

} }
#endif
