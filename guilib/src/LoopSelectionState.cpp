#include "LoopSelectionState.h"
#include <QCryptographicHash>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QFile>
#include <QSaveFile>

namespace rtabmap {
QString LoopSelectionState::fingerprint(const Link & input)
{
    const Link l=input.from()<input.to()?input.inverse():input;
    // Quantize only the identity check, not the constraint: inverse round trips
    // introduce float noise. Five significant digits tolerate that noise.
    QByteArray data=QByteArray::number(l.from())+":"+QByteArray::number(l.to())+":"+QByteArray::number(int(l.type()));
    for(int i=0;i<12;++i) data+=":"+QByteArray::number(l.transform().data()[i],'g',5);
    for(int r=0;r<6;++r) for(int c=0;c<6;++c) data+=":"+QByteArray::number(l.infMatrix().at<double>(r,c),'g',5);
    return QCryptographicHash::hash(data,QCryptographicHash::Sha256).toHex();
}
void LoopSelectionState::remember(const Link & l,double value,int inliers,int matches)
{
    if(!loopSelection::isLoop(l) || !l.isValid() || !std::isfinite(value) || value<=0) return;
    Score s; s.fingerprint=fingerprint(l); s.value=value; s.inliers=inliers; s.matches=matches;
    scores[loopSelection::ordered(l.from(),l.to())]=s; dirty=true;
}
const LoopSelectionState::Score * LoopSelectionState::score(const Link & l) const
{
    auto it=scores.find(loopSelection::ordered(l.from(),l.to()));
    return it!=scores.end() && it->second.fingerprint==fingerprint(l)?&it->second:nullptr;
}
bool LoopSelectionState::disabled(const Link & l) const
{
    if(!loopSelection::isLoop(l)) return false;
    auto it=selection.disabled.find(loopSelection::ordered(l.from(),l.to()));
    return it!=selection.disabled.end() && it->second==fingerprint(l);
}
void LoopSelectionState::filterLinks(std::multimap<int,Link> & links) const
{
    if(selection.disabled.empty()) return;
    // Choose one canonical record per pair, then suppress both orientations.
    // Refined constraints can reactivate a pair; never leave a one-way mask.
    std::map<Pair,const Link *> canonical;
    for(const auto & e:links) if(loopSelection::isLoop(e.second))
    {
        Pair p=loopSelection::ordered(e.second.from(),e.second.to());
        if(selection.disabled.count(p) && (!canonical.count(p) || e.second.from()>e.second.to())) canonical[p]=&e.second;
    }
    std::set<Pair> excluded;
    for(const auto & e:canonical) if(disabled(*e.second)) excluded.insert(e.first);
    for(auto it=links.begin();it!=links.end();)
        if(loopSelection::isLoop(it->second) && excluded.count(loopSelection::ordered(it->second.from(),it->second.to()))) it=links.erase(it);
        else ++it;
}
bool LoopSelectionState::load(QString * error)
{
    scores.clear(); selection=Snapshot(); undo.clear(); dirty=false; readable=true;
    QFile file(path);
    if(!file.exists()) return true;
    auto fail=[&](QString text) { readable=false; if(error) *error=text; return false; };
    if(!file.open(QIODevice::ReadOnly)) return fail(file.errorString());
    if(file.size()>128*1024*1024) return fail("Loop selection file exceeds 128 MiB.");
    QJsonParseError parse;
    QJsonDocument doc=QJsonDocument::fromJson(file.readAll(),&parse);
    if(parse.error!=QJsonParseError::NoError || !doc.isObject()) return fail("Invalid loop selection JSON: "+parse.errorString());
    QJsonObject root=doc.object();
    if(root.value("version").toInt()!=1 || root.value("scoreModel").toString()!="reprojection-logdet-v1" || !root.value("loops").isArray()) return fail("Unsupported loop selection format.");
    std::map<Pair,Score> loadedScores; Snapshot loaded;
    for(const QJsonValue & v : root.value("loops").toArray())
    {
        if(!v.isObject()) return fail("Invalid loop record.");
        auto o=v.toObject(); int a=o.value("from").toInt(),b=o.value("to").toInt();
        if(a<=0 || b<=a) return fail("Invalid loop endpoints.");
        Pair p(a,b); QString f=o.value("fingerprint").toString();
        if(o.contains("score") || o.value("disabled").toBool())
        {
            if(f.size()!=64 || QByteArray::fromHex(f.toLatin1()).size()!=32) return fail("Invalid constraint fingerprint.");
        }
        if(o.contains("score"))
        {
            Score s; s.value=o.value("score").toDouble(-1); s.fingerprint=f;
            s.inliers=o.value("inliers").toInt(); s.matches=o.value("matches").toInt();
            if(!std::isfinite(s.value) || s.value<=0 || s.inliers<0 || s.matches<0) return fail("Invalid quality score.");
            loadedScores[p]=s;
        }
        if(o.value("disabled").toBool()) loaded.disabled[p]=f;
        if(o.value("protected").toBool()) loaded.protectedLinks.insert(p);
    }
    scores.swap(loadedScores); selection=loaded; return true;
}
bool LoopSelectionState::save(QString * error)
{
    if(!dirty) return true;
    if(!readable || path.isEmpty()) { if(error) *error="Existing metadata could not be read; it has not been overwritten."; return false; }
    std::set<Pair> pairs=selection.protectedLinks;
    for(const auto & e : scores) pairs.insert(e.first);
    for(const auto & e : selection.disabled) pairs.insert(e.first);
    QJsonArray loops;
    for(Pair p : pairs)
    {
        QJsonObject o; o["from"]=p.first; o["to"]=p.second;
        auto s=scores.find(p); auto d=selection.disabled.find(p);
        // A stale saved score must never inherit the fingerprint of a refined,
        // disabled constraint. Retain the mask and omit that obsolete score.
        if(s!=scores.end() && (d==selection.disabled.end() || d->second==s->second.fingerprint))
        {
            o["fingerprint"]=s->second.fingerprint; o["score"]=s->second.value;
            o["inliers"]=s->second.inliers; o["matches"]=s->second.matches;
        }
        if(d!=selection.disabled.end()) { o["disabled"]=true; o["fingerprint"]=d->second; }
        if(selection.protectedLinks.count(p)) o["protected"]=true;
        loops.append(o);
    }
    QJsonObject root; root["version"]=1; root["scoreModel"]="reprojection-logdet-v1"; root["loops"]=loops;
    QSaveFile file(path); QByteArray bytes=QJsonDocument(root).toJson(QJsonDocument::Compact);
    if(!file.open(QIODevice::WriteOnly) || file.write(bytes)!=bytes.size() || !file.commit())
    { if(error) *error=file.errorString(); return false; }
    dirty=false; return true;
}
}
