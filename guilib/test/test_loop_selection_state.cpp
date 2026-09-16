#include "../src/LoopSelectionState.h"
#include <gtest/gtest.h>
#include <QTemporaryDir>
#include <QFile>
#include <random>
#include <QApplication>
#include <QPushButton>
#include <QTableView>
#include <QDialogButtonBox>
#include <QDoubleSpinBox>
using namespace rtabmap;
namespace {
struct Graph {
    std::map<int,Transform> poses;
    std::multimap<int,Link> links;
    void edge(int a,int b,Link::Type type,double length=1) {
        poses[a]=Transform::getIdentity(); poses[b]=Transform::getIdentity();
        Link l(a,b,type,Transform(length,0,0,0,0,0));
        links.emplace(a,l); links.emplace(b,l.inverse());
    }
    Graph() {edge(1,2,Link::kNeighbor); edge(2,3,Link::kNeighbor); edge(11,12,Link::kNeighbor); edge(12,13,Link::kNeighbor);}
};
}
TEST(WholeGraphSelection, RanksQualityDeduplicatesDirectionsAndProtectsLinks)
{
    Graph g; g.edge(1,11,Link::kGlobalClosure); g.edge(2,12,Link::kUserClosure); g.edge(3,13,Link::kLocalSpaceClosure);
    auto r=loopSelection::select(g.poses,g.links,{},{{{1,11},1},{{2,12},8},{{3,13},4}},{},5);
    ASSERT_EQ(r.reasons.size(),3u); EXPECT_EQ(r.reasons.at({2,12}),loopSelection::Quality);
    EXPECT_EQ(r.reasons.at({1,11}),loopSelection::Redundant); EXPECT_EQ(r.reasons.at({3,13}),loopSelection::Redundant);
    r=loopSelection::select(g.poses,g.links,{},{{{2,12},8}},{{1,11}},5);
    EXPECT_EQ(r.reasons.at({1,11}),loopSelection::Protected); EXPECT_EQ(r.reasons.at({2,12}),loopSelection::Redundant);
}
TEST(WholeGraphSelection, SpacingBoundaryAndMissingGeometryAreConservative)
{
    Graph g; g.edge(1,11,Link::kGlobalClosure); g.edge(3,13,Link::kGlobalClosure);
    auto r=loopSelection::select(g.poses,g.links,{},{},{},2);
    EXPECT_NE(r.reasons.at({3,13}),loopSelection::Redundant);
    g.poses.erase(3); r=loopSelection::select(g.poses,g.links,{},{},{},5);
    EXPECT_EQ(r.reasons.at({3,13}),loopSelection::MissingGeometry);
}
TEST(WholeGraphSelection, UnscoredOrderingIsDeterministicAndCancellationDoesNotSucceed)
{
    Graph g; g.edge(3,13,Link::kGlobalClosure); g.edge(1,11,Link::kGlobalClosure);
    auto r=loopSelection::select(g.poses,g.links,{},{},{},5);
    EXPECT_EQ(r.reasons.at({1,11}),loopSelection::Quality);
    EXPECT_EQ(r.reasons.at({3,13}),loopSelection::Redundant);
    EXPECT_TRUE(loopSelection::select(g.poses,g.links,{},{},{},5,[]{return true;}).canceled);
}
TEST(WholeGraphSelection, EveryExcludedEdgeRemainsConnectedAndNonLoopsAreUntouched)
{
    Graph g; std::mt19937 rng(41);
    for(int chain=0;chain<8;++chain) for(int i=1;i<20;++i) g.edge(100+chain*30+i,101+chain*30+i,Link::kNeighbor);
    for(int k=0;k<500;++k) {
        int a=101+(rng()%8)*30+rng()%20,b=101+(rng()%8)*30+rng()%20;
        if(a!=b) g.edge(a,b,Link::kGlobalClosure);
    }
    // Non-loop constraints must be available to connectivity checks.
    g.edge(1,101,Link::kNeighborMerged);
    auto r=loopSelection::select(g.poses,g.links,{},{},{},5);
    std::map<int,std::set<int>> adjacency;
    int excluded=0;
    for(const auto & e:g.links) {
        auto p=loopSelection::ordered(e.second.from(),e.second.to());
        if(loopSelection::isLoop(e.second) && r.reasons.at(p)==loopSelection::Redundant) {++excluded;continue;}
        adjacency[e.second.from()].insert(e.second.to());
        adjacency[e.second.to()].insert(e.second.from());
    }
    ASSERT_GT(excluded,0);
    for(const auto & e:r.reasons) if(e.second==loopSelection::Redundant) {
        std::set<int> visited; std::vector<int> queue={e.first.first};
        while(!queue.empty()) {int id=queue.back();queue.pop_back(); if(!visited.insert(id).second) continue;
            for(int n:adjacency[id]) if(!visited.count(n)) queue.push_back(n);}
        EXPECT_TRUE(visited.count(e.first.second));
    }
}
TEST(LoopSelectionState, AtomicRoundTripAndChangedConstraintInvalidation)
{
    QTemporaryDir dir; ASSERT_TRUE(dir.isValid());
    LoopSelectionState a; a.path=dir.path()+"/scan.db.loop-selection.json";
    Link link(12,2,Link::kUserClosure,Transform(1,2,3,.1,.2,.3));
    a.remember(link,7.5,30,45); a.selection.disabled[{2,12}]=a.fingerprint(link); a.selection.protectedLinks.insert({3,13});
    QString error; ASSERT_TRUE(a.save(&error))<<error.toStdString();
    LoopSelectionState b; b.path=a.path; ASSERT_TRUE(b.load(&error));
    ASSERT_NE(b.score(link),nullptr); EXPECT_DOUBLE_EQ(b.score(link)->value,7.5); EXPECT_EQ(b.score(link)->inliers,30);
    EXPECT_TRUE(b.disabled(link)); EXPECT_TRUE(b.selection.protectedLinks.count({3,13}));
    Link changed(12,2,Link::kUserClosure,Transform(1.1,2,3,.1,.2,.3));
    EXPECT_FALSE(b.disabled(changed)); EXPECT_EQ(b.score(changed),nullptr);
    changed=Link(12,2,Link::kUserClosure,link.transform(),2*cv::Mat::eye(6,6,CV_64FC1));
    EXPECT_FALSE(b.disabled(changed)); EXPECT_EQ(b.score(changed),nullptr);
    b.selection.disabled.clear(); b.dirty=true; ASSERT_TRUE(b.save(&error)); ASSERT_TRUE(a.load(&error)); EXPECT_FALSE(a.disabled(link));
}
TEST(LoopSelectionState, BadMetadataCannotBeOverwrittenAndPartialDataIsNotLoaded)
{
    QTemporaryDir dir; LoopSelectionState state; state.path=dir.path()+"/bad.json";
    QFile file(state.path); ASSERT_TRUE(file.open(QIODevice::WriteOnly)); file.write("{\"version\":99,\"loops\":[]}"); file.close();
    QString error; EXPECT_FALSE(state.load(&error)); EXPECT_FALSE(state.readable);
    state.dirty=true; EXPECT_FALSE(state.save(&error)); ASSERT_TRUE(file.open(QIODevice::ReadOnly)); EXPECT_TRUE(file.readAll().contains("99"));
}
TEST(LoopSelectionState, StaleScoreDoesNotAcquireNewDisabledFingerprint)
{
    QTemporaryDir dir; LoopSelectionState a; a.path=dir.path()+"/state.json";
    Link old(12,2,Link::kUserClosure,Transform::getIdentity()),changed(12,2,Link::kUserClosure,Transform(2,0,0,0,0,0));
    a.remember(old,4); a.selection.disabled[{2,12}]=a.fingerprint(changed);
    QString error; ASSERT_TRUE(a.save(&error)); ASSERT_TRUE(a.load(&error));
    EXPECT_EQ(a.score(changed),nullptr); EXPECT_TRUE(a.disabled(changed));
}

TEST(LoopSelectionDialog, PreviewProtectApplyAndCancelDoNotMutateSourceState)
{
    int argc=1; char name[]="loop-selection-test"; char * argv[]={name,nullptr};
    qputenv("QT_QPA_PLATFORM","offscreen"); QApplication app(argc,argv);
    Graph g; g.edge(1,11,Link::kGlobalClosure); g.edge(2,12,Link::kGlobalClosure);
    LoopSelectionState state; state.remember(g.links.find(2)->second,5);
    const std::map<int,int> maps;
    LoopSparsificationDialog dialog(g.poses,g.links,maps,state,5);
    auto table=dialog.findChild<QTableView*>(); ASSERT_NE(table,nullptr);
    auto box=dialog.findChild<QDialogButtonBox*>(); ASSERT_NE(box,nullptr);
    auto apply=box->button(QDialogButtonBox::Apply); ASSERT_FALSE(apply->isEnabled());
    QPushButton * preview=nullptr;
    for(auto button:dialog.findChildren<QPushButton*>()) if(button->text()=="Preview") preview=button;
    ASSERT_NE(preview,nullptr);
    dialog.show(); QApplication::processEvents(); preview->click();
    ASSERT_TRUE(apply->isEnabled()); EXPECT_EQ(dialog.resultSelection().disabled.size(),1u);
    // Protect the second row: it overrides the deterministic first-row choice.
    ASSERT_TRUE(table->model()->setData(table->model()->index(1,0),Qt::Checked,Qt::CheckStateRole));
    EXPECT_FALSE(apply->isEnabled()); preview->click(); EXPECT_TRUE(apply->isEnabled());
    EXPECT_TRUE(dialog.resultSelection().protectedLinks.count({2,12}));
    EXPECT_TRUE(dialog.resultSelection().disabled.count({1,11}));
    QApplication::processEvents();
    if(qEnvironmentVariableIsSet("RTABMAP_TEST_SCREENSHOT"))
    {
        EXPECT_TRUE(dialog.grab().save(QString::fromLocal8Bit(qgetenv("RTABMAP_TEST_SCREENSHOT"))));
    }
    apply->click(); EXPECT_EQ(dialog.result(),QDialog::Accepted);
    EXPECT_TRUE(state.selection.disabled.empty()); EXPECT_TRUE(state.selection.protectedLinks.empty());
    // Editing spacing invalidates the preview; cancel leaves the caller untouched.
    dialog.findChild<QDoubleSpinBox*>()->setValue(1);
    EXPECT_FALSE(apply->isEnabled()); box->button(QDialogButtonBox::Cancel)->click();
    EXPECT_EQ(dialog.result(),QDialog::Rejected); EXPECT_TRUE(state.selection.disabled.empty());
}

TEST(LoopSelectionState, MaskRemovesBothDirectionsAndKeepsOdometryAndChangedLinks)
{
    LoopSelectionState state;
    Link l(12,2,Link::kGlobalClosure,Transform(1,2,3,.1,.2,.3));
    Link odom(1,2,Link::kNeighbor,Transform::getIdentity());
    state.selection.disabled[{2,12}]=state.fingerprint(l);
    std::multimap<int,Link> links={{12,l},{2,l.inverse()},{1,odom},{2,odom.inverse()}};
    state.filterLinks(links); ASSERT_EQ(links.size(),2u);
    for(const auto & e:links) EXPECT_EQ(e.second.type(),Link::kNeighbor);
    Link changed(12,2,Link::kGlobalClosure,Transform(2,2,3,.1,.2,.3));
    links={{12,changed},{2,changed.inverse()}}; state.filterLinks(links); EXPECT_EQ(links.size(),2u);
    links={{12,l}}; state.filterLinks(links); EXPECT_TRUE(links.empty());
}
