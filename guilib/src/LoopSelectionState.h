#ifndef RTABMAP_GUI_LOOP_SELECTION_STATE_H
#define RTABMAP_GUI_LOOP_SELECTION_STATE_H

#include <rtabmap/core/LoopClosureSelection.h>
#include <QString>
#include <QDialog>

namespace rtabmap {

class LoopSelectionState
{
public:
    typedef loopSelection::Pair Pair;
    struct Score { QString fingerprint; double value=0; int inliers=0; int matches=0; };
    struct Snapshot { std::map<Pair,QString> disabled; std::set<Pair> protectedLinks; };
    std::map<Pair,Score> scores;
    Snapshot selection;
    std::vector<Snapshot> undo;
    QString path;
    bool dirty=false;
    bool readable=true;

    static QString fingerprint(const Link & link);
    void remember(const Link & link,double score,int inliers=0,int matches=0);
    const Score * score(const Link & link) const;
    bool disabled(const Link & link) const;
    void filterLinks(std::multimap<int,Link> & links) const;
    bool load(QString * error);
    bool save(QString * error);
};

class LoopSparsificationDialog : public QDialog
{
public:
    LoopSparsificationDialog(const std::map<int,Transform> & poses,
        const std::multimap<int,Link> & links,const std::map<int,int> & maps,
        const LoopSelectionState & state,double spacing,QWidget * parent=0);
    LoopSelectionState::Snapshot resultSelection() const;
private:
    LoopSelectionState::Snapshot result_;
};

}
#endif
