#include "LoopSelectionState.h"
#include <QAbstractTableModel>
#include <QDialogButtonBox>
#include <QDoubleSpinBox>
#include <QHeaderView>
#include <QLabel>
#include <QPushButton>
#include <QTableView>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QProgressDialog>
#include <QApplication>
#include <QElapsedTimer>
#include <QSortFilterProxyModel>

namespace rtabmap {
namespace {
class SelectionModel : public QAbstractTableModel
{
public:
    struct Row { Link link; double score=0; bool active=true; };
    std::vector<Row> rows;
    std::set<loopSelection::Pair> protectedLinks;
    loopSelection::Result preview;
    std::function<void()> invalidate;
    explicit SelectionModel(QObject * parent):QAbstractTableModel(parent) {}
    int rowCount(const QModelIndex & p=QModelIndex()) const override {return p.isValid()?0:int(rows.size());}
    int columnCount(const QModelIndex & p=QModelIndex()) const override {return p.isValid()?0:7;}
    QVariant headerData(int c,Qt::Orientation o,int role) const override
    {
        if(o!=Qt::Horizontal || role!=Qt::DisplayRole) return {};
        return QStringList({"Protect","From","To","Type","Quality score","Currently","Preview"}).value(c);
    }
    QVariant data(const QModelIndex & i,int role) const override
    {
        if(!i.isValid()) return {};
        const auto & r=rows.at(i.row()); auto p=loopSelection::ordered(r.link.from(),r.link.to());
        if(i.column()==0 && role==Qt::CheckStateRole) return protectedLinks.count(p)?Qt::Checked:Qt::Unchecked;
        if(role!=Qt::DisplayRole) return {};
        switch(i.column())
        {
        case 1:return p.first; case 2:return p.second;
        case 3:return QString::fromStdString(r.link.typeName());
        case 4:return r.score>0?QVariant(r.score):QVariant("Unavailable");
        case 5:return r.active?"Active":"Excluded";
        case 6:
            if(!preview.reasons.count(p)) return "Preview needed";
            switch(preview.reasons.at(p)) {
            case loopSelection::Protected:return "Keep: protected";
            case loopSelection::Connectivity:return "Keep: connectivity";
            case loopSelection::MissingGeometry:return "Keep: missing odometry";
            case loopSelection::Redundant:return "Exclude: spacing";
            default:return r.score>0?"Keep: quality / spacing":"Keep: ID fallback / spacing";
            }
        }
        return {};
    }
    Qt::ItemFlags flags(const QModelIndex & i) const override
    {return QAbstractTableModel::flags(i) | (i.column()==0?Qt::ItemIsUserCheckable:Qt::NoItemFlags);}
    bool setData(const QModelIndex & i,const QVariant & value,int role) override
    {
        if(!i.isValid() || i.column()!=0 || role!=Qt::CheckStateRole) return false;
        auto p=loopSelection::ordered(rows[i.row()].link.from(),rows[i.row()].link.to());
        if(value.toInt()==Qt::Checked) protectedLinks.insert(p); else protectedLinks.erase(p);
        Q_EMIT dataChanged(i,i); if(invalidate) invalidate(); return true;
    }
    void refresh() {if(!rows.empty()) Q_EMIT dataChanged(index(0,0),index(rows.size()-1,6));}
};
}
LoopSparsificationDialog::LoopSparsificationDialog(const std::map<int,Transform> & poses,
    const std::multimap<int,Link> & links,const std::map<int,int> & maps,
    const LoopSelectionState & state,double spacing,QWidget * parent):QDialog(parent)
{
    setWindowTitle(tr("Sparsify loop closures")); resize(1050,650);
    auto layout=new QVBoxLayout(this);
    auto explanation=new QLabel(tr("Select a sparse set of active loops across the entire database. Odometry and other non-loop constraints stay active.\n"
        "Existing connectivity is preserved; protected loops are always kept. Higher saved quality scores are considered first.\n"
        "Unscored loops use node-ID order. No image matching is run. Selection is saved beside the database; other RTAB-Map tools ignore it."),this);
    explanation->setWordWrap(true); layout->addWidget(explanation);
    auto controls=new QHBoxLayout; layout->addLayout(controls);
    controls->addWidget(new QLabel(tr("Spacing along both odometry paths:"),this));
    auto distance=new QDoubleSpinBox(this); distance->setRange(0.01,10000); distance->setDecimals(2);
    distance->setSuffix(" m"); distance->setValue(spacing); controls->addWidget(distance); controls->addStretch();
    auto previewButton=new QPushButton(tr("Preview"),this); controls->addWidget(previewButton);
    auto summary=new QLabel(tr("Press Preview to compute the selection."),this); summary->setWordWrap(true); layout->addWidget(summary);
    auto model=new SelectionModel(this); model->protectedLinks=state.selection.protectedLinks;
    std::map<loopSelection::Pair,Link> unique;
    for(const auto & e : links) if(loopSelection::isLoop(e.second))
    {
        auto p=loopSelection::ordered(e.second.from(),e.second.to());
        if(!unique.count(p) || e.second.from()>e.second.to()) unique[p]=e.second;
    }
    std::map<loopSelection::Pair,double> scores;
    for(const auto & e : unique)
    {
        SelectionModel::Row r; r.link=e.second; r.active=!state.disabled(r.link);
        const auto * s=state.score(r.link); if(s) {r.score=s->value; scores[e.first]=r.score;}
        model->rows.push_back(r);
    }
    auto table=new QTableView(this); auto proxy=new QSortFilterProxyModel(this); proxy->setSourceModel(model);
    table->setModel(proxy); table->setSortingEnabled(true); table->sortByColumn(1,Qt::AscendingOrder);
    table->verticalHeader()->hide(); table->horizontalHeader()->setStretchLastSection(true);
    table->setColumnWidth(4,130); table->setColumnWidth(3,140); layout->addWidget(table);
    auto buttons=new QDialogButtonBox(QDialogButtonBox::Apply|QDialogButtonBox::Cancel,this); layout->addWidget(buttons);
    auto apply=buttons->button(QDialogButtonBox::Apply); apply->setText(tr("Apply selection")); apply->setEnabled(false);
    model->invalidate=[model,apply,summary]() {model->preview={}; apply->setEnabled(false); summary->setText(tr("Selection changed. Press Preview again.")); model->refresh();};
    connect(distance,QOverload<double>::of(&QDoubleSpinBox::valueChanged),this,[model](double){model->invalidate();});
    connect(buttons,&QDialogButtonBox::rejected,this,&QDialog::reject);
    connect(previewButton,&QPushButton::clicked,this,[=,&poses,&links,&maps]() {
        table->setEnabled(false); distance->setEnabled(false); previewButton->setEnabled(false); apply->setEnabled(false);
        QProgressDialog progress(tr("Selecting loops…"),tr("Cancel"),0,0,this); progress.setWindowModality(Qt::WindowModal); progress.setMinimumDuration(0);
        QElapsedTimer timer; timer.start(); QElapsedTimer events; events.start();
        auto result=loopSelection::select(poses,links,maps,scores,model->protectedLinks,distance->value(),[&]() {
            if(events.elapsed()>30) {QApplication::processEvents(); events.restart();} return progress.wasCanceled();
        });
        table->setEnabled(true); distance->setEnabled(true); previewButton->setEnabled(true);
        if(result.canceled) {model->invalidate(); summary->setText(tr("Preview canceled. Current active loops are unchanged.")); return;}
        model->preview=result; model->refresh(); result_=LoopSelectionState::Snapshot(); result_.protectedLinks=model->protectedLinks;
        int excluded=0,protect=0,bridge=0,missing=0;
        for(const auto & e : result.reasons) {
            if(e.second==loopSelection::Redundant) {++excluded; result_.disabled[e.first]=LoopSelectionState::fingerprint(unique.at(e.first));}
            if(e.second==loopSelection::Protected) ++protect;
            if(e.second==loopSelection::Connectivity) ++bridge;
            if(e.second==loopSelection::MissingGeometry) ++missing;
        }
        summary->setText(tr("Keep %1 / %2 loops; exclude %3. Protected: %4; connectivity exceptions: %5; missing odometry: %6.\n"
            "%7 loops have no saved quality score. Preview took %8 s. Apply keeps all database constraints intact.")
            .arg(result.reasons.size()-excluded).arg(result.reasons.size()).arg(excluded).arg(protect).arg(bridge).arg(missing)
            .arg(unique.size()-scores.size()).arg(timer.elapsed()/1000.0,0,'f',2));
        apply->setEnabled(true);
    });
    connect(apply,&QPushButton::clicked,this,&QDialog::accept);
}
LoopSelectionState::Snapshot LoopSparsificationDialog::resultSelection() const {return result_;}
}
