#include "rtabmap/gui/DatabaseViewer.h"
#include "rtabmap/utilite/ULogger.h"
#include "rtabmap/core/util3d.h"
#include "rtabmap/core/util3d_transforms.h"
#include "rtabmap/core/util3d_filtering.h"
#include "rtabmap/core/Graph.h"
#include "rtabmap/core/Optimizer.h"
#include "rtabmap/core/DBDriver.h"

#include "ui_DatabaseViewer.h"

#include <QInputDialog>
#include <QMessageBox>

namespace rtabmap {


void DatabaseViewer::handleImageAClicked(float x, float y, float depth, const cv::Point3f & pt3d) 
{
    this->handleImageClicked(0, x, y, depth, pt3d);
}

void DatabaseViewer::handleImageBClicked(float x, float y, float depth, const cv::Point3f & pt3d) 
{
    this->handleImageClicked(1, x, y, depth, pt3d);
}

void DatabaseViewer::handleImageClicked(int imageIndex, float x, float y, float depth, const cv::Point3f & pt3d) 
{
    UDEBUG("handleImageClicked at pixel (%f,%f) with depth=%f and 3D point (%f,%f,%f)", 
           x, y, depth, pt3d.x, pt3d.y, pt3d.z);

    if(depth <= 0 || std::isnan(depth) || 
       std::isnan(pt3d.x) || std::isnan(pt3d.y) || std::isnan(pt3d.z))
    {
        QMessageBox::warning(this, tr("Add Pose Prior"),
                            tr("Cannot add pose prior, the selected image point has no valid depth."));
        return;
    }

    if(ids_.empty())
    {
        QMessageBox::warning(this, tr("Add Pose Prior"),
                            tr("Cannot add pose prior, no nodes loaded."));
        return;
    }
    int currentId = ids_.at(ui_->horizontalSlider_A->value());
    if(currentId <= 0)
    {
        QMessageBox::warning(this, tr("Add Pose Prior"),
                            tr("Cannot add pose prior, invalid node ID."));
        return;
    }

    std::map<int, Transform>::iterator poseIter = odomPoses_.find(currentId);
    if(poseIter == odomPoses_.end())
    {
        QMessageBox::warning(this, tr("Add Pose Prior"),
                            tr("Cannot add pose prior, node %1 doesn't have a pose.").arg(currentId));
        return;
    }

    this->editConstraint(imageIndex, pt3d.x, pt3d.y, pt3d.z);
}

}