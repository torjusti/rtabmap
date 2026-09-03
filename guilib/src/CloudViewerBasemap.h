/*
Copyright (c) 2010-2026, Mathieu Labbe - IntRoLab - Universite de Sherbrooke
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Universite de Sherbrooke nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef RTABMAP_CLOUDVIEWERBASEMAP_H_
#define RTABMAP_CLOUDVIEWERBASEMAP_H_

#include "rtabmap/gui/BasemapTileSource.h"

#include <QObject>
#include <QTimer>
#include <vtkSmartPointer.h>
#include <map>
#include <memory>
#include <vector>

class vtkRenderer;
class vtkActor;
class vtkProp;
class vtkObject;

namespace rtabmap {

/**
 * Renders a BasemapTileSource in a VTK renderer as a set of textured quads
 * at a fixed elevation, refining the quadtree where the camera looks
 * (level of detail from the projected tile size) and requesting missing
 * tiles asynchronously. Coarser (already cached) ancestors are shown while
 * finer tiles are loading so the ground is never empty.
 */
class CloudViewerBasemap : public QObject
{
	Q_OBJECT

public:
	CloudViewerBasemap(vtkRenderer * renderer, QObject * parent = 0);
	virtual ~CloudViewerBasemap();

	void setSource(const std::shared_ptr<BasemapTileSource> & source);
	const std::shared_ptr<BasemapTileSource> & source() const {return source_;}
	bool isActive() const {return source_ && source_->isLoaded();}

	void setZ(double z);
	double z() const {return z_;}
	void setOpacity(double opacity);
	double opacity() const {return opacity_;}
	void setVisible(bool visible);
	bool isVisible() const {return visible_;}
	// Upper bound on displayed tiles (memory/GPU guard)
	void setMaxTiles(int maxTiles) {maxTiles_ = maxTiles;}
	// Screen pixels per texture pixel above which a tile is refined (1 = full detail)
	void setDetail(double screenPixelsPerTexel) {detail_ = screenPixelsPerTexel;}

	bool isBasemapActor(vtkProp * prop) const;
	std::vector<vtkProp*> actors() const;
	int displayedTiles() const {return (int)actors_.size();}

	// Picks the basemap under display coordinates (x, y), ignoring everything
	// else in the renderer. Returns the point in map coordinates.
	bool pick(int displayX, int displayY, double & mapX, double & mapY) const;

	// Recomputes the set of displayed tiles from the current camera (called
	// automatically when the camera moves; force ignores the camera check).
	void update(bool force = false);

Q_SIGNALS:
	// Emitted when displayed tiles changed and the view should be re-rendered
	void changed();

private Q_SLOTS:
	void poll();
	void onTileReady(const rtabmap::BasemapTileKey & key);
	void onSourceCleared();
	void onSourceLoaded();

private:
	void clearActors();
	// Widens the camera clipping range so that the displayed tiles (which are
	// excluded from the automatic bounds computation) are not cut off.
	void extendClippingRange() const;
	static void onResetClippingRange(vtkObject * caller, unsigned long eventId, void * clientData, void * callData);
	vtkSmartPointer<vtkActor> createActor(const BasemapTileKey & key, const cv::Mat & image) const;
	void applyProperties(vtkActor * actor, int level) const;
	bool computeTargets(std::vector<BasemapTileKey> & targets, double & focalX, double & focalY) const;

private:
	vtkRenderer * renderer_;
	std::shared_ptr<BasemapTileSource> source_;
	double z_;
	double opacity_;
	bool visible_;
	int maxTiles_;
	double detail_;
	QTimer timer_;
	unsigned long lastCameraMTime_;
	int lastSize_[2];
	bool dirty_;
	unsigned long clippingObserverTag_;
	std::map<BasemapTileKey, vtkSmartPointer<vtkActor> > actors_;
};

} // namespace rtabmap

#endif /* RTABMAP_CLOUDVIEWERBASEMAP_H_ */
