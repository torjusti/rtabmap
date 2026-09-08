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
