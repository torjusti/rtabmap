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

#ifndef RTABMAP_BASEMAPTILESOURCE_H_
#define RTABMAP_BASEMAPTILESOURCE_H_

#include "rtabmap/gui/rtabmap_gui_export.h" // DLL export/import defines

#include <QObject>
#include <QString>
#include <QMetaType>
#include <opencv2/core/core.hpp>
#include <memory>
#include <vector>
#include <list>
#include <map>
#include <set>
#include <thread>
#include <mutex>
#include <condition_variable>

namespace rtabmap {

class GeoRaster;

/**
 * Address of a tile in the quadtree pyramid built over the raster's pixel grid.
 * Level 0 is full resolution; level L tiles cover tileSize*2^L source pixels.
 */
struct RTABMAP_GUI_EXPORT BasemapTileKey
{
	int level;
	int x;
	int y;
	BasemapTileKey() : level(0), x(0), y(0) {}
	BasemapTileKey(int l, int tx, int ty) : level(l), x(tx), y(ty) {}
	bool operator<(const BasemapTileKey & o) const
	{
		if(level != o.level) return level < o.level;
		if(x != o.x) return x < o.x;
		return y < o.y;
	}
	bool operator==(const BasemapTileKey & o) const {return level==o.level && x==o.x && y==o.y;}
	bool operator!=(const BasemapTileKey & o) const {return !(*this == o);}
	BasemapTileKey parent() const {return BasemapTileKey(level+1, x>>1, y>>1);}
};

/**
 * Progressive, cached access to a huge georeferenced raster as a pyramid of
 * fixed-size tiles. Tiles are decoded on worker threads (through GDAL
 * overviews) and delivered to the GUI thread with tileReady(). The raster is
 * never loaded whole.
 *
 * Coordinates: "CRS" coordinates are the raster's projected coordinates.
 * "Map" coordinates are CRS coordinates minus a frame offset (the anchor
 * frame origin), so that they stay small enough for float32 rendering.
 *
 * All methods must be called from the thread owning the object (GUI thread),
 * except for the internal worker threads.
 */
class RTABMAP_GUI_EXPORT BasemapTileSource : public QObject
{
	Q_OBJECT

public:
	static const int kDefaultTileSize = 512;

public:
	BasemapTileSource(QObject * parent = 0);
	virtual ~BasemapTileSource();

	/**
	 * Opens the raster and starts worker threads. Returns false with an error
	 * message on failure (previous raster, if any, is closed anyway).
	 */
	bool load(const QString & path, QString * error = 0, int tileSize = kDefaultTileSize, int workerThreads = 3);
	void clear();
	bool isLoaded() const {return raster_.get() != 0;}

	QString path() const;
	QString crsName() const;
	QString crsAuthority() const;
	bool hasOverviews() const;
	bool isGeographic() const;
	int width() const;
	int height() const;
	int tileSize() const {return tileSize_;}
	// Coarsest level (a single tile covers the whole raster)
	int maxLevel() const {return maxLevel_;}
	// Ground size of a source pixel at given level (CRS units per texture pixel)
	double pixelSize(int level = 0) const;

	// Map frame = CRS - offset
	void setFrameOffset(double x, double y);
	double frameOffsetX() const {return offsetX_;}
	double frameOffsetY() const {return offsetY_;}
	void crsToMap(double crsX, double crsY, double & mapX, double & mapY) const;
	void mapToCrs(double mapX, double mapY, double & crsX, double & crsY) const;

	// Center of the raster in CRS coordinates
	void rasterCenterCrs(double & x, double & y) const;
	// Axis-aligned bounds of the whole raster in map coordinates
	void rasterBoundsMap(double & xMin, double & yMin, double & xMax, double & yMax) const;

	bool tileExists(const BasemapTileKey & key) const;
	// Source pixel window covered by a tile (clipped to the raster)
	bool tilePixelWindow(const BasemapTileKey & key, int & x0, int & y0, int & w, int & h) const;
	// Texture size of that tile (equals tileSize except on the right/bottom edges)
	bool tileTextureSize(const BasemapTileKey & key, int & w, int & h) const;
	// Corners in map coordinates: top-left, top-right, bottom-right, bottom-left
	// (in raster pixel orientation, i.e. matching texture coordinates (0,0),(1,0),(1,1),(0,1))
	bool tileCornersMap(const BasemapTileKey & key, double corners[4][2]) const;
	// Axis-aligned bounds in map coordinates
	bool tileBoundsMap(const BasemapTileKey & key, double & xMin, double & yMin, double & xMax, double & yMax) const;
	// The (up to 4) existing children of a tile (empty at level 0)
	std::vector<BasemapTileKey> children(const BasemapTileKey & key) const;
	BasemapTileKey rootKey() const {return BasemapTileKey(maxLevel_, 0, 0);}

	// Cached tile image (BGRA), empty if not (yet) loaded. Marks it as recently used.
	cv::Mat tile(const BasemapTileKey & key);
	bool isCached(const BasemapTileKey & key) const;
	/**
	 * Finest cached ancestor of key (may be key itself). Returns false if
	 * nothing usable is cached (not even the root).
	 */
	bool bestCached(const BasemapTileKey & key, BasemapTileKey & cached);

	/**
	 * Replaces the pending request set: tiles not in "wanted" that were pending
	 * are dropped (in-flight reads finish anyway). Lower priority value = read
	 * first. Already cached tiles are ignored.
	 */
	void request(const std::vector<std::pair<BasemapTileKey, double> > & wanted);
	void cancelPending();

	void setCacheLimitBytes(size_t bytes);
	size_t cacheLimitBytes() const {return cacheLimitBytes_;}
	size_t cacheBytes() const {return cacheBytes_;}
	int cachedTiles() const {return (int)cache_.size();}
	int pendingTiles() const;

Q_SIGNALS:
	void tileReady(const rtabmap::BasemapTileKey & key);
	void loaded();
	void cleared();

private:
	void workerLoop(int index);
	void stopWorkers();
	void deliverResults(); // GUI thread
	void touch(const BasemapTileKey & key);
	void evict();

private:
	std::shared_ptr<GeoRaster> raster_;
	int tileSize_;
	int maxLevel_;
	double offsetX_;
	double offsetY_;

	// cache (GUI thread only)
	struct CacheEntry
	{
		cv::Mat image;
		std::list<BasemapTileKey>::iterator lruIter;
	};
	std::map<BasemapTileKey, CacheEntry> cache_;
	std::list<BasemapTileKey> lru_; // front = most recently used
	size_t cacheBytes_;
	size_t cacheLimitBytes_;

	// worker state (shared)
	struct Result
	{
		BasemapTileKey key;
		cv::Mat image;
	};
	mutable std::mutex mutex_;
	std::condition_variable cond_;
	std::map<BasemapTileKey, double> pending_;
	std::set<BasemapTileKey> inFlight_;
	std::vector<Result> results_;
	bool resultsScheduled_;
	bool stopping_;
	unsigned int generation_; // bumped on clear() so stale results are dropped
	std::vector<std::thread> workers_;
	std::vector<std::shared_ptr<GeoRaster> > workerRasters_;
};

} // namespace rtabmap

Q_DECLARE_METATYPE(rtabmap::BasemapTileKey)

#endif /* RTABMAP_BASEMAPTILESOURCE_H_ */
