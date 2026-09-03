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

#include "rtabmap/gui/BasemapTileSource.h"
#include "rtabmap/gui/GeoRaster.h"
#include <rtabmap/utilite/ULogger.h>
#include <rtabmap/utilite/UTimer.h>
#include <rtabmap/utilite/UConversion.h>

#include <QMetaObject>
#include <algorithm>
#include <cmath>

namespace rtabmap {

BasemapTileSource::BasemapTileSource(QObject * parent) :
		QObject(parent),
		tileSize_(kDefaultTileSize),
		maxLevel_(0),
		offsetX_(0.0),
		offsetY_(0.0),
		cacheBytes_(0),
		cacheLimitBytes_(512*1024*1024),
		resultsScheduled_(false),
		stopping_(false),
		generation_(0)
{
	qRegisterMetaType<rtabmap::BasemapTileKey>("rtabmap::BasemapTileKey");
	qRegisterMetaType<rtabmap::BasemapTileKey>("BasemapTileKey");
}

BasemapTileSource::~BasemapTileSource()
{
	stopWorkers();
}

bool BasemapTileSource::load(const QString & path, QString * error, int tileSize, int workerThreads)
{
	clear();

	std::shared_ptr<GeoRaster> raster(new GeoRaster());
	std::string err;
	if(!raster->open(path.toStdString(), &err))
	{
		if(error)
		{
			*error = QString::fromStdString(err);
		}
		return false;
	}

	tileSize_ = std::max(64, tileSize);
	raster_ = raster;

	int maxDim = std::max(raster_->width(), raster_->height());
	maxLevel_ = 0;
	while(((long long)tileSize_ << maxLevel_) < maxDim)
	{
		++maxLevel_;
	}

	if(!raster_->hasOverviews() && maxLevel_ > 2)
	{
		UWARN("Raster \"%s\" (%dx%d) has no overviews: coarse tiles will be slow to read. "
			  "Build them with: gdaladdo -r average \"%s\" 2 4 8 16 32 64 128 256",
			  raster_->path().c_str(), raster_->width(), raster_->height(), raster_->path().c_str());
	}

	workerThreads = std::max(1, std::min(workerThreads, 8));
	{
		std::lock_guard<std::mutex> lock(mutex_);
		stopping_ = false;
	}
	for(int i=0; i<workerThreads; ++i)
	{
		workerRasters_.push_back(i==0?raster_:raster_->clone());
		if(!workerRasters_.back()->isOpen())
		{
			UWARN("Could not open an extra GDAL handle for worker %d, falling back to %d worker(s).", i, i);
			workerRasters_.pop_back();
			break;
		}
	}
	// the shared handle (index 0) is only used by worker 0, the GUI thread
	// never reads pixels itself.
	for(size_t i=0; i<workerRasters_.size(); ++i)
	{
		workers_.push_back(std::thread(&BasemapTileSource::workerLoop, this, (int)i));
	}

	UINFO("Basemap loaded: %s, %d levels (tile=%d px, root covers %.0f m), %d worker(s)",
			raster_->path().c_str(), maxLevel_+1, tileSize_,
			pixelSize(maxLevel_)*tileSize_, (int)workers_.size());
	Q_EMIT loaded();
	return true;
}

void BasemapTileSource::clear()
{
	stopWorkers();
	{
		std::lock_guard<std::mutex> lock(mutex_);
		++generation_;
		pending_.clear();
		inFlight_.clear();
		results_.clear();
		resultsScheduled_ = false;
	}
	cache_.clear();
	lru_.clear();
	cacheBytes_ = 0;
	workerRasters_.clear();
	bool wasLoaded = raster_.get() != 0;
	raster_.reset();
	maxLevel_ = 0;
	if(wasLoaded)
	{
		Q_EMIT cleared();
	}
}

void BasemapTileSource::stopWorkers()
{
	{
		std::lock_guard<std::mutex> lock(mutex_);
		stopping_ = true;
	}
	cond_.notify_all();
	for(size_t i=0; i<workers_.size(); ++i)
	{
		if(workers_[i].joinable())
		{
			workers_[i].join();
		}
	}
	workers_.clear();
	std::lock_guard<std::mutex> lock(mutex_);
	stopping_ = false;
}

QString BasemapTileSource::path() const
{
	return raster_?QString::fromStdString(raster_->path()):QString();
}
QString BasemapTileSource::crsName() const
{
	return raster_?QString::fromStdString(raster_->crsName()):QString();
}
QString BasemapTileSource::crsAuthority() const
{
	return raster_?QString::fromStdString(raster_->crsAuthority()):QString();
}
bool BasemapTileSource::hasOverviews() const
{
	return raster_ && raster_->hasOverviews();
}
bool BasemapTileSource::isGeographic() const
{
	return raster_ && raster_->isGeographic();
}
int BasemapTileSource::width() const
{
	return raster_?raster_->width():0;
}
int BasemapTileSource::height() const
{
	return raster_?raster_->height():0;
}

double BasemapTileSource::pixelSize(int level) const
{
	if(!raster_)
	{
		return 0.0;
	}
	return std::max(raster_->pixelSizeX(), raster_->pixelSizeY()) * double(1LL << std::max(0, level));
}

void BasemapTileSource::setFrameOffset(double x, double y)
{
	offsetX_ = x;
	offsetY_ = y;
}

void BasemapTileSource::crsToMap(double crsX, double crsY, double & mapX, double & mapY) const
{
	mapX = crsX - offsetX_;
	mapY = crsY - offsetY_;
}

void BasemapTileSource::mapToCrs(double mapX, double mapY, double & crsX, double & crsY) const
{
	crsX = mapX + offsetX_;
	crsY = mapY + offsetY_;
}

void BasemapTileSource::rasterCenterCrs(double & x, double & y) const
{
	x = y = 0.0;
	if(raster_)
	{
		raster_->pixelToWorld(raster_->width()/2.0, raster_->height()/2.0, x, y);
	}
}

void BasemapTileSource::rasterBoundsMap(double & xMin, double & yMin, double & xMax, double & yMax) const
{
	xMin = yMin = xMax = yMax = 0.0;
	if(raster_)
	{
		tileBoundsMap(rootKey(), xMin, yMin, xMax, yMax);
	}
}

bool BasemapTileSource::tileExists(const BasemapTileKey & key) const
{
	int x0, y0, w, h;
	return tilePixelWindow(key, x0, y0, w, h);
}

bool BasemapTileSource::tilePixelWindow(const BasemapTileKey & key, int & x0, int & y0, int & w, int & h) const
{
	if(!raster_ || key.level < 0 || key.level > maxLevel_ || key.x < 0 || key.y < 0)
	{
		return false;
	}
	long long span = (long long)tileSize_ << key.level;
	long long sx = key.x * span;
	long long sy = key.y * span;
	if(sx >= raster_->width() || sy >= raster_->height())
	{
		return false;
	}
	x0 = (int)sx;
	y0 = (int)sy;
	w = (int)std::min<long long>(span, raster_->width() - sx);
	h = (int)std::min<long long>(span, raster_->height() - sy);
	return w > 0 && h > 0;
}

bool BasemapTileSource::tileTextureSize(const BasemapTileKey & key, int & w, int & h) const
{
	int x0, y0, sw, sh;
	if(!tilePixelWindow(key, x0, y0, sw, sh))
	{
		return false;
	}
	long long f = 1LL << key.level;
	w = (int)((sw + f - 1) / f);
	h = (int)((sh + f - 1) / f);
	return true;
}

bool BasemapTileSource::tileCornersMap(const BasemapTileKey & key, double corners[4][2]) const
{
	int x0, y0, w, h;
	if(!tilePixelWindow(key, x0, y0, w, h))
	{
		return false;
	}
	double px[4] = {double(x0), double(x0+w), double(x0+w), double(x0)};
	double py[4] = {double(y0), double(y0),   double(y0+h), double(y0+h)};
	for(int i=0; i<4; ++i)
	{
		double wx, wy;
		raster_->pixelToWorld(px[i], py[i], wx, wy);
		crsToMap(wx, wy, corners[i][0], corners[i][1]);
	}
	return true;
}

bool BasemapTileSource::tileBoundsMap(const BasemapTileKey & key, double & xMin, double & yMin, double & xMax, double & yMax) const
{
	double c[4][2];
	if(!tileCornersMap(key, c))
	{
		return false;
	}
	xMin = xMax = c[0][0];
	yMin = yMax = c[0][1];
	for(int i=1; i<4; ++i)
	{
		xMin = std::min(xMin, c[i][0]);
		xMax = std::max(xMax, c[i][0]);
		yMin = std::min(yMin, c[i][1]);
		yMax = std::max(yMax, c[i][1]);
	}
	return true;
}

std::vector<BasemapTileKey> BasemapTileSource::children(const BasemapTileKey & key) const
{
	std::vector<BasemapTileKey> out;
	if(key.level <= 0)
	{
		return out;
	}
	for(int dy=0; dy<2; ++dy)
	{
		for(int dx=0; dx<2; ++dx)
		{
			BasemapTileKey child(key.level-1, key.x*2+dx, key.y*2+dy);
			if(tileExists(child))
			{
				out.push_back(child);
			}
		}
	}
	return out;
}

cv::Mat BasemapTileSource::tile(const BasemapTileKey & key)
{
	std::map<BasemapTileKey, CacheEntry>::iterator iter = cache_.find(key);
	if(iter == cache_.end())
	{
		return cv::Mat();
	}
	touch(key);
	return iter->second.image;
}

bool BasemapTileSource::isCached(const BasemapTileKey & key) const
{
	return cache_.find(key) != cache_.end();
}

bool BasemapTileSource::bestCached(const BasemapTileKey & key, BasemapTileKey & cached)
{
	if(!raster_)
	{
		return false;
	}
	BasemapTileKey k = key;
	for(;;)
	{
		if(isCached(k))
		{
			cached = k;
			touch(k);
			return true;
		}
		if(k.level >= maxLevel_)
		{
			return false;
		}
		k = k.parent();
	}
}

void BasemapTileSource::touch(const BasemapTileKey & key)
{
	std::map<BasemapTileKey, CacheEntry>::iterator iter = cache_.find(key);
	if(iter != cache_.end())
	{
		lru_.splice(lru_.begin(), lru_, iter->second.lruIter);
		iter->second.lruIter = lru_.begin();
	}
}

void BasemapTileSource::evict()
{
	BasemapTileKey root = rootKey();
	int skipped = 0;
	while(cacheBytes_ > cacheLimitBytes_ && (int)lru_.size() > 1 + skipped)
	{
		BasemapTileKey key = lru_.back();
		if(key == root)
		{
			// keep the root tile forever, there is always something to show
			lru_.splice(lru_.begin(), lru_, --lru_.end());
			cache_[key].lruIter = lru_.begin();
			++skipped;
			continue;
		}
		std::map<BasemapTileKey, CacheEntry>::iterator iter = cache_.find(key);
		if(iter != cache_.end())
		{
			cacheBytes_ -= iter->second.image.total()*iter->second.image.elemSize();
			cache_.erase(iter);
		}
		lru_.pop_back();
	}
}

void BasemapTileSource::setCacheLimitBytes(size_t bytes)
{
	cacheLimitBytes_ = bytes;
	evict();
}

int BasemapTileSource::pendingTiles() const
{
	std::lock_guard<std::mutex> lock(mutex_);
	return (int)(pending_.size() + inFlight_.size());
}

void BasemapTileSource::request(const std::vector<std::pair<BasemapTileKey, double> > & wanted)
{
	if(!raster_)
	{
		return;
	}
	bool notify = false;
	{
		std::lock_guard<std::mutex> lock(mutex_);
		pending_.clear();
		for(size_t i=0; i<wanted.size(); ++i)
		{
			const BasemapTileKey & key = wanted[i].first;
			if(cache_.find(key) == cache_.end() && inFlight_.find(key) == inFlight_.end() && tileExists(key))
			{
				std::map<BasemapTileKey, double>::iterator iter = pending_.find(key);
				if(iter == pending_.end())
				{
					pending_.insert(std::make_pair(key, wanted[i].second));
				}
				else if(wanted[i].second < iter->second)
				{
					iter->second = wanted[i].second;
				}
			}
		}
		notify = !pending_.empty();
	}
	if(notify)
	{
		cond_.notify_all();
	}
}

void BasemapTileSource::cancelPending()
{
	std::lock_guard<std::mutex> lock(mutex_);
	pending_.clear();
}

void BasemapTileSource::workerLoop(int index)
{
	ULogger::registerCurrentThread(uFormat("BasemapTile%d", index));
	std::shared_ptr<GeoRaster> raster = workerRasters_[index];
	for(;;)
	{
		BasemapTileKey key;
		unsigned int generation;
		{
			std::unique_lock<std::mutex> lock(mutex_);
			cond_.wait(lock, [this]{return stopping_ || !pending_.empty();});
			if(stopping_)
			{
				ULogger::unregisterCurrentThread();
				return;
			}
			std::map<BasemapTileKey, double>::iterator best = pending_.begin();
			for(std::map<BasemapTileKey, double>::iterator iter = pending_.begin(); iter != pending_.end(); ++iter)
			{
				if(iter->second < best->second)
				{
					best = iter;
				}
			}
			key = best->first;
			pending_.erase(best);
			inFlight_.insert(key);
			generation = generation_;
		}

		cv::Mat image;
		int x0, y0, w, h, tw, th;
		if(tilePixelWindow(key, x0, y0, w, h) && tileTextureSize(key, tw, th))
		{
			UTimer timer;
			image = raster->read(x0, y0, w, h, tw, th);
			UDEBUG("Tile L%d (%d,%d) %dx%d -> %dx%d read in %.3fs", key.level, key.x, key.y, w, h, tw, th, timer.ticks());
		}

		bool schedule = false;
		{
			std::lock_guard<std::mutex> lock(mutex_);
			inFlight_.erase(key);
			if(stopping_)
			{
				ULogger::unregisterCurrentThread();
				return;
			}
			if(generation == generation_)
			{
				Result r;
				r.key = key;
				r.image = image;
				results_.push_back(r);
				if(!resultsScheduled_)
				{
					resultsScheduled_ = true;
					schedule = true;
				}
			}
		}
		if(schedule)
		{
			QMetaObject::invokeMethod(this, [this](){this->deliverResults();}, Qt::QueuedConnection);
		}
	}
}

void BasemapTileSource::deliverResults()
{
	std::vector<Result> results;
	{
		std::lock_guard<std::mutex> lock(mutex_);
		results.swap(results_);
		resultsScheduled_ = false;
	}
	if(!raster_)
	{
		return;
	}
	for(size_t i=0; i<results.size(); ++i)
	{
		const BasemapTileKey & key = results[i].key;
		if(results[i].image.empty())
		{
			UWARN("Failed to read basemap tile L%d (%d,%d)", key.level, key.x, key.y);
			continue;
		}
		std::map<BasemapTileKey, CacheEntry>::iterator iter = cache_.find(key);
		if(iter != cache_.end())
		{
			cacheBytes_ -= iter->second.image.total()*iter->second.image.elemSize();
			iter->second.image = results[i].image;
			touch(key);
		}
		else
		{
			lru_.push_front(key);
			CacheEntry entry;
			entry.image = results[i].image;
			entry.lruIter = lru_.begin();
			cache_.insert(std::make_pair(key, entry));
		}
		cacheBytes_ += results[i].image.total()*results[i].image.elemSize();
	}
	evict();
	for(size_t i=0; i<results.size(); ++i)
	{
		if(!results[i].image.empty())
		{
			Q_EMIT tileReady(results[i].key);
		}
	}
}

} // namespace rtabmap
