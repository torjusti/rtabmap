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

#ifndef RTABMAP_GEORASTER_H_
#define RTABMAP_GEORASTER_H_

#include "rtabmap/gui/rtabmap_gui_export.h" // DLL export/import defines

#include <opencv2/core/core.hpp>
#include <memory>
#include <string>

namespace rtabmap {

/**
 * Thin read-only wrapper over a GDAL raster dataset (GeoTIFF, COG, VRT...).
 *
 * A GeoRaster instance is NOT thread-safe: GDAL dataset handles must be used
 * from a single thread at a time. Use clone() to get an independent handle
 * on the same file for another thread.
 *
 * Pixel <-> world mapping follows the GDAL geotransform convention:
 *   X = gt[0] + px*gt[1] + py*gt[2]
 *   Y = gt[3] + px*gt[4] + py*gt[5]
 * where (px, py) is the pixel/line coordinate of the top-left corner of a pixel.
 */
class RTABMAP_GUI_EXPORT GeoRaster
{
public:
	// true when RTAB-Map was built with GDAL
	static bool isAvailable();
	// File filter for QFileDialog-like usage (without Qt dependency)
	static std::string fileFilter();

public:
	GeoRaster();
	~GeoRaster();

	bool open(const std::string & path, std::string * error = 0);
	void close();
	bool isOpen() const;

	// Opens another handle on the same file (for use in another thread).
	std::shared_ptr<GeoRaster> clone() const;

	const std::string & path() const {return path_;}
	int width() const {return width_;}
	int height() const {return height_;}
	int bands() const {return bands_;}
	bool hasOverviews() const {return overviews_ > 0;}
	int overviewCount() const {return overviews_;}
	bool hasGeoTransform() const {return hasGeoTransform_;}
	const double * geoTransform() const {return geoTransform_;}
	// Human readable CRS name (e.g. "RGF93 v1 / Lambert-93"), empty if unknown.
	const std::string & crsName() const {return crsName_;}
	// "EPSG:2154" style identifier, empty if unknown.
	const std::string & crsAuthority() const {return crsAuthority_;}
	// true if the CRS is geographic (lat/long in degrees): distances are not metres.
	bool isGeographic() const {return isGeographic_;}

	// Approximate ground size of one pixel (in CRS units, metres for projected CRS)
	double pixelSizeX() const;
	double pixelSizeY() const;

	void pixelToWorld(double px, double py, double & wx, double & wy) const;
	void worldToPixel(double wx, double wy, double & px, double & py) const;

	/**
	 * Reads the source window [x0, x0+w) x [y0, y0+h) (pixels, must be inside
	 * the raster) resampled to an outW x outH BGRA (CV_8UC4) image. Nodata and
	 * mask bands are written to the alpha channel. GDAL picks the closest
	 * overview level automatically when outW < w, so reading a coarse view of a
	 * huge raster is cheap as long as overviews exist.
	 * Returns an empty matrix on error.
	 */
	cv::Mat read(int x0, int y0, int w, int h, int outW, int outH) const;

private:
	void * dataset_; // GDALDatasetH
	std::string path_;
	int width_;
	int height_;
	int bands_;
	int overviews_;
	bool hasGeoTransform_;
	double geoTransform_[6];
	double invGeoTransform_[6];
	std::string crsName_;
	std::string crsAuthority_;
	bool isGeographic_;
	// Byte rasters are read as is. Other types are linearly scaled to 8 bits
	// using approximate statistics computed at open().
	bool isByte_;
	bool hasPalette_;
	double scaleMin_;
	double scaleMax_;
};

} // namespace rtabmap

#endif /* RTABMAP_GEORASTER_H_ */
