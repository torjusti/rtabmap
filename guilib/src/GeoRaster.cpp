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

#include "rtabmap/gui/GeoRaster.h"
#include "rtabmap/core/Version.h"
#include <rtabmap/utilite/ULogger.h>
#include <rtabmap/utilite/UConversion.h>

#include <opencv2/imgproc/imgproc.hpp>

#include <cmath>
#include <mutex>
#include <vector>

#ifdef RTABMAP_GDAL
#include <gdal.h>
#include <gdal_version.h>
#include <ogr_srs_api.h>
#include <cpl_conv.h>
#include <cpl_error.h>
#endif

namespace rtabmap {

bool GeoRaster::isAvailable()
{
#ifdef RTABMAP_GDAL
	return true;
#else
	return false;
#endif
}

std::string GeoRaster::fileFilter()
{
	return "Georeferenced rasters (*.tif *.tiff *.vrt *.jp2 *.img *.ecw *.png *.jpg);;All files (*)";
}

#ifdef RTABMAP_GDAL
namespace {
void registerGdalOnce()
{
	static std::once_flag flag;
	std::call_once(flag, [](){
		GDALAllRegister();
		// GDAL prints errors to stderr by default, route them to our logger
		CPLSetErrorHandler([](CPLErr eErrClass, CPLErrorNum, const char * msg)
		{
			if(eErrClass == CE_Failure || eErrClass == CE_Fatal)
			{
				UERROR("GDAL: %s", msg);
			}
			else if(eErrClass == CE_Warning)
			{
				UWARN("GDAL: %s", msg);
			}
			else
			{
				UDEBUG("GDAL: %s", msg);
			}
		});
	});
}
}
#endif

GeoRaster::GeoRaster() :
		dataset_(0),
		width_(0),
		height_(0),
		bands_(0),
		overviews_(0),
		hasGeoTransform_(false),
		isGeographic_(false),
		isByte_(true),
		hasPalette_(false),
		scaleMin_(0.0),
		scaleMax_(255.0)
{
	for(int i=0; i<6; ++i)
	{
		geoTransform_[i] = 0.0;
		invGeoTransform_[i] = 0.0;
	}
	geoTransform_[1] = 1.0;
	geoTransform_[5] = 1.0;
	invGeoTransform_[1] = 1.0;
	invGeoTransform_[5] = 1.0;
}

GeoRaster::~GeoRaster()
{
	close();
}

bool GeoRaster::isOpen() const
{
	return dataset_ != 0;
}

void GeoRaster::close()
{
#ifdef RTABMAP_GDAL
	if(dataset_)
	{
		GDALClose((GDALDatasetH)dataset_);
	}
#endif
	dataset_ = 0;
}

std::shared_ptr<GeoRaster> GeoRaster::clone() const
{
	std::shared_ptr<GeoRaster> other(new GeoRaster());
	if(isOpen())
	{
		other->open(path_);
	}
	return other;
}

bool GeoRaster::open(const std::string & path, std::string * error)
{
	close();
#ifndef RTABMAP_GDAL
	if(error)
	{
		*error = "RTAB-Map was not built with GDAL support (WITH_GDAL=ON is required to load georeferenced rasters).";
	}
	return false;
#else
	registerGdalOnce();

	GDALDatasetH ds = GDALOpenEx(path.c_str(), GDAL_OF_RASTER | GDAL_OF_READONLY | GDAL_OF_VERBOSE_ERROR, 0, 0, 0);
	if(!ds)
	{
		if(error)
		{
			*error = uFormat("Cannot open \"%s\": %s", path.c_str(), CPLGetLastErrorMsg());
		}
		return false;
	}
	dataset_ = ds;
	path_ = path;
	width_ = GDALGetRasterXSize(ds);
	height_ = GDALGetRasterYSize(ds);
	bands_ = GDALGetRasterCount(ds);
	if(width_ <= 0 || height_ <= 0 || bands_ <= 0)
	{
		if(error)
		{
			*error = uFormat("\"%s\" has no raster data (%dx%d, %d bands).", path.c_str(), width_, height_, bands_);
		}
		close();
		return false;
	}

	hasGeoTransform_ = GDALGetGeoTransform(ds, geoTransform_) == CE_None;
	if(!hasGeoTransform_)
	{
		// Pixel coordinates, Y down: keep the same orientation convention as
		// a north-up raster so the rest of the code does not need to care.
		geoTransform_[0] = 0.0; geoTransform_[1] = 1.0; geoTransform_[2] = 0.0;
		geoTransform_[3] = 0.0; geoTransform_[4] = 0.0; geoTransform_[5] = -1.0;
		UWARN("\"%s\" has no georeferencing (geotransform), assuming 1 m pixels at origin.", path.c_str());
	}
	if(!GDALInvGeoTransform(geoTransform_, invGeoTransform_))
	{
		if(error)
		{
			*error = uFormat("\"%s\" has a singular geotransform.", path.c_str());
		}
		close();
		return false;
	}

	crsName_.clear();
	crsAuthority_.clear();
	isGeographic_ = false;
#if GDAL_VERSION_NUM >= GDAL_COMPUTE_VERSION(3,0,0)
	OGRSpatialReferenceH srs = GDALGetSpatialRef(ds);
	bool ownSrs = false;
#else
	const char * wkt = GDALGetProjectionRef(ds);
	OGRSpatialReferenceH srs = (wkt && wkt[0])?OSRNewSpatialReference(wkt):0;
	bool ownSrs = srs != 0;
#endif
	if(srs)
	{
#if GDAL_VERSION_NUM >= GDAL_COMPUTE_VERSION(2,3,0)
		const char * name = OSRGetName(srs);
		if(name) crsName_ = name;
#endif
		const char * authName = OSRGetAuthorityName(srs, 0);
		const char * authCode = OSRGetAuthorityCode(srs, 0);
		if(authName && authCode)
		{
			crsAuthority_ = uFormat("%s:%s", authName, authCode);
		}
		isGeographic_ = OSRIsGeographic(srs) != 0;
		if(ownSrs)
		{
			OSRDestroySpatialReference(srs);
		}
	}

	GDALRasterBandH band1 = GDALGetRasterBand(ds, 1);
	overviews_ = GDALGetOverviewCount(band1);
	GDALDataType type = GDALGetRasterDataType(band1);
	isByte_ = type == GDT_Byte;
	hasPalette_ = GDALGetRasterColorInterpretation(band1) == GCI_PaletteIndex && GDALGetRasterColorTable(band1) != 0;
	scaleMin_ = 0.0;
	scaleMax_ = 255.0;
	if(!isByte_ && !hasPalette_)
	{
		double minMax[2] = {0.0, 255.0};
		if(GDALComputeRasterMinMax(band1, TRUE, minMax) == CE_None && minMax[1] > minMax[0])
		{
			scaleMin_ = minMax[0];
			scaleMax_ = minMax[1];
		}
		else
		{
			scaleMax_ = type == GDT_UInt16?65535.0:255.0;
		}
	}

	UINFO("Opened raster \"%s\": %dx%d px, %d band(s), type=%s, overviews=%d, pixel size=%.4f x %.4f, CRS=%s (%s)%s",
			path.c_str(), width_, height_, bands_, GDALGetDataTypeName(type), overviews_,
			pixelSizeX(), pixelSizeY(),
			crsName_.empty()?"unknown":crsName_.c_str(),
			crsAuthority_.empty()?"no authority":crsAuthority_.c_str(),
			isGeographic_?" [geographic: units are degrees, not metres!]":"");
	return true;
#endif
}

double GeoRaster::pixelSizeX() const
{
	return std::sqrt(geoTransform_[1]*geoTransform_[1] + geoTransform_[4]*geoTransform_[4]);
}

double GeoRaster::pixelSizeY() const
{
	return std::sqrt(geoTransform_[2]*geoTransform_[2] + geoTransform_[5]*geoTransform_[5]);
}

void GeoRaster::pixelToWorld(double px, double py, double & wx, double & wy) const
{
	wx = geoTransform_[0] + px*geoTransform_[1] + py*geoTransform_[2];
	wy = geoTransform_[3] + px*geoTransform_[4] + py*geoTransform_[5];
}

void GeoRaster::worldToPixel(double wx, double wy, double & px, double & py) const
{
	px = invGeoTransform_[0] + wx*invGeoTransform_[1] + wy*invGeoTransform_[2];
	py = invGeoTransform_[3] + wx*invGeoTransform_[4] + wy*invGeoTransform_[5];
}

cv::Mat GeoRaster::read(int x0, int y0, int w, int h, int outW, int outH) const
{
#ifndef RTABMAP_GDAL
	return cv::Mat();
#else
	if(!dataset_ || w <= 0 || h <= 0 || outW <= 0 || outH <= 0 ||
	   x0 < 0 || y0 < 0 || x0 + w > width_ || y0 + h > height_)
	{
		UERROR("Invalid read window (%d,%d,%d,%d -> %dx%d) for raster %dx%d", x0, y0, w, h, outW, outH, width_, height_);
		return cv::Mat();
	}
	GDALDatasetH ds = (GDALDatasetH)dataset_;

	GDALRasterIOExtraArg extra;
	INIT_RASTERIO_EXTRA_ARG(extra);
	extra.eResampleAlg = hasPalette_?GRIORA_NearestNeighbour:GRIORA_Average;

	cv::Mat out(outH, outW, CV_8UC4, cv::Scalar(0, 0, 0, 255));

	// Figure out which bands hold B, G, R (in that order) and alpha.
	std::vector<int> colorBands;
	int alphaBand = 0;
	if(hasPalette_)
	{
		colorBands.push_back(1);
	}
	else if(bands_ >= 3)
	{
		int r = 0, g = 0, b = 0;
		for(int i=1; i<=bands_; ++i)
		{
			GDALColorInterp ci = GDALGetRasterColorInterpretation(GDALGetRasterBand(ds, i));
			if(ci == GCI_RedBand && !r) r = i;
			else if(ci == GCI_GreenBand && !g) g = i;
			else if(ci == GCI_BlueBand && !b) b = i;
			else if(ci == GCI_AlphaBand && !alphaBand) alphaBand = i;
		}
		if(!r || !g || !b)
		{
			r = 1; g = 2; b = 3;
		}
		if(!alphaBand && bands_ >= 4)
		{
			alphaBand = 4;
		}
		colorBands.push_back(b);
		colorBands.push_back(g);
		colorBands.push_back(r);
	}
	else
	{
		colorBands.push_back(1);
		if(bands_ == 2)
		{
			alphaBand = 2;
		}
	}

	int n = (int)colorBands.size();
	if(hasPalette_)
	{
		cv::Mat indices(outH, outW, CV_8UC1);
		if(GDALRasterIOEx(GDALGetRasterBand(ds, 1), GF_Read, x0, y0, w, h,
				indices.data, outW, outH, GDT_Byte, 1, (GSpacing)indices.step, &extra) != CE_None)
		{
			return cv::Mat();
		}
		GDALColorTableH ct = GDALGetRasterColorTable(GDALGetRasterBand(ds, 1));
		int count = GDALGetColorEntryCount(ct);
		bool rgb = GDALGetPaletteInterpretation(ct) == GPI_RGB;
		cv::Mat lut(1, 256, CV_8UC4, cv::Scalar(0, 0, 0, 255));
		for(int i=0; i<256 && i<count; ++i)
		{
			const GDALColorEntry * e = GDALGetColorEntry(ct, i);
			if(rgb)
			{
				lut.at<cv::Vec4b>(0, i) = cv::Vec4b((uchar)e->c3, (uchar)e->c2, (uchar)e->c1, (uchar)e->c4);
			}
			else
			{
				lut.at<cv::Vec4b>(0, i) = cv::Vec4b((uchar)e->c1, (uchar)e->c1, (uchar)e->c1, 255);
			}
		}
		cv::Mat indices4;
		cv::cvtColor(indices, indices4, cv::COLOR_GRAY2BGRA);
		cv::LUT(indices4, lut, out);
	}
	else if(isByte_)
	{
		if(GDALDatasetRasterIOEx(ds, GF_Read, x0, y0, w, h,
				out.data, outW, outH, GDT_Byte, n, &colorBands[0],
				4, (GSpacing)out.step, 1, &extra) != CE_None)
		{
			return cv::Mat();
		}
		if(n == 1)
		{
			// replicate gray to the 3 color channels
			int fromTo[] = {0,1, 0,2};
			cv::mixChannels(&out, 1, &out, 1, fromTo, 2);
		}
		if(alphaBand)
		{
			if(GDALRasterIOEx(GDALGetRasterBand(ds, alphaBand), GF_Read, x0, y0, w, h,
					out.data + 3, outW, outH, GDT_Byte, 4, (GSpacing)out.step, &extra) != CE_None)
			{
				return cv::Mat();
			}
		}
	}
	else
	{
		cv::Mat tmp(outH, outW, CV_32FC(n));
		if(GDALDatasetRasterIOEx(ds, GF_Read, x0, y0, w, h,
				tmp.data, outW, outH, GDT_Float32, n, &colorBands[0],
				4*n, (GSpacing)tmp.step, 4, &extra) != CE_None)
		{
			return cv::Mat();
		}
		double alpha = 255.0/(scaleMax_-scaleMin_);
		cv::Mat tmp8;
		tmp.convertTo(tmp8, CV_8U, alpha, -scaleMin_*alpha);
		if(n == 1)
		{
			int fromTo[] = {0,0, 0,1, 0,2};
			cv::mixChannels(&tmp8, 1, &out, 1, fromTo, 3);
		}
		else
		{
			int fromTo[] = {0,0, 1,1, 2,2};
			cv::mixChannels(&tmp8, 1, &out, 1, fromTo, 3);
		}
		if(alphaBand)
		{
			cv::Mat a(outH, outW, CV_32FC1);
			if(GDALRasterIOEx(GDALGetRasterBand(ds, alphaBand), GF_Read, x0, y0, w, h,
					a.data, outW, outH, GDT_Float32, 4, (GSpacing)a.step, &extra) != CE_None)
			{
				return cv::Mat();
			}
			cv::Mat a8;
			a.convertTo(a8, CV_8U, alpha, -scaleMin_*alpha);
			int fromTo[] = {0,3};
			cv::mixChannels(&a8, 1, &out, 1, fromTo, 1);
		}
	}

	if(!alphaBand && !hasPalette_)
	{
		// nodata / internal mask -> alpha
		GDALRasterBandH band1 = GDALGetRasterBand(ds, 1);
		int flags = GDALGetMaskFlags(band1);
		if(!(flags & GMF_ALL_VALID))
		{
			GDALRasterIOExtraArg maskExtra;
			INIT_RASTERIO_EXTRA_ARG(maskExtra);
			maskExtra.eResampleAlg = GRIORA_Average;
			if(GDALRasterIOEx(GDALGetMaskBand(band1), GF_Read, x0, y0, w, h,
					out.data + 3, outW, outH, GDT_Byte, 4, (GSpacing)out.step, &maskExtra) != CE_None)
			{
				UWARN("Failed to read mask band, ignoring nodata.");
			}
		}
	}
	return out;
#endif
}

} // namespace rtabmap
