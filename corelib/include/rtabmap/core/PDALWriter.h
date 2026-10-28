/*
Copyright (c) 2010-2021, Mathieu Labbe - IntRoLab - Universite de Sherbrooke
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
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef CORELIB_INCLUDE_RTABMAP_CORE_PDALWRITER_H_
#define CORELIB_INCLUDE_RTABMAP_CORE_PDALWRITER_H_

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace rtabmap {

// Optional georeferencing of the exported coordinates. The offset (e.g., the
// CRS coordinates of the map origin) is added in double precision, so the
// cloud can stay in small local coordinates in memory. Only meaningful for
// LAS/LAZ outputs: the offset is also stored as the header offset, the
// srs (e.g., "EPSG:2154") as the coordinate reference system.
struct PDALGeoreference
{
	PDALGeoreference() : offsetX(0.0), offsetY(0.0), offsetZ(0.0) {}
	bool isNull() const {return offsetX==0.0 && offsetY==0.0 && offsetZ==0.0 && srs.empty();}
	double offsetX;
	double offsetY;
	double offsetZ;
	std::string srs;
};

std::string getPDALSupportedWriters();

int savePDALFile(const std::string & filePath, const pcl::PointCloud<pcl::PointXYZ> & cloud,          const std::vector<int> & cameraIds = std::vector<int>(), bool binary = false, const PDALGeoreference & georef = PDALGeoreference());
int savePDALFile(const std::string & filePath, const pcl::PointCloud<pcl::PointXYZRGB> & cloud,       const std::vector<int> & cameraIds = std::vector<int>(), bool binary = false, const std::vector<float> & intensities = std::vector<float>(), const PDALGeoreference & georef = PDALGeoreference());
int savePDALFile(const std::string & filePath, const pcl::PointCloud<pcl::PointXYZRGBNormal> & cloud, const std::vector<int> & cameraIds = std::vector<int>(), bool binary = false, const std::vector<float> & intensities = std::vector<float>(), const PDALGeoreference & georef = PDALGeoreference());
int savePDALFile(const std::string & filePath, const pcl::PointCloud<pcl::PointXYZI> & cloud,         const std::vector<int> & cameraIds = std::vector<int>(), bool binary = false, const PDALGeoreference & georef = PDALGeoreference());
int savePDALFile(const std::string & filePath, const pcl::PointCloud<pcl::PointXYZINormal> & cloud,   const std::vector<int> & cameraIds = std::vector<int>(), bool binary = false, const PDALGeoreference & georef = PDALGeoreference());

}


#endif /* CORELIB_INCLUDE_RTABMAP_CORE_PDALWRITER_H_ */
