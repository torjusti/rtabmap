/*
Copyright (c) 2010-2016, Mathieu Labbe - IntRoLab - Universite de Sherbrooke
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

#ifndef RTABMAP_CORE_COLMAP_H_
#define RTABMAP_CORE_COLMAP_H_

#include "rtabmap/core/rtabmap_core_export.h"

#include <rtabmap/core/Transform.h>
#include <rtabmap/core/Link.h>
#include <rtabmap/core/Parameters.h>
#include <rtabmap/core/Optimizer.h>
#include <rtabmap/core/Signature.h>
#include <rtabmap/core/ProgressState.h>
#include <map>
#include <string>

namespace rtabmap {

class DBDriver;

// Matches the export dialog's camera model combo box order.
enum ColmapCameraModel
{
	kColmapPinhole = 0,
	kColmapOpenCV = 1,
	kColmapSimpleRadial = 2
};

/**
 * Options for exportColmap(), shared by ExportColmapDialog and
 * rtabmap-export --colmap so the two cannot drift apart.
 *
 * OPENCV is the default camera model because exported images are the stored
 * (possibly distorted) frames. PINHOLE drops distortion; use it only when the
 * images are already rectified or D is empty.
 */
struct RTABMAP_CORE_EXPORT ColmapExportOptions
{
	int cameraModel;             // ColmapCameraModel
	bool exportPoints;           // run bundle adjustment / correspondences and write points3D.txt
	bool exportImages;           // write images/<id>.jpg
	bool allKeypoints;           // write every detected keypoint, not just the ones with a 3D point
	Optimizer::Type sbaType;
	int sbaIterations;           // 0 = build correspondences only, skip the solve
	double sbaVariance;
	bool sbaRematchFeatures;
	double maxLinearVelocity;    // m/s, 0=disabled
	double maxAngularVelocity;   // rad/s, 0=disabled
	double laplacianThr;         // 0=disabled

	ColmapExportOptions();
};

struct ColmapExportResult
{
	bool ok;
	std::string error;
	int camerasExported;
	int posesTotal;
	int pointsExported;
	int droppedPoints;         // dropped for having < 2 remaining views after camera filtering / outliers
	int outliersRemoved;       // observations dropped after BA's robust kernel

	ColmapExportResult() :
		ok(false), camerasExported(0), posesTotal(0),
		pointsExported(0), droppedPoints(0), outliersRemoved(0) {}
};

/**
 * Writes a COLMAP text model (cameras.txt, images.txt, points3D.txt, and
 * optionally an images/ directory of jpgs) to outputPath.
 *
 * @p poses is updated in place with the bundle-adjusted poses only when the
 * solve succeeds. A failed solve leaves the input poses untouched.
 *
 * Provide @p signatures, @p dbDriver, or both. When signatures are empty the
 * frames are loaded from the database (without images) for BA; images are
 * then streamed one at a time. When signatures already hold the frames
 * (MainWindow cache), the database is optional.
 */
ColmapExportResult RTABMAP_CORE_EXPORT exportColmap(
		std::map<int, Transform> & poses,
		const std::multimap<int, Link> & links,
		const std::map<int, Signature> & signatures,
		DBDriver * dbDriver,
		const ParametersMap & parameters,
		const ColmapExportOptions & options,
		const std::string & outputPath,
		const ProgressState * progressState = 0);

} // namespace rtabmap

#endif // RTABMAP_CORE_COLMAP_H_
