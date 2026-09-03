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

#include "CloudViewerBasemap.h"
#include <rtabmap/utilite/ULogger.h>

#include <opencv/vtkImageMatSource.h>

#include <vtkVersion.h>
#include <vtkRenderer.h>
#include <vtkCamera.h>
#include <vtkActor.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkFloatArray.h>
#include <vtkPointData.h>
#include <vtkTexture.h>
#include <vtkProperty.h>
#include <vtkMatrix4x4.h>
#include <vtkPropCollection.h>
#include <vtkCallbackCommand.h>
#include <vtkCommand.h>

#include <cmath>
#include <set>
#include <algorithm>

namespace rtabmap {

// Vertical separation between pyramid levels so that finer tiles win the
// depth test over the coarser ancestors drawn underneath while they load.
static const double kLevelZStep = 0.002;

CloudViewerBasemap::CloudViewerBasemap(vtkRenderer * renderer, QObject * parent) :
		QObject(parent),
		renderer_(renderer),
		z_(0.0),
		opacity_(1.0),
		visible_(true),
		maxTiles_(400),
		detail_(1.0),
		lastCameraMTime_(0),
		dirty_(false),
		clippingObserverTag_(0)
{
	lastSize_[0] = lastSize_[1] = 0;
	timer_.setInterval(100);
	connect(&timer_, SIGNAL(timeout()), this, SLOT(poll()));
	if(renderer_)
	{
		vtkSmartPointer<vtkCallbackCommand> callback = vtkSmartPointer<vtkCallbackCommand>::New();
		callback->SetClientData(this);
		callback->SetCallback(&CloudViewerBasemap::onResetClippingRange);
		clippingObserverTag_ = renderer_->AddObserver(vtkCommand::ResetCameraClippingRangeEvent, callback);
	}
}

CloudViewerBasemap::~CloudViewerBasemap()
{
	timer_.stop();
	clearActors();
	if(renderer_ && clippingObserverTag_)
	{
		renderer_->RemoveObserver(clippingObserverTag_);
	}
}

void CloudViewerBasemap::onResetClippingRange(vtkObject *, unsigned long, void * clientData, void *)
{
	static_cast<CloudViewerBasemap*>(clientData)->extendClippingRange();
}

void CloudViewerBasemap::extendClippingRange() const
{
	if(!renderer_ || !visible_ || actors_.empty())
	{
		return;
	}
	vtkCamera * camera = renderer_->GetActiveCamera();
	double pos[3], dop[3], range[2];
	camera->GetPosition(pos);
	camera->GetDirectionOfProjection(dop);
	camera->GetClippingRange(range);
	double tMin = 1e300, tMax = -1e300;
	for(std::map<BasemapTileKey, vtkSmartPointer<vtkActor> >::const_iterator iter=actors_.begin(); iter!=actors_.end(); ++iter)
	{
		double b[6];
		iter->second->GetBounds(b);
		for(int c=0; c<8; ++c)
		{
			double x = (c&1)?b[1]:b[0];
			double y = (c&2)?b[3]:b[2];
			double z = (c&4)?b[5]:b[4];
			double t = (x-pos[0])*dop[0] + (y-pos[1])*dop[1] + (z-pos[2])*dop[2];
			tMin = std::min(tMin, t);
			tMax = std::max(tMax, t);
		}
	}
	if(tMax <= 0.0)
	{
		return; // entirely behind the camera
	}
	double nearPlane = std::min(range[0], tMin*0.99);
	double farPlane = std::max(range[1], tMax*1.01);
	double tolerance = renderer_->GetNearClippingPlaneTolerance();
	if(nearPlane < tolerance*farPlane)
	{
		nearPlane = tolerance*farPlane;
	}
	if(nearPlane != range[0] || farPlane != range[1])
	{
		camera->SetClippingRange(nearPlane, farPlane);
	}
}

void CloudViewerBasemap::setSource(const std::shared_ptr<BasemapTileSource> & source)
{
	if(source_ == source)
	{
		return;
	}
	if(source_)
	{
		disconnect(source_.get(), 0, this, 0);
	}
	clearActors();
	source_ = source;
	if(source_)
	{
		connect(source_.get(), SIGNAL(tileReady(const rtabmap::BasemapTileKey &)), this, SLOT(onTileReady(const rtabmap::BasemapTileKey &)));
		connect(source_.get(), SIGNAL(cleared()), this, SLOT(onSourceCleared()));
		connect(source_.get(), SIGNAL(loaded()), this, SLOT(onSourceLoaded()));
		timer_.start();
	}
	else
	{
		timer_.stop();
	}
	update(true);
}

void CloudViewerBasemap::setZ(double z)
{
	if(z_ != z)
	{
		z_ = z;
		for(std::map<BasemapTileKey, vtkSmartPointer<vtkActor> >::iterator iter=actors_.begin(); iter!=actors_.end(); ++iter)
		{
			applyProperties(iter->second, iter->first.level);
		}
		if(renderer_ && !actors_.empty())
		{
			renderer_->ResetCameraClippingRange();
		}
		Q_EMIT changed();
	}
}

void CloudViewerBasemap::setOpacity(double opacity)
{
	opacity = std::max(0.0, std::min(1.0, opacity));
	if(opacity_ != opacity)
	{
		opacity_ = opacity;
		for(std::map<BasemapTileKey, vtkSmartPointer<vtkActor> >::iterator iter=actors_.begin(); iter!=actors_.end(); ++iter)
		{
			applyProperties(iter->second, iter->first.level);
		}
		Q_EMIT changed();
	}
}

void CloudViewerBasemap::setVisible(bool visible)
{
	if(visible_ != visible)
	{
		visible_ = visible;
		update(true);
	}
}

bool CloudViewerBasemap::isBasemapActor(vtkProp * prop) const
{
	for(std::map<BasemapTileKey, vtkSmartPointer<vtkActor> >::const_iterator iter=actors_.begin(); iter!=actors_.end(); ++iter)
	{
		if(iter->second.GetPointer() == prop)
		{
			return true;
		}
	}
	return false;
}

std::vector<vtkProp*> CloudViewerBasemap::actors() const
{
	std::vector<vtkProp*> out;
	for(std::map<BasemapTileKey, vtkSmartPointer<vtkActor> >::const_iterator iter=actors_.begin(); iter!=actors_.end(); ++iter)
	{
		out.push_back(iter->second.GetPointer());
	}
	return out;
}

bool CloudViewerBasemap::pick(int displayX, int displayY, double & mapX, double & mapY) const
{
	if(actors_.empty() || !renderer_ || !isActive())
	{
		return false;
	}
	// Analytic ray/plane intersection: the basemap is a plane at z_, so this
	// is exact, sees through the clouds and does not depend on the camera
	// clipping range (the tiles are excluded from the bounds used to set it).
	double p0[3], p1[3], w0[4], w1[4];
	renderer_->SetDisplayPoint(displayX, displayY, 0.0);
	renderer_->DisplayToWorld();
	renderer_->GetWorldPoint(w0);
	renderer_->SetDisplayPoint(displayX, displayY, 1.0);
	renderer_->DisplayToWorld();
	renderer_->GetWorldPoint(w1);
	if(std::fabs(w0[3]) < 1e-12 || std::fabs(w1[3]) < 1e-12)
	{
		return false;
	}
	for(int i=0; i<3; ++i)
	{
		p0[i] = w0[i]/w0[3];
		p1[i] = w1[i]/w1[3];
	}
	double dz = p1[2] - p0[2];
	if(std::fabs(dz) < 1e-12)
	{
		return false; // looking parallel to the basemap
	}
	double t = (z_ - p0[2]) / dz;
	if(t < 0.0)
	{
		return false; // plane is behind the camera
	}
	double x = p0[0] + t*(p1[0]-p0[0]);
	double y = p0[1] + t*(p1[1]-p0[1]);
	double xMin, yMin, xMax, yMax;
	if(!source_->tileBoundsMap(source_->rootKey(), xMin, yMin, xMax, yMax) ||
	   x < xMin || x > xMax || y < yMin || y > yMax)
	{
		return false; // outside the raster
	}
	mapX = x;
	mapY = y;
	return true;
}

void CloudViewerBasemap::poll()
{
	if(!renderer_ || !isActive() || !visible_)
	{
		return;
	}
	vtkCamera * camera = renderer_->GetActiveCamera();
	int * size = renderer_->GetSize();
	if(dirty_ ||
	   camera->GetMTime() != lastCameraMTime_ ||
	   size[0] != lastSize_[0] || size[1] != lastSize_[1])
	{
		update(true);
	}
}

void CloudViewerBasemap::onTileReady(const rtabmap::BasemapTileKey &)
{
	// Re-evaluate on the next poll so that a burst of tiles is handled once
	dirty_ = true;
}

void CloudViewerBasemap::onSourceCleared()
{
	clearActors();
	Q_EMIT changed();
}

void CloudViewerBasemap::onSourceLoaded()
{
	update(true);
}

void CloudViewerBasemap::clearActors()
{
	if(renderer_)
	{
		for(std::map<BasemapTileKey, vtkSmartPointer<vtkActor> >::iterator iter=actors_.begin(); iter!=actors_.end(); ++iter)
		{
			renderer_->RemoveActor(iter->second);
		}
	}
	actors_.clear();
}

void CloudViewerBasemap::applyProperties(vtkActor * actor, int level) const
{
	actor->SetPosition(0.0, 0.0, z_ - level*kLevelZStep);
	actor->GetProperty()->SetOpacity(opacity_);
	actor->SetVisibility(visible_);
}

vtkSmartPointer<vtkActor> CloudViewerBasemap::createActor(const BasemapTileKey & key, const cv::Mat & image) const
{
	double corners[4][2];
	if(!source_->tileCornersMap(key, corners) || image.empty())
	{
		return vtkSmartPointer<vtkActor>();
	}

	vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
	points->SetDataTypeToDouble();
	points->SetNumberOfPoints(4);
	// z is applied through the actor position so that it can be changed
	// without rebuilding the geometry
	for(int i=0; i<4; ++i)
	{
		points->SetPoint(i, corners[i][0], corners[i][1], 0.0);
	}

	vtkSmartPointer<vtkCellArray> polys = vtkSmartPointer<vtkCellArray>::New();
	vtkIdType ids[4] = {0, 1, 2, 3};
	polys->InsertNextCell(4, ids);

	// vtkImageMatSource flips the image vertically (row 0 -> t=1), so the
	// raster's top-left corner maps to (0,1).
	vtkSmartPointer<vtkFloatArray> tcoords = vtkSmartPointer<vtkFloatArray>::New();
	tcoords->SetNumberOfComponents(2);
	tcoords->SetNumberOfTuples(4);
	tcoords->SetTuple2(0, 0.0, 1.0);
	tcoords->SetTuple2(1, 1.0, 1.0);
	tcoords->SetTuple2(2, 1.0, 0.0);
	tcoords->SetTuple2(3, 0.0, 0.0);
	tcoords->SetName("TCoords");

	vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
	polydata->SetPoints(points);
	polydata->SetPolys(polys);
	polydata->GetPointData()->SetTCoords(tcoords);

	vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	mapper->SetInputData(polydata);
	mapper->ScalarVisibilityOff();

	vtkSmartPointer<vtkImageMatSource> imageSource = vtkSmartPointer<vtkImageMatSource>::New();
	imageSource->SetImage(image);
	imageSource->Update();
	vtkSmartPointer<vtkTexture> texture = vtkSmartPointer<vtkTexture>::New();
	texture->SetInputConnection(imageSource->GetOutputPort());
	texture->InterpolateOn();
#if VTK_MAJOR_VERSION >= 9
	texture->SetWrap(vtkTexture::ClampToEdge);
#else
	texture->RepeatOff();
	texture->EdgeClampOn();
#endif

	vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
	actor->SetMapper(mapper);
	actor->SetTexture(texture);
	actor->GetProperty()->SetLighting(false);
	actor->GetProperty()->SetAmbient(1.0);
	actor->GetProperty()->SetDiffuse(0.0);
	actor->GetProperty()->SetSpecular(0.0);
	actor->GetProperty()->SetInterpolationToFlat();
	actor->GetProperty()->BackfaceCullingOff();
	actor->GetProperty()->FrontfaceCullingOff();
	actor->PickableOff();
	// Keep the (possibly huge) basemap out of the bounds used to reset or
	// center the camera on the data.
	actor->UseBoundsOff();
	applyProperties(actor, key.level);
	return actor;
}

namespace {
struct Projector
{
	Projector(vtkCamera * camera, double aspect, int width, int height) :
		width_(width), height_(height)
	{
		vtkMatrix4x4 * m = camera->GetCompositeProjectionTransformMatrix(aspect, -1.0, 1.0);
		for(int i=0; i<16; ++i)
		{
			m_[i] = m->GetData()[i];
		}
	}
	// Returns false if the point is behind the camera
	bool project(double x, double y, double z, double & dx, double & dy) const
	{
		double px = m_[0]*x + m_[1]*y + m_[2]*z + m_[3];
		double py = m_[4]*x + m_[5]*y + m_[6]*z + m_[7];
		double pw = m_[12]*x + m_[13]*y + m_[14]*z + m_[15];
		if(pw <= 1e-9)
		{
			return false;
		}
		dx = (px/pw + 1.0) * 0.5 * width_;
		dy = (py/pw + 1.0) * 0.5 * height_;
		return true;
	}
	double m_[16];
	int width_;
	int height_;
};

// Axis-aligned box (at fixed z) against the 4 side planes of the frustum
bool intersectsFrustum(const double planes[24], double xMin, double yMin, double xMax, double yMax, double z)
{
	for(int p=0; p<4; ++p)
	{
		const double * pl = planes + p*4;
		// most inside corner
		double x = pl[0] >= 0 ? xMax : xMin;
		double y = pl[1] >= 0 ? yMax : yMin;
		if(pl[0]*x + pl[1]*y + pl[2]*z + pl[3] < 0.0)
		{
			return false;
		}
	}
	return true;
}
}

bool CloudViewerBasemap::computeTargets(std::vector<BasemapTileKey> & targets, double & focalX, double & focalY) const
{
	targets.clear();
	if(!renderer_ || !isActive())
	{
		return false;
	}
	vtkCamera * camera = renderer_->GetActiveCamera();
	int * size = renderer_->GetSize();
	if(!camera || size[0] <= 0 || size[1] <= 0)
	{
		return false;
	}
	double aspect = double(size[0]) / double(size[1]);
	double planes[24];
	camera->GetFrustumPlanes(aspect, planes);
	Projector projector(camera, aspect, size[0], size[1]);
	double focal[3];
	camera->GetFocalPoint(focal);
	focalX = focal[0];
	focalY = focal[1];

	std::vector<BasemapTileKey> current;
	current.push_back(source_->rootKey());
	while(!current.empty())
	{
		std::vector<BasemapTileKey> refine;
		for(size_t i=0; i<current.size(); ++i)
		{
			const BasemapTileKey & key = current[i];
			double xMin, yMin, xMax, yMax;
			if(!source_->tileBoundsMap(key, xMin, yMin, xMax, yMax))
			{
				continue;
			}
			if(!intersectsFrustum(planes, xMin, yMin, xMax, yMax, z_))
			{
				continue;
			}
			if(key.level == 0)
			{
				targets.push_back(key);
				continue;
			}
			// projected size of the tile on screen
			double corners[4][2];
			source_->tileCornersMap(key, corners);
			double dxMin = 1e30, dxMax = -1e30, dyMin = 1e30, dyMax = -1e30;
			bool behind = false;
			for(int c=0; c<4; ++c)
			{
				double dx, dy;
				if(!projector.project(corners[c][0], corners[c][1], z_, dx, dy))
				{
					behind = true;
					break;
				}
				dxMin = std::min(dxMin, dx); dxMax = std::max(dxMax, dx);
				dyMin = std::min(dyMin, dy); dyMax = std::max(dyMax, dy);
			}
			int texW, texH;
			source_->tileTextureSize(key, texW, texH);
			bool shouldRefine = behind ||
					(dxMax - dxMin) > texW*detail_ ||
					(dyMax - dyMin) > texH*detail_;
			if(shouldRefine)
			{
				refine.push_back(key);
			}
			else
			{
				targets.push_back(key);
			}
		}
		current.clear();
		if(refine.empty())
		{
			break;
		}
		if((int)(targets.size() + refine.size()*4) > maxTiles_)
		{
			// out of budget: keep this level
			targets.insert(targets.end(), refine.begin(), refine.end());
			break;
		}
		for(size_t i=0; i<refine.size(); ++i)
		{
			std::vector<BasemapTileKey> children = source_->children(refine[i]);
			current.insert(current.end(), children.begin(), children.end());
		}
	}
	return true;
}

void CloudViewerBasemap::update(bool force)
{
	dirty_ = false;
	if(!renderer_)
	{
		return;
	}
	if(!isActive() || !visible_)
	{
		if(!actors_.empty())
		{
			clearActors();
			Q_EMIT changed();
		}
		if(source_ && source_->isLoaded())
		{
			source_->cancelPending();
		}
		return;
	}
	vtkCamera * camera = renderer_->GetActiveCamera();
	int * size = renderer_->GetSize();
	if(!force &&
	   camera->GetMTime() == lastCameraMTime_ &&
	   size[0] == lastSize_[0] && size[1] == lastSize_[1])
	{
		return;
	}
	lastCameraMTime_ = camera->GetMTime();
	lastSize_[0] = size[0];
	lastSize_[1] = size[1];

	std::vector<BasemapTileKey> targets;
	double focalX = 0.0, focalY = 0.0;
	if(!computeTargets(targets, focalX, focalY))
	{
		return;
	}

	// What to display now (targets or their best cached ancestors) and what
	// to request (missing targets and their missing ancestors, coarse first
	// so that the view refines progressively).
	std::set<BasemapTileKey> display;
	std::vector<std::pair<BasemapTileKey, double> > wanted;
	int maxLevel = source_->maxLevel();
	for(size_t i=0; i<targets.size(); ++i)
	{
		const BasemapTileKey & key = targets[i];
		double xMin, yMin, xMax, yMax;
		source_->tileBoundsMap(key, xMin, yMin, xMax, yMax);
		double cx = (xMin+xMax)*0.5, cy = (yMin+yMax)*0.5;
		double dist = std::sqrt((cx-focalX)*(cx-focalX) + (cy-focalY)*(cy-focalY));

		if(source_->isCached(key))
		{
			display.insert(key);
			continue;
		}
		BasemapTileKey k = key;
		for(;;)
		{
			if(source_->isCached(k))
			{
				display.insert(k);
				break;
			}
			wanted.push_back(std::make_pair(k, double(maxLevel - k.level)*1e7 + dist));
			if(k.level >= maxLevel)
			{
				break;
			}
			k = k.parent();
		}
	}
	source_->request(wanted);

	bool modified = false;
	// remove actors no longer displayed
	for(std::map<BasemapTileKey, vtkSmartPointer<vtkActor> >::iterator iter=actors_.begin(); iter!=actors_.end();)
	{
		if(display.find(iter->first) == display.end())
		{
			renderer_->RemoveActor(iter->second);
			actors_.erase(iter++);
			modified = true;
		}
		else
		{
			++iter;
		}
	}
	// add new ones
	for(std::set<BasemapTileKey>::const_iterator iter=display.begin(); iter!=display.end(); ++iter)
	{
		if(actors_.find(*iter) == actors_.end())
		{
			cv::Mat image = source_->tile(*iter);
			vtkSmartPointer<vtkActor> actor = createActor(*iter, image);
			if(actor)
			{
				renderer_->AddActor(actor);
				actors_.insert(std::make_pair(*iter, actor));
				modified = true;
			}
		}
	}
	if(modified)
	{
		renderer_->ResetCameraClippingRange();
		UDEBUG("Basemap: %d tiles displayed (%d targets, %d requested, cache=%d tiles / %.0f MB)",
				(int)actors_.size(), (int)targets.size(), (int)wanted.size(),
				source_->cachedTiles(), source_->cacheBytes()/(1024.0*1024.0));
		Q_EMIT changed();
	}
}

} // namespace rtabmap
