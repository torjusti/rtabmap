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

#include "hnswlib/HnswIndex.h"

#include <rtabmap/utilite/ULogger.h>

#include "hnswlib/hnswlib.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace rtabmap {

class HnswIndexImpl
{
public:
	HnswIndexImpl(int dim, size_t maxElements, int m, int efConstruction) :
		space(dim),
		hnsw(&space,
			maxElements,
			m,
			efConstruction,
			100, // random seed
			true), // allow searches to reuse the slots of removed points
		nextIndex(0)
	{}

	hnswlib::L2Space space;
	hnswlib::HierarchicalNSW<float> hnsw;
	// The index given out for the next added point. Indexes are hnswlib
	// labels: they are never reused, a removed point's slot being relabeled
	// when it is reused.
	size_t nextIndex;
};

HnswIndex::HnswIndex() :
		index_(0),
		featuresDim_(0)
{
}

HnswIndex::~HnswIndex()
{
	this->release();
}

void HnswIndex::release()
{
	delete index_;
	index_ = 0;
	featuresDim_ = 0;
}

void HnswIndex::buildIndex(
		const cv::Mat & features,
		int m,
		int efConstruction)
{
	this->release();
	UASSERT_MSG(features.type() == CV_32FC1, "HNSW only supports float descriptors (CV_32FC1), convert binary descriptors first.");
	UASSERT(features.rows > 0 && features.cols > 0);

	featuresDim_ = features.cols;
	// Capacity is grown by addPoints() when needed; start with room over the
	// initial batch so that the first additions don't resize right away.
	size_t capacity = (size_t)features.rows + (size_t)features.rows/2 + 8192;
	index_ = new HnswIndexImpl(featuresDim_, capacity, m, efConstruction);

	#pragma omp parallel for schedule(dynamic, 8)
	for(int i=0; i<features.rows; ++i)
	{
		index_->hnsw.addPoint(features.ptr<float>(i), (hnswlib::labeltype)i, true);
	}
	index_->nextIndex = (size_t)features.rows;
}

size_t HnswIndex::indexedFeatures() const
{
	if(!index_)
	{
		return 0;
	}
	return index_->hnsw.getCurrentElementCount() - index_->hnsw.getDeletedCount();
}

size_t HnswIndex::memoryUsed() const
{
	if(!index_)
	{
		return 0;
	}
	// The serialized size covers the point data, the level-0 links and the
	// upper-level links, which is the bulk of what the graph allocates.
	return index_->hnsw.indexFileSize();
}

std::vector<unsigned int> HnswIndex::addPoints(const cv::Mat & features)
{
	if(!index_)
	{
		UERROR("HNSW index not yet created!");
		return std::vector<unsigned int>();
	}
	UASSERT(features.type() == CV_32FC1);
	UASSERT(features.cols == featuresDim_);

	// Points replace removed slots first; only the net growth needs capacity.
	size_t vacant = index_->hnsw.getDeletedCount();
	size_t grow = (size_t)features.rows > vacant ? (size_t)features.rows - vacant : 0;
	size_t needed = index_->hnsw.getCurrentElementCount() + grow;
	if(needed > index_->hnsw.getMaxElements())
	{
		size_t newCapacity = std::max(needed + 8192, index_->hnsw.getMaxElements() + index_->hnsw.getMaxElements()/2);
		UDEBUG("Resizing HNSW index: %ld -> %ld", index_->hnsw.getMaxElements(), newCapacity);
		index_->hnsw.resizeIndex(newCapacity);
	}

	size_t baseIndex = index_->nextIndex;
	#pragma omp parallel for schedule(dynamic, 8)
	for(int i=0; i<features.rows; ++i)
	{
		index_->hnsw.addPoint(features.ptr<float>(i), (hnswlib::labeltype)(baseIndex + i), true);
	}
	index_->nextIndex += (size_t)features.rows;

	std::vector<unsigned int> indexes(features.rows);
	for(int i=0; i<features.rows; ++i)
	{
		indexes[i] = (unsigned int)(baseIndex + i);
	}
	return indexes;
}

void HnswIndex::removePoint(unsigned int index)
{
	if(!index_)
	{
		UERROR("HNSW index not yet created!");
		return;
	}
	index_->hnsw.markDelete((hnswlib::labeltype)index);
}

void HnswIndex::knnSearch(
		const cv::Mat & query,
		cv::Mat & indices,
		cv::Mat & dists,
		int knn,
		int ef,
		int cores) const
{
	if(!index_)
	{
		UERROR("HNSW index not yet created!");
		return;
	}
	UASSERT(query.type() == CV_32FC1);
	UASSERT(query.cols == featuresDim_);
	UASSERT(knn > 0);

	indices.create(query.rows, knn, CV_32S);
	dists.create(query.rows, knn, CV_32F);
	indices.setTo(-1);
	dists.setTo(-1.0f);

	if(indexedFeatures() == 0)
	{
		return;
	}

	// searchKnn() reads its budget from the index, clamped to knn at least.
	index_->hnsw.setEf((size_t)(ef > knn ? ef : knn));

	if(cores <= 0)
	{
#ifdef _OPENMP
		cores = omp_get_max_threads();
#else
		cores = 1;
#endif
	}

	#pragma omp parallel for schedule(dynamic, 16) num_threads(cores)
	for(int i=0; i<query.rows; ++i)
	{
		// max-heap of (squared distance, index), worst on top, at most knn
		// entries (fewer if not that many points are indexed)
		std::priority_queue<std::pair<float, hnswlib::labeltype> > result =
			index_->hnsw.searchKnn(query.ptr<float>(i), (size_t)knn);
		int * indicesRow = indices.ptr<int>(i);
		float * distsRow = dists.ptr<float>(i);
		int j = (int)result.size() - 1;
		for(; !result.empty(); result.pop(), --j)
		{
			indicesRow[j] = (int)result.top().second;
			distsRow[j] = result.top().first;
		}
	}
}

} /* namespace rtabmap */
