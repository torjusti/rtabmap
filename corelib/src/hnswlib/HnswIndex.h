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

#ifndef CORELIB_SRC_HNSWLIB_HNSWINDEX_H_
#define CORELIB_SRC_HNSWLIB_HNSWINDEX_H_

#include <opencv2/opencv.hpp>

namespace rtabmap {

class HnswIndexImpl;

/**
 * HNSW (Hierarchical Navigable Small World) graph backed by the vendored
 * hnswlib, held by FlannIndex when its HNSW_INDEX algorithm is selected.
 *
 * An approximate index for high-dimensional float descriptors whose recall
 * doesn't degrade as points are added and removed: unlike the rtflann kd-trees,
 * the graph never needs a full rebuild, so there is no rebuild stall however
 * large the dictionary gets, and removals (lazy, like the other structures)
 * don't erode the search quality of what remains.
 *
 * - addPoints() inserts the given batch in parallel over the available cores:
 *   inserting point by point is much more expensive per point than a kd-tree
 *   append, batching is what makes the cost per node comparable.
 * - removePoint() marks the point deleted; searches skip it and a later
 *   insertion reuses its slot.
 * - knnSearch() is approximate, its "ef" budget (the size of the candidate
 *   list kept while walking the graph, >= knn) playing the role of rtflann's
 *   "checks".
 *
 * Only CV_32FC1 features with the (squared) L2 metric are supported: binary
 * descriptors have to be converted to float first, as for the kd-trees.
 * Indexes returned by addPoints() are stable for the lifetime of the index.
 */
class HnswIndex
{
public:
	HnswIndex();
	~HnswIndex();

	HnswIndex(const HnswIndex &) = delete;
	HnswIndex & operator=(const HnswIndex &) = delete;

	void release();

	// features must be a CV_32FC1 matrix, one point per row, inserted in
	// parallel. "m" is the number of graph links per point and
	// "efConstruction" the candidate list size used while inserting: both
	// trade insertion time and memory for recall.
	void buildIndex(
			const cv::Mat & features,
			int m = 16,
			int efConstruction = 100);

	bool isBuilt() const {return index_ != 0;}

	// removed points excluded
	size_t indexedFeatures() const;

	// return Bytes
	size_t memoryUsed() const;

	// return the index assigned to each added point; the features are indexed
	// in parallel and the graph grows its capacity as needed
	std::vector<unsigned int> addPoints(const cv::Mat & features);

	void removePoint(unsigned int index);

	// return squared L2 distances, indices and distances are set to -1 for the
	// neighbors that couldn't be found. "ef" is the search budget (clamped to
	// knn at least), queries are searched in parallel ("cores" <= 0 for all).
	void knnSearch(
			const cv::Mat & query,
			cv::Mat & indices,
			cv::Mat & dists,
			int knn,
			int ef = 32,
			int cores = 1) const;

private:
	// hnswlib is template/header-only, kept out of this header behind the
	// implementation.
	HnswIndexImpl * index_;
	int featuresDim_;
};

} /* namespace rtabmap */

#endif /* CORELIB_SRC_HNSWLIB_HNSWINDEX_H_ */
